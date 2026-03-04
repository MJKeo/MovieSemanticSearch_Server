# Stage 2: TMDB Detail Fetching & Storage — Technical Product Guide

## Document Purpose

This is the complete specification for fetching movie data from TMDB's API and persisting it locally in a SQLite database. It covers everything an engineer needs to implement this stage from scratch: what we fetch, why we fetch it, what we store, how we store it, rate limiting, error handling, and how this stage feeds into the downstream quality funnel.

This stage is one piece of a larger ingestion pipeline that ultimately produces ~100K fully enriched movie records across Postgres, Qdrant, and Redis. The pipeline stages, in order:

```
Stage 1: TMDB Export Download & Initial Filtering     → ~1M → ~950K movie IDs
Stage 2: TMDB Detail Fetching (this document)          → ~950K movies fetched + stored
Stage 3: TMDB Quality Funnel                           → ~950K → ~100K candidates
Stage 4: IMDB Scraping                                 → ~100K movies enriched
Stage 5–6: LLM Generation (two Batch API phases)       → ~100K movies with embeddings text
Stage 7: Embedding                                     → ~100K movies with vectors
Stage 8: Database Ingestion (Postgres + Qdrant)         → Production-ready catalog
```

Stage 2 is the first stage that touches an external API. It runs after Stage 1 has produced a list of ~950K valid (non-adult, non-video) TMDB movie IDs in the checkpoint database with `status = 'pending'`.

---

## Why This Stage Exists

The pipeline needs to reduce ~950K candidate movies down to ~100K before running the expensive downstream stages (IMDB scraping, LLM generation, embedding). The only way to make that filtering decision intelligently is to fetch structured data from TMDB for every candidate.

The critical cost insight: TMDB fetching is free. TMDB's API requires only an API key (no payment, no proxy infrastructure). The only constraint is a rate limit (~40 requests/second on the standard tier). This means we can afford to fetch details for all ~950K movies — the cost is measured in hours of wall time, not dollars.

By contrast, IMDB scraping requires residential proxy infrastructure at ~$8/GB, costing ~$300–400 for 100K movies and ~$2,800–3,800 for 950K. Fetching TMDB data first and filtering aggressively before touching IMDB is the 10x cost lever for the entire pipeline.

---

## What We Fetch From TMDB

We hit the TMDB expanded detail endpoint once per movie:

```
GET https://api.themoviedb.org/3/movie/{tmdb_id}?append_to_response=release_dates,keywords,watch/providers,credits
```

The `append_to_response` parameter bundles four sub-resources into a single API call, which is critical: without it, we'd need 5 separate requests per movie (5x the rate limit consumption).

### What the endpoint returns

The response is a JSON object containing the core movie record plus the four appended sub-resources. The fields we care about fall into two categories.

**Fields we persist for downstream pipeline use (survive past Stage 3):**

These are fields that later stages (IMDB scraping, LLM generation, database ingestion) will need regardless of whether the movie passes the quality funnel. We store these at full fidelity.

| Field | Source location in response | Why we need it downstream |
|-------|---------------------------|--------------------------|
| `tmdb_id` | `id` | Universal movie identifier across all systems (Postgres PK, Qdrant point ID) |
| `imdb_id` | `imdb_id` | Required key for IMDB scraping (Stage 4); also used as a cross-reference throughout the pipeline |
| `title` | `title` | Lexical search indexing, movie card display, LLM prompt context |
| `release_date` | `release_date` | Metadata filtering (hard filter field in Qdrant and Postgres), display |
| `duration` | `runtime` | Metadata filtering (hard filter field), display |
| `poster_url` | `poster_path` | Movie card rendering; stored as TMDB path string (e.g., `"/8z7rC8uIDaTM91X0ZfkRf04ydj2.jpg"`) |
| `watch_provider_keys` | `watch/providers.results.US` | Metadata hard filtering in Qdrant and Postgres; encoded as integer keys |

**Fields we persist solely for quality filtering (used in Stage 3, discarded conceptually after):**

These fields exist in the database only to power the quality funnel's scoring and hard filter logic. They occupy minimal space because we store derived values, not raw data.

| Field | Source location in response | What we store | Why |
|-------|---------------------------|---------------|-----|
| `vote_count` | `vote_count` | Integer as-is | Strongest quality signal; log-scaled into engagement score |
| `popularity` | `popularity` | Float as-is | TMDB's composite metric (page views, votes, watchlist adds); secondary quality signal |
| `vote_average` | `vote_average` | Float as-is | Rating quality component, weighted by vote_count confidence |
| `overview_length` | `len(overview)` | Integer (character count) | Hard filter (overview must exist and be ≥ 20 chars); also a data completeness signal |
| `genre_count` | `len(genres)` | Integer (count) | Hard filter (must have ≥ 1 genre); data completeness signal |
| `has_revenue` | `revenue > 0` | Boolean (0 or 1) | Data completeness bonus in quality scoring |
| `has_budget` | `budget > 0` | Boolean (0 or 1) | Data completeness signal; movies with known budgets tend to be professionally produced |
| `has_production_companies` | `len(production_companies) > 0` | Boolean (0 or 1) | Data completeness signal; presence of a named studio indicates a legitimate, distributed film |
| `has_production_countries` | `len(production_countries) > 0` | Boolean (0 or 1) | Data completeness signal; country data indicates a properly cataloged release |
| `has_keywords` | `len(keywords.keywords) > 0` | Boolean (0 or 1) | Data completeness signal; keyword-tagged movies have been curated by TMDB contributors, suggesting notability |
| `has_cast_and_crew` | `len(credits.cast) > 0 AND len(credits.crew) > 0` | Boolean (0 or 1) | Data completeness signal; requires both cast and crew to be non-empty. A movie with no credited people is almost certainly a low-quality entry. |

### What we intentionally do NOT store from TMDB

The following data is available in the TMDB response but is not persisted. The rationale: we get higher-quality versions of all of this from IMDB scraping in Stage 4, and storing it from both sources would create confusing duplication.

| Data | Why we skip the full data |
|------|--------------------------|
| `overview` (full text) | We store only the character count. The LLM stages use IMDB's plot summaries, which are richer and more detailed than TMDB's overview. |
| `credits` (full cast/crew lists) | We store only a boolean (`has_cast_and_crew`) indicating both lists are non-empty. IMDB's full credits page provides complete cast, crew, and character names. TMDB credits are a subset. |
| `keywords` (full keyword objects) | We store only a boolean (`has_keywords`) indicating the list is non-empty. IMDB keywords are more comprehensive. TMDB keywords are useful but redundant once IMDB data exists. |
| `genres` (full objects) | We store only the count for filtering. Genre IDs for the final database come from a normalized mapping at ingestion time (Stage 8), not raw TMDB genre objects. |
| `release_dates` (certification data) | Maturity rating (`maturity_rank`) is derived from IMDB's parental guide, which is more detailed. |
| `budget` (dollar amount) | We store only a boolean (`has_budget`) indicating a nonzero value exists. The actual dollar figure isn't needed for filtering or downstream stages. |
| `production_companies` (full objects) | We store only a boolean (`has_production_companies`) indicating the list is non-empty. Full company names for lexical search indexing come from IMDB, which has cleaner data. |
| `production_countries` (full objects) | We store only a boolean (`has_production_countries`) indicating the list is non-empty. The actual country data for the production database is sourced from IMDB for consistency. |
| `spoken_languages`, `origin_country` | Used in the final database but sourced from IMDB for consistency. |
| `original_title` | Only needed if it differs from `title`; handled at ingestion time. |

### TMDB response extraction paths for boolean filter fields

The five new boolean fields derive from different locations in the TMDB response JSON. Some are top-level fields, others are nested inside `append_to_response` sub-resources. Here is the exact path to each, along with the gotchas.

**`has_budget`** — Top-level field.
```
response["budget"]  →  integer (0 if missing or unknown)
```
TMDB defaults `budget` to `0` when the value is unknown, not `null`. This means a simple null check is insufficient — you must check `budget > 0`.

**`has_production_companies`** — Top-level field.
```
response["production_companies"]  →  list of objects, each with "id", "name", etc.
```
An empty list `[]` means no companies are associated. Check `len(production_companies) > 0`.

**`has_production_countries`** — Top-level field.
```
response["production_countries"]  →  list of objects, each with "iso_3166_1", "name"
```
An empty list `[]` means no countries are associated. Check `len(production_countries) > 0`.

**`has_keywords`** — Nested inside the `keywords` append sub-resource.
```
response["keywords"]["keywords"]  →  list of objects, each with "id", "name"
```
Note the double nesting: the `append_to_response=keywords` expansion adds a top-level `"keywords"` key to the response, and the actual keyword list is under `"keywords"` within that object. If the append expansion is missing or the movie has no keywords, this path may yield an empty list or the `"keywords"` key may be absent entirely. Defensive access: `response.get("keywords", {}).get("keywords", [])`.

**`has_cast_and_crew`** — Nested inside the `credits` append sub-resource.
```
response["credits"]["cast"]  →  list of cast member objects
response["credits"]["crew"]  →  list of crew member objects
```
Both lists must be non-empty for `has_cast_and_crew` to be `True`. A movie might have crew but no credited cast (e.g., a documentary), or cast but no crew entries — either case should evaluate to `False`. Defensive access: `response.get("credits", {}).get("cast", [])` and `response.get("credits", {}).get("crew", [])`.

---

## SQLite Schema

All fetched data is stored in a single SQLite database (`./ingestion_data/tracker.db`) that also serves as the checkpoint tracker for the entire pipeline. Stage 2 writes to two tables: the main progress tracker (shared across all stages) and a new TMDB data table.

### Progress tracker (shared, created in Stage 1)

```sql
CREATE TABLE movie_progress (
    tmdb_id         INTEGER PRIMARY KEY,
    imdb_id         TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    -- Status progression:
    --   pending → tmdb_fetched → quality_passed → imdb_scraped →
    --   phase1_complete → phase2_complete → embedded → ingested
    --
    -- Terminal statuses:
    --   filtered_out         (failed a hard filter or scrape)
    --   below_quality_cutoff (didn't make the top ~100K)
    quality_score   REAL,
    batch1_custom_id TEXT,
    batch2_custom_id TEXT,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_progress_status ON movie_progress(status);
```

### Filter log (shared, created in Stage 1)

```sql
CREATE TABLE filter_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id     INTEGER NOT NULL,
    title       TEXT,
    year        INTEGER,
    stage       TEXT NOT NULL,
    reason      TEXT NOT NULL,
    details     TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_filter_log_stage ON filter_log(stage);
CREATE INDEX idx_filter_log_tmdb ON filter_log(tmdb_id);
```

### TMDB data table (new, created by Stage 2)

```sql
CREATE TABLE tmdb_data (
    tmdb_id             INTEGER PRIMARY KEY,
    imdb_id             TEXT,
    title               TEXT,
    release_date        TEXT,         -- ISO 8601 date string: "YYYY-MM-DD"
    duration            INTEGER,      -- Runtime in minutes; NULL if TMDB doesn't have it
    poster_url          TEXT,         -- TMDB poster path, e.g. "/8z7rC8uIDaTM91X0ZfkRf04ydj2.jpg"
    watch_provider_keys BLOB,         -- Packed list of int32s (see encoding above)

    -- Quality filter fields (minimal storage, used by Stage 3 only)
    vote_count          INTEGER DEFAULT 0,
    popularity          REAL DEFAULT 0.0,
    vote_average        REAL DEFAULT 0.0,
    overview_length     INTEGER DEFAULT 0,
    genre_count         INTEGER DEFAULT 0,
    has_revenue         INTEGER DEFAULT 0,  -- 0 or 1
    has_budget          INTEGER DEFAULT 0,  -- 0 or 1
    has_production_companies INTEGER DEFAULT 0,  -- 0 or 1
    has_production_countries INTEGER DEFAULT 0,  -- 0 or 1
    has_keywords        INTEGER DEFAULT 0,  -- 0 or 1
    has_cast_and_crew   INTEGER DEFAULT 0   -- 0 or 1; True only if BOTH cast and crew are non-empty
);
```

### Why a single SQLite database (not flat JSON files)

The original pipeline design stored TMDB responses as one JSON file per movie (~950K files totaling 3–5 GB). The SQLite approach is better for several reasons:

**Storage efficiency.** We're storing ~195 bytes per row instead of ~3–5 KB per JSON file. Total footprint: ~310–360 MB vs. 3–5 GB. This is roughly a 10x reduction.

**No filesystem overhead.** 950K small files create real problems: `ls` becomes unusable, `find` is slow, some filesystems degrade with this many inodes in a single directory, and backup/copy operations crawl. A single SQLite file has none of these issues.

**Atomic queries.** Stage 3 (the quality funnel) needs to rank all ~950K movies by a composite score and select the top 100K. With flat files, this means loading and parsing 950K JSON files sequentially. With SQLite, it's a single query that completes in seconds:

```sql
SELECT tmdb_id, vote_count, popularity, vote_average, overview_length, genre_count, has_revenue
FROM tmdb_data
WHERE tmdb_id IN (SELECT tmdb_id FROM movie_progress WHERE status = 'tmdb_fetched')
```

**Checkpoint integration.** The progress tracker is already in SQLite. Having the TMDB data in the same database means Stage 2's write (data + status update) can be a single transaction, which prevents inconsistencies if the process crashes between writing data and updating status.

### Watch provider key storage format

The `watch_provider_keys` column uses a BLOB containing packed 32-bit integers (`struct.pack` in Python). This is more compact than storing a JSON array or a TEXT comma-separated list:

```python
import struct

def pack_provider_keys(keys: list[int]) -> bytes:
    """Pack a list of watch_offer_keys into a compact BLOB."""
    return struct.pack(f'<{len(keys)}I', *keys)

def unpack_provider_keys(blob: bytes) -> list[int]:
    """Unpack a BLOB back into a list of watch_offer_keys."""
    if not blob:
        return []
    count = len(blob) // 4
    return list(struct.unpack(f'<{count}I', blob))
```

Typical movie has 5–15 watch provider keys → 20–60 bytes per movie. A JSON text representation of the same data would be 40–120 bytes. The savings add up at 950K rows but the real motivation is avoiding the need to parse/serialize text on every read.

---

## Storage Budget Analysis

Worst-case analysis for 1,000,000 rows (rounding up from ~950K for safety):

| Column | Avg bytes/row | Total (1M rows) |
|--------|--------------|-----------------|
| tmdb_id | 8 | 8 MB |
| imdb_id | 12 | 12 MB |
| title | 40 | 40 MB |
| release_date | 10 | 10 MB |
| duration | 4 | 4 MB |
| poster_url | 38 | 38 MB |
| watch_provider_keys | 50 | 50 MB |
| vote_count | 4 | 4 MB |
| popularity | 8 | 8 MB |
| vote_average | 8 | 8 MB |
| overview_length | 4 | 4 MB |
| genre_count | 2 | 2 MB |
| has_revenue | 1 | 1 MB |
| has_budget | 1 | 1 MB |
| has_production_companies | 1 | 1 MB |
| has_production_countries | 1 | 1 MB |
| has_keywords | 1 | 1 MB |
| has_cast_and_crew | 1 | 1 MB |
| **Row total** | **~195** | **~195 MB** |

Add SQLite per-row overhead (~25 bytes/row) and index storage (primary key on `tmdb_data`, primary key + status index on `movie_progress`):

| Component | Estimate |
|-----------|----------|
| Raw row data | ~195 MB |
| SQLite row overhead | ~25 MB |
| Primary key index (tmdb_data) | ~30 MB |
| movie_progress table + indexes | ~40 MB |
| filter_log table + indexes | ~20 MB |
| Page fragmentation overhead | ~30 MB |
| **Total database file** | **~310–360 MB** |

This fits comfortably in RAM on any modern laptop, which means SQLite read queries for Stage 3 will be served entirely from the OS page cache — effectively instant.

---

## TMDB API Configuration

### Authentication

TMDB uses bearer token authentication (not query parameter API keys). Every request must include:

```
Authorization: Bearer {TMDB_API_TOKEN}
```

The token is obtained from your TMDB account settings under API → API Read Access Token (v4 auth). This is different from the older v3 API key — the v4 bearer token is what you want.

### Rate limits

TMDB rate limits are keyed to your API token, not your IP address. This means proxies are irrelevant for TMDB — every request is identified by the same token regardless of origin IP.

The stated limit varies by tier:

| Tier | Limit | Effective throughput |
|------|-------|---------------------|
| Standard (most common) | ~40 requests/second | ~36–39 req/s with adaptive headroom |
| Lower tier | 40 requests per 10 seconds (= 4 req/s) | ~3.6 req/s with adaptive headroom |

**Before running the pipeline, verify your tier.** Make 50 rapid requests in a test script and observe where 429 responses begin. If they start around 40/s, you're on the standard tier. If they start around 4/s, you're on the lower tier and should request an upgrade from TMDB for bulk use.

### Duration estimates

| Tier | Movies to fetch | Effective rate | Wall time |
|------|----------------|---------------|-----------|
| Standard (40/s) | 950,000 | ~36–39 req/s | ~7–8 hours |
| Lower (4/s) | 950,000 | ~3.6 req/s | ~3 days |

At the standard tier, this stage runs unattended overnight. At the lower tier, the timeline becomes the dominant bottleneck in the entire pipeline, and upgrading your TMDB tier is worth pursuing.

---

## Rate Limiting Strategy

### Why a rate limiter (not a concurrency semaphore)

The constraint on TMDB is requests per second, not concurrent connections. A concurrency semaphore (e.g., `asyncio.Semaphore(35)`) limits how many requests are in-flight simultaneously, but says nothing about how many are initiated per second. If TMDB responds in 50ms on average, a semaphore of 35 would allow ~700 requests/second — far exceeding the limit.

What we need is a token bucket rate limiter: a mechanism that enforces "no more than N requests per second" regardless of response latency.

### Adaptive rate limiter

The rate limiter starts conservatively (90% of the stated limit) and adjusts based on 429 signals from TMDB. This creates a feedback loop that converges on the real server-side limit without manual tuning.

**Behavior:**

```
Start at 36 req/s
  → No 429s for 2 minutes → increase rate by 5% → ~37.8 req/s
  → No 429s for 2 minutes → increase rate by 5% → ~39.7 req/s
  → 429 received → decrease rate by 10% → ~35.7 req/s
  → (cycle repeats, hovering near the real limit)
```

**Parameters:**

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `initial_rate` | 36 req/s | 90% of stated 40/s limit; conservative start |
| `max_rate` | 40 req/s | Never exceed stated limit |
| `burst` | 5 | Max tokens in bucket; allows brief bursts after idle periods |
| `clean_window` | 120 seconds | Time without 429s before attempting a rate increase |
| Rate decrease on 429 | 10% | Aggressive enough to recover quickly |
| Rate increase on clean window | 5% | Gentle increase to avoid oscillation |
| Rate floor | 1 req/s | Never drop below this regardless of 429 frequency |

**Why starting conservative is nearly free:** At 36 req/s vs 40 req/s, 950K requests takes ~7.3 hours instead of ~6.6 hours. The 40-minute difference is irrelevant in a multi-day pipeline, but the cost of being too aggressive (bursts of 429s, forced backoffs, potential temporary bans) is much higher.

### Implementation

```python
import asyncio
import time

class AdaptiveRateLimiter:
    """
    Token-bucket rate limiter with adaptive rate adjustment.
    
    Maintains a bucket of tokens. Each request consumes one token.
    Tokens refill at `current_rate` per second, up to `burst` max.
    If the bucket is empty, callers sleep until a token is available.
    
    On 429 signals: rate drops by 10%.
    After clean_window seconds with no 429s: rate increases by 5%, up to max_rate.
    """
    
    def __init__(self, initial_rate: float, max_rate: float, burst: int = 5):
        self.current_rate = initial_rate
        self.max_rate = max_rate
        self.burst = burst
        self.tokens = burst
        self.last_refill = time.monotonic()
        self.last_429_time = 0.0
        self.clean_window = 120
        self._lock = asyncio.Lock()
        
        # Metrics
        self.total_requests = 0
        self.total_429s = 0

    async def acquire(self):
        """Block until a rate-limit token is available."""
        async with self._lock:
            now = time.monotonic()
            
            # Refill tokens based on elapsed time
            elapsed = now - self.last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.current_rate)
            self.last_refill = now
            
            # If no tokens available, sleep until one is
            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.current_rate
                await asyncio.sleep(wait)
                self.tokens = 0.0
                self.last_refill = time.monotonic()
            else:
                self.tokens -= 1.0
            
            # Check if we should increase rate
            if (now - self.last_429_time > self.clean_window 
                    and self.last_429_time > 0
                    and self.current_rate < self.max_rate):
                old_rate = self.current_rate
                self.current_rate = min(self.max_rate, self.current_rate * 1.05)
                if self.current_rate != old_rate:
                    print(f"  Rate limiter: increased to {self.current_rate:.1f} req/s")
            
            self.total_requests += 1

    def report_429(self):
        """Call when a 429 response is received. Reduces rate by 10%."""
        self.last_429_time = time.monotonic()
        self.total_429s += 1
        old_rate = self.current_rate
        self.current_rate = max(1.0, self.current_rate * 0.90)
        print(f"  Rate limiter: 429 received, reduced {old_rate:.1f} → "
              f"{self.current_rate:.1f} req/s "
              f"(total 429s: {self.total_429s}/{self.total_requests})")

    def stats(self) -> str:
        pct = (self.total_429s / max(1, self.total_requests)) * 100
        return (f"Rate: {self.current_rate:.1f} req/s | "
                f"Total: {self.total_requests:,} | "
                f"429s: {self.total_429s:,} ({pct:.2f}%)")
```

### Concurrency semaphore (separate concern)

In addition to the rate limiter, cap the number of simultaneous TCP connections. This protects your local machine and `aiohttp`'s connection pool — it is not a throughput control.

```python
# Rate limiter: controls requests per second (the TMDB constraint)
tmdb_limiter = AdaptiveRateLimiter(initial_rate=36, max_rate=40, burst=5)

# Concurrency semaphore: caps open TCP connections (local resource protection)
tmdb_semaphore = asyncio.Semaphore(100)
```

The semaphore is set to 100 (well above 36–40 req/s) so it almost never blocks. It exists purely as a safety net against local resource exhaustion.

---

## Per-Movie Fetch Process

For each movie with `status = 'pending'` in `movie_progress`:

### Step 1: Fetch from TMDB

Acquire a rate limiter token, then fire the request through the concurrency semaphore:

```python
async def fetch_tmdb_details(
    session: aiohttp.ClientSession, 
    tmdb_id: int,
    max_retries: int = 3
) -> dict | None:
    """Fetch TMDB expanded details for a single movie."""
    
    for attempt in range(max_retries):
        await tmdb_limiter.acquire()
        
        try:
            async with session.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                params={
                    "append_to_response": "release_dates,keywords,watch/providers,credits"
                },
                headers={"Authorization": f"Bearer {TMDB_API_TOKEN}"},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                elif resp.status == 429:
                    tmdb_limiter.report_429()
                    retry_after = int(resp.headers.get("Retry-After", 2))
                    await asyncio.sleep(retry_after)
                    continue
                elif resp.status == 404:
                    return None
                else:
                    raise FetchError(f"HTTP {resp.status}")
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                raise FetchError(f"Failed after {max_retries} attempts: {e}")
            await asyncio.sleep(1.0 * (attempt + 1))
    
    raise FetchError(f"Exhausted retries for tmdb_id={tmdb_id}")
```

### Step 2: Extract and transform fields

Parse the response into the flat structure matching our SQLite schema:

```python
def extract_tmdb_fields(raw: dict) -> dict:
    """Transform raw TMDB response into our storage schema."""
    
    # Watch provider extraction (US region only)
    providers = raw.get("watch/providers", {}).get("results", {}).get("US", {})
    watch_keys = set()
    for provider in providers.get("flatrate", []):
        watch_keys.add(provider["provider_id"] * 100 + 1)  # subscription
    for provider in providers.get("rent", []):
        watch_keys.add(provider["provider_id"] * 100 + 2)  # rent
    for provider in providers.get("buy", []):
        watch_keys.add(provider["provider_id"] * 100 + 3)  # buy
    
    overview = raw.get("overview", "") or ""
    genres = raw.get("genres", []) or []
    revenue = raw.get("revenue") or 0
    
    # --- Boolean filter field extraction ---
    # budget: top-level integer, defaults to 0 (not null) when unknown
    budget = raw.get("budget") or 0
    
    # production_companies: top-level list of objects
    production_companies = raw.get("production_companies", []) or []
    
    # production_countries: top-level list of objects
    production_countries = raw.get("production_countries", []) or []
    
    # keywords: nested under append_to_response expansion
    # response["keywords"]["keywords"] → list of keyword objects
    keywords_list = raw.get("keywords", {}).get("keywords", []) or []
    
    # credits: nested under append_to_response expansion
    # response["credits"]["cast"] → list, response["credits"]["crew"] → list
    credits = raw.get("credits", {})
    cast_list = credits.get("cast", []) or []
    crew_list = credits.get("crew", []) or []
    
    return {
        "tmdb_id":             raw["id"],
        "imdb_id":             raw.get("imdb_id"),
        "title":               raw.get("title"),
        "release_date":        raw.get("release_date"),
        "duration":            raw.get("runtime"),
        "poster_url":          raw.get("poster_path"),
        "watch_provider_keys": sorted(watch_keys),
        
        # Quality filter fields
        "vote_count":          raw.get("vote_count", 0),
        "popularity":          raw.get("popularity", 0.0),
        "vote_average":        raw.get("vote_average", 0.0),
        "overview_length":     len(overview),
        "genre_count":         len(genres),
        "has_revenue":         1 if revenue > 0 else 0,
        "has_budget":          1 if budget > 0 else 0,
        "has_production_companies": 1 if len(production_companies) > 0 else 0,
        "has_production_countries": 1 if len(production_countries) > 0 else 0,
        "has_keywords":        1 if len(keywords_list) > 0 else 0,
        "has_cast_and_crew":   1 if len(cast_list) > 0 and len(crew_list) > 0 else 0,
    }
```

### Step 3: Write to SQLite (single transaction)

The data insert and the status update happen in one transaction. If the process crashes between these two operations, neither takes effect, so resumption is clean.

```python
def persist_tmdb_movie(db: sqlite3.Connection, fields: dict):
    """Write extracted TMDB data and update progress status atomically."""
    
    packed_keys = pack_provider_keys(fields["watch_provider_keys"])
    
    db.execute("""
        INSERT OR REPLACE INTO tmdb_data (
            tmdb_id, imdb_id, title, release_date, duration, poster_url,
            watch_provider_keys, vote_count, popularity, vote_average,
            overview_length, genre_count, has_revenue, has_budget,
            has_production_companies, has_production_countries,
            has_keywords, has_cast_and_crew
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        fields["tmdb_id"], fields["imdb_id"], fields["title"],
        fields["release_date"], fields["duration"], fields["poster_url"],
        packed_keys, fields["vote_count"], fields["popularity"],
        fields["vote_average"], fields["overview_length"],
        fields["genre_count"], fields["has_revenue"], fields["has_budget"],
        fields["has_production_companies"], fields["has_production_countries"],
        fields["has_keywords"], fields["has_cast_and_crew"]
    ))
    
    db.execute("""
        UPDATE movie_progress 
        SET imdb_id = ?, status = 'tmdb_fetched', updated_at = CURRENT_TIMESTAMP
        WHERE tmdb_id = ?
    """, (fields["imdb_id"], fields["tmdb_id"]))
```

### Step 4: Handle missing IMDB ID

If `imdb_id` is null or missing, the movie cannot proceed to IMDB scraping. Filter it out immediately:

```python
if not fields["imdb_id"]:
    log_filter(db, fields["tmdb_id"], "tmdb_fetch", "missing_imdb_id")
    # log_filter sets status to 'filtered_out' internally
```

This check happens after writing to `tmdb_data` so the filter log can reference the movie's title and year.

---

## Error Handling

Every error state has a defined behavior. No errors are silently swallowed.

| Error | Behavior |
|-------|----------|
| **HTTP 200** | Parse, extract, persist. Normal path. |
| **HTTP 404** | Movie doesn't exist on TMDB. Write whatever data was returned (often none), log to `filter_log` with `reason='tmdb_404'`, set `status='filtered_out'`. |
| **HTTP 429** | Signal the rate limiter (`report_429()`), which reduces the rate by 10%. Sleep for `Retry-After` header duration (default 2s). Retry automatically. Not logged as a failure. |
| **HTTP 5xx** | Retry up to 3 times with exponential backoff (1s, 2s, 3s). If all retries fail, log to `filter_log` with `reason='tmdb_fetch_error'`, set `status='filtered_out'`. |
| **Network timeout** | Same as HTTP 5xx — retry with backoff, then filter out. |
| **JSON parse error** | Log to `filter_log` with `reason='tmdb_parse_error'`, set `status='filtered_out'`. |

### The `log_filter` helper

Every call to filter out a movie goes through this central helper, which ensures the filter log always includes the movie's title and year (if available) for human-readable sanity checking:

```python
def log_filter(db, tmdb_id: int, stage: str, reason: str, details: str = None):
    """Log a filtered movie with title/year for debugging."""
    title = None
    year = None
    
    # Pull title/year from tmdb_data if available
    row = db.execute(
        "SELECT title, release_date FROM tmdb_data WHERE tmdb_id = ?", (tmdb_id,)
    ).fetchone()
    if row:
        title = row[0]
        release_date = row[1] or ""
        year = int(release_date[:4]) if len(release_date) >= 4 else None
    
    db.execute(
        """INSERT INTO filter_log (tmdb_id, title, year, stage, reason, details) 
           VALUES (?, ?, ?, ?, ?, ?)""",
        (tmdb_id, title, year, stage, reason, details)
    )
    db.execute(
        """UPDATE movie_progress 
           SET status = 'filtered_out', updated_at = CURRENT_TIMESTAMP 
           WHERE tmdb_id = ?""",
        (tmdb_id,)
    )
```

---

## Orchestration

### Main loop

The stage processes all `pending` movies in chunks for progress reporting and periodic database commits:

```python
async def stage2_tmdb_fetch(db: sqlite3.Connection):
    """Main Stage 2 entry point. Fetches TMDB details for all pending movies."""
    
    rows = db.execute(
        "SELECT tmdb_id FROM movie_progress WHERE status = 'pending'"
    ).fetchall()
    
    print(f"Stage 2: {len(rows):,} movies to fetch from TMDB")
    
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        
        CHUNK_SIZE = 500
        
        for i in range(0, len(rows), CHUNK_SIZE):
            chunk = rows[i:i + CHUNK_SIZE]
            
            tasks = [
                process_single_movie(session, tmdb_id, db)
                for (tmdb_id,) in chunk
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
            
            db.commit()
            
            if (i // CHUNK_SIZE) % 20 == 0:
                print(f"  Progress: {min(i + CHUNK_SIZE, len(rows)):,}/{len(rows):,} "
                      f"| {tmdb_limiter.stats()}")
    
    # Final summary
    counts = db.execute("""
        SELECT status, COUNT(*) FROM movie_progress 
        WHERE status IN ('tmdb_fetched', 'filtered_out')
        GROUP BY status
    """).fetchall()
    print(f"Stage 2 complete: {dict(counts)}")
```

### Commit strategy

We commit after each chunk of 500 movies. This means:

- If the process crashes, we lose at most ~500 movies of progress (re-fetched on resume, costing ~14 seconds of TMDB API time at 36 req/s).
- We don't hold a massive transaction open for hours, which would cause the SQLite WAL to grow unboundedly.
- Progress is visible in real-time if you query the database from another process.

### aiohttp connector settings

| Setting | Value | Why |
|---------|-------|-----|
| `limit` | 100 | Max total TCP connections. Well above the rate limiter ceiling, so it never bottlenecks throughput. Safety net only. |
| `limit_per_host` | 0 (unlimited) | All connections go to the same TMDB host. Setting a per-host limit would be redundant with the global limit. |

---

## Resumability

If the process stops for any reason (crash, Ctrl+C, laptop sleep), restarting picks up exactly where it left off. The resume logic is simple:

1. Query `SELECT tmdb_id FROM movie_progress WHERE status = 'pending'`
2. Process only those movies.

Movies that were already fetched have `status = 'tmdb_fetched'` and are skipped. Movies that were filtered out have `status = 'filtered_out'` and are skipped. The rate limiter reinitializes at 36 req/s and re-adapts within the first few minutes.

There is no need to detect or handle partial writes. The `INSERT OR REPLACE` on `tmdb_data` is idempotent — if a movie's data was written but the status wasn't updated (crash between the two statements in a committed chunk), the re-fetch simply overwrites the existing row with identical data.

---

## How Stage 3 Uses This Data

Stage 3 (the quality funnel) reads directly from `tmdb_data` to score and rank movies. Here's a summary of how each stored field feeds into the funnel:

### Hard filters (disqualify immediately)

These checks run first. A movie failing any hard filter is logged and set to `filtered_out` regardless of its quality score:

| Filter | Logic | Field used |
|--------|-------|------------|
| Missing IMDB ID | `imdb_id IS NULL` | `imdb_id` |
| Missing title | `title IS NULL OR title = ''` | `title` |
| Missing overview | `overview_length < 20` | `overview_length` |
| No genres | `genre_count = 0` | `genre_count` |
| Too short | `duration IS NOT NULL AND duration > 0 AND duration < 40` | `duration` |
| No runtime | `duration IS NULL OR duration = 0` | `duration` |
| Not yet released | `release_date > today or release_date IS NULL` | `release_date` |

### Quality scoring (rank survivors)

Movies passing hard filters receive a composite score (0–100 scale) built from four components:

| Component | Range | Weight | Primary field | Formula |
|-----------|-------|--------|---------------|---------|
| Engagement | 0–50 | Dominant | `vote_count` | `min(log1p(vote_count) * 5.0, 50.0)` |
| Popularity | 0–25 | Secondary | `popularity` | `min(log1p(popularity) * 4.0, 25.0)` |
| Rating quality | 0–15 | Conditional | `vote_average`, `vote_count` | `(vote_average / 10) * 15 * min(vote_count / 10, 1.0)` |
| Data completeness | 0–10 | Tiebreaker | `poster_url`, `has_revenue`, `has_budget`, `overview_length`, `genre_count`, `has_production_companies`, `has_production_countries`, `has_keywords`, `has_cast_and_crew` | Sum of boolean/scaled bonuses |

The hierarchy is deliberate: a movie with 500 votes and a 5.0 average is a better candidate than a movie with 2 votes and a 9.0 average, because real engagement data means IMDB will have actual reviews, ratings, and parental guide data to scrape — producing much better downstream LLM and vector outputs.

After scoring, Stage 3 sorts descending and selects the top 100,000. The score of the 100,000th movie becomes the `cutoff_score`, which is saved to disk for the daily update pipeline to use as a minimum bar for new movie admission.

---

## Relationship to the Daily Update Pipeline

After the initial bulk ingest, a daily cron job on the EC2 instance discovers recently-released movies via the TMDB changes endpoint. These candidates go through the same Stage 2 fetch process (hitting the same TMDB API endpoint, writing to the same `tmdb_data` table schema — though in the EC2 context, this would be a Postgres table rather than local SQLite).

The daily pipeline uses the saved `cutoff_score` from the initial funnel as its quality threshold: any new movie scoring above the cutoff gets the full IMDB → LLM → embed → ingest treatment. Movies below the cutoff are parked as `below_quality_cutoff` and re-evaluated monthly.

This design means the Stage 2 fetch logic is reusable across both the bulk ingest (local laptop, 950K movies) and the daily update (EC2, typically <100 movies/day).

---

## Pre-Flight Checklist

Before running Stage 2, verify the following:

1. **TMDB API token is set.** The v4 bearer token from your TMDB account settings. Test with a single request to confirm it returns 200.

2. **Rate limit tier is identified.** Run a quick burst test (50 rapid requests) and observe where 429s begin. This determines whether you're looking at ~8 hours or ~3 days of wall time.

3. **Stage 1 is complete.** The `movie_progress` table should contain ~950K rows with `status = 'pending'`. Verify with:
   ```sql
   SELECT COUNT(*) FROM movie_progress WHERE status = 'pending';
   ```

4. **`tmdb_data` table is created.** Run the CREATE TABLE statement from the schema section above. This is idempotent if the table already exists.

5. **Disk space is available.** The database will grow to ~310–360 MB. Verify you have at least 1 GB of free space on the volume containing `./ingestion_data/`.

6. **Network is stable.** This stage runs for hours. If running on a laptop, ensure sleep/hibernate is disabled during the run. If running over WiFi, a wired connection is preferred for reliability (though not required — the resumability design handles intermittent drops).

---

## Monitoring During the Run

The stage emits progress logs every 10,000 movies (20 chunks × 500 movies/chunk). Each log line includes:

- Movies processed so far vs. total
- Current adaptive rate (req/s)
- Total requests fired
- Total 429s received and their percentage

A healthy run looks like:

```
Stage 2: 948,217 movies to fetch from TMDB
  Progress: 10,000/948,217 | Rate: 37.8 req/s | Total: 10,000 | 429s: 3 (0.03%)
  Progress: 20,000/948,217 | Rate: 38.9 req/s | Total: 20,012 | 429s: 5 (0.02%)
  ...
```

Warning signs:

- 429 rate above 1% → the rate limiter should be converging down, but if it's persistently high, TMDB may have tightened limits. Monitor for a few minutes; the adaptive algorithm should self-correct.
- 429 rate above 5% → something is wrong. Kill the process, wait 10 minutes, and restart (resumability handles this cleanly).
- Many `filtered_out` entries with `reason='tmdb_fetch_error'` → TMDB may be experiencing an outage. Check status.themoviedb.org.

---

## Summary of Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Storage format | SQLite (not flat JSON files) | 10x smaller footprint, atomic queries for Stage 3, transactional consistency with checkpoint tracker |
| Data stored from TMDB | 7 persistent fields + 11 filter-only fields | Persistent fields are needed downstream; filter fields enable quality funnel without re-fetching |
| Watch provider encoding | Packed int32 BLOB | Compact, avoids JSON parse overhead, directly compatible with the integer-key format used in Qdrant and Postgres |
| Overview storage | Length only (not full text) | Full text comes from IMDB; TMDB overview is only needed for the ≥20 char hard filter check |
| Genre storage | Count only (not IDs/names) | Genre IDs for the production database are derived from a normalized mapping at Stage 8; the count is sufficient for the hard filter and completeness scoring |
| Credits, keywords, production companies | Not stored | Higher-quality versions come from IMDB scraping; storing TMDB versions creates redundancy |
| Rate limiting | Adaptive token bucket starting at 90% of stated limit | Converges on real limit automatically; starting conservative costs minutes, not hours |
| Commit granularity | Every 500 movies | Bounds data loss on crash to ~14 seconds of re-fetch time; prevents WAL bloat |
| Pre-filtering by TMDB popularity | Not used | Popularity is volatile and narrow; the composite quality score in Stage 3 uses 6 dimensions and makes a better selection. Pre-filtering risks excluding classics with low recent activity but high engagement. |