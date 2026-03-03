# Movie Ingestion Pipeline — Final Architecture

## Executive Summary

This document defines the end-to-end pipeline for ingesting ~100K movies into Postgres, Qdrant, and Redis. The TMDB bulk export contains ~1M entries; the pipeline fetches TMDB details for all of them (free, just rate-limited), applies a quality-scoring funnel to select the top ~100K, then runs the expensive stages (IMDB scraping, LLM generation, embedding) only on those. The pipeline runs on your local laptop, targeting remote database instances.

It is designed around four core principles:

1. **Cost efficiency** — TMDB-first funnel avoids ~$2,500+ in unnecessary proxy spend; OpenAI Batch API saves 50% on LLM generation
2. **Resilience** — SQLite checkpoint tracker with per-movie status, enabling stop/resume at any point
3. **Throughput** — Aggressive parallelism at every stage, with rate limiting adapted to each external service
4. **Continuity** — Daily update pipeline keeps the catalog current using TMDB's changes endpoint

**Expected timeline:** ~3–4 calendar days (dominated by two ~24hr Batch API waits + TMDB fetching for ~950K movies), with roughly 12–20 hours of active compute time spread across the gaps.

**Expected LLM cost:** ~$250–350 total (after 50% batch discount), depending on actual token volumes.

---

## Pipeline Stages Overview

```
Stage 1: TMDB Export Download & Initial Filtering          (~2 min)        ~1M → ~950K
Stage 2: TMDB Detail Fetching (rate-limited, adaptive)     (~1–8 hrs)      ~950K movies
Stage 3: TMDB Quality Funnel                               (~5 min)        ~950K → ~100K
Stage 4: IMDB Scraping (parallel, proxied)                 (~4–8 hrs)      ~100K movies
Stage 5: LLM Phase 1 — Batch API submission                (~24 hrs wait)   ~100K movies
Stage 6: LLM Phase 2 — Batch API submission                (~24 hrs wait)   ~100K movies
Stage 7: Embedding (real-time batched calls)                (~20–40 min)    ~100K movies
Stage 8: Database Ingestion (Postgres + Qdrant bulk)        (~1–2 hrs)      ~100K movies
```

**The critical cost insight:** TMDB fetching is free (just rate-limited, no proxies needed). IMDB scraping costs ~$3–4 per thousand movies in proxy fees. By doing TMDB fetching for all ~950K movies first and filtering down to ~100K *before* touching IMDB, we avoid ~$2,500–3,000 in unnecessary proxy spend. The TMDB stage adds a few hours of wall time but saves an order of magnitude in cost.

Stages 5 and 6 are sequential because Phase 2 LLM generations depend on Phase 1 outputs. Stage 7 runs in real-time (not batched) because embedding costs are negligible and the extra 24hr wait isn't worth the ~$4 savings.

---

## Why Two Batch API Calls (Not One)

You noted that Phase 2 generations (`plot_analysis`, `viewer_experience`, `narrative_techniques`, `production_metadata`) depend on Phase 1 outputs (`plot_events_metadata` specifically). This creates an unavoidable data dependency:

```
Phase 1 (independent):  plot_events, watch_context, reception
                              │
                              ▼
Phase 2 (depends on Phase 1): plot_analysis, viewer_experience, 
                               narrative_techniques, production_metadata
```

A single batch call would require all prompts to be constructed up front, but Phase 2 prompts can't be built until Phase 1 results exist. Two batch submissions is the minimum.

**The cost math still works well:**
- 2 batches at 50% discount = 50% savings on all LLM spend
- Compared to real-time: saves ~$250–350 on a ~$500–700 total LLM bill
- The tradeoff is ~48 hours of wall-clock wait (during which you do nothing)

**If you later find a way to remove the Phase 1→2 dependency** (e.g., by passing raw source text into Phase 2 prompts instead of Phase 1 outputs), you could collapse to a single batch. Worth considering as an optimization, but expect some quality degradation on Phase 2 outputs.

---

## Checkpoint & Resume System

The backbone of resilience. A local SQLite database tracks every movie's progress through the pipeline. If the process crashes at any point, you restart and it picks up where it left off.

### Schema

```sql
-- Main progress tracker
CREATE TABLE movie_progress (
    tmdb_id       INTEGER PRIMARY KEY,
    imdb_id       TEXT,
    status        TEXT NOT NULL DEFAULT 'pending',
    -- Statuses (in order):
    --   pending → tmdb_fetched → quality_passed → imdb_scraped → 
    --   phase1_complete → phase2_complete →
    --   embedded → ingested
    --
    -- Terminal statuses:
    --   filtered_out        (missing essential data, scrape failures, etc.)
    --   below_quality_cutoff (didn't make the top ~100K in the TMDB quality funnel)
    quality_score  REAL,     -- Composite TMDB quality score (for funnel ranking)
    batch1_custom_id TEXT,   -- OpenAI batch custom_id for phase 1 (for result matching)
    batch2_custom_id TEXT,   -- OpenAI batch custom_id for phase 2
    updated_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Every movie that gets dropped, with why
-- title and year come from TMDB and are populated for any movie that made it
-- past Stage 1 (the initial adult/video filter). For movies filtered at
-- Stage 1, these will be NULL since we only have the tmdb_id at that point.
CREATE TABLE filter_log (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    tmdb_id    INTEGER NOT NULL,
    title      TEXT,             -- From TMDB, NULL if filtered before Stage 2
    year       INTEGER,          -- Release year from TMDB, NULL if filtered before Stage 2
    stage      TEXT NOT NULL,
    -- Stages: 'tmdb_export_filter', 'tmdb_fetch', 'tmdb_quality_funnel',
    --         'imdb_scrape', 'essential_data_check', 'llm_phase1', 
    --         'llm_phase2', 'embedding', 'ingestion'
    reason     TEXT NOT NULL,
    details    TEXT,          -- Optional JSON blob with extra context
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_progress_status ON movie_progress(status);
CREATE INDEX idx_filter_log_stage ON filter_log(stage);
CREATE INDEX idx_filter_log_tmdb ON filter_log(tmdb_id);
```

### `log_filter` Helper

Every call to filter out a movie goes through this helper. It automatically attaches title and year from the TMDB data file if it exists (i.e., the movie made it past Stage 1):

```python
def log_filter(db, tmdb_id: int, stage: str, reason: str, details: str = None):
    """Log a filtered movie with title/year for human-readable sanity checking."""
    title = None
    year = None
    
    # Pull title/year from TMDB data if available (exists for anything past Stage 1)
    tmdb_path = f"./ingestion_data/tmdb/{tmdb_id}.json"
    if os.path.exists(tmdb_path):
        tmdb_data = load_json(tmdb_path)
        title = tmdb_data.get("title")
        release_date = tmdb_data.get("release_date", "")
        year = int(release_date[:4]) if release_date and len(release_date) >= 4 else None
    
    db.execute(
        """INSERT INTO filter_log (tmdb_id, title, year, stage, reason, details) 
           VALUES (?, ?, ?, ?, ?, ?)""",
        (tmdb_id, title, year, stage, reason, details)
    )
    db.execute(
        "UPDATE movie_progress SET status = 'filtered_out', updated_at = CURRENT_TIMESTAMP WHERE tmdb_id = ?",
        (tmdb_id,)
    )
```

### Data Storage Convention

Each stage writes its output to a local directory structure. The checkpoint DB tracks status; the actual data lives on disk as JSON files:

```
./ingestion_data/
├── tmdb/                    # Stage 2 outputs (~950K files, ~3-5 GB)
│   ├── 9377.json            # One file per tmdb_id
│   ├── 12345.json
│   └── ...
├── imdb/                    # Stage 4 outputs (~100K files)
│   ├── 9377.json
│   └── ...
├── llm_phase1/              # Stage 5 outputs
│   ├── 9377.json
│   └── ...
├── llm_phase2/              # Stage 6 outputs
│   ├── 9377.json
│   └── ...
├── batch_files/             # Batch API .jsonl files (input + output)
│   ├── phase1_input.jsonl
│   ├── phase1_output.jsonl
│   ├── phase2_input.jsonl
│   └── phase2_output.jsonl
├── embeddings/              # Stage 7 outputs (binary packed vectors)
│   ├── 9377.bin
│   └── ...
└── tracker.db               # SQLite checkpoint database
```

**Why flat JSON files per movie (not one giant file)?**
- Atomic writes: a crash mid-write only corrupts one movie's file
- Easy to inspect/debug individual movies
- No memory pressure from loading 100K movies into RAM at once
- Trivial to resume: if the file exists and status is correct, skip it

---

## Stage 1: TMDB Export Download & Initial Filtering

**Goal:** Get the list of all non-adult, non-video movie IDs from TMDB's daily export.

**Input:** TMDB daily export URL (e.g., `https://files.tmdb.org/p/exports/movie_ids_MM_DD_YYYY.json.gz`)

**Process:**
1. Download and decompress the .gz file (stream-decompress to avoid loading entirely into memory)
2. Parse each line as JSON
3. Filter: keep entries where `adult == false` AND `video == false` AND `popularity > 0`
4. Insert all passing `tmdb_id` values into `movie_progress` with `status = 'pending'`
5. Log all filtered-out entries to `filter_log` with `stage = 'tmdb_export_filter'` and `reason = 'adult'` or `reason = 'video'` or `reason = 'popularity'`

**Rate limits:** None (single file download).

**Duration:** ~1–2 minutes.

**Resumability:** Idempotent. Re-running just does an `INSERT OR IGNORE` into `movie_progress`.

---

## Stage 2: TMDB Detail Fetching

**Goal:** For every `pending` movie, fetch expanded TMDB details and extract core fields.

**Input:** All `movie_progress` rows where `status = 'pending'`

**API endpoint:** `GET https://api.themoviedb.org/3/movie/{tmdb_id}?append_to_response=release_dates,keywords,watch/providers,credits`

### Why TMDB Needs a Rate Limiter (Not a Semaphore)

TMDB rate limits are **token-based (keyed to your API key)**, not IP-based. Every request carries your API key in the `Authorization` header, and TMDB tracks your request count against that key regardless of what IP it originates from. This means:

- **Proxies are useless here.** Unlike IMDB (where rotating IPs avoids per-IP bans), TMDB sees through proxies because the identifying factor is your API key, not your IP.
- **A concurrency semaphore is the wrong tool.** A semaphore limits how many requests are *in flight simultaneously*, but says nothing about how many are *initiated per second*. If TMDB responds in 50ms on average, a `Semaphore(35)` would allow 35 requests to complete and re-fire in 50ms — that’s 700 requests/second, blowing past a 40/s limit instantly.

What you need is a **rate limiter**: a mechanism that enforces "no more than N requests per second," regardless of how fast responses come back. The standard algorithm for this is a **token bucket**.

### Rate Limiter vs. Concurrency Semaphore — When to Use Which

| Scenario | Tool | Why |
|----------|------|-----|
| **TMDB** (API key rate limit: N req/s) | Token bucket rate limiter | The constraint is *requests per unit of time* against a single identity. You need to meter the rate of outgoing requests. |
| **IMDB** (IP-based blocking, proxied) | Concurrency semaphore | The constraint is *total concurrent connections* through the proxy gateway. The proxy provider handles per-IP distribution; you just need to not overwhelm the gateway with too many simultaneous TCP connections. |
| **OpenAI embedding** (RPM + TPM limits) | Rate limiter + concurrency semaphore | Both constraints apply: requests per minute AND concurrent connections. Use a rate limiter to stay under RPM and a semaphore to cap in-flight requests. |

### Maximizing Throughput Without Hitting 429s

The goal is to stay as close to the rate limit ceiling as possible without crossing it. The challenge is that "40 requests per second" isn't a precise sliding window on TMDB's side — it's an approximation, and their actual enforcement might use a different windowing strategy (fixed window, sliding window, token bucket) that you can't observe directly.

**Strategy: target 90–95% of the stated limit, then adapt based on 429 signals.**

1. **Start at 36 req/s** (90% of the stated 40/s limit). This gives a comfortable margin for clock drift, network jitter, and any imprecision in TMDB's enforcement window.

2. **Monitor 429 responses.** If you go 5 minutes without a single 429, you're probably leaving headroom on the table. If you're getting 429s on more than 1% of requests, you're too aggressive.

3. **Adaptive adjustment.** When a 429 arrives, immediately reduce the rate by 10% and respect the `Retry-After` header. After 2 minutes of clean responses, nudge the rate back up by 5%. This creates a natural oscillation that hugs the real limit:

```
36/s → (no 429s for 2 min) → 37.8/s → (no 429s for 2 min) → 39.7/s → (429!) → 35.7/s → ...
```

**Important: verify your actual tier first.** Some TMDB API tiers enforce 40 requests per **10 seconds** (= 4/s), not 40/s. Check your API key's tier before running the pipeline. If you're on the slower tier, request an upgrade from TMDB for bulk use, or accept a ~7-hour Stage 2 runtime. Since the quality funnel (Stage 3) needs all TMDB data before it can rank movies, this adds real wall time to the pipeline.

### Implementation

```python
import asyncio
import time

class AdaptiveRateLimiter:
    """
    Token-bucket rate limiter with adaptive rate adjustment.
    
    How it works:
    - Maintains a bucket of "tokens." Each request consumes one token.
    - Tokens refill at `current_rate` per second, up to `burst` max.
    - If a caller tries to acquire when the bucket is empty, they sleep
      until a token becomes available.
    - On 429 signals: rate drops by 10%.
    - After clean_window seconds with no 429s: rate increases by 5%,
      up to max_rate.
    
    This creates a feedback loop that converges on the real server limit.
    """
    
    def __init__(self, initial_rate: float, max_rate: float, burst: int = 5):
        self.current_rate = initial_rate   # tokens per second
        self.max_rate = max_rate           # never exceed this
        self.burst = burst                 # max tokens available at once
        self.tokens = burst
        self.last_refill = time.monotonic()
        self.last_429_time = 0.0
        self.clean_window = 120            # seconds without 429 before rate increase
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
            
            # Check if we should increase rate (no 429s in clean_window)
            if (now - self.last_429_time > self.clean_window 
                    and self.last_429_time > 0
                    and self.current_rate < self.max_rate):
                old_rate = self.current_rate
                self.current_rate = min(self.max_rate, self.current_rate * 1.05)
                if self.current_rate != old_rate:
                    print(f"  Rate limiter: increased to {self.current_rate:.1f} req/s")
            
            self.total_requests += 1

    def report_429(self):
        """Call this when a 429 response is received. Reduces rate by 10%."""
        self.last_429_time = time.monotonic()
        self.total_429s += 1
        old_rate = self.current_rate
        self.current_rate = max(1.0, self.current_rate * 0.90)  # Floor at 1 req/s
        print(f"  Rate limiter: 429 received, reduced {old_rate:.1f} → {self.current_rate:.1f} req/s "
              f"(total 429s: {self.total_429s}/{self.total_requests})")

    def stats(self) -> str:
        pct = (self.total_429s / max(1, self.total_requests)) * 100
        return (f"Rate: {self.current_rate:.1f} req/s | "
                f"Total: {self.total_requests:,} | "
                f"429s: {self.total_429s:,} ({pct:.2f}%)")


# Initialize: start at 36 req/s, allow up to 40, burst of 5
tmdb_limiter = AdaptiveRateLimiter(initial_rate=36, max_rate=40, burst=5)

async def fetch_tmdb_details(
    session: aiohttp.ClientSession, 
    tmdb_id: int,
    max_retries: int = 3
) -> dict:
    """Fetch TMDB details for a single movie, respecting rate limits."""
    
    for attempt in range(max_retries):
        await tmdb_limiter.acquire()  # Blocks until a rate token is available
        
        try:
            async with session.get(
                f"https://api.themoviedb.org/3/movie/{tmdb_id}",
                params={"append_to_response": "release_dates,keywords,watch/providers,credits"},
                headers={"Authorization": f"Bearer {TMDB_API_TOKEN}"},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                
                elif resp.status == 429:
                    # Signal the rate limiter to back off
                    tmdb_limiter.report_429()
                    retry_after = int(resp.headers.get("Retry-After", 2))
                    await asyncio.sleep(retry_after)
                    continue  # Retry (acquire will enforce the new lower rate)
                
                elif resp.status == 404:
                    return None  # Caller handles as filter
                
                else:
                    raise FetchError(f"HTTP {resp.status}")
                    
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            if attempt == max_retries - 1:
                raise FetchError(f"Failed after {max_retries} attempts: {e}")
            await asyncio.sleep(1.0 * (attempt + 1))
    
    raise FetchError(f"Exhausted retries for tmdb_id={tmdb_id}")
```

**Why the adaptive approach works well for this use case:**

- **You can't just hardcode 40/s.** TMDB's stated limit is approximate. Their server-side enforcement window might not align perfectly with your client-side clock. Network latency means requests arrive at TMDB slightly clustered even if you space them evenly on your end.
- **Starting conservative (36/s) is nearly free.** At 36/s vs 40/s, 100K requests takes 46 minutes instead of 42 minutes. That 4-minute difference is meaningless in a multi-day pipeline. But the cost of being too aggressive (bursts of 429s, forced backoffs, potential temporary bans) is much higher.
- **The feedback loop converges quickly.** After the first few minutes, the adaptive rate finds the real ceiling and hovers just below it. You get ~95%+ of maximum throughput without manual tuning.
- **429s aren't failures — they're information.** A well-designed rate limiter treats occasional 429s as a natural part of the calibration process, not as errors. The goal isn't zero 429s; it's a 429 rate below ~0.5%.

### Concurrency Note

The rate limiter handles *when* requests fire. You still want a concurrency semaphore to limit *how many TCP connections are open simultaneously* — this protects your local machine and `aiohttp`'s connection pool:

```python
# Rate limiter: controls requests per second (the important constraint)
tmdb_limiter = AdaptiveRateLimiter(initial_rate=36, max_rate=40, burst=5)

# Concurrency semaphore: caps open TCP connections (local resource protection)
# Set higher than the rate — this shouldn't be the bottleneck
tmdb_semaphore = asyncio.Semaphore(100)

async def fetch_tmdb_details_with_both(session, tmdb_id):
    async with tmdb_semaphore:           # Limit concurrent connections
        await tmdb_limiter.acquire()      # Limit requests per second
        async with session.get(...) as resp:
            ...
```

The semaphore here is set to 100 (well above 36–40 req/s) so it almost never blocks — it's purely a safety net for local resource exhaustion, not a throughput control.

### Duration estimate

| TMDB tier | Effective rate | Time for 100K movies | Notes |
|-----------|---------------|---------------------|-------|
| 40 req/s | ~36–39 req/s (adaptive) | ~45–60 minutes | Standard tier, most likely |
| 4 req/s (40 per 10s) | ~3.6 req/s (adaptive) | ~7–8 hours | Slower tier, verify first |

Either way, Stage 2 runs unattended. At the slower tier, the quality funnel (Stage 3) can’t begin until all TMDB data has been fetched — it needs the full set to rank and select the top 100K — so the 7–8 hours is real wall time added to the pipeline. This is the one scenario where requesting a higher TMDB rate limit from their team is worth doing.

**Before running: verify your tier.** Make 50 rapid requests in a test script and see where 429s start. If they start at ~40/s, you're on the standard tier. If they start at ~4/s, you're on the slower tier.

### Per-movie process

1. Fetch TMDB details (through the rate limiter)
2. Extract: `imdb_id`, `title`, `original_title`, `release_date`, `runtime` (duration), `poster_path` (→ full poster URL), `overview`, `genres`, `keywords`, `credits` (cast/crew), `watch/providers` (US region), `release_dates` (for maturity rating), `production_companies`, `budget`, `revenue`, `spoken_languages`, `origin_country`
3. Save extracted data to `./ingestion_data/tmdb/{tmdb_id}.json` — always save first, even for movies about to be filtered, so `log_filter` can read title/year from disk
4. **Filter check:** If `imdb_id` is missing/null → call `log_filter` (which reads title/year from the file saved in step 3) with `stage='tmdb_fetch', reason='missing_imdb_id'` and set `status='filtered_out'`
5. Update `movie_progress`: set `imdb_id` and `status = 'tmdb_fetched'`

### Error handling

- HTTP 404 → movie doesn't exist on TMDB. Save whatever data was returned (for title/year), then log to `filter_log` with `reason='tmdb_404'`, set `status='filtered_out'`
- HTTP 429 → handled by the adaptive rate limiter (backs off automatically, retries with lower rate)
- HTTP 5xx → retry up to 3 times with exponential backoff, then log with `reason='tmdb_fetch_error'` and set `status='filtered_out'`
- Network timeout → same as 5xx

### Resumability

On restart, query `SELECT tmdb_id FROM movie_progress WHERE status = 'pending'` and process only those. The rate limiter reinitializes at 36 req/s and re-adapts within the first few minutes.

---

## Stage 3: TMDB Quality Funnel

**Goal:** Reduce the ~950K `tmdb_fetched` movies down to ~100K worth scraping on IMDB.

**Input:** All `movie_progress` rows where `status = 'tmdb_fetched'`

### Why This Stage Exists

IMDB scraping is the single most expensive step in the pipeline — not in compute, but in proxy costs. Each movie requires 6 page fetches through residential proxies at ~$8/GB. Scraping 950K movies would cost ~$2,800–3,800 in proxy fees alone; scraping 100K costs ~$300–400. This stage is the 10x cost lever.

TMDB data is free to fetch (just rate-limited). And TMDB provides several strong signals that can distinguish "real, notable movies that users might search for" from "obscure entries with minimal data that would produce garbage vectors anyway."

### Hard Filters (Disqualify Immediately)

These movies cannot produce useful search results regardless of popularity:

```python
HARD_FILTERS = {
    "missing_imdb_id":    lambda m: not m.get("imdb_id"),
    "missing_title":      lambda m: not m.get("title"),
    "missing_overview":   lambda m: not m.get("overview") or len(m["overview"]) < 20,
    "missing_genres":     lambda m: not m.get("genres") or len(m["genres"]) == 0,
    "too_short":          lambda m: m.get("runtime") is not None and 0 < m["runtime"] < 40,
    "no_runtime":         lambda m: m.get("runtime") is None or m["runtime"] == 0,
    "not_yet_released":   lambda m: is_future_release(m.get("release_date")),
}
```

Movies failing hard filters get logged to `filter_log` with `stage='tmdb_quality_funnel'` and the specific reason, then set to `status='filtered_out'`.

**Unreleased movies are filtered out, not tracked.** There's no need to park them in a special status and re-check them. When an unreleased movie eventually releases, it accumulates TMDB activity (votes, reviews, streaming provider additions) that surfaces it in the TMDB changes endpoint. The daily pipeline catches it then, re-fetches its TMDB data, sees that it's now released, and evaluates it fresh. This avoids maintaining a growing list of unreleased movies to poll.

### Quality Scoring (Rank the Rest)

After hard filters, you'll have somewhere in the range of 200K–500K movies. Now you need to rank them and take the top ~100K. The ranking uses a composite score built from TMDB signals:

```python
import math

def compute_quality_score(movie: dict) -> float:
    """
    Composite quality score from TMDB data.
    
    Design principles:
    - vote_count is the strongest signal (indicates real human engagement)
    - popularity is a useful secondary signal (TMDB's own metric, based on 
      page views, votes, watchlist adds, etc.)
    - vote_average matters but only when vote_count is meaningful
    - Each component is log-scaled to prevent outliers from dominating
    """
    
    vote_count = movie.get("vote_count", 0)
    popularity = movie.get("popularity", 0)
    vote_average = movie.get("vote_average", 0)
    has_poster = 1.0 if movie.get("poster_path") else 0.0
    has_revenue = 1.0 if movie.get("revenue") and movie["revenue"] > 0 else 0.0
    overview_len = len(movie.get("overview", ""))
    
    # --- Component 1: Engagement (0-50 points) ---
    # vote_count is the single best proxy for "is this a real movie people watch?"
    # log-scaled: 1 vote = 0, 10 votes = 11.5, 100 = 23, 1000 = 34.5, 10000 = 46
    engagement = math.log1p(vote_count) * 5.0
    engagement = min(engagement, 50.0)
    
    # --- Component 2: Popularity (0-25 points) ---
    # TMDB popularity scores range from ~0 to ~1000+ for massive blockbusters.
    # Most movies cluster between 0-20. Log scale compresses the range.
    pop_score = math.log1p(popularity) * 4.0
    pop_score = min(pop_score, 25.0)
    
    # --- Component 3: Rating quality (0-15 points) ---
    # Only meaningful when vote_count is high enough to be reliable.
    # Below 10 votes, the average is noise — weight it down.
    rating_confidence = min(vote_count / 10.0, 1.0)
    rating_score = (vote_average / 10.0) * 15.0 * rating_confidence
    
    # --- Component 4: Data completeness bonus (0-10 points) ---
    # Movies with more metadata tend to be more notable and will produce
    # better vectors. This is a gentle nudge, not a hard requirement.
    completeness = 0.0
    completeness += has_poster * 3.0          # Has a poster image
    completeness += has_revenue * 2.0          # Has box office data
    completeness += min(overview_len / 200.0, 1.0) * 3.0  # Longer overview = more context
    completeness += min(len(movie.get("genres", [])) / 3.0, 1.0) * 2.0
    
    return engagement + pop_score + rating_score + completeness
```

**Why these weights?**

The hierarchy is deliberate: **engagement >> popularity >> rating >> completeness**. A movie with 500 votes and a 5.0 average is a better candidate than a movie with 2 votes and a 9.0 average, because the first movie has real audience data to build vectors from, while the second might be a data entry artifact with no IMDB reviews, no plot synopsis, and no parental guide — meaning LLM generation would produce low-quality results anyway.

### The Cutoff Decision

```python
def apply_quality_funnel(db, target_count: int = 100_000):
    """Rank all tmdb_fetched movies and promote the top N to quality_passed."""
    
    rows = db.execute(
        "SELECT tmdb_id FROM movie_progress WHERE status = 'tmdb_fetched'"
    ).fetchall()
    
    scored_movies = []
    for (tmdb_id,) in rows:
        tmdb_data = load_json(f"./ingestion_data/tmdb/{tmdb_id}.json")
        
        # Apply hard filters first
        filter_reason = check_hard_filters(tmdb_data)
        if filter_reason:
            log_filter(db, tmdb_id, "tmdb_quality_funnel", filter_reason)
            continue
        
        score = compute_quality_score(tmdb_data)
        scored_movies.append((tmdb_id, score))
    
    # Sort by score descending, take top N
    scored_movies.sort(key=lambda x: x[1], reverse=True)
    
    cutoff_idx = min(target_count, len(scored_movies))
    cutoff_score = scored_movies[cutoff_idx - 1][1] if cutoff_idx > 0 else 0
    
    promoted = scored_movies[:cutoff_idx]
    below_cutoff = scored_movies[cutoff_idx:]
    
    # Promote winners
    for tmdb_id, score in promoted:
        db.execute(
            """UPDATE movie_progress 
               SET status = 'quality_passed', quality_score = ?, updated_at = CURRENT_TIMESTAMP 
               WHERE tmdb_id = ?""",
            (score, tmdb_id)
        )
    
    # Log movies below the cutoff (these are valid movies, just not top-100K)
    for tmdb_id, score in below_cutoff:
        db.execute(
            """UPDATE movie_progress 
               SET status = 'below_quality_cutoff', quality_score = ?, updated_at = CURRENT_TIMESTAMP 
               WHERE tmdb_id = ?""",
            (score, tmdb_id)
        )
        log_filter(db, tmdb_id, "tmdb_quality_funnel", "below_cutoff",
                   details=json.dumps({"score": round(score, 2), "cutoff": round(cutoff_score, 2)}))
    
    db.commit()
    
    print(f"  Quality funnel results:")
    print(f"    Hard-filtered:      {len(rows) - len(scored_movies):,}")
    print(f"    Promoted (top {target_count:,}): {len(promoted):,}")
    print(f"    Below cutoff:       {len(below_cutoff):,}")
    print(f"    Cutoff score:       {cutoff_score:.2f}")
    
    # Save the cutoff score for the daily pipeline to reference
    save_json("./ingestion_data/quality_cutoff.json", {
        "cutoff_score": cutoff_score,
        "target_count": target_count,
        "total_scored": len(scored_movies),
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return cutoff_score
```

### What the Cutoff Score Means for the Daily Pipeline

The `quality_cutoff.json` file records the score threshold of the 100,000th-ranked movie. The daily update pipeline uses this as a minimum bar: any new movie scoring above this threshold gets the full IMDB → LLM → embed → ingest treatment. This keeps the catalog growing with quality entries without re-running the full ranking.

Over time, as new high-profile movies are added, the effective quality bar drifts slightly. This is fine — if anything, it means your catalog gets incrementally better. If you ever want to tighten the bar, re-run the quality funnel on the full `tmdb_fetched` set.

### Storage Impact

Keeping TMDB data for all ~950K movies on disk costs roughly:
- ~950K files × ~3–5KB each (shaped TMDB detail JSON) = ~3–5 GB
- This is well within laptop storage limits
- The files are useful long-term: re-running the funnel with different criteria or a different target count doesn't require re-fetching from TMDB

### Resumability

On restart, the funnel re-scores only movies still in `tmdb_fetched` status. Movies already promoted or filtered retain their status.

---

## Stage 4: IMDB Scraping (with Proxies)

**Goal:** For every `quality_passed` movie, scrape 6 IMDB pages and extract structured data.

**Input:** All `movie_progress` rows where `status = 'quality_passed'`

### Proxy Education

IMDB has aggressive anti-scraping measures. At 100K movies × 6 pages = 600K requests, you'll get IP-banned within minutes without proxies. Here's what you need to know:

#### What proxies are and why you need them

A proxy is an intermediary server that makes HTTP requests on your behalf. When you route your request through a proxy, IMDB sees the proxy's IP address instead of yours. By rotating through thousands of different proxy IPs, you make it appear as if thousands of different users are each making a small number of requests — which avoids triggering rate limits or IP bans.

#### Types of proxies

| Type | What it is | Cost | Detection risk | Best for |
|------|-----------|------|---------------|----------|
| **Datacenter** | IPs from cloud providers (AWS, GCP, etc.) | Cheapest (~$1–2/GB) | High — IMDB actively blocks known datacenter IP ranges | Low-security targets, not recommended for IMDB |
| **Residential** | IPs from real ISP customers (via opt-in SDKs in consumer apps) | Medium (~$8–15/GB) | Low — looks like real home users | IMDB scraping (recommended) |
| **ISP/Static residential** | Datacenter-hosted IPs registered to real ISPs | Medium-high | Low-medium | When you need consistent IPs |
| **Mobile** | IPs from mobile carriers | Most expensive (~$20–30/GB) | Lowest | Only if residential gets blocked |

**For IMDB: use residential rotating proxies.** They provide the best balance of cost, reliability, and stealth.

#### Recommended proxy providers

| Provider | Residential price | Why consider it |
|----------|------------------|-----------------|
| **Bright Data** | ~$8–10/GB | Largest proxy network, most reliable. Has a "Web Unlocker" product specifically for anti-bot bypass. More complex setup but most feature-rich. |
| **SmartProxy** | ~$7–8/GB | Simpler API, good documentation. Slightly cheaper. Very good for IMDB-scale scraping. |
| **Oxylabs** | ~$10–12/GB | Enterprise-grade, excellent for large volumes. Has "Web Scraper API" that handles retries/rotation for you. |
| **ScraperAPI** | ~$1–3 per 1K requests | Pay-per-request model instead of per-GB. Handles proxy rotation, headers, and retries automatically. Simplest integration but most expensive at scale. |

**Recommended starting point: SmartProxy or Bright Data residential.** Both offer pay-as-you-go plans.

#### Cost estimate for IMDB scraping

- 600K page fetches
- Average page size: ~50–80KB (IMDB pages with `__NEXT_DATA__` JSON are relatively large)
- Total bandwidth: ~30–50 GB
- At $8/GB residential: **~$240–400**
- At $3/1K requests (ScraperAPI): **~$1,800** (much more expensive but zero setup)

#### How proxy integration works in code

Most residential proxy providers give you a single gateway endpoint. You authenticate with your credentials, and they handle rotation automatically:

```python
import aiohttp

PROXY_URL = "http://USERNAME:PASSWORD@gate.smartproxy.com:7777"
# or for Bright Data:
# PROXY_URL = "http://USERNAME:PASSWORD@brd.superproxy.io:22225"

async def fetch_imdb_page(session: aiohttp.ClientSession, url: str) -> str:
    headers = {
        "User-Agent": random.choice(USER_AGENT_LIST),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    }
    async with session.get(url, proxy=PROXY_URL, headers=headers, timeout=30) as resp:
        if resp.status == 200:
            return await resp.text()
        elif resp.status == 403:
            # Proxy IP got blocked, retry (rotation gives new IP automatically)
            raise RetryableError("403 from IMDB, will retry with new proxy IP")
        elif resp.status == 429:
            # Rate limited — back off
            await asyncio.sleep(random.uniform(2, 5))
            raise RetryableError("429 rate limited")
        else:
            raise FetchError(f"HTTP {resp.status}")
```

**Key integration points:**
- The proxy provider rotates IPs for you on every request (or per-session, configurable)
- You still want to rotate `User-Agent` headers yourself (keep a list of 20+ real browser UAs)
- Add small random delays between requests (0.1–0.5s) even with proxies — helps avoid pattern detection
- The `proxy` parameter in `aiohttp` routes all traffic through the gateway

#### Anti-detection best practices

1. **Rotate User-Agents** — maintain a list of 20+ real Chrome/Firefox/Safari UA strings, pick randomly per request
2. **Randomize request timing** — add `asyncio.sleep(random.uniform(0.1, 0.5))` between requests, even within the same movie's 6 pages
3. **Don't hammer sequentially** — stagger the 6 pages for a single movie across different times rather than all at once
4. **Handle soft blocks gracefully** — if you get a CAPTCHA page or empty response, back off that IP (the proxy rotation handles this, but log it)
5. **Respect the `__NEXT_DATA__` approach** — this is actually great for anti-detection because you're making normal page requests, not hitting internal APIs. IMDB serves the same HTML to everyone

#### Session management with proxies

For IMDB scraping, you want a **new IP per request** (not sticky sessions). Configure your proxy provider accordingly:

```python
# SmartProxy example — each request gets a fresh IP
PROXY_URL = "http://user-USERNAME:PASSWORD@gate.smartproxy.com:7777"

# Bright Data example — add session randomization
PROXY_URL = f"http://USERNAME-session-{random.randint(1,999999)}:PASSWORD@brd.superproxy.io:22225"
```

### IMDB Scraping Pipeline

**Parallelism model: fully async with `asyncio` + `aiohttp`.**

The industry standard for high-concurrency I/O-bound work is `asyncio`, not threads. The reasons:

- **Memory:** Each Python thread carries ~8MB of stack space. 100 threads = ~800MB just for stacks. An async coroutine is ~1KB. 10,000 coroutines = ~10MB.
- **Coordination overhead:** Threads require OS context switches and GIL acquisition for any Python-level work. Coroutines yield cooperatively with zero context-switch cost.
- **Backpressure:** A single `asyncio.Semaphore` naturally limits total in-flight HTTP requests across all movies, regardless of how many movie-level coroutines are active. With threads, you'd need to carefully size both an outer pool (movies) and inner pool (pages per movie), and they compete for resources in ways that are hard to reason about.
- **Industry norm:** Modern Python scraping frameworks (Scrapy, crawlee, httpx-based tooling) are all async-first. This is the pattern you'd see in any production scraping system.

**Architecture: one global semaphore, one coroutine per movie.**

```python
import asyncio
import aiohttp
import random

# Global concurrency control — this is the ONLY throttle you need.
# Every HTTP request (across all movies) must acquire this before firing.
SCRAPE_SEMAPHORE = asyncio.Semaphore(50)  # Start here, tune based on error rates

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 ...",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ...",
    # ... 20+ real browser UA strings
]

async def fetch_imdb_page(
    session: aiohttp.ClientSession, 
    url: str, 
    max_retries: int = 3
) -> str:
    """Fetch a single IMDB page through the proxy, with retries."""
    for attempt in range(max_retries):
        async with SCRAPE_SEMAPHORE:
            # Small random delay — avoids burst patterns that trigger detection
            await asyncio.sleep(random.uniform(0.05, 0.3))
            
            headers = {
                "User-Agent": random.choice(USER_AGENTS),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            }
            try:
                async with session.get(
                    url, proxy=PROXY_URL, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        return await resp.text()
                    elif resp.status in (403, 429):
                        # Proxy rotation gives a new IP on next attempt automatically
                        wait = random.uniform(1, 3) * (attempt + 1)
                        await asyncio.sleep(wait)
                        continue
                    else:
                        raise FetchError(f"HTTP {resp.status} for {url}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == max_retries - 1:
                    raise FetchError(f"Failed after {max_retries} attempts: {e}")
                await asyncio.sleep(random.uniform(1, 3))
    
    raise FetchError(f"Exhausted retries for {url}")


async def scrape_single_movie(
    session: aiohttp.ClientSession,
    tmdb_id: int,
    imdb_id: str,
    db: sqlite3.Connection
) -> bool:
    """Scrape all 6 IMDB pages for one movie. Returns True on success."""
    
    pages = {
        "main_page":    f"https://www.imdb.com/title/{imdb_id}/",
        "summary":      f"https://www.imdb.com/title/{imdb_id}/plotsummary/",
        "keywords":     f"https://www.imdb.com/title/{imdb_id}/keywords/",
        "parental":     f"https://www.imdb.com/title/{imdb_id}/parentalguide/",
        "credits":      f"https://www.imdb.com/title/{imdb_id}/fullcredits/",
        "reviews":      f"https://www.imdb.com/title/{imdb_id}/reviews/",
    }
    
    # Fan out all 6 page fetches concurrently.
    # They each independently acquire the global semaphore, so total
    # concurrency stays bounded regardless of how many movies are in-flight.
    async def fetch_page(name: str, url: str):
        html = await fetch_imdb_page(session, url)
        return (name, html)
    
    try:
        results = await asyncio.gather(
            *[fetch_page(name, url) for name, url in pages.items()]
        )
    except FetchError as e:
        log_filter(db, tmdb_id, "imdb_scrape", "page_fetch_failed", str(e))
        return False
    
    # Parse all pages (CPU-bound but fast — no need to offload to threads)
    html_by_name = dict(results)
    try:
        parsed = {
            "imdb_data":        extract_imdb_attributes(html_by_name["main_page"]),
            "summary_data":     extract_summary_attributes(html_by_name["summary"]),
            "plot_keywords":    extract_plot_keywords(html_by_name["keywords"]),
            "parental_data":    extract_parental_guide(html_by_name["parental"]),
            "cast_crew_data":   extract_cast_crew(html_by_name["credits"]),
            "featured_reviews": extract_featured_reviews(html_by_name["reviews"]),
        }
    except Exception as e:
        log_filter(db, tmdb_id, "imdb_scrape", "parse_error", str(e))
        return False
    
    save_json(f"./ingestion_data/imdb/{tmdb_id}.json", parsed)
    return True


async def stage4_imdb_scrape(db: sqlite3.Connection):
    """Main Stage 4 entry point. Scrapes all quality_passed movies."""
    
    rows = db.execute(
        "SELECT tmdb_id, imdb_id FROM movie_progress WHERE status = 'quality_passed'"
    ).fetchall()
    
    # Single aiohttp session for connection pooling (the proxy handles IP rotation)
    connector = aiohttp.TCPConnector(limit=100, limit_per_host=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        
        # Process all movies concurrently — the semaphore handles throttling.
        # asyncio.gather would launch all 100K coroutines at once (fine — they're
        # lightweight and block on the semaphore). But for progress reporting,
        # we use asyncio.as_completed or batch into chunks.
        
        CHUNK_SIZE = 500  # Process in chunks for progress reporting + DB commits
        
        for i in range(0, len(rows), CHUNK_SIZE):
            chunk = rows[i:i + CHUNK_SIZE]
            
            tasks = [
                scrape_single_movie(session, tmdb_id, imdb_id, db)
                for tmdb_id, imdb_id in chunk
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update statuses for successful movies in this chunk
            for (tmdb_id, _), result in zip(chunk, results):
                if result is True:
                    db.execute(
                        "UPDATE movie_progress SET status = 'imdb_scraped', updated_at = CURRENT_TIMESTAMP WHERE tmdb_id = ?",
                        (tmdb_id,)
                    )
                elif isinstance(result, Exception):
                    log_filter(db, tmdb_id, "imdb_scrape", "unexpected_error", str(result))
            
            db.commit()
            print(f"  IMDB scrape progress: {min(i + CHUNK_SIZE, len(rows)):,}/{len(rows):,}")
```

**Why this pattern works well:**

- The semaphore is the single source of truth for concurrency. You tune one number (50 → 100 → 30) based on error rates, and everything else adapts.
- Each movie's 6 pages fan out independently within the semaphore. If the semaphore is 50 and 10 movies are each trying to fetch 6 pages, at most 50 of those 60 fetches run simultaneously — the rest queue up automatically.
- Chunking at 500 movies gives you natural commit points and progress reporting without adding any throttling beyond the semaphore.
- The `aiohttp.TCPConnector` with `limit=100` prevents opening too many TCP connections to the proxy gateway, which is a separate concern from request-level concurrency.

**Tuning guide:**
- Start with semaphore=50. Monitor your proxy provider's dashboard for error rates.
- If error rate < 5%, increase to 75, then 100.
- If error rate > 15%, decrease to 30.
- The proxy provider's dashboard is your primary signal — not your local logs.

**Essential vs. non-essential data check:**

After parsing, validate that essential fields exist. This mirrors your existing filtering logic:

```python
ESSENTIAL_FIELDS = [
    "overview",      # Need some form of plot description
    "genres",        # Required for filtering and vector generation
    "imdb_rating",   # Required for reception tier calculation
]

# If essential fields missing → filter out
# If non-essential fields missing → fill defaults
NON_ESSENTIAL_DEFAULTS = {
    "filming_locations": [],
    "budget": None,
    "metacritic_rating": None,
    "reception_summary": None,
    "featured_reviews": [],
    "review_themes": [],
    "maturity_reasoning": [],
    "parental_guide_items": [],
}
```

Movies failing essential checks get logged to `filter_log` with `stage='essential_data_check'` and `status='filtered_out'`.

**Error handling per page:**
- If any of the 6 pages fails after 3 retries → log to `filter_log` with `stage='imdb_scrape', reason='page_fetch_failed', details='{"page": "keywords", "error": "..."}'` and set `status='filtered_out'`
- Parse errors (unexpected HTML structure) → same treatment

**Resumability:** On restart, query `WHERE status = 'quality_passed'` and process only those.

---

## Stage 5: LLM Phase 1 — Batch API

**Goal:** Generate `plot_events_metadata`, `watch_context_metadata`, and `reception_metadata` for all scraped movies.

**Input:** All `movie_progress` rows where `status = 'imdb_scraped'`

### OpenAI Batch API Primer

The Batch API lets you submit a file of requests and retrieve results later (typically within 24 hours). It costs 50% less than real-time API calls.

**How it works:**
1. Create a `.jsonl` file where each line is a complete API request
2. Upload the file to OpenAI
3. Create a "batch" pointing to that file
4. Poll for completion (or set up a webhook)
5. Download the results file (another `.jsonl` with responses keyed by `custom_id`)

**Request format (each line of the .jsonl):**
```json
{
  "custom_id": "phase1_plot_events_9377",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "gpt-5-mini",
    "max_tokens": 2000,
    "response_format": { "type": "json_schema", "json_schema": { ... } },
    "messages": [
      { "role": "system", "content": "..." },
      { "role": "user", "content": "..." }
    ]
  }
}
```

**The `custom_id` is how you match results back to movies.** Use a convention like `phase1_{generation_type}_{tmdb_id}` so you can parse it programmatically.

### Batch File Construction

For Phase 1, each movie produces 3 lines in the batch file:

```
phase1_plot_events_{tmdb_id}     → plot_events generation prompt
phase1_watch_context_{tmdb_id}   → watch_context generation prompt  
phase1_reception_{tmdb_id}       → reception generation prompt
```

100K movies × 3 = 300K lines.

**Batch size limits:** OpenAI currently allows up to 50,000 requests per batch. With 300K requests, you'll need to split across **6 batch files**. Submit them all at once — they process in parallel on OpenAI's side.

```python
BATCH_SIZE = 50_000

def build_phase1_batches(movies: list[dict]) -> list[str]:
    """Build .jsonl files for Phase 1, split into 50K-request chunks."""
    all_requests = []
    for movie in movies:
        tmdb_id = movie["tmdb_id"]
        tmdb_data = load_json(f"./ingestion_data/tmdb/{tmdb_id}.json")
        imdb_data = load_json(f"./ingestion_data/imdb/{tmdb_id}.json")
        
        # Build the 3 prompts using your existing prompt templates
        all_requests.append(build_plot_events_request(tmdb_id, tmdb_data, imdb_data))
        all_requests.append(build_watch_context_request(tmdb_id, tmdb_data, imdb_data))
        all_requests.append(build_reception_request(tmdb_id, tmdb_data, imdb_data))
    
    # Split into chunks and write .jsonl files
    batch_paths = []
    for i in range(0, len(all_requests), BATCH_SIZE):
        chunk = all_requests[i:i + BATCH_SIZE]
        path = f"./ingestion_data/batch_files/phase1_batch_{i // BATCH_SIZE}.jsonl"
        write_jsonl(path, chunk)
        batch_paths.append(path)
    
    return batch_paths
```

### Submission Flow

```python
import openai

client = openai.OpenAI()

def submit_batch(jsonl_path: str) -> str:
    """Upload file and create batch. Returns batch ID."""
    # 1. Upload the file
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    
    # 2. Create the batch
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    return batch.id

def poll_batch(batch_id: str) -> str:
    """Poll until complete. Returns output file ID."""
    while True:
        batch = client.batches.retrieve(batch_id)
        if batch.status == "completed":
            return batch.output_file_id
        elif batch.status == "failed":
            raise BatchError(f"Batch {batch_id} failed: {batch.errors}")
        elif batch.status in ("expired", "cancelled"):
            raise BatchError(f"Batch {batch_id} status: {batch.status}")
        
        # Log progress
        print(f"Batch {batch_id}: {batch.request_counts.completed}/{batch.request_counts.total}")
        time.sleep(60)  # Check every minute
```

### Result Processing

When a batch completes, download the output file and match results back to movies:

```python
def process_phase1_results(output_file_id: str):
    """Download results and update per-movie JSON files + tracker DB."""
    content = client.files.content(output_file_id)
    
    for line in content.text.strip().split("\n"):
        result = json.loads(line)
        custom_id = result["custom_id"]          # e.g., "phase1_plot_events_9377"
        
        # Parse the custom_id
        _, gen_type, tmdb_id_str = custom_id.split("_", 2)
        # Actually, since gen_type may have underscores (plot_events), use a different approach:
        parts = custom_id.split("_")
        tmdb_id = int(parts[-1])
        gen_type = "_".join(parts[1:-1])  # "plot_events", "watch_context", "reception"
        
        if result["response"]["status_code"] == 200:
            body = result["response"]["body"]
            content = json.loads(body["choices"][0]["message"]["content"])
            
            # Load existing phase1 file (or create new)
            phase1_path = f"./ingestion_data/llm_phase1/{tmdb_id}.json"
            phase1_data = load_json_or_default(phase1_path, {})
            phase1_data[f"{gen_type}_metadata"] = content
            save_json(phase1_path, phase1_data)
        else:
            # Log the failure for retry
            log_filter(tmdb_id, "llm_phase1", f"{gen_type}_failed", 
                       details=json.dumps(result["error"]))
```

### Handling Failures and Retries

After processing all batch results, identify movies with incomplete Phase 1 data:

```python
def get_phase1_incomplete_movies() -> list[tuple[int, list[str]]]:
    """Returns [(tmdb_id, [missing_generation_types])] for retry."""
    incomplete = []
    for row in db.execute("SELECT tmdb_id FROM movie_progress WHERE status = 'imdb_scraped'"):
        tmdb_id = row[0]
        phase1_path = f"./ingestion_data/llm_phase1/{tmdb_id}.json"
        phase1_data = load_json_or_default(phase1_path, {})
        
        missing = []
        for gen_type in ["plot_events", "watch_context", "reception"]:
            if f"{gen_type}_metadata" not in phase1_data:
                missing.append(gen_type)
        
        if missing:
            incomplete.append((tmdb_id, missing))
    
    return incomplete
```

**Retry strategy:**
- Collect all incomplete generations
- Build a small retry batch file (likely a few hundred to low thousands of requests)
- Submit as another batch (processes quickly since it's small)
- Repeat up to 3 total attempts
- After 3 attempts, any still-incomplete movies get logged to `filter_log` with `stage='llm_phase1', reason='exhausted_retries'` and `status='filtered_out'`

**After all retries resolve:** Update all successful movies to `status = 'phase1_complete'`.

---

## Stage 6: LLM Phase 2 — Batch API

**Goal:** Generate `plot_analysis_metadata`, `viewer_experience_metadata`, `narrative_techniques_metadata`, and `production_metadata` using Phase 1 outputs.

**Input:** All `movie_progress` rows where `status = 'phase1_complete'`

**Process is identical to Stage 5** with these differences:

- Each movie produces **4** lines in the batch file (not 3)
- Prompts include Phase 1 outputs (especially `plot_events_metadata`) as additional context
- `custom_id` format: `phase2_{gen_type}_{tmdb_id}`
- ~100K movies × 4 = 400K requests → **8 batch files** (at 50K per batch)

**Prompt construction must load Phase 1 results:**
```python
def build_plot_analysis_request(tmdb_id: int, tmdb_data: dict, imdb_data: dict) -> dict:
    phase1_data = load_json(f"./ingestion_data/llm_phase1/{tmdb_id}.json")
    plot_events = phase1_data["plot_events_metadata"]
    
    # plot_events feeds into the plot_analysis prompt
    prompt = build_plot_analysis_prompt(
        title=tmdb_data["title"],
        overview=imdb_data["overview"],
        plot_events=plot_events,  # <-- Phase 1 dependency
        genres=imdb_data["genres"],
        keywords=imdb_data["keywords"],
        # ... etc
    )
    
    return {
        "custom_id": f"phase2_plot_analysis_{tmdb_id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": { ... }
    }
```

Same retry logic as Stage 5. After all retries: successful movies → `status = 'phase2_complete'`.

---

## Stage 7: Embedding (Real-Time Batched)

**Goal:** Generate 8 vector embeddings per movie using OpenAI's embedding API.

**Input:** All `movie_progress` rows where `status = 'phase2_complete'`

**Why not use the Batch API here?**

Embedding is cheap. At `text-embedding-3-small` pricing ($0.02/1M tokens):
- 100K movies × 8 vectors × ~400 tokens average = ~320M tokens
- Cost: ~$6.40
- Batch discount would save ~$3.20 — not worth an extra 24-hour wait

**Approach: real-time API calls with batched inputs.**

OpenAI's embedding endpoint accepts arrays of strings (up to 2048 inputs per call). This dramatically reduces the number of HTTP requests:

```python
EMBED_BATCH_SIZE = 2048  # Max inputs per API call

async def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed up to 2048 texts in a single API call."""
    response = await async_client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]
```

**Pipeline:**

1. For each `phase2_complete` movie, generate all 8 vector text strings using your existing `create_*_vector_text()` functions
2. Collect texts into batches of 2048
3. Fire embedding API calls with moderate concurrency (semaphore of 10–20)
4. Store results as packed binary float32 arrays (6,144 bytes per vector × 8 = 49,152 bytes per movie)
5. Update `status = 'embedded'`

**Math:**
- 100K movies × 8 vectors = 800K texts
- 800K / 2048 per call = ~391 API calls
- At 10 concurrent calls, ~1–2 seconds each = ~40–80 seconds
- With overhead, budget 20–40 minutes total

**Storage format (matches your Redis convention):**
```python
import struct

def pack_embedding(embedding: list[float]) -> bytes:
    return struct.pack(f'{len(embedding)}f', *embedding)

def save_movie_embeddings(tmdb_id: int, embeddings: dict[str, list[float]]):
    """Save all 8 embeddings for a movie as a single binary file."""
    path = f"./ingestion_data/embeddings/{tmdb_id}.bin"
    with open(path, 'wb') as f:
        for vector_name in VECTOR_NAMES:  # consistent ordering
            f.write(pack_embedding(embeddings[vector_name]))
```

**Resumability:** On restart, process only `WHERE status = 'phase2_complete'` (not yet `embedded`).

---

## Stage 8: Database Ingestion

**Goal:** Bulk-insert all `embedded` movies into Postgres and Qdrant.

**Input:** All `movie_progress` rows where `status = 'embedded'`

### 8a: Postgres Ingestion

Insert into both `public.movie_card` and all `lex.*` tables.

**Approach: batch inserts using `COPY` or multi-row `INSERT` with `executemany`.**

For `movie_card`, construct the row from combined TMDB + IMDB + LLM data:

```python
# Pseudocode for building a movie_card row
def build_movie_card_row(tmdb_id: int) -> dict:
    tmdb = load_json(f"./ingestion_data/tmdb/{tmdb_id}.json")
    imdb = load_json(f"./ingestion_data/imdb/{tmdb_id}.json")
    phase1 = load_json(f"./ingestion_data/llm_phase1/{tmdb_id}.json")
    phase2 = load_json(f"./ingestion_data/llm_phase2/{tmdb_id}.json")
    
    return {
        "movie_id": tmdb_id,
        "title": tmdb["title"],
        "year": extract_year(tmdb["release_date"]),
        "poster_url": build_poster_url(tmdb["poster_path"]),
        "release_ts": date_to_unix(tmdb["release_date"]),
        "runtime_minutes": tmdb["runtime"],
        "maturity_rank": maturity_to_rank(imdb["maturity_rating"]),
        "genre_ids": map_genres_to_ids(imdb["genres"]),
        "watch_offer_keys": encode_watch_offer_keys(tmdb["watch_providers"]),
        "audio_language_ids": map_languages_to_ids(imdb["languages"]),
        "reception_score": compute_reception_score(imdb["imdb_rating"], imdb["metacritic_rating"]),
    }
```

**Lexical tables** (`lex.lexical_dictionary`, `lex.inv_*_postings`, `lex.title_token_doc_frequency`):

Process in bulk after all `movie_card` rows are inserted:

1. Collect all unique normalized strings across all movies (people, characters, studios, title tokens)
2. Batch-insert into `lex.lexical_dictionary` with `ON CONFLICT DO NOTHING` to get/create `string_id`s
3. Batch-insert posting rows into each `lex.inv_*_postings` table
4. Compute and insert `lex.title_token_doc_frequency`

**Batch size:** Insert in batches of 1,000–5,000 rows. Use `psycopg2.extras.execute_values()` for efficient multi-row inserts.

**Transaction strategy:** Commit per batch (not per movie, not one giant transaction). This avoids lock contention and provides incremental progress.

### 8b: Qdrant Ingestion

Upsert points with all 8 named vectors + payload fields.

**Approach: batch upsert using Qdrant's batch upload.**

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

QDRANT_BATCH_SIZE = 100  # Points per upsert call

def build_qdrant_point(tmdb_id: int) -> PointStruct:
    embeddings = load_embeddings(f"./ingestion_data/embeddings/{tmdb_id}.bin")
    tmdb = load_json(f"./ingestion_data/tmdb/{tmdb_id}.json")
    imdb = load_json(f"./ingestion_data/imdb/{tmdb_id}.json")
    
    return PointStruct(
        id=tmdb_id,
        vector={
            "anchor": embeddings["anchor"],
            "plot_events": embeddings["plot_events"],
            "plot_analysis": embeddings["plot_analysis"],
            "viewer_experience": embeddings["viewer_experience"],
            "watch_context": embeddings["watch_context"],
            "narrative_techniques": embeddings["narrative_techniques"],
            "production": embeddings["production"],
            "reception": embeddings["reception"],
        },
        payload={
            "release_ts": date_to_unix(tmdb["release_date"]),
            "runtime_minutes": tmdb["runtime"],
            "maturity_rank": maturity_to_rank(imdb["maturity_rating"]),
            "genre_ids": map_genres_to_ids(imdb["genres"]),
            "watch_offer_keys": encode_watch_offer_keys(tmdb["watch_providers"]),
        }
    )

# Batch upsert
client = QdrantClient(host="your-ec2-ip", port=6333)
batch = []
for tmdb_id in embedded_movie_ids:
    batch.append(build_qdrant_point(tmdb_id))
    if len(batch) >= QDRANT_BATCH_SIZE:
        client.upsert(collection_name="movies_v1", points=batch)
        batch = []
if batch:
    client.upsert(collection_name="movies_v1", points=batch)
```

**Qdrant batch sizing:** 100 points per upsert is a good balance. Each point has 8 × 1536 floats = ~49KB of vector data, so 100 points = ~5MB per request. Qdrant handles this well.

**After both Postgres and Qdrant ingestion complete:** Update `status = 'ingested'` for each movie.

**Resumability:** On restart, process only `WHERE status = 'embedded'`. Qdrant upserts are idempotent (same point ID overwrites), and Postgres inserts should use `ON CONFLICT DO UPDATE` or `INSERT ... ON CONFLICT DO NOTHING`.

---

## Cost Estimates

### LLM Generation (Batch API, 50% discount applied)

| Phase | Calls | Avg input tokens | Avg output tokens | Model | Est. cost |
|-------|-------|-----------------|-------------------|-------|-----------|
| Phase 1 | 300K | ~2,500 | ~500 | gpt-5-mini/nano | ~$100–150 |
| Phase 2 | 400K | ~3,500 | ~500 | gpt-5-mini/nano | ~$130–180 |
| Retries (~2%) | ~14K | ~3,000 | ~500 | gpt-5-mini/nano | ~$5–10 |
| **Total LLM** | | | | | **~$235–340** |

*Note: Exact costs depend on gpt-5-mini/nano pricing tiers. The above estimates use gpt-4o-mini-equivalent rates as a baseline since gpt-5-mini/nano pricing may differ.*

### Embedding

| Item | Count | Tokens | Cost |
|------|-------|--------|------|
| Vectors | 800K | ~320M | ~$6.40 |

### Proxy / Scraping

| Item | Estimate |
|------|----------|
| Residential proxies (~40GB) | ~$300–400 |

### Total

| Category | Cost |
|----------|------|
| LLM generations | $235–340 |
| Embeddings | ~$6 |
| Proxies | $300–400 |
| **Grand total** | **~$540–750** |

---

## Orchestrator Script Structure

The pipeline is driven by a single orchestrator that calls each stage and respects the checkpoint DB:

```python
async def run_pipeline():
    db = init_tracker_db("./ingestion_data/tracker.db")
    
    # Stage 1
    print("=== Stage 1: TMDB Export Filtering ===")
    await stage1_tmdb_export(db)
    
    # Stage 2
    print("=== Stage 2: TMDB Detail Fetching ===")
    await stage2_tmdb_fetch(db)
    
    # Stage 3
    print("=== Stage 3: TMDB Quality Funnel ===")
    cutoff = stage3_quality_funnel(db, target_count=100_000)
    
    # Stage 4
    print("=== Stage 4: IMDB Scraping ===")
    await stage4_imdb_scrape(db)
    
    # Stage 5
    print("=== Stage 5: LLM Phase 1 (Batch API) ===")
    batch_ids = stage5_build_and_submit_phase1(db)
    print(f"Submitted {len(batch_ids)} batches. Waiting for completion...")
    print("Run this script again after batches complete, or wait here.")
    await stage5_poll_and_process(db, batch_ids)
    stage5_retry_failures(db, max_retries=3)
    
    # Stage 6
    print("=== Stage 6: LLM Phase 2 (Batch API) ===")
    batch_ids = stage6_build_and_submit_phase2(db)
    print(f"Submitted {len(batch_ids)} batches. Waiting for completion...")
    await stage6_poll_and_process(db, batch_ids)
    stage6_retry_failures(db, max_retries=3)
    
    # Stage 7
    print("=== Stage 7: Embedding ===")
    await stage7_embed(db)
    
    # Stage 8
    print("=== Stage 8: Database Ingestion ===")
    await stage8_ingest(db)
    
    # Summary
    print_final_summary(db)
```

Each `stageN_*` function queries the tracker DB for movies in the appropriate status, processes only those, and updates statuses as it goes. You can run the orchestrator multiple times safely — it always picks up where it left off.

### Running Individual Stages

Since batch waits take ~24 hours, you'll likely run the script in stages:

```bash
# Day 0: Export, TMDB fetch, quality funnel, IMDB scrape, submit Phase 1 LLM batch
python ingest.py --through stage5_submit

# Day 1: Process Phase 1 results, submit Phase 2 LLM batch  
python ingest.py --through stage6_submit

# Day 2: Process Phase 2 results, embed, ingest into databases
python ingest.py --through stage8
```

---

## Monitoring & Observability

### Progress Dashboard (Terminal)

Print periodic status summaries during long-running stages:

```python
def print_progress(db):
    counts = db.execute("""
        SELECT status, COUNT(*) FROM movie_progress GROUP BY status
    """).fetchall()
    for status, count in counts:
        print(f"  {status}: {count:,}")
    
    filter_counts = db.execute("""
        SELECT stage, reason, COUNT(*) FROM filter_log 
        GROUP BY stage, reason ORDER BY COUNT(*) DESC
    """).fetchall()
    if filter_counts:
        print("\nFiltered out:")
        for stage, reason, count in filter_counts:
            print(f"  [{stage}] {reason}: {count:,}")
    
    # Show a sample of recently filtered movies (for sanity checking)
    recent_filters = db.execute("""
        SELECT tmdb_id, title, year, stage, reason 
        FROM filter_log 
        WHERE title IS NOT NULL
        ORDER BY created_at DESC LIMIT 10
    """).fetchall()
    if recent_filters:
        print("\nRecent filter samples:")
        for tmdb_id, title, year, stage, reason in recent_filters:
            print(f"  {title} ({year}) [tmdb:{tmdb_id}] — {stage}: {reason}")
```

### Post-Ingestion Validation

After Stage 8, run sanity checks:

```python
def validate_ingestion(db):
    ingested_count = db.execute(
        "SELECT COUNT(*) FROM movie_progress WHERE status = 'ingested'"
    ).fetchone()[0]
    
    # Check Postgres
    pg_count = pg_conn.execute("SELECT COUNT(*) FROM public.movie_card").fetchone()[0]
    assert pg_count == ingested_count, f"Postgres mismatch: {pg_count} vs {ingested_count}"
    
    # Check Qdrant
    collection_info = qdrant_client.get_collection("movies_v1")
    assert collection_info.points_count == ingested_count
    
    # Spot check: random sample of 10 movies, verify all 8 vectors exist
    sample_ids = random.sample(range(ingested_count), 10)
    for point_id in sample_ids:
        point = qdrant_client.retrieve(
            collection_name="movies_v1",
            ids=[point_id],
            with_vectors=True
        )
        assert len(point[0].vector) == 8, f"Point {point_id} missing vectors"
    
    print(f"Validation passed: {ingested_count:,} movies in all databases")
```

---

## Re-Running for Previously Filtered Movies

After the initial ingest, you can query the `filter_log` to find movies that might have been incorrectly filtered:

```sql
-- Movies filtered due to transient errors (worth retrying)
SELECT tmdb_id, title, year, stage, reason, details 
FROM filter_log 
WHERE reason LIKE '%error%' OR reason LIKE '%timeout%' OR reason LIKE '%rate_limit%'
ORDER BY stage;

-- Movies filtered for missing data (check if IMDB has updated)
SELECT tmdb_id, title, year, stage, reason 
FROM filter_log 
WHERE reason LIKE '%missing%'
ORDER BY stage;

-- Sanity check: any well-known movies that got filtered?
SELECT tmdb_id, title, year, stage, reason
FROM filter_log
WHERE title IS NOT NULL
ORDER BY year DESC, title
LIMIT 50;
```

To retry filtered movies:

```python
def reset_filtered_movies(db, tmdb_ids: list[int]):
    """Reset filtered movies back to the appropriate stage for retry."""
    for tmdb_id in tmdb_ids:
        # Find what stage they failed at
        last_filter = db.execute(
            "SELECT stage FROM filter_log WHERE tmdb_id = ? ORDER BY created_at DESC LIMIT 1",
            (tmdb_id,)
        ).fetchone()
        
        # Reset to the status just before the failed stage
        stage_to_status = {
            "tmdb_fetch": "pending",
            "tmdb_quality_funnel": "tmdb_fetched",
            "imdb_scrape": "quality_passed",
            "essential_data_check": "quality_passed",
            "llm_phase1": "imdb_scraped",
            "llm_phase2": "phase1_complete",
            "embedding": "phase2_complete",
            "ingestion": "embedded",
        }
        
        reset_status = stage_to_status.get(last_filter[0], "pending")
        db.execute(
            "UPDATE movie_progress SET status = ? WHERE tmdb_id = ?",
            (reset_status, tmdb_id)
        )
    
    db.commit()
```

Then re-run the orchestrator — it picks them up at the appropriate stage.


---

## Daily Update Pipeline

After the initial bulk ingest, you need a lightweight process to keep the catalog current with new releases.

### Discovery via TMDB Changes Endpoint

The TMDB `/movie/changes` endpoint returns IDs of all movies that have been modified or **newly added** within a date range (confirmed by TMDB staff, max 14-day window).

**Why this is sufficient on its own:** Any movie worth adding to your catalog will have ongoing edits on TMDB — poster uploads, cast/crew additions, translations, vote accumulation. Each edit is another appearance in the changes feed. You don't need to catch a movie on its first change; you just need to catch it once. For a movie notable enough to pass your quality cutoff, the odds of it never appearing in the changes feed across weeks of daily polling are effectively zero.

```python
async def fetch_tmdb_changes(session, start_date: str, end_date: str) -> list[int]:
    """
    Fetch all movie IDs that changed/were added in the given date range.
    Returns may include movies you already have — caller deduplicates.
    
    The endpoint is paginated (100 results per page).
    """
    all_ids = []
    page = 1
    
    while True:
        await tmdb_limiter.acquire()
        async with session.get(
            "https://api.themoviedb.org/3/movie/changes",
            params={"start_date": start_date, "end_date": end_date, "page": page},
            headers={"Authorization": f"Bearer {TMDB_API_TOKEN}"}
        ) as resp:
            data = await resp.json()
            results = data.get("results", [])
            if not results:
                break
            
            for entry in results:
                if not entry.get("adult", False):
                    all_ids.append(entry["id"])
            
            if page >= data.get("total_pages", 1):
                break
            page += 1
    
    return all_ids
```

**Typical volume:** A few hundred to a few thousand IDs per day. Most will be edits to existing movies (which you skip); a smaller subset will be genuinely new candidates.

### The Release Date Age Cutoff

The changes endpoint returns IDs for *any* movie that was edited, including a 1987 film where someone fixed a typo in the Portuguese overview. You don't want to re-evaluate those daily — the monthly `below_quality_cutoff` re-eval handles the rare case where an old film legitimately surges in popularity.

**Rule: only consider movies released within the last 8 weeks.**

This single filter solves three problems at once:

1. **Unreleased movies are automatically excluded.** No need to track them in a special status. When they eventually release and accumulate TMDB activity (votes, reviews, streaming additions), they'll appear in changes again, pass the release date filter, and get evaluated fresh.

2. **Old catalog edits are excluded.** A minor metadata fix on an old movie doesn't trigger expensive IMDB scraping and LLM generation. These movies were already evaluated during the bulk ingest or a previous monthly re-eval.

3. **The window is wide enough to catch everything that matters.** IMDB data takes a few weeks to accumulate after release (reviews trickle in, vote counts stabilize). Streaming debuts often lag theatrical by 4–6 weeks. An 8-week window catches both theatrical and streaming premieres with buffer for data to be meaningful.

```python
from datetime import datetime, timedelta

RELEASE_AGE_CUTOFF_DAYS = 56  # 8 weeks — tune as needed

def passes_release_date_filter(release_date: str) -> bool:
    """
    Returns True if the movie was released within the last RELEASE_AGE_CUTOFF_DAYS
    and is not a future release.
    """
    if not release_date or len(release_date) < 10:
        return False  # No release date = skip
    
    try:
        rd = datetime.strptime(release_date[:10], "%Y-%m-%d")
    except ValueError:
        return False
    
    now = datetime.utcnow()
    if rd > now:
        return False  # Not yet released
    
    age = (now - rd).days
    return age <= RELEASE_AGE_CUTOFF_DAYS
```

### Daily Update Flow

```python
async def daily_update(db):
    """
    Daily pipeline: discover recently-released movies and ingest qualifying ones.
    Runs on the EC2 instance as a cron job.
    """
    cutoff = load_json("./ingestion_data/quality_cutoff.json")
    cutoff_score = cutoff["cutoff_score"]
    
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    
    # Step 1: Get candidate IDs from TMDB changes
    async with aiohttp.ClientSession() as session:
        changed_ids = await fetch_tmdb_changes(session, yesterday, today)
    
    print(f"TMDB changes returned {len(changed_ids)} IDs")
    
    # Step 2: Skip movies already ingested into our databases.
    # Everything else is fair game — even movies we've previously filtered out,
    # because their data may have changed (newly released, gained votes, etc.).
    already_ingested = set(
        row[0] for row in db.execute(
            "SELECT tmdb_id FROM movie_progress WHERE status = 'ingested'"
        ).fetchall()
    )
    candidate_ids = [tid for tid in changed_ids if tid not in already_ingested]
    
    if not candidate_ids:
        print("No new candidates today.")
        return
    
    print(f"{len(candidate_ids)} candidates after filtering out already-ingested")
    
    # Step 3: Fetch TMDB details for candidates we haven't seen before.
    # For candidates we've already fetched TMDB data for, re-fetch to get
    # current vote_count/popularity (they may have changed since bulk ingest).
    for tmdb_id in candidate_ids:
        db.execute(
            "INSERT OR REPLACE INTO movie_progress (tmdb_id, status) VALUES (?, 'pending')",
            (tmdb_id,)
        )
    db.commit()
    
    await stage2_tmdb_fetch(db)  # Processes all 'pending' rows
    
    # Step 4: Apply release date filter + hard filters + quality scoring.
    # This is where the heavy filtering happens.
    newly_fetched = db.execute(
        "SELECT tmdb_id FROM movie_progress WHERE status = 'tmdb_fetched'"
    ).fetchall()
    
    promoted = []
    skipped_release_date = 0
    skipped_hard_filter = 0
    skipped_quality = 0
    
    for (tmdb_id,) in newly_fetched:
        tmdb_data = load_json(f"./ingestion_data/tmdb/{tmdb_id}.json")
        
        # Release date age cutoff — the primary daily filter
        if not passes_release_date_filter(tmdb_data.get("release_date")):
            # Don't even log these — they're expected noise from the changes feed.
            # Just reset to filtered_out silently.
            db.execute(
                "UPDATE movie_progress SET status = 'filtered_out' WHERE tmdb_id = ?",
                (tmdb_id,)
            )
            skipped_release_date += 1
            continue
        
        # Hard filters (no IMDB ID, no overview, etc.)
        filter_reason = check_hard_filters(tmdb_data)
        if filter_reason:
            log_filter(db, tmdb_id, "tmdb_quality_funnel", filter_reason)
            skipped_hard_filter += 1
            continue
        
        # Quality scoring against the saved cutoff
        score = compute_quality_score(tmdb_data)
        if score >= cutoff_score:
            db.execute(
                """UPDATE movie_progress 
                   SET status = 'quality_passed', quality_score = ? WHERE tmdb_id = ?""",
                (score, tmdb_id)
            )
            promoted.append(tmdb_id)
        else:
            db.execute(
                """UPDATE movie_progress 
                   SET status = 'below_quality_cutoff', quality_score = ? WHERE tmdb_id = ?""",
                (score, tmdb_id)
            )
            skipped_quality += 1
    
    db.commit()
    
    print(f"  Release date filter: {skipped_release_date:,} skipped (not recent enough)")
    print(f"  Hard filters:        {skipped_hard_filter:,} skipped")
    print(f"  Below quality cutoff:{skipped_quality:,} skipped")
    print(f"  Promoted:            {len(promoted):,}")
    
    if not promoted:
        print("No new movies passed all filters today.")
        return
    
    print(f"Running full pipeline for {len(promoted)} movies...")
    
    # Step 5: Full pipeline for promoted movies (IMDB → LLM → embed → ingest)
    # For small daily batches (<100 movies), use real-time LLM calls instead
    # of the Batch API — the cost difference is negligible and you avoid 24hr waits.
    await stage4_imdb_scrape(db)
    await daily_llm_generate(db)  # Phase 1 + Phase 2 in real-time
    await stage7_embed(db)
    await stage8_ingest(db)
    
    print(f"Daily update complete: {len(promoted)} new movies ingested.")
```

### Why `INSERT OR REPLACE` for Step 3

When a movie shows up in the changes feed and we've seen it before (e.g., it was `filtered_out` or `below_quality_cutoff` from a previous run), we reset it to `pending` so `stage2_tmdb_fetch` re-fetches its TMDB data. This is intentional:

- The movie's vote_count, popularity, and other signals may have changed since we last evaluated it
- Re-fetching TMDB data is free (just rate-limited) and gives us current scores
- The release date filter and quality cutoff then re-evaluate with fresh data
- If it still doesn't qualify, it goes right back to `filtered_out` or `below_quality_cutoff`

The only status we protect is `ingested` — once a movie is fully in our databases, the daily pipeline doesn't touch it.

### Key Differences from Bulk Pipeline

| Aspect | Bulk Ingest | Daily Update |
|--------|------------|--------------|
| **Movie discovery** | TMDB bulk export (1M+ IDs) | TMDB changes endpoint (~100s of IDs) |
| **Release date filter** | None (evaluate everything) | Released within last 8 weeks only |
| **Quality threshold** | Top 100K by rank | Score >= saved cutoff |
| **LLM generation** | Batch API (50% cheaper, 24hr wait) | Real-time API (costs more per call, but volume is tiny) |
| **IMDB scraping** | Residential proxies required | Still proxied, but volume so small cost is negligible |
| **Runs on** | Local laptop (one-time) | EC2 cron job (daily) |
| **Duration** | ~3-4 days | ~10-30 minutes |

### Handling `below_quality_cutoff` Movies (Monthly Re-evaluation)

The daily pipeline's release date filter intentionally ignores older movies. But some movies gain popularity long after release (viral moments, streaming debuts, anniversary resurgence). Monthly, you can re-score all `below_quality_cutoff` movies to catch these:

```python
async def monthly_reeval_cutoff(db):
    """Re-evaluate movies that were below cutoff — they may have gained popularity."""
    
    below_cutoff = db.execute(
        "SELECT tmdb_id FROM movie_progress WHERE status = 'below_quality_cutoff'"
    ).fetchall()
    
    cutoff = load_json("./ingestion_data/quality_cutoff.json")
    cutoff_score = cutoff["cutoff_score"]
    
    # Re-fetch TMDB data for these movies (vote_count/popularity may have changed)
    for (tmdb_id,) in below_cutoff:
        db.execute("UPDATE movie_progress SET status = 'pending' WHERE tmdb_id = ?", (tmdb_id,))
    db.commit()
    
    await stage2_tmdb_fetch(db)  # Re-fetches TMDB details
    
    # Re-score and promote any that now pass
    # (reuses the quality check logic — no release date filter for monthly re-eval)
```

This creates a natural promotion path: a movie starts below the cutoff → gains streaming popularity → gets re-scored monthly → passes the threshold → gets the full IMDB + LLM treatment.