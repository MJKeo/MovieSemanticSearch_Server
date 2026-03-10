# movie_ingestion/ — Ingestion Pipeline

Multi-stage pipeline that processes ~1M TMDB movies down to ~100K
high-quality movies through fetching, quality filtering, and IMDB
enrichment. All stages are crash-safe and idempotent.

## What This Module Does

Manages the first five stages of the ingestion pipeline: TMDB daily
export download, TMDB detail fetching, quality scoring/filtering,
IMDB scraping, and IMDB quality filtering. Stages 6+ (LLM generation,
embedding, database ingestion) live outside this module.

## Key Files

| File | Purpose |
|------|---------|
| `tracker.py` | Shared backbone — SQLite database at `./ingestion_data/tracker.db`. Manages `movie_progress`, `filter_log`, and `tmdb_data` tables. Defines `MovieStatus`/`PipelineStage` enums. Provides `log_filter()` and `batch_log_filter()` helpers. |
| `tmdb_fetching/daily_export.py` | Stage 1: Stream-download gzipped JSONL (~1M entries), filter (adult=False, video=False, popularity>0), insert ~800K as 'pending'. |
| `tmdb_fetching/tmdb_fetcher.py` | Stage 2: Async TMDB detail fetch for all pending movies. Extracts fields into `tmdb_data` table. Filters movies missing IMDB ID. Uses adaptive rate limiting from `db/tmdb.py`. HTTP fetching and DB writes are separated: async tasks return result NamedTuples, all DB writes happen via `executemany` after `asyncio.gather()`. |
| `tmdb_quality_scoring/tmdb_filter.py` | Stage 3: Apply 5 hard filters + quality score threshold. |
| `tmdb_quality_scoring/tmdb_quality_scorer.py` | Quality scoring model: 10 weighted signals (weights sum to 1.0). |
| `tmdb_quality_scoring/tmdb_data_analysis.py` | Diagnostic: per-attribute distributions from tmdb_data. |
| `tmdb_quality_scoring/plot_quality_scores.py` | Diagnostic: Gaussian-smoothed survival curve + derivatives (determines threshold). |
| `imdb_scraping/run.py` | Stage 4 entry point: batch orchestration with commit-per-batch. HTTP fetching and DB writes are separated — async tasks return result NamedTuples, all DB writes happen via `executemany` after each batch. |
| `imdb_scraping/scraper.py` | Per-movie: fetch GraphQL → transform → return result (does not write to DB). |
| `imdb_scraping/http_client.py` | Async GraphQL client with proxy, retry, semaphore, random UA rotation. |
| `imdb_scraping/parsers.py` | GraphQL response → `IMDBScrapedMovie` transformer. |
| `imdb_scraping/models.py` | Pydantic models for IMDB scraped data. |
| `imdb_scraping/fix_stale_statuses.py` | One-off reconciliation script for stuck `tmdb_quality_passed` movies. |
| `imdb_quality_scoring/imdb_quality_scorer.py` | Stage 5: IMDB essential data hard-filter + quality score soft threshold. |
| `imdb_quality_scoring/analyze_imdb_quality.py` | Diagnostic: per-field coverage and distribution report for scraped IMDB data. |
| `scoring_utils.py` | Shared scoring utilities: `unpack_provider_keys()`, `score_vote_count()`, `score_popularity()`, `validate_weights()`, age-adjustment constants. Used by both Stage 3 and Stage 5 scorers. |

## Boundaries

- **In scope**: TMDB export, TMDB detail fetching, quality scoring,
  IMDB scraping, pipeline state tracking.
- **Out of scope**: LLM metadata generation (Stage 6,
  implementation/llms/), embedding (Stage 7,
  implementation/vectorize.py), database ingestion (Stage 8,
  db/ingest_movie.py).

## Pipeline Stages

```
Stage 1: TMDB Daily Export     ~1M → ~800K     (~2 min)
Stage 2: TMDB Detail Fetch     ~800K fetched   (~1-8 hrs)
Stage 3: Quality Funnel        ~800K → ~100K   (~5 min)
Stage 4: IMDB Scraping         ~100K enriched  (~4-8 hrs)
Stage 5: IMDB Quality Filter   ~100K filtered  (~5 min)
```

## Stage 3: Quality Scoring Model

**Hard filters** (applied first, any failure = filtered_out):
1. `zero_vote_count` — vote_count = 0
2. `missing_or_zero_duration` — duration IS NULL or 0
3. `missing_overview` — overview_length = 0
4. `no_genres` — genre_count = 0
5. `future_release` — release_date > today

**Soft threshold**: quality_score < -0.0441

**10-signal weighted model** (weights sum to 1.0):

| Signal | Weight | Scoring |
|--------|--------|---------|
| vote_count | 0.38 | Log-scaled, capped at 2000. Recency boost (up to 2x, peaking at ≤1yr, decaying to 1x at 2yr), classic boost (1.5x for >20yr) |
| watch_providers | 0.25 | Tiered [-1, +1]. Harsh -1 for no US streaming |
| popularity | 0.12 | Log-scaled, capped at 10 |
| has_revenue | 0.05 | Rare (7%), bonus when present |
| poster_url | 0.05 | Common (95%), penalty when absent |
| overview_length | 0.04 | Tiered by character count |
| has_keywords | 0.04 | Symmetric [-0.5, +0.5] |
| has_production_companies | 0.03 | Penalty when absent |
| has_budget | 0.02 | Bonus when present |
| has_cast_and_crew | 0.02 | Penalty when absent |

## Stage 4: IMDB Scraping

Uses a single GraphQL query to `api.graphql.imdb.com` per movie
(replaces the original 6 HTML page fetch approach). Routed through
DataImpulse residential proxies with US geo-targeting.

**Extracts**: credits (directors, writers, actors, characters,
producers, composers), keywords (with community vote scoring),
synopses/plot summaries, parental guide, featured reviews,
maturity rating, reception data, filming locations, budget,
languages, countries, production companies.

**Keyword scoring formula**: `score = usersInterested - 0.75 * dislikes`.
Dynamic threshold: `min(0.75 * N, N - 2)` where N = top keyword score.
Floor of 5, cap of 15 keywords per movie.

**Output**: Per-movie JSON at `ingestion_data/imdb/{tmdb_id}.json`.

**Proxy tuning**: Successful fetches complete in <1s; the request
timeout is set aggressively (2s) to fail fast on flagged IPs and
trigger rotation. Optimal semaphore ceiling is ~35; beyond that,
timeout rates increase without throughput gain — the bottleneck is
IP quality, not parallelism.

## Stage 5: Combined Quality Scoring Model

Stage 5 applies both hard filters (essential data checks) and a combined
TMDB+IMDB quality scorer. IMDB data is primary; TMDB is fallback for
overlapping fields. See ADR-016 for design rationale and alternatives
considered.

The score answers two questions: (1) does this movie have sufficient data
for high-quality LLM metadata generation across the 7 vector spaces?
(2) is this movie a credible search target worth including?

**Hard filters** (applied first, any failure = filtered_out):
1. `missing_imdb_json` — no JSON file on disk
2. `no_imdb_rating` — no audience engagement signal
3. `no_directors` — fundamental lexical entity missing
4. `no_poster_url` — display requirement
5. `no_actors` — critical lexical entity
6. `no_characters` — character-based search support
7. `no_overall_keywords` — limits lexical and semantic signals
8. `no_languages` — metadata filtering requirement
9. `no_countries_of_origin` — region-based query support
10. `no_release_date` — date-range filtering requirement

### 8-signal weighted model (weights sum to 1.0)

| Signal | Weight | Range | Scale |
|--------|--------|-------|-------|
| imdb_vote_count | 0.22 | [0, 1] | Log-scaled + recency/classic adj. |
| watch_providers | 0.20 | [-1, +1] | Binary with theater-window logic |
| featured_reviews_chars | 0.16 | [-1, +1] | Tiered by total char count |
| plot_text_depth | 0.12 | [0, 1] | Log-scaled composite |
| lexical_completeness | 0.10 | [-1, +1] | Capped entity-type composite |
| data_completeness | 0.10 | [-1, +1] | Tiered/binary field composite |
| tmdb_popularity | 0.06 | [0, 1] | Log-scaled |
| metacritic_rating | 0.04 | [0, +1] | Binary bonus |

Raw score range: approximately -0.55 to +0.95.

**Soft threshold**: determined via survival-curve derivative analysis
(same approach as Stage 3's `plot_quality_scores.py`).

### Signal details

**imdb_vote_count (0.22)** — Primary notability proxy. IMDB only (100%
present, no TMDB fallback). Log cap at 12,001 (just above p90).
Formula: `min(log10(vc + 1) / log10(12001), 1.0)`. Two age multipliers
(mutually exclusive, larger applied): recency boost up to 2.0x for
films < 2yr (hyperbolic decay, floored at 0.5yr), classic boost
linearly ramping from 1.0x at 20yr to 1.5x cap at 35yr. Films 2–20yr
receive no adjustment. Result clamped to [0, 1].

**watch_providers (0.20)** — Binary: +1 if ≥1 US provider or within
75-day theater window; -1 otherwise. Null release date conservatively
treated as past theater window. The -0.20 penalty makes it nearly
impossible for unwatchable movies to survive.

**featured_reviews_chars (0.16)** — Total chars across all review texts.
IMDB `featured_reviews` primary; TMDB reviews JSON fallback. Reviews
feed 6/7 vector spaces — absence is a red flag. Tiers: 0 chars → -1.0,
1–3,000 → 0.0, 3,001–8,000 → 0.5, 8,001+ → 1.0.

**plot_text_depth (0.12)** — Log-scaled composite of overview +
plot_summaries + synopses total chars. IMDB primary per component;
TMDB `overview_length` fallback for overview only. Log cap: 5,001.
Formula: `min(log10(total + 1) / log10(5001), 1.0)`. These fields
are substitutes — total text budget matters, not which field contributes.

**lexical_completeness (0.10)** — 6 entity types, each capped at 0–1.0
sub-score, total mapped [0, 6] → [-1, +1] via `(total - 3) / 3`.
Actors/characters: <5 → 0.5, 5+ → 1.0. Writers, composers, producers:
binary (0 → 0.0, 1+ → 1.0). Production companies: IMDB primary, TMDB
`has_production_companies` fallback (binary).

**data_completeness (0.10)** — 6 vector-search-readiness fields, mapped
[0, 6] → [-1, +1] via `(total - 3) / 3`. plot_keywords: 0 → 0.0,
1–4 → 0.5, 5+ → 1.0. overall_keywords: 1 → 0.25, 2–3 → 0.5, 4+ → 1.0.
filming_locations, parental_guide_items: binary. maturity_rating, budget:
IMDB primary, TMDB fallback (binary).

**tmdb_popularity (0.06)** — Log-scaled TMDB activity score. Log cap: 11
(just above p99). Formula: `min(log10(pop + 1) / log10(11), 1.0)`.
Captures short-term buzz/momentum complementary to vote count.

**metacritic_rating (0.04)** — Binary bonus (0.0 absent, 1.0 present).
At 15.2% presence, indicates professional critical coverage.

### Shared scoring utilities (scoring_utils.py)

`score_vote_count()` and `score_popularity()` are shared between
Stage 3 and Stage 5 via `scoring_utils.py`. Stage 3 passes
`VoteCountSource.TMDB` (log cap 2,001), Stage 5 passes
`VoteCountSource.IMDB` (log cap 12,001). Age-adjustment constants
(`THEATER_WINDOW_DAYS=75`, `VC_RECENCY_BOOST_MAX=2.0`,
`VC_CLASSIC_START_YEARS=20`, `VC_CLASSIC_RAMP_YEARS=30`,
`VC_CLASSIC_BOOST_CAP=1.5`) and `validate_weights()` are also shared.

### Downstream vector space coverage

How the scored fields map to the 7 LLM-generated vector spaces:

| Vector Space | Scored Input Fields |
|---|---|
| plot_events | plot_text_depth, data_completeness (plot_keywords) |
| plot_analysis | plot_text_depth, featured_reviews, data_completeness |
| viewer_experience | featured_reviews, data_completeness (parental_guide, maturity_rating, overall_keywords, plot_keywords) |
| watch_context | featured_reviews, data_completeness (overall_keywords, plot_keywords) |
| narrative_techniques | featured_reviews, plot_text_depth, data_completeness (overall_keywords, plot_keywords) |
| production | featured_reviews, plot_text_depth, data_completeness (overall_keywords, plot_keywords, filming_locations) |
| reception | featured_reviews, metacritic_rating |

Non-LLM channels: **lexical search** ← lexical_completeness;
**metadata scoring** ← imdb_vote_count, watch_providers, tmdb_popularity.

## Tracker System

The `tracker.py` module is the shared backbone. Key rules:
- Always use `log_filter()` or `batch_log_filter()` for filtering —
  they atomically update both `filter_log` and `movie_progress`.
- Never write to `filter_log` or update status to `filtered_out`
  directly from stage modules.
- `filter_log` does NOT store `title` or `year` — JOIN on `tmdb_data`
  to get those when needed for display.
- Status progression: `pending` → `tmdb_fetched` →
  `tmdb_quality_passed` → `imdb_scraped` → `essential_data_passed` →
  `phase1_complete` → `phase2_complete` → `embedded` → `ingested`
- Terminal statuses: `filtered_out`

### Durability settings

`init_db()` enables both `PRAGMA journal_mode=WAL` and
`PRAGMA synchronous=FULL`. FULL sync ensures every commit is
fsynced to disk, preventing corruption if the process is killed
during a batch. Performance cost is negligible at the ~500-movie
commit cadence used by Stages 2 and 4.

## HTTP / DB Write Separation Pattern

Stages 2 and 4 follow the same pattern to avoid mixing async HTTP
with synchronous SQLite writes:

1. `asyncio.gather()` dispatches all HTTP fetches; each task returns
   a NamedTuple (success or failure) rather than writing to the DB.
2. After gather completes, all results are written to SQLite in bulk
   via `executemany`.
3. A single `db.commit()` closes the batch.

This was chosen over aiosqlite, semaphore-guarded per-row writes,
or write queues because it is simpler, easier to reason about, and
eliminates any uncertainty about concurrent DB access.

## Gotchas

- TMDB fetching is free (API key only). IMDB scraping requires
  proxies (~$5-17 for 100K movies). This cost asymmetry drives
  the TMDB-first funnel design.
- The SQLite tracker serves both as checkpoint DB and data store
  for Stage 3 quality scoring. Single file at ~310-360 MB.
- Watch provider keys in `tmdb_data` are stored as packed uint32
  BLOBs, not JSON arrays.
- IMDB scraping uses graceful defaults — no movie is filtered out
  based on missing IMDB data. All fields default to None/[].
- DataImpulse proxy requires `DATA_IMPULSE_LOGIN` and
  `DATA_IMPULSE_PASSWORD` in `.env`.
- IMDB can block datacenter proxy IPs while accepting residential
  ones. Symptom: mass ConnectTimeouts and tiny response payloads
  (1-2 MB/min vs 18+ MB/min when healthy). Switching to residential
  proxies resolves this — planned for the daily-update pipeline.
  See ADR-015 and `memory/imdb-scraping.md` for tuning history.
