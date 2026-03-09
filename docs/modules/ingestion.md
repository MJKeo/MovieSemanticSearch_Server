# movie_ingestion/ — Ingestion Pipeline

Multi-stage pipeline that processes ~1M TMDB movies down to ~100K
high-quality movies through fetching, quality filtering, and IMDB
enrichment. All stages are crash-safe and idempotent.

## What This Module Does

Manages the first four stages of the ingestion pipeline: TMDB daily
export download, TMDB detail fetching, quality scoring/filtering,
and IMDB scraping. Stages 5+ (LLM generation, embedding, database
ingestion) live outside this module.

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
| `imdb_quality_scoring/analyze_imdb_quality.py` | Diagnostic: per-field coverage and distribution report for scraped IMDB data. |

## Boundaries

- **In scope**: TMDB export, TMDB detail fetching, quality scoring,
  IMDB scraping, pipeline state tracking.
- **Out of scope**: LLM metadata generation (implementation/llms/),
  embedding (implementation/vectorize.py), database ingestion
  (db/ingest_movie.py).

## Pipeline Stages

```
Stage 1: TMDB Daily Export     ~1M → ~800K     (~2 min)
Stage 2: TMDB Detail Fetch     ~800K fetched   (~1-8 hrs)
Stage 3: Quality Funnel        ~800K → ~100K   (~5 min)
Stage 4: IMDB Scraping         ~100K enriched  (~4-8 hrs)
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

## Tracker System

The `tracker.py` module is the shared backbone. Key rules:
- Always use `log_filter()` or `batch_log_filter()` for filtering —
  they atomically update both `filter_log` and `movie_progress`.
- Never write to `filter_log` or update status to `filtered_out`
  directly from stage modules.
- `filter_log` does NOT store `title` or `year` — JOIN on `tmdb_data`
  to get those when needed for display.
- Status progression: `pending` → `tmdb_fetched` →
  `tmdb_quality_passed` → `imdb_scraped` → `phase1_complete` →
  `phase2_complete` → `embedded` → `ingested`
- Terminal statuses: `filtered_out`, `below_quality_cutoff`

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
