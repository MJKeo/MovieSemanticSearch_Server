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
| `tmdb_quality_scoring/tmdb_quality_scorer.py` | Stage 3 scorer: edge cases (unreleased → 0.0, has providers → 1.0) + 4-signal weighted formula for no-provider movies. |
| `tmdb_quality_scoring/tmdb_filter.py` | Stage 3 filter: apply quality score threshold (0.2344) to scored movies. |
| `tmdb_quality_scoring/tmdb_data_analysis.py` | Diagnostic: per-attribute distributions from tmdb_data, split into two output files by watch-provider availability (`tmdb_data_analysis_with_providers.json` and `tmdb_data_analysis_no_providers.json`). |
| `tmdb_quality_scoring/plot_tmdb_quality_scores.py` | Diagnostic: survival curve + derivative analysis for Stage 3 scores. Plots both all-movies and no-provider-only populations using `survival_curve_utils`. |
| `imdb_scraping/run.py` | Stage 4 entry point: batch orchestration with commit-per-batch. HTTP fetching and DB writes are separated — async tasks return result NamedTuples, all DB writes happen via `executemany` after each batch. |
| `imdb_scraping/scraper.py` | Per-movie: fetch GraphQL → transform → return result (does not write to DB). |
| `imdb_scraping/http_client.py` | Async GraphQL client with proxy, retry, semaphore, random UA rotation. |
| `imdb_scraping/parsers.py` | GraphQL response → `IMDBScrapedMovie` transformer. |
| `imdb_scraping/models.py` | Pydantic models for IMDB scraped data. |
| `imdb_scraping/fix_stale_statuses.py` | One-off reconciliation script for stuck `tmdb_quality_passed` movies. |
| `imdb_scraping/reconcile_cached.py` | Advances `tmdb_quality_passed` movies to `imdb_scraped` when their IMDB JSON already exists on disk — recovers from runs that wrote the cache file but crashed before committing the status update. |
| `imdb_quality_scoring/imdb_quality_scorer.py` | Stage 5: 8-signal combined TMDB+IMDB quality scorer (v2). No hard filters — score is the sole filtering mechanism. |
| `imdb_quality_scoring/analyze_imdb_quality.py` | Diagnostic: per-field coverage and distribution report for scraped IMDB data, split into 3 groups matching the Stage 5 threshold groups (has_providers, recent_no_providers, old_no_providers). Produces `imdb_data_analysis_{group}.json` output files. |
| `imdb_quality_scoring/plot_quality_scores.py` | Diagnostic: survival curve + derivative analysis for Stage 5 scores across 3 groups (with providers, no providers recent, no providers old). Thin wrapper around `survival_curve_utils`. |
| `imdb_quality_scoring/sample_threshold_candidates.py` | Diagnostic: samples 15 movies below and above each of N candidate thresholds, writes full TMDB+IMDB data to `ingestion_data/threshold_candidate_samples.json` for manual review. |
| `imdb_quality_scoring/top_no_providers.py` | Diagnostic: lists top-scoring movies with no US watch providers (>1yr past theater window) to audit watch_providers signal distortion. |
| `scoring_utils.py` | Shared scoring utilities: `unpack_provider_keys()`, `score_vote_count()`, `score_popularity()`, `validate_weights()`, age-adjustment constants. Used by both Stage 3 and Stage 5 scorers. |
| `survival_curve_utils.py` | Shared Gaussian-smoothed survival curve plotting utility. Provides normalization, zero-crossing detection, survival count interpolation at extrema, and parameterized plotting. Used by the TMDB and IMDB `plot_quality_scores.py` wrappers. |

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

Stage 3 is deliberately lenient — it removes obvious gunk, not
borderline candidates. The real quality gate is Stage 5 after
IMDB data is available. Scoring and filtering are separate scripts
(run scorer first, then filter). See ADR-017 for design rationale
and the decision to simplify from the original 10-signal model.

**Edge cases** (bypass the formula entirely):
- Unreleased movies (release_date > today) → automatic score 0.0
- Movies with ≥1 US watch provider → automatic score 1.0

**4-signal weighted formula** (no-provider population only, weights
sum to 1.0, all outputs in [0, 1]):

| Signal | Weight | Scoring |
|--------|--------|---------|
| vote_count | 0.50 | Log-scaled, log cap 101 (calibrated to no-provider p99=72). Recency boost (up to 2x for <2yr), classic boost (up to 1.5x for >20yr) |
| popularity | 0.20 | Log-scaled, log cap 11 |
| overview_length | 0.15 | Tiered by character count (0→0.0, 1–50→0.2, 51–100→0.5, 101–200→0.8, 201+→1.0) |
| data_completeness | 0.15 | Average of 8 binary metadata indicators (genres, poster, cast/crew, countries, companies, keywords, budget, revenue) |

**Soft threshold**: stage_3_quality_score < 0.2344 (inflection point
from survival curve derivative analysis on the no-provider population)

**Status progression**: tmdb_fetched → tmdb_quality_calculated (scored)
→ tmdb_quality_passed (filtered)

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

**Proxy tuning (residential proxies)**: Successful fetches complete
in <1s. With residential proxies, each retry arrives from a fresh IP,
so fast failure and immediate retry is better than exponential backoff.
Current constants: request timeout 5s (fail fast without over-rotation),
flat retry delay 0.2–0.3s (vs. former exponential 2^n + rand), semaphore
60 (increasing to 100 raised timeout rates without throughput gain —
bottleneck is IP quality, not concurrency). See ADR-018 for the
residential-vs-datacenter tuning tradeoffs.

## Stage 5: Combined Quality Scoring Model (v2)

Stage 5 computes a combined TMDB+IMDB quality score for every
`imdb_scraped` movie. IMDB data is primary; TMDB is fallback for
overlapping fields. See `quality_score_design.md` for full design
rationale. See ADR-019 for the v2 redesign decisions.

**No hard filters.** The quality score is the sole filtering mechanism.
Movies with missing IMDB JSON are skipped (status unchanged).

The score answers two questions: (1) is this movie relevant — would
users search for it or choose it? (2) is the data sufficient for
reliable LLM metadata generation and multi-channel search?

**Three separate thresholds** (determined after scoring via
survival-curve analysis, applied per provider group):
- **has_providers** — movies with ≥1 US watch provider (lenient)
- **recent_no_providers** — no providers, released ≤75 days ago (lenient)
- **old_no_providers** — no providers, released >75 days ago (strictest)

**Status progression:** `imdb_scraped` → `imdb_quality_calculated` (scored)
→ `imdb_quality_passed` (after threshold filtering).
Stage name for filter_log: `imdb_quality_funnel`.
See ADR-020 for the rationale behind the two-step status pattern.

### 8-signal weighted model (weights sum to 1.0, all [0, 1])

| Signal | Weight | Category | Scale |
|--------|--------|----------|-------|
| imdb_vote_count | 0.27 | Relevance | Log-scaled + recency/classic adj. |
| critical_attention | 0.12 | Relevance | Presence of metacritic + reception_summary |
| community_engagement | 0.08 | Relevance | Weighted presence of reviews/synopses/summaries/keywords |
| tmdb_popularity | 0.08 | Relevance | Log-scaled (lowered cap, ~p75 saturates) |
| featured_reviews_chars | 0.15 | Data sufficiency | Tiered by total char count |
| plot_text_depth | 0.12 | Data sufficiency | Log-scaled composite |
| lexical_completeness | 0.10 | Data sufficiency | Averaged entity-type composite |
| data_completeness | 0.08 | Data sufficiency | Averaged field composite |

Score range: [0, 1].

### Signal details

**imdb_vote_count (0.27)** — Primary notability proxy. IMDB only.
Log cap 12,001. Age multipliers: recency boost up to 2.0x for
films <2yr, classic boost up to 1.5x for >20yr. Unchanged from v1.

**critical_attention (0.12)** — Count presence of `metacritic_rating`
(not None) + `reception_summary` (truthy string). 0/2→0.0, 1/2→0.5,
2/2→1.0. Pure bonus — absence is normal.

**community_engagement (0.08)** — Weighted presence composite.
Sub-weights: plot_keywords→1, featured_reviews→2 (IMDB or TMDB),
plot_summaries→3, synopses→4. Score = sum of present sub-weights / 10.

**tmdb_popularity (0.08)** — Log-scaled TMDB activity score. Lowered
log cap (STAGE5_POP_LOG_CAP=4.0) so ~p75 of has_providers saturates.

**featured_reviews_chars (0.15)** — Total chars across review texts.
IMDB primary; TMDB fallback. Tiers: 0→0.0, 1–3K→0.33, 3K–8K→0.67,
8K+→1.0.

**plot_text_depth (0.12)** — Log-scaled composite of overview +
plot_summaries + synopses. IMDB primary; TMDB overview_length fallback.
Log cap 5,001. Unchanged from v1.

**lexical_completeness (0.10)** — 6 entity types averaged to [0, 1].
Actors/characters: 0→0.0, 1–4→0.5, 5+→1.0. Writers, composers,
producers: binary. Production companies: IMDB primary, TMDB fallback.

**data_completeness (0.08)** — 6 fields averaged to [0, 1].
plot_keywords: tiered. overall_keywords: tiered. filming_locations,
parental_guide_items: binary. maturity_rating, budget: IMDB/TMDB.

### Changes from v1

- Removed all 10 hard filters — score is the sole filter
- Removed `watch_providers` signal (0.20) — handled by per-group thresholds
- Removed `metacritic_rating` signal (0.04) — folded into `critical_attention`
- Added `critical_attention` (0.12) and `community_engagement` (0.08)
- All signals normalised to [0, 1] (was mixed [-1, +1] and [0, 1])
- Score range now [0, 1] (was ~[-0.55, +0.95])
- Three thresholds per provider group (was single threshold)

### Shared scoring utilities (scoring_utils.py)

`score_vote_count()` and `score_popularity()` are shared between
Stage 3 and Stage 5 via `scoring_utils.py`. `score_popularity()` accepts
an optional `log_cap` parameter; Stage 5 passes `STAGE5_POP_LOG_CAP=4.0`.
Stage 3 uses the default (11). Age-adjustment constants and
`validate_weights()` are also shared.

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
| reception | featured_reviews, critical_attention |

Non-LLM channels: **lexical search** ← lexical_completeness;
**metadata scoring** ← imdb_vote_count, tmdb_popularity.

## Tracker System

The `tracker.py` module is the shared backbone. Key rules:
- Always use `log_filter()` or `batch_log_filter()` for filtering —
  they atomically update both `filter_log` and `movie_progress`.
- Never write to `filter_log` or update status to `filtered_out`
  directly from stage modules.
- `filter_log` does NOT store `title` or `year` — JOIN on `tmdb_data`
  to get those when needed for display.
- Status progression: `pending` → `tmdb_fetched` →
  `tmdb_quality_calculated` → `tmdb_quality_passed` → `imdb_scraped` →
  `imdb_quality_calculated` → `imdb_quality_passed` → `phase1_complete` →
  `phase2_complete` → `embedded` → `ingested`
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
- IMDB blocks datacenter proxy IPs while accepting residential
  ones. The bulk scrape already uses DataImpulse residential proxies
  (see ADR-015). For the future daily-update pipeline, ensure
  residential proxies remain configured.
  See ADR-015 and `memory/imdb-scraping.md` for tuning history.
- Survival curve plot scripts call `plt.show()` which blocks in
  headless environments. Use the `Agg` backend or redirect output
  to a file when running without a display.
