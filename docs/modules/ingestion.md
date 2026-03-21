# movie_ingestion/ — Ingestion Pipeline

Multi-stage pipeline that processes ~1M TMDB movies down to ~100K
high-quality movies through fetching, quality filtering, and IMDB
enrichment. All stages are crash-safe and idempotent.

## What This Module Does

Manages the first five stages of the ingestion pipeline: TMDB daily
export download, TMDB detail fetching, quality scoring/filtering,
IMDB scraping, and IMDB quality filtering. Also houses Stage 6
(LLM metadata generation via Batch API) in the `metadata_generation/`
subpackage. Stages 7+ (embedding, database ingestion) live outside
this module.

## Key Files

| File | Purpose |
|------|---------|
| `tracker.py` | Shared backbone — SQLite database at `./ingestion_data/tracker.db`. Manages `movie_progress`, `filter_log`, `tmdb_data`, and `imdb_data` tables. Defines `MovieStatus`/`PipelineStage` enums. Provides `log_filter()`, `batch_log_filter()`, `serialize_imdb_movie()`, `deserialize_imdb_row()`, and `IMDB_DATA_COLUMNS`. |
| `tmdb_fetching/daily_export.py` | Stage 1: Stream-download gzipped JSONL (~1M entries), filter (adult=False, video=False, popularity>0), insert ~800K as 'pending'. |
| `tmdb_fetching/tmdb_fetcher.py` | Stage 2: Async TMDB detail fetch for all pending movies. Extracts fields into `tmdb_data` table. Filters movies missing IMDB ID. Uses adaptive rate limiting from `db/tmdb.py`. HTTP fetching and DB writes are separated: async tasks return result NamedTuples, all DB writes happen via `executemany` after `asyncio.gather()`. |
| `tmdb_quality_scoring/tmdb_quality_scorer.py` | Stage 3 scorer: edge cases (unreleased → 0.0, has providers → 1.0) + 4-signal weighted formula for no-provider movies. |
| `tmdb_quality_scoring/tmdb_filter.py` | Stage 3 filter: apply quality score threshold (0.2344) to scored movies. |
| `tmdb_quality_scoring/tmdb_data_analysis.py` | Diagnostic: per-attribute distributions from tmdb_data, split into two output files by watch-provider availability (`tmdb_data_analysis_with_providers.json` and `tmdb_data_analysis_no_providers.json`). |
| `tmdb_quality_scoring/plot_tmdb_quality_scores.py` | Diagnostic: survival curve + derivative analysis for Stage 3 scores. Plots both all-movies and no-provider-only populations using `survival_curve_utils`. |
| `imdb_scraping/run.py` | Stage 4 entry point: batch orchestration with commit-per-batch. HTTP fetching and DB writes are separated — async tasks return result NamedTuples, all DB writes happen via `executemany` after each batch. |
| `imdb_scraping/scraper.py` | Per-movie: fetch GraphQL → transform → return result dict (does not write to DB). `MovieResult.data` is `dict | None` (not a JSON string). |
| `imdb_scraping/http_client.py` | Async GraphQL client with proxy, retry, semaphore, random UA rotation. Fetches `titleType { id }` as part of the standard query. |
| `imdb_scraping/parsers.py` | GraphQL response → `IMDBScrapedMovie` transformer. Extracts `imdb_title_type`, `plot_summaries` (always extracted independently from synopses), and `plot_keywords` (floor 10, cap 15). |
| `imdb_scraping/models.py` | Pydantic models for IMDB scraped data. `IMDBScrapedMovie` includes `imdb_title_type: str | None`. |
| `imdb_scraping/fix_stale_statuses.py` | One-off reconciliation script for stuck `tmdb_quality_passed` movies. Bulk-queries `imdb_data` table. |
| `imdb_scraping/reconcile_cached.py` | Advances `tmdb_quality_passed` movies to `imdb_scraped` when their IMDB data already exists in the `imdb_data` table — recovers from runs that wrote data but crashed before committing the status update. |
| `imdb_scraping/migrate_json_to_sqlite.py` | One-off migration script: reads per-movie JSON files from `ingestion_data/imdb/` and inserts rows into the `imdb_data` table via `serialize_imdb_movie()`. |
| `imdb_quality_scoring/imdb_quality_scorer.py` | Stage 5 scorer: two hard gates (title-type, missing-text) + 8-signal combined TMDB+IMDB quality scorer (v4). See ADR-037. Advances `imdb_scraped` → `imdb_quality_calculated`. |
| `imdb_quality_scoring/imdb_filter.py` | Stage 5 filter: applies per-group quality-score thresholds from `scoring_utils.IMDB_QUALITY_THRESHOLDS`. Advances `imdb_quality_calculated` → `imdb_quality_passed` (or `filtered_out`). |
| `imdb_quality_scoring/analyze_imdb_quality.py` | Diagnostic: per-field coverage and distribution report for scraped IMDB data, split into 3 groups matching the Stage 5 threshold groups (has_providers, recent_no_providers, old_no_providers). Produces `imdb_data_analysis_{group}.json` output files. |
| `imdb_quality_scoring/plot_quality_scores.py` | Diagnostic: survival curve + derivative analysis for Stage 5 scores across 3 groups (with providers, no providers recent, no providers old). Thin wrapper around `survival_curve_utils`. |
| `imdb_quality_scoring/sample_threshold_candidates.py` | Diagnostic: samples movies around each candidate threshold per group, writes full TMDB+IMDB data to per-group JSON files in `ingestion_data/` for manual review. |
| `metadata_generation/run.py` | Stage 6 CLI entry point: `submit --wave 1/2`, `status`, `process`. Two-wave Batch API flow. |
| `metadata_generation/wave1_runner.py` | Direct (non-batch) runner for Wave 1 generation. Generates plot_events and reception outputs for a movie set, stores results immediately to `wave1_results` table in tracker.db. Used to pre-generate Wave 1 outputs for Wave 2 evaluation. Async tasks return result tuples; all DB writes happen after `asyncio.gather()`. |
| `metadata_generation/state.py` | SQLite helpers for `metadata_batches` and `metadata_results` tables (added to tracker.db). |
| `metadata_generation/request_builder.py` | Assembles JSONL files for batch submission. Wraps generator output in `{custom_id, method, url, body}` format. |
| `metadata_generation/batch_manager.py` | OpenAI Files API + Batch API wrapper: upload, create, check status, download. No movie/generation knowledge. |
| `metadata_generation/result_processor.py` | Parses downloaded result JSONL, routes to generators, stores in `metadata_results`. Auto-submits Wave 2 after Wave 1 processing. |
| `metadata_generation/inputs.py` | `MovieInputData`, `ConsolidatedInputs`, `SkipAssessment` dataclasses + `build_user_prompt()` + `MultiLineList`. `MovieInputData` provides `merged_keywords()` (deduplicated union of plot + overall keywords) and `maturity_summary()` (delegates to `pre_consolidation.consolidate_maturity()`). Contract between SQLite loading and pre-consolidation. |
| `metadata_generation/schemas.py` | Pydantic output schemas for each LLM generation type. Base variants have justification fields removed; each Wave 2 type also has a `WithJustificationsOutput` variant for evaluation (identical `__str__()` to the base). `PlotEventsOutput` and `MajorCharacter` use minimal neutral field descriptions — behavioral instructions live in branch-specific prompts, not the schema (see ADR-036). `ReceptionOutput` includes `review_insights_brief` intermediate. See ADR-025. |
| `metadata_generation/pre_consolidation.py` | Pre-consolidation: keyword routing + normalization, maturity consolidation, eligibility checks (Wave 1: `check_plot_events`, `check_reception` — both public; Wave 2: 6 private `_check_*`), `assess_skip_conditions()` orchestrator, `run_pre_consolidation()` entry point. `source_of_inspiration` skip check uses only `merged_keywords` and `review_insights_brief` (no `plot_synopsis`). |
| `metadata_generation/analyze_eligibility.py` | Diagnostic: loads all `imdb_quality_passed` movies, runs Wave 1 + Wave 2 skip assessments, estimates token sizes, saves per-group eligibility report to `ingestion_data/eligibility_report.json`. |
| `metadata_generation/generators/` | 8 generator files (one per generation type). All fully implemented as real-time async callers. `plot_events.py`: two-branch generation (synopsis condensation vs. synthesis) — see Stage 6 section. All other generators default to OpenAI gpt-5-mini; Wave 2 generators accept `system_prompt`/`response_format` overrides for evaluation. See ADR-026, ADR-027. |
| `metadata_generation/evaluations/` | Reference-free pointwise evaluation pipeline for comparing LLM candidates before production commits. See ADR-028, ADR-034. |
| `metadata_generation/evaluations/shared.py` | `EvaluationCandidate` frozen dataclass, `EVALUATION_TEST_SET_TMDB_IDS` and subset lists (`ORIGINAL_SET_TMDB_IDS`, `MEDIUM_SPARSITY_TMDB_IDS`, `HIGH_SPARSITY_TMDB_IDS`) — 70 movies stratified by sparsity, `get_eval_connection()`, `create_candidates_table()`, `store_candidate()`, `load_movie_input_data()`, `compute_score_summary()` (supports `score_weights` for weighted overall_mean). |
| `metadata_generation/evaluations/plot_events.py` | Reference-free candidate evaluation (candidate generation + Opus 4.6 judge scoring, 2-run sequential with prompt caching) + `PLOT_EVENTS_CANDIDATES` list. 4-dimension judge rubric: groundedness, plot_summary, character_quality, setting. Accepts `--branch synopsis\|synthesis` flag to filter test corpus by branch. |
| `metadata_generation/evaluations/openai_oauth.py` | ChatGPT OAuth2 PKCE token lifecycle: browser consent, JWT decode for account_id/expiry, token persistence to `evaluation_data/openai_oauth_tokens.json`, auto-refresh. `get_valid_auth()` is the sole entry point for WHAM callers. (WHAM is no longer used by the plot_events evaluator — retained for any future WHAM callers.) |
| `metadata_generation/evaluations/run_evaluations_pipeline.py` | CLI entry point: loads 70-movie corpus, filters ineligible movies via `check_plot_events`, runs candidate evaluation. Accepts `--branch synopsis\|synthesis` flag. |
| `metadata_generation/evaluations/analyze_results.py` | Read-only analysis: merges scores + token counts + model pricing into quality and cost tables. Value ranking table shows separate dense/sparse cost/1K columns (dense = ORIGINAL_SET movies, sparse = MEDIUM+HIGH_SPARSITY). Run via `python -m movie_ingestion.metadata_generation.evaluations.analyze_results`. |
| `metadata_generation/prompts/` | 8 system prompt files (one per LLM call). Each prompt file exports a `SYSTEM_PROMPT` constant; Wave 2 generators also export `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` for evaluation. `plot_events.py` exports `SYSTEM_PROMPT_SYNOPSIS` and `SYSTEM_PROMPT_SYNTHESIS` for the two branches. |
| `scoring_utils.py` | Shared scoring utilities: `unpack_provider_keys()`, `score_vote_count()`, `score_popularity()`, `validate_weights()`, age-adjustment constants. Also the canonical group classification: `MovieGroup` enum, `classify_movie_group()`, `passes_imdb_quality_threshold()`, `IMDB_QUALITY_THRESHOLDS`, and SQL fragment constants (`HAS_PROVIDERS_SQL`, `NO_PROVIDERS_SQL`, `THEATER_WINDOW_SQL_PARAM`). |
| `survival_curve_utils.py` | Shared Gaussian-smoothed survival curve plotting utility. Provides normalization, zero-crossing detection, survival count interpolation at extrema, and parameterized plotting. Used by the TMDB and IMDB `plot_quality_scores.py` wrappers. |

## Boundaries

- **In scope**: TMDB export, TMDB detail fetching, quality scoring,
  IMDB scraping, pipeline state tracking, LLM metadata generation
  (Stage 6, `metadata_generation/` subpackage).
- **Out of scope**: Embedding (Stage 7, `implementation/vectorize.py`),
  database ingestion (Stage 8, `db/ingest_movie.py`).

## Pipeline Stages

```
Stage 1: TMDB Daily Export     ~1M → ~800K     (~2 min)
Stage 2: TMDB Detail Fetch     ~800K fetched   (~1-8 hrs)
Stage 3: Quality Funnel        ~800K → ~100K   (~5 min)
Stage 4: IMDB Scraping         ~100K enriched  (~4-8 hrs)
Stage 5: IMDB Quality Filter   ~100K filtered  (~5 min)
Stage 6: LLM Metadata Gen      ~112K movies    (Batch API, up to 48h)
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
languages, countries, production companies, and `imdb_title_type`
(IMDB's `titleType.id` — e.g. "movie", "videoGame", "tvSeries").

**Keyword scoring formula**: `score = usersInterested - 0.75 * dislikes`.
Dynamic threshold: `min(0.75 * N, N - 2)` where N = top keyword score.
Floor of 10, cap of 15 keywords per movie (`_MIN_PLOT_KEYWORDS = 10`,
`_MAX_PLOT_KEYWORDS = 15`).

**plot_summaries extraction**: Always extracted independently of synopses.
Previously, plot_summaries were discarded when a synopsis was present;
now both are extracted since they serve different downstream purposes
(synopses go to the condensation branch; summaries serve as fallback).

**Output**: Stored in `imdb_data` table in tracker.db (see ADR-023).
Previously per-movie JSON files at `ingestion_data/imdb/{tmdb_id}.json`;
those files are no longer the primary data store.

**Data serialization**: `scraper.py` returns `model_dump()` dict in
`MovieResult.data`. `run.py` calls `serialize_imdb_movie()` to convert
to INSERT tuple, then writes via `executemany`. Readers call
`deserialize_imdb_row()` to get back the same dict shape (JSON columns
parsed back to Python lists/dicts). `IMDB_DATA_COLUMNS` in `tracker.py`
is the single source of truth for column order.

**Proxy tuning (residential proxies)**: Successful fetches complete
in <1s. With residential proxies, each retry arrives from a fresh IP,
so fast failure and immediate retry is better than exponential backoff.
Current constants: request timeout 5s (fail fast without over-rotation),
flat retry delay 0.2–0.3s (vs. former exponential 2^n + rand), semaphore
60 (increasing to 100 raised timeout rates without throughput gain —
bottleneck is IP quality, not concurrency). See ADR-018 for the
residential-vs-datacenter tuning tradeoffs.

## Stage 5: Combined Quality Scoring Model (v4)

Stage 5 computes a combined TMDB+IMDB quality score for every
`imdb_scraped` movie. IMDB data is primary; TMDB is fallback for
overlapping fields. See ADR-019 for the v2 redesign decisions,
ADR-021 for the v4 notability signal change.

**Two hard gates (early return 0.0 before signal computation)**:
1. **Title-type gate**: `ALLOWED_TITLE_TYPES = {"movie", "tvMovie", "short", "video"}`. Movies whose `imdb_title_type` is not in this set (including None) score 0.0. Catches tvSeries, tvEpisode, videoGame, etc. that passed TMDB's classification. See ADR-037.
2. **Missing-text gate**: Movies with no `plot_summaries`, no `synopses`, AND no `featured_reviews` score 0.0. Without any text source, LLM metadata generation cannot produce meaningful vector space content.

**No other hard filters.** After the two gates, the quality score is the
sole filtering mechanism. Movies with missing IMDB data in the `imdb_data`
table are skipped (status unchanged).

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
| imdb_notability | 0.31 | Relevance | Vote count × Bayesian-adjusted rating blend + recency/classic adj. |
| critical_attention | 0.08 | Relevance | Presence of metacritic + reception_summary |
| community_engagement | 0.08 | Relevance | Weighted linear-to-cap composite |
| tmdb_popularity | 0.08 | Relevance | Log-scaled (lowered cap, ~p75 saturates) |
| featured_reviews_chars | 0.15 | Data sufficiency | Linear chars+count blend |
| plot_text_depth | 0.12 | Data sufficiency | Log-scaled composite |
| lexical_completeness | 0.10 | Data sufficiency | 5-entity average + classic boost |
| data_completeness | 0.08 | Data sufficiency | 5-field average |

Score range: [0, 1].

**Design principle (v3+):** Non-binary attributes use linear growth to a
"good enough" cap, where full credit (1.0) is reached once the data is
sufficient for the app's needs. Avoids over-rewarding movies with
excess data.

### Signal details

**imdb_notability (0.31)** — Blends log-scaled vote count with a
Bayesian-adjusted IMDB rating. Three confidence tiers determine the
blend weights based on rating stability analysis:
- Low (< 100 votes): 95/5 vote/rating — ratings are noise (std ~1.37)
- Medium (100–999): 70/30 — rating has real signal
- High (≥ 1000): 85/15 — vote count dominates, rating modulates

Bayesian rating formula (m=500, C=6.0): shrinks noisy low-vote ratings
toward the dataset mean before blending. Falls back to pure vote count
when imdb_rating is absent. Age multipliers (recency up to 2.0x for
films <2yr, classic up to 1.5x for >20yr) applied after blending.
See ADR-021 for rationale.

**critical_attention (0.08)** — Count presence of `metacritic_rating`
(not None) + `reception_summary` (truthy string). 0/2→0.0, 1/2→0.5,
2/2→1.0. Pure bonus — absence is normal.

**community_engagement (0.08)** — Weighted composite with linear-to-cap
sub-scores. Sub-weights: plot_keywords→1 (cap 5), featured_reviews→2
(cap 5, IMDB or TMDB fallback), plot_summaries→3 (binary), synopses→4
(binary). Score = sum of (sub-weight × sub-score) / 10.

**tmdb_popularity (0.08)** — Log-scaled TMDB activity score. Lowered
log cap (STAGE5_POP_LOG_CAP=4.0) so ~p75 of has_providers saturates.

**featured_reviews_chars (0.15)** — Linear blend of total chars and
review count. char_score = min(chars/5000, 1.0), count_score =
min(count/5, 1.0), averaged. IMDB primary; TMDB fallback.

**plot_text_depth (0.12)** — Log-scaled composite of overview +
plot_summaries + synopses. IMDB primary; TMDB overview_length fallback.
Log cap 5,001.

**lexical_completeness (0.10)** — 5 entity types averaged to [0, 1]
with classic-film age boost (1.0× at 20yr → 1.5× at 50yr).
Actors/characters: linear to 10. Writers, producers: binary.
Production companies: binary (IMDB primary, TMDB fallback).
Composers removed (not useful for search).

**data_completeness (0.08)** — 5 fields averaged to [0, 1].
plot_keywords: linear to 5. overall_keywords: linear to 6.
parental_guide_items: linear to 3. maturity_rating, budget:
binary (IMDB/TMDB fallback). Filming_locations removed.

### Changes from v3

- **imdb_vote_count → imdb_notability**: pure log-scaled vote count
  replaced with a vote-count × Bayesian-adjusted-rating blend. Three
  confidence tiers based on rating-stability analysis of has_providers
  movies control how much the IMDB rating modulates the base score.
- **Weights**: imdb_notability 0.25→0.31, critical_attention 0.12→0.08,
  community_engagement 0.10→0.08

### Changes from v2

- featured_reviews: 4-tier char-only → linear chars+count blend
- community_engagement: plot_keywords/reviews linear-to-cap (was binary)
- lexical_completeness: composers removed, actors/characters linear to
  10 (was 3-tier at 1/5), classic-film age boost added
- data_completeness: filming_locations removed, plot_keywords/
  overall_keywords/parental_guide_items linear-to-cap (was coarse tiers)
- Weights: imdb_vote_count 0.27→0.25, community_engagement 0.08→0.10

### Changes from v1

- Removed all 10 hard filters — score is the sole filter (supplemented
  by the two hard gates added for title-type and missing-text, ADR-037)
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

The canonical group classification for Stage 5 lives in `scoring_utils.py`:
`MovieGroup` enum, `classify_movie_group()`, `passes_imdb_quality_threshold()`,
`IMDB_QUALITY_THRESHOLDS`, and SQL constants for the three groups. All
analysis/diagnostic scripts import from here rather than reimplementing
the bucketing logic. See ADR-022.

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
**metadata scoring** ← imdb_notability, tmdb_popularity.

## Stage 6: LLM Metadata Generation (Batch API)

Stage 6 generates 8 types of LLM metadata per movie (7 embedded vector
spaces + `reception` which produces a non-embedded intermediate) via
OpenAI's Batch API. Lives in `metadata_generation/` subpackage.
See ADR-024 for the Batch API architecture decision and ADR-025 for
schema design decisions.

**CLI workflow**:
```
python -m movie_ingestion.metadata_generation.run submit --wave 1
python -m movie_ingestion.metadata_generation.run status
python -m movie_ingestion.metadata_generation.run process   # auto-submits Wave 2
python -m movie_ingestion.metadata_generation.run status
python -m movie_ingestion.metadata_generation.run process
```

**Two-wave design**: Wave 1 generates `plot_events` and `reception` in
parallel. Wave 2 uses `plot_synopsis` (from plot_events) and
`review_insights_brief` (from reception) to generate the remaining 6
generations: `plot_analysis`, `viewer_experience`, `watch_context`,
`narrative_techniques`, `production_keywords`, `source_of_inspiration`.
Completion window is up to 24h per wave; `process` auto-submits Wave 2
after Wave 1 processing.

**State tracking**: Two tables in tracker.db — `metadata_batches`
(one row per batch submission) and `metadata_results` (one row per
tmdb_id × generation_type). `plot_synopsis` and `review_insights_brief`
are scalar columns in `metadata_results`, not buried in `result_json`,
so Wave 2 request building can query them directly.

**Generator contract**: All 8 generators are fully implemented as async
real-time callers — each takes `MovieInputData` plus optional
`provider`/`model`/`system_prompt`/`response_format`/`**kwargs`,
calls `generate_llm_response_async`, and returns `Tuple[Output, TokenUsage]`.
Wave 2 generators additionally accept `plot_synopsis` and/or
`review_insights_brief` Wave 1 outputs as parameters. This diverges from
the original Batch API body-dict design in ADR-024; the Batch API
scaffolding in `request_builder.py` and `run.py` was built for a different
interface; alignment is pending. See ADR-026, ADR-027.

**Status progression**: `imdb_quality_passed` → `phase1_complete`
(after Wave 1) → `phase2_complete` (after Wave 2).

### plot_events Generator: Two-Branch Design

The plot_events generator branches on synopsis presence and quality.
See ADR-033 for the full design rationale.

**`build_plot_events_prompts(movie)` returns `(user_prompt, system_prompt)`.**
All branching logic is contained here; callers do not need to know about
the branching.

**Branch A — synopsis condensation** (`MIN_SYNOPSIS_CHARS = 1000`):
When a synopsis exists and is ≥ 1,000 chars, route to condensation.
Inputs: synopsis + overview + plot_keywords. System prompt:
`SYSTEM_PROMPT_SYNOPSIS` (frames task as abbreviation of a rich source).

**Synopsis quality gate**: When a synopsis exists but is under 1,000 chars
(consistently non-plot text in practice), demote it into the summaries
list and route to Branch B instead. Prevents condensation branch (which
prohibits model knowledge) from fabricating content from an inadequate
source.

**Branch B — synthesis/consolidation**: When no synopsis (or thin synopsis
demoted above). Inputs: summaries (first 3) + overview + plot_keywords.
System prompt: `SYSTEM_PROMPT_SYNTHESIS` — frames task as text
consolidation, not narrative creation. Model is told it "has no knowledge
of any film" to eliminate the self-assessment problem that allowed
parametric knowledge to undermine anti-fabrication guardrails.
Output: 4K token soft cap + `max_tokens=5000` hard safety net.
See ADR-035.

**Provider default**: OpenAI `gpt-5-mini`. (Gemini cannot be the default
because it requires `max_output_tokens` not `max_tokens`; the generic
router does not normalize this parameter.)

**`source_of_inspiration` no longer receives `plot_synopsis`**: Saves
~83.6M input tokens. Keywords + reviews + title are sufficient.

### Pre-consolidation (pre_consolidation.py)

Pure data processing before any LLM calls. Called by
`request_builder.build_wave1_requests()` via `run_pre_consolidation()`.

**Keyword routing** (`route_keywords()`): normalizes each keyword
(`.lower().strip()`) and deduplicates within each list before merging.
Produces three routed lists:
- `plot_keywords` → plot_events only
- `overall_keywords` → narrative_techniques only
- `merged_keywords` (union, plot first) → plot_analysis, viewer_experience,
  watch_context, production_keywords, source_of_inspiration

Note: `plot_analysis` was changed from `plot_keywords` to `merged_keywords`
because overall keywords provide additional thematic signal. `narrative_techniques`
uses `overall_keywords` only — structural tags like "nonlinear timeline" live
there and plot keywords add noise without structural signal.

**Maturity consolidation** (`consolidate_maturity()`): 4-step priority
chain: reasoning list → parental guide items → MPAA rating definition
→ None. Produces a single `maturity_summary` string passed to
viewer_experience and watch_context. `MovieInputData.maturity_summary()`
delegates to this function via lazy import (avoids circular import).

**Eligibility checks**: Wave 1 checks (`check_plot_events`, `check_reception`)
are public functions — called by evaluation runners and `wave1_runner` to
pre-filter ineligible movies. Wave 2 has 6 private `_check_<type>()` methods.
All return `str | None` (None = eligible, str = skip reason), composed by
`assess_skip_conditions()` into a `SkipAssessment`. Called twice: before
Wave 1 (Wave 1 checks only), before Wave 2 (Wave 2 checks using actual
Wave 1 outputs).

Key skip thresholds:
- `plot_events`: skips if all text sources are absent OR all sparse
  (overview < 10 chars, each synopsis < 50 chars, combined summaries
  < 50 chars)
- `reception`: skips if no `reception_summary`, no
  `audience_reception_attributes`, AND combined review text < 25 chars
  (`_MIN_REVIEWS_CHARS`)
- Wave 2 checks require `plot_synopsis` or `review_insights_brief`
  as primary fallback paths; many have secondary fallbacks via
  genre/keyword/maturity data
- `source_of_inspiration` check requires only `merged_keywords` or
  `review_insights_brief` (no `plot_synopsis` dependency)

### Output schemas (schemas.py)

Pydantic `BaseModel` schemas for each generation. Key design decisions
(see ADR-025, ADR-036):

- **Base variants have justification fields removed** from all section
  models. The existing `implementation/classes/schemas.py` schemas (used
  by the search pipeline) retain their original structure; these
  generation-side schemas diverge intentionally to reduce token cost.
- **`WithJustificationsOutput` variants** exist for all Wave 2 generation
  types (e.g., `PlotAnalysisWithJustificationsOutput`,
  `ViewerExperienceWithJustificationsOutput`, etc.) for evaluation use.
  `TermsWithJustificationSection` (adds a `justification` field to
  `TermsSection`) is the shared sub-model for these variants. The
  `__str__()` of each `WithJustificationsOutput` produces identical
  embedding text to its base variant — this invariant is tested.
- **`PlotEventsOutput` and `MajorCharacter` use minimal neutral field
  descriptions** ("Chronological plot summary.", "Character name.", etc.).
  Behavioral instructions — what the model should do with each field —
  live in the branch-specific system prompts (SYSTEM_PROMPT_SYNOPSIS
  and SYSTEM_PROMPT_SYNTHESIS), not the schema. This prevents schema
  field descriptions from competing with prompt-level constraints.
  See ADR-036.
- **`review_insights_brief`** (on `ReceptionOutput`) is a ~150-250 token
  dense paragraph extracted from reviews. It is an intermediate output:
  stored as a scalar column in `metadata_results`, consumed by Wave 2
  generator prompts, and **excluded from `__str__()`** so it is never
  embedded into Qdrant.
- **`ProductionKeywordsOutput` and `SourceOfInspirationOutput` are
  separate schemas** (and separate LLM calls), unlike the existing
  `ProductionMetadata` which merged them.
- All `__str__()` methods lowercase their output to match the embedding
  text convention used by the search-side schemas.

### Model Evaluation (evaluations/)

Before committing to a model/provider for any generation type, the
evaluation pipeline runs systematic side-by-side comparisons across
candidates. See ADR-028 for the overall design, ADR-034 for the
current reference-free approach.

**Reference-free pointwise evaluation (current):** For each (candidate,
movie) pair, the pipeline generates the candidate's output, then scores
it using Claude Opus 4.6 as the judge. The judge sees raw source data
(the labeled input fields from `build_plot_events_user_prompt`) and a
quality rubric ("A HIGH-QUALITY OUTPUT should:"), not a reference output
or the generation prompt itself. Each evaluation runs the judge 2 times
sequentially — run 1 primes the Anthropic prompt cache, run 2 benefits
from cached reads at 90% discount. Scores are averaged; reasoning is
concatenated. Idempotent — rerunning skips rows that already exist.
See ADR-031 for the multi-run averaging decision, ADR-034 for the
reference removal and judge switch decision.

**Thinking disabled, caveman-speak reasoning**: Judge runs with
`thinking: disabled`. Reasoning fields are constrained to one sentence,
max 30 words, caveman-speak (no articles, filler) to compress output
tokens.

**Rate-limit retry**: Judge calls retry indefinitely on 429, sleeping
30s between attempts.

**`--branch` flag**: `run_evaluations_pipeline.py` accepts
`--branch synopsis|synthesis` to filter the test corpus by synopsis
presence, enabling separate evaluation of each generation branch.

**Evaluation DB (`evaluation_data/eval.db`)**: Separate from tracker.db.
Each metadata type gets its own set of tables
(e.g., `plot_events_candidate_outputs`, `plot_events_evaluations`).
Scoring dimensions are individual REAL columns (not JSON blobs) so SQL
can aggregate per-dimension. `judge_runs` column records how many runs
contributed to each score. WAL journal mode enabled for crash-safety.

**plot_events candidates**: 8 active candidates (remaining candidates from
the original 19-candidate set are commented out after evaluation runs narrowed
the field). Active set: Gemini 2.5 Flash Lite (think-1k, think-4k),
GPT-5-mini (reason-low), GPT-5.4-nano, plus short-prompt variants of each
(4 total). Candidate IDs use `{type}__{model}__{variant}` naming. Evaluation
complete — winner is `gemini-2.5-flash-lite__think-1k__short-prompt`.
Production default has not been committed to the generator; callers must
pass provider/model explicitly (see `generators/plot_events.py`).

**Short-prompt variants**: `SYSTEM_PROMPT_SHORT` in
`metadata_generation/prompts/plot_events.py` is a ~37% shorter version of
the full system prompt. Four `__short-prompt` candidates (one per active
non-WHAM candidate) isolate prompt length as the sole independent variable.

**4-dimension judge rubric for plot_events**: groundedness, plot_summary,
character_quality, setting. Each scored 1–4. Rubric defines quality criteria
aligned with the generation intent — evaluates against source data and
quality standards, not the generation prompt.

**CLI**:
```
python -m movie_ingestion.metadata_generation.evaluations.run_evaluations_pipeline
python -m movie_ingestion.metadata_generation.evaluations.run_evaluations_pipeline --branch synopsis
python -m movie_ingestion.metadata_generation.evaluations.run_evaluations_pipeline --branch synthesis
python -m movie_ingestion.metadata_generation.evaluations.analyze_results
```

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

### Tables

- `movie_progress` — one row per movie, status + quality scores +
  `batch1_custom_id` / `batch2_custom_id` columns.
- `filter_log` — append-only audit trail of every filtered movie.
- `tmdb_data` — TMDB fields for quality scoring and downstream stages.
- `imdb_data` — IMDB scraped fields: 8 scalar columns (including
  `imdb_title_type`) + 19 JSON TEXT columns (empty lists stored as NULL).
  See ADR-023.
- `metadata_batches` — OpenAI batch submissions (managed by `state.py`).
- `metadata_results` — per-generation results (managed by `state.py`).
- `wave1_results` — direct generation results for evaluation use (managed
  by `wave1_runner.py`). One row per tmdb_id, nullable `plot_events` and
  `reception` JSON columns.

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

`wave1_runner.py` follows the same pattern for LLM generation: async
tasks return result tuples (no DB access inside tasks); all DB writes
happen in a synchronous batch after `asyncio.gather()`.

## Gotchas

- TMDB fetching is free (API key only). IMDB scraping requires
  proxies (~$5-17 for 100K movies). This cost asymmetry drives
  the TMDB-first funnel design.
- The SQLite tracker serves as checkpoint DB, data store for quality
  scoring, and IMDB data store. Single file; size grows substantially
  with `imdb_data` (28 columns × 425K rows of JSON text).
- Watch provider keys in `tmdb_data` are stored as packed uint32
  BLOBs, not JSON arrays.
- `MovieResult.data` in `scraper.py` is `dict | None` (not a JSON
  string). Any code that unpacks it positionally needs awareness of
  this type.
- IMDB scraping uses graceful defaults — no movie is filtered out
  based on missing IMDB data at scrape time. All fields default to None/[].
  Hard gates in Stage 5 handle content-type and text-availability filtering.
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
- The `ingestion_data/imdb/` JSON file directory is now obsolete as
  the primary data store. `migrate_json_to_sqlite.py` was used for
  the one-time migration of 425,345 rows.
- `analyze_eligibility.py` uses character-length estimation for token
  counts (~3.7 chars/token) and `orjson.loads` for JSON column
  deserialization — both chosen for performance over 111K+ movies.
  System prompt tokens still use exact tiktoken (computed once at import).
- The generation-side schemas in `metadata_generation/schemas.py`
  intentionally diverge from the search-side schemas in
  `implementation/classes/schemas.py`. When deploying, align the
  search-side schemas to match generation outputs.
- Generator kwargs must be provider-specific. Do not pass a shared
  `_DEFAULT_KWARGS` dict across providers — OpenAI-specific params
  (e.g., `verbosity`, `reasoning_effort`) cause 400 errors on Gemini/Groq.
  Each caller passes exactly the kwargs its provider needs.
- `MIN_SYNOPSIS_CHARS = 1000` threshold in `generators/plot_events.py`
  must also be applied at embedding time when the pipeline uses
  `plot_synopsis` from `imdb_data` directly. If a short synopsis is
  passed raw to the embedding step, it will not match the generated
  `plot_summary` that was produced by Branch B.
- The operator precedence bug (`edge.get("node") or {} if isinstance(...)`
  parsing incorrectly) was fixed in `parsers.py` — correct form is
  `(edge.get("node") or {}) if isinstance(edge, dict) else {}`. Latent
  in practice (GraphQL edges are always dicts) but any new edge-parsing
  code must use the parenthesized form.
