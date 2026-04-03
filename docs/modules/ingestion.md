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
| `imdb_scraping/http_client.py` | Async GraphQL client with proxy, retry, semaphore, random UA rotation. Fetches `titleType { id }` as part of the standard query. Artificial limits on credits, companies, composers, and plots fields were removed — `plots` was previously `first: 10`, which silently dropped synopses (often entry 11+); now set to 100,000 so synopses are always reached. |
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
| `metadata_generation/batch_generation/run.py` | Stage 6 CLI entry point: `eligibility`, `submit`, `status`, `process`, `autopilot`. `eligibility`/`submit`/`autopilot` require `--metadata` arg; `status`/`process` handle all types. Autopilot generates/submits for specified type but processes all types' batches. The `eligibility` command re-evaluates ALL `imdb_quality_passed` rows on every run (not just NULL rows), so reruns pick up changed upstream inputs without manual flag clearing. |
| `metadata_generation/batch_generation/generator_registry.py` | Maps each `MetadataType` to its `GeneratorConfig` (schema, eligibility checker, prompt builder, live generator, model config). Thin adapter wrappers normalize different generator prompt interfaces into a common `(user_prompt, system_prompt)` tuple contract. `get_config(metadata_type)` is the lookup entry point. |
| `metadata_generation/batch_generation/request_builder.py` | Builds per-type Batch API request lists. `build_requests(metadata_type)` loads eligible movies in chunks of 5K to avoid OOM, uses registry config for prompts/schema/model. Returns `list[list[dict]]`; serialization to JSONL is done in `openai_batch_manager.py`. |
| `metadata_generation/batch_generation/openai_batch_manager.py` | OpenAI Files API + Batch API wrapper: upload, create, check status (`BatchStatus` dataclass includes batch-level `errors`), download. In-memory JSONL upload/download (no temp files). No movie/generation knowledge. |
| `metadata_generation/batch_generation/result_processor.py` | Parses downloaded result JSONL, determines metadata type from custom_id, validates against correct schema via `SCHEMA_BY_TYPE`, stores results in the type's column in `generated_metadata`. Records per-request failures to `generation_failures`. Handles `"response": null` entries from expired batches via `or {}` pattern (not `.get()` default). |
| `metadata_generation/inputs.py` | `MovieInputData`, `ConsolidatedInputs`, `SkipAssessment`, `Wave1Outputs` dataclasses + `build_user_prompt()` + `MultiLineList`. `MovieInputData` provides `merged_keywords()`, `maturity_summary()`, `best_plot_fallback()`, and `batch_id()`. `build_custom_id(tmdb_id, MetadataType)` / `parse_custom_id(str) -> tuple[MetadataType, int]` encode/decode Batch API `custom_id` as `{metadata_type}_{tmdb_id}`. `load_movie_input_data()` loads raw data from tracker.db. `load_wave1_outputs(tmdb_id)` returns a `Wave1Outputs` with all Wave 1 fields (plot_summary, thematic/emotional/craft observations, source_material_hint) — callers pick whichever subset they need. |
| `metadata_generation/schemas.py` | Pydantic output schemas for each LLM generation type. Base variants have justification fields removed; each Wave 2 type also has a `WithJustificationsOutput` variant for evaluation (identical `__str__()` to the base). `PlotEventsOutput` contains only `plot_summary` (setting and major_characters removed after 42-movie evaluation; see ADR-040). `ReceptionOutput` uses dual-zone structure: extraction zone (4 observation fields for Wave 2, not embedded) + synthesis zone (summary + quality tags, embedded). `NarrativeTechniquesOutput` uses 9-section schema (removed `thematic_delivery` and merged `meta_techniques` into `additional_narrative_devices`; see ADR-048). `WatchContextWithIdentityNoteOutput` replaces `viewing_appeal_summary` (20-30 word anchor) with `identity_note` (2-8 word classification). `TermsWithJustificationSection.justification` renamed to `evidence_basis` — framed as evidence inventory, not post-hoc explanation (see ADR-049). `SourceOfInspirationOutput` uses two fields: `source_material` (adaptations, remakes, reboots, reimaginings, spinoffs) and `franchise_lineage` (sequel, prequel, trilogy position); `production_mediums` removed (derived deterministically at embedding time); see ADR-051, ADR-052. `SourceOfInspirationWithReasoningOutput` adds `source_evidence`/`lineage_evidence` inventory fields (non-gating); aliased as `SourceOfInspirationWithJustificationsOutput` for test compatibility. See ADR-025, ADR-050. |
| `metadata_generation/batch_generation/pre_consolidation.py` | Pre-consolidation: keyword routing + normalization, maturity consolidation, eligibility checks (Wave 1: `check_plot_events`, `check_reception` — both public; Wave 2: 6 private `_check_*`), `assess_skip_conditions()` orchestrator, `run_pre_consolidation()` entry point. Public shared functions: `resolve_viewer_experience_narrative()`, `filter_viewer_experience_observations()`, `resolve_narrative_techniques_narrative()` — each used by both eligibility and prompt building. `narrative_techniques` uses tiered eligibility (plot_summary / fallback >= 500 / craft >= 400 standalone / fallback >= 300 + craft >= 300 combined). `watch_context` now requires genre data AND ≥1 observation field (emotional/craft/thematic); genre-only movies are ineligible (~0.7% of pipeline). |
| `metadata_generation/errors.py` | Custom exception classes (`MetadataGenerationError`, `MetadataGenerationEmptyResponseError`) imported by all generators. |
| `metadata_generation/helper_scripts/report_bucket_axis_performance.py` | Diagnostic CLI: reads `*_evaluation.json` files and prints per-bucket tables of average candidate performance per scoring axis. Supports both bucket file shapes. |
| `metadata_generation/helper_scripts/estimate_generation_cost.py` | Diagnostic CLI: projects per-candidate generation cost to the full corpus using evaluation token-usage data, with optional per-bucket breakdown. |
| `metadata_generation/generators/` | 8 generator files (one per generation type). All use `MetadataType.<VARIANT>` for `GENERATION_TYPE`. All 8 generators are now locked (provider/model are module constants, no caller params): `plot_events.py` (gpt-5-mini, reasoning_effort=minimal), `reception.py` (gpt-5-mini, reasoning_effort=low), `plot_analysis.py` (gpt-5-mini, reasoning_effort=minimal, justifications schema), `viewer_experience.py` (gpt-5-mini, reasoning_effort=minimal, justifications schema, GPO-only narrative), `narrative_techniques.py` (gpt-5-mini, reasoning_effort=minimal, justifications schema, 9-section schema), `watch_context.py` (gpt-5-mini, reasoning_effort=minimal, WatchContextWithIdentityNoteOutput), `source_of_inspiration.py` (gpt-5-mini, reasoning_effort=low, base SourceOfInspirationOutput schema, see ADR-053), `production_keywords.py` (gpt-5-mini, reasoning_effort=low, base ProductionKeywordsOutput schema — only Wave 2 generator using base schema in production, see ADR-054). See ADR-026, ADR-027, ADR-045, ADR-048, ADR-049, ADR-053, ADR-054. |
| `metadata_generation/prompts/` | 8 system prompt files (one per LLM call). Each prompt file exports a `SYSTEM_PROMPT` constant. All generators are now locked; no unlocked generators remain. `production_keywords.py` exports both `SYSTEM_PROMPT` and `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` (retained for evaluation notebook backward compatibility). `source_of_inspiration.py` exports both `SYSTEM_PROMPT` and `SYSTEM_PROMPT_WITH_REASONING` for potential future evaluation. Locked generators use the base (non-reasoning) prompt as production. `plot_events.py` exports `SYSTEM_PROMPT_SYNOPSIS` and `SYSTEM_PROMPT_SYNTHESIS` for the two branches. |
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
overlapping fields. See ADR-021 for the v4 notability signal change.

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

**CLI workflow** (per metadata type — plot_events first):
```
python -m movie_ingestion.metadata_generation.run eligibility
python -m movie_ingestion.metadata_generation.run submit [--batch-size N] [--max-batches N]
python -m movie_ingestion.metadata_generation.run status
python -m movie_ingestion.metadata_generation.run process
# Or use autopilot to run the full cycle automatically:
python -m movie_ingestion.metadata_generation.run autopilot [--batch-size 1000] [--max-concurrent 3] [--live-batch-size 25] [--live-concurrency 5]
```

**Per-type design**: Each metadata type is processed independently. No wave
dependency is modeled in the CLI — the dependency between generation types
(plot_events must precede plot_analysis etc.) is managed at the operator
level by running types in the right order.

**`autopilot` command**: Requires `--metadata <type>`. Interleaves live
generator calls for the specified type with Batch API polling. Each iteration:
(1) run N live generations at configurable concurrency, (2) check batch
statuses across all types, (3) process completed/failed batches for all
types, (4) submit new batches for the specified type. Terminates when no
batches remain and no eligible movies exist. Live generation (~30-60s for
25 requests) serves as a natural poll interval; falls back to 60s sleep
when no live-eligible movies remain.

**State tracking**: Three tables in tracker.db —
- `metadata_batch_ids` — one row per movie (tmdb_id PK), 8 batch_id columns
  (one per metadata type). Populated when batch requests are submitted;
  cleared after processing.
- `generated_metadata` — one row per movie, 8 JSON result columns + 8
  `eligible_for_<type>` integer columns (NULL = not evaluated, 1 = eligible,
  0 = ineligible).
- `generation_failures` — per-request failures (tmdb_id, metadata_type,
  error_message). A movie with eligible=1, result=NULL, and a failure row
  clearly failed; without a failure row it has not been attempted.

**Generator contract**: All 8 generators are fully implemented as async
real-time callers — each takes `MovieInputData`, calls
`generate_llm_response_async`, and returns `Tuple[Output, TokenUsage]`.
All 8 are registered in `generator_registry.py` and reachable via the
batch pipeline CLI (`run.py`).
All 8 generators are locked: provider/model are module-level constants
(`_PROVIDER`, `_MODEL`), not caller params.
See ADR-026, ADR-039, ADR-042, ADR-043, ADR-048, ADR-049, ADR-053, ADR-054.

**Locking finalized generators**: When a generator's evaluation is
complete, remove all configurable model parameters from its production
function signature — take only `movie: MovieInputData`. Hardcode
provider, model, schema, and kwargs as module-level constants with a
comment citing the evaluation winner. The playground notebook is the
only place that needs configurable model params.

**No provider-specific default kwargs**: Generators must not define
default kwargs that span providers. The generic LLM router passes kwargs
through without normalization — provider-specific params (e.g., `max_tokens`
for OpenAI vs `max_output_tokens` for Gemini) cause 400 errors on other
providers. Document at the call site.

**Status progression**: `imdb_quality_passed` → `phase1_complete`
(after plot_events + reception) → `phase2_complete` (after remaining 6).

### plot_events Generator: Two-Branch Design

The plot_events generator branches on synopsis presence and quality.
See ADR-033 for the full design rationale.

**`build_plot_events_prompts(movie)` returns `(user_prompt, system_prompt)`.**
All branching logic is contained here; callers do not need to know about
the branching.

**Branch A — synopsis condensation** (`MIN_SYNOPSIS_CHARS = 2500`):
When a synopsis exists and is ≥ 2,500 chars, route to condensation.
Inputs: synopsis + overview. (plot_keywords removed — evaluation showed
they act as hallucination springboards.) System prompt: `SYSTEM_PROMPT_SYNOPSIS`.

**Synopsis quality gate**: When a synopsis exists but is under 2,500 chars,
demote it into the summaries list and route to Branch B. (Raised from 1,000
to 2,500 after 42-movie evaluation showed 67% hallucination rate on ~1K char
synopses; 0% at 4K+; 2,500 is the observed quality threshold.) See ADR-040.

**Branch B — synthesis/consolidation**: When no synopsis (or thin synopsis
demoted above). Inputs: summaries (first 3) + overview. (plot_keywords
removed for same reason as Branch A.) System prompt: `SYSTEM_PROMPT_SYNTHESIS`
— frames task as text consolidation, model is told it "has no knowledge of any
film." Output: 4K token soft cap + `max_tokens=5000` hard safety net. See ADR-035.

**Output**: `plot_summary` only. `setting` and `major_characters` removed after
42-movie evaluation (setting exceeded ≤10 word constraint 83% of the time;
character motivations/roles added analytical burden to a consolidation task).
See ADR-040.

**Provider**: OpenAI `gpt-5-mini` with `reasoning_effort=minimal, verbosity=low`.
Locked at module level — not a caller param. Selected after 21-movie 6-candidate
evaluation (4.93/5.0 overall, 4.86 groundedness). See ADR-039.

**`source_of_inspiration` no longer receives `plot_synopsis`**: Saves
~83.6M input tokens. Keywords + `source_material_hint` + title are sufficient.
`source_material_hint` (from reception extraction zone) replaced the blunt
`review_insights_brief` concatenation — it's a targeted classifying phrase
for adaptation/remake/source status, not a blob of thematic/emotional observations.

### Pre-consolidation (pre_consolidation.py)

Pure data processing before any LLM calls. Called by the batch pipeline
via `run_pre_consolidation()`.

**Keyword routing** (`route_keywords()`): normalizes each keyword
(`.lower().strip()`) and deduplicates within each list before merging.
Produces three routed lists:
- `plot_keywords` → (no longer passed to plot_events; kept for other Wave 2 generators)
- `overall_keywords` → narrative_techniques only
- `merged_keywords` (union, plot first) → plot_analysis, watch_context,
  production_keywords, source_of_inspiration

Note: `plot_analysis` was changed from `plot_keywords` to `merged_keywords`
because overall keywords provide additional thematic signal. `narrative_techniques`
uses `overall_keywords` only — structural tags like "nonlinear timeline" live
there and plot keywords add noise without structural signal.
`viewer_experience` does not receive merged_keywords — Round 3 evaluation
showed <2% citation rate; removed to reduce noise (Tier 1 pruning).

**Maturity consolidation** (`consolidate_maturity()`): 4-step priority
chain: reasoning list → parental guide items → MPAA rating definition
→ None. Produces a single `maturity_summary` string passed to
viewer_experience and watch_context. `MovieInputData.maturity_summary()`
delegates to this function via lazy import (avoids circular import).

**Shared logic rule**: When eligibility checking and prompt building
apply the same logic (threshold filtering, input resolution, fallback
ladders), that logic must live in exactly one place — a public function
in `pre_consolidation.py`. Generators import and call it. Never mirror
threshold constants or filtering logic between the two layers.

**Eligibility checks**: Wave 1 checks (`check_plot_events`, `check_reception`)
are public functions — called by evaluation runners and the batch pipeline
to pre-filter ineligible movies before LLM spend. Wave 2 has 6 private `_check_<type>()` methods.
All return `str | None` (None = eligible, str = skip reason), composed by
`assess_skip_conditions()` into a `SkipAssessment`. Called twice: before
Wave 1 (Wave 1 checks only), before Wave 2 (Wave 2 checks using actual
Wave 1 outputs).

Key skip thresholds:
- `plot_events`: skips if the longest text among the first synopsis
  entry (if any) and all plot_summaries entries is < 600 chars
  (`_MIN_PLOT_TEXT_CHARS`). Overview is excluded — too short to anchor
  plot event extraction on its own.
- `reception`: skips if no `reception_summary`, no
  `audience_reception_attributes`, AND combined review text < 400 chars
  (`_MIN_REVIEWS_CHARS`). Raised from 25 after evaluation showed movies
  below 400 chars produce observations that merely paraphrase the overview
  rather than adding genuine review signal (~779 movies affected, 0.7%
  of corpus). See ADR-042.
- Wave 2 checks use per-type tiered logic based on the quality of available
  inputs; eligibility is not shared across types. Key thresholds:
  - `plot_analysis`: plot_summary → plot_fallback >= 400 → (fallback >= 250 + thematic >= 300)
  - `viewer_experience`: GPO >= 350 standalone → obs standalone → GPO >= 200 + obs
  - `narrative_techniques`: plot_summary → fallback >= 500 → craft >= 400 → (fallback >= 300 + craft >= 300)
  - `source_of_inspiration`: requires only `merged_keywords` or `source_material_hint`
  - `production_keywords`: requires >= 1 keyword in merged_keywords
  - `watch_context`: requires >= 1 genre_signature or genre AND >= 1 observation field (emotional/craft/thematic). See ADR-049.
  - `source_of_inspiration` and `production_keywords` do not depend on plot data or review observations

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
  `TermsWithJustificationSection` (adds an `evidence_basis` field — formerly
  `justification` — to `TermsSection`) is the shared sub-model for these variants.
  The field is framed as an evidence inventory (quote input phrases that constrain
  the terms) rather than a post-hoc explanation (see ADR-049). The
  `__str__()` of each `WithJustificationsOutput` produces identical
  embedding text to its base variant — this invariant is tested.
- **`PlotEventsOutput` and `MajorCharacter` use minimal neutral field
  descriptions** ("Chronological plot summary.", "Character name.", etc.).
  Behavioral instructions — what the model should do with each field —
  live in the branch-specific system prompts (SYSTEM_PROMPT_SYNOPSIS
  and SYSTEM_PROMPT_SYNTHESIS), not the schema. This prevents schema
  field descriptions from competing with prompt-level constraints.
  See ADR-036.
- **`ReceptionOutput` dual-zone structure**: extraction zone has 4
  nullable observation fields (`source_material_hint`, `thematic_observations`,
  `emotional_observations`, `craft_observations`) consumed by Wave 2
  generators, excluded from `__str__()` and never embedded. Synthesis zone
  has `reception_summary`, `praised_qualities` (0-6), `criticized_qualities`
  (0-6) — these ARE embedded.
- **`ProductionKeywordsOutput` and `SourceOfInspirationOutput` are
  separate schemas** (and separate LLM calls), unlike the existing
  `ProductionMetadata` which merged them.
- All `__str__()` methods lowercase their output to match the embedding
  text convention used by the search-side schemas.

### Model Evaluation (evaluations/)

Before committing to a model/provider for any generation type, the
evaluation pipeline runs systematic side-by-side comparisons across
candidates.

**Reference-free pointwise evaluation (current):** For each (candidate,
movie) pair, the pipeline generates the candidate's output, then scores
it using Claude Opus 4.6 as the judge. The judge sees raw source data
(the labeled input fields from `build_plot_events_prompts()`) and a
quality rubric ("A HIGH-QUALITY OUTPUT should:"), not a reference output
or the generation prompt itself. Each evaluation runs the judge 2 times
sequentially — run 1 primes the Anthropic prompt cache, run 2 benefits
from cached reads at 90% discount. Scores are averaged; reasoning is
concatenated. Idempotent — rerunning skips rows that already exist.
**Thinking disabled, caveman-speak reasoning**: Judge runs with
`thinking: disabled`. Reasoning fields are constrained to one sentence,
max 30 words, caveman-speak (no articles, filler) to compress output
tokens.

**Rate-limit retry**: Judge calls retry indefinitely on 429, sleeping
30s between attempts.

**plot_events evaluation**: Complete. 21-movie 6-candidate evaluation selected
gpt-5-mini (4.93/5.0 overall, near-zero hallucinations). Subsequently, a
42-movie evaluation refined the schema (removed `setting`/`major_characters`
and raised `MIN_SYNOPSIS_CHARS` to 2,500). See ADR-039, ADR-040.
The `evaluations/` subpackage was removed after evaluation completed;
`load_movie_input_data()` was moved to `inputs.py`.

**Eval guide append-only rule**: When adding a new candidate to an
evaluation guide round, append its documentation to the existing
candidates section. Never remove or replace previous candidate docs
when adding a new experiment to the same round, even if the code
changes that produced those candidates were later reverted. Generated
results are permanent artifacts independent of the current codebase state.

**4-dimension judge rubric for plot_events** (historical, now inactive):
groundedness, plot_summary, character_quality, setting. Each scored 1–4.
Evaluated via Claude Opus 4.6 judge with 2 sequential runs + prompt caching.

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

- `movie_progress` — one row per movie, status + quality scores. The
  `batch1_custom_id` / `batch2_custom_id` columns were removed; batch
  tracking lives in `metadata_batch_ids`.
- `filter_log` — append-only audit trail of every filtered movie.
- `tmdb_data` — TMDB fields for quality scoring and downstream stages.
- `imdb_data` — IMDB scraped fields: 8 scalar columns (including
  `imdb_title_type`) + 19 JSON TEXT columns (empty lists stored as NULL).
  See ADR-023.
- `metadata_batch_ids` — one row per movie (tmdb_id PK), 8 batch_id
  columns (one per metadata type, e.g. `plot_events_batch_id`). Populated
  on `submit`, cleared after `process`. No auto-seeded rows.
- `generated_metadata` — one row per movie, 8 JSON result columns
  (`plot_events`, `reception`, etc.) + 8 `eligible_for_<type>` integer
  columns (NULL = not evaluated, 1 = eligible, 0 = ineligible).
- `generation_failures` — per-request failures: `tmdb_id`, `metadata_type`,
  `error_message`. Separate from `generated_metadata` — a failure row
  distinguishes "attempted and failed" from "not yet attempted."

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
  For the future daily-update pipeline, ensure
  residential proxies remain configured.
  See `memory/imdb-scraping.md` for tuning history.
- Survival curve plot scripts call `plt.show()` which blocks in
  headless environments. Use the `Agg` backend or redirect output
  to a file when running without a display.
- The `ingestion_data/imdb/` JSON file directory is now obsolete as
  the primary data store. `migrate_json_to_sqlite.py` was used for
  the one-time migration of 425,345 rows.
- The generation-side schemas in `metadata_generation/schemas.py`
  intentionally diverge from the search-side schemas in
  `implementation/classes/schemas.py`. When deploying, align the
  search-side schemas to match generation outputs.
- Generator kwargs must be provider-specific. Do not pass a shared
  `_DEFAULT_KWARGS` dict across providers — OpenAI-specific params
  (e.g., `verbosity`, `reasoning_effort`) cause 400 errors on Gemini/Groq.
  Each caller passes exactly the kwargs its provider needs.
- `MIN_SYNOPSIS_CHARS = 2500` threshold in `generators/plot_events.py`
  must also be applied at embedding time when the pipeline uses
  `plot_synopsis` from `imdb_data` directly. If a short synopsis is
  passed raw to the embedding step, it will not match the generated
  `plot_summary` that was produced by Branch B.
- The operator precedence bug (`edge.get("node") or {} if isinstance(...)`
  parsing incorrectly) was fixed in `parsers.py` — correct form is
  `(edge.get("node") or {}) if isinstance(edge, dict) else {}`. Latent
  in practice (GraphQL edges are always dicts) but any new edge-parsing
  code must use the parenthesized form.
