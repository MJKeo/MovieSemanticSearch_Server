# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Rewrite notebook cell 4 for multi-candidate per-movie evaluation
Files: `movie_ingestion/metadata_generation/metadata_generation_playground.ipynb` (cell 4)

### Intent
Replace single-candidate, per-group JSON output with multi-candidate, per-movie JSON output for comparing plot_events generation across all non-Gemini models.

### Key Decisions
- **Halved evaluation groups** from 6 to 3 movies each (first half of each array) — 21 movies total.
- **Per-movie JSON files** at `evaluation_data/plot_events_{tmdb_id}.json` instead of per-group files. Each file stores: tmdb_id, title, user_prompt (once), and candidate_results array with model name, result, tokens, and cost.
- **Excluded Gemini candidates** by filtering on `provider != LLMProvider.GEMINI`.
- **Cost calculation** uses `MODEL_PRICING` from `analyze_results.py` — same pricing source used by the evaluation analysis.
- **Concurrent generation** — all candidates for a given movie fire simultaneously via `asyncio.gather()`, spreading rate-limit pressure across providers. Movies processed sequentially.

## Add JSON output instruction to branch-specific plot_events prompts
Files: `movie_ingestion/metadata_generation/prompts/plot_events.py` | Added `JSON with a single field: plot_summary.` to OUTPUT section of both SYSTEM_PROMPT_SYNOPSIS and SYSTEM_PROMPT_SYNTHESIS. Alibaba/Qwen requires "json" in messages when using `json_object` response format; the legacy SYSTEM_PROMPT already had this but the ADR-033 branch prompts didn't.

## Fix plot_events provider/model to gpt-5-mini
Files: `movie_ingestion/metadata_generation/generators/plot_events.py`

### Intent
Lock `generate_plot_events()` to OpenAI gpt-5-mini with `{"reasoning_effort": "minimal", "verbosity": "low"}` instead of accepting caller-specified provider/model/kwargs.

### Key Decisions
- **Model choice based on 21-movie evaluation** across 6 candidates (gpt-5-mini, gpt-5-nano, gpt-5.4-nano, qwen3.5-flash, gpt-oss-120b, llama-4-scout). gpt-5-mini scored 4.93/5.0 overall with 4.86 groundedness — near-zero errors. Next-best (qwen3.5-flash at $10.98) scored 4.56 with consistent small inference leaps. With 50% batch pricing, the premium is ~$17 total for meaningfully higher reliability on the pipeline's most critical field.
- **Removed provider/model/kwargs params** from `generate_plot_events()`. Production caller (`wave1_runner.py`) already used no-args form. Playground notebook cell 4 passes explicit args for multi-candidate evaluation — that's evaluation-specific code, not production.
- **Module-level constants** `_PROVIDER`, `_MODEL`, `_MODEL_KWARGS` keep the config visible and greppable.

## Simplify plot_events generation based on 42-movie evaluation
Files: `movie_ingestion/metadata_generation/generators/plot_events.py`, `movie_ingestion/metadata_generation/prompts/plot_events.py`, `movie_ingestion/metadata_generation/schemas.py`, `movie_ingestion/metadata_generation/evaluations/plot_events.py`, `movie_ingestion/metadata_generation/evaluations/analyze_results.py`

### Intent
Reduce hallucination and simplify the plot_events output based on evaluation of 42 movies across 7 size-based buckets.

### Key Decisions
- **Raised MIN_SYNOPSIS_CHARS from 1000 to 2500**: The condensation path hallucinated at 67% rate with ~1K char synopses (model fills gaps from training knowledge). At 4K+ synopses: 0% hallucination. 2500 is the threshold where synopses are consistently detailed enough for faithful condensation. Synopses below this are demoted into the summaries list and routed through the synthesis path, which has 0% hallucination at all input sizes.
- **Removed plot_keywords from input**: Evaluation showed keywords act as hallucination springboards (e.g., "intersex" keyword in Bol triggered fabrication of entire character arcs) and get incorrectly treated as plot events (e.g., "Oreos" in Drunks became a narrative detail). The overview already provides high-level framing.
- **Removed setting and major_characters from output**: Setting was redundant (already in plot_summary, exceeded ≤10 word constraint in 83% of results). Structured characters with motivations/roles added analytical burden to a consolidation task — character names appear naturally in plot_summary, and motivation analysis fits better in downstream plot_analysis. Output is now just plot_summary.
- **implementation/classes/schemas.py left unchanged** (reference only for future pipeline work).
- **Eligibility check unchanged**: `_MIN_PLOT_TEXT_CHARS = 600` in pre_consolidation.py stays — it gates on whether there's any substantial plot text, not on which branch to use.

### Testing Notes
- Unit tests referencing MajorCharacter, setting, or major_characters will need updating.
- Evaluation DB tables have changed schema — existing eval DBs need to be recreated (clean slate confirmed).
- Notebook cell 4 will produce simplified output (just plot_summary in result.model_dump()).

## Restructure tracker DB for batch generation
Files: `movie_ingestion/tracker.py`, `movie_ingestion/metadata_generation/inputs.py`

### Intent
Clean up evaluation-phase artifacts and add proper tables for batch metadata generation at scale.

### Key Decisions
- **Deleted `wave1_runner.py`**: Direct (non-batch) runner built for evaluation. Obsolete now that batch generation uses the new tables.
- **Deleted `evaluations/` folder**: Evaluations are now done via Claude Code directly. `load_movie_input_data()` was the only reusable function — moved to `inputs.py` where it lives alongside the `MovieInputData` dataclass it produces.
- **Deleted `state.py`**: Docstring-only stub describing a normalized table design that was never implemented, now superseded by the wide-table design.
- **Removed `batch1_custom_id`/`batch2_custom_id`** from `movie_progress` — replaced by `metadata_batch_ids` table.
- **`metadata_batch_ids` table**: Per-movie table with `tmdb_id` as PK and 8 batch_id columns (one per metadata type). Tracks which OpenAI batch each movie's request belongs to. No auto-seeded rows — populated when batch requests are submitted.
- **`generated_metadata` table**: One row per movie. 8 JSON columns for generated results + 8 `eligible_for_<type>` integer columns (NULL = not evaluated, 1 = eligible, 0 = ineligible).
- **Migrations** added: DROP COLUMN for batch columns, DROP TABLE for wave1_results.
- **Notebook cell 5** imports `MODEL_PRICING` from deleted `analyze_results.py` — will break but is analysis code to fix separately.
- **`unit_tests/test_wave1_runner.py`** exists but not touched per test-boundaries rule.

## Build plot_events batch generation pipeline
Files: `movie_ingestion/metadata_generation/run.py`, `movie_ingestion/metadata_generation/request_builder.py`, `movie_ingestion/metadata_generation/openai_batch_manager.py`, `movie_ingestion/metadata_generation/result_processor.py`, `movie_ingestion/metadata_generation/inputs.py`, `movie_ingestion/tracker.py`, `movie_ingestion/metadata_generation/__init__.py`

### Intent
Implement the full CLI pipeline for generating plot_events metadata at scale via OpenAI's Batch API (50% cost savings). Four commands: eligibility → submit → status → process.

### Key Decisions
- **Renamed `batch_manager.py` → `openai_batch_manager.py`**: Other metadata types may use non-OpenAI providers later, so the OpenAI-specific wrapper gets a provider-prefixed name.
- **Deleted old `batch_manager.py`** stub (wave-based design, never implemented).
- **`generation_failures` table** added to tracker schema: tracks individual request failures with tmdb_id, metadata_type, error_message. Separate from generated_metadata — a movie with eligible=1, result=NULL, and a failure row clearly failed; without a failure row it hasn't been attempted.
- **`build_custom_id()` / `parse_custom_id()`** added to `inputs.py`: deterministic format `{metadata_type}_{tmdb_id}` (e.g. `plot_events_12345`). `rsplit('_', 1)` parsing is safe because tmdb_id is always a pure integer. `MovieInputData.batch_id()` updated to delegate.
- **In-memory JSONL upload**: request_builder returns `list[dict]`, openai_batch_manager serializes to `io.BytesIO`. No temp files. ~20MB per 10K-request batch.
- **In-memory result download**: `download_results()` returns parsed dicts. ~5MB for 10K results.
- **Sync OpenAI client**: Batch API calls are infrequent (per-batch, not per-movie). Async adds no benefit.
- **10K batch size**: Balances completion speed vs expiration risk. Remaining movies below 10K go in a final smaller batch.
- **Eligibility as separate step**: `cmd_eligibility()` evaluates all `imdb_quality_passed` movies in chunks of 5000, stores `eligible_for_plot_events` = 1/0 in `generated_metadata`. Separates population inspection from batch submission.
- **Batch_id clearing after processing**: `plot_events_batch_id` set to NULL after results are stored. Failed movies retain `plot_events IS NULL` with `eligible=1` so re-running `submit` picks them up.
- **Plot_events only**: No wave concept. Each metadata type handled individually. Generalization comes later.
- **Replaced all four stub files** (`run.py`, `request_builder.py`, `batch_manager.py`, `result_processor.py`) — old docstrings described a wave-based design that was never implemented.

## Replace hard-coded GENERATION_TYPE strings with MetadataType enum
Files: all 7 generators in `movie_ingestion/metadata_generation/generators/` (viewer_experience, watch_context, narrative_techniques, plot_analysis, production_keywords, reception, source_of_inspiration) | Added `MetadataType` import and replaced `GENERATION_TYPE = "..."` with `MetadataType.<VARIANT>`. `plot_events.py` already used the enum. No behavioral change since `MetadataType` is a `StrEnum`.

## Code review fixes for batch pipeline
Files: `request_builder.py`, `run.py`, `result_processor.py`, `inputs.py`

- **Chunked loading in request_builder**: `build_plot_events_requests()` now loads MovieInputData in chunks of 5K instead of all ~109K at once (would OOM — each movie holds full synopses/reviews).
- **Deduplicated custom_id parsing**: `_record_batch_ids()` in run.py now uses `parse_custom_id()` instead of inline `rsplit`. Removed unused `build_custom_id` import from run.py.
- **Guarded batch_id clearing**: `cmd_process()` now skips clearing batch_ids if `output_file_id` is None (prevents losing track of unprocessed movies).
- **Accurate error count**: `process_error_file()` now tracks actual successfully-parsed errors instead of returning `len(errors)`.
- **Removed unused import**: `GENERATION_TYPE` import removed from result_processor.py.
- **Typed custom_id functions**: `build_custom_id()` now takes `MetadataType` (not str), `parse_custom_id()` returns `MetadataType` (raises ValueError if invalid). `MovieInputData.batch_id()` signature updated to match.
