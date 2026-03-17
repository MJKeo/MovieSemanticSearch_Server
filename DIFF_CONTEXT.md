# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Add Anthropic provider support and LLM evaluation pipeline for plot_events metadata

Files:
- `implementation/llms/generic_methods.py`
- `movie_ingestion/metadata_generation/evaluations/shared.py`
- `movie_ingestion/metadata_generation/evaluations/plot_events.py`
- `pyproject.toml` / `uv.lock` (anthropic==0.85.0 added)

### Intent
Extends the LLM infrastructure to support Anthropic (Claude) as a first-class provider, and builds out the first concrete evaluation pipeline for the plot_events metadata type. This enables systematic comparison of LLM candidates for each metadata generation task before committing to a production model.

### Key Decisions

**Anthropic client in generic_methods.py, not eval-only**
Claude is added to the shared `_PROVIDER_DISPATCH` router so it can be used as a generation candidate in any context, not just evaluations. Auth uses `ANTHROPIC_OAUTH_KEY` env var. Structured output uses the tool-use pattern (Anthropic's way to force JSON-shaped responses). `max_tokens` defaults to 4096 to match the verbosity of plot_events outputs.

**SQLite at evaluation_data/eval.db with per-type tables**
Results stored in a dedicated eval DB (already gitignored) separate from the ingestion tracker DB. Each metadata type gets its own set of tables (e.g., `plot_events_references`, `plot_events_candidate_outputs`, `plot_events_evaluations`). Scoring dimensions are individual columns, not JSON blobs — enables SQL queries over individual scores. Consistent with the "queryable storage over opaque blobs" preference.

**EvaluationCandidate frozen dataclass as standard config unit**
Defined in `shared.py`. Contains candidate_id, provider, model, system_prompt, response_format, kwargs. All evaluation scripts take a list of these as their primary input. Prevents magic string proliferation at call sites.

**Two-phase idempotent evaluation structure**
Phase 0 generates reference responses using Claude Opus (via OAuth) as the gold-standard judge. Phase 1 runs each candidate and scores against the reference. Both phases check for existing DB rows before inserting, making reruns safe. Phase 0 must complete before Phase 1; enforced by assertion in run_evaluation().

**Phase 1 calls generate_llm_response_async() directly**
Evaluation candidates may override the system prompt, so the eval runner bypasses the higher-level `generate_plot_events()` function and calls the generic dispatch router directly with the candidate's own system_prompt. This honours the candidate's configuration faithfully.

**4-dimension judge rubric for plot_events (revised)**
Dimensions: groundedness, plot_summary, character_quality, setting. Each scored Literal[1,2,3,4]. Key changes from initial version: (1) groundedness promoted to its own first-class dimension instead of an awkward bolt-on penalty — gives clean diagnostic signal for hallucination rate vs. quality; (2) character_selection and character_description merged into character_quality — they evaluate the same output field and splitting gave characters 50% of total weight vs. 25% for plot_summary; (3) sparse-input guidance added to plot_summary rubric — don't penalize shorter outputs from limited inputs; (4) scoring instruction added to evaluate dimensions independently (no double-counting hallucination across groundedness + other dimensions).

### Testing Notes
- Run Phase 0 on 5 movies first; inspect DB to verify reference quality before full run
- Check judge reasoning column in plot_events_evaluations table before running Phase 1 at scale
- `request_builder.py` in the evaluations package is still a stub — not required for plot_events but will be needed for future evaluation types

## Code review fixes for evaluation pipeline

Files: `movie_ingestion/metadata_generation/generators/plot_events.py`, `movie_ingestion/metadata_generation/evaluations/plot_events.py`, `movie_ingestion/metadata_generation/evaluations/shared.py`, `movie_ingestion/metadata_generation/evaluations/__init__.py` (new)

Why: /review-code identified 3 warnings and 4 suggestions across the evaluation pipeline code.

Fixes applied:
- **Shared prompt builder**: Extracted `build_plot_events_user_prompt()` as a public function in `generators/plot_events.py`. The eval pipeline now imports it instead of duplicating the logic. Prevents silent drift if the generator's prompt construction changes.
- **`db_path` type mismatch**: `get_eval_connection` now accepts `Path | str` and coerces to `Path`. All callers in `plot_events.py` changed from `str | None` to `Path | None`.
- **Return type fix**: `print_score_summary` changed from `-> None` to `-> pd.DataFrame | None`.
- **WAL mode on eval DB**: `get_eval_connection` now sets `PRAGMA journal_mode=WAL` for crash-safety.
- **Dead variables removed**: Removed unused `gen_input_tokens`/`gen_output_tokens` in the already-exists branch.
- **`__init__.py` added**: Created empty `evaluations/__init__.py` for explicit package.
- **Debug print removed**: The `print(f"user_prompt: ...")` in the generator was removed as a side effect of the prompt extraction refactor.

## Add evaluation pipeline CLI runner and CANDIDATES list

Files: `movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py` (new), `movie_ingestion/metadata_generation/evaluations/shared.py`

Why: No CLI entry point existed to run evaluations from the terminal.

Approach:
- Added `CANDIDATES` list to `shared.py` (after `EvaluationCandidate` definition) with 3 plot_events candidates: gpt-4.1-mini, gpt-4.1-nano, gemini-2.5-flash. All use the default plot_events system prompt.
- Created `run_evaluations_pipeline.py` with `__main__` guard. Filters `CANDIDATES` by `response_format`, loads first movie only (pipeline validation before full run), then calls Phase 0 and Phase 1 sequentially.
- Run via: `python -m movie_ingestion.metadata_generation.evaluations.run_evaluations_pipeline`

## Move CANDIDATES to plot_events.py and update candidate list

Files: `movie_ingestion/metadata_generation/evaluations/plot_events.py`, `movie_ingestion/metadata_generation/evaluations/shared.py`, `movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py`

Why: Candidates should be defined per-metadata-type, not in the shared module. Candidate list updated to match the models being evaluated.

Approach:
- Moved candidates list from `shared.py` to `plot_events.py` as `PLOT_EVENTS_CANDIDATES` (7 candidates: qwen3.5-flash, gemini-2.5-flash, gemini-2.5-flash-lite, gpt-5-mini, gpt-5-nano, gpt-oss-120b, llama-4-scout).
- Updated `run_evaluations_pipeline.py` to import `PLOT_EVENTS_CANDIDATES` from `plot_events` instead of `CANDIDATES` from `shared`.

## Expand plot_events evaluation candidates from 7 to 16

Files: `movie_ingestion/metadata_generation/evaluations/plot_events.py`

Why: Original list had 1 candidate per model. Expanded to 2-3 per model to test the most impactful knob per provider — primarily reasoning/thinking depth, since the task's core challenges (source conflict resolution, character selection discipline) are reasoning problems.

Approach:
- Gemini 2.5 Flash: 3 candidates — thinking_budget 0 / 1024 / 4096 (continuous slider, most to explore)
- GPT-5-mini: 3 candidates — reasoning_effort × verbosity (two independent axes)
- Qwen 3.5 Flash: 2 candidates — thinking off / on (binary toggle)
- GPT-5-nano: 2 candidates — reasoning_effort minimal / low
- GPT-oss-120b: 2 candidates — reasoning_effort low / medium
- Gemini 2.5 Flash Lite: 2 candidates — thinking_budget 0 / 1024
- Llama 4 Scout: 2 candidates — temperature 0.2 / 0.0 (only knob available)

Candidate IDs use `{type}__{model}__{variant}` naming; baseline candidates keep the original ID (no variant suffix) for DB continuity.

## Rewrite JUDGE_SYSTEM_PROMPT to align with generation prompt

Files: `movie_ingestion/metadata_generation/evaluations/plot_events.py`

Why: The judge rubric had 9 inconsistencies with the generation SYSTEM_PROMPT (source of truth for what a good output looks like). Key clashes: rubric rewarded "narrative arc" structure while generation prompt asks for "events and facts"; rubric penalized collapsing secondary threads while generation prompt says "only 1-3 core conflicts"; rubric had no signal for compactness, filler, or theme talk which the generation prompt explicitly prohibits.

Approach:
- Added generation prompt summary to rubric context so the judge knows exactly what the generator was instructed to do
- **groundedness**: Added source priority hierarchy (synopsis > summaries > overview) so judge can adjudicate conflicting inputs
- **plot_summary**: Reframed from "narrative arc" to "chronological event coverage"; removed secondary-thread penalty; added compactness and no-theme-talk criteria at each score level
- **character_quality**: Aligned with "ABSOLUTELY ESSENTIAL ONLY" — no penalty for aggressive minimalism; added primary_motivations format criterion (1 short sentence); added "short, plot-relevant description" criterion
- **setting**: Clarified that omitting unknown dimensions is correct behavior, not a gap
- Added scoring instruction #7: filler/moralizing penalized in plot_summary, not groundedness
- No schema/DB changes — same 4 dimensions, same PlotEventsJudgeOutput fields

## Add analyze_results.py for evaluation result analysis

Files: `movie_ingestion/metadata_generation/evaluations/analyze_results.py` (new)

Why: No way to view combined quality scores + token usage + cost per candidate from the eval DB.

Approach: Read-only script that merges three queries — `plot_events_evaluations` (scores via `compute_score_summary`), `plot_events_candidate_outputs` (mean token counts), and `candidates` (model/provider) — and prints two sorted tables: scores by overall_mean descending, cost by cost_per_movie ascending. Per-movie cost computed from a hard-coded `MODEL_PRICING` dict (input/output price per million tokens) keyed by model name. Run via `python -m movie_ingestion.metadata_generation.evaluations.analyze_results`.

## Run full 70-movie evaluation corpus with eligibility pre-filter

Files: `movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py`, `movie_ingestion/metadata_generation/pre_consolidation.py`

Why: The pipeline runner only loaded 1 movie as a smoke-test guard (`[:1]`). That guard is now removed. Added a pre-filter to skip movies lacking sufficient plot text data before incurring any LLM spend.

Approach:
- Removed `[:1]` slice — all 70 `EVALUATION_TEST_SET_TMDB_IDS` are loaded.
- Added `_filter_plot_events_eligible()` helper that calls `check_plot_events(movie_input)` on each loaded movie; excludes it and prints `SKIPPED <id> (<title>): <reason>` if the check returns a non-None reason. Passes only eligible movies to Phase 0 and Phase 1.
- Made `_check_plot_events` public as `check_plot_events` in `pre_consolidation.py`. Left `_check_plot_events = check_plot_events` alias so existing test imports are not broken. Updated the internal call in `assess_skip_conditions` to use the public name. Updated module docstring accordingly.

## Add gpt-5.4-nano candidates to plot_events evaluation
Files: `movie_ingestion/metadata_generation/evaluations/plot_events.py` | 3 candidates testing reasoning_effort (minimal vs low) and verbosity (low vs medium) to find the quality-cost sweet spot for the model upgrade from gpt-5-nano.

## Tune temperature parameters across plot_events evaluation candidates
Files: `movie_ingestion/metadata_generation/evaluations/plot_events.py`

Why: Several candidates were missing explicit temperature settings, defaulting to provider defaults (e.g., Gemini defaults to 1.0) which is too high for a factual extraction task where groundedness is the #1 scoring dimension.

Changes:
- Qwen 3.5 Flash (thinking disabled): temperature 0.2 → 0.0 (maximize determinism for the no-thinking baseline)
- All 5 Gemini candidates (3x Flash, 2x Flash Lite): added `"temperature": 0.2` (previously unset, defaulting to 1.0)
- Both GPT-oss-120b candidates (Groq): added `"temperature": 0.2` (Groq docs confirm temperature is supported for this reasoning model, default 0.6)

Verified via research: Gemini's `GenerateContentConfig` declares `temperature: Optional[float]` in the google-genai SDK. Groq's reasoning model docs list temperature as a recommended config param with range 0.0-2.0.

## Rewrite test_imdb_quality_scorer.py to match current source (v4)

Files: `unit_tests/test_imdb_quality_scorer.py`

### Intent
The test file was written against v2 of `imdb_quality_scorer.py` and never updated through
v3 and v4 rewrites. 14 of 21 imports were broken (functions removed/renamed), and many
surviving tests asserted wrong expected values (old tiered scoring, [-1,1] ranges, 6-entity
system). Full audit and rewrite to achieve comprehensive coverage of the current source.

### Key Decisions
- **Deleted 10 obsolete test classes** (~250 lines): all `TestFails*` filter predicates,
  `TestEvaluateFilters`, `TestScoreImdbVoteCount`, `TestScoreWatchProviders`,
  `TestScoreMetacriticRating` — all tested functions that no longer exist.
- **Added 3 new test classes**: `TestScoreImdbNotability` (22 tests covering 3 confidence
  tiers, Bayesian shrinkage, recency/classic boosts, edge cases), `TestScoreCriticalAttention`
  (6 tests including `is not None` vs truthiness edge cases), `TestScoreCommunityEngagement`
  (12 tests covering weighted sub-scores, TMDB fallback, partial contributions).
- **Rewrote 3 test classes**: `TestScoreFeaturedReviewsChars` (tiered → linear blend),
  `TestScoreLexicalCompleteness` (6→5 entities, [-1,1]→[0,1], added `today` param and
  classic boost tests), `TestScoreDataCompleteness` (6→5 fields, [-1,1]→[0,1], linear-to-cap).
- **Updated helper factory** `_make_ctx()`: removed `imdb_none` param, `struct` import,
  `composers`/`filming_locations` defaults; added `reception_summary`; increased actors/
  characters to 10, overall_keywords to 6, parental_guide_items to 3 (matching new caps).

### Testing Notes
98 tests, all passing. Coverage spans every scoring function, TMDB fallback path,
age multiplier, tier boundary, and edge case in the current source.
