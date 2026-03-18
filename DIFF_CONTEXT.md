# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Clean up dead WHAM parameter handling
Files: implementation/llms/generic_methods.py | Removed misleading max_output_tokens/temperature extraction from generate_wham_response_async — WHAM rejects both. Now explicitly pops and discards them with a comment explaining why, keeping only verbosity and reasoning_effort which WHAM supports.

## TODO cleanup
Files: docs/TODO.md | Removed 4 completed items (Stage 5 filter run, debug print, WHAM e2e verification, WHAM param cleanup).

## Weighted overall_mean for plot_events evaluations
Files: movie_ingestion/metadata_generation/evaluations/shared.py, plot_events.py, analyze_results.py | Added score_weights parameter to compute_score_summary so overall_mean is a weighted average instead of a simple mean. Plot events weights: summary 3x, grounded 2x, characters 1x, setting 1x.

## Test plan + test implementation for current changes
Files: unit_tests/TEST_PLAN.md, unit_tests/test_generic_methods.py, unit_tests/test_eval_plot_events.py, unit_tests/test_eval_run_pipeline.py

### New tests added
- **WHAM provider tests** (16 tests in test_generic_methods.py): parameter stripping (max_tokens, max_output_tokens, temperature), supported param forwarding (verbosity, reasoning_effort as nested object), validation errors (missing api_key/account_id, null parsed output), exception wrapping, return value shape, client construction (base_url, headers), stream call structure (store=False, instructions param).
- **WHAM router tests** (3 tests): dispatch via _PROVIDER_DISPATCH, presence in dispatch table, LLMProvider enum includes WHAM.
- **Short prompt invariant tests** (4 tests in test_eval_plot_events.py): non-empty, contains no-hallucination rule, mentions all output fields, shorter than default.
- **Short-prompt candidate configuration tests** (3 tests): naming convention consistency, prompt assignment verification, non-short candidates use default.

### Stale tests fixed
- test_eval_plot_events.py: Replaced 3 tests that asserted max_tokens=4096 and temperature=0.2 in LLM calls (source code never passes these). Now test reasoning_effort='low' (which IS passed) and verify temperature is absent.
- test_eval_run_pipeline.py: Replaced dead-code test (validated commented-out temp_evaluation_set) with tests verifying EVALUATION_TEST_SET_TMDB_IDS contains all sparsity subsets and has no duplicates.

## Documentation staleness fixes (docs audit)
Files: docs/decisions/ADR-028-llm-evaluation-pipeline-design.md, docs/decisions/ADR-029-anthropic-extended-thinking-integration.md, docs/decisions/ADR-026-multi-provider-llm-routing.md, docs/decisions/ADR-009-imdb-graphql-migration.md, docs/decisions/ADR-016-combined-imdb-quality-scorer.md, docs/PROJECT.md, docs/TODO.md, movie_ingestion/metadata_generation/evaluations/plot_events.py, implementation/llms/generic_methods.py

### Intent
Fix 10 stale documentation items identified by the docs-auditor scan.

### Key changes
- ADR-028: Updated Phase 0/1 descriptions from Claude Opus/Anthropic to GPT-5.4/WHAM (reflects ADR-030 switch)
- ADR-029: Corrected Consequences section — judge no longer uses Anthropic
- ADR-026: Noted expansion from 5 to 7 providers (ANTHROPIC via ADR-029, WHAM via ADR-030)
- ADR-009: Updated output format from per-movie JSON to SQLite (reflects ADR-023 migration)
- ADR-016: Updated data loading description from per-movie JSON to tracker DB tables (reflects ADR-023)
- PROJECT.md: Fixed Stage 5 (removed "hard filters"), Stage 6 (partially implemented, not "needs to be fleshed out"), and LLM provider (evaluation running, no model selected yet)
- TODO.md: Corrected request_builder.py path from evaluations/ to metadata_generation/
- plot_events.py: Fixed module docstring (Claude Opus → GPT-5.4/WHAM)
- generic_methods.py: Fixed WHAM docstring endpoint (wham/v1 → codex)

## Split value ranking cost column into dense/sparse
Files: movie_ingestion/metadata_generation/evaluations/analyze_results.py | The final value ranking table now shows two cost/1K columns (dense and sparse) instead of one overall cost. Token averages are queried separately by filtering candidate_outputs on the dense (ORIGINAL_SET) and sparse (MEDIUM+HIGH_SPARSITY) movie ID sets, so costs reflect actual input size differences.

## Migrate plot_events generator to evaluation winner
Files: movie_ingestion/metadata_generation/generators/plot_events.py | Switched production defaults to evaluation winner: Gemini 2.5 Flash Lite + short prompt + 1k thinking budget. Changed prompt import from SYSTEM_PROMPT to SYSTEM_PROMPT_SHORT. Added default provider/model/kwargs so callers don't need to specify them. Caller overrides still supported via **kwargs.
