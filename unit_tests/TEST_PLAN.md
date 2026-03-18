# Test Plan

Generated: 2026-03-18
Base: main branch
Changes analyzed: `implementation/llms/generic_methods.py`, `movie_ingestion/metadata_generation/evaluations/plot_events.py`, `movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py`, `movie_ingestion/metadata_generation/prompts/plot_events.py`

Non-code files changed (excluded from analysis): `DIFF_CONTEXT.md`, `docs/TODO.md`, `docs/modules/ingestion.md`, `docs/modules/llms.md`, `docs/decisions/ADR-030-wham-backend-oauth-integration.md` (new), `docs/decisions/ADR-031-multi-run-judge-averaging.md` (new)

## Summary

The changes span two areas:

1. **WHAM provider parameter handling** (`generic_methods.py`): `generate_wham_response_async` no longer extracts and forwards `max_tokens`, `max_output_tokens`, or `temperature` to the WHAM API. Instead, these are silently popped and discarded. Previously, `max_tokens` was mapped to `max_output_tokens` and `temperature` was forwarded. This is a behavioral change that ensures these unsupported parameters don't leak into WHAM API calls.

2. **Evaluation pipeline expansion** (`plot_events.py`, `run_evaluations_pipeline.py`, `prompts/plot_events.py`): Four new "short prompt" evaluation candidates added using a new `SYSTEM_PROMPT_SHORT` constant. The pipeline runner switches from a temporary subset of movies to the full evaluation test set, and reduces concurrency from 10 to 5.

## Changed Files Analysis

### `implementation/llms/generic_methods.py`
**What changed:** In `generate_wham_response_async`, `max_tokens`, `max_output_tokens`, and `temperature` are now popped from kwargs and discarded (previously `max_tokens`/`max_output_tokens` were forwarded as `max_output_tokens`, and `temperature` was forwarded as `temperature` to the API call).
**Existing coverage:** `unit_tests/test_generic_methods.py`
**Gaps identified:**
- No tests exist for `generate_wham_response_async` at all -- the entire WHAM provider is untested. The `TestGenerateLLMResponseAsync` router tests don't include WHAM dispatch.
- No test verifying that `max_tokens` is silently dropped when passed to WHAM
- No test verifying that `max_output_tokens` is silently dropped when passed to WHAM
- No test verifying that `temperature` is silently dropped when passed to WHAM
- No test verifying that `verbosity` and `reasoning_effort` are still correctly forwarded
- No test verifying the `api_key`/`account_id` validation (ValueError when missing)
- No test verifying the per-call client creation with correct base_url and headers
- No test verifying the streaming response path (`responses.stream()` with `text_format`)
- No test verifying the `parsed is None` error path
- No router dispatch test for `LLMProvider.WHAM`

### `movie_ingestion/metadata_generation/prompts/plot_events.py`
**What changed:** New constant `SYSTEM_PROMPT_SHORT` added -- a shorter variant of the existing `SYSTEM_PROMPT` (~37% token reduction). Same semantic instructions, compressed.
**Existing coverage:** None (no test file for prompts)
**Gaps identified:**
- No test verifying `SYSTEM_PROMPT_SHORT` is a non-empty string
- No test verifying the no-hallucination instruction is preserved in the short variant (this is called out as a critical invariant in the module docstring)
- No test verifying the short prompt mentions all 3 output fields (plot_summary, setting, major_characters)

### `movie_ingestion/metadata_generation/evaluations/plot_events.py`
**What changed:**
- New import of `SYSTEM_PROMPT_SHORT` as `SHORT_SYSTEM_PROMPT`
- 4 new `EvaluationCandidate` entries added to `PLOT_EVENTS_CANDIDATES` using `SHORT_SYSTEM_PROMPT`: Gemini Flash Lite (1k think + short prompt), Gemini Flash Lite (4k think + short prompt), GPT-5-mini (low reasoning + short prompt), GPT-5.4-nano (short prompt)
**Existing coverage:** `unit_tests/test_eval_plot_events.py`
**Gaps identified:**
- Existing test `test_plot_events_candidates_have_unique_ids` covers uniqueness -- this should still pass with the new candidates (no new tests needed for this)
- No test verifying that short-prompt candidates actually reference `SHORT_SYSTEM_PROMPT` (as opposed to `DEFAULT_SYSTEM_PROMPT`)
- No test verifying that each short-prompt candidate has a `candidate_id` containing `"__short-prompt"` suffix (naming convention consistency)
- No test verifying that short-prompt candidates mirror the kwargs of their long-prompt counterparts (ensuring the only variable is the prompt)

### `movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py`
**What changed:**
- Switched from temporary evaluation subset (11 movies) to full `EVALUATION_TEST_SET_TMDB_IDS` (70 movies)
- Reduced concurrency from 10 to 5
- Old `temp_evaluation_set` line is now commented out
**Existing coverage:** `unit_tests/test_eval_run_pipeline.py`
**Gaps identified:**
- Existing test `test_temp_evaluation_set_is_proper_subset_of_full_set` verifies the temp set is 11 IDs -- this test is now stale because the temp set is commented out and the full set is used instead. The test itself still passes (it constructs its own slice), but it tests dead code.
- No test verifying that `main()` uses `EVALUATION_TEST_SET_TMDB_IDS` (the full set)
- No test verifying the concurrency value passed to `run_evaluation`

## Test Plan

### New Tests Needed

#### 1. WHAM Provider: Parameter Stripping
- **File:** `unit_tests/test_generic_methods.py`
- **Tests:**
  - [ ] test_wham_async_strips_max_tokens_from_kwargs -- Pass `max_tokens=4096` in kwargs; verify it does NOT appear in the `responses.stream()` call params
  - [ ] test_wham_async_strips_max_output_tokens_from_kwargs -- Pass `max_output_tokens=2048` in kwargs; verify it does NOT appear in the stream call
  - [ ] test_wham_async_strips_temperature_from_kwargs -- Pass `temperature=0.5` in kwargs; verify it does NOT appear in the stream call
  - [ ] test_wham_async_strips_all_unsupported_params_simultaneously -- Pass all three (`max_tokens`, `max_output_tokens`, `temperature`) at once; verify none leak through
  - [ ] test_wham_async_forwards_verbosity -- Pass `verbosity="low"`; verify it appears in the stream call
  - [ ] test_wham_async_forwards_reasoning_effort_as_nested_object -- Pass `reasoning_effort="low"`; verify stream call gets `reasoning={"effort": "low"}`
  - [ ] test_wham_async_reasoning_effort_none_not_forwarded -- When `reasoning_effort` is not passed, `reasoning` key should not appear in stream params
- **Fixtures/mocks needed:** Mock `AsyncOpenAI` constructor and its `responses.stream()` context manager to capture kwargs. Use a dummy Pydantic model as `response_format`. Need to mock the streaming context manager to return a response with `output_parsed` and `usage`.

#### 2. WHAM Provider: Validation and Error Paths
- **File:** `unit_tests/test_generic_methods.py`
- **Tests:**
  - [ ] test_wham_async_raises_without_api_key -- Call with `api_key=None`; expect `ValueError` mentioning OAuth
  - [ ] test_wham_async_raises_without_account_id -- Call with `account_id=None`; expect `ValueError`
  - [ ] test_wham_async_raises_when_parsed_is_none -- Mock stream to return `output_parsed=None`; expect `ValueError` mentioning "did not contain parsed output"
  - [ ] test_wham_async_returns_tuple_of_three -- Verify return type is `(parsed_response, input_tokens, output_tokens)`
  - [ ] test_wham_async_wraps_exceptions_as_value_error -- Mock stream to raise generic exception; verify it's wrapped as `ValueError` with "WHAM async failed" prefix
- **Fixtures/mocks needed:** Same as above. The streaming mock needs to support both the success and error paths.

#### 3. WHAM Provider: Router Integration
- **File:** `unit_tests/test_generic_methods.py`
- **Tests:**
  - [ ] test_router_dispatches_to_wham -- Verify `generate_llm_response_async(provider=LLMProvider.WHAM, ...)` dispatches to `generate_wham_response_async`
  - [ ] test_provider_dispatch_includes_wham -- Verify `LLMProvider.WHAM` is in `_PROVIDER_DISPATCH`
  - [ ] test_llm_provider_includes_wham -- Verify `LLMProvider.WHAM` exists in the enum
- **Fixtures/mocks needed:** Patch `generate_wham_response_async` as an AsyncMock and verify it receives the expected args.

#### 4. Short Prompt Invariants
- **File:** `unit_tests/test_eval_plot_events.py` (or a new `unit_tests/test_plot_events_prompts.py`)
- **Tests:**
  - [ ] test_system_prompt_short_is_nonempty_string -- `SYSTEM_PROMPT_SHORT` is a non-empty string
  - [ ] test_system_prompt_short_contains_no_hallucination_rule -- Must contain the critical instruction: "Only describe what is evident from the provided data" (or equivalent)
  - [ ] test_system_prompt_short_mentions_all_output_fields -- Must mention `plot_summary`, `setting`, and `major_characters`
  - [ ] test_system_prompt_short_is_shorter_than_default -- `len(SYSTEM_PROMPT_SHORT) < len(SYSTEM_PROMPT)` (the whole point is token reduction)
- **Fixtures/mocks needed:** None -- these are pure string constant tests. Import both `SYSTEM_PROMPT` and `SYSTEM_PROMPT_SHORT` from the prompts module.

#### 5. Short-Prompt Candidates Configuration
- **File:** `unit_tests/test_eval_plot_events.py`
- **Tests:**
  - [ ] test_short_prompt_candidates_use_short_system_prompt -- All candidates with `"__short-prompt"` in their ID use `SHORT_SYSTEM_PROMPT`, not `DEFAULT_SYSTEM_PROMPT`
  - [ ] test_short_prompt_candidates_naming_convention -- Every candidate using `SHORT_SYSTEM_PROMPT` has `"__short-prompt"` suffix in its `candidate_id`
  - [ ] test_short_prompt_candidates_have_long_prompt_counterparts -- For each short-prompt candidate, there exists a corresponding candidate (active or commented) with the same provider/model/kwargs but using the default prompt
- **Fixtures/mocks needed:** Import `PLOT_EVENTS_CANDIDATES` and the two prompt constants.

### Existing Tests to Update

#### `unit_tests/test_eval_plot_events.py`
- [ ] `test_max_tokens_4096_in_reference_generation_calls` -- **STALE**: This test asserts that `generate_llm_response_async` is called with `max_tokens=4096` during reference generation, but the source code on BOTH main and the current branch does NOT pass `max_tokens` in `generate_reference_responses`. The test will fail because `captured_kwargs` won't contain a `max_tokens` key. Either the test was written anticipating a feature that was never merged, or the source was changed after the test was written. This test should be removed or updated to match actual behavior.
- [ ] `test_judge_call_uses_max_tokens_4096` -- **STALE**: Same issue. The judge call in `run_evaluation` does not pass `max_tokens`. This test asserts something that isn't true in the source. Should be removed or updated.
- [ ] `test_judge_call_uses_temperature_0_2` -- **STALE**: Same issue. The judge call does not pass `temperature`. Should be removed or updated.

#### `unit_tests/test_eval_run_pipeline.py`
- [ ] `test_temp_evaluation_set_is_proper_subset_of_full_set` -- **STALE**: Tests the commented-out `temp_evaluation_set` slice logic. The code now uses `EVALUATION_TEST_SET_TMDB_IDS` directly. This test still passes in isolation (it constructs its own slice), but it validates dead code that no longer executes in `main()`. Should be removed or replaced with a test that verifies the pipeline uses the full set.

#### `unit_tests/test_generic_methods.py`
- [ ] `TestLLMProvider.test_llm_provider_values` -- May need updating if it asserts an exact set of provider values and doesn't include WHAM (need to verify)

### Stale Tests
- `test_eval_plot_events.py::TestGenerateReferenceResponses::test_max_tokens_4096_in_reference_generation_calls` -- Asserts `max_tokens=4096` in kwargs, but source code never passes it
- `test_eval_plot_events.py::TestJudgeCallParameters::test_judge_call_uses_max_tokens_4096` -- Asserts `max_tokens=4096` in judge kwargs, but source code never passes it
- `test_eval_plot_events.py::TestJudgeCallParameters::test_judge_call_uses_temperature_0_2` -- Asserts `temperature=0.2` in judge kwargs, but source code never passes it
- `test_eval_run_pipeline.py::TestTempEvaluationSet::test_temp_evaluation_set_is_proper_subset_of_full_set` -- Tests dead (commented-out) code path

## Priority Order
1. **WHAM parameter stripping tests** (groups 1-2) -- Highest priority. This is the only behavioral code change (not just configuration). The WHAM function has zero test coverage, and the change modifies how parameters are handled. A regression here would cause silent API failures or rejected requests. The stripping behavior is the specific fix for a production issue (WHAM rejecting `temperature` when `reasoning_effort != "none"`).
2. **Stale test cleanup** -- High priority. Three tests in `test_eval_plot_events.py` assert behavior that does not exist in the source code. These likely fail when run. They should be removed or corrected before any new tests are added, to avoid confusion about test suite health.
3. **WHAM router integration tests** (group 3) -- Medium priority. Ensures the WHAM provider is correctly wired into the unified dispatch table.
4. **Short prompt invariants** (group 4) -- Medium priority. The no-hallucination instruction is called out as critical in the module docstring. A regression (accidentally removing it during prompt compression) would cause downstream data quality issues.
5. **Short-prompt candidate configuration tests** (group 5) -- Lower priority. These are configuration correctness tests. The candidates are data declarations, not logic, so the risk of subtle bugs is lower. But verifying the naming convention and prompt assignment prevents misconfiguration.
6. **Run pipeline stale test** -- Lower priority. The test passes but validates dead code. Low risk but should be cleaned up for test suite hygiene.
