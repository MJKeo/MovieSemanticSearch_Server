# Test Plan
Generated from diff against main branch (last 5 commits).

## Summary
The recent commits introduced a complete metadata generation pipeline under `movie_ingestion/metadata_generation/`. This includes Pydantic response schemas, input data structures, pre-consolidation logic, 8 generator modules (Wave 1: plot_events, reception; Wave 2: plot_analysis, viewer_experience, watch_context, narrative_techniques, production_keywords, source_of_inspiration), a Wave 1 runner with SQLite persistence, and a WHAM LLM provider in `generic_methods.py`. Existing test files cover the core modules well but have specific gaps identified below.

## Changed Files

| File | Description |
|------|-------------|
| `movie_ingestion/metadata_generation/schemas.py` | Pydantic schemas for all 8 generation types + WithJustifications variants |
| `movie_ingestion/metadata_generation/inputs.py` | MovieInputData, ConsolidatedInputs, SkipAssessment, build_user_prompt, MultiLineList |
| `movie_ingestion/metadata_generation/pre_consolidation.py` | route_keywords, consolidate_maturity, 8 eligibility checks, assess_skip_conditions, run_pre_consolidation |
| `movie_ingestion/metadata_generation/errors.py` | MetadataGenerationError, MetadataGenerationEmptyResponseError |
| `movie_ingestion/metadata_generation/generators/plot_events.py` | Wave 1 generator: build prompt + async LLM call |
| `movie_ingestion/metadata_generation/generators/reception.py` | Wave 1 generator: _truncate_reviews, _format_attributes, build prompt + async LLM call |
| `movie_ingestion/metadata_generation/generators/plot_analysis.py` | Wave 2 generator |
| `movie_ingestion/metadata_generation/generators/viewer_experience.py` | Wave 2 generator |
| `movie_ingestion/metadata_generation/generators/watch_context.py` | Wave 2 generator |
| `movie_ingestion/metadata_generation/generators/narrative_techniques.py` | Wave 2 generator |
| `movie_ingestion/metadata_generation/generators/production_keywords.py` | Wave 2 generator |
| `movie_ingestion/metadata_generation/generators/source_of_inspiration.py` | Wave 2 generator |
| `movie_ingestion/metadata_generation/wave1_runner.py` | SQLite-backed runner for Wave 1 generation + fetch helper |
| `implementation/llms/generic_methods.py` | Added WHAM LLMProvider enum value + generate_wham_response_async |

## Test Coverage Analysis

### schemas.py
**Changed behavior:** All output schemas with `__str__()` methods, extra="forbid" on all models, WithJustifications variants that must produce identical `__str__()` to base variants.
**Existing coverage:** `test_metadata_schemas.py` -- good coverage of `__str__()` parity for all 6 WithJustifications pairs, ReceptionOutput field rename, CharacterArc updates.
**Gaps:**
- [ ] PlotEventsOutput.__str__() -- no direct test. Verify lowercasing, newline joining, and that major_characters are included via MajorCharacter.__str__()
- [ ] PlotEventsOutput.__str__() with empty major_characters list
- [ ] MajorCharacter.__str__() -- no direct test for its format ("name: description Motivations: motivations")
- [ ] MajorCharacter extra="forbid" -- verify extra fields rejected (only CharacterArc is tested, not MajorCharacter)
- [ ] PlotAnalysisOutput.__str__() directly -- only tested via parity, not independently for content correctness (e.g., verify "conflict" suffix on conflict_scale)
- [ ] ViewerExperienceOutput.__str__() independently -- verify comma-separated format, lowercasing
- [ ] WatchContextOutput.__str__() independently -- verify comma-separated format
- [ ] NarrativeTechniquesOutput.__str__() independently -- verify all 11 sections contribute terms
- [ ] ProductionKeywordsOutput.__str__() with empty terms list -- should return ""
- [ ] SourceOfInspirationOutput.__str__() with empty lists -- should return ""
- [ ] TermsSection / TermsWithNegationsSection extra="forbid" validation
- [ ] OptionalTermsWithNegationsSection with should_skip=True -- verify section_data is still required by schema even when skipped
- [ ] constr/conlist constraints -- verify min_length/max_length on PlotAnalysisOutput fields (e.g., genre_signatures min_length=2, character_arcs min_length=1)
- [ ] ReceptionOutput with empty praise_attributes and complaint_attributes -- verify __str__() handles gracefully

### inputs.py
**Changed behavior:** MovieInputData dataclass, build_user_prompt utility, MultiLineList, SkipAssessment, ConsolidatedInputs.
**Existing coverage:** `test_metadata_inputs.py` -- comprehensive coverage of batch_id, title_with_year, defaults, build_user_prompt (basic, None skipping, lists, MultiLineList), merged_keywords, maturity_summary.
**Gaps:**
- [ ] build_user_prompt with integer values -- verify non-string scalars are formatted correctly
- [ ] build_user_prompt field ordering -- verify kwargs order is preserved in output
- [ ] MovieInputData.batch_id with different tmdb_id types (large int, 0)
- [ ] MultiLineList inherits from list -- verify isinstance(MultiLineList([]), list) is True

### pre_consolidation.py
**Changed behavior:** Keyword routing, maturity consolidation, 8 eligibility checks, skip condition orchestrator, run_pre_consolidation.
**Existing coverage:** `test_pre_consolidation.py` -- thorough coverage of route_keywords, consolidate_maturity priority chain, all 8 check functions, assess_skip_conditions for both waves, run_pre_consolidation.
**Gaps:**
- [ ] _all_text_sources_sparse() -- boundary tests: overview exactly 10 chars (should pass), exactly 9 chars (should be sparse)
- [ ] _all_text_sources_sparse() -- combined summaries exactly at 50 char threshold
- [ ] _all_text_sources_sparse() -- multiple short synopses where each is < 50 chars but combined would be >= 50 (function checks each individually, not sum)
- [ ] consolidate_maturity with multiple parental_guide_items -- verify comma-separated formatting
- [ ] assess_skip_conditions Wave 2: verify that when plot_events_output is not None but reception_output IS None, review_insights_brief correctly becomes None
- [ ] assess_skip_conditions Wave 2: verify that when reception_output is not None but plot_events_output IS None, plot_synopsis correctly becomes None
- [ ] run_pre_consolidation -- verify merged_keywords uses normalized/deduped keywords (not raw)
- [ ] check_reception: multiple reviews whose combined text is exactly at threshold (25 chars)
- [ ] _check_source_of_inspiration: eligible via review_insights_brief alone (tested only for keywords and synopsis)

### errors.py
**Changed behavior:** Two custom exception classes with structured attributes.
**Existing coverage:** Tested indirectly via generator error path tests.
**Gaps:**
- [ ] MetadataGenerationError -- direct test: verify attributes (generation_type, title, cause) and message format
- [ ] MetadataGenerationEmptyResponseError -- direct test: verify attributes and message format
- [ ] Both exceptions inherit from Exception (not a custom base)

### generators/plot_events.py
**Changed behavior:** build_plot_events_user_prompt (first synopsis only, newline collapse, summary cap at 3), generate_plot_events (Gemini defaults, _DEFAULT_KWARGS merge).
**Existing coverage:** `test_plot_events_generator.py` -- good coverage: prompt building (synopsis selection, newline collapse, summary cap, empty fields), LLM delegation, return values, error paths.
**Gaps:**
- [ ] _DEFAULT_KWARGS merge: verify that caller-provided kwargs override defaults (e.g., temperature=0.5 overrides default 0.2)
- [ ] _DEFAULT_KWARGS merge: verify that thinking_config default is present when no override
- [ ] Prompt building: verify plot_summaries use MultiLineList format (dash-prefixed items)
- [ ] Prompt building: verify plot_keywords are comma-separated (not dash-prefixed)

### generators/reception.py
**Changed behavior:** _truncate_reviews, _format_attributes, build_reception_user_prompt (review formatting, newline collapse), generate_reception.
**Existing coverage:** `test_reception_generator.py` -- comprehensive: truncation logic, attribute formatting, prompt building, LLM delegation, error paths.
**Gaps:**
- [ ] _truncate_reviews: single review exactly at char limit -- verify it is included
- [ ] build_reception_user_prompt: verify empty reception_summary string ("") is treated as None (omitted)
- [ ] generate_reception: does NOT have system_prompt/response_format override params (unlike Wave 2 generators) -- verify the hardcoded system_prompt and response_format are used

### generators/plot_analysis.py
**Changed behavior:** build_plot_analysis_user_prompt (uses merged_keywords not plot_keywords), generate_plot_analysis (supports system_prompt and response_format override).
**Existing coverage:** `test_plot_analysis_generator.py` -- covers prompt building, LLM delegation, kwargs, error paths.
**Gaps:**
- [ ] Prompt uses merged_keywords (not plot_keywords) -- the test checks "plot_keywords" label but the actual prompt uses "merged_keywords" label. Verify the test is correct or stale.
- [ ] system_prompt override -- verify custom system_prompt is forwarded to LLM
- [ ] response_format override -- verify custom response_format (e.g., PlotAnalysisWithJustificationsOutput) is forwarded

### generators/viewer_experience.py
**Changed behavior:** build_viewer_experience_user_prompt (includes maturity_summary), generate_viewer_experience (system_prompt/response_format override).
**Existing coverage:** `test_viewer_experience_generator.py` -- covers prompt fields, LLM delegation, error paths.
**Gaps:**
- [ ] Prompt includes maturity_summary -- verify it appears in the prompt when available
- [ ] Prompt omits maturity_summary when None -- verify it is excluded
- [ ] system_prompt override -- verify custom system_prompt is forwarded
- [ ] response_format override -- verify custom response_format is forwarded

### generators/watch_context.py
**Changed behavior:** No plot_synopsis parameter, includes maturity_summary, system_prompt/response_format override.
**Existing coverage:** `test_watch_context_generator.py` -- covers prompt fields, no-plot-info constraint, LLM delegation, error paths.
**Gaps:**
- [ ] Stale test: `test_default_reasoning_effort_is_medium` (line 138-149) asserts `reasoning_effort == "medium"` but the generator does NOT define _DEFAULT_KWARGS and does NOT inject reasoning_effort. This test may be checking behavior that was intended but not implemented. **Needs investigation.**
- [ ] system_prompt override -- verify custom system_prompt is forwarded
- [ ] response_format override -- verify custom response_format is forwarded
- [ ] SYSTEM_PROMPT_WITH_JUSTIFICATIONS is imported and re-exported -- verify import does not fail

### generators/narrative_techniques.py
**Changed behavior:** Uses overall_keywords (not merged_keywords), system_prompt/response_format override.
**Existing coverage:** `test_narrative_techniques_generator.py`
**Gaps:**
- [ ] Verify prompt uses overall_keywords (not merged_keywords or plot_keywords)
- [ ] system_prompt override -- verify custom system_prompt is forwarded
- [ ] response_format override -- verify custom response_format is forwarded

### generators/production_keywords.py
**Changed behavior:** Simplest generator -- only title + merged_keywords, no Wave 1 outputs.
**Existing coverage:** `test_production_keywords_generator.py`
**Gaps:**
- [ ] Verify prompt includes only title and merged_keywords (no other fields)
- [ ] Verify prompt omits merged_keywords when empty

### generators/source_of_inspiration.py
**Changed behavior:** Uses merged_keywords + plot_synopsis + review_insights_brief.
**Existing coverage:** `test_source_of_inspiration_generator.py`
**Gaps:**
- [ ] system_prompt override -- verify custom system_prompt is forwarded
- [ ] response_format override -- verify custom response_format is forwarded
- [ ] Verify no genres field in prompt (source_of_inspiration does not use genres)

### wave1_runner.py
**Changed behavior:** SQLite table creation, generate_and_store for plot_events and reception, get_wave1_results fetch helper, idempotent re-runs.
**Existing coverage:** No test file exists.
**Gaps:**
- [ ] init_wave1_table: creates table, is idempotent (call twice without error)
- [ ] generate_and_store_plot_events: skips movies already in DB (idempotent)
- [ ] generate_and_store_plot_events: stores JSON result in correct column
- [ ] generate_and_store_plot_events: handles LLM failures gracefully (failed movies not stored)
- [ ] generate_and_store_plot_events: skips ineligible movies via check_plot_events
- [ ] generate_and_store_plot_events: empty input dict returns early
- [ ] generate_and_store_reception: same set of tests as plot_events above
- [ ] get_wave1_results: returns deserialized PlotEventsOutput and ReceptionOutput
- [ ] get_wave1_results: handles NULL columns (returns None in dict)
- [ ] get_wave1_results: omits movies not in table from result dict
- [ ] get_wave1_results: empty tmdb_ids list returns empty dict

### implementation/llms/generic_methods.py (WHAM provider)
**Changed behavior:** Added LLMProvider.WHAM enum value, generate_wham_response_async function.
**Existing coverage:** No tests for WHAM provider.
**Gaps:**
- [ ] LLMProvider.WHAM enum value exists
- [ ] generate_wham_response_async: raises ValueError when api_key is None
- [ ] generate_wham_response_async: raises ValueError when account_id is None
- [ ] generate_wham_response_async: strips max_tokens, max_output_tokens, temperature from kwargs
- [ ] generate_wham_response_async: constructs correct base_url and headers
- [ ] generate_wham_response_async: forwards reasoning_effort correctly
- [ ] generate_llm_response_async routing: LLMProvider.WHAM dispatches to generate_wham_response_async

## Stale Tests

1. **`test_watch_context_generator.py::TestGenerateWatchContext::test_default_reasoning_effort_is_medium`** -- This test asserts that `reasoning_effort == "medium"` appears in the LLM call kwargs by default. However, `generate_watch_context` does NOT define `_DEFAULT_KWARGS` and does NOT inject reasoning_effort. The test may be checking for behavior that was intended but not implemented, or may rely on a caller convention. Needs investigation.

2. **`test_plot_analysis_generator.py::TestBuildPlotAnalysisUserPrompt::test_includes_plot_keywords`** -- The test checks for `plot_keywords` label but `build_plot_analysis_user_prompt` passes `merged_keywords=movie.merged_keywords()` to `build_user_prompt`. The label in the output will be "merged_keywords", not "plot_keywords". This test may be asserting the wrong field name. Needs investigation.

## Priority

### Critical (correctness of core pipeline)
1. **wave1_runner.py** -- zero test coverage for SQLite persistence, idempotent re-runs, and the fetch helper that Wave 2 evaluation depends on
2. **Stale test investigation** -- watch_context reasoning_effort and plot_analysis keyword label tests may be silently passing with wrong assertions or failing

### High (schema correctness for embedding)
3. **PlotEventsOutput.__str__()** and **MajorCharacter.__str__()** -- these directly affect vector embeddings
4. **Schema constraint validation** -- conlist min_length/max_length enforcement on PlotAnalysisOutput
5. **Empty list edge cases** in __str__() methods across all schemas

### Medium (generator robustness)
6. **system_prompt/response_format override** tests across Wave 2 generators -- needed for evaluation pipeline
7. **_DEFAULT_KWARGS merge** behavior in plot_events generator
8. **errors.py** direct tests

### Low (defense in depth)
9. **WHAM provider** tests in generic_methods.py
10. **build_user_prompt** edge cases (integer values, field ordering)
11. **MultiLineList** isinstance checks
