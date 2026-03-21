# Test Plan
Generated: 2026-03-21
Based on: uncommitted changes vs main

## Summary

The changes span three areas: (1) adding `imdb_title_type` across the IMDB scraping, tracker, and quality scoring pipeline, (2) a major refactor of the plot_events generator from a single-prompt to a two-branch (synopsis/synthesis) design per ADR-033, and (3) smaller signature changes removing `plot_synopsis` from source_of_inspiration and adding `**kwargs` passthrough to OpenAI generation methods. Several operator-precedence bugs in parsers.py were also fixed.

The highest-risk change is the new early-return guards in `compute_imdb_quality_score()` -- existing test fixtures that lack `imdb_title_type` will silently return 0.0 instead of computing the real score. The plot_events generator refactor renames the public prompt-building function and changes its return type, which will break existing tests.

## Changed Files Analysis

### movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py
**What changed:** Added two early-return guards to `compute_imdb_quality_score()`: (a) returns 0.0 if `imdb_title_type` is not in `ALLOWED_TITLE_TYPES` {"movie", "tvMovie", "short", "video"}, (b) returns 0.0 if the movie has no text sources (no plot_summaries, synopses, or featured_reviews). Added `ALLOWED_TITLE_TYPES` constant.
**Current test coverage:** `unit_tests/test_imdb_quality_scorer.py`
**Recommended tests:**
- [ ] Test `compute_imdb_quality_score()` returns 0.0 for `imdb_title_type="tvSeries"` -- validates the title type filter rejects non-movie content
- [ ] Test `compute_imdb_quality_score()` returns 0.0 for `imdb_title_type="videoGame"` -- validates another non-movie type
- [ ] Test `compute_imdb_quality_score()` returns 0.0 for `imdb_title_type=None` -- None is not in ALLOWED_TITLE_TYPES
- [ ] Test `compute_imdb_quality_score()` returns >0.0 for `imdb_title_type="movie"` with sufficient data -- validates allowed types pass through
- [ ] Test `compute_imdb_quality_score()` returns >0.0 for each of "tvMovie", "short", "video" -- validates all allowed types
- [ ] Test returns 0.0 when all text sources are empty (no plot_summaries, no synopses, no featured_reviews) but title type is valid -- validates the text source guard
- [ ] Test returns >0.0 when only featured_reviews are present (no summaries/synopses) -- validates that any single text source is sufficient
- [ ] Test returns >0.0 when only plot_summaries are present -- validates alternate text source
- [ ] Verify `ALLOWED_TITLE_TYPES` contains exactly {"movie", "tvMovie", "short", "video"} -- guards against accidental changes

### movie_ingestion/metadata_generation/generators/plot_events.py
**What changed:** Major refactor: (a) Renamed `build_plot_events_user_prompt()` to `build_plot_events_prompts()` which now returns `(user_prompt, system_prompt)` tuple. (b) Added two-branch logic: synopsis branch (>= 1000 chars) uses `SYSTEM_PROMPT_SYNOPSIS`, synthesis branch uses `SYSTEM_PROMPT_SYNTHESIS`. (c) Short synopses (< 1000 chars) are demoted into the summaries list as the first entry. (d) Removed default provider/model -- now required args. (e) Added `MIN_SYNOPSIS_CHARS = 1000` constant.
**Current test coverage:** `unit_tests/test_plot_events_generator.py`
**Recommended tests:**
- [ ] Test `build_plot_events_prompts()` with a long synopsis (>= 1000 chars) returns `SYSTEM_PROMPT_SYNOPSIS` and includes "plot_synopsis" label in user prompt
- [ ] Test `build_plot_events_prompts()` with a long synopsis does NOT include "plot_summaries" label in user prompt
- [ ] Test `build_plot_events_prompts()` with no synopsis returns `SYSTEM_PROMPT_SYNTHESIS` and includes "plot_summaries" label
- [ ] Test `build_plot_events_prompts()` with no synopsis does NOT include "plot_synopsis" label
- [ ] Test `build_plot_events_prompts()` with a short synopsis (< 1000 chars) returns `SYSTEM_PROMPT_SYNTHESIS` and demotes the synopsis into plot_summaries
- [ ] Test short synopsis demotion prepends to summaries list (appears first before existing summaries)
- [ ] Test `build_plot_events_prompts()` with exactly 1000 chars synopsis selects synopsis branch -- boundary test for `>=`
- [ ] Test `build_plot_events_prompts()` with exactly 999 chars synopsis selects synthesis branch -- boundary test
- [ ] Test newline collapsing in synopsis -- `\n` characters should be replaced with spaces before length check and before inclusion
- [ ] Test summaries capped at 3 entries in synthesis branch -- 4+ summaries should be truncated to 3 (before demotion prepend)
- [ ] Test `generate_plot_events()` requires provider and model args (no defaults) -- call without them should raise TypeError

### movie_ingestion/metadata_generation/generators/source_of_inspiration.py
**What changed:** Removed `plot_synopsis` parameter from `build_source_of_inspiration_user_prompt()` and `generate_source_of_inspiration()` per ADR-033.
**Current test coverage:** `unit_tests/test_source_of_inspiration_generator.py`
**Recommended tests:**
- [ ] Verify `build_source_of_inspiration_user_prompt()` does not accept `plot_synopsis` as a positional or keyword arg -- passing it should raise TypeError
- [ ] Verify the user prompt does not contain a "plot_synopsis" section
- [ ] Verify `generate_source_of_inspiration()` does not accept `plot_synopsis` -- same concern

### movie_ingestion/imdb_scraping/parsers.py
**What changed:** (a) Fixed operator-precedence bug in 3 places: `edge.get("node") or {} if isinstance(edge, dict) else {}` corrected to `(edge.get("node") or {}) if isinstance(edge, dict) else {}`. (b) Added `imdb_title_type` extraction to `transform_graphql_response()`.
**Current test coverage:** `unit_tests/test_imdb_parsers.py`
**Recommended tests:**
- [ ] Test `_extract_synopses_and_summaries()` with a non-dict edge element (e.g., None, int) -- validates the precedence fix prevents AttributeError
- [ ] Test `_score_and_filter_keywords()` with a non-dict edge element -- same precedence fix
- [ ] Test featured reviews extraction with non-dict edge element in the reviews edges list
- [ ] Test `transform_graphql_response()` with `titleType: {"id": "movie"}` -- verifies `imdb_title_type` is extracted as "movie"
- [ ] Test `transform_graphql_response()` with `titleType` absent -- verifies graceful fallback to None
- [ ] Test `transform_graphql_response()` with `titleType: {"id": null}` -- verifies None handling
- [ ] Test `transform_graphql_response()` with `titleType: {"id": "  tvSeries  "}` -- verifies stripping

### movie_ingestion/tracker.py
**What changed:** (a) Added `imdb_title_type TEXT` column to `imdb_data` CREATE TABLE. (b) Added migration `ALTER TABLE imdb_data ADD COLUMN imdb_title_type TEXT`. (c) Added `"imdb_title_type"` as first entry in `IMDB_DATA_COLUMNS`. (d) Replaced an old migration with the new column migration.
**Current test coverage:** `unit_tests/test_tracker.py`
**Recommended tests:**
- [ ] Verify `IMDB_DATA_COLUMNS[0]` is `"imdb_title_type"` -- ensures column order matches the CREATE TABLE definition
- [ ] Verify `"imdb_title_type"` is NOT in `IMDB_JSON_COLUMNS` -- it's a scalar TEXT field, not a JSON array
- [ ] Verify `serialize_imdb_movie()` includes `imdb_title_type` in the output tuple at the correct position (index 1, after tmdb_id)
- [ ] Verify `deserialize_imdb_row()` passes `imdb_title_type` through as a plain string (not JSON-parsed)
- [ ] Verify end-to-end: serialize then deserialize round-trips `imdb_title_type` correctly

### movie_ingestion/imdb_scraping/models.py
**What changed:** Added `imdb_title_type: Optional[str] = None` field to `IMDBScrapedMovie`.
**Current test coverage:** `unit_tests/test_imdb_parsers.py` (indirect)
**Recommended tests:**
- [ ] Verify `IMDBScrapedMovie` defaults `imdb_title_type` to None when not provided -- backwards compatibility
- [ ] Verify `IMDBScrapedMovie(imdb_title_type="movie")` sets the field correctly

### movie_ingestion/imdb_scraping/http_client.py
**What changed:** Added `titleType { id }` to the `_GRAPHQL_QUERY` string.
**Current test coverage:** `unit_tests/test_imdb_http_client.py`
**Recommended tests:**
- [ ] Verify `_GRAPHQL_QUERY` contains `titleType` -- simple assertion on the constant to prevent accidental removal

### movie_ingestion/metadata_generation/pre_consolidation.py
**What changed:** Removed `plot_synopsis` parameter from `_check_source_of_inspiration()`. Updated `assess_skip_conditions()` to match the new signature.
**Current test coverage:** `unit_tests/test_pre_consolidation.py`
**Recommended tests:**
- [ ] Test `_check_source_of_inspiration()` returns None when only `merged_keywords` is provided
- [ ] Test `_check_source_of_inspiration()` returns None when only `review_insights_brief` is provided
- [ ] Test `_check_source_of_inspiration()` returns skip reason string when both are empty/None
- [ ] Test `assess_skip_conditions()` Wave 2 path correctly forwards to updated `_check_source_of_inspiration` without plot_synopsis

### movie_ingestion/metadata_generation/schemas.py
**What changed:** Simplified field descriptions on `MajorCharacter` and `PlotEventsOutput` to minimal text (behavioral instructions moved to branch-specific system prompts per ADR-033).
**Current test coverage:** `unit_tests/test_metadata_schemas.py`
**Recommended tests:**
- [ ] Verify `PlotEventsOutput.__str__()` still produces expected format -- ensures simplified descriptions didn't break embedding text
- [ ] Verify `MajorCharacter.__str__()` still produces expected format
- [ ] Verify `PlotEventsOutput` schema validates with minimal valid data -- `constr(min_length=1)` constraints intact

### movie_ingestion/metadata_generation/prompts/plot_events.py
**What changed:** Added `SYSTEM_PROMPT_SYNOPSIS` (condensation task) and `SYSTEM_PROMPT_SYNTHESIS` (consolidation task with strict no-hallucination framing). Legacy prompts kept for backwards compatibility.
**Current test coverage:** Indirectly via `unit_tests/test_plot_events_generator.py`
**Recommended tests:**
- [ ] Verify `SYSTEM_PROMPT_SYNOPSIS` is non-empty and contains "condense" or "condensation" -- basic sanity
- [ ] Verify `SYSTEM_PROMPT_SYNTHESIS` is non-empty and contains "consolidat" -- basic sanity
- [ ] Verify `SYSTEM_PROMPT_SYNOPSIS` mentions plot_synopsis as PRIMARY source
- [ ] Verify `SYSTEM_PROMPT_SYNTHESIS` contains anti-hallucination constraint ("no knowledge of any film" or similar)

### movie_ingestion/metadata_generation/prompts/source_of_inspiration.py
**What changed:** Added `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` variant. Updated preamble to describe `review_insights_brief` and document ADR-033 removal of plot_synopsis.
**Current test coverage:** Indirectly via `unit_tests/test_source_of_inspiration_generator.py`
**Recommended tests:**
- [ ] Verify `SYSTEM_PROMPT` does not mention "justification" in the OUTPUT section
- [ ] Verify `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` mentions "justification" in the OUTPUT section
- [ ] Verify both prompts contain the parametric knowledge allowance text

### implementation/llms/generic_methods.py
**What changed:** Added `**kwargs` passthrough to `generate_openai_response()` and `generate_openai_response_async()`.
**Current test coverage:** `unit_tests/test_generic_methods.py`
**Recommended tests:**
- [ ] Verify `generate_openai_response()` forwards extra kwargs (e.g., `max_completion_tokens=500`) to the underlying API call
- [ ] Verify `generate_openai_response_async()` forwards extra kwargs similarly

### movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py
**What changed:** Added branch filtering (`--branch synopsis|synthesis`). Added `_filter_plot_events_eligible()` function. Updated imports.
**Current test coverage:** `unit_tests/test_eval_run_pipeline.py`
**Recommended tests:**
- [ ] Test `_filter_plot_events_eligible()` with `branch="synopsis"` skips movies without synopsis
- [ ] Test `_filter_plot_events_eligible()` with `branch="synthesis"` skips movies with synopsis
- [ ] Test `_filter_plot_events_eligible()` with `branch=None` includes all eligible movies
- [ ] Test `_filter_plot_events_eligible()` respects `check_plot_events()` eligibility before branch filtering

### movie_ingestion/metadata_generation/evaluations/plot_events.py
**What changed:** Updated evaluation candidates to use the new two-branch design.
**Current test coverage:** `unit_tests/test_eval_plot_events.py`
**Recommended tests:**
- [ ] Verify `PLOT_EVENTS_CANDIDATES` is non-empty and each candidate has expected attributes

## Stale Tests

The following existing tests are likely broken or need updating:

1. **`test_plot_events_generator.py`** -- **WILL BREAK.** Function renamed from `build_plot_events_user_prompt()` to `build_plot_events_prompts()` with different return type (tuple vs string). Default provider/model removed from `generate_plot_events()`. All tests calling the old API will fail.

2. **`test_source_of_inspiration_generator.py`** -- **WILL BREAK.** Tests calling `build_source_of_inspiration_user_prompt(movie, plot_synopsis, ...)` or `generate_source_of_inspiration(movie, plot_synopsis=..., ...)` will get TypeError since `plot_synopsis` parameter was removed.

3. **`test_pre_consolidation.py`** -- **WILL BREAK.** Tests for `_check_source_of_inspiration()` that pass `plot_synopsis` as an argument will fail. Tests for `assess_skip_conditions()` Wave 2 path may also need updating if they verify the skip reason string (changed from "No keywords, review insights, or plot synopsis available" to "No keywords or review insights available").

4. **`test_imdb_quality_scorer.py`** -- **SILENT FAILURES.** Existing test fixtures that construct `MovieContext` without `imdb_title_type` in the imdb dict will now hit the early-return guard (None not in ALLOWED_TITLE_TYPES) and return 0.0. Tests that assert specific non-zero scores will fail; tests that only assert score > 0 will also fail. All test fixtures need `"imdb_title_type": "movie"` added to the imdb dict. Tests also need `"plot_summaries"`, `"synopses"`, or `"featured_reviews"` to be non-empty to pass the second guard.

5. **`test_imdb_parsers.py`** -- **MAY BREAK.** Tests that assert the exact fields of `transform_graphql_response()` output need to expect `imdb_title_type` in the result. Tests using real-ish GraphQL response fixtures need `titleType` in the input.

6. **`test_tracker.py`** -- **MAY BREAK.** Tests that verify `IMDB_DATA_COLUMNS` length or exact content, or that verify `serialize_imdb_movie()` tuple length/content, will break since a new column was added at the beginning.

7. **`test_eval_run_pipeline.py`** -- **MAY BREAK.** Imports or function references may be stale if they reference the old API.

## Priority Order

1. **`test_imdb_quality_scorer.py` fixture updates + new guard tests** -- Highest risk: silent score changes break all existing scorer tests. Every existing fixture needs `imdb_title_type` and text source data added. Then add new tests for both early-return guards.
2. **`test_plot_events_generator.py` rewrite** -- All tests broken by rename and signature change. Rewrite to test two-branch `build_plot_events_prompts()` with boundary cases (999/1000 chars, demotion, newline collapsing).
3. **`test_source_of_inspiration_generator.py` update** -- Remove `plot_synopsis` from all test calls. Verify prompt no longer includes plot_synopsis section.
4. **`test_pre_consolidation.py` update** -- Remove `plot_synopsis` from `_check_source_of_inspiration` tests. Update skip reason string assertions.
5. **`test_imdb_parsers.py` update** -- Add `imdb_title_type` to transform tests. Add non-dict edge element tests for precedence fix validation.
6. **`test_tracker.py` update** -- Update column list assertions and serialization tuple assertions for new `imdb_title_type` column.
7. **`test_generic_methods.py`** -- Add kwargs passthrough verification for both sync and async OpenAI functions.
8. **`test_metadata_schemas.py`** -- Verify `__str__()` methods still produce expected output after description simplification.
9. **`test_eval_run_pipeline.py`** -- Add tests for `_filter_plot_events_eligible()` branch filtering.
10. **Prompt constant tests** -- Verify new `SYSTEM_PROMPT_SYNOPSIS` / `SYSTEM_PROMPT_SYNTHESIS` / `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` exist and contain expected keywords.
