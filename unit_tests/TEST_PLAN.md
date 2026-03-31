# Test Plan

Generated: 2026-03-31
Source: DIFF_CONTEXT.md (committed changes)

## Summary of Changes

**Schema redesign (schemas.py):**
- PlotAnalysisOutput removed entirely; replaced by PlotAnalysisWithJustificationsOutput as the sole production schema. Fields renamed: `core_concept_label` -> `elevator_pitch`, `conflict_scale` -> `conflict_type` (now a list), `themes_primary` + `lessons_learned` merged into `thematic_concepts`. `CharacterArc` replaced by `CharacterArcWithReasoning`. `CoreConceptWithJustification` -> `ElevatorPitchWithJustification`. `MajorThemeWithJustification` + `MajorLessonLearnedWithJustification` -> `ThematicConceptWithJustification`.
- `OptionalTermsWithNegationsSection` removed from ViewerExperience; all 8 sections now use flat `TermsWithNegationsSection` directly (no `should_skip` wrapper).
- NarrativeTechniques field order changed (reordered for autoregressive generation).

**Generator signature changes:**
- `build_plot_analysis_user_prompt(movie, plot_summary, thematic_observations)` -- 3rd param renamed from `review_insights_brief` to `thematic_observations`.
- `build_viewer_experience_user_prompt(movie, generalized_plot_overview, emotional_observations, craft_observations, thematic_observations, genre_signatures)` -- completely new signature; was `(movie, plot_synopsis, review_insights_brief)`.
- `build_narrative_techniques_user_prompt(movie, plot_summary, craft_observations)` -- was `(movie, plot_synopsis, review_insights_brief)`.
- `build_source_of_inspiration_user_prompt(movie, source_material_hint)` -- 2nd param renamed from `review_insights_brief` to `source_material_hint`.

**Generator registry:** Now registers 5 types (PLOT_EVENTS, RECEPTION, PLOT_ANALYSIS, PRODUCTION_KEYWORDS, VIEWER_EXPERIENCE); was 2.

**Pre-consolidation:** `_check_source_of_inspiration` 2nd param renamed from `review_insights_brief` to `source_material_hint`. New tiered skip logic, unified observation filtering, new narrative resolution functions.

**Result processor:** SCHEMA_BY_TYPE now includes 5 types (added PLOT_ANALYSIS, PRODUCTION_KEYWORDS, VIEWER_EXPERIENCE).

**Inputs:** New `Wave1Outputs` dataclass, `load_wave1_outputs()`, `load_plot_analysis_output()`, `best_plot_fallback()` on `MovieInputData`.

**IMDB scraping:** Removed artificial query limits in http_client.py and interests cap in parsers.py.

---

## Broken Tests (Immediate Fixes Needed)

### test_metadata_schemas.py

1. **Import errors -- removed classes/symbols (FATAL):**
   - Line 19: `CharacterArc` -- removed, replaced by `CharacterArcWithReasoning`
   - Line 20: `CoreConceptWithJustification` -- removed, replaced by `ElevatorPitchWithJustification`
   - Line 21: `MajorThemeWithJustification` -- removed, replaced by `ThematicConceptWithJustification`
   - Line 22: `MajorLessonLearnedWithJustification` -- removed (merged into `ThematicConceptWithJustification`)
   - Line 23: `PlotAnalysisOutput` -- removed (production schema is `PlotAnalysisWithJustificationsOutput` only)
   - Line 28: `OptionalTermsWithNegationsSection` -- removed from ViewerExperience schemas
   - Line 29: `OptionalTermsWithNegationsAndJustificationSection` -- removed

2. **TestCharacterArcUpdated (lines 182-207):** References `CharacterArc` which no longer exists. Must use `CharacterArcWithReasoning` which likely has different field names.

3. **TestCoreConceptWithJustificationStr (lines 214-221):** References `CoreConceptWithJustification` which is now `ElevatorPitchWithJustification`. Field `core_concept_label` renamed to `elevator_pitch`.

4. **TestMajorThemeWithJustificationStr (lines 224-230):** Class removed, replaced by `ThematicConceptWithJustification` with field `concept_label` instead of `theme_label`.

5. **TestMajorLessonLearnedWithJustificationStr (lines 234-240):** Class entirely removed.

6. **_PLOT_ANALYSIS_DATA shared test data (lines 249-260):** Uses `CharacterArc` (removed), `conflict_scale` (renamed to `conflict_type`, now a list). All `PlotAnalysisOutput` tests below this are broken.

7. **TestPlotAnalysisWithJustificationsStrParity (lines 263-343):** All tests broken. `PlotAnalysisOutput` removed. `PlotAnalysisWithJustificationsOutput` now has different fields: `elevator_pitch_with_justification` instead of `core_concept`, `thematic_concepts` instead of `themes_primary`/`lessons_learned`, `conflict_type` instead of `conflict_scale`, `CharacterArcWithReasoning` instead of `CharacterArc`.

8. **TestPlotAnalysisOutputStrContent (lines 643-710):** `PlotAnalysisOutput` removed. `conflict_scale` field removed (was testing " conflict" suffix). All assertions reference deleted schema.

9. **TestViewerExperienceWithJustificationsStrParity (lines 371-478):** Uses `OptionalTermsWithNegationsSection` and `OptionalTermsWithNegationsAndJustificationSection` which are removed. ViewerExperience sections no longer use skip wrappers.

10. **TestViewerExperienceOutputStrContent (lines 717-745):** Uses `OptionalTermsWithNegationsSection` which is removed.

11. **TestOptionalSectionDataRequired (lines 833-837):** Tests `OptionalTermsWithNegationsSection` which is removed.

### test_plot_analysis_generator.py

12. **`_make_plot_analysis_output()` helper (lines 51-66):** Creates `PlotAnalysisOutput` which no longer exists. Uses `CharacterArc` (removed), `core_concept_label` (removed), `conflict_scale` (removed), `themes_primary` (removed), `lessons_learned` (removed).

13. **`build_plot_analysis_user_prompt` signature change (line 76-127):** Tests call `build_plot_analysis_user_prompt(movie, None, None)` -- the 3rd parameter was `review_insights_brief`, now it's `thematic_observations`. Tests that check for `"review_insights_brief:"` in prompt output (line 97) will fail because the label is now different.

14. **`generate_plot_analysis` return type:** Tests assert `parsed is expected` where `expected` is `PlotAnalysisOutput` -- if mock returns this, validation may mismatch with actual expected type.

### test_viewer_experience_generator.py

15. **`build_viewer_experience_user_prompt` signature change (lines 77-138):** Tests call `build_viewer_experience_user_prompt(movie, None, None)` -- old signature was `(movie, plot_synopsis, review_insights_brief)`, new is `(movie, generalized_plot_overview, emotional_observations, craft_observations, thematic_observations, genre_signatures)`. All calls with 3 positional args will get wrong parameter mapping. The `test_includes_review_insights_brief` test (line 111) references a removed parameter.

16. **`_make_ve_output()` helper (lines 55-69):** Uses `OptionalTermsWithNegationsSection` which is removed. All 8 sections should now be `TermsWithNegationsSection` directly.

### test_narrative_techniques_generator.py

17. **`build_narrative_techniques_user_prompt` signature change (lines 73-118):** Tests call `build_narrative_techniques_user_prompt(movie, None, None)` -- old was `(movie, plot_synopsis, review_insights_brief)`, new is `(movie, plot_summary, craft_observations)`. Test `test_includes_review_insights_brief` (line 95) references removed parameter. Label checks may fail.

### test_source_of_inspiration_generator.py

18. **`build_source_of_inspiration_user_prompt` label change:** `test_includes_review_insights_brief` (line 72) asserts `"review_insights_brief:"` appears in the prompt -- label is now `"source_material_hint:"`. Positional calls still work, but the assertion on the output label is wrong.

### test_pre_consolidation.py

19. **`_make_plot_events_output()` helper (lines 74-82):** Creates `PlotEventsOutput` with `setting` and `major_characters` fields -- these were removed from `PlotEventsOutput` (extra="forbid"). Will raise `ValidationError` at construction.

20. **`_make_reception_output()` helper (lines 85-94):** Creates `ReceptionOutput` with `new_reception_summary`, `praise_attributes`, `complaint_attributes`, `review_insights_brief` -- all old field names. `ReceptionOutput` has `extra="forbid"` so this raises `ValidationError`.

21. **`TestAssessSkipConditionsWave2Partial` (lines 648-675):** Uses `_make_reception_output()` which is broken (see item 20).

### test_generator_registry.py

22. **`TestGeneratorRegistry.test_has_exactly_registered_types` (line 64):** Asserts registry has exactly `{PLOT_EVENTS, RECEPTION}` -- now has 5 types (added PLOT_ANALYSIS, PRODUCTION_KEYWORDS, VIEWER_EXPERIENCE). Will fail.

23. **`TestGetConfig.test_raises_key_error_for_unregistered_type` (line 98):** Uses `MetadataType.PLOT_ANALYSIS` as example of unregistered type -- PLOT_ANALYSIS is now registered. Will fail.

### test_result_processor.py

24. **`TestSchemaByType.test_includes_both_types` (line 141-143):** Asserts SCHEMA_BY_TYPE has PLOT_EVENTS and RECEPTION -- it now has 5 types. Test passes but is incomplete/stale.

25. **`TestProcessResultsReception.test_unknown_metadata_type_records_failure` (line 405-420):** Uses `"narrative_techniques_1"` as unregistered type example. Currently still passes (narrative_techniques not in SCHEMA_BY_TYPE), but fragile if more types get registered.

---

## New Test Coverage Needed

### Priority 1 -- Schema correctness (replaces broken tests)

1. **PlotAnalysisWithJustificationsOutput field validation:** Test `elevator_pitch_with_justification`, `thematic_concepts`, `conflict_type` (list), `character_arcs` (now `CharacterArcWithReasoning`), `genre_signatures`, `generalized_plot_overview`. Test `__str__()` produces correct embedding text.
2. **ElevatorPitchWithJustification `__str__()`:** Returns only the pitch, not the justification.
3. **ThematicConceptWithJustification `__str__()`:** Returns only `concept_label`.
4. **CharacterArcWithReasoning `__str__()`:** Returns only the label.
5. **ViewerExperienceOutput simplified schema:** All 8 sections are `TermsWithNegationsSection` directly (no skip wrappers). Test `__str__()` with the flattened structure.
6. **ViewerExperienceWithJustificationsOutput `__str__()` parity** with the simplified base variant.
7. **PlotAnalysisWithJustificationsOutput constraint validation:** `conflict_type` max_length=2, `character_arcs` min_length=0 max_length=3, `thematic_concepts` max_length=5. Test boundary values.

### Priority 2 -- Generator prompt/signature changes

8. **`build_plot_analysis_user_prompt`:** Test with `thematic_observations` parameter. Verify `"thematic_observations:"` label in output. Test plot_summary fallback via `best_plot_fallback()` and the `"plot_text"` vs `"plot_summary"` label distinction.
9. **`build_viewer_experience_user_prompt`:** Test new 6-parameter signature: `generalized_plot_overview`, `emotional_observations`, `craft_observations`, `thematic_observations`, `genre_signatures`. Test that `genre_signatures` overrides raw genres. Test explicit "not available" signals.
10. **`build_narrative_techniques_user_prompt`:** Test with `craft_observations` parameter. Verify explicit "not available" signals for absent inputs. Test craft_observations minimum length filtering.
11. **`build_source_of_inspiration_user_prompt`:** Test `source_material_hint` label in output.

### Priority 3 -- New modules and logic

12. **`Wave1Outputs` dataclass:** Test construction, default None fields, field access.
13. **`load_wave1_outputs()`:** Test DB loading, missing data handling, JSON parsing, field extraction from parsed output objects.
14. **`load_plot_analysis_output()`:** Test DB loading and parsing.
15. **`MovieInputData.best_plot_fallback()`:** Test priority chain (synopsis > summary > overview), empty inputs, length-based selection.
16. **Generator registry expansion:** Test that all 5 registered types return valid configs with correct schema_class, model, and prompt builders. Update the exact-set assertion.
17. **`SCHEMA_BY_TYPE` coverage:** Test all 5 registered types are present with correct schema classes.
18. **`result_processor` for new types:** Test processing plot_analysis, production_keywords, viewer_experience batch results (similar to existing reception tests).
19. **`request_builder` `max_batches` parameter:** Test early truncation limits number of batches returned.
20. **Result processor expired batch handling:** Test behavior when batch status is expired.

### Priority 4 -- Pre-consolidation changes

21. **Tiered skip conditions for plot_analysis:** The check may have new logic beyond the simple "synopsis or insights" test.
22. **Tiered skip conditions for viewer_experience:** Test new multi-signal eligibility (GPO, observations, contextual data).
23. **Tiered skip conditions for narrative_techniques:** Test new eligibility criteria.
24. **Unified observation filtering in pre_consolidation:** Test `craft_observations` minimum length filtering.
25. **Shared narrative resolution functions:** Test plot text fallback ladder used across generators.

---

## Edge Cases and Risk Areas

1. **`conflict_type` is now a list (0-2 items), not a scalar string.** Any code or test that appends " conflict" to a scalar will break. The `__str__()` method now does `", ".join(self.conflict_type).lower()` -- test with 0, 1, and 2 items.

2. **`character_arcs` min_length changed from 1 to 0.** Tests that assert `min_length=1` validation will fail. The new schema allows empty character_arcs when input data is sparse.

3. **ViewerExperience `should_skip` removal.** The old schema had `OptionalTermsWithNegationsSection` wrappers with `should_skip=True/False` for disturbance_profile, sensory_load, emotional_volatility. The new schema uses flat `TermsWithNegationsSection` -- these sections simply produce empty term lists when not applicable. Any test relying on `should_skip` semantics is broken.

4. **NarrativeTechniques field ordering in `__str__()`.** The old `__str__()` iterated sections in the declaration order `pov_perspective, narrative_delivery, narrative_archetype, ...`. The new order is `narrative_archetype, narrative_delivery, additional_plot_devices, pov_perspective, ...`. While the sections all contribute to a comma-separated output, the ORDER of terms changes, which could affect string equality tests.

5. **`best_plot_fallback()` priority chain.** This is new and used by Wave 2 generators when plot_events output is unavailable. The fallback order (synopsis > summary > overview) and length-based selection need thorough testing.

6. **Prompt "not available" signals.** Narrative techniques now explicitly includes `"plot_synopsis: not available"` rather than omitting the field. Tests that assert field absence need updating.

7. **`PlotAnalysisOutput` removal ripple effects.** Any test file that imports `PlotAnalysisOutput` will fail at import time, causing the entire test file to be skipped. This affects test_metadata_schemas.py and test_plot_analysis_generator.py completely.

---

## Testing Conventions Observed

- **Factory helpers:** Each test file has `_make_movie(**overrides)` and type-specific output helpers like `_make_ve_output()`. Defaults are minimal; tests override only what they need.
- **Async tests:** No `@pytest.mark.asyncio` decorator needed (autodetected via `asyncio_mode = "auto"`). Tests are plain `async def`.
- **LLM mocking:** All LLM calls mocked via `patch(MODULE_PATH.generate_llm_response_async)` with `AsyncMock`. Mock returns `(output, input_tokens, output_tokens)` tuple.
- **DB fixtures:** `tracker_db` fixture creates temp SQLite DB with `_SCHEMA_SQL` schema. Seeds movie rows via `_seed_movie()` helper.
- **Assertion style:** Direct `assert` statements, `pytest.raises` for error paths. No assertion messages unless they provide unique diagnostic value.
- **Test class grouping:** Related tests grouped in classes (`TestBuildPlotAnalysisUserPrompt`, `TestGeneratePlotAnalysisErrors`). No `setUp`/`tearDown`.
- **Import style:** Explicit named imports, no wildcard imports.
- **Extra="forbid" testing:** Tests verify that passing old/removed field names raises `ValidationError`.
- **Parity testing:** WithJustifications variants tested for `__str__()` parity with base variants.
