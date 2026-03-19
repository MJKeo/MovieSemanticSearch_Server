# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Implement reception metadata generator (Wave 1)
Files: movie_ingestion/metadata_generation/generators/reception.py, movie_ingestion/metadata_generation/prompts/reception.py, movie_ingestion/metadata_generation/schemas.py, movie_ingestion/metadata_generation/analyze_eligibility.py

### Intent
Second Wave 1 generator — produces evaluative reception metadata for embedding + review_insights_brief intermediate for Wave 2 consumption. Follows the same async real-time pattern as plot_events.py for evaluation before batch API integration.

### Key Decisions
- Renamed `ReceptionOutput.reception_summary` → `new_reception_summary` to differentiate from the scraped `MovieInputData.reception_summary` input field. Matches search-side `ReceptionMetadata` naming.
- Review truncation: adds reviews one at a time, stops when 5 reviews or 2500 combined chars is reached. The review crossing the threshold IS included.
- System prompt based on legacy RECEPTION_SYSTEM_PROMPT with additions for review_insights_brief (evaluative vs descriptive distinction, source material extraction directive).
- Defaults: OpenAI gpt-5-mini, reasoning_effort: low — pending evaluation.
- Two test files (test_metadata_schemas.py, test_pre_consolidation.py) will break from the field rename — not modified per test-boundaries rule.

### Testing Notes
- Import checks pass for generator, schema, and prompt
- Prompt builder smoke-tested with full and sparse movie data
- Review truncation logic verified for char limit, count limit, and empty cases

## Implement plot_analysis metadata generator (Wave 2)
Files: movie_ingestion/metadata_generation/generators/plot_analysis.py, movie_ingestion/metadata_generation/prompts/plot_analysis.py, movie_ingestion/metadata_generation/schemas.py, docs/TODO.md

### Intent
First Wave 2 generator — extracts thematic meaning (core concept, genre signatures, character arcs, themes, lessons, generalized overview). Consumes plot_synopsis from Wave 1 plot_events and review_insights_brief from Wave 1 reception. Follows the same async real-time pattern as plot_events.py.

### Key Decisions
- Added `arc_transformation_description` to generation-side `CharacterArc` to match search-side schema. Not a justification — helps produce accurate arc labels. Not embedded (excluded from `__str__`).
- Created `PlotAnalysisWithJustificationsOutput` variant with structured sub-models (`CoreConceptWithJustification`, `MajorThemeWithJustification`, `MajorLessonLearnedWithJustification`) for evaluation comparison. `__str__()` produces identical embedding text to `PlotAnalysisOutput`.
- Skip condition deliberately more lenient than spec: `plot_synopsis OR review_insights_brief` (spec requires only plot_synopsis). Discussed and confirmed — review_insights_brief contains useful thematic signal.
- System prompt adapted from legacy `PLOT_ANALYSIS_SYSTEM_PROMPT`: removed overview/reception_summary/featured_reviews inputs, added review_insights_brief, title as "Title (Year)", added arc_transformation_description to character_arcs field instructions.
- Two system prompt variants: `SYSTEM_PROMPT` (flat labels) and `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` (sub-objects with explanation_and_justification). Shared sections extracted into private constants; only fields 1, 5, 6 differ between variants.
- Defaults: OpenAI gpt-5-mini, reasoning_effort: low — matching current system, pending evaluation.
- Added `ConfigDict(extra="forbid")` to `CharacterArc` for consistency with all other sub-models.
- User also added `ViewerExperienceWithJustificationsOutput` variant to schemas.py (with `TermsWithNegationsAndJustificationSection` and `OptionalTermsWithNegationsAndJustificationSection` sub-models) outside of this implementation task.

### Testing Notes
- Import checks pass for generator, schema variants, and both prompt variants
- Prompt builder smoke-tested with full and sparse movie data
- `__str__()` parity between PlotAnalysisOutput and PlotAnalysisWithJustificationsOutput verified
- Both prompt variants verified: shared sections identical, variant-specific content correct

## Implement viewer_experience metadata generator (Wave 2)
Files: movie_ingestion/metadata_generation/generators/viewer_experience.py, movie_ingestion/metadata_generation/prompts/viewer_experience.py, movie_ingestion/metadata_generation/schemas.py, movie_ingestion/metadata_generation/inputs.py

### Intent
Second Wave 2 generator — extracts what it FEELS LIKE to watch the movie (emotional palette, tension, tone, cognitive complexity, disturbance, sensory load, emotional volatility, ending aftertaste). Consumes plot_synopsis from Wave 1 plot_events and review_insights_brief from Wave 1 reception. Can run without plot data if review data exists.

### Key Decisions
- Added `merged_keywords()` and `maturity_summary()` methods to `MovieInputData` so generators can derive consolidated fields directly from the input object without needing `ConsolidatedInputs`. Logic matches `pre_consolidation.route_keywords()` and `consolidate_maturity()` respectively. `_MPAA_DEFINITIONS` constant duplicated in inputs.py to avoid circular import with pre_consolidation.py.
- Created `ViewerExperienceWithJustificationsOutput` variant with `TermsWithNegationsAndJustificationSection` and `OptionalTermsWithNegationsAndJustificationSection` sub-models. Justification field placed FIRST in field order for chain-of-thought ordering. `__str__()` produces identical embedding text to `ViewerExperienceOutput`.
- System prompt adapted from legacy `VIEWER_EXPERIENCE_SYSTEM_PROMPT`: replaced 11 input descriptions with 6 consolidated inputs (merged_keywords, maturity_summary, review_insights_brief). All 8 section descriptions with examples preserved verbatim.
- Two prompt variants: `SYSTEM_PROMPT` (no justifications) and `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` (adds per-section justification field). Shared sections extracted into private constants (`_OPENING_AND_CONTEXT`, `_INPUTS_AND_RULES`, `_SECTIONS`); only "Primary goal" and "Output expectations" differ between variants.
- Defaults: OpenAI gpt-5-mini, reasoning_effort: low — matching current system, pending evaluation.

### Testing Notes
- Import checks pass for generator, both schema variants, and prompt
- MovieInputData methods verified: merged_keywords dedup + normalization, maturity_summary all 4 priority chain paths
- User prompt builder verified with full and sparse data (None optional inputs correctly omitted)
- `__str__()` parity between ViewerExperienceOutput and ViewerExperienceWithJustificationsOutput verified
- All 103 existing tests pass (1 pre-existing failure in test_eval_shared.py unrelated to changes)

## Implement watch_context metadata generator (Wave 2)
Files: movie_ingestion/metadata_generation/generators/watch_context.py, movie_ingestion/metadata_generation/prompts/watch_context.py, movie_ingestion/metadata_generation/schemas.py, docs/TODO.md

### Intent
Third Wave 2 generator — extracts WHY and WHEN someone would choose to watch this movie (self-experience motivations, external motivations, key feature draws, watch scenarios). Consumes review_insights_brief from Wave 1 reception. Deliberately receives ZERO plot information (Decision 2 in the redesigned flow spec).

### Key Decisions
- Created `TermsWithJustificationSection` sub-model: adds `justification` to `TermsSection` (no negations). Mirrors search-side `GenericTermsSection` structure. Reusable by any future generation type with TermsSection-based output.
- Created `WatchContextWithJustificationsOutput` variant using `TermsWithJustificationSection`. `__str__()` produces identical embedding text to `WatchContextOutput`.
- Watch context receives `merged_keywords` (not `overall_keywords` as spec originally stated — corrected during spec understanding conversation).
- No `plot_synopsis` parameter on the generator — only Wave 2 generator with no plot input.
- System prompt adapted from legacy `WATCH_CONTEXT_SYSTEM_PROMPT`: replaced 8 input descriptions with 5 (merged_keywords, maturity_summary, review_insights_brief, title as "Title (Year)", genres). Section 3 updated "user reviews" → "review_insights_brief". All 4 section descriptions with examples preserved verbatim otherwise.
- Two prompt variants: `SYSTEM_PROMPT` (no justifications) and `SYSTEM_PROMPT_WITH_JUSTIFICATIONS`. Shared sections extracted into `_PREAMBLE` and `_SECTIONS`; only `_OUTPUT_*` paragraph differs.
- Defaults: OpenAI gpt-5-mini, reasoning_effort: medium — matching legacy system, pending evaluation.
- Added TODO for search-side `WatchContextMetadata.__str__()` lowercase alignment.

### Testing Notes
- Import checks needed for generator, schema variants, and prompt
- Prompt builder needs smoke-test with full and sparse movie data
- `__str__()` parity between WatchContextOutput and WatchContextWithJustificationsOutput should be verified

## Implement narrative_techniques metadata generator (Wave 2)
Files: movie_ingestion/metadata_generation/generators/narrative_techniques.py, movie_ingestion/metadata_generation/prompts/narrative_techniques.py, movie_ingestion/metadata_generation/schemas.py

### Intent
Fourth Wave 2 generator — extracts HOW the story is told (POV, temporal structure, narrative archetype, information control, characterization, character arcs, audience perception, conflict design, thematic delivery, meta techniques, plot devices). Consumes plot_synopsis from Wave 1 plot_events and review_insights_brief from Wave 1 reception.

### Key Decisions
- Reuses existing `TermsWithJustificationSection` (created for watch_context) for the with-justifications variant. No new sub-models needed.
- `NarrativeTechniquesWithJustificationsOutput` uses `TermsWithJustificationSection` for all 11 sections. `__str__()` produces identical embedding text to `NarrativeTechniquesOutput`.
- Uses `overall_keywords` only (not `merged_keywords`) — structural tags like "nonlinear timeline" live in overall keywords. Plot keywords add noise without structural signal.
- System prompt adapted from legacy `NARRATIVE_TECHNIQUES_SYSTEM_PROMPT`: updated inputs (title as "Title (Year)", added genres, overall_keywords only, review_insights_brief replaces reception_summary + featured_reviews). All 11 category guidance sections preserved verbatim.
- Two prompt variants: `SYSTEM_PROMPT` (no justifications) and `SYSTEM_PROMPT_WITH_JUSTIFICATIONS`. Only the OUTPUT EXPECTATIONS paragraph differs.
- Added "Do not supplement with your own knowledge of this film" to HOW TO USE section (consistent with no-hallucination stance for non-source_of_inspiration generators).
- Defaults: OpenAI gpt-5-mini, reasoning_effort: medium — matching legacy system, pending evaluation.

### Testing Notes
- Import checks needed for generator, schema variants, and prompt
- Prompt builder needs smoke-test with full and sparse movie data
- `__str__()` parity between NarrativeTechniquesOutput and NarrativeTechniquesWithJustificationsOutput should be verified

## Implement production_keywords and source_of_inspiration generators (Wave 2)
Files: movie_ingestion/metadata_generation/generators/production_keywords.py (new),
       movie_ingestion/metadata_generation/generators/source_of_inspiration.py (new),
       movie_ingestion/metadata_generation/generators/production.py (deleted),
       movie_ingestion/metadata_generation/schemas.py,
       movie_ingestion/metadata_generation/prompts/production_keywords.py,
       movie_ingestion/metadata_generation/prompts/source_of_inspiration.py,
       movie_ingestion/metadata_generation/generators/__init__.py,
       movie_ingestion/metadata_generation/generators/plot_analysis.py,
       movie_ingestion/metadata_generation/generators/reception.py,
       movie_ingestion/metadata_generation/generators/viewer_experience.py,
       movie_ingestion/metadata_generation/generators/watch_context.py,
       movie_ingestion/metadata_generation/generators/narrative_techniques.py

### Intent
Final two Wave 2 generators — production_keywords (keyword classification task) and source_of_inspiration (source material + production medium identification). Also fixes missing verbosity="low" across all OpenAI generators.

### Key Decisions
- Split into two separate generator files (production_keywords.py, source_of_inspiration.py) replacing the single production.py scaffold. Follows one-file-per-generation-type pattern consistent with all other generators.
- production_keywords takes only title + merged_keywords (simplest generator — no Wave 1 outputs needed). source_of_inspiration takes title + merged_keywords + plot_synopsis + review_insights_brief.
- Both prompts adapted from legacy prompts with: "Title (Year)" format, merged_keywords, parametric knowledge allowance for source_of_inspiration only, no justification in base variant.
- Added `ProductionKeywordsWithJustificationsOutput` and `SourceOfInspirationWithJustificationsOutput` to schemas.py for evaluation comparison. Both use a single justification field (matching legacy search-side GenericTermsSection and SourceOfInspirationSection). `__str__()` produces identical embedding text to base variants.
- Added `"verbosity": "low"` to `_DEFAULT_KWARGS` for all 5 OpenAI-provider generators (plot_analysis, reception, viewer_experience, watch_context, narrative_techniques). Not plot_events (Gemini provider, doesn't support verbosity). Matches legacy sync calls which all passed verbosity="low".
- Both new generators default to OpenAI gpt-5-mini, reasoning_effort: low, verbosity: low.
- Updated generators/__init__.py docstring to reflect current async real-time caller contract (was stale, described old batch API body-dict interface).

### Testing Notes
- Import checks pass for both new generators, schema variants, and all 4 prompt exports
- Prompt builders smoke-tested with full and sparse MovieInputData
- `__str__()` parity verified between all Output and WithJustificationsOutput variants (including empty lists)
- Confirmed production.py scaffold deleted and not imported anywhere

## Implement test suite for metadata generation pipeline
Files: unit_tests/test_metadata_schemas.py, unit_tests/test_metadata_inputs.py, unit_tests/test_pre_consolidation.py,
       unit_tests/test_reception_generator.py (new), unit_tests/test_plot_analysis_generator.py (new),
       unit_tests/test_viewer_experience_generator.py (new), unit_tests/test_watch_context_generator.py (new),
       unit_tests/test_narrative_techniques_generator.py (new), unit_tests/test_production_keywords_generator.py (new),
       unit_tests/test_source_of_inspiration_generator.py (new)

### Intent
Comprehensive test coverage for all new metadata generation code. Ran test planner subagent to analyze diff, then implemented all recommended tests.

### Key Decisions
- Fixed 3 stale tests broken by `reception_summary` → `new_reception_summary` rename and empty-list behavior change
- Added 6 WithJustifications `__str__()` parity tests (critical invariant: embedding text must be identical)
- Added `merged_keywords()` tests (9 cases: dedup, normalization, ordering, edge cases)
- Added `maturity_summary()` tests (5 cases including delegation verification)
- Added `MultiLineList` formatting tests (3 cases)
- Created 7 new generator test files following `test_plot_events_generator.py` pattern (prompt building, LLM delegation, error paths)
- Reception generator tests include dedicated coverage for `_truncate_reviews()` (8 cases) and `_format_attributes()` (3 cases)
- All 222 tests pass

## Reception generator: strip review newlines + raise char cap
Files: movie_ingestion/metadata_generation/generators/reception.py
Why: Featured reviews contain embedded newlines that waste tokens and can confuse the LLM (same issue plot_events already solved for synopses). Char cap raised from 2500→5000 to allow more review text.
Approach: Added `re.sub(r'\n+', ' ', ...)` on review text during formatting (matching plot_events pattern). Bumped `_MAX_REVIEW_CHARS` constant from 2500 to 5000.

## Deduplicate maturity_summary logic
Files: movie_ingestion/metadata_generation/inputs.py
Why: `MovieInputData.maturity_summary()` duplicated the logic and MPAA_DEFINITIONS dict from `pre_consolidation.consolidate_maturity()`.
Approach: Removed the duplicated `_MPAA_DEFINITIONS` dict and method body from inputs.py. `maturity_summary()` now delegates to `consolidate_maturity()` via a lazy import (avoids circular import since pre_consolidation imports from inputs).

## Switch playground notebook to reception + remove _DEFAULT_KWARGS
Files: movie_ingestion/metadata_generation/metadata_generation_playground.ipynb, movie_ingestion/metadata_generation/generators/reception.py
Why: Needed to experiment with reception generation across providers. Discovered that `_DEFAULT_KWARGS = {"reasoning_effort": "low", "verbosity": "low"}` leaked OpenAI-specific params to Gemini/Groq providers, causing 400 errors.
Approach: Switched notebook import + all 7 runners from `generate_plot_events` to `generate_reception`. Added `run_gpt54_nano` runner. Updated movie summary cell to print reception-relevant fields (reception_summary, audience_reception_attributes, featured_reviews). Removed `_DEFAULT_KWARGS` from reception.py entirely — each caller now passes exactly the kwargs their provider needs, avoiding cross-provider leakage.

## Add wave1_runner for direct generation + storage of Wave 1 results
Files: movie_ingestion/metadata_generation/wave1_runner.py (new)
Why: Wave 2 evaluation needs pre-generated plot_events and reception outputs. The batch API pipeline is separate; this provides direct async generation with immediate SQLite storage.
Approach: New `wave1_results` table in tracker.db (tmdb_id PK, plot_events TEXT, reception TEXT — both nullable). Per-type `generate_and_store_*` functions with semaphore concurrency, UPSERT semantics, and per-movie error isolation. `get_wave1_results()` fetch helper deserializes JSON back to Pydantic models. Ran on EVALUATION_TEST_SET_TMDB_IDS: 68/70 succeeded (2 Gemini content-filter failures).
Design: Async tasks return result tuples (no DB access inside tasks); all DB writes happen in a single synchronous batch after `gather()` — follows the async I/O separation convention. SQLite connections use `_open_connection()` helper with WAL + FULL synchronous pragmas.

## Restructure playground notebook into candidates + per-generation-type cells
Files: movie_ingestion/metadata_generation/metadata_generation_playground.ipynb,
       movie_ingestion/metadata_generation/generators/plot_analysis.py,
       movie_ingestion/metadata_generation/generators/viewer_experience.py,
       movie_ingestion/metadata_generation/generators/watch_context.py,
       movie_ingestion/metadata_generation/generators/narrative_techniques.py,
       movie_ingestion/metadata_generation/generators/production_keywords.py,
       movie_ingestion/metadata_generation/generators/source_of_inspiration.py,
       movie_ingestion/metadata_generation/generators/reception.py

### Intent
Replace the reception-only notebook (8 `run_<model>` wrapper functions) with a structured multi-generation-type playground covering all 8 metadata types.

### Key Decisions
- Introduced `PlaygroundCandidate` dataclass (label, provider, model, kwargs) — lighter than `EvaluationCandidate` since generators handle system_prompt/response_format internally.
- 8 candidates with kwargs cross-referenced against `evaluations/plot_events.py` `PLOT_EVENTS_CANDIDATES` for accuracy.
- Shared `run_candidates()` helper eliminates duplication across 8 generation cells (loop, error handling, printing).
- `print_all_fields()` iterates `model_fields` to print every attribute including ones `__str__()` omits (e.g., `review_insights_brief`).
- Wave 2 cells pass `db_path=project_root / "ingestion_data" / "tracker.db"` to `get_wave1_results()` to avoid relative-path OperationalError in notebook CWD.
- Removed `_DEFAULT_KWARGS` and `effective_kwargs` from all generators except plot_events. These 6 generators now pass `**kwargs` directly (matching reception's existing pattern). Plot_events unchanged — retains `{**_DEFAULT_KWARGS, **kwargs}` merge.

### Testing Notes
- Notebook runs end-to-end for plot_events and reception cells
- Wave 2 cells require wave1_results table to be populated (via wave1_runner.py)
- Alibaba/Qwen candidate expected to fail (known API issue with JSON response_format)

## Make check_reception public, remove _check_plot_events alias
Files: movie_ingestion/metadata_generation/pre_consolidation.py, unit_tests/test_pre_consolidation.py, docs/modules/ingestion.md, docs/decisions/ADR-028-llm-evaluation-pipeline-design.md
Why: `_check_reception` was private but needed by wave1_runner. `_check_plot_events` was a backward-compat alias for `check_plot_events` that served no purpose.
Approach: Renamed `_check_reception` → `check_reception`. Removed `_check_plot_events = check_plot_events` alias. Updated all callsites (pre_consolidation internal, wave1_runner, test file). Removed `TestCheckPlotEventsPublicName` test class (tested alias identity, now tautological). Updated docs.

## Add system_prompt/response_format overrides to Wave 2 generators + notebook variant cells
Files: movie_ingestion/metadata_generation/generators/{plot_analysis,narrative_techniques,production_keywords,source_of_inspiration,viewer_experience,watch_context}.py, movie_ingestion/metadata_generation/metadata_generation_playground.ipynb
Why: Need to test justification vs non-justification prompt/schema variants on gpt-5.1-mini in the playground notebook.
Approach: Added `system_prompt` and `response_format` parameters to all 6 Wave 2 generator functions, defaulting to the non-justification versions in the parameter signature. Added 6 new notebook cells (one per generation type) that run both variants on gpt-5.1-mini.

## Switch plot_analysis from plot_keywords to merged_keywords
Files: movie_ingestion/metadata_generation/generators/plot_analysis.py, movie_ingestion/metadata_generation/prompts/plot_analysis.py
Why: plot_analysis was passing only `movie.plot_keywords` while other Wave 2 generators (viewer_experience, watch_context, production_keywords, source_of_inspiration) pass `movie.merged_keywords()` — the deduplicated union of plot + overall keywords. Overall keywords provide additional thematic signal useful for plot analysis.
Approach: Changed `build_plot_analysis_user_prompt` to pass `merged_keywords=movie.merged_keywords() or None` (was `plot_keywords=movie.plot_keywords or None`). Updated system prompt INPUTS section to describe `merged_keywords` matching the convention used by other generators.
