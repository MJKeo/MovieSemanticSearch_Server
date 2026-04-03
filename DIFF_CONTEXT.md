# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Docs audit: fix stale documentation across 10 files

Files: CLAUDE.md, docs/conventions.md, docs/modules/ingestion.md, docs/modules/llms.md, docs/modules/classes.md, docs/modules/README.md, docs/decisions/ADR-012-llm-generation-cost-optimization.md, implementation/classes/enums.py, movie_ingestion/tracker.py

Why: Full docs-auditor scan found 16 issues — factual errors in CLAUDE.md (wrong IMDB data path, wrong Stage 3/5 descriptions, wrong signal count, wrong filename), stale conventions (missing status step, nonexistent eval.db), broken ADR cross-references (10 deleted ADRs still cited), and minor inaccuracies in module docs.

Changes:
- CLAUDE.md: Fixed IMDB output path (JSON files → imdb_data table), Stage 5 hard filters (old list → 2 gates), Stage 3 description (removed phantom hard filters), tmdb_quality_scorer signal count (10 → 4), batch_manager.py → openai_batch_manager.py
- conventions.md: Added tmdb_quality_calculated/imdb_quality_calculated to status progression; rewrote eval storage section (eval.db → per-movie JSON files)
- ingestion.md: Fixed generator lock claim (all 8 now locked), build_custom_id arg order, parse_custom_id return type, replaced analyze_evaluations.py entry with errors.py, removed references to deleted ADRs (015, 019, 028, 031, 034), removed deleted run_evaluations_pipeline.py and eval.db sections
- llms.md: Removed references to deleted ADRs (024, 029, 030)
- classes.md: Added note that ProductionMetadata is legacy and not aligned with generation-side split schemas
- enums.py: VectorCollectionName docstring ChromaDB → Qdrant
- tracker.py: Added tmdb_quality_calculated to docstring and SQL comment progressions
- README.md: Relaxed 60-line guideline to allow complex modules to exceed
- ADR-012: Moved watch_context from Wave 1 to Wave 2 to match code

## Move ingest_movie.py from db/ to movie_ingestion/final_ingestion/
Files: db/ingest_movie.py → movie_ingestion/final_ingestion/ingest_movie.py, 5 notebooks, unit_tests/test_ingest_movie.py, docs/modules/db.md, docs/modules/ingestion.md, CLAUDE.md
Why: Consolidate all ingestion pipeline logic under movie_ingestion/. The new final_ingestion/ subpackage will house all logic for taking movie data from the SQLite tracker and upserting into Postgres/Qdrant.
Approach: Moved the file, created __init__.py, updated all import paths (5 notebooks, 1 test file), updated db.md/ingestion.md module docs and CLAUDE.md architecture docs.

## Add top-level schemas/ package and refactor cross-cutting types out of metadata_generation
Files: schemas/__init__.py, schemas/metadata.py, schemas/enums.py, schemas/data_types.py, schemas/movie_input.py (all new), movie_ingestion/metadata_generation/schemas.py (deleted), movie_ingestion/metadata_generation/inputs.py (trimmed), all 8 generators, all 4 batch_generation modules, CLAUDE.md

Why: Need a home for shared Pydantic models/data classes importable by db/, api/, and movie_ingestion/. The implementation/ folder is being phased out.

Moved to schemas/:
- All Output schema classes (PlotEventsOutput, ReceptionOutput, etc.) → schemas/metadata.py
- MetadataType enum → schemas/enums.py
- MultiLineList → schemas/data_types.py
- MovieInputData + load_movie_input_data → schemas/movie_input.py

Kept in inputs.py (generation-pipeline-specific):
- build_custom_id (now overloaded: accepts int or MovieInputData), parse_custom_id
- WAVE1_TYPES, WAVE2_TYPES, ALL_GENERATION_TYPES
- ConsolidatedInputs, SkipAssessment, build_user_prompt
- Wave1Outputs, load_wave1_outputs, load_plot_analysis_output

Removed: MovieInputData.batch_id() method — callers use build_custom_id() directly.
All 12 consumer files updated to import from new canonical locations (no re-export shims).

## Extract vector text generation into movie_ingestion/final_ingestion/vector_text.py
Files: movie_ingestion/final_ingestion/vector_text.py (new), movie_ingestion/final_ingestion/ingest_movie.py, CLAUDE.md, docs/modules/ingestion.md, docs/llm_metadata_generation_report.md
Why: Vector text generation functions (8 total, one per vector space) were in implementation/vectorize.py alongside legacy ChromaDB code. Extracting them into final_ingestion/ co-locates them with the ingestion pipeline that consumes them.
Approach: Copied all 8 create_*_vector_text functions and the budget_size_to_vector_text helper verbatim into the new file. Updated ingest_movie.py import to point to the new location (linter normalized to relative import `from .vector_text import ...`). Updated all docs that referenced the old path: ingestion.md, llm_metadata_generation_report.md, CLAUDE.md, PROJECT.md, and TODO.md (6 references).

## Update test imports for schemas package split
Files: unit_tests/test_pre_consolidation.py, unit_tests/test_source_of_inspiration_generator.py, unit_tests/test_viewer_experience_generator.py, unit_tests/test_metadata_schemas.py, unit_tests/test_plot_analysis_generator.py, unit_tests/test_plot_events_generator.py, unit_tests/test_production_keywords_generator.py, unit_tests/test_narrative_techniques_generator.py, unit_tests/test_watch_context_generator.py, unit_tests/test_reception_generator.py, unit_tests/test_generator_registry.py
Why: The generation-side schema models moved from `movie_ingestion.metadata_generation.schemas` to the top-level `schemas.metadata` module, and a set of tests were still importing the deleted module path.
Approach: Repointed test imports to `schemas.metadata` without changing test intent. Re-ran collection in the `uv` environment to verify the pure import-path drift was fixed; those updated files now collect successfully.
Testing notes: `unit_tests/test_prompt_constants.py` still fails collection because `SYSTEM_PROMPT_WITH_REASONING` is missing from `movie_ingestion.metadata_generation.prompts.source_of_inspiration`, which appears to be a real API/constant change rather than a path rename.
