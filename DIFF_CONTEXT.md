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
