# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Documentation staleness fixes from full audit

Files: CLAUDE.md, docs/PROJECT.md, docs/conventions.md, docs/modules/classes.md, docs/modules/ingestion.md, docs/modules/api.md, docs/modules/db.md, docs/decisions/ADR-060-basemovie-to-movie-migration.md

Why: docs-auditor subagent identified 11 stale claims across permanent docs.

Fixes applied:
- Status chain updated from removed `phase1_complete`/`phase2_complete` to `metadata_generated` + `ingestion_failed` (conventions.md, CLAUDE.md)
- Stage 7 embedding corrected: it's integrated into Stage 8 inside `ingest_movie.py`, not a separate unimplemented step (CLAUDE.md, PROJECT.md, ingestion.md)
- `implementation/vectorize.py` correctly identified as legacy ChromaDB (CLAUDE.md, PROJECT.md, ingestion.md)
- Wrong filename `plot_quality_scores.py` → `plot_tmdb_quality_scores.py` (CLAUDE.md)
- BaseMovie no longer described as used in ingestion or db/ (classes.md, ADR-060)
- Dangling ADR-027 reference removed (ingestion.md)
- `cli_search.py` added to api.md key files table
- `batch_upsert_*_dictionary()` wording clarified to exclude `batch_upsert_lexical_dictionary` (db.md)
