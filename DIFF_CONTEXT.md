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

## Search system analysis and improvement planning

Files: search_improvement_planning/current_search_flaws.md, search_improvement_planning/types_of_searches.md, search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/open_questions.md

### Intent
Deep analysis of why the current search pipeline fails on multi-constraint queries
(e.g., "iconic twist ending" returns mid-tier thrillers instead of Fight Club, The
Sixth Sense, etc.) and planning for a redesigned search architecture.

### Key Findings
- Compared generated metadata for Fight Club, The Sixth Sense, Wild Things, and A
  Perfect Getaway — metadata quality is comparable across all four, ruling out
  metadata as the cause
- Root cause is architectural: additive scoring at every layer (vector scoring,
  channel merging) creates disjunctive results, rewarding movies that excel at one
  attribute over movies that satisfy multiple attributes simultaneously
- Secondary cause: embedding density effect — movies whose identity revolves around
  a single attribute (Wild Things = twists) have higher cosine similarity for that
  attribute than movies with richer, more distributed embeddings (Fight Club)

### Planning Decisions
- Proposed deal-breaker / preference / implicit hierarchy to replace flat additive
  scoring — deal-breakers gate the candidate set (conjunctive), preferences rank
  within it (additive)
- Proposed threshold + flatten approach for semantic deal-breakers: once a candidate
  passes retrieval threshold, its deal-breaker score is flattened to 1.0 to prevent
  embedding density bias
- Proposed 4-phase pipeline: Phase 0 (query understanding) → Phase 1 (deal-breaker
  retrieval) → Phase 2 (preference scoring) → Phase 3 (result assembly) → Phase 4
  (exploratory extension)
- Identified 6 query type categories with distinct retrieval needs
- Multiple open questions documented around threshold selection, LLM classification
  reliability, graceful degradation strategy

### Testing Notes
- Theories to validate: run "iconic twist ending" through notebook to inspect actual
  subqueries/weights/per-space scores; simulate threshold+flatten to test if
  reranking by reception would surface expected movies
