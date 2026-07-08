# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Docs-audit remediation (docs-auditor findings 1–8)
Files: docs/modules/api.md, docs/modules/search_v2.md, CLAUDE.md,
docs/decisions/ADR-012-*.md, docs/conventions.md, docs/PROJECT.md,
search_v2/endpoint_fetching/entity_query_generation.py; deleted
search_v2/reranking/ (8 files)
Why: A full docs-auditor staleness scan surfaced 8 doc/code drift items;
user directed applying all of them.
Approach:
- api.md: documented the seventh hard filter `keywords` (was omitted from
  the six-filter enumeration) on /query_search + /attribute_search;
  added the third CORS origin `http://localhost:3001`.
- CLAUDE.md: corrected Stage 3 endpoint count 7 → 10 (verified by
  `ls search_v2/endpoint_fetching/*_execution.py`).
- search_v2.md: added a "Deterministic Endpoint Backers" subsection +
  Key Files rows for attribute_search.py, title_search.py,
  query_input_validation.py (three live endpoint/validation modules that
  were entirely absent).
- Deleted search_v2/reranking/ — dead, broken code (every file imported
  the deleted search_v2.stage_4 package; `import search_v2.reranking`
  raised ModuleNotFoundError). No live importer (confirmed via repo-wide
  grep). Updated the now-stale stub rationale in entity_query_generation.py
  (its only former importer was reranking/dispatch.py).
- ADR-012: re-statused Proposed → Superseded + appended a 2026-07
  postscript; its roadmap diverged (Batch API shipped under
  ADR-025/036/041/044; GPT-4o-mini swap and 8→5 consolidation NOT
  adopted — pipeline grew to 12 generation types, models are
  gpt-5-mini/gpt-5.4-mini per ADR-039/043/044). ADR body preserved
  verbatim (append-only).
- conventions.md: documented the codebase-wide `extra="forbid"` default
  on wire-boundary/LLM-output Pydantic models (per ADR-102).
- PROJECT.md: added a "Secondary criterion: skill transferability" note
  acknowledging the resume-credibility factor that explicitly shaped
  ADR-101, scoped as a tooling-only tiebreaker below the four product
  priorities.
Design context: docs-awareness rule normally reserves decisions/,
conventions.md, PROJECT.md for the dedicated skill workflows; these three
were edited under explicit user direction, not autonomously.
Testing notes: docs/comment-only except the reranking/ deletion — verified
no remaining `search_v2.reranking` imports anywhere in the tree.
