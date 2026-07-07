# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Created the /query_search observability plan (1c-1 planning, no code)
Files: observability_context/query_search_planning.md (new), observability_context/observability_todos.md, CLAUDE.md
Why: 1c-1 is too large for one session; the plan doc lets the work be taken in
small bites across sessions with no conversation memory.
Approach: mapped the live V2 streaming pipeline end to end (streaming_orchestrator
→ steps 0/1/2/3 → handler-LLM query generation → stage_4_execution → the three
Qdrant primitives in semantic_query_execution.py), then captured the agreed
per-phase span/attribute plan, locked cross-cutting decisions (gen_ai span once
in the LLM router, prompt-version hash, payload capture as sampled span events,
SSE-adapted outcome semantics: only validation rejection or Step-0 fatal failure
flips success; branch failures are counted degradations), 7 open questions, and
a 9-bite implementation checklist. Notable finding recorded in the doc: the
"metadata" channel is Postgres-fetch + in-memory scoring (auto-traced), so Qdrant
is the only backend blind spot; Stage 4's 25s dispatch timeout is a silent
soft-fail that must become a span event.
Supporting edits: observability_todos.md 1c-1 now points at the new doc and marks
its old V1-vocabulary sub-checklist historical; CLAUDE.md's observability pointer
lists the new file.
