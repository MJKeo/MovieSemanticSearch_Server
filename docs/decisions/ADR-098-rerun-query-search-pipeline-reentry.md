# [098] — `/rerun_query_search`: re-enter pipeline post-Steps-0/1 to apply new filters

## Status
Active

## Context
Re-running a query only to change hard filters (e.g. toggle streaming services
after seeing results) re-paid for Step 0 (flow routing) and Step 1 (spin
generation) — two Gemini calls — even though filters never enter query
understanding. The `fetches_ready` SSE event already carries all branch data
needed to replay the downstream machine from that point.

## Decision
Extract the shared execution kernel from `stream_full_pipeline` into
`_stream_from_branch_plan()` — everything after the six `to_*_flow_data()` calls
(build fetches → launch tasks → merge loop). A new public
`stream_rerun_pipeline(rerun_plan, metadata_filters)` accepts a `RerunPlan`
dataclass (branch_plan + six entity FlowData objects, constructed at the API
boundary from the client's re-submitted branch data) and delegates directly to
`_stream_from_branch_plan()`, bypassing Steps 0/1.

Standard branches re-enter at Step 2 (the branch query is the Step-2 input; Step 2
is not skipped because skipping would require passing the full `QueryAnalysis`
through the wire). Entity branches re-enter at their executor (post-Step-0).

API wire format: `RerunSearchBody` with a discriminated-union `branches` list
(type-tagged, one entry per flow type) + optional `filters`. Branch kinds are
assigned positionally (original / spin_1 / spin_2), not on the wire — BranchKind
is identity/label-only and not load-bearing in scoring.

## Alternatives Considered
- **Skip Step 2 for standard branches** (pass `QueryAnalysis` on the wire):
  rejected. `QueryAnalysis` is large and internal; the frontend would need to
  cache it opaquely per query. The `query` string is sufficient to re-run Step 2.
- **Cache the full pipeline output in Redis, re-run only Stage 4**: rejected.
  Stage 4 input (`GeneratedEndpointSpec` lists) is even larger and more internal.
  A filter change legitimately changes Stage 4 results (different candidate
  pool); caching the spec list would yield stale specs.
- **Add `new_filters` param to `/query_search`**: rejected. Would require the
  client to re-send the original query + all original branch data + the new
  filter — same wire volume as `/rerun_query_search` with no latency saving.

## Consequences
- `stream_full_pipeline` now delegates to `_stream_from_branch_plan` — the
  existing Step-0/1 path is refactored, not duplicated.
- `RerunPlan` is a frozen dataclass so the seven-field positional tuple it
  replaces can't silently misbind on field reorder.
- Standard branch queries are capped at `MAX_QUERY_CHARS + MAX_CLARIFICATION_CHARS`
  (the failed-clarification merge ceiling) so any branch the pipeline can emit
  round-trips cleanly; entity names capped at `MAX_QUERY_CHARS`.
- SSE event stream is byte-identical to `/query_search` — same event sequence,
  same shape per event.

## References
- `search_v2/streaming_orchestrator.py` (`stream_rerun_pipeline`,
  `_stream_from_branch_plan`, `RerunPlan`)
- `api/main.py` (`/rerun_query_search` endpoint, `_to_rerun_plan`, branch models)
- `search_v2/query_input_validation.py` (shared input caps)
- `docs/modules/api.md` (`/rerun_query_search` endpoint section)
- `docs/modules/search_v2.md` (`streaming_orchestrator.py` key file entry)
