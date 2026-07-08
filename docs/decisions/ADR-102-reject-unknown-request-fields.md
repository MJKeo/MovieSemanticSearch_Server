# [102] — Reject unknown request fields; record framework-422s as outcomes

## Status
Active (drafted by docs-maintainer from DIFF_CONTEXT — pending human review)

## Context
Pydantic's default behavior silently ignores extra fields on a request
body. An unknown or typo'd filter key (e.g. `genrez` instead of `genres`)
was therefore dropped before it ever reached `MetadataFiltersInput` —
the request looked well-formed and simply behaved as "no filter on that
axis." This is a silent client/server contract drift: the caller believes
a filter is applied; the server silently ignores it. It surfaced while
building out `/query_search` observability (`observability_context/
query_search_planning.md`), which also exposed a second gap — FastAPI's
own request-validation 422s (malformed JSON, wrong types, and now the new
unknown-field 422s) never passed through `@record_outcome`, so they were
invisible in traces (flagged as an open gap in `observability_architecture.md`
§8).

## Decision
- Set `model_config = ConfigDict(extra="forbid")` on `QuerySearchBody` and
  the shared `MetadataFiltersInput` (the latter is reused by
  `/similarity_search` and `/attribute_search`, so the fix applies there
  too). An unknown field now 422s at the boundary instead of being
  silently dropped.
- Added an app-level `@app.exception_handler(RequestValidationError)`
  (`_on_request_validation_error` in `api/main.py`) that stamps
  `request.success=false` + `FailureReason.INVALID_PARAMETERS` and a
  `request rejected` span event naming the offending field(s)/error
  type(s) (`loc` + `msg` + `type` only — never the input value, so no PII
  or oversized payloads land on the span; capped at 5 errors), then
  delegates to FastAPI's default `request_validation_exception_handler`
  so the HTTP response body is byte-for-byte unchanged. The handler is
  global, so every endpoint's malformed-body 422s now carry a verdict,
  not just the ones that already used `@record_outcome`.
- Reused the existing `FailureReason.INVALID_PARAMETERS` for framework
  validation failures (framework body/param validation is the same
  actionable class as existing "bad/missing request params" failures) —
  no new enum member for this path. A distinct `INVALID_FILTERS` member
  already existed for the narrower "unknown filter enum value" case and
  was left as-is.

## Alternatives Considered
- **Leave `extra="ignore"` (the Pydantic default).** Rejected: this is
  the silent-drift behavior the decision fixes. Cheapest option but
  directly contradicts the "correctness over convenience" priority
  ordering in `docs/PROJECT.md` (search quality depends on filters
  actually being applied when the caller believes they are).
- **Per-model custom validators that raise `EndpointFailure` directly.**
  Rejected in favor of a single app-level exception handler: FastAPI's
  own request parsing rejects the body *before* any endpoint handler or
  per-field validator runs, so a per-model approach can't intercept it —
  the rejection happens at the framework layer, so the fix has to live
  there too.
- **Silently drop and log server-side only.** Rejected: doesn't fix the
  caller-facing behavior (the filter is still silently ignored from the
  client's point of view) and the observability motivation is a symptom
  of the same underlying looseness, not a separate problem to solve
  independently.

## Consequences
- Any future field added to `QuerySearchBody` or `MetadataFiltersInput`
  must be added to the actual model — a client can no longer send extra
  fields without a 422, which is a stricter (breaking) contract for any
  caller currently relying on lenient parsing. No known callers depended
  on this, per the diff-context testing notes.
- Every endpoint (not just the four with `@record_outcome`) now records
  a `request.success=false` / `invalid_parameters` verdict on a
  framework-level 422, closing the observability blind spot noted in
  `observability_architecture.md` §8.
- Establishes the pattern for future request models: default to
  `extra="forbid"` rather than opting in per-model as gaps are
  discovered.

## References
- `observability_context/observability_architecture.md` §5.2 (single-write
  outcome model), §8 (the now-closed framework-422 gap)
- `observability_context/query_search_planning.md` (the /query_search
  instrumentation plan this was discovered while building)
- `docs/conventions.md` — "Observability Conventions" (per-request
  outcome model this decision extends)
- `docs/modules/api.md` — `/query_search` Telemetry note, Observability
  section (outcome verdict + `FailureReason` enum)
- `docs/PROJECT.md` — priority #1 (search quality / correctness): a
  silently-ignored filter is a correctness bug wearing an API-contract
  disguise
