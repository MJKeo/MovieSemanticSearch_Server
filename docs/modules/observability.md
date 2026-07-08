# observability/ — OpenTelemetry Bootstrap & Naming Registry

## What This Module Does

Central home for the API's OpenTelemetry (OTel) tracing setup. `tracing.py`
builds and wires the OTel SDK (TracerProvider, OTLP exporter, auto-
instrumentation) into the FastAPI app; `names.py` is the single source of
truth for every manual span name and attribute key emitted by hand-written
instrumentation elsewhere in the codebase. Both are consumed by
`api/main.py` (and `api/outcome.py`, which builds on `names.py`).

## Key Files

| File | Purpose |
|------|---------|
| `tracing.py` | `setup_tracing(app)` — builds a `TracerProvider` (`service.name` / `deployment.environment` resource attrs from env), a `BatchSpanProcessor` + OTLP **gRPC** exporter (endpoint from `OTEL_EXPORTER_OTLP_ENDPOINT`), and enables auto-instrumentation for FastAPI, httpx, **psycopg v3**, and redis. Idempotent via a module-level `_configured` guard so `--reload` / repeated imports don't double-instrument. Calls `load_dotenv()` (non-overriding) so a bare host `uvicorn` run resolves `OTEL_*` vars from `.env`. |
| `names.py` | `Name(str)` — a telemetry name that is also a namespace; `.child()` derives deeper names so a namespace root is written once and never retyped. Every manual span name / attribute key used in `api/main.py` and `api/outcome.py` is declared here as a module-level constant. Deliberately stdlib-only (no `opentelemetry` import) so any module can reference a name without pulling in the SDK. The module docstring is the source-of-truth naming ruleset (structure, root ownership, dot-vs-underscore, leaf conventions, values-vs-names, cardinality) — see it before adding a new name. |
| `cost_tracking.py` | Request-scoped LLM + embedding cost **and token** accumulation. A `ContextVar`-held `RequestCostAccumulator` entered once per request via `track_request_cost()` (before the pipeline spawns tasks, so every branch inherits it by reference); `add_request_cost(cost_usd)` **and** `add_request_tokens(input, cached_input, output)` are called by each LLM provider (per billed attempt) and by `generate_vector_embedding`, and are no-ops outside a tracked request. The `/query_search` handler reads the totals at stream end and writes them on the server span as `query_search.cost_usd` + `query_search.usage.{input,cached_input,output}_tokens`. Cost gates on the pricing table (unpriced models add 0); tokens accumulate unconditionally, so the two can diverge. `cached_input` is a subset of `input`. Stdlib-only, mirroring `names.py`. |

## Boundaries

- **In scope**: OTel SDK bootstrap/config, auto-instrumentation
  enablement, the manual span/attribute name registry.
- **Out of scope**: the instrumentation call sites themselves (span
  creation, attribute setting) — those live in the consuming modules
  (`api/main.py`, `api/outcome.py`), not here. This package is a
  "plumbing" layer other modules import from; it never imports app code.
- `api/outcome.py` (the per-request outcome mechanism — `FailureReason`,
  `EndpointFailure`, `@record_outcome`) deliberately sits under `api/`,
  not here: it imports FastAPI + the OTel SDK, whereas `names.py` stays
  dependency-free so a non-API module (e.g. a future `search_v2`/`db`
  span) could reference a name without pulling in the SDK.

## Internal Patterns

- **Idempotent bootstrap.** `setup_tracing` guards itself with a
  module-level flag — safe to call more than once (uvicorn `--reload`,
  repeated test-client imports); re-instrumenting would otherwise raise
  (FastAPI) or double-wrap (httpx/psycopg/redis).
- **Config-not-code.** Every environment-specific value (service name,
  deployment environment, OTLP endpoint) is read from env vars, never
  hardcoded — local → Grafana Cloud is a `.env` change, not a code
  change.
- **`Name` is-a `str`.** `Name` subclasses `str`, so a constant hands
  straight to `start_as_current_span(...)` / `set_attribute(...)` with
  no unwrapping, while `.child(segment)` derives a deeper name from any
  existing one (root or leaf) so names can grow subtokens later without
  migrating off a raw string.

## Interactions

- `api/main.py` calls `setup_tracing(app)` immediately after constructing
  the `FastAPI` app (before it serves traffic, so every request is a root
  span), and imports span/attribute constants from `names.py` for all its
  manual instrumentation (see `docs/modules/api.md`'s Observability
  section for the current catalog).
- `api/outcome.py` imports `OUTCOME_SUCCESS` / `OUTCOME_FAILURE_REASON`
  from `names.py`.

## Gotchas

- **The auto-instrumented library set is fixed to what this stack
  actually uses**: psycopg **v3**, not asyncpg (the DB layer uses
  `psycopg_pool.AsyncConnectionPool`) and not a generic ORM instrumentor.
  Adding a new outbound client (a different HTTP library, a new DB
  driver) needs its own instrumentor added in `tracing.py`.
- **Qdrant is deliberately NOT auto-instrumented.** Its async client
  speaks gRPC, which the available auto-instrumentors don't wrap
  cleanly. Qdrant timing is expected to come from a manual vector-search
  span once `/similarity_search` / `/query_search` are instrumented
  (`observability_context/observability_todos.md` Phase 1c-1/1c-3) — not
  from this module.
- **The docker-compose `api` service is not tracing-ready.**
  `observability/` is not volume-mounted into the container and the OTel
  packages are not in `api/requirements.txt`, so importing this package
  in the container crashes (`ModuleNotFoundError: observability`). The
  only instrumented path today is host-run (`uv run uvicorn
  api.main:app --reload`). Making the container tracing-ready (deps +
  mount + endpoint pointed at the host network) is deferred — see
  `observability_context/observability_todos.md` Phase 5.
- **Backend is local-only today.** `OTEL_EXPORTER_OTLP_ENDPOINT` points
  at a local `grafana/otel-lgtm` container (run standalone, outside the
  prod compose file); there is no production trace export configured
  yet.

## References

- `observability_context/observability_architecture.md` — full as-built
  catalog (bootstrap, conventions, per-endpoint span/attribute list);
  this doc is a summary, that is the source of truth and should be kept
  ahead of this one.
- `observability_context/initial_implementation_context.md` — locked
  tooling decisions (OTel + Grafana stack, alternatives rejected).
- `observability_context/observability_todos.md` — phased rollout plan
  and current status.
- `docs/modules/api.md` — per-endpoint telemetry that consumes this
  package.
- `docs/decisions/ADR-101-otel-grafana-observability-stack.md` — the
  OTel/Grafana stack selection.
- `docs/conventions.md` — "Observability Conventions": the manual
  instrumentation conventions (naming registry + per-request outcome +
  span/error contract). Applies codebase-wide, so it lives in
  conventions rather than a module doc.
