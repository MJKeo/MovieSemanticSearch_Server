# Observability Architecture (as-built)

**Purpose.** A factual record of the observability system *as actually
implemented* in the codebase — the source of truth for what exists today. Use it
to (a) understand the current tracing setup without re-deriving it from code,
and (b) drive updates to the permanent project docs (see
[§9](#9-permanent-docs-that-should-be-updated)), which do not yet describe any
of this.

This is a companion to the planning docs in this folder — `observability_logs_plan.md`
(the *why*), `initial_implementation_context.md` (locked decisions + standing
guidelines), and `observability_todos.md` (the ordered checklist + per-item
status). Where those describe intent, **this describes what shipped**.

**Last updated:** 2026-07-03 · **Phase:** 1 (traces) — partially complete.
Keep this doc in sync whenever manual instrumentation is added or the bootstrap
changes.

---

## 1. Status at a glance

| Area | State |
|------|-------|
| OTel SDK bootstrap + OTLP export | ✅ implemented (`observability/tracing.py`) |
| Auto-instrumentation (FastAPI, httpx, psycopg v3, redis) | ✅ implemented |
| Local trace backend (`grafana/otel-lgtm`) | ✅ running locally (dev only) |
| Manual spans/attrs: `/title_search` | ✅ implemented (1c-5) |
| Manual spans/attrs: `/movie_details`, `/movie_credits` | ✅ implemented (1c-6, 1c-7) |
| Manual spans: `/query_search`, `/rerun_query_search`, `/similarity_search`, `/attribute_search` | ❌ not started (1c-1…1c-4) |
| LLM `gen_ai.*` attributes | ❌ not started (folded into 1c-1) |
| Metrics (RED/USE) | ❌ not started (Phase 3) |
| Structured logs + trace correlation | ❌ not started (Phase 4) |
| Production export (Grafana Cloud on EC2) | ❌ not started (Phase 5) |

**One-line summary:** every inbound request already produces a trace with
auto-instrumented network spans; three of the eight endpoints additionally carry
hand-written pipeline spans + semantic attributes. Nothing but traces exists yet
(no metrics, no log correlation), and telemetry is local-only.

---

## 2. Bootstrap & configuration

**Module:** `observability/tracing.py` — `setup_tracing(app)`.

- Builds a `TracerProvider` with a `Resource` carrying `service.name` (from
  `OTEL_SERVICE_NAME`) and `deployment.environment` (from `DEPLOY_ENV`).
- Exports via `BatchSpanProcessor` + OTLP **gRPC** exporter. The exporter reads
  its endpoint from the environment (`OTEL_EXPORTER_OTLP_ENDPOINT`), so
  local↔prod is a config change, not a code change.
- **Idempotent** via a module-level `_configured` guard, so `uvicorn --reload`
  re-imports and test clients don't double-instrument (which raises for FastAPI
  / double-wraps the clients).
- Calls `load_dotenv()` (non-overriding) so a bare host `uvicorn` run resolves
  the OTEL_* vars from `.env`.

**Wiring:** `api/main.py` calls `setup_tracing(app)` at import time, right after
`app = FastAPI(...)` is constructed — so instrumentation is live before the app
serves traffic and every request is a root span.

**Environment variables** (set in `.env`):

| Var | Role | Observed value |
|-----|------|----------------|
| `OTEL_SERVICE_NAME` | `service.name` resource attr | `cinemind-api` |
| `DEPLOY_ENV` | `deployment.environment` resource attr | (env-specific) |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | OTLP gRPC target | local otel-lgtm `:4317` |

**Smoke test:** `scripts/otel_smoke_test.py` emits a hand-made parent+child span
straight to the OTLP endpoint — used to confirm the backend is reachable before
trusting app traces.

---

## 3. Auto-instrumentation

Enabled process-wide in `setup_tracing`. Covers every network hop this service
makes, for free, on **all** endpoints:

| Instrumentor | Covers | Span shape |
|--------------|--------|-----------|
| FastAPI | inbound HTTP | one **root/server span** per request (`GET /movie_details/{tmdb_id}`, …) with method, route, status code, duration |
| httpx | outbound TMDB / LLM / proxy calls | child span per outbound request |
| psycopg v3 | Postgres | child span per query (carries the SQL) |
| redis | cache GET/SET | child span per command |

**Deliberate gap — Qdrant.** Its async client speaks gRPC, which the
auto-instrumentation doesn't wrap cleanly. Qdrant timing will come from a manual
vector-search span when `/similarity_search` / `/query_search` are instrumented
(1c-1, 1c-3). Not a bug — a known, documented omission.

**What auto-instrumentation does NOT give you** (the reason manual spans exist):
semantic facts no client library can see — cache hit vs. miss, why a 404
happened, cache-write success, result counts, LLM tokens/cost.

---

## 4. Backends

- **Local (dev/test only):** the all-in-one `grafana/otel-lgtm` container —
  OTel Collector + Tempo (traces) + Prometheus + Loki + Grafana. Grafana UI on
  `:3000` (admin/admin); OTLP on `:4317` (gRPC) / `:4318` (HTTP). View traces via
  the Tempo datasource in Grafana. Run **outside** the prod compose.
- **Production (EC2):** Grafana Cloud free tier — **not wired yet** (Phase 5). It
  becomes a change to `OTEL_EXPORTER_OTLP_ENDPOINT` + credentials, no app code.

**Operational caveat (known).** The instrumented API is currently run on the
**host** (`uv run uvicorn api.main:app --reload`). The docker-compose `api`
service is **not** tracing-ready: `observability/` isn't volume-mounted and the
OTel packages aren't in `api/requirements.txt`, so the container crashes on
import (`ModuleNotFoundError: observability`). Making the container tracing-ready
is deferred to Phase 5 (add deps, mount the package, point the endpoint at the
host/otel-lgtm network).

---

## 5. Conventions & naming

These are the patterns the first manual instrumentation set; new endpoints follow
them (candidates for `docs/conventions.md` — see §9). The naming rules (§5.1) are
codified in `observability/names.py`'s module docstring, which is the source of
truth; the below is the summary.

### 5.1 Naming rules

Manual span names and attribute keys are defined **once** as `Name` constants in
`observability/names.py` (never inline literals) and derived from a namespace
root via `.child()`, so a root token is written once and a typo can't silently
split a metric. `Name` subclasses `str`, so a constant hands straight to
`start_as_current_span(...)` / `set_attribute(...)` with no unwrapping.

- **A. Structure.** Every name is `namespace.leaf` — at least one namespace,
  never a bare key (attribute keys share one flat global keyspace across every
  span, metric, and log). Two levels is the default depth.
- **B. Root.** The domain/endpoint that conceptually owns the thing (`movie`,
  `movie_details`, `movie_credits`, `cache`, `title_search`). For a span, its
  home endpoint even when it runs under another — `movie_credits.build_and_cache`
  keeps that name under a `/movie_details` request (runtime parent ≠ name). Never
  author or reuse a standard OTel root (`http`/`db`/`net`/`gen_ai`/…).
- **C. Dot vs. underscore.** Add a namespace level only when it already groups ≥2
  emitted attributes, or has ≥2 concrete siblings you'll *actually emit* on the
  roadmap; else fold into the leaf with `_`. "Expansion" = telemetry we'll emit,
  not properties the real object happens to have. Tie → stay flat (promoting
  later is a one-line edit).

  | Name | Decision | Why |
  |------|----------|-----|
  | `cache.write_ok` | dot | `cache` is a growing telemetry space (hit, read_ok, ttl_seconds are nameable siblings) |
  | `outcome.failure_reason` | dot | `outcome` already groups ≥2 emitted attributes (`success` + `failure_reason`) |
  | `movie.payload_source` | underscore | one measured fact (provenance); we'll never emit `payload.size`/`.bytes` |
  | `gen_ai.usage.input_tokens` (future) | dot | `usage` groups input/output/total (illustrative — a standard key, not ours) |

- **D. Leaf.** snake_case; counts end `_count`, booleans read as a predicate
  (`_ok`), measures carry a unit suffix (`_seconds`/`_ms`/`_chars`); never bake a
  varying/high-cardinality value into the key.
- **E. Values are not names.** A closed value set is a `str`-Enum in its owning
  module (`MoviePayloadSource` in `api/main.py`, `FailureReason` in
  `api/outcome.py`), set via `.value` — not a `Name`.
- **F. Cardinality.** High-cardinality values (`movie.tmdb_id`, query text) are
  fine as **span attributes** but must **never** become metric labels (Phase 3);
  only low-cardinality keys (`movie.payload_source`, `outcome.success`,
  `outcome.failure_reason`) are label-eligible.

### 5.2 Instrumentation behavior

- **Per-module tracer.** `tracer = trace.get_tracer(__name__)` at module level in
  `api/main.py`. (First app code to create manual child spans; previously only
  `scripts/otel_smoke_test.py`.)
- **Request-scoped facts go on the server span, not child spans.** Capture it
  once at the top of the handler — `request_span = trace.get_current_span()` —
  because inside a `start_as_current_span(...)` block `get_current_span()`
  returns the *child*. Never mutate the auto-instrumentation child spans.
- **Child spans only for genuine sub-units of work.** A span is worth a bar in
  the waterfall only if it takes meaningful time or can fail independently.
  `/title_search` is one unit → attributes only, no child span.
- **Per-request outcome, recorded once (`api/outcome.py`).** Every instrumented
  endpoint carries `outcome.success` (bool) on its server span, plus
  `outcome.failure_reason` (`FailureReason`) whenever `success` is false. It is
  written in exactly ONE place — the `@record_outcome` decorator — not at each
  exit site: a known failure site raises `EndpointFailure(failure_reason=…)` and
  the reason *bubbles up* to the decorator, which records it and re-raises;
  reaching the end of the handler with nothing raised is the `success=true`
  verdict. "Failure" is broader than "error" — a 404 for an un-indexed id is a
  failure (`not_indexed`) but not a span error. `FailureReason` is kept coarse
  (`invalid_parameters` / `not_indexed` / `tmdb_removed` / `tmdb_fetch_failed` /
  `internal_error`) — one member per actionable class; the specifics (which
  param, which id, the stack) live on other span attributes and the recorded
  exception. (Framework-level 422s — a missing/mis-typed query param rejected by
  FastAPI *before* the handler runs — are not yet covered; see §8.)
- **Error contract on manual spans** (distinct from the outcome above — this is
  the *span status*, that is the *request verdict*):
  - Expected **404** → span left **UNSET** (not an error); the request verdict is
    `outcome.success=false` + `not_indexed`/`tmdb_removed`.
  - **502** (upstream TMDB failure) → span marked **ERROR** + `record_exception`;
    verdict `tmdb_fetch_failed`.
  - **Unexpected** exception → also marked **ERROR** + recorded, so the span that
    contains the failing op never reads green; verdict `internal_error`.
  - Swallowed best-effort failures (cache read/write, cross-populate) → span
    **events**, never span errors, and **not** request failures — they're
    degradations, so `outcome.success` stays true.
- **`source` is set only at a success point**, never optimistically → a 404/502
  carries no `movie.payload_source` (absent = no payload served), keeping
  cache-hit-rate honest.

---

## 6. Span & attribute catalog (implemented endpoints)

**Universal (every endpoint below).** On the request/server span, via
`@record_outcome` (§5.2):

| Attribute | Type | Meaning |
|-----------|------|---------|
| `outcome.success` | bool | request verdict; set on every path |
| `outcome.failure_reason` | `FailureReason` | set only when `success` is false: `invalid_parameters`\|`not_indexed`\|`tmdb_removed`\|`tmdb_fetch_failed`\|`internal_error` |

### `/title_search` (GET) — 1c-5
No child span (single unit of work). Attributes on the **request span**:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `title_search.query` | string | raw query text (attribute-only; never a metric label) |
| `title_search.limit` | int | requested limit |
| `title_search.result_count` | int | hydrated cards returned (incl. 0) |
| `title_search.fuzzy_result_count` | int | how many results came from the fuzzy fallback tier (>0 = likely typo / catalog gap) |

Supporting change: `run_title_search` returns a `TitleSearchResult` NamedTuple
`(movie_ids, fuzzy_count)` (was `list[int]`) so the endpoint can report the fuzzy
count without re-deriving discarded tier data (`search_v2/title_search.py`).

### `/movie_details` (GET) — 1c-6 · `/movie_credits` (GET) — 1c-7
Share one fetch helper and one credits build helper, so their spans/attributes
are consistent.

**Child spans (cold path only):**

| Span | Where | Notes |
|------|-------|-------|
| `movie_details.payload_creation` / `movie_credits.payload_creation` | `_fetch_movie_payload` | wraps card existence-gate + TMDB fetch; auto psycopg + httpx spans nest inside |
| `movie_details.cache_write` | `movie_details` handler | wraps the details cache SET; carries `cache.write_ok` |
| `movie_credits.build_and_cache` | `_encode_and_cache_credits` | build + encode + credits cache SET. Appears under **both** endpoints — as the main path in `/movie_credits` and as the cross-populate step nested under `/movie_details` |

Warm path (cache hit) = request span + one auto redis GET, no child spans.

**Attributes (shared keys, both endpoints):**

| Attribute | On | Type | Meaning |
|-----------|----|------|---------|
| `movie.tmdb_id` | request span | int | set unconditionally; queryable on every path |
| `movie.payload_source` | request span | `cache`\|`tmdb` (`MoviePayloadSource`) | origin of the served payload; set only on success (absent on 404/502) |
| `cache.write_ok` | `cache_write` / `build_and_cache` span | bool | best-effort cache write outcome |
| `movie_credits.cast_count` | `movie_credits.build_and_cache` span | int | billed cast size |
| `movie_credits.crew_count` | `movie_credits.build_and_cache` span | int | total crew members (summed across department groups) |

**Span events:** `cache read failed`, `details cache write failed`,
`credits cache write failed`, `credits cross-populate failed` — mark swallowed
best-effort degradations without failing the request.

---

## 7. Shared building blocks

| Symbol | Where | Kind | Role |
|--------|-------|------|------|
| `tracer` | `api/main.py` | module tracer | `trace.get_tracer(__name__)` |
| `MoviePayloadSource(str, Enum)` | `api/main.py` | enum | `CACHE`/`TMDB` values for `movie.payload_source` |
| `FailureReason(str, Enum)` | `api/outcome.py` | enum | the coarse `outcome.failure_reason` vocabulary (replaces the old `NotFoundReason`, folding its two 404 reasons into the broader set) |
| `EndpointFailure(HTTPException)` | `api/outcome.py` | exception | request-terminating failure that carries its `FailureReason`; raised at each known failure site so the reason bubbles up |
| `record_outcome` | `api/outcome.py` | decorator | wraps each endpoint; writes `outcome.success`/`outcome.failure_reason` on the server span exactly once (the single write point) |
| `Name` + constants | `observability/names.py` | name registry | span names + attribute keys, each derived once from a namespace root via `.child()`; imported by `api/main.py`. Replaces the former inline `_ATTR_*` / literal-string approach |
| `_fetch_movie_payload(...)` | `api/main.py` | helper | opens the `*.payload_creation` span, existence-gate + TMDB fetch; raises `EndpointFailure` (404 `not_indexed`/`tmdb_removed`, 502 `tmdb_fetch_failed`) and owns the child-span error marking; returns `(card_row, tmdb_payload)` |
| `_encode_and_cache_credits(...)` | `api/main.py` | helper | opens `movie_credits.build_and_cache`, builds/encodes/caches credits; shared by both endpoints |

---

## 8. What is deliberately NOT instrumented yet

So downstream docs don't overclaim coverage:

- **Endpoints without manual spans:** `/query_search`, `/rerun_query_search`,
  `/similarity_search`, `/attribute_search`. They still get the auto server/DB/
  cache/HTTP spans, but no pipeline-stage spans, no `gen_ai.*` LLM attributes,
  no Qdrant span. (`/health` intentionally excluded entirely.)
- **No metrics** (RED per endpoint, USE for the box) — Phase 3.
- **No structured/JSON logs and no trace↔log correlation** — Phase 4.
- **No LLM token/cost/`gen_ai.*` attributes** — lands with 1c-1.
- **Framework-level 422s carry no `outcome.*`.** A missing or mis-typed query/path
  param (e.g. no `q`, a non-int `tmdb_id`) is rejected by FastAPI's validation
  *before* the handler runs, so `@record_outcome` never sees it. Covering these
  would need an app-level exception handler; deferred (in-handler validation
  already reports `invalid_parameters`).
- **No production telemetry** — local otel-lgtm only.

---

## 9. Permanent docs that should be updated

This file is transient context in `observability_context/`. The permanent project
docs currently say **nothing** about observability; when this work is stable they
should be reconciled (via the normal doc-maintenance flow — not autonomously):

- **`CLAUDE.md`** — the Architecture Overview has no observability section; add a
  short "Observability" subsection (OTel bootstrap, backends, what's traced).
- **`docs/modules/api.md`** — document the per-endpoint telemetry (spans +
  attributes) alongside each endpoint's behavior.
- **`docs/conventions.md`** — promote the §5 conventions (the naming rules,
  request-span vs child span, cardinality rule, the 404/502/unexpected error
  contract) into cross-codebase invariants. The naming rules themselves live in
  `observability/names.py`'s docstring (the source of truth); the promotion
  should reference that module rather than restate it.
- **Consider** a dedicated `docs/modules/observability.md` for the bootstrap
  module + backend topology, if the tracing surface keeps growing.

Until then, **this document is the reference** for what the observability system
actually does.
