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

**Last updated:** 2026-07-06 · **Phase:** 1 (traces) — partially complete.
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
| Manual attrs: `/query_search` **Stage 0 only** (request-boundary input capture + failure-only outcome) | 🟡 partial (1c-1 Stage 0) |
| Manual spans: `/query_search` pipeline, `/rerun_query_search`, `/similarity_search`, `/attribute_search` | ❌ not started (1c-1 Bites 2–9, 1c-2…1c-4) |
| LLM router span (`llm.generate`) + `gen_ai.*` + cost/prompt-hash/attempt-count + retry/payload events | ✅ implemented (1c-1 Bite 1; **awaiting user manual verification** 2026-07-06) |
| Metrics (RED/USE) | ❌ not started (Phase 3) |
| Structured logs + trace correlation | ❌ not started (Phase 4) |
| Production export (Grafana Cloud on EC2) | ❌ not started (Phase 5) |

**One-line summary:** every inbound request already produces a trace with
auto-instrumented network spans; three of the eight endpoints additionally carry
hand-written pipeline spans + semantic attributes, `/query_search` carries
request-boundary (Stage 0) input attributes, and **every routed LLM call now
carries a manual `llm.generate` span** (tokens/cost/prompt-hash/attempt-count +
retry/payload events) via the shared router — though `/query_search`'s pipeline
spans that would *parent* those LLM spans are still pending (Bites 2–9). Nothing
but traces exists yet (no metrics, no log correlation), and telemetry is
local-only.

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
  exception. (Framework-level 422s — a missing/mis-typed param or an unknown
  body field rejected by FastAPI *before* the handler runs — are covered by an
  app-level `RequestValidationError` handler that stamps `invalid_parameters`;
  see §8.)
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
| `outcome.success` | bool | request verdict; set on every path (**exception:** streaming endpoints under `record_outcome(success_on_return=False)` write it only on failure — see `/query_search` below) |
| `outcome.failure_reason` | `FailureReason` | set only when `success` is false: `invalid_parameters`\|`invalid_filters`\|`not_indexed`\|`tmdb_removed`\|`tmdb_fetch_failed`\|`internal_error` |

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

### `/query_search` (POST) — 1c-1 **Stage 0 only**
Partial coverage: the **request boundary** (input validation + filter
translation) is instrumented; the streaming pipeline below it (Steps 0–3,
query generation, Stage 4 execution, Qdrant, scoring) is **not yet** — those
are Bites 1–9 in `query_search_planning.md`. No manual child spans in Stage 0
(validation + translation are microsecond-scale); everything hangs on the
FastAPI **request span**.

**Input attributes** — written at handler entry from the RAW wire body, *before*
either validator runs (via `_record_query_search_inputs`), so a rejected 400/422
trace still carries the input that caused it. Text attrs are defensively
truncated at 300 chars (`_INPUT_ATTR_MAX_CHARS` — Pydantic enforces no max on
these fields; 300 > the 200-char validation caps so valid input is never
truncated); the `*_chars` attrs carry the true pre-truncation length.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `query_search.query` | string | raw query text (truncated); high-cardinality span attr, never a metric label |
| `query_search.query_chars` | int | true length of raw query pre-truncation |
| `query_search.clarification` | string | raw clarification text (truncated); set only when the field was sent |
| `query_search.clarification_chars` | int | true length of raw clarification; set only when sent |
| `filters.min_release_ts` / `max_release_ts` | int | raw wire bound; set only when sent |
| `filters.min_runtime` / `max_runtime` | int | raw wire bound; set only when sent |
| `filters.min_maturity_rank` / `max_maturity_rank` | int | raw wire bound; set only when sent |
| `filters.genres` / `audio_languages` / `keywords` / `streaming_services` | string[] | raw wire enum values, PRE-translation; set only when the list is non-empty |
| `filters.active_count` | int | number of active filter fields; **always set** (0 = none). The one low-cardinality, metric-label-eligible member of `filters.*` |

Per-field `filters.*` attrs are always-on (not error-path-only) deliberately:
the numeric fields have no validation, so a valid-but-wrong filter (wrong-unit
timestamp, min>max) yields empty results on a *successful* trace — the case
these attrs are most needed on never trips an error hook. Attribute existence =
"this filter is active"; the typed value carries the debugging detail.

**Stream-end rollup — request cost.** One rollup attribute is written on the
server span at stream end (the first non-input `query_search.*` fact):

| Attribute | Type | Meaning |
|-----------|------|---------|
| `query_search.cost_usd` | float | total USD cost of the request = every LLM call's cost + every embedding call's cost, summed across **all billed attempts** (a retried/failed-but-billed attempt still counts) |

Mechanism: the handler's `event_stream()` generator enters
`observability/cost_tracking.py::track_request_cost()` *before* the pipeline
spawns any `asyncio` task, so a `ContextVar`-held mutable accumulator is shared
by reference into every branch; each provider self-accounts its per-attempt cost
(`_account_llm_call_cost`) and `generate_vector_embedding` accounts its cost, all
no-ops outside a tracked request. The total is written on the server span in the
generator's `finally` (which runs while the ASGI server span is still open —
that span ends only after the generator fully drains), so partial-failure and
client-disconnect paths still report the cost incurred. This is a superset of the
per-call `llm.cost_usd` (which reflects only each call's successful attempt).
Unpriced models contribute `0` (they emit no `llm.cost_usd` and log a warning),
so the rollup under-reports rather than fabricates when a model is missing from
the pricing table.

**Span events:** `request rejected` (with a `detail` attribute) — emitted on the
400 (empty/over-length query or clarification) and 422 (unknown filter enum
value) paths. Expected 4xx rejections get no `record_exception` per the §5.2
error contract, so this event is where the offending field/value lands on the
trace.

**Outcome semantics — failure-only (interim).** The endpoint uses
`@record_outcome(success_on_return=False)` (see §7): failures are recorded, but a
clean handler return writes **no** `outcome.success=true`. Rationale: the handler
returns a `StreamingResponse` *before* the pipeline runs, so "returned cleanly"
only means the request passed the boundary — stamping success there would read as
true even if the stream later fails fatally. The success verdict (and the
stream-end rollups: total result count, fetch/failed-branch counts) is deferred
to a stream-aware mechanism (`query_search_planning.md` Bite 2). Until then:

| Path | `outcome.success` | `outcome.failure_reason` |
|------|-------------------|--------------------------|
| Empty/over-length query or clarification (400) | false | `invalid_parameters` |
| Unknown body field, e.g. typo'd filter key (422, framework) | false | `invalid_parameters` |
| Unknown filter enum value (422) | false | `invalid_filters` |
| Pre-stream crash | false | `internal_error` |
| Passed the boundary (200, stream begins) | *(absent — deferred to Bite 2)* | — |

### LLM router span — `llm.generate` (1c-1 Bite 1) — cross-cutting
Not an endpoint: one span emitted by
`implementation/llms/generic_methods.py::generate_llm_response_async` — the
single codepath every routed structured-generation call passes through — so
one instrumentation point covers every step's LLM call on every endpoint. The
span wraps the **whole retry loop** (not one span per attempt); step identity
comes from the parent span nesting, never duplicated onto this span. It nests
under whatever pipeline span is active (once those land in Bites 2–9); today,
with no `/query_search` pipeline spans yet, it hangs directly off the auto httpx
child under the request span.

Uses `record_exception=False, set_status_on_exception=False` so a *recovered*
retry never reads as an error and only an exhausted retry marks ERROR (§5.2
error contract, applied to LLM calls).

**Attributes:**

| Attribute | Type | Meaning |
|-----------|------|---------|
| `gen_ai.system` | string | provider (`openai`/`kimi`/`gemini`/`groq`/`alibaba`/`anthropic`/`wham`). Standard OTel GenAI key — not authored in `names.py` |
| `gen_ai.request.model` | string | requested model. Standard OTel GenAI key |
| `gen_ai.usage.input_tokens` / `output_tokens` | int | token usage; set on the successful attempt only. Standard OTel GenAI keys |
| `gen_ai.usage.cache_read.input_tokens` | int | input tokens served from the provider's prompt cache (a subset of `input_tokens`, billed at a discount); set on every span (even `0`) so cache-hit rate is queryable. Standard OTel GenAI key (Anthropic reports cache reads separately and is deliberately excluded → always `0`) |
| `llm.cost_usd` | float | computed dollar cost via `implementation/llms/pricing.py::compute_llm_cost_usd` (cache-adjusted: cached input priced at the discounted rate); **omitted** (with a logged warning) when the model is unpriced — never a fabricated `$0`. Reflects the successful attempt; the request-level superset is `query_search.cost_usd` |
| `llm.prompt_version` | string | 12-char sha256 of the SYSTEM prompt (`@lru_cache`d); changes iff the prompt text changes — lets evals slice by prompt revision |
| `llm.attempt_count` | int | attempts made: `1` = clean first try, `>1` = retried, `LLM_MAX_ATTEMPTS` on exhaustion. With span status this separates clean / recovered / exhausted |
| `error.type` | string | set **only on exhaustion** (span-level): normalized failure class — `timeout` for the `asyncio.wait_for` ceiling, else the exception class name. Standard OTel key |

**Failure marking (three states — `query_search_planning.md` §2.8):**

| State | Span status | `llm.attempt_count` | Events |
|-------|-------------|---------------------|--------|
| Clean success | UNSET | 1 | `llm.payload` (if sampled) |
| Failed but recovered | UNSET (green) | > 1 | one `llm.retry` per failed attempt + `llm.payload` (if sampled) |
| Failed all retries | ERROR + `record_exception` | `LLM_MAX_ATTEMPTS` | `llm.retry`(s) + `exception` + prompt-only `llm.payload` (always-on-error) |

An ERROR `llm.generate` span does **not** flip the request verdict — per §2.5
per-call failures are degradations (the branch will carry `branch_error`; the
request span stays `success=true`). Recovered = `attempt_count > 1 && status !=
error`; exhausted = `status = error`; either sliceable by `error.type`.

**Span events:**

| Event | When | Attributes |
|-------|------|-----------|
| `llm.retry` | each failed-but-retried attempt (always-on) | `attempt` (int), `error.type` (string), `timeout` (bool), `backoff_seconds` (float) |
| `llm.payload` | successful call at the sampled rate; **failure** captures prompt-only whenever capture is enabled | `system_prompt`, `user_prompt`, `response` (JSON; absent on the failure path) |

**Payload capture config.** `LLM_PAYLOAD_CAPTURE_SAMPLE_RATE` (env var, float,
read once at module import; default `1.0`). `>=1.0` captures every successful
call, `0` disables entirely, else Bernoulli(rate) per successful call; when the
rate is `>0`, a terminal failure captures the prompt unconditionally
(always-on-error). Full payloads ride span **events** (not attributes) so the
sampling dial can drop them and attribute size limits don't apply.

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
| `FailureReason(str, Enum)` | `api/outcome.py` | enum | the coarse `outcome.failure_reason` vocabulary (replaces the old `NotFoundReason`, folding its two 404 reasons into the broader set). Members: `invalid_parameters`, `invalid_filters` (unknown hard-filter enum value — UI/server taxonomy drift, distinct from `invalid_parameters`), `not_indexed`, `tmdb_removed`, `tmdb_fetch_failed`, `internal_error` |
| `EndpointFailure(HTTPException)` | `api/outcome.py` | exception | request-terminating failure that carries its `FailureReason`; raised at each known failure site so the reason bubbles up |
| `record_outcome` | `api/outcome.py` | decorator | wraps each endpoint; writes `outcome.success`/`outcome.failure_reason` on the server span exactly once (the single write point). Dual-form: bare `@record_outcome` (default, `success_on_return=True`) stamps `success=true` on clean return; `@record_outcome(success_on_return=False)` records failures only and leaves success absent on clean return — for streaming (SSE) endpoints whose handler returns before the pipeline runs (`/query_search`) |
| `Name` + constants | `observability/names.py` | name registry | span names + attribute keys, each derived once from a namespace root via `.child()`; imported by `api/main.py`. Replaces the former inline `_ATTR_*` / literal-string approach |
| `_fetch_movie_payload(...)` | `api/main.py` | helper | opens the `*.payload_creation` span, existence-gate + TMDB fetch; raises `EndpointFailure` (404 `not_indexed`/`tmdb_removed`, 502 `tmdb_fetch_failed`) and owns the child-span error marking; returns `(card_row, tmdb_payload)` |
| `_encode_and_cache_credits(...)` | `api/main.py` | helper | opens `movie_credits.build_and_cache`, builds/encodes/caches credits; shared by both endpoints |
| `_tracer` | `implementation/llms/generic_methods.py` | module tracer | `trace.get_tracer(__name__)`; a no-op `ProxyTracer` when `setup_tracing` hasn't run (offline ingestion/eval imports), so the `llm.generate` span is a cheap no-op there |
| `generate_llm_response_async` | `implementation/llms/generic_methods.py` | router + LLM instrumentation point | the single codepath every routed LLM call passes through; owns the `llm.generate` span, `gen_ai.*`/`llm.*` attrs, retry/payload events, and the three-state failure marking |
| `compute_llm_cost_usd` | `implementation/llms/pricing.py` | cost util | canonical `(input, cached_input, output)`-per-million pricing table → cache-adjusted USD cost; returns `None` for unpriced models (caller omits `llm.cost_usd` + warns). A parallel table in `estimate_generation_cost.py` should later import from here (see `docs/TODO.md`) |
| `track_request_cost` / `add_request_cost` | `observability/cost_tracking.py` | cost rollup | `ContextVar`-scoped per-request accumulator summing every LLM + embedding call's cost (all billed attempts) → written on the server span as `query_search.cost_usd`; no-op outside a tracked request |

---

## 8. What is deliberately NOT instrumented yet

So downstream docs don't overclaim coverage:

- **Endpoints without manual pipeline spans:** `/query_search` (has Stage 0
  request-boundary *attributes* but no pipeline-stage spans yet),
  `/rerun_query_search`, `/similarity_search`, `/attribute_search`. They still
  get the auto server/DB/cache/HTTP spans **and** the `llm.generate` span on any
  LLM call they make (it rides the shared router), but no pipeline-stage spans
  to parent those LLM spans, and no Qdrant span. (`/health` intentionally
  excluded entirely.)
- **No metrics** (RED per endpoint, USE for the box) — Phase 3.
- **No structured/JSON logs and no trace↔log correlation** — Phase 4.
- ✅ **LLM token/cost/`gen_ai.*` attributes now land** on the shared
  `llm.generate` span (1c-1 Bite 1) — see §6. What's still missing is the
  *pipeline* spans that would parent them (Bites 2–9).
- **Framework-level 422s now carry `outcome.*`** (app-wide). A missing or
  mis-typed query/path param (e.g. no `q`, a non-int `tmdb_id`) or an unknown
  body field is rejected by FastAPI's validation *before* the handler runs, so
  `@record_outcome` never sees it. An app-level `RequestValidationError` handler
  (`_on_request_validation_error` in `api/main.py`) closes this gap: it stamps
  `outcome.success=false` + `invalid_parameters` and adds a `request rejected`
  event whose `detail` names the offending field(s) (location + message + type,
  never the input value — PII-safe), then delegates to FastAPI's default handler
  so the HTTP response is unchanged. Unknown body fields are only *reachable* now
  that `QuerySearchBody` and `MetadataFiltersInput` set `extra="forbid"` (a typo'd
  filter key like `genrez` used to be silently dropped; it now 422s and is
  recorded). This handler is app-wide, so even the not-yet-instrumented endpoints
  get a verdict on malformed-body requests.
- **No production telemetry** — local otel-lgtm only.

---

## 9. Permanent docs reconciliation — DONE (2026-07-04)

This file is transient context in `observability_context/`. The permanent project
docs originally said nothing about observability; the reconciliation below has now
been applied (docs-maintainer pass + follow-up), so it should **not** be redone —
only kept current as the tracing surface grows:

- ✅ **`CLAUDE.md`** — Architecture Overview now has an "Observability" subsection
  (OTel bootstrap, Grafana backends, what's traced, partial coverage), pointing
  here and to the two docs below.
- ✅ **`docs/modules/api.md`** — per-endpoint telemetry (spans + attributes)
  documented inline per endpoint, plus a summary "Observability" section.
- ✅ **`docs/conventions.md`** — the §5 conventions (naming registry, request-span
  vs child span, cardinality rule, per-request outcome, 404/502/unexpected error
  contract) are promoted into a codebase-wide "Observability Conventions" section,
  worded to generalize beyond `api/` as instrumentation spreads. It references
  `observability/names.py`'s docstring as the naming source of truth rather than
  restating it. (This subsumed the former ADR-102 draft, which was promoted to a
  convention rather than kept as a standalone decision record.)
- ✅ **`docs/modules/observability.md`** — created for the `observability/` package
  (bootstrap module + name registry + backend topology).
- ✅ **`docs/decisions/ADR-101`** — the OTel/Grafana stack-selection decision record
  is finalized (`Status: Active`).

**This document remains the reference** (source of truth) for what the
observability system actually does; the permanent docs above summarize and point
back here.
