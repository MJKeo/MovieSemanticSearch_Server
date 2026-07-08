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

**Last updated:** 2026-07-08 · **Phase:** 1 (traces) — partially complete.
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
| Manual attrs: `/query_search` — request-boundary input capture + outcome (boundary failures + mid-stream fatal Step 0) + stream-end `request.cost_usd` rollup | 🟡 partial (1c-1 Stage 0 + cost rollup + Step-0 fatal verdict) |
| Manual spans: `/query_search` pipeline — Steps 0 + 1 (`query_search.step_0` / `.step_1`) | ✅ implemented + verified (1c-1 Bite 3) |
| Manual spans: `/query_search` pipeline — per-branch (`query_search.branch`, `branch_type` + `branch_uses_original_text`); also emitted on `/rerun_query_search` | 🟡 spans + attrs landed (1c-1 Bite 4); `branch_error`/result-count attrs pending |
| Manual spans: `/query_search` trait pipeline — Step 2 / trait / Step 3 / query generation (`query_search.step_2` / `.trait` / `.step_3` / `.query_generation`, incl. `query_generation_endpoints` + `"solo trim"` event) | ✅ implemented + verified (1c-1 Bite 4) |
| Manual spans/attrs: `/query_search` entity flows (person / similarity / exact_title / studio / franchises) — `branch_*` attributes + per-flow child spans (person resolution, char-franchise resolutions, similarity Qdrant + per-lane candidate fetches) | 🟡 code landed, awaiting Tempo verification (1c-1 Bite 8) |
| Manual spans: `/query_search` Stage 4 semantic Qdrant probes (`query_search.semantic_qdrant` × 3 primitives + retry/failed events) | ✅ implemented + verified (1c-1 Bite 6; now nests under the `query_search.dispatch` span) |
| Manual spans: `/query_search` implicit-prior generation + application (`query_search.implicit_expectations` + `query_search.implicit_prior_rerank`, `boost_axis` + policy/selection attrs + failure events) | ✅ implemented (application span now nests under `scoring`, applied inside Stage 4's `_run_branch`) |
| Manual spans: `/query_search` Stage 4 execution — six per-branch groups: `step_2` (A) / `decomposition` (B) / `candidate_generation` (C, wraps `.generators` / `.promotion` / `.neutral_seed` / `.auxiliary_shorts_exclusion`) / `rerankers` (D, positive **and** negative in parallel; `dispatch.polarity`) / `scoring` (E, `implicit_prior_rerank` nested) / `hydration` (F). Per-stage `cost_usd` on A–D via nested cost scopes; `.dispatch` + fallback/dedup/shorts/timeout events | 🟡 code landed, awaiting Tempo verification |
| Manual spans/attrs: `/similarity_search` — flow-neutral `similarity.*` signal set + `similarity.qdrant`/`.fetch` child spans (from the shared engine) on the server span, plus endpoint `similarity_search.cache_hit` + cross-endpoint `request.result_count` + `filters.*` + `@record_outcome` verdict | ✅ implemented (1c-3) |
| Manual spans/attrs: `/attribute_search` — flow-neutral `person.resolve` child span (shared with the `/query_search` person branch) per supplied name, plus endpoint `attribute_search.*` path/people/pool skeleton + cross-endpoint `request.result_count` + `filters.*` + `@record_outcome` verdict | ✅ implemented (1c-4) |
| Manual span: shared `person.resolve` — one per resolved person (`resolve_person_traced`), flow-neutral root, emitted by both the `/query_search` person branch and `/attribute_search` | ✅ implemented (1c-4) |
| Cross-endpoint `request.*` rollups (`cost_usd` / `usage.*` / `result_count`) — on any endpoint's server span; the no-LLM endpoints emit only `result_count` | ✅ implemented (`/query_search`, `/rerun_query_search`, `/similarity_search`, `/attribute_search`, `/title_search`) |
| Manual attrs/spans: `/rerun_query_search` — server-span input capture (`rerun_query_search.*` + `filters.*`), `trace.use_span` so the reused `query_search.*` pipeline spans nest (not orphan), stream-end `request.*` rollups + verdict + branch-count, `EndpointFailure` boundary rejections | 🟡 code landed (1c-2), awaiting Tempo verification |
| Manual attrs: `/query_search` terminal rollups (success verdict + branch/result counts) | ✅ implemented (1c-1 Bite 2; Bite 9 end-to-end validation pending) |
| LLM router span (`llm.generate`) + `gen_ai.*` (incl. `usage.cache_read.input_tokens`) + cache-adjusted cost/prompt-hash/attempt-count + retry/payload events; per-request cost rollup (`request.cost_usd`) | ✅ implemented + verified (1c-1 Bite 1 + cost rollup; retry/recovery + exhaustion paths exercised 2026-07-07) |
| Embedding router span (`embedding.generate`) + `gen_ai.*` (system/model/operation/input_tokens) + `embedding.cost_usd` / `embedding.input_count`; single-attempt error contract | ✅ implemented + verified (shared `generate_vector_embedding`; covers search + offline ingestion; success + error paths + accumulator reconciliation exercised 2026-07-07) |
| Metrics (RED/USE) | ❌ not started (Phase 3) |
| Structured logs + trace correlation | ❌ not started (Phase 4) |
| Production export (Grafana Cloud on EC2) | ❌ not started (Phase 5) |

**One-line summary:** every inbound request already produces a trace with
auto-instrumented network spans; most of the endpoints additionally carry
hand-written pipeline spans + semantic attributes (`/attribute_search` is the
newest — 1c-4, sharing the flow-neutral `person.resolve` span), `/query_search` carries
request-boundary (Stage 0) input attributes, and **every routed LLM call now
carries a manual `llm.generate` span** (tokens/cost/prompt-hash/attempt-count +
retry/payload events) via the shared router — with the matching
**`embedding.generate` span** on every embedding call (search + offline ingestion),
carrying `gen_ai.*` + `embedding.cost_usd` / `embedding.input_count`.
`/query_search` now has its first
pipeline spans — `query_search.step_0` / `.step_1` (flow routing + spin
generation) parenting their `llm.generate` children under the server span — plus a
stream-end `request.cost_usd` rollup and a mid-stream fatal-Step-0 outcome
verdict on the server span; the per-branch spans (Bite 4), the standard-branch
**trait pipeline** (`query_search.step_2` / `.trait` / `.step_3` /
`.query_generation`, Bite 4), the entity-flow `branch_*` attributes (Bite 8),
the Stage-4 semantic Qdrant probe spans (`query_search.semantic_qdrant`, Bite 6),
the implicit-prior generation + application spans
(`query_search.implicit_expectations` / `.implicit_prior_rerank`), and the full
six-group Stage-4 branch tree — `candidate_generation` / `rerankers` (positive +
negative in parallel) / `scoring` (implicit rerank nested) / `hydration`, with
per-stage `cost_usd` — have also landed, as have `/similarity_search` (1c-3) and
`/attribute_search` (1c-4, with the shared `person.resolve` span). Still pending:
the success verdict/rollups and `/rerun_query_search` (Bites 2 remainder, 9, 1c-2).
Nothing but traces exists yet (no metrics, no log correlation), and telemetry is
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
| psycopg v3 | Postgres | child span per query (carries the SQL **and** bound params — `instrument(capture_parameters=True)`, so `db.statement.parameters` disambiguates otherwise-identical parameterized statements; see the note below §3) |
| redis | cache GET/SET | child span per command |

**Deliberate gap — Qdrant (now closing).** Its async client speaks gRPC, which
the auto-instrumentation doesn't wrap cleanly, so Qdrant timing comes from
hand-written spans. The similar-movies engine's **both** Qdrant calls are covered
by `similarity.qdrant` (1c-3; renamed from `query_search.similarity_qdrant` when
the spans were made flow-neutral) — the `anchor_vectors` retrieve and the `shape`
batched probe, split by `probe_kind` (see §6) — and since they're emitted inside
the shared engine, they fire on **both** `/query_search`'s similarity branch and
the pure `/similarity_search` endpoint. Alongside these, the Stage-4 semantic
probes `query_search.semantic_qdrant` (Bite 6, three primitives) cover the SEMANTIC
path. Any remaining `/query_search` vector paths are still uninstrumented. Not a
bug — a known, documented omission being retired flow by flow. (`/attribute_search`
is pure Postgres — no Qdrant call sites, so nothing to close there.)

**SQL parameter capture.** `PsycopgInstrumentor().instrument(capture_parameters=True)`
adds `db.statement.parameters` (the bound param sequence) to every psycopg query
span, so structurally identical parameterized statements — e.g. `… FROM
public.movie_card WHERE movie_id = ANY($1)` fired for anchors (2 IDs) vs candidates
vs hydration — are told apart by their arrays. Global (all queries, all endpoints);
our params are movie/trait IDs + filter values, no secrets/PII. High-cardinality →
span-attr only, never a metric label; non-standard attribute, intended for debugging.

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
  | `request.failure_reason` | dot | `request` already groups ≥2 emitted attributes (`success`, `failure_reason`, `cost_usd`, …); `failure_reason` names WHY a request failed, paired with the `request.success` bool |
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
  only low-cardinality keys (`movie.payload_source`, `request.success`,
  `request.failure_reason`) are label-eligible.

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
  endpoint carries `request.success` (bool) on its server span, plus
  `request.failure_reason` (`FailureReason`) whenever `success` is false. It is
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
    `request.success=false` + `not_indexed`/`tmdb_removed`.
  - **502** (upstream TMDB failure) → span marked **ERROR** + `record_exception`;
    verdict `tmdb_fetch_failed`.
  - **Unexpected** exception → also marked **ERROR** + recorded, so the span that
    contains the failing op never reads green; verdict `internal_error`.
  - Swallowed best-effort failures (cache read/write, cross-populate) → span
    **events**, never span errors, and **not** request failures — they're
    degradations, so `request.success` stays true.
- **`source` is set only at a success point**, never optimistically → a 404/502
  carries no `movie.payload_source` (absent = no payload served), keeping
  cache-hit-rate honest.

---

## 6. Span & attribute catalog (implemented endpoints)

**Universal (every endpoint below).** On the request/server span, via
`@record_outcome` (§5.2):

| Attribute | Type | Meaning |
|-----------|------|---------|
| `request.success` | bool | request verdict; set on every path (**exception:** streaming endpoints under `record_outcome(success_on_return=False)` write it only on failure — see `/query_search` below) |
| `request.failure_reason` | `FailureReason` | set only when `success` is false: `invalid_parameters`\|`invalid_filters`\|`not_indexed`\|`tmdb_removed`\|`tmdb_fetch_failed`\|`internal_error` |

### `/title_search` (GET) — 1c-5
No child span (single unit of work). Attributes on the **request span**:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `title_search.query` | string | raw query text (attribute-only; never a metric label) |
| `title_search.limit` | int | requested limit |
| `request.result_count` | int | hydrated cards returned (incl. 0) — the cross-endpoint rollup (see §"Stream-end rollup") |
| `title_search.fuzzy_result_count` | int | how many results came from the fuzzy fallback tier (>0 = likely typo / catalog gap) |

Supporting change: `run_title_search` returns a `TitleSearchResult` NamedTuple
`(movie_ids, fuzzy_count)` (was `list[int]`) so the endpoint can report the fuzzy
count without re-deriving discarded tier data (`search_v2/title_search.py`).

### `/query_search` (POST) — 1c-1 **request boundary + stream-end cost rollup**
Partial coverage: the **request boundary** (input validation + filter
translation) is instrumented, and a **stream-end cost rollup**
(`request.cost_usd`) is written when the stream completes. The streaming
pipeline *between* them is now broadly covered — Steps 0/1 spans (Bite 3),
per-branch spans + entity-flow attributes (Bites 4/8), the Stage-4 semantic
Qdrant probe spans (`query_search.semantic_qdrant`, Bite 6), Steps 2/3, and the
full six-group Stage-4 branch tree (`candidate_generation` / `rerankers` /
`scoring` / `hydration` + per-stage `cost_usd`) have landed — but the terminal
success verdict/rollups are still pending (Bites 2 remainder, 9 in
`query_search_planning.md`). No manual child spans exist here (validation +
translation are microsecond-scale, and the cost rollup is a server-span
attribute, not a span); every manual fact hangs on the FastAPI **request span**.

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

**Stream-end rollup — request cost + token usage + result count (cross-endpoint
`request.*`).** These rollups are **not** `query_search`-owned — they mean the
same thing on any endpoint, so they live under a generic `request.*` root (which
also carries the per-request verdict — see §5.2), written on the server span at
stream/handler end. `/query_search`
and `/rerun_query_search` write all of them; the no-LLM endpoints
(`/similarity_search`, `/attribute_search`, `/title_search`) write only
`request.result_count` (they incur no LLM/embedding spend, so `cost_usd` /
`usage.*` are simply absent — absence == "spent nothing").

| Attribute | Type | Meaning |
|-----------|------|---------|
| `request.cost_usd` | float | total USD cost of the request = every LLM call's cost + every embedding call's cost, summed across **all billed attempts** (a retried/failed-but-billed attempt still counts) |
| `request.usage.input_tokens` | int | total input tokens across every LLM + embedding call (all billed attempts). Cache-inclusive — cached tokens are a subset, per the next row |
| `request.usage.cached_input_tokens` | int | input tokens served from the provider prompt cache — a **subset** of `input_tokens`, never additive (same `cached ⊆ input` convention as the per-call `gen_ai.usage.cache_read.input_tokens`) |
| `request.usage.output_tokens` | int | total output/generation tokens across every LLM call (all billed attempts; embeddings contribute 0) |
| `request.result_count` | int | results the client received. On the streaming branch-plan endpoints (`/query_search`, `/rerun_query_search`) it's the pre-dedup **sum** across branches (no server-side cross-branch merge); on the single-response endpoints it's the hydrated card count |

The per-request **branch counts** stay branch-plan-owned under `query_search.*`
(only the two branch-plan endpoints have branches) and `/rerun_query_search`
**reuses** them — the same rule-B move as reusing the branch spans:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `query_search.succeeded_branch_count` | int | branches that executed without a `branch_error` (empty-but-clean counts); backs the `request.success` verdict (success = >= 1) |
| `query_search.failed_branch_count` | int | branches that soft-failed with a `branch_error` (a degradation, not a request failure) |

Mechanism: the handler's `event_stream()` generator enters
`observability/cost_tracking.py::track_request_cost()` *before* the pipeline
spawns any `asyncio` task, so a `ContextVar`-held mutable accumulator is shared
by reference into every branch; each provider self-accounts its per-attempt cost
**and tokens** (`_account_llm_call_cost` → `add_request_cost` + `add_request_tokens`)
and `generate_vector_embedding` accounts its embedding cost + input tokens, all
no-ops outside a tracked request. The totals are written on the server span in the
generator's `finally` (which runs while the ASGI server span is still open —
that span ends only after the generator fully drains), so partial-failure and
client-disconnect paths still report the usage incurred. The token rollup is a
superset of any single `llm.generate` span's `gen_ai.usage.*` (those reflect each
call's successful attempt only; the rollup counts every billed attempt).

Cost and tokens diverge deliberately on unpriced models: cost gates on the
pricing table — an unpriced model contributes `0` (emits no `llm.cost_usd`, logs
a warning), so `cost_usd` under-reports rather than fabricates — but tokens do
**not** gate on price (`add_request_tokens` is unconditional), so an unpriced
model's real tokens still count toward `request.usage.*`.

**Pipeline spans — Steps 0 + 1 (Bite 3).** The first per-stage spans on the
streaming path: two manual child spans wrapping the parallel LLM pair at the head
of `stream_full_pipeline` (`search_v2/streaming_orchestrator.py`). Both are
children of the server span and overlap in the waterfall (they launch together);
each parents the router's `llm.generate` child, so a step's tokens/cost/prompt
hash/payload come free from that child and are never duplicated onto the step
span. Owned by `query_search` (rule B / OQ #1 resolved): /rerun_query_search
reuses Step 2 → Stage 4, not routing.

| Span | Attribute | Type | Meaning |
|------|-----------|------|---------|
| `query_search.step_0` | `query_search.step_0_flows` | string[] | activated flow names — the entity flow's `SearchFlow` value (`person`, `exact_title`, …) + `standard` when it co-fires; never empty on success. Low-cardinality closed set |
| `query_search.step_0` | `query_search.step_0_standard_branch_count` | int | standard-flow branch budget (0 = standard didn't fire). **Always set** on success; low-cardinality, label-eligible |
| `query_search.step_1` | `query_search.step_1_unused` | bool | true when routing left no budget for spins (`not _step1_needed`), so Step 1's output feeds no branch. Derivable from the two step_0 attrs, recorded directly for legibility on the span |
| `query_search.branch` | `query_search.branch_type` | string | the fetch type — one of `standard` / `exact_title` / `similarity` / `non_character_franchise` / `character_franchise` / `studio` / `person` (mirrors `SearchFlow` values). Low-cardinality closed set; **set on every branch span** |
| `query_search.branch` | `query_search.branch_uses_original_text` | bool | true only for the first standard branch (`standard:original`) of the non-clarification flow, which searches the typed query verbatim rather than a generated/rewritten query. Set on **standard** branches only; false for spins and for every branch on the rerun path |

**Trait-pipeline spans — Step 2 / Step 3 / query generation (1c-1 Bite 4,
partial).** The standard-branch trait pipeline, from Step 2 through
query-generation completion (Stage 4 execution is deliberately out of scope — see
below). All are owned by `query_search`, sit under the `query_search.branch` span,
and each parents its own `llm.generate` child (tokens/cost/prompt hash/payload
come free from that child, never duplicated onto the stage span). Created in place
with plain `tracer.start_as_current_span(...)` in `full_pipeline_orchestrator.py`
(`_run_step2_for_branch`, `_decompose_and_generate`) and `handler.py`
(`run_query_generation`) — those modules run with the branch span already current
(each branch task is `_run_under_span`-wrapped in the streaming layer), so no
non-current `use_span` dance is needed. `step_2` and the per-`trait` spans are
**siblings** under the branch: `step_2` closes at its LLM return, the `trait` spans
start after. Each `trait` span brackets that one trait's Step 3 → query generation;
`step_3` wraps the Step-3 LLM call; `query_generation` wraps only the handler-LLM
call.

| Span | Attribute | Type | Meaning |
|------|-----------|------|---------|
| `query_search.step_2` | `query_search.step_2_trait_count` | int | number of traits Step 2 committed |
| `query_search.step_2` | `query_search.step_2_contextualized_phrases` | string[] | the traits' `contextualized_phrase` strings in order — at-a-glance "what did Step 2 decide". High-cardinality span attr, never a metric label |
| `query_search.trait` | `query_search.trait_phrase` | string | the trait's `contextualized_phrase` (high-cardinality) |
| `query_search.trait` | `query_search.trait_polarity` | string | `Polarity` value — `positive` / `negative` |
| `query_search.trait` | `query_search.trait_commitment` | string | `required` / `elevated` / `neutral` / `supporting` / `diminished` |
| `query_search.trait` | `query_search.trait_step_3_error` | string | set only on a Step-3 soft-fail (retries exhausted). Degradation — the trait span stays UNSET; the request verdict is untouched |
| `query_search.step_3` | `query_search.step_3_combine_mode` | string | `TraitCombineMode` value — `solo` / `framings` / `facets` |
| `query_search.step_3` | `query_search.step_3_categories` | string[] | category names of the surviving category calls, recorded **POST-SOLO-trim** (the calls that actually reach retrieval, not the raw committed set) |
| `query_search.query_generation` | `query_search.query_generation_category` | string | the category name this handler-LLM call routes |
| `query_search.query_generation` | `query_search.query_generation_endpoints` | string[] | the `EndpointRoute` names that actually fired for this call — just which endpoints activated (the detailed params ride the nested `llm.generate` payload). Empty `[]` when the handler fired nothing |

**Span events (trait span).** `"solo trim"` — emitted when the SOLO trim in
`_decompose_and_generate` drops extra category calls (attributes `kept_category`,
`dropped_count`); previously a log-only line.

**Error / soft-fail marking.** A Step-2 soft-fail marks the `step_2` span ERROR +
`record_exception` (the branch still degrades via `branch_error`). A Step-3
soft-fail marks the `step_3` span ERROR + records the exception, and sets
`trait_step_3_error` on the `trait` span, which itself stays UNSET. Neither flips
the request verdict — per-call/per-trait failures are degradations (§5.2 error contract),
and the underlying `llm.generate` child already carries the ERROR status.

**Deliberately still out of scope (per the Bite-4 scope cut).** The
`query_generation` span exists only for handler-LLM calls: the `EXPLICIT_NO_OP` and
`NO_LLM_PURE_CODE` buckets return before the span, so deterministic/no-op calls get
no span. **Stage 4 query execution is not instrumented here** — its candidate pool
is a branch-level union (not per-trait), so execution can't nest under a `trait`
span; it becomes a separate branch-level span in a later bite, correlated back to a
trait by attribute rather than nesting. (The `implicit_expectations` LLM call is
now instrumented — see the implicit-prior subsection below.)

**Implicit-prior spans — generation + application (1c-1 Bite 4 [deferred] +
Bite 7, verified 2026-07-08).** The implicit-prior mechanism is a two-location flow, so it gets two
manual spans, both **per standard branch** (entity flows never run it), both
under the `query_search.branch` span. Generation is instrumented in
`_run_implicit_expectations_for_branch` (`full_pipeline_orchestrator.py`);
application in `apply_implicit_prior_rerank_for_branch`
(`search_v2/implicit_prior_rerank.py`), which Stage 4's `_run_branch` now calls
**inside its `scoring` span** — so the `implicit_prior_rerank` span nests under
`scoring`, and both the streaming API and the batch/CLI path pick it up through
`_run_branch` (the separate orchestrator-level rerank calls were removed). Design note:
the rerank is **single-axis** (ADR-087 — popularity primary, quality fallback),
so `boost_axis` names the ONE axis that fired; both priors' direction+strength
are recorded on the **application** span (not the generation span) so the LLM's
proposal sits beside the code's selection in one read. Two value enums
(`BoostAxis`, `PriorNoopReason`) live in `full_pipeline_orchestrator.py` per
names.py rule E.

`query_search.implicit_expectations` (generation) wraps the policy LLM call; it
carries **no manual attributes** (the nested `llm.generate` child has
tokens/cost/prompt-hash/full payload) and exists to name that child's parent,
time the call (on Stage 4's critical path), and anchor the soft-fail event.

| Span | Attribute | Type | Meaning |
|------|-----------|------|---------|
| `query_search.implicit_prior_rerank` | `implicit_prior_rerank.boost_axis` | string | which axis fired — `popularity`\|`quality`\|`none` (`BoostAxis`). Low-cardinality |
| ″ | `implicit_prior_rerank.popularity_direction` / `.popularity_strength` | string | policy output for the popularity axis (`none`/`positive`/`inverse`; `none`/`light`/`normal`/`strong`) |
| ″ | `implicit_prior_rerank.quality_direction` / `.quality_strength` | string | policy output for the quality axis |
| ″ | `implicit_prior_rerank.popularity_cap` / `.quality_cap` | float | strength→boost ceiling the code resolved (0 disables the axis). Span-attr-only (rule F) |
| ″ | `implicit_prior_rerank.popularity_active` / `.quality_active` | bool | the selection variables — which axis the code chose to fire |
| ″ | `implicit_prior_rerank.inverse_applied` | bool | fired axis used the inverse direction (rewards LOW popularity/reception); false when `boost_axis=none` |
| ″ | `implicit_prior_rerank.noop_reason` | string | set **only** when `boost_axis=none` (`PriorNoopReason`): `policy_unavailable`\|`branch_error`\|`empty_pool`\|`both_axes_off` — disambiguates the four no-op causes the active flags can't (the first three return before caps/active exist) |
| ″ | `implicit_prior_rerank.signal_missing_count` | int | candidates whose FIRED-axis signal was NULL in Postgres (→ 0 boost) — the data-coverage risk. Set only when an axis fired |

The span is started **before the gate** so a skipped branch still emits a
legible span (`boost_axis=none` + `noop_reason`). The Postgres signal fetch
(`fetch_quality_popularity_signals`) runs inside the span, so its auto psycopg
child nests here. Rerank behavior is byte-for-byte unchanged — telemetry only.

**Span events (implicit-prior).** `implicit_expectations_failed` (on the
generation span; generation soft-fail — span ERROR + `record_exception`, with
`error.type` = `schema_mismatch` for output-validation/Pydantic `ValueError`s
vs. the provider/timeout exception class). `implicit_prior_apply_failed` (on the
application span; the signal-fetch Postgres call threw — the event is added and
the exception re-raised so branch soft-fail propagation is unchanged; the context
manager records the exception + ERROR status). Neither flips the request verdict — both
are per-branch degradations (§5.2).

**Per-branch spans (`query_search.branch`, 1c-1 Bite 4).** One span per fetch —
up to three standard branches (main query + spins) plus at most one entity flow —
brackets that branch's full lifecycle (launch → terminal `branch_results`) and
**parents its work**: each of the branch's asyncio tasks (standard: Step 2 → Step
3 → Stage 4; entity flow: one wrapper task) runs under the span via
`_run_under_span` (`use_span(end_on_exit=False)`), so the branch's `llm.generate`
children nest beneath it instead of directly under the server span. Started
non-current in `_stream_from_branch_plan` right after `fetches_ready`, keyed by
`fetch_id`; closed centrally in the merge loop when no live task remains for that
`fetch_id` (the signal that the branch reached its terminal `branch_results`),
with a `finally`-block safety net for the client-disconnect path. Both the full
pipeline and the `/rerun_query_search` replay emit these (they share
`_stream_from_branch_plan`); `branch_uses_original_text` defaults false on
replays. A per-branch soft-fail (`branch_error`) is a **degradation** — it does
not set the span to ERROR or touch the request verdict (span status stays UNSET); a
`branch_error` attribute is a later bite.

**Entity-flow attributes on `query_search.branch` (1c-1 Bite 8).** The six
non-standard flows record — on the branch span — what they searched, how it
resolved, and how the result was composed. All are set INSIDE each flow's entry
executor (`run_*_search`), which runs under the branch span (`get_current_span()`
there IS the branch span), so they never touch the shared Stage-4 executors that
standard branches use, and no-op outside a traced request. Aliases are always-on
(low traffic). The empty-result **event** is the message string `entity flow
empty`.

Universal skeleton (every entity flow):

| Attribute | On | Type | Meaning |
|-----------|----|------|---------|
| `query_search.branch_entities` | branch span | string[] | Step 0 canonical identity (person names / reference titles / studio names / franchise / character / exact title) |
| `query_search.branch_entity_resolved_counts` | branch span | int[] | per-entity PRE-union resolved candidate count, index-aligned with `branch_entities`; a `0` marks a silently-dropped entity |
| `query_search.branch_unresolved_entity_count` | branch span | int | count of entities that resolved to 0 (the low-cardinality, alertable form) |
| `query_search.branch_result_count` | branch span | int | post-hydration card total; set in the orchestrator wrapper (`_stamp_branch_outcome`) where the count is known |
| `query_search.branch_aliases` | branch span | string[] | LLM-expanded surface forms (studio brand+freeform names; non-char-franchise expanded `franchise_names`) |
| `query_search.branch_top_tier` / `_count` | branch span | string / int | top populated prominence tier + its size (person, both franchises) |
| **event** `entity flow empty` | branch span | — | fired only when `branch_result_count == 0` |

Per-flow attributes + child spans:

| Flow | Attributes | Child spans |
|------|-----------|-------------|
| person | (skeleton) + `branch_top_tier`(`bucket_1_lead`…) | `person.resolve` (flow-neutral, shared with `/attribute_search` via `resolve_person_traced` — see the `person.resolve` subsection) — one per named person (parallel `fetch_person_buckets`) |
| similarity | **Flow-neutral `similarity.*` root (1c-3):** the signals + child spans below are emitted **inside the shared engine** (`run_similar_movies_for_ids`), so they're owned by neither caller (renamed off `query_search.branch_*` / `query_search.similarity_*`). Here they land on the `query_search.branch` span; on the pure `/similarity_search` endpoint the identical set lands on the server span (see that endpoint's subsection). The reference-resolution skeleton (`branch_entities`/`branch_entity_resolved_counts`/`branch_unresolved_entity_count`) stays `query_search.branch_*` — it's specific to resolving NL titles and doesn't apply to the endpoint. `similarity.anchor_count` (int) is the single-vs-multi-anchor discriminator (stands in for the entity skeleton on the endpoint path). Map-shaped signals are **single JSON-string attributes** (OTel drops raw dicts — a JSON string is kept and renders readably; tradeoff: no numeric TraceQL filter on an inner key). **Both flows:** `similarity.retrieval_lanes` (JSON `{lane:count}` — every candidate-fetch query that ran, seed non-empty; fired-but-empty present at 0, gated-off absent) + `similarity.retrieval_total` (int, deduped union); `similarity.lane_weights` (JSON `{lane:weight}`) + `similarity.vector_space_weights` (JSON `{space:weight}`); `similarity.weave_targets` (JSON `{bucket:slots}` — the **desired** allocation `_compute_bucket_targets` set before weaving, best_overall first; a signal bucket absent from the map didn't instantiate. This is the intended reservation, NOT the realized draw — multi-bucket credit lets an instantiated bucket's films enter via best_overall, so a per-slot seat tally reads as "best_overall took everything" for franchise-dominant cohorts; the target answers what the weave meant to reserve); `similarity.low_cohesion_fallback` (bool); `similarity.additional_boosts` (JSON array — currently `["director_signature"]` for the auteur multiplier, which the weights/fetch map can't reveal; **omitted when empty**). **Single-anchor only:** `similarity.shape_modifiers` (JSON array of enacted additive lane-weight-delta types — cult_garbage/prestige/franchise_dominant/source_material; always set, `"[]"` when none) + `similarity.anchor_shape` (str reach×quality bucket, `"none"` when shapeless). **Multi-anchor only:** `similarity.anchor_shape_cohesion` (JSON `{shape:M_s/N}`, dominant first, with a `"none"` key for the shapeless fraction so it sums to 1) + `similarity.lane_cohesion` (JSON `{lane:cohesion}`) + `similarity.vector_space_cohesion` (JSON `{space:cohesion}`). | `similarity.qdrant` — closes the gRPC gap for the flow's **two** Qdrant calls, discriminated by `similarity.qdrant.probe_kind`: `anchor_vectors` (the `retrieve` loading the anchors' stored vectors — `requested_count` / `returned_count`, where returned < requested flags an anchor missing from Qdrant) and `shape` (the batched `query_batch_points` named-vector probe — `space_count` + `spaces` JSON list, `limit_per_space` surfacing the 2× filter over-fetch, `filter_active`, `hit_count` total + `hits_by_space` JSON per-space recall). Leaf vocab (`probe_kind`/`filter_active`/`hit_count`) matches `semantic_qdrant`. `similarity.fetch` — one span per Postgres candidate lane that actually ran (director/franchise/studio/source/quality/themes_recall/rare_medium; qdrant shape probe excluded), carrying `similarity.fetch.lane` (str), `similarity.fetch.match` (JSON of the concrete IDs/bucket the lane queried on — the bound IN-list the SQL span parameterizes away), and `similarity.fetch.result_count` (int). Gated on the same seed predicates as `similarity.retrieval_lanes`, so a span exists iff that lane fired; runs inside the gathered task so it nests under the current span (branch span here, server span on `/similarity_search`) and the lane's own SQL span nests under it |
| exact_title | `branch_exact_title_year` (int, only when Step 0 gave a year); `branch_source.{seed,close,fanout}_count` (int, always-on); `branch_source.title_only_count` (int, only when a year was given) | — |
| studio | `branch_studio_llm_fallback` (bool); `branch_studio_entity_paths` (str[] `brand`/`freeform` per ref); `branch_studio_brand_names` (str[]); `branch_studio_{brand,freeform}_match_count` (int — via the opt-in `path_match_counts` side channel on `execute_studio_query`; brand refs with `brand_match_count==0` = the no-fall-through dead-end) | — |
| non_character_franchise | `branch_franchise_llm_fallback` (bool); `branch_top_tier`=`primary`/`_count`; `branch_secondary_count` (int) | — |
| character_franchise | `branch_character_forms` + `branch_franchise_forms` (str[]); `branch_character_franchise_llm_failed` (bool — the fanout has NO fallback, so this distinguishes LLM-died from catalog-gap); `branch_tier_counts` (int[7]); `branch_top_tier`=`tier_1_lineage_mainline` | `query_search.franchise_resolution` (incl. the folded-in lineage-mainline split) + `query_search.character_resolution` — the two parallel resolution axes |

Supporting return-shape changes: `_weave_candidates` returns `(woven, target)` (the desired per-bucket allocation); `_resolve_similarity_anchors` returns `(anchor_ids, per_ref_resolved)`; `_translate_studio_query` returns `(spec, llm_fallback)`; `_expand_canonical_names` (non-char franchise) returns `(names, llm_fallback)`; `execute_studio_query` / `_execute_any` take an opt-in `path_match_counts` mutable dict (default None → no change for standard-branch callers). Verification vehicle: `search_v2/testing_nonstandard_flows.ipynb`.

Span mechanics: started non-current (`tracer.start_span`) and activated inside
each asyncio task via `use_span(end_on_exit=False)` so the LLM child nests while
the span outlives the call — `step_1_unused` is only known after Step 0 returns.
step_0 closes at its true completion (accurate duration); step_1 closes at
resolution, so its duration = launch → consumed/cancelled (overstates only when
step_1 finished *before* step_0, i.e. when it isn't the long pole). Fatal Step 0
(retries exhausted) marks its span **ERROR** + `record_exception`; a
needed-but-failed Step 1 is a degradation read off the nested `llm.generate`
ERROR status, **not** flipped to `unused`. A `try/finally` closes any span left
open on the client-disconnect path. To make the server span the current parent
for the whole streamed pipeline, `event_stream()` in `api/main.py` wraps the
pipeline loop in `use_span(request_span, end_on_exit=False)` — during SSE
iteration the server span is alive but not guaranteed current (same reason the
cost rollup writes through a captured handle); this also re-parents the existing
`llm.generate` spans under the server span. *(Verified 2026-07-07: step_0 /
step_1 render as overlapping children of the server span with the `llm.generate`
child nested in each; recovered-retry and retry-exhaustion paths exercised via a
forced-failure harness.)*

**Span events:** `request rejected` (with a `detail` attribute) — emitted on the
400 (empty/over-length query or clarification) and 422 (unknown filter enum
value) paths. Expected 4xx rejections get no `record_exception` per the §5.2
error contract, so this event is where the offending field/value lands on the
trace.

**Outcome semantics — failure-only (interim).** The endpoint uses
`@record_outcome(success_on_return=False)` (see §7): failures are recorded, but a
clean handler return writes **no** `request.success=true`. Rationale: the handler
returns a `StreamingResponse` *before* the pipeline runs, so "returned cleanly"
only means the request passed the boundary — stamping success there would read as
true even if the stream later fails fatally. The success verdict (and the
remaining stream-end rollups: total result count, fetch/failed-branch counts) is
deferred to a stream-aware mechanism (`query_search_planning.md` Bite 2) — one
stream-end rollup, `request.cost_usd`, already lands early via the same
generator-`finally` write point (see the Stream-end rollup subsection above).

**Mid-stream fatal failure records a verdict (early Bite 2 slice, verified
2026-07-07).** A fatal Step 0 (LLM retries exhausted) is emitted by the pipeline
as a terminal SSE `error` event *after* HTTP 200, so `@record_outcome` can't see
it. The stream consumer in `event_stream()` watches for that `error` event and
writes `request.success=false` + `query_understanding_failed` (a new
`FailureReason`, OQ #2 — distinct from `internal_error`: upstream LLM exhaustion,
not our bug) on the server span. Only the *failure* verdict lands this way; the
success verdict is still deferred, so a clean stream stays absent. Until the rest
of Bite 2:

| Path | `request.success` | `request.failure_reason` |
|------|-------------------|--------------------------|
| Empty/over-length query or clarification (400) | false | `invalid_parameters` |
| Unknown body field, e.g. typo'd filter key (422, framework) | false | `invalid_parameters` |
| Unknown filter enum value (422) | false | `invalid_filters` |
| Pre-stream crash | false | `internal_error` |
| Fatal Step 0 mid-stream (SSE `error` event) | false | `query_understanding_failed` |
| Passed the boundary, stream completes (200) | *(absent — success deferred to Bite 2)* | — |

**Semantic-endpoint Qdrant probes — `query_search.semantic_qdrant` (Bite 6,
verified 2026-07-07).** The first spans inside Stage 4 execution, and the second
place the gRPC gap is closed (after the similarity flow). One manual span wraps
each of the three `query_points` primitives in
`search_v2/endpoint_fetching/semantic_query_execution.py` — a SEMANTIC call fans
these out across vector spaces in parallel, and one span per primitive renders
that fan-out with **no per-vector-space wrapper spans** (the ~130–150 span/request
budget stays healthy). The single span name is discriminated by `probe_kind`.
Throwing probes auto-mark ERROR + `record_exception` (default span settings) — a
Qdrant failure is a real error at the probe level. Parenting is automatic: the
primitives run under `asyncio.gather`, and asyncio Tasks copy the active OTel
span via contextvars, so probes nest under whatever span is current — the notebook
parent span standalone; today the server/Step span in the live pipeline; the
Bite-5 `_dispatch_call` span once it exists.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `query_search.semantic_qdrant.probe_kind` | string | which primitive fired — `calibration` (`_run_corpus_topn`, unfiltered elbow probe), `pool` (`_run_corpus_topn_filtered`, filtered candidate pool), `hasid_score` (`_run_filtered_score`, HasId reranker). Low-cardinality closed set (`QdrantProbeKind`, owned by the call-site module per rule E) |
| `query_search.semantic_qdrant.vector_space` | string | the named vector queried (`using=vector_name.value`) |
| `query_search.semantic_qdrant.query_params` | string (JSON) | the space body that produced this probe's vector, `model_dump_json(exclude_defaults=True)` — only the populated fields that feed `embedding_text`, so each probe is self-describing. High-cardinality span attr, never a metric label |
| `query_search.semantic_qdrant.limit` | int | the `limit` arg — `CORPUS_PROBE_LIMIT` (2000) on the corpus probes; pool size on the reranker |
| `query_search.semantic_qdrant.filter_active` | bool | whether the USER HARD FILTER was applied. **True only on the `pool` probe**; False on `calibration` (unfiltered by design) and on `hasid_score` (its `HasIdCondition` is a pool restriction, not the hard filter) |
| `query_search.semantic_qdrant.hit_count` | int | `len(response.points)` returned — the diagnostic that explains downstream elbow/pathology behavior (a short calibration probe forces the fallback) |

**Span events (on the ambient span, not the probe span).** `execute_semantic_query`'s
one-retry-then-empty contract previously soft-failed with a log line only; it now
emits `semantic_query_retry` (first-attempt failure) and `semantic_query_failed`
(terminal, before returning the empty `EndpointResult`) as events carrying an
`exception.type`. These land on the current span (dispatch/Step/server span in the
pipeline; notebook span standalone) and no-op when no recording span is active. The
per-branch `branch_error` attribute and the request-level failed-branch count stay
out of scope here — they belong to the Bite-5 branch/dispatch span. Verification
vehicle: `search_v2/test_semantic_qdrant_span.ipynb` (app-free tracing bootstrap +
an editable sample vector space + a Grafana walkthrough).

**Stage 4 execution spans.** All in `search_v2/stage_4_execution.py`. A module
tracer (`trace.get_tracer(__name__)`, a no-op `ProxyTracer` offline) creates each span
with `start_as_current_span`, so they nest under whatever span is current — in the
live pipeline the `query_search.branch` span Stage 4 runs under via `_run_under_span`.
Each standard branch collapses into **six groups** directly under `query_search.branch`:
`step_2` (A), `decomposition` (B), `candidate_generation` (C), `rerankers` (D),
`scoring` (E), `hydration` (F). A/B live in the orchestrator (front half); C–F here.

- **`candidate_generation` (C)** is the Phase-B wrapper span (`_define_candidate_pool`):
  the `generators` / `promotion` / `neutral_seed` / `auxiliary_shorts_exclusion` spans
  all nest under it. It records `fetch_count` (dispatch calls issued, post-dedup),
  `candidate_count` (deduped union size), and `cost_usd` (semantic-generator embeddings).
- **`rerankers` (D)** now covers **positive AND negative** reranker dispatch, run in
  parallel against the finalized union (one shared span; there is no separate
  `negatives` span). Polarity is recorded per dispatch via `dispatch.polarity`.
- **`scoring` (E)** consumes the already-dispatched reranker score maps (positive combine
  + negative gate×fuzzy fold + aggregation); the `implicit_prior_rerank` span now nests
  **under scoring** (applied via `apply_implicit_prior_rerank_for_branch` from
  `search_v2/implicit_prior_rerank.py`, not the orchestrator).

| Span | Attribute | Type | Meaning |
|------|-----------|------|---------|
| `query_search.candidate_generation` | `candidate_generation.fetch_count` | int | dispatch calls actually issued this phase (post-dedup generators + promotion re-dispatches + shorts + any neutral-seed fetch) |
| `query_search.candidate_generation` | `candidate_generation.candidate_count` | int | deduped union size at phase end (0 ⇒ branch returns empty) |
| `query_search.candidate_generation` | `candidate_generation.cost_usd` | float | semantic-generator embedding cost for this phase (per-stage nested accumulator) |
| `query_search.generators` | `generators.raw_union_count` | int | deduped candidate IDs after the initial parallel generator dispatch, **pre-shorts** |
| `query_search.generators` | `generators.shorts_removed_count` | int | how many the default shorts blocklist removed |
| `query_search.generators` | `generators.final_pool_count` | int | pool remaining post-shorts (the initial generator phase's final size; the promotion loop may grow it further) |
| `query_search.promotion` | `promotion.tier` | string | the promoted `PromotionTier.name` this round (tiered loop, filter-active) |
| `query_search.promotion` | `promotion.pool_count_before` / `_after` | int | union size entering / leaving the round |
| `query_search.promotion` | `promotion.promoted_spec_count` | int | specs flipped to generator this round |
| `query_search.promotion` | `promotion.shorts_removed_count` | int | shorts removed after this round's dispatch |
| `query_search.neutral_seed` | `neutral_seed.reason` | string | `no_candidate_generators` (aux-spec arm) / `under_floor_exhausted` (filter-active direct arm) — `NeutralSeedReason` |
| `query_search.neutral_seed` | `neutral_seed.seed_count` | int | seed candidates fetched; **0 = fetch failed ⇒ branch returns empty** |
| `query_search.rerankers` | `rerankers.call_count` / `rerankers.pool_count` | int | total reranker dispatches (positive specs + negative-trait endpoint calls) / union size scored against |
| `query_search.rerankers` | `rerankers.cost_usd` | float | semantic-reranker embedding cost for this phase (per-stage nested accumulator) |
| `query_search.dispatch` | `dispatch.route` | string | `EndpointRoute.value` |
| `query_search.dispatch` | `dispatch.operation_type` | string | `OperationType.value` |
| `query_search.dispatch` | `dispatch.polarity` | string | `Polarity.value` — `positive` / `negative`. Distinguishes positive vs negative reranker dispatch (both `POOL_RERANKER`) now that they share the `rerankers` span; defaults `positive` on generators/promotions/shorts |
| `query_search.dispatch` | `dispatch.was_promoted` | bool | true if this spec was promoted (either fallback path) |
| `query_search.dispatch` | `dispatch.result_count` | int | candidates this call returned (success path) |
| `query_search.dispatch` | `dispatch.query_params` | string (JSON) | the committed query the executor runs — `spec.params` dumped (`exclude_none=True`) with the LLM analysis/reasoning scaffolding stripped (`thinking` / `*_exploration` / `*_candidates` / `*_intent` / `*_reasoning` / `request_overview` / `search_picture` / keyword `attributes` walk). Shows *what* was queried without generation-assist noise or the SQL child spans' movie-ID bloat (the `restrict` set is a separate arg, never in `params`). Omitted when there are no params (TRENDING) or nothing query-relevant survives. High-cardinality span attr, never a metric label |
| `query_search.dispatch` | `error.type` | string | on soft-fail: `timeout` (the `asyncio.wait_for` ceiling) or the exception class name; span marked ERROR + `record_exception` (standard OTel key, emitted raw) |

`query_search.dispatch` is the per-unique-call child span, created inside `_dispatch_call`
so it nests under whichever container gather is current (`generators` / `promotion` /
`rerankers` — all themselves under `candidate_generation` except `rerankers`); the
semantic Qdrant probes + embedding span nest one level deeper under it. Dedup folds structurally-identical `(route, params)` calls onto one
dispatch, so a dispatch span can serve multiple trait/category coordinates — **per-call
trait attribution is deliberately omitted** for that reason. The default shorts-exclusion
fetch runs through the same `_dispatch_call` but overrides the span name to
`query_search.auxiliary_shorts_exclusion` (via a `span_name` param) so it reads as the
auxiliary exclusion rather than an anonymous `dispatch`; it carries the same `dispatch.*`
attributes (route = media_type, result_count = shorts fetched, timeout handling) and — run
before the `generators` span opens — parents directly to the branch span.

**Span events (Stage 4 execution).**
- `generator_dedup` — on the current container span when the within-batch/cross-iteration fold collapsed ≥1 call; attr `deduped_routes` (list of `route.value`). Rare; emitted independent of log level.
- `aux_shorts_exclusion` — on the `generators` span; the default shorts blocklist (distinct from a user negative MEDIA_TYPE trait). Attrs `formats` (always `["short"]`), `shorts_pool_count`, `removed_count`, `remaining_count`.
- `reranker_fallback_promotion` — one event per promotion. Attrs `reason` (`PromotionReason`: `under_candidate_floor` on each `promotion` round span, or `no_candidate_generators` on the **branch span**, emitted from the orchestrator for the pre-Stage-4 single-shot case) + `tier`; the under-floor form also carries `count_so_far` (pool size at the promote decision).
- `thin_pool_accepted` — on the branch span when a filter-active branch exhausts every promotable tier but the union is non-empty and still `< CANDIDATE_FLOOR`; attr `final_count`. Mutually exclusive with the `neutral_seed` span (which fires only on `union == 0`).
- `dispatch_soft_fail` — on the `dispatch` span for **every** dropped call (not just timeouts); attrs `error.type` + `timeout` (bool). The old silent 25s-timeout soft-fail is now queryable.

**Supporting side channel.** `_apply_reranker_only_candidate_fallback` /
`_compute_branch_auxiliary` (`full_pipeline_orchestrator.py`) gained an opt-in
`fallback_outcome: dict | None` param recording the promoted `tier` name when the
no-generator fallback promotes; `streaming_orchestrator.py::_handle_finished_task`
passes a dict and stamps the branch-span `reranker_fallback_promotion` event. The tier
can't be re-derived post-promotion (`determine_promotion_tier` → `NEVER_PROMOTE` for a
`CANDIDATE_GENERATOR`). The two `PromotionReason` / `NeutralSeedReason` value enums live
in `search_v2/promotion_tiers.py` (OTel-free, imported by both emitters; rule E).

### `/rerun_query_search` (POST) — 1c-2
Replays a prior search's branch set against a fresh filter set, bypassing Steps
0/1 by calling `stream_rerun_pipeline` → the shared `_stream_from_branch_plan`.
Because it reuses that shared downstream half, it **inherits every
`query_search.*` pipeline span for free** — `query_search.branch` (+ entity-flow
attrs), `step_2` / `decomposition` / trait / `step_3` / `query_generation`, the
full six-group Stage-4 tree, the `query_search.semantic_qdrant` probes, and the
implicit-prior spans — plus the `llm.generate` router children. **The one thing
that made this real:** the handler's `event_stream()` wraps the loop in
`trace.use_span(request_span, end_on_exit=False)` (identical to `/query_search`).
Without it those reused spans attach to the ambient context during SSE iteration,
not the rerun server span — i.e. they orphan. There are **no routing spans** (Steps
0/1 are bypassed) and no `error` SSE event / `query_understanding_failed` verdict
(nothing fatal upstream of the branch plan).

Server-span facts written by the handler:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `rerun_query_search.branch_count` | int | total fetches replayed. Low-cardinality, label-eligible |
| `rerun_query_search.branch_types` | string[] | the `type` tag of each replayed branch (`standard` / `exact_title` / `similarity` / …) — the rerun analog of `step_0_flows` ("what shape was replayed"). Closed set |
| `rerun_query_search.standard_queries` | string[] | the up-to-3 standard branch queries (defensively truncated); set only when ≥1 standard branch present. High-cardinality span attr, never a metric label. Entity anchor names are NOT duplicated here — they land on the entity branch spans |
| `filters.*` | (see `/query_search`) | raw hard-filter inputs via the shared `_record_filter_attributes` helper |
| `request.cost_usd` / `request.usage.*` / `request.result_count` | — | cross-endpoint stream-end rollups (see §"Stream-end rollup") |
| `query_search.succeeded_branch_count` / `.failed_branch_count` | int | branch-plan-owned counts, **reused** here (rule B) |
| `request.success` / `request.failure_reason` | — | success (≥1 branch executed clean) or `all_branches_failed`, written from the generator `finally`; boundary rejections write `invalid_parameters` / `invalid_filters` via `@record_outcome(success_on_return=False)` |

All input attrs are captured at handler entry (before `_to_rerun_plan`), so a
rejected trace still carries its input. The boundary helpers raise
`EndpointFailure` (via `_rerun_rejection`, `invalid_parameters`) with a `request
rejected` span event, so a malformed branch/name/duplicate/over-cap request is
classified on the server span rather than logged as `internal_error`.

### `/similarity_search` (POST) — 1c-3
Ranked "similar to" from a caller-supplied TMDB-ID anchor set — no NLP/LLM. It
reuses the same engine (`run_similar_movies_for_ids`) as `/query_search`'s
similarity branch, so it **inherits every `similarity.*` fact for free**: the
Qdrant/fetch child spans (`similarity.qdrant` / `similarity.fetch`, the gRPC gap
closer — §3) and the full signal set + `similarity.anchor_count`, all recorded by
the engine onto the current span (here the **FastAPI server span**, since the
handler wraps nothing). See the similarity row in the entity-flow table above for
the exhaustive `similarity.*` attribute list; the reference-resolution skeleton is
absent here (anchors are supplied, not resolved from NL titles).

The endpoint adds only what the handler alone knows, on the server span:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `similarity_search.cache_hit` | bool | Redis result-cache disposition — `true` on a warm return (before any engine work), `false` at the cold-path success point. Set only at a success point (mirrors `movie.payload_source`), so a failed request carries neither and cache-hit rate stays honest. Low-cardinality, label-eligible |
| `request.result_count` | int | post-hydration `MovieCard` count returned — the cross-endpoint rollup (see §"Stream-end rollup"). Not set on the warm cache-hit return (no hydration happens) |
| `filters.*` | (see `/query_search`) | the raw hard-filter inputs, via the shared `_record_filter_attributes` helper (one typed attr per sent field + always-on `filters.active_count`) — same convention as `/query_search`'s boundary capture |

**Outcome semantics.** Plain `@record_outcome` (non-streaming — the handler does all
its work before returning, so `success_on_return=True` is correct; unlike
`/query_search`'s streaming form). The two boundary error paths raise
`EndpointFailure` so the verdict is recorded: unknown/absent anchor id (`LookupError`
→ 422) and empty anchor set (`ValueError` → 400, unreachable behind the pydantic
`min_length=1` guard) both map to `invalid_parameters`; an unknown filter enum value
already raises `EndpointFailure(422, invalid_filters)` inside `_to_metadata_filters`.

**Span events.** `cache read failed` / `cache write failed` — the two swallowed
best-effort Redis degradations (§5.2): a span event, not a span error, and the
request still succeeds (the read falls through to the cold path; the write loss
doesn't lose the already-built response).

### `/attribute_search` (POST) — 1c-4
Deterministic hard-attribute browse — no NLP/LLM/vector, pure Postgres +
in-memory ranking, so every DB call is auto-traced (§3) and the manual work is
the outcome verdict, the request-input "traits," a per-person grouping span, and
the path/count skeleton. Two paths: **browse** (no people → one
`fetch_neutral_reranker_seed_ids` query) and **people** (per name resolution →
union/sum → rank). Like `/similarity_search`, the per-person span is
**flow-neutral and shared** with the `/query_search` person branch (the
`person.resolve` subsection below), not written twice.

**Endpoint-owned attributes on the server span** (`attribute_search.*`, flat
leaves per the `branch_*` grouped-facts precedent). The `path` + people/pool
skeleton is stamped by the orchestrator `run_attribute_search` on the current
span (the server span at that call depth); the input `filters.*` / people attrs +
`result_count` are stamped by the handler:

| Attribute | Type | Meaning |
|-----------|------|---------|
| `attribute_search.path` | string | `browse` \| `people` (`AttributeSearchPath`) — which ranking path ran; the single low-cardinality slice discriminator |
| `attribute_search.people_requested_count` | int | named people received (post blank-strip at the boundary). **Always set** (0 on browse); low-cardinality |
| `attribute_search.people_names` | string[] | the raw supplied names, per-element truncated at `_INPUT_ATTR_MAX_CHARS`; set only when people were sent. High-cardinality span attr, never a metric label |
| `attribute_search.people_searched_count` | int | names actually resolved after normalize + dedupe (== number of `person.resolve` spans). `requested − searched` = blank/duplicate names dropped. People path only |
| `attribute_search.people_unresolved_count` | int | searched names that resolved to **zero** credits — the silent-drop signal in aggregate (alertable; also flagged per-name by the `"person unresolved"` event). `searched − unresolved` = names that fed the pool. People path only |
| `attribute_search.pool_count` | int | union pool size before hydration; `0` here with `searched_count > 0` is the **empty-pool** case (resolved names, nothing survived the union/filter) — distinct from the all-blank return where `searched_count == 0`. People path only |
| `request.result_count` | int | post-hydration `MovieCard` count returned (both paths, incl. 0) — the cross-endpoint rollup (see §"Stream-end rollup") |
| `filters.*` | (see `/query_search`) | raw hard-filter inputs via the shared `_record_filter_attributes` helper |

**Outcome semantics.** Plain `@record_outcome` (non-streaming). The only boundary
failure path is an unknown filter enum value, which already raises
`EndpointFailure(422, invalid_filters)` inside `_to_metadata_filters`; a clean
return records `success=true`. Unresolvable person names are **not** failures —
they degrade silently (an empty or partial result on a `success=true` trace),
which is exactly why the `people_unresolved_count` attr + `"person unresolved"`
event exist.

### Shared `person.resolve` span — cross-endpoint (1c-4)
Not an endpoint: one span emitted by
`search_v2/person_search.py::resolve_person_traced` — the single instrumented
wrapper around the pure resolver `fetch_person_buckets` — so both person callers
(`/query_search`'s person branch via `run_person_search`, and `/attribute_search`
via `run_attribute_search`) emit **identical** per-person telemetry from one
definition. It hangs off a flow-neutral `person` root (rule B — the domain that
owns the work, not either caller), mirroring the `similarity.*` split.
`fetch_person_buckets` stays tracer-free; the span nests under whatever span is
current (the `query_search.branch` span on the branch path, the server span on
the endpoint path) and its 6 psycopg calls (phrase-term lookup + 5 role tables)
nest under it. Carries only the **intrinsic** facts the resolution knows about
itself; caller-specific context (branch identity, request path, union pool) stays
on the caller's span.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `person.resolve.name` | string | the name resolved. High-cardinality span attr, never a metric label |
| `person.resolve.movie_count` | int | movies this person resolved to; `0` = no filter-eligible credit (the silent-drop miss) |
| `person.resolve.best_bucket` | int | most-prominent bucket reached (1=lead … 4=minor); **omitted when `movie_count == 0`**. An all-`4` result for a well-known name is the fingerprint of a resolution collision |

**Span event.** `"person unresolved"` — emitted on the span when `movie_count`
is 0, so the zero-credit miss is queryable without scanning every `movie_count`
attribute. Not a span error and never touches the request verdict — a silent drop is a
degradation, not a failure (§5.2).

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
| `llm.cost_usd` | float | computed dollar cost via `implementation/llms/pricing.py::compute_llm_cost_usd` (cache-adjusted: cached input priced at the discounted rate); **omitted** (with a logged warning) when the model is unpriced — never a fabricated `$0`. Reflects the successful attempt; the request-level superset is `request.cost_usd` |
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
(always-on-error). Full payloads ride span **events** (not span attributes) so
the sampling dial can drop them.

**⚠️ Known limitation — payloads are truncated in Tempo (fix deferred to Phase
4).** Riding on span events does NOT exempt the payload from size limits: Tempo's
distributor truncates every attribute key/value — **including span-event
attributes** — to `max_attribute_bytes` (default **2048 bytes**) before storage,
so `system_prompt` / `user_prompt` / `response` are silently cut off at ~2 KB in
Grafana. This is a backend default (not our SDK — we set no `SpanLimits`, and the
OTel value-length limit defaults to unlimited), observable via
`tempo_distributor_attributes_truncated_total{scope="event"}`. **Decision
(2026-07-08):** the durable fix is to move the payload OFF the span event onto an
OTel **log record** correlated by trace/span id (Loki, 256 KB line limit),
reversing the "large payloads on span events" tactic in
`observability_logs_plan.md`. Scheduled as a Phase-4 item (see
`observability_todos.md` Phase 4). Interim dev workaround: raise Tempo's
`max_attribute_bytes` in the local `otel-lgtm` config.

### Embedding router span — `embedding.generate` — cross-cutting
Not an endpoint: one span emitted by
`implementation/llms/generic_methods.py::generate_vector_embedding` — the single
codepath every embedding call passes through (semantic-endpoint retrieval AND
offline ingestion) — so one instrumentation point covers every embedding on every
endpoint, the exact parallel to `llm.generate` for the LLM router. It nests under
whatever pipeline span is active (the semantic Qdrant probes' ambient span / the
server span today); offline, `_tracer` is a no-op `ProxyTracer`, so ingestion pays
no span cost.

Simpler than `llm.generate` in one way: the embedding client runs `max_retries=0`
(retries are the caller's concern), so the span wraps a **single attempt** — no
`attempt_count`, no `llm.retry` events. Uses
`record_exception=False, set_status_on_exception=False`; a failure sets
`error.type`, marks the span **ERROR**, and records the exception before the call
re-raises as `ValueError` (§5.2 error contract).

Standard facts ride the OTel GenAI keys (not authored in `names.py`); embeddings
produce no output tokens and have no prompt caching, so
`gen_ai.usage.output_tokens` / `.cache_read.input_tokens` are deliberately **not
emitted** (an always-0 key is noise). The two project-owned facts are
`embedding.cost_usd` and `embedding.input_count`.

| Attribute | Type | Meaning |
|-----------|------|---------|
| `gen_ai.system` | string | always `openai` (the only embedding provider). Standard OTel GenAI key |
| `gen_ai.request.model` | string | requested model (`text-embedding-3-large`). Standard OTel GenAI key |
| `gen_ai.operation.name` | string | always `embeddings` — disambiguates this span from chat generation within the `gen_ai.*` namespace. Standard OTel GenAI key |
| `gen_ai.usage.input_tokens` | int | `usage.total_tokens` for the batch (pure input — embeddings have no output/cache tokens). Standard OTel GenAI key; set only when the response carries `usage` |
| `embedding.cost_usd` | float | computed dollar cost via `compute_llm_cost_usd(model, total_tokens, 0)`; **omitted** (with a logged warning) when the model is unpriced — never a fabricated `$0`. The same cost value feeds the request-level `request.cost_usd` rollup |
| `embedding.input_count` | int | number of texts in the batch (a single call embeds up to 2048 inputs), so cost/tokens-per-input is derivable. Always set; low-cardinality, metric-label-eligible |
| `error.type` | string | set **only on failure**: normalized failure class (`timeout` for the asyncio ceiling, else the exception class name). Standard OTel key |

The per-request rollup is unchanged: the same computed cost + `total_tokens` still
feed `add_request_cost` / `add_request_tokens` (→ `request.cost_usd` +
`request.usage.input_tokens`), no-ops outside a tracked request. An ERROR
`embedding.generate` span does not flip the request verdict — a failed embedding is
a degradation surfaced on the branch, per §5.2.

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
| `FailureReason(str, Enum)` | `api/outcome.py` | enum | the coarse `request.failure_reason` vocabulary (replaces the old `NotFoundReason`, folding its two 404 reasons into the broader set). Members: `invalid_parameters`, `invalid_filters` (unknown hard-filter enum value — UI/server taxonomy drift, distinct from `invalid_parameters`), `not_indexed`, `tmdb_removed`, `tmdb_fetch_failed`, `internal_error` |
| `EndpointFailure(HTTPException)` | `api/outcome.py` | exception | request-terminating failure that carries its `FailureReason`; raised at each known failure site so the reason bubbles up |
| `record_outcome` | `api/outcome.py` | decorator | wraps each endpoint; writes `request.success`/`request.failure_reason` on the server span exactly once (the single write point). Dual-form: bare `@record_outcome` (default, `success_on_return=True`) stamps `success=true` on clean return; `@record_outcome(success_on_return=False)` records failures only and leaves success absent on clean return — for streaming (SSE) endpoints whose handler returns before the pipeline runs (`/query_search`, `/rerun_query_search`), which write the success verdict + stream-end rollups from the generator's `finally` instead |
| `Name` + constants | `observability/names.py` | name registry | span names + attribute keys, each derived once from a namespace root via `.child()`; imported by `api/main.py`. Replaces the former inline `_ATTR_*` / literal-string approach |
| `_fetch_movie_payload(...)` | `api/main.py` | helper | opens the `*.payload_creation` span, existence-gate + TMDB fetch; raises `EndpointFailure` (404 `not_indexed`/`tmdb_removed`, 502 `tmdb_fetch_failed`) and owns the child-span error marking; returns `(card_row, tmdb_payload)` |
| `_encode_and_cache_credits(...)` | `api/main.py` | helper | opens `movie_credits.build_and_cache`, builds/encodes/caches credits; shared by both endpoints |
| `_tracer` | `implementation/llms/generic_methods.py` | module tracer | `trace.get_tracer(__name__)`; a no-op `ProxyTracer` when `setup_tracing` hasn't run (offline ingestion/eval imports), so the `llm.generate` span is a cheap no-op there |
| `generate_llm_response_async` | `implementation/llms/generic_methods.py` | router + LLM instrumentation point | the single codepath every routed LLM call passes through; owns the `llm.generate` span, `gen_ai.*`/`llm.*` attrs, retry/payload events, and the three-state failure marking |
| `generate_vector_embedding` | `implementation/llms/generic_methods.py` | embedding instrumentation point | the single codepath every embedding call passes through (search + ingestion); owns the `embedding.generate` span, `gen_ai.*` (system/model/operation/input_tokens) + `embedding.cost_usd` / `embedding.input_count` attrs, and single-attempt ERROR marking. Also feeds the per-request cost/token rollup (unchanged) |
| `compute_llm_cost_usd` | `implementation/llms/pricing.py` | cost util | canonical `(input, cached_input, output)`-per-million pricing table → cache-adjusted USD cost; returns `None` for unpriced models (caller omits `llm.cost_usd` + warns). A parallel table in `estimate_generation_cost.py` should later import from here (see `docs/TODO.md`) |
| `track_request_cost` / `track_stage_cost` / `add_request_cost` / `add_request_tokens` | `observability/cost_tracking.py` | cost + token rollup | `ContextVar`-scoped **stack** of `RequestCostAccumulator`s. `track_request_cost` pushes the request root (→ server-span `request.cost_usd` + `request.usage.*`); `track_stage_cost` pushes a per-stage child so a pipeline stage (A step_2 / B decomposition / C candidate_generation / D rerankers) captures only its own subtree's spend for its `*.cost_usd` attr. `add_*` fan out to **every** accumulator on the stack, so the root stays complete while children isolate — and because `create_task`/`gather` snapshot context at creation, concurrent branches' stages don't bleed. No-op outside a tracked request (empty stack). Tokens accumulate unconditionally; cost skips unpriced models |

---

## 8. What is deliberately NOT instrumented yet

So downstream docs don't overclaim coverage:

- `/rerun_query_search` **is now covered (1c-2)** — see its subsection above. It
  replays the shared branch plan, so it inherits every `query_search.*` pipeline
  span (branch / step_2 / Stage 4 / semantic Qdrant) and the `llm.generate`
  children, kept nested under its server span by the `trace.use_span` wrapper
  (without which they orphaned); it also writes the cross-endpoint `request.*`
  rollups + verdict, reuses the branch-plan `query_search.{succeeded,failed}_branch_count`,
  and stamps its `rerun_query_search.*` input attrs. No routing spans (Steps 0/1
  are bypassed by construction).
  (`/query_search` is now broadly covered — Steps 0/1/2/3, per-branch +
  entity-flow attrs, the Stage-4 semantic Qdrant probes, and the full six-group
  Stage-4 branch tree (`candidate_generation` / `rerankers` / `scoring` /
  `hydration`) — but still lacks the terminal success verdict/rollups.
  `/similarity_search` is now covered (1c-3): it inherits the shared engine's
  flow-neutral `similarity.*` signal set + `similarity.qdrant`/`.fetch` child spans
  on its server span, plus endpoint-owned `similarity_search.cache_hit`, the
  cross-endpoint `request.result_count`, `filters.*`, and the `@record_outcome`
  verdict.
  `/attribute_search` is now covered (1c-4): pure Postgres, so the manual work is
  the shared `person.resolve` span per name, the `attribute_search.*` path/people/
  pool skeleton + cross-endpoint `request.result_count`, `filters.*`, and the
  `@record_outcome` verdict — no Qdrant call sites exist to cover. `/health`
  intentionally excluded entirely.)
- **No metrics** (RED per endpoint, USE for the box) — Phase 3.
- **No structured/JSON logs and no trace↔log correlation** — Phase 4.
- ✅ **LLM token/cost/`gen_ai.*` attributes now land** on the shared
  `llm.generate` span (1c-1 Bite 1) — including `gen_ai.usage.cache_read.input_tokens`
  and a cache-adjusted `llm.cost_usd` — see §6. ✅ **Per-request cost + token
  usage also roll up** to `request.cost_usd` and `request.usage.*`
  (input/cached-input/output tokens) on the server span. What's still missing is
  the *pipeline* spans that would parent the LLM spans (Bites 2–9).
- ✅ **Embedding calls now carry a span.** `generate_vector_embedding` emits a
  cross-cutting `embedding.generate` span (the exact parallel to `llm.generate`)
  carrying `gen_ai.*` (system/model/operation/input_tokens) + `embedding.cost_usd`
  / `embedding.input_count`, in addition to feeding the `request.cost_usd` /
  `request.usage.input_tokens` rollups it already fed — see §6. One
  instrumentation point covers every embedding on search and offline ingestion;
  offline the span is a no-op (`_tracer` is a `ProxyTracer`).
- **Framework-level 422s now carry the request verdict** (app-wide). A missing or
  mis-typed query/path param (e.g. no `q`, a non-int `tmdb_id`) or an unknown
  body field is rejected by FastAPI's validation *before* the handler runs, so
  `@record_outcome` never sees it. An app-level `RequestValidationError` handler
  (`_on_request_validation_error` in `api/main.py`) closes this gap: it stamps
  `request.success=false` + `invalid_parameters` and adds a `request rejected`
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
