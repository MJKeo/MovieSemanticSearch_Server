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

**Last updated:** 2026-07-07 · **Phase:** 1 (traces) — partially complete.
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
| Manual attrs: `/query_search` — request-boundary input capture + outcome (boundary failures + mid-stream fatal Step 0) + stream-end `query_search.cost_usd` rollup | 🟡 partial (1c-1 Stage 0 + cost rollup + Step-0 fatal verdict) |
| Manual spans: `/query_search` pipeline — Steps 0 + 1 (`query_search.step_0` / `.step_1`) | ✅ implemented + verified (1c-1 Bite 3) |
| Manual spans: `/query_search` pipeline — per-branch (`query_search.branch`, `branch_type` + `branch_uses_original_text`); also emitted on `/rerun_query_search` | 🟡 spans + attrs landed (1c-1 Bite 4); `branch_error`/result-count attrs pending |
| Manual spans: `/query_search` trait pipeline — Step 2 / trait / Step 3 / query generation (`query_search.step_2` / `.trait` / `.step_3` / `.query_generation`, incl. `query_generation_endpoints` + `"solo trim"` event) | ✅ implemented + verified (1c-1 Bite 4) |
| Manual spans/attrs: `/query_search` entity flows (person / similarity / exact_title / studio / franchises) — `branch_*` attributes + per-flow child spans (person resolution, char-franchise resolutions, similarity Qdrant + per-lane candidate fetches) | 🟡 code landed, awaiting Tempo verification (1c-1 Bite 8) |
| Manual spans: `/query_search` Stage 4 semantic Qdrant probes (`query_search.semantic_qdrant` × 3 primitives + retry/failed events) | ✅ implemented + verified (1c-1 Bite 6; nests under ambient span until Bite 5's dispatch span lands) |
| Manual spans: `/query_search` pipeline (Stage 4 dispatch/scoring, terminal), `/rerun_query_search`, `/similarity_search`, `/attribute_search` | ❌ not started (1c-1 Bites 2 remainder, 5, 7, 9, 1c-2…1c-4) |
| LLM router span (`llm.generate`) + `gen_ai.*` (incl. `usage.cache_read.input_tokens`) + cache-adjusted cost/prompt-hash/attempt-count + retry/payload events; per-request cost rollup (`query_search.cost_usd`) | ✅ implemented + verified (1c-1 Bite 1 + cost rollup; retry/recovery + exhaustion paths exercised 2026-07-07) |
| Embedding router span (`embedding.generate`) + `gen_ai.*` (system/model/operation/input_tokens) + `embedding.cost_usd` / `embedding.input_count`; single-attempt error contract | ✅ implemented + verified (shared `generate_vector_embedding`; covers search + offline ingestion; success + error paths + accumulator reconciliation exercised 2026-07-07) |
| Metrics (RED/USE) | ❌ not started (Phase 3) |
| Structured logs + trace correlation | ❌ not started (Phase 4) |
| Production export (Grafana Cloud on EC2) | ❌ not started (Phase 5) |

**One-line summary:** every inbound request already produces a trace with
auto-instrumented network spans; three of the eight endpoints additionally carry
hand-written pipeline spans + semantic attributes, `/query_search` carries
request-boundary (Stage 0) input attributes, and **every routed LLM call now
carries a manual `llm.generate` span** (tokens/cost/prompt-hash/attempt-count +
retry/payload events) via the shared router — with the matching
**`embedding.generate` span** on every embedding call (search + offline ingestion),
carrying `gen_ai.*` + `embedding.cost_usd` / `embedding.input_count`.
`/query_search` now has its first
pipeline spans — `query_search.step_0` / `.step_1` (flow routing + spin
generation) parenting their `llm.generate` children under the server span — plus a
stream-end `query_search.cost_usd` rollup and a mid-stream fatal-Step-0 outcome
verdict on the server span; the per-branch spans (Bite 4), the standard-branch
**trait pipeline** (`query_search.step_2` / `.trait` / `.step_3` /
`.query_generation`, Bite 4), the entity-flow `branch_*` attributes (Bite 8), and
the Stage-4 semantic Qdrant probe spans (`query_search.semantic_qdrant`, Bite 6)
have also landed. Still pending: the Stage-4 dispatch/scoring spans and the success
verdict/rollups (Bites 2 remainder, 5, 7, 9). Nothing but traces exists yet (no
metrics, no log correlation), and telemetry is local-only.

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
hand-written spans. On `/query_search` the similarity flow's **both** Qdrant calls
are now covered by `query_search.similarity_qdrant` (Bite 8) — the `anchor_vectors`
retrieve and the `shape` batched probe, split by `probe_kind` (see §6) — alongside
the Stage-4 semantic probes `query_search.semantic_qdrant` (Bite 6, three
primitives). The remaining Qdrant calls (`/similarity_search`, `/attribute_search`,
and any other `/query_search` vector paths) are still uninstrumented (1c-2…1c-4).
Not a bug — a known, documented omission being retired flow by flow.

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

### `/query_search` (POST) — 1c-1 **request boundary + stream-end cost rollup**
Partial coverage: the **request boundary** (input validation + filter
translation) is instrumented, and a **stream-end cost rollup**
(`query_search.cost_usd`) is written when the stream completes. The streaming
pipeline *between* them is now partially covered — Steps 0/1 spans (Bite 3),
per-branch spans + entity-flow attributes (Bites 4/8), and the Stage-4 semantic
Qdrant probe spans (`query_search.semantic_qdrant`, Bite 6) have landed — but
Steps 2/3, the Stage-4 dispatch/scoring spans, and the terminal rollups are still
pending (Bites 2 remainder, 5, 7, 9 in `query_search_planning.md`). No manual child spans exist here (validation +
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

**Stream-end rollup — request cost + token usage.** Four rollup attributes are
written on the server span at stream end (the first non-input `query_search.*`
facts):

| Attribute | Type | Meaning |
|-----------|------|---------|
| `query_search.cost_usd` | float | total USD cost of the request = every LLM call's cost + every embedding call's cost, summed across **all billed attempts** (a retried/failed-but-billed attempt still counts) |
| `query_search.usage.input_tokens` | int | total input tokens across every LLM + embedding call (all billed attempts). Cache-inclusive — cached tokens are a subset, per the next row |
| `query_search.usage.cached_input_tokens` | int | input tokens served from the provider prompt cache — a **subset** of `input_tokens`, never additive (same `cached ⊆ input` convention as the per-call `gen_ai.usage.cache_read.input_tokens`) |
| `query_search.usage.output_tokens` | int | total output/generation tokens across every LLM call (all billed attempts; embeddings contribute 0) |

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
model's real tokens still count toward `query_search.usage.*`.

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
`outcome.*` — per-call/per-trait failures are degradations (§5.2 error contract),
and the underlying `llm.generate` child already carries the ERROR status.

**Deliberately still out of scope (per the Bite-4 scope cut).** The
`query_generation` span exists only for handler-LLM calls: the `EXPLICIT_NO_OP` and
`NO_LLM_PURE_CODE` buckets return before the span, so deterministic/no-op calls get
no span. **Stage 4 query execution is not instrumented here** — its candidate pool
is a branch-level union (not per-trait), so execution can't nest under a `trait`
span; it becomes a separate branch-level span in a later bite, correlated back to a
trait by attribute rather than nesting. The `implicit_expectations` LLM call (runs
concurrently with the trait fan-out) also has no manual span yet.

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
not set the span to ERROR or touch `outcome.*` (span status stays UNSET); a
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
| person | (skeleton) + `branch_top_tier`(`bucket_1_lead`…) | `query_search.person_resolution` — one per named person (parallel `fetch_person_buckets`) |
| similarity | Redesigned around four reader questions; map-shaped signals are **single JSON-string attributes** (OTel drops raw dicts — a JSON string is kept and renders readably; tradeoff: no numeric TraceQL filter on an inner key). **Both flows:** `branch_retrieval_lanes` (JSON `{lane:count}` — every candidate-fetch query that ran, seed non-empty; fired-but-empty present at 0, gated-off absent) + `branch_retrieval_total` (int, deduped union); `branch_lane_weights` (JSON `{lane:weight}`) + `branch_vector_space_weights` (JSON `{space:weight}`); `branch_weave_targets` (JSON `{bucket:slots}` — the **desired** allocation `_compute_bucket_targets` set before weaving, best_overall first; a signal bucket absent from the map didn't instantiate. This is the intended reservation, NOT the realized draw — multi-bucket credit lets an instantiated bucket's films enter via best_overall, so a per-slot seat tally reads as "best_overall took everything" for franchise-dominant cohorts; the target answers what the weave meant to reserve); `branch_low_cohesion_fallback` (bool); `branch_additional_boosts` (JSON array — currently `["director_signature"]` for the auteur multiplier, which the weights/fetch map can't reveal; **omitted when empty**). **Single-anchor only:** `branch_shape_modifiers` (JSON array of enacted additive lane-weight-delta types — cult_garbage/prestige/franchise_dominant/source_material; always set, `"[]"` when none) + `branch_anchor_shape` (str reach×quality bucket, `"none"` when shapeless). **Multi-anchor only:** `branch_anchor_shape_cohesion` (JSON `{shape:M_s/N}`, dominant first, with a `"none"` key for the shapeless fraction so it sums to 1) + `branch_lane_cohesion` (JSON `{lane:cohesion}`) + `branch_vector_space_cohesion` (JSON `{space:cohesion}`). | `query_search.similarity_qdrant` — closes the gRPC gap for the flow's **two** Qdrant calls, discriminated by `similarity_qdrant.probe_kind`: `anchor_vectors` (the `retrieve` loading the anchors' stored vectors — `requested_count` / `returned_count`, where returned < requested flags an anchor missing from Qdrant) and `shape` (the batched `query_batch_points` named-vector probe — `space_count` + `spaces` JSON list, `limit_per_space` surfacing the 2× filter over-fetch, `filter_active`, `hit_count` total + `hits_by_space` JSON per-space recall). Leaf vocab (`probe_kind`/`filter_active`/`hit_count`) matches `semantic_qdrant`. `query_search.similarity_fetch` — one span per Postgres candidate lane that actually ran (director/franchise/studio/source/quality/themes_recall/rare_medium; qdrant shape probe excluded), carrying `similarity_fetch.lane` (str), `similarity_fetch.match` (JSON of the concrete IDs/bucket the lane queried on — the bound IN-list the SQL span parameterizes away), and `similarity_fetch.result_count` (int). Gated on the same seed predicates as `branch_retrieval_lanes`, so a span exists iff that lane fired; runs inside the gathered task so it nests under the branch span and the lane's own SQL span nests under it |
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
clean handler return writes **no** `outcome.success=true`. Rationale: the handler
returns a `StreamingResponse` *before* the pipeline runs, so "returned cleanly"
only means the request passed the boundary — stamping success there would read as
true even if the stream later fails fatally. The success verdict (and the
remaining stream-end rollups: total result count, fetch/failed-branch counts) is
deferred to a stream-aware mechanism (`query_search_planning.md` Bite 2) — one
stream-end rollup, `query_search.cost_usd`, already lands early via the same
generator-`finally` write point (see the Stream-end rollup subsection above).

**Mid-stream fatal failure records a verdict (early Bite 2 slice, verified
2026-07-07).** A fatal Step 0 (LLM retries exhausted) is emitted by the pipeline
as a terminal SSE `error` event *after* HTTP 200, so `@record_outcome` can't see
it. The stream consumer in `event_stream()` watches for that `error` event and
writes `outcome.success=false` + `query_understanding_failed` (a new
`FailureReason`, OQ #2 — distinct from `internal_error`: upstream LLM exhaustion,
not our bug) on the server span. Only the *failure* verdict lands this way; the
success verdict is still deferred, so a clean stream stays absent. Until the rest
of Bite 2:

| Path | `outcome.success` | `outcome.failure_reason` |
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
| `embedding.cost_usd` | float | computed dollar cost via `compute_llm_cost_usd(model, total_tokens, 0)`; **omitted** (with a logged warning) when the model is unpriced — never a fabricated `$0`. The same cost value feeds the request-level `query_search.cost_usd` rollup |
| `embedding.input_count` | int | number of texts in the batch (a single call embeds up to 2048 inputs), so cost/tokens-per-input is derivable. Always set; low-cardinality, metric-label-eligible |
| `error.type` | string | set **only on failure**: normalized failure class (`timeout` for the asyncio ceiling, else the exception class name). Standard OTel key |

The per-request rollup is unchanged: the same computed cost + `total_tokens` still
feed `add_request_cost` / `add_request_tokens` (→ `query_search.cost_usd` +
`query_search.usage.input_tokens`), no-ops outside a tracked request. An ERROR
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
| `FailureReason(str, Enum)` | `api/outcome.py` | enum | the coarse `outcome.failure_reason` vocabulary (replaces the old `NotFoundReason`, folding its two 404 reasons into the broader set). Members: `invalid_parameters`, `invalid_filters` (unknown hard-filter enum value — UI/server taxonomy drift, distinct from `invalid_parameters`), `not_indexed`, `tmdb_removed`, `tmdb_fetch_failed`, `internal_error` |
| `EndpointFailure(HTTPException)` | `api/outcome.py` | exception | request-terminating failure that carries its `FailureReason`; raised at each known failure site so the reason bubbles up |
| `record_outcome` | `api/outcome.py` | decorator | wraps each endpoint; writes `outcome.success`/`outcome.failure_reason` on the server span exactly once (the single write point). Dual-form: bare `@record_outcome` (default, `success_on_return=True`) stamps `success=true` on clean return; `@record_outcome(success_on_return=False)` records failures only and leaves success absent on clean return — for streaming (SSE) endpoints whose handler returns before the pipeline runs (`/query_search`) |
| `Name` + constants | `observability/names.py` | name registry | span names + attribute keys, each derived once from a namespace root via `.child()`; imported by `api/main.py`. Replaces the former inline `_ATTR_*` / literal-string approach |
| `_fetch_movie_payload(...)` | `api/main.py` | helper | opens the `*.payload_creation` span, existence-gate + TMDB fetch; raises `EndpointFailure` (404 `not_indexed`/`tmdb_removed`, 502 `tmdb_fetch_failed`) and owns the child-span error marking; returns `(card_row, tmdb_payload)` |
| `_encode_and_cache_credits(...)` | `api/main.py` | helper | opens `movie_credits.build_and_cache`, builds/encodes/caches credits; shared by both endpoints |
| `_tracer` | `implementation/llms/generic_methods.py` | module tracer | `trace.get_tracer(__name__)`; a no-op `ProxyTracer` when `setup_tracing` hasn't run (offline ingestion/eval imports), so the `llm.generate` span is a cheap no-op there |
| `generate_llm_response_async` | `implementation/llms/generic_methods.py` | router + LLM instrumentation point | the single codepath every routed LLM call passes through; owns the `llm.generate` span, `gen_ai.*`/`llm.*` attrs, retry/payload events, and the three-state failure marking |
| `generate_vector_embedding` | `implementation/llms/generic_methods.py` | embedding instrumentation point | the single codepath every embedding call passes through (search + ingestion); owns the `embedding.generate` span, `gen_ai.*` (system/model/operation/input_tokens) + `embedding.cost_usd` / `embedding.input_count` attrs, and single-attempt ERROR marking. Also feeds the per-request cost/token rollup (unchanged) |
| `compute_llm_cost_usd` | `implementation/llms/pricing.py` | cost util | canonical `(input, cached_input, output)`-per-million pricing table → cache-adjusted USD cost; returns `None` for unpriced models (caller omits `llm.cost_usd` + warns). A parallel table in `estimate_generation_cost.py` should later import from here (see `docs/TODO.md`) |
| `track_request_cost` / `add_request_cost` / `add_request_tokens` | `observability/cost_tracking.py` | cost + token rollup | `ContextVar`-scoped per-request accumulator (`RequestCostAccumulator`) summing every LLM + embedding call's cost **and** input/cached-input/output tokens (all billed attempts) → written on the server span as `query_search.cost_usd` + `query_search.usage.*`; no-op outside a tracked request. Tokens accumulate unconditionally (not gated on pricing); cost skips unpriced models |

---

## 8. What is deliberately NOT instrumented yet

So downstream docs don't overclaim coverage:

- **Endpoints without manual pipeline spans:** `/rerun_query_search`,
  `/similarity_search`, `/attribute_search`. They still get the auto
  server/DB/cache/HTTP spans **and** the `llm.generate` span on any LLM call they
  make (it rides the shared router), but no pipeline-stage spans to parent those
  LLM spans, and no Qdrant span. (`/query_search` is now partially covered —
  Steps 0/1, per-branch + entity-flow attrs, and the Stage-4 semantic Qdrant
  probes — but still lacks Steps 2/3 and the Stage-4 dispatch/scoring spans;
  `/health` intentionally excluded entirely.)
- **No metrics** (RED per endpoint, USE for the box) — Phase 3.
- **No structured/JSON logs and no trace↔log correlation** — Phase 4.
- ✅ **LLM token/cost/`gen_ai.*` attributes now land** on the shared
  `llm.generate` span (1c-1 Bite 1) — including `gen_ai.usage.cache_read.input_tokens`
  and a cache-adjusted `llm.cost_usd` — see §6. ✅ **Per-request cost + token
  usage also roll up** to `query_search.cost_usd` and `query_search.usage.*`
  (input/cached-input/output tokens) on the server span. What's still missing is
  the *pipeline* spans that would parent the LLM spans (Bites 2–9).
- ✅ **Embedding calls now carry a span.** `generate_vector_embedding` emits a
  cross-cutting `embedding.generate` span (the exact parallel to `llm.generate`)
  carrying `gen_ai.*` (system/model/operation/input_tokens) + `embedding.cost_usd`
  / `embedding.input_count`, in addition to feeding the `query_search.cost_usd` /
  `query_search.usage.input_tokens` rollups it already fed — see §6. One
  instrumentation point covers every embedding on search and offline ingestion;
  offline the span is a no-op (`_tracer` is a `ProxyTracer`).
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
