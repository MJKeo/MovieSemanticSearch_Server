# /query_search Observability Planning

**Purpose.** The working plan for instrumenting `POST /query_search`
(`observability_todos.md` item 1c-1, plus groundwork 1c-2 will reuse). This doc
is designed to be consumed in fresh sessions with no memory of the planning
conversation: it carries (1) the as-built pipeline map, (2) the locked
cross-cutting decisions, (3) the per-phase span/attribute plan, (4) open
questions, and (5) a bite-sized implementation checklist so the work can be
taken in small increments without re-deriving context.

**Read alongside:** `observability_architecture.md` §5 (conventions — request
span vs child span, error contract, cardinality) and
`observability/names.py`'s docstring (naming ruleset). All names in this doc
are **indicative** — finalize each against the registry ruleset when its bite
is implemented.

**Last updated:** 2026-07-07 · **Status:** in progress. Bite 1 (LLM router) and
Bite 3 (Steps 0 + 1 spans) are landed + verified; Bite 2 is partial (request
boundary + `query_search.cost_usd` + the Step-0-fatal failure verdict done; the
success verdict + remaining rollups + the OQ #5 walkthrough remain). Check off
bites in §5 as they land and keep `observability_architecture.md` §6 in sync per
bite.

---

## 1. Pipeline map (as-built)

The linear "3-channel merge" diagram in CLAUDE.md describes the **V1** search
shape. The live `/query_search` path is the **V2 streaming, branch-parallel**
architecture — the three retrieval channels (lexical / vector / metadata) live
*inside* each branch's Stage 4 execution, not as one top-level merge. Note
also: the "metadata" channel is a **Postgres fetch + in-memory scoring**, not
an in-memory store — so lexical and metadata are both auto-traced by psycopg;
only **Qdrant (vector)** is an instrumentation blind spot.

### Phase 0 — Request boundary (serial)
`api/main.py::query_search`
1. `clean_query` / `clean_clarification` — validate + normalize (400 on
   empty/over-length).
2. `_to_metadata_filters(body.filters)` — wire filters → internal
   `MetadataFilters` (422 on unknown enum; collapses to None when unset).
3. Hand off to `stream_full_pipeline(...)` wrapped as an SSE
   `StreamingResponse` (HTTP 200 begins *before* the pipeline finishes —
   see the outcome-semantics decision in §2).

### Phase 1 — Query understanding front-end (parallel pair)
`search_v2/streaming_orchestrator.py::stream_full_pipeline`
- **Step 0 — flow routing** (`search_v2/step_0.py::run_step_0`) — one LLM
  call. Decides which entity flows fire + the standard-branch budget.
  **Fatal if it fails** (after retry): the stream emits an `error` SSE event
  and ends.
- **Step 1 — spin generation** (`search_v2/step_1.py::run_step_1`) — one LLM
  call, launched concurrently with Step 0. When Step 0's routing proves the
  spins unused (`_step1_needed` false), the task is cancelled — possibly
  mid-generation, possibly after completing — and its result is disregarded
  either way.
- Then `_plan_step2_branches` builds the fetch list (N standard branches + up
  to one of each entity flow) and the `fetches_ready` SSE event fires.

### Phase 2 — Fan-out (everything below runs concurrently)
The orchestrator launches every fetch as an independent asyncio task; a merge
loop (`asyncio.wait`, FIRST_COMPLETED) streams SSE events as each lands.

**Family 1 — standard branches.** Per branch (all branches overlap):
1. **Step 2 — trait extraction** (`search_v2/step_2.py::run_step_2`) — one
   LLM call → `branch_traits` event.
2. **Post-Step-2 parallel pair**
   (`search_v2/full_pipeline_orchestrator.py::_finish_branch_after_step2`):
   - **Implicit-expectations policy** — one LLM call (popularity/quality
     prior; soft-fail).
   - **Step 3 → query generation, fanned out per trait**
     (`_decompose_and_generate`): each trait runs
     `search_v2/step_3.py::run_step_3` (one LLM call), then immediately fans
     out its category calls to **query generation**
     (`search_v2/endpoint_fetching/category_handlers/handler.py::run_query_generation`)
     — one handler-LLM call per category call, *except* `EXPLICIT_NO_OP`
     (skipped) and `NO_LLM_PURE_CODE` (deterministic code, e.g. TRENDING /
     MEDIA_TYPE).
   - → `branch_categories` event.
   - **Pipelining nuance:** the conceptual linear flow is step 2 → step 3 →
     query generation → query execution, but step 3 and generation are
     pipelined *per trait* — a trait's handler calls start the moment ITS
     step 3 returns, without waiting for sibling traits. There is no
     cross-trait barrier between the two.
3. **Query execution + scoring** — Phase 3 below.

**Family 2 — entity flows** (launched immediately after Step 0, concurrent
with Family 1). Each is resolve → hydrate → `branch_results`; no LLM, no
traits: `exact_title`, `similarity`, `non_character_franchise`,
`character_franchise`, `studio`, `person`
(`search_v2/streaming_orchestrator.py::_run_*_with_hydration`).

### Phase 3 — Query execution (Stage 4), per branch
`search_v2/stage_4_execution.py::_run_branch`, dispatched per branch the
moment that branch's step 3 completes; runs after both the linear chain AND
implicit expectations resolve (the post-Step-2 gather awaits both).
- **Phase B — pool definition:** shorts-blocklist fetch
  (`_fetch_shorts_ids`, parallel), then `asyncio.gather` over **every
  positive candidate-generator spec** (`_dispatch_generator_specs`) → union →
  shorts subtraction → *edge cases:* tiered reranker→generator **promotion
  loop** when a filter is active and the pool is under the candidate floor
  (serial per tier, parallel within a tier), and **neutral-seed fallback**
  when zero generators ran pipeline-wide.
- **Phase C — reranker pass:** `asyncio.gather` over every positive reranker
  spec against the finalized union.
- **Phase D — per-trait scoring:** positive traits scored in pure
  CPU/numpy; negative traits dispatch their calls in a parallel gather
  (`_dispatch_negative_trait`) then score gate×fuzzy.
- **Phase E — branch aggregation:** `_finalize_scores` — CPU only.
- **Implicit-prior rerank**
  (`full_pipeline_orchestrator.py::_apply_implicit_prior_rerank_for_branch`):
  one Postgres signals fetch + boost + re-sort.
- **Hydration:** `fetch_movie_card_summaries` (single auto-traced Postgres
  call) → `branch_results` event.

**Dispatch chokepoint:** every channel call funnels through
`stage_4_execution.py::_dispatch_call`, which wraps each executor in
`asyncio.wait_for(timeout=25s)` and **silently soft-fails on timeout** (log
warning only). Routes diverge one spec at a time in
`search_v2/endpoint_fetching/endpoint_executors.py::build_endpoint_coroutine`
— the Phase B/C gathers mix lexical/vector/metadata routes in flight
simultaneously; there is no per-channel batching layer.

**Qdrant call sites** (the gRPC auto-instrumentation gap): exactly three
primitives in `search_v2/endpoint_fetching/semantic_query_execution.py`, all
`qdrant_client.query_points`:
- `_run_corpus_topn` — unfiltered calibration probe
- `_run_corpus_topn_filtered` — filtered candidate-pool probe
- `_run_filtered_score` — HasId reranker scoring

The four role/restrict executors above them gather these probes across vector
spaces in parallel. A SEMANTIC call first **embeds via OpenAI**
(`generate_vector_embedding` — httpx, auto-traced).

### Phase 4 — Stream terminal
The merge loop drains remaining progress events and emits `done` with
`total_elapsed`. **⚠ Flagged as not-yet-understood** — see Open Question #5:
before implementing the request-level rollups, walk through how the merge
loop / generator ends (including client-disconnect cancellation) and verify
the FastAPI server span stays open for the full stream duration.

---

## 2. Locked cross-cutting decisions

1. **Duration is never an attribute — it's the span.** Every "how long did X
   take" requirement is answered by making X a span. The `elapsed` values
   `run_step_*` return predate tracing; do not copy them into attributes.
2. **Instrument the LLM router once, not each call site.** The `gen_ai.*`
   span lives in
   `implementation/llms/generic_methods.py::generate_llm_response_async` —
   the single codepath every LLM call in this flow passes through. It
   carries `gen_ai.system` (provider), `gen_ai.request.model`,
   `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`, and computed
   dollar cost. The *step identity* comes from the parent span nesting (the
   `step_2` span wraps its LLM child) — never duplicated onto the LLM span.
3. **Prompt version = content hash.** A short hash of the system-prompt text,
   computed once at module load, emitted as an attribute on the LLM span.
   Zero maintenance; changes exactly when the prompt changes; lets evals
   slice by prompt revision. The attribute key must live under **our own
   namespace root, not `gen_ai.`** (the naming ruleset forbids authoring new
   keys under standard OTel roots) — finalize the key in `names.py` at
   implementation.
4. **Model-performance payloads: span events, config-gated, sample-dialable.**
   Full prompt + response captured as span **events** on the LLM span (not
   attributes — size limits, and events are what the sampling dial drops).
   Behind a config flag; capture 100% now; design the flag as a sample *rate*
   (future: always-on-error + small random %), not an on/off boolean.
5. **SSE outcome semantics.** `outcome.success = false` in exactly two cases:
   - Phase-0 validation rejection (400/422) → `invalid_parameters`.
   - Step 0 fatal failure (the `error` SSE event path).
   Everything else is success — **including all-branches-empty**: an empty
   result set is not inherently a failure (same ruling as `/title_search`),
   and per-branch failures are *degradations*: tracked on the branch span
   (`branch_error`) plus a failed-branch count on the request span, never
   flipping the request verdict.
6. **The waterfall must read as the conceptual model:**
   `[step 0 ‖ step 1]` → per branch `[(step 2 → step 3 → query generation) ‖
   implicit priors]` → query execution → rescoring/scoring. Query generation
   is a **named phase distinct from step 3** in span naming, even though the
   two are pipelined per-trait at runtime (§1 Phase 2 nuance).
7. **Raw query at the top, branch queries per branch.** The user-provided
   query (+ clarification + active-filters summary) lives on the **request
   span**; each standard branch's *expanded* sub-query lives on that
   **branch's span**.
8. **LLM failure marking: span status is the terminal verdict; the retry
   story is an attribute + per-attempt events (decided 2026-07-06).** The
   router (`generate_llm_response_async`) wraps its whole retry loop in one
   `llm.generate` span, so three states must be separable without conflating a
   *recovered* call with a *failed* one — a retry that eventually succeeds is a
   **successful** call and must NOT mark the span ERROR (doing so poisons
   error-rate metrics/alerts with calls that worked). The scheme, off two
   signals — an always-on `llm.attempt_count` attribute and the native span
   status:
   - **Clean success:** status UNSET, `attempt_count = 1`, no retry events.
   - **Failed but recovered** (fails then succeeds within the retry limit):
     status stays UNSET (it succeeded), `attempt_count > 1`, and **one
     `llm.retry` span event per failed attempt** carrying the attempt index,
     transient `error.type`, a `timeout` bool, and the backoff slept. Events
     (not attributes) because there can be N and they're timestamped; this is
     also what makes the recovered retry visible in the waterfall behind an
     otherwise-green span.
   - **Failed all retries** (exhausts the limit): status **ERROR** +
     `record_exception(last_exc)` + a span-level `error.type` (the standard
     OTel key, **normalized** — `timeout` for the `asyncio.wait_for` ceiling
     vs. the provider/validation class), `attempt_count = LLM_MAX_ATTEMPTS`.
   Query separation: recovered = `attempt_count > 1 && status != error`;
   exhausted = `status = error`; either sliceable by `error.type`. This layers
   cleanly under decision §2.5: an ERROR `llm.generate` span does **not** flip
   the request verdict — the branch records `branch_error`, the request span
   stays `success=true`; each status is true at its own level. Retry/exception
   events are **always-on** (class names + counts are tiny and
   cardinality-bounded) — kept separate from the config-gated payload events of
   §2.4. No explicit `retry_outcome` enum: it's derivable from
   `(attempt_count, status)`, and a third key would drift out of sync with the
   two it summarizes.

---

## 3. Per-phase instrumentation plan

### Phase 0 — Request boundary (request span)

> **✅ IMPLEMENTED 2026-07-06.** All Phase 0 input capture below is live in
> `api/main.py` (`_record_query_search_inputs` + the two rejection paths),
> `observability/names.py` (`query_search.*` + `filters.*` constants), and
> `api/outcome.py` (new `INVALID_FILTERS` member +
> `record_outcome(success_on_return=False)` failure-only mode). The endpoint is
> decorated `@record_outcome(success_on_return=False)`, so the two rejection
> paths + pre-stream crashes record `outcome.*` but a clean return does NOT
> (the `success=true` verdict + stream-end rollups remain Bite 2). Verified in
> Tempo: valid / 400 / 422 traces all carry the expected attrs + events, and
> `/title_search` still records `success=true` (bare-decorator form intact).
> User confirmed full end-to-end testing on 2026-07-06; the TEMP
> `_STAGE0_TEST_SHORT_CIRCUIT` scaffold has been removed and the handler now
> runs the real downstream pipeline. `filters.active_count` counts distinct
> filter *groups* (min/max ranges count once). Also added: `extra="forbid"` on
> `QuerySearchBody`/`MetadataFiltersInput` + an app-level
> `RequestValidationError` handler so unknown body fields 422 with an
> `invalid_parameters` verdict + `request rejected` event (previously silently
> dropped / uninstrumented). As-built catalog now in
> `observability_architecture.md` §6.

**No child spans** — validation + filter translation are microsecond-scale;
Phase 0 contributes attributes on the request span only. **Capture timing
(decided 2026-07-06):** the input attributes below are written at **handler
entry, from the raw wire body, BEFORE either validator runs** — so a 400/422
trace still carries the full input that caused it.

**Input capture — the always-on vs. on-error framework (decided
2026-07-06).** The deciding test is *which failure modes have an error hook*:
the numeric filter fields (`min/max_release_ts`, `min/max_runtime`,
`min/max_maturity_rank`) have **no validation at all** — a wrong-unit
timestamp or min>max range sails through and yields an empty-result trace
with `outcome.success = true`. There is no error path to capture on, so raw
filters must be always-on. Additionally, Stage 4's tier-promotion /
candidate-floor edge cases fire only when a filter is active — cross-trace
analysis needs filter facts on every trace. Tests for always-on: (1) would a
*successful* trace need it, (2) queryable in aggregate, (3) bounded size +
cardinality-safe.

| What | Shape | Notes |
|------|-------|-------|
| Raw query text | attr, always-on | high-cardinality OK as span attr, never a metric label. ⚠ Pydantic has `min_length=1` but **no max** — pre-validation input is unbounded, so the attr needs its own defensive truncation (~300 chars, above MAX_QUERY_CHARS=200), not the validator's cap. Never sampled — sampling is reserved for LLM payloads |
| Raw query length | attr (int), always-on | free; makes over-length 400s self-explanatory (truncated text alone can't show true size); length distribution as a byproduct |
| Clarification text + raw length | attrs, always-on when sent | identical treatment to query — clarification has its own independent 200-char cap + rejection path (MAX_CLARIFICATION_CHARS, `query_input_validation.py`) and is equally unbounded at the Pydantic layer, so every query-side argument applies |
| Per-field `filters.*` typed attrs | attrs, always-on, each set only when the client sent that field | **resolves OQ #4.** One typed attr per wire field: 6 ints (`filters.min_release_ts`, …) + 4 string arrays (`filters.genres`, …), raw wire values pre-translation. Existence = "filter active" (queryable per field), typed values = debuggable and range-queryable (`min_runtime > 180`). Subsumes both a names array and a JSON blob — see note below |
| `filters.active_count` | attr (int), always-on (0 = none) | one cheap key so "any filter active?" isn't a ten-way `!= nil` OR |
| Rejection detail | **span event, error path only** | at the 400/422 raise site, carrying the exception detail (already names the offending field + value). `outcome.failure_reason` classifies; the event pinpoints |
| Filter-translation failure flag | via `outcome.failure_reason` = **`INVALID_FILTERS`** (new member, decided 2026-07-06) | the 422 from `_to_metadata_filters` gets its own FailureReason member rather than sharing `invalid_parameters` with the free-text 400s. Rationale: distinct actionable classes per the enum's own design rule — a bad query is user behavior, an unknown filter enum value is UI/server taxonomy drift (filters come from UI dropdowns), i.e. *our* bug and alertable. Don't rely on the implicit 400-vs-422 status split. Unexpected crashes inside translation still fall through to `internal_error` |
| Total result count | attr | sum of cards across ALL fetches (standard + entity), written at stream end |
| Fetch count / failed-branch count | attrs | the degradation signals per decision §2.5 |
| `outcome.success` / `outcome.failure_reason` | attrs | via an SSE-adapted outcome mechanism — see OQ #3 |

**Why per-field typed attrs, not names-array + JSON-blob (the earlier
lean):** Tempo treats JSON inside an attribute as an opaque string —
nothing inside it is filterable — so that design needed two attributes
(one to query, one to read). Per-field typed attrs are one representation
doing both jobs, and go further (value-range queries the names array never
allowed). Registry cost is 10 keys, but the set mirrors
`MetadataFiltersInput`, a closed wire schema — churn is bounded and each
key is independently meaningful. Known limit in BOTH designs: a
single-query "rank all filters by usage" isn't clean in TraceQL (weak
group-by on arrays) — that's ten existence-rate queries or a future
metric either way.

**Deliberately never captured:** translated internals (expanded
`watch_offer_keys`, resolved `MetadataFilters`) — derivable from raw input +
code version, and raw wire values are the ground truth when translation
itself is the bug.

### Phase 1 — Steps 0 + 1

- **`step_0` span** (child of request span): attrs `flows` (array of
  activated flow names) + `standard_branch_count`. Both low-cardinality and
  explain the shape of everything downstream. Tokens/cost/payloads free via
  the router span (§2.2–2.4).
- **`step_1` span**: attr **`unused`** (bool) — true when Step 0's routing
  made the spins unneeded. Named "unused" rather than "cancelled": the
  meaning is "the result was not consumed," which covers both
  cancelled-in-flight and completed-then-disregarded. (Note: in code the
  task IS cancelled at that point and may not have finished generating —
  the attribute is about consumption, not about how far generation got.)
- **Step 0 fatal failure**: request verdict `success=false`;
  `failure_reason` vocabulary needs a decision — see OQ #2.

### Phase 2 — Standard branches

- **Branch span** (one per standard fetch, spanning launch →
  `branch_results`; its duration = "step 1 done → results returned"):
  - `fetch_id`, `kind`
  - branch query full text (the expanded sub-query)
  - `branch_error` (when the branch soft-failed)
  - branch result count
  - **activated-categories array**: the category of every category call
    across ALL the branch's traits, **duplicates preserved** — each trait
    dedupes internally, but the same category can activate in multiple
    traits and per-category counts across a branch must be queryable.
    Include deterministic (NO_LLM_PURE_CODE) categories in the array even
    though they get no span (OQ #7).
- **`step_2` span**: attrs `trait_count` (overall) +
  `negative_trait_count` (subset of overall).
- **`implicit_expectations` span**: attrs for the end result —
  `quality_prior` and `popularity_prior` direction/strength. Small closed
  enums → the rare attributes that are safe as future metric labels.
- **`step_3` span** — one per trait.
- **`query_generation` span** — one per **handler-LLM** call, attr
  `category`. Deterministic and no-op calls get no span (microsecond
  units — see the "genuine sub-units of work" convention); they are counted
  via the branch-level categories array.

### Phase 2 — Entity flows

- **Uniform per-fetch span** (mirrors how `outcome.*` unified the read
  endpoints): `fetch.type`, result count, error state, and an **explicit
  zero-results flag** — routing to an entity flow means Step 0 believed a
  match exists, so zero results is actionable (taxonomy gap / resolution
  miss), not neutral.
- **One flow-specific signal each**, chosen by "what silently degrades":
  - `similarity` / `studio` / `person` / `exact_title`:
    **references requested vs. resolved** — failed anchor/name resolution
    returns plausible-looking garbage; resolution rate is the only signal
    that catches it.
  - franchises (both): **empty primary bucket / top tiers** despite a
    canonical-name match.
- Deepen a flow's telemetry only when a real question shows up.

### Phase 3 — Query execution (Stage 4)

- **`stage_4` span** per branch, with thin child spans for the phase
  boundaries: **pool definition (Phase B)** and **reranker pass (Phase C)**.
- **Phase B span attrs** (the semantic facts no auto-instrumentation sees):
  - `neutral_seed_fired` (bool)
  - promotion tiers used (int; 0 = the tiered-promotion edge case never
    fired) + pool size before/after promotion
  - candidate-floor-hit (bool)
- **Per-spec span at the `_dispatch_call` chokepoint** — this is what gives
  the per-endpoint breakdown, since the B/C gathers mix routes: attrs
  `route`, `operation_type`, result/pool size, `timed_out` (bool). A timeout
  additionally records a **span event** — today it's a silent soft-fail and
  must become queryable.
- **Qdrant manual spans at exactly the three primitives** in
  `semantic_query_execution.py` (`_run_corpus_topn`,
  `_run_corpus_topn_filtered`, `_run_filtered_score`). One span name
  (`query_search.semantic_qdrant`) discriminated by a `probe_kind` attribute.
  As-built attrs (finalized against `names.py`): `probe_kind`
  (`calibration`/`pool`/`hasid_score`), `vector_space`, `query_params` (the
  space body as `model_dump_json(exclude_defaults=True)` — only the populated
  fields that feed `embedding_text`, so each probe is self-describing),
  `limit`, `filter_active`, `hit_count`. **`filter_active` = the USER HARD
  FILTER was applied** (True only on the pool probe; False on calibration and
  on the HasId reranker, whose `HasIdCondition` is a pool restriction, not the
  hard filter). **No wrapper spans per vector space** — the probe spans already
  render the fan-out, and the per-request span budget (~130–150 spans for a
  3-branch/4-trait query) is healthy only if we don't double-wrap.
- **Scoring span** around Phase D/E (pure CPU/numpy — the one place a big
  union stalls the *event loop*, invisible to network spans).
- **Implicit-prior rerank span**: attr `boost_axis`
  (`popularity` | `quality` | `none`). Contains one auto-traced Postgres
  fetch.
- **Hydration**: no manual span — a single auto-traced Postgres call.

### Phase 4 — Stream terminal

**Blocked on understanding (OQ #5).** This is where the request-level
rollups (outcome verdict, total result count, failed-branch count) get
written, and where the server span must be confirmed to cover the full
stream. Do the walkthrough before implementing Bite 2; nothing else in this
phase gets designed until then.

---

## 4. Open questions

1. **Namespace root for the shared pipeline spans.** **PARTIALLY RESOLVED
   (2026-07-06, Bite 3)** — the Step 0/1 routing spans are owned by
   `query_search` (`query_search.step_0` / `.step_1`), because
   `/rerun_query_search` (1c-2) reuses the **Step 2 → Stage 4** spans, *not*
   routing, so per rule B query_search is routing's home endpoint. The open
   part is the genuinely shared Step 2 → Stage 4 spans (Bites 4–7): those may
   still deserve a pipeline-owned root rather than `query_search.*` — decide
   against `names.py`'s docstring when Bite 4 lands.
2. **`FailureReason` vocabulary for Step 0 fatal failure.** **RESOLVED
   (2026-07-06)** — new member `query_understanding_failed` added to
   `api/outcome.py::FailureReason`, per the lean: an actionable class distinct
   from "our code crashed" (`internal_error`) — a fatal Step 0 is an upstream
   LLM/provider exhaustion. Written on the server span from `event_stream()`
   when the pipeline emits its terminal SSE `error` event (see OQ #3). (The
   other new member, `INVALID_FILTERS`, already landed with the Phase 0
   boundary work.)
3. **How the outcome mechanism adapts to SSE.** **PARTIALLY RESOLVED
   (2026-07-06) — failure path only.** The stream consumer (`event_stream()`
   in `api/main.py`) watches the `(event_name, payload)` stream and, on the
   terminal `error` event, writes `outcome.success=false` +
   `query_understanding_failed` on the server span (the handle it already
   holds) — the orchestrator stays transport-agnostic and keeps emitting the
   same `error` event. **Still open:** the *success* verdict
   (`outcome.success=true`) + the stream-end rollups (total result count,
   fetch/failed-branch counts), which must be written at clean generator
   completion including the client-disconnect path. Do that as the rest of
   Bite 2, reusing the generator-`finally` write point that
   `query_search.cost_usd` already uses.
4. ~~**Active-filters attribute shape.**~~ **RESOLVED (2026-07-06)** — see
   the Phase 0 table in §3: per-field typed `filters.*` attributes (raw
   wire values, always-on, each present only when sent — existence =
   active, values queryable) + `filters.active_count`, plus an
   error-path-only span event at the 400/422 raise site with the
   rejection detail. Always-on is forced by the unvalidated numeric
   filter fields, whose failure mode is empty results on a *successful*
   trace, not a 422. Input attrs are written at handler entry from the
   raw wire body, pre-validation. (Earlier names-array + JSON-blob lean
   superseded — rationale in the §3 table note.)
5. **Phase 4 (stream terminal) mechanics.** Explicitly flagged as not yet
   understood — walk through the merge-loop shutdown, `done` emission, and
   client-disconnect cancellation, and verify the FastAPI server span stays
   open for the whole stream, before wiring rollups. (Owner: do this
   together at the start of Bite 2.) **PARTIALLY EXERCISED (2026-07-06):** the
   `query_search.cost_usd` request-cost rollup already writes on the server span
   from `event_stream()`'s `finally` on the assumption that the ASGI server span
   ends only after the generator fully drains. That assumption is *designed-for
   but not yet Tempo-verified* — the end-to-end check (does `query_search.cost_usd`
   actually appear on the server span?) is the concrete test that resolves this OQ.
6. ~~**Payload-capture flag mechanics.**~~ **RESOLVED (2026-07-06, Bite 1)** —
   env var `LLM_PAYLOAD_CAPTURE_SAMPLE_RATE`, a float 0.0–1.0 read once at
   module import in `generic_methods.py`. `>= 1.0` captures every call
   (default), `0` disables, else Bernoulli(rate) per successful call; when the
   rate is > 0, a terminal failure captures the prompt unconditionally
   (always-on-error). Payloads ride `llm.payload` span events, not attributes.
7. **Deterministic category calls and spans.** Current lean (recorded in
   §3): no span for NO_LLM_PURE_CODE / EXPLICIT_NO_OP calls; they're
   counted in the branch-level categories array. Revisit only if their
   (currently microsecond) cost ever grows.

---

## 5. Implementation bites

Ordered so each bite is a self-contained session with its own verification.
Check off as landed; after each bite update `observability_architecture.md`
§6 (the span/attribute catalog) and, when it changes endpoint behavior,
`docs/modules/api.md`.

- [x] **Bite 1 — LLM router telemetry (the multiplier).** *(code landed
      2026-07-06; awaiting the user's manual Tempo verification.)*
      `llm.generate` span in `generate_llm_response_async` wrapping the whole
      retry loop, carrying standard `gen_ai.*` (system, request.model,
      usage.input_tokens, usage.output_tokens, usage.cache_read.input_tokens) +
      our cache-adjusted `llm.cost_usd`, `llm.prompt_version` (system-prompt
      content hash, `@lru_cache`d), and `llm.attempt_count`. Failure marking per decision §2.8: `llm.retry`
      events per recovered attempt (status stays UNSET); ERROR + `error.type`
      + `record_exception` on exhaustion. Payload capture as `llm.payload` span
      events (system + user prompt + response JSON) gated by
      `LLM_PAYLOAD_CAPTURE_SAMPLE_RATE` (float, default `1.0` = capture all;
      `0` disables; else Bernoulli per call) — **resolves OQ #6**: env var,
      float rate, read once at module import, with always-on-error prompt
      capture when the rate is > 0. Cost via a new canonical
      `implementation/llms/pricing.py` (`compute_llm_cost_usd`); unpriced
      models emit no `llm.cost_usd` and log a warning (no fabricated cost).
      Verify: run one `/query_search` and confirm every LLM call in the trace
      carries tokens/cost/hash/attempt_count; force a transient failure and
      confirm `llm.retry` + green span; exhaust retries and confirm ERROR +
      `error.type`; set the rate to `0` and confirm payload events vanish.
- [~] **Bite 2 — Terminal mechanics + request boundary.**
      **Request-boundary input capture is DONE (2026-07-06)** — see the ✅
      note in §3 Phase 0: pre-validation query/clarification text + lengths,
      per-field `filters.*` typed attrs + `active_count`, the `request
      rejected` span event, and the 400/422 failure verdicts
      (`invalid_parameters` / `invalid_filters`) all landed via
      `record_outcome(success_on_return=False)`. **The Step-0-fatal failure
      verdict is also DONE + verified (2026-07-07):** `event_stream()` watches
      the SSE stream and, on the terminal `error` event, writes
      `outcome.success=false` + the new `query_understanding_failed`
      `FailureReason` (OQ #2 resolved) on the server span — the failure half of
      the SSE-adapted outcome mechanism (OQ #3). **Remaining in this bite:**
      the OQ #5 walkthrough (understand Phase 4; verify the server span
      covers the full stream), then the SSE-adapted *success* mechanism
      (OQ #3) that writes `outcome.success=true` + the stream-end rollups
      (total result count, fetch/failed-branch counts) at generator
      completion. **One stream-end rollup landed early
      (2026-07-06): `query_search.cost_usd`** — the summed LLM + embedding cost
      (all billed attempts) written on the server span from `event_stream()`'s
      `finally` via `observability/cost_tracking.py`. It rides the same
      server-span-stays-open assumption OQ #5 must confirm; the remaining
      rollups can reuse the same write point once that's verified. (The TEMP `_STAGE0_TEST_SHORT_CIRCUIT` scaffold
      was removed 2026-07-06 after the boundary was fully tested; the handler
      runs the real pipeline again.)
      Verify: success case, forced Step-0 failure case, and a forced
      single-branch failure (success stays true, count increments). (400/422
      already verified.)
- [x] **Bite 3 — Steps 0 + 1 spans.** *(landed + verified 2026-07-07.)*
      `query_search.step_0` (`step_0_flows` — SearchFlow
      values + `standard`; `step_0_standard_branch_count` — always-set budget)
      + `query_search.step_1` (`step_1_unused`, recorded directly at the user's
      request rather than derived). **OQ #1 resolved:** query_search owns these
      (rerun reuses Step 2 → Stage 4, not routing); attrs flat under the
      query_search root. Spans started non-current + `use_span(end_on_exit=False)`
      inside each task so the `llm.generate` child nests and step_1's span
      outlives its call (unused known only post-Step-0); fatal Step 0 → span
      ERROR; try/finally closes spans on disconnect. `api/main.py` wraps the
      stream loop in `use_span(request_span)` so the server span parents the
      whole pipeline. Added `_standard_branch_count` as the single budget source
      (`_step1_needed` / `_plan_step2_branches` now read it). Rejected in design:
      int-enum flows, spin-text attr, `[step0 ‖ step1]` wrapper span. Verify:
      waterfall shows the two overlapping under the server span with an
      `llm.generate` child nested in each; `unused=true` on an entity-only query;
      forced Step-0 failure shows step_0 ERROR + green request span.
- [~] **Bite 4 — Standard-branch skeleton.** Branch span landed earlier
      (kind + uses_original_text). **Trait pipeline landed + verified:**
      `step_2` span (`step_2_trait_count` +
      `step_2_contextualized_phrases`; negative_trait_count dropped per the
      user), a per-`trait` span (`trait_phrase` / `trait_polarity` /
      `trait_commitment`, `trait_step_3_error` on soft-fail, `"solo trim"`
      event) bracketing step 3 → generation, `step_3` span
      (`step_3_combine_mode` + `step_3_categories` recorded **POST-SOLO-trim**,
      not pre-trim), and `query_generation` per handler-LLM call (`category` +
      `query_generation_endpoints` — the EndpointRoute names that fired;
      no span for EXPLICIT_NO_OP / NO_LLM_PURE_CODE). Built with plain
      `start_as_current_span` in `full_pipeline_orchestrator.py` +
      `handler.py` (both gained a module tracer; they already run under the
      branch span). **Still pending in this bite:** `implicit_expectations`
      span (skipped for now per the user), and the branch-level
      `branch_error` / result-count attrs (deferred with Bite 5). Verify:
      per-trait pipelining visible (a trait's generation bar starts before
      sibling step-3 bars end); `step_3_categories` shows the trimmed set +
      `"solo trim"` event on a SOLO-with-extras trait; forced Step-3 failure
      shows `trait_step_3_error` + step_3 ERROR with the request still
      succeeding; no `query_generation` span for a NO_LLM_PURE_CODE category.
- [ ] **Bite 5 — Stage 4 execution.** `stage_4` span + Phase B/C child
      spans + edge-case attrs (neutral seed, tier promotion, floor) +
      per-spec dispatch spans with route/operation_type/`timed_out` and the
      timeout span event. Verify: per-route breakdown queryable in Tempo;
      artificially low timeout produces the event.
- [x] **Bite 6 — Qdrant primitives.** *(code landed; verified via
      `search_v2/test_semantic_qdrant_span.ipynb` — spans reach Tempo with the
      right per-space fan-out and attributes.)* Manual `query_search.semantic_qdrant`
      span on each of the three `query_points` call sites (attrs: `probe_kind`,
      `vector_space`, `query_params`, `limit`, `filter_active`, `hit_count`);
      throwing probes auto-mark ERROR. Also converted the previously log-only
      soft-fail in `execute_semantic_query` into queryable `semantic_query_retry`
      / `semantic_query_failed` span **events** on the ambient span. Built ahead
      of Bite 5: the probes nest under whatever span is current (the notebook's
      parent span standalone; today the server/Step span in the live pipeline),
      and will nest under the Bite-5 dispatch span once it exists — at which point
      the branch-level `branch_error` attribute + failed-branch count get wired
      there (deliberately out of scope for Bite 6).
- [ ] **Bite 7 — Scoring + rerank.** Phase D/E scoring span,
      implicit-prior rerank span (`boost_axis`). Verify: CPU bar visible
      on a large-union query; boost_axis matches the implicit policy.
- [x] **Bite 8 — Entity flows.** *(code landed; awaiting Tempo verification via
      `search_v2/testing_nonstandard_flows.ipynb`.)* Attributes hang on the
      existing `query_search.branch` span, set INSIDE each entity-flow entry
      executor (`run_*_search`, which runs under the branch span) — not the
      shared Stage-4 executors, so standard branches / attribute_search stay
      unpolluted. **Universal skeleton:** `branch_entities`,
      `branch_entity_resolved_counts` (per-entity pre-union), `branch_unresolved_entity_count`
      (set in executors); `branch_result_count` + an `entity flow empty` span
      **event** (set post-hydration in the orchestrator wrappers via
      `_stamp_branch_outcome`). Aliases are **always-on**. **Per flow:** person —
      per-entity resolution child spans + `branch_top_tier`/`_count`; similarity —
      redesigned into JSON-string maps organized by four reader questions
      (`branch_retrieval_lanes`/`branch_retrieval_total`, `branch_lane_weights`/
      `branch_vector_space_weights`, `branch_weave_targets` as one `{bucket:slots}`
      desired-allocation map (the pre-weave target ratio, not the realized draw),
      `branch_low_cohesion_fallback`, `branch_additional_boosts`; single-only
      `branch_shape_modifiers` + scalar `branch_anchor_shape`; multi-only
      `branch_anchor_shape_cohesion` + `branch_lane_cohesion` +
      `branch_vector_space_cohesion`) — see observability_architecture.md §6 for the
      full catalog — + a manual **Qdrant span** on the shape probe + one
      **`similarity_fetch` span per Postgres candidate lane that ran** (lane +
      the concrete match IDs/bucket + result_count; qdrant excluded); exact_title —
      `branch_exact_title_year` + `branch_source.{seed,close,fanout}_count` (always-on)
      + `branch_source.title_only_count` (conditional on a supplied year); studio —
      `branch_studio_llm_fallback` + per-ref `branch_studio_entity_paths` /
      `branch_studio_brand_names` + `branch_studio_{brand,freeform}_match_count`
      (via a new opt-in `path_match_counts` side channel on `execute_studio_query`);
      non_character_franchise — aliases + `branch_franchise_llm_fallback` +
      `branch_top_tier`(primary)/`_count` + `branch_secondary_count`;
      character_franchise — `branch_character_forms` / `branch_franchise_forms`,
      `branch_character_franchise_llm_failed`, two resolution child spans
      (lineage-mainline split folded into the franchise span), `branch_tier_counts`
      + `branch_top_tier`(tier_1). Verify: one query per flow type via the notebook;
      force a zero-result entity query and confirm the empty event +
      `branch_unresolved_entity_count` / studio `brand_match_count==0`.
- [ ] **Bite 9 — End-to-end validation + docs reconciliation.**
      Representative multi-branch query: single trace ID end to end,
      parallel stages render as overlapping bars (validates the asyncio
      fan-out), rollup counts match the SSE payloads. Capture the first
      real latency finding (which stage/provider dominates — the original
      Phase 1 goal). Reconcile `observability_architecture.md`,
      `docs/modules/api.md`, and check off 1c-1 in
      `observability_todos.md`.
