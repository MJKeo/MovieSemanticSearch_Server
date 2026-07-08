# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Add `embedding.generate` span at the embedding chokepoint
Files: observability/names.py, implementation/llms/generic_methods.py, observability_context/observability_architecture.md
Why: every routed LLM call carried a manual `llm.generate` span, but embedding calls emitted none of their own — per-call latency/tokens/cost/batch-size were invisible in the trace waterfall (the request-level cost/token rollup already included embeddings, but not as their own spans). Architecture doc §8 flagged this gap explicitly.
Approach: wrapped `generate_vector_embedding` (the single chokepoint every embedding passes through — search + ingestion) in one `embedding.generate` span, the exact parallel to `generate_llm_response_async` owning `llm.generate`. Reused the module `_tracer` (no-op offline, so ingestion pays nothing), `_error_type`, `compute_llm_cost_usd`. Standard facts on the OTel GenAI keys (`gen_ai.system`/`request.model`/`operation.name`=`embeddings`/`usage.input_tokens`); two project-owned attrs (`embedding.cost_usd`, `embedding.input_count`) in names.py. New `embedding` root earns a dot (rule C: 2 emitted siblings). Cost computed once, feeds both the existing `add_request_cost` rollup and the span attr (no double-count — no retry loop, since the embedding client runs max_retries=0). Deliberately omit output/cache-read token attrs (always 0 for embeddings). Single-attempt error contract: `record_exception=False`/`set_status_on_exception=False` on start; on failure set `error.type` + ERROR + `record_exception`, then re-raise the existing ValueError wrap.
Design context: mirrors the `llm.generate` span contract (§5.2 error contract, names.py naming rules). An ERROR embedding span is a degradation, does not flip the request verdict. Module doc docs/modules/observability.md left untouched (it points to the architecture catalog rather than enumerating spans, same as the llm.generate precedent).
Testing notes: verify via a scratch script with a ConsoleSpanExporter — call under `track_request_cost()` and confirm the span carries the GenAI attrs + `embedding.input_count` + `embedding.cost_usd`, and that the accumulator totals match; force a bad model to confirm ERROR + `error.type`. Live confirm in Tempo that a `/query_search` renders `embedding.generate` spans reconciling with `query_search.usage.*`.

## Per-branch span for /query_search (1c-1 Bite 4)
Files: observability/names.py, search_v2/streaming_orchestrator.py, observability_context/observability_architecture.md
Why: the streaming pipeline fans out into concurrent branches (up to 3 standard + at most one entity flow), but the trace had no per-branch span — every branch's `llm.generate` calls nested directly under the request span, so you couldn't see per-branch duration/type or attribute LLM cost to a branch.
Approach: one `query_search.branch` span per fetch, keyed by `fetch_id`, started non-current in `_stream_from_branch_plan` right after `fetches_ready` and closed centrally in the merge loop when no live task remains for that `fetch_id` (the terminal-`branch_results` signal), with a `finally` safety net for client disconnect. Chose central merge-loop closing over ending at each of the 6 terminal `_branch_results_event` sites — the "no remaining task for this fetch_id" invariant holds uniformly across standard (multi-task Step2→3→4) and entity-flow (single-task) branches, success and soft-fail alike. Nesting via a `_run_under_span(span, coro)` helper wrapping every branch task launch (standard Step 2, six entity flows, Step 3, Stage 4); the span is threaded into `_handle_finished_task`/`_handle_step2_done` for the follow-on launches. Attributes: `branch_type` (existing fetch `type` string, reused — closed low-cardinality set mirroring SearchFlow) on every branch; `branch_uses_original_text` on standard branches only, true for exactly `standard:original` when the non-clarification path uses the raw query verbatim — computed in `stream_full_pipeline` mirroring `_plan_step2_branches`'s `else` shape and threaded down as `original_branch_uses_raw_query` (rerun defaults false). Names flat-underscore per the step_0_*/step_1_* precedent (names.py rule C).
Design context: streaming path only (matches existing query_search span coverage; non-streaming orchestrator stays uninstrumented). Rerun shares `_stream_from_branch_plan` so it emits branch spans for free. Per-branch soft-fail is a degradation — span status stays UNSET, no `outcome.*` touch (docs/conventions.md span-vs-verdict rule); `branch_error`/result-count attrs deferred to a later bite. Follows the query_search_planning.md Bite 4 spec.
Testing notes: verify in Tempo — standard-no-clarification query shows N branch spans, exactly one `branch_uses_original_text=true`, each parenting its own `llm.generate` children; entity-flow query shows `branch_type=person` (etc.) with no uses_original_text attr; clarification flow → original branch false; rerun → all standard branches false; client-disconnect mid-stream leaves no unclosed span.

## Roll up per-request LLM+embedding token usage onto the /query_search span
Files: observability/cost_tracking.py, implementation/llms/generic_methods.py, observability/names.py, api/main.py, observability_context/observability_architecture.md
Why: the request cost rollup (`query_search.cost_usd`) already existed, but the token counts behind that cost were not exposed on the span — the user wants input (cache-inclusive), cached_input, and output token totals per request to see what they're paying for.
Approach: extended the existing `RequestCostAccumulator` (already threaded through the whole request via ContextVar) with three token fields + an `add_tokens` method, rather than building a parallel mechanism. Added `add_request_tokens` beside `add_request_cost` and called it from the same two accounting sites (`_account_llm_call_cost` for all LLM providers, `generate_vector_embedding` for embeddings). Three new `query_search.usage.*` name constants; three `set_attribute` calls in the same stream-end `finally` that writes cost.
Design context: three semantics locked with the user — (1) tokens sum over ALL billed attempts (accounting fires per-attempt before the parse that can fail), consistent with cost; (2) cached_input is a SUBSET of input, never additive (mirrors the per-call `gen_ai.usage.cache_read.input_tokens` convention); (3) tokens roll up UNCONDITIONALLY even for unpriced models (add_tokens is not gated on price, unlike add_cost) so token totals stay honest when the dollar total under-reports. `usage` earns a dotted namespace (names.py rule C: groups 3 emitted siblings, mirrors standard `gen_ai.usage.*`).
Testing notes: token rollup is a superset of any single `llm.generate` span's `gen_ai.usage.*` (all-attempts vs successful-attempt); unpriced-model tokens counted while their cost is 0, so token total and cost can diverge by design. Verify in Tempo that a /query_search trace's server span carries all three `query_search.usage.*` attrs and that cached ≤ input.

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

## Implemented /query_search Stage 0 observability (1c-1, first bite)
Files: observability/names.py, api/outcome.py, api/main.py,
observability_context/observability_architecture.md,
observability_context/query_search_planning.md, docs/modules/api.md
Why: first implementation slice of the /query_search instrumentation plan —
the request boundary (input validation + filter translation), instrumented so
even rejected requests carry the full input that caused them.
Approach:
- names.py: added `query_search.*` (query/clarification text + `_chars`) and a
  new `filters.*` root (one typed attr per wire field + `active_count`). Filters
  got their own root, not a `query_search` child, because the same
  MetadataFiltersInput shape is reused by three sibling endpoints (1c-2..4).
- outcome.py: added `FailureReason.INVALID_FILTERS` (unknown hard-filter enum =
  UI/server taxonomy drift, distinct actionable class from `invalid_parameters`);
  made `record_outcome` a dual-form decorator with `success_on_return=False`
  (failure-only) for SSE endpoints whose handler returns before the pipeline runs.
- main.py: `_record_query_search_inputs` writes the raw wire body to the server
  span at handler entry, BEFORE validation (per the §3 Phase 0 decision — the
  unvalidated numeric filter fields fail as empty-results on a *successful*
  trace, so on-error capture is structurally blind to them). Text attrs
  defensively truncated at 300 (`_INPUT_ATTR_MAX_CHARS`) since Pydantic sets no
  max. Both rejection paths (400 clean_query, 422 _to_metadata_filters) now raise
  `EndpointFailure` + emit a `request rejected` span event with the detail.
  `_to_metadata_filters`'s four raises consolidated through `_reject_invalid_filters`.
Design context: query_search_planning.md §3 Phase 0 (the decided capture table),
§2.5 (SSE outcome semantics), and the user's failure-only interim decision.
Verified in Tempo: valid / 400 / 422 traces carry the expected attrs + events;
/title_search still records success=true (bare-decorator form intact).
Testing notes: no unit tests added (test-boundaries rule); the SSE-adapted
success verdict + stream-end rollups remain Bite 2.
Update (2026-07-06): user completed end-to-end testing; the TEMP
`_STAGE0_TEST_SHORT_CIRCUIT` scaffold has been REMOVED — query_search runs the
real downstream pipeline again.

## filters.active_count now counts filter groups, not wire fields
Files: api/main.py | `_record_query_search_inputs` counts distinct user-facing
filter groups (the three min/max ranges — release_date, runtime, maturity —
each collapse to one) instead of one-per-set-field, so setting both bounds of a
range is one active filter, not two. Per-field `filters.*` attrs unchanged.

## Forbid unknown request fields + record framework-422 outcomes
Files: api/main.py, observability_context/observability_architecture.md
Why: an unknown/typo'd parameter (e.g. filter key `genrez`) was silently
dropped by Pydantic's default extra-ignore, reading downstream as "no filter on
that axis" — a silent client/server drift — and framework-level 422s never
reached `@record_outcome`, so they were a trace blind spot (was documented as
uncovered in architecture §8).
Approach:
- Set `model_config = ConfigDict(extra="forbid")` on `QuerySearchBody` and the
  shared `MetadataFiltersInput` (latter also affects /similarity_search) so an
  unknown field 422s at the boundary instead of being dropped.
- Added an app-level `@app.exception_handler(RequestValidationError)`
  (`_on_request_validation_error`) that stamps `outcome.success=false` +
  `invalid_parameters` and a `request rejected` event whose detail names the
  offending field(s) via `_summarize_validation_errors` (loc + msg + type only,
  never the input value — PII-safe; capped at 5 errors), then delegates to
  FastAPI's default `request_validation_exception_handler` so the HTTP response
  is byte-for-byte unchanged. Reused the existing `INVALID_PARAMETERS` reason
  (framework body/param validation == "bad/missing request params"), no new
  enum member. Handler is app-wide, so all endpoints' malformed-body 422s now
  carry a verdict.
Design context: architecture §5.2 (single-write outcome model), §8 (the
now-closed framework-422 gap). Chose a global exception handler over per-model
validators because the rejection fires before any handler runs.
Testing notes: no unit tests (test-boundaries rule). Manual: `import api.main`
imports clean. Verify in Tempo — unknown top-level field and unknown filter key
both 422 with `invalid_parameters` + a `request rejected` event naming the bad
field.

## LLM router telemetry (observability Bite 1) — the `llm.generate` span
Files: implementation/llms/generic_methods.py, implementation/llms/pricing.py (new),
observability/names.py, observability_context/query_search_planning.md,
observability_context/observability_architecture.md,
observability_context/observability_todos.md, docs/TODO.md

### Intent
Instrument every routed LLM call at the single codepath they share
(`generate_llm_response_async`) so every step's call carries tokens, cost,
prompt version, retry/attempt facts, and (sample-gated) full prompt/response —
the "multiplier" bite that lights up all of `/query_search`'s LLM fan-out from
one place. This is `query_search_planning.md` §5 Bite 1. Code landed; the user
will manually verify in Tempo.

### Key Decisions
- **One span wraps the whole retry loop** (not one per attempt) — decision §2.2.
  Step identity comes from parent-span nesting, never duplicated onto the LLM
  span. `record_exception=False, set_status_on_exception=False` so error marking
  is fully hand-controlled (mirrors `_fetch_movie_payload`).
- **Failure marking = span status + `llm.attempt_count`** (new decision §2.8,
  added to the plan this session per the user's ask). Three states: clean
  success (UNSET, count 1), failed-but-recovered (UNSET/green, count >1, one
  `llm.retry` event per failed attempt), failed-all-retries (ERROR +
  `record_exception` + normalized `error.type`, count at ceiling). A recovered
  retry is deliberately NOT an ERROR — that would poison error-rate metrics. No
  `retry_outcome` enum (derivable from status+count). An ERROR LLM span does not
  flip the request verdict (§2.5 layering).
- **Standard vs. owned keys.** `gen_ai.system/request.model/usage.*` and
  `error.type` are OTel standard keys emitted as spec strings (module constants
  for typo-safety), NOT authored in `names.py` (which never re-spells a standard
  root). `llm.cost_usd`, `llm.prompt_version`, `llm.attempt_count` + the
  `llm.generate` span name are new `Name`s under a new `llm` root in names.py.
- **Cost in a new canonical `pricing.py`** (`compute_llm_cost_usd`), seeded from
  the existing `estimate_generation_cost.py` table. Chose a new reusable module
  over importing the helper script (wrong dependency direction: the low-level
  router must not depend on a metadata-gen helper). Unpriced model → `None` →
  attribute omitted + warning (never a fabricated $0). This duplicates one table
  temporarily — flagged in docs/TODO.md to repoint the helper later (kept out of
  scope to avoid touching metadata-gen).
- **Prompt version = 12-char sha256 of the system prompt, `@lru_cache`d.**
  Computed per-call at the router (the router only sees the string, not the
  caller's module constant) — cache makes it effectively "once per prompt".
- **Payload capture (§2.4, resolves OQ #6):** env var
  `LLM_PAYLOAD_CAPTURE_SAMPLE_RATE` (float, read once at import; default 1.0).
  Full prompt+response on `llm.payload` span **events** (size + sampling);
  Bernoulli per successful call; always-on-error prompt capture when rate > 0.

### Testing Notes
No unit tests (test-boundaries rule). Verified via an in-memory-exporter harness
(scratchpad) driving a fake provider through all four states — clean/recovered/
exhausted/unpriced — asserting span status, attempt_count, gen_ai attrs, cost,
retry/exception/payload events, and prompt-only-on-failure. `py_compile` clean;
module imports clean (tracer is a no-op `ProxyTracer` when `setup_tracing`
hasn't run, so offline ingestion/eval imports pay nothing). Risk/edge cases for
the manual pass: confirm `llm.generate` spans appear under a real `/query_search`
trace and nest sensibly; the Gemini search-flow models resolve a cost (others
like `qwen-plus`/`claude-opus-4-6`/`gpt-5.4` are unpriced → warning, expected);
setting the rate to 0 removes payload events.

## Scope pricing table to live serving models + embedding
Files: implementation/llms/pricing.py | Replaced the stale table (seeded from the offline batch-estimate list) with only the models the live search path routes to — `gemini-3.5-flash` (steps 0/1/2), `gemini-3-flash-preview` (implicit expectations), `gpt-5.4-mini` (step 3 + query generation + entity flows) — plus `text-embedding-3-large` for embeddings. LLM prices set to `(0.0, 0.0)` placeholders (real values unknown; to be filled manually); embedding priced at the known $0.13/1M input, $0 output. Note: `0.0` placeholders report `llm.cost_usd=0.0` rather than tripping the unpriced-model warning, so they must be remembered and updated.

## Switch implicit-expectations model to gemini-3.5-flash
Files: search_v2/implicit_expectations.py, implementation/llms/pricing.py | Replaced the only `gemini-3-flash-preview` use with `gemini-3.5-flash` (consolidating the search flow onto one Gemini model) and dropped the now-unused pricing row. Live serving set is now `gemini-3.5-flash`, `gpt-5.4-mini`, `text-embedding-3-large`.

## Cache-aware LLM cost: capture cached tokens + price the discount
Files: implementation/llms/pricing.py, implementation/llms/generic_methods.py, observability/names.py, implementation/llms/query_understanding_methods.py

### Intent
Providers bill input tokens served from their prompt cache at a discount, but the cost calc ignored it (flat input rate) and the cached count was thrown away (Gemini only `print`ed it). Now the router captures cached-input tokens from every provider and prices them at a separate discounted rate.

### Key Decisions
- Pricing tuple widened `(input, output)` → `(input, cached_input, output)` per 1M; `compute_llm_cost_usd` gains `cached_input_tokens=0` and computes `cached*cached_rate + (input-cached)*input_rate + output*output_rate`, clamping uncached at ≥0. cached is a SUBSET of input (OpenAI/Gemini/compatible accounting), not additive. Cached rates seeded as `0.0` placeholders for the user to fill (user set gpt-5.4-mini to 0.075).
- Cached extraction centralized in one defensive helper `_extract_cached_tokens(usage)` probing `prompt_tokens_details.cached_tokens` (Chat Completions/compatible), `input_tokens_details.cached_tokens` (Responses API/WHAM), and `cached_content_token_count` (Gemini), else 0. Anthropic deliberately not probed — it reports cache reads separately and does NOT fold them into input_tokens, breaking the cached⊆input model (and it's unpriced anyway).
- Contract: all 7 ASYNC provider fns now return a uniform 4-tuple `(parsed, input, output, cached)` so the router unpacks any provider identically. The 2 SYNC helpers (`generate_openai_response`, `generate_kimi_response`) stay 3-tuple — they're an offline/ingestion path off the cost telemetry (sync openai has 8 vector_metadata callers expecting 3-tuple). Router keeps its own 3-tuple return to the pipeline, so no search_v2 caller changes.
- New span attr `llm.cached_input_tokens` (registered in names.py under the `llm.` root — no stable GenAI semconv key exists yet), set on every llm.generate span (even 0) so cache-hit rate is queryable; cost is now cache-adjusted. Dropped the Gemini debug `print`.
- Updated the one non-test direct async caller (`query_understanding_methods.py`, 5 kimi unpacks → `parsed,_,_,_`).

### Testing Notes
Verified: py_compile clean on all four files; cost math (1000in/400cached/200out gpt-5.4-mini = $0.00138 vs $0.00165 uncached; unpriced → None). NOT yet run against live provider responses — the per-provider cached field paths in `_extract_cached_tokens` are best-effort and should be confirmed in a real trace (esp. WHAM Responses API and Qwen/DashScope). `unit_tests/test_generic_methods.py` + `test_query_understanding_unpacking.py` assert the old 3-tuple provider contract and will fail until updated in the testing phase (per test-boundaries rule, not touched here).

## Move cached tokens to gen_ai key + request-level /query_search cost rollup
Files: observability/cost_tracking.py (new), observability/names.py, implementation/llms/generic_methods.py, api/main.py, observability_context/observability_architecture.md, observability_context/query_search_planning.md, docs/modules/observability.md

### Intent
Two follow-ons to the cache-aware cost work. (1) The GenAI semconv has since added `gen_ai.usage.cache_read.input_tokens`, so cached tokens no longer need a project-authored `llm.` key. (2) Per-call `llm.cost_usd` existed but nothing summed a whole `/query_search` request; the user wanted a request-level total on the main (server) span, counting every LLM attempt that returned a token count (including retried/failed-but-billed attempts) plus every embedding call.

### Key Decisions
- Cached tokens moved `llm.cached_input_tokens` -> standard `gen_ai.usage.cache_read.input_tokens` (inline string constant at the call site, not authored in names.py per the standard-root rule). Removed `LLM_CACHED_INPUT_TOKENS` from names.py + updated the LLM-root comment (dot still justified by attempt_count/cost_usd/prompt_version). Confirmed via web search the semconv key exists and means a cache READ/hit. `llm.cost_usd` unchanged (already under `llm.`).
- Request rollup via a new dependency-free `observability/cost_tracking.py`: a `ContextVar`-held mutable `RequestCostAccumulator`, entered once with `track_request_cost()` at the TOP of the handler's `event_stream()` generator (before the pipeline spawns tasks) so every branch inherits it by reference; `asyncio.create_task` snapshots context, and the whole /query_search path is pure asyncio, so unlocked in-place `+=` is safe. `add_request_cost()` is a no-op outside a tracked request, so offline/ingestion callers are unaffected.
- To count every BILLED attempt (user requirement), accounting happens INSIDE each of the 7 async providers, right after usage extraction and BEFORE the parse/validate that can raise. Reordered Kimi/Gemini/Groq/Anthropic/WHAM (they extracted usage after validation); OpenAI/Alibaba already usage-first. One shared helper `_account_llm_call_cost(model,in,out,cached)` -> `compute_llm_cost_usd` -> `add_request_cost`. The router does NOT add (avoids double-counting the successful attempt the provider already accounted); its per-span `llm.cost_usd` stays the successful-attempt cost, and `query_search.cost_usd` is the all-attempts superset.
- Embeddings: `generate_vector_embedding` now reads `response.usage.total_tokens` (previously discarded) and accounts `compute_llm_cost_usd(model, total_tokens, 0)` (no output, no caching). `text-embedding-3-large` already priced.
- Rollup written on the server span in `event_stream()`'s `finally` as `query_search.cost_usd` (new name, stays flat per rule C, `_usd` suffix). `finally` covers partial-failure/disconnect; the ASGI server span stays open until the generator drains, so the write lands.

### Planning Context
Ties to query_search_planning.md OQ #5 (Phase-4 stream terminal): the rollup relies on the server-span-stays-open assumption, which is designed-for but NOT yet Tempo-verified. Docs updated to mark it partially exercised and to name the end-to-end check that resolves the OQ.

### Testing Notes
py_compile clean on all changed files. NOT yet run live: (1) confirm `query_search.cost_usd` actually appears on the server span (resolves OQ #5) and roughly equals the sum of per-span `llm.cost_usd` + embedding cost; (2) force a retry and confirm a billed-but-retried attempt still increments the total; (3) confirm `gen_ai.usage.cache_read.input_tokens` replaced `llm.cached_input_tokens`. `unit_tests/test_generic_methods.py` + `test_query_understanding_unpacking.py` still assert the old provider contract and reference the removed `LLM_CACHED_INPUT_TOKENS`; out of scope here (test-boundaries), to reconcile in the testing phase.

## Step 0 / Step 1 pipeline spans (query_search Bite 3)
Files: observability/names.py, search_v2/streaming_orchestrator.py, search_v2/full_pipeline_orchestrator.py, api/main.py

### Intent
First pipeline spans for /query_search's streaming path: manual `query_search.step_0` (flow routing) and `query_search.step_1` (spin generation) spans wrapping the parallel LLM pair at the head of `stream_full_pipeline`, so the router's `llm.generate` child nests under the right step and the waterfall shows the two overlapping. Resolves the design questions worked through this session on which attributes each step earns.

### Key Decisions
- Attributes (kept deliberately minimal — everything else is free on the nested llm.generate child): step_0 carries `query_search.step_0_flows` (activated flow names — the entity flow's SearchFlow value + "standard" when it co-fires; never empty on success) and `query_search.step_0_standard_branch_count` (the standard-flow budget, ALWAYS set on success, 0 = standard didn't fire). step_1 carries only `query_search.step_1_unused` (bool).
- `unused` is derivable from step_0's two attrs + clarification presence, but recorded directly at the user's request: the verdict is legible on the span itself instead of reconstructed across sources. It means "routing left no budget for spins" (`not _step1_needed`); a needed-but-failed Step 1 is a separate degradation read off the nested llm.generate ERROR status, not this attr.
- Considered and REJECTED: int-enum for flows (kept human-readable strings — tiny cardinality, array attr so contains-only matching, matches existing string-enum convention); spin texts on step_1 (used spins already appear on branch spans, full output already in the llm.payload event); a wrapper span over [step_0 ‖ step_1] (no independent work/failure — violates the "genuine sub-units of work" convention; parallelism already legible as overlapping siblings).
- Namespace (OQ #1): spans owned by `query_search`, not a shared pipeline root — /rerun_query_search reuses Step 2 -> Stage 4, NOT routing, so per rule B query_search is routing's home. Attrs flat under the query_search root (underscore leaves) to match existing query_search.* input attrs.
- Span mechanics: spans started non-current via `tracer.start_span`, activated inside each asyncio task with `use_span(end_on_exit=False)` so the LLM child parents correctly while the span outlives run_step_1 (needed because `unused` is only known after step_0 returns). step_0 span closed at its true completion (accurate duration); step_1 span closed after resolution — duration = launch->resolution (consumed or cancelled), which overstates only when step_1 finished before step_0, i.e. when it is NOT the long pole. Fatal step_0 marks its span ERROR + record_exception. A try/finally closes any span left open on the client-disconnect path (is_recording() guards double-end).
- Parenting fix in api/main.py: wrapped `event_stream()`'s pipeline loop in `use_span(request_span, end_on_exit=False)` so the server span is the current parent for the whole streamed pipeline. During SSE iteration the server span is alive but not guaranteed current (same reason the cost rollup writes through a captured handle); this makes step spans AND the existing llm.generate spans nest under the server span rather than orphaning as roots. Rides the same server-span-stays-open assumption as the cost rollup (query_search_planning.md OQ #5) — not a new dependency.
- DRY refactor: added `_standard_branch_count(step0)` as the single source of truth for the 3-minus-non-standard budget; `_step1_needed` and `_plan_step2_branches` now read it (provably equivalent to the prior inline `3 - _non_standard_firing_count` expressions). It also serves as the telemetry value.

### Testing Notes
py_compile clean; import-checked new names/helpers/enums; validated `_activated_flow_names`/`_standard_branch_count`/`_step1_needed` across 5 constructed Step0Response cases (none_of_the_above, entity-only, entity+ambiguity, specific_title, similarity) — all match the designed flows/budget/unused values. NOT yet Tempo-verified (Bite 3 verification, like Bite 1, is manual): confirm a real /query_search trace shows step_0 and step_1 as overlapping children of the server span with the llm.generate child nested under each; `unused=true` on an entity-only query; step_0 ERROR + green request on a forced routing failure. Risk: the api/main.py use_span wrapper touches the streaming boundary flagged in OQ #5 — the end-to-end check (do step spans actually parent to the server span?) is the concrete resolution of that OQ.

## Step-0-fatal outcome verdict on /query_search (early Bite 2 slice)
Files: api/outcome.py, api/main.py, observability_context/observability_architecture.md, observability_context/query_search_planning.md

### Intent
Record `outcome.success=false` on the server span when Step 0 fails fatally. Previously nothing landed: `@record_outcome(success_on_return=False)` only writes on exceptions that bubble out of the handler, but a fatal Step 0 is caught inside stream_full_pipeline and delivered as a terminal SSE `error` event AFTER the handler already returned the StreamingResponse — so no verdict was recorded for the failure the user was probing.

### Key Decisions
- Write from the API layer, not the orchestrator: `event_stream()` watches the `(event_name, payload)` stream and, on `event_name == "error"`, sets `outcome.success=false` + failure_reason on `request_span` (the handle it already holds for the cost rollup). Keeps the orchestrator transport-agnostic (it keeps emitting the same `error` event); puts the outcome write where outcome + request_span ownership already live. Keying on `error` is precise — the orchestrator emits it ONLY on the Step 0 fatal path; per-fetch failures ride `branch_error` inside `branch_results`, not an `error` event.
- New FailureReason `query_understanding_failed` (resolves OQ #2), distinct from `internal_error`: a fatal Step 0 is upstream LLM/provider retry-exhaustion, not a bug in our code — different action to take. Low-cardinality, metric-label-safe like the rest of the enum.
- Scope: FAILURE verdict only (resolves OQ #3 partially). The success verdict + remaining stream-end rollups (total result count, fetch/failed-branch counts) stay deferred to the rest of Bite 2; the happy path is still absent under success_on_return=False. This is a coherent increment — we record the failures we can hook, leave success for the stream-completion mechanism.
- No double-write risk: the decorator writes nothing on the streaming path's clean return (success_on_return=False), so the manual failure write in event_stream is the only outcome write.

### Testing Notes
py_compile clean; confirmed the new enum member serializes as `query_understanding_failed`. With the TEMP always-fail injection active, a /query_search should now show the server span carrying outcome.success=false + outcome.failure_reason=query_understanding_failed alongside the existing `error` SSE event and the ERROR query_search.step_0 span. NOT yet Tempo-verified. Note the request/server span itself stays non-ERROR (auto-instrumentation status untouched) — only the semantic outcome.* attributes mark the failure, per the §5.2 contract.

## Removed retry-testing scaffold; Steps 0/1 observability section complete
Files: implementation/llms/generic_methods.py, search_v2/step_0.py, search_v2/step_1.py, observability_context/observability_architecture.md, observability_context/query_search_planning.md
Removed the TEMP `_debug_fail_once` param + one-shot failure injection from the router and the two call-site opt-ins (step_0/step_1) after the user thoroughly tested the retry-recovery and retry-exhaustion paths end to end. Marked Bite 1 + Bite 3 landed+verified and the Step-0-fatal outcome verdict done+verified in observability_architecture.md (§1 table, §6 catalog + outcome table) and query_search_planning.md (header status, Bite 2/3, OQ #2 resolved / OQ #3 partial). Still open: Bite 2 success verdict + stream-end rollups + OQ #5 walkthrough, and Bites 4–9.

## Entity-flow (non-standard branch) observability — 1c-1 Bite 8
Files: observability/names.py, search_v2/streaming_orchestrator.py, search_v2/person_search.py, search_v2/similar_movies.py, search_v2/exact_title_search.py, search_v2/studio_search.py, search_v2/endpoint_fetching/studio_query_execution.py, search_v2/non_character_franchise_search.py, search_v2/character_franchise_search.py, search_v2/testing_nonstandard_flows.ipynb, observability_context/query_search_planning.md, observability_context/observability_architecture.md

### Intent
Make the six entity flows (person / similarity / exact_title / studio / non_character_franchise / character_franchise) legible in traces. Unlike a standard branch, an entity flow empties only when something concrete broke (name unresolved, wrong entity, filters, catalog gap), so each flow now records what it searched, how it resolved, and how the result was composed — hung on the existing `query_search.branch` span (Bite 4) plus a few sub-operation child spans.

### Key Decisions
- Instrument INSIDE each entity-flow entry executor (run_*_search), not the shared Stage-4 executors: those entry fns run under the branch span (via `_run_under_span`), so `trace.get_current_span()` there IS the branch span. Setting attrs in `execute_studio_query` / `fetch_person_buckets` would pollute standard-branch / attribute_search call sites; keeping them in the entity modules confines the entity-flow `branch_*` attrs to entity flows (and no-ops outside a traced request).
- Universal skeleton split: `branch_result_count` + the empty span EVENT are stamped in the orchestrator wrappers (`_stamp_branch_outcome`, post-hydration where the card count is known); `branch_entities` + `branch_entity_resolved_counts` + `branch_unresolved_entity_count` + all per-flow attrs are set in the executors (where resolution internals live and alignment is guaranteed). Aliases are ALWAYS-ON (user's call — low traffic favors depth over the earlier empty-event-only lean).
- Studio brand/freeform yield split: added an opt-in `path_match_counts` mutable-dict side channel to `execute_studio_query`/`_execute_any` (default None → no behavior change for standard-branch callers). The entity flow passes a dict and reads brand vs freeform match counts; brand refs with brand_count==0 is the silent dead-end (brand wins per ref, no fall-through). `_translate_studio_query` now returns `(spec, llm_fallback)`.
- Similarity weave seats: `_weave_candidates` now returns `(woven, seats_by_bucket)` counting top-section placements per weaver bucket; surfaced by `_build_results` (the single weaver call site, under the branch span). `_resolve_similarity_anchors` now returns per-reference resolution flags for the requested-vs-resolved skeleton. active_anchor_types + low_cohesion_fallback read off the existing result/debug.
- Child spans only where parallel work genuinely warrants latency/nesting: person per-entity resolution (one per `fetch_person_buckets`), character_franchise's two parallel resolutions (franchise — with the sequential lineage-mainline split folded in — and character), and the similarity Qdrant probe (the gRPC auto-instrumentation gap). Non-char-franchise/studio/exact-title get attributes only.
- Bucketed flows record `branch_top_tier` + count (top populated tier) rather than a full histogram (per design discussion: prominence-tier distribution isn't worth monitoring; the top-tier occupancy catches wrong-entity / thin-coverage). exact_title records source composition by RETRIEVAL MECHANISM (seed/close/fanout always-on, title_only conditional on a supplied year) — that IS worth it because the sources differ in trust.

### Planning Context
Implements the consolidated per-flow spec aligned over the prior design discussion (see query_search_planning.md §3 Phase 2 entity flows). Names follow the flat `QUERY_SEARCH.child("branch_*")` precedent (rule C), with dot-namespaces only for the two genuine groups (`branch_weave_seats.*`, `branch_source.*`).

### Testing Notes
py_compile + import-checked all 9 modules; verified every new name constant resolves; smoke-tested the recording helpers (exact_title sources/entities incl. conditional title_only + year, person top-tier, weave seats 5-emitted-0-safe, nc-franchise buckets, char-franchise tiers/forms, studio brand/freeform/aliases) under an InMemorySpanExporter — all attributes correct. NOT yet Tempo-verified end to end: the notebook `search_v2/testing_nonstandard_flows.ipynb` (in-process `stream_full_pipeline` + `setup_tracing(FastAPI())` export) is the verification vehicle — one parameterized cell per flow, each confirming the intended fetch type fired. Confirm in Tempo: branch_* attrs per flow, person per-entity spans, char-franchise's two resolution spans, similarity Qdrant span + weave seats, and the `entity flow empty` event on a zero-result query. Risk: attribute alignment for studio `entity_paths`/`brand_names` is keyed to the LLM's StudioRef list (usually 1:1 with canonical names, not guaranteed); `branch_entities` stays the canonical names.

## Qdrant probe-kind spans for the SEMANTIC endpoint (Bite 6) + standalone test notebook
Files: observability/names.py, search_v2/endpoint_fetching/semantic_query_execution.py, search_v2/test_semantic_qdrant_span.ipynb

### Intent
Close the gRPC auto-instrumentation blind spot on the SEMANTIC endpoint by adding a manual OTel span to each of the three `query_points` primitives, and give the user a notebook to exercise the endpoint and verify the spans in Grafana Tempo.

### Key Decisions
- One span (`query_search.semantic_qdrant`) per primitive, discriminated by a `probe_kind` attribute (`QdrantProbeKind` str-enum owned by the call-site module per names.py rule E) — NOT one span name per primitive, and NO per-vector-space wrapper spans (locked "no double-wrap" decision protecting the ~130-150 span/request budget). Parenting is automatic: the primitives run under `asyncio.gather`, and asyncio Tasks copy the active OTel span via contextvars, so probes nest under the ambient span (dispatch span in prod, notebook span standalone).
- Attributes (5): probe_kind, vector_space, limit, filter_active, hit_count. `filter_active` = whether the USER HARD FILTER was applied (user's call this session): True only on the pool probe, False on calibration and on the HasId reranker (its HasIdCondition is a pool restriction, not the hard filter). Chosen over "drop it" and over "any query_filter present" — the latter would have conflated HasId with the hard filter.
- Default span settings (record_exception/set_status_on_exception left on): a throwing Qdrant call auto-marks its probe span ERROR + records the exception, which is correct at the probe level with zero extra code.
- Retry/soft-fail in `execute_semantic_query` now emits `semantic_query_retry` / `semantic_query_failed` span EVENTS on the ambient span (`trace.get_current_span()`), converting the previously log-only silent soft-fail into something queryable. Names follow the `semantic_qdrant.*` dotted namespace (rule C: 5 grouped attrs).
- Notebook uses the app-free tracing bootstrap from scripts/otel_smoke_test.py (setup_tracing needs a FastAPI app); service.name=query_search_notebook; tracing cell must run before the search_v2 imports (module tracers captured at import).

### Follow-up dependency (Bite 5)
`branch_error` attribute and the request-level failed-branch COUNT are deliberately out of scope here — they belong to the branch/dispatch span created in Bite 5 (Stage 4 execution). Until Bite 5 lands, the retry/failed events attach to the nearest existing ancestor (server/Step span) in the live pipeline, and there is no branch_error attribute or failed-branch rollup. Wire those onto the dispatch span when Bite 5 is implemented.

### Testing Notes
Import-checked both modules; all new name constants resolve; notebook validates under nbformat. NOT yet Tempo-verified end to end — the notebook is the verification vehicle: run with grafana/otel-lgtm up + QDRANT_COLLECTION_ALIAS/OPENAI_API_KEY set, then confirm in Tempo the notebook.semantic_query parent with one query_search.semantic_qdrant child per space (probe_kind=calibration on the generator path), plus the filtered (pool, filter_active=True) and reranker (hasid_score) variants, and the retry/failed events on a forced Qdrant error.

## Add per-space query_params JSON to the Qdrant probe spans
Files: observability/names.py, search_v2/endpoint_fetching/semantic_query_execution.py, search_v2/test_semantic_qdrant_span.ipynb
Why: make each probe span self-describing — read the exact query that produced its vector without cross-referencing the request body.
Approach: new attr `query_search.semantic_qdrant.query_params`; the per-space body serialized via `model_dump_json(exclude_defaults=True)` (only populated fields, mirroring what embedding_text consumes) precomputed once in `_unpack_inputs` as `_CallInputs.query_params_json`, threaded through all 3 primitives (new optional `query_params_json` kwarg) at every `_execute_*` call site. High-cardinality span-attr, never a metric label (rule F). Verified with an in-memory exporter: 2-space carver query → 2 calibration probe spans each carrying its space's body JSON.

## Reconcile observability context docs with the shipped Bite 6 work
Files: observability_context/observability_architecture.md, observability_context/query_search_planning.md
What: architecture doc — new §6 `query_search.semantic_qdrant` catalog subsection (6 attrs + retry/failed events), status-table row (Bite 6 ✅), one-line summary / §3 Qdrant-gap / /query_search intro / §8 all updated to stop claiming Qdrant is uninstrumented on /query_search. Planning doc — Bite 6 checked off with the as-built note (built ahead of Bite 5; branch_error deferred), and the Bite 6 attribute list finalized (probe_kind/vector_space/query_params/limit/filter_active/hit_count; filter_active = hard-filter-applied). No code change.

## Similarity branch observability — Tier 1 gaps (anchor shape, lane counts) + director_signature multi fix
Files: observability/names.py, search_v2/similar_movies.py
Why: three highest-value gaps in the similarity (Step-0 SIMILARITY_TO_TITLES) branch telemetry, chosen against the "depth over cardinality at low traffic" value function. `active_anchor_types` collapses every ordinary film to `standard_shape` regardless of reach, so the obscure-vs-blockbuster axis the user kept reaching for was untraceable; nothing distinguished a retrieval/tagging gap (lane returned 0 candidates) from a ranking miss (lane returned many that scored low); and the multi-anchor branch silently omitted `director_signature` from `active_anchor_types`, making the one already-trusted classification attribute mean different things single vs multi.
Approach:
- Anchor reach×quality shape: surfaced the already-computed `anchor_shape_cohesion` dict ({shape: M_s/N}) onto `SimilarMoviesDebug` (new field, populated at both the single- and multi-anchor return sites — single is {shape:1.0}/empty, multi is the cohort fraction). Recorded as two index-aligned span arrays `branch_anchor_shape` (str[]) + `branch_anchor_shape_cohesion` (float[]), dominant shape first. Chose the uniform two-array shape over a scalar-for-single / dict-for-multi split so single and multi are queryable together; empty arrays are a legitimate "shapeless anchor" signal, not a missing attribute.
- Per-lane candidate counts: recorded the `candidate_counts_by_lane` dict (already on `result.debug`, previously discarded) as `branch_candidate_lanes` (str[]) + `branch_candidate_counts` (int[]), lane-name-sorted. Pure plumbing of an existing value — the retrieval-vs-ranking triage signal.
- director_signature multi fix: append `director_signature` to the multi-anchor `active_anchor_types` when `has_auteur_anchor` (union of anchor directors ∩ curated auteur set) — mirrors the single-anchor `anchor_directors & auteur_term_ids` gate. Correctness fix, not just telemetry.
All recording lives in `_record_similarity_signals` (runs under the `query_search.branch` span). Deliberately deferred (Tier 2/3): the seven-multiplier stack (per-candidate detail already in `LaneEvidence`; only a systemic rollup would belong on the span), vector/lane weights + cohesion, franchise fatigue, format lock, shorts_dominant, per_anchor_active_anchor_types — all already in the debug payload and either high-cardinality or low-aggregation-value at current traffic.
Testing notes: py_compile + import clean; all four new name constants resolve; smoke-tested `_record_similarity_signals` under InMemorySpanExporter — multi cohort (2 prestige + 1 blockbuster) emits shape arrays dominant-first, lane arrays index-aligned (franchise=0 distinguishable from themes=143), shapeless single-anchor emits empty arrays, director_signature present in active_anchor_types. NOT yet Tempo-verified end to end (needs live services + otel-lgtm; the testing_nonstandard_flows notebook similarity cell is the vehicle).

## Similarity branch: shapeless-shape sentinel + director_signature semantics note
Files: search_v2/similar_movies.py, observability/names.py, observability_context/observability_architecture.md
Why: `branch_anchor_shape` emitted empty arrays for the common shapeless anchor (reception 50–80, sub-100K reach, no award shift — e.g. Blue Ruin, Barbie), and empty-array span attributes don't render in trace UIs, making "shapeless" indistinguishable from "instrumentation didn't run".
Approach: when `anchor_shape_cohesion` is empty, `_record_similarity_signals` now emits the explicit `["shapeless"]`/`[1.0]` sentinel (module const `_ANCHOR_SHAPE_SHAPELESS`) instead of empty arrays. `shapeless` is not one of the five real shapes, so the token is unambiguous and the verdict is always a visible, first-class value. Verified under InMemorySpanExporter (shapeless→`('shapeless',)/(1.0,)`, shaped→unchanged).
Note (no code change): `director_signature` in multi-anchor `active_anchor_types` means "≥1 anchor directed by a curated auteur" (`has_auteur_anchor` = union-of-anchor-directors ∩ auteur set), NOT "anchors share a director". Interstellar (Nolan, curated) makes it fire regardless of the other anchors — correct, and identical to single-anchor semantics. The "anchors agree on a director" quantity is director cohesion (`cohesion_by_lane["director"]`), which is a different, currently-untraced signal.

## Step 2 / Step 3 / query-generation spans on /query_search (1c-1 Bite 4, partial)
Files: observability/names.py, search_v2/full_pipeline_orchestrator.py, search_v2/endpoint_fetching/category_handlers/handler.py, observability_context/observability_architecture.md, observability_context/query_search_planning.md

### Intent
Give the standard-branch trait pipeline a per-trait waterfall from Step 2 through query-generation completion, with low-cardinality summary attributes that stay queryable when LLM payload sampling is dialed down. Scope stops at query generation — Stage 4 execution stays a separate later branch-level span (the pool is a branch-level union, not per-trait, so execution can't nest under a trait span; correlate it by a trait_ref attribute in a future bite).

### Key Decisions
- New spans (all `query_search.*`, flat leaves per rule C, under the branch span): `step_2` (one per standard branch), `trait` (one per Step-2 trait), `step_3` (wraps the Step-3 LLM call), `query_generation` (one per handler-LLM call). step_2 and the trait spans are siblings under the branch (step_2 closes at its LLM return; trait spans start after).
- Attributes: step_2 → `step_2_trait_count` + `step_2_contextualized_phrases[]` (dropped negative_trait_count per user). trait → `trait_phrase` / `trait_polarity` / `trait_commitment`. step_3 → `step_3_combine_mode` + `step_3_categories[]` recorded POST-SOLO-trim (the calls that actually reach retrieval, not the raw committed set). query_generation → `query_generation_category` + `query_generation_endpoints[]` (the EndpointRoute names that fired — just which endpoints activated; params ride the nested `llm.generate` payload; empty `[]` when nothing fired).
- SOLO trim (`_decompose_and_generate`) now also emits a `"solo trim"` span EVENT on the trait span (kept category + dropped count) — previously log-only.
- step_3 soft-fail (retries exhausted): `trait_step_3_error` attribute on the trait span (degradation — trait span stays UNSET), step_3 span marked ERROR; step_2 soft-fail marks the step_2 span ERROR + records exception, branch degrades via existing branch_error. Neither flips the request verdict.
- query_generation span wraps ONLY the handler-LLM branch of `run_query_generation`; EXPLICIT_NO_OP / NO_LLM_PURE_CODE return before it, so deterministic/no-op calls get no span (per design).
- Plain `with tracer.start_as_current_span(...)` context managers (not the step_0/1 non-current `use_span` dance) — these units start/end in one scope and already run with the branch span current; nesting is automatic. Added a module tracer to full_pipeline_orchestrator.py and handler.py (neither had OTel before). Enum emission: polarity/combine_mode via `.value` (StrEnum), commitment is a Literal[str] set directly, category names via `.name`.

### Testing Notes
Syntax-checked all three modules; names.py constants resolve to the expected dotted strings; handler.py imports clean with a live tracer. full_pipeline_orchestrator could NOT be import-tested end to end because of a PRE-EXISTING, UNRELATED breakage in the uncommitted Bite 8 work: `search_v2/similar_movies.py:2508-2512` references per-bucket constants `QUERY_SEARCH_BRANCH_WEAVE_SEATS_{BEST_OVERALL,AUTEUR,FRANCHISE,RARE_KEYWORD,LEAD_ACTOR}` that don't exist in names.py (only the consolidated `QUERY_SEARCH_BRANCH_WEAVE_SEATS` json-map constant does) and aren't imported — a NameError at module load. Not caused by and not in scope of this change; flagged for the Bite 8 author. NOT yet Tempo-verified end to end — verification vehicle is a live `/query_search` with grafana/otel-lgtm up (see plan verification section): confirm per-branch step_2 + N sibling trait spans, each trait span parenting a step_3 (combine_mode + post-trim categories) and one query_generation per LLM handler call, the "solo trim" event on a SOLO-with-extras trait, trait_step_3_error + step_3 ERROR on a forced Step-3 failure with the request still succeeding, and no query_generation span for a NO_LLM_PURE_CODE category.

## Redesign similarity-flow branch telemetry into JSON-string maps (1c-1 Bite 8, similarity)
Files: observability/names.py, search_v2/similar_movies.py, observability_context/observability_architecture.md, observability_context/query_search_planning.md

### Intent
The similarity `query_search.branch` span attributes had grown into hard-to-read index-aligned parallel arrays (`branch_candidate_lanes`+`_counts`, split `branch_anchor_shape`/`_cohesion`), five separate `branch_weave_seats.*` ints, and a `branch_active_anchor_types` array that silently meant different things single vs multi. Reorganized the whole set around four reader questions (traits marked important / fetch avenues + counts / scoring weights / final-weave paths) and switched map-shaped signals to single JSON-string attributes.

### Key Decisions
- JSON-string maps over parallel arrays: empirically confirmed OTel span attributes accept only `str|bool|int|float|Sequence[those]` — a raw `dict` is dropped with a warning; a JSON string is kept and renders readably (label+value side by side). Accepted tradeoff: no numeric TraceQL filter on an inner key (these are for reading traces, not per-lane alerting). Added `_set_json_map` helper (key-sorted, stable output) + `import json`.
- Final attribute set — both flows: `branch_retrieval_lanes` (JSON {lane:count}), `branch_retrieval_total` (int), `branch_lane_weights`/`branch_vector_space_weights` (JSON maps), `branch_weave_seats` (JSON {bucket:seats>0}), `branch_low_cohesion_fallback` (bool), `branch_additional_boosts` (JSON array, omitted when empty). Single-only: `branch_shape_modifiers` (JSON array, always set, "[]" when none), scalar `branch_anchor_shape` ("none" when shapeless). Multi-only: `branch_anchor_shape_cohesion` (JSON {shape:frac} with a "none" key so it sums to 1), `branch_lane_cohesion`, `branch_vector_space_cohesion`.
- Retrieval counts are a NEW fetch-side signal (`debug.retrieval_counts_by_lane` + `retrieval_total`), distinct from the scoring-side `candidate_counts_by_lane` (left untouched — consumed by `run_similar_movies_batch.py`). Built at each flow's fetch site keyed on the lane's existing fetch-gate predicate (single: seed non-empty; multi: the `cohesion>0`/auteur/consensus gates), so a fired-but-empty lane is present at 0 and a gated-off lane is absent. Also surfaced multi `cohesion_by_lane` → `debug.lane_cohesion`.
- `branch_additional_boosts` per user decision: the auteur multiplier isn't recoverable from the fetch map (the director lane fires on ANY director, not just auteurs) nor from the weights (director lane is weight-0), so it gets a dedicated extensible array carrying `"director_signature"` when `"director_signature" in active_anchor_types`; omitted entirely when empty.
- Dropped `active_anchor_types` as a span attribute (its content is now covered by shape + fetch map + weights, and the single/multi ambiguity was a footgun); removed the per-bucket weave-seat constants + `_WEAVE_SEAT_ATTR_BY_BUCKET`, collapsed `_record_weave_seats` to one JSON map. This also RESOLVES the pre-existing module-load breakage the Bite-4 DIFF_CONTEXT entry flagged (similar_movies referenced per-bucket constants that had already been consolidated) — `search_v2.full_pipeline_orchestrator` and `search_v2.similar_movies` now both import clean.

### Testing Notes
Syntax + import verified (both modules clean); names.py constants resolve to expected dotted strings. Focused InMemorySpanExporter smoke test (scratchpad) exercised `_record_similarity_signals` + `_record_weave_seats` on fabricated single/multi/shapeless results and asserted: single emits shape_modifiers + scalar shape and NOT the cohesion maps; multi emits the three cohesion maps (shape map sums to 1 incl. a "none" fraction); both emit retrieval map (fired-empty present at 0), weights, weave_seats (>0 only); additional_boosts present iff auteur, omitted otherwise; every map value round-trips through json.loads. NOT yet Tempo-verified end to end — vehicle is the `testing_nonstandard_flows.ipynb` similarity cell with grafana/otel-lgtm up.

## Similarity weave telemetry: log desired bucket-target ratio instead of realized seats
Files: search_v2/similar_movies.py, observability/names.py, observability_context/observability_architecture.md, observability_context/query_search_planning.md
Why: `branch_weave_seats` tallied which bucket *drew* each top-section slot. For franchise-dominant cohorts (e.g. two Avengers anchors) it collapsed to `{best_overall: 10}` even with strong franchise overlap, because multi-bucket full credit routes signal-bucket films in via best_overall — so the seat map read as "franchise did nothing" when franchise was highly influential. The desired allocation (`_compute_bucket_targets` output) answers the more useful reader question: what the weave *meant* to reserve per bucket.
Approach: `_weave_candidates` now returns the pre-weave `target` dict instead of accumulating `seats_by_bucket` (removed the tally init + per-placement increment — it had no other consumer). Renamed `_record_weave_seats` → `_record_weave_targets` and the attribute/Name `branch_weave_seats`/`QUERY_SEARCH_BRANCH_WEAVE_SEATS` → `branch_weave_targets`/`QUERY_SEARCH_BRANCH_WEAVE_TARGETS` (keeping the "seats" name for target data would misrepresent it). Recorder emits every bucket present in `target` (best_overall always; instantiated signal buckets; a bucket that instantiated but rounded to 0 slots is kept as signal), ordered by ALL_BUCKETS with best_overall first. A signal bucket absent from the map never instantiated.
Testing notes: updated scratchpad smoke_similarity_signals.py to drive `_record_weave_targets` and assert the target map (present keys incl. 0 kept, ALL_BUCKETS order) — all asserts pass. AST + import clean. Not Tempo-verified end-to-end.

## Per-lane candidate-fetch spans for the similarity flow
Files: search_v2/similar_movies.py, observability/names.py, observability_context/observability_architecture.md, observability_context/query_search_planning.md
Why: The branch span records per-lane retrieval *counts* but not which concrete IDs each lane matched on, and the auto-instrumented Postgres span parameterizes the IN-list away — so a reader can't see WHY a lane returned what it did. Add one manual span per fetch that names the lane and records its match values.
Approach: New `query_search.similarity_fetch` span (Names: `.lane` str, `.match` json, `.result_count` int). Helper `_traced_lane_fetch(lane, match, coro)` wraps a fetch coroutine, sets lane + JSON match + result_count, awaits and returns the result unchanged (exception-transparent). Applied to director/franchise/studio/source/quality/themes_recall/rare_medium in both `_run_single_anchor_similarity` and `_run_multi_anchor_similarity`; the qdrant shape probe is excluded per request (its params never vary; it already has `similarity_qdrant`). Each wrap is gated on the SAME seed/cohesion predicate as `retrieval_counts`, so a span exists iff the lane's query fired. `match` is a JSON-string object (keys vary by lane: lineage_entry_ids for franchise, company_ids for studio, bucket+limit for quality, per-kind ID legs for themes_recall) — OTel can't hold a dict; ID lists are high-cardinality span-only values. Match builders `_franchise_match` / `_themes_recall_match` sort IDs and omit empty dimensions.
Concurrency note: the lanes run in `asyncio.gather`, which wraps each coro in a Task that copies the current OTel context at creation (branch span current) — so each lane span parents to the branch span with no cross-lane context bleed, and each lane's own SQL span nests under its lane span. Verified with an InMemorySpanExporter gather test (3 lanes → 3 lane spans under branch, 3 db spans each under their own lane span).
Testing notes: scratchpad smoke_lane_fetch_spans.py covers attribute set, result pass-through, exception→ERROR status, match-helper JSON shape, and the gather nesting — all pass. AST + import clean. Not Tempo-verified end-to-end.

## Instrument the similarity anchor-vector retrieve + enrich the shape probe span; capture SQL params
Files: search_v2/similar_movies.py, observability/names.py, observability/tracing.py, observability_context/observability_architecture.md
Why: The similarity flow makes two Qdrant calls but only the shape probe (`similarity_qdrant`) was spanned — the `_load_anchor_vectors` retrieve was an untraced blind spot, so only one Qdrant span showed up. And identical-looking `movie_card WHERE movie_id = ANY($1)` SQL spans (anchors vs candidates vs hydration) were indistinguishable without their bound params.
Approach: (1) Kept ONE `similarity_qdrant` span name and added a `probe_kind` discriminator (`SimilarityQdrantProbeKind` enum: anchor_vectors | shape), mirroring the semantic path's `QdrantProbeKind`. The retrieve now gets a span (probe_kind=anchor_vectors, requested_count/returned_count — returned<requested flags a missing anchor); the shape probe gained space_count/spaces/limit_per_space/filter_active/hit_count/hits_by_space (batched across N spaces, so it describes the batch rather than semantic's single vector_space). Leaf vocab (probe_kind/filter_active/hit_count) matches semantic_qdrant so a reader query spans both. Wrapping the retrieve in-place does NOT touch the parallel prefetch — it's a sibling of the shape span under branch, no perf change. (2) Enabled `PsycopgInstrumentor().instrument(capture_parameters=True)` — built-in dbapi flag (not custom wrapper); sets `db.statement.parameters` on every query span. Global switch; params here are IDs/filter values (no PII).
Testing notes: scratchpad smoke_qdrant_spans.py monkeypatches qdrant_client and asserts both probe kinds' attributes (missing-anchor gap, per-space recall, filter over-fetch, filter-inactive path) — all pass. AST + import clean. Not Tempo-verified end-to-end. Note: capture_parameters is a global debugging switch — revisit before prod if any sensitive values ever flow through SQL params.

## Similarity title path: reuse resolved anchor rows instead of re-fetching them by ID
Files: search_v2/similar_movies.py
Why: A title-based similarity search issued three movie_card reads where two suffice. Title resolution (`_resolve_similarity_reference`) fetched the full signal row for every title match to rank and pick a winner, then discarded everything but the winning `movie_id`; the engine (`run_similar_movies_for_ids`) immediately re-fetched that same row (plus the other anchor) via `fetch_similarity_signal_rows(anchor_ids)`. Same function, same columns — a pure re-read of rows already in hand, and a duplicate movie_card span in every similarity trace.
Approach: Modeled the two entry paths explicitly as "two ways to get populated anchor rows": (1) titles → resolve to winning rows, (2) raw IDs → fetch rows. `_resolve_similarity_reference` now returns the winning row dict (was `int | None`); `_resolve_similarity_anchors` returns `(anchor_ids, anchor_rows, per_ref_resolved)`, using the anchor_rows dict as its dedupe set while keeping first-seen order via a parallel list. `run_similarity_search` threads those rows into a new `prefetched_anchor_rows` param. `run_similar_movies_for_ids` reuses supplied rows and fetches only `missing_ids`; `fetch_similarity_signal_rows([])` short-circuits to `{}` with no round-trip, so the title path fetches — and traces — nothing there. The anchor fetch stays a leg of the 4-way gather (raw-ID callers unchanged: batch runner run_similar_movies_batch.py, direct API endpoint api/main.py:1243), so it could NOT be removed outright — only made conditional. Latency is ~unchanged (that read was already overlapped with vectors/studio/director fetches); the win is one fewer Postgres query + a cleaner trace on the title path.
Testing notes: scratchpad smoke_anchor_prefetch.py monkeypatches the fetch/resolve/pipeline seams and asserts: resolution returns deduped winning rows in first-seen order; title path (rows supplied) issues only an empty anchor fetch; raw-ID path fetches all anchors; partial prefetch fetches only the missing anchor; single-anchor raw ID routes single. All pass. AST + import clean. No behavior change for raw-ID callers.
