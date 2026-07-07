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
