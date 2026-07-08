# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Consolidate per-request verdict under request.* (outcome.success/failure_reason → request.*)
Files: observability/names.py, api/outcome.py, api/main.py, docs/modules/api.md, observability_context/{observability_architecture,query_search_planning,observability_todos}.md
Why: the per-request verdict (success + failure_reason) is a request-level fact, so it now lives on the `request.*` rollup root beside cost_usd/result_count/usage.*. The `outcome.*` attribute namespace is retired entirely.
Approach: registry constants `OUTCOME_SUCCESS`/`OUTCOME_FAILURE_REASON` → `REQUEST_SUCCESS`/`REQUEST_FAILURE_REASON` (both now `REQUEST.child(...)`); the `OUTCOME = Name("outcome")` root is deleted. Amended the REQUEST comment (verdict pair is the low-cardinality label-eligible exception among the continuous measures) and the rule-C example. In docs, the specific attribute string was renamed, and the wildcard `outcome.*` phrases (which meant "the verdict") were reworded to "the request verdict" rather than blanket-swapped to `request.*` (that root now also holds cost/usage). Mechanism names kept as-is: the `api/outcome.py` module, `@record_outcome` decorator, and `FailureReason` enum are the recording mechanism, not attribute keys.
Protected docs (docs/conventions.md, docs/decisions/ADR-102) were edited directly at the user's explicit instruction, bypassing the usual human-only channels (/solidify-draft-conventions, /extract-finalized-decisions) for this rename.
Testing notes: no test references the old constants; all three edited .py files parse.

## Switch implicit-expectations LLM to gpt-5.4-mini
Files: search_v2/implicit_expectations.py | Changed the executor's provider/model from Gemini (gemini-3.5-flash, thinking_budget=0, temp 0.35) to the OpenAI gpt-5.4-mini / reasoning_effort=low / verbosity=low setup used by step_3 and the entity/query-generation callsites, for consistency. Orchestrator calls run_implicit_expectations without a model override, so it picks up the new constants automatically; gpt-5.4-mini is already priced in pricing.py.

## Docs-audit remediation (docs-auditor findings 1–8)
Files: docs/modules/api.md, docs/modules/search_v2.md, CLAUDE.md,
docs/decisions/ADR-012-*.md, docs/conventions.md, docs/PROJECT.md,
search_v2/endpoint_fetching/entity_query_generation.py; deleted
search_v2/reranking/ (8 files)
Why: A full docs-auditor staleness scan surfaced 8 doc/code drift items;
user directed applying all of them.
Approach:
- api.md: documented the seventh hard filter `keywords` (was omitted from
  the six-filter enumeration) on /query_search + /attribute_search;
  added the third CORS origin `http://localhost:3001`.
- CLAUDE.md: corrected Stage 3 endpoint count 7 → 10 (verified by
  `ls search_v2/endpoint_fetching/*_execution.py`).
- search_v2.md: added a "Deterministic Endpoint Backers" subsection +
  Key Files rows for attribute_search.py, title_search.py,
  query_input_validation.py (three live endpoint/validation modules that
  were entirely absent).
- Deleted search_v2/reranking/ — dead, broken code (every file imported
  the deleted search_v2.stage_4 package; `import search_v2.reranking`
  raised ModuleNotFoundError). No live importer (confirmed via repo-wide
  grep). Updated the now-stale stub rationale in entity_query_generation.py
  (its only former importer was reranking/dispatch.py).
- ADR-012: re-statused Proposed → Superseded + appended a 2026-07
  postscript; its roadmap diverged (Batch API shipped under
  ADR-025/036/041/044; GPT-4o-mini swap and 8→5 consolidation NOT
  adopted — pipeline grew to 12 generation types, models are
  gpt-5-mini/gpt-5.4-mini per ADR-039/043/044). ADR body preserved
  verbatim (append-only).
- conventions.md: documented the codebase-wide `extra="forbid"` default
  on wire-boundary/LLM-output Pydantic models (per ADR-102).
- PROJECT.md: added a "Secondary criterion: skill transferability" note
  acknowledging the resume-credibility factor that explicitly shaped
  ADR-101, scoped as a tooling-only tiebreaker below the four product
  priorities.
Design context: docs-awareness rule normally reserves decisions/,
conventions.md, PROJECT.md for the dedicated skill workflows; these three
were edited under explicit user direction, not autonomously.
Testing notes: docs/comment-only except the reranking/ deletion — verified
no remaining `search_v2.reranking` imports anywhere in the tree.

## Implicit-prior observability — two spans (generation + application)
Files: observability/names.py, search_v2/full_pipeline_orchestrator.py

### Intent
Instrument the implicit-prior mechanism (the deferred Bite-4
`implicit_expectations` span + Bite-7 rerank span) so the policy the LLM
proposes and the single axis the code actually applies are both queryable.

### Key Decisions
- Two per-standard-branch spans, both under `query_search.branch`:
  `query_search.implicit_expectations` (generation, wraps the policy LLM
  call in `_run_implicit_expectations_for_branch`) and
  `query_search.implicit_prior_rerank` (application, wraps
  `_apply_implicit_prior_rerank_for_branch`). Instrumenting the shared
  per-branch functions covers both the streaming API and the batch/CLI path.
- Policy direction/strength recorded on the APPLICATION span (not the
  generation span) so they sit beside `boost_axis` — proposed-vs-applied in
  one read. Generation span carries no semantic attrs of its own; the nested
  `llm.generate` child already has tokens/cost/prompt-hash/full payload.
- ADR-087 model is single-axis: `boost_axis` (BoostAxis enum) names the one
  axis that fired. Recorded the selection variables verbatim
  (`*_cap` floats, `*_active` bools), `inverse_applied`, and
  `signal_missing_count` (fired-axis NULL-signal count = the data-coverage
  risk). Span created before the gate so skipped branches still emit a
  legible span (`boost_axis=none` + `noop_reason`).
- `noop_reason` (PriorNoopReason enum) disambiguates the four no-op causes
  the active flags can't: three gate-skips return before caps/active exist.
- Failure events on both spans: `implicit_expectations_failed` (generation
  soft-fail; `error.type` = `schema_mismatch` for output-validation/Pydantic
  ValueErrors vs. the provider/timeout class) and `implicit_prior_apply_failed`
  (signal-fetch throw; annotate + re-raise, propagation unchanged).
- Two value enums (BoostAxis, PriorNoopReason) live in
  full_pipeline_orchestrator.py per names.py rule E; span/attr names in
  observability/names.py.

### Testing Notes
Rerank behavior is byte-for-byte unchanged on the happy path (same boost
math, same resort); only telemetry added. The signal-fetch try/except
re-raises, so branch soft-fail propagation is preserved. See the test
checklist handed to the user (happy-path variants: popularity-positive,
popularity-inverse, quality-fallback, quality-inverse; no-op variants:
both-axes-off, policy_unavailable, branch_error, empty_pool; failure
events on both spans; signal_missing_count with NULL signals).

## Stage 4 execution observability (Bite 5 execution slice: Phase B pool + Phase C rerank + negative dispatch)
Files: search_v2/stage_4_execution.py, observability/names.py, search_v2/promotion_tiers.py, search_v2/full_pipeline_orchestrator.py, search_v2/streaming_orchestrator.py

### Intent
Close the tracing blind spot between query generation and scoring: instrument the
Stage 4 EXECUTION that builds + scores the candidate pool. Previously the tiered
promotion loop, reranker-only fallback, neutral-seed fallback, and the per-call 25s
timeout all soft-failed with log lines only — no way in Tempo to tell why a branch
returned a thin/empty result set. Scope is pool definition + reranker dispatch +
negative dispatch ONLY; Phase D scoring / Phase E aggregation / implicit rerank /
hydration (and per-branch result_count) are a later bite.

### Key Decisions
- No wrapping `stage_4` span; spans sit directly under the existing
  `query_search.branch` span (Stage 4 already runs under it via `_run_under_span`).
  New spans use `start_as_current_span`, nesting via asyncio contextvar propagation
  (same mechanism as the Bite 6 semantic_qdrant probes).
- Six spans: `query_search.generators` (initial dispatch; raw_union_count /
  shorts_removed_count / final_pool_count), `query_search.promotion` (one per tiered
  round — tier / before / promoted_spec_count / after / shorts_removed), 
  `query_search.neutral_seed` (reason + seed_count; both empty-pool arms),
  `query_search.rerankers` (call_count / pool_count), `query_search.negatives`
  (trait_count / pool_count; span kept tight around the dispatch gather, Phase-D
  weight fold left outside), and `query_search.dispatch` (per-unique-call child;
  route / operation_type / was_promoted / result_count).
- Events: `generator_dedup` (deduped_routes list; moved out of the isEnabledFor
  guard so it fires regardless of log level), `aux_shorts_exclusion` (default shorts
  blocklist; distinct from a user negative MEDIA_TYPE trait), `reranker_fallback_
  promotion` (reason + tier; emitted from BOTH the orchestrator for the no-generator
  single-shot case and Stage 4 for each under-floor round), `thin_pool_accepted`
  (filter-active, 0<union<floor, tiers exhausted — mutually exclusive with the
  neutral_seed span which requires union==0), `dispatch_soft_fail` (every dropped
  call; TimeoutError split from other exceptions → error.type=timeout vs class name,
  span marked ERROR).
- Side channel (Bite-8 pattern): `_apply_reranker_only_candidate_fallback` /
  `_compute_branch_auxiliary` gain opt-in `fallback_outcome: dict|None`; when a tier
  is promoted the fallback records `fallback_outcome["tier"]=lowest_tier.name`. The
  tier can't be re-derived post-promotion (determine_promotion_tier returns
  NEVER_PROMOTE for a CANDIDATE_GENERATOR), and this promotion is decided pre-Stage-4
  outside the branch span's current context, so `_handle_finished_task` reads the
  dict and stamps the event on the branch_span handle. Two non-streaming call sites
  keep the default None → no behavior change.
- Reason vocabularies (PromotionReason, NeutralSeedReason) live in promotion_tiers.py
  — OTel-free, imported by both emitting modules, prevents value drift (rule E).
- Per-call trait attribution deliberately omitted from dispatch spans: dedup fan-out
  makes one dispatch serve multiple trait/category coords, so a single trait attr
  would mislead.

### Testing Notes
Behavior unchanged — spans/events wrap existing call sites without altering returns
or control flow (soft-fail still returns None). Verify in Tempo: happy path (generators
+ rerankers + dispatch nesting; embedding/semantic_qdrant under the right dispatch);
filter-active thin pool (promotion rounds with escalating tier + thin_pool_accepted or
neutral_seed); all-reranker branch (branch-span no_candidate_generators event or
neutral_seed); forced low EXECUTOR_TIMEOUT_SECONDS (dispatch ERROR + dispatch_soft_fail
error.type=timeout); negative trait (negatives span after rerankers). Edge: shorts-fetch
dispatch span parents to the branch (fetch runs before the generators span); dedup event
lands on whichever container is current (generators or a promotion round).

## Observability: Stage 4 scoring/aggregation (D+E), hydration (G), request outcome (I)
Files: observability/names.py, api/outcome.py, search_v2/stage_4_execution.py,
search_v2/streaming_orchestrator.py, api/main.py

### Intent
Completes the /query_search Stage-4 instrumentation the A–C bite deferred (see the
"later bite" notes in names.py + the negatives-span comment): scoring/aggregation
spans, hydration span, and the request-level success verdict. A trace now answers
how each trait was weighted (incl. corpus-relative rarity), the scored pool size +
whether it carried real signal, which candidates failed hydration, and whether the
request as a whole succeeded.

### Key Decisions
- `query_search.scoring` span wraps Phase D + Phase E (positive combine/weight fold,
  negative gate×fuzzy, aggregation). The existing `query_search.negatives` dispatch
  span now NESTS under scoring (negative-trait scoring includes its own dispatch) —
  chosen over a pure-compute-only scoring span because D+E is one contiguous unit and
  this keeps the diff to an enclosing block. Attributes: `scoring.trait_weights`
  (JSON array, branch order, all scored traits), `scoring.ranked_count`,
  `scoring.top_score` (base/pre-implicit-prior; ~0 flags a filler/neutral-seed pool).
- `_positive_trait_weight` now returns `_TraitWeight` (weight + commitment_multiplier
  + classification + match_count + rarity_factor) instead of a bare float, so the span
  reports the decomposition without recomputing (surface-decomposition preference).
  Rarity fields ride `trait_weights` only for pure-generator positives (the sole class
  rarity applies to); rerankers/negatives carry weight without rarity. Single call
  site; no cross-module/test callers.
- `query_search.hydration` span in streaming_orchestrator `_run_stage4_with_implicit_prior`
  wraps `fetch_movie_card_summaries`: `hydration.requested_count` /
  `.returned_count` + a `hydration missing cards` event carrying `missing_ids`
  (scored ids with no movie_card row, silently dropped). No failed-count attr
  (redundant: requested−returned, and the event enumerates them). Standard branch
  only — entity flows already report branch_result_count via _stamp_branch_outcome.
- Phase I (api/main.py event_stream): accumulate succeeded/failed branch counts +
  total_result_count from branch_results events; write `outcome.success` at clean
  completion. success = ≥1 branch executed without branch_error (empty-but-clean
  counts as success). New FailureReason.ALL_BRANCHES_FAILED when all branches errored
  OR none ran (empty plan). Guards: `fatal` preserves the Step-0
  query_understanding_failed verdict; `completed` ensures a client disconnect writes
  no verdict (cost rollup stays unconditional). New request-level names:
  succeeded_branch_count / failed_branch_count / total_result_count (flat leaves).

### Planning Context
Design walked and approved in-conversation (Phases D–I of the query-search walkthrough);
plan at ~/.claude/plans/looks-good-implement-it-eager-eclipse.md. A–C (generators/
promotion/neutral_seed/rerankers/negatives-dispatch/dispatch) + F (implicit_prior_rerank)
were instrumented in a parallel conversation and are already in the tree.

### Testing Notes
Behavior unchanged — spans/events wrap existing call sites; only `_positive_trait_weight`'s
return type changed (internal). Verify in Tempo: multi-trait query (scoring.trait_weights
shows rarity on a pure-generator trait, none on rerankers; ranked_count/top_score set;
hydration requested==returned); negative-trait query (negatives nests under scoring; neg
trait in trait_weights without rarity); hydration gap (missing cards event with ids);
success verdict on normal query (outcome.success=true, counts, total_result_count);
all-fail/empty-plan (outcome.success=false + all_branches_failed); fatal Step 0 keeps
query_understanding_failed. FOLLOW-UP: sync observability_context/observability_architecture.md
(§6 catalog + §1 status table) — scoring/hydration/outcome are the last Stage-4 bite.

## Stage 4 shorts-fetch gets a dedicated span name (follow-up to the entry above)
Files: search_v2/stage_4_execution.py, observability/names.py | `_dispatch_call` gained a `span_name` param (default `query_search.dispatch`); `_fetch_shorts_ids` passes the new `query_search.auxiliary_shorts_exclusion` name so the default shorts fetch is identifiable in the waterfall instead of reading as an anonymous dispatch (same `dispatch.*` attrs).

## Amend: query_search.scoring trait_weights trimmed to scoring-relevant fields
Files: search_v2/stage_4_execution.py | Per-trait records now carry only what
drives scoring — `trait` (surface text), `polarity`, `used_as`
(generator/mixed/reranker via new `_USED_AS_LABEL`), `weight`, and (pure-generator
only) `match_count` + `rarity_multiplier` (renamed from rarity_factor on the wire).
Dropped `commitment` and the emitted `trait_index` (kept internally only to sort
the array into branch order). Negatives report `used_as="reranker"` (gate × fuzzy,
no rarity).

## Stage 4 dispatch span: committed-query params attribute
Files: search_v2/stage_4_execution.py, observability/names.py | Added `dispatch.query_params` (compact JSON) to the `query_search.dispatch` span via a new `_query_params_json` helper: `spec.params.model_dump(mode="json", exclude_none=True)` then a recursive `_strip_generation_assist` that drops the LLM analysis/reasoning layers (exact `thinking`/`exploration`/`search_picture`/`request_overview`/`attributes`; suffixes `_exploration`/`_reasoning`/`_candidates`/`_intent`), leaving only the commitment layer the executor queries on (finalized_keywords / column_spec / space_queries / formats / franchise_names / forms / chrono fields). Purpose: read what was queried without generation-assist noise or the SQL child spans' movie-ID bloat. Best-effort (never raises); omitted for params=None (TRENDING). Verified against the shorts MediaType spec and a synthetic keyword/semantic/metadata/franchise tree.

## Restructure /query_search branch spans into 6 collapsible groups + per-stage cost
Files: observability/cost_tracking.py, observability/names.py, search_v2/implicit_prior_rerank.py (NEW), search_v2/stage_4_execution.py, search_v2/full_pipeline_orchestrator.py, search_v2/streaming_orchestrator.py, observability_context/observability_architecture.md

### Intent
Collapse each standard branch's ~11 sibling spans under `query_search.branch` into 6
groups: A `step_2`, B `decomposition`, C `candidate_generation`, D `rerankers`,
E `scoring`, F `hydration`. Two behavioral changes the restructure enables: positive +
negative rerankers now dispatch in parallel under one span (was serialized); the
implicit-prior rerank now runs inside Stage 4's scoring span (was a separate
orchestrator pass). Plus per-stage `cost_usd` on A–D.

### Key Decisions
- Per-stage cost: generalized `cost_tracking._request_cost` from a single accumulator to
  a ContextVar STACK. `add_request_cost/tokens` fan out to every frame, so the request
  root stays complete (api/main.py unchanged) while each `track_stage_cost()` child
  isolates its subtree — correct across concurrent branches because create_task/gather
  snapshot context at push time. Verified with an inline concurrency test.
- Rerankers merge: pos + neg dispatched in one `asyncio.gather(pos_coro, neg_coro)` under
  `rerankers`; negatives no longer dispatch inside scoring (scoring only folds the
  captured maps). Dropped the `query_search.negatives` span + names.
- Polarity: user's assumption that dispatch already labels polarity was wrong (both are
  POOL_RERANKER). A trivial concurrent sub-span split isn't possible (context-manager vs
  async parentage), so added `dispatch.polarity` (Polarity.value); `_dispatch_call` gained
  a `polarity` kwarg (default POSITIVE), negatives pass NEGATIVE.
- Implicit-prior relocation: moved the rerank + helpers/enums/boost-dicts out of
  full_pipeline_orchestrator into new search_v2/implicit_prior_rerank.py (public
  `apply_implicit_prior_rerank_for_branch`). Type-only imports (Step2BranchResult,
  BranchRankedResults) under TYPE_CHECKING break the cycle. stage_4._run_branch calls it
  inside the scoring span (top_score set BEFORE it, stays pre-boost). Removed the 3 old
  call sites (2 batch in full_pipeline, 1 per-branch in streaming) + the batch wrapper.
- Phase B extracted into `_define_candidate_pool` (owns the `candidate_generation` span +
  fetch_count/candidate_count/cost_usd) to avoid re-indenting ~240 lines with early
  returns; `_dispatch_generator_specs` now returns its dispatched count.

### Testing Notes
All 7 files compile + import (incl. api.main). Behavior parity expected: ranking unchanged
(rerank relocated, not altered); request-level cost total unchanged (root still sums all).
Verify in Tempo: branch collapses to 6 children; no top-level `negatives`;
`implicit_prior_rerank` under `scoring`; pos+neg dispatch overlap under `rerankers` with
`dispatch.polarity`; A–D carry `cost_usd` summing ≤ `query_search.cost_usd`;
`candidate_generation.candidate_count == len(union)`. `_dispatch_call` gained a defaulted
`polarity` kwarg and implicit-prior symbols moved modules (test impact if referenced).

## Observability for POST /similarity_search (1c-3)
Files: observability/names.py, search_v2/similar_movies.py, api/main.py,
observability_context/observability_architecture.md, docs/modules/api.md,
observability_context/observability_todos.md

### Intent
Give the pure /similarity_search endpoint first-class telemetry with parity to the
/query_search similarity branch, and fix that the shared engine's Qdrant/fetch spans
+ signal attributes were mis-rooted under `query_search.*` even when no query_search
ran (names.py rule B: root = owning endpoint).

### Key Decisions
- Flow-neutral `similarity.*` root (user-chosen over caller-parameterized / leave-as-is).
  Renamed the engine-produced spans (`query_search.similarity_qdrant`/`_fetch` →
  `similarity.qdrant`/`.fetch`) and the 12 similarity signal attributes (off
  `query_search.branch_*` → `similarity.*`, stripping the `branch_` prefix) since
  they're emitted inside `run_similar_movies_for_ids`, shared by both callers. Added
  `similarity.anchor_count` (engine-set) as the single-vs-multi discriminator.
- Relocated `_record_similarity_signals(result)` OUT of `run_similarity_search`
  (query_search branch executor) and INTO the engine's return path (both single- and
  multi-anchor), so it writes to `trace.get_current_span()` — the branch span on the
  query_search path, the server span on the endpoint path — giving parity with zero
  per-caller wiring. The reference-resolution skeleton (`_record_similarity_entities`
  → `query_search.branch_entities/…`) stays query_search-only (anchors are supplied,
  not resolved, on the endpoint).
- Endpoint-owned facts on a new `similarity_search.*` root (only the handler knows
  them): `cache_hit` (Redis disposition, set at success points à la
  movie.payload_source) and `result_count` (post-hydration cards, analog of the
  orchestrator's branch_result_count). Added `@record_outcome` (plain, non-streaming),
  converted the two bare HTTPExceptions to `EndpointFailure(invalid_parameters)` so the
  verdict is labeled (filter-enum 422s already raise EndpointFailure(invalid_filters)),
  and promoted the two swallowed Redis warnings to `cache read/write failed` span events.
- Extracted `_record_filter_attributes(filters, span)` from `_record_query_search_inputs`
  so the endpoint records the same `filters.*` breakdown (+ active_count) as /query_search.

### Testing Notes
- Verify in Tempo (warm + cold): server span carries `similarity.*` signals +
  `similarity.anchor_count`, `similarity_search.cache_hit`/`.result_count`,
  `filters.active_count`, `outcome.success`, with `similarity.qdrant`/`.fetch` child
  spans nested (no `query_search.*` names). Warm repeat → cache_hit=true, no engine
  child spans.
- Error paths: unknown tmdb_id → 422 + outcome invalid_parameters; unknown filter enum
  → 422 + invalid_filters.
- No /query_search regression: its similarity branch now carries the same facts under
  the renamed `similarity.*` keys; branch_type=similarity + entity skeleton unchanged.
- Offline (batch/notebook) path: engine writes are no-ops (ProxyTracer), behavior
  unchanged. Constant renames touch only observability/names.py + search_v2/similar_movies.py
  (grep-confirmed no other importers) — test impact if any test asserts on the old
  `query_search.branch_*`/`query_search.similarity_*` attribute keys.

## Observability for /attribute_search (1c-4) + shared person.resolve span
Files: observability/names.py, search_v2/person_search.py, search_v2/attribute_search.py, api/main.py, observability_context/observability_architecture.md, observability_context/observability_todos.md

### Intent
Instrument the last uninstrumented search endpoint (POST /attribute_search) — pure
Postgres, so its DB calls were already auto-traced but it had no outcome verdict, no
request-input attrs, no result counts, and no per-person grouping span. Also fixes the
root cause: the /query_search person-branch per-person span lived only in the
run_person_search wrapper (named query_search.person_resolution) and the shared resolver
fetch_person_buckets — which /attribute_search calls directly — carried nothing.

### Key Decisions
- Extracted the per-person span into a shared, flow-neutral wrapper
  resolve_person_traced(name, *, metadata_filters) beside fetch_person_buckets
  (person_search.py), emitting a `person.resolve` span (renamed off the
  query_search.person_resolution root, rule B) with intrinsic attrs person.resolve.name /
  .movie_count / .best_bucket (omitted when 0) + a "person unresolved" event on a
  zero-credit miss. fetch_person_buckets stays tracer-free. This mirrors the 1c-3
  similarity refactor (shared engine → neutral `similarity.*` root). run_person_search now
  calls the wrapper and keeps its branch-level aggregate attrs (branch_entities etc.).
  Chosen over pushing the span into fetch_person_buckets (keeps the resolver pure) and over
  a second hand-copied span in attribute_search (single source of truth, no drift).
- Endpoint-owned facts (attribute_search.*, flat leaves per the branch_* precedent):
  path (browse|people, AttributeSearchPath enum), people_requested_count, people_names
  (truncated), people_searched_count, people_unresolved_count, pool_count, result_count.
  The path + people/pool skeleton is stamped by run_attribute_search on the current span
  (server span at that call depth, same get_current_span pattern run_person_search uses);
  the input filters.*/people attrs + result_count are stamped by the handler.
- Reused the EXISTING _record_filter_attributes helper (already shared by /query_search +
  /similarity_search) for the filters.* inputs — the plan's "extract _record_filter_inputs"
  was redundant; the extraction already existed. Added @record_outcome (bare, non-streaming)
  so the invalid_filters 422 (already an EndpointFailure inside _to_metadata_filters) and a
  clean success=true are recorded. Mirrors /similarity_search's handler shape.

### Testing Notes
- Verify in Tempo: (browse) {} → attribute_search.path=browse + result_count +
  success=true, no person.resolve spans; (people) one person.resolve span per name nesting
  its 6 psycopg spans + people_searched/pool/result counts; misspelled name → that span's
  movie_count=0 + "person unresolved" event + people_unresolved_count increments, request
  still success=true; unknown genre → 422 + outcome invalid_filters.
- Regression: /query_search person query still emits the per-person span under the branch
  span, now named person.resolve, with branch aggregate attrs intact.
- Constant rename QUERY_SEARCH_PERSON_RESOLUTION → PERSON_RESOLVE touches only
  observability/names.py + search_v2/person_search.py (grep-confirmed no other importers);
  test impact only if a test asserts on the old query_search.person_resolution span name.

## /rerun_query_search observability + generic request.* rollup root (1c-2)
Files: observability/names.py, api/main.py, implementation/llms/generic_methods.py,
observability_context/{observability_architecture.md,observability_todos.md,query_search_planning.md},
docs/modules/{api.md,observability.md}

### Intent
Bring /rerun_query_search to full server-span parity with /query_search, and promote the
request-level rollups off the query_search.* root to a generic cross-endpoint request.*
root. Before this, the rerun handler set NOTHING on its server span and — critically —
omitted the trace.use_span(request_span) wrapper, so every reused query_search.* pipeline
span (branch/step_2/Stage 4/llm.generate) orphaned instead of nesting under the rerun
server span. That orphaning (todo 1c-2's core concern) is the highest-value fix here.

### Key Decisions
- New generic `request` root (observability/names.py, sibling to `outcome`): request.cost_usd,
  request.result_count, request.usage.{input,cached_input,output}_tokens. Rationale: these
  mean the same thing on any endpoint, so the owner is the request, not query_search (rule B,
  like outcome.*). Removed QUERY_SEARCH_COST_USD / _TOTAL_RESULT_COUNT / _USAGE_* and repointed
  the /query_search finally block.
- result_count UNIFIED across all five result-returning endpoints (user decision): removed
  SIMILARITY_SEARCH_RESULT_COUNT / ATTRIBUTE_SEARCH_RESULT_COUNT / TITLE_SEARCH_RESULT_COUNT,
  repointed all three set_attribute calls to REQUEST_RESULT_COUNT. cost_usd/usage.* don't
  affect those three — they make no LLM/embedding calls, so the attrs are simply absent.
- Branch counts stayed query_search.* (succeeded/failed_branch_count): "branch" is a
  branch-plan concept only /query_search + /rerun have. Rerun REUSES the keys — same rule-B
  move as reusing the branch spans.
- New rerun_query_search.* root for rerun's input capture (branch_count / branch_types /
  standard_queries) — rerun has no raw query/clarification, its input IS the replayed branch
  plan. Entity anchor names deliberately NOT duplicated here (already on entity branch spans).
- Rerun handler: added @record_outcome(success_on_return=False); copied /query_search's
  event_stream scaffold (track_request_cost → use_span → branch_results accumulation → finally
  rollup), dropping the Step-0-fatal/error-event branch (rerun has no Steps 0/1). New helper
  _record_rerun_inputs (reuses shared _record_filter_attributes).
- Converted the rerun boundary helpers' raw HTTPException raises (_clean_branch_query /
  _enforce_name_cap / _clean_one / _clean_names / _to_rerun_plan) to EndpointFailure via a new
  _rerun_rejection helper (invalid_parameters + "request rejected" span event), so
  @record_outcome classifies rejections instead of internal_error.
- Deferred (low value): an explicit span around the _to_rerun_plan / _to_metadata_filters
  boundary translation.

### Testing Notes
- Tempo, rerun happy path: every reused query_search.* + llm.generate span nests under the
  rerun server span (the orphaning fix); server span carries rerun_query_search.* + filters.* +
  request.* + query_search.{succeeded,failed}_branch_count + outcome.success=true.
- Rerun rejections: blank query → 400 invalid_parameters + "request rejected"; bad filter enum
  → 422 invalid_filters; >3 standard branches → 422; confirm NO internal_error.
- All-branches-failed → outcome.success=false + all_branches_failed (verdict, not crash).
- Cross-endpoint: /query_search, /similarity_search, /attribute_search, /title_search all now
  emit request.result_count; similarity/attribute carry no request.cost_usd (no LLM calls).
- Grep-confirmed zero dangling refs to the six removed constants; both edited .py files parse;
  new names.py constants resolve and old ones are gone.
