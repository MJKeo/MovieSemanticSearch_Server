# [104] — /query_search branch span restructure: 6 groups, parallel rerankers, relocated implicit-prior

## Status
Active (drafted by docs-maintainer from DIFF_CONTEXT — pending human review)

## Context
Each standard branch's Stage 3/4 execution had grown to ~11 sibling
observability spans directly under `query_search.branch` (step_2, trait,
step_3, query_generation, generators, promotion, neutral_seed, rerankers,
negatives, dispatch, implicit_expectations, implicit_prior_rerank), built
up incrementally across several instrumentation bites. Two of those
siblings also reflected an implementation quirk rather than a real
architectural split: positive and negative rerankers dispatched serially
through separate spans, and the implicit-prior rerank ran as a distinct
pass in `full_pipeline_orchestrator.py` after Stage 4 returned, even
though it is part of the same scoring step.

## Decision
- Collapse the sibling spans into six named groups under
  `query_search.branch`: `step_2`, `decomposition` (Step 3 fan-out +
  implicit-prior policy generation), `candidate_generation` (Stage 4
  Phase B), `rerankers` (Stage 4 Phase C), `scoring` (Stage 4 Phase D+E),
  `hydration`. Add per-stage `cost_usd` on the first four groups.
- Dispatch positive and negative rerankers together in one
  `asyncio.gather` under `rerankers` (previously serialized, negatives as
  a separate span); a new `dispatch.polarity` attribute disambiguates
  them post hoc, since a true concurrent sub-span split isn't possible
  across the context-manager/gather boundary.
- Move the implicit-prior rerank out of the orchestrator's separate
  post-Stage-4 pass into `stage_4_execution._run_branch`'s `scoring` span
  (new module `search_v2/implicit_prior_rerank.py`), so `top_score` is
  captured pre-boost in the same place it's computed.
- Generalize `cost_tracking._request_cost` from a single accumulator to a
  `ContextVar` stack: `add_request_cost/tokens` fan out to every open
  frame, so `track_stage_cost()` isolates a subtree while the request
  root still sums everything — correct under concurrent branches because
  `asyncio.gather`/`create_task` snapshot context at push time.

## Alternatives Considered
- **Leave rerankers serialized / negatives as a separate span.**
  Rejected — no reason to pay serial-dispatch latency once both share
  one `rerankers` scope.
- **Keep implicit-prior rerank as an orchestrator-level pass.** Rejected
  — it is scoring, not orchestration, and living outside Stage 4 meant
  `top_score` couldn't reflect it without an awkward second write.

## Consequences
- Ranking output is unchanged (same boost math, same resort) — a
  structural + latency change, not a scoring change.
- Future work on reranker dispatch or implicit-prior boosting now lives
  in `stage_4_execution.py` / `implicit_prior_rerank.py`, not
  `full_pipeline_orchestrator.py`.
- Future per-stage cost instrumentation elsewhere should reuse
  `track_stage_cost()` / the ContextVar-stack pattern rather than a
  second bespoke accumulator.

## References
- `observability/cost_tracking.py`, `observability/names.py`
- `search_v2/implicit_prior_rerank.py` (new), `search_v2/stage_4_execution.py`,
  `search_v2/full_pipeline_orchestrator.py`, `search_v2/streaming_orchestrator.py`
- `docs/modules/search_v2.md` (Observability section), `docs/modules/api.md`
  (`/query_search` Telemetry)
- `observability_context/observability_architecture.md`
