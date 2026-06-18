# [099] — Follow-up clarification support in `/query_search`

## Status
Active

## Context
After seeing initial results, users want to issue a correction ("less violent",
"more 80s", "actually I meant the comedy one") without retyping the full query.
The challenge is where to inject the clarification: injecting it everywhere
(including Steps 2+) would require new schema fields throughout the pipeline;
injecting it only at Steps 0/1 keeps all existing prompt discipline (intent_exploration,
trait commitment, polarity, positioning) intact.

## Decision
Inject clarification only at the query-understanding layer (Steps 0 and 1).
Steps 2+ are unchanged and receive a single natural-language query per branch:

- **Step 0** uses a separate `CLARIFICATION_SYSTEM_PROMPT` that teaches the model
  to merge original + clarification: "clarification is authoritative on conflict;
  original is preserved where clarification is silent; retraction drops material;
  flow choice is re-evaluated from the merged intent." The rest of the prompt
  (zones, coverage, resolution, qualifier, ambiguity, similarity, output-field)
  is shared. `run_step_0(query, clarification=None)` dispatches on presence.

- **Step 1** has a parallel clarification path producing `Step1ClarificationResponse`
  (`exploration` + `main_rewrite: Spin` + `spins: list[Spin]`). `main_rewrite`
  is a faithful translation of the merged intent (no hallucinated detail, no
  resolving descriptions to specific titles); spins are creative angles on the
  rewritten intent. The `Spin` class is shared so downstream branch-plan unpacking
  is identical for both response types.

- **Slot-1 branch plan**: `_plan_step2_branches` no longer hard-codes
  `"Original Query"` as the label. With clarification: slot 1 carries
  `main_rewrite.query` + `main_rewrite.ui_label`. Without: slot 1 carries
  the raw query verbatim in both query and label fields.

## Alternatives Considered
- **Inject at Step 2** (per-branch): rejected. Step 2 already handles a single
  branch query; injecting a second string would require schema changes to every
  Atom/Trait/QueryAnalysis type and all downstream fields.
- **Re-run the whole pipeline with a merged prompt constructed client-side**:
  rejected. Client-side merging is lossy (client may not have all query context)
  and moves product logic outside the server.
- **LLM-merged query before Step 0**: rejected. Adds a pre-Step-0 LLM call (extra
  latency + cost) and the Step 0 prompt already has the teaching to handle the merge.

## Consequences
- `clarification` is a nullable field on `QuerySearchBody`; whitespace-only
  collapses to `None` at the API boundary so the no-clarification fast path is
  stable.
- Existing unit tests that fixture the literal `"Original Query"` label or call
  `run_step_0(query)` / `run_step_1(query)` positionally still compile (new param
  defaults to `None`) but will need label/output fixture updates in the testing phase.
- The `MAX_CLARIFICATION_CHARS = 200` cap (shared with query) bounds per-request
  LLM input cost for free.

## References
- `search_v2/step_0.py` (CLARIFICATION_SYSTEM_PROMPT, run_step_0)
- `search_v2/step_1.py` (Step1ClarificationResponse, run_step_1)
- `schemas/step_1.py` (Step1ClarificationResponse, Spin)
- `search_v2/streaming_orchestrator.py` (_plan_step2_branches)
- `search_v2/query_input_validation.py` (MAX_CLARIFICATION_CHARS)
- `docs/modules/search_v2.md` (Step 1 pipeline step description)
- `docs/modules/api.md` (/query_search clarification field)
