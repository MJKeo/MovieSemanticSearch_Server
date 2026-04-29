# [075] — `creative_alternatives` as a separate Stage 1 field for productive sub-angle spins

## Status
Superseded by ADR-076. The original `search_v2/stage_1.py` module and
its `FlowRoutingResponse.creative_alternatives` field have been replaced
by the new Step 0 / Step 1 split: `search_v2/step_0.py` carries flow
routing, and `search_v2/step_1.py` produces two distinct creative spins
as `Step1Response`. The schema decisions described here informed the
Step 1 spin design but are no longer the live schema.

## Context
The prior `alternative_intents` field was strictly for competing
*readings* of the user's words (e.g. "Scary Movie" → exact title vs
collection). User testing showed that broad single-intent queries like
"Best Christmas movies for families" produced a clean primary but no
useful exploratory branches — the alternative_intents semantics
correctly excluded "spin on a broad set" cases, leaving UI consumers
with nothing to render in an "explore more" slot.

## Decision
Add a separate `creative_alternatives: list[CreativeSpin]` field to
`FlowRoutingResponse` (Stage 1 output schema). Spins are productive
narrowings *within* a single broad primary intent (e.g. animated
Christmas movies, modern streaming-era Christmas movies). They are
distinct from alternative_intents (which are different *readings*).

Design choices, each explicitly confirmed by user:
- **In-Stage-1, not a separate LLM call.** A parallel call duplicates
  routing work and risks interpreting the query differently; a serial
  call adds latency. Placing it at the END of the Stage 1 schema means
  structured-output generation has already committed primary + true
  alternatives before the spin field — decoupling the reasoning by
  construction.
- **Separate field, not folded into `alternative_intents`.** Downstream
  consumers can render "Did you mean..." (alternatives) differently from
  "You might also like..." (spins). Prompt discipline diverges between
  the two.
- **Separate `CreativeSpin` class, not reusing `AlternativeIntent`.**
  `spin_angle` replaces `difference_rationale`. Keeps type signal at
  the edge for downstream code.
- **Cap of 2 spins**, matching `alternative_intents`. Soft
  "be more conservative when alternative_intents already exist" rule
  in the prompt (no validator).
- **`alternative_intents` semantics tightened.** "Broad request where
  an adjacent exploratory branch would add value" case removed from
  `_BRANCHING_POLICY` — it now belongs to spins. Spins and
  alternatives can coexist for queries with genuine reading ambiguity.

## Alternatives Considered
- **Parallel second LLM call for spins**: Duplicates routing context
  and risks independent interpretation drift.
- **Serial second LLM call**: Adds latency with no benefit; the spin
  task can be done in the first pass with the same context.
- **Fold spins into `alternative_intents`**: Loses the semantic
  distinction; downstream can't render the two differently.

## Consequences
- Stage 1 prompt grew from ~16KB to ~21KB.
- Broad queries ("Best Christmas movies for families", "good horror
  movies") now produce 0 true alts + 2 spins.
- Spin emission is aggressive when alts exist — the "be more conservative"
  rule is guidance, not constraint. May want to tighten if 4-branch
  outputs feel cluttered.
- `schemas/flow_routing.py` carries both `AlternativeIntent` and
  `CreativeSpin`; `_validate_title_for_flow` helper applies to both.

## References
- schemas/flow_routing.py (legacy V1 schema, retained for reference)
- search_v2/step_1.py (active spin-generation module under ADR-076)
- search_improvement_planning/v3_step_2_rethinking.md (replaces the
  earlier `steps_1_2_improving.md` planning doc)
- ADR-076 (five-step query understanding pipeline — supersedes this)
- ADR-074-stage-2a-interpret-verdict-decompose-first.md
