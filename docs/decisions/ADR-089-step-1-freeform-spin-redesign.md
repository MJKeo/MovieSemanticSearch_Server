# [089] — Step 1 freeform spin redesign

## Status
Active

## Context
Step 1 produced spins under a rigid schema: `hard_commitments`,
`soft_interpretations`, `open_dimensions`, `original_query_label`,
`branching_opportunity`, `distinctness`. Every spin was forced to
preserve all hard commitments, so all three branches collapsed onto
the same dominant anchor (e.g. "dark Marvel movies" → all three spins
returned Marvel movies with slight adjective tweaks). The model was
treating spin generation as slot-filling on the original query rather
than as genuine result-set diversification.

## Decision
Stripped `Step1Response` to `exploration: str` (2-3 telegraphic sentences
of freeform brainstorm, generated first) + `spins: list[Spin]` where
`Spin` has only `query` + `ui_label`. Rewrote the system prompt as
principle-based guidance: vague queries commit to a specific reading;
specific queries are built from scratch as semantic neighbors, not
textual variations; the two spins must produce visibly different result
sets from each other and from the original. Model switched to
`gemini-3.5-flash` with `thinking_level="minimal"`. `exploration` serves
as a visible scratchpad, eliminating the need for internal thinking budget.

## Alternatives Considered
- **Tighter structured decomposition with better instructions**: rejected
  because the hard-commitments field was architecturally forcing anchor
  preservation. No prompt change could fix a schema that required every
  spin to share the dominant constraint.
- **Hidden thinking budget (thinking_budget=1024)**: tried as an intermediate
  step; replaced when the visible `exploration` field made reasoning observable
  and externalized, making hidden-thinking redundant and wasteful.

## Consequences
- **Enables**: genuine result-set diversification for single-anchor queries
  (Tom Hanks, Pixar, Scorsese) — anchors are dropped and sibling searches
  built from scratch.
- **Known ceiling**: multi-anchor composite queries ("sci-fi with mentorship
  arc") still tend to anchor-lock because the combination is the viewer's
  taste itself; and evaluative language ("heartfelt", "magical") still leaks
  into spins. Both deferred.
- `full_pipeline_orchestrator` hard-codes the original-branch label as
  "Original Query" (was sourced from `step1.original_query_label`).

## References
- `search_v2/step_1.py`, `schemas/step_1.py`
- `docs/personal_preferences.md` — principle-based prompt instructions;
  examples in field descriptions become templates
