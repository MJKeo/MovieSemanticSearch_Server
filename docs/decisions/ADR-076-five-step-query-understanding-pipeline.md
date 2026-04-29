# [076] — Five-step query understanding pipeline decomposition

## Status
Active

## Context
The original Step 2 schema treated trait identification and category
grounding as a single LLM pass: identify spans, decide if they are
atomic, and classify each into a category simultaneously. Two
structural failure modes emerged:

1. **Lost structural context after decomposition.** Once spans were
   flattened into atoms, the relationship between them was gone. "John
   Wick but with kids" became two separate traits; the information
   that the second modifies the first as a kept-whole anchor was not
   recorded anywhere downstream could use it.

2. **Parametric opacity.** Phrases like "like Inception" or "Criterion
   Collection films" needed outside-knowledge expansion before they
   could route to a search endpoint. The old schema deferred this to
   special-purpose catch-all categories (Cat 43, Cat 45) rather than
   resolving it in-pipeline, giving the LLM an escape hatch that
   bypassed resolution.

## Decision
Decompose the monolithic Step 2 into five sequential steps, each with
narrow scope:

| Step | Purpose |
|------|---------|
| 1 | Holistic read — faithful prose read, relationships recorded |
| 2 | Atom + candidate boundary identification |
| 3 | Reconstruction test — would additive combination reconstruct intent? |
| 4 | Literal test — does the unit's literal text route correctly, or does it need parametric expansion? |
| 5 | Trait commitment — final category, role, polarity, salience |

Step 1 (holistic read) is landed. Steps 2–5 are not yet implemented.

Two parametric-expansion categories (Cat 43 and Cat 45) were removed
from the taxonomy in parallel. Their work now belongs to Step 4.
Numbering gaps are preserved for cross-reference stability.

## Alternatives Considered
- **Keep a single-pass schema with more fields**: Adding kept-whole
  flags and parametric-expansion fields to the existing schema. Rejected
  because a single pass cannot catch reconstruction failures — the model
  needs to commit to decomposition before it can test whether the
  decomposition was valid.
- **Two passes (structural + categorical)**: Simpler than five steps,
  but still collapses reconstruction testing and literal testing into
  one operation. Both tests require reviewing the previous step's
  output; they are not the same operation and interact differently with
  the model's failure modes.

## Consequences
- Step 1 is the only place structural relationships between query wants
  are explicitly recorded. Downstream steps consume this as a ground
  truth; they can override its decomposition but should trust its
  structural markings.
- The pipeline prompt size drops significantly per step because the
  category taxonomy (the largest single block) only loads at Step 5
  (and possibly a lightweight version at Step 4). Steps 1–3 run without it.
- Steps 2–5 need to be authored and evaluated before the holistic-read
  path replaces the old concept-extraction flow end-to-end.

## References
- search_improvement_planning/v3_step_2_rethinking.md
- schemas/step_2.py (Step 1 schema)
- search_v2/step_2.py (Step 1 executor)
- schemas/trait_category.py (43-category taxonomy for Step 5)
- docs/modules/search_v2.md
