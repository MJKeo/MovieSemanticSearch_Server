# [096] — Step 3 consolidation redesign: minimum-viable-call-set + inclusion-only routing

## Status
Active

## Context
Step 3 over-fragmented single-concept traits into FACETS (products) of multiple
category calls. A single coherent trait like "family friendly" would decompose
into both a SENSITIVE_CONTENT (avoidance framing) and a TARGET_AUDIENCE call —
the former violating the presence-only invariant, the latter correct but
redundant. More broadly, aspects were being enumerated as the whole restated
alongside its parts, and the consolidation step was too binary (SOLO vs
non-SOLO) to decide the minimum call count continuously.

## Decision
Three interlocking changes to Step 3's prompt architecture:

1. **Aspects redesign**: aspects now enumerate distinct, non-overlapping,
   comprehensive PARTS of the trait (not the whole restated alongside its parts).
   Described plot/story shapes are kept whole as one aspect. Reassembly into calls
   is the consolidation step's job; over-enumeration in aspects is safe because
   consolidation can reduce it.

2. **`CandidateFit` enum on each CategoryCandidate**: `CLEAN_OWNERSHIP` (category
   squarely claims this aspect) / `COULD_CONSOLIDATE` (usable but better folded
   into a broader call) / `LIKELY_DISREGARD` (category can only describe the
   trait by describing an absence — inclusion-ineligible). Fit is decided at
   candidate enumeration time. The schema floor (min_length=5) is preserved;
   `LIKELY_DISREGARD` distinguishes real fits from floor-filler without lowering it.

3. **`routing_exploration` renamed to `consolidation_analysis`** and rewritten:
   explore options first, then place the trait on the breadth↔single-shape
   spectrum (continuous, not binary), then select the minimum call set. Broader
   calls that cover multiple aspects preferred over spawning brittle separate
   facets. `combine_mode` now reads off the spectrum placement.

## Alternatives Considered
- **Raise the SOLO threshold / tune combine_mode rules**: rejected — the over-
  fragmentation root cause was at the aspect level (wrong parts enumerated) and
  the candidate-fit level (absence-framed categories offered as options). Changing
  combine_mode rules would leave those causes in place.
- **Add schema constraints to block LIKELY_DISREGARD calls**: rejected (user
  decision) — the LLM should reason about fit before committing, not be blocked
  at schema validation. `LIKELY_DISREGARD` as a label is more informative than
  a schema error.
- **Per-query few-shot examples**: evaluated but not adopted — caused a
  regression on "scream" (ambiguity case) and the principled guidance
  generalized better than pattern-matched examples.

## Consequences
- Field `routing_exploration` renamed to `consolidation_analysis` — any unit
  test or serializer referencing the old name needs updating.
- `CategoryCandidate` now requires a `fit` field — tests asserting on
  `TraitDecomposition` shape need the new field.
- `SENSITIVE_CONTENT` presence-only boundary and `MATURITY_RATING` loosened
  proxy use (separate category-definition audit, same session) are companion
  changes that enable correct fit assignment for suitability queries.

## References
- `search_v2/step_3.py` (prompt), `schemas/step_3.py` (CandidateFit, TraitDecomposition)
- `schemas/trait_category.py` (category definitions, SENSITIVE_CONTENT/MATURITY_RATING)
- `search_v2/category_candidates_experiment/CONSOLIDATION_EXPERIMENT.md` (validation)
- `docs/modules/search_v2.md` (Step 3 Trait Decomposition Design, Category Taxonomy)
