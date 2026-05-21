# [088] — FACETS soft fold: geometric mean with floor=0.1 replacing strict PRODUCT

## Status
Active

## Context
The FACETS `TraitCombineMode` is used for traits whose multiple categories are
independent facets (e.g., genre + tone + period), where each facet must be satisfied
for the film to match. The original V4 design used a strict PRODUCT of all category
scores: any near-zero score would collapse the combined score to near-zero, regardless
of how well the film matched the other facets. In practice this produced over-penalization:
a film that matched 3 of 4 facets extremely well received a near-zero trait score due
to one weak facet.

## Decision
Replace strict PRODUCT with a geometric mean with floor=0.1 (V5 Phase 7 "soft FACETS
fold"):
- Each category score is floored at 0.1 before the geometric mean is computed.
- Geometric mean over N categories: `(∏ scores_i)^(1/N)`.
- Floor ensures a weak-but-nonzero facet still contributes proportionally rather than
  dominating the product.

The geometric mean preserves FACETS semantics (all facets must contribute) while
reducing over-penalization for partial matches.

## Alternatives Considered
- **Strict PRODUCT (V4)**: Mathematically clean but too harsh in practice; a single
  0.05 score in a 4-facet trait produces a combined score of ~0.05 * avg^3 ≈ tiny.
- **Arithmetic mean**: Loses FACETS semantics entirely; a film scoring 1.0 on one
  facet and 0.0 on all others would average to 0.25, appearing to partially satisfy
  the trait.
- **Weighted harmonic mean**: Theoretically sound but more complex with no practical
  benefit over geometric mean with floor.

## Consequences
- Partial-match films receive meaningfully non-zero FACETS scores rather than being
  collapsed to near-zero.
- Floor=0.1 is a hyperparameter; if it's too high, FACETS stops penalizing weak facets
  sufficiently. Current value chosen empirically.
- Behavior change from V4 is significant: downstream scoring expectations based on
  V4 PRODUCT semantics are invalid.

## References
- docs/modules/search_v2.md — Stage 4 Phase D, TraitCombineMode section
- search_v2/full_pipeline_orchestrator.py
- ADR-083: SOLO combine mode (companion TraitCombineMode change)
