# Objective

This category maps to **a single metadata endpoint**, introduced above. Metadata answers questions through typed values, ranges, thresholds, ordinal selections, and priors over a structured attribute — not through search prose.

Your task: decide whether the endpoint should fire for the target requirement, and — if it should — emit typed parameters that match the kind of signal the user is expressing.

Metadata signals come in distinct shapes. Treat them as separate, not interchangeable:

- **Hard filters** exclude any movie outside the bound from consideration entirely.
- **Soft preferences** rank movies higher or lower without excluding any.
- **Additive priors** add a structured score signal that layers on top of other channels.
- **Ordinal selection** picks by position within a sorted set (first, latest, n-th).

Work through the decision in order:

1. Break the target requirement into its discrete aspects. For each aspect, identify which signal shape the user is expressing and what concrete value, range, or ordinal it implies.
2. Decide whether the endpoint can express those signals at the strength and shape the user implied.
3. If it can, populate the parameter payload — typed values, bounds, decay, or ordinal selection as appropriate. The category-specific notes that follow define attribute-specific shaping.

**Declining to fire is a valid and preferred outcome.** If the expression doesn't pin a typed value the endpoint can carry, abstaining is always better than inventing thresholds or bounds.
