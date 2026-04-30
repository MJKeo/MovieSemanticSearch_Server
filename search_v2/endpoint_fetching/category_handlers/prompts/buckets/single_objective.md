# Objective

This category maps to **a single retrieval endpoint**, introduced above. Your task: decide whether that endpoint should fire for the target requirement, and — if it should — fill in its parameters so the retrieved results best match the user's intent.

Work through the decision in order:

1. Break the target requirement into its discrete aspects. For each aspect, describe what the user is asking for and what the endpoint can concretely do toward satisfying it. Name any gaps where the endpoint cannot cover the aspect.
2. Use that analysis to decide whether the endpoint is a genuine fit overall.
3. If it is, populate the parameter payload. If it is not, leave parameters null.

**No-fire is a valid and preferred outcome.** Upstream dispatch is not infallible — if the endpoint does not genuinely fit the requirement, declining to fire is always better than inventing parameters for a bad match.
