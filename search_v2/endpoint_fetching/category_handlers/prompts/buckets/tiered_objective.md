# Objective

This category maps to **multiple candidate endpoints in a biased order**. Earlier entries in the endpoint list above are the preferred channels for this category — they are the authoritative hit when they apply. Later entries are fallbacks for requirements that fall outside the earlier endpoints' canonical vocabulary.

Your task: pick which single endpoint (if any) best fits the target requirement, and fill in its parameters.

Work through the decision in order:

1. Break the target requirement into its discrete aspects. For each aspect, describe how every candidate endpoint could cover it, then name the best-covering endpoint and any gaps it leaves.
2. In `performance_vs_bias_analysis`, reason through how the candidates stack up against the bias: is one candidate clearly better on its own merits, or is the bias what breaks a close call?
3. Commit to the single endpoint that best fits the requirement overall — or decide that none of them does.
4. If one is picked, populate its parameter payload.

**No-fire is a valid and preferred outcome.** If no candidate is a genuine fit, returning `endpoint_to_run: "None"` is always better than forcing a weak match. The bias does not force a pick.

**How to apply the bias:**

- The bias is a **tiebreaker** when multiple endpoints fit roughly equally — not a veto on lower-preference endpoints.
- A **clearly-better lower-preference endpoint wins.** When the requirement falls outside the higher-preference endpoint's canonical vocabulary, the lower-preference endpoint's broader reach wins decisively despite the bias.
- The bias only exists to prevent picking a plausible-but-less-authoritative endpoint when a canonical one would fit equally well. That is all.
