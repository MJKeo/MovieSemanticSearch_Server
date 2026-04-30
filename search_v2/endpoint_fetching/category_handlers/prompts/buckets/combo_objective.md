# Objective

This category maps to **multiple candidate endpoints that can each fire in parallel**. Each endpoint carries distinct, complementary signal that cannot be collapsed into a single call — the kind of requirement this category covers is inherently multi-faceted, and forcing a single endpoint would drop real signal.

Your task: determine which combination of endpoints — including the empty combination (none fire) — best covers the target requirement, and fill in parameters for each endpoint you choose to fire.

Work through the decision in order:

1. Break the target requirement into its discrete aspects. For each aspect, describe how every candidate endpoint could cover it.
2. In `overall_endpoint_fits`, synthesize which endpoints fit, why, and how they complement each other.
3. In `per_endpoint_breakdown`, address **every** candidate endpoint explicitly with a `should_run_endpoint` decision. Every endpoint must be addressed — silently skipping one is the failure mode this bucket is designed to prevent.
4. For each firing endpoint, populate its parameter payload. Leave parameters null for endpoints that should not fire.

**No-fire — per-endpoint or for the whole combination — is a valid and preferred outcome.** Only fire endpoints that carry real signal toward this specific requirement. Firing every endpoint "because they are available" dilutes the result pool; firing only one when multiple genuinely apply drops signal the bucket exists to preserve.
