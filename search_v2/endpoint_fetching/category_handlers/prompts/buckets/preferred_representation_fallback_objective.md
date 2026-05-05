# Objective

This category maps to **multiple candidate endpoints with an explicit priority order**, introduced above. The first endpoint listed is the preferred representation when its grounded analysis surfaces a clean fit; later endpoints are fallbacks for parts the preferred can't reach.

The category-specific notes that follow define what each representation can carry and where the boundary between them lies. Trust those notes — they encode the coverage judgment for the specific representations involved.

Your task: walk every declared endpoint concretely, decide who fires on what slice, then fill thin parameters for each endpoint that fires. Three sequential phases, each grounded in the prior one:

1. **Per-endpoint walks.** For each declared endpoint, fill its `{route}_walk` block with concrete candidates the endpoint could plausibly carry — registry members for keyword, vector spaces for semantic, structured columns for metadata. The walks are independent: each asks "what could this endpoint cover for the call's intent?" without considering what the others might do. An empty / no-match walk is a valid signal that this endpoint has nothing useful for the call.

2. **Coverage commitment.** Read every walk above and decide who owns what. Add one entry to `coverage_assignments` per endpoint that should fire, naming the slice it owns. List anything no endpoint can cleanly serve in `intentionally_uncovered`. Priority order (the order endpoints appear in the schema) is a tiebreaker only — when two endpoints could cover the same slice equally well, prefer the earlier one. If no endpoint cleanly fits, `coverage_assignments` may be empty.

3. **Thin per-endpoint parameters.** For each endpoint with an assignment, fill its `{route}_parameters` block. The wrapper's `{route}_retrieval_intent` mirrors the slice_description from the matching assignment; the inner parameters draw on that intent and the endpoint's walk above to commit the route-specific translation.

**Declining to fire any endpoint is a valid and preferred outcome.** If no representation genuinely fits the requirement, an empty `coverage_assignments` plus honest `intentionally_uncovered` entries is always better than padding the response with plausible-but-noisy signal.
