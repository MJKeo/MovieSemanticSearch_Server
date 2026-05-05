# Objective

This category routes to **multiple candidate endpoints that should each fire in parallel whenever they carry a real signal**, introduced above. Audience-suitability requirements are inherently multi-faceted: a single concept like "suitable for kids" or "no gore" is genuinely better served by a deterministic gate, an inclusion or exclusion scoring signal, and a semantic intensity / watch-context query all running in parallel than by any one of them alone. Overlap across endpoints is welcome — every endpoint that has a real signal to contribute should fire, even when its angle is already partially captured by another.

Your task: walk every declared endpoint concretely, decide who fires on which slice of the suitability concept, then fill thin parameters for each endpoint that fires. Three sequential phases, each grounded in the prior one:

1. **Per-endpoint walks.** For each declared endpoint, fill its `{route}_walk` block with concrete candidates — registry members for keyword, structured columns (especially maturity_rating ceilings) for metadata, vector spaces for semantic. Each walk asks "what could this endpoint cover for the suitability requirement?" Surface every plausibly useful candidate so the commitment phase reads off real options rather than abstract optimism.

2. **Coverage commitment.** Read every walk and decide who owns what. Suitability requirements typically result in **multiple assignments**: a hard maturity ceiling on metadata, an inclusion / exclusion tag set on keyword, an intensity or watch-context query on semantic. Overlap is welcome — different channels catch different signal sharpness. List anything no endpoint can cleanly serve in `intentionally_uncovered`. Skip an endpoint only when its walk surfaced no clean signal; don't skip it just because another endpoint already touches the same angle.

3. **Thin per-endpoint parameters.** For each endpoint with an assignment, fill its `{route}_parameters` block. The wrapper's `{route}_retrieval_intent` mirrors the slice_description from the matching assignment; the inner parameters draw on that intent and the endpoint's walk above to commit the route-specific translation.

**Polarity rule: emit presence of an attribute, not direction.** Endpoint parameters describe what the content has. Whether that presence helps or hurts the user is decided when the signals are combined later — do not encode that decision into the parameter itself.

**Declining to fire any endpoint is a valid outcome** when no walk surfaces a clean signal. But the default posture for this bucket is to fire every endpoint that carries real complementary signal, not to collapse to a single one out of conservatism.
