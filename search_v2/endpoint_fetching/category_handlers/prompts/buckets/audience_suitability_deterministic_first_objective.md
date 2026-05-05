# Objective

This category routes to **multiple candidate endpoints**, introduced above. Audience-suitability requirements are inherently multi-faceted: a single concept like "suitable for kids" or "no gore" is genuinely better served by a deterministic gate, an inclusion or exclusion scoring signal, and a semantic intensity / watch-context query all running in parallel than by any one of them alone. Endpoints are puzzle pieces — overlap is the design, every endpoint that has a real signal to contribute should fire.

Your task: walk every declared endpoint concretely with strengths + weaknesses, argue out the composition, then fill thin parameters for each endpoint that fires. Four sequential phases, each grounded in the prior one:

1. **Per-endpoint walks.** For each declared endpoint, fill its `{route}_walk` block with concrete candidates — registry members for keyword, structured columns (especially maturity_rating ceilings) for metadata, vector spaces for semantic. Each candidate carries `strengths` (what it genuinely OWNS at retrieval time) and `weaknesses` (under-coverage gaps AND over-coverage breadth — both belong here). Surface every plausibly useful candidate so the commitment phase reads off real options.

2. **Coverage exploration.** Argue which endpoints should fire to compose coverage. Read off the strengths + weaknesses already written; do not re-derive. The local tests:
   - **Fire test:** "Does this endpoint contribute a strength the others don't, OR fill a weakness another has?" Yes → fire it.
   - **Drop test:** "Does another endpoint dominate this one's strengths AND weaknesses?" Yes → drop the dominated one.
   - **Sharpness layering:** a hard maturity ceiling on metadata, an inclusion/exclusion tag set on keyword, and an intensity or watch-context query on semantic each catch different signal sharpness on the same slice. When each has useful candidates, all fire.

3. **Coverage assignments.** Mechanical commit of the choice argued above. Suitability requirements typically result in **multiple assignments**. Empty `coverage_assignments` is valid only when ALL declared endpoint walks surfaced no useful candidate.

4. **Thin per-endpoint parameters.** For each endpoint with an assignment, fill its `{route}_parameters` block. The wrapper's `{route}_retrieval_intent` mirrors the slice_description from the matching assignment; the inner parameters draw on that intent and the endpoint's walk above to commit the route-specific translation.

**Invariant:** every aspect surfaced in the walks must be owned by some assignment, OR the candidate that surfaced it should have been dropped per the local tests. There is no soft-out for "this slice is unservable."

**Polarity rule: emit presence of an attribute, not direction.** Endpoint parameters describe what the content has. Whether that presence helps or hurts the user is decided when the signals are combined later — do not encode that decision into the parameter itself.

**Whole-call abstain is valid only when no walk surfaces a clean candidate.** The default posture for this bucket is to fire every endpoint that carries real complementary signal.
