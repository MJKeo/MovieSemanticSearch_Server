# Objective

This category maps to **multiple candidate endpoints**, introduced above. They are declared in priority order, but priority is a tiebreaker, not a license for the first endpoint to absorb the whole call. Endpoints are puzzle pieces — one may add specificity another lacks; one may fill a gap another leaves; and overlap is the design.

The category-specific notes that follow define what each representation can carry and where the boundary between them lies. Trust those notes — they encode the coverage judgment for the specific representations involved.

Your task: walk every declared endpoint concretely with strengths + weaknesses, argue out the composition that covers the call's intent, then fill thin parameters for each endpoint that fires. Four sequential phases, each grounded in the prior one:

1. **Per-endpoint walks.** For each declared endpoint, fill its `{route}_walk` block with concrete candidates the endpoint could plausibly carry — registry members for keyword, vector spaces for semantic, structured columns for metadata. Each candidate carries `strengths` (what it genuinely OWNS at retrieval time) and `weaknesses` (under-coverage gaps AND over-coverage breadth — both belong here). The walks are independent: each asks "what could this endpoint cover for the call's intent?" without considering what the others might do. An empty / no-match walk is a valid signal that this endpoint has nothing useful for the call.

2. **Coverage exploration.** Argue which endpoints should fire to compose coverage of the call's intent. Read off the strengths + weaknesses already written; do not re-derive. Frame the endpoints as puzzle pieces. The local tests:
   - **Fire test:** "Does this endpoint contribute a strength the others don't, OR fill a weakness another has?" Yes → fire it.
   - **Drop test:** "Does another endpoint dominate this one's strengths AND weaknesses (capture the same content strictly better)?" Yes → drop the dominated one.
   - **Over-coverage refinement:** when one endpoint's `weaknesses` names over-coverage and another endpoint's `strengths` isolate the slice, fire BOTH. The refining endpoint narrows what the broader one pulls.

3. **Coverage assignments.** Mechanical commit of the choice argued above. One entry per endpoint that should fire, naming the slice it owns. Multiple assignments are expected when the call is genuinely compound or when one endpoint refines another. Empty `coverage_assignments` is valid only when ALL declared endpoint walks surfaced no useful candidate; in that case the whole call abstains.

4. **Thin per-endpoint parameters.** For each endpoint with an assignment, fill its `{route}_parameters` block. The wrapper's `{route}_retrieval_intent` mirrors the slice_description from the matching assignment; the inner parameters draw on that intent and the endpoint's walk above to commit the route-specific translation.

**Invariant:** every aspect surfaced in the walks must be owned by some assignment, OR the candidate that surfaced it should have been dropped per the local tests. If you find yourself wanting to "leave an aspect unserved," that's either (a) a candidate whose strengths weren't real (drop it) or (b) a routing problem upstream (not something to memorialize here).

**Declining to fire any endpoint is valid only when all walks surfaced no useful candidate.** It is not a fallback for "this is hard." Use the local tests above before abstaining.
