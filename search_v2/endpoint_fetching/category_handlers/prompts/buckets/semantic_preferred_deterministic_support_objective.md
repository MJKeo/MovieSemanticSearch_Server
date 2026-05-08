# Objective

This category routes to **semantic plus one or more deterministic candidates**, introduced above. Semantic typically carries the graded / experiential / tonal core; the deterministic endpoints catch binary or canonical signals that the embeddings blur across (a named season, a known holiday phrase, a registry-tagged tone). Endpoints are puzzle pieces — they overlap, refine each other, fill each other's gaps; that's the design.

Your task: walk every declared endpoint concretely with strengths + weaknesses, argue out the composition that covers the call's intent, then fill thin parameters for each endpoint that fires. Four sequential phases, each grounded in the prior one:

1. **Per-endpoint walks.** For each declared endpoint, fill its `{route}_walk` block with concrete candidates — vector spaces for semantic, registry members / columns for the deterministic candidates. Each candidate carries `strengths` (what it genuinely OWNS at retrieval time) and `weaknesses` (under-coverage gaps AND over-coverage breadth — both belong here). Surface every plausibly useful candidate so the commitment phase reads off real options.

2. **Coverage exploration.** Argue which endpoints should fire to compose coverage. Read off the strengths + weaknesses already written; do not re-derive. The local tests:
   - **Fire test:** "Does this endpoint contribute a strength the others don't, OR fill a weakness another has?" Yes → fire it.
   - **Drop test:** "Does another endpoint dominate this one's strengths AND weaknesses?" Yes → drop the dominated one.
   - **Sharpness layering:** semantic carries graded experiential signal; a deterministic endpoint with a clean canonical/binary candidate adds gate-style sharpness that semantic alone blurs. When BOTH have useful candidates, BOTH fire.
   - **Superset test (per endpoint):** apply the endpoint's own commitment principle to its candidates. If the candidates fail the endpoint's commitment criteria (e.g., the keyword candidates fail the keyword endpoint's superset test — gaps in coverage of the user's attribute, or stretching intent beyond what the registry names), drop that endpoint from `coverage_assignments` even if other endpoints fire. This is partial abstention — sanctioned alongside the existing whole-call abstention pathway.

3. **Coverage assignments.** Mechanical commit of the choice argued above. One entry per endpoint that should fire. Empty `coverage_assignments` is valid when no endpoint walk surfaced a candidate that passes both the local fire/drop tests and the endpoint's own commitment criteria. Partial commitment (some endpoints fire, others abstain via the superset test or equivalent) is also valid — the per-endpoint criteria are independent.

4. **Thin per-endpoint parameters.** For each endpoint with an assignment, fill its `{route}_parameters` block. The wrapper's `{route}_retrieval_intent` mirrors the slice_description from the matching assignment; the inner parameters draw on that intent and the endpoint's walk above to commit the route-specific translation.

**Invariant:** every aspect surfaced in the walks must be owned by some assignment, OR the candidate that surfaced it should have been dropped per the local tests. There is no soft-out for "this slice is unservable."

**Semantic abstaining is unusual but valid** — only when semantic's walk surfaces no useful space coverage. Do not force a semantic assignment when nothing in the 7 spaces carries the call's intent.
