# Objective

This category maps to **multiple candidate endpoints**, introduced above. They are declared in priority order, but priority is a tiebreaker, not a license for the first endpoint to absorb the whole call. Endpoints are puzzle pieces — one may add specificity another lacks; one may fill a gap another leaves; and overlap is the design.

The category-specific notes that follow define what each representation can carry and where the boundary between them lies. Trust those notes — they encode the coverage judgment for the specific representations involved.

Your task: walk every declared endpoint concretely with strengths + weaknesses, argue out the composition that covers the call's intent, then fill thin parameters for each endpoint that fires. Four sequential phases, each grounded in the prior one:

1. **Per-endpoint walks.** For each declared endpoint, fill its `{route}_walk` block with concrete candidates the endpoint could plausibly carry — registry members for keyword (each with `verdict_reason` → `verdict`), vector spaces for semantic, structured columns for metadata. Each candidate carries `strengths` (what it genuinely OWNS at retrieval time) and `weaknesses` (under-coverage gaps AND over-coverage breadth — both belong here). The walks are independent: each asks "what could this endpoint cover for the call's intent?" without considering what the others might do. An empty / no-match walk is a valid signal that this endpoint has nothing useful for the call. For keyword candidates, the per-candidate verdict (commit/abstain) is what server-derives `finalized_keywords` downstream — the candidates the LLM verdicts as commit ARE the commit list.

2. **Coverage exploration.** Argue which endpoints should fire to compose coverage of the call's intent. Read off the strengths + weaknesses already written; do not re-derive. Frame the endpoints as puzzle pieces. The local tests:
   - **Fire test:** "Does this endpoint contribute a strength the others don't, OR fill a weakness another has?" Yes → fire it.
   - **Drop test:** "Does another endpoint dominate this one's strengths AND weaknesses (capture the same content strictly better)?" Yes → drop the dominated one.
   - **Over-coverage refinement:** when one endpoint's `weaknesses` names over-coverage and another endpoint's `strengths` isolate the slice, fire BOTH. The refining endpoint narrows what the broader one pulls.
   - **Superset test (per endpoint):** apply the endpoint's own commitment principle to its candidates. If the candidates fail the endpoint's commitment criteria (e.g., every keyword candidate verdict-abstained, or all keyword candidates stretch beyond what the registry names), set that endpoint's `coverage_commitments.{route}.verdict` to `abstain` even if other endpoints fire. Partial abstention is sanctioned alongside whole-call abstention — they are independent decisions per endpoint.

3. **Coverage commitments.** Mechanical commit of the choice argued above, written into a fixed-shape `coverage_commitments` object with one required slot per declared endpoint. Each slot carries `verdict_reason` (cite the walk's strengths/weaknesses) → `verdict` (commit/abstain) → `slice_description` (required iff verdict == "commit"). Multiple commits are expected when the call is genuinely compound or when one endpoint refines another. All-endpoint abstain is valid when no endpoint walk surfaced a candidate that passes both the local fire/drop tests and the endpoint's own commitment criteria. Partial commitment (some endpoints commit, others abstain) is also valid — the per-endpoint criteria are independent. **You cannot abstain on an endpoint by silence — every declared endpoint must receive an explicit verdict.**

4. **Thin per-endpoint parameters.** For each endpoint where `coverage_commitments.{route}.verdict == "commit"`, fill its `{route}_parameters` block. Leave `{route}_parameters` null when verdict is `abstain`. The wrapper's `{route}_retrieval_intent` mirrors the slice_description from the matching commitment; the inner parameters draw on that intent and the endpoint's walk above to commit the route-specific translation. For keyword, `finalized_keywords` is server-derived from the walk's verdict commits — emit `[]` and let the derivation fill it.

**Invariant:** every aspect surfaced in the walks must be owned by some commit, OR the candidate that surfaced it should have been dropped per the local tests. If you find yourself wanting to "leave an aspect unserved," that's either (a) a candidate whose strengths weren't real (drop it) or (b) a routing problem upstream (not something to memorialize here).

**Declining to fire any endpoint is valid when no walk surfaces a candidate that passes both the local fire/drop tests and the endpoint's own commitment criteria.** It is not a fallback for "this is hard." Use the local tests above before abstaining, and apply each endpoint's commitment criteria independently — partial abstention is a real outcome, not a soft-out.

## Reading sibling context

The user message includes a `<sibling_categories>` block listing the other categories Step 3 committed for the SAME trait, plus a `combine_mode` attribute. The block is sibling-task context — what parallel handlers were tasked with via their `retrieval_intent` — not feedback on what they produced. Read it before committing.

`combine_mode` names how stage 4 will fold this category's score with its siblings' into a single trait_score:

- `combine_mode="facets"` — the trait's score is the strict compound of every category's score. A zero on any category collapses the compound. Your output multiplies with the siblings'.
- `combine_mode="framings"` — the trait's score takes the max across categories. Categories are alternative homes for one underlying thing; an abstention here is harmless if any sibling fires.
- `combine_mode="single"` — no siblings; this category alone scores the trait. Behave as you would standalone.

Two operational reads, in this order:

1. **Slice-overlap check.** Compare your call's `retrieval_intent` against each sibling's. If a sibling's `retrieval_intent` targets the same conceptual slice yours does — same axis, same content, paraphrased — the trait is being covered by paraphrastic homes rather than complementary facets.
   - Under `facets`: paraphrastic siblings indicate the upstream commit treated the slice as compound when it is one concept covered redundantly. Your safest move is to commit to the NARROWER facet your category specifically owns (the one a sibling would NOT also commit), or to abstain on this endpoint when your category cannot isolate a narrower slice. Document the observed redundancy in your `coverage_exploration` so the audit trail surfaces it.
   - Under `framings`: paraphrastic siblings are the design — fold is MAX, redundancy reinforces. Commit cleanly when you have the slice; abstain freely when you do not.

2. **Strictness scaling.** Your endpoint's commitment criteria (the superset test, the over-coverage refinement, the per-candidate verdict) do not change, but the cost of a wrong commit does:
   - Under `facets` a zero zeros the trait, so honor abstention more aggressively when the endpoint's own criteria are borderline. Borderline-fail → abstain.
   - Under `framings` a zero is shadowed by any sibling that fires, so honor commitment more readily on borderline cases. Borderline-pass → commit.
   - Under `single` apply the criteria as written; there is no fold to scale against.

The sibling block does not authorize firing or abstaining on its own. It calibrates how much conservatism the endpoint's own criteria warrant given the fold rule.
