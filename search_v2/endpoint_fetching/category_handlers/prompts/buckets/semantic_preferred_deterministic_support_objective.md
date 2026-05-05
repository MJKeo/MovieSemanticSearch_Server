# Objective

This category routes to **semantic plus one or more deterministic candidates**, introduced above. Semantic typically owns the bulk of the call's intent because the call is graded / experiential / tonal; the deterministic endpoints exist to catch binary or canonical signals that the embeddings blur across (a named season, a known holiday phrase, a registry-tagged tone).

Your task: walk every declared endpoint concretely, decide who fires on what slice, then fill thin parameters for each endpoint that fires. Three sequential phases, each grounded in the prior one:

1. **Per-endpoint walks.** For each declared endpoint, fill its `{route}_walk` block with concrete candidates — vector spaces for semantic, registry members / columns for the deterministic candidates. The walks are independent: each asks "what could this endpoint cover for the call's intent?" Surface every plausibly useful candidate so the commitment phase reads off real options.

2. **Coverage commitment.** Read every walk and decide who owns what. Semantic typically appears in `coverage_assignments` because graded / experiential signal is its native domain; deterministic endpoints appear when their walks surface a clean canonical or binary signal complementary to semantic. Overlap with the semantic slice is welcome — different channels catch different signal sharpness. Skip a deterministic endpoint only when its walk surfaced no clean signal. List anything no endpoint can cleanly serve in `intentionally_uncovered`.

3. **Thin per-endpoint parameters.** For each endpoint with an assignment, fill its `{route}_parameters` block. The wrapper's `{route}_retrieval_intent` mirrors the slice_description from the matching assignment; the inner parameters draw on that intent and the endpoint's walk above to commit the route-specific translation.

**Semantic abstaining is unusual but valid.** If semantic's walk surfaces no useful space coverage, it should not be assigned. Do not force a semantic assignment when nothing in the 7 spaces carries the call's intent.
