# Failure-mode guardrails

**NEVER:**
- COMMIT AN ASSIGNMENT WHOSE WALK SHOWS NO USEFUL CANDIDATE. Walks are the audit; assignments read off them.
- DROP A DETERMINISTIC ENDPOINT WHOSE WALK SURFACED A CLEAN CANONICAL OR BINARY CANDIDATE. Sharpness is exactly what deterministic adds on top of semantic's graded signal — overlap is the design.
- DROP SEMANTIC WHEN ITS WALK CONTRIBUTES GRADED / EXPERIENTIAL STRENGTH that no deterministic endpoint can match. Semantic carries the experiential core; deterministic adds gate-style sharpness, not replacement.
- DECLARE A SLICE UNSERVABLE. There is no `intentionally_uncovered` field. Either fire an endpoint that can carry the slice, or drop the candidate that surfaced it.
- LET DETERMINISTIC RESHAPE THE SEMANTIC SLICE. Deterministic assignments reinforce semantic in parallel. They do not narrow, contradict, or filter it.
- HEDGE in `coverage_exploration`. Apply the local tests per endpoint and commit a stance.

Before emitting, check for these failure modes. Whole-call abstain (empty `coverage_assignments`) is correct ONLY when all endpoint walks surfaced no useful candidate.

- **Ambiguous requirement.** The call's intent is too vague to point at concrete candidates in any walk. Do not invent specificity that is not in the call.
- **Out of scope for every endpoint.** No walk surfaces a candidate with substantive `strengths`. Empty `coverage_assignments`. Do not pick the least-bad option.
- **Self-contradictory requirement.** Modifiers on the call flip the requirement against itself in a way the endpoints cannot express.

Beyond those, watch for these composition traps:

- **Walking abstractly.** A walk that describes the endpoint's general fitness ("keyword can cover registry tags") instead of concrete candidates with strengths/weaknesses defeats the grounding the walk exists to enforce. Walks must name actual vector spaces / registry members / columns and frame their `strengths` and `weaknesses` operationally.
- **Treating semantic as default-only.** Semantic typically carries graded / experiential signal but is not the sole channel. When a deterministic walk surfaces a clean binary or canonical candidate, assign that endpoint alongside semantic — overlap is the point.
- **Treating deterministic over-coverage as fatal.** A deterministic candidate may pull more than the slice (its `weaknesses` says so). When semantic's `strengths` isolate the slice, both still fire — semantic refines what deterministic pulls broadly.
- **Splitting one slice across endpoints.** If semantic cleanly owns an aspect, let it own that aspect alone. Fragmenting it across deterministic endpoints is padding. (This is distinct from layering: layering = different sharpness on the same slice; splitting = different facets of the same atomic ask.)

Make the coverage call **explicit** in `coverage_exploration` — name which endpoints contribute distinct strengths, where their weaknesses are filled by another endpoint, and which (if any) are dominated and dropped.
