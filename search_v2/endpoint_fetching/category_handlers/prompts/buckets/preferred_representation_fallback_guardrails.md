# Failure-mode guardrails

**NEVER:**
- COMMIT AN ENDPOINT WHOSE WALK SHOWS NO USEFUL CANDIDATE. The walks are the audit; verdicts in `coverage_commitments` must be readable off them.
- DROP AN ENDPOINT WHOSE WALK CONTRIBUTES A DISTINCT STRENGTH OR FILLS ANOTHER'S WEAKNESS. Drop only when another endpoint dominates the same content strictly better, or its walk surfaced no useful candidate.
- FIRE A SINGLE ENDPOINT WHEN ANOTHER ENDPOINT'S WALK CLEANLY FILLS A WEAKNESS. The most common failure: keyword's walk shows over-coverage (e.g. SPORT pulls football and basketball alongside running), semantic's walk has a clean candidate that isolates the slice, but only keyword fires. The over-coverage signal exists for exactly this case.
- DECLARE A SLICE UNSERVABLE. There is no `intentionally_uncovered` field. Either fire an endpoint that can carry the slice, or drop the candidate that surfaced it.
- HEDGE in `coverage_exploration`. Apply the local tests per endpoint and commit a stance.
- SPLIT ONE SLICE ACROSS ENDPOINTS to look thorough. Let the endpoint that owns the slice own it cleanly.
- ABSTAIN ON AN ENDPOINT BY OMISSION. Every declared endpoint requires an explicit `verdict_reason` → `verdict` in `coverage_commitments`. Silence is not abstention.

Before emitting, check for these failure modes. Abstaining at the whole-call level (every endpoint's `coverage_commitments.{route}.verdict == "abstain"`) is correct ONLY when ALL endpoint walks surfaced no useful candidate.

- **Ambiguous requirement.** The call's intent is too vague to point at concrete candidates in any walk. Do not invent specificity that is not in the call.
- **Out of scope for every endpoint.** No walk surfaces a candidate with substantive `strengths`. Verdict abstain on every endpoint. Do not pick the least-bad option.
- **Self-contradictory requirement.** Modifiers on the call flip the requirement against itself in a way no representation can express.

Beyond those, watch for these walk-vs-commitment traps:

- **Walking abstractly.** A walk that describes the endpoint's general fitness ("keyword can cover registry-style elements") instead of concrete candidates with strengths/weaknesses defeats the grounding the walk exists to enforce. Walks must name actual registry members / vector spaces / columns and frame their `strengths` and `weaknesses` operationally.
- **Treating "fully covered" as license to drop other endpoints.** A keyword candidate marked as fully covering may STILL pull more than the slice — its `weaknesses` should call out the over-coverage. If it does, a sibling endpoint that isolates the slice still fires.
- **Treating priority order as a license to skip the fallback.** Priority is a tiebreaker for slices that two endpoints could equivalently own. It does not override the fire test (does this endpoint contribute net signal?).

Make the coverage call **explicit** in `coverage_exploration` — name which endpoints contribute distinct strengths, where their weaknesses are filled by another endpoint, and which (if any) are dominated and dropped.
