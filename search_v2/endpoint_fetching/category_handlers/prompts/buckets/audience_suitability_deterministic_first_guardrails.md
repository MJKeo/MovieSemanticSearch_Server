# Failure-mode guardrails

**NEVER:**
- COMMIT AN ENDPOINT WHOSE WALK SHOWS NO USEFUL CANDIDATE. Walks are the audit; verdicts in `coverage_commitments` read off them.
- DROP AN ENDPOINT WHOSE WALK CONTRIBUTES A DISTINCT STRENGTH OR FILLS ANOTHER'S WEAKNESS. Suitability is multi-faceted; deduplicating to a single endpoint loses real signal.
- COLLAPSE TO A SINGLE COMMIT OUT OF CONSERVATISM. The default posture for this bucket is multiple commits. Drop only when another endpoint dominates the same content strictly better, or its walk surfaced no useful candidate.
- DECLARE A SLICE UNSERVABLE. There is no `intentionally_uncovered` field. Either fire an endpoint that can carry the slice, or drop the candidate that surfaced it.
- FLIP POLARITY AT THE PARAMETER LEVEL. Parameters describe presence of an attribute. Whether that presence helps the user is decided when signals are combined later.
- HEDGE in `coverage_exploration`. Apply the local tests per endpoint and commit a stance.
- ABSTAIN ON AN ENDPOINT BY OMISSION. Every declared endpoint requires an explicit `verdict_reason` → `verdict` in `coverage_commitments`. Silence is not abstention.

Before emitting, check for these failure modes. Whole-call abstain (every endpoint's `coverage_commitments.{route}.verdict == "abstain"`) is correct ONLY when all endpoint walks surfaced no useful candidate.

- **Ambiguous requirement.** The call's intent is too vague to point at concrete candidates in any walk. Do not invent specificity that is not in the call.
- **Out of scope for every endpoint.** No walk surfaces a candidate with substantive `strengths`. Verdict abstain on every endpoint. Do not pick the least-bad option.
- **Self-contradictory requirement.** Modifiers on the call flip the requirement against itself in a way no endpoint can express.

Beyond those, watch for these composition traps:

- **Walking abstractly.** A walk that describes the endpoint's general fitness ("metadata can pin a maturity ceiling") instead of concrete candidates with strengths/weaknesses defeats the grounding the walk exists to enforce. Walks must name actual columns / registry members / vector spaces and frame their `strengths` and `weaknesses` operationally.
- **Mismatching strength to channel.** A clear maturity ceiling or hard exclusion belongs on metadata (gate-style sharpness). A soft preference belongs on keyword or semantic scoring (graded signal). Match the strength the call actually carries — but use this to choose WHICH endpoint, not whether to drop a softer overlapping signal that another endpoint can carry.
- **Treating one endpoint's clean coverage as license to skip another.** Overlap is the point. A maturity gate (metadata) and a wholesome-tone semantic query both pulling on "suitable for kids" is the design. Skip an endpoint only when its walk surfaced no useful candidate, not when its angle is partially captured elsewhere.
- **Treating an endpoint as a rubber stamp** — committed or abstained without consideration of its walk. Each `verdict=commit` in `coverage_commitments` must point at a real candidate from its walk; each `verdict=abstain` should be justifiable per the drop test.

Make the coverage call **explicit** in `coverage_exploration` — name which endpoints contribute distinct strengths, where their weaknesses are filled by another endpoint, and which (if any) are dominated and dropped.
