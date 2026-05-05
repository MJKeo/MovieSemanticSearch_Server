# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **leaving the affected endpoint unassigned with the rationale captured in `intentionally_uncovered` is the correct response** — not fabricated parameters on endpoints that do not genuinely fit.

- **Ambiguous requirement.** The call's intent is too vague to point at concrete candidates in any walk. Do not invent specificity that is not in the call.
- **Out of scope for every endpoint.** No walk surfaces a clean fit. Leave `coverage_assignments` empty and name the unaddressable aspects in `intentionally_uncovered`. Do not fire every endpoint "just in case."
- **Self-contradictory requirement.** Modifiers on the call flip the requirement against itself in a way no endpoint can express.

Beyond those, watch for these combination-specific traps:

- **Walking abstractly.** A walk that describes the endpoint's general fitness ("metadata can pin a maturity ceiling") instead of concrete candidates ("maturity_rating capped at PG-13 covers the suitability ceiling; no column captures the parental-consent angle") defeats the grounding the walk exists to enforce. Walks must name actual columns / registry members / vector spaces and explicit covers/misses prose.
- **Committing past the walk.** An assignment whose endpoint's walk surfaced no clean signal is a fabrication. The walks are the audit; commitments must be readable off them.
- **Skipping a real signal because another endpoint already covers it.** Overlap is the point. A maturity gate (metadata) and a wholesome-tone semantic query both pulling on "suitable for kids" is the design — do not deduplicate them at the commitment phase. Skip an endpoint only when its walk has nothing distinct to add, not when its angle is already partially captured by another assignment.
- **Mismatching strength to channel.** A clear maturity ceiling or hard exclusion belongs on metadata. A soft preference belongs on keyword or semantic scoring. Match the strength the call actually carries — but do not use this as cover for skipping a softer overlapping signal that another endpoint can carry.
- **Flipping polarity at the parameter level.** Parameters describe presence of an attribute. Whether that presence helps the user is decided when the signals are combined later — do not encode the direction into the parameter itself.
- **Silently skipping an endpoint without naming the gap.** If an endpoint's walk surfaced potential candidates but you choose not to assign, there should be a corresponding entry in `intentionally_uncovered` naming the aspect you walked away from. Treating any endpoint as a rubber stamp — fired or skipped without consideration — is the failure this bucket is designed to prevent.

Beyond those: each fired endpoint's assignment must point at a real candidate from its walk — overlap with another assignment is welcome, but firing on no candidate is not. Do not collapse to a single assignment out of conservatism when multiple endpoints genuinely apply.
