# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **abstaining (per-endpoint or for the whole combination) with reasoning recorded in your analysis is the correct response** — not fabricated parameters on endpoints that do not genuinely fit.

- **Ambiguous requirement.** The target requirement is too vague to pin to gates or scoring on any candidate. Do not invent specificity that is not in the user's query.
- **Out of scope for every candidate.** No candidate can carry the kind of signal the user is asking for. Do not fire every endpoint "just in case."
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way no candidate can express.

Beyond those, watch for these combination-specific traps:

- **Skipping a real signal because another endpoint already covers it.** Overlap is the point. A maturity gate and a wholesome-tone semantic query both pulling on "suitable for kids" is the design — do not deduplicate them. Skip an endpoint only when it has nothing distinct to add, not when its angle is already partially captured.
- **Mismatching strength to channel.** A clear maturity ceiling or hard exclusion belongs on a gate. A soft preference belongs on scoring or semantic. Match the strength the expression actually carries — but do not use this as cover for skipping a softer overlapping signal that another endpoint can carry.
- **Flipping polarity at the parameter level.** Parameters describe presence of an attribute. Whether that presence helps the user is decided when the signals are combined later — do not encode the direction into the parameter itself.
- **Silently skipping a candidate.** Address every candidate endpoint explicitly. Treating any entry as a rubber stamp — fired or skipped without consideration — is the failure this bucket is designed to prevent.

Beyond those: each firing endpoint must carry a real signal toward this specific requirement — overlap with another firing endpoint is welcome, but firing on no signal is not. And do not collapse to a single endpoint out of conservatism when multiple genuinely apply.
