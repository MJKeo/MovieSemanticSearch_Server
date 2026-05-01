# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **abstaining (per-endpoint or for the whole combination) with reasoning recorded in your analysis is the correct response** — not fabricated parameters on endpoints that do not genuinely fit.

- **Ambiguous requirement.** The target requirement is too vague to pin to gates or scoring on any candidate. Do not invent specificity that is not in the user's query.
- **Out of scope for every candidate.** No candidate can carry the kind of signal the user is asking for. Do not fire every endpoint "just in case."
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way no candidate can express.

Beyond those, watch for these combination-specific traps:

- **Substituting scoring for a gate (or vice versa).** A clear maturity ceiling or hard exclusion belongs in a gate, not a soft scoring signal. Conversely, a soft preference does not belong in a gate. Match the strength the expression actually carries.
- **Flipping polarity at the parameter level.** Parameters describe presence of an attribute. Whether that presence helps the user is decided when the signals are combined later — do not encode the direction into the parameter itself.
- **Using semantic intensity to bypass an explicit gate.** When the user states a hard boundary, the gate fires. Adding semantic intensity around the same content does not replace the gate.
- **Silently skipping a candidate.** Address every candidate endpoint explicitly. Treating any entry as a rubber stamp — fired or skipped without consideration — is the failure this bucket is designed to prevent.

Beyond those: do not fire an endpoint just because it is available — each firing endpoint must carry distinct, complementary signal toward this specific requirement. And do not collapse to a single endpoint out of conservatism when multiple genuinely apply.
