# Failure-mode guardrails

Before emitting, check for these general failure modes. In all three cases, **an empty combination (every endpoint's `should_run_endpoint: false`) with reasoning captured in `requirement_aspects` and `overall_endpoint_fits` is the correct response** — not fabricated parameters on endpoints that do not genuinely fit.

- **Ambiguous requirement.** The target requirement is too vague to point at concrete parameters on any candidate.
- **Out of scope for every candidate.** No candidate can actually answer the requirement. Do not fire every endpoint "just in case."
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way no candidate can express.

**Combo-specific pitfalls:**

- **Address every candidate explicitly.** The schema enforces it; the reasoning should reflect it. Every endpoint gets a considered `should_run_endpoint` decision — do not treat any entry as a rubber stamp.
- **Do not fire an endpoint just because it is available.** Each firing endpoint must carry distinct, complementary signal toward this specific requirement. Over-firing dilutes the result pool.
- **Do not fire only one endpoint out of conservatism** when multiple genuinely apply. Some requirements are inherently multi-faceted — collapsing to a single endpoint drops real signal that this bucket exists to preserve.
