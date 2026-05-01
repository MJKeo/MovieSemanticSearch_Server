# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **abstaining or limiting the response with reasoning recorded in your coverage analysis is the correct response** — not fabricated parameters.

- **Ambiguous requirement.** The target requirement is too vague to point at concrete parameters on any representation. Do not invent specificity that is not in the user's query.
- **Out of scope for every representation.** No representation can answer the requirement. Do not pick the least-bad option.
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way no representation can express.

Beyond those, watch for these coverage-specific traps:

- **Firing fallback when the preferred representation already covers.** When the preferred representation cleanly captures the requirement, adding a fallback mixes interpretations and adds noisy duplicate signal. Stop at the preferred representation.
- **Stopping at the preferred representation when uncovered intent remains.** When the preferred representation handles only part of the requirement and meaningful qualifiers, long-tail terms, or spectrum/intensity remain uncovered, the fallback is required to capture them. Do not flatten those parts away.

Make the coverage call **explicit** in your reasoning — name which parts of the requirement each fired representation handles, and which (if any) intentionally go uncovered.
