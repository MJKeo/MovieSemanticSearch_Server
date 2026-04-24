# Failure-mode guardrails

Before emitting, check for these failure modes. In all three cases, **`endpoint_to_run: "None"` with reasoning captured in `requirement_aspects` is the correct response** — not picking the least bad candidate and fabricating parameters.

- **Ambiguous requirement.** The target requirement is too vague to point at concrete parameters on any candidate. Do not invent specificity that is not in the user's query.
- **Out of scope for every candidate.** Every candidate endpoint leaves substantial gaps; none is a genuine fit. Dispatch was wrong — do not pick the "least bad" option.
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way no candidate can express.

**When a candidate does fit, pick exactly one.** Mutually exclusive means what it says: firing more than one would mix answers to different questions rather than reinforce one.
