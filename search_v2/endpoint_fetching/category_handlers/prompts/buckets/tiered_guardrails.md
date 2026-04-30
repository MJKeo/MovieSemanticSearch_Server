# Failure-mode guardrails

Before emitting, check for these general failure modes. In all three cases, **`endpoint_to_run: "None"` with reasoning in `requirement_aspects` and `performance_vs_bias_analysis` is the correct response** — not picking a weak candidate and fabricating parameters.

- **Ambiguous requirement.** The target requirement is too vague to point at concrete parameters on any candidate.
- **Out of scope for every candidate.** No candidate can actually answer the requirement. The bias does not force a pick when no candidate genuinely fits.
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way no candidate can express.

**Bias-specific pitfalls:**

- **Do not pick the highest-biased endpoint when it leaves a real gap** that a lower-biased endpoint cleanly fills. The bias is a tiebreaker, not a veto on lower-preference endpoints.
- **Do not over-correct against the bias.** When endpoints fit roughly equally, the bias *is* the tiebreaker — use it.
- **Make the bias reasoning explicit in `performance_vs_bias_analysis`.** State whether one endpoint wins on its own merits or whether the bias decided a close call. Do not leave this implicit.

**When a candidate does fit, pick exactly one.**
