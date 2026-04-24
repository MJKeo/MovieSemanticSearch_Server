# Failure-mode guardrails

Before emitting, check for these failure modes. In all three cases, **`should_run_endpoint: false` with reasoning captured in `requirement_aspects.coverage_gaps` is the correct response** — not fabricated parameters.

- **Ambiguous requirement.** The target requirement is too vague to point at any concrete parameter. Do not invent specificity that is not in the user's query.
- **Out of scope for this endpoint.** The endpoint's vocabulary, vector space, or metadata column does not cover what the user is asking for. Dispatch was wrong — say so in `coverage_gaps` rather than forcing a degraded query through the wrong channel.
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way the endpoint cannot express (e.g. a polarity modifier inverts the atomic rewrite and the endpoint has no negation semantics). Record the contradiction in `coverage_gaps`.
