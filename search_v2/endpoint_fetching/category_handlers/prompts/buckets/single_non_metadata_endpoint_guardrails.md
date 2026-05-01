# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **declining to fire with reasoning recorded in your gap analysis is the correct response** — not fabricated parameters.

- **Ambiguous requirement.** The target requirement is too vague to point at any concrete parameter. Do not invent specificity that is not in the user's query.
- **Out of scope for this endpoint.** The endpoint's vocabulary, vector space, or retrieval surface does not cover what the user is asking for. Dispatch was wrong — say so in your gap analysis rather than forcing a degraded query through the wrong channel.
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way the endpoint cannot express. Record the contradiction in your gap analysis.

The endpoint is fixed for this category — there is no routing decision to make beyond fire-or-abstain.
