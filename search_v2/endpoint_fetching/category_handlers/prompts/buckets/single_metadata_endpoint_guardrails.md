# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **declining to fire with reasoning recorded in your gap analysis is the correct response** — not fabricated parameters.

- **Ambiguous requirement.** The target requirement is too vague to pin to a typed value, bound, or ordinal. Do not invent specificity that is not in the user's query.
- **Out of scope for this endpoint.** The endpoint's attribute does not carry the kind of signal the user is asking for, or the strength the user implied (e.g. a hard cutoff requested where the endpoint can only soft-rank). Record the mismatch in your gap analysis rather than forcing the wrong shape through.
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way the endpoint cannot express.

Beyond those, watch for these metadata-specific traps:

- **Conflating signal shape.** Hard filter, soft preference, additive prior, and ordinal selection are not interchangeable. Match the parameters to the strength the expression actually carries — do not promote a soft preference into a gate or vice versa.
- **Generalizing across attributes.** Decay, gating, and bucket choices are attribute-specific and live in the category notes. Do not apply the shape from one attribute to another.
