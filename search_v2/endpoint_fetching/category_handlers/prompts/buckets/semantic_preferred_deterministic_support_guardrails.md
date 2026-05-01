# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **emitting only the semantic query, or declining to fire entirely, with reasoning recorded in your gap analysis is the correct response** — not fabricated deterministic parameters.

- **Ambiguous requirement.** The target requirement is too vague to phrase as semantic prose. Do not invent specificity that is not in the user's query.
- **Out of scope.** The semantic channel cannot capture what the user is asking for at all. Dispatch was wrong — record this rather than forcing a degraded query through.
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way the channel cannot express.

Beyond those, watch for these semantic-specific traps:

- **Leading with deterministic tags.** Starting from a tag-first read flattens evaluative, experiential, and canonical-stature meaning into a categorical label. The semantic query is the primary read — generate it first, supporting signals only after.
- **Adding speculative support.** Add deterministic support only when the expression cleanly implies it. "Plausible adjacency" is not enough — if the user did not name a tag, pin a number, or state a popularity prior, do not add one.
- **Letting deterministic support shrink the semantic read.** Supporting signals reinforce the semantic query. They do not narrow, contradict, or filter it. If a deterministic match would change what the semantic query retrieves, drop the support rather than degrade the primary read.
