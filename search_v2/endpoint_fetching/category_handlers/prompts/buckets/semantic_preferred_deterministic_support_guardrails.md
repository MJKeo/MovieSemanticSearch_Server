# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **emitting only the semantic query, or declining to fire entirely, with reasoning recorded in your gap analysis is the correct response** — not fabricated deterministic parameters.

- **Ambiguous requirement.** The target requirement is too vague to phrase as semantic prose. Do not invent specificity that is not in the user's query.
- **Out of scope.** The semantic channel cannot capture what the user is asking for at all. Dispatch was wrong — record this rather than forcing a degraded query through.
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the requirement against itself in a way the channel cannot express.

Beyond those, watch for these augmentation-specific traps:

- **Leading with deterministic tags.** Starting from a tag-first read flattens evaluative, experiential, and canonical-stature meaning into a categorical label. The semantic query is the primary read — generate it first, deterministic augmentation only after.
- **Adding speculative augmentation.** Add a deterministic signal only when the expression cleanly implies it. "Plausible adjacency" is not enough — if the user did not name a tag, pin a number, or state a popularity prior, do not add one.
- **Suppressing augmentation because semantic already covers it.** Overlap is the point. When a clean deterministic signal exists, fire it alongside the semantic query even when the meaning is already in the semantic read — the deterministic surface catches binary or canonical signals that semantic retrieval blurs across.
- **Letting deterministic augmentation reshape the semantic read.** Deterministic signals reinforce the semantic query in parallel. They do not narrow, contradict, or filter it. If a deterministic match would change what the semantic query retrieves, drop the augmentation rather than degrade the primary read.
