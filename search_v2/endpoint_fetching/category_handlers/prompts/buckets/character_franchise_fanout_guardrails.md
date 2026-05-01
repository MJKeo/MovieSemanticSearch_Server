# Failure-mode guardrails

Before emitting, check for these failure modes. In each case, **declining both paths with reasoning recorded in your analysis is the correct response** — not fabricated parameters.

- **Ambiguous requirement.** The expression does not actually name a referent that is both a character and a franchise. A character whose name does not anchor a franchise (or vice versa) belongs to a different category — say so rather than forcing a fan-out.
- **Out of scope.** The referent is not retrievable through these paths at all (e.g. the named entity is not present in the registries the paths search).
- **Self-contradictory requirement.** Modifiers on the parent fragment flip the referent against itself in a way the paths cannot express.

Beyond those, watch for these fan-out-specific traps:

- **Dropping a path because the other "already covers it."** Both retrievals are required by design — collapsing to one drops the signal this fan-out exists to preserve. If both paths fit the referent, both must fire.
- **Diverging on the referent.** Resolve the referent once. Do not let the two paths target subtly different entities (the character version vs. some adjacent universe variant) — that defeats the purpose of fanning out from a single named referent.
- **Folding adaptation or source-medium signals into either path.** Those belong to separate category calls. Keep them out of the character and franchise payloads here.
