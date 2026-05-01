# Objective

This category maps to **a semantic retrieval channel that always fires, plus any deterministic surfaces that can run alongside it**, introduced above. The user is expressing meaning that is broad, evaluative, or experiential — semantic prose is the authoritative read. Deterministic tags, numeric priors, or popularity priors are layered on top whenever they can catch a binary or canonical signal that semantic search tends to flatten. Running both in parallel is strictly stronger than running either alone, so overlap with the semantic read is welcome.

Your task: generate the semantic query that captures the expression's core meaning, and additionally generate every deterministic signal the expression cleanly implies — even when it overlaps with what the semantic query already covers.

Work through the decision in order:

1. Read the target requirement and articulate its core meaning. Phrase the semantic query to surface that meaning against the relevant ingestion-text style — the category-specific notes that follow give the template.
2. Scan each available deterministic surface and ask: would this endpoint catch a binary or canonical signal (a named tag, a pinned number, a stated popularity prior) that semantic retrieval may blur across? If yes, generate it. Overlap with the semantic read is not a reason to skip.
3. Emit the semantic query plus every deterministic signal that clears that bar.

**Semantic-only is still a valid outcome** when no deterministic surface has a clean signal to add. Missing a clean deterministic match is not a failure. But when a clean deterministic signal exists, fire it alongside the semantic query — do not suppress it on the grounds that semantic "already covers" the meaning.
