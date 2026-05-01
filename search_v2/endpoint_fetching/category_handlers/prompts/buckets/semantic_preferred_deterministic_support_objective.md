# Objective

This category maps to **a primary semantic retrieval channel with optional deterministic support**, introduced above. The user is expressing meaning that is broad, evaluative, or experiential — the kind of signal a tag or numeric prior tends to flatten. Semantic prose is the authoritative read; deterministic tags, numeric priors, or popularity priors only enter when the expression cleanly implies a structured signal.

Your task: generate the semantic query that captures the expression's core meaning, and add deterministic support only where the expression genuinely calls for it.

Work through the decision in order:

1. Read the target requirement and articulate its core meaning. Phrase the semantic query to surface that meaning against the relevant ingestion-text style — the category-specific notes that follow give the template.
2. Scan the expression for any structured signal that is cleanly implied — a tag the expression names, a numeric range it pins, a popularity prior it states. Add support only for signals that pass that bar.
3. Emit the semantic query plus whatever supporting signals genuinely apply. Use the minimum supporting set.

**Semantic-only is a valid and common outcome.** Missing a clean deterministic match is not a failure. Deterministic support is optional reinforcement, not a required component of every response.
