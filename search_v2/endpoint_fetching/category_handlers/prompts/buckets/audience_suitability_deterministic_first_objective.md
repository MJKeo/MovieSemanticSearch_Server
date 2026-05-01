# Objective

This category maps to **multiple candidate endpoints that can each fire in parallel**, introduced above. Audience-suitability requirements are inherently multi-faceted: the user often expresses both a clear-cut boundary (a maturity ceiling, content the user wants to avoid) and a softer fit (a tone the user wants more of, a watch-context that should feel right). Deterministic gates capture the first kind; scoring signals capture the second; semantic intensity fills in what the gates and scoring cannot reach.

Your task: determine which combination of endpoints — including the empty combination (none fire) — best covers the target requirement, and fill in parameters for each endpoint that should fire.

Work through the decision in this priority:

1. **Deterministic gates first.** Where the expression states a clear maturity ceiling, content exclusion, or suitability boundary, emit gate-style parameters. Gates carry the strongest commitments the user makes.
2. **Inclusion scoring next.** Where the expression names content, tone, or fit the user wants more of, emit inclusion-scoring parameters.
3. **Exclusion scoring next.** Where the expression names content the user wants to avoid that is not strict enough to gate, emit exclusion-scoring parameters.
4. **Semantic intensity or watch-context last.** Where the expression carries intensity, mood, or watch-context that gates and scoring cannot fully capture, add semantic-channel parameters to fill the gap.

Address every candidate endpoint with an explicit fire-or-abstain decision. Silent skipping is the failure mode this bucket exists to prevent — if an endpoint should not fire, say so deliberately rather than passing it over.

**Polarity rule: emit presence of an attribute, not direction.** Endpoint parameters describe what the content has. Whether that presence helps or hurts the user is decided when the signals are combined later — do not encode that decision into the parameter itself.

**Declining to fire — per-endpoint or for the whole combination — is a valid outcome.** Only fire endpoints that carry real signal for this specific requirement.
