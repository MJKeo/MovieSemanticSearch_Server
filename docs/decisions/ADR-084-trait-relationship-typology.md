# [084] — V4 trait relationship typology: INDEPENDENT / POSITIONING_REFERENCE / POSITIONING_QUALIFIER

## Status
Active

## Context
Step 2 previously used a role+salience pair to describe how a trait related to the
query's intent. Role values like "primary", "secondary", "contextual" were ambiguous and
often produced inconsistent LLM output. More importantly, the system had no way to
distinguish between a trait the user actually wants (INDEPENDENT) and a reference film
used only to position what the user wants (POSITIONING_REFERENCE), or a qualifier that
modifies another trait without being retrievable on its own (POSITIONING_QUALIFIER).

## Decision
Replace role+salience with a `relationship_role` enum having three values:
- `INDEPENDENT` — the trait stands alone as a direct retrieval target.
- `POSITIONING_REFERENCE` — the trait is a reference (e.g., "like Inception") used to
  anchor what the user wants; Step 3 applies identity-vs-attribute rule: extract only
  attributes of the reference, not the identity itself.
- `POSITIONING_QUALIFIER` — the trait modifies or qualifies another trait in context
  but is not itself a retrieval target.

Orphaned POSITIONING_REFERENCE/QUALIFIER traits (not associated with an INDEPENDENT
trait) are coerced to INDEPENDENT by a Step 2 validator self-heal pass.

## Alternatives Considered
- **Keep role+salience with clearer definitions**: Tested but the multi-axis encoding
  still produced LLM inconsistency; the fundamental issue is that role and salience
  conflate two orthogonal dimensions.
- **Binary PRIMARY/SECONDARY**: Loses the POSITIONING distinction entirely; reference
  films would be retrieved as direct targets.

## Consequences
- POSITIONING_REFERENCE enables correct handling of "like X" queries: X's identity
  is not retrieved; X's attributes (themes, tone, period) inform retrieval.
- Orphan coercion prevents silent score gaps when the LLM misclassifies.
- Step 3 must respect the identity-vs-attribute rule for POSITIONING_REFERENCE traits
  — this is a non-obvious downstream constraint.

## References
- docs/modules/search_v2.md — Step 2 section, Step 3 gotchas
- search_v2/step_2.py, search_v2/step_3.py
- ADR-074: interpret-verdict-decompose-first (predecessor design)
