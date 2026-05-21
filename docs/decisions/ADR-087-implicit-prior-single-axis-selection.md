# [087] — Implicit prior: single-axis selection (popularity primary, quality fallback)

## Status
Active

## Context
Stage 4's implicit prior injects a baseline score for every candidate based on the
film's intrinsic quality, to prevent obscure films from ranking above well-known ones
when trait scores are otherwise equal. Two axes were originally considered: popularity
(engagement signal) and quality (critical/reception signal). Using both as separate
weighted axes made the prior formula complex and introduced a tuning parameter with no
principled value.

## Decision
Use a single-axis implicit prior with fallback:
- **Primary axis**: popularity, when the query's context makes popularity relevant
  (i.e., `popularity` signal is active in the query understanding output).
- **Fallback axis**: quality, when popularity is inactive (e.g., prestige or arthouse
  queries where popularity is a poor proxy for what the user wants).
- The sigmoid curves used to map raw popularity/quality scores to prior weights are
  shared with the metadata endpoint to maintain consistency.
- `prior_base = positive_total if positive_total > 0 else 1.0` — prior weight is
  anchored to the sum of positive trait weights so it scales with query complexity.

## Alternatives Considered
- **Always use quality as prior**: Quality is available for all films; popularity has
  nulls for obscure titles. But for mainstream queries, quality alone over-ranks
  critically acclaimed obscurities at the expense of well-known films the user
  is more likely to have heard of.
- **Blend popularity and quality at fixed ratio**: Adds a tuning parameter with no
  principled value; single-axis with context-driven selection is simpler and more
  interpretable.
- **No implicit prior**: Results become very sensitive to small trait score differences
  among equally-matching candidates, producing arbitrary tie-breaking.

## Consequences
- Prior behavior differs by query type; popularity-inactive queries use quality prior.
  This distinction must be correctly signaled by Step 0/1.
- `prior_base` scaling means the prior's influence shrinks for simple one-trait queries
  and grows for complex multi-trait queries — a desirable property.
- Sigmoid curves shared with metadata endpoint must not diverge; changes in one path
  affect both.

## References
- docs/modules/search_v2.md — Stage 4 implicit prior section
- search_v2/step_2.py (popularity activation signal), search_v2/full_pipeline_orchestrator.py
