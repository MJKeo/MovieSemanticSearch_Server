# ADR-007: Soft Metadata Scoring (Not Hard Filters)

**Status:** Active

## Context

The LLM extracts metadata preferences from user queries (genre,
decade, runtime, maturity, etc.). These could be implemented as
hard filters (exclude non-matching) or soft scores (penalize but
don't exclude).

## Decision

All LLM-extracted metadata preferences are treated as **soft
signals** in a weighted average, never hard filters. The UI
provides separate hard filter controls that users set directly.

### Scoring Formula

```
metadata_score = Σ(weight_i × score_i) / Σ(weight_i)
```

Sums over active preferences only. Inactive preferences (None
result, false boolean, no_preference) excluded from both
numerator and denominator.

### Preference Weights (static)

| Preference | Weight | Score Range |
|-----------|--------|-------------|
| genres | 5 | [-2, 1] (exclusion = -2.0) |
| release_date | 4 | [0, 1] (linear decay from range edges) |
| watch_providers | 4 | [0, 1] (0.5 for wrong method, right provider) |
| audio_language | 3 | [-2, 1] (exclusion = -2.0) |
| maturity_rating | 3 | [0, 1] (ordinal distance: off-by-1 = 0.5) |
| reception | 3 | [0, 1] (linear in [55,95] or [10,50] range) |
| duration | 2 | [0, 1] (30-minute grace period) |
| trending | 2 | [0, 1] (binary: in set or not) |
| popular | 2 | [0, 1] (precomputed sigmoid of percentile) |
| budget_size | 3 | [0, 1] (binary match) |

### Key Scoring Details

**Release date**: Grace period adapts to range width.
`BETWEEN` = clamp(range_width × 0.5, 365, 1825 days).
`AFTER/BEFORE` = 1095 days (3 years). `EXACT` = 730 days.

**Genres/Language**: Exclusion violations produce -2.0 — an
intentional hard penalty that drags the final score negative.

**Watch providers**: 1.0 if exact method+provider match, 0.5 if
same provider but different access method, 0.0 if no overlap.

**Trending**: Uses precomputed scores from Redis Hash (concave
decay: `1 - (rank/500)^0.5`). See ADR-010.

**Popularity**: Uses precomputed sigmoid of vote count percentile
from Postgres. See ADR-010.

## Alternatives Considered

1. **Hard filters from LLM output**: Over-penalizes candidates
   when the LLM misinterprets the query. A user saying "something
   from the 90s" might still enjoy a 2001 film.
2. **Equal weights for all preferences**: Genre and decade matter
   more than duration for most queries. Static weights reflect this.
3. **Dynamic weights from LLM**: Added complexity; the channel-level
   `metadata_relevance` weight already controls how much the entire
   metadata score matters vs vector and lexical.

## Consequences

- Exclusion penalties (-2.0) can make metadata_score negative,
  which actively drags down the final score. This is intentional.
- The `channel_weights.metadata_relevance` controls the overall
  weight of metadata in the final formula. The per-preference
  weights here control relative importance within metadata.

## References

- docs/modules/db.md (metadata scoring description and constants)
