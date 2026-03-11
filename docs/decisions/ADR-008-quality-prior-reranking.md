# ADR-008: Quality-Prior Reranking

**Status:** Active

## Context

Without quality awareness, low-quality movies can rank above
high-quality ones when their relevance scores happen to tie or
nearly tie. Users expect that among equally relevant results,
well-received movies appear first.

## Decision

Apply a quality-prior reranking step after the composite relevance
score is computed:

1. **Normalize reception**: Map `reception_score` (0-100 scale
   from IMDb/Metacritic) to [0, 1] using floor=30.0, ceil=90.0.
   Missing scores default to 0.5 (neutral).

2. **Bucket by relevance**: Round `final_score` to 2 decimal
   places (BUCKET_PRECISION=2).

3. **Sort within buckets by reception**: Within each 0.01-wide
   relevance band, sort by normalized reception score descending.

### Special Cases

- If `reception_preference` is `poorly_received`, disable the
  quality prior (set to 0.0) — user wants bad movies, don't
  fight them.
- When `reception_preference` is `critically_acclaimed`, reception
  already participates as a metadata preference AND as the quality
  prior. This stacking is fine because bucketing means the prior
  only matters among candidates with similar relevance.

## Alternatives Considered

1. **Add reception as a score component**: Would allow low-quality
   but highly relevant results to be pushed down by a great movie
   that's only slightly relevant. Bucketing avoids this.
2. **No quality prior**: Ties broken arbitrarily, leading to
   inconsistent result ordering and occasional low-quality results
   appearing prominently.
3. **Finer buckets (3 decimal places)**: Too granular — almost
   no ties, quality prior rarely activates.

## Consequences

- The 0.5 default for missing reception scores prevents movies
  without ratings from systematically sinking to the bottom.
- This is the final sorting step before returning results.

## References

- docs/modules/db.md (reranking constants and gotchas)
