# ADR-005: Vector Scoring Pipeline Design

**Status:** Active

## Context

Raw cosine similarity scores from Qdrant across 8 vector spaces
need to be combined into a single [0, 1] final vector score per
candidate. The pipeline must handle: spaces that didn't participate,
dual search results (original + subquery), varying relevance weights,
and score distributions that differ across spaces.

## Decision

5-stage pipeline:

### Stage 1: Execution Flags
Determine which searches ran per space. Key rule: if relevance is
NOT_RELEVANT but a subquery exists, promote effective relevance
to SMALL for weighting purposes (but `did_run_original` stays
False — the search decision was already made).

### Stage 2: Blend (80/20)
For spaces with both original and subquery results:
`blended = 0.8 * subquery + 0.2 * original`

Subquery-only spaces use 100% subquery. Original-only use 100%
original. The 80/20 ratio reflects that LLM-generated subqueries
are more targeted than the raw user query.

### Stage 3: Normalize (Exponential Decay)
Within each space, normalize blended scores using exponential
decay from the best score:

```
gap = (s_max - s_i) / (s_max - s_min)
normalized = exp(-k * gap)     where k = 3.0
```

Candidates with blended score 0.0 (not found in any search for
that space) get normalized score 0.0 and are excluded from the
statistical pool.

### Stage 4: Weight Normalization
Map RelevanceSize enums to raw weights (SMALL=1, MEDIUM=2,
LARGE=3). Anchor weight = 80% of the mean of active non-anchor
weights. Normalize all weights to sum to 1.0.

### Stage 5: Weighted Sum
`final_vector_score = Σ weight[space] * normalized[space]`

## Alternatives Considered

1. **Z-score normalization**: Distribution-dependent, requires
   clamping and rescaling. Exponential decay directly encodes
   "distance from best" which is more intuitive.
2. **Simple average across spaces**: Ignores LLM-assigned
   relevance weights. A query about "cozy rainy day movie"
   should weight viewer_experience higher than production.
3. **Re-fetching vectors for reranking**: Expensive and
   unnecessary. Qdrant scores are authoritative.

## Consequences

- Candidates appearing in few spaces are naturally penalized
  (zeroes for non-participating spaces). This is by design.
- The anchor vector always participates, providing general recall
  even when no specialized space matches.
- Tunable parameters: SUBQUERY_BLEND_WEIGHT (0.8), DECAY_K (3.0),
  ANCHOR_MEAN_FRACTION (0.8), RELEVANCE_RAW_WEIGHTS
  ({SMALL: 1.0, MEDIUM: 2.0, LARGE: 3.0}).

## References

- docs/modules/db.md (scoring pipeline constants and stages)
