# [097] — Filter-active tiered promotion loop + semantic elbow calibration decoupling

## Status
Active

## Context
Two interacting issues surfaced when diagnosing empty results on queries like
"super violent and bloody" + rated-G hard filter:

1. The semantic elbow probe (`_run_corpus_topn`) was passing `metadata_filters`
   into the Qdrant `must` condition, so the elbow threshold was calibrated
   against the filtered slice. Small/noisy filtered slices trigger Kneedle's
   flat-distribution pathology or produce fake elbows in noise — manufacturing
   false positives or rejecting valid candidates.

2. The reranker→generator promotion fallback ran once pre-execution and fired
   only when no CANDIDATE_GENERATOR spec existed in the branch. When hard filters
   narrowed keyword/metadata generators to empty candidate pools, structural
   generator specs were still present and promotion never triggered — leaving the
   user with empty results even though a promoted semantic generator could have
   produced filter-compatible candidates.

## Decision
Two independent fixes that compose:

**Semantic elbow decoupling**: `_run_corpus_topn` is unconditionally unfiltered
(calibration-only). A new `_run_corpus_topn_filtered` is the separate filtered
probe used to populate the candidate pool on carver-unrestricted / promoted-
generator paths. In the no-filter case, one probe fires per space (byte-identical
to pre-fix); in the filter-active case, two parallel probes fire per space
(calibration + pool). The elbow threshold is now an absolute "is-about-X" bar
on the global corpus — not on the filtered slice.

**Per-branch tiered promotion loop in Phase B, filter-active only**: after the
initial generator dispatch, when `metadata_filters.is_active` and
`len(union) < CANDIDATE_FLOOR (25)`, find the lowest promotable tier among
the branch's rerankers not yet promoted, flip those specs to CANDIDATE_GENERATOR
(in place, `was_promoted=True`), re-dispatch the newly promoted specs through
the shared dedup-dispatch helper, merge into the union, and loop. Tiers are
parallel within a tier, serial across tiers. Tiers and their category memberships
live in `search_v2/promotion_tiers.py` (extracted to avoid an import cycle with
Stage 4). Promoted specs are removed from the reranker pass so their generator
score isn't overwritten. The unfiltered base case ("doesn't exist means doesn't
exist") is unchanged.

## Alternatives Considered
- **Always filter the elbow probe**: rejected. The elbow is meant to be a global
  "is this content about X" bar. A tight filter (e.g. rated-G only) would
  calibrate the elbow against a noise floor, inflating it and rejecting all
  candidates from a sparse-but-real filtered slice.
- **Single pre-execution promotion sweep (old behavior)**: insufficient — it
  only fired when no generator specs existed at all. Filter-narrowed generators
  that return empty pools look like generators that ran, not like missing generators.
- **Infinite promotion loop**: risk of runaway. Loop terminates when `CANDIDATE_FLOOR`
  is met or all promotable tiers are exhausted, whichever comes first. A neutral
  seed fallback handles the tier-exhausted case.

## Consequences
- No-filter invariant: any query without filters produces a byte-identical result
  list to pre-fix. The `filter_active` short-circuit in the semantic executor and
  Stage 4's filter-active gate together guarantee this.
- `was_promoted=True` propagates through `ScoreBreakdown` / `TraitContribution`
  for diagnostics.
- Cross-iteration dispatch dedup: a spec dispatched in an earlier tier is not
  re-dispatched if a later tier holds a structurally identical `(route, params)`
  spec (the dispatch cache carries results across iterations).

## References
- `search_v2/endpoint_fetching/semantic_query_execution.py`
- `search_v2/stage_4_execution.py` (Phase B loop, `_run_branch`)
- `search_v2/promotion_tiers.py` (`PromotionTier`, `_SEMANTIC_PROMOTION_TIERS`,
  `determine_promotion_tier`)
- `docs/modules/search_v2.md` (Stage 4 Phase B, Hard Filters sections)
- ADR-092 (original hard-filter threading)
