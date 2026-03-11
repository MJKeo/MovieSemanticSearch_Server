# ADR-017 — Stage 3 TMDB Quality Scorer Redesign (4-signal + edge-case model)

## Status
Active

## Context

The original Stage 3 quality scorer (ADR-002) used a 10-signal weighted
model with 5 hard filters. As the pipeline matured, two things changed:

1. **IMDB scraping cost dropped substantially** (GraphQL migration in
   ADR-009 eliminated the need for 6 HTML page fetches per movie,
   reducing proxy bandwidth ~6x). This changed the cost-of-error
   calculus: a false exclusion at Stage 3 now costs much less to avoid.

2. **Stage 5 became the real quality gate.** With a rich combined
   TMDB+IMDB scorer (ADR-016) available downstream, Stage 3 only needs
   to remove obvious junk — not make fine-grained judgments from
   TMDB-only data.

The old design had several problems:
- 5 hard filters (zero_vote_count, missing duration, missing overview,
  no genres, future release) excluded edge cases that Stage 5 handles
  better with IMDB data.
- 10 signals introduced weight-calibration complexity that wasn't
  justified by Stage 3's actual role.
- With-provider movies were scored through the same formula despite
  providers being a near-decisive signal, creating false precision.

## Decision

Replace the 10-signal + 5-hard-filter Stage 3 scorer with a 4-signal
weighted formula plus two edge-case bypasses. Remove all hard filters
from Stage 3. Separate scoring and filtering into two distinct scripts
(scorer first, then filter), matching the Stage 5 pattern.

**Edge cases** (bypass formula):
- Unreleased (release_date > today) → score 0.0 (auto-exclude)
- Has ≥1 US watch provider → score 1.0 (auto-pass)

**4-signal formula** (no-provider population only):

| Signal | Weight | Notes |
|--------|--------|-------|
| vote_count | 0.50 | Log cap 101 (calibrated to no-provider p99=72) |
| popularity | 0.20 | Log cap 11 |
| overview_length | 0.15 | 5-tier by character count |
| data_completeness | 0.15 | Average of 8 binary metadata indicators |

**Threshold**: 0.2344 — the f' minimum (peak attrition rate) from
Gaussian-smoothed survival curve derivative analysis on the
no-provider population.

## Alternatives Considered

1. **Keep 5 hard filters, simplify soft scorer**: Rejected because
   the hard filters duplicate logic better handled by Stage 5, which
   has IMDB data. "No overview" at Stage 3 might still produce a
   passable movie after IMDB enrichment.

2. **Keep 10-signal scorer, adjust weights**: Rejected because the
   with-provider bypass makes many signals redundant for that
   population, and 4 signals are sufficient to rank the no-provider
   population for a lenient gate.

3. **Inline scoring inside the filter script** (original design):
   Rejected in favor of the two-script pattern, which allows survival
   curve re-analysis without re-running the scorer. Matches the
   Stage 5 pattern for consistency.

4. **Use TMDB watch_providers as a tiered signal** (0/1-2/3+ platforms):
   Rejected — the Stage 3 question is simply "does this movie have any
   US distribution?" The binary bypass is cleaner and avoids calibrating
   tier boundaries on limited data.

## Consequences

- Stage 3 now passes all movies with US watch providers unconditionally,
  regardless of vote count, overview quality, or other signals. This
  is intentional: Stage 5 handles that discrimination with richer data.
- The no-provider population's quality distribution is bimodal: a long
  tail of low-engagement movies and a smaller cluster of genuine films
  without US streaming (rights gaps, older releases). The 0.2344
  threshold removes the long tail while keeping the rights-gap cluster
  for Stage 5 to adjudicate.
- Removing hard filters means Stage 3 no longer writes `filtered_out`
  statuses for structural data problems — only for score < threshold.
- `VoteCountSource.TMDB_NO_PROVIDER` (log cap 101) was added to
  `scoring_utils.py` alongside the existing IMDB source (log cap 12,001).
- ADR-002's scoring model is superseded by this ADR. ADR-002's
  architectural rationale (TMDB-first cost funnel) remains valid.

## References

- ADR-002 (TMDB-first quality funnel) — superseded scoring model;
  cost funnel rationale still active
- ADR-009 (IMDB GraphQL migration) — cost reduction that changed
  the error calculus for Stage 3 exclusions
- ADR-016 (combined IMDB quality scorer) — the downstream gate
  that makes Stage 3 leniency safe
- `docs/modules/ingestion.md` (Stage 3 section) — full signal
  specifications and threshold derivation
- `movie_ingestion/tmdb_quality_scoring/tmdb_quality_scorer.py`
- `movie_ingestion/tmdb_quality_scoring/tmdb_filter.py`
- `movie_ingestion/tmdb_quality_scoring/plot_tmdb_quality_scores.py`
