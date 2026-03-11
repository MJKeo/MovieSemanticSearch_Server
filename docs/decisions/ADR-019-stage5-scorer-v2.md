# ADR-019 — Stage 5 Quality Scorer v2: Signal Redesign and Per-Group Thresholds

## Status
Active

## Context

ADR-016 introduced the first combined TMDB+IMDB quality scorer at Stage 5.
After building diagnostic tooling (analyze_imdb_quality.py, plot_quality_scores.py)
and inspecting the score distribution, two structural problems emerged:

1. **Mixed signal ranges broke threshold reasoning.** Signals ranged from
   [-1, +1] (watch_providers, featured_reviews_chars, lexical_completeness,
   data_completeness) to [0, 1] (others). A single threshold on the composite
   score was hard to reason about, and absence of common fields produced negative
   sub-scores that punished data-sparse but otherwise notable movies.

2. **watch_providers signal (weight 0.20) conflated two different populations.**
   Movies with and without US watch providers have fundamentally different
   quality distributions. Penalizing no-provider movies in the score — rather
   than routing them to a stricter threshold — made it impossible to tune
   separately for streaming vs. non-streaming populations.

## Decision

Redesign the Stage 5 scorer to use 8 signals all in [0, 1], remove all 10
hard filters (score is the sole filter mechanism), replace the watch_providers
signal with per-group thresholds, and add two new signals.

**Three provider groups with separate thresholds** (determined via survival
curve derivative analysis after scoring):
- `has_providers` — movies with ≥1 US watch provider (lenient threshold)
- `recent_no_providers` — no providers, released ≤75 days ago (lenient)
- `old_no_providers` — no providers, released >75 days ago (strictest)

**8-signal model** (weights sum to 1.0, all [0, 1]):

| Signal | Weight | Category |
|--------|--------|----------|
| imdb_vote_count | 0.27 | Relevance |
| critical_attention | 0.12 | Relevance |
| community_engagement | 0.08 | Relevance |
| tmdb_popularity | 0.08 | Relevance |
| featured_reviews_chars | 0.15 | Data sufficiency |
| plot_text_depth | 0.12 | Data sufficiency |
| lexical_completeness | 0.10 | Data sufficiency |
| data_completeness | 0.08 | Data sufficiency |

Two new signals replace the removed watch_providers and metacritic_rating:
- **critical_attention (0.12)**: presence of metacritic_rating + reception_summary.
  0/2→0.0, 1/2→0.5, 2/2→1.0. Pure bonus; absence is normal.
- **community_engagement (0.08)**: weighted presence of plot_keywords (1),
  featured_reviews (2), plot_summaries (3), synopses (4). Score = sum of
  present sub-weights / 10.

## Alternatives Considered

1. **Keep hard filters, adjust weights**: Rejected. Hard filters on "no directors"
   or "no runtime" pre-empt movies that Stage 5's richer data can score fairly.
   Removing them delegates all filtering to the score, which is more calibrated.

2. **Keep watch_providers as a signal**: Rejected. Encoding provider presence
   as a weight distorts the score in ways that are hard to compensate with
   a single threshold. Per-group thresholds cleanly separate the populations
   without contaminating the score itself.

3. **Normalize the composite score post-hoc**: Considered but rejected. The
   per-group threshold approach makes normalization unnecessary; each group
   is analyzed independently via its own survival curve.

4. **Keep metacritic_rating as a standalone signal (0.04)**: Replaced with
   critical_attention (0.12) because metacritic alone missed the directional
   signal. reception_summary presence is a stronger proxy for critical engagement
   and the combined signal better captures the "does anyone care?" dimension.

## Consequences

- ADR-016's scoring model is superseded. ADR-016's motivation (combined
  TMDB+IMDB scorer, IMDB primary) remains valid.
- Scorer only writes scores and advances to `imdb_quality_calculated`. A
  separate threshold-filter script (not yet written) reads scores and
  advances to `imdb_quality_passed` per provider group.
- score_popularity() in scoring_utils.py is now parameterized with `log_cap`;
  Stage 5 passes STAGE5_POP_LOG_CAP=4.0. Stage 3 still uses the default (11).
- lexical_completeness and data_completeness remapped from [-1, +1] to [0, 1]
  (total/6 instead of (total-3)/3), making all signals interpretable as 0=worst
  to 1=best.
- featured_reviews_chars remapped from [-1, +1] tiers to [0, 1] tiers.
- All existing `essential_data_passed` movies were reset to `imdb_scraped`
  for re-scoring under the v2 model (one-time migration in tracker.py).

## References

- ADR-016 (combined IMDB quality scorer v1) — superseded scoring model
- ADR-017 (Stage 3 scorer redesign) — precedent for per-group threshold analysis
- docs/modules/ingestion.md (Stage 5 section) — full signal specifications
- movie_ingestion/imdb_quality_scoring/quality_score_design.md — design spec
- movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py
- movie_ingestion/scoring_utils.py
