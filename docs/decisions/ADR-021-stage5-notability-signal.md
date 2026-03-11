# ADR-021 — Stage 5 Scorer v4: Vote Count × Bayesian Rating Notability Signal

## Status
Active

## Context

Manual threshold evaluation on the `has_providers` group (v3 scorer)
revealed movies with very low IMDB ratings (e.g. 2.4/10) but moderate
vote counts scoring comparably to genuinely notable films. The v3
`imdb_vote_count` signal was a pure log-scaled vote count — it had no
quality discrimination. Two movies with the same vote count scored
identically regardless of whether one was rated 2/10 and the other 8/10.

## Decision

Replace the `imdb_vote_count` signal with `imdb_notability`: a
vote-count × Bayesian-adjusted-rating blend with three confidence tiers.

**Three tiers** (derived from rating-stability analysis of ~113K
has_providers movies):
- Low (< 100 votes): blend 95/5 vote/rating. Std ~1.37; 8+ ratings
  inflated 3–7× by self-selection — rating is noise.
- Medium (100–999 votes): blend 70/30. 8+ inflation gone; 13–15%
  sub-4.0 — rating has real signal.
- High (≥ 1000 votes): blend 85/15. Movie is already notable; vote
  count dominates, rating modulates modestly.

**Bayesian rating formula** (m=500, C=6.0): shrinks noisy low-vote
ratings toward the population mean before blending. Falls back to
pure vote count when `imdb_rating` is absent.

**Weight changes**: imdb_notability 0.25→0.31, critical_attention
0.12→0.08, community_engagement 0.10→0.08. The higher weight reflects
notability being the primary relevance discriminator.

## Alternatives Considered

1. **Hard filter on IMDB rating (e.g. < 4.0 = filtered_out)**: Rejected.
   Hard filters are brittle for low-vote movies where ratings are
   unreliable. A weighted blend degrades gracefully across vote count.

2. **Add IMDB rating as a separate signal (new weight)**: Rejected.
   Rating without vote-count context is misleading. A 2/10 from 5 votes
   is noise; the same rating from 50,000 votes is a strong signal.
   Blending them into one signal (with tier weights) correctly models
   this dependency.

3. **Simple linear combination of vote count and raw rating**: Rejected.
   A raw 2/10 from 10 votes would still penalise movies that haven't
   accumulated enough votes yet. The Bayesian prior correctly shrinks
   extreme ratings toward the mean at low vote counts.

## Consequences

- Scorer is now v4. Existing `imdb_quality_calculated` movies need to be
  reset to `imdb_scraped` and re-scored if a threshold recalibration is
  required.
- `imdb_quality_scorer.py` imports only `score_popularity` from
  `scoring_utils.py` for the notability signal; the Bayesian blend logic
  is self-contained in `_score_imdb_notability()`.
- Per-group thresholds from v3 analysis are no longer valid — the score
  distribution shifts upward because high-vote-count garbage movies are
  now penalised. New thresholds: has_providers=0.486,
  no_providers_new=0.55, no_providers_old=0.654.

## References

- ADR-019 (Stage 5 scorer v2) — introduced the 8-signal model this builds on
- ADR-016 (combined IMDB quality scorer v1) — original scorer
- docs/modules/ingestion.md (Stage 5 section) — full signal specification
- movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py
- movie_ingestion/scoring_utils.py — IMDB_QUALITY_THRESHOLDS
