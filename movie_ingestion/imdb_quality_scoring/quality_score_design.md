# Stage 5 Quality Score Design

## Overview

A single quality score in [0, 1] determines whether a movie is worth
the cost of LLM metadata generation, embedding, and ingestion into
the search engine. The score is a weighted sum of 8 signals, each
normalized to [0, 1].

**No hard filters are applied.** The quality score is the sole filtering
mechanism. A movie passes or fails based entirely on whether its score
exceeds a threshold.

**Three separate thresholds** are applied, one per movie group:

1. **has_providers** — movies with at least one US watch provider.
2. **recent_no_providers** — no providers, released within the last
   75 days (theater window).
3. **old_no_providers** — no providers, released more than 75 days ago
   (or missing release date).

The grouping logic mirrors the watch-provider classification in
`analyze_imdb_quality.py` and uses `THEATER_WINDOW_DAYS` from
`scoring_utils.py`. Thresholds will be determined via survival-curve
analysis on the scored dataset.

## Design Criteria

A movie is worth including if both of the following are true:

1. **Relevant** — users would either search for it directly or would
   be likely to choose it as a movie to watch. Relevance does not
   require the movie to be critically acclaimed; cult status, notoriety,
   or niche appeal all count.
2. **Data-sufficient** — enough data exists to reliably generate LLM
   metadata and support all three search channels (lexical, vector,
   metadata). If key fields are missing or sparse, the movie cannot be
   surfaced reliably through non-title searches, making it a net
   negative in search results.

Movies that fail either criterion cost money (LLM + embedding) for no
gain and degrade search quality by cluttering results.

## Weight Allocation

| Category | Total Weight | Rationale |
|----------|-------------|-----------|
| Relevance | 0.55 | A movie must be worth finding. Slightly higher weight because even a data-rich movie that nobody wants is wasted cost. |
| Data sufficiency | 0.45 | A relevant movie with sparse data can't be surfaced through the search channels it needs to serve. |

## Signals

### Relevance Signals (0.55)

#### 1. imdb_vote_count — 0.27

The strongest single proxy for "people know this movie exists." A movie
with thousands of IMDB votes is definitionally relevant to some audience.
Movies with very few votes are obscure by definition.

- **Source:** IMDB `imdb_vote_count` field.
- **Normalization:** Log-scaled with age-adjustment multipliers for
  recency and classic status. Uses existing `score_vote_count()` from
  `scoring_utils` with the IMDB log cap (12,001).
- **Output:** [0, 1].
- **Implementation:** Existing, no changes.

#### 2. critical_attention — 0.12

Presence of professional critical coverage is a strong signal that a
movie crossed a mainstream attention threshold. These fields are rare
(metacritic: ~16% of has_providers, ~1% of old_no_providers;
reception_summary: ~1% of has_providers) so their presence is
highly discriminating.

This is a bonus signal — absence is normal and not penalized. But
presence is a clear positive indicator.

- **Source:** IMDB `metacritic_rating` and `reception_summary`.
- **Calculation:** Count presence of two fields:
  - metacritic_rating is not None → +1
  - reception_summary is non-empty string → +1
- **Scoring:** 0/2 → 0.0, 1/2 → 0.5, 2/2 → 1.0.
- **Output:** [0, 1].
- **Implementation:** New signal.

#### 3. community_engagement — 0.08

Measures whether a movie was important enough for people to voluntarily
contribute data to its IMDB page. Each contributing field type represents
someone's time investment — writing a review, a synopsis, a plot summary,
or tagging keywords.

No single field is a strong enough indicator on its own, but together
they form a picture of community investment. Fields are weighted
inversely to their prevalence: a synopsis (present on ~4% of
old_no_providers) represents far more engagement signal than plot
keywords (present on ~65%).

- **Source:** IMDB `featured_reviews`, `synopses`, `plot_summaries`,
  `plot_keywords`; TMDB `reviews` as fallback for featured_reviews.
- **Calculation:** Unequal-weighted presence composite:

  | Sub-field | Presence check | Sub-weight | Approx prevalence (old_no_providers) |
  |-----------|---------------|------------|--------------------------------------|
  | plot_keywords | IMDB list non-empty | 1 | 65% |
  | featured_reviews | IMDB list non-empty OR TMDB reviews non-empty | 2 | 45% |
  | plot_summaries | IMDB list non-empty | 3 | 18% |
  | synopses | IMDB list non-empty | 4 | 4% |

- **Scoring:** Sum sub-weights for present fields (max 10), divide by
  10. Example: reviews + plot_keywords present → (2 + 1) / 10 = 0.3.
- **Output:** [0, 1].
- **Implementation:** New signal.
- **Note on overlap with data sufficiency:** Some of these fields also
  feed data sufficiency signals (featured_reviews_chars, plot_text_depth).
  The overlap is intentional — presence here measures "someone cared
  enough to contribute" (relevance) while the data sufficiency signals
  measure "is there enough text to embed well" (depth). A movie with
  zero reviews gets correctly penalized on both dimensions.

#### 4. tmdb_popularity — 0.08

TMDB's algorithmic activity score captures short-term buzz and current
momentum that vote count alone doesn't reflect. Useful for surfacing
trending and newly-released movies.

However, TMDB popularity is biased toward a global audience and can be
distorted by extreme outliers. The saturation cap is set low so that
moderately popular movies already score 1.0, preventing viral outliers
from dominating the signal. A low score here doesn't necessarily mean
a movie isn't relevant — just that it isn't trending.

- **Source:** TMDB `popularity` field.
- **Normalization:** Log-scaled via `score_popularity()` from
  `scoring_utils`, with a lowered saturation cap so ~p75 of
  has_providers saturates at 1.0.
- **Output:** [0, 1].
- **Implementation:** Existing signal, lower the log cap in
  `score_popularity()`.

### Data Sufficiency Signals (0.45)

#### 5. featured_reviews_chars — 0.15

Total character count of review text. Reviews are the single most
important data source for LLM metadata generation — they feed 6 of
7 vector spaces (all except dense_anchor). A movie with no review
text will produce thin, unreliable embeddings for reception, viewer
experience, narrative techniques, watch context, plot analysis, and
production vectors.

- **Source:** IMDB `featured_reviews` text lengths (primary); TMDB
  `reviews` JSON (fallback when IMDB contributes zero chars).
- **Calculation:** Sum character lengths across all review texts.
  Apply tiered scoring:

  | Total chars | Score |
  |-------------|-------|
  | 0 | 0.0 |
  | 1–3,000 | 0.33 |
  | 3,001–8,000 | 0.67 |
  | 8,001+ | 1.0 |

- **Output:** [0, 1].
- **Implementation:** Existing signal, remap tiers from [-1, +1]
  to [0, 1].

#### 6. plot_text_depth — 0.12

Total character count of the movie's narrative text: overview +
plot summaries + synopses. This is the semantic backbone for
embeddings — the LLM needs enough plot and thematic material to
generate meaningful vector metadata.

These fields are substitutes, not complements. The total text budget
available to the LLM determines quality, regardless of which field
contributes it.

- **Source:** IMDB `overview`, `plot_summaries`, `synopses` (primary);
  TMDB `overview_length` as fallback for overview only.
- **Normalization:** log10(total + 1) / log10(5001), capped at 1.0.
  The 5,001-char cap means movies with rich synopses (~p75) saturate
  while overview-only movies (~150 chars) score ~0.59.
- **Output:** [0, 1].
- **Implementation:** Existing, no changes.

#### 7. lexical_completeness — 0.10

Measures how well the movie's named entities support lexical search.
Users search by actor, director, character name, franchise keywords.
Missing entity types mean entire query categories can't match this
movie.

Six entity types, each contributing a capped sub-score of 0.0–1.0.
Actors and characters use a threshold (≥5 for full score) because
having 1 actor vs 10 is a data quality concern, while 10 vs 80 is
a budget difference.

- **Source:** IMDB entity lists.
- **Calculation:**

  | Entity | Scoring |
  |--------|---------|
  | actors | 0 → 0.0, 1–4 → 0.5, 5+ → 1.0 |
  | characters | 0 → 0.0, 1–4 → 0.5, 5+ → 1.0 |
  | writers | 0 → 0.0, 1+ → 1.0 |
  | composers | 0 → 0.0, 1+ → 1.0 |
  | producers | 0 → 0.0, 1+ → 1.0 |
  | production_companies | 0 → 0.0, 1+ → 1.0 (IMDB primary, TMDB fallback) |

- **Scoring:** Sum sub-scores (max 6), divide by 6.
- **Output:** [0, 1].
- **Implementation:** Existing signal, remap from [-1, +1] to [0, 1].

#### 8. data_completeness — 0.08

Measures supplementary fields that enrich LLM-generated vector
metadata beyond the core entities and text. These fields improve
embedding quality for production, viewer experience, and watch
context vectors.

Distinct from lexical_completeness: covers supplementary attributes
(keywords depth, filming locations, content advisories) rather than
named entities for lexical matching.

- **Source:** IMDB fields with TMDB fallback where noted.
- **Calculation:**

  | Field | Scoring |
  |-------|---------|
  | plot_keywords | 0 → 0.0, 1–4 → 0.5, 5+ → 1.0 |
  | overall_keywords | 1 → 0.25, 2–3 → 0.5, 4+ → 1.0 |
  | filming_locations | 0 → 0.0, 1+ → 1.0 |
  | parental_guide_items | 0 → 0.0, 1+ → 1.0 |
  | maturity_rating | absent → 0.0, present → 1.0 (IMDB primary, TMDB fallback) |
  | budget | absent → 0.0, present → 1.0 (IMDB primary, TMDB fallback) |

- **Scoring:** Sum sub-scores (max 6), divide by 6.
- **Output:** [0, 1].
- **Implementation:** Existing signal, remap from [-1, +1] to [0, 1].

## Formula

```
quality_score = (
    0.27 * imdb_vote_count
  + 0.12 * critical_attention
  + 0.08 * community_engagement
  + 0.08 * tmdb_popularity
  + 0.15 * featured_reviews_chars
  + 0.12 * plot_text_depth
  + 0.10 * lexical_completeness
  + 0.08 * data_completeness
)
```

All signals ∈ [0, 1], weights sum to 1.0, so quality_score ∈ [0, 1].

## Threshold Strategy

After scoring all movies, three thresholds will be determined
independently via survival-curve analysis (Gaussian-smoothed derivative
to find inflection points):

- **has_providers threshold** — expected to be relatively low since
  these movies already have streaming availability (a strong relevance
  signal not captured in the score).
- **recent_no_providers threshold** — expected to be lenient since
  these are new releases that may gain providers soon.
- **old_no_providers threshold** — expected to be the strictest since
  these movies have no streaming path and need to justify inclusion on
  data quality and notability alone.

## Changes from Previous Scorer (v1)

| Aspect | v1 | v2 |
|--------|----|----|
| Hard filters | 10 predicates (no rating, no directors, etc.) | None — score is the sole filter |
| watch_providers signal | 0.20 weight, binary ±1.0 | Removed — handled by per-group thresholds |
| metacritic_rating signal | 0.04 weight, standalone | Folded into critical_attention (0.12) alongside reception_summary |
| Signal ranges | Mixed [0, 1] and [-1, +1] | All [0, 1] |
| Threshold | Single threshold for all movies | 3 separate thresholds by provider group |
| New signals | — | critical_attention, community_engagement |
