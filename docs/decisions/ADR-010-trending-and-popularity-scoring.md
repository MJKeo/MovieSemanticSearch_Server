# ADR-010: Trending and Popularity Scoring Designs

**Status:** Active

## Context

The system needs two distinct audience signals beyond reception
quality: "trending right now" (temporal) and "widely known"
(permanent). These are independent axes — a cult classic can be
high popularity but not trending; a viral meme movie can be
trending but not popular.

## Decision

### Trending: Concave Decay from TMDB Weekly Ranking

**Source**: TMDB weekly trending endpoint, top N=500 movies.
**Formula**: `score = 1 - (rank / 500) ^ 0.5`
**Storage**: Redis Hash `trending:current` with precomputed scores.
**Refresh**: Daily via cron, atomic RENAME (no TTL — stale data
is better than missing data).
**Write pattern**: DEL staging key → HSET staging → RENAME to
current. Never DEL+HSET the live key (creates a gap).

| Rank | Score |
|------|-------|
| 1 | 0.96 |
| 10 | 0.86 |
| 50 | 0.68 |
| 100 | 0.55 |
| 250 | 0.29 |
| 500 | 0.00 |

**Why N=500**: Hard metadata filters eliminate 85-90% of candidates.
With N=100, aggressive filtering leaves only 10-15 trending results.
With N=500, 50-75 typically survive.

**Why concave, not linear**: Linear wastes discriminative power at
the top. Users see #1 and #10 as both "definitely trending." Square
root curve is flat at the top, steep at the bottom.

**Why not sigmoid**: Trending is already a curated list. The
question is "how much trending," not "whether trending." Decay
fits better than threshold.

**Scores are absolute, not re-normalized per search result set.**

### Popularity: Sigmoid of Vote Count Percentile

**Source**: IMDb vote count → global percentile rank → sigmoid.
**Formula**: `popularity_score = 1 / (1 + exp(-15 * (percentile - 0.70)))`
**Storage**: Precomputed on `movie_card.popularity_score` in Postgres.
**Refresh**: Daily materialized view refresh + UPDATE.

| Percentile | Score |
|-----------|-------|
| 0.90+ | ~0.97 |
| 0.80 | ~0.82 |
| 0.70 | ~0.50 |
| 0.60 | ~0.18 |
| < 0.40 | ~0.00 |

**Why sigmoid, not raw percentile**: Most metadata preferences
produce near-binary distributions (match or not). A uniform
percentile has disproportionate variance in the weighted average,
dominating other preferences. Sigmoid compresses toward 0/1.

**Why threshold at 0.70**: "Popular" is a high bar. The median
movie is not "popular" in the way users mean it. Top 30% by
engagement begins receiving meaningful credit.

**Why IMDb vote count, not TMDB popularity**: TMDB popularity is
a rolling/decaying metric (current attention), which is what
trending already captures. Vote count is lifetime engagement —
a distinct signal.

### Vote Count Refresh Cadence

| Tier | Movie Age | Frequency |
|------|-----------|-----------|
| 1 | ≤ 30 days | Daily |
| 2 | 31-180 days | Weekly |
| 3 | > 180 days | Monthly |

## Alternatives Considered

1. **Binary trending (in/out of set)**: The 500th movie would
   equal #1. That's wrong.
2. **Box office revenue for popularity**: Requires inflation
   adjustment, era normalization. TMDB's revenue field is
   incomplete (many zeros pre-1990).
3. **TMDB vote count**: Correlated with IMDb but sparser for
   older films.

## Consequences

- Trending is volatile by design — can change weekly.
- Popularity is stable — changes slowly as votes accumulate.
- Both participate as metadata preference components alongside
  genre, decade, etc., only when the LLM sets
  `prefers_trending_movies` or `prefers_popular_movies` to true.

## References

- docs/modules/db.md (trending_movies.py and scoring constants)
