# Popularity Metric Guide

## What It Represents

A measure of **long-term mainstream cultural reach** — how widely known a movie is in the general population. This is distinct from the two other audience signals in the system:

| Signal | Question it answers | Data source |
|--------|-------------------|-------------|
| **Trending** (`prefers_trending_movies`) | Is this movie buzzing *right now*? | Redis `trending:current` set (daily cron) |
| **Reception** (`reception_preference`) | Was this movie *well-made* in critics'/audiences' eyes? | `reception_score` on `movie_card` (IMDb + Metacritic) |
| **Popularity** (`prefers_popular_movies`) | Would a random person on the street *recognize* this movie? | `popularity_score` on `movie_card` (IMDb vote count) |

A blockbuster can have terrible reviews (high popularity, low reception). A cult classic can be universally acclaimed but obscure (low popularity, high reception). A viral meme movie can be trending without being a household name (high trending, low popularity). These three axes are independent.

---

## Raw Data: IMDb Vote Count

**Why vote count:** It's a lifetime engagement metric. A movie accumulates votes only when someone watches it and cares enough to rate it. High vote counts correlate with cultural awareness — passive recognition alone doesn't generate votes, but movies people have actually seen and discussed do. It is the most accessible, consistently populated proxy for mainstream reach available from the APIs we already use.

**Why not other signals:**

- TMDB `popularity` is a rolling/decaying metric (recent page views, edits, social activity). It measures current attention, not lifetime reach. It's a second trending signal, not a popularity signal.
- Box office revenue requires inflation adjustment, era-aware normalization, and TMDB's `revenue` field is incomplete (many entries are 0 or null, especially pre-1990 and non-Hollywood). Too unreliable for a core metric.
- TMDB vote count is correlated with IMDb vote count but sparser and noisier for older films. Not worth the added complexity as a starting signal. Can be reconsidered later if failure cases emerge.

### Schema

```sql
ALTER TABLE public.movie_card
ADD COLUMN imdb_vote_count INT,
ADD COLUMN popularity_score FLOAT NOT NULL DEFAULT 0;
```

`imdb_vote_count` stores the raw count fetched from IMDb. `popularity_score` stores the final precomputed reranking-ready score (see Scoring section below). Both live directly on `movie_card` so they're returned in the existing Step 5a bulk fetch with no additional queries or joins.

---

## Percentile Calculation

The raw vote count is converted to a global percentile rank across the entire catalog:

```sql
CREATE MATERIALIZED VIEW mv_popularity_percentile AS
SELECT
  movie_id,
  PERCENT_RANK() OVER (ORDER BY imdb_vote_count ASC) AS percentile
FROM public.movie_card
WHERE imdb_vote_count IS NOT NULL;
```

`PERCENT_RANK()` produces a value in [0, 1] representing the fraction of movies with fewer votes. It is purely ordinal — only relative position matters, not magnitude. This means:

- Heavy-tailed distributions are handled automatically (no log transform needed).
- Outliers at the top (a single movie with 3M votes) don't compress everyone else downward.
- The output is stable across catalog changes — adding movies shifts neighbors slightly, never drastically.

Movies with `imdb_vote_count IS NULL` are excluded from ranking and default to `popularity_score = 0`.

---

## Scoring: Sigmoid Transform (Precomputed)

The raw percentile is **not** used directly for reranking. A sigmoid transform is applied to convert it into a score that behaves like the other metadata preference components.

**Why not use the percentile directly:** Most metadata preferences (genre match, decade match, maturity match) produce near-binary scores — a movie either matches or it doesn't. Their distributions cluster near 0 and 1. A raw percentile is uniformly distributed across [0, 1], giving it disproportionate variance in the equal-weight metadata average. It would dominate the other preferences. The sigmoid compresses the score toward 0 or 1, matching the distribution shape of the other components so equal averaging works correctly.

**Why threshold at 0.70:** "Popular" is a high bar. The median movie in the catalog is not "popular" in the way users mean it. A threshold at the 70th percentile means roughly the top 30% of movies by engagement begin receiving meaningful credit, which aligns with the user intent behind "blockbuster" and "household name."

### Formula

```
popularity_score = 1 / (1 + exp(-k * (percentile - threshold)))
```

With `threshold = 0.70` and `k = 15`, this produces:

| Percentile | Score | Interpretation |
|-----------|-------|----------------|
| 0.90+ | ~0.97 | Mega-blockbuster, household name |
| 0.80 | ~0.82 | Well-known mainstream film |
| 0.70 | ~0.50 | Transition zone |
| 0.60 | ~0.18 | Below the "popular" threshold |
| 0.50 | ~0.04 | Not mainstream |
| < 0.40 | ~0.00 | Obscure |

### Precomputation

The sigmoid is computed once after the percentile view refreshes, then written back to `movie_card`. It is **never** computed at query time.

```sql
-- Step 1: Refresh percentile view
REFRESH MATERIALIZED VIEW mv_popularity_percentile;

-- Step 2: Write sigmoid-transformed scores back to movie_card
UPDATE public.movie_card mc
SET popularity_score = 1.0 / (1.0 + exp(-15.0 * (mv.percentile - 0.70)))
FROM mv_popularity_percentile mv
WHERE mc.movie_id = mv.movie_id;

-- Step 3: Zero out any movies that weren't in the percentile view
UPDATE public.movie_card
SET popularity_score = 0
WHERE imdb_vote_count IS NULL;
```

At query time, `popularity_score` is just a float that comes back in the existing bulk fetch. No computation, no joins, no additional queries.

---

## Refresh Cadence

Two independent refresh cycles: how often `imdb_vote_count` is re-fetched from the API per movie, and how often the percentile view + score UPDATE runs.

### Vote count refresh (API fetches)

| Tier | Movie age | Refresh frequency | Rationale |
|------|-----------|-------------------|-----------|
| 1 | Released ≤ 30 days ago | Daily | Initial accumulation spike; counts can jump by orders of magnitude in the first month. API cost is small (30–100 movies). |
| 2 | Released 31–180 days ago | Weekly | Accumulation has slowed but the movie hasn't settled. Weekly captures the gradual climb without wasting API calls. |
| 3 | Released > 180 days ago | Monthly | Essentially static. A monthly sweep is data hygiene, not a meaningful signal update. |

### Percentile + score refresh (materialized view + UPDATE)

Runs **daily** as part of the existing ingestion cycle, immediately after any vote count updates have been written. The computation is a single sort + window function over the catalog followed by a bulk UPDATE — takes seconds regardless of catalog size. It runs against whatever counts are currently stored, so Tier 2/3 movies with slightly stale counts still get correct relative positioning.

---

## Reranking Integration

When `prefers_popular_movies` is `true` in the query understanding output:

1. **No additional data fetch.** `popularity_score` is already on `movie_card` and comes back in the existing Step 5a Postgres bulk query.
2. **Score the candidate.** Read `popularity_score` directly — it's already in [0, 1] and sigmoid-shaped. No per-candidate computation.
3. **Average into metadata score.** Fold `popularity_score` in as one metadata preference component alongside genre, decade, runtime, maturity, watch providers, audio language, reception, and (if active) the trending bonus. All components are averaged equally.

When `prefers_popular_movies` is `false` (the default), `popularity_score` is ignored entirely — it does not participate in the metadata average.

### Example

Query: *"Popular 80s action movies"*

Active metadata preferences for a candidate:
- Decade match (1980–1989): **1.0** (released 1984, matches)
- Genre match (Action): **1.0** (is an action film)
- Popularity: **0.82** (80th percentile, sigmoid-transformed)

Metadata score = (1.0 + 1.0 + 0.82) / 3 = **0.94**

This then feeds into the final score formula:

```
score = w_L * lexical + w_V * vector + w_M * 0.94 + P * session_penalty
```