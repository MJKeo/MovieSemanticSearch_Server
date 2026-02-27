# Trending Metric Guide

## What It Represents

A measure of **current cultural attention** — what movies people are watching, discussing, and searching for right now. This is distinct from the two other audience signals in the system:

| Signal | Question it answers | Data source |
|--------|-------------------|-------------|
| **Trending** (`prefers_trending_movies`) | Is this movie buzzing *right now*? | TMDB weekly trending endpoint → Redis Hash |
| **Reception** (`reception_preference`) | Was this movie *well-made* in critics'/audiences' eyes? | `reception_score` on `movie_card` (IMDb + Metacritic) |
| **Popularity** (`prefers_popular_movies`) | Would a random person on the street *recognize* this movie? | `popularity_score` on `movie_card` (IMDb vote count) |

Trending is volatile by nature. A movie can be #1 trending this week and off the list entirely next month. This is the correct behavior — the signal captures temporal relevance, not lasting fame.

---

## Data Source

**TMDB Weekly Trending endpoint** — returns movies ordered by engagement over the past 7 days (page views, votes, watchlist additions, social activity). The top **N = 500** movies are fetched. The rank position within this list is the only input to scoring.

**Why N = 500:** When a user says "trending," they're picturing maybe 20–50 movies. But hard metadata filters (genre, decade, provider, maturity) applied in the UI can eliminate 85–90% of candidates. With N = 100, aggressive filtering leaves 10–15 trending candidates — too thin for the reranker. With N = 500, 50–75 typically survive, which is workable. Beyond 500, TMDB's trending tail stops representing anything a user would call "trending."

---

## Score Computation: Concave Decay

Each movie's trending score is computed from its rank position using a concave (square root) decay:

```
score = 1 - (rank / 500) ^ 0.5
```

Where `rank` is 1-indexed (rank 1 = most trending).

| Rank | Score | Interpretation |
|------|-------|----------------|
| 1 | 0.96 | Top trending, near-max boost |
| 10 | 0.86 | Definitely trending, strong boost |
| 25 | 0.78 | Clearly trending, strong boost |
| 50 | 0.68 | Trending, solid boost |
| 100 | 0.55 | Moderately trending, meaningful boost |
| 250 | 0.29 | Marginally trending, small boost |
| 400 | 0.11 | Barely trending, negligible boost |
| 500 | 0.00 | Bottom of list, no boost |

**Why concave decay over binary:** Binary scoring (in/out of set) says the 500th movie is equally trending as #1. That's wrong. But binary with a small N (100) breaks under hard filtering.

**Why concave decay over linear:** Linear (`1 - rank/N`) wastes discriminative power at the top. From the user's perspective, #1 and #10 are both "definitely trending" — the top ~100 should all score high with gentle differentiation. The square root curve is flat at the top and steep at the bottom, matching this intuition.

**Why concave decay over sigmoid:** Sigmoid solves a threshold problem (popularity needed one because "widely known" is a binary concept applied to a continuous input). Trending is already a curated list where every entry has some claim to being trending. The question is how much, not whether. Decay fits better than threshold here.

**Scores are absolute, not relative to the search result set.** If hard filters leave only 8 trending candidates, they keep their original scores. Re-normalizing to [0, 1] within the filtered set would inflate a barely-trending movie to 1.0 just because it was the most trending survivor.

---

## Redis Storage

### Key and Type

```
trending:current   — Hash
```

Fields map `movie_id` (string) → precomputed trending score (float as string). Scores are computed at write time so the reranking loop does a single dict lookup per candidate with no math.

### Write Pattern — Atomic RENAME

The daily cron job refreshes the key atomically to avoid any window where the key is empty or partially populated:

```
1. Fetch top 500 weekly trending movie IDs from TMDB (in rank order)
2. Compute score per movie: 1 - (rank / 500) ^ 0.5
3. Write to staging key:
     DEL trending:next
     HSET trending:next <id1> <score1> <id2> <score2> ... <idN> <scoreN>
4. Atomically swap:
     RENAME trending:next trending:current
```

`RENAME` is atomic in Redis — it replaces `trending:current` in a single operation with no gap. **Do not** use `DEL trending:current` + `HSET trending:current ...` in sequence — that creates a window where the key is absent, causing concurrent requests to silently skip the trending bonus.

### TTL: None

`trending:current` carries no TTL. The key lives indefinitely and is replaced atomically by each daily run. This is deliberate:

- With a TTL, a delayed or failed cron job causes the key to expire before the replacement arrives, creating a silent gap.
- With no TTL, a failed job leaves a stale-but-valid trending set in place until the next successful run. Stale trending data is a much better failure mode than missing trending data.

Under the `volatile-lru` memory policy, this TTL-less key is immune to eviction.

### Storage Cost

500 entries × ~20 bytes per entry (movie ID + float string) ≈ 10 KB. Negligible.

---

## Refresh Cadence

**Daily**, as part of the existing cron cycle. The cron job fetches the weekly trending list from TMDB and overwrites the Redis Hash each run. Even though the source data is TMDB's "weekly" trending window, refreshing daily captures the rolling nature of that window — the list shifts as new engagement data accumulates throughout the week.

---

## Read Pattern (Per Search Request)

During Step 5b of the request flow, the server fetches the entire Hash once and loads it into an in-memory dict:

```python
# Step 5b — one call per request, concurrent with Postgres bulk fetch
raw = redis.hgetall("trending:current")
trending_scores = {int(k): float(v) for k, v in raw.items()}
```

If the key is missing (cron has never run, or the key was manually deleted), treat the dict as empty and log a `WARNING` — do not fail the request. Missing trending data is a graceful degradation.

All membership checks and score lookups during reranking are O(1) in-process dict lookups. Redis is never queried per-candidate.

---

## Reranking Integration

When `prefers_trending_movies` is `true` in the query understanding output:

1. **No additional data fetch.** The trending dict was loaded in Step 5b.
2. **Score the candidate.** Look up `trending_scores.get(candidate.movie_id, 0.0)`. The value is already in [0, 1] and precomputed — no per-candidate math.
3. **Average into metadata score.** Fold the trending score in as one metadata preference component alongside genre, decade, runtime, maturity, watch providers, audio language, reception, and (if active) the popularity score. All components are averaged equally.

When `prefers_trending_movies` is `false` (the default), the trending dict is still fetched in Step 5b (it's cheap and runs concurrently) but the score is not included in the metadata average.

```python
# Step 6 — reranking loop
for candidate in candidates:
    trending_score = trending_scores.get(candidate.movie_id, 0.0)  # O(1)
```

### Example

Query: *"Trending horror movies"*

Active metadata preferences for a candidate:
- Genre match (Horror): **1.0** (is a horror film)
- Trending: **0.86** (rank 10 in trending list)

Metadata score = (1.0 + 0.86) / 2 = **0.93**

This feeds into the final score formula:

```
score = w_L * lexical + w_V * vector + w_M * 0.93 + P * session_penalty
```