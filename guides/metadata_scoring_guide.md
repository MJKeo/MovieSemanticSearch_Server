# Metadata Preference Scoring Guide

Computes a single `metadata_score` in roughly [-1, 1] for each candidate during reranking (Step 6). Only preferences actively set by the query contribute.

---

## Formula

```
metadata_score = Σ(weight_i × score_i) / Σ(weight_i)
```

Sums are taken over **active preferences only**. A preference is inactive (excluded from both numerator and denominator) when:

- Its `.result` is `None`
- `prefers_trending_movies` is `false`
- `prefers_popular_movies` is `false`
- `reception_type` is `no_preference`

---

## Weights

| Preference | Weight |
|---|---|
| `genres` | 5 |
| `release_date` | 4 |
| `watch_providers` | 4 |
| `audio_language` | 3 |
| `maturity_rating` | 3 |
| `reception` | 3 |
| `duration` | 2 |
| `trending` | 2 |
| `popular` | 2 |

Weights are static. The LLM-generated `channel_weights.metadata_relevance` already controls how much the entire metadata score matters relative to vector and lexical scores in the final formula.

---

## Per-Preference Scorers

### Release Date

**Score: [0, 1]**

Full credit if the movie's `release_ts` falls within the preferred range. Linear decay outside the range based on distance from the nearest edge.

```
grace_days:
  AFTER / BEFORE  → 1095 (3 years, fixed)
  EXACT           → 730  (2 years, fixed)
  BETWEEN         → clamp(range_width_days × 0.5, lower=365, upper=1825)

distance_days = distance from movie's release_ts to nearest range edge

score = max(0, 1 - distance_days / grace_days)
```

| Query | Range | Grace | 1yr out | 3yr out |
|---|---|---|---|---|
| "80s movies" | 10yr | 5yr | 0.80 | 0.40 |
| "2015 films" | 1yr | 1yr | 0.0 | 0.0 |
| "after 1980" | open | 3yr | 0.67 | 0.0 |

### Duration

**Score: [0, 1]**

Same shape as release date. Distance measured in minutes from nearest range edge.

```
GRACE_MINUTES = 30

score = max(0, 1 - distance_minutes / 30)
```

### Genres

**Score: [-2, 1]**

Check exclusions first. Any excluded genre present → **-2.0** (hard penalty). Otherwise, score inclusions as a fraction.

```
if any excluded genre is in movie's genre_ids → -2.0
if should_include is set → matched_count / len(should_include)
if only exclusions were set and none matched → 1.0
```

### Audio Language

**Score: [-2, 1]**

Same pattern as genres for exclusions.

```
if any excluded language is in movie's audio_language_ids → -2.0
if should_include is set → 1.0 if any match, else 0.0
if only exclusions were set and none matched → 1.0
```

### Watch Providers

**Score: [0, 1]**

Uses encoded `watch_offer_keys` (provider_id << 2 | method_id).

```
1. If should_exclude: build all (excluded_provider, any_method) keys.
   Any overlap with movie's keys → 0.0

2. If should_include:
   a. Build desired keys using preferred_access_type (or all 3 methods if unset)
   b. Any overlap → 1.0
   c. No overlap on desired method, but overlap on same provider + any method → 0.5
   d. No overlap at all → 0.0

3. No include preference set → 1.0
```

### Maturity Rating

**Score: [0, 1]**

Uses ordinal `maturity_rank` (G=1, PG=2, PG-13=3, R=4, NC-17=5). Unrated movies always score 0 unless unrated is the exact target.

```
if movie is unrated → 0.0 (unless target is unrated → 1.0)
if in range → 1.0
if off by 1 → 0.5
if off by 2+ → 0.0
```

Range is determined by `match_operation` (EXACT, GREATER_THAN, LESS_THAN, etc.). "Off by 1" means distance of 1 from the nearest edge of the valid range.

### Reception

**Score: [0, 1]**

Uses `reception_score` (0–100) from Postgres. Inactive when `reception_type` is `no_preference`.

```
critically_acclaimed:
  score = clamp((reception_score - 55) / 40, 0, 1)
  55 → 0.0 | 65 → 0.25 | 75 → 0.5 | 85 → 0.75 | 95 → 1.0

poorly_received:
  score = clamp((50 - reception_score) / 40, 0, 1)
  50 → 0.0 | 40 → 0.25 | 30 → 0.5 | 20 → 0.75 | 10 → 1.0

Null reception_score → 0.0
```

### Trending

**Score: binary 1 or 0**

Active only when `prefers_trending_movies` is `true`. Check `movie_id` membership in the in-memory trending set (fetched once from Redis `trending:current` per request).

```
in trending set → 1.0
not in set → 0.0
```

### Popular

**Score: [0, 1]**

Active only when `prefers_popular_movies` is `true`. Uses pre-computed `popularity_score` (0–1) from Postgres `movie_card`. Value is passed through directly.

```
score = movie.popularity_score
```

---

## Integration with Final Score

The metadata score feeds into the reranking formula:

```
final_score = w_L × lexical_score + w_V × vector_score + w_M × metadata_score + P
```

Because exclusion penalties produce negative scorer outputs, `metadata_score` can go below zero. This is intentional — a violated exclusion actively drags down the candidate's final score.

---

## Design Rationale

All metadata preferences extracted by the LLM are treated as **soft signals**, never hard filters. The app's UI provides separate hard filter controls that the user can set directly. Soft scoring avoids over-penalizing candidates when the LLM misinterprets the query, while still meaningfully boosting candidates that match inferred preferences.