Here's the implementation guide.

---

## Quality Prior in Reranking — Implementation Guide

### 1. Normalize reception score

You already have `reception_score` as a float on `movie_card`, fetched during Step 5a. Normalize it to [0, 1] at reranking time:

```python
RECEPTION_FLOOR = 3.0
RECEPTION_CEIL  = 9.0

def normalize_reception(raw: float | None) -> float:
    if raw is None:
        return 0.5  # unknown = neutral, don't penalize or reward
    clamped = max(RECEPTION_FLOOR, min(RECEPTION_CEIL, raw))
    return (clamped - RECEPTION_FLOOR) / (RECEPTION_CEIL - RECEPTION_FLOOR)
```

The 0.5 default for missing scores is important — you don't want films without ratings data systematically sinking to the bottom, since many newer or international films may not have populated scores yet.

### 2. Compute relevance score (unchanged from current formula)

```python
relevance = w_L * lexical_score + w_V * vector_score + w_M * metadata_score
```

This is your existing composite. Nothing changes here.

### 3. Bucket the relevance score

```python
BUCKET_PRECISION = 2  # round to 0.01

bucketed_relevance = round(relevance, BUCKET_PRECISION)
```

### 4. Determine quality prior weight

Check the QU output's `reception_preference` to decide whether the quality prior should be active:

```python
reception_pref = qu_output.metadata_preferences.reception_preference.reception_type

if reception_pref == "poorly_received":
    quality_prior = 0.0  # user wants bad movies — don't fight them
else:
    quality_prior = normalize_reception(candidate.reception_score)
```

When `reception_type` is `CRITICALLY_ACCLAIMED`, the reception score is already participating as a full metadata preference component in `w_M * metadata_score`. The quality prior stacking on top of that is fine — both signals agree and the bucketing means the prior only matters among candidates with similar relevance anyway.

### 5. Sort with lexicographic ordering

```python
def sort_key(candidate):
    return (
        candidate.bucketed_relevance,   # primary: relevance band
        candidate.quality_prior,         # secondary: quality tiebreaker
    )

candidates.sort(key=sort_key, reverse=True)
```

---

### Summary of constants

| Constant | Value | Notes |
|---|---|---|
| `RECEPTION_FLOOR` | 3.0 | Scores below this all map to 0 |
| `RECEPTION_CEIL` | 9.0 | Scores above this all map to 1 |
| `BUCKET_PRECISION` | 2 | 0.01-wide relevance bands |
| Missing score default | 0.5 | Neutral — no penalty, no reward |