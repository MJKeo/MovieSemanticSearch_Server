# Vector Score Calculation — Logical Plan

This document specifies the complete scoring pipeline that converts per-collection cosine similarity scores from Qdrant into a single `[0, 1]` final vector score for each candidate movie.

---

## Overview: Pipeline Stages

```
Stage 1: Determine search execution flags per vector space
Stage 2: Blend original + subquery scores per space per candidate
Stage 3: Normalize blended scores within each space (exponential decay from best)
Stage 4: Compute normalized weight array across all 8 spaces
Stage 5: Weighted sum → final vector score per candidate
```

---

## Stage 1: Determine Search Execution Flags

Before scoring, we need to know which searches actually ran for each vector space. This is determined by **system-level conditions**, not by individual candidate scores.

### Inputs

- `VectorWeights` — relevance per non-anchor space (`RelevanceSize` enum)
- `VectorSubqueries` — subquery text per non-anchor space (`Optional[str]`)

### Logic per space

```python
# Anchor is special — always original only
anchor.did_run_original = True
anchor.did_run_subquery = False

# For each non-anchor space:
for space in [plot_events, plot_analysis, viewer_experience,
              watch_context, narrative_techniques, production, reception]:

    # Promotion rule: not_relevant + subquery exists → treat as small
    effective_relevance = space.relevance
    if space.relevance == NOT_RELEVANT and space.subquery_text is not None:
        effective_relevance = SMALL

    space.effective_relevance = effective_relevance
    space.did_run_original = (effective_relevance != NOT_RELEVANT)
    space.did_run_subquery = (space.subquery_text is not None)
```

### Possible states per non-anchor space

| Effective Relevance | Subquery Text | did_run_original | did_run_subquery | Participates in scoring? |
|---|---|---|---|---|
| `NOT_RELEVANT` | `None` | ❌ | ❌ | **No** — weight is 0, space skipped entirely |
| `SMALL` (promoted) | exists | ❌ | ✅ | **Yes** — subquery-only, 100% subquery |
| `SMALL` / `MEDIUM` / `LARGE` | `None` | ✅ | ❌ | **Yes** — original-only, 100% original |
| `SMALL` / `MEDIUM` / `LARGE` | exists | ✅ | ✅ | **Yes** — 80/20 blend |

> **Important edge case — promoted `SMALL`:** When relevance is `NOT_RELEVANT` but subquery text exists, the effective relevance becomes `SMALL`. However, the original query was NOT searched for this space (because the pre-promotion relevance was `NOT_RELEVANT`, which means Step 3b of the architecture never ran the original query against this collection). So `did_run_original = False`. The original search execution decision was already made before scoring — we can't retroactively search. We only promote the weight.

> **Wait — clarification needed on the promoted SMALL case.** The architecture doc says: *"Also search every channel with relevance > not_relevant using the original query embedding."* The promotion from `NOT_RELEVANT` → `SMALL` happens during scoring, but the search execution already happened using the *original* relevance. So if relevance was `NOT_RELEVANT` at search time, the original query was never searched in that space, even though we're now promoting the weight. This means:
>
> - `did_run_original` should be based on the **pre-promotion relevance** (i.e., the relevance that was active when Qdrant searches were dispatched)
> - `effective_relevance` (for weighting purposes) uses the **post-promotion value**
>
> Updated logic:

```python
for space in non_anchor_spaces:
    # Search execution flags — based on ORIGINAL relevance (what was used at search time)
    space.did_run_original = (space.relevance != NOT_RELEVANT)
    space.did_run_subquery = (space.subquery_text is not None)

    # Weight calculation — uses PROMOTED relevance
    if space.relevance == NOT_RELEVANT and space.subquery_text is not None:
        space.effective_relevance = SMALL
    else:
        space.effective_relevance = space.relevance
```

### Corrected state table

| Original Relevance | Subquery Text | did_run_original | did_run_subquery | Effective Relevance | Blend Mode |
|---|---|---|---|---|---|
| `NOT_RELEVANT` | `None` | ❌ | ❌ | `NOT_RELEVANT` | N/A — excluded |
| `NOT_RELEVANT` | exists | ❌ | ✅ | `SMALL` | 100% subquery |
| `SMALL`+ | `None` | ✅ | ❌ | same | 100% original |
| `SMALL`+ | exists | ✅ | ✅ | same | 80% sub / 20% orig |

### Example

Query: *"cozy 90s movie for a rainy night, great soundtrack"*

| Space | Relevance | Subquery Text | did_run_original | did_run_subquery | Effective Relevance | Blend |
|---|---|---|---|---|---|---|
| anchor | (always) | (never) | ✅ | ❌ | (special) | 100% original |
| plot_events | `NOT_RELEVANT` | `None` | ❌ | ❌ | `NOT_RELEVANT` | excluded |
| plot_analysis | `SMALL` | `"cozy comforting atmosphere nostalgic warmth"` | ✅ | ✅ | `SMALL` | 80/20 |
| viewer_experience | `LARGE` | `"cozy, warm, comforting, relaxing, not intense"` | ✅ | ✅ | `LARGE` | 80/20 |
| watch_context | `LARGE` | `"rainy day movie, relaxation, great soundtrack"` | ✅ | ✅ | `LARGE` | 80/20 |
| narrative_techniques | `NOT_RELEVANT` | `None` | ❌ | ❌ | `NOT_RELEVANT` | excluded |
| production | `MEDIUM` | `"1990s, 90s"` | ✅ | ✅ | `MEDIUM` | 80/20 |
| reception | `NOT_RELEVANT` | `"praised for soundtrack, beloved"` | ❌ | ✅ | `SMALL` (promoted) | 100% subquery |

Active spaces: anchor, plot_analysis, viewer_experience, watch_context, production, reception (6 of 8).

---

## Stage 2: Blend Original + Subquery Scores Per Space Per Candidate

### Purpose

Each candidate may have up to two raw cosine similarity scores per vector space (one from searching with the original query embedding, one from the subquery embedding). This stage produces a single **blended score** per space per candidate.

### Inputs

- `CandidateVectorScores` for each candidate (raw cosine similarities from Qdrant, 0.0 if not in top-N)
- Search execution flags from Stage 1

### Logic

```python
SUBQUERY_WEIGHT = 0.8
ORIGINAL_WEIGHT = 0.2

def blend_score(space, candidate_scores) -> float:
    original = getattr(candidate_scores, f"{space}_score_original")
    subquery = getattr(candidate_scores, f"{space}_score_subquery")

    if space.did_run_original and space.did_run_subquery:
        return SUBQUERY_WEIGHT * subquery + ORIGINAL_WEIGHT * original
    elif space.did_run_original:
        return original
    elif space.did_run_subquery:
        return subquery
    else:
        return 0.0  # space doesn't participate — this shouldn't be called
```

### Important: what "0.0" means after blending

After blending, a score of 0.0 means one of two things:
1. **The candidate wasn't returned in any executed search for that space.** Both original and subquery (whichever ran) returned it outside the top-N. This is a legitimate "not relevant" signal.
2. **The space didn't execute any search.** But this case is handled by Stage 1 — non-participating spaces are excluded before blending is called.

A blended score of 0.0 for a participating space is a **real score** — it means the candidate was not found relevant by Qdrant in that space. It will receive a normalized score of 0.0 in Stage 3.

### Example

Candidate movie "You've Got Mail" for the example query above:

| Space | original score | subquery score | Blend mode | Blended score |
|---|---|---|---|---|
| anchor | 0.72 | — | 100% original | **0.720** |
| plot_analysis | 0.41 | 0.68 | 80/20 | 0.8(0.68) + 0.2(0.41) = **0.626** |
| viewer_experience | 0.65 | 0.81 | 80/20 | 0.8(0.81) + 0.2(0.65) = **0.778** |
| watch_context | 0.58 | 0.79 | 80/20 | 0.8(0.79) + 0.2(0.58) = **0.748** |
| production | 0.55 | 0.71 | 80/20 | 0.8(0.71) + 0.2(0.55) = **0.678** |
| reception | — | 0.44 | 100% subquery | **0.440** |

Candidate movie "The Shawshank Redemption" (not found in some spaces):

| Space | original score | subquery score | Blend mode | Blended score |
|---|---|---|---|---|
| anchor | 0.38 | — | 100% original | **0.380** |
| plot_analysis | 0.0 | 0.22 | 80/20 | 0.8(0.22) + 0.2(0.0) = **0.176** |
| viewer_experience | 0.31 | 0.0 | 80/20 | 0.8(0.0) + 0.2(0.31) = **0.062** |
| watch_context | 0.0 | 0.0 | 80/20 | **0.000** |
| production | 0.47 | 0.60 | 80/20 | 0.8(0.60) + 0.2(0.47) = **0.574** |
| reception | — | 0.0 | 100% subquery | **0.000** |

Note how Shawshank's viewer_experience score is heavily penalized: it appeared in the original search with a mediocre 0.31, didn't appear in the (more targeted) subquery search at all, and the 80/20 blend reflects that it's not a strong match for the "cozy, warm, comforting" subquery.

---

## Stage 3: Normalize Blended Scores Within Each Space

### Purpose

Transform blended cosine similarity scores into `[0, 1]` normalized scores that reflect **relative quality within that space's candidate pool**, using exponential decay from the best score.

### Which candidates participate in normalization

For a given space, the **normalization pool** consists of only the candidates whose blended score is **> 0.0** for that space. Candidates with blended score = 0.0 (they didn't appear in any of that space's searches) are assigned a normalized score of 0.0 and excluded from the statistical calculations.

### Formula: Exponential Decay from Best

For each active space, given the set of blended scores `S = {s₁, s₂, ..., sₙ}` where each `sᵢ > 0`:

```
s_max = max(S)
s_min = min(S)
range = s_max - s_min

if range == 0:
    # All candidates in this space have the same blended score
    normalized_score(sᵢ) = 1.0 for all i

else:
    # Linear gap: how far this candidate is from the best, as a fraction of the range
    gap(sᵢ) = (s_max - sᵢ) / range    # gap ∈ [0, 1], where 0 = best

    # Exponential decay: steep dropoff for candidates far from best
    normalized_score(sᵢ) = exp(-k * gap(sᵢ))
```

Where `k` is the **decay steepness parameter**.

### Behavior of the decay parameter `k`

| k value | Behavior | When candidates are clustered near top | When there's a clear gap |
|---|---|---|---|
| 1.0 | Very gentle — most candidates score 0.37+ | Barely differentiates | Mild separation |
| 2.0 | Moderate — bottom of range scores ~0.14 | Moderate clustering near 1.0 | Decent separation |
| **3.0** | **Recommended start** — bottom scores ~0.05 | Top cluster stays high (~0.7+) | Clear separation |
| 5.0 | Aggressive — bottom scores ~0.007 | Very tight top cluster | Extreme winner-take-all |

### Why `k = 3.0` is a good starting point

With `k = 3.0`:
- Best candidate: `exp(-3 * 0) = 1.0`
- Candidate at 90% of best's score (gap=0.1): `exp(-3 * 0.1) = 0.74`
- Candidate at 75% of best's score (gap=0.25): `exp(-3 * 0.25) = 0.47`
- Candidate at 50% of best's score (gap=0.5): `exp(-3 * 0.5) = 0.22`
- Worst candidate in pool (gap=1.0): `exp(-3 * 1.0) = 0.05`

This satisfies the requirement: if all candidates are similarly close to the best (small range, small gaps), they all get high normalized scores. If there's a clear winner group, the dropoff is steep.

### Numerical walkthrough

**Tight cluster example** (viewer_experience space, 5 candidates):

| Candidate | Blended score | gap | normalized |
|---|---|---|---|
| A | 0.81 | 0.000 | 1.000 |
| B | 0.79 | 0.118 | 0.701 |
| C | 0.78 | 0.176 | 0.589 |
| D | 0.77 | 0.235 | 0.494 |
| E | 0.64 | 1.000 | 0.050 |

range = 0.81 - 0.64 = 0.17. A–D are clustered (scores 0.49–1.0); E is clearly worse (0.05). This is the desired behavior.

**Spread example** (anchor space, 5 candidates):

| Candidate | Blended score | gap | normalized |
|---|---|---|---|
| A | 0.72 | 0.000 | 1.000 |
| B | 0.55 | 0.500 | 0.223 |
| C | 0.45 | 0.794 | 0.092 |
| D | 0.38 | 1.000 | 0.050 |

range = 0.72 - 0.38 = 0.34. Clear winner, steep falloff. Also desired.

**All-same example** (production space, 3 candidates with identical blended score of 0.71):

range = 0. All get normalized score = 1.0.

### Edge cases

| Scenario | Handling |
|---|---|
| Only 1 candidate in pool | `range = 0` → normalized = 1.0 |
| All candidates have identical blended scores | `range = 0` → all get 1.0 |
| Very small pool (2–3 candidates) | Normalization proceeds normally; accept noise |
| Candidate's blended score = 0.0 | Not in pool; gets normalized score = 0.0 directly |

### Pseudocode

```python
DECAY_K = 3.0

def normalize_space_scores(blended_scores: dict[int, float]) -> dict[int, float]:
    """
    Input: movie_id → blended cosine similarity for one vector space.
    Output: movie_id → normalized score ∈ [0, 1].
    """
    # Separate participants from non-participants
    pool = {mid: score for mid, score in blended_scores.items() if score > 0.0}
    result = {mid: 0.0 for mid in blended_scores}  # default all to 0

    if not pool:
        return result

    s_max = max(pool.values())
    s_min = min(pool.values())
    score_range = s_max - s_min

    for mid, score in pool.items():
        if score_range == 0:
            result[mid] = 1.0
        else:
            gap = (s_max - score) / score_range
            result[mid] = math.exp(-DECAY_K * gap)

    return result
```

### Why this is better than z-score for this use case

Traditional z-score (`(x - mean) / std`) has two problems here:
1. **It's distribution-dependent.** If most candidates cluster near the mean with a few outliers, z-scores spread the bulk into a narrow band and give extreme values to outliers. You'd need clamping and rescaling, which defeats the purpose.
2. **It requires computing mean and std across potentially thousands of candidates**, then clamping and rescaling to [0,1]. The exponential decay approach only needs `max` and `min` — two O(n) passes vs. a full statistical computation.

The exponential decay directly encodes what you asked for: "score based on how close to the BEST, where the best gets 1.0."

---

## Stage 4: Compute Normalized Weight Array

### Purpose

Convert the `RelevanceSize` enums (and anchor's special case) into a float array of length 8 that sums to 1.0.

### Step 4a: Assign raw numeric weights

```python
RELEVANCE_TO_RAW = {
    NOT_RELEVANT: 0.0,   # after promotion check — if still not_relevant, weight is 0
    SMALL:        1.0,
    MEDIUM:       2.0,
    LARGE:        3.0,
}
```

Map each non-anchor space's `effective_relevance` to its raw weight.

### Step 4b: Compute anchor's raw weight

Anchor should be "slightly below average" of the active (non-zero) weights.

```python
active_weights = [w for w in non_anchor_raw_weights if w > 0.0]

if len(active_weights) == 0:
    # Only anchor is active (all others are not_relevant with no subqueries)
    anchor_raw = 1.0
else:
    anchor_raw = mean(active_weights) * 0.8  # 80% of the average active weight
```

### Why 80% of the mean?

The anchor vector is a broad "movie card" embedding — it provides useful general recall but should never dominate over purpose-built spaces that directly match the query intent. Setting it at 80% of the mean ensures:

- If all active spaces are `LARGE` (raw=3.0): anchor gets `3.0 * 0.8 = 2.4`
- If active spaces are mixed `SMALL`/`MEDIUM`/`LARGE` (mean≈2.0): anchor gets `2.0 * 0.8 = 1.6`
- If only one space is `SMALL` (raw=1.0): anchor gets `1.0 * 0.8 = 0.8`
- Anchor is always present but never the loudest voice

### Step 4c: Normalize to sum to 1.0

```python
all_raw = [anchor_raw, plot_events_raw, plot_analysis_raw, ..., reception_raw]
total = sum(all_raw)
normalized_weights = [w / total for w in all_raw]
```

### Full example

Using the "cozy 90s movie" query from Stage 1:

| Space | Effective Relevance | Raw Weight |
|---|---|---|
| plot_events | `NOT_RELEVANT` | 0.0 |
| plot_analysis | `SMALL` | 1.0 |
| viewer_experience | `LARGE` | 3.0 |
| watch_context | `LARGE` | 3.0 |
| narrative_techniques | `NOT_RELEVANT` | 0.0 |
| production | `MEDIUM` | 2.0 |
| reception | `SMALL` (promoted) | 1.0 |

Active non-anchor weights: `[1.0, 3.0, 3.0, 2.0, 1.0]` → mean = `2.0`
Anchor raw: `2.0 * 0.8 = 1.6`

All raw weights: `[1.6, 0.0, 1.0, 3.0, 3.0, 0.0, 2.0, 1.0]` → sum = `11.6`

| Space | Raw | Normalized Weight |
|---|---|---|
| anchor | 1.6 | **0.138** |
| plot_events | 0.0 | **0.000** |
| plot_analysis | 1.0 | **0.086** |
| viewer_experience | 3.0 | **0.259** |
| watch_context | 3.0 | **0.259** |
| narrative_techniques | 0.0 | **0.000** |
| production | 2.0 | **0.172** |
| reception | 1.0 | **0.086** |
| **Total** | | **1.000** |

### Edge case: all non-anchor spaces are NOT_RELEVANT with no subqueries

This means only anchor is active. `active_weights` is empty, so `anchor_raw = 1.0`.

Weight array: `[1.0, 0, 0, 0, 0, 0, 0, 0]` → normalized: `[1.0, 0, 0, 0, 0, 0, 0, 0]`

The final vector score is purely the anchor's normalized score. This is correct — the query understanding system decided nothing was relevant, so we fall back to broad recall.

### Edge case: single non-anchor space active

Say only `viewer_experience = LARGE` (raw=3.0). Active mean = 3.0. Anchor = `3.0 * 0.8 = 2.4`.

Weights: `[2.4, 0, 0, 3.0, 0, 0, 0, 0]` → normalized: `[0.444, 0, 0, 0.556, 0, 0, 0, 0]`

Anchor gets 44.4%, viewer_experience gets 55.6%. Anchor still provides meaningful recall diversity even when one space dominates.

---

## Stage 5: Final Weighted Sum

### Purpose

Combine normalized per-space scores with normalized weights to produce a single `[0, 1]` final vector score per candidate.

### Formula

```
final_vector_score(movie) = Σ over all 8 spaces: weight[space] * normalized_score[space][movie]
```

### Why this is already in [0, 1]

- Each `normalized_score` is in `[0, 1]` (Stage 3 guarantees this)
- Each `weight` is in `[0, 1]` and they sum to 1.0 (Stage 4 guarantees this)
- A weighted sum of `[0, 1]` values with weights summing to 1.0 produces a value in `[0, 1]`

A score of 1.0 would mean: the candidate was the top result in every active space. Realistic top scores will be well below 1.0.

### Full numerical example

Using "You've Got Mail" from the earlier stages:

**Stage 2 blended scores → Stage 3 normalized scores** (assume the following after normalization against all candidates in each space):

| Space | Blended | Normalized (after exp decay) | Weight |
|---|---|---|---|
| anchor | 0.720 | 0.85 | 0.138 |
| plot_analysis | 0.626 | 0.47 | 0.086 |
| viewer_experience | 0.778 | 0.92 | 0.259 |
| watch_context | 0.748 | 0.88 | 0.259 |
| production | 0.678 | 0.71 | 0.172 |
| reception | 0.440 | 0.35 | 0.086 |

```
final = (0.138 × 0.85) + (0.086 × 0.47) + (0.259 × 0.92) +
        (0.259 × 0.88) + (0.172 × 0.71) + (0.086 × 0.35)
      = 0.117 + 0.040 + 0.238 + 0.228 + 0.122 + 0.030
      = 0.776
```

**"The Shawshank Redemption"** (weaker match for this query):

| Space | Blended | Normalized | Weight |
|---|---|---|---|
| anchor | 0.380 | 0.05 | 0.138 |
| plot_analysis | 0.176 | 0.08 | 0.086 |
| viewer_experience | 0.062 | 0.05 | 0.259 |
| watch_context | 0.000 | 0.00 | 0.259 |
| production | 0.574 | 0.52 | 0.172 |
| reception | 0.000 | 0.00 | 0.086 |

```
final = (0.138 × 0.05) + (0.086 × 0.08) + (0.259 × 0.05) +
        (0.259 × 0.00) + (0.172 × 0.52) + (0.086 × 0.00)
      = 0.007 + 0.007 + 0.013 + 0.000 + 0.089 + 0.000
      = 0.116
```

Shawshank scores 0.116 vs You've Got Mail's 0.776 — appropriately large gap for a "cozy 90s rainy night" query. Shawshank's only meaningful contribution comes from the production space (it is indeed a 90s film), but it fails across the experience and context dimensions.

---

## Appendix A: Candidate Coverage and Penalization

A candidate that appears in few vector searches is **not** re-normalized. If a candidate only appears in the anchor search and has a 0.0 for all other spaces, it receives:

```
final = weight_anchor * normalized_anchor + 0 + 0 + 0 + ...
```

This is by design — it means the candidate was only relevant to broad recall, not to any specific query dimension. A candidate that shows up in multiple relevant spaces earns a compounding advantage.

### Why this is correct

Consider two candidates for "cozy 90s movie for a rainy night":
- **Candidate A**: appears in anchor + viewer_experience + watch_context. It's genuinely cozy and fits the scenario.
- **Candidate B**: appears only in anchor + production (it's from the 90s). It's a 90s movie but not cozy.

Without re-normalization, Candidate B gets zeroes for viewer_experience and watch_context, which together carry ~52% of the weight. It would need a perfect anchor and production score to compete, which is exactly the intended behavior.

---

## Appendix B: Data Structures Summary

### Input to the scoring pipeline

```python
@dataclass
class VectorScoringInput:
    candidates: dict[int, CandidateVectorScores]  # from Qdrant
    vector_weights: VectorWeights                  # from query understanding
    vector_subqueries: VectorSubqueries            # from query understanding
```

### Internal intermediate structures

```python
@dataclass
class SpaceExecutionContext:
    """Computed once per space per request."""
    space_name: str
    did_run_original: bool
    did_run_subquery: bool
    effective_relevance: RelevanceSize
    normalized_weight: float           # from Stage 4

@dataclass
class SpaceBlendedScores:
    """Intermediate result from Stage 2, per space."""
    space_name: str
    blended: dict[int, float]          # movie_id → blended cosine similarity

@dataclass
class SpaceNormalizedScores:
    """Intermediate result from Stage 3, per space."""
    space_name: str
    normalized: dict[int, float]       # movie_id → normalized score ∈ [0, 1]
```

### Output

```python
@dataclass
class VectorScoringResult:
    final_scores: dict[int, float]                    # movie_id → final vector score ∈ [0, 1]
    space_contexts: list[SpaceExecutionContext]        # for debug logging
    per_space_normalized: dict[str, dict[int, float]] # space_name → {movie_id: score} for debug
```

---

## Appendix C: Tunable Parameters

| Parameter | Default | What it controls | Tuning guidance |
|---|---|---|---|
| `SUBQUERY_WEIGHT` | 0.8 | How much the subquery search dominates the blend | Increase if subqueries are consistently higher quality than original |
| `ORIGINAL_WEIGHT` | 0.2 | How much the original query search contributes | = 1 - SUBQUERY_WEIGHT |
| `DECAY_K` | 3.0 | Steepness of exponential decay from best candidate | Increase for more winner-take-all; decrease if too many candidates get near-zero |
| `ANCHOR_MEAN_FRACTION` | 0.8 | Anchor's weight as a fraction of active-space mean | Decrease if anchor is drowning out specialized spaces; increase if recall is suffering |
| `RELEVANCE_SMALL` | 1.0 | Raw weight for SMALL relevance | Adjust ratio between SMALL/MEDIUM/LARGE |
| `RELEVANCE_MEDIUM` | 2.0 | Raw weight for MEDIUM relevance | — |
| `RELEVANCE_LARGE` | 3.0 | Raw weight for LARGE relevance | — |

---

## Appendix D: Walkthrough of Every Possible Candidate Scenario

To ensure completeness, here is every meaningful combination a single candidate can encounter:

### Scenario 1: Candidate appears in all active searches (best case)

Movie shows up in top-N for every search that ran. Gets blended scores > 0 everywhere. All spaces contribute to final score. This is the highest-scoring scenario.

### Scenario 2: Candidate appears only in anchor

Likely a broadly relevant movie that doesn't match any specific query dimension. Final score = `weight_anchor * normalized_anchor`. Will be a low score unless anchor is the only active space.

### Scenario 3: Candidate appears in subquery search but not original search (for a blended space)

Blended = `0.8 * subquery + 0.2 * 0.0 = 0.8 * subquery`. The candidate is penalized 20% vs a candidate that appeared in both, which is correct — the original query didn't find it relevant.

### Scenario 4: Candidate appears in original search but not subquery search (for a blended space)

Blended = `0.8 * 0.0 + 0.2 * original = 0.2 * original`. Heavy penalty — the more targeted subquery didn't find this candidate relevant, which is a strong negative signal.

### Scenario 5: Candidate appears in a space where only subquery ran (promoted SMALL or subquery-only)

Blended = `1.0 * subquery`. No penalty since there was no original search to miss.

### Scenario 6: Candidate doesn't appear in any search for an active space

Blended = 0.0. Normalized = 0.0. The full weight of that space contributes nothing. This is the intended penalty for not being found relevant.

### Scenario 7: Candidate appears in a non-participating space

Impossible by construction — if the space isn't participating (effective_relevance = NOT_RELEVANT and no subquery), no search was run, so no candidates exist for it.