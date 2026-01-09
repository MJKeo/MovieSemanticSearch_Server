# Movie Vectorization Evaluation Guide (Movie→Movie, K=5)

This guide defines **(1) the source-of-truth schema** (your similarity JSON) and **(2) a step-by-step, formula-defined evaluation process** for measuring embedding quality for:
- **DenseAnchor** (overall similarity)
- **DenseContent** (content/plot similarity)
- **DenseVibe** (vibe/watch-context similarity)
- **Combined retrieval** (your existing **weighted RRF**), evaluated against **overall similarity**

Target audience: a **coding LLM** implementing the evaluation pipeline.

---

## 1) Source-of-truth (Ground Truth) Schema

### 1.1 JSON file shape

The ground truth is a JSON array of objects:

```json
[
  {
    "movie_id": "tt0123456",
    "5_most_similar_movies_overall_ordered": ["tt...", "tt...", "tt...", "tt...", "tt..."],
    "5_most_similar_movies_content_ordered": ["tt...", "tt...", "tt...", "tt...", "tt..."],
    "5_most_similar_movies_vibes_ordered": ["tt...", "tt...", "tt...", "tt...", "tt..."]
  }
]
```

### 1.2 Field definitions

- `movie_id` *(string)*  
  The IMDb title ID for the **query** movie.

- `5_most_similar_movies_overall_ordered` *(list[string], length=5)*  
  The **ranked** list of the 5 most similar movies (overall recommendation similarity).  
  **Used as ground truth for DenseAnchor** and **for combined evaluation**.

- `5_most_similar_movies_content_ordered` *(list[string], length=5)*  
  The **ranked** list of the 5 closest matches by **content/plot/themes/story beats**.  
  **Used as ground truth for DenseContent**.

- `5_most_similar_movies_vibes_ordered` *(list[string], length=5)*  
  The **ranked** list of the 5 closest matches by **vibe/watch context** (how it feels to watch).  
  **Used as ground truth for DenseVibe**.

### 1.3 Hard constraints (must validate; fail fast)

Let `Q` = set of all `movie_id` entries in the JSON array.

For each object with query `q`:
- Each list must have **exactly 5** items.
- `q` must **not appear** in any of the 3 lists.
- Each list must contain **unique IDs** (no duplicates).
- Every listed ID must exist in `Q` (since recommendations must come only from the same set).
- The JSON array must contain exactly one entry per `movie_id` (no duplicates).

If any constraint fails: raise an error and stop.

---

## 2) Evaluation Setup

### 2.1 Dataset

- Total movies: **50**
- Queries: **all 50 movies**
- For each query movie `q`, candidates are **all other movies**:
  - `C(q) = Q \ {q}` (49 candidates)

### 2.2 Embedding spaces (inputs)

You have 3 embedding vectors per movie:

- DenseAnchor embeddings: `E_A[movie_id] -> vector`
- DenseContent embeddings: `E_C[movie_id] -> vector`
- DenseVibe embeddings: `E_V[movie_id] -> vector`

### 2.3 Retrieval similarity

Use **cosine similarity**.

For vectors `u`, `v`:

\[
\cos(u, v) = \frac{u \cdot v}{\|u\|\,\|v\|}
\]

Implementation recommendation:
1. Pre-normalize each vector to unit length once:

\[
u_{norm} = \frac{u}{\|u\| + \epsilon}
\]

2. Then cosine is just dot product:

\[
\cos(u, v) = u_{norm} \cdot v_{norm}
\]

### 2.4 Cutoff

Start with **K = 5**.

All evaluation metrics below are computed at `K=5` unless otherwise parameterized.

---

## 3) Build Ranked Lists (per vector space)

For each space `S ∈ {A, C, V}` and each query `q`:

1. Compute scores for all candidates `d ∈ C(q)`:

\[
score_S(q, d) = \cos(E_S[q], E_S[d])
\]

2. Sort candidates by:
- primary: `score_S(q, d)` descending
- tie-breaker: `d` ascending (string compare) **for determinism**

3. Produce:
- `ranked_S[q] = [d1, d2, ..., d49]` (full ranking)
- `topK_S[q] = ranked_S[q][:K]`

---

## 4) Relevance Definition (Graded, derived from GT order)

Because your GT lists are ordered and you want the **top ~2** to matter more, use **graded relevance**.

Let `GT[q] = [g1, g2, g3, g4, g5]`.

Define `pos(q, x)`:
- `pos(q, x) = i` if `x == gi` for some `i ∈ {1..5}`
- otherwise `pos(q, x) = None`

Define graded relevance:

\[
rel(q, x) =
\begin{cases}
6 - pos(q, x) & \text{if } x \in GT[q] \\
0 & \text{otherwise}
\end{cases}
\]

So:
- `g1` has `rel=5`
- `g2` has `rel=4`
- …
- `g5` has `rel=1`
- non-GT items have `rel=0`

---

## 5) Metrics (Per Query and Aggregated)

Let `pred = [p1, p2, ..., pK]` be the predicted top-K list for query `q`.

### 5.1 Coverage: Hits@K (aka Precision@K and Recall@K here)

Because `K=5` and `|GT|=5`, **Precision@5 = Recall@5 = Hits@5/5** in this setup.

\[
Hits@K(q) = \frac{|\{p_i\}_{i=1..K} \cap GT[q]|}{|GT[q]|}
\]

Since `|GT[q]| = 5`, this is simply:

\[
Hits@5(q) = \frac{\#hits}{5}
\]

### 5.2 MRR@K (first relevant early)

Let `r` be the smallest index such that `p_r ∈ GT[q]`, with `r ∈ {1..K}`.

\[
RR@K(q) =
\begin{cases}
\frac{1}{r} & \text{if } \exists r \le K \text{ with } p_r \in GT[q] \\
0 & \text{otherwise}
\end{cases}
\]

Aggregate:

\[
MRR@K = \frac{1}{|Q|} \sum_{q \in Q} RR@K(q)
\]

### 5.3 nDCG@K (top-weighted relevance + ordering)

Use exponential gain:

\[
DCG@K(q) = \sum_{i=1}^{K} \frac{2^{rel(q,p_i)} - 1}{\log_2(i+1)}
\]

Compute ideal DCG using the **ground-truth order**:

- `ideal = GT[q]` (already best→worst)
- `IDCG@K(q) = DCG@K(q)` computed over `ideal[:K]`

Normalize:

\[
nDCG@K(q) =
\begin{cases}
\frac{DCG@K(q)}{IDCG@K(q)} & \text{if } IDCG@K(q) > 0 \\
0 & \text{otherwise}
\end{cases}
\]

Aggregate:

\[
mean\_nDCG@K = \frac{1}{|Q|} \sum_{q \in Q} nDCG@K(q)
\]

---

## 6) Per-Vector Evaluation Passes

Map each embedding space to its GT list:

- DenseAnchor (`S=A`) uses: `5_most_similar_movies_overall_ordered`
- DenseContent (`S=C`) uses: `5_most_similar_movies_content_ordered`
- DenseVibe (`S=V`) uses: `5_most_similar_movies_vibes_ordered`

For each space `S`:
1. For each query `q`:
   - compute `topK_S[q]`
   - compute:
     - `hits_S[q] = Hits@K(q)`
     - `rr_S[q]   = RR@K(q)`
     - `ndcg_S[q] = nDCG@K(q)`
2. Aggregate means across queries.

---

## 7) Combined Evaluation (Weighted RRF — keep your implementation)

You already have **weighted RRF**. The evaluation must reproduce that exact combined ranking.

### 7.1 Weighted RRF formula (generic reference)

Let `rank_S(q, d)` be the 1-indexed position of candidate `d` in `ranked_S[q]`.

Given weights `w_A`, `w_C`, `w_V` and constant `c`:

\[
RRF(q,d) = \sum_{S \in \{A,C,V\}} \frac{w_S}{c + rank_S(q,d)}
\]

Important implementation detail:
- If your production implementation uses **top-N truncation** (e.g., only consider top 50 from each list), then for candidates absent from a list, treat their contribution as `0`.

### 7.2 Build combined top-K list

For each query `q`:
1. Compute `RRF(q,d)` for all `d ∈ C(q)` using your existing method.
2. Sort by:
   - `RRF(q,d)` descending
   - tie-breaker: `d` ascending
3. Take `combined_topK[q] = first K items`.

### 7.3 Combined evaluation target

Evaluate combined ranking **only** against the overall GT list:

- `GT = 5_most_similar_movies_overall_ordered`

Compute:
- `Hits@K`
- `MRR@K`
- `nDCG@K`

Aggregate means across all queries.

---

## 8) Implementation Plan (Step-by-Step)

### Step 1 — Load GT + validate
- Parse JSON array
- Build:
  - `Q` set of all movie IDs
  - `GT_overall[q]`, `GT_content[q]`, `GT_vibes[q]`
- Validate constraints (Section 1.3).

### Step 2 — Load embeddings
- Load `E_A`, `E_C`, `E_V` for all `q ∈ Q`.
- Verify all movie IDs in `Q` exist in each embedding dict.

### Step 3 — Normalize embeddings
- Pre-normalize vectors to unit length for cosine-as-dot:
  - `E_S_norm[id] = E_S[id] / (||E_S[id]|| + eps)`

### Step 4 — Compute rankings for each space
For each `S ∈ {A, C, V}` and each query `q`:
- compute dot product scores for all candidates `d ≠ q`
- sort descending by score, tiebreak by `movie_id`
- store:
  - `ranked_S[q]` full list (49 items)
  - `topK_S[q]` first K items

### Step 5 — Implement metric functions
Implement the following pure functions:

- `hits_at_k(pred: list[str], gt: list[str], k: int) -> float`
  - return `len(set(pred[:k]) ∩ set(gt)) / len(gt)`

- `rr_at_k(pred: list[str], gt_set: set[str], k: int) -> float`
  - find smallest `i` where `pred[i] in gt_set`
  - return `1/(i+1)` else `0`

- `ndcg_at_k(pred: list[str], gt: list[str], k: int) -> float`
  - build `rel_map` from gt positions:
    - `rel_map[gt[i]] = 5 - i`
  - compute DCG with exponential gain:
    - `sum( (2**rel - 1)/log2(i+2) )`
  - compute IDCG using `gt[:k]`
  - return `dcg/idcg` else `0`

### Step 6 — Score per vector space
For each `S`:
- choose GT list mapping
- for each query `q`, compute metrics on `topK_S[q]`
- average over queries

### Step 7 — Combined ranking via weighted RRF
- For each query `q`, compute combined ranking using your existing weighted RRF
- take top K
- evaluate against overall GT

### Step 8 — Produce a report artifact
Emit a JSON report with:
- run metadata: `K`, `weights`, `c`, timestamp/model tag
- aggregated metrics per retriever (A/C/V/combined)
- per-query metrics + predicted topK lists (for debugging)

Recommended report schema:

```json
{
  "K": 5,
  "denseanchor": {"mean_hits": 0.0, "mean_mrr": 0.0, "mean_ndcg": 0.0},
  "densecontent": {"mean_hits": 0.0, "mean_mrr": 0.0, "mean_ndcg": 0.0},
  "densevibe": {"mean_hits": 0.0, "mean_mrr": 0.0, "mean_ndcg": 0.0},
  "combined_weighted_rrf": {"mean_hits": 0.0, "mean_mrr": 0.0, "mean_ndcg": 0.0},
  "per_query": {
    "tt....": {
      "denseanchor": {"hits": 0.0, "rr": 0.0, "ndcg": 0.0, "topK": ["..."]},
      "densecontent": {"hits": 0.0, "rr": 0.0, "ndcg": 0.0, "topK": ["..."]},
      "densevibe": {"hits": 0.0, "rr": 0.0, "ndcg": 0.0, "topK": ["..."]},
      "combined": {"hits": 0.0, "rr": 0.0, "ndcg": 0.0, "topK": ["..."]}
    }
  }
}
```

---

## 9) Notes specific to your constraints

- With K=5 and GT size=5, `Hits@5` is the main “coverage %” metric.
- `nDCG@5` is the primary quality metric when you care that the top 1–2 are much better than the rest.
- `MRR@5` is a useful early-signal metric for whether your #1/#2 are landing.

---

## 10) Optional extensions (keep out of MVP)

- Evaluate multiple K values: K ∈ {1, 2, 3, 5, 10} for sensitivity.
- Add a “random baseline” for sanity checks.
- Add per-cluster breakdowns (animation/horror/crime) to find weak genres.
