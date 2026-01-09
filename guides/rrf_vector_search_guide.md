# Vector-only Retrieval Baseline: 3× Chroma Collections + Weighted RRF Fusion (Conceptual Guide)

This document is a **conceptual**, step-by-step implementation guide for a **vector-search-only** baseline that:
- embeds the user query **once**
- searches **three Chroma collections**
- fuses the three ranked lists with **Weighted Reciprocal Rank Fusion (RRF)**
- breaks ties with **average cosine similarity** across the three collections
- exposes all key knobs in a **Gradio** UI for rapid testing

Target audience: **another coding LLM**.

---

## 0) System Context (Your Setup)

### Chroma collections
You have three separate Chroma collections:

- `dense_anchor_vectors`
- `dense_content_vectors`
- `dense_vibe_vectors`

Each collection contains **one embedding per movie** for a different textual representation (anchor/content/vibe).

### Stored metadata (per vector)
Each vector entry stores at least:

```python
metadata = {
    "movie_id": movie.id,
    "tmdb_id": movie.tmdb_id,
    "title": movie.title,
    "release_date": movie.release_date,
    "genres": ", ".join(movie.genres) if movie.genres else "",
}
```

**Important identity assumption:** `movie_id` is consistent across all three collections and is the key used for fusion.

---

## 1) Goal

Given a user query, return the **best overall match** (and top-N list) using **only**:
- vector similarity retrieval
- rank fusion (RRF)
- cosine tie-breaking

No metadata filtering, no lexical search, no learned reranker.

---

## 2) Tunable Variables (Expose in Gradio)

Design the Gradio interface so you can adjust these live.

### A) Retrieval depth per collection
How deep to retrieve from each list before fusion:
- `n_anchor`  (top-K from `dense_anchor_vectors`)
- `n_content` (top-K from `dense_content_vectors`)
- `n_vibe`    (top-K from `dense_vibe_vectors`)

Optional convenience:
- `n_all` to set all three at once.

### B) RRF parameters
RRF uses ranks only (not cosine values) for the fused score.

- `rrf_k` (a.k.a. “c” in some descriptions): **rank dampening constant**
  - Smaller → more top-heavy (rank #1 matters a lot)
  - Larger  → more consensus-y (rank #1 vs #10 matters less)

- Weights per list:
  - `w_anchor`, `w_content`, `w_vibe` (non-negative floats)

### C) Candidate pool / safety knobs (optional but recommended)
- `max_candidates` (cap the union set size, if needed)
- `return_top_n` (how many fused results to display)

### D) Debug toggles (highly recommended)
- `show_raw_lists` (show the three raw top-K lists)
- `show_score_breakdown` (show ranks, per-list RRF terms, cosines)
- `show_metadata` (title/date/genres/tmdb_id)

---

## 3) Data Structures (Conceptual)

### Ranked list from a collection
For each collection X (anchor/content/vibe), represent results as an ordered list:

- `Lx = [(movie_id, similarity_or_distance, metadata), ...]` of length `nx`

Also build a quick lookup:
- `rankX[movie_id] = 1..nx` (1-indexed rank position)

### Candidate set
- `Candidates = IDs(L_anchor) ∪ IDs(L_content) ∪ IDs(L_vibe)`

### Per-candidate score record
For each candidate `m`, you will compute:
- ranks: `rankA(m), rankC(m), rankV(m)` (or missing)
- RRF terms: `termA, termC, termV`
- fused score: `rrf_score(m)`
- cosines: `simA, simC, simV`
- tie-break: `avg_sim(m)`
- display metadata: title, tmdb_id, release_date, genres (pulled from any collection’s metadata, but consistent)

---

## 4) End-to-End Algorithm (Exact Order of Operations)

### Step 1 — Embed the query once
1. Input: `query_text`
2. Compute one embedding vector: `q = embed(query_text)`
3. Use `q` for searching all three collections.

**Why:** same embedding model across all collections; only the stored movie text differs.

---

### Step 2 — Search each Chroma collection independently (top-K)
For each collection:

1. Query with `q`
2. Retrieve top `n_*` nearest neighbors (cosine similarity or equivalent distance)
3. Store the ranked list in order

Outputs:
- `L_anchor` from `dense_anchor_vectors` (length `n_anchor`)
- `L_content` from `dense_content_vectors` (length `n_content`)
- `L_vibe` from `dense_vibe_vectors` (length `n_vibe`)

Also build rank maps:
- `rankA`, `rankC`, `rankV`

**Note on score semantics:** Chroma may return *distance* rather than *similarity*. For RRF you only need the **rank order**, so this doesn’t matter for fusion—only for tie-break later (where you’ll compute cosine explicitly).

---

### Step 3 — Build the candidate pool (union)
1. `Candidates = union of movie_id across the three lists`
2. If `max_candidates` is enabled and `|Candidates|` exceeds it, apply a deterministic cap rule (optional), e.g.:
   - keep all during early testing (recommended)
   - or cap by taking the union of only the first M from each list

During testing, favor **no cap** unless performance forces it.

---

### Step 4 — Compute Weighted RRF score for each candidate
For each candidate `m`:

1. Determine ranks:
   - `rA = rankA.get(m)` else missing
   - `rC = rankC.get(m)` else missing
   - `rV = rankV.get(m)` else missing

2. Compute per-list RRF term (missing → 0):
   - If present in list X with rank `rX`:
     - `termX = wX / (rrf_k + rX)`
   - Else:
     - `termX = 0`

3. Sum:
   - `rrf_score(m) = termA + termC + termV`

**Interpretation:**
- A movie appearing in multiple lists accumulates score.
- A #1 in one list can still compete strongly, especially if `rrf_k` is small.
- Larger `rrf_k` pushes winners toward multi-list consensus.

---

### Step 5 — Compute cosine similarities for tie-breaking (cheap for you)
You chose to compute these for *all* candidates (good for stability and debugging).

For each candidate `m`:

1. Retrieve the stored embedding for `m` from each collection:
   - `vecA(m)` from `dense_anchor_vectors`
   - `vecC(m)` from `dense_content_vectors`
   - `vecV(m)` from `dense_vibe_vectors`

2. Compute:
   - `simA(m) = cosine(q, vecA(m))`
   - `simC(m) = cosine(q, vecC(m))`
   - `simV(m) = cosine(q, vecV(m))`

3. Tie-break value:
   - `avg_sim(m) = (simA + simC + simV) / 3`

**Missing embedding policy (should be explicit):**
- If a movie might be absent from a collection, choose one:
  - **Strict:** treat missing sim as very low (e.g., 0 or -1)
  - **Lenient:** average only across available sims
- Prefer strict for stability if your data should be complete; prefer lenient if incompleteness is expected.

---

### Step 6 — Final sorting: fused score + tie-break
Sort candidates by:

1. Primary: `rrf_score` descending
2. Secondary (tie-break): `avg_sim` descending
3. Optional stability key: `movie_id` ascending

Return top `return_top_n`.

---

## 5) Output Formatting (What to Display per Result)

Since your metadata is already stored, use it for display.

### Recommended result row fields
- `final_rank`
- `movie_id`
- `tmdb_id`
- `title`
- `release_date`
- `genres`

Scoring / debug fields:
- `rrf_score`
- `avg_sim`
- `rank_anchor`, `rank_content`, `rank_vibe` (or blank if missing)
- `term_anchor`, `term_content`, `term_vibe`
- `sim_anchor`, `sim_content`, `sim_vibe`

---

## 6) Gradio UI: Suggested Controls and Views

### Controls
- Query textbox
- Numeric inputs or sliders:
  - `n_anchor`, `n_content`, `n_vibe`
  - `rrf_k`
  - `w_anchor`, `w_content`, `w_vibe`
  - `return_top_n`
- Checkboxes:
  - `show_raw_lists`
  - `show_score_breakdown`
  - `show_metadata`

### Views
1. **Fused results table** (top-N)
2. Optional: **three raw list tables** side-by-side (anchor/content/vibe)
   - Each row: rank, title, similarity/distance, movie_id
3. Optional: **winner explanation panel**
   - “This movie won because…” show terms and ranks and avg_sim

---

## 7) Practical Tuning Playbook (What to Try First)

### A) Start with balanced settings
- `w_anchor = w_content = w_vibe = 1.0`
- `n_anchor = n_content = n_vibe = 50` (or 100 if you can afford it)
- `rrf_k` in a moderate range (test low vs high)

### B) Use query categories to probe behavior
Test a handful of query types:
- **Plot-driven:** “a slow-burn thriller about surveillance and paranoia”
- **Vibe-driven:** “cozy, whimsical, gentle, found-family animation”
- **Entity-driven:** “like The Dark Knight but more psychological”
- **Hybrid:** “time-loop romance with melancholy tone”

Adjust:
- If you want consensus across lists → increase `rrf_k`
- If you want “best single-list hit can win” → decrease `rrf_k`
- If a list is noisy → reduce its weight

### C) Use the debug columns to diagnose failures
Look for patterns:
- Winner has huge `term_content` but weak others → content dominating; adjust weights or k
- Winner appears mid-rank in all lists but wins by consensus → k may be too large for your taste
- Ties frequent → increase `return_top_n` while inspecting; avg_sim will stabilize ordering

---

## 8) Implementation Checklist (Conceptual)

1. **Embed query once** → `q`
2. **Query each collection** with `q` → `L_anchor`, `L_content`, `L_vibe`
3. **Build rank maps** → `rankA`, `rankC`, `rankV`
4. **Union candidates** → `Candidates`
5. **Compute RRF** for each candidate using `rrf_k`, `w_*`
6. **Compute cosines** `simA/simC/simV` and `avg_sim`
7. **Sort** by `(rrf_score desc, avg_sim desc, movie_id asc)`
8. **Render** fused results (+ optional raw lists) with metadata columns
9. **Log** parameters + top results for each run for reproducibility

---

## 9) Notes on Metadata Usage

Your stored metadata is primarily for:
- display
- lightweight filtering later (not in this baseline)
- debugging (“what genre did this match come from?”)

Even in a vector-only baseline, showing:
- `title`, `release_date`, `genres`
helps you quickly judge if the retrieval “feels right.”

---

## 10) Recommended Minimal Logging for Experiments

For each search run, log:
- query text
- parameters: `n_anchor/n_content/n_vibe`, `rrf_k`, `w_*`
- top-N fused results: `movie_id`, `rrf_score`, `avg_sim`, ranks, sims
- optionally, raw lists LA/LC/LV

This makes it easy to compare runs and discover good defaults.
