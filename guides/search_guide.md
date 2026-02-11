# Movie Search Pipeline (MVP) — Implementation Guide

## Goal and non-goals

**Goal:** Given a freeform `query_text`, return the best-matching movies (initially: just “best matches,” not a full conversational agent). 
**Non-goals (for MVP):** diversification/MMR, “more like this,” cross-session personalization, and advanced learning-to-rank training (all reserved for post-MVP followups).

---

## Data sources and inputs

### Client → Server inputs

The client sends these values on every search request: 

* `query_text: str` — the user’s raw query.
* `times_shown_in_session: Dict[tmdb_id, int]` — number of times each movie has appeared in the current session.
* `selected_filters: Dict[...]` — **explicit UI filters** that pre-filter which movies may be returned (these are the only true “hard filters”).

### Server-side data stores

1. **Vector DB**

* 8 separate vector collections (searched independently). Each returns:
  `top_n_candidates_<vector_collection>: Dict[tmdb_id, distance_or_similarity]` 

2. **Movie Metadata DB**

* Static movie attributes:
  `{ tmdb_id: metadata_json }` 

3. **Dynamic DB**

* Frequently updated, not embedded in vectors. For MVP:
  `trending_movie_ids: List[tmdb_id]` 

4. **Lexical DB**

* Lexicographical store for entity search by category.
* Also stores:
  `top_candidates_by_entity: Dict[entity_id, ranked_candidates]` 

---

# Phase 1 — Query Understanding (parallel)

Everything in this phase runs **in parallel** to minimize latency. 

## 1) Lexical Entity Extraction

Purpose: identify “entity-like” strings the user likely intended as *specific* identifiers (title, person, studio, etc.), and expand them.

Output schema (list of extracted entities): 

```json
[
  {
    "candidate_entity_phrase": "direct substring from query",
    "most_likely_category": "PERSON | CHARACTER | TITLE | STUDIO",
    "include_or_exclude": "INCLUDE | EXCLUDE",
    "expansions": ["alias1", "alias2", "phonetic_variant1"]
  }
]
```

**Why this exists:** entity searches should not depend on embeddings to “happen to catch” exact names. Entity extraction also lets you do alias/phonetic expansions (while still keeping the original text).

## 2) Metadata Preferences (structured)

Purpose: extract **concrete metadata constraints/preferences** from `query_text` and express them in a structured way that later scoring can evaluate on a [0,1] scale.

Example schema pieces include: 

* `genre.must_contain`, `genre.must_not_contain` (as enums)
* `release_year.value_1`, `release_year.operation (EXACT|BEFORE|AFTER|BETWEEN)`, `release_year.value_2`
* plus:

  * `prefers_trending_now`
  * `prefers_popular_overall`
  * `prefers_critically_acclaimed` 

**Key MVP rule:** anything “hard” the user types (e.g., “on Netflix”) becomes *metadata preference scoring*, not a pre-filter. Only `selected_filters` from the UI are pre-filters. 

## 3) Vector Routing (per vector)

Purpose:

* decide which vector collections matter for this query (relative importance),
* produce an expanded/rewritten query for each vector.

Output schema: 

```json
{
  "<vector-name>_data": {
    "relevance_to_query": "NONE | SMALL | MEDIUM | LARGE",
    "vector_query": "rewritten + expanded query text for this vector"
  }
}
```

**Why:** different vectors represent different semantic spaces; the router determines which collections are worth querying and how to phrase the query for that space.

## 4) Channel Weights (lexical vs semantic vs metadata)

Purpose: decide how much each scoring channel should influence the final ranking.

Output schema: 

```json
{
  "lexical_match_importance": "NONE | SMALL | MEDIUM | LARGE",
  "semantic_match_importance": "NONE | SMALL | MEDIUM | LARGE",
  "metadata_match_importance": "NONE | SMALL | MEDIUM | LARGE"
}
```

**Why:** not every query should be dominated by the same channel. (“Brad Pitt movies” should lean lexical; “movies that will make me cry” should lean semantic; “before 2000” should add metadata pressure.)

---

# Additional initial computations

## Query Complexity

This is a single scalar multiplier used to scale how many candidates you pull.

Definitions: 

* `V = # active vectors`
* `E = # lexical entities extracted`
* `M = # metadata prefs extracted`

Default formula: 
[
complexity = clamp(1.0 + 0.15*(V-1) + 0.10*E + 0.10*M, 1.0, 2.5)
]

**Why:** multi-constraint queries need deeper candidate retrieval to avoid missing good intersections.

---

# Candidate generation

All retrieval is subject to **UI pre-filters** (`selected_filters`) first. 

## 1) Lexical search

**Inputs:** extracted entities + categories. 
**Outputs:**

* `lexical_query_candidates`: ranked list of `(tmdb_id, lexical_score)` 
  **Candidate count scaling:** depends on `lexical_match_importance` and `query_complexity`. 

**Implementation note:** your lexical DB is entity-category aware, so you should be issuing category-specific queries rather than throwing all tokens at title matching.

## 2) Vector search (per vector collection)

For each active vector, you do two fetches: 

* `original_query_candidates`: using the **full original query text**
* `expanded_query_candidates`: using the **expanded/rewritten vector_query** from routing

Total candidates returned per vector scales based on channel weights + complexity (the PDF says “based on lexical_match_importance and query_complexity,” but conceptually it should scale with semantic importance; treat the PDF as source-of-truth, but you’ll likely fix that naming mismatch in code). 

**80/20 mix rule:** 

* 80% of vector candidates come from the expanded query
* 20% from the original query

**Why:** you preserve recall for literal user phrasing while letting the rewritten query carry most of the semantic load.

---

# Unioning candidates

Goal: produce up to **1000 deduped candidates** across all sources.

Algorithm: 

1. Take the first **50%** of candidates from each source (lexical + each vector fetch list), union and dedupe.
2. If still `< 1000`, take the next **5%** from each source.
3. Repeat until:

   * 1000 unique candidates are collected, or
   * every source is exhausted.

**Why:** this approximates “progressive deepening” without fancy gating logic, and it’s simple to implement while still robust.

---

# Reranking value generation

## Phase 1: Prefetch all data for reranking

For each candidate movie: 

* Fetch full metadata JSON.
* Fetch full vector(s) for each active vector collection where relevance is not NONE.

**Why:** reranking requires exact similarity computations and metadata scoring.

---

## Vector similarity recomputation (exact)

For each active vector collection:

* compute similarity between movie vector and query vector for:

  * original query text
  * expanded query text 

If vectors are normalized, cosine similarity can be dot product. 

**Critical implementation detail:** your vector DB might return “distance” rather than “similarity.” Your recomputation step should standardize to “higher is better” similarity before z-scoring. (E.g., if you have cosine *distance*, convert with `sim = 1 - dist`.)

---

## Z-score normalization per vector (and per query version)

For each vector collection and for each of the two query variants (OG vs expanded): 
[
z = \frac{sim - mean}{std}
]
Clamp `z` to [-5, 5], then normalize to [0,1]: 
[
z_norm = \frac{z_clamped + 5}{10}
]

Then combine OG + expanded with fixed weights: 
[
collection_z = 0.2 * z_{og} + 0.8 * z_{expanded}
]

**Why:**

* z-scoring makes vector scores comparable within the candidate set.
* the 0.2/0.8 mix matches your “expanded query is primary” decision.

---

## Final vector score (across active vectors)

Combine per-collection z-scores using vector weights (derived from routing relevance): 
[
vector_score = \sum_{v \in enabled} weight[v] * collection_z[v]
]
Max possible is 1.0 (if weights sum to 1 and each collection_z is [0,1]). 

**Why:** this fuses multiple semantic spaces into a single bounded signal.

---

## Lexical score normalization

You will have lexical match scores for some candidates (those returned by lexical search). For candidates without lexical data, you explicitly compute lexical match score (how depends on your lexical DB implementation). 

Then apply the same z-score → clamp → [0,1] normalization. 

**Why:** lexical scores need to live on a comparable scale to vector scores.

---

## Metadata preference score

For each extracted metadata preference:

* compute a score in [0,1] depending on the preference type (categorical match, range score, before/after logic, etc.) 
  Then average equally: 
  [
  metadata_score = \sum_{p=1..n} \frac{1}{n} * score[p]
  ]
  Max is 1.0. 

**Why:** you get a single bounded “metadata alignment” signal without training.

---

## Session penalty

Penalty is based on how many times the movie was shown in this session: 
[
penalty = \frac{min(t,5)}{5}
]
where `t = times_shown_in_session[tmdb_id]`.

**Why:** repeated exposure without selection should reduce rank, capped to avoid infinite punishment.

---

# Final scoring and ranking

Final score formula: 
[
score = weights[L] * lexical_score + weights[V] * vector_score + weights[M] * metadata_score + P * penalty
]

**Important note:** `penalty` increases with repeated exposure, so `P` should typically be **negative** (or you subtract the penalty term) if the goal is to downrank repeated movies. The PDF writes it as `+ P * penalty`; implement it as either:

* `score += P * penalty` with `P < 0`, or
* `score -= P * penalty` with `P > 0`.

Then:

1. sort candidates by `score` descending
2. return top results

---

# Why these decisions are defensible (short rationale)

* **Parallel query understanding** keeps latency low and isolates responsibilities (entity extraction vs vector routing vs metadata scoring).
* **Expanded vector query dominates retrieval (80%) and scoring (0.8 weight)** to turn noisy user text into high-signal search phrases while keeping fallback recall.
* **Progressive unioning** is a simple, robust approximation to more complex candidate orchestration.
* **Per-query z-score normalization** solves “one vector always dominates” and makes channel fusion feasible without training.
* **Session penalty** nudges exploration while still allowing strong matches to reappear (cap at 5).

---

# Post-MVP followups to explore (threads not fully closed)

These are the main “next” items that weren’t finalized and will matter once MVP is working:

1. **Exact numeric mappings**

   * Mapping `"NONE|SMALL|MEDIUM|LARGE"` to numeric weights for:

     * channel weights (L/V/M)
     * per-vector weights
     * penalty coefficient `P`

2. **Lexical DB design details**

   * Index fields per entity category (title variants, people aliases, studio normalization)
   * Fuzzy matching thresholds
   * How to compute lexical match scores for candidates not returned by lexical retrieval

3. **Metadata scoring functions**

   * Precise scoring shapes for:

     * BEFORE/AFTER/BETWEEN vs EXACT
     * discrete vs continuous (runtime, year)
     * “must_not_contain” penalties
   * Whether some metadata prefs should be weighted more than others (vs equal averaging)

4. **Trending / popularity / acclaim integration**

   * The PDF includes `prefers_trending_now/popular/critically_acclaimed` extraction, but the exact data sources and scoring are TBD.
   * Decide whether “requests_trending” triggers:

     * a restricted candidate universe retrieval pass, and/or
     * purely metadata boosting, and how those interact.

5. **Reranker evolution**

   * MVP is deterministic fusion, but later:

     * learn weights from clicks/saves (LTR)
     * replace or augment with a small MLP / LambdaMART

6. **Diversity (MMR)**

   * We parked it earlier; once browsing UX matters, define:

     * when to diversify (broad/vibe queries)
     * similarity function for redundancy (single vector vs blended)

7. **“More like this” / precomputed neighbor graph**

   * Later UX feature, likely built from a blended multi-vector similarity.

8. **Evaluation + regression harness**

   * A test set of queries + expected outcomes
   * Offline metrics (MRR/NDCG) and “query class” breakdowns (entity queries vs vibe queries)
