# Movie Search Guide: Hybrid Lexical + Vector Retrieval with Confidence-Gated Filters

## Background and goal

You’re building a **movie search system** (movie-catalog “RAG” in the sense of retrieval + ranking) that must handle **many query styles**:

- **Pure lexical**: `"bloodsport"` → likely an exact title match.
- **Structured constraints**: `"Tom Hanks comedies"` → star name + genre.
- **Pure semantic**: `"cozy date night movies"` → vibe/tone.
- **Mixed semantic + constraints**: `"Pixar movies that will make me cry"` → (studio mention) + emotional vibe.
- **Messy / misinformed queries**: `"leandro dicaprio boat movie 2001"` → misspellings + wrong year, but still should surface *Titanic*.

Key design choices:

1. Use **hybrid retrieval**: **BM25** (sparse lexical) + **dense embeddings** (semantic).
2. Avoid “one-vector blur” by storing **multiple dense vectors per movie**:
   - `DenseAnchor` (broad identity / recall safety net)
   - `DenseContent` (aboutness / topics)
   - `DenseVibe` (tone / emotion)
3. Handle “hard filters” only for a **small allowed metadata set** and apply them **only when the system is confident**.
4. Run two retrieval groups:
   - **Group 1 (RAW)**: embed & search the raw query with **no metadata filters** to remain tolerant to user errors.
   - **Group 2 (SOFT)**: embed & search a “soft topics/vibes” query, applying **high-confidence metadata filters** as hard constraints.
5. Fuse results across all retrieval lists using **Reciprocal Rank Fusion (RRF)**.
6. Rerank fused candidates using a **deterministic feature-weighted scorer** (no training for MVP) and then optionally apply an **LLM reranker on the top 50**.

This guide specifies the schemas, derived fields, text templates, query extraction format, retrieval steps, fusion, and reranking—so agentic coders can build the product in Python.

---

## 1) Data available

### 1.1 Movie payload schema (from the upstream movie API)

Assume every movie record starts as the following structure:

```python
class MovieAPIResult(BaseModel):
    id: int
    title: str
    overview: str
    genres: list[str]
    keywords: list[str]

    # People
    cast: list[CastMember]  # includes up to 15 actors, all directors, all producers

    # Production / locale
    production_companies: list[str]
    origin_countries: list[str]
    production_countries: list[str]
    original_language: str                 # full English name (e.g., "English")
    spoken_languages: list[str]            # full English names (e.g., ["English", "Spanish"])

    # Time & length
    release_date: str                      # ISO date string "YYYY-MM-DD"
    runtime: int                           # minutes

    # Ratings / popularity
    maturity_rating: MaturityRating        # enum
    popularity: float
    vote_average: float
    vote_count: int
    is_trending: bool

    # Availability
    watch_providers: list[WatchProvider]
```

```python
class WatchProvider(BaseModel):
    id: int
    name: str
    logo_path: str
    display_priority: int
    types: list[StreamingAccessType]     # subscription / rent / buy
```

```python
class CastMember(BaseModel):
    id: int
    name: str
    role: CastMemberRole                   # actor / director / producer
    character: str | None                  # What character they play (only if actor)
    render_order: int
    profile_image_path: str
```

```python
class MaturityRating(Enum):
    G="G"; PG="PG"; PG_13="PG-13"; R="R"; NC_17="NC-17"; NR="NR"

class StreamingAccessType(Enum):
    SUBSCRIPTION="subscription"; BUY="buy"; RENT="rent"

class CastMemberRole(Enum):
    ACTOR="actor"; DIRECTOR="director"; PRODUCER="producer"
```

---

## 2) What gets indexed

You index each movie in three ways:

1. **Sparse (lexical) index** for BM25
2. **Dense vectors** (3 embedding fields) for semantic retrieval
3. **Metadata fields** for (limited) confidence-gated filtering + rerank features

### 2.1 Allowed metadata filters (and only these)

The system **never** hard-filters on anything outside this set:

- `min_release_date_ts` and/or `max_release_date_ts` (timestamps)
- `max_maturity_rating` (excludes any movie above that rating; if active, exclude NR)
- `min_runtime` and/or `max_runtime` (minutes)
- `watch_provider_ids` (must contain at least one of these provider ids)
- `spoken_languages` (must include one of these languages; see notes below)
- `is_trending` (boolean)

**Important project rule:** People / company / title mentions are **never hard filters** (even if confident), because user typos or fictional character mentions could over-constrain incorrectly.

### 2.2 Sparse fields for BM25

BM25 is used to win on:
- exact/near-exact title matches,
- names (actors/directors/producers),
- studio/provider names,
- rare keywords.

Index a single “searchable text” field (or multiple weighted fields, depending on your engine) built from:

- `title`
- `cast` (names + role + character)
- `production_companies`
- `watch_providers` (names)
- `genres`
- `keywords`
- `overview`

(No numeric data is needed here.)

---

## 3) Derived fields and bucketing rules

### 3.1 Bucketing philosophy

For any **numeric** attribute used in vector text, **bucket it** into a small set of human-readable categories so embeddings don’t overfit exact numbers.

The system uses these bucketed strings in dense text templates; raw numbers are still stored for filtering/reranking.

### 3.2 Required derived fields for DenseAnchor (and often reused elsewhere)

The following are derived from the upstream payload:

#### A) `maturity_text` (from `maturity_rating`)
English description of what the rating means.

Example mapping (edit as needed):
- G → “Suitable for all audiences”
- PG → “Parental guidance suggested”
- PG-13 → “Some material may be inappropriate for children under 13”
- R → “Restricted; under 17 requires accompanying parent or adult guardian”
- NC-17 → “Adults only”
- NR → “Not rated”

#### B) `runtime_bucket` (from `runtime` minutes)
Example bucket scheme:
- `< 80` → “Very short”
- `80–99` → “Short”
- `100–129` → “Medium length”
- `130–159` → “Long”
- `>= 160` → “Very long”

#### C) `budget_bucket` (from `budget`)
Because budget distributions are long-tailed, use log-scale buckets:
- `budget <= 0 or missing` → “Unknown budget”
- `< $5M` → “Micro budget”
- `$5M–$20M` → “Small budget”
- `$20M–$80M` → “Medium budget”
- `$80M–$150M` → “Big budget”
- `>= $150M` → “Blockbuster budget”

(Exact cutoffs can be tuned later; the important part is **few stable buckets**.)

#### D) `release_era_bucket` (from `release_date`)
Bucket into eras that match how users speak:
- If year missing → “Unknown release era”
- `pre-1970` → “Classic (pre-1970)”
- `1970s`, `1980s`, `1990s`, `2000s`, `2010s`
- `2020–present` → “2020s”

Optionally add “early/mid/late” decade if you want more granularity.

#### E) `production_summary` (from `original_language`, `origin_countries`, `production_countries`, `production_companies`, `cast`)
A compact description for identity:
- Original language
- Origin country (or “multiple countries”)
- Production countries
- Production companies
- Key people (actors/directors/producers)

This is NOT a hard filter; it is descriptive text.

#### F) `reception_text` (from `vote_average`, `vote_count`, `popularity`, `is_trending`)
This is an English description:
- how well received,
- how “high confidence” that rating is (vote volume),
- and whether it’s trending.

Example (one or two sentences):
- “Well reviewed (8.2/10) with high review volume (120k votes). Not currently trending.”
- “Mixed reviews (6.4/10) with moderate review volume (8k votes). Trending now.”

---

## 4) Dense vector storage (how movies become vectors)

You store **three embedding vectors per movie**. The embedding model is an open-source sentence embedding model; dimensionality is model-dependent and treated generically.

### 4.1 DenseAnchor vector

**Purpose:** broad recall / “overall identity” vector that prevents missing good matches when other specialized vectors fail.

**DenseAnchor text template (exact structure):**

```text
Title: {title}

Overview: {overview}

Genres: {genres_csv}

Maturity: {maturity_text}

Runtime: {runtime_bucket}

Budget size: {budget_bucket}

Release era: {release_era_bucket}

Production: {production_summary}

Reception: {reception_text}
```

Notes:
- `genres_csv` = comma-joined genre names.
- `production_summary` should list:
  - Directors: …
  - Producers: …
  - Lead cast: …
  - Production companies: …
  - Original language: …
  - Origin countries: …
  - Production countries: …
- This template intentionally uses **bucketed** values for numeric fields.

### 4.2 DenseContent vector

**Purpose:** “aboutness” retrieval (plot/topic/keywords) without tone or numeric distractions.

**DenseContent text template:**

```text
Title: {title}
Overview: {overview}
Genres: {genres_csv}
Keywords: {keywords_csv}
```

Where `keywords_csv` is a comma-joined list of keywords (cap length, e.g., top 30).

### 4.3 DenseVibe vector (LLM-derived)

**Purpose:** tone/emotion/pacing retrieval for queries like “cozy date night,” “make me cry,” “high adrenaline,” etc.

**Inputs available:** `title`, `overview`, `genres`, `maturity_rating` (bucketed), `runtime_bucket`.

**Derived fields (LLM-based):**
- `vibe_summary`: 1–2 sentences describing how it feels to watch.
- `tonal_keywords`: 8–20 keywords/phrases capturing tone (e.g., “cozy”, “bittersweet”, “tense”, “heartwarming”, “campy”).
- `intensity_tags`: 3–6 coarse tags, each categorical:
  - stress: low/medium/high
  - fear: none/mild/strong
  - violence: none/mild/strong
  - sadness: low/medium/high
  - humor: low/medium/high
  (exact tag set can be tuned; keep it small and stable)

**DenseVibe text template:**

```text
Vibe summary: {vibe_summary}

Tonal keywords: {tonal_keywords_csv}

Intensity: stress={stress}; fear={fear}; violence={violence}; sadness={sadness}; humor={humor}

(Genres: {genres_csv}; Maturity: {maturity_text}; Runtime: {runtime_bucket})
```

### 4.4 LLM schema for DenseVibe derivation (exact JSON)

Use a small LLM to generate:

```json
{
  "vibe_summary": "string (1-2 sentences)",
  "tonal_keywords": ["string", "..."],
  "intensity_tags": {
    "stress": "low|medium|high",
    "fear": "none|mild|strong",
    "violence": "none|mild|strong",
    "sadness": "low|medium|high",
    "humor": "low|medium|high"
  }
}
```

**Guardrails:**
- Must be grounded in `overview`/`genres` (no new factual claims).
- Must not invent cast/plot points not present in `overview`.

---

## 5) Query handling

### 5.1 High-level approach

Every user search produces **two query representations**:

1. `raw_query`: the exact user text
2. `soft_query_text`: a merged, expanded string representing:
   - vibe/topics,
   - plus **low-confidence numeric constraints phrased softly**.

Only `soft_query_text` can apply **hard metadata filters** (and only if confidence is high).

### 5.2 Structured query extraction schema (exact)

Use a small LLM to extract the following structured object:

```json
{
  "raw_query": "string",
  "soft_query_text": "string",

  "metadata_filters": {
    "release_date": {
      "min_ts": "int|null",
      "max_ts": "int|null",
      "confidence_bucket": "HIGH|MEDIUM|LOW"
    },
    "runtime": {
      "min_minutes": "int|null",
      "max_minutes": "int|null",
      "confidence_bucket": "HIGH|MEDIUM|LOW"
    },
    "max_maturity_rating": {
      "value": "G|PG|PG-13|R|NC-17|NR|null",
      "confidence_bucket": "HIGH|MEDIUM|LOW"
    },
    "watch_provider_ids": {
      "values": ["int", "..."],
      "confidence_bucket": "HIGH|MEDIUM|LOW"
    },
    "spoken_languages": {
      "values": ["string", "..."],
      "confidence_bucket": "HIGH|MEDIUM|LOW"
    },
    "is_trending": {
      "value": "true|false|null",
      "confidence_bucket": "HIGH|MEDIUM|LOW"
    }
  },

  "soft_entities": {
    "people": ["string", "..."],
    "companies": ["string", "..."],
    "titles": ["string", "..."],
    "fictional_characters": ["string", "..."]
  }
}
```

### 5.3 Confidence buckets and the hard-filter rule

The LLM should output **buckets**, not numeric confidences.

Interpretation:
- `HIGH`  → apply as a **hard metadata filter** (equivalent to confidence > 0.8)
- `MEDIUM/LOW` → do **not** hard filter; instead incorporate into `soft_query_text`

**Global rule:** same threshold for all filter types.

### 5.4 How to build `soft_query_text` (exact behavior)

`soft_query_text` is a single merged string that includes:

- Core semantic topic and vibe words extracted from the query
- Query expansion (synonyms / related phrases) from a small LLM
- **Low-confidence filter hints** expressed softly, e.g.:
  - “short-ish runtime” rather than “under 86 minutes”
  - “around the 2000s” rather than “2000–2009”

Example:
User: `"leandro dicaprio boat movie 2001"`
- raw_query: unchanged
- metadata filters:
  - release_date: LOW (because year is likely wrong in such queries)
  - runtime: none
- soft entities: people=["leandro dicaprio"] (do not hard filter)
- soft_query_text might be:
  - `"boat movie, ship, ocean voyage, romance, disaster; around early 2000s; starring leandro dicaprio"`

---

## 6) Retrieval: two groups × four retrieval lists

### 6.1 Constants (MVP defaults)

- `K_PER_LIST = 500`
- `RRF_K = 60` (RRF constant; configurable)
- Dedup key: `movie_id` only
- Rerank candidate cap: recommended 2000–4000 after fusion + dedup

### 6.2 Group 1: RAW retrieval (no metadata filters)

Input text: `raw_query`

Run these four retrievals, each returning top `K_PER_LIST`:

1. **BM25(raw_query)**
2. **DenseAnchor(raw_query embedding)**
3. **DenseVibe(raw_query embedding)**
4. **DenseContent(raw_query embedding)**

Purpose: tolerate typos, misinformation, and odd query forms.

### 6.3 Group 2: SOFT retrieval (confidence-gated hard metadata filters)

Input text: `soft_query_text`

Build a hard filter expression using only metadata filters where `confidence_bucket == HIGH`.

Apply these filters **before retrieval** for Group 2 (i.e., pre-filtered candidate space).

Run the same four retrievals (top K each) *within the filtered space*:

5. **BM25(soft_query_text, filters=HIGH)**
6. **DenseAnchor(soft_query_text, filters=HIGH)**
7. **DenseVibe(soft_query_text, filters=HIGH)**
8. **DenseContent(soft_query_text, filters=HIGH)**

#### Maturity filter exact semantics
If `max_maturity_rating` is active:
- Exclude any movie whose rating is **above** the threshold in this order:
  `G < PG < PG-13 < R < NC-17`
- Exclude `NR` always when the filter is active.

#### Watch provider filter semantics
If provider filter is active:
- movie must contain **at least one** provider id in the requested set.

#### Spoken languages filter semantics
If spoken languages filter is active:
- movie must have **at least one** spoken language in the requested set.

#### is_trending semantics
If active:
- include only movies where `is_trending == True` (or False, if requested).

### 6.4 Output lists

- “Exact” list is based on **Group 2** (filtered) candidates only.
- “Similar” list is based on **Group 1** (unfiltered) candidates.

(You can choose to always show both lists, or only show “Similar” when “Exact” is sparse; the MVP can always show both.)

---

## 7) Candidate fusion with RRF

### 7.1 Why RRF
You have 8 ranked lists with different score scales (BM25 vs cosine similarities). RRF fuses **rank positions**, avoiding score normalization issues.

### 7.2 RRF algorithm (exact)

For each ranked list `L` and each movie at rank `r` (1-indexed):

```
rrf_score[movie_id] += 1 / (RRF_K + r)
```

Then sort movies descending by `rrf_score`.

Run RRF separately for:
- **Exact RRF**: fuse lists 5–8 (Group 2 only)
- **Similar RRF**: fuse lists 1–4 (Group 1 only)

Finally:
- Deduplicate by `movie_id` (RRF already aggregates per id)

---

## 8) Final reranking (no training) + top-50 LLM rerank

### 8.1 Deterministic feature-weighted rerank (MVP)

After RRF, take the top `N_RERANK` candidates from each lane (Exact / Similar), where `N_RERANK` is in the low 4-digits.

Compute a `final_score` for each candidate as a weighted sum of:

#### A) Retrieval agreement & strength
- `rrf_score` (primary backbone)

#### B) Semantic similarity features (optional but recommended)
If your retrieval engine returns similarity scores:
- cosine(query_embedding, movie_dense_anchor)
- cosine(query_embedding, movie_dense_content)
- cosine(query_embedding, movie_dense_vibe)

Use both query representations where applicable:
- Exact lane: use `soft_query_text` similarities
- Similar lane: use `raw_query` similarities

If you don’t have scores, use rank-derived proxies:
- `rank_feature = 1 / log(2 + rank)` for each list.

#### C) Soft entity match features (never hard filters)
- Presence of any extracted `soft_entities.people` in cast names (string match / fuzzy)
- Presence of `soft_entities.companies` in production companies
- Title similarity to any `soft_entities.titles`
- Fictional character mentions: treat as plain text signals against overview/keywords

#### D) Popularity/quality boost (single metric)
Compute a single boost metric from `popularity`, `vote_average`, `vote_count`.

Suggested (configurable) computation:

1) Weighted rating:
- `R = vote_average`
- `v = vote_count`
- `C = global_mean_vote_average` (precomputed across dataset)
- `m = vote_count_threshold` (e.g., 80th percentile vote_count)

`WR = (v/(v+m))*R + (m/(v+m))*C`

2) Normalize and blend:
`boost = 0.7 * normalize(WR) + 0.3 * normalize(log1p(popularity))`

Add `boost` as a small tie-breaker weight (do not overpower relevance).

### 8.2 LLM rerank on the top 50 (precision pass)

After deterministic scoring:
- Take top 50 results from each lane (or just Exact lane) and run an LLM reranker.

**Inputs provided to the LLM per candidate:**
- title
- overview (truncated)
- genres
- runtime_bucket
- maturity_text
- release_era_bucket
- key cast + directors + producers
- provider names (optional)
- reception_text (optional)

**LLM task:**
- Rerank candidates by best match to the **user’s raw query intent**, without introducing new constraints.
- Return:
  - ordered list of `movie_id`s
  - optional short notes for debugging (not shown to users)

**LLM rerank output schema:**
```json
{
  "ranked_movie_ids": [123, 456, 789],
  "notes": "optional short debugging notes"
}
```

---

## 9) End-to-end flow summary (what the search “does”)

1. **Ingest** movie payloads and compute derived bucketed fields.
2. Build three dense texts (`DenseAnchor`, `DenseContent`, `DenseVibe`) and embed them.
3. Index:
   - BM25 text fields
   - three vectors
   - allowed metadata filter fields
   - popularity/quality boost metric components
4. At query time:
   1) Run small LLM extraction → `raw_query`, `soft_query_text`, metadata filter buckets, soft entities.
   2) **Group 1**: run 4 retrieval lists on `raw_query` (no filters).
   3) **Group 2**: build hard filter expression from `HIGH` buckets and run 4 retrieval lists on `soft_query_text` within the filtered space.
   4) Fuse with **RRF** into:
      - “Exact” candidates (Group 2)
      - “Similar” candidates (Group 1)
   5) Deterministic rerank (feature-weighted scoring, no training).
   6) Optional LLM rerank of top 50 for final polish.
5. Return:
   - `exact_results` (strict constraints honored where confident)
   - `similar_results` (tolerant fallback that can recover from user mistakes)

---

## 10) Reasoning recap (why each decision exists)

- **Two groups (RAW + SOFT):** ensures you don’t “delete the answer” when users are wrong about details (RAW), while still respecting constraints when they seem reliable (SOFT).
- **Confidence-gated metadata filters:** prevents hard-filtering on fragile inferences and keeps recall high.
- **Never hard-filter on people/companies/titles:** avoids overcommitting to misspellings, partial memories, and fictional character names.
- **Three vectors:** enough separation to reduce “one-vector blur” without exploding into dozens of vectors that might miss top-K.
- **RRF fusion:** combines heterogeneous retrieval methods reliably using rank-only fusion.
- **Deterministic rerank + top-50 LLM:** cheap and scalable for MVP, yet still allows an LLM to apply “spirit of the query” judgment where it matters most.
