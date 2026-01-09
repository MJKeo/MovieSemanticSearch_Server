# Movie Search Guide (Source of Truth: Movie Search Overview)

This document defines the **movie search system** for a large-scale catalog. It is intended to be handed to coding agents implementing the search pipeline in **Python**.

The system supports a wide range of user query styles:

- **Lexical**: `"bloodsport"` → title match
- **Constrained**: `"Tom Hanks comedies"` → actor + genre
- **Semantic**: `"cozy date night movies"` → vibe/tone
- **Mixed**: `"Pixar movies that will make me cry"` → brand term + emotional vibe
- **Messy/misinformed**: `"leandro dicaprio boat movie 2001"` → misspellings + wrong facts

The core approach is **hybrid retrieval + fusion + reranking**:
- **Sparse retrieval (BM25)** for exact/rare-term matches
- **Dense retrieval (embeddings)** for semantics
- **Multiple vectors per movie** to avoid “one-vector blur”
- Two retrieval groups: **RAW** (no hard filters) and **SOFT** (confidence-gated hard filters)
- **RRF** (Reciprocal Rank Fusion) to combine ranked lists
- Deterministic reranking for MVP + optional LLM rerank on the top 50

> **Important constraint**: This guide is **stack-agnostic**. It describes data structures and algorithms, not vendor products.

---

## 1) Data available (base movie object)

All fields below are the canonical base movie object. If it is not listed here, you must not rely on it.

### 1.1 Base schema (Python-style)

```python
from dataclasses import dataclass
from typing import Optional, Literal

ProviderType = Literal["subscription", "rent", "buy"]

@dataclass(frozen=True)
class WatchProvider:
    id: int
    name: str
    logo_path: str
    display_priority: int
    types: list[ProviderType]

@dataclass(frozen=True)
class ParentalGuideItem:
    category: str
    severity: str  # e.g., "Mild", "Moderate", "Severe"

@dataclass(frozen=True)
class Movie:
    # Identity
    id: str  # e.g., "tt0405094" (IMDB id)
    tmdb_id: int # the ID corresponding to the TMDB database
    title: str
    original_title: Optional[str]  # written in English letters if present

    # High-level descriptors
    overall_keywords: list[str]
    genres: list[str]

    # Time & length
    release_date: str  # "YYYY-MM-DD"
    duration: int      # minutes

    # Locale / production
    countries_of_origin: list[str]
    languages: list[str]           # audio languages; index 0 = original language
    filming_locations: list[str]
    production_companies: list[str]

    # Money / availability
    budget: Optional[int]          # nominal USD, not inflation-adjusted
    watch_providers: list[WatchProvider]

    # Maturity guidance
    maturity_rating: str           # "G", "PG", "PG-13", "R", "NC-17", "Unrated"
    maturity_reasoning: list[str]
    parental_guide_items: list[ParentalGuideItem]

    # Plot
    overview: str
    synopsis: str | None = None
    plot_keywords: list[str]

    # People / characters
    directors: list[str]
    writers: list[str]
    producers: list[str]
    composers: list[str]
    actors: list[str]
    characters: list[str]

    # Popularity / reception
    imdb_rating: float             # 0..10
    metacritic_rating: float       # 0..100
    reception_summary: Optional[str]  # AI-generated
```

---

## 2) What gets indexed

You index each movie into:

1. **Sparse text fields** for BM25
2. **Dense vectors**: `DenseAnchor`, `DenseContent`, `DenseVibe`
3. **Metadata fields** for a limited, confidence-gated hard-filter set
4. **Derived scalar features** for reranking

---

## 3) What is used for what

### 3.1 Metadata to hard-filter on (and only these)

These are the only fields that can be used as **hard constraints**:

- Release date window: `minimum_release_date_timestamp`, `maximum_release_date_timestamp`
- Duration window: `min_duration`, `max_duration`
- `genres`
- `watch_providers`
- Maturity rating: `min_maturity_rating`

**Non-negotiable rule:** never hard-filter on people, titles, companies, or characters (they can be misspelled or user-misremembered).

#### 3.1.1 Maturity “min” semantics

To support queries like “R-rated horror”:
- Define rating order: `G < PG < PG-13 < R < NC-17`
- `min_maturity_rating = R` means include only movies rated **R or NC-17**
- When `min_maturity_rating` is active, treat `"Unrated"` as **not comparable** and exclude it from the exact (filtered) lane.

For family-friendly queries (“for kids”), do **not** hard filter; rely on semantic matching against maturity guidance text in vectors and lexical matching on the maturity reasoning / parental categories.

### 3.2 Lexical search fields (BM25 weighting tiers)

Lexical search is driven by these fields, with weight tiers:

**Strong matches (highest boost)**
- `title`
- `original_title`

**Medium matches**
- `genres`
- `languages`
- `countries_of_origin`
- `filming_locations`
- `watch_providers.name`
- `overview`
- `directors`
- `writers`
- `producers`
- `composers`
- `actors`
- `characters`
- `production_companies`

**Weaker matches (lowest boost)**
- `overall_keywords`
- `plot_keywords`

> Implementation note: you can either (a) store separate weighted text fields, or (b) store one concatenated text field with manual repetitions for high-boost fields. The algorithmic intent is: title/original title dominate lexical matching; keywords help but don’t override.

---

## 4) Derived attributes for vector search (how they are built)

The doc specifies a set of **vector-search attributes** and how they are derived from movie fields. Below are the exact formats.

### 4.1 Title string

```text
If original_title exists:
  "Movie: <title> (<original_title>)"
Else:
  "Movie: <title>"
```

### 4.2 Overview

Use `overview` as-is.

### 4.3 Keywords

Use `overall_keywords` as-is.

### 4.4 Genres

Use `genres` as-is.

### 4.5 Release decade bucket

Convert `release_date` into a decade label, e.g. `"80s"`.

Suggested function:

```python
def decade_bucket(release_date: str) -> str:
    # release_date: "YYYY-MM-DD"
    year = int(release_date[:4])
    decade = (year // 10) * 10
    return f"{str(decade)[-2:]}s"  # 1980 -> "80s"
```

### 4.6 Duration bucket

Bucket `duration` into short descriptive categories.

Recommended buckets (tunable but must be stable and few):
- `< 102` → `"short, quick watch"`
- `102–118` → `"Standard length"`
- `118–144` → `"Long"`
- `> 144` → `"Very long"`

### 4.7 Budget bucket for era (small vs blockbuster vs none)

Budget is encoded only as whether it is **exceptionally small** or **exceptionally large for its era**, otherwise “not noteworthy.”

**Key requirement:** it must be **era-aware** (release-year adjusted), but avoid overclaiming semantics.

Output categories:
- `"small budget"`
- `"big budget, blockbuster"`
- `""` (when budget is neither big nor small)

Recommended approach:
Below is a researched threshold for what is considered a small and large budget. Only make note of the budget if it falls under the threshold to be considered small budget or above the threshold to be considered big budget.
```python
_DECADE_THRESHOLDS: dict[int, tuple[int, int]] = {
    1920: (100_000, 1_000_000),
    1930: (150_000, 2_000_000),
    1940: (250_000, 3_000_000),
    1950: (750_000, 10_000_000),
    1960: (1_000_000, 15_000_000),
    1970: (2_000_000, 20_000_000),
    1980: (7_000_000, 40_000_000),
    1990: (20_000_000, 100_000_000),
    2000: (25_000_000, 150_000_000),
    2010: (25_000_000, 200_000_000),
    2020: (25_000_000, 250_000_000),
}
```
If the movie was made before the 1920s just use the 1920s numbers. If produced 2030 or later just use 2020 numbers.


### 4.8 Plot (LLM-derived)

Overviews are often vague, so plot is derived from:
- `plot_keywords`, `plot_summaries`, `synopsis`, `overview`

A (larger, offline) LLM generates:
- `plot_synopsis`: a brief but complete summary (spoilers allowed)
- `plot_keyphrases`: a list of key terms and phrases related to the plot (ex. ["haunted hotel","labyrinth","isolation","based on novel","hedge maze","domestic violence","surrealism"])

**Output schema:**
```json
{
  "plot_synopsis": "string",
  "plot_keyphrases": ["string", "..."]
}
```

- plot_keyphrases will be combined with plot_keywords fetched from IMDB to form one single plot_keywords variable
- plot_synopsis replaces plot_summaries and synopsis (fetched from IMDB) to as the movie's "synopsis" variable


### 4.9 Maturity guidance (derived formatting)

If `maturity_rating == "Unrated"`:
- generate one string per parental guide item: `"<severity> <category>"`
- then join them (comma or newline)

Otherwise:
- map `maturity_rating` to a brief description (configurable mapping)
- append maturity reasoning

Example output:
- `"Suitable for all audiences. Rated PG for mild peril."`

### 4.10 Production string

Format:

```text
Produced in <countries_of_origin> by <production_companies>. Filming happened in <filming_locations>
```

### 4.11 Languages string

Format:

```text
Primary language: <languages[0]>. Audio also available for <languages[1:]>
```

Do not assume len(languages) > 0

### 4.12 Cast string (truncated)

Format:

```text
Directed by <directors>. Written by <writers>. Produced by <producers[:4]>. Music composed by <composers>. Main actors: <actors[:8]>
```

### 4.13 Characters string (truncated)

Format:

```text
Main characters: <characters[:8]>
```

### 4.14 Reception score + tier label

Compute numeric score:

```python
reception_score = (0.4 * 10 * imdb_rating) + (0.6 * metacritic_rating)
# Note: imdb_rating is 0..10, metacritic is 0..100
```

Tier label mapping:

- `>= 81` → `"Universally acclaimed"`
- `>= 61` → `"Generally favorable reviews"`
- `>= 41` → `"Mixed or average reviews"`
- `>= 21` → `"Generally unfavorable reviews"`
- else → `"Overwhelming dislike"`

**Important:** use the numeric score for ranking; the string tier is primarily for embedding + UX.

### 4.15 Reception summary (optional)

If `reception_summary` exists, include:

```text
Review summary: <reception_summary>
```

---

## 5) Dense vectors (3 embeddings per movie)

You store **three vectors per movie** in the same “movie document/row,” differentiated by field name:

- `dense_anchor_vector`
- `dense_content_vector`
- `dense_vibe_vector`

### 5.1 DenseAnchor (broad identity / recall safety net)

**Goal:** capture the full “movie card” identity so at least one dense retriever has good recall for most queries.

**DenseAnchor text template:**
```text
{TitleString}

Overview: {overview}

Genres: {genres_csv}
Keywords: {overall_keywords_csv}

Release decade: {decade_bucket}
Duration: {duration_bucket}
Budget scale: {budget_bucket_for_era}

Maturity guidance: {maturity_guidance_text}

Production: {production_text}
Languages: {languages_text}

Cast: {cast_text}
Characters: {characters_text}

Reception: {reception_tier} (score={reception_score})
{optional_review_summary_line}
```

### 5.2 DenseContent (aboutness / plot & themes)

**Goal:** match “what happens” and major themes without being dominated by cast/production.

**DenseContent text template:**
```text
{TitleString}

Plot synopsis: {plot_synopsis}

Plot keyphrases: {plot_keyphrases_csv}

Genres: {genres_csv}
Keywords: {overall_keywords_csv}
```

### 5.3 DenseVibe (viewing experience / suitability)

**Goal:** match “how it feels to watch” queries (e.g., *chill date night*, *edge-of-your-seat thrillers*, *gross-out horror*, *comfort watch*, *background-friendly*).

**Non-goal:** do **not** encode themes/messages or restate plot events. DenseContent already owns “what happens + themes”; DenseVibe is the *viewer experience*.

#### 5.3.1 LLM-derived vibe fields (offline, very small, cheap)

A single-pass, fast, low-cost LLM generates three fields:

```json
{
  "vibe_summary": "string",
  "vibe_keywords": ["string", "..."],
  "suitability_keywords": ["string", "..."]
}
```

**Field purposes (what each represents and why it exists):**
- `vibe_summary`  
  A **single short sentence** capturing the *felt viewing experience* (mood, pacing, intensity, scare/gross/humor style, emotional feel, and how “locked in” you need to be). It should be spoiler-light (no plot events), but it *may be spoiler-aware* (e.g., “surprisingly intense for a family film”).
- `vibe_keywords`  
  **12–24 short phrases** (1–3 words each) that act as “semantic hooks” for the vibe embedding. These should be **viewer-experience descriptors** (tense, cozy, bleak, jump-scarey, slow-burn, high-energy, etc.), not plot nouns or themes.
- `suitability_keywords`  
  **0–10 short phrases** describing **watch context** (date-night friendly, group-watch, background watch, requires full attention, family movie night, etc.). Keeping suitability separate improves coverage for “what should I watch *in this situation*?” queries while still embedding everything together.

**Hard rules for generation:**
- Do **not** describe plot events, story topics, character names, or locations.
- Do **not** describe themes/messages (e.g., “grief”, “justice”, “friendship”).
- Avoid “absence” statements like “not scary”, “no violence”. Prefer presence/degree descriptors (e.g., “light scares”, “minimal gore”, “intense violence”, “gory”).
- Do not copy input keywords verbatim unless they are already clearly vibe-words; prefer rephrasing.

#### 5.3.2 Input to the DenseVibe LLM

Provide these labeled inputs (some may be empty). The model should treat **genres** as a light prior, not the main signal.

- `overview`
- `genres`
- `overall_keywords` (hint only; do not copy theme-heavy phrases directly)
- `plot_keywords` (hint only; do not copy plot nouns directly)
- `imdb_story_text` (optional, raw IMDB scrape; **use the whole string**):
  - If `imdb_synopsis_list[0]` exists and is non-empty → use it
  - Else if `imdb_plot_summaries_list[0]` exists and is non-empty → use it
  - Else omit `imdb_story_text`

**Important:** `imdb_synopsis_list` / `imdb_plot_summaries_list` are **raw scraped IMDB fields used only during preprocessing**. They are **not** the same as the derived `Movie.synopsis` you store later (which is generated by your plot-derivation step).

Also include (for suitability hints; do not over-index):
- `maturity_rating`, `maturity_reasoning`, `parental_guide_items`
- `reception_summary` (optional; can hint pacing, crowd-pleaser vs bleak, etc.)

#### 5.3.3 Post-processing (only this)

After the LLM returns JSON:
- `vibe_summary = vibe_summary.strip().lower()`
- For every item in `vibe_keywords` and `suitability_keywords`: `item = item.strip().lower()`

No other normalization, deduping, filtering, or rewriting.

#### 5.3.4 DenseVibe text template (embedded)

```text
Vibe summary: {vibe_summary}

Vibe keywords: {vibe_keywords_csv}

Suitability: {suitability_keywords_csv}

Maturity guidance: {maturity_guidance_text}
Duration: {duration_bucket}
Genres: {genres_csv}
```

---

## 6) Query understanding (RAW + SOFT)

Every user query produces:
- `raw_query`: the original text (unchanged)
- `soft_query_text`: an expanded, merged semantic query intended for better dense retrieval and soft constraint handling
- a set of **candidate metadata filters** with **confidence buckets** (HIGH/MED/LOW)
- “soft entities” (people, titles, companies, fictional characters) for lexical/semantic boosting (never hard filters)

### 6.1 Structured query extraction schema (exact)

A **small, cheap model** (used online at query time) returns:

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
    "duration": {
      "min_minutes": "int|null",
      "max_minutes": "int|null",
      "confidence_bucket": "HIGH|MEDIUM|LOW"
    },
    "genres": {
      "values": ["string", "..."],
      "confidence_bucket": "HIGH|MEDIUM|LOW"
    },
    "watch_provider_ids": {
      "values": ["int", "..."],
      "confidence_bucket": "HIGH|MEDIUM|LOW"
    },
    "min_maturity_rating": {
      "value": "G|PG|PG-13|R|NC-17|null",
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

### 6.2 Confidence handling (hard vs soft)

- **HIGH** → apply as a **hard filter** (only within the SOFT group; never within RAW)
- **MEDIUM/LOW** → do not hard filter; instead incorporate into `soft_query_text` as soft constraints

The confidence is bucketed, not numeric, because LLMs are unreliable at precise probabilities.

### 6.3 Building `soft_query_text`

`soft_query_text` is a single merged string that includes:
- semantic topic/vibe terms extracted from the query
- synonyms / paraphrases (from the small model)
- low-confidence constraints expressed softly:
  - e.g., “around the 2000s” rather than “2000–2009”
  - “shorter runtime” rather than “< 90 minutes”

Example:
Query: `"leandro dicaprio boat movie 2001"`
- hard filters: likely NONE (year is low confidence)
- soft entities: people=["leandro dicaprio"]
- soft_query_text:
  - `"boat movie, ship, ocean voyage, romance, disaster; around early 2000s; starring leandro dicaprio"`

---

## 7) Retrieval (two groups × four ranked lists)

### 7.1 Constants (MVP defaults)

- `K_PER_LIST = 500`
- `RRF_K = 60` (tunable)
- Dedup key: `movie_id` only
- LLM rerank size: `TOP_LLM = 50`

### 7.2 Group 1: RAW retrieval (no hard filters)

Input: `raw_query`

Run these ranked retrieval lists:
1. `BM25(raw_query)`
2. `DenseAnchor(raw_query_embedding)`
3. `DenseContent(raw_query_embedding)`
4. `DenseVibe(raw_query_embedding)`

Purpose: keep recall high when the user is wrong about details.

### 7.3 Group 2: SOFT retrieval (confidence-gated hard filters)

Input: `soft_query_text`

Build a filter expression from any metadata filter where `confidence_bucket == HIGH` and apply it **before** retrieval.

Run:
5. `BM25(soft_query_text, filters=HIGH_ONLY)`
6. `DenseAnchor(soft_query_text_embedding, filters=HIGH_ONLY)`
7. `DenseContent(soft_query_text_embedding, filters=HIGH_ONLY)`
8. `DenseVibe(soft_query_text_embedding, filters=HIGH_ONLY)`

Purpose: provide an “exact lane” where confident constraints are honored.

---

## 8) Fusion with Reciprocal Rank Fusion (RRF)

You have heterogeneous scores (BM25 vs cosine), so use rank-only fusion.

### 8.1 RRF formula (exact)

For each ranked list and each movie at rank `r` (1-indexed):

```python
rrf_score[movie_id] += 1 / (RRF_K + r)
```

Sort by `rrf_score` descending.

### 8.2 Apply RRF separately per lane

- **Exact lane**: fuse lists (5–8)
- **Similar lane**: fuse lists (1–4)

Equal weights are used for MVP.

---

## 9) Reranking (no training for MVP)

After RRF, rerank the candidates deterministically, then optionally apply an LLM rerank on the top 50.

### 9.1 Deterministic feature-weighted rerank

Compute `final_score` for each candidate as a weighted sum of:

**A) Fusion strength**
- `rrf_score` (backbone)

**B) Dense similarity features**
If available, include similarity scores from each dense list; otherwise use rank-based proxies.

Recommended rank proxy:
```python
rank_feature = 1 / (1 + rank)
```

**C) Lexical match features**
Compute match strength in tiers:
- strong: title/original_title
- medium: (genres, languages, countries, filming locations, providers, overview, people, characters, companies)
- weak: overall_keywords, plot_keywords

These do not hard-filter; they boost.

**D) Quality/popularity tie-breakers**
- Use `reception_score` as a small positive weight.

> Principle: tie-breakers must not override relevance.

### 9.2 LLM rerank on top 50 (optional precision pass)

After deterministic scoring, take the top 50 results per lane (or just Exact) and ask a stronger LLM to rerank.

Inputs per candidate should include:
- title + original title
- overview (truncated)
- genres
- plot synopsis (truncated)
- vibe summary/keywords (if available)
- maturity guidance
- decade + duration bucket + budget bucket
- key cast/characters (truncated)
- reception tier + numeric score

**Output schema:**
```json
{
  "ranked_movie_ids": [123, 456, 789],
  "notes": "optional brief notes for debugging"
}
```

---

## 10) Final outputs

Return two lists to the product:

1. **Exact results**
   - Derived from Group 2 (SOFT) with HIGH-confidence hard filters applied
   - Movies violating confident filters never appear here (they were filtered out pre-retrieval)

2. **Similar results**
   - Derived from Group 1 (RAW) with no hard filters
   - Intended to rescue results when the user is wrong or too specific

This gives the UX behavior:
- “Here are the best matches to what you asked”
- “Here are similar matches (even if not exact)”

---

## 11) Rationale (why these decisions exist)

- **Multiple vectors per movie** reduce the “one vector blurs everything” problem:
  - Content retrieval and vibe retrieval are separable.
  - Anchor provides recall and identity.
- **Two groups (RAW + SOFT)** prevent hard filters from deleting the true intent when the user is mistaken.
- **Hard filters are limited** to a small set to avoid brittle assumptions.
- **Never hard-filter on names** (people/companies/titles/characters) because typos and memory errors are common.
- **RRF** is a robust, simple fusion method across heterogeneous retrieval lists.
- **Deterministic reranking** is fast, cheap, and sufficient for MVP; LLM rerank only touches the top 50.

---

## 12) Implementation checklist (for coding agents)

At indexing time:
- validate base Movie schema
- derive all vector-search attributes (decade, duration bucket, budget bucket, maturity guidance, production/language/cast/character strings)
- run offline LLMs to derive plot synopsis/keyphrases
- run an offline (very small) LLM to derive DenseVibe fields: vibe_summary, vibe_keywords, suitability_keywords (lowercase + trim only)
- build 3 dense texts and embed them
- index sparse fields with boosts

At query time:
- run small model to produce structured extraction + soft_query_text
- build HIGH-only hard filter expression (SOFT lane)
- run 8 retrieval lists (2 groups × 4 retrievers)
- run RRF per lane
- deterministic rerank per lane
- optional LLM rerank top 50
- return exact + similar lists
