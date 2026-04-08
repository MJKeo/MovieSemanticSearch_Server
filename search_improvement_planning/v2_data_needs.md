# V2 Data Needs

Data that must be captured, generated, or restructured before V2 search can be
implemented. Organized by dependency order where possible.

---

## Prerequisites (unblocks other work)

### 1. IMDB keyword vocabulary audit

**What:** Extract and catalog ALL distinct `overall_keywords` from the scraped
IMDB data in `tracker.db`. Produce frequency counts, coverage patterns, and
a mapping of which keywords correspond to which deal-breaker concepts.

**Why:** The keyword-based deal-breaker filtering design (production medium
search, holiday themes, structural attributes) is blocked until we know the
actual keyword vocabulary. Determines whether static mapping, dynamic LLM
translation, or a hybrid approach is appropriate.

**How:** Query `imdb_data.overall_keywords` (JSON column) across all movies
in the tracker DB. Parse, deduplicate, count frequencies.

**Output:** Keyword vocabulary report + proposed concept→keyword mappings.

### 2. Source material type enum derivation

**What:** Evaluate ALL current `source_of_inspiration.source_material` values
from `generated_metadata` to determine the final enum taxonomy.

**Why:** The brainstorm doc proposed a draft enum (ORIGINAL_SCREENPLAY,
NOVEL_ADAPTATION, SHORT_STORY_ADAPTATION, TRUE_STORY, BIOGRAPHY,
COMIC_BOOK_ADAPTATION, VIDEO_GAME_ADAPTATION, REMAKE, STAGE_PLAY_ADAPTATION,
TV_ADAPTATION) but the final values must be validated against what the LLM
actually generated. There may be categories we missed or categories that
should be merged.

**How:** Extract all distinct `source_material` values from
`generated_metadata.source_of_inspiration`, cluster them, and finalize the
enum. Then re-generate with enum-constrained output.

**Output:** Finalized `SourceMaterialType` enum definition.

### 3. Country enum derivation

**What:** Create a `Country` enum based on the set of countries IMDB supports
for filtering, following the same pattern as the `Language` enum (int ID +
display name + normalized name).

**Why:** Required for `country_of_origin_ids` on movie_card and
`inv_country_origin_postings` in lexical schema.

**How:** Scrape or extract the IMDB country list. Cross-reference with the
`countries_of_origin` values already in our `imdb_data` table to ensure
coverage. Assign stable integer IDs alphabetically (same convention as Genre
and Language enums).

**Output:** `Country` enum in `implementation/classes/enums.py` (or dedicated
file like `languages.py`).

---

## New scraping targets

### 4. IMDB awards scraping

**What:** Scrape award nominations and wins from the IMDB GraphQL API for
all movies that pass quality filtering.

**Scope:** Major ceremonies only — Academy Awards, Golden Globes, BAFTA,
Cannes, Venice, Berlin, SAG, Critics Choice, Sundance.

**Per-movie output:**
```json
[
  {
    "ceremony": "Academy Awards",
    "category": "Best Picture",
    "outcome": "winner",
    "year": 2020
  },
  ...
]
```

**Storage:**
- Tracker DB: `imdb_data.awards` (JSON column) — raw scraped data
- Postgres: `movie_awards` table — structured for deterministic lookup
- Reception vector: generated prose summary appended to embedding text

**Pipeline integration:** New scraping step, likely after Stage 4 (IMDB
scraping) or as an extension of it. Needs to handle the same proxy/retry
infrastructure as existing IMDB scraping.

### 5. TMDB collection name capture

**What:** Capture `belongs_to_collection.name` from the TMDB detail API
response during Stage 2 (TMDB detail fetching).

**Why:** Input to the franchise LLM generation step. Currently not stored
anywhere in the pipeline.

**How:** Add `collection_name TEXT` column to `tmdb_data` table. Extract from
the TMDB detail response in `tmdb_fetcher.py`.

**Impact:** Requires a tracker DB migration (new column on tmdb_data) and a
code change to the TMDB fetcher extraction logic.

---

## New LLM generation tasks

### 6. Franchise generation

**What:** For each movie, generate `franchise_name`, `franchise_role`
(`FranchiseRole` enum: STARTER, MAINLINE, SPINOFF, PREBOOT, REMAKE), and
`culturally_recognized_group` (only when internet has established the term).

**Inputs to LLM:**
- Title
- Release year
- TMDB `collection_name` (from #5 above, may be null)
- Production companies
- Overall keywords
- Any other helpful context from TMDB/IMDB data

**LLM approach:** Pass structured input, generate structured output. The LLM
uses parametric knowledge to fill gaps that TMDB collection data doesn't cover
(spinoffs, brand-level groupings like MCU, etc.). `culturally_recognized_group`
must never be hallucinated — only used when established internet terminology
exists.

**Canonical naming convention:** The LLM is instructed to output the most
common, fully expanded form of the franchise name — no abbreviations. The
search extraction LLM follows the same convention (same pattern as the lexical
entity extractor for person names). This ensures both sides converge on the
same canonical string without needing alias tables.

**Output storage:**
- Tracker DB: `generated_metadata.franchise` (JSON)
- Postgres: `franchise_membership` table
- Lexical: `franchise_name_normalized` → `lex.lexical_dictionary` →
  `lex.inv_franchise_postings`

**Note:** Many movies won't have a franchise. The LLM should output null/empty
when a movie is standalone.

### 7. Source of inspiration re-generation

**What:** Re-generate `source_of_inspiration` metadata with enum-constrained
output instead of the current free-text `source_material` and
`franchise_lineage` fields.

**Why:** The current free-text output can't be reliably mapped to
`SourceMaterialType` enum IDs. Re-generation with enum constraints ensures
consistent, filterable output.

**Changes to generation:**
- `source_material` output becomes `source_material_types: SourceMaterialType[]`
  (enum values, not free text)
- `franchise_lineage` field is removed entirely (replaced by franchise
  generation in #6)

**Depends on:** #2 (enum derivation must be finalized first).

### 8. Production technique keyword re-generation

**What:** Re-generate the `production_keywords` metadata with a tightened
scope: only filming locations + production technique keywords.

**Tightened scope includes:**
- Visual techniques: black-and-white, IMAX, 3D, found-footage, single-take,
  handheld-camera
- Structural formats: anthology, vignette, nonlinear-timeline, mockumentary
- Production processes: stop-motion, rotoscope, practical-effects,
  motion-capture

**Tightened scope excludes (moved to structured fields):**
- Countries of origin → `country_of_origin_ids`
- Production companies → `inv_studio_postings`
- Languages → `audio_language_ids`
- Budget/revenue → `budget_bucket` / `box_office_bucket`
- Source material → `source_material_type_ids`
- Franchise/ecosystem → `franchise_membership`
- Decade/era → derivable from `release_ts`
- Animation/live action → keyword search

**Output:** Renamed column `generated_metadata.production_techniques` to
distinguish from the V1 `production_keywords`.

---

## New computed fields

### 9. Box office bucket calculation

**What:** Compute `box_office_bucket` (`HIT` / `FLOP` / null) for each movie
based on TMDB revenue data, era-adjusted using the same pattern as the
existing `budget_bucket`.

**Rules:**
- Movies < 75 days old → always `NULL` (too early to judge)
- Movies with no revenue data → `NULL`
- Era-adjusted thresholds (like budget_bucket) to account for inflation

**Depends on:** #5 (TMDB revenue capture — check if revenue is already
available in TMDB detail data; `has_revenue` exists on `tmdb_data` but actual
revenue value may not be stored).

### 10. Country of origin ID mapping

**What:** Map each movie's `imdb_data.countries_of_origin` (list of country
name strings) to `Country` enum IDs and store as `movie_card.country_of_origin_ids`.

**Depends on:** #3 (Country enum must exist first).

### 11. Keyword ID mapping

**What:** Map each movie's `imdb_data.overall_keywords` to
`lex.lexical_dictionary` string IDs and store as `movie_card.keyword_ids`.

**Depends on:** #1 (keyword vocabulary audit, to understand what we're working
with).

**Note:** Only `overall_keywords`, NOT `plot_keywords`.

---

## Embedding regeneration

### 12. Structured-label embedding format (all 7 vector spaces)

**What:** Convert all vector space embedding text from flat comma-separated
lists to section-labeled structured text. Regenerate all embeddings.

**Why:** Identified as the single highest-leverage change. Prerequisite for
cross-space rescoring to work effectively. Flat lists lose per-attribute
signal; structured labels preserve it.

**Scope:** All ~100K movies × 7 vector spaces (anchor dropped from V2).
Search subquery generation must also produce structured-format output.

**Cost:** 7 × 100K embedding API calls = 700K embeddings. Estimate before
executing.

### 13. Production vector re-embedding

**What:** After #8 (production technique re-generation), re-embed the
production vector with the tightened content.

**Open question:** After regeneration, is the thinned content (filming
locations + technique keywords only) enough to justify a dedicated vector
space? Options: keep as lean space, fold into anchor, or repurpose slot.

### 14. Reception vector re-embedding

**What:** After #4 (awards scraping), append generated awards prose summary
to reception vector embedding text and re-embed.

---

## New Postgres tables and schema changes

### 15. movie_awards table

```sql
CREATE TABLE IF NOT EXISTS public.movie_awards (
    movie_id    BIGINT NOT NULL REFERENCES movie_card,
    ceremony    TEXT NOT NULL,
    category    TEXT NOT NULL,
    outcome     TEXT NOT NULL,
    year        INT,
    PRIMARY KEY (movie_id, ceremony, category, year)
);
CREATE INDEX idx_awards_ceremony_outcome
    ON public.movie_awards (ceremony, outcome);
```

### 16. franchise_membership table

```sql
CREATE TABLE IF NOT EXISTS public.franchise_membership (
    movie_id                    BIGINT NOT NULL REFERENCES movie_card,
    franchise_name              TEXT NOT NULL,
    franchise_name_normalized   TEXT NOT NULL,
    culturally_recognized_group TEXT,
    franchise_role              TEXT NOT NULL,
    PRIMARY KEY (movie_id, franchise_name_normalized)
);
```

### 17. Role-specific person posting tables

Replace `lex.inv_person_postings` with:

```sql
CREATE TABLE IF NOT EXISTS lex.inv_actor_postings (
    term_id           BIGINT NOT NULL,
    movie_id          BIGINT NOT NULL,
    billing_position  INT NOT NULL,
    cast_size         INT NOT NULL,
    PRIMARY KEY (term_id, movie_id)
);
CREATE TABLE IF NOT EXISTS lex.inv_director_postings (
    term_id   BIGINT NOT NULL,
    movie_id  BIGINT NOT NULL,
    PRIMARY KEY (term_id, movie_id)
);
CREATE TABLE IF NOT EXISTS lex.inv_writer_postings (
    term_id   BIGINT NOT NULL,
    movie_id  BIGINT NOT NULL,
    PRIMARY KEY (term_id, movie_id)
);
CREATE TABLE IF NOT EXISTS lex.inv_producer_postings (
    term_id   BIGINT NOT NULL,
    movie_id  BIGINT NOT NULL,
    PRIMARY KEY (term_id, movie_id)
);
CREATE TABLE IF NOT EXISTS lex.inv_composer_postings (
    term_id   BIGINT NOT NULL,
    movie_id  BIGINT NOT NULL,
    PRIMARY KEY (term_id, movie_id)
);
```

### 18. Franchise posting table

```sql
CREATE TABLE IF NOT EXISTS lex.inv_franchise_postings (
    term_id   BIGINT NOT NULL,
    movie_id  BIGINT NOT NULL,
    PRIMARY KEY (term_id, movie_id)
);
```

### 19. New movie_card columns

```sql
ALTER TABLE public.movie_card
    ADD COLUMN IF NOT EXISTS country_of_origin_ids INT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS box_office_bucket TEXT,
    ADD COLUMN IF NOT EXISTS source_material_type_ids INT[] NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS keyword_ids INT[] NOT NULL DEFAULT '{}';

CREATE INDEX IF NOT EXISTS idx_movie_card_country_ids
    ON public.movie_card USING GIN (country_of_origin_ids gin__int_ops);
CREATE INDEX IF NOT EXISTS idx_movie_card_source_material
    ON public.movie_card USING GIN (source_material_type_ids gin__int_ops);
CREATE INDEX IF NOT EXISTS idx_movie_card_keyword_ids
    ON public.movie_card USING GIN (keyword_ids gin__int_ops);
```

### 20. New inverse lookup tables for country and source material

```sql
CREATE TABLE IF NOT EXISTS lex.inv_country_origin_postings (
    term_id   BIGINT NOT NULL,
    movie_id  BIGINT NOT NULL,
    PRIMARY KEY (term_id, movie_id)
);
CREATE TABLE IF NOT EXISTS lex.inv_source_material_postings (
    term_id   BIGINT NOT NULL,
    movie_id  BIGINT NOT NULL,
    PRIMARY KEY (term_id, movie_id)
);
```

### 21. New enums

```python
class Country(Enum):
    # TBD — derive from IMDB's supported country list
    # Pattern: country_id: int, value: str, normalized_name: str
    # Same structure as Genre and Language enums
    pass

class BoxOfficeBucket(Enum):
    HIT = "hit"
    FLOP = "flop"

class SourceMaterialType(Enum):
    # TBD — derive from current generated values (see #2)
    # Draft: ORIGINAL_SCREENPLAY, NOVEL_ADAPTATION, SHORT_STORY_ADAPTATION,
    # TRUE_STORY, BIOGRAPHY, COMIC_BOOK_ADAPTATION, VIDEO_GAME_ADAPTATION,
    # REMAKE, STAGE_PLAY_ADAPTATION, TV_ADAPTATION
    # Pattern: source_material_type_id: int, value: str
    pass

class FranchiseRole(Enum):
    STARTER = "starter"
    MAINLINE = "mainline"
    SPINOFF = "spinoff"
    PREBOOT = "preboot"
    REMAKE = "remake"
```

---

## Dependency graph

```
#1 Keyword audit ──────────────────────────────→ #11 Keyword ID mapping
#2 Source material enum derivation ────────────→ #7 Source of inspiration re-gen
#3 Country enum derivation ────────────────────→ #10 Country ID mapping
#5 TMDB collection capture ───────┬────────────→ #6 Franchise generation
                                  └────────────→ #9 Box office calculation
#4 Awards scraping ────────────────────────────→ #14 Reception re-embedding
#6 Franchise generation ───────────────────────→ #16 franchise_membership table
#7 Source of inspiration re-gen ───────────────→ #19 source_material_type_ids
#8 Production technique re-gen ────────────────→ #13 Production re-embedding
#12 Structured-label format ───────────────────→ All embedding regeneration

No dependencies (can start immediately):
  #3 Country enum
  #4 Awards scraping
  #5 TMDB collection capture
  #8 Production technique re-gen
  #15 movie_awards table (schema only)
  #16-20 Table/schema creation (schema only)
  #21 Enum definitions (except SourceMaterialType)
  #17 Role-specific postings (data already exists, just restructuring)
```
