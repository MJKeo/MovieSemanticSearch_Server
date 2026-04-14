# V2 Data Needs

Data that must be captured, generated, or restructured before V2 search can be
implemented. Organized by dependency order where possible.

---

## Prerequisites (unblocks other work)

### 1. IMDB keyword vocabulary audit — COMPLETED

**Status:** Finalized. See
[keyword_vocabulary_audit.md](keyword_vocabulary_audit.md) for the full
report with concept→keyword mappings.

**What was done:** Extracted and cataloged all `overall_keywords` and
`plot_keywords` from 109,238 qualifying movies in tracker.db.

**Key findings:**
- `overall_keywords` is a compact curated genre taxonomy of exactly 225
  terms (not a free-form tagging system). 100% of movies have at least
  one tag. Zero overlap with `plot_keywords`.
- `plot_keywords` is the free-form community system (114,547 terms) but
  its value is already absorbed by the metadata generation pipeline. No
  need to index it separately.
- **Static mapping is trivially feasible** — no LLM translation or
  hybrid approach needed. The full vocabulary can be provided as context
  to the QU LLM for user query→keyword mapping.
- Primary deal-breaker value: sub-genre precision across the entire
  genre space (16 horror sub-types, 17 comedy sub-types, etc.) — exactly
  the deterministic signal that vector search is weakest at.
- Language/nationality tags (30 terms) complement structured
  `country_of_origin_ids` / `audio_language_ids` fields.
- Holiday tags capture curated editorial judgment ("this is a holiday
  movie") including non-obvious cases (Die Hard, Harry Potter).

**Output:** Full vocabulary report + concept→keyword mappings in
[keyword_vocabulary_audit.md](keyword_vocabulary_audit.md).

### 2. Source material type enum derivation — COMPLETED

**Status:** Finalized. See
[source_material_type_enum.md](source_material_type_enum.md) for the full
enum definition with boundary notes, encompassed values, and re-generation
guidance.

**What was done:** Extracted all 4,311 unique free-text `source_material`
values from `generated_metadata.source_of_inspiration`, analyzed the top 100
by movie count, and clustered them into 10 distinct enum values:
```
NOVEL_ADAPTATION, SHORT_STORY_ADAPTATION, TRUE_STORY, BIOGRAPHY,
COMIC_ADAPTATION, FOLKLORE_ADAPTATION, STAGE_ADAPTATION,
VIDEO_GAME_ADAPTATION, REMAKE, TV_ADAPTATION
```
`ORIGINAL_SCREENPLAY` was removed during implementation — original screenplays
are identified by an empty `source_material_type_ids` array, not an explicit
enum value. Queries for "original screenplays" filter for empty arrays.

**Key changes from the draft enum:**
- Added `FOLKLORE_ADAPTATION` (~560 movies: fairy tales, mythology, religious
  texts, legends) — was a gap in the original proposal
- Renamed `COMIC_BOOK_ADAPTATION` → `COMIC_ADAPTATION` (includes manga,
  manhwa, graphic novels, comic strips)
- Renamed `STAGE_PLAY_ADAPTATION` → `STAGE_ADAPTATION` (includes opera,
  ballet, musicals — not just plays)
- Songs, toys, spinoffs, reboots, documentary, and parody were deliberately
  excluded with documented reasoning

**Output:** Finalized `SourceMaterialType` enum definition in
[source_material_type_enum.md](source_material_type_enum.md).

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

**Scope:** 12 major ceremonies — Academy Awards, Golden Globes, BAFTA,
Cannes, Venice, Berlin, SAG, Critics Choice, Sundance, Razzie Awards,
Film Independent Spirit Awards, Gotham Awards.

**GraphQL source:** `awardNominations(first: 500)` field on the `title` type.
Movies can have 300-570+ total nominations across all award bodies, but
filtering to the 12 in-scope ceremonies produces ~10-50 rows per movie.

**GraphQL response structure per nomination:**
```json
{
  "award": {
    "text": "Oscar",
    "event": { "text": "Academy Awards, USA" },
    "year": 2020
  },
  "isWinner": true,
  "category": { "text": "Best Picture" }
}
```

**Field mapping:**
- `ceremony` ← `award.event.text` (filtered to 12 known values)
- `category` ← `category.text` (nullable — festival grand prizes like
  Palme d'Or, Golden Lion have no category; the award name IS the category)
- `outcome` ← `isWinner` → `"winner"` / `"nominee"`
- `year` ← `award.year`

**IMDB `event.text` → ceremony mapping:**

| `event.text` | Ceremony |
|--------------|----------|
| `Academy Awards, USA` | Academy Awards |
| `Golden Globes, USA` | Golden Globes |
| `BAFTA Awards` | BAFTA |
| `Cannes Film Festival` | Cannes |
| `Venice Film Festival` | Venice |
| `Berlin International Film Festival` | Berlin |
| `Actor Awards` | SAG |
| `Critics Choice Awards` | Critics Choice |
| `Sundance Film Festival` | Sundance |
| `Razzie Awards` | Razzie Awards |
| `Film Independent Spirit Awards` | Spirit Awards |
| `Gotham Awards` | Gotham Awards |

**Per-movie output:**
```json
[
  {
    "ceremony": "Academy Awards, USA",
    "category": "Best Picture",
    "outcome": "winner",
    "year": 2020
  },
  {
    "ceremony": "Cannes Film Festival",
    "category": null,
    "outcome": "winner",
    "year": 2019
  },
  ...
]
```

**Storage:**
- Tracker DB: `imdb_data.awards` (JSON column) — raw scraped data
- Postgres: `movie_awards` table — structured for deterministic lookup
- Reception vector: generated prose summary appended to embedding text

**Pipeline integration:** Extension of Stage 4 (IMDB scraping). The
`awardNominations` field can be added to the existing GraphQL query in
`http_client.py` — no separate request needed. Filtering to the 12
ceremonies happens in the parser.

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

**What:** For each movie, generate the finalized `FranchiseOutput` shape:
`lineage`, `shared_universe`, `recognized_subgroups`, `launched_subgroup`,
`lineage_position`, `is_spinoff`, `is_crossover`, and `launched_franchise`.

**Franchise definition:** Any recognizable intellectual property or brand that
originated in any medium — film series, video games, toys, books, comics, TV
shows, board games, theme parks, etc. — where the movie is an adaptation,
extension, or product of that IP. Examples: "Mario" (video game), "Barbie"
(toy), "Transformers" (toy/cartoon), "Harry Potter" (book), "Marvel Cinematic
Universe" (comic/film). The franchise name should be the **IP name**, not the
film series name.

**Inputs to LLM:**
- Title
- Release year
- Overview (as an identification aid — helps the LLM correctly identify which
  movie this is; NOT for inferring franchise from plot similarity)
- TMDB `collection_name` (from #5 above, may be null)
- Production companies
- Overall keywords
- Characters

This is a compact, high-signal input set with no generated-metadata
dependencies. The overview was included because title + year alone can be
ambiguous (multiple movies share titles), and the overview helps the LLM
confidently access its parametric knowledge about franchise membership.
Characters are critical for IP-based franchises (e.g., "Mario", "Luigi" →
Mario franchise; "Optimus Prime" → Transformers).

**LLM approach:** Pass structured input, generate structured output. The LLM
uses parametric knowledge to fill gaps that TMDB collection data doesn't cover
(spinoffs, brand-level groupings like MCU, real-world IP franchises, etc.).
`recognized_subgroups` must never be hallucinated — only used when established
terminology exists in any market globally. If multiple names exist across
markets for the same grouping, prefer the American-market term.

**Storage simplification:** Named identity fields (`lineage`,
`shared_universe`, subgroup labels) are stored normalized. Display-form names
can be derived at the UI layer if ever needed.

**Canonical naming convention:** The LLM is instructed to output the most
common, fully expanded form of the franchise/IP name — no abbreviations. The
search extraction LLM follows the same convention (same pattern as the lexical
entity extractor for person names). This ensures both sides converge on the
same canonical string without needing alias tables.

**Output storage:**
- Tracker DB: `generated_metadata.franchise` (JSON)
- Postgres: `movie_franchise_metadata` table
- Lexical: `lineage` and `shared_universe` → `lex.lexical_dictionary` →
  `lex.inv_franchise_postings`

**Note:** Many movies won't have a franchise. The LLM should output null/empty
when a movie is standalone.

### 7. Source of inspiration re-generation

**What:** Re-generate `source_of_inspiration` metadata with enum-constrained
output instead of the current free-text `source_material` and
`franchise_lineage` fields.

**Why:** The current free-text output produced 4,311 unique strings with
massive duplication (e.g., "based on true events" vs "based on a true story")
and concept leakage (spinoff/reboot appearing in source_material). Can't be
reliably mapped to enum IDs. Re-generation with enum constraints ensures
consistent, filterable output.

**Changes to generation:**
- `source_material` output becomes `source_material_types: SourceMaterialType[]`
  (array of enum values, not free text). Must remain an array — movies
  frequently have multiple applicable types (e.g., NOVEL_ADAPTATION +
  TRUE_STORY).
- `franchise_lineage` field is removed entirely (replaced by franchise
  generation in #6)
- The LLM prompt must include the full enum definitions with boundary notes
  from [source_material_type_enum.md](source_material_type_enum.md) so the
  model understands each category's scope and exclusions

**Depends on:** #2 (enum derivation — now complete).

### 8. Production techniques generation

**What:** Generate `production_techniques` metadata with a tightened scope:
only concrete production-technique terms. Filming locations remain scraped
raw data and are paired with these terms in the production vector.

**Status clarification:** This work remains in scope. It is NOT superseded by
concept tags. Concept tags answer binary content questions like "does this
movie have X?" for story, character, ending, and experiential deal-breakers.
Production techniques serve a different purpose: they capture real-world
making-of signals like black-and-white, single-take, stop-motion, and
practical-effects. Those are production-context retrieval signals, not the
same kind of binary movie-concept filter.

**Relationship to scraped data:** the generation step is for production
technique terms only. Scraped filming locations remain a separate raw input
and pair with those generated technique terms in the production vector.

**Tightened scope includes:**
- Animation modalities/sub-techniques: hand-drawn animation, 2d animation,
  3d animation, traditional animation, computer animation, cgi animation,
  stop-motion, rotoscope, motion-capture, hybrid/partial animation labels
- Visual capture/rendering techniques: black-and-white, 3d, single-take,
  long take, handheld-camera
- Special exception: found-footage

**Tightened scope excludes (moved to structured fields):**
- Countries of origin → `country_of_origin_ids`
- Production companies → `inv_studio_postings`
- Languages → `audio_language_ids`
- Budget/revenue → `budget_bucket` / `box_office_bucket`
- Source material → `source_material_type_ids`
- Franchise/ecosystem → `movie_franchise_metadata`
- Decade/era → derivable from `release_ts`
- Animation/live action → keyword search
- IMAX, anthology, vignette, mockumentary, nonlinear timeline → not part of
  the finalized production-techniques schema

**Output:** Renamed column `generated_metadata.production_techniques` to
distinguish from the V1 `production_keywords`.

---

## New computed fields

### 9. Box office bucket calculation

**What:** Compute `box_office_bucket` (`HIT` / `FLOP` / null) for each movie
based on IMDB box office data, era-adjusted using the same pattern as the
existing `budget_bucket`.

**Data source:** IMDB GraphQL API — `lifetimeGross(boxOfficeArea: ...)` field
on the `title` type. Two variants:
- `lifetimeGross(boxOfficeArea: DOMESTIC)` — US + Canada + Puerto Rico
- `lifetimeGross(boxOfficeArea: WORLDWIDE)` — domestic + international (inclusive)

Both return `{ total: { amount: int, currency: str } }` or `null`. Amount is
whole dollars (not cents). Also available:
`openingWeekendGross(boxOfficeArea: DOMESTIC)` with `{ gross: { total: ... }, weekendEndDate }`.

**Worldwide is inclusive of domestic** — verified via Box Office Mojo glossary.
International-only = worldwide - domestic.

**Rules:**
- Movies < 75 days old → always `NULL` (too early to judge)
- Movies with no revenue data → `NULL`
- Era-adjusted thresholds (like budget_bucket) to account for inflation

**Pipeline integration:** Box office fields can be added to the existing IMDB
GraphQL query alongside awards data. No separate request needed.

**Depends on:** IMDB box office scraping (extension of Stage 4). Previously
depended on #5 (TMDB revenue) but IMDB is the better source — more complete
data and already fetched via GraphQL.

### 10. Country of origin ID mapping

**What:** Map each movie's `imdb_data.countries_of_origin` (list of country
name strings) to `Country` enum IDs and store as `movie_card.country_of_origin_ids`.

**Depends on:** #3 (Country enum must exist first).

### 11. Keyword ID mapping

**What:** Map each movie's `imdb_data.overall_keywords` to
`lex.lexical_dictionary` string IDs and store as `movie_card.keyword_ids`.

**Depends on:** #1 (keyword vocabulary audit — now complete). Audit
confirmed: only 225 distinct terms, static mapping is trivial. See
[keyword_vocabulary_audit.md](keyword_vocabulary_audit.md).

**Note:** Only `overall_keywords`, NOT `plot_keywords`. Audit confirmed
`plot_keywords` value is already absorbed by the metadata generation
pipeline.

---

## Embedding regeneration

### 12. Structured-label embedding format (all 8 vector spaces)

**What:** Convert all vector space embedding text from flat comma-separated
lists to section-labeled structured text. Regenerate all embeddings.

**Why:** Identified as the single highest-leverage change. Prerequisite for
cross-space rescoring to work effectively. Flat lists lose per-attribute
signal; structured labels preserve it.

**Scope:** All ~100K movies × 8 vector spaces. Anchor is retained in V2 in a
reduced labeled form alongside the 7 specialized spaces.
Search subquery generation must also produce structured-format output.

**Cost:** 8 × 100K embedding API calls = 800K embeddings. Estimate before
executing.

**Model choice (decided 2026-04-10):** Upgrade from `text-embedding-3-small`
to OpenAI `text-embedding-3-large` as part of this re-embed. Rationale:
lowest-friction upgrade path (same provider, same SDK, same batch API,
Matryoshka-compatible so Qdrant collection dims can stay at 1536 if desired),
modest but real MTEB retrieval lift over 3-small, and minimal new vendor risk.
The structured-label format fix removes one major compensation layer in the
architecture, so the embedder's raw quality becomes more visible — worth a
cheap upgrade while we're already rebuilding. Cost delta (~$0.02 → $0.13 per
M tokens) is trivial at this scale.

**Fallback plan:** If retrieval quality evaluation after this migration still
shows embedding-quality-attributable failures (known-relevant movies not
ranking despite correct metadata, correct format, and aligned queries), switch
to Voyage-3-large. Migration path is tractable: centralize model string in
`generate_vector_embedding`, add a Voyage client branch, rebuild Qdrant at
1024 dims, re-embed via Voyage Batch API. Redis embedding cache is keyed by
model so old/new coexist safely. Do not switch preemptively — wait for eval
signal that the embedder is the next bottleneck.

### 13. Production vector re-embedding

**What:** After #8 (production technique re-generation), re-embed the
production vector with the tightened content.

**Open question:** After regeneration, is the thinned content (filming
locations + technique keywords only) enough to justify a dedicated vector
space? Options: keep as lean space, fold into anchor, or repurpose slot.

### 14. Reception vector re-embedding

**What:** After #4 (awards scraping), append deterministic `major_award_wins`
ceremony summary text to reception vector embedding text and re-embed.

**Definition:** Use winners only, collapse to distinct ceremony names in fixed
priority order, and exclude nominations from the vector entirely. Precise
award/category/nominee queries stay deterministic via `movie_awards`.

---

## New Postgres tables and schema changes

### 15. movie_awards table

```sql
CREATE TABLE IF NOT EXISTS public.movie_awards (
    movie_id    BIGINT NOT NULL REFERENCES movie_card,
    ceremony    TEXT NOT NULL,
    award_name  TEXT NOT NULL,     -- specific prize name ("Oscar", "Palme d'Or", etc.)
    category    TEXT,
    outcome     TEXT NOT NULL,
    year        INT,
    PRIMARY KEY (movie_id, ceremony, award_name, COALESCE(category, ''), year)
);
CREATE INDEX idx_awards_ceremony_outcome
    ON public.movie_awards (ceremony, outcome);
```

### 16. movie_franchise_metadata table

```sql
CREATE TABLE IF NOT EXISTS public.movie_franchise_metadata (
    movie_id               BIGINT PRIMARY KEY REFERENCES movie_card,
    lineage                TEXT,
    shared_universe        TEXT,
    recognized_subgroups   TEXT[] NOT NULL DEFAULT '{}',
    launched_subgroup      BOOLEAN NOT NULL DEFAULT FALSE,
    lineage_position       SMALLINT,
    is_spinoff             BOOLEAN NOT NULL DEFAULT FALSE,
    is_crossover           BOOLEAN NOT NULL DEFAULT FALSE,
    launched_franchise     BOOLEAN NOT NULL DEFAULT FALSE
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

CREATE INDEX IF NOT EXISTS idx_movie_card_country_of_origin_ids
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

class SourceMaterialType(str, Enum):
    # Finalized — 10 values. Empty array = original screenplay.
    # NOVEL_ADAPTATION, SHORT_STORY_ADAPTATION, TRUE_STORY, BIOGRAPHY,
    # COMIC_ADAPTATION, FOLKLORE_ADAPTATION, STAGE_ADAPTATION,
    # VIDEO_GAME_ADAPTATION, REMAKE, TV_ADAPTATION
    # Pattern: (str value, source_material_type_id: int)
    # Implemented in schemas/enums.py
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
#1 Keyword audit (DONE) ───────────────────────→ #11 Keyword ID mapping
#2 Source material enum derivation ────────────→ #7 Source of inspiration re-gen
#3 Country enum derivation ────────────────────→ #10 Country ID mapping
#5 TMDB collection capture ───────┬────────────→ #6 Franchise generation
                                  └────────────→ #9 Box office calculation
#4 Awards scraping ────────────────────────────→ #14 Reception re-embedding
#6 Franchise generation ───────────────────────→ #16 movie_franchise_metadata table
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
