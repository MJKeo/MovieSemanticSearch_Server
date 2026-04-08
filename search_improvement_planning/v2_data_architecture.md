# V2 Data Architecture

Complete inventory of all data in the V2 search system. Organized by storage
location and access pattern.

---

## 1. Postgres: movie_card (display + filtering)

Canonical per-movie metadata. Used for UI display, metadata preference scoring,
and hard filtering via GIN-indexed array overlap.

### Existing columns (unchanged)

| Column | Type | Description |
|--------|------|-------------|
| `movie_id` | `BIGINT` PK | TMDB movie ID. Universal join key. |
| `title` | `TEXT` | Display title |
| `poster_url` | `TEXT?` | Poster image URL |
| `release_ts` | `BIGINT?` | Unix timestamp (seconds) of release date |
| `runtime_minutes` | `INT?` | Duration in minutes |
| `maturity_rank` | `SMALLINT?` | Ordinal 1-5 for G/PG/PG-13/R/NC-17, 999=UNRATED |
| `genre_ids` | `INT[]` | Genre enum IDs (1-27). GIN `gin__int_ops` index. |
| `watch_offer_keys` | `INT[]` | Encoded `(provider_id << 2) \| method_id`. GIN index. |
| `audio_language_ids` | `INT[]` | Language enum IDs (1-334). GIN index. |
| `imdb_vote_count` | `INT` | Raw IMDB vote count |
| `popularity_score` | `FLOAT` | Sigmoid-transformed percentile from vote count [0,1] |
| `reception_score` | `FLOAT?` | 0-100 composite: 40% IMDB (scaled) + 60% Metacritic |
| `budget_bucket` | `TEXT?` | `"small"` / `"large"` / `NULL`. Era-adjusted. |
| `title_token_count` | `INT` | Token count of normalized title (for title search F-scoring) |
| `created_at` | `TIMESTAMP` | Record creation time |
| `updated_at` | `TIMESTAMP` | Last update time |

### New columns (V2)

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `country_of_origin_ids` | `INT[]` | IMDB `countries_of_origin` mapped to `Country` enum | GIN index. Postgres-only, NOT in Qdrant payload. Hard filter for "Korean movies" etc. |
| `box_office_bucket` | `TEXT?` | IMDB box office + era adjustment | `"hit"` / `"flop"` / `NULL`. Same pattern as budget_bucket. Movies < 75 days old always `NULL` (too early to judge). Source: IMDB GraphQL `lifetimeGross(boxOfficeArea: DOMESTIC)` and `lifetimeGross(boxOfficeArea: WORLDWIDE)`. Worldwide is inclusive of domestic. |
| `source_material_type_ids` | `INT[]` | LLM source_of_inspiration (re-generated with enum) | `SourceMaterialType` enum IDs (11 values, finalized — see [source_material_type_enum.md](source_material_type_enum.md)). Array because movies can have multiple (e.g. Schindler's List = NOVEL_ADAPTATION + TRUE_STORY). GIN index. |
| `keyword_ids` | `INT[]` | IMDB `overall_keywords` mapped to `lex.lexical_dictionary` IDs | GIN index. Used for keyword-based deal-breaker boost. Only `overall_keywords`, NOT `plot_keywords`. |

### Indexes

Existing: range index on `(release_ts, runtime_minutes, maturity_rank)`, individual
B-tree indexes on each of those three, GIN `gin__int_ops` on `genre_ids`,
`watch_offer_keys`, `audio_language_ids`.

New: GIN `gin__int_ops` on `country_of_origin_ids`, `source_material_type_ids`,
`keyword_ids`.

---

## 2. Postgres: movie_awards (structured award lookup)

New table. Stores award nominations and wins for deterministic retrieval. Designed
for inverse lookup: given an award, find movies.

```sql
movie_awards (
    movie_id    BIGINT NOT NULL REFERENCES movie_card,
    ceremony    TEXT NOT NULL,
    award_name  TEXT NOT NULL,     -- specific prize name (e.g., "Oscar", "Palme d'Or", "Golden Lion")
    category    TEXT,              -- nullable: festival grand prizes have no category
    outcome     TEXT NOT NULL,     -- "winner" | "nominee"
    year        INT,               -- ceremony year
    PRIMARY KEY (movie_id, ceremony, award_name, COALESCE(category, ''), year)
)
```

**Note on `award_name`:** The specific prize name users search for — "Oscar",
"Palme d'Or", "Golden Lion", "Golden Globe", etc. Distinct from `ceremony`
(the organization) and `category` (the specific category within the prize).
A single ceremony can give out differently-named awards (e.g., BAFTA gives
both "BAFTA Film Award" and "David Lean Award for Direction").

**Note on nullable `category`:** Festival grand prizes (Palme d'Or, Golden Lion,
Golden Bear, etc.) have no category — the award name IS the category. These
return `null` from the IMDB GraphQL API. The PK uses `COALESCE(category, '')`
to handle this.

**Index:** `idx_awards_ceremony_outcome (ceremony, outcome)` for queries like
"Oscar winners", "Cannes Palme d'Or nominees".

**Ceremonies in scope (12):** Academy Awards, Golden Globes, BAFTA, Cannes,
Venice, Berlin, SAG, Critics Choice, Sundance, Razzie Awards, Film Independent
Spirit Awards, Gotham Awards.

**IMDB GraphQL `event.text` mapping:**

| Ceremony | `event.text` value |
|----------|-------------------|
| Academy Awards | `"Academy Awards, USA"` |
| Golden Globes | `"Golden Globes, USA"` |
| BAFTA | `"BAFTA Awards"` |
| Cannes | `"Cannes Film Festival"` |
| Venice | `"Venice Film Festival"` |
| Berlin | `"Berlin International Film Festival"` |
| SAG | `"Actor Awards"` |
| Critics Choice | `"Critics Choice Awards"` |
| Sundance | `"Sundance Film Festival"` |
| Razzie Awards | `"Razzie Awards"` |
| Spirit Awards | `"Film Independent Spirit Awards"` |
| Gotham Awards | `"Gotham Awards"` |

After filtering to these 12 ceremonies, per-movie row counts are ~10-50 (vs
300-570 total nominations across all regional critics circles).

**Also embedded in reception vector** as generated prose summary for semantic
queries like "award-winning thriller."

---

## 3. Postgres: franchise_membership (structured franchise data)

New table. Replaces the current title-token + character-matching franchise
heuristic in lexical search.

```sql
franchise_membership (
    movie_id                    BIGINT NOT NULL REFERENCES movie_card,
    franchise_name              TEXT NOT NULL,
    franchise_name_normalized   TEXT NOT NULL,
    culturally_recognized_group TEXT,
    franchise_role              TEXT NOT NULL,
    PRIMARY KEY (movie_id, franchise_name_normalized)
)
```

**Fields:**
- `franchise_name` — display name ("Star Wars", "Marvel Cinematic Universe")
- `franchise_name_normalized` — `normalize_string()` applied, for matching
- `culturally_recognized_group` — only when internet has established the term
  (e.g. "original trilogy", "MCU Phase 1"). Never hallucinated by LLM.
- `franchise_role` — `FranchiseRole` enum value stored as integer ordinal:
  `STARTER`, `MAINLINE`, `SPINOFF`, `PREBOOT`, `REMAKE`. The search extraction
  LLM receives the same enum definition for consistent output.

**Data sources:**
1. TMDB `belongs_to_collection` — reliable base for ~25% of movies
2. LLM enrichment — receives title, year, TMDB collection name (if any),
   production companies, keywords. Generates franchise_name, franchise_role,
   and culturally_recognized_group using parametric knowledge.

**Canonical naming convention:** The franchise generation LLM is instructed to
output the most common, fully expanded form of the franchise name — no
abbreviations. The search extraction LLM follows the same convention (same
pattern as the lexical entity extractor for person names). This ensures both
sides converge on the same canonical string without needing alias tables.

**Lexical access:** `franchise_name_normalized` is inserted into
`lex.lexical_dictionary` and a new `lex.inv_franchise_postings` table maps
`term_id → movie_id` for text-based franchise lookup.

**Search strategy:**
- `franchise_name` — trigram matching via lexical dictionary. Both LLMs use
  the same canonical naming convention, so no enum or alias table needed.
- `franchise_role` — integer WHERE clause on the post-lookup result set.
- `culturally_recognized_group` — trigram similarity on the post-franchise-lookup
  result set (3-30 movies). No separate index needed at this scale.

---

## 4. Postgres Lexical Schema (lex)

Inverted index for entity-based text search. Uses trigram matching on
dictionary tables to resolve fuzzy user input to term IDs, then posting
tables to find matching movies.

### Dictionaries (shared across V1 and V2)

| Table | Purpose |
|-------|---------|
| `lex.lexical_dictionary` | Master dictionary: `string_id BIGINT PK`, `norm_str TEXT UNIQUE`. Used for person names, studio names, franchise names, keyword strings. |
| `lex.title_token_strings` | Normalized title tokens. `string_id → norm_str`. Trigram GIN index. |
| `lex.character_strings` | Normalized character names. `string_id → norm_str`. Trigram GIN index. |

### Posting tables: V1 (current)

| Table | Columns | Notes |
|-------|---------|-------|
| `lex.inv_person_postings` | `term_id, movie_id` | All roles in one table, binary presence |
| `lex.inv_character_postings` | `term_id, movie_id` | |
| `lex.inv_studio_postings` | `term_id, movie_id` | |
| `lex.inv_title_token_postings` | `term_id, movie_id` | |

### Posting tables: V2 (replacing inv_person_postings, adding franchise)

| Table | Columns | Notes |
|-------|---------|-------|
| `lex.inv_actor_postings` | `term_id, movie_id, billing_position INT, cast_size INT` | Enables prominence scoring |
| `lex.inv_director_postings` | `term_id, movie_id` | |
| `lex.inv_writer_postings` | `term_id, movie_id` | |
| `lex.inv_producer_postings` | `term_id, movie_id` | |
| `lex.inv_composer_postings` | `term_id, movie_id` | |
| `lex.inv_franchise_postings` | `term_id, movie_id` | Franchise name from `franchise_membership` |
| `lex.inv_character_postings` | `term_id, movie_id` | Unchanged |
| `lex.inv_studio_postings` | `term_id, movie_id` | Unchanged |
| `lex.inv_title_token_postings` | `term_id, movie_id` | Unchanged |
| `lex.inv_country_origin_postings` | `term_id, movie_id` | Country name from `Country` enum |
| `lex.inv_source_material_postings` | `term_id, movie_id` | Source material type from `SourceMaterialType` enum |

### Materialized views

| View | Purpose |
|------|---------|
| `lex.title_token_doc_frequency` | `term_id → doc_frequency` for max-df stop-word filtering |

### Actor prominence scoring modes (Phase 0 determines)

1. **Exclude non-major** — top min(2-3, 10-15% of `cast_size`). "Starring" language.
2. **Boost by position** — `1.0 - (position / cast_size)`. Default.
3. **Binary** — all actors equal. Fallback when data missing.
4. **Reverse** — boost deep credits. "Minor roles" queries.

### Entity extraction categories

| Category | V1 Lookup | V2 Lookup |
|----------|-----------|-----------|
| `PERSON` | `inv_person_postings` | Role-specific tables; `role_hint` field routes to correct table(s) |
| `CHARACTER` | `inv_character_postings` | Unchanged |
| `STUDIO` | `inv_studio_postings` | Unchanged |
| `MOVIE_TITLE` | `inv_title_token_postings` | Unchanged |
| `FRANCHISE` | Title tokens + character heuristic | `inv_franchise_postings` via `franchise_membership` |

---

## 5. Metadata Filters: Hard (UI-set)

Applied as Qdrant payload conditions and/or Postgres WHERE clauses. Non-negotiable.
Results failing any hard filter are excluded entirely.

### V1 (current) — MetadataFilters dataclass

| Field | Type | Qdrant Condition |
|-------|------|------------------|
| `min_release_ts` | `INT?` | `release_ts >= value` |
| `max_release_ts` | `INT?` | `release_ts <= value` |
| `min_runtime` | `INT?` | `runtime_minutes >= value` |
| `max_runtime` | `INT?` | `runtime_minutes <= value` |
| `min_maturity_rank` | `INT?` | `maturity_rank >= value` |
| `max_maturity_rank` | `INT?` | `maturity_rank <= value` |
| `genres` | `Genre[]?` | Any `genre_id` in list (OR) |
| `audio_languages` | `Language[]?` | Any `language_id` in list (OR) |
| `watch_offer_keys` | `INT[]?` | Any key in list (OR) |

### V2 additions

| Field | Type | Filter Location |
|-------|------|-----------------|
| `countries` | `Country[]?` | Postgres-only (pre-filter, pass IDs to Qdrant) |
| `source_material_types` | `SourceMaterialType[]?` | Postgres array overlap |
| `keywords` | `INT[]?` | Postgres array overlap (deal-breaker boost, not hard pre-filter) |

### Qdrant payload (stored per-point)

V1 and V2 identical — country/source_material/keyword filtering happens in
Postgres before Qdrant, not in Qdrant payload:

```json
{
  "movie_id": int,
  "release_ts": int | null,
  "runtime_minutes": int | null,
  "maturity_rank": int | null,
  "genre_ids": [int],
  "watch_offer_keys": [int],
  "audio_language_ids": [int]
}
```

### V2 three-tier constraint strictness

| Tier | Source | Behavior |
|------|--------|----------|
| Hard filter | UI controls | Strict WHERE clause. Overrides LLM inferences. |
| Soft constraint | NLP-extracted | Generous gate + preference decay. Attribute softness varies (dates soft, named entities hard). |
| Semantic constraint | Vector similarity | Threshold + flatten for deal-breakers. |

---

## 6. Metadata Preferences: Soft (LLM-inferred)

Preferences extracted from natural language by the query understanding LLM.
Scored as weighted average against movie_card data. Range [0,1] per candidate.

### Preference weights

| Preference | Weight | LLM Fields | Score Logic |
|------------|--------|------------|-------------|
| **Genres** | 5 | `should_include: Genre[]`, `should_exclude: Genre[]` | Exclusion → -2.0; inclusion → fraction matched |
| **Release Date** | 4 | `first_date`, `second_date?`, `match_op: EXACT\|BEFORE\|AFTER\|BETWEEN` | Linear decay from range, 1-5yr grace |
| **Watch Providers** | 4 | `should_include: StreamingService[]`, `should_exclude: StreamingService[]`, `preferred_access_type: SUBSCRIPTION\|BUY\|RENT?` | Desired method = 1.0; any method = 0.5 |
| **Audio Language** | 3 | `should_include: Language[]`, `should_exclude: Language[]` | Exclusion → -2.0; any included present → 1.0 |
| **Maturity Rating** | 3 | `rating: MaturityRating`, `match_op: EXACT\|GT\|LT\|GTE\|LTE` | In range → 1.0; 1 rank away → 0.5 |
| **Reception** | 3 | `reception_type: CRITICALLY_ACCLAIMED\|POORLY_RECEIVED\|NO_PREFERENCE` | Linear ramp on reception_score |
| **Budget Size** | 3 | `budget_size: SMALL\|LARGE\|NO_PREFERENCE` | Binary match vs budget_bucket |
| **Duration** | 2 | `first_value`, `second_value?`, `match_op: EXACT\|BETWEEN\|LT\|GT` | Linear decay, 30min grace |
| **Trending** | 2 | `prefers_trending_movies: bool` | Redis trending score pass-through |
| **Popular** | 2 | `prefers_popular_movies: bool` | popularity_score pass-through |

**Formula:** `metadata_score = sum(weight_i * score_i) / sum(active weights)`

### V2 scoring function modes (Phase 0 determines per-attribute)

| Mode | Use Case | Behavior |
|------|----------|----------|
| Threshold+flatten | Deal-breakers | >= threshold → 1.0; below → decay |
| Preserved similarity | Superlatives ("scariest") | Raw cosine similarity preserved |
| Diminishing returns | Preferences | Additive but compressed |
| Sort-by | Ranking axes ("most popular") | Used as primary sort key |

---

## 7. Enums

### Existing (unchanged)

| Enum | Values | ID Field |
|------|--------|----------|
| `Genre` | 27 values: ACTION(1) through WESTERN(27) | `genre_id` |
| `MaturityRating` | G(1), PG(2), PG-13(3), R(4), NC-17(5), UNRATED(999) | `maturity_rank` |
| `StreamingAccessType` | SUBSCRIPTION(1), BUY(2), RENT(3) | `type_id` |
| `StreamingService` | 20 values: NETFLIX through VUDU | string-valued |
| `Language` | 334 values: ABKHAZIAN(1) through ZULU(334) | `language_id` |
| `VectorName` | ANCHOR, PLOT_EVENTS, PLOT_ANALYSIS, VIEWER_EXPERIENCE, WATCH_CONTEXT, NARRATIVE_TECHNIQUES, PRODUCTION, RECEPTION | string-valued |
| `VectorCollectionName` | 8 Qdrant collection names | string-valued |
| `RelevanceSize` | NOT_RELEVANT, SMALL, MEDIUM, LARGE | string-valued |
| `EntityCategory` | MOVIE_TITLE, PERSON, CHARACTER, FRANCHISE, STUDIO | string-valued |
| `DateMatchOperation` | EXACT, BEFORE, AFTER, BETWEEN | string-valued |
| `NumericalMatchOperation` | EXACT, BETWEEN, LESS_THAN, GREATER_THAN | string-valued |
| `RatingMatchOperation` | EXACT, GT, LT, GTE, LTE | string-valued |
| `ReceptionType` | CRITICALLY_ACCLAIMED, POORLY_RECEIVED, NO_PREFERENCE | string-valued |
| `BudgetSize` | SMALL, LARGE, NO_PREFERENCE | string-valued |

### New (V2)

| Enum | Values | ID Field | Notes |
|------|--------|----------|-------|
| `Country` | TBD — derive from IMDB's country list | `country_id: int` | Same pattern as Language enum. ~100-200 values. |
| `BoxOfficeBucket` | HIT, FLOP | string-valued | Movies < 75 days old → NULL (not enough data). |
| `SourceMaterialType` | TBD — derive from current generated source_of_inspiration values | `source_material_type_id: int` | Brainstorm draft: ORIGINAL_SCREENPLAY, NOVEL_ADAPTATION, SHORT_STORY_ADAPTATION, TRUE_STORY, BIOGRAPHY, COMIC_BOOK_ADAPTATION, VIDEO_GAME_ADAPTATION, REMAKE, STAGE_PLAY_ADAPTATION, TV_ADAPTATION. Final values must be validated against actual generated data. |
| `FranchiseRole` | STARTER, MAINLINE, SPINOFF, PREBOOT, REMAKE | string-valued | Stored on `franchise_membership.franchise_role`. |

---

## 8. Vector Spaces

V2 has 7 spaces (anchor dropped). OpenAI `text-embedding-3-small`, 1536 dims,
8191 token limit. Stored in Qdrant with scalar quantization + memmap.

### V2 embedding format change (applies to all spaces)

V1 (flat list):
```
plot twist / reversal, planted-foreshadowing clues, slow-burn reveal, ...
```

V2 (structured labels — PREREQUISITE for cross-space rescoring):
```
information_control: plot twist / reversal, planted-foreshadowing clues
pacing_and_structure: slow-burn reveal, flashback storytelling
perspective_and_voice: multiple-perspective narration, ...
```

Search subqueries generated in the same structured shape as embedded text.

### ~~8.1 Anchor (`dense_anchor_vectors`)~~ DROPPED FROM V2

**Decision:** The anchor vector is dropped from V2. Its generalist role is
superseded by deterministic candidate generation (Phase 1) and cross-space
rescoring across specialized vectors (Phase 2). Similarity queries use a
weighted mix of targeted vectors instead. See open_questions.md for full
reasoning.

If clear failure examples emerge during V2 testing, the anchor vector can be
reintroduced in a reduced role. The embedded content definition is preserved
below for reference.

<details>
<summary>V1 embedded content (reference only)</summary>

- Title + original title
- Elevator pitch (6 words, from plot_analysis)
- Generalized plot overview (1-3 sentences, no proper nouns)
- Deduplicated genres (LLM + IMDB merged)
- Overall keywords (not plot keywords)
- Thematic concepts (from plot_analysis)
- Source material + franchise lineage (from source_of_inspiration)
- Subsampled experiential signals: emotional palette + key movie feature draws
- Release decade bucket + semantic era label
- Languages
- Budget scale relative to era
- Maturity rating
- Reception summary (prose) + reception tier label
</details>

### 8.2 Plot Events (`plot_events_vectors`)

**Purpose:** What literally happens. Chronological narrative prose.

**Embedded content** (priority fallback):
1. Full original scraped synopsis (longest IMDB synopsis)
2. LLM-generated `plot_summary`
3. Longest scraped plot_summary
4. IMDB overview (last resort)

**Subquery:** Third-person past-tense narrative. Synonym expansion + register
rephrasing. Null for pure vibes/production/technique queries.

**Boundary:** Plot EVENTS only, NOT themes.

### 8.3 Plot Analysis (`plot_analysis_vectors`)

**Purpose:** What type of story — thematic territory, genres, concepts. Generalized
terms only, no proper nouns.

**Embedded content** (from PlotAnalysisOutput):
- Elevator pitch (6 words max)
- Generalized plot overview (1-3 sentences, thematically saturated) — largest segment
- Genre signatures (2-6 compound phrases)
- Conflict type (0-2 phrases)
- Character arcs (0-3 transformation labels)
- Thematic concepts (0-5 labels)
- TMDB genres merged into genre_signatures

**Subquery:** Thematic territory. Genre phrases, concept labels, arc labels.
Translates events to thematic equivalents. Null for production/logistics/pure
technique/pure experience.

**Boundary:** What story IS ABOUT thematically, NOT how it FEELS.

### 8.4 Viewer Experience (`viewer_experience_vectors`)

**Purpose:** What it FEELS like to watch. Emotional, sensory, cognitive.

**Embedded content** (from ViewerExperienceOutput) — 8 sections, each with
`terms` + `negations`:

| Section | Examples |
|---------|----------|
| Emotional palette | uplifting, cozy, tearjerker, nostalgic |
| Tension/adrenaline | edge of seat, slow burn, anxiety inducing |
| Tone/self-seriousness | earnest, campy, deadpan, cynical |
| Cognitive complexity | thought provoking, digestible, draining |
| Disturbance profile | creepy, gory, nightmare fuel (often empty) |
| Sensory load | overstimulating, soothing (90% empty) |
| Emotional volatility | tonal whiplash, gets dark fast (empty when consistent) |
| Ending aftertaste | satisfying, gut punch, haunting, cliffhanger |

**Key feature:** Negations embedded directly ("no jump scares", "not too dark").

**Subquery:** Aggressively translates ANY query to experiential language.
Null ONLY for purely logistical queries.

**Boundary:** Emotional REACTION / viewing SENSATION, NOT thematic territory.

### 8.5 Watch Context (`watch_context_vectors`)

**Purpose:** WHY and WHEN to watch. Viewing occasions, motivations.

**Embedded content** (from WatchContextOutput) — 4 sections:

| Section | Count | Examples |
|---------|-------|----------|
| Self-experience motivations | 4-8 | mood booster, good cry, escape reality |
| External motivations | 1-4 | learn something, cultural significance |
| Key movie feature draws | 1-4 | amazing soundtrack, incredible acting |
| Watch scenarios | 3-6 | date night, solo watch, halloween movie |

**Design:** Receives ZERO plot information. `identity_note` NOT embedded.

**Subquery:** Viewing occasion language. No plot/characters/proper nouns.

**Boundary:** WHY/WHEN to watch, NOT what happens or what it's about.

### 8.6 Narrative Techniques (`narrative_techniques_vectors`)

**Purpose:** HOW the story is told. Craft, structure, narrative mechanics.

**Embedded content** (from NarrativeTechniquesOutput) — 9 sections:

| Section | Count | Examples |
|---------|-------|----------|
| Narrative archetype | 1 | cautionary tale, heist, whodunit |
| Narrative delivery | 1-2 | non-linear timeline, time loop |
| POV/perspective | 1-2 | unreliable narrator, multiple POV |
| Information control | 1-2 | plot twist, dramatic irony, red herrings |
| Characterization methods | 1-3 | show don't tell, character foil |
| Character arcs | 1-3 | redemption arc, coming-of-age |
| Audience-character perception | 1-3 | lovable rogue, morally gray lead |
| Conflict/stakes design | 1-2 | ticking clock, no-win dilemma |
| Additional narrative devices | varies | found footage, cold open, framed story |

**Subquery:** Shares routing with plot_analysis (technique → thematic translation).

**Boundary:** HOW the story is told, NOT what happens.

### 8.7 Production (`production_vectors`)

**Purpose:** How/where the film was made.

**V1 embedded content:**
- Countries of origin, production companies, filming locations
- Languages (labeled)
- Release decade + era label
- Budget scale relative to era
- Production medium (animation / live action)
- Source material + franchise lineage (from source_of_inspiration)
- Production keywords (from ProductionKeywordsOutput)

**V2 embedded content (tightened):**
- Filming locations only (not countries of origin or production companies)
- Production technique keywords: visual (black-and-white, IMAX, found-footage,
  single-take, handheld-camera), structural (anthology, mockumentary),
  process (stop-motion, rotoscope, practical-effects, motion-capture)

**V2 removes:** Countries → `country_of_origin_ids`. Companies → already in
`inv_studio_postings`. Languages → `audio_language_ids`. Budget → `budget_bucket` /
`box_office_bucket`. Source material → `source_material_type_ids`. Franchise →
`franchise_membership`. Decade → derivable from `release_ts`. Animation →
keyword search.

**Open question:** After regeneration, is the thinned content enough to justify
a dedicated vector space? With anchor dropped, options are: keep as lean space,
eliminate entirely, or repurpose the slot.

### 8.8 Reception (`reception_vectors`)

**Purpose:** What people thought. Critical/audience reception.

**V1 embedded content** (from ReceptionOutput):
- Reception tier label
- Reception summary (2-3 sentence evaluation)
- Praised qualities (0-6 tags)
- Criticized qualities (0-6 tags)

**V2 addition:**
- Awards summary text generated from `movie_awards` structured data
  (e.g. "Won Academy Award for Best Picture (2020). Nominated for Golden Globe
  for Best Director."). Appended alongside praised/criticized qualities.

---

## 9. Redis

| Key Pattern | Type | Purpose |
|-------------|------|---------|
| `emb:{model}:{hash}` | binary | Embedding cache. Case-sensitive keys. |
| `qu:v{N}:{hash}` | JSON | Query understanding cache. Lowercased. Version-bumped on prompt change. |
| `tmdb:detail:{id}` | JSON | TMDB detail cache. TTL 1 day. |
| `trending` | sorted set | Trending movie scores. Fetched once per request. Atomic RENAME. |

No V2 changes to Redis.

---

## 10. Tracker DB (SQLite — ingestion_data/tracker.db)

Pipeline state management. Not queried at search time.

### Existing tables (relevant to V2)

| Table | Key Columns | Notes |
|-------|-------------|-------|
| `movie_progress` | `tmdb_id, status, stage_3_quality_score, combined_quality_score` | Status progression through pipeline |
| `filter_log` | `tmdb_id, stage, reason, details` | Append-only audit trail |
| `tmdb_data` | 20 columns including title, release_date, budget, watch_provider_keys (BLOB) | TMDB API data |
| `imdb_data` | All IMDBScrapedMovie fields as JSON columns | IMDB GraphQL data |
| `generated_metadata` | Per-type JSON columns: plot_events, reception, plot_analysis, viewer_experience, watch_context, narrative_techniques, production_keywords, source_of_inspiration | LLM outputs |

### V2 additions to tmdb_data

| Column | Type | Purpose |
|--------|------|---------|
| `collection_name` | `TEXT?` | TMDB `belongs_to_collection.name`. Input to franchise LLM. |
| `revenue` | `INTEGER?` | TMDB revenue. Superseded by IMDB box office data for bucket calculation, but already captured — kept as supplementary signal. |

### V2 additions to imdb_data

| Column | Type | Purpose |
|--------|------|---------|
| `awards` | `TEXT` (JSON) | Scraped award nominations/wins from IMDB GraphQL `awardNominations`. Filtered to 12 in-scope ceremonies at parse time. |
| `box_office_worldwide` | `INTEGER?` | IMDB GraphQL `lifetimeGross(boxOfficeArea: WORLDWIDE)`. Whole USD, inclusive of domestic. |

### V2 additions to generated_metadata

| Column | Type | Purpose |
|--------|------|---------|
| `franchise` | `TEXT` (JSON) | LLM-generated franchise_name, franchise_role, culturally_recognized_group |
| `production_techniques` | `TEXT` (JSON) | Replaces `production_keywords` with tightened scope |
| `source_material_v2` | `TEXT` (JSON) | Re-generated with enum-constrained output |
