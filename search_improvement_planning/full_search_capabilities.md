# Full Search Capabilities

Catalog of every individual data source available for search, with a brief note
on how each can be queried independently. The finalized_search_proposal.md will
define how these are composed into the full search pipeline.

---

## 1. PostgreSQL: movie_card (structured metadata)

The canonical per-movie metadata table. Supports hard filtering (WHERE clauses /
GIN array overlap), soft preference scoring (weighted distance functions), and
sort-based ranking.

### 1.1 Scalar columns

| Column | Type | What it represents | Search utility |
|--------|------|--------------------|----------------|
| `movie_id` | BIGINT PK | TMDB movie ID. Universal join key across all stores. | Join key — not directly searchable. |
| `title` | TEXT | Display title. | Display only — title *search* is handled by the lexical schema (§3). |
| `poster_url` | TEXT? | Poster image URL. | Display only. |
| `release_ts` | BIGINT? | Unix timestamp of release date. | **Hard filter** (range) or **soft preference** (linear decay from target date, 1-5yr grace window). Enables "80s movies", "movies from 2020", "recent movies". |
| `runtime_minutes` | INT? | Duration in minutes. | **Hard filter** (range) or **soft preference** (linear decay, 30min grace). Enables "short movies", "under 2 hours", "epic runtime". |
| `maturity_rank` | SMALLINT? | Ordinal 1-5: G(1) / PG(2) / PG-13(3) / R(4) / NC-17(5). NULL = unrated. | **Hard filter** (range) or **soft preference** (1 rank away = 0.5). Enables "family friendly", "rated R". |
| `imdb_vote_count` | INT | Raw IMDB vote count. | Proxy for notability. Not directly exposed as a filter, but feeds `popularity_score`. |
| `popularity_score` | FLOAT | Sigmoid-normalized percentile from vote count, range [0, 1]. | **Soft preference** (pass-through when user prefers popular movies). Also feeds the default quality composite (`0.4 × popularity`). |
| `reception_score` | FLOAT? | Blended reception: 40% IMDB (scaled to 0-100) + 60% Metacritic. 0-100 range. | **Soft preference** (linear ramp for "critically acclaimed" / "poorly received"). Feeds default quality composite (`0.6 × reception`). Enables "well reviewed", "best movies", "hidden gems" (low popularity, high reception). |
| `budget_bucket` | TEXT? | Era-adjusted budget classification: `"small"` / `"large"` / NULL (mid-range or unknown). | **Soft preference** (binary match). Enables "low budget indie", "big budget blockbuster". |
| `box_office_bucket` | TEXT? | Box office classification: `"hit"` / `"flop"` / NULL. Movies < 75 days old always NULL. | **Soft preference** (binary match). Enables "box office hit", "commercial flop". |
| `title_token_count` | INT | Count of normalized title tokens. | Used internally by title search F-scoring (longer titles get slightly penalized to avoid false matches on common words). |
| `created_at` / `updated_at` | TIMESTAMP | Record timestamps. | Not searchable. |

### 1.2 Array columns (GIN-indexed, overlap matching)

All use `gin__int_ops` for efficient `&&` (overlap) queries.

| Column | Type | Enum | What it represents | Search utility |
|--------|------|------|--------------------|----------------|
| `genre_ids` | INT[] | Genre (27 values) | TMDB genre IDs (Action, Drama, Horror, etc.) | **Hard filter** or **soft preference** with include/exclude. Enables "action movies", "not horror". Most common filter. |
| `watch_offer_keys` | INT[] | Composite `(provider_id << 2 \| method_id)` | Where to watch + how (subscription/buy/rent). | **Hard filter** or **soft preference** (desired method = 1.0, any method = 0.5). Enables "on Netflix", "free to stream". |
| `audio_language_ids` | INT[] | Language (334 values) | Languages the movie has audio in. | **Hard filter** or **soft preference** with include/exclude. Enables "French language films", "not English". |
| `country_of_origin_ids` | INT[] | Country (262 values) | Country of origin. | **Hard filter** (Postgres-only, not in Qdrant payload). Enables "Korean movies", "British films". |
| `source_material_type_ids` | INT[] | SourceMaterialType (10 values) | What the movie is based on: novel, true story, remake, comic, etc. Empty = original screenplay. | **Hard filter** or **soft preference**. Enables "based on a true story", "book adaptations", "remakes". |
| `keyword_ids` | INT[] | OverallKeyword (225 values) | Curated IMDB genre/sub-genre taxonomy with definitions. | **Deal-breaker boost** (not hard pre-filter). Enables "zombie movies", "heist", "coming-of-age". More specific than genres but deterministic — avoids vector search noise for well-defined sub-genres. |
| `concept_tag_ids` | INT[] | 7 category enums (25 tags total) | Binary classification tags across narrative structure, plot archetype, setting, character, ending, experiential, content flags. | **Deal-breaker filter** or **preference boost**. Each tag is a yes/no signal. See §1.3 for the full tag list. |
| `award_ceremony_win_ids` | SMALLINT[] | AwardCeremony (12 values) | Which major ceremonies the movie has won at (denormalized from movie_awards). | **Hard filter** or **soft preference**. Enables "Oscar winners", "Cannes winners", "award-winning". Fast check without joining movie_awards. |

### 1.3 Concept tags (concept_tag_ids reference)

25 binary tags, never embedded — deterministic filtering/boosting only.

| Category | ID | Tag | Search example |
|----------|----|-----|----------------|
| Narrative Structure | 1 | plot_twist | "movies with a twist" |
| | 2 | twist_villain | "twist villain movies" |
| | 3 | time_loop | "Groundhog Day style time loop" |
| | 4 | nonlinear_timeline | "nonlinear storytelling" |
| | 5 | unreliable_narrator | "unreliable narrator" |
| | 6 | open_ending | "ambiguous ending" |
| | 7 | single_location | "bottle movies", "single location" |
| | 8 | breaking_fourth_wall | "breaks the fourth wall" |
| | 9 | cliffhanger_ending | "cliffhanger ending" |
| Plot Archetype | 11 | revenge | "revenge movie" |
| | 12 | underdog | "underdog story" |
| | 13 | kidnapping | "kidnapping movie" |
| | 14 | con_artist | "con artist / heist" |
| Setting | 21 | post_apocalyptic | "post-apocalyptic" |
| | 22 | haunted_location | "haunted house" |
| | 23 | small_town | "small town setting" |
| Character | 31 | female_lead | "female-led movies" |
| | 32 | ensemble_cast | "ensemble cast" |
| | 33 | anti_hero | "anti-hero protagonist" |
| Ending | 41 | happy_ending | "happy ending" |
| | 42 | sad_ending | "sad ending" |
| | 43 | bittersweet_ending | "bittersweet ending" |
| Experiential | 51 | feel_good | "feel-good movie" |
| | 52 | tearjerker | "tearjerker" |
| Content Flag | 61 | animal_death | "does the dog die?" |

---

## 2. PostgreSQL: movie_awards (structured award lookup)

Normalized award nominations and wins. Designed for inverse lookup (given an
award specification, find matching movies).

| Column | Type | What it represents |
|--------|------|--------------------|
| `movie_id` | BIGINT FK | Movie this award belongs to. |
| `ceremony_id` | SMALLINT | AwardCeremony enum ID (1-12). |
| `award_name` | TEXT | Specific prize name: "Oscar", "Palme d'Or", "Golden Lion", etc. |
| `category` | TEXT? | Category within the prize: "Best Picture", "Best Director". NULL for festival grand prizes where the award name IS the category. |
| `outcome_id` | SMALLINT | 1 = winner, 2 = nominee. |
| `year` | SMALLINT | Award year. |

**Search utility:** Deterministic lookup for specific award queries. Indexed on
`(ceremony_id, award_name, category, outcome_id, year)` for queries like:
- "Oscar Best Picture winners" → ceremony=1, award_name="Oscar", category="Best Picture", outcome=1
- "Cannes Palme d'Or nominees" → ceremony=4, award_name="Palme d'Or", outcome=2
- "2023 Oscar nominees" → ceremony=1, year=2023
- "Movies that won at Sundance" → ceremony=9, outcome=1

The denormalized `award_ceremony_win_ids` on movie_card handles the simpler
"award-winning" filter without needing this join.

**Ceremonies in scope (12):** Academy Awards (1), Golden Globes (2), BAFTA (3),
Cannes (4), Venice (5), Berlin (6), SAG (7), Critics Choice (8), Sundance (9),
Razzie (10), Spirit Awards (11), Gotham (12).

---

## 3. PostgreSQL: Lexical Schema (lex.*)

Inverted index infrastructure for entity-based text search. The pattern is:
user text → normalize → fuzzy-match against dictionary → get term_id → look up
posting table → get movie_ids.

### 3.1 Dictionaries

| Table | What it stores | Matching method |
|-------|----------------|-----------------|
| `lex.lexical_dictionary` | Master dictionary of all normalized strings (people, studios, franchises). `string_id → norm_str`. | Exact match on `norm_str` after normalization. |
| `lex.title_token_strings` | Normalized individual title tokens. `string_id → norm_str`. | **Trigram GIN** (`gin_trgm_ops`) for fuzzy LIKE/similarity matching. Handles typos and partial matches. |
| `lex.character_strings` | Normalized character names. `string_id → norm_str`. | **Trigram GIN** for fuzzy matching. |

### 3.2 Posting tables (entity → movie mapping)

| Table | Extra columns | Search utility |
|-------|---------------|----------------|
| `lex.inv_title_token_postings` | — | Title search. Multi-token title queries resolve each token independently and intersect. `title_token_doc_frequency` materialized view provides IDF-like stop-word filtering. |
| `lex.inv_actor_postings` | `billing_position`, `cast_size` | Actor search with **prominence scoring**. Enables 4 modes: exclude non-major (top 10-15%), boost by position (default: `1 - pos/cast_size`), binary (all equal), reverse (boost deep credits for "minor roles" queries). |
| `lex.inv_director_postings` | — | Director search. Binary presence. |
| `lex.inv_writer_postings` | — | Writer/screenwriter search. Binary presence. |
| `lex.inv_producer_postings` | — | Producer search. Binary presence. |
| `lex.inv_composer_postings` | — | Composer/musician search. Binary presence. |
| `lex.inv_character_postings` | — | Character name search (fuzzy-resolved via character_strings). Enables "movies with a character named X". |
| `lex.inv_studio_postings` | — | Production company search. Enables "Pixar movies", "A24 films". |
| `lex.inv_franchise_postings` | — | Franchise search via `movie_franchise_metadata.lineage` and `.shared_universe`. Both are projected into the same posting table. Fuzzy resolution handles naming variation. |

### 3.3 Materialized views

| View | Purpose |
|------|---------|
| `lex.title_token_doc_frequency` | `term_id → doc_frequency`. Used for max-df stop-word filtering — extremely common tokens (like "the") are downweighted or excluded from title matching. |

### 3.4 Entity extraction categories (query understanding → posting table routing)

| Entity type | Resolved via | Notes |
|-------------|-------------|-------|
| MOVIE_TITLE | `inv_title_token_postings` | Fuzzy trigram on title_token_strings, then posting lookup. |
| PERSON | Role-specific tables | `role_hint` from LLM routes to actor/director/writer/producer/composer. Falls back to searching all when role unclear. |
| CHARACTER | `inv_character_postings` | Fuzzy trigram on character_strings. |
| FRANCHISE | `inv_franchise_postings` | Fuzzy resolution against franchise strings. |
| STUDIO | `inv_studio_postings` | Exact-after-normalization match. |

---

## 4. PostgreSQL: movie_franchise_metadata (structured franchise data)

Per-movie franchise classification. Enables structured franchise queries beyond
simple name matching.

| Column | Type | Search utility |
|--------|------|----------------|
| `movie_id` | BIGINT PK | Join key. |
| `lineage` | TEXT? | Franchise name (e.g., "Marvel Cinematic Universe", "James Bond"). Projected into `inv_franchise_postings` for text search. |
| `shared_universe` | TEXT? | Shared universe name when distinct from lineage. Also projected into `inv_franchise_postings`. |
| `recognized_subgroups` | TEXT[] | Named subgroups (e.g., ["Iron Man", "Avengers"] within MCU). Searched via trigram similarity on the post-franchise-lookup result set (3-30 movies, no index needed). |
| `launched_subgroup` | BOOLEAN | Did this film launch a subgroup? Boolean filter. |
| `lineage_position` | SMALLINT? | 1=sequel, 2=prequel, 3=remake, 4=reboot. Enables "sequels in the X franchise", "remakes". |
| `is_spinoff` | BOOLEAN | Boolean filter for "spinoff movies". |
| `is_crossover` | BOOLEAN | Boolean filter for "crossover movies". |
| `launched_franchise` | BOOLEAN | Boolean filter for "movies that started a franchise". |

**Search flow:** Franchise name → fuzzy resolve via `inv_franchise_postings` →
get candidate movie_ids → optionally narrow with `lineage_position`, `is_spinoff`,
etc. on the `movie_franchise_metadata` table.

---

## 5. Qdrant: 8 Named Vector Spaces

Each movie has up to 8 vectors (OpenAI `text-embedding-3-small`, 1536 dims).
Searched via cosine similarity. Each space targets a specific semantic dimension.

Subqueries are LLM-generated reformulations of the user query, tailored to
each space's semantic territory. The blend ratio is 80% subquery / 20% original.

### 5.1 Anchor (`dense_anchor_vectors`)

**What it captures:** Lean holistic movie fingerprint — the movie's overall identity.

**Embedded content:** title, original_title, identity_pitch (6-word elevator pitch),
identity_overview (thematic overview), genre_signatures, themes, emotional_palette
(positive terms only), key_draws, maturity_summary, reception_summary.

**No subquery** — always searched with the original user query only.

**When useful:** Broad "movies like X" queries, general vibes queries that don't
emphasize any single dimension, catch-all similarity when no specialized space
is clearly dominant.

### 5.2 Plot Events (`plot_events_vectors`)

**What it captures:** What literally happens. Chronological narrative prose.

**Embedded content (priority fallback):** full IMDB synopsis → LLM plot_summary →
longest scraped plot_summary → IMDB overview.

**Has subquery:** Third-person past-tense narrative with synonym expansion.

**When useful:** "Movie where a guy wakes up in a different body", "the one with
the heist on a train", any query describing specific plot events or scenes.

### 5.3 Plot Analysis (`plot_analysis_vectors`)

**What it captures:** What type of story — thematic territory, genre signatures,
concepts. Generalized terms, no proper nouns.

**Embedded content:** elevator_pitch → generalized_plot_overview → genre_signatures
→ conflict → themes → character_arcs (thematic arc labels).

**Has subquery:** Thematic territory in structured-label format.

**When useful:** "Redemption stories", "movies about grief", "man vs nature
conflict", "coming of age with dark themes". Thematic searches where the user
cares about what the movie *means*, not what literally happens.

### 5.4 Viewer Experience (`viewer_experience_vectors`)

**What it captures:** What it FEELS like to watch. Emotional, sensory, cognitive.

**Embedded content:** 8 experiential dimensions, each with positive terms and
negations as separate labeled lines:
- emotional_palette, tension_adrenaline, tone_self_seriousness,
  cognitive_complexity, disturbance_profile, sensory_load,
  emotional_volatility, ending_aftertaste

**Has subquery:** Aggressively translates query to experiential language.

**When useful:** "Feel-good movies", "something unsettling but not gory",
"slow burn suspense", "movies that leave you thinking", "visually overwhelming".
Any query focused on the *experience* of watching rather than what the movie
is about.

### 5.5 Watch Context (`watch_context_vectors`)

**What it captures:** WHY and WHEN to watch. Viewing occasions and motivations.

**Embedded content:** 4 sections — self_experience_motivations, external_motivations,
key_movie_feature_draws, watch_scenarios.

**Has subquery:** Viewing occasion language, no plot/characters/proper nouns.

**When useful:** "Date night movie", "good background movie", "something to watch
with my parents", "culturally important films everyone should see", "movie for
a rainy Sunday". Occasion-driven and motivation-driven queries.

### 5.6 Narrative Techniques (`narrative_techniques_vectors`)

**What it captures:** HOW the story is told. Craft, structure, storytelling mechanics.

**Embedded content:** 9 sections — narrative_archetype, narrative_delivery,
pov_perspective, characterization_methods, character_arcs (film-language labels),
audience_character_perception, information_control, conflict_stakes_design,
additional_narrative_devices.

**Has subquery:** Technique + thematic translation.

**When useful:** "Movies with unreliable narrators", "nonlinear storytelling",
"found footage style", "movies that use dramatic irony well", "cold open into
flashback structure". Queries about storytelling craft.

### 5.7 Production (`production_vectors`)

**What it captures:** How/where the film was physically made.

**Embedded content (V2 thinned):** filming_locations (up to 3, omitted for
animation) + production_techniques (concrete making/rendering/capture methods:
black-and-white, stop-motion, practical effects, hand-held camera, etc.).

**Has subquery.**

**When useful:** "Movies filmed in New Zealand", "stop-motion animated films",
"shot on 16mm", "black and white movies", "practical effects heavy". Physical
production queries. Note: this space is deliberately thin in V2 — many things
formerly here moved to deterministic Postgres columns.

### 5.8 Reception (`reception_vectors`)

**What it captures:** What people thought. Critical and audience reception.

**Embedded content:** reception_summary (2-3 sentence evaluation), praised_qualities
(0-6 tags), criticized_qualities (0-6 tags), major_award_wins (deterministic
ceremony list from movie_awards, Razzie excluded).

**Has subquery.**

**When useful:** "Universally acclaimed movies", "movies praised for cinematography",
"controversial films critics hated but audiences loved", "award-winning thriller".
Reception-focused queries. For *specific* award lookups ("2023 Oscar Best Picture
nominees"), use movie_awards deterministically instead.

---

## 6. Qdrant: Payload (hard filters applied at vector search time)

Minimal payload stored per Qdrant point for pre-filtering before cosine similarity.
These are a subset of movie_card fields — only the ones worth filtering in Qdrant
to avoid scanning irrelevant vectors.

| Field | Type | Mirrors |
|-------|------|---------|
| `release_ts` | int? | movie_card.release_ts |
| `runtime_minutes` | int? | movie_card.runtime_minutes |
| `maturity_rank` | int? | movie_card.maturity_rank |
| `genre_ids` | int[] | movie_card.genre_ids |
| `watch_offer_keys` | int[] | movie_card.watch_offer_keys |
| `audio_language_ids` | int[] | movie_card.audio_language_ids |

**Not in Qdrant payload** (filtered in Postgres before/after vector search):
country_of_origin_ids, source_material_type_ids, keyword_ids, concept_tag_ids,
award_ceremony_win_ids, box_office_bucket, budget_bucket.

---

## 7. Redis

### 7.1 Trending scores

| Key pattern | Type | What it represents |
|-------------|------|--------------------|
| `{ENV}:trending:current` | Hash (movie_id → score) | Precomputed trending scores [0, 1] for all trending movies. Atomically replaced via staging key + RENAME. |

**Search utility:** **Soft preference** — score passed through directly when user
wants trending movies ("what's popular right now", "trending movies"). Also feeds
the default sort order when no explicit ranking criteria is given.

### 7.2 Caches (not search data, but relevant to pipeline)

| Key pattern | Type | Purpose |
|-------------|------|---------|
| `emb:{model}:{hash}` | binary | Embedding cache. Avoids redundant OpenAI API calls. Case-sensitive keys. |
| `qu:v{N}:{hash}` | JSON | Query understanding cache. Lowercased + normalized. Version-bumped on prompt change. |
| `tmdb:detail:{id}` | JSON | TMDB detail cache. TTL 1 day. Ingestion-time only. |

---

## 8. SQLite Tracker DB (ingestion-time only, not queried at search time)

The tracker DB contains raw source data that feeds the production databases.
Listed here because some fields exist in the tracker that are *not* currently
promoted to search-time stores, and could be if needed.

### 8.1 Data already promoted to search-time stores

These fields flow from tracker → Postgres/Qdrant and are covered in §1-6:
- All movie_card columns
- All lexical posting data (names, titles, characters, studios, franchises)
- All vector text (8 spaces of LLM-generated metadata)
- All movie_awards rows
- All movie_franchise_metadata rows

### 8.2 Data available in tracker but NOT in search-time stores

Fields that could be promoted if a search use case justified it:

| Source table | Field(s) | What it represents | Potential search use |
|-------------|----------|--------------------|---------------------|
| imdb_data | `featured_reviews` (up to 10) | Full user review text with summaries. | Full-text review search, review-based semantic matching. Currently only the LLM-distilled reception_summary and praised/criticized tags reach Qdrant. |
| imdb_data | `plot_keywords` | 10-15 community-voted plot keywords with engagement scoring. | More granular than overall_keywords. Currently only overall_keywords are promoted to keyword_ids. |
| imdb_data | `parental_guide_items` | Content warnings: {category, severity}. | Granular content filtering beyond maturity_rank (e.g., "no sexual content", "mild violence only"). Currently only the concept_tag `animal_death` addresses content specifics. |
| imdb_data | `maturity_reasoning` | Textual explanation for the rating. | Could augment content-based filtering. |
| imdb_data | `filming_locations` (full list) | All real-world filming locations. | Currently only first 3 are embedded in production vectors. Full list could support more location queries. |
| imdb_data | `review_themes` | Structured {name, sentiment} theme analysis from reviews. | Already partially captured in reception vector via LLM processing. |
| tmdb_data | `vote_average` | TMDB user rating (0-10). | Already subsumed by reception_score (which blends IMDB + Metacritic). |
| tmdb_data | `collection_name` | TMDB franchise collection name. | Already subsumed by movie_franchise_metadata.lineage. |
| generated_metadata | Reception extraction zone: `source_material_hint`, `thematic_observations`, `emotional_observations`, `craft_observations` | LLM-extracted reception signals passed to Wave 2 generators. | Not embedded or stored in Postgres. These intermediate signals feed other generators but aren't directly searchable. |
| generated_metadata | `watch_context.identity_note` | 2-8 word classification of viewing appeal (e.g., "sincere emotional drama"). | Not embedded. Could serve as a compact movie archetype label. |
| generated_metadata | All CoT reasoning fields | Justification text from every generator. | Not embedded. Diagnostic value only. |
| generated_metadata | Eligibility flags (`eligible_for_*`) | Whether input data was sufficient for each generation type. | Could be used to flag movies with sparse metadata at search time. |

---

## 9. Runtime External Sources (search-time API calls)

Data sources that are not stored locally but are called at search time or
feed search-time stores via periodic refresh.

### 9.1 TMDB Trending API

The trending refresh job (`db/trending_movies.py`) calls the TMDB
`/trending/movie/week` endpoint to fetch the top-500 weekly trending movie IDs
in rank order. These are scored with a concave-decay formula and written
atomically to the Redis `trending:current` hash. This is the sole source of
trending data — if the refresh job hasn't run, trending scores are empty.

**Refresh cadence:** Manual / scheduled (not on every search request).

### 9.2 OpenAI Embedding API

The user's query text is embedded at search time via `text-embedding-3-small`
(1536 dims) to produce the query vector used for Qdrant similarity search.
Cached in Redis (`emb:{model}:{hash}`) to avoid redundant calls for repeated
queries.

### 9.3 Query Understanding LLM

The LLM is called at search time (4 parallel calls in V1) to produce:
- **Channel weights** — how much to weight lexical vs. vector vs. metadata channels
- **Lexical entity extraction** — identified people, titles, characters, franchises, studios with role hints
- **Vector space weights + subqueries** — per-space relevance sizing and reformulated search queries
- **Metadata preference extraction** — structured preferences for date, genre, runtime, streaming, etc.

The LLM's parametric knowledge is itself a search capability — it knows that
"Nolan" is a director (routing to the correct posting table), what "feel-good"
means (generating appropriate subqueries), and can infer implicit preferences
(a "date night movie" query implies certain genres and tones).

Cached in Redis (`qu:v{N}:{hash}`) keyed by normalized query text.

---

## 10. Static Knowledge Bases (in-code enums with embedded definitions)

### 10.1 OverallKeyword definitions

The `OverallKeyword` enum (225 members in `implementation/classes/overall_keywords.py`)
carries a per-keyword `definition` string designed for LLM context. Each
definition clarifies what the keyword means and when it should apply. These
definitions can be passed to the query understanding LLM to help it accurately
map natural language queries to keyword IDs (e.g., understanding that "zombie
movies" maps to a specific keyword ID rather than requiring vector search).

### 10.2 Concept tag definitions

The 25 concept tags (§1.3) have implicit definitions baked into the LLM
classification prompt. At query time, the same semantic understanding can
inform the query understanding LLM's ability to route queries like "movies
with a twist ending" to the `plot_twist` concept tag rather than relying
solely on vector similarity.

---

## 11. Cross-cutting: Enums and ID systems

For reference, the enum ID systems that underlie array columns and filters:

| Enum | Count | Used in |
|------|-------|---------|
| Genre | 27 | genre_ids, Qdrant payload |
| Language | 334 | audio_language_ids, Qdrant payload |
| Country | 262 | country_of_origin_ids |
| StreamingService / AccessType | ~20 services × 3 methods | watch_offer_keys, Qdrant payload |
| MaturityRating | 5 + UNRATED | maturity_rank, Qdrant payload |
| SourceMaterialType | 10 | source_material_type_ids |
| OverallKeyword | 225 | keyword_ids |
| Concept tags (7 categories) | 25 | concept_tag_ids |
| AwardCeremony | 12 | award_ceremony_win_ids, movie_awards.ceremony_id |
| AwardOutcome | 2 | movie_awards.outcome_id |
| LineagePosition | 4 | movie_franchise_metadata.lineage_position |
| RelevanceSize | 4 | Vector space weight assignment (query-time only) |
| EntityCategory | 5 | Query understanding entity extraction routing |

---

## 12. Summary: Search capability by query intent

Quick reference for which data sources serve which user intents.

| Query intent | Primary data source | Secondary / boosting source |
|-------------|--------------------|-----------------------------|
| **Known movie by title** | lex.inv_title_token_postings (trigram) | — |
| **Known movie by plot description** | plot_events vectors | plot_analysis vectors |
| **Actor/director/crew filmography** | lex.inv_actor/director/writer/producer/composer_postings | Actor prominence scoring (billing_position) |
| **Character name** | lex.inv_character_postings (trigram) | — |
| **Franchise / sequel / spinoff** | lex.inv_franchise_postings + movie_franchise_metadata | lineage_position, is_spinoff, is_crossover filters |
| **Studio / production company** | lex.inv_studio_postings | — |
| **Genre filtering** | movie_card.genre_ids (GIN) | Qdrant payload genre_ids |
| **Thematic / conceptual** | plot_analysis vectors | concept_tag_ids (deterministic boost) |
| **Emotional / experiential** | viewer_experience vectors | concept_tag_ids (feel_good, tearjerker) |
| **Viewing occasion** | watch_context vectors | — |
| **Storytelling technique** | narrative_techniques vectors | concept_tag_ids (plot_twist, unreliable_narrator, etc.) |
| **Production style / technique** | production vectors | — |
| **Filming location** | production vectors | — |
| **Critical reception / acclaim** | reception vectors | reception_score, movie_awards |
| **Specific award lookup** | movie_awards table | award_ceremony_win_ids for broad "award-winning" |
| **Streaming availability** | movie_card.watch_offer_keys (GIN) | Qdrant payload |
| **Release date range** | movie_card.release_ts | Qdrant payload |
| **Country of origin** | movie_card.country_of_origin_ids (GIN) | — |
| **Source material / adaptation** | movie_card.source_material_type_ids (GIN) | — |
| **Content safety / warnings** | concept_tag_ids (animal_death) + maturity_rank | parental_guide_items (tracker, not promoted) |
| **Trending / popular** | Redis trending scores + popularity_score | — |
| **Sub-genre keyword** | movie_card.keyword_ids (GIN, 225 terms) | — |
| **"Movies like X"** | anchor vectors (holistic similarity) | All specialized vectors weighted by relevant dimensions |
| **Superlative ("scariest", "funniest")** | Relevant vector space with preserved similarity scoring | concept_tag_ids, genre_ids as pre-filters |
| **Box office performance** | movie_card.box_office_bucket | — |
| **Budget scale** | movie_card.budget_bucket | — |
| **Hidden gem / underrated** | High reception_score + low popularity_score | — |
