# LLM Metadata Generation Pipeline Report

End-to-end audit of how movies are enriched with LLM-generated metadata after passing IMDB quality filtering, through embedding and database ingestion.

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Input Data: What a Movie Looks Like at This Stage](#input-data)
3. [LLM Generation Orchestration](#llm-generation-orchestration)
4. [The 7 Metadata Types (Detailed)](#the-7-metadata-types)
5. [Vector Text Generation: From Metadata to Embeddable Text](#vector-text-generation)
6. [Embedding](#embedding)
7. [Database Ingestion](#database-ingestion)
8. [Error Handling](#error-handling)
9. [Key File Reference](#key-file-reference)

---

## 1. Pipeline Overview <a name="pipeline-overview"></a>

After a movie passes IMDB quality filtering (tracker status: `imdb_quality_passed`), it goes through three stages before it's searchable:

```
BaseMovie (enriched with TMDB + IMDB data)
    │
    ▼
Stage 1: LLM Metadata Generation
    │  7 metadata types generated via gpt-5-mini
    │  8 total LLM calls per movie (production splits into 2 sub-calls)
    │  Two-wave parallel execution
    │
    ▼
Stage 2: Vector Text Generation + Embedding
    │  8 text representations created (one per vector space)
    │  All 8 embedded in a single batched OpenAI API call
    │  Model: text-embedding-3-small (1536 dimensions)
    │
    ▼
Stage 3: Database Ingestion
    ├── PostgreSQL: movie_card row + lexical posting tables
    └── Qdrant: single point with 8 named vectors + hard-filter payload
```

**Model used for all LLM generation:** `gpt-5-mini` (OpenAI reasoning model) via `openai_client.chat.completions.parse()` with structured output validation.

**Model used for all embeddings:** `text-embedding-3-small` (OpenAI, 1536 dimensions).

---

## 2. Input Data: What a Movie Looks Like at This Stage <a name="input-data"></a>

By the time a movie reaches LLM metadata generation, it has been enriched with data from both TMDB and IMDB. The `BaseMovie` object contains:

### Core Movie Info
| Field | Source | Description |
|-------|--------|-------------|
| `title` | TMDB | Movie title |
| `overview` | TMDB | Marketing summary / premise (not a full plot) |
| `genres` | TMDB | List of genre strings (e.g., ["Drama", "Thriller"]) |
| `release_date` | TMDB | YYYY-MM-DD format |
| `duration` | TMDB | Runtime in minutes |
| `budget` | TMDB | Production budget (may be None) |

### Cast & Crew
| Field | Source | Description |
|-------|--------|-------------|
| `directors`, `writers`, `producers`, `composers` | TMDB + IMDB | Lists of people names |
| `actors`, `characters` | TMDB + IMDB | Lead cast and their character names |
| `production_companies` | TMDB | Studios that produced the film |

### Plot Content (from IMDB)
| Field | Source | Description |
|-------|--------|-------------|
| `plot_keywords` | IMDB | Short phrases viewers tag as important plot elements (community-voted) |
| `overall_keywords` | IMDB | High-level keywords about the movie (broader than plot) |
| `plot_summaries` | IMDB | Shorter user-written plot summaries |
| `plot_synopses` | IMDB | Longest, most detailed plot recounts (primary truth source) |

### Audience Reception (from IMDB)
| Field | Source | Description |
|-------|--------|-------------|
| `reception_summary` | IMDB | Summary of what people think about the movie |
| `featured_reviews` | IMDB | User-written reviews (each has `summary` + `text`) |
| `review_themes` | IMDB | Key attributes with audience sentiment (positive/negative/neutral) |

### Content Rating (from IMDB)
| Field | Source | Description |
|-------|--------|-------------|
| `maturity_rating` | IMDB | G / PG / PG-13 / R / NC-17 |
| `maturity_reasoning` | IMDB | Why it received that rating |
| `parental_guide_items` | IMDB | Category + severity pairs (e.g., Violence: Moderate) |

### Other
| Field | Source | Description |
|-------|--------|-------------|
| `imdb_rating`, `metacritic_rating` | IMDB | Numerical ratings |
| `imdb_vote_count` | IMDB | Number of IMDB votes |
| `watch_providers` | TMDB | Streaming services, rental platforms |
| `countries_of_origin`, `languages`, `filming_locations` | TMDB | Production geography |

---

## 3. LLM Generation Orchestration <a name="llm-generation-orchestration"></a>

**Entry point:** `generate_llm_metadata()` in `implementation/llms/vector_metadata_generation_methods.py`

The function generates all 7 metadata types using a two-wave parallel execution strategy:

```
Wave 1 (3 tasks in parallel, ThreadPoolExecutor max_workers=3):
┌─────────────────────────────────┐
│ generate_plot_events_metadata   │ ← CRITICAL: must succeed
│ generate_watch_context_metadata │ ← graceful failure OK
│ generate_reception_metadata     │ ← graceful failure OK
└─────────────────────────────────┘
                │
                │ plot_events_metadata.plot_summary used as plot_synopsis input
                ▼
Wave 2 (4 tasks in parallel, ThreadPoolExecutor max_workers=4):
┌──────────────────────────────────────────┐
│ generate_plot_analysis_metadata          │
│ generate_viewer_experience_metadata      │
│ generate_narrative_techniques_metadata   │
│ generate_production_metadata             │
│   ├── generate_production_keywords       │
│   └── generate_source_of_inspiration    │
└──────────────────────────────────────────┘
```

**Why two waves?** Plot events generation produces a detailed `plot_summary` field which serves as the `plot_synopsis` input for all Wave 2 functions. This is because the IMDB `plot_synopses` may be incomplete or absent, so the LLM first synthesizes a comprehensive plot summary from all available sources, then that summary feeds the deeper analysis tasks.

**Why is plot_events critical?** If it fails, there's no plot synopsis for Wave 2 to analyze. The entire function raises `RuntimeError`. All other failures are graceful — the metadata key is set to `None` and the movie can still be partially indexed.

### LLM Call Configuration

All 8 LLM calls (7 metadata types, with production split into 2 sub-calls) use the same pattern:

```python
generate_openai_response(
    user_prompt=<assembled input text>,
    system_prompt=<task-specific system prompt>,
    response_format=<Pydantic model class>,
    model="gpt-5-mini",
    reasoning_effort=<varies by task>,
    verbosity="low"
)
```

The call uses OpenAI's native structured output via `chat.completions.parse()`, which automatically validates that the response matches the Pydantic schema.

**Reasoning effort by task:**

| Task | Reasoning Effort | Rationale |
|------|-----------------|-----------|
| Plot Events | `"minimal"` | Extraction task — pulling facts from inputs, minimal analysis |
| Plot Analysis | `"low"` | Structured thematic analysis |
| Viewer Experience | `"low"` | Structured emotional analysis |
| Watch Context | `"medium"` | Requires more inferential reasoning about real-world contexts |
| Narrative Techniques | `"medium"` | Technical film analysis requiring deeper reasoning |
| Production Keywords | `"low"` | Simple filtering task |
| Source of Inspiration | `"low"` | Pattern matching against known categories |
| Reception | `"low"` | Summarization and extraction task |

---

## 4. The 7 Metadata Types (Detailed) <a name="the-7-metadata-types"></a>

### 4.1 Plot Events Metadata

**Purpose:** Capture WHAT HAPPENS in the movie — concrete events, characters, and settings. This powers queries like "movie where a detective investigates a murder in 1940s LA."

**System prompt key directive:** "Extract a HIGH-SIGNAL, SPOILER-CONTAINING representation of WHAT HAPPENS."

**Input fields provided:**
- `title` — movie title
- `overview` — marketing summary (not full plot)
- `plot_summaries` — shorter user-written IMDB summaries
- `plot_synopses` — longest/most detailed plot recounts (primary truth source)
- `plot_keywords` — IMDB community-tagged plot elements

**Output schema (`PlotEventsMetadata`):**

| Field | Type | Description |
|-------|------|-------------|
| `plot_summary` | str | Detailed chronological spoiler-containing summary of the entire film. Preserves character names, locations, key events. **This field is reused as input to Wave 2 functions.** |
| `setting` | str | 10 words or less — where/when the story takes place (e.g., "1912, RMS Titanic crossing the Atlantic") |
| `major_characters` | list[MajorCharacter] | Only the absolutely essential characters. Each has: `name`, `description` (who they are), `role` (protagonist/antagonist/etc.), `primary_motivations` (1 sentence) |

**How it becomes embedding text (`__str__`):** Concatenates `plot_summary` + `setting` + character descriptions, all lowercased, separated by newlines.

---

### 4.2 Plot Analysis Metadata

**Purpose:** Capture WHAT TYPE OF STORY this is — themes, lessons, core concepts. This powers queries like "movie about the cost of revenge" or "redemption story."

**System prompt key directive:** "Generalize: replace proper nouns with generic terms for better vector embedding."

**Input fields provided:**
- `title`, `genres`, `overview`
- `plot_synopsis` — **from plot_events_metadata.plot_summary** (Wave 1 output)
- `plot_keywords`
- `reception_summary` (optional) — strong source of thematic meaning
- `featured_reviews` (up to 5) — strongest source of thematic meaning

**Output schema (`PlotAnalysisMetadata`):**

| Field | Type | Description |
|-------|------|-------------|
| `core_concept` | CoreConcept | The single dominant story concept (6 words or less). Has `core_concept_label` + `explanation_and_justification`. E.g., "Investigation reveals escalating truth" |
| `genre_signatures` | list[str] (2-6) | Search-query-like genre phrases, 1-4 words each. E.g., "murder mystery", "survival thriller" |
| `conflict_scale` | str | Scale of consequences: personal / small-group / community / large-scale / mass-casualty / global |
| `character_arcs` | list[CharacterArc] (1-3) | Key character transformations. Each has `character_name`, `arc_transformation_label` (generic term like "redemption"), `arc_transformation_description` |
| `themes_primary` | list[MajorTheme] (1-3) | Core concepts the story explores (NOT the answers). E.g., "Love constrained by power". Each has `theme_label` + `explanation_and_justification` |
| `lessons_learned` | list[MajorLessonLearned] (0-3) | Key takeaways, phrased as moral lessons. E.g., "Freedom costs safety". Optional — only if strongly supported by plot. |
| `generalized_plot_overview` | str | 1-3 sentences describing the story with generalized terms, heavily emphasizing themes and lessons throughout. |

**How it becomes embedding text (`__str__`):** Concatenates `generalized_plot_overview` + core concept + genre signatures + conflict scale + character arc labels + theme labels + lesson labels, all lowercased.

---

### 4.3 Viewer Experience Metadata

**Purpose:** Capture what it FEELS LIKE to watch the movie — the emotional, sensory, and cognitive experience from the viewer's perspective. This powers queries like "edge of your seat thriller" or "cozy feel-good movie."

**System prompt key directive:** "Produce search-query-like phrases that match how real users actually type queries."

**Input fields provided:**
- `title`, `genres`
- `plot_synopsis` — **from plot_events_metadata.plot_summary** (Wave 1 output)
- `plot_keywords`, `overall_keywords`
- `maturity_rating`, `maturity_reasoning`, `parental_guide_items`
- `reception_summary`, `audience_reception_attributes`, `featured_reviews` (up to 5)

**Output schema (`ViewerExperienceMetadata`):** 8 sections, each with `justification`, `terms` (3-10 phrases), and `negations` (3-10 phrases):

| Section | What It Captures | Example Terms | Example Negations |
|---------|-----------------|---------------|-------------------|
| `emotional_palette` | Dominant emotions while watching | "uplifting and hopeful", "tearjerker", "cozy" | "not too sad", "not cheesy" |
| `tension_adrenaline` | Stress, energy, suspense | "edge of your seat", "slow burn suspense" | "not too intense", "not stressful" |
| `tone_self_seriousness` | Movie's attitude — earnest vs ironic | "campy", "dark comedy", "grounded and realistic" | "not cringey", "not corny" |
| `cognitive_complexity` | Mental effort required to follow | "thought provoking", "digestible", "confusing" | "not hard to follow", "not draining" |
| `disturbance_profile`* | Unsettling elements — horror, gore | "psychological horror", "jump scares", "nightmare fuel" | "not too gory", "not scary" |
| `sensory_load`* | Visual/auditory intensity | "overstimulating", "soothing" | "not too loud" |
| `emotional_volatility`* | How emotional tone changes over time | "tonal whiplash", "gets dark fast" | "consistent tone" |
| `ending_aftertaste` | Final emotion after watching | "satisfying ending", "gut punch ending" | "not a downer ending" |

\* Sections marked with `*` are optional (`OptionalViewerExperienceSection`) — they have a `should_skip` field set to `True` when not applicable (e.g., a rom-com skips `disturbance_profile`).

**Design rationale for negations:** The embedding model needs to understand what a movie is NOT, so vector search can distinguish "scary but not gory" from "gory but not scary." Negation phrases always contain "not" or "no."

**Design rationale for redundancy:** The prompts explicitly request "redundant near-duplicates" (synonyms, slang, paraphrases) to maximize recall in vector similarity search.

**How it becomes embedding text (`__str__`):** Concatenates all `terms` + all `negations` from non-skipped sections as a comma-separated list.

---

### 4.4 Watch Context Metadata

**Purpose:** Capture WHY and WHEN someone would choose to watch this movie — motivations, occasions, and contexts. This powers queries like "date night movie" or "something to watch high."

**System prompt key directive:** "Extract what would motivate someone to watch this movie or real-world occasions in which this movie would be a good fit."

**Input fields provided:**
- `title`, `genres`, `overview`
- `plot_keywords`, `overall_keywords`
- `reception_summary`, `audience_reception_attributes`, `featured_reviews` (up to 5)

Note: This function does NOT receive `plot_synopsis` — it runs in Wave 1 (in parallel with plot_events), so the synthesized plot summary isn't available yet. It uses `overview` (the marketing summary) instead.

**Output schema (`WatchContextMetadata`):** 4 sections, each a `GenericTermsSection` with `justification` + `terms`:

| Section | Phrase Count | What It Captures | Example Terms |
|---------|-------------|-----------------|---------------|
| `self_experience_motivations` | 4-8 | Self-focused experiential reasons to watch | "need a laugh", "mood booster", "cathartic watch", "feed my adrenaline addiction" |
| `external_motivations` | 1-4 | Value beyond the viewing experience itself | "sparks conversation", "cult classic", "culturally iconic characters" |
| `key_movie_feature_draws` | 1-4 | Standout attributes that attract viewers | "incredible soundtrack", "visually stunning", "compelling characters" |
| `watch_scenarios` | 3-6 | Real-world occasions and contexts | "romantic date night", "watch with the boys", "stoned movie", "halloween movie" |

**How it becomes embedding text (`__str__`):** Concatenates all `terms` from all 4 sections as a comma-separated list.

---

### 4.5 Narrative Techniques Metadata

**Purpose:** Capture HOW the story is told — cinematic narrative craft, structure, and storytelling mechanics. This powers queries like "movie with an unreliable narrator" or "non-linear timeline."

**System prompt key directive:** "Focus on HOW the story is told (POV/structure/info control/theme delivery), not what happens."

**Input fields provided:**
- `title`
- `plot_synopsis` — **from plot_events_metadata.plot_summary** (Wave 1 output)
- `plot_keywords`, `overall_keywords`
- `featured_reviews` (up to 5), `reception_summary`

**Output schema (`NarrativeTechniquesMetadata`):** 11 sections, each a `GenericTermsSection` with `justification` + `terms`:

| Section | Phrase Count | What It Captures | Example Terms |
|---------|-------------|-----------------|---------------|
| `pov_perspective` | 1-2 | Who the audience experiences through | "unreliable narrator", "multiple pov switching" |
| `narrative_delivery` | 1-2 | How time is arranged / manipulated | "non-linear timeline", "flashback-driven structure" |
| `narrative_archetype` | 1 | The classic whole-plot label | "revenge spiral", "whodunit mystery", "underdog rise" |
| `information_control` | 1-2 | How the story controls audience knowledge | "plot twist", "dramatic irony", "red herrings" |
| `characterization_methods` | 1-3 | How characters are conveyed to the audience | "show don't tell", "character foil contrast" |
| `character_arcs` | 1-3 | How characters change (technique labels) | "redemption arc", "corruption arc", "flat arc" |
| `audience_character_perception` | 1-3 | How viewers read/judge characters | "lovable rogue", "morally gray lead" |
| `conflict_stakes_design` | 1-2 | How pressure is created | "ticking clock deadline", "no-win dilemma" |
| `thematic_delivery` | 1-2 | How deeper meaning is communicated | "moral argument embedded in choices", "contrast pairs" |
| `meta_techniques` | 0-2 | Self-awareness and deconstruction | "fourth-wall breaks", "genre deconstruction" |
| `additional_plot_devices` | misc | Extra narrative mechanisms | "cold open", "framed story", "chaptered structure" |

**Strict confidence rule:** The prompt instructs the model to only include techniques it is "strongly confident" are present. Uncertain = omit.

**How it becomes embedding text (`__str__`):** Concatenates all `terms` from all 11 sections as a comma-separated list.

---

### 4.6 Production Metadata

**Purpose:** Capture how the movie was produced in the real world — source material, production medium, production-related keywords. This powers queries like "based on a true story" or "stop motion animation."

This metadata type is unique: it's generated by **two sub-functions running in parallel** within a `ThreadPoolExecutor(max_workers=2)`, then combined.

#### Sub-function A: Production Keywords

**System prompt key directive:** "Take a list of keywords and return every keyword that relates to the production of the movie."

**Input fields:** `title`, `overall_keywords` (the full IMDB keyword list)

**What it does:** Filters the existing keyword list to only production-relevant keywords. The LLM is NOT generating new keywords — it's classifying which existing keywords relate to how the movie was produced (not plot events, themes, or genres).

**Output:** `GenericTermsSection` with `justification` + `terms` (filtered subset of input keywords)

#### Sub-function B: Source of Inspiration

**System prompt key directive:** "Determine what real-world sources of inspiration the movie is based on and how the film was produced visually."

**Input fields:** `title`, `plot_synopsis` (from Wave 1), `plot_keywords`, `overall_keywords`, `featured_reviews` (up to 5)

**Output (`SourceOfInspirationSection`):**

| Field | Type | Description |
|-------|------|-------------|
| `justification` | str | 2-sentence justification citing concrete evidence |
| `sources_of_inspiration` | list[str] | E.g., "based on a true story", "based on a novel" — only direct adaptations, not loose inspiration |
| `production_mediums` | list[str] | E.g., "live action", "hand-drawn animation", "stop motion" |

**Combined output (`ProductionMetadata`):**
- `production_keywords`: from Sub-function A
- `sources_of_inspiration`: from Sub-function B
- Token usage is summed across both sub-calls

**How it becomes embedding text (`__str__`):** Concatenates production keywords + sources of inspiration + production mediums as a comma-separated list.

---

### 4.7 Reception Metadata

**Purpose:** Capture what audiences and critics think about the movie — consensus praise and complaints. This powers queries like "well-acted drama" or "poorly written dialogue."

**System prompt key directive:** "Extract high-relevance attributes exemplifying the audience reception."

**Input fields provided:**
- `title`
- `reception_summary` — externally generated summary of audience opinion
- `audience_reception_attributes` — key attributes with sentiment labels
- `featured_reviews` (up to 5) — user-written reviews

Note: This function runs in Wave 1 (no dependency on plot_events), so it does NOT receive `plot_synopsis`.

**Output schema (`ReceptionMetadata`):**

| Field | Type | Description |
|-------|------|-------------|
| `new_reception_summary` | str | 2-3 sentences — concise summary of what viewers thought (not a rating, but qualitative opinion) |
| `praise_attributes` | list[str] (0-4) | Short tag-like phrases for what audiences enjoyed. E.g., "masterful cinematography", "compelling lead performance" |
| `complaint_attributes` | list[str] (0-4) | Short tag-like phrases for what audiences disliked. E.g., "predictable plot", "wooden dialogue" |

**How it becomes embedding text:** `new_reception_summary` (lowercased) + `"Praises: ..."` + `"Complaints: ..."` as separate lines.

---

## 5. Vector Text Generation: From Metadata to Embeddable Text <a name="vector-text-generation"></a>

After LLM metadata generation, each movie's 7 metadata objects are transformed into 8 text strings (one per vector space) that will be embedded. The 8th vector space ("anchor") is a reduced holistic fingerprint built from a narrow set of movie-wide fields plus selected metadata.

Text generation functions are in `movie_ingestion/final_ingestion/vector_text.py`.

### The 8 Vector Spaces

| Vector Space | Text Generator Function | Primary Content |
|-------------|------------------------|-----------------|
| **anchor** | `create_anchor_vector_text()` | Lean holistic fingerprint — labeled title/original title, identity pitch/overview, genre signatures, themes, emotional palette, key draws, maturity summary, and reception summary. |
| **plot_events** | `create_plot_events_vector_text()` | `PlotEventsMetadata.__str__()` — plot summary + setting + character descriptions |
| **plot_analysis** | `create_plot_analysis_vector_text()` | `PlotAnalysisMetadata.__str__()` + genre subset + plot keywords |
| **viewer_experience** | `create_viewer_experience_vector_text()` | `ViewerExperienceMetadata.__str__()` — all terms + negations from non-skipped sections |
| **watch_context** | `create_watch_context_vector_text()` | `WatchContextMetadata.__str__()` — all terms from all 4 sections |
| **narrative_techniques** | `create_narrative_techniques_vector_text()` | `NarrativeTechniquesMetadata.__str__()` — all terms from all 11 sections |
| **production** | `create_production_vector_text()` | Base movie production info (countries, companies, locations, languages, decade, budget) + production keywords + sources + cast + maturity rating |
| **reception** | `create_reception_vector_text()` | Labeled reception summary + praise/complaint attributes + deterministic major award wins by ceremony |

### Anchor Vector — The Lean Holistic One

The anchor vector is special. It's designed to provide good recall for general
"movie as a whole" queries without duplicating structured facts or specialized
spaces. It emits only labeled lines, in stable order:

```
title: <tmdb title>
original_title: <imdb original title, only when different>
identity_pitch: <plot-analysis elevator pitch>
identity_overview: <plot-analysis generalized overview, or imdb overview fallback>
genre_signatures: <plot-analysis genre signatures>
themes: <plot-analysis thematic concept labels>
emotional_palette: <viewer-experience positive terms>
key_draws: <watch-context key movie feature draws>
maturity_summary: <maturity reasoning or rating-derived semantic summary>
reception_summary: <reception summary>
```

Deliberately excluded from anchor: keywords, source material, franchise
position, languages, decade, budget/box office, awards, reception tier, and
other structured/filterable facts that have better homes elsewhere.

### Production Vector — Also Composite

The production vector text draws from both `ProductionMetadata` (LLM-generated) and base movie fields:

```
# Production:
<countries, companies, filming locations (lowercased)>
<languages (lowercased)>
<decade bucket (lowercased)>
<budget bucket (lowercased)>
<production keywords, production mediums, sources of inspiration>

# Cast and Characters:
<directors, writers, producers, composers, top actors>
<character names>
<maturity rating> maturity rating
```

### `__str__()` Methods: What Gets Embedded vs. What Doesn't

Each Pydantic model has a `__str__()` method that controls what text goes into the embedding. Critically, **justification fields are NOT included** in the embedding text. Justifications exist only for internal LLM reasoning consistency — they're never embedded.

For `ViewerExperienceMetadata`, sections with `should_skip=True` are excluded from the `__str__()` output.

---

## 6. Embedding <a name="embedding"></a>

**Model:** `text-embedding-3-small` (OpenAI)
**Dimensions:** 1536
**API:** `openai.embeddings.create()`

### Single Movie Ingestion
For a single movie, all 8 text representations are embedded in one batched API call:
```python
embeddings = await generate_vector_embedding(model="text-embedding-3-small", text=filtered_texts)
```

### Batch Ingestion (Production)
For batch ingestion (`ingest_movies_to_qdrant_batched()`), all N×8 texts for an entire batch (default 50 movies = up to 400 texts) are embedded in a single API call, minimizing network round-trips.

---

## 7. Database Ingestion <a name="database-ingestion"></a>

### PostgreSQL (movie_card + lexical data)

**Function:** `ingest_movie()` in `db/ingest_movie.py`

Runs within a single transaction for atomicity:

1. **movie_card upsert** — stores: tmdb_id, title, poster_url, release_ts, runtime_minutes, maturity_rank, genre_ids, watch_offer_keys, audio_language_ids, imdb_vote_count, reception_score, title_token_count, budget_bucket
2. **lexical data upsert** — populates posting tables for:
   - Title tokens (normalized words from the title)
   - People (actors, directors, writers, composers, producers)
   - Characters (character names)
   - Studios (production companies)

These posting tables power the lexical search channel.

### Qdrant (vector embeddings)

**Function:** `ingest_movie_to_qdrant()` or `ingest_movies_to_qdrant_batched()`

Each movie becomes a single Qdrant point containing:

**Named vectors (8):**
- `anchor`, `plot_events`, `plot_analysis`, `viewer_experience`, `watch_context`, `narrative_techniques`, `production`, `reception`

**Hard-filter payload (for Qdrant filtering at search time):**
- `release_ts` — Unix timestamp
- `runtime_minutes` — integer
- `maturity_rank` — ordinal integer (or null for unrated)
- `genre_ids` — list of genre integers
- `watch_offer_keys` — list of streaming service offer keys
- `audio_language_ids` — list of language integers

**Point ID:** `tmdb_id` (integer)

The payload is kept minimal — only fields needed for hard filtering during search. Full display metadata lives in PostgreSQL.

---

## 8. Error Handling <a name="error-handling"></a>

### LLM Generation
- **Plot events failure → entire movie fails.** RuntimeError is raised. Without a plot synopsis, Wave 2 can't run.
- **Any other metadata failure → graceful degradation.** The metadata key is set to `None`, token usage is omitted, and the movie proceeds with partial metadata.
- Each function raises `ValueError` if the OpenAI `parse()` call fails to produce a valid response matching the schema.

### Embedding
- If the batched embedding call fails, all movies in that batch are marked as failed.

### Database Ingestion
- PostgreSQL ingestion is wrapped in a transaction — if any step fails, the entire transaction is rolled back (no partial data).
- Qdrant batch upsert failures are logged and counted but don't block subsequent batches.

---

## 9. Key File Reference <a name="key-file-reference"></a>

| File | Purpose |
|------|---------|
| `implementation/llms/vector_metadata_generation_methods.py` | All 7 generation functions + parallel orchestration (`generate_llm_metadata()`) |
| `implementation/prompts/vector_metadata_generation_prompts.py` | All 8 system prompts (one per LLM call) |
| `implementation/llms/generic_methods.py` | `generate_openai_response()` — the base LLM call wrapper using `chat.completions.parse()` |
| `implementation/classes/schemas.py` | All Pydantic output schemas (PlotEventsMetadata, PlotAnalysisMetadata, etc.) |
| `implementation/classes/movie.py` | `BaseMovie` model with computed methods (release_decade_bucket, reception_tier, etc.) |
| `movie_ingestion/final_ingestion/vector_text.py` | 8 text generation functions (one per vector space) |
| `db/ingest_movie.py` | PostgreSQL + Qdrant ingestion functions |
| `implementation/classes/enums.py` | `VectorName` enum defining the 8 named vector spaces |
