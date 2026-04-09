# schemas/ — Shared Pydantic Models & Data Types

Top-level package for shared Pydantic models, data classes, and enums
that are importable across `db/`, `api/`, and `movie_ingestion/`. Created
to give cross-cutting types a stable home as `implementation/` is phased
out.

## What This Module Does

Defines canonical types for:
- LLM output schemas used in both generation and embedding pipelines
- The `Movie` tracker-backed data loader for ingestion-time access
- Shared enums (`MetadataType`, `SourceMaterialType`) and utility types (`MultiLineList`)
- MovieInputData for tracker loading (generation-pipeline entry point)

## Key Files

| File | Purpose |
|------|---------|
| `metadata.py` | `EmbeddableOutput` base class + 9 `*Output` schema classes (one per embeddable generation type) + `ConceptTagsOutput` / `TagEvidence` (non-embeddable concept tag classification). Each `EmbeddableOutput` subclass implements `embedding_text()` returning normalized text for vector embedding. `ConceptTagsOutput` produces integer concept_tag_ids via `all_concept_tag_ids()`, not embedding text. Legacy `__str__()` methods are retained for backward compatibility. Class docstrings are written as `#` comment blocks above each class — not as Python docstrings — to prevent them from leaking into the JSON schema payload sent to the LLM via `model_json_schema()`. |
| `movie.py` | `Movie`, `TMDBData`, `IMDBData` Pydantic models + `Movie.from_tmdb_id()` single-movie loader and `Movie.from_tmdb_ids()` batch loader. Joins `tmdb_data`, `imdb_data`, and `generated_metadata` from tracker.db in one query and returns fully typed objects with parsed metadata. |
| `enums.py` | `MetadataType` StrEnum (one value per generation type, 10 total including `SOURCE_MATERIAL_V2` and `CONCEPT_TAGS`), `SourceMaterialType` enum (10 values with stable integer IDs for GIN-indexed storage), and `ConceptTag` IntEnum (concept tag IDs grouped by category via `CONCEPT_TAG_CATEGORIES` dict). |
| `data_types.py` | `MultiLineList` — a constrained list type used in generation schemas. |
| `movie_input.py` | `MovieInputData` dataclass + `load_movie_input_data()` — loads raw tracker data into the form consumed by generator prompt builders. |

## Boundaries

- **In scope**: Shared data structures with no business logic.
  No database access (except `Movie.from_tmdb_id()` / `Movie.from_tmdb_ids()`,
  which are pure loaders). No LLM calls.
- **Out of scope**: Generation-pipeline-specific types that are not
  shared across modules. These remain in
  `movie_ingestion/metadata_generation/inputs.py`:
  `build_custom_id`, `parse_custom_id`, `WAVE1_TYPES`, `WAVE2_TYPES`,
  `ConsolidatedInputs`, `SkipAssessment`, `build_user_prompt`,
  `Wave1Outputs`, `load_wave1_outputs`, `extract_narrative_technique_terms`,
  `load_plot_analysis_output`.

## Key Types

**`EmbeddableOutput`** (`metadata.py`): Abstract base class for 9
embeddable `*Output` schemas. Declares `embedding_text() -> str` as the
canonical contract for producing normalized vector embedding text.
Replaces the previous `__str__()`-based convention, which was implicit
and inconsistently applied. `ConceptTagsOutput` intentionally does not
subclass `EmbeddableOutput` — it produces integer IDs, not embedding text.

**`*Output` schemas** (`metadata.py`): 9 embeddable Pydantic models for LLM
structured output — `PlotEventsOutput`, `ReceptionOutput`,
`PlotAnalysisOutput`, `ViewerExperienceOutput`, `WatchContextOutput`,
`NarrativeTechniquesOutput`, `ProductionKeywordsOutput`,
`SourceOfInspirationOutput`, `SourceMaterialV2Output`. Each
`embedding_text()` produces the text for its corresponding vector space.
`WithJustificationsOutput` variants exist for evaluation (identical
`embedding_text()` output to base variant).

**`ConceptTagsOutput`** (`metadata.py`): Multi-label binary classification
of concept tags by category. Contains 7 category fields (each a
`list[TagEvidence]`): `narrative_structure`, `plot_archetypes`, `settings`,
`characters`, `endings`, `experiential`, `content_flags`. A
`validate_tag_categories` model validator ensures each category only
contains tags from its allowed set (defined in `CONCEPT_TAG_CATEGORIES`).
`all_concept_tag_ids()` extracts a sorted, deduplicated `list[int]` for
storage in `movie_card.concept_tag_ids`.

**`ConceptTag`** (`enums.py`): IntEnum with a `concept_tag_id` attribute
per member. Grouped into 7 categories via `CONCEPT_TAG_CATEGORIES` dict
(maps category field name → frozenset of allowed `ConceptTag` values).

`SourceMaterialV2Output` outputs a `list[SourceMaterialType]`. An empty
list signals original screenplay — no enum value is assigned for that case.
`embedding_text()` returns `""` for empty lists.

**`SourceMaterialType`** (`enums.py`): `(str, Enum)` with both string
values (for Pydantic JSON schema enum constraints in LLM structured output)
and stable integer IDs (for future `movie_card.source_material_type_ids`
GIN-indexed storage). 10 values: `NOVEL_ADAPTATION`, `SHORT_STORY_ADAPTATION`,
`STAGE_ADAPTATION`, `TRUE_STORY`, `BIOGRAPHY`, `COMIC_ADAPTATION`,
`FOLKLORE_ADAPTATION`, `VIDEO_GAME_ADAPTATION`, `REMAKE`, `TV_ADAPTATION`.
See `search_improvement_planning/source_material_type_enum.md` for boundary notes.

**`Movie`** (`movie.py`): Central ingestion-time data object. Loads a
fully typed row from `tracker.db` including parsed IMDB JSON columns,
TMDB review JSON, provider-key blob unpacking, and all generated
metadata objects. Includes helper methods:
- `title_with_original()` — "Title (Original Title)" or just "Title"
- `maturity_text_short()` — IMDB reasoning prose or MPA description fallback
- `deduplicated_genres()` — genre_signatures + IMDB genres, substring-deduped
- `reception_score()` / `reception_tier()` — blended IMDB + Metacritic score and tier label
- `is_animation()` — binary genre check
- `production_text()` — labeled format with 3-location cap
- `languages_text()` — labeled primary + additional languages
- `release_decade_bucket()` — semantic era label
- `budget_bucket_for_era()` — era-adjusted budget classification
- `resolved_box_office_revenue()` — IMDB worldwide gross when positive, falls back to TMDB revenue. Zero and negative values treated as missing.

**Ingestion-compatible methods** added to `Movie` (mirrors `BaseMovie`
interface for the ingestion pipeline):
- `release_ts()` — unix timestamp for Postgres/Qdrant payload
- `maturity_rating_and_rank()` — tuple of `(MaturityRating, int)` for payload
- `normalized_title_tokens()` — deduplicated token set from both primary and
  original_title (original_title tokens merged after primary; first-seen order
  preserved). Foreign-language films are now searchable by original title.
- `genre_ids()` — list of integer genre IDs
- `watch_offer_keys()` — list of decoded uint32 provider keys (pre-decoded in TMDBData)
- `audio_language_ids()` — list of integer language IDs

`Movie.from_tmdb_ids(tmdb_ids, tracker_db_path?)` is the batch loader: executes
one SQLite query with `json_each()` for N movies instead of N individual queries.
Reuses the same `_QUERY` column definitions and `_build_*` parsers as the
single-movie loader.

Default tracker DB path is resolved from the file's own location
(`schemas/movie.py` → repo root → `ingestion_data/tracker.db`),
making it stable across notebooks, shells, and other entry points.

**`TMDBData`** (`movie.py`): Typed sub-model for TMDB data. Now includes
`revenue: int | None` (raw TMDB revenue in USD; 0 stored as None) in addition
to the existing `has_revenue` boolean. Also includes `collection_name: str | None`
(the `belongs_to_collection.name` from TMDB, needed for franchise generation).

**`IMDBData`** (`movie.py`): Typed sub-model for IMDB data. Now includes
`awards: list[AwardNomination]` (nominations from 12 in-scope ceremonies,
parsed from JSON) and `box_office_worldwide: int | None` (IMDB worldwide
lifetime gross in USD; non-USD flagged as 0). `AwardNomination` sub-model
has fields: `ceremony`, `award_name`, `category` (nullable for festival
grand prizes), `outcome`, `year`.

## Interaction with Other Modules

- `movie_ingestion/metadata_generation/` — all generators and all
  batch pipeline modules import `*Output` schemas from `schemas.metadata`
  and `MetadataType` from `schemas.enums`.
- `movie_ingestion/final_ingestion/vector_text.py` — all vector text
  functions accept `Movie` and call `embedding_text()` on metadata
  objects.
- `movie_ingestion/final_ingestion/ingest_movie.py` — all ingestion
  functions accept `Movie` exclusively. `BaseMovie` is no longer used
  in either Postgres or Qdrant ingestion paths.
- `movie_ingestion/metadata_generation/inputs.py` — imports
  `MovieInputData` from `schemas.movie_input`.
- `implementation/classes/schemas.py` (search-side schemas) — remains
  separate. Will need alignment with generation-side schemas before
  deployment.

## Gotchas

- `EmbeddableOutput.embedding_text()` is abstract — any new `*Output`
  schema that does not implement it will raise `TypeError` at
  instantiation (not at import time).
- **Class docstrings are deliberately avoided in `metadata.py` and
  `enums.py`.** Pydantic's `model_json_schema()` propagates Python
  docstrings into JSON schema `description` fields, which OpenAI's
  `to_strict_json_schema()` then sends to the LLM. Use `#` comment
  blocks above classes instead. Field-level `Field(description=...)`
  is intentional and kept.
- `Movie.from_tmdb_id()` applies a narrow compatibility normalization
  for known legacy key drift in stored metadata JSON:
  `justification` → `evidence_basis`, and obsolete
  source-of-inspiration evidence fields. This lets existing tracker
  rows validate against the current schema without requiring a
  database migration.
- Generation-side schemas (`schemas/metadata.py`) intentionally diverge
  from search-side schemas (`implementation/classes/schemas.py`). When
  deploying, align the search-side schemas.
- `schemas/testing.ipynb` has been deleted — use the unit tests or a
  notebook in `implementation/notebooks/` for manual inspection.
- `normalized_title_tokens()` on `Movie` merges tokens from both
  `title` and `original_title`. Tests that only cover primary title
  tokens may need updating.
- `SourceMaterialType` uses `(str, Enum)` pattern (not `StrEnum`) because
  it needs a secondary `source_material_type_id: int` attribute per member.
  `__new__` carries both values.
