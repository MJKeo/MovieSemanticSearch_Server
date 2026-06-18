# implementation/classes/ — Data Models & Enums

Pydantic data models, enums, and reference data used across the
entire codebase. This is a pure data-definition module with no
business logic.

**Note**: Cross-cutting types that need to be shared between
`db/`, `api/`, and `movie_ingestion/` are migrating to the new
top-level `schemas/` package. `implementation/` is being phased out
as the canonical home for shared types. See `docs/modules/schemas.md`.

## What This Module Does

Defines the canonical data structures for movies, query
understanding outputs, vector metadata, and all enum types used
in scoring and filtering.

## Key Files

| File | Purpose |
|------|---------|
| `movie.py` | `BaseMovie` — the core movie data model with all fields from TMDB + IMDB + LLM-generated metadata. Includes era-aware budget classification. |
| `schemas.py` | All Pydantic models: LLM output schemas (plot events, plot analysis, viewer experience, watch context, narrative techniques, production, reception), query understanding responses (entities, channel weights, metadata preferences, vector subqueries/weights), and metadata filter types. |
| `enums.py` | All enums: `MaturityRating` (with maturity_rank + alias resolution via `_missing_`), `StreamingAccessType` (with type_id), `VectorName`/`VectorCollectionName` (8 vector spaces, backed by Qdrant), `RelevanceSize` (NOT_RELEVANT/SMALL/MEDIUM/LARGE), `Genre` (with `from_id()` backed by `_GENRE_BY_ID` reverse dict — used by `/movie_details` to map `genre_ids` INT[] to display names), `EntityCategory`, `BudgetSize`, match operation enums. |
| `languages.py` | `Language` enum with stable integer IDs (`language_id`) plus ISO 639-1 codes (`iso_code`, ~140/334 covered), a `from_iso()` classmethod, and a `from_id()` classmethod backed by a module-level `_LANGUAGE_BY_ID` reverse dict. The LLM-facing metadata schema speaks ISO codes; the executor maps back to `language_id` via `from_iso()`. `from_id()` is used by the `/movie_details` endpoint to map `audio_language_ids` INT[] back to display names. Long-tail languages without a 639-1 code keep `iso_code = None` and are unreachable from the LLM schema by design. `LANGUAGE_BY_NORMALIZED_NAME` remains for non-LLM display-name lookup. |
| `watch_providers.py` | Streaming service catalog with provider IDs. |
| `countries.py` | `Country` enum with stable integer IDs (`country_id`) plus ISO 3166-1 alpha-2 codes (`iso_code`, ~249/262 covered) and a `from_iso()` classmethod. Same ISO-code contract as `languages.py` — LLM-facing schema uses alpha-2, retired entities (Yugoslavia, USSR, etc.) keep `iso_code = None`. `country_from_string()` and `_COUNTRY_ALIASES` remain for non-LLM display-name lookup (used by `schemas/movie.py` for IMDB country names). Imported by `schemas/movie.py`, `schemas/metadata_translation.py`, and `search_v2/endpoint_fetching/metadata_query_execution.py`. |
| `overall_keywords.py` | `OverallKeyword` enum (225 values). Extended with `from_id()` backed by `_KEYWORD_BY_ID` reverse dict, and `keyword_names_from_ids(ids, exclude_names)` helper that returns display names for a list of integer IDs, dropping unknown IDs and optionally skipping names duplicated by the genres list (hyphen-insensitive dedup via `_dedup_key`). Used by `/movie_details` to surface keyword tags. Imported by `schemas/movie.py` and `schemas/unified_classification.py` for the V2 keyword endpoint vocabulary. |

## Boundaries

- **In scope**: Data structures, validation, enum definitions,
  reference data catalogs.
- **Out of scope**: No database access, no API calls, no scoring
  logic. Models are pure data containers.

## Key Models

**BaseMovie** (`movie.py`): Legacy movie representation with fields
from all pipeline stages — TMDB basics (title, date, duration),
IMDB enrichment (credits, keywords, reviews, maturity), and 7
LLM-generated metadata objects. No longer used in the active
ingestion pipeline (replaced by `Movie` in `schemas/movie.py`
per ADR-060). Only remaining use is the `base_movie_factory`
test fixture in `unit_tests/conftest.py`. The legacy consumers
(`implementation/vectorize.py`, `implementation/scraping/`) have
been removed from the codebase.

**LLM Metadata Schemas** (`schemas.py`): Seven metadata types
generated at ingestion time, each feeding a specific vector space.
These are the **search-side** schemas consumed by the search pipeline
when reading metadata from Qdrant:
- `PlotEventsMetadata` → plot_events vector
- `PlotAnalysisMetadata` → plot_analysis vector
- `ViewerExperienceMetadata` → viewer_experience vector
- `WatchContextMetadata` → watch_context vector
- `NarrativeTechniquesMetadata` → narrative_techniques vector
- `ProductionMetadata` → production vector (**legacy combined schema**;
  the active generation-side production-vector input is now
  `ProductionTechniquesOutput` in `schemas/metadata.py`, paired with scraped
  filming locations. `ProductionKeywordsOutput` is a historical predecessor,
  and `SourceOfInspirationOutput` no longer feeds the production vector.
  Search-side schema still needs alignment before deployment.)
- `ReceptionMetadata` → reception vector

The generation-side counterparts live in `schemas/metadata.py` and
diverge intentionally from these search-side schemas. When deploying,
align the search-side schemas to match generation outputs.

**Query Understanding Schemas** (`schemas.py`): Four response types
from the search-time LLM DAG:
- `ExtractedEntitiesResponse` — actors, directors, franchises, characters
- `ChannelWeightsResponse` — lexical/vector/metadata relevance
- `MetadataPreferencesResponse` — genre, date, runtime, etc.
- `VectorSubqueriesResponse` + `VectorWeightsResponse` — per-space queries and weights

## Gotchas

- Several schema fields exist for LLM chain-of-thought but are
  NOT included in vector text: `justification` fields on all
  GenericTermsSection/ViewerExperienceSection models,
  `explanation_and_justification` on CoreConcept/MajorTheme/
  MajorLessonLearned, `role` on MajorCharacter,
  `character_name`/`arc_transformation_description` on CharacterArc.
- `MaturityRating` enum includes `maturity_rank` property for
  ordinal comparison (G=1, PG=2, PG-13=3, R=4, NC-17=5). Extended with
  per-member `aliases` sets and a `_missing_` hook that resolves TV,
  legacy, and foreign-cert strings (TV-MA→R, TV-14→PG-13, GP→PG, X→NC-17,
  etc.) to canonical members. `from_string_with_default` logs a warning
  for non-empty unresolved values and is the safe construction path for
  ingestion and search. `_build_maturity_alias_map` builds the reverse
  lookup once at module load. Display methods (`maturity_text_short`,
  `maturity_guidance_text`) now resolve through the enum before branching
  on `UNRATED`, so TV-MA movies display the R description.
- Watch provider keys encode both provider and access method in
  a single uint32: `provider_id << 4 | method_id`.
- `VectorCollectionName` docstring references Qdrant (not ChromaDB,
  which was the previous vector store).
