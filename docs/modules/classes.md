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
| `enums.py` | All enums: `MaturityRating` (with maturity_rank), `StreamingAccessType` (with type_id), `VectorName`/`VectorCollectionName` (8 vector spaces, backed by Qdrant), `RelevanceSize` (NOT_RELEVANT/SMALL/MEDIUM/LARGE), `Genre`, `EntityCategory`, `BudgetSize`, match operation enums. |
| `languages.py` | Language catalog with IDs for audio language filtering. |
| `watch_providers.py` | Streaming service catalog with provider IDs. |

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
  ordinal comparison (G=1, PG=2, PG-13=3, R=4, NC-17=5).
- Watch provider keys encode both provider and access method in
  a single uint32: `provider_id << 2 | method_id`.
- `VectorCollectionName` docstring references Qdrant (not ChromaDB,
  which was the previous vector store).
