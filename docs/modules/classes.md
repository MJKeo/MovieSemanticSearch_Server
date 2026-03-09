# implementation/classes/ — Data Models & Enums

Pydantic data models, enums, and reference data used across the
entire codebase. This is a pure data-definition module with no
business logic.

## What This Module Does

Defines the canonical data structures for movies, query
understanding outputs, vector metadata, and all enum types used
in scoring and filtering.

## Key Files

| File | Purpose |
|------|---------|
| `movie.py` | `BaseMovie` — the core movie data model with all fields from TMDB + IMDB + LLM-generated metadata. Includes era-aware budget classification. |
| `schemas.py` | All Pydantic models: LLM output schemas (plot events, plot analysis, viewer experience, watch context, narrative techniques, production, reception), query understanding responses (entities, channel weights, metadata preferences, vector subqueries/weights), and metadata filter types. |
| `enums.py` | All enums: `MaturityRating` (with maturity_rank), `StreamingAccessType` (with type_id), `VectorName`/`VectorCollectionName` (8 vector spaces), `RelevanceSize` (NOT_RELEVANT/SMALL/MEDIUM/LARGE), `Genre`, `EntityCategory`, `BudgetSize`, match operation enums. |
| `languages.py` | Language catalog with IDs for audio language filtering. |
| `watch_providers.py` | Streaming service catalog with provider IDs. |

## Boundaries

- **In scope**: Data structures, validation, enum definitions,
  reference data catalogs.
- **Out of scope**: No database access, no API calls, no scoring
  logic. Models are pure data containers.

## Key Models

**BaseMovie** (`movie.py`): Central movie representation with fields
from all pipeline stages — TMDB basics (title, date, duration),
IMDB enrichment (credits, keywords, reviews, maturity), and 7
LLM-generated metadata objects. Used during ingestion to construct
vector text and populate databases.

**LLM Metadata Schemas** (`schemas.py`): Seven metadata types
generated at ingestion time, each feeding a specific vector space:
- `PlotEventsMetadata` → plot_events vector
- `PlotAnalysisMetadata` → plot_analysis vector
- `ViewerExperienceMetadata` → viewer_experience vector
- `WatchContextMetadata` → watch_context vector
- `NarrativeTechniquesMetadata` → narrative_techniques vector
- `ProductionMetadata` → production vector (via sub-schemas)
- `ReceptionMetadata` → reception vector

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
