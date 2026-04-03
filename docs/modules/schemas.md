# schemas/ — Shared Pydantic Models & Data Types

Top-level package for shared Pydantic models, data classes, and enums
that are importable across `db/`, `api/`, and `movie_ingestion/`. Created
to give cross-cutting types a stable home as `implementation/` is phased
out.

## What This Module Does

Defines canonical types for:
- LLM output schemas used in both generation and embedding pipelines
- The `Movie` tracker-backed data loader for ingestion-time access
- Shared enums (`MetadataType`) and utility types (`MultiLineList`)
- MovieInputData for tracker loading (generation-pipeline entry point)

## Key Files

| File | Purpose |
|------|---------|
| `metadata.py` | `EmbeddableOutput` base class + 8 `*Output` schema classes (one per generation type). Each implements `embedding_text()` returning normalized text for vector embedding. Legacy `__str__()` methods are retained for backward compatibility. Moved from `movie_ingestion/metadata_generation/schemas.py`. |
| `movie.py` | `Movie`, `TMDBData`, `IMDBData` Pydantic models + `Movie.from_tmdb_id()` loader. Joins `tmdb_data`, `imdb_data`, and `generated_metadata` from tracker.db in one query and returns a fully typed object with parsed metadata. |
| `enums.py` | `MetadataType` enum (one value per generation type). |
| `data_types.py` | `MultiLineList` — a constrained list type used in generation schemas. |
| `movie_input.py` | `MovieInputData` dataclass + `load_movie_input_data()` — loads raw tracker data into the form consumed by generator prompt builders. |

## Boundaries

- **In scope**: Shared data structures with no business logic.
  No database access (except `Movie.from_tmdb_id()`, which is a
  pure loader). No LLM calls.
- **Out of scope**: Generation-pipeline-specific types that are not
  shared across modules. These remain in
  `movie_ingestion/metadata_generation/inputs.py`:
  `build_custom_id`, `parse_custom_id`, `WAVE1_TYPES`, `WAVE2_TYPES`,
  `ConsolidatedInputs`, `SkipAssessment`, `build_user_prompt`,
  `Wave1Outputs`, `load_wave1_outputs`.

## Key Types

**`EmbeddableOutput`** (`metadata.py`): Abstract base class for all 8
`*Output` schemas. Declares `embedding_text() -> str` as the canonical
contract for producing normalized vector embedding text. Replaces the
previous `__str__()`-based convention, which was implicit and
inconsistently applied.

**`*Output` schemas** (`metadata.py`): 8 Pydantic models for LLM
structured output — `PlotEventsOutput`, `ReceptionOutput`,
`PlotAnalysisOutput`, `ViewerExperienceOutput`, `WatchContextOutput`,
`NarrativeTechniquesOutput`, `ProductionKeywordsOutput`,
`SourceOfInspirationOutput`. Each `embedding_text()` produces the text
for its corresponding vector space. `WithJustificationsOutput` variants
exist for evaluation (identical `embedding_text()` output to base variant).

**`Movie`** (`movie.py`): Central ingestion-time data object. Loads a
fully typed row from `tracker.db` including parsed IMDB JSON columns,
TMDB review JSON, provider-key blob unpacking, and all 8 generated
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

Default tracker DB path is resolved from the file's own location
(`schemas/movie.py` → repo root → `ingestion_data/tracker.db`),
making it stable across notebooks, shells, and other entry points.

**`TMDBData`** / **`IMDBData`** (`movie.py`): Typed sub-models for
the TMDB and IMDB data sections of a `Movie`. Source column names are
preserved exactly.

## Interaction with Other Modules

- `movie_ingestion/metadata_generation/` — all 8 generators and all
  batch pipeline modules import `*Output` schemas from `schemas.metadata`
  and `MetadataType` from `schemas.enums`.
- `movie_ingestion/final_ingestion/vector_text.py` — all vector text
  functions accept `Movie` and call `embedding_text()` on metadata
  objects.
- `movie_ingestion/metadata_generation/inputs.py` — imports
  `MovieInputData` from `schemas.movie_input`.
- `implementation/classes/schemas.py` (search-side schemas) — remains
  separate. Will need alignment with generation-side schemas before
  deployment.

## Gotchas

- `EmbeddableOutput.embedding_text()` is abstract — any new `*Output`
  schema that does not implement it will raise `TypeError` at
  instantiation (not at import time).
- `Movie.from_tmdb_id()` applies a narrow compatibility normalization
  for known legacy key drift in stored metadata JSON:
  `justification` → `evidence_basis`, and obsolete
  source-of-inspiration evidence fields. This lets existing tracker
  rows validate against the current schema without requiring a
  database migration.
- Generation-side schemas (`schemas/metadata.py`) intentionally diverge
  from search-side schemas (`implementation/classes/schemas.py`). When
  deploying, align the search-side schemas.
- `schemas/testing.ipynb` is a manual inspection notebook — not part
  of the test suite. Run via Jupyter with the project root in `sys.path`
  or use the `find_project_root()` helper already in the first cell.
