# [056] — Movie: Tracker-Backed Schema Loader for Ingestion-Time Access

## Status
Active

## Context

The ingestion pipeline (Stage 7+) needs to load a fully-typed movie object
from the SQLite tracker in order to generate vector text and ingest into
Postgres/Qdrant. Previously, `BaseMovie` in `implementation/classes/movie.py`
served this role, but it required callers to construct it manually from
multiple tracker queries and handle JSON deserialization themselves.

The vector text functions (`final_ingestion/vector_text.py`) needed a
single object type that could provide typed access to TMDB data, IMDB data,
and all 8 generated metadata objects without per-caller boilerplate.

## Decision

Add `Movie`, `TMDBData`, and `IMDBData` Pydantic models to `schemas/movie.py`
with a `Movie.from_tmdb_id(tmdb_id, tracker_db_path?)` class method that:
- Executes one joined SQLite query with aliased columns (tmdb_data + imdb_data
  + generated_metadata)
- Parses IMDB JSON TEXT columns, TMDB review JSON, and provider-key blobs
  into typed Python values
- Parses all 8 generated metadata columns into their `schemas.metadata`
  output models
- Applies a narrow compatibility normalization for known legacy key drift
  (`justification` → `evidence_basis`; obsolete source-of-inspiration
  evidence fields) so existing tracker rows validate against the current
  schema without a DB migration

Default tracker DB path is resolved from `schemas/movie.py`'s own location
(repo root → `ingestion_data/tracker.db`) so notebooks and shells both
work without passing an explicit path.

Helper methods on `Movie` encapsulate derived signals used by vector text
functions: `title_with_original()`, `maturity_text_short()`,
`deduplicated_genres()`, `reception_score()`, `reception_tier()`,
`is_animation()`, `production_text()`, `languages_text()`,
`release_decade_bucket()`, `budget_bucket_for_era()`.

## Alternatives Considered

1. **Continue using `BaseMovie`**: `BaseMovie` requires callers to pass all
   fields at construction time — it is not a loader. Adapting it to load
   from the tracker would mix data-model concerns with persistence concerns,
   and `BaseMovie` already has a different field layout from the tracker
   columns.

2. **Load fields lazily (on-demand SQLite queries per attribute)**: Simpler
   initially but causes N+1 query patterns when vector text functions access
   multiple fields. One joined query is more predictable.

3. **Return a plain dict from a loader function**: Loses static typing and
   makes IDE navigation and test assertions harder. Pydantic validation also
   catches schema drift between the stored JSON and the current output models.

## Consequences

- All vector text functions accept `Movie` instead of `BaseMovie` or
  individual output models — a single parameter type across all 8 functions.
- `Movie` is the canonical ingestion-time data object; `BaseMovie` is
  effectively legacy for ingest use.
- The compatibility normalization for legacy key drift means `Movie` silently
  upgrades old tracker rows. If the schema diverges further, additional
  normalization entries will be needed here.
- `schemas/testing.ipynb` provides a manual inspection entry point for
  loading a single movie and examining all fields grouped by data source.

## References

- `schemas/movie.py`
- `docs/modules/schemas.md`
- `movie_ingestion/final_ingestion/vector_text.py`
- ADR-055 (schemas/ package — where Movie lives)
- ADR-023 (IMDB data SQLite migration — the source table Movie reads from)
