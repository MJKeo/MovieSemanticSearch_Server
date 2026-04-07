# [060] — Complete Migration from BaseMovie to Movie in Ingestion Pipeline

## Status
Active

## Context

The ingestion pipeline had two parallel movie type hierarchies:
- `BaseMovie` (`implementation/classes/movie.py`): a flat Pydantic model
  constructed manually by callers; used since the original pipeline.
- `Movie` (`schemas/movie.py`): tracker-backed loader with nested `TMDBData` /
  `IMDBData` sub-models; introduced in ADR-056 for vector text generation.

`vector_text.py` had already migrated to `Movie`. The Qdrant ingestion functions
in `ingest_movie.py` then migrated, but Postgres ingestion functions still used
`BaseMovie`. Helper methods required by ingestion (`release_ts`,
`maturity_rating_and_rank`, `normalized_title_tokens`, `genre_ids`,
`watch_offer_keys`, `audio_language_ids`) only existed on `BaseMovie`.

The parallel hierarchy created a type mismatch: calling code had to hold both
a `Movie` (for vector text) and a `BaseMovie` (for Postgres ingestion) from the
same tracker row. `BaseMovie` also used `getattr()` on flat fields — a pattern
incompatible with `Movie`'s nested structure.

## Decision

1. Add all 6 missing ingestion-compatible methods to `Movie` in `schemas/movie.py`,
   with implementations that delegate to `movie.tmdb_data` / `movie.imdb_data`.
2. Migrate all remaining `BaseMovie`-typed functions in `ingest_movie.py` to
   accept `Movie`. Update all field access from flat `getattr(movie, "field")`
   to nested `movie.tmdb_data.field` / `movie.imdb_data.field`.
3. Remove now-unused `BaseMovie` imports and defensive `isinstance` guards
   from `ingest_movie.py`.
4. Add `Movie.from_tmdb_ids()` batch loader that executes one SQLite query for
   N movies using `json_each()`, reusing the existing `_QUERY` column definitions
   and `_build_*` parsers.

`BaseMovie` is retained in `implementation/classes/movie.py` — it is still used
by the `base_movie_factory` test fixture (`unit_tests/conftest.py`) and legacy
files (`implementation/vectorize.py`, `implementation/scraping/gather_data.py`).
It is not imported by the search-side pipeline (`db/`).

## Alternatives Considered

1. **Keep BaseMovie as the canonical ingestion type**: Would require keeping two
   separate loaders in sync. `BaseMovie` is a construction-time model (callers
   build it), not a loader — adapting it to load from tracker columns would mix
   data-model and persistence concerns.

2. **Merge BaseMovie and Movie into one type**: Premature until the search-side
   schemas are aligned. The search side uses flat field access patterns that differ
   from the nested tracker-backed model.

3. **Migrate only Qdrant functions, leave Postgres on BaseMovie**: Left an
   explicit type mismatch. Any CLI entry point that calls both Postgres and
   Qdrant ingestion would need to construct both types from the same tracker row.

## Consequences

- `ingest_movie.py` uses `Movie` exclusively. `BaseMovie` is no longer imported.
- The batch CLI (`cmd_ingest`) can load movies once and pass the same object to
  both Postgres and Qdrant ingestion functions.
- `duration` is now `int | None` on `Movie.tmdb_data` (was `int` on `BaseMovie`);
  an existing None guard in `_build_qdrant_payload` handles this correctly.
- `create_people_list` return type was corrected from `List[str]` to `set[str]`
  as part of this migration (was a latent annotation bug).
- `normalized_title_tokens()` on `Movie` merges tokens from both `title` and
  `original_title`, enabling search by foreign-language original titles.

## References

- `schemas/movie.py`, `movie_ingestion/final_ingestion/ingest_movie.py`
- ADR-056 (Movie tracker-backed loader — the type this migration consolidates around)
- `docs/modules/schemas.md`, `docs/modules/ingestion.md`
