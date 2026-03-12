# ADR-023 — Migrate IMDB Scraped Data from JSON Files to SQLite imdb_data Table

## Status
Active

## Context

Stage 4 (IMDB scraping) previously wrote per-movie results as individual
JSON files at `ingestion_data/imdb/{tmdb_id}.json`. After scraping 425K
movies, this produced 425K files. The downstream Stage 5 quality scorer
and diagnostic scripts loaded these files using a `ThreadPoolExecutor`
file I/O pattern, which was slow and made SQL-based analysis impossible.
The design also created a split state: pipeline status in SQLite,
actual data in the filesystem — two sources of truth to keep in sync.

## Decision

Replace the JSON file store with a single `imdb_data` table in tracker.db.
Each `IMDBScrapedMovie` field gets its own column: 8 scalar columns
(TEXT/REAL/INTEGER) + 19 JSON TEXT columns for list/object fields.
Empty lists are stored as NULL so `IS NOT NULL` works as a presence check.

**Centralized helpers in tracker.py**:
- `IMDB_DATA_COLUMNS` — tuple defining column order, single source of truth.
- `serialize_imdb_movie(model_dict)` — converts model_dump dict to INSERT tuple.
- `deserialize_imdb_row(row)` — converts sqlite3.Row back to dict with JSON
  columns parsed to Python lists/dicts.

**Writer change**: `scraper.py` returns `model_dump()` dict in
`MovieResult.data` (was a JSON string). `run.py` calls
`serialize_imdb_movie()` and writes via `executemany`.

**Reader change**: All readers use `sqlite3.Row` factory +
`deserialize_imdb_row()` to get back the same dict shape as before.
Downstream code required no changes.

**Auto-migration**: `_migrate_imdb_data_blob_to_columns()` ran on
`init_db()`, detected the old single-blob `data` column via
`PRAGMA table_info`, and used `json_extract()` to expand 425,345 rows
into individual columns. This migration path is now a no-op on fresh DBs.

## Alternatives Considered

1. **Keep JSON files, add a SQLite index table**: Rejected. Two sources of
   truth (file presence + DB status) were the existing problem. This
   would enshrine that split permanently.

2. **Single BLOB column in SQLite** (store raw JSON as one value): Simpler
   migration, but precludes SQL queries on individual fields (e.g., `WHERE
   imdb_vote_count > 1000`, `WHERE plot_keywords IS NOT NULL`). The whole
   point of moving to SQLite is enabling field-level queries.

3. **Dedicated SQLite DB file for IMDB data**: Rejected. All ingestion data
   lives in one tracker.db; splitting would require managing cross-DB
   connections in every stage script. ADR-006 established the single-file
   pattern.

## Consequences

- Stage 5, diagnostic scripts, and `reconcile_cached.py` all read from
  `imdb_data` via SQL instead of filesystem I/O — queries are faster and
  field-level filtering is now possible without loading entire rows.
- `MovieResult.data` in `scraper.py` is now `dict | None` (was `str | None`).
  Any new code touching this field must account for the type change.
- The `ingestion_data/imdb/` directory is now obsolete as a data store.
  The one-time migration script (`migrate_json_to_sqlite.py`) can be
  retained for reference but is no longer needed.
- tracker.db file size increases substantially with 28 columns × 425K rows
  of JSON text stored inline.

## References

- ADR-006 (SQLite checkpoint tracker) — established the single-file pattern
- docs/modules/ingestion.md (Stage 4 section, Tracker System section)
- movie_ingestion/tracker.py — imdb_data schema, serialize/deserialize helpers
- movie_ingestion/imdb_scraping/scraper.py, run.py
