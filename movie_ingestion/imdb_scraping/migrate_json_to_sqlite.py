"""
One-time migration: backfill the imdb_data SQLite table from JSON files.

Reads all per-movie JSON files at ingestion_data/imdb/{tmdb_id}.json and
inserts each field into its own column in the imdb_data table. Existing
rows are skipped (INSERT OR IGNORE) so the script is safe to re-run.

After verifying the migration, the ingestion_data/imdb/ directory can be
deleted manually.

Usage:
    python -m movie_ingestion.imdb_scraping.migrate_json_to_sqlite
"""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import orjson

from movie_ingestion.tracker import IMDB_INSERT_SQL, INGESTION_DATA_DIR, init_db, serialize_imdb_movie

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_IMDB_DIR = INGESTION_DATA_DIR / "imdb"
_BATCH_SIZE = 1_000
_THREAD_WORKERS = 12
_PROGRESS_INTERVAL = 10_000


# ---------------------------------------------------------------------------
# File loading (reuses the same orjson + ThreadPoolExecutor pattern that
# the old readers used, since we need it one last time for the migration)
# ---------------------------------------------------------------------------


def _load_one_json(path: Path) -> tuple | None:
    """Load a single IMDB JSON file and return an INSERT-ready tuple.

    Returns a tuple matching IMDB_INSERT_SQL column order (tmdb_id +
    all 27 individual fields), with list/object fields serialized to
    JSON TEXT strings. Returns None on parse failure.
    """
    try:
        tmdb_id = int(path.stem)
    except ValueError:
        return None
    try:
        with open(path, "rb") as f:
            raw = f.read()
        data = orjson.loads(raw)
        return serialize_imdb_movie(tmdb_id, data)
    except (FileNotFoundError, orjson.JSONDecodeError) as exc:
        print(f"  WARNING: skipping {path.name}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("\n" + "=" * 60)
    print("  Migrate IMDB JSON files → imdb_data SQLite table")
    print("=" * 60)

    db = init_db()

    # Discover all JSON files in the IMDB directory.
    if not _IMDB_DIR.exists():
        print(f"\n  Directory {_IMDB_DIR} does not exist. Nothing to migrate.")
        db.close()
        return

    json_files = sorted(_IMDB_DIR.glob("*.json"))
    total_files = len(json_files)

    if total_files == 0:
        print("\n  No JSON files found. Nothing to migrate.")
        db.close()
        return

    print(f"\n  Found {total_files:,} JSON files to migrate")

    start = time.monotonic()
    inserted = 0
    skipped = 0
    errors = 0

    # Process in batches: load files with ThreadPoolExecutor, then bulk-insert.
    for batch_start in range(0, total_files, _BATCH_SIZE):
        batch_files = json_files[batch_start : batch_start + _BATCH_SIZE]

        # Parallel file reads
        rows: list[tuple] = []
        with ThreadPoolExecutor(max_workers=_THREAD_WORKERS) as pool:
            for result in pool.map(_load_one_json, batch_files):
                if result is not None:
                    rows.append(result)
                else:
                    errors += 1

        # Bulk insert — INSERT OR IGNORE skips existing rows.
        # Each row is a tuple matching IMDB_INSERT_SQL column order.
        if rows:
            # Replace INSERT OR REPLACE with INSERT OR IGNORE for migration
            # (we don't want to overwrite existing rows on re-run).
            ignore_sql = IMDB_INSERT_SQL.replace("OR REPLACE", "OR IGNORE")
            cursor = db.executemany(ignore_sql, rows)
            batch_inserted = cursor.rowcount
            inserted += batch_inserted
            skipped += len(rows) - batch_inserted

        db.commit()

        # Progress reporting
        processed = min(batch_start + _BATCH_SIZE, total_files)
        if processed % _PROGRESS_INTERVAL < _BATCH_SIZE or processed <= _BATCH_SIZE:
            elapsed = time.monotonic() - start
            rate = processed / max(elapsed, 0.001)
            print(f"  Progress: {processed:,}/{total_files:,} files "
                  f"({rate:.0f} files/sec) — "
                  f"inserted={inserted:,}, skipped={skipped:,}, errors={errors:,}")

    elapsed = time.monotonic() - start
    db.close()

    # Verify final count
    db2 = init_db()
    row_count = db2.execute("SELECT COUNT(*) FROM imdb_data").fetchone()[0]
    db2.close()

    print(f"\n  {'=' * 60}")
    print(f"  Migration complete in {elapsed:.1f}s")
    print(f"  Files processed:   {total_files:,}")
    print(f"  Rows inserted:     {inserted:,}")
    print(f"  Rows skipped:      {skipped:,} (already existed)")
    print(f"  Errors:            {errors:,}")
    print(f"  Total rows in DB:  {row_count:,}")
    print(f"  {'=' * 60}")


if __name__ == "__main__":
    main()
