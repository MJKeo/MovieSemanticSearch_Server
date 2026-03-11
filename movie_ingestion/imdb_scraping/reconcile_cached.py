"""
Reconcile cached IMDB JSON files with the tracker database.

Finds movies at 'tmdb_quality_passed' status that already have cached
IMDB data on disk (ingestion_data/imdb/{tmdb_id}.json) and advances
them to 'imdb_scraped' without re-fetching. This handles the case
where a prior scraping run wrote the JSON but crashed before committing
the status update to SQLite.

Usage:
    python -m movie_ingestion.imdb_scraping.reconcile_cached
"""

import time

from movie_ingestion.tracker import (
    INGESTION_DATA_DIR,
    MovieStatus,
    init_db,
)

# Commit after this many status updates to bound transaction size.
_COMMIT_BATCH_SIZE = 500

# Directory where Stage 4 writes per-movie IMDB JSON.
_IMDB_CACHE_DIR = INGESTION_DATA_DIR / "imdb"

# SQL for bulk-updating movie status after confirming cached data exists.
_UPDATE_STATUS_SQL = """
    UPDATE movie_progress
    SET status = ?, updated_at = CURRENT_TIMESTAMP
    WHERE tmdb_id = ?
"""


def run() -> None:
    """
    Scan tmdb_quality_passed movies for existing IMDB cache files
    and promote matching movies to imdb_scraped.
    """
    print("\n" + "=" * 60)
    print("  Reconcile cached IMDB data with tracker status")
    print("=" * 60)

    db = init_db()

    try:
        rows = db.execute(
            "SELECT tmdb_id FROM movie_progress WHERE status = ?",
            (MovieStatus.TMDB_QUALITY_PASSED,),
        ).fetchall()
        candidates = [row[0] for row in rows]

        if not candidates:
            print("\n  No movies at tmdb_quality_passed status. Nothing to do.")
            return

        print(f"\n  {len(candidates):,} movies at tmdb_quality_passed")

        # Check which candidates already have cached IMDB JSON on disk.
        promoted = 0
        batch: list[int] = []

        for tmdb_id in candidates:
            cache_path = _IMDB_CACHE_DIR / f"{tmdb_id}.json"
            if not cache_path.exists():
                continue

            batch.append(tmdb_id)

            # Flush batch when it reaches the commit size.
            if len(batch) >= _COMMIT_BATCH_SIZE:
                db.executemany(
                    _UPDATE_STATUS_SQL,
                    [(MovieStatus.IMDB_SCRAPED, tid) for tid in batch],
                )
                db.commit()
                promoted += len(batch)
                print(f"    Promoted {promoted:,} so far...")
                batch.clear()

        # Flush remaining partial batch.
        if batch:
            db.executemany(
                _UPDATE_STATUS_SQL,
                [(MovieStatus.IMDB_SCRAPED, tid) for tid in batch],
            )
            db.commit()
            promoted += len(batch)

        skipped = len(candidates) - promoted

        print(f"\n  Done.")
        print(f"  Total candidates:  {len(candidates):,}")
        print(f"  Promoted:          {promoted:,}")
        print(f"  Skipped (no cache):{skipped:,}")
        print("=" * 60)

    finally:
        db.close()
        print("  Database connection closed.")


if __name__ == "__main__":
    run()
