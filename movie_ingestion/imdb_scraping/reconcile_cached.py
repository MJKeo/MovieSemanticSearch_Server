"""
Reconcile cached IMDB data with the tracker database.

Finds movies at 'tmdb_quality_passed' status that already have IMDB data
in the imdb_data SQLite table and advances them to 'imdb_scraped' without
re-fetching. This handles the case where a prior scraping run persisted
the data but crashed before committing the status update.

Usage:
    python -m movie_ingestion.imdb_scraping.reconcile_cached
"""

from movie_ingestion.tracker import (
    MovieStatus,
    init_db,
)

# Commit after this many status updates to bound transaction size.
_COMMIT_BATCH_SIZE = 500

# SQL for bulk-updating movie status after confirming cached data exists.
_UPDATE_STATUS_SQL = """
    UPDATE movie_progress
    SET status = ?, updated_at = CURRENT_TIMESTAMP
    WHERE tmdb_id = ?
"""


def run() -> None:
    """
    Scan tmdb_quality_passed movies for existing IMDB data in SQLite
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

        # Find which candidates already have IMDB data in the imdb_data table.
        # Use a single query to get the set of IDs that exist.
        placeholders = ",".join("?" * len(candidates))
        cached_rows = db.execute(
            f"SELECT tmdb_id FROM imdb_data WHERE tmdb_id IN ({placeholders})",  # noqa: S608
            candidates,
        ).fetchall()
        cached_ids = {row[0] for row in cached_rows}

        # Promote cached movies in batches.
        promoted = 0
        batch: list[int] = []

        for tmdb_id in candidates:
            if tmdb_id not in cached_ids:
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
        print(f"  Skipped (no data): {skipped:,}")
        print("=" * 60)

    finally:
        db.close()
        print("  Database connection closed.")


if __name__ == "__main__":
    run()
