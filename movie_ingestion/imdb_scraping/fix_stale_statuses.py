"""
One-off script to reconcile movies stuck at tmdb_quality_passed status.

For each movie still at tmdb_quality_passed:
  - If IMDB data exists in the imdb_data table, update status to imdb_scraped
  - Otherwise, print the tmdb_id and imdb_id so they can be re-scraped
"""

from movie_ingestion.tracker import init_db, MovieStatus


def main() -> None:
    db = init_db()

    rows = db.execute(
        "SELECT tmdb_id, imdb_id FROM movie_progress WHERE status = ?",
        (MovieStatus.TMDB_QUALITY_PASSED,),
    ).fetchall()

    if not rows:
        print("No movies at tmdb_quality_passed status.")
        return

    print(f"Found {len(rows):,} movies still at tmdb_quality_passed.\n")

    # Bulk-check which candidates have IMDB data in SQLite.
    candidate_ids = [row[0] for row in rows]
    placeholders = ",".join("?" * len(candidate_ids))
    cached_rows = db.execute(
        f"SELECT tmdb_id FROM imdb_data WHERE tmdb_id IN ({placeholders})",  # noqa: S608
        candidate_ids,
    ).fetchall()
    cached_ids = {row[0] for row in cached_rows}

    updated = 0
    missing = 0

    for tmdb_id, imdb_id in rows:
        if tmdb_id in cached_ids:
            # Scraped data exists — promote status to imdb_scraped
            db.execute(
                "UPDATE movie_progress SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE tmdb_id = ?",
                (MovieStatus.IMDB_SCRAPED, tmdb_id),
            )
            updated += 1
        else:
            # No scraped data — report for manual re-scraping
            print(f"  Missing: tmdb_id={tmdb_id}  imdb_id={imdb_id}")
            missing += 1

    db.commit()
    print(f"\nDone. Updated {updated:,} to imdb_scraped, {missing:,} still missing scraped data.")


if __name__ == "__main__":
    main()
