"""
One-off script to reconcile movies stuck at tmdb_quality_passed status.

For each movie still at tmdb_quality_passed:
  - If scraped JSON exists in ingestion_data/imdb/, update status to imdb_scraped
  - Otherwise, print the tmdb_id and imdb_id so they can be re-scraped
"""

from pathlib import Path

from movie_ingestion.tracker import init_db, MovieStatus, INGESTION_DATA_DIR

_IMDB_JSON_DIR = INGESTION_DATA_DIR / "imdb"


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

    updated = 0
    missing = 0

    for tmdb_id, imdb_id in rows:
        json_path = _IMDB_JSON_DIR / f"{tmdb_id}.json"

        if json_path.exists():
            # Scraped data exists — promote status to imdb_scraped
            db.execute(
                "UPDATE movie_progress SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE tmdb_id = ?",
                (MovieStatus.IMDB_SCRAPED, tmdb_id),
            )
            updated += 1
        else:
            # No scraped data on disk — report for manual re-scraping
            print(f"  Missing: tmdb_id={tmdb_id}  imdb_id={imdb_id}")
            missing += 1

    db.commit()
    print(f"\nDone. Updated {updated:,} to imdb_scraped, {missing:,} still missing scraped data.")


if __name__ == "__main__":
    main()
