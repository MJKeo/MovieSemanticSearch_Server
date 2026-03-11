"""Quick diagnostic: top 10 movies by stage_5_quality_score with no watch providers."""

import sqlite3
from movie_ingestion.tracker import TRACKER_DB_PATH


def main() -> None:
    db = sqlite3.connect(TRACKER_DB_PATH)

    rows = db.execute("""
        SELECT p.imdb_id, p.stage_5_quality_score
        FROM movie_progress p
        JOIN tmdb_data d ON d.tmdb_id = p.tmdb_id
        WHERE p.stage_5_quality_score IS NOT NULL
          AND (d.watch_provider_keys IS NULL OR LENGTH(d.watch_provider_keys) = 0)
          AND d.release_date IS NOT NULL
          AND d.release_date <= DATE('now', '-365 days')
        ORDER BY p.stage_5_quality_score DESC
        LIMIT 10
    """).fetchall()

    db.close()

    print(f"{'Rank':<6} {'IMDB ID':<12} {'Stage 5 Score'}")
    print("-" * 35)
    for i, (imdb_id, score) in enumerate(rows, 1):
        print(f"{i:<6} {imdb_id or 'N/A':<12} {score:.4f}")


if __name__ == "__main__":
    main()
