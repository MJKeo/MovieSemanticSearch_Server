"""
One-off backfill: set release_format = ReleaseFormat.SHORT on movie_card
for every ingested movie whose runtime_minutes <= 40.

The source of truth for "ingested" is the SQLite tracker
(./ingestion_data/tracker.db, movie_progress.status = 'ingested'),
matching the rule in CLAUDE.md. We bulk-fetch those tmdb_ids, then
issue a single parameterized UPDATE against public.movie_card filtered
by movie_id = ANY(...) and runtime_minutes <= 40.

Mirrors the ingest-time rule in
movie_ingestion/final_ingestion/ingest_movie.py: shorts (<= 40 min) are
classified as ReleaseFormat.SHORT regardless of IMDB's reported title
type, since IMDB occasionally tags genuinely short content as 'movie'.
"""

import asyncio

from dotenv import load_dotenv

from db.postgres import pool
from movie_ingestion.tracker import MovieStatus, init_db
from schemas.enums import ReleaseFormat

load_dotenv()

# Threshold for SHORT classification. 40 minutes is the IMDB / Academy
# convention dividing line between a short film and a feature.
SHORT_RUNTIME_MAX_MINUTES = 40


def fetch_ingested_tmdb_ids() -> list[int]:
    """Read all tmdb_ids with status='ingested' from the SQLite tracker."""
    db = init_db()
    try:
        cur = db.execute(
            "SELECT tmdb_id FROM movie_progress WHERE status = ?",
            (MovieStatus.INGESTED.value,),
        )
        return [row[0] for row in cur.fetchall()]
    finally:
        db.close()


async def backfill() -> int:
    """Update release_format -> SHORT for ingested short-runtime movies.

    Returns the number of rows updated.
    """
    ingested_ids = fetch_ingested_tmdb_ids()
    if not ingested_ids:
        return 0

    short_id = ReleaseFormat.SHORT.release_format_id  # 3

    # Single bulk UPDATE: filter to ingested ids AND short runtime, and
    # skip rows already at SHORT so the row count reflects real changes.
    query = """
    UPDATE public.movie_card
    SET release_format = %s,
        updated_at = now()
    WHERE movie_id = ANY(%s)
      AND runtime_minutes IS NOT NULL
      AND runtime_minutes <= %s
      AND release_format <> %s
    """

    await pool.open()
    try:
        async with pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    query,
                    (short_id, ingested_ids, SHORT_RUNTIME_MAX_MINUTES, short_id),
                )
                updated = cur.rowcount
            await conn.commit()
        return updated
    finally:
        await pool.close()


async def main() -> None:
    updated = await backfill()
    print(f"Updated {updated} movie_card rows to release_format=SHORT")


if __name__ == "__main__":
    asyncio.run(main())
