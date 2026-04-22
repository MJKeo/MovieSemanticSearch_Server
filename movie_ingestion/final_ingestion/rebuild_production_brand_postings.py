"""One-shot backfill: truncate and rebuild `lex.inv_production_brand_postings`.

Why this exists: the production-brand registry in
`schemas/production_brands.py` was retuned from a "catalog recall"
principle to a "casual-viewer brand identity" principle. Existing
postings were written against the old registry, so queries like
"Disney movies" over-return (e.g. No Country for Old Men via the
old Disney→Miramax umbrella). Re-ingesting every movie just to
refresh brand tags would be wasteful since the raw production
company strings and release dates live in the tracker; this script
replays the brand-resolution step against the current registry.

Flow:
  1. TRUNCATE lex.inv_production_brand_postings.
  2. Iterate tracker movies with status == INGESTED, pulling
     `production_companies` from imdb_data and `release_date` from
     tmdb_data in a single join.
  3. For each movie, derive release_year (YYYY from the date prefix,
     mirroring `ingest_production_data`), call resolve_brands_for_movie,
     and hand the resulting BrandTags to batch_insert_brand_postings.
  4. Commit in batches for progress visibility and crash safety.

Re-run semantics: the TRUNCATE at the top wipes any partial previous
run, so the script always starts from scratch. Tracker data is
read-only here.

Run as:
    python -m movie_ingestion.final_ingestion.rebuild_production_brand_postings
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

import orjson

from db.postgres import batch_insert_brand_postings, pool
from movie_ingestion.final_ingestion.brand_resolver import resolve_brands_for_movie
from movie_ingestion.tracker import (
    MovieStatus,
    TRACKER_DB_PATH,
)

# Commit roughly every N movies so progress is visible and a crash
# mid-run loses only the most recent batch. Brand resolution is pure
# Python (no network/LLM calls), so this is a pure observability knob.
_COMMIT_BATCH_SIZE: int = 500

# TRUNCATE (vs DELETE) because we're replacing every row wholesale.
# No FKs reference this table, so TRUNCATE is safe and much faster
# than a full-table DELETE.
_TRUNCATE_SQL = "TRUNCATE TABLE lex.inv_production_brand_postings;"


def _iter_ingested_movies() -> list[tuple[int, list[str], int | None]]:
    """Stream (tmdb_id, production_companies, release_year) for every movie at status INGESTED.

    Reads production_companies from imdb_data (JSON TEXT array) and
    release_date from tmdb_data, joining on tmdb_id. A LEFT JOIN on
    tmdb_data is used so movies without a tmdb_data row still resolve
    (release_year becomes None, and the resolver drops windowed
    memberships accordingly).

    Returns a materialized list rather than a generator so the SQLite
    cursor closes promptly — the per-row payload is small.
    """
    tracker = sqlite3.connect(str(TRACKER_DB_PATH))
    tracker.row_factory = sqlite3.Row
    try:
        rows = tracker.execute(
            """
            SELECT
                mp.tmdb_id,
                i.production_companies,
                t.release_date
            FROM movie_progress mp
            LEFT JOIN imdb_data i ON i.tmdb_id = mp.tmdb_id
            LEFT JOIN tmdb_data t ON t.tmdb_id = mp.tmdb_id
            WHERE mp.status = ?
            """,
            (MovieStatus.INGESTED.value,),
        ).fetchall()
    finally:
        tracker.close()

    out: list[tuple[int, list[str], int | None]] = []
    for row in rows:
        raw_json = row["production_companies"]
        # production_companies is stored as JSON TEXT or NULL when empty.
        if raw_json:
            try:
                companies = orjson.loads(raw_json)
            except Exception:
                companies = []
        else:
            companies = []
        # Filter out non-string entries defensively; the resolver expects
        # strings and bad data should be dropped rather than crash the run.
        companies = [c for c in companies if isinstance(c, str)]

        # Derive release_year from YYYY-MM-DD prefix; mirror the logic in
        # ingest_production_data so the rebuild produces identical tags.
        release_date = row["release_date"]
        release_year: int | None = None
        if release_date:
            try:
                release_year = int(str(release_date)[:4])
            except (ValueError, TypeError):
                release_year = None

        out.append((int(row["tmdb_id"]), companies, release_year))
    return out


async def _truncate_postings() -> None:
    """Wipe all rows from lex.inv_production_brand_postings."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_TRUNCATE_SQL)
        await conn.commit()


async def _process_movie(
    movie_id: int,
    production_companies: list[str],
    release_year: int | None,
    conn,
) -> int:
    """Resolve brands and write postings for one movie.

    Returns the number of brand rows written (0 when the movie has
    no production companies or none resolve to a brand).
    """
    brand_tags = resolve_brands_for_movie(production_companies, release_year)
    if not brand_tags:
        # Still call the helper so an idempotent run with a now-empty
        # tag set clears any stale rows. Post-TRUNCATE there are none,
        # so skip the round trip for efficiency.
        return 0

    rows = [(tag.brand_id, tag.first_matching_index) for tag in brand_tags]
    await batch_insert_brand_postings(movie_id, rows, conn=conn)
    return len(rows)


async def rebuild_production_brand_postings() -> None:
    """End-to-end rebuild: truncate, iterate movies, write postings."""
    await pool.open()
    try:
        print("Truncating lex.inv_production_brand_postings...")
        await _truncate_postings()

        print("Loading ingested movies from tracker...")
        movies = _iter_ingested_movies()
        total = len(movies)
        print(f"Rebuilding postings for {total} movies...")

        start = time.perf_counter()
        processed = 0
        total_rows = 0
        for batch_start in range(0, total, _COMMIT_BATCH_SIZE):
            batch = movies[batch_start : batch_start + _COMMIT_BATCH_SIZE]
            async with pool.connection() as conn:
                for movie_id, companies, release_year in batch:
                    try:
                        rows_written = await _process_movie(
                            movie_id, companies, release_year, conn
                        )
                        total_rows += rows_written
                    except Exception as exc:
                        # A single bad row should not sink the rebuild;
                        # log and continue. batch_insert_brand_postings
                        # does DELETE-then-INSERT inside the same conn,
                        # so a failure mid-movie rolls back cleanly when
                        # the outer commit runs.
                        print(f"  movie_id={movie_id} failed: {exc!r}")
                await conn.commit()
            processed += len(batch)
            elapsed = time.perf_counter() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(
                f"  committed {processed}/{total} "
                f"({rate:.1f} movies/sec, {total_rows} postings so far)"
            )
    finally:
        await pool.close()

    print(
        f"Done. {total_rows} brand postings written across {processed} movies."
    )


def main() -> None:
    asyncio.run(rebuild_production_brand_postings())


if __name__ == "__main__":
    main()
