"""One-shot backfill: drop and rebuild `lex.inv_character_postings`.

Why this exists: the original schema was `(term_id, movie_id)` only
(binary match semantics). The new schema adds `billing_position` and
`character_cast_size` so that stage 3 can score character matches with
a prominence curve parallel to actor scoring. Since raw IMDB data
lives in the tracker's `imdb_data.characters` JSON column, we can
recompute every row from source rather than attempting an in-place
ALTER + backfill.

Flow:
  1. DROP + CREATE the postings table with the new schema.
  2. Iterate tracker movies with status == INGESTED.
  3. For each movie, load the ordered `characters` list from tracker
     imdb_data, run the same normalization + dedup pipeline as
     `ingest_movie.write_lexical_metadata`, resolve term IDs via the
     idempotent lexical upserts, and insert postings.
  4. Commit in batches for progress visibility and crash safety.

Re-run semantics: the DROP is at the top, so any partial previous
run is wiped and the script always starts from scratch. Tracker data
is read-only here.

Run as:
    python -m movie_ingestion.final_ingestion.rebuild_character_postings
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

from db.postgres import (
    batch_insert_character_postings,
    batch_upsert_character_strings,
    batch_upsert_lexical_dictionary,
    pool,
)
from implementation.misc.helpers import normalize_string
from movie_ingestion.tracker import (
    IMDB_DATA_COLUMNS,
    MovieStatus,
    TRACKER_DB_PATH,
    deserialize_imdb_row,
)

# Commit roughly every N movies so progress is visible and a crash
# mid-run loses only the most recent batch. The DROP+CREATE wipes
# nothing on re-run (we always start fresh), so the batch boundary
# is a pure observability/latency knob.
_COMMIT_BATCH_SIZE: int = 500

# DDL duplicated from db/init/01_create_postgres_tables.sql so the
# script is self-contained. Keep in sync with the init script.
_DROP_AND_CREATE_SQL = """
DROP TABLE IF EXISTS lex.inv_character_postings;

CREATE TABLE lex.inv_character_postings (
  term_id              BIGINT NOT NULL,
  movie_id             BIGINT NOT NULL,
  billing_position     INT    NOT NULL,
  character_cast_size  INT    NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX idx_character_postings_movie
  ON lex.inv_character_postings (movie_id);
"""


def _iter_ingested_movies() -> list[tuple[int, list[str]]]:
    """Stream (tmdb_id, characters) for every movie at status INGESTED.

    Returns a materialized list rather than a generator so the SQLite
    cursor closes promptly — character lists are tiny, so memory is
    not a concern.
    """
    tracker = sqlite3.connect(str(TRACKER_DB_PATH))
    tracker.row_factory = sqlite3.Row
    try:
        select_cols = "tmdb_id, " + ", ".join(IMDB_DATA_COLUMNS)
        rows = tracker.execute(
            f"""
            SELECT {select_cols}
            FROM imdb_data
            WHERE tmdb_id IN (
                SELECT tmdb_id FROM movie_progress WHERE status = ?
            )
            """,
            (MovieStatus.INGESTED.value,),
        ).fetchall()
    finally:
        tracker.close()

    out: list[tuple[int, list[str]]] = []
    for row in rows:
        data = deserialize_imdb_row(row)
        # `characters` is stored as JSON; deserialize_imdb_row returns
        # [] when the column is NULL.
        characters = data.get("characters") or []
        # Filter out non-string entries defensively — normalization
        # will reject them anyway, but the type narrowing avoids
        # ambiguity downstream.
        characters = [c for c in characters if isinstance(c, str)]
        out.append((int(data["tmdb_id"]), characters))
    return out


async def _rebuild_schema() -> None:
    """Drop and recreate the character postings table with the new schema."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_DROP_AND_CREATE_SQL)
        await conn.commit()


async def _process_movie(movie_id: int, raw_characters: list[str], conn) -> int:
    """Normalize, dedup, upsert, and insert postings for one movie.

    Mirrors the character section of
    `ingest_movie.write_lexical_metadata` so the backfill produces
    identical rows to a fresh ingestion.

    Returns the number of posting rows written (0 when the movie has
    no characters or none survive normalization).
    """
    # Preserve cast-edge order; normalize in place.
    normalized: list[str] = []
    for character in raw_characters:
        norm = normalize_string(str(character))
        if not norm:
            continue
        normalized.append(norm)
    if not normalized:
        return 0

    # Idempotent upsert into the shared lexical dictionary; resolves
    # norm_str → string_id for every distinct character name.
    string_id_map = await batch_upsert_lexical_dictionary(normalized, conn=conn)

    # Build (string_id, norm_str) pairs in billing order, keeping only
    # strings that resolved. Mirrors ingest_movie's pattern.
    character_string_pairs = [
        (string_id_map[c], c) for c in normalized if c in string_id_map
    ]
    if not character_string_pairs:
        return 0

    # Upsert into lex.character_strings with aligned (string_id,
    # norm_str) arrays — same shape as the ingestion call.
    await batch_upsert_character_strings(
        [sid for sid, _ in character_string_pairs],
        [s for _, s in character_string_pairs],
        conn=conn,
    )

    # Dedup preserves the first (topmost-billed) occurrence of each
    # term_id — same dict.fromkeys pattern as the ingestion path.
    character_term_ids = [sid for sid, _ in character_string_pairs]
    deduped_term_ids = list(dict.fromkeys(character_term_ids))

    await batch_insert_character_postings(
        deduped_term_ids,
        movie_id,
        character_cast_size=len(deduped_term_ids),
        conn=conn,
    )
    return len(deduped_term_ids)


async def rebuild_character_postings() -> None:
    """End-to-end rebuild: drop schema, iterate movies, write postings."""
    await pool.open()
    try:
        print("Dropping and recreating lex.inv_character_postings...")
        await _rebuild_schema()

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
                for movie_id, raw_characters in batch:
                    try:
                        rows_written = await _process_movie(
                            movie_id, raw_characters, conn
                        )
                        total_rows += rows_written
                    except Exception as exc:
                        # A single bad row should not sink the rebuild;
                        # log and continue. PK conflicts from retries
                        # are absorbed by ON CONFLICT DO NOTHING.
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

    print(f"Done. {total_rows} character postings written across {processed} movies.")


def main() -> None:
    asyncio.run(rebuild_character_postings())


if __name__ == "__main__":
    main()
