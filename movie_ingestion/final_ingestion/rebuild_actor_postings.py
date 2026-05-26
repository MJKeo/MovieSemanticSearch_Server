"""One-shot backfill: drop and rebuild `lex.inv_actor_postings`.

Why this exists: the IMDB scrape was just re-run for ~109K movies to
pick up `self` credits (documentary subjects, concert performers,
archive-footage cameos) that the previous `["actor", "actress"]`
filter dropped. The new `actors` list now sits in the tracker's
`imdb_data.actors` column in IMDB's billing order — but
`lex.inv_actor_postings` is still populated from the old, narrower
list. Every other table is correct from the prior full ingest.

Rather than re-run the full Stage 8 ingest (which would needlessly
re-embed 8 vector spaces, re-upsert movie_card, character postings,
brand postings, awards, etc.), this script does the surgical thing:
drop and repopulate exactly one table, in IMDB's billing order, from
the actors data already in the tracker.

Flow:
  1. DROP + CREATE `lex.inv_actor_postings` (same schema as
     db/init/01_create_postgres_tables.sql:227-237) to guarantee a
     clean starting state and to clear the secondary index so bulk
     inserts run with an empty index.
  2. Load every (tmdb_id, actors) pair at status `imdb_scraped` from
     the tracker.
  3. Process in batches of 500 movies. Per batch:
       a. Normalize + hyphen-expand actor names in-memory (no DB).
       b. ONE `batch_upsert_lexical_dictionary` call across the whole
          batch's variants — collapses ~500 round-trips into one.
       c. ONE big `unnest()`-based INSERT for the whole batch's
          postings — collapses another ~500 round-trips into one.
       d. Commit Postgres, then UPDATE the tracker for all movies in
          the batch from `imdb_scraped` → `ingested`.

Re-run semantics: DROP is at the top, so a previous partial run is
wiped. Tracker status filter (`imdb_scraped`) means re-runs after a
crash naturally pick up the not-yet-promoted slice — but because the
table is dropped fresh on every invocation, the only useful re-run
strategy is from-scratch.

Mirrors the actor section of
`movie_ingestion.final_ingestion.ingest_movie.ingest_lexical_data()`
by re-importing the same private helpers (`_normalize_name_list`,
`_expand_positioned_names`, `_term_ids_and_positions`) so the rows
this script writes are byte-identical to a fresh Stage 8 ingest.

Run as:
    python -m movie_ingestion.final_ingestion.rebuild_actor_postings
"""

from __future__ import annotations

import asyncio
import sqlite3
import time

from db.postgres import batch_upsert_lexical_dictionary, pool
from movie_ingestion.tracker import (
    IMDB_DATA_COLUMNS,
    MovieStatus,
    TRACKER_DB_PATH,
    deserialize_imdb_row,
)
from movie_ingestion.final_ingestion.ingest_movie import (
    _expand_positioned_names,
    _normalize_name_list,
    _term_ids_and_positions,
)

# 500 movies per batch matches rebuild_character_postings.py and is
# the sweet spot: large enough that the per-batch DB round-trips
# (lexical upsert + INSERT + tracker UPDATE = 3 total) dominate over
# Python-side normalization, small enough that a crash mid-rebuild
# loses at most ~500 movies of progress.
_BATCH_SIZE: int = 500

# DDL duplicated from db/init/01_create_postgres_tables.sql:227-237.
# Keep in sync with the init script. The DROP + CREATE pair gives us
# an empty index for the bulk insert and trivial idempotency on re-run.
_DROP_AND_CREATE_SQL = """
DROP TABLE IF EXISTS lex.inv_actor_postings;

CREATE TABLE lex.inv_actor_postings (
  term_id          BIGINT NOT NULL,
  movie_id         BIGINT NOT NULL,
  billing_position INT    NOT NULL,
  cast_size        INT    NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX idx_actor_postings_movie ON lex.inv_actor_postings (movie_id);
"""

# Single-statement INSERT that takes four parallel arrays and emits
# one row per array index. Lets the whole batch (every movie × every
# variant) land in one round-trip. ON CONFLICT DO NOTHING is purely
# defensive — after the DROP the table is empty, so collisions can
# only come from intra-batch duplicates, which `_term_ids_and_positions`
# already filters via `string_id_map` lookups.
_INSERT_SQL = """
INSERT INTO lex.inv_actor_postings
    (term_id, movie_id, billing_position, cast_size)
SELECT
    unnest(%s::bigint[]),
    unnest(%s::bigint[]),
    unnest(%s::int[]),
    unnest(%s::int[])
ON CONFLICT (term_id, movie_id) DO NOTHING
"""


def _iter_scraped_movies() -> list[tuple[int, list[str]]]:
    """Return (tmdb_id, actors) for every movie at status IMDB_SCRAPED.

    Materialized up front rather than streamed: ~109K rows × ~10
    short strings each ≈ tens of MB, well within memory, and lets
    the SQLite cursor close immediately so the tracker DB isn't held
    open for the whole Postgres run.
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
            (MovieStatus.IMDB_SCRAPED.value,),
        ).fetchall()
    finally:
        tracker.close()

    out: list[tuple[int, list[str]]] = []
    for row in rows:
        data = deserialize_imdb_row(row)
        # `actors` is JSON; deserialize_imdb_row returns [] for NULL.
        # Defensive `isinstance` filter mirrors the char-rebuild — the
        # downstream normalization would reject non-strings anyway.
        actors = [a for a in (data.get("actors") or []) if isinstance(a, str)]
        out.append((int(data["tmdb_id"]), actors))
    return out


async def _rebuild_schema() -> None:
    """Drop and recreate the actor postings table on its own connection."""
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(_DROP_AND_CREATE_SQL)
        await conn.commit()


async def _process_batch(
    batch: list[tuple[int, list[str]]],
    conn,
) -> int:
    """Normalize, expand, upsert lexical, and INSERT postings for one batch.

    Returns the number of posting rows written.
    """
    # Phase A: in-memory normalization + variant expansion. No DB calls.
    # `per_movie` is parallel to `batch`; entries with no surviving
    # normalized names are kept (with positioned=[]) so we can still
    # advance their tracker status — empty actor list is a valid
    # outcome for some films and shouldn't block status promotion.
    per_movie: list[tuple[int, list[tuple[str, int]], int]] = []
    all_variants: dict[str, None] = {}  # ordered dedup across the batch
    for movie_id, raw_actors in batch:
        actors = _normalize_name_list(raw_actors)
        if not actors:
            per_movie.append((movie_id, [], 0))
            continue
        # cast_size is frozen pre-variant-expansion to match ingest_movie.py:488.
        cast_size = len(actors)
        variants, positioned = _expand_positioned_names(actors)
        for v in variants:
            all_variants.setdefault(v, None)
        per_movie.append((movie_id, positioned, cast_size))

    # Phase B: one lexical-dictionary upsert for the entire batch.
    # `batch_upsert_lexical_dictionary` is the same idempotent helper
    # ingest_movie.py uses — existing string_ids are returned, new
    # variants get INSERTed under their existing UNIQUE constraint.
    string_id_map: dict[str, int] = {}
    if all_variants:
        string_id_map = await batch_upsert_lexical_dictionary(
            list(all_variants.keys()), conn=conn,
        )

    # Phase C: build four parallel arrays for one big INSERT.
    all_term_ids: list[int] = []
    all_movie_ids: list[int] = []
    all_positions: list[int] = []
    all_cast_sizes: list[int] = []
    for movie_id, positioned, cast_size in per_movie:
        if not positioned:
            continue
        term_ids, positions = _term_ids_and_positions(positioned, string_id_map)
        n = len(term_ids)
        if n == 0:
            continue
        all_term_ids.extend(term_ids)
        all_movie_ids.extend([movie_id] * n)
        all_positions.extend(positions)
        all_cast_sizes.extend([cast_size] * n)

    # Phase D: one INSERT for the whole batch. `batch_insert_actor_postings`
    # from db/postgres.py is per-movie (takes a single movie_id +
    # cast_size); we do the cross-movie variant inline here.
    if all_term_ids:
        async with conn.cursor() as cur:
            await cur.execute(
                _INSERT_SQL,
                (all_term_ids, all_movie_ids, all_positions, all_cast_sizes),
            )

    return len(all_term_ids)


def _advance_tracker_status(movie_ids: list[int]) -> None:
    """Mark the batch as `ingested` in the tracker.

    Runs AFTER the Postgres commit so the tracker promotion never
    races ahead of the postings write. A crash between the two leaves
    movies at `imdb_scraped` (replayable on next run after a fresh
    DROP); a crash before the Postgres commit leaves them at
    `imdb_scraped` with no postings (also replayable). Either way,
    `status='imdb_scraped'` remains the source of truth for work
    remaining.

    The `AND status = 'imdb_scraped'` guard is defensive — if some
    other process advanced the status concurrently, don't clobber it.
    """
    if not movie_ids:
        return
    tracker = sqlite3.connect(str(TRACKER_DB_PATH))
    try:
        placeholders = ",".join("?" * len(movie_ids))
        tracker.execute(
            f"UPDATE movie_progress SET status = ? "
            f"WHERE tmdb_id IN ({placeholders}) AND status = ?",
            [MovieStatus.INGESTED.value, *movie_ids, MovieStatus.IMDB_SCRAPED.value],
        )
        tracker.commit()
    finally:
        tracker.close()


async def rebuild_actor_postings() -> None:
    """End-to-end rebuild: drop schema, iterate movies, write postings."""
    await pool.open()
    try:
        print("Dropping and recreating lex.inv_actor_postings...")
        await _rebuild_schema()

        print("Loading scraped movies from tracker...")
        movies = _iter_scraped_movies()
        total = len(movies)
        print(f"Rebuilding postings for {total} movies (batch size {_BATCH_SIZE})...")

        start = time.perf_counter()
        processed = 0
        total_rows = 0
        for batch_start in range(0, total, _BATCH_SIZE):
            batch = movies[batch_start : batch_start + _BATCH_SIZE]
            try:
                async with pool.connection() as conn:
                    rows_written = await _process_batch(batch, conn)
                    await conn.commit()
                total_rows += rows_written
                # Only advance tracker status after Postgres commits cleanly.
                _advance_tracker_status([m for m, _ in batch])
            except Exception as exc:
                # One bad batch shouldn't sink the rebuild. Log and
                # continue — the affected movies stay at `imdb_scraped`,
                # so the next run will retry them (after a fresh DROP).
                print(
                    f"  batch starting at index {batch_start} failed: {exc!r}"
                )
            processed += len(batch)
            elapsed = time.perf_counter() - start
            rate = processed / elapsed if elapsed > 0 else 0.0
            print(
                f"  committed {processed}/{total} "
                f"({rate:.1f} movies/sec, {total_rows} postings so far)"
            )
    finally:
        await pool.close()

    print(f"Done. {total_rows} actor postings written across {processed} movies.")


def main() -> None:
    asyncio.run(rebuild_actor_postings())


if __name__ == "__main__":
    main()
