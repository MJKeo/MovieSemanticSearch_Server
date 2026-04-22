"""
One-shot backfill: migrate the lexical posting schema to the
hyphen-variant + normalized-title world documented in
[search_improvement_planning/entity_improvement.md](../../search_improvement_planning/entity_improvement.md)
and the ingestion-side plan in
[.claude/plans/mighty-sauteeing-cookie.md](../../.claude/plans/mighty-sauteeing-cookie.md).

What it does
------------
1. Drops the retired v1 title-token SQL objects
   (``lex.title_token_doc_frequency``, ``lex.inv_title_token_postings``,
   ``lex.title_token_strings``) and ``movie_card.title_token_count``.
2. Adds ``movie_card.title_normalized`` plus the
   ``idx_movie_card_title_normalized_trgm`` GIN and
   ``idx_movie_card_title_normalized_prefix`` btree indexes.
3. Truncates every per-movie role posting table
   (``inv_actor_postings``, ``inv_director_postings``,
   ``inv_writer_postings``, ``inv_producer_postings``,
   ``inv_composer_postings``, ``inv_character_postings``).
   The shared ``lex.lexical_dictionary`` / ``lex.character_strings``
   registries are NOT truncated — they are shared with franchises,
   studios, and awards; their ``UNIQUE(norm_str)`` constraint makes
   re-upserts safe, and orphaned term_ids are harmless.
4. Walks every row in ``public.movie_card``, loads the corresponding
   ``Movie`` object from ``tracker.db``, and rewrites:
     - ``movie_card.title_normalized`` via
       ``normalize_string(movie.tmdb_data.title)`` (a narrow UPDATE —
       other card columns are left alone).
     - All lexical postings via
       :func:`movie_ingestion.final_ingestion.ingest_movie.ingest_lexical_data`,
       which now emits hyphen-variant rows sharing billing positions.

Prerequisites
-------------
- The v1 search path (``db/lexical_search.py`` / ``db/search.py``) must
  either be retired or patched so it no longer queries
  ``lex.title_token_strings`` / ``lex.inv_title_token_postings`` /
  ``lex.title_token_doc_frequency``. This changeset already stubs the
  relevant imports inside ``db/lexical_search.py`` so the module loads;
  live v1 queries against title tokens will simply return empty
  results after the backfill drops the tables.
- ``tracker.db`` (``movie_ingestion.tracker.TRACKER_DB_PATH``) must
  still contain rows for every movie that has a ``movie_card`` entry,
  because per-movie rebuild reads them back from ``imdb_data`` and
  ``tmdb_data``.

Usage
-----
::

    # Schema migration + full rebuild (default).
    python -m movie_ingestion.backfill.rebuild_lexical_postings

    # Schema migration only.
    python -m movie_ingestion.backfill.rebuild_lexical_postings --schema-only

    # Smaller smoke slice (first N movie_card rows, ordered by movie_id).
    python -m movie_ingestion.backfill.rebuild_lexical_postings --max-movies 100

    # Dry run — just report counts, no destructive actions.
    python -m movie_ingestion.backfill.rebuild_lexical_postings --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from db.postgres import (
    _execute_on_conn,
    _execute_read,
    pool,
)
from implementation.misc.helpers import normalize_string
from movie_ingestion.final_ingestion.ingest_movie import ingest_lexical_data
from movie_ingestion.tracker import TRACKER_DB_PATH
from schemas.movie import Movie

logger = logging.getLogger(__name__)


# Cap concurrent per-movie tasks. Each task holds one connection for the
# duration of a (UPDATE title_normalized, ingest_lexical_data) pair; the
# pool tops out at max_size=10 so we stay one below to leave room for
# incidental reads from other code paths running alongside.
_CONCURRENCY = 8

# How many movies to load per tracker-DB batch. Movie.from_tmdb_ids is a
# single SQLite query so a larger batch trades less overhead for more
# memory; 200 matches the default used by the main ingest CLI.
_TRACKER_BATCH = 200


# ---------------------------------------------------------------------------
# Step 1 — schema migration.
# ---------------------------------------------------------------------------

# Pre-rebuild schema statements: add the normalized-title column and
# drop the retired v1 title-token objects. Indexes on
# title_normalized are deliberately deferred to _POST_REBUILD_STATEMENTS
# so the GIN isn't built against the column's DEFAULT '' placeholder
# values and then immediately reshuffled when the rebuild writes real
# values — building once after the rebuild is cheaper and yields a
# well-shaped index.
_PRE_REBUILD_SCHEMA_STATEMENTS: list[str] = [
    # DEFAULT '' satisfies NOT NULL during the ALTER on already-populated
    # rows; the per-movie UPDATE below overwrites it with the real
    # normalized value.
    "ALTER TABLE public.movie_card "
    "ADD COLUMN IF NOT EXISTS title_normalized TEXT NOT NULL DEFAULT '';",
    # Retire the v1 title-token structures. Drop in dependency order:
    # the MV depends on inv_title_token_postings, which in turn is
    # referenced by nothing else that survives the migration.
    "DROP MATERIALIZED VIEW IF EXISTS lex.title_token_doc_frequency;",
    "DROP TABLE IF EXISTS lex.inv_title_token_postings;",
    "DROP TABLE IF EXISTS lex.title_token_strings;",
    # Drop the precomputed title token count — its only readers were
    # the v1 title-scoring CTE and ingest_movie_card, both of which are
    # now updated.
    "ALTER TABLE public.movie_card DROP COLUMN IF EXISTS title_token_count;",
]

# Indexes that back Stage 3 title_pattern (CONTAINS / STARTS_WITH).
# Applied after the per-movie rebuild has populated title_normalized so
# the trigram and prefix indexes are built against final data.
_POST_REBUILD_SCHEMA_STATEMENTS: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_movie_card_title_normalized_trgm "
    "ON public.movie_card USING GIN (title_normalized gin_trgm_ops);",
    "CREATE INDEX IF NOT EXISTS idx_movie_card_title_normalized_prefix "
    "ON public.movie_card (title_normalized text_pattern_ops);",
]


async def _apply_schema_statements(statements: list[str]) -> None:
    """Apply ``statements`` inside a single transaction.

    Rolls back on any statement failure so a partial migration cannot
    leave the DB in a state where the init script can't recreate it.
    """
    async with pool.connection() as conn:
        try:
            for stmt in statements:
                await _execute_on_conn(conn, stmt, ())
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


async def apply_pre_rebuild_schema_migration() -> None:
    """Add the normalized-title column and drop retired v1 objects."""
    await _apply_schema_statements(_PRE_REBUILD_SCHEMA_STATEMENTS)


async def apply_post_rebuild_schema_migration() -> None:
    """Create the title_normalized trgm + btree indexes post-rebuild."""
    await _apply_schema_statements(_POST_REBUILD_SCHEMA_STATEMENTS)


async def delete_role_postings_for_movies(movie_ids: list[int]) -> None:
    """Delete posting rows for the movies about to be rebuilt.

    Scoped to ``movie_ids`` so ``--max-movies N`` smoke slices don't
    wipe postings for every other movie in the DB. The shared
    registries (``lex.lexical_dictionary`` / ``lex.character_strings``)
    are NOT touched — they back franchises / studios / awards too, and
    their UNIQUE(norm_str) constraint makes re-upserts idempotent.
    Orphaned term_ids are harmless.

    Runs the six DELETEs in a single transaction so partial deletion
    across role tables can't leak into the rebuild phase.
    """
    if not movie_ids:
        return
    ids_param = list(movie_ids)
    async with pool.connection() as conn:
        try:
            for table in (
                "lex.inv_actor_postings",
                "lex.inv_director_postings",
                "lex.inv_writer_postings",
                "lex.inv_producer_postings",
                "lex.inv_composer_postings",
                "lex.inv_character_postings",
            ):
                await _execute_on_conn(
                    conn,
                    f"DELETE FROM {table} WHERE movie_id = ANY(%s::bigint[])",
                    (ids_param,),
                )
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


# ---------------------------------------------------------------------------
# Step 2 — per-movie rebuild.
# ---------------------------------------------------------------------------


async def _fetch_movie_ids(max_movies: int | None) -> list[int]:
    """Return movie_ids to rebuild, ordered for deterministic resumption."""
    limit_clause = f" LIMIT {int(max_movies)}" if max_movies else ""
    rows = await _execute_read(
        f"SELECT movie_id FROM public.movie_card ORDER BY movie_id{limit_clause}"
    )
    return [int(row[0]) for row in rows]


async def _rebuild_one(
    movie: Movie,
    movie_id: int,
    semaphore: asyncio.Semaphore,
    counters: dict[str, int],
) -> None:
    """Rebuild one movie's lexical postings and title_normalized column.

    Runs inside ``semaphore`` so the pool never exhausts. Exceptions are
    caught and counted rather than bubbling up — a single bad movie
    shouldn't kill a multi-hour backfill.
    """
    async with semaphore:
        try:
            raw_title = movie.tmdb_data.title
            if not raw_title:
                counters["skipped_no_title"] += 1
                return
            normalized_title = normalize_string(str(raw_title))

            async with pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE public.movie_card "
                        "SET title_normalized = %s, updated_at = now() "
                        "WHERE movie_id = %s",
                        (normalized_title, movie_id),
                    )
                await ingest_lexical_data(movie, conn=conn)
                await conn.commit()

            counters["rebuilt"] += 1
        except Exception as exc:  # noqa: BLE001 — report and continue
            logger.warning("rebuild failed for movie_id=%s: %s", movie_id, exc)
            counters["failed"] += 1


async def rebuild_all(all_movie_ids: list[int]) -> dict[str, int]:
    """Rebuild title_normalized + lexical postings for each movie_id.

    Loads movies from tracker.db in ``_TRACKER_BATCH``-sized groups, fans
    out per-movie rebuild tasks inside each group under a semaphore, then
    logs progress between groups.
    """
    total = len(all_movie_ids)
    print(f"Rebuilding {total:,} movies...")

    counters = {"rebuilt": 0, "failed": 0, "skipped_no_title": 0, "total": total}
    if total == 0:
        return counters

    semaphore = asyncio.Semaphore(_CONCURRENCY)
    start = time.monotonic()

    for group_start in range(0, total, _TRACKER_BATCH):
        group_ids = all_movie_ids[group_start : group_start + _TRACKER_BATCH]
        movies_map = Movie.from_tmdb_ids(group_ids, TRACKER_DB_PATH)

        missing = set(group_ids) - set(movies_map.keys())
        if missing:
            # Movies with a movie_card row but no tracker record can't be
            # rebuilt — log once per occurrence and count them as failures.
            for movie_id in missing:
                logger.warning(
                    "movie_id=%s in movie_card but missing from tracker.db", movie_id
                )
            counters["failed"] += len(missing)

        tasks = [
            _rebuild_one(movies_map[movie_id], movie_id, semaphore, counters)
            for movie_id in group_ids
            if movie_id in movies_map
        ]
        if tasks:
            await asyncio.gather(*tasks)

        done = min(group_start + _TRACKER_BATCH, total)
        elapsed = time.monotonic() - start
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (total - done) / rate if rate > 0 else float("inf")
        print(
            f"  {done:,}/{total:,} processed "
            f"(rebuilt={counters['rebuilt']:,} failed={counters['failed']:,} "
            f"skipped={counters['skipped_no_title']:,}) "
            f"| {rate:.1f} movies/s | eta {eta:.0f}s",
            flush=True,
        )

    return counters


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


async def run(
    *,
    schema_only: bool = False,
    dry_run: bool = False,
    max_movies: int | None = None,
) -> None:
    # Pool is open=False at import time; this standalone script manages
    # its own open/close so it can run without the FastAPI lifespan.
    await pool.open()
    try:
        start = time.monotonic()

        if dry_run:
            movie_ids = await _fetch_movie_ids(max_movies)
            print(
                "[dry-run] Would apply pre-rebuild schema migration, "
                f"delete role postings for {len(movie_ids):,} movies, "
                "rebuild them, then create title_normalized indexes. "
                "No destructive actions taken."
            )
            return

        print("Step 1: applying pre-rebuild schema migration...")
        await apply_pre_rebuild_schema_migration()
        print("  done.")

        if schema_only:
            print(
                "--schema-only set; skipping rebuild and post-rebuild "
                "index creation. Re-run without --schema-only to finish."
            )
            return

        print("Step 2: resolving target movie set...")
        movie_ids = await _fetch_movie_ids(max_movies)
        print(f"  {len(movie_ids):,} movies selected.")

        print("Step 3: deleting existing role postings for target movies...")
        await delete_role_postings_for_movies(movie_ids)
        print("  done.")

        print("Step 4: rebuilding per-movie postings + title_normalized...")
        counters = await rebuild_all(movie_ids)

        print("Step 5: creating title_normalized indexes...")
        await apply_post_rebuild_schema_migration()
        print("  done.")

        elapsed = time.monotonic() - start
        print(
            f"Done in {elapsed:.1f}s. "
            f"rebuilt={counters['rebuilt']:,} "
            f"failed={counters['failed']:,} "
            f"skipped_no_title={counters['skipped_no_title']:,} "
            f"total={counters['total']:,}"
        )
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate movie_card + lex schema to title_normalized + "
            "hyphen-variant postings, then rebuild every row."
        ),
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Apply the schema migration only; skip truncation + rebuild.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without making any changes.",
    )
    parser.add_argument(
        "--max-movies",
        type=int,
        default=None,
        help="Cap the number of movies processed during rebuild (smoke-test slice).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(
        run(
            schema_only=args.schema_only,
            dry_run=args.dry_run,
            max_movies=args.max_movies,
        )
    )


if __name__ == "__main__":
    main()
