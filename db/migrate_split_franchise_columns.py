"""
One-shot migration: split movie_card.franchise_name_entry_ids into
separate lineage_entry_ids + shared_universe_entry_ids columns, then
backfill both from the untouched raw TEXT values on
movie_franchise_metadata.

Idempotent end-to-end. Safe to re-run. Doesn't touch lex.franchise_entry
/ lex.franchise_token (those are unchanged by this migration — the
lineage-vs-universe distinction is a property of the movie-to-entry
relationship, not of the entry itself), and doesn't touch any other
movie_card column. Movies with no movie_franchise_metadata row keep
the default empty arrays, which is correct.

Usage:
    python -m db.migrate_split_franchise_columns

Order of operations:
  1. ALTER TABLE movie_card: drop the old column and index if present,
     add the two new columns and their GIN indexes if absent.
  2. Read every row from movie_franchise_metadata.
  3. For each row, call write_franchise_data (which is idempotent —
     the ON CONFLICT DO NOTHING upserts against lex.franchise_entry /
     lex.franchise_token leave already-populated registry data alone)
     to get a FranchiseEntryIds dataclass, then call
     update_movie_card_franchise_ids to stamp the three arrays on the
     card row. Movies with no corresponding movie_card row (should not
     happen in practice) are logged and skipped.

Concurrency: movies are processed in parallel via asyncio.gather with
a small semaphore cap so the pool's max_size=10 isn't exhausted.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

from db.postgres import (
    _execute_read,
    _execute_on_conn,
    pool,
    update_movie_card_franchise_ids,
)
from movie_ingestion.final_ingestion.ingest_movie import write_franchise_data

logger = logging.getLogger(__name__)

# Cap concurrent per-movie backfill tasks. Each task holds one connection
# for the duration of its write_franchise_data + update_movie_card_franchise_ids
# call; the pool tops out at max_size=10 so we stay a hair below that to
# leave room for other activity.
_CONCURRENCY = 8


# ---------------------------------------------------------------------------
# Step 1 — schema change.
# ---------------------------------------------------------------------------

_SCHEMA_STATEMENTS: list[str] = [
    # Drop the old GIN index first so its dependency on the column is gone
    # before the column itself is dropped. `IF EXISTS` makes the statement
    # safe on both a fresh DB (where the index was never created — the
    # updated init script doesn't build it) and a partially-migrated DB.
    "DROP INDEX IF EXISTS idx_movie_card_franchise_name_entry_ids;",
    "ALTER TABLE public.movie_card DROP COLUMN IF EXISTS franchise_name_entry_ids;",
    # New columns with the same BIGINT[] shape + non-null default the
    # updated init script declares. `IF NOT EXISTS` makes the add idempotent.
    "ALTER TABLE public.movie_card "
    "ADD COLUMN IF NOT EXISTS lineage_entry_ids BIGINT[] NOT NULL DEFAULT '{}';",
    "ALTER TABLE public.movie_card "
    "ADD COLUMN IF NOT EXISTS shared_universe_entry_ids BIGINT[] NOT NULL DEFAULT '{}';",
    # Matching GIN indexes for the && overlap predicates in
    # fetch_franchise_movie_ids. Plain GIN — same reason as the preserved
    # indexes: the column is BIGINT[], and gin__int_ops is INT[]-only.
    "CREATE INDEX IF NOT EXISTS idx_movie_card_lineage_entry_ids "
    "ON public.movie_card USING GIN (lineage_entry_ids);",
    "CREATE INDEX IF NOT EXISTS idx_movie_card_shared_universe_entry_ids "
    "ON public.movie_card USING GIN (shared_universe_entry_ids);",
]


async def apply_schema_change() -> None:
    """Apply the movie_card schema change in a single transaction."""
    async with pool.connection() as conn:
        try:
            for stmt in _SCHEMA_STATEMENTS:
                await _execute_on_conn(conn, stmt, ())
            await conn.commit()
        except Exception:
            await conn.rollback()
            raise


# ---------------------------------------------------------------------------
# Step 2 — backfill.
# ---------------------------------------------------------------------------


async def _fetch_franchise_metadata_rows() -> list[tuple[int, str | None, str | None, list[str]]]:
    """Read every row from movie_franchise_metadata.

    Returns a list of (movie_id, lineage, shared_universe, recognized_subgroups)
    tuples. recognized_subgroups is normalized to a list (defaulting to []
    when the column is NULL) so downstream code doesn't need to re-check.
    """
    rows = await _execute_read(
        "SELECT movie_id, lineage, shared_universe, recognized_subgroups "
        "FROM public.movie_franchise_metadata"
    )
    normalized: list[tuple[int, str | None, str | None, list[str]]] = []
    for movie_id, lineage, shared_universe, subgroups in rows:
        normalized.append((
            int(movie_id),
            lineage,
            shared_universe,
            list(subgroups) if subgroups is not None else [],
        ))
    return normalized


async def _backfill_one(
    movie_id: int,
    lineage: str | None,
    shared_universe: str | None,
    subgroups: list[str],
    semaphore: asyncio.Semaphore,
    counters: dict[str, int],
) -> None:
    """Resolve entry ids for one movie and stamp the three card columns.

    Runs under `semaphore` to bound pool usage. Missing movie_card rows
    (ValueError from update_movie_card_franchise_ids when rowcount=0)
    are logged and counted rather than aborting the whole backfill.
    """
    async with semaphore:
        try:
            entry_ids = await write_franchise_data(
                lineage=lineage,
                shared_universe=shared_universe,
                recognized_subgroups=subgroups,
            )
            await update_movie_card_franchise_ids(
                movie_id=movie_id,
                lineage_entry_ids=entry_ids.lineage,
                shared_universe_entry_ids=entry_ids.shared_universe,
                subgroup_entry_ids=entry_ids.subgroup,
            )
            counters["updated"] += 1
        except ValueError as e:
            # Movie has franchise metadata but no movie_card row — log
            # and skip. Unexpected in a healthy DB, but not fatal for
            # the migration.
            logger.warning("Skipping movie_id=%s: %s", movie_id, e)
            counters["skipped"] += 1


async def backfill_franchise_columns() -> dict[str, int]:
    """Backfill lineage_entry_ids + shared_universe_entry_ids on every card.

    Returns a dict of counters: {"updated": N, "skipped": M, "total": T}.
    """
    rows = await _fetch_franchise_metadata_rows()
    total = len(rows)
    print(f"Backfilling {total:,} movies with franchise metadata...")

    counters = {"updated": 0, "skipped": 0, "total": total}
    semaphore = asyncio.Semaphore(_CONCURRENCY)

    # Fire all tasks and let the semaphore shape the actual concurrency.
    # asyncio.gather preserves exception propagation; ValueError cases are
    # caught inside _backfill_one so they don't abort the batch.
    tasks = [
        _backfill_one(movie_id, lineage, shared_universe, subgroups, semaphore, counters)
        for movie_id, lineage, shared_universe, subgroups in rows
    ]

    # Progress reporting in chunks rather than per-task to keep stdout quiet.
    chunk = 2000
    for i in range(0, len(tasks), chunk):
        await asyncio.gather(*tasks[i : i + chunk])
        done = min(i + chunk, len(tasks))
        print(f"  {done:,}/{total:,} processed", flush=True)

    return counters


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


async def run(*, schema_only: bool = False) -> None:
    # Pool is created with open=False in db.postgres so it doesn't
    # open at import time; FastAPI opens it in its lifespan. Standalone
    # scripts like this one have to open and close it explicitly.
    await pool.open()
    try:
        start = time.monotonic()
        print("Step 1: applying schema change...")
        await apply_schema_change()
        print("  done.")

        if schema_only:
            print("--schema-only set; skipping backfill.")
            return

        print("Step 2: backfilling franchise columns...")
        counters = await backfill_franchise_columns()
        elapsed = time.monotonic() - start
        print(
            f"Done in {elapsed:.1f}s. "
            f"updated={counters['updated']:,} "
            f"skipped={counters['skipped']:,} "
            f"total={counters['total']:,}"
        )
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Apply the ALTER TABLE statements only; skip the backfill.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run(schema_only=args.schema_only))


if __name__ == "__main__":
    main()
