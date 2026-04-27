"""
One-shot backfill: populate ``movie_card.release_format`` for every
existing row.

What it does
------------
1. Applies an idempotent ``ALTER TABLE`` that adds ``release_format``
   to ``movie_card`` (no-op if the column is already present, e.g. on
   a fresh install where the init script already created it).
2. Reads ``(tmdb_id, imdb_title_type)`` for every row in the SQLite
   tracker's ``imdb_data`` table.
3. Buckets those tmdb_ids by their mapped :class:`ReleaseFormat` int
   id and issues one bulk ``UPDATE ... WHERE movie_id = ANY(...)`` per
   bucket. UNKNOWN (0) is skipped — the column default already covers
   it, so we only write the four buckets that need a non-default value.

Idempotency: the ALTER is ``IF NOT EXISTS``, and the UPDATE is by
movie_id with a constant value, so re-running the script is safe and
produces no diff once the column is converged.

Source of truth: ``movie.imdb_data.imdb_title_type`` lives in the
tracker, not in Postgres. We deliberately read it from the tracker so
this backfill doesn't depend on the (yet-to-be-populated) Postgres
column it's writing.

Usage
-----
::

    # Add the column (if missing) and run the full backfill.
    python -m movie_ingestion.backfill.backfill_release_format

    # Add the column and exit — no UPDATEs.
    python -m movie_ingestion.backfill.backfill_release_format --schema-only

    # Report bucket counts and skip every write.
    python -m movie_ingestion.backfill.backfill_release_format --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import time
from collections import defaultdict

from db.postgres import _execute_on_conn, _execute_read, pool
from movie_ingestion.tracker import TRACKER_DB_PATH
from schemas.enums import ReleaseFormat, release_format_id_for_imdb_type

logger = logging.getLogger(__name__)


# Idempotent: re-running on a fresh install (where the init script
# already created the column) is a no-op. NOT NULL DEFAULT 0 means
# every existing row materializes as UNKNOWN immediately, then the
# per-bucket UPDATEs below promote them to their real values.
_ADD_COLUMN_SQL = (
    "ALTER TABLE public.movie_card "
    "ADD COLUMN IF NOT EXISTS release_format SMALLINT NOT NULL DEFAULT 0;"
)

# Precomputed {int id -> enum member name} for human-readable progress
# output. Mirrors the lookup-dict idiom established for AwardCeremony
# (CEREMONY_BY_EVENT_TEXT) — avoids re-scanning the enum on every line.
_LABEL_BY_RELEASE_FORMAT_ID: dict[int, str] = {
    member.release_format_id: member.name for member in ReleaseFormat
}


def _load_tracker_title_types() -> dict[int, str | None]:
    """Read every (tmdb_id, imdb_title_type) pair from the tracker.

    Returns a dict so downstream bucketing can look up a movie_id even
    when the same id appears in movie_card but has no tracker row (we
    skip those — they default to UNKNOWN via the column default).
    """
    conn = sqlite3.connect(TRACKER_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT tmdb_id, imdb_title_type FROM imdb_data"
        )
        return {int(row["tmdb_id"]): row["imdb_title_type"] for row in cur}
    finally:
        conn.close()


def _bucket_by_release_format(
    title_types: dict[int, str | None],
) -> dict[int, list[int]]:
    """Group tmdb_ids by their mapped ReleaseFormat int id.

    The grouping uses ``release_format_id_for_imdb_type`` so the bucket
    boundaries here stay in lockstep with what the live ingest path
    writes via ``ingest_movie_card``.
    """
    buckets: dict[int, list[int]] = defaultdict(list)
    for tmdb_id, imdb_type in title_types.items():
        fmt_id = release_format_id_for_imdb_type(imdb_type)
        buckets[fmt_id].append(tmdb_id)
    return buckets


async def _fetch_movie_card_ids() -> set[int]:
    """Return every movie_id currently present in movie_card.

    Used to filter the tracker buckets to rows that actually exist in
    Postgres — UPDATEs against missing ids would be no-ops, but the
    intersection lets the dry-run report meaningful "would update"
    counts.
    """
    rows = await _execute_read("SELECT movie_id FROM public.movie_card")
    return {int(row[0]) for row in rows}


def _format_label(fmt_id: int) -> str:
    """Human label for a ReleaseFormat int id (UNKNOWN included)."""
    return _LABEL_BY_RELEASE_FORMAT_ID.get(fmt_id, f"id={fmt_id}")


async def _update_bucket(fmt_id: int, movie_ids: list[int]) -> int:
    """Run one bulk UPDATE for ``fmt_id`` over ``movie_ids``.

    Returns the number of rows the update touched. The bulk
    ``ANY(%s::bigint[])`` form mirrors the pattern used by the
    existing rebuild_lexical_postings backfill — single round trip per
    bucket regardless of size.
    """
    if not movie_ids:
        return 0
    async with pool.connection() as conn:
        try:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE public.movie_card "
                    "SET release_format = %s, updated_at = now() "
                    "WHERE movie_id = ANY(%s::bigint[])",
                    (fmt_id, movie_ids),
                )
                touched = cur.rowcount
            await conn.commit()
            return int(touched) if touched is not None else 0
        except Exception:
            await conn.rollback()
            raise


async def _report_final_counts() -> None:
    """Log the final per-format row counts straight from movie_card."""
    rows = await _execute_read(
        "SELECT release_format, COUNT(*) "
        "FROM public.movie_card "
        "GROUP BY release_format "
        "ORDER BY release_format"
    )
    print("Final movie_card.release_format distribution:")
    for fmt_id, count in rows:
        print(f"  {_format_label(int(fmt_id)):>9} ({fmt_id}): {int(count):>7,}")


async def run(*, schema_only: bool = False, dry_run: bool = False) -> None:
    # The pool defaults to open=False at import; this script runs
    # outside the FastAPI lifespan, so we manage open/close ourselves.
    await pool.open()
    try:
        start = time.monotonic()

        print("Step 1: ensuring release_format column exists...")
        if dry_run:
            print(f"  [dry-run] would execute: {_ADD_COLUMN_SQL}")
        else:
            async with pool.connection() as conn:
                try:
                    await _execute_on_conn(conn, _ADD_COLUMN_SQL, ())
                    await conn.commit()
                except Exception:
                    await conn.rollback()
                    raise
            print("  done.")

        if schema_only:
            print("--schema-only set; skipping backfill UPDATEs.")
            return

        print("Step 2: loading imdb_title_type from tracker...")
        title_types = _load_tracker_title_types()
        print(f"  loaded {len(title_types):,} tracker rows.")

        print("Step 3: bucketing tmdb_ids by release format...")
        buckets = _bucket_by_release_format(title_types)
        # Report all buckets including UNKNOWN so the operator can
        # eyeball coverage; the UPDATEs below skip UNKNOWN (default).
        for fmt_id in sorted(buckets):
            print(
                f"  {_format_label(fmt_id):>9} ({fmt_id}): "
                f"{len(buckets[fmt_id]):>7,} tracker rows"
            )

        print("Step 4: intersecting with movie_card...")
        existing_ids = await _fetch_movie_card_ids()
        print(f"  {len(existing_ids):,} rows in movie_card.")

        # Surface movie_card rows with no tracker imdb_data row. After
        # the ALTER, these stay release_format=UNKNOWN because we have
        # no source data to classify them — exactly the case the UNKNOWN
        # sentinel exists to flag. Logging here lets an operator decide
        # whether to re-scrape before re-running.
        tracker_ids = set(title_types.keys())
        orphans = existing_ids - tracker_ids
        if orphans:
            logger.warning(
                "%d movie_card rows have no tracker imdb_data row "
                "(will remain release_format=UNKNOWN). Sample movie_ids: %s",
                len(orphans), sorted(orphans)[:5],
            )

        # Filter each bucket to ids actually present in movie_card.
        # Tracker has more rows than movie_card (filtered_out movies,
        # in-flight statuses, etc.), so this avoids reporting inflated
        # "would update" numbers in the dry-run.
        for fmt_id in list(buckets):
            buckets[fmt_id] = [mid for mid in buckets[fmt_id] if mid in existing_ids]

        # True post-backfill UNKNOWN total = intersected UNKNOWN bucket
        # (movie_card rows whose tracker imdb_title_type maps to UNKNOWN)
        # PLUS orphans (movie_card rows with no tracker row at all). The
        # dry-run uses this so its preview matches what _report_final_counts
        # will print after a live run.
        unknown_id = ReleaseFormat.UNKNOWN.release_format_id
        effective_unknown = len(buckets.get(unknown_id, [])) + len(orphans)

        print("Step 5: running per-bucket UPDATEs...")
        total_touched = 0
        for fmt_id, movie_ids in sorted(buckets.items()):
            label = _format_label(fmt_id)
            if fmt_id == unknown_id:
                # UNKNOWN matches the column default — no UPDATE needed.
                # Report the effective count (intersected bucket + orphans),
                # not just the intersected bucket, so the operator sees the
                # actual audit number that will land in the column.
                print(
                    f"  {label:>9} ({fmt_id}): {effective_unknown:>7,} "
                    "movie_card rows (skipped — column default)"
                )
                continue

            if dry_run:
                print(
                    f"  [dry-run] {label:>9} ({fmt_id}): would update "
                    f"{len(movie_ids):,} rows."
                )
                continue

            touched = await _update_bucket(fmt_id, movie_ids)
            total_touched += touched
            print(f"  {label:>9} ({fmt_id}): updated {touched:>7,} rows")

        if not dry_run:
            print(f"Step 6: final distribution check...")
            await _report_final_counts()

        elapsed = time.monotonic() - start
        suffix = " (dry-run)" if dry_run else f" (wrote {total_touched:,} rows)"
        print(f"Done in {elapsed:.1f}s{suffix}.")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill movie_card.release_format from the SQLite "
            "tracker's imdb_title_type column."
        ),
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Run the ALTER TABLE only; skip the per-bucket UPDATEs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report per-bucket counts without writing anything.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(run(schema_only=args.schema_only, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
