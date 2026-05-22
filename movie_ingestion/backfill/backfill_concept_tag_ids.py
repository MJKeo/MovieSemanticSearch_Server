"""
One-shot backfill: write ``movie_card.concept_tag_ids`` from the
majority-vote merge of the three concept_tags runs stored in the
tracker's ``generated_metadata`` table.

What it does
------------
1. Streams ``(tmdb_id, concept_tags, concept_tags_run_2, concept_tags_run_3)``
   from the SQLite tracker for every eligible movie that has all three
   runs populated, ordered by ``tmdb_id``. An optional ``--limit N``
   caps the read for smoke testing.
2. For each row: parses the three JSON strings into ``ConceptTagsOutput``,
   majority-merges them via
   ``movie_ingestion.metadata_generation.concept_tags_merge`` (≥2/3 votes
   for list categories; mode-with-first-run-tiebreak for endings),
   re-applies ``apply_deterministic_fixups()`` (TWIST_VILLAIN → PLOT_TWIST
   implication + per-list dedup — matches the live generator's contract,
   guards against any legacy run rows that predate the fixup), and
   flattens to a sorted/deduped ``list[int]`` via
   ``ConceptTagsOutput.all_concept_tag_ids()`` (also filters out
   ``EndingTag.NO_CLEAR_CHOICE`` id=-1).
3. Buffers merged rows up to ``--chunk-size`` then flushes a single
   binary-format COPY into a connection-scoped TEMP TABLE followed by
   one ``UPDATE ... FROM`` join against ``public.movie_card``. ``updated_at``
   is bumped on every touched row.

Memory profile: peak holds ``chunk_size`` merged rows (plus a few
transient ``ConceptTagsOutput`` instances during the per-row merge) —
not the full ~101K eligible corpus.

Idempotency: the merge is deterministic (categories sorted by
``concept_tag_id``, endings tiebreak by first run) so re-running the
backfill on the same tracker state produces byte-identical column
values; only ``updated_at`` advances.

The chunked write reuses a single dedicated Postgres connection so the
TEMP TABLE is created exactly once. Each chunk runs in its own
transaction so a mid-run failure only loses the in-flight chunk.

Usage
-----
::

    # Full backfill (all eligible movies with 3 runs).
    python -m movie_ingestion.backfill.backfill_concept_tag_ids

    # Smoke test on a few movies.
    python -m movie_ingestion.backfill.backfill_concept_tag_ids --limit 5

    # Preview what would be written without touching Postgres.
    python -m movie_ingestion.backfill.backfill_concept_tag_ids --dry-run --limit 25

    # Tune chunk size if needed.
    python -m movie_ingestion.backfill.backfill_concept_tag_ids --chunk-size 10000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import time
from collections.abc import Iterator

from db.postgres import pool
from movie_ingestion.metadata_generation.concept_tags_merge import majority_merge
from movie_ingestion.tracker import TRACKER_DB_PATH
from schemas.metadata import ConceptTagsOutput

logger = logging.getLogger(__name__)


# Temp table used as the join target for each chunk's UPDATE. Created
# once on our dedicated connection and reused per chunk via TRUNCATE,
# so we do not pay create/drop cost N times. The default
# ON COMMIT PRESERVE ROWS keeps the table alive across our per-chunk
# commits; the table dies with the connection when the pool closes.
_CREATE_TEMP_SQL = (
    "CREATE TEMP TABLE _concept_tag_updates ("
    "    movie_id BIGINT PRIMARY KEY,"
    "    tag_ids  INT[] NOT NULL"
    ")"
)

_TRUNCATE_TEMP_SQL = "TRUNCATE _concept_tag_updates"

# Binary format COPY avoids the text-mode array-literal encoding
# (`{1,2,3}`) on every row — psycopg3's built-in adapters handle
# list[int] → int4[] natively in binary mode.
_COPY_BINARY_SQL = (
    "COPY _concept_tag_updates (movie_id, tag_ids) "
    "FROM STDIN WITH (FORMAT BINARY)"
)

_UPDATE_FROM_TEMP_SQL = (
    "UPDATE public.movie_card AS m "
    "SET concept_tag_ids = u.tag_ids, updated_at = now() "
    "FROM _concept_tag_updates u "
    "WHERE m.movie_id = u.movie_id"
)


def _stream_eligible_runs(
    limit: int | None,
    fetch_size: int = 1000,
) -> Iterator[tuple[int, str, str, str]]:
    """Yield ``(tmdb_id, run_1_json, run_2_json, run_3_json)`` one row at a time.

    Iterates the SQLite cursor with ``fetchmany`` so we never hold more
    than ``fetch_size`` raw JSON triples in memory at once. ``ORDER BY
    tmdb_id`` makes ``--limit N`` reproducible across re-runs while
    metadata generation is still landing new rows.
    """
    if not TRACKER_DB_PATH.exists():
        raise FileNotFoundError(
            f"Tracker DB not found at {TRACKER_DB_PATH}. "
            "Run from the project root and ensure metadata generation "
            "has populated tracker.db."
        )

    sql = (
        "SELECT tmdb_id, concept_tags, concept_tags_run_2, concept_tags_run_3 "
        "FROM generated_metadata "
        "WHERE eligible_for_concept_tags = 1 "
        "  AND concept_tags        IS NOT NULL "
        "  AND concept_tags_run_2  IS NOT NULL "
        "  AND concept_tags_run_3  IS NOT NULL "
        "ORDER BY tmdb_id"
    )
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)

    db = sqlite3.connect(str(TRACKER_DB_PATH))
    try:
        cur = db.execute(sql, params)
        while True:
            rows = cur.fetchmany(fetch_size)
            if not rows:
                break
            for row in rows:
                yield int(row[0]), row[1], row[2], row[3]
    finally:
        db.close()


def _merge_and_flatten(
    tmdb_id: int,
    run_1: str,
    run_2: str,
    run_3: str,
) -> tuple[int, list[int]]:
    """Parse three JSON runs, majority-merge, fixup, flatten to id list.

    Raises whatever ``model_validate_json`` / ``majority_merge`` raise on
    malformed input — the caller logs + skips.
    """
    outputs = [
        ConceptTagsOutput.model_validate_json(run_1),
        ConceptTagsOutput.model_validate_json(run_2),
        ConceptTagsOutput.model_validate_json(run_3),
    ]
    # Re-apply deterministic fixups on the merged result. Each input run
    # was already fixed up at generation time, but the majority threshold
    # can drop the implied PLOT_TWIST if the upstream rows were written
    # before the fixup logic existed. apply_deterministic_fixups() is
    # idempotent and cheap, so we run it unconditionally to match the
    # live generator's contract.
    merged = majority_merge(outputs).apply_deterministic_fixups()
    return tmdb_id, merged.all_concept_tag_ids()


async def _write_chunk(
    conn,
    chunk: list[tuple[int, list[int]]],
) -> int:
    """COPY ``chunk`` into the temp table, run UPDATE FROM, truncate, commit.

    Returns the number of ``movie_card`` rows touched by the UPDATE.
    """
    if not chunk:
        return 0
    try:
        async with conn.cursor() as cur:
            async with cur.copy(_COPY_BINARY_SQL) as copy:
                # Binary COPY does not negotiate column types over the
                # wire, so psycopg has to be told what Postgres types to
                # encode for. Python int defaults to int4 — wrong for
                # the BIGINT movie_id column — and triggers a
                # "insufficient data left in message" ProtocolViolation
                # without this call.
                copy.set_types(["bigint", "int4[]"])
                for movie_id, tag_ids in chunk:
                    await copy.write_row((movie_id, tag_ids))

            await cur.execute(_UPDATE_FROM_TEMP_SQL)
            touched = cur.rowcount or 0

            await cur.execute(_TRUNCATE_TEMP_SQL)
        await conn.commit()
    except Exception:
        await conn.rollback()
        raise
    return int(touched)


async def run(
    *,
    limit: int | None,
    chunk_size: int,
    dry_run: bool,
) -> None:
    await pool.open()
    try:
        start = time.monotonic()

        # --------------------------------------------------------------
        # Dry-run path: stream + merge, collect a small sample, no DB write.
        # --------------------------------------------------------------
        if dry_run:
            print("Step 1: streaming + merging (dry-run, no Postgres writes)...")
            total_processed = 0
            parse_failures = 0
            sample: list[tuple[int, list[int]]] = []

            for tmdb_id, r1, r2, r3 in _stream_eligible_runs(limit):
                try:
                    merged_row = _merge_and_flatten(tmdb_id, r1, r2, r3)
                except Exception as exc:
                    parse_failures += 1
                    logger.warning(
                        "tmdb_id=%d: failed to parse/merge concept_tags runs (%s); skipping",
                        tmdb_id, exc,
                    )
                    continue

                total_processed += 1
                if len(sample) < 5:
                    sample.append(merged_row)

            print(
                f"  would process {total_processed:,} movies "
                f"({parse_failures:,} parse/merge failures)."
            )
            if sample:
                print(f"  sample of {len(sample)} merged rows:")
                for movie_id, tag_ids in sample:
                    print(f"    movie_id={movie_id} → concept_tag_ids={tag_ids}")

            elapsed = time.monotonic() - start
            print(f"Done in {elapsed:.1f}s (dry-run).")
            return

        # --------------------------------------------------------------
        # Live path: stream + merge + flush chunks under one connection.
        # --------------------------------------------------------------
        print(
            f"Step 1: streaming + merging + bulk-updating "
            f"(chunk_size={chunk_size:,})..."
        )

        total_processed = 0
        total_touched = 0
        parse_failures = 0

        async with pool.connection() as conn:
            # Create the temp table once on this connection.
            async with conn.cursor() as cur:
                await cur.execute(_CREATE_TEMP_SQL)
            await conn.commit()

            buffer: list[tuple[int, list[int]]] = []
            chunk_index = 0

            async def _flush() -> None:
                nonlocal chunk_index, total_touched, total_processed
                if not buffer:
                    return
                chunk_index += 1
                touched = await _write_chunk(conn, buffer)
                total_touched += touched
                total_processed += len(buffer)
                print(
                    f"  chunk {chunk_index}: copied {len(buffer):,} rows, "
                    f"updated {touched:,} movie_card rows"
                )
                buffer.clear()

            for tmdb_id, r1, r2, r3 in _stream_eligible_runs(limit):
                try:
                    merged_row = _merge_and_flatten(tmdb_id, r1, r2, r3)
                except Exception as exc:
                    parse_failures += 1
                    logger.warning(
                        "tmdb_id=%d: failed to parse/merge concept_tags runs (%s); skipping",
                        tmdb_id, exc,
                    )
                    continue

                buffer.append(merged_row)
                if len(buffer) >= chunk_size:
                    await _flush()

            # Flush the trailing partial chunk.
            await _flush()

        # Operators care about the rows-prepared vs rows-touched gap:
        # eligible movies that haven't reached movie_card yet show up
        # here as a delta, the same way backfill_release_format.py
        # surfaces "orphans".
        missing_in_movie_card = total_processed - total_touched
        if missing_in_movie_card > 0:
            logger.warning(
                "%d merged rows did not match any movie_card row "
                "(eligible movies upstream of final ingestion).",
                missing_in_movie_card,
            )

        elapsed = time.monotonic() - start
        print(
            f"Done in {elapsed:.1f}s "
            f"(processed {total_processed:,} movies, "
            f"updated {total_touched:,} movie_card rows, "
            f"{parse_failures:,} parse failures, "
            f"{missing_in_movie_card:,} not in movie_card)."
        )
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill movie_card.concept_tag_ids by majority-merging the "
            "three concept_tags runs stored in the SQLite tracker."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap the number of tracker rows processed. Use a small value "
            "(e.g. 5) for smoke testing before the full run."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5_000,
        help="Rows per COPY+UPDATE round-trip (default: 5000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse + merge, then exit without writing to Postgres.",
    )
    args = parser.parse_args()

    if args.chunk_size <= 0:
        parser.error("--chunk-size must be a positive integer")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer when provided")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(
        run(
            limit=args.limit,
            chunk_size=args.chunk_size,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
