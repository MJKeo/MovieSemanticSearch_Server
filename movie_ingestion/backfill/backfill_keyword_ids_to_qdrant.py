"""
One-shot backfill: write the ``keyword_ids`` payload key onto every
existing Qdrant point from the authoritative ``movie_card.keyword_ids``
column.

Why this exists
---------------
``keyword_ids`` (OverallKeyword tag IDs) was added to the Qdrant
hard-filter payload (``_build_qdrant_payload`` in
``movie_ingestion/final_ingestion/ingest_movie.py``) so the vector
channel can honor keyword filters via ``build_qdrant_filter``
(``db/vector_search.py``). New points get the key at ingest time, but
the ~100K points ingested before that change carry no ``keyword_ids``
payload. A ``MatchAny`` keyword filter treats an absent key as a
non-match, so **without this backfill a keyword filter on the vector
channel would silently exclude every already-ingested movie.** This
script closes that gap.

Postgres is the source of truth: ``movie_card.keyword_ids`` is already
populated for every ingested movie, so this is a Qdrant-only write (no
Postgres mutation) — unlike ``backfill_maturity_rank.py``, which keeps
both stores in sync. The genre/language backfill direction is the same
shape: read the column, push it into the matching payload key.

What it does
------------
1. Reads ``(movie_id, keyword_ids)`` for every row in ``public.movie_card``
   (ordered by ``movie_id`` so ``--limit`` is reproducible).
2. Chunks the rows and issues one ``batch_update_points`` per chunk, each
   carrying a ``SetPayloadOperation`` per movie (distinct ``keyword_ids``
   list per point — a single ``set_payload`` call can't express that).
   Empty arrays are written as ``[]`` to match what ingest writes, so the
   payload is uniform across old and new points.

Idempotency: ``set_payload`` is a merge-overwrite of just the
``keyword_ids`` key (other payload keys are untouched), so re-running is
safe and converges to the current ``movie_card`` state.

Ordering / durability: Qdrant ops use ``wait=False`` — the op is durably
accepted into Qdrant's WAL and applied asynchronously, matching
``backfill_maturity_rank.py``. Because Postgres already holds the
authoritative value, a crash mid-run just means a re-run finishes the
job; nothing can be stranded.

Usage
-----
::

    # Preview only — counts rows, writes nothing to Qdrant.
    python -m movie_ingestion.backfill.backfill_keyword_ids_to_qdrant --dry-run --limit 25

    # Full backfill.
    python -m movie_ingestion.backfill.backfill_keyword_ids_to_qdrant

    # Smoke test against a few points, or a non-default collection.
    python -m movie_ingestion.backfill.backfill_keyword_ids_to_qdrant --limit 50
    python -m movie_ingestion.backfill.backfill_keyword_ids_to_qdrant --collection movies --chunk-size 5000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from collections.abc import Iterator

from qdrant_client import models

from db.postgres import _execute_read, pool
from db.qdrant import qdrant_client

logger = logging.getLogger(__name__)

# Qdrant collection the ingester writes points to (ingest_movie.COLLECTION_ALIAS).
# Hardcoded rather than imported because importing the ingester pulls in the
# OpenAI/LLM client stack at module load; this is the write target by which
# every existing payload was created. Mirrors backfill_maturity_rank.py.
_DEFAULT_COLLECTION = "movies"

# Movies per batch_update_points round trip. Bounds request size so memory and
# latency stay flat regardless of corpus size.
_DEFAULT_BATCH_SIZE = 2000

# Per-call timeout for the Qdrant payload ops. The shared client defaults to a
# 10s search-tuned deadline (db/qdrant.py); a bulk payload op needs more
# headroom even with wait=False.
_QDRANT_OP_TIMEOUT = 60

# Payload key written by this backfill — must match the key set in
# _build_qdrant_payload and read in build_qdrant_filter.
_PAYLOAD_KEY = "keyword_ids"


async def _load_keyword_ids(limit: int | None) -> list[tuple[int, list[int]]]:
    """Return ``[(movie_id, keyword_ids), ...]`` from movie_card.

    movie_card.keyword_ids is the authoritative source (already populated
    at ingest). ``ORDER BY movie_id`` makes ``--limit`` reproducible. The
    full result is small — ~100K rows of short int arrays — so a single
    read is fine; chunking happens on the Qdrant write side.
    """
    query = "SELECT movie_id, keyword_ids FROM public.movie_card ORDER BY movie_id"
    params: tuple = ()
    if limit is not None:
        query += " LIMIT %s"
        params = (limit,)
    rows = await _execute_read(query, params)
    # psycopg3 decodes int[] columns to list[int]; coerce defensively in case
    # a NULL slips through (column is NOT NULL DEFAULT '{}', so this is belt
    # and suspenders) so the payload always carries a real list.
    return [(int(movie_id), list(keyword_ids or [])) for movie_id, keyword_ids in rows]


def _chunked(
    items: list[tuple[int, list[int]]],
    size: int,
) -> Iterator[list[tuple[int, list[int]]]]:
    """Yield successive ``size``-length slices of ``items``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def _write_chunk(
    chunk: list[tuple[int, list[int]]],
    *,
    collection: str,
) -> None:
    """Set the keyword_ids payload for every movie in ``chunk`` in one request.

    Each movie carries a distinct keyword_ids list, so we build one
    SetPayloadOperation per point and submit them together via
    batch_update_points (a single set_payload call applies the same payload
    to all its points, which can't express per-point lists).
    """
    operations = [
        models.SetPayloadOperation(
            set_payload=models.SetPayload(
                payload={_PAYLOAD_KEY: keyword_ids},
                points=[movie_id],
            )
        )
        for movie_id, keyword_ids in chunk
    ]
    await qdrant_client.batch_update_points(
        collection_name=collection,
        update_operations=operations,
        wait=False,
        timeout=_QDRANT_OP_TIMEOUT,
    )


async def run(
    *,
    limit: int | None,
    chunk_size: int,
    collection: str,
    dry_run: bool,
) -> None:
    await pool.open()
    try:
        start = time.monotonic()

        print("Step 1: reading movie_card.keyword_ids...")
        rows = await _load_keyword_ids(limit)
        print(f"  loaded {len(rows):,} movies.")

        if dry_run:
            sample = rows[:5]
            if sample:
                print("  sample of rows that would be written:")
                for movie_id, keyword_ids in sample:
                    print(f"    movie_id={movie_id} → keyword_ids={keyword_ids}")
            elapsed = time.monotonic() - start
            print(
                f"Done in {elapsed:.1f}s (dry-run, no Qdrant writes; "
                f"would update {len(rows):,} points in '{collection}')."
            )
            return

        print(
            f"Step 2: writing keyword_ids payload to Qdrant "
            f"(collection='{collection}', chunk_size={chunk_size:,})..."
        )
        total = 0
        chunk_index = 0
        for chunk in _chunked(rows, chunk_size):
            chunk_index += 1
            await _write_chunk(chunk, collection=collection)
            total += len(chunk)
            print(f"  chunk {chunk_index}: submitted {len(chunk):,} points (total {total:,})")

        elapsed = time.monotonic() - start
        print(
            f"Done in {elapsed:.1f}s "
            f"(submitted keyword_ids for {total:,} points to '{collection}', "
            f"wait=False — applied asynchronously by Qdrant)."
        )
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill the keyword_ids payload onto existing Qdrant points "
            "from the authoritative movie_card.keyword_ids column."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Cap the number of movie_card rows processed. Use a small value "
            "(e.g. 25) for smoke testing before the full run."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Points per batch_update_points round trip (default {_DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=_DEFAULT_COLLECTION,
        help=f"Target Qdrant collection (default '{_DEFAULT_COLLECTION}').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read + count, then exit without writing to Qdrant.",
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
            collection=args.collection,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
