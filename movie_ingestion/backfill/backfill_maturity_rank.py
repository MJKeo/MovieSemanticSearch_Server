"""One-shot backfill: re-resolve ``maturity_rank`` for every already-ingested
movie and write the corrected value to both stores that hold it.

Why this exists
---------------
``MaturityRating`` gained an alias table (TV / legacy / foreign certificates →
G/PG/PG-13/R/NC-17). Movies ingested before that change had any non-canonical
certificate silently collapse to UNRATED → ``maturity_rank`` stored as NULL,
hiding them from rank-range filters. This backfill recomputes the rank from the
*same* tracker source the ingester reads and converges the stored value.

``maturity_rank`` is a hard-filter field read from BOTH stores — Postgres
``movie_card`` (db/postgres.py range conditions; search_v2 metadata gate) and
the Qdrant point payload (db/vector_search.py range filter). Updating only one
would make the two retrieval paths disagree, so this script writes both.

Efficiency
----------
* Source ranks are resolved once from the SQLite tracker in a single query.
* We diff against the current ``movie_card`` value and write ONLY the movies
  whose rank actually changes (the alias addition only ever moves NULL → a real
  rank, so the changed set is small relative to the ~109K ingested corpus).
  Re-running after convergence writes nothing.
* Writes are bucketed by target rank and chunked: one bulk
  ``UPDATE ... WHERE movie_id = ANY(%s::bigint[])`` and one Qdrant
  ``set_payload(points=[...])`` per chunk — a single round trip each, no
  per-movie calls. This mirrors the efficient bucket pattern in
  ``backfill_release_format`` (one set-based UPDATE per bucket).

Crash safety / idempotency
--------------------------
Per chunk we write Qdrant FIRST, Postgres SECOND. The Postgres column is the
diff source, so writing it last means an interrupted run leaves any unfinished
movie still in the diff; a re-run repairs it (``set_payload`` is idempotent).
The reverse order could strand a stale Qdrant payload the next diff never
revisits. This assumes the two stores were consistent before the run — true by
construction, since the ingester writes both from the same value.

Usage
-----
::

    # Full backfill of both stores.
    python -m movie_ingestion.backfill.backfill_maturity_rank

    # Report what would change; write nothing.
    python -m movie_ingestion.backfill.backfill_maturity_rank --dry-run

    # Tune chunk size / target a non-default Qdrant collection.
    python -m movie_ingestion.backfill.backfill_maturity_rank --batch-size 5000
    python -m movie_ingestion.backfill.backfill_maturity_rank --collection movies
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sqlite3
import time
from collections import defaultdict
from typing import Iterator

from db.postgres import _execute_read, pool
from db.qdrant import qdrant_client
from implementation.classes.enums import MaturityRating
from movie_ingestion.tracker import TRACKER_DB_PATH

logger = logging.getLogger(__name__)

# Rank that means "no usable rating". Stored as NULL in Postgres and as a
# null/absent payload value in Qdrant — both excluded from rank-range filters.
_UNRATED_RANK = MaturityRating.UNRATED.maturity_rank

# Qdrant collection the ingester writes points to (ingest_movie.COLLECTION_ALIAS).
# Hardcoded here rather than imported because importing the ingester pulls in
# the OpenAI/LLM client stack at module load; this is the write target by which
# every existing payload was created.
_DEFAULT_COLLECTION = "movies"

# Chunk size for both the Postgres ANY(...) array and the Qdrant points list.
# Bounds statement/request size so memory and round-trip latency stay flat
# regardless of how many movies change.
_DEFAULT_BATCH_SIZE = 2000

# Per-call timeout for the Qdrant payload ops. The shared client defaults to a
# 10s deadline (db/qdrant.py) tuned for search; a bulk payload op needs more
# headroom even with wait=False, so we override it here.
_QDRANT_OP_TIMEOUT = 60


def _resolve_rank(imdb_rating: str | None, tmdb_rating: str | None) -> int | None:
    """Resolve a movie's stored rank exactly as ingestion does.

    Mirrors ``Movie.resolved_maturity_rating`` (IMDB-preferred, TMDB fallback)
    feeding ``MaturityRating.from_string_with_default``, then collapses UNRATED
    to ``None`` — identical to ingest_movie.py's ``maturity_rank == UNRATED ->
    None`` for both the Postgres column and the Qdrant payload.
    """
    raw = imdb_rating if imdb_rating else tmdb_rating
    rank = MaturityRating.from_string_with_default(raw).maturity_rank
    return None if rank == _UNRATED_RANK else rank


def _load_target_ranks() -> dict[int, int | None]:
    """Resolve the target rank for every 'ingested' movie from the tracker.

    Reads the IMDB and TMDB maturity strings in one join so resolution uses the
    same precedence as the live ingest path. Returns ``{tmdb_id: rank|None}``.
    """
    conn = sqlite3.connect(TRACKER_DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT mp.tmdb_id        AS tmdb_id, "
            "       i.maturity_rating AS imdb_rating, "
            "       t.maturity_rating AS tmdb_rating "
            "FROM movie_progress mp "
            "LEFT JOIN imdb_data i ON i.tmdb_id = mp.tmdb_id "
            "LEFT JOIN tmdb_data t ON t.tmdb_id = mp.tmdb_id "
            "WHERE mp.status = 'ingested'"
        )
        return {
            int(row["tmdb_id"]): _resolve_rank(row["imdb_rating"], row["tmdb_rating"])
            for row in cur
        }
    finally:
        conn.close()


async def _load_current_ranks() -> dict[int, int | None]:
    """Return ``{movie_id: maturity_rank|None}`` from movie_card.

    This is the diff baseline — comparing against it lets us skip movies whose
    rank is already correct, so a converged corpus produces zero writes.
    """
    rows = await _execute_read("SELECT movie_id, maturity_rank FROM public.movie_card")
    return {int(mid): (int(rank) if rank is not None else None) for mid, rank in rows}


def _chunked(items: list[int], size: int) -> Iterator[list[int]]:
    """Yield ``items`` in lists of at most ``size``."""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _rank_sort_key(rank: int | None) -> tuple[bool, int]:
    """Sort key that orders ``None`` last (None can't compare with ints)."""
    return (rank is None, rank if rank is not None else 0)


def _rank_label(rank: int | None) -> str:
    """Human label for progress output."""
    return "NULL (unrated)" if rank is None else f"rank {rank}"


async def _apply_bucket(
    rank: int | None,
    movie_ids: list[int],
    *,
    collection: str,
    batch_size: int,
    dry_run: bool,
) -> int:
    """Write ``rank`` to both stores for ``movie_ids``; return rows written.

    Per chunk: Qdrant payload first (idempotent — only ``maturity_rank`` is
    touched, other payload keys are preserved), then the Postgres column as the
    commit marker. See the module docstring for why this order is crash-safe.

    Qdrant ops use ``wait=False``: the op is durably accepted into Qdrant's WAL
    before the call returns (then applied/indexed asynchronously), so it is
    still ordered before the Postgres commit. ``wait=True`` would block on
    apply+index and blow past the client deadline on batches this size.

    A ``None`` target removes the Qdrant key via ``delete_payload`` rather than
    writing a null value: a missing key is excluded from rank-range filters
    exactly like NULL in Postgres, with no ambiguity over how Qdrant stores a
    null payload value.
    """
    total = 0
    for chunk in _chunked(movie_ids, batch_size):
        if dry_run:
            total += len(chunk)
            continue

        # 1) Qdrant: merge-update (or clear) only the maturity_rank payload key.
        if rank is None:
            await qdrant_client.delete_payload(
                collection_name=collection,
                keys=["maturity_rank"],
                points=chunk,
                wait=False,
                timeout=_QDRANT_OP_TIMEOUT,
            )
        else:
            await qdrant_client.set_payload(
                collection_name=collection,
                payload={"maturity_rank": rank},
                points=chunk,
                wait=False,
                timeout=_QDRANT_OP_TIMEOUT,
            )

        # 2) Postgres: bulk UPDATE in a single round trip for the whole chunk.
        async with pool.connection() as conn:
            try:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE public.movie_card "
                        "SET maturity_rank = %s, updated_at = now() "
                        "WHERE movie_id = ANY(%s::bigint[])",
                        (rank, chunk),
                    )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise

        total += len(chunk)
    return total


async def run(
    *,
    dry_run: bool = False,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    collection: str = _DEFAULT_COLLECTION,
) -> None:
    # The pool defaults to open=False at import; this script runs outside the
    # FastAPI lifespan, so we own open/close.
    await pool.open()
    try:
        start = time.monotonic()

        print("Step 1: resolving target ranks from tracker...")
        targets = _load_target_ranks()
        print(f"  {len(targets):,} ingested movies in tracker.")

        print("Step 2: reading current ranks from movie_card...")
        current = await _load_current_ranks()
        print(f"  {len(current):,} movie_card rows.")

        print("Step 3: diffing target vs current...")
        buckets: dict[int | None, list[int]] = defaultdict(list)
        missing = 0  # 'ingested' in tracker but absent from movie_card
        for tmdb_id, target_rank in targets.items():
            if tmdb_id not in current:
                missing += 1
                continue
            if current[tmdb_id] != target_rank:
                buckets[target_rank].append(tmdb_id)

        changed = sum(len(ids) for ids in buckets.values())
        print(f"  {changed:,} movies need an update.")
        for rank in sorted(buckets, key=_rank_sort_key):
            print(f"    {_rank_label(rank):>15}: {len(buckets[rank]):>7,}")
        if missing:
            logger.warning(
                "%d tracker 'ingested' movies have no movie_card row (skipped).",
                missing,
            )

        if not changed:
            print("Stores already converged; nothing to write.")
            return

        verb = "would update" if dry_run else "updated"
        print(f"Step 4: applying updates (batch={batch_size}, collection={collection})...")
        total = 0
        for rank in sorted(buckets, key=_rank_sort_key):
            written = await _apply_bucket(
                rank,
                buckets[rank],
                collection=collection,
                batch_size=batch_size,
                dry_run=dry_run,
            )
            total += written
            print(f"  {_rank_label(rank):>15}: {verb} {written:>7,}")

        elapsed = time.monotonic() - start
        suffix = " (dry-run)" if dry_run else ""
        print(f"Done in {elapsed:.1f}s — {total:,} movies{suffix}.")
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Re-resolve movie_card.maturity_rank (and the Qdrant payload) for "
            "every ingested movie from the tracker's maturity strings."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the per-rank change counts without writing anything.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help=f"Movies per UPDATE / set_payload chunk (default {_DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        help=f"Qdrant collection to update (default '{_DEFAULT_COLLECTION}').",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    asyncio.run(
        run(
            dry_run=args.dry_run,
            batch_size=args.batch_size,
            collection=args.collection,
        )
    )


if __name__ == "__main__":
    main()
