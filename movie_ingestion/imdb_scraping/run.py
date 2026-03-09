"""
Stage 4: IMDB Scraping via GraphQL API.

Scrapes IMDB data for all movies that passed the TMDB quality funnel
(status='tmdb_quality_passed'). Uses a single GraphQL query per movie
via DataImpulse residential/datacenter proxies with async concurrency
controlled by a global semaphore.

This is a one-time bulk operation. The daily update pipeline will have
its own separate (and much cheaper) scraping strategy designed later.

Usage:
    python -m movie_ingestion.imdb_scraping.run
"""

import asyncio
import time
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

from movie_ingestion.tracker import (
    INGESTION_DATA_DIR,
    MovieStatus,
    PipelineStage,
    batch_log_filter,
    init_db,
)
from .http_client import create_client, create_ua_generator
from .scraper import MovieResult, process_movie

# Load .env for proxy credentials (DATA_IMPULSE_LOGIN, etc.)
load_dotenv()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Number of movies per batch. After each batch we commit the SQLite
# transaction, bounding data loss on crash to at most one batch.
_COMMIT_BATCH_SIZE = 500

# Print a progress line every N movies processed.
_PROGRESS_INTERVAL = 50

# Initial concurrency limit — the single knob for throttle control.
# Start at 60 with DataImpulse IP rotation; tune based on error rate
# metrics from the DataImpulse dashboard.
_INITIAL_SEMAPHORE = 60

_STAGE = PipelineStage.IMDB_SCRAPE

# SQL for bulk-updating movie status after successful scraping
_UPDATE_STATUS_SQL = """
    UPDATE movie_progress
    SET status = ?, updated_at = CURRENT_TIMESTAMP
    WHERE tmdb_id = ?
"""

# Append-only error log for exceptions that escape process_movie.
# These would otherwise be silently lost since process_movie's own
# try/except only covers expected failure modes.
_ERROR_LOG_PATH = INGESTION_DATA_DIR / "stage4_errors.log"


# ---------------------------------------------------------------------------
# Error logging (same pattern as tmdb_fetcher.py)
# ---------------------------------------------------------------------------


def _log_unexpected_error(tmdb_id: int, exc: Exception) -> None:
    """
    Append a structured error entry to the Stage 4 debug log file.

    Called when asyncio.gather captures an exception that process_movie
    did not handle internally. Each entry includes the movie ID,
    timestamp, exception type, message, and full traceback so the
    error is diagnosable without re-running the pipeline.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
    tb_str = "".join(tb).rstrip()

    entry = (
        f"[{timestamp}] tmdb_id={tmdb_id} | "
        f"{type(exc).__name__}: {exc}\n"
        f"{tb_str}\n"
        f"{'-' * 80}\n"
    )

    with open(_ERROR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(entry)


# ---------------------------------------------------------------------------
# Batch orchestration
# ---------------------------------------------------------------------------


async def _scrape_all(db, candidates: list[tuple[int, str]]) -> dict:
    """
    Process all candidate movies in batches of _COMMIT_BATCH_SIZE.

    Creates a shared httpx client with proxy configuration, a global
    semaphore, and a UserAgent generator. Each batch is processed
    concurrently via asyncio.gather and committed to SQLite afterward.

    Args:
        db: Open SQLite connection.
        candidates: List of (tmdb_id, imdb_id) tuples to process.

    Returns:
        Counters dict with keys: scraped, filtered, errors.
    """
    counters = {"scraped": 0, "filtered": 0, "errors": 0}
    total = len(candidates)

    semaphore = asyncio.Semaphore(_INITIAL_SEMAPHORE)
    ua = create_ua_generator()

    async with create_client() as client:
        for batch_start_idx in range(0, total, _COMMIT_BATCH_SIZE):
            chunk = candidates[batch_start_idx: batch_start_idx + _COMMIT_BATCH_SIZE]
            batch_num = batch_start_idx // _COMMIT_BATCH_SIZE + 1
            batch_start = time.monotonic()

            print(f"\n{'=' * 60}")
            print(f"  Batch {batch_num}: processing {len(chunk)} movies "
                  f"(position {batch_start_idx + 1}-"
                  f"{batch_start_idx + len(chunk)} of {total})")
            print(f"{'=' * 60}\n")

            # --- Phase 1: Async HTTP fetching (no DB writes) ---
            tasks = [
                process_movie(client, semaphore, ua, tmdb_id, imdb_id)
                for tmdb_id, imdb_id in chunk
            ]

            # return_exceptions=True prevents one movie's unexpected exception
            # from aborting the entire chunk. We inspect results afterward to
            # log and count any exceptions that process_movie didn't handle.
            results = await asyncio.gather(*tasks, return_exceptions=True)

            batch_elapsed = time.monotonic() - batch_start

            # --- Phase 2: Collect outcomes and do bulk DB writes ---
            scraped_ids: list[int] = []
            filtered_entries: list[tuple[int, str, str, str | None]] = []
            unexpected_errors = 0

            for (tmdb_id, imdb_id), result in zip(chunk, results):
                if isinstance(result, Exception):
                    _log_unexpected_error(tmdb_id, result)
                    counters["errors"] += 1
                    unexpected_errors += 1
                    print(f"  [tmdb={tmdb_id}] UNEXPECTED ERROR: "
                          f"{type(result).__name__}: {result}")
                elif result.status == "scraped":
                    scraped_ids.append(result.tmdb_id)
                    counters["scraped"] += 1
                elif result.status == "filtered":
                    filtered_entries.append(
                        (result.tmdb_id, _STAGE, result.reason, None)
                    )
                    counters["filtered"] += 1
                elif result.status == "error":
                    counters["errors"] += 1

            # Bulk DB writes — all writes happen here, sequentially,
            # after async HTTP is complete.
            db.executemany(
                _UPDATE_STATUS_SQL,
                [(MovieStatus.IMDB_SCRAPED, tid) for tid in scraped_ids],
            )
            batch_log_filter(db, filtered_entries)

            commit_start = time.monotonic()
            db.commit()
            commit_elapsed = time.monotonic() - commit_start

            # Batch summary
            print(f"\n  Batch {batch_num} complete in {batch_elapsed:.1f}s "
                  f"({len(chunk) / max(batch_elapsed, 0.001):.1f} movies/sec) — "
                  f"scraped={counters['scraped']:,}, "
                  f"filtered={counters['filtered']:,}, "
                  f"errors={counters['errors']:,}")
            if unexpected_errors:
                print(f"  WARNING: {unexpected_errors} unexpected errors "
                      f"in this batch (see {_ERROR_LOG_PATH})")
            if commit_elapsed > 0.1:
                print(f"  DB commit took {commit_elapsed:.2f}s")

            # Progress reporting at _PROGRESS_INTERVAL boundaries
            # (and always for the first batch to confirm startup).
            processed = min(batch_start_idx + _COMMIT_BATCH_SIZE, total)
            if processed % _PROGRESS_INTERVAL < _COMMIT_BATCH_SIZE or processed <= _COMMIT_BATCH_SIZE:
                print(
                    f"\n  Progress: {processed:,}/{total:,} | "
                    f"scraped={counters['scraped']:,} "
                    f"filtered={counters['filtered']:,} "
                    f"errors={counters['errors']:,}"
                )

    return counters


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """
    Execute Stage 4: scrape IMDB data for all quality-passed movies.

    1. Initialize the tracker DB.
    2. Query all movies with status='tmdb_quality_passed'.
    3. Fetch, transform, and persist in batched async chunks.
    4. Print summary statistics.

    On restart after a crash, this same query naturally returns only
    unprocessed movies since successful movies have already been moved
    to imdb_scraped or filtered_out.
    """
    print("\n" + "=" * 60)
    print("Stage 4: IMDB Scraping Pipeline (GraphQL)")
    print("=" * 60)

    print("\n  Initializing tracker database...")
    db_start = time.monotonic()
    db = init_db()
    print(f"  Database initialized in {time.monotonic() - db_start:.2f}s")

    try:
        rows = db.execute(
            "SELECT tmdb_id, imdb_id FROM movie_progress WHERE status = ?",
            (MovieStatus.TMDB_QUALITY_PASSED,),
        ).fetchall()
        candidates = [(row[0], row[1]) for row in rows]

        if not candidates:
            print("  No movies at tmdb_quality_passed status. Nothing to do.")
            return

        print(f"\n  {len(candidates):,} movies to scrape from IMDB")
        print(f"  Settings: batch_size={_COMMIT_BATCH_SIZE}, "
              f"semaphore={_INITIAL_SEMAPHORE}")

        start = time.monotonic()

        counters = asyncio.run(_scrape_all(db, candidates))

        elapsed = time.monotonic() - start
        hours = elapsed / 3600
        movies_per_sec = len(candidates) / max(elapsed, 0.001)
        avg_per_movie = elapsed / max(len(candidates), 1)

        print("\n" + "=" * 60)
        print("Stage 4 Complete")
        print("=" * 60)
        print(f"  Total candidates:  {len(candidates):,}")
        print(f"  Scraped:           {counters['scraped']:,}")
        print(f"  Filtered out:      {counters['filtered']:,}")
        print(f"  Errors:            {counters['errors']:,}")
        print(f"  Duration:          {elapsed:,.0f}s ({hours:.1f}h)")
        print(f"  Throughput:        {movies_per_sec:.1f} movies/sec")
        print(f"  Avg per movie:     {avg_per_movie:.2f}s")
        print("=" * 60)
    finally:
        # Ensure the connection is always closed, even on Ctrl+C or crash,
        # so we don't leave a dangling lock on the database file.
        db.close()
        print("  Database connection closed.")


if __name__ == "__main__":
    run()
