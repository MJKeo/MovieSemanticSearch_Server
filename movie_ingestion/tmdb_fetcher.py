"""
Stage 2: TMDB Detail Fetching.

Fetches expanded movie details from the TMDB API for every movie with
``status='pending'`` in the tracker database, extracts/transforms fields
into the ``tmdb_data`` SQLite table, and advances each movie's status to
``tmdb_fetched``.

All direct TMDB API calls go through ``db.tmdb.fetch_movie_details()`` —
this module owns only data transformation, persistence, and orchestration.

Usage:
    python -m movie_ingestion.tmdb_fetcher
"""

import asyncio
import json
from pathlib import Path
import struct
import time
import traceback
from datetime import datetime, timezone

import httpx

from db.tmdb import (
    AdaptiveRateLimiter,
    TMDBFetchError,
    access_token,
    fetch_movie_details,
)
from implementation.classes.enums import StreamingAccessType
from implementation.misc.helpers import create_watch_provider_offering_key
from movie_ingestion.tracker import (
    INGESTION_DATA_DIR,
    PipelineStage,
    init_db,
    log_filter,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGE = PipelineStage.TMDB_FETCH

# Number of movies per async batch.  After each batch completes we commit the
# SQLite transaction, bounding data loss on crash to at most one batch.
_COMMIT_BATCH_SIZE = 100

# Print a progress line every N movies processed.
_PROGRESS_INTERVAL = 500

# httpx connection pool ceiling — protects the local machine; the rate limiter
# (not this) is what enforces TMDB's per-second throughput cap.
_MAX_CONNECTIONS = 100

# Per-request timeout in seconds.
_REQUEST_TIMEOUT = 15.0

# Append-only debug log for unexpected exceptions that asyncio.gather captures
# via return_exceptions=True.  These would otherwise be silently lost since
# _process_movie's own try/except only covers expected failure modes.
_ERROR_LOG_PATH = INGESTION_DATA_DIR / "stage2_errors.log"

# Maps TMDB provider category names to our StreamingAccessType enum.
# TMDB's "free" and "ads" categories are intentionally excluded — we only
# track subscription, rent, and buy offerings.
_TMDB_CATEGORY_TO_ACCESS_TYPE: dict[str, StreamingAccessType] = {
    "flatrate": StreamingAccessType.SUBSCRIPTION,
    "rent": StreamingAccessType.RENT,
    "buy": StreamingAccessType.BUY,
}

# ---------------------------------------------------------------------------
# SQL templates (module-level to avoid repeated string construction)
# ---------------------------------------------------------------------------

_INSERT_TMDB_DATA_SQL = """
    INSERT OR REPLACE INTO tmdb_data (
        tmdb_id, imdb_id, title, release_date, duration, poster_url,
        watch_provider_keys, vote_count, popularity, vote_average,
        overview_length, genre_count, has_revenue, has_budget,
        has_production_companies, has_production_countries,
        has_keywords, has_cast_and_crew
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_PROGRESS_SQL = """
    UPDATE movie_progress
    SET imdb_id = ?, status = 'tmdb_fetched', updated_at = CURRENT_TIMESTAMP
    WHERE tmdb_id = ?
"""


# ---------------------------------------------------------------------------
# Debug error logging — append-only file for unexpected exceptions
# ---------------------------------------------------------------------------


def _log_unexpected_error(tmdb_id: int, exc: Exception) -> None:
    """
    Append a structured error entry to the Stage 2 debug log file.

    Called when asyncio.gather captures an exception that _process_movie did
    not handle internally (e.g. a SQLite operational error, disk-full, or an
    unanticipated data shape).  Each entry includes the movie ID, timestamp,
    exception type, message, and full traceback so the error is diagnosable
    without re-running the pipeline.
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
# Watch provider key encoding
# ---------------------------------------------------------------------------


def _pack_provider_keys(keys: list[int]) -> bytes | None:
    """
    Pack a sorted list of watch-provider offering keys into a compact BLOB.

    Each key is a 32-bit unsigned int.  Returns ``None`` for an empty list
    (stored as NULL in SQLite to save space).
    """
    if not keys:
        return None
    return struct.pack(f"<{len(keys)}I", *keys)


def _extract_watch_provider_keys(raw_providers: dict) -> list[int]:
    """
    Extract US-region watch provider offering keys from a TMDB response.

    Encodes each (provider_id, access_type) pair into the integer key format
    used by Qdrant and Postgres via ``create_watch_provider_offering_key``.
    Deduplicates and returns a deterministic sorted list so the packed BLOB
    is stable across re-fetches.
    """
    us_data = raw_providers.get("results", {}).get("US", {})
    keys: set[int] = set()

    for category, access_type in _TMDB_CATEGORY_TO_ACCESS_TYPE.items():
        for provider in us_data.get(category, []):
            key = create_watch_provider_offering_key(
                provider["provider_id"], access_type.type_id
            )
            keys.add(key)

    return sorted(keys)


# ---------------------------------------------------------------------------
# Field extraction — TMDB JSON → flat dict matching tmdb_data columns
# ---------------------------------------------------------------------------


def _extract_fields(raw: dict) -> dict:
    """
    Transform a raw TMDB expanded-detail response into a flat dict whose
    keys match the ``tmdb_data`` table columns.

    ``raw["id"]`` is accessed directly (not ``.get()``) because a response
    without an ID is a structural API violation that should raise loudly.
    """
    # Top-level fields with safe defaults
    overview = raw.get("overview") or ""
    genres = raw.get("genres") or []
    revenue = raw.get("revenue") or 0
    budget = raw.get("budget") or 0
    production_companies = raw.get("production_companies") or []
    production_countries = raw.get("production_countries") or []

    # Nested append_to_response fields — defensive .get() chains because
    # sub-resources may be absent or partially populated.
    keywords_list = raw.get("keywords", {}).get("keywords", []) or []
    credits = raw.get("credits") or {}
    cast_list = credits.get("cast") or []
    crew_list = credits.get("crew") or []

    # Watch providers are under "watch/providers" (the slash is literal)
    raw_providers = raw.get("watch/providers") or {}
    watch_keys = _extract_watch_provider_keys(raw_providers)

    return {
        "tmdb_id": raw["id"],
        "imdb_id": raw.get("imdb_id"),
        "title": raw.get("title"),
        "release_date": raw.get("release_date"),
        "duration": raw.get("runtime"),
        "poster_url": raw.get("poster_path"),
        "watch_provider_keys": watch_keys,
        # Quality filter fields
        "vote_count": raw.get("vote_count", 0),
        "popularity": raw.get("popularity", 0.0),
        "vote_average": raw.get("vote_average", 0.0),
        "overview_length": len(overview),
        "genre_count": len(genres),
        "has_revenue": 1 if revenue > 0 else 0,
        "has_budget": 1 if budget > 0 else 0,
        "has_production_companies": 1 if len(production_companies) > 0 else 0,
        "has_production_countries": 1 if len(production_countries) > 0 else 0,
        "has_keywords": 1 if len(keywords_list) > 0 else 0,
        "has_cast_and_crew": 1 if (len(cast_list) > 0 and len(crew_list) > 0) else 0,
    }


# ---------------------------------------------------------------------------
# Persistence — write extracted fields into SQLite (no commit)
# ---------------------------------------------------------------------------


def _persist_movie(db, fields: dict) -> None:
    """
    Insert one movie's extracted data into ``tmdb_data`` and advance its
    ``movie_progress`` status to ``tmdb_fetched``.

    Does NOT commit — the caller batches commits for efficiency.
    """
    packed_keys = _pack_provider_keys(fields["watch_provider_keys"])

    db.execute(
        _INSERT_TMDB_DATA_SQL,
        (
            fields["tmdb_id"],
            fields["imdb_id"],
            fields["title"],
            fields["release_date"],
            fields["duration"],
            fields["poster_url"],
            packed_keys,
            fields["vote_count"],
            fields["popularity"],
            fields["vote_average"],
            fields["overview_length"],
            fields["genre_count"],
            fields["has_revenue"],
            fields["has_budget"],
            fields["has_production_companies"],
            fields["has_production_countries"],
            fields["has_keywords"],
            fields["has_cast_and_crew"],
        ),
    )

    db.execute(_UPDATE_PROGRESS_SQL, (fields["imdb_id"], fields["tmdb_id"]))


# ---------------------------------------------------------------------------
# Single-movie async processing
# ---------------------------------------------------------------------------


async def _process_movie(
    client: httpx.AsyncClient,
    rate_limiter: AdaptiveRateLimiter,
    tmdb_id: int,
    db,
    counters: dict,
) -> None:
    """
    Fetch, extract, persist, and filter-check a single movie.

    All exception paths call ``log_filter`` so nothing is silently lost.
    Updates ``counters`` dict in-place (safe in single-threaded asyncio).
    """
    # --- Step 1: Fetch from TMDB ---
    try:
        raw = await fetch_movie_details(client, rate_limiter, tmdb_id)
    except TMDBFetchError:
        log_filter(db, tmdb_id, _STAGE, reason="tmdb_fetch_error")
        counters["errors"] += 1
        return
    except ValueError:
        log_filter(db, tmdb_id, _STAGE, reason="tmdb_parse_error")
        counters["errors"] += 1
        return

    # --- Step 2: Handle 404 (movie deleted / not found on TMDB) ---
    if raw is None:
        log_filter(db, tmdb_id, _STAGE, reason="tmdb_404")
        counters["filtered"] += 1
        return

    # --- Step 3: Extract fields ---
    try:
        fields = _extract_fields(raw)
    except Exception:
        log_filter(db, tmdb_id, _STAGE, reason="tmdb_extract_error")
        counters["errors"] += 1
        return

    # --- Step 4: Persist BEFORE the IMDB ID check so log_filter can read
    #     title/year from the tmdb_data table. ---
    _persist_movie(db, fields)

    # --- Step 5: Filter out movies without an IMDB ID (can't proceed to
    #     Stage 4 IMDB scraping). ---
    # NOTE: If the process crashes after this log_filter executes but before
    # the batch db.commit(), the movie remains 'pending' and will be re-
    # processed on restart.  _persist_movie is idempotent (INSERT OR REPLACE),
    # but log_filter appends a new filter_log row, producing a duplicate entry.
    # This is an accepted trade-off — the filter_log is a debugging aid, and
    # deduplication can be done at query time with DISTINCT or GROUP BY.
    if not fields["imdb_id"]:
        log_filter(db, tmdb_id, _STAGE, reason="missing_imdb_id")
        counters["filtered"] += 1
        return

    counters["fetched"] += 1


# ---------------------------------------------------------------------------
# Batch orchestration — chunked async loop with periodic commits
# ---------------------------------------------------------------------------


async def _fetch_all(db, pending_ids: list[int]) -> dict:
    """
    Process all pending movies in batches of ``_COMMIT_BATCH_SIZE``.

    Creates a shared httpx client and adaptive rate limiter, dispatches each
    batch concurrently via ``asyncio.gather``, and commits after every batch.
    Prints progress at ``_PROGRESS_INTERVAL`` boundaries.

    Returns:
        Counters dict with keys: fetched, filtered, errors.
    """
    counters = {"fetched": 0, "filtered": 0, "errors": 0}
    total = len(pending_ids)

    rate_limiter = AdaptiveRateLimiter(
        initial_rate=36.0, max_rate=40.0, burst=5, clean_window=120.0
    )

    headers = {"Authorization": f"Bearer {access_token()}"}
    limits = httpx.Limits(max_connections=_MAX_CONNECTIONS)
    timeout = httpx.Timeout(_REQUEST_TIMEOUT)

    async with httpx.AsyncClient(
        headers=headers, limits=limits, timeout=timeout
    ) as client:
        for i in range(0, total, _COMMIT_BATCH_SIZE):
            chunk = pending_ids[i : i + _COMMIT_BATCH_SIZE]

            tasks = [
                _process_movie(client, rate_limiter, tmdb_id, db, counters)
                for tmdb_id in chunk
            ]
            # return_exceptions=True prevents one unexpected exception from
            # aborting the entire chunk.  We inspect results afterward to
            # log and count any exceptions that _process_movie didn't handle.
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Pair each result back to its tmdb_id so the debug log
            # clearly identifies which movie caused the failure.
            for tmdb_id, result in zip(chunk, results):
                if isinstance(result, Exception):
                    _log_unexpected_error(tmdb_id, result)
                    counters["errors"] += 1

            db.commit()

            print(f"Committed batch {i}")

            # Progress reporting at _PROGRESS_INTERVAL boundaries
            processed = min(i + _COMMIT_BATCH_SIZE, total)
            if processed % _PROGRESS_INTERVAL < _COMMIT_BATCH_SIZE:
                print(
                    f"  Progress: {processed:,}/{total:,} | "
                    f"{rate_limiter.stats()}"
                )

    return counters


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """
    Execute Stage 2: fetch TMDB details for all pending movies.

    1. Initialize the tracker DB (creates tmdb_data table if needed).
    2. Query all movies with status='pending'.
    3. Fetch, extract, and persist in batched async chunks.
    4. Print summary statistics.
    """
    db = init_db()

    rows = db.execute(
        "SELECT tmdb_id FROM movie_progress WHERE status = 'pending'"
    ).fetchall()
    pending_ids = [row[0] for row in rows[:10000]]

    if not pending_ids:
        print("Stage 2: No movies found in saved_imdb_movies.json.")
        return

    print(f"Stage 2: {len(pending_ids):,} movies to fetch from TMDB")
    start = time.monotonic()

    counters = asyncio.run(_fetch_all(db, pending_ids))

    elapsed = time.monotonic() - start
    hours = elapsed / 3600

    print("\n" + "=" * 60)
    print("Stage 2 Complete")
    print("=" * 60)
    print(f"  Total pending:  {len(pending_ids):,}")
    print(f"  Fetched:        {counters['fetched']:,}")
    print(f"  Filtered out:   {counters['filtered']:,}")
    print(f"  Errors:         {counters['errors']:,}")
    print(f"  Duration:       {elapsed:,.0f}s ({hours:.1f}h)")
    print("=" * 60)


if __name__ == "__main__":
    run()
