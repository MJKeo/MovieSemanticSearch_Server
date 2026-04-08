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
from typing import NamedTuple
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
    MovieStatus,
    PipelineStage,
    batch_log_filter,
    init_db,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGE = PipelineStage.TMDB_FETCH

# Number of movies per async batch.  After each batch completes we commit the
# SQLite transaction, bounding data loss on crash to at most one batch.
_COMMIT_BATCH_SIZE = 500

# Print a progress line every N movies processed.
_PROGRESS_INTERVAL = 1_000

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
        has_keywords, has_cast_and_crew,
        budget, maturity_rating, reviews,
        collection_name, revenue
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

_UPDATE_PROGRESS_SQL = """
    UPDATE movie_progress
    SET imdb_id = ?, status = ?, updated_at = CURRENT_TIMESTAMP
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
# Maturity rating extraction — US certification from release_dates
# ---------------------------------------------------------------------------


def _extract_us_maturity_rating(raw: dict) -> str | None:
    """
    Extract the US maturity rating (e.g. "PG-13", "R") from the TMDB
    release_dates append_to_response data.

    Iterates the US release date entries and returns the first non-empty
    certification string.  Returns None if the US region has no entries or
    none of its release dates carry a certification.
    """
    release_dates_data = raw.get("release_dates") or {}
    results = release_dates_data.get("results") or []

    for country_entry in results:
        if country_entry.get("iso_3166_1") != "US":
            continue

        # Found the US entry — scan its release dates for a certification.
        for rd in country_entry.get("release_dates") or []:
            certification = (rd.get("certification") or "").strip()
            if certification:
                return certification

    return None


# ---------------------------------------------------------------------------
# Review content extraction — plain text from TMDB reviews
# ---------------------------------------------------------------------------


def _extract_review_contents(raw: dict) -> str | None:
    """
    Extract review text from the TMDB reviews append_to_response data.

    Returns a JSON-encoded list of review content strings, or None if no
    reviews exist.  Only the review body text is preserved — author info,
    ratings, and other metadata are intentionally discarded.
    """
    reviews_data = raw.get("reviews") or {}
    results = reviews_data.get("results") or []

    contents = [
        r["content"]
        for r in results
        if r.get("content")
    ]

    return json.dumps(contents, ensure_ascii=False) if contents else None


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
    collection = raw.get("belongs_to_collection")
    collection_name = collection.get("name") if collection else None
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
        # TMDB backup fields — used as fallbacks when IMDB data is missing
        "budget": budget,
        "maturity_rating": _extract_us_maturity_rating(raw),
        "reviews": _extract_review_contents(raw),
        # Additional fields for downstream use
        "collection_name": collection_name,
        "revenue": revenue,
    }


# ---------------------------------------------------------------------------
# Result type returned by _process_movie
# ---------------------------------------------------------------------------


class _MovieResult(NamedTuple):
    """Outcome of processing a single TMDB movie."""
    tmdb_id: int
    status: str          # "fetched", "missing_imdb_id", "filtered", or "error"
    reason: str | None   # filter/error reason (None for "fetched")
    fields: dict | None  # extracted fields (present for "fetched" and "missing_imdb_id")


# ---------------------------------------------------------------------------
# Persistence — bulk-write extracted fields into SQLite
# ---------------------------------------------------------------------------


def _persist_movies(db, results: list[_MovieResult]) -> None:
    """
    Bulk-insert tmdb_data and update movie_progress for all movies that
    were successfully fetched (both "fetched" and "missing_imdb_id").

    Does NOT commit — the caller handles the commit.
    """
    # Collect all results that have extracted fields to persist
    to_persist = [r for r in results if r.fields is not None]
    if not to_persist:
        return

    # Bulk insert into tmdb_data
    tmdb_rows = []
    for r in to_persist:
        f = r.fields
        packed_keys = _pack_provider_keys(f["watch_provider_keys"])
        tmdb_rows.append((
            f["tmdb_id"], f["imdb_id"], f["title"], f["release_date"],
            f["duration"], f["poster_url"], packed_keys,
            f["vote_count"], f["popularity"], f["vote_average"],
            f["overview_length"], f["genre_count"], f["has_revenue"],
            f["has_budget"], f["has_production_companies"],
            f["has_production_countries"], f["has_keywords"],
            f["has_cast_and_crew"], f["budget"], f["maturity_rating"],
            f["reviews"], f["collection_name"], f["revenue"],
        ))
    db.executemany(_INSERT_TMDB_DATA_SQL, tmdb_rows)

    # Bulk update movie_progress status to tmdb_fetched (only for movies
    # that have an IMDB ID — missing_imdb_id movies get filtered below)
    fetched_rows = [
        (r.fields["imdb_id"], MovieStatus.TMDB_FETCHED, r.tmdb_id)
        for r in to_persist if r.status == "fetched"
    ]
    if fetched_rows:
        db.executemany(_UPDATE_PROGRESS_SQL, fetched_rows)


# ---------------------------------------------------------------------------
# Single-movie async processing
# ---------------------------------------------------------------------------


async def _process_movie(
    client: httpx.AsyncClient,
    rate_limiter: AdaptiveRateLimiter,
    tmdb_id: int,
) -> _MovieResult:
    """
    Fetch and extract data for a single movie from TMDB.

    Performs only HTTP fetching and field extraction. Returns a result
    describing the outcome — all database writes are handled by the
    caller in bulk after the async batch completes.
    """
    # --- Step 1: Fetch from TMDB ---
    try:
        raw = await fetch_movie_details(client, rate_limiter, tmdb_id)
    except TMDBFetchError:
        return _MovieResult(tmdb_id, "error", "tmdb_fetch_error", None)
    except ValueError:
        return _MovieResult(tmdb_id, "error", "tmdb_parse_error", None)

    # --- Step 2: Handle 404 (movie deleted / not found on TMDB) ---
    if raw is None:
        return _MovieResult(tmdb_id, "filtered", "tmdb_404", None)

    # --- Step 3: Extract fields ---
    try:
        fields = _extract_fields(raw)
    except Exception:
        return _MovieResult(tmdb_id, "error", "tmdb_extract_error", None)

    # --- Step 4: Check for IMDB ID ---
    # Movies without an IMDB ID can't proceed to Stage 4 (IMDB scraping).
    # We still persist their tmdb_data (fields are included in the result)
    # so the data is available for diagnostics.
    if not fields["imdb_id"]:
        return _MovieResult(tmdb_id, "missing_imdb_id", "missing_imdb_id", fields)

    return _MovieResult(tmdb_id, "fetched", None, fields)


# ---------------------------------------------------------------------------
# Batch orchestration — chunked async loop with periodic commits
# ---------------------------------------------------------------------------


async def _fetch_all(db, pending_ids: list[int]) -> dict:
    """
    Process all pending movies in batches of ``_COMMIT_BATCH_SIZE``.

    Creates a shared httpx client and adaptive rate limiter, dispatches each
    batch concurrently via ``asyncio.gather``. After each batch's HTTP work
    completes, writes all results to SQLite in bulk and commits.

    Returns:
        Counters dict with keys: fetched, filtered, errors.
    """
    counters = {"fetched": 0, "filtered": 0, "errors": 0}
    total = len(pending_ids)

    rate_limiter = AdaptiveRateLimiter(
        initial_rate=45.0, max_rate=100.0, burst=5, clean_window=120.0, increase_interval=8.0
    )

    headers = {"Authorization": f"Bearer {access_token()}"}
    limits = httpx.Limits(max_connections=_MAX_CONNECTIONS)
    timeout = httpx.Timeout(_REQUEST_TIMEOUT)

    async with httpx.AsyncClient(
        headers=headers, limits=limits, timeout=timeout
    ) as client:
        for i in range(0, total, _COMMIT_BATCH_SIZE):
            chunk = pending_ids[i : i + _COMMIT_BATCH_SIZE]

            # --- Phase 1: Async HTTP fetching (no DB writes) ---
            tasks = [
                _process_movie(client, rate_limiter, tmdb_id)
                for tmdb_id in chunk
            ]
            # return_exceptions=True prevents one unexpected exception from
            # aborting the entire chunk.  We inspect results afterward to
            # log and count any exceptions that _process_movie didn't handle.
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)

            # --- Phase 2: Collect outcomes ---
            movie_results: list[_MovieResult] = []
            filtered_entries: list[tuple[int, str, str, str | None]] = []

            for tmdb_id, result in zip(chunk, raw_results):
                if isinstance(result, Exception):
                    _log_unexpected_error(tmdb_id, result)
                    counters["errors"] += 1
                    continue

                movie_results.append(result)

                if result.status == "fetched":
                    counters["fetched"] += 1
                elif result.status == "missing_imdb_id":
                    # Persisted to tmdb_data but filtered out (no IMDB ID)
                    filtered_entries.append(
                        (result.tmdb_id, _STAGE, "missing_imdb_id", None)
                    )
                    counters["filtered"] += 1
                elif result.status == "filtered":
                    filtered_entries.append(
                        (result.tmdb_id, _STAGE, result.reason, None)
                    )
                    counters["filtered"] += 1
                elif result.status == "error":
                    filtered_entries.append(
                        (result.tmdb_id, _STAGE, result.reason, None)
                    )
                    counters["errors"] += 1

            # --- Phase 3: Bulk DB writes ---
            _persist_movies(db, movie_results)
            batch_log_filter(db, filtered_entries)
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

    try:
        rows = db.execute(
            "SELECT tmdb_id FROM movie_progress WHERE status = ?",
            (MovieStatus.IMDB_SCRAPED,)
        ).fetchall()
        pending_ids = [row[0] for row in rows]

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
    finally:
        # Ensure the connection is always closed, even on Ctrl+C or crash,
        # so we don't leave a dangling lock on the database file.
        db.close()


if __name__ == "__main__":
    run()
