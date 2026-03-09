"""
Per-movie IMDB scraping orchestration.

Coordinates the GraphQL fetch, response transformation, and JSON persistence
for a single movie. Called by the run module's batch loop.

Returns a result describing the outcome — the caller (run.py) is responsible
for all database writes, which happen in bulk after the async batch completes.

Failure routing:
  - GraphQL returns null title → filtered (movie doesn't exist on IMDB)
  - Fetch failed after retries → filtered (infrastructure failure)
  - Transform exception → error, movie skipped
  - Full success → save JSON, return scraped result
"""

import time
from typing import NamedTuple

import httpx
from fake_useragent import UserAgent

from movie_ingestion.tracker import (
    INGESTION_DATA_DIR,
    PipelineStage,
    save_json,
)
from .http_client import FetchResult, fetch_movie
from .parsers import transform_graphql_response


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGE = PipelineStage.IMDB_SCRAPE
_IMDB_JSON_DIR = INGESTION_DATA_DIR / "imdb"


# ---------------------------------------------------------------------------
# Result type returned by process_movie
# ---------------------------------------------------------------------------


class MovieResult(NamedTuple):
    """Outcome of processing a single movie. Used by run.py for bulk DB writes."""
    tmdb_id: int
    status: str        # "scraped", "filtered", or "error"
    reason: str | None  # filter reason (e.g., "imdb_404"), None for scraped/error


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------


async def process_movie(
    client: httpx.AsyncClient,
    semaphore,
    ua: UserAgent,
    tmdb_id: int,
    imdb_id: str,
) -> MovieResult:
    """
    Fetch, transform, and persist IMDB data for a single movie.

    Performs only HTTP fetching, transformation, and JSON file persistence.
    Returns a MovieResult describing the outcome — all database writes
    are handled by the caller in bulk after the async batch completes.

    Args:
        client: Shared httpx.AsyncClient with proxy configuration.
        semaphore: Global concurrency semaphore.
        ua: UserAgent generator for random desktop UA strings.
        tmdb_id: TMDB movie ID (used for JSON filename and tracker updates).
        imdb_id: IMDB title ID (e.g., "tt0137523", used for GraphQL variable).
    """
    movie_start = time.monotonic()
    tag = f"[tmdb={tmdb_id}]"

    # Step 1: Fetch all movie data via a single GraphQL query
    result_type, title_data = await fetch_movie(client, semaphore, ua, imdb_id)

    # Step 2: Handle fetch failures
    if result_type == FetchResult.HTTP_404:
        elapsed = time.monotonic() - movie_start
        print(f"  {tag} Not found on IMDB — filtering out ({elapsed:.2f}s)")
        return MovieResult(tmdb_id, "filtered", "imdb_404")

    if result_type == FetchResult.FAILED:
        elapsed = time.monotonic() - movie_start
        print(f"  {tag} Fetch failed — filtering out ({elapsed:.2f}s)")
        return MovieResult(tmdb_id, "filtered", "fetch_failed")

    # Step 3: Transform the GraphQL response into the output model.
    # Wrap in try/except so a parser bug on one movie doesn't crash the batch.
    try:
        merged = transform_graphql_response(title_data)
    except Exception as exc:
        elapsed = time.monotonic() - movie_start
        print(f"  {tag} Transform FAILED: {type(exc).__name__}: "
              f"{str(exc)[:120]} ({elapsed:.2f}s)")
        return MovieResult(tmdb_id, "error", None)

    # Step 4: Save the merged JSON file (atomic write-then-rename, crash-safe)
    json_path = _IMDB_JSON_DIR / f"{tmdb_id}.json"
    save_json(json_path, merged.model_dump(mode="json"))

    return MovieResult(tmdb_id, "scraped", None)
