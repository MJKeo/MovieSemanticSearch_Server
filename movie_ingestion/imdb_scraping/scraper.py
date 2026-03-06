"""
Per-movie IMDB scraping orchestration.

Coordinates the GraphQL fetch, response transformation, and persistence
for a single movie. Called by the run module's batch loop.

Failure routing:
  - GraphQL returns null title → FILTERED_OUT (movie doesn't exist on IMDB)
  - Fetch failed after retries → FILTERED_OUT (infrastructure failure)
  - Transform exception → counted as error, movie skipped
  - Full success → save merged data, mark imdb_scraped
"""

import sqlite3
import time

import httpx
from fake_useragent import UserAgent

from movie_ingestion.tracker import (
    INGESTION_DATA_DIR,
    MovieStatus,
    PipelineStage,
    log_filter,
    save_json,
)
from .http_client import FetchResult, fetch_movie
from .parsers import transform_graphql_response


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_STAGE = PipelineStage.IMDB_SCRAPE
_IMDB_JSON_DIR = INGESTION_DATA_DIR / "imdb"

# SQL for updating movie status after successful scraping
_UPDATE_STATUS_SQL = """
    UPDATE movie_progress
    SET status = ?, updated_at = CURRENT_TIMESTAMP
    WHERE tmdb_id = ?
"""


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------


async def process_movie(
    client: httpx.AsyncClient,
    semaphore,
    ua: UserAgent,
    tmdb_id: int,
    imdb_id: str,
    db: sqlite3.Connection,
    counters: dict,
) -> None:
    """
    Fetch, transform, and persist IMDB data for a single movie.

    This is the main per-movie coroutine called from the batch loop in
    run.py. Updates counters dict in-place (safe in single-threaded asyncio).

    Args:
        client: Shared httpx.AsyncClient with proxy configuration.
        semaphore: Global concurrency semaphore.
        ua: UserAgent generator for random desktop UA strings.
        tmdb_id: TMDB movie ID (used for JSON filename and tracker updates).
        imdb_id: IMDB title ID (e.g., "tt0137523", used for GraphQL variable).
        db: Open SQLite connection for tracker updates.
        counters: Mutable dict tracking scraped/filtered/errors counts.
    """
    movie_start = time.monotonic()
    tag = f"[tmdb={tmdb_id}]"

    # Step 1: Fetch all movie data via a single GraphQL query
    result_type, title_data = await fetch_movie(client, semaphore, ua, imdb_id)

    # Step 2: Handle fetch failures
    if result_type == FetchResult.HTTP_404:
        elapsed = time.monotonic() - movie_start
        print(f"  {tag} Not found on IMDB — filtering out ({elapsed:.2f}s)")
        log_filter(db, tmdb_id, _STAGE, reason="imdb_404")
        counters["filtered"] += 1
        return

    if result_type == FetchResult.FAILED:
        elapsed = time.monotonic() - movie_start
        print(f"  {tag} Fetch failed — filtering out ({elapsed:.2f}s)")
        log_filter(db, tmdb_id, _STAGE, reason="fetch_failed")
        counters["filtered"] += 1
        return

    # Step 3: Transform the GraphQL response into the output model.
    # Wrap in try/except so a parser bug on one movie doesn't crash the batch.
    try:
        merged = transform_graphql_response(title_data)
    except Exception as exc:
        elapsed = time.monotonic() - movie_start
        print(f"  {tag} Transform FAILED: {type(exc).__name__}: "
              f"{str(exc)[:120]} ({elapsed:.2f}s)")
        counters["errors"] += 1
        return

    # Step 4: Save the merged JSON file
    json_path = _IMDB_JSON_DIR / f"{tmdb_id}.json"
    save_json(json_path, merged.model_dump(mode="json"))

    # Step 5: Update tracker status to imdb_scraped
    db.execute(_UPDATE_STATUS_SQL, (MovieStatus.IMDB_SCRAPED, tmdb_id))
    counters["scraped"] += 1
