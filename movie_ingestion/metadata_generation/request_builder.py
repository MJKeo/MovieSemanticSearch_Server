"""
Builds Batch API request dicts for metadata generation.

Currently handles plot_events only. Other metadata types will follow
the same pattern — each gets its own build function that queries for
eligible movies, constructs prompts, and returns chunked request lists.

Usage:
    batches = build_plot_events_requests()
    # batches is a list of lists — each inner list is up to BATCH_SIZE
    # request dicts ready for upload_and_create_batch().
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from openai.lib._pydantic import to_strict_json_schema

from movie_ingestion.tracker import TRACKER_DB_PATH
from movie_ingestion.metadata_generation.inputs import (
    build_custom_id,
    load_movie_input_data,
    MovieInputData,
)
from movie_ingestion.metadata_generation.generators.plot_events import (
    GENERATION_TYPE,
    build_plot_events_prompts,
)
from movie_ingestion.metadata_generation.schemas import PlotEventsOutput


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 10_000

# How many movies to load into memory at once during request building.
# Each MovieInputData holds full synopses, summaries, reviews — several KB.
# 5K × ~5KB ≈ 25MB per chunk, well within limits.
_LOAD_CHUNK_SIZE = 5_000

# Model config for plot_events — matches generators/plot_events.py constants.
# Duplicated here rather than importing private vars because these are the
# values that go into the Batch API request body (not the live API call).
_MODEL = "gpt-5-mini"
_MODEL_KWARGS = {"reasoning_effort": "minimal", "verbosity": "low"}

# Compute the JSON schema once — it's identical for every plot_events request.
_PLOT_EVENTS_SCHEMA = to_strict_json_schema(PlotEventsOutput)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_plot_events_requests(
    tracker_db_path: Path = TRACKER_DB_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[dict]]:
    """Build Batch API request dicts for all eligible movies needing plot_events.

    Queries generated_metadata for movies with eligible_for_plot_events=1
    and plot_events IS NULL (no result yet). Loads movie data in chunks
    (each MovieInputData holds full synopses/summaries/reviews, so loading
    all ~109K at once would use 500MB+). Builds prompts via the existing
    two-branch logic and wraps each in the Batch API request format.

    Args:
        tracker_db_path: Path to the tracker SQLite database.
        batch_size: Max requests per batch. Smaller batches help stay
            within OpenAI's enqueued token limits.

    Returns a list of batches, where each batch is a list of up to
    batch_size request dicts. Returns an empty list if no movies need
    generation.
    """
    # Find movies that are eligible but don't have a result yet
    tmdb_ids = _get_pending_tmdb_ids(tracker_db_path)
    if not tmdb_ids:
        return []

    print(f"  Building requests for {len(tmdb_ids)} movies...")

    # Build request dicts in chunks to keep memory bounded.
    # MovieInputData holds full plot text, reviews, etc. — several KB each.
    all_requests: list[dict] = []
    for chunk_start in range(0, len(tmdb_ids), _LOAD_CHUNK_SIZE):
        chunk_ids = tmdb_ids[chunk_start : chunk_start + _LOAD_CHUNK_SIZE]
        movies = load_movie_input_data(chunk_ids, tracker_db_path)

        for tmdb_id in chunk_ids:
            movie = movies.get(tmdb_id)
            if movie is None:
                # Movie data couldn't be loaded (missing from tmdb_data/imdb_data)
                continue

            request = _build_single_request(movie)
            all_requests.append(request)

    print(f"  Built {len(all_requests)} requests.")

    # Chunk into batches of batch_size
    return _chunk(all_requests, batch_size)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pending_tmdb_ids(tracker_db_path: Path) -> list[int]:
    """Query for movies eligible for plot_events that don't have a result yet.

    Excludes movies that already have an active batch_id in metadata_batch_ids,
    preventing duplicate submissions if submit is run before process completes.
    """
    with sqlite3.connect(str(tracker_db_path)) as db:
        rows = db.execute(
            """
            SELECT gm.tmdb_id FROM generated_metadata gm
            LEFT JOIN metadata_batch_ids mb ON gm.tmdb_id = mb.tmdb_id
            WHERE gm.eligible_for_plot_events = 1
              AND gm.plot_events IS NULL
              AND (mb.plot_events_batch_id IS NULL OR mb.tmdb_id IS NULL)
            """,
        ).fetchall()

    return [row[0] for row in rows]


def _build_single_request(movie: MovieInputData) -> dict:
    """Build one Batch API request dict for a single movie's plot_events."""
    user_prompt, system_prompt = build_plot_events_prompts(movie)
    custom_id = build_custom_id(movie.tmdb_id, GENERATION_TYPE)

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": _MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "PlotEventsOutput",
                    "strict": True,
                    "schema": _PLOT_EVENTS_SCHEMA,
                },
            },
            **_MODEL_KWARGS,
        },
    }


def _chunk(items: list, size: int) -> list[list]:
    """Split a list into sublists of at most `size` items."""
    return [items[i : i + size] for i in range(0, len(items), size)]
