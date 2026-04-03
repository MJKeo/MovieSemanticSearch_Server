"""
Builds Batch API request dicts for metadata generation.

Generic over MetadataType — the caller specifies which type to build
requests for, and the registry provides the schema, prompt builder,
and model config.

Usage:
    batches = build_requests(MetadataType.RECEPTION)
    # batches is a list of lists — each inner list is up to BATCH_SIZE
    # request dicts ready for upload_and_create_batch().
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from openai.lib._pydantic import to_strict_json_schema

from movie_ingestion.tracker import TRACKER_DB_PATH
from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    build_custom_id,
    load_movie_input_data,
    MovieInputData,
)
from movie_ingestion.metadata_generation.batch_generation.generator_registry import (
    GeneratorConfig,
    get_config,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_BATCH_SIZE = 10_000

# How many movies to load into memory at once during request building.
# Each MovieInputData holds full synopses, summaries, reviews — several KB.
# 5K × ~5KB ≈ 25MB per chunk, well within limits.
_LOAD_CHUNK_SIZE = 5_000


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_requests(
    metadata_type: MetadataType,
    tracker_db_path: Path = TRACKER_DB_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int | None = None,
) -> list[list[dict]]:
    """Build Batch API request dicts for eligible movies needing generation.

    Queries generated_metadata for movies with eligible_for_{type}=1 and
    {type} IS NULL (no result yet). Loads movie data in chunks (each
    MovieInputData holds full synopses/summaries/reviews, so loading all
    ~109K at once would use 500MB+). Builds prompts via the type's
    registered prompt builder and wraps each in the Batch API request format.

    Args:
        metadata_type: Which metadata type to build requests for.
        tracker_db_path: Path to the tracker SQLite database.
        batch_size: Max requests per batch. Smaller batches help stay
            within OpenAI's enqueued token limits.
        max_batches: If set, only build requests for enough movies to
            fill this many batches. Avoids loading data and building
            prompts for movies that would be discarded by the caller.

    Returns a list of batches, where each batch is a list of up to
    batch_size request dicts. Returns an empty list if no movies need
    generation.
    """
    config = get_config(metadata_type)

    # Find movies that are eligible but don't have a result yet
    tmdb_ids = _get_pending_tmdb_ids(metadata_type, tracker_db_path)
    if not tmdb_ids:
        return []

    # Truncate early to avoid loading data for movies we won't submit.
    if max_batches is not None:
        max_movies = max_batches * batch_size
        if len(tmdb_ids) > max_movies:
            print(f"  [{metadata_type}] {len(tmdb_ids)} eligible movies, limiting to {max_movies}.")
            tmdb_ids = tmdb_ids[:max_movies]

    print(f"  [{metadata_type}] Building requests for {len(tmdb_ids)} movies...")

    # Compute the JSON schema once for this type — identical for every request.
    json_schema = to_strict_json_schema(config.schema_class)

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

            request = _build_single_request(movie, config, json_schema)
            all_requests.append(request)

    print(f"  [{metadata_type}] Built {len(all_requests)} requests.")

    # Chunk into batches of batch_size
    return _chunk(all_requests, batch_size)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pending_tmdb_ids(
    metadata_type: MetadataType,
    tracker_db_path: Path,
) -> list[int]:
    """Query for movies eligible for a metadata type that don't have a result yet.

    Excludes movies that already have an active batch_id in metadata_batch_ids,
    preventing duplicate submissions if submit is run before process completes.

    Column names are interpolated from MetadataType (a StrEnum with fixed
    values matching DB columns exactly — never from untrusted input).
    """
    # Column names derived from the MetadataType StrEnum value.
    eligible_col = f"eligible_for_{metadata_type}"
    result_col = str(metadata_type)
    batch_col = f"{metadata_type}_batch_id"

    with sqlite3.connect(str(tracker_db_path)) as db:
        rows = db.execute(
            f"""
            SELECT gm.tmdb_id FROM generated_metadata gm
            LEFT JOIN metadata_batch_ids mb ON gm.tmdb_id = mb.tmdb_id
            WHERE gm.{eligible_col} = 1
              AND gm.{result_col} IS NULL
              AND (mb.{batch_col} IS NULL OR mb.tmdb_id IS NULL)
            """,
        ).fetchall()

    return [row[0] for row in rows]


def _build_single_request(
    movie: MovieInputData,
    config: GeneratorConfig,
    json_schema: dict,
) -> dict:
    """Build one Batch API request dict for a single movie."""
    user_prompt, system_prompt = config.prompt_builder(movie)
    custom_id = build_custom_id(movie.tmdb_id, config.metadata_type)

    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": config.schema_class.__name__,
                    "strict": True,
                    "schema": json_schema,
                },
            },
            **config.model_kwargs,
        },
    }


def _chunk(items: list, size: int) -> list[list]:
    """Split a list into sublists of at most `size` items."""
    return [items[i : i + size] for i in range(0, len(items), size)]
