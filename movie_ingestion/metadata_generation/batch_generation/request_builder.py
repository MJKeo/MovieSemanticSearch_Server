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
from schemas.enums import MetadataType
from schemas.movie_input import MovieInputData, load_movie_input_data
from ..inputs import build_custom_id
from ..batch_generation.generator_registry import (
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
    at least one NULL slot among the type's result_columns. Loads movie
    data in chunks (each MovieInputData holds full synopses/summaries/
    reviews, so loading all ~109K at once would use 500MB+). Builds
    prompts via the type's registered prompt builder and wraps each in
    the Batch API request format.

    For multi-run types (config.runs_per_movie > 1), a movie with K NULL
    slots receives K requests, each with a distinct custom_id suffix
    (`_r{N}`). The run_index is purely a uniqueness token for OpenAI's
    per-batch custom_id constraint and for debug traceability — the
    result writer picks the next NULL column at ingest time, independent
    of which run produced which response.

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

    # Find eligible movies with at least one NULL result slot.
    # Each tuple is (tmdb_id, n_runs_needed) — 1 for single-run types,
    # 1..runs_per_movie for multi-run types.
    pending = _get_pending_tmdb_ids(config, tracker_db_path)
    if not pending:
        return []

    # Truncate early to avoid loading data for movies we won't submit.
    # Multi-run movies vary in request cost, so accumulate until the
    # request budget is reached rather than slicing by movie count.
    if max_batches is not None:
        max_requests = max_batches * batch_size
        truncated: list[tuple[int, int]] = []
        running = 0
        for tmdb_id, n_needed in pending:
            if running + n_needed > max_requests:
                break
            truncated.append((tmdb_id, n_needed))
            running += n_needed
        if len(truncated) < len(pending):
            print(
                f"  [{metadata_type}] {len(pending)} eligible movies "
                f"({sum(n for _, n in pending)} requests), limiting to "
                f"{len(truncated)} movies ({running} requests)."
            )
        pending = truncated

    total_requests = sum(n for _, n in pending)
    print(
        f"  [{metadata_type}] Building {total_requests} requests for "
        f"{len(pending)} movies (runs_per_movie={config.runs_per_movie})..."
    )

    # Compute the JSON schema once for this type — identical for every request.
    json_schema = to_strict_json_schema(config.schema_class)

    # Build request dicts in chunks to keep memory bounded.
    # MovieInputData holds full plot text, reviews, etc. — several KB each.
    all_requests: list[dict] = []
    for chunk_start in range(0, len(pending), _LOAD_CHUNK_SIZE):
        chunk = pending[chunk_start : chunk_start + _LOAD_CHUNK_SIZE]
        chunk_ids = [tid for tid, _ in chunk]
        movies = load_movie_input_data(chunk_ids, tracker_db_path)

        for tmdb_id, n_needed in chunk:
            movie = movies.get(tmdb_id)
            if movie is None:
                # Movie data couldn't be loaded (missing from tmdb_data/imdb_data)
                continue

            # Emit one request per missing run. Single-run types pass
            # run_index=None so the custom_id keeps its legacy
            # `{type}_{tmdb_id}` shape; multi-run types pass 1..n_needed.
            if config.runs_per_movie == 1:
                all_requests.append(_build_single_request(movie, config, json_schema, run_index=None))
            else:
                for run_index in range(1, n_needed + 1):
                    all_requests.append(_build_single_request(movie, config, json_schema, run_index=run_index))

    print(f"  [{metadata_type}] Built {len(all_requests)} requests.")

    # Chunk into batches of batch_size
    return _chunk(all_requests, batch_size)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_pending_tmdb_ids(
    config: GeneratorConfig,
    tracker_db_path: Path,
) -> list[tuple[int, int]]:
    """Query for movies eligible for a metadata type with at least one NULL slot.

    Returns (tmdb_id, n_runs_needed) pairs. For single-run types
    n_runs_needed is always 1; for multi-run types it equals the count
    of NULL columns among config.result_columns (1..runs_per_movie).

    Excludes movies that already have an active batch_id in
    metadata_batch_ids, preventing duplicate submissions if submit is
    run before process completes. The batch_id gate is per-type, so a
    movie with a pending batch (regardless of how many slots are
    actually mid-flight) is held until the batch finishes and the gate
    clears.

    Column names are interpolated from config.metadata_type (StrEnum
    with fixed values matching DB columns exactly) and from
    config.result_columns (whitelist defined in the registry) — never
    from untrusted input.
    """
    metadata_type = config.metadata_type
    eligible_col = f"eligible_for_{metadata_type}"
    batch_col = f"{metadata_type}_batch_id"

    # Per-column "NULL = 1" terms summed into n_needed. Works for both
    # single-run (one column, value 0 or 1) and multi-run (N columns,
    # value 0..N) types with a single uniform query shape.
    null_terms = " + ".join(
        f"CASE WHEN gm.{col} IS NULL THEN 1 ELSE 0 END"
        for col in config.result_columns
    )

    with sqlite3.connect(str(tracker_db_path)) as db:
        rows = db.execute(
            f"""
            SELECT gm.tmdb_id, ({null_terms}) AS n_needed
            FROM generated_metadata gm
            LEFT JOIN metadata_batch_ids mb ON gm.tmdb_id = mb.tmdb_id
            WHERE gm.{eligible_col} = 1
              AND ({null_terms}) > 0
              AND (mb.{batch_col} IS NULL OR mb.tmdb_id IS NULL)
            """,
        ).fetchall()

    return [(row[0], row[1]) for row in rows]


def _build_single_request(
    movie: MovieInputData,
    config: GeneratorConfig,
    json_schema: dict,
    run_index: int | None = None,
) -> dict:
    """Build one Batch API request dict for a single movie.

    For multi-run types, pass run_index 1..runs_per_movie so the
    custom_id is unique within the batch. For single-run types, pass
    None to keep the legacy `{type}_{tmdb_id}` custom_id shape.
    """
    user_prompt, system_prompt = config.prompt_builder(movie)
    custom_id = build_custom_id(movie.tmdb_id, config.metadata_type, run_index=run_index)

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
