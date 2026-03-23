"""
Parses batch result files and stores results to SQLite.

Currently handles plot_events only. Other metadata types will follow
the same pattern — each gets its own process function that validates
against the correct Pydantic schema and stores to the right column.

Usage:
    results = download_results(output_file_id)  # from openai_batch_manager
    summary = process_plot_events_results(results)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from movie_ingestion.tracker import TRACKER_DB_PATH
from movie_ingestion.metadata_generation.inputs import parse_custom_id
from movie_ingestion.metadata_generation.schemas import PlotEventsOutput


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ProcessingSummary:
    """Summary of batch result processing."""
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# How many rows to accumulate before committing. Balances crash safety
# (don't lose too much work) with performance (fewer commits).
_COMMIT_INTERVAL = 500


def process_plot_events_results(
    results: list[dict],
    tracker_db_path: Path = TRACKER_DB_PATH,
) -> ProcessingSummary:
    """Parse batch results and store plot_events in generated_metadata.

    For each result dict from the output JSONL:
    1. Parse custom_id to get tmdb_id
    2. Check response status_code == 200
    3. Extract the content string from choices[0].message.content
    4. Validate with PlotEventsOutput.model_validate_json()
    5. On success: store raw JSON in generated_metadata.plot_events
    6. On failure: record in generation_failures table

    Commits in batches of _COMMIT_INTERVAL for crash safety.

    Args:
        results: Parsed result dicts from download_results().
        tracker_db_path: Path to the tracker SQLite database.

    Returns:
        ProcessingSummary with counts and token usage.
    """
    summary = ProcessingSummary(total=len(results))

    with sqlite3.connect(str(tracker_db_path)) as db:
        pending_since_commit = 0

        for result in results:
            custom_id = result.get("custom_id", "")
            try:
                metadata_type, tmdb_id = parse_custom_id(custom_id)
            except (ValueError, IndexError):
                summary.failed += 1
                continue

            # Extract response body
            response = result.get("response", {})
            status_code = response.get("status_code", 0)
            body = response.get("body", {})

            if status_code != 200:
                # HTTP-level failure from OpenAI
                error_msg = f"HTTP {status_code}: {body.get('error', {}).get('message', 'unknown error')}"
                _record_failure(db, tmdb_id, metadata_type, error_msg)
                summary.failed += 1
                pending_since_commit += 1
            else:
                # Try to extract and validate the content
                success = _process_single_result(
                    db, tmdb_id, metadata_type, body, summary,
                )
                pending_since_commit += 1

            # Periodic commit for crash safety
            if pending_since_commit >= _COMMIT_INTERVAL:
                db.commit()
                pending_since_commit = 0

        # Final commit for any remaining rows
        if pending_since_commit > 0:
            db.commit()

    return summary


def process_error_file(
    errors: list[dict],
    metadata_type: str,
    tracker_db_path: Path = TRACKER_DB_PATH,
) -> int:
    """Process the error file from a batch, recording failures.

    Each error line contains a custom_id and error details. Inserts
    into the generation_failures table for later retry.

    Args:
        errors: Parsed error dicts from download_results().
        metadata_type: The metadata type for these errors (e.g. 'plot_events').
        tracker_db_path: Path to the tracker SQLite database.

    Returns:
        Count of errors processed.
    """
    if not errors:
        return 0

    recorded = 0
    with sqlite3.connect(str(tracker_db_path)) as db:
        for error_line in errors:
            custom_id = error_line.get("custom_id", "")
            try:
                _, tmdb_id = parse_custom_id(custom_id)
            except (ValueError, IndexError):
                continue

            # Extract error message from the response
            response = error_line.get("response", {})
            body = response.get("body", {})
            error_obj = body.get("error", {})
            error_msg = error_obj.get("message", str(error_line))

            _record_failure(db, tmdb_id, metadata_type, error_msg)
            recorded += 1

        db.commit()

    return recorded


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_single_result(
    db: sqlite3.Connection,
    tmdb_id: int,
    metadata_type: str,
    body: dict,
    summary: ProcessingSummary,
) -> bool:
    """Process a single successful (HTTP 200) result.

    Extracts content, validates against PlotEventsOutput, stores result
    or records failure. Updates summary in place.

    Returns True if the result was successfully stored.
    """
    # Accumulate token usage regardless of validation outcome
    usage = body.get("usage", {})
    summary.total_input_tokens += usage.get("prompt_tokens", 0)
    summary.total_output_tokens += usage.get("completion_tokens", 0)

    # Extract the content string from the response
    choices = body.get("choices", [])
    if not choices:
        _record_failure(db, tmdb_id, metadata_type, "No choices in response")
        summary.failed += 1
        return False

    content = choices[0].get("message", {}).get("content")
    if not content:
        _record_failure(db, tmdb_id, metadata_type, "Empty content in response")
        summary.failed += 1
        return False

    # Validate the content against the Pydantic schema
    try:
        PlotEventsOutput.model_validate_json(content)
    except ValidationError as e:
        _record_failure(
            db, tmdb_id, metadata_type,
            f"Validation failed: {e.error_count()} error(s) — {str(e)[:500]}",
        )
        summary.failed += 1
        return False

    # Store the raw JSON content string (already valid, no need to re-serialize)
    db.execute(
        "UPDATE generated_metadata SET plot_events = ? WHERE tmdb_id = ?",
        (content, tmdb_id),
    )
    summary.succeeded += 1
    return True


def _record_failure(
    db: sqlite3.Connection,
    tmdb_id: int,
    metadata_type: str,
    error_message: str,
) -> None:
    """Insert a row into generation_failures."""
    db.execute(
        """INSERT INTO generation_failures (tmdb_id, metadata_type, error_message)
           VALUES (?, ?, ?)""",
        (tmdb_id, metadata_type, error_message),
    )
