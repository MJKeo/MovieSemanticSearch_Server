"""
Parses batch result files and stores results to SQLite.

Generic over MetadataType — determines the type from each result's
custom_id, validates against the correct Pydantic schema, and stores
to the corresponding column in generated_metadata.

Usage:
    results = download_results(output_file_id)  # from openai_batch_manager
    summary = process_results(results)
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ValidationError

from movie_ingestion.tracker import TRACKER_DB_PATH
from movie_ingestion.metadata_generation.inputs import MetadataType, parse_custom_id
from movie_ingestion.metadata_generation.schemas import (
    PlotEventsOutput,
    ReceptionOutput,
    PlotAnalysisWithJustificationsOutput,
    NarrativeTechniquesWithJustificationsOutput,
    ProductionKeywordsOutput,
    SourceOfInspirationOutput,
    WatchContextWithIdentityNoteOutput,
    ViewerExperienceWithJustificationsOutput,
)


# ---------------------------------------------------------------------------
# Schema lookup
# ---------------------------------------------------------------------------
# Maps MetadataType to its Pydantic output schema for validation.
# Kept separate from the full generator registry to avoid pulling in
# generator modules (prompt builders, LLM callers) that aren't needed here.

SCHEMA_BY_TYPE: dict[MetadataType, type[BaseModel]] = {
    MetadataType.PLOT_EVENTS: PlotEventsOutput,
    MetadataType.RECEPTION: ReceptionOutput,
    MetadataType.PLOT_ANALYSIS: PlotAnalysisWithJustificationsOutput,
    MetadataType.NARRATIVE_TECHNIQUES: NarrativeTechniquesWithJustificationsOutput,
    MetadataType.PRODUCTION_KEYWORDS: ProductionKeywordsOutput,
    MetadataType.SOURCE_OF_INSPIRATION: SourceOfInspirationOutput,
    MetadataType.VIEWER_EXPERIENCE: ViewerExperienceWithJustificationsOutput,
    MetadataType.WATCH_CONTEXT: WatchContextWithIdentityNoteOutput,
}


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


def process_results(
    results: list[dict],
    tracker_db_path: Path = TRACKER_DB_PATH,
    batch_id: str | None = None,
) -> ProcessingSummary:
    """Parse batch results and store metadata in generated_metadata.

    For each result dict from the output JSONL:
    1. Parse custom_id to get (metadata_type, tmdb_id)
    2. Check response status_code == 200
    3. Extract the content string from choices[0].message.content
    4. Validate with the type's Pydantic schema
    5. On success: store raw JSON in the type's column in generated_metadata
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
                _record_failure(db, tmdb_id, metadata_type, error_msg, batch_id)
                summary.failed += 1
                pending_since_commit += 1
            else:
                # Try to extract and validate the content
                _process_single_result(
                    db, tmdb_id, metadata_type, body, summary, batch_id,
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
    batch_id: str | None = None,
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

            # Extract error message from the response. Expired batch
            # entries have "response": null (key present, value None),
            # so we fall back to {} when the value is falsy.
            response = error_line.get("response") or {}
            body = response.get("body") or {}
            error_obj = body.get("error") or {}
            error_msg = error_obj.get("message", "Expired or missing response")

            _record_failure(db, tmdb_id, metadata_type, error_msg, batch_id)
            recorded += 1

        db.commit()

    return recorded


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _process_single_result(
    db: sqlite3.Connection,
    tmdb_id: int,
    metadata_type: MetadataType,
    body: dict,
    summary: ProcessingSummary,
    batch_id: str | None = None,
) -> bool:
    """Process a single successful (HTTP 200) result.

    Looks up the schema for the metadata_type, validates the content,
    and stores the result to the correct column. Updates summary in place.

    Returns True if the result was successfully stored.
    """
    # Accumulate token usage regardless of validation outcome
    usage = body.get("usage", {})
    summary.total_input_tokens += usage.get("prompt_tokens", 0)
    summary.total_output_tokens += usage.get("completion_tokens", 0)

    # Extract the content string from the response
    choices = body.get("choices", [])
    if not choices:
        _record_failure(db, tmdb_id, metadata_type, "No choices in response", batch_id)
        summary.failed += 1
        return False

    content = choices[0].get("message", {}).get("content")
    if not content:
        _record_failure(db, tmdb_id, metadata_type, "Empty content in response", batch_id)
        summary.failed += 1
        return False

    # Look up the correct schema for this metadata type
    schema_class = SCHEMA_BY_TYPE.get(metadata_type)
    if schema_class is None:
        _record_failure(
            db, tmdb_id, metadata_type,
            f"No schema registered for metadata type '{metadata_type}'",
            batch_id,
        )
        summary.failed += 1
        return False

    # Validate the content against the Pydantic schema
    try:
        schema_class.model_validate_json(content)
    except ValidationError as e:
        _record_failure(
            db, tmdb_id, metadata_type,
            f"Validation failed: {e.error_count()} error(s) — {str(e)[:500]}",
            batch_id,
        )
        summary.failed += 1
        return False

    # Store the raw JSON content string to the type's column.
    # Column name from MetadataType StrEnum — fixed values, not user input.
    result_col = str(metadata_type)
    db.execute(
        f"UPDATE generated_metadata SET {result_col} = ? WHERE tmdb_id = ?",
        (content, tmdb_id),
    )
    summary.succeeded += 1
    return True


def _record_failure(
    db: sqlite3.Connection,
    tmdb_id: int,
    metadata_type: str,
    error_message: str,
    batch_id: str | None = None,
) -> None:
    """Insert a row into generation_failures."""
    db.execute(
        """INSERT INTO generation_failures (tmdb_id, metadata_type, error_message, batch_id)
           VALUES (?, ?, ?, ?)""",
        (tmdb_id, metadata_type, error_message, batch_id),
    )
