"""
CLI entry point for the metadata generation pipeline.

Each metadata type is handled individually (no wave grouping).
Currently supports plot_events; other types will follow the same pattern.

Commands:
    eligibility    Evaluate plot_events eligibility for all imdb_quality_passed
                   movies and store flags in generated_metadata.

    submit         Build JSONL request dicts for eligible movies, upload to
                   OpenAI Files API, and create batch(es) of up to 10K each.

    status         Check the status of all active plot_events batches.

    process        Download results from completed batches, parse, validate,
                   and store in generated_metadata. Records failures in
                   generation_failures table.

Usage:
    python -m movie_ingestion.metadata_generation.run eligibility
    python -m movie_ingestion.metadata_generation.run submit
    python -m movie_ingestion.metadata_generation.run status
    python -m movie_ingestion.metadata_generation.run process
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from movie_ingestion.tracker import TRACKER_DB_PATH, init_db
from movie_ingestion.metadata_generation.inputs import (
    load_movie_input_data,
    parse_custom_id,
)
from movie_ingestion.metadata_generation.pre_consolidation import check_plot_events
from movie_ingestion.metadata_generation.request_builder import (
    build_plot_events_requests,
    DEFAULT_BATCH_SIZE,
)
from movie_ingestion.metadata_generation.openai_batch_manager import (
    upload_and_create_batch,
    check_batch_status,
    download_results,
)
from movie_ingestion.metadata_generation.result_processor import (
    process_plot_events_results,
    process_error_file,
)
from movie_ingestion.metadata_generation.generators.plot_events import GENERATION_TYPE


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many movies to load at once during eligibility evaluation.
# Keeps memory bounded while allowing efficient SQL batch operations.
_ELIGIBILITY_CHUNK_SIZE = 5000


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_eligibility(tracker_db_path: Path = TRACKER_DB_PATH) -> None:
    """Evaluate plot_events eligibility for all imdb_quality_passed movies.

    1. Query movie_progress for all tmdb_ids at imdb_quality_passed status.
    2. Ensure each has a row in generated_metadata (INSERT OR IGNORE).
    3. For movies where eligible_for_plot_events IS NULL, load data in
       chunks and run check_plot_events() to determine eligibility.
    4. Store results as 1 (eligible) or 0 (ineligible) in generated_metadata.
    """
    db = init_db()

    # Step 1: Get all imdb_quality_passed movies
    all_tmdb_ids = _get_quality_passed_tmdb_ids(db)
    print(f"Found {len(all_tmdb_ids)} movies at imdb_quality_passed status.")

    if not all_tmdb_ids:
        db.close()
        return

    # Step 2: Ensure every movie has a generated_metadata row
    _ensure_generated_metadata_rows(db, all_tmdb_ids)

    # Step 3: Find movies that haven't been evaluated yet
    unevaluated = db.execute(
        """
        SELECT tmdb_id FROM generated_metadata
        WHERE eligible_for_plot_events IS NULL
          AND tmdb_id IN (
              SELECT tmdb_id FROM movie_progress
              WHERE status = 'imdb_quality_passed'
          )
        """,
    ).fetchall()
    unevaluated_ids = [row[0] for row in unevaluated]

    if not unevaluated_ids:
        # Print current counts even if nothing new to evaluate
        _print_eligibility_summary(db)
        db.close()
        return

    print(f"Evaluating eligibility for {len(unevaluated_ids)} movies...")

    # Step 4: Process in chunks to manage memory
    eligible_count = 0
    ineligible_count = 0

    for chunk_start in range(0, len(unevaluated_ids), _ELIGIBILITY_CHUNK_SIZE):
        chunk_ids = unevaluated_ids[chunk_start : chunk_start + _ELIGIBILITY_CHUNK_SIZE]

        # Load movie data for this chunk
        movies = load_movie_input_data(chunk_ids, tracker_db_path)

        # Evaluate each movie and collect updates
        eligible_updates: list[tuple[int, int]] = []
        for tmdb_id in chunk_ids:
            movie = movies.get(tmdb_id)
            if movie is None:
                # Couldn't load data — mark ineligible
                eligible_updates.append((0, tmdb_id))
                ineligible_count += 1
                continue

            skip_reason = check_plot_events(movie)
            if skip_reason is None:
                eligible_updates.append((1, tmdb_id))
                eligible_count += 1
            else:
                eligible_updates.append((0, tmdb_id))
                ineligible_count += 1

        # Bulk update this chunk
        db.executemany(
            "UPDATE generated_metadata SET eligible_for_plot_events = ? WHERE tmdb_id = ?",
            eligible_updates,
        )
        db.commit()

        processed = min(chunk_start + _ELIGIBILITY_CHUNK_SIZE, len(unevaluated_ids))
        print(f"  Processed {processed}/{len(unevaluated_ids)}...")

    print(f"\nNewly evaluated: {eligible_count} eligible, {ineligible_count} ineligible.")
    _print_eligibility_summary(db)
    db.close()


def cmd_submit(
    tracker_db_path: Path = TRACKER_DB_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int | None = None,
) -> None:
    """Build and submit plot_events batch(es) to OpenAI.

    1. Build request dicts for all eligible movies without results.
    2. Upload each batch (up to batch_size requests) and create OpenAI batches.
    3. Store batch_ids in metadata_batch_ids table.

    If max_batches is set, only the first N batches are submitted.
    Remaining movies stay eligible for the next submit run.
    """
    # Ensure DB is initialized
    init_db().close()

    batches = build_plot_events_requests(tracker_db_path, batch_size=batch_size)
    if not batches:
        print("Nothing to submit — no eligible movies without results.")
        return

    # Limit to max_batches if specified — remaining movies stay eligible
    # for the next submit run since they won't have a batch_id yet.
    if max_batches is not None and len(batches) > max_batches:
        print(f"  {len(batches)} batch(es) available, limiting to {max_batches}.")
        batches = batches[:max_batches]

    print(f"Submitting {len(batches)} batch(es)...")

    with sqlite3.connect(str(tracker_db_path)) as db:
        for i, batch_requests in enumerate(batches, 1):
            print(f"\n  Batch {i}/{len(batches)}: {len(batch_requests)} requests")

            # Upload and create the OpenAI batch
            batch_id = upload_and_create_batch(batch_requests)
            print(f"  Created batch: {batch_id}")

            # Record the batch_id for each movie in this batch
            _record_batch_ids(db, batch_requests, batch_id)
            db.commit()

    print(f"\nDone. Submitted {sum(len(b) for b in batches)} total requests.")


def cmd_status(tracker_db_path: Path = TRACKER_DB_PATH) -> None:
    """Check the status of all active plot_events batches."""
    batch_ids = _get_active_batch_ids(tracker_db_path)

    if not batch_ids:
        print("No active plot_events batches.")
        return

    print(f"{'Batch ID':<30} {'Status':<15} {'Completed':<12} {'Failed':<8} {'Total':<8}")
    print("-" * 73)

    for bid in batch_ids:
        status = check_batch_status(bid)
        print(
            f"{status.batch_id:<30} {status.status:<15} "
            f"{status.completed:<12} {status.failed:<8} {status.total:<8}"
        )


def cmd_process(tracker_db_path: Path = TRACKER_DB_PATH) -> None:
    """Download and process results from completed plot_events batches.

    For each active batch:
    - If not completed, print its current status and skip.
    - If completed, download results, process them, handle errors,
      and clear the batch_id from metadata_batch_ids.
    """
    batch_ids = _get_active_batch_ids(tracker_db_path)

    if not batch_ids:
        print("No active plot_events batches to process.")
        return

    for bid in batch_ids:
        status = check_batch_status(bid)
        print(f"\nBatch {bid}: {status.status}")

        if status.status == "failed":
            print(f"  FAILED — {status.completed}/{status.total} completed, {status.failed} failed")
            if status.errors:
                for err in status.errors:
                    print(f"  Error [{err.get('code', 'unknown')}]: {err.get('message', 'no message')}")
            else:
                print("  No batch-level error details available.")
            # Clear batch_ids so these movies are eligible for resubmission
            _clear_batch_id(tracker_db_path, bid)
            print(f"  Cleared batch_id — movies are available for resubmission.")
            continue

        if status.status != "completed":
            print(f"  Skipping — {status.completed}/{status.total} completed, {status.failed} failed")
            continue

        # Download and process the output file
        if status.output_file_id:
            print(f"  Downloading results ({status.completed} completed)...")
            results = download_results(status.output_file_id)
            summary = process_plot_events_results(results, tracker_db_path)
            print(
                f"  Results: {summary.succeeded} succeeded, {summary.failed} failed"
                f" | Tokens: {summary.total_input_tokens:,} in, {summary.total_output_tokens:,} out"
            )
        else:
            # Completed batch with no output file — unusual. Don't clear
            # batch_ids since we can't confirm results were processed.
            print(f"  Warning: completed batch has no output file. Skipping.")
            continue

        # Download and process the error file if present
        if status.error_file_id:
            print(f"  Downloading error file ({status.failed} failed)...")
            errors = download_results(status.error_file_id)
            error_count = process_error_file(errors, GENERATION_TYPE, tracker_db_path)
            print(f"  Recorded {error_count} error(s) in generation_failures.")

        # Clear batch_ids for processed movies so they don't show up in
        # future status/process calls. Failed movies keep plot_events IS NULL
        # and can be retried via a new submit.
        _clear_batch_id(tracker_db_path, bid)
        print(f"  Cleared batch_id for processed movies.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_quality_passed_tmdb_ids(db: sqlite3.Connection) -> list[int]:
    """Get all tmdb_ids at imdb_quality_passed status."""
    rows = db.execute(
        "SELECT tmdb_id FROM movie_progress WHERE status = 'imdb_quality_passed'",
    ).fetchall()
    return [row[0] for row in rows]


def _ensure_generated_metadata_rows(
    db: sqlite3.Connection, tmdb_ids: list[int],
) -> None:
    """Ensure every tmdb_id has a row in generated_metadata.

    Uses INSERT OR IGNORE so existing rows are untouched.
    Commits after all inserts.
    """
    db.executemany(
        "INSERT OR IGNORE INTO generated_metadata (tmdb_id) VALUES (?)",
        [(tid,) for tid in tmdb_ids],
    )
    db.commit()


def _print_eligibility_summary(db: sqlite3.Connection) -> None:
    """Print current eligibility counts from generated_metadata."""
    row = db.execute(
        """
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN eligible_for_plot_events = 1 THEN 1 ELSE 0 END) AS eligible,
            SUM(CASE WHEN eligible_for_plot_events = 0 THEN 1 ELSE 0 END) AS ineligible,
            SUM(CASE WHEN eligible_for_plot_events IS NULL THEN 1 ELSE 0 END) AS unevaluated,
            SUM(CASE WHEN eligible_for_plot_events = 1 AND plot_events IS NOT NULL THEN 1 ELSE 0 END) AS already_generated
        FROM generated_metadata
        """,
    ).fetchone()
    print(
        f"\nOverall plot_events status:"
        f"\n  Total movies:       {row[0]}"
        f"\n  Eligible:           {row[1]}"
        f"\n  Ineligible:         {row[2]}"
        f"\n  Unevaluated:        {row[3]}"
        f"\n  Already generated:  {row[4]}"
    )


def _record_batch_ids(
    db: sqlite3.Connection,
    batch_requests: list[dict],
    batch_id: str,
) -> None:
    """Record the batch_id for each movie in a submitted batch.

    Ensures a metadata_batch_ids row exists for each movie, then sets
    the plot_events_batch_id column.
    """
    # Extract tmdb_ids from custom_ids in the request dicts
    tmdb_ids = []
    for req in batch_requests:
        _, tmdb_id = parse_custom_id(req["custom_id"])
        tmdb_ids.append(tmdb_id)

    # Ensure rows exist
    db.executemany(
        "INSERT OR IGNORE INTO metadata_batch_ids (tmdb_id) VALUES (?)",
        [(tid,) for tid in tmdb_ids],
    )

    # Set the batch_id
    db.executemany(
        "UPDATE metadata_batch_ids SET plot_events_batch_id = ? WHERE tmdb_id = ?",
        [(batch_id, tid) for tid in tmdb_ids],
    )


def _get_active_batch_ids(tracker_db_path: Path) -> list[str]:
    """Get distinct non-NULL plot_events_batch_id values."""
    with sqlite3.connect(str(tracker_db_path)) as db:
        rows = db.execute(
            """
            SELECT DISTINCT plot_events_batch_id
            FROM metadata_batch_ids
            WHERE plot_events_batch_id IS NOT NULL
            """,
        ).fetchall()

    return [row[0] for row in rows]


def _clear_batch_id(tracker_db_path: Path, batch_id: str) -> None:
    """Clear plot_events_batch_id for all movies in a processed batch."""
    with sqlite3.connect(str(tracker_db_path)) as db:
        db.execute(
            "UPDATE metadata_batch_ids SET plot_events_batch_id = NULL WHERE plot_events_batch_id = ?",
            (batch_id,),
        )
        db.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metadata generation pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "eligibility",
        help="Evaluate plot_events eligibility for all imdb_quality_passed movies",
    )
    submit_parser = subparsers.add_parser(
        "submit",
        help="Build and submit plot_events batch(es) to OpenAI",
    )
    submit_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Max requests per batch (default: {DEFAULT_BATCH_SIZE}). "
             "Smaller batches help stay within OpenAI's enqueued token limits.",
    )
    submit_parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Max number of batches to submit in this run. "
             "Unsubmitted movies remain eligible for the next run.",
    )
    subparsers.add_parser(
        "status",
        help="Check status of active plot_events batches",
    )
    subparsers.add_parser(
        "process",
        help="Download and process results from completed batches",
    )

    args = parser.parse_args()

    if args.command == "eligibility":
        cmd_eligibility()
    elif args.command == "submit":
        cmd_submit(batch_size=args.batch_size, max_batches=args.max_batches)
    elif args.command == "status":
        cmd_status()
    elif args.command == "process":
        cmd_process()


if __name__ == "__main__":
    main()
