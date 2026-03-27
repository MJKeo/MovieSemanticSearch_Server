"""
CLI entry point for the metadata generation pipeline.

Each metadata type is handled individually. The `eligibility`, `submit`, and
`autopilot` commands require a `--metadata` argument to specify which type.
The `status` and `process` commands operate across all metadata types.

Commands:
    eligibility    Evaluate eligibility for a specific metadata type for all
                   imdb_quality_passed movies and store flags in generated_metadata.

    submit         Build JSONL request dicts for eligible movies, upload to
                   OpenAI Files API, and create batch(es) of up to 10K each.

    status         Check the status of all active batches (all metadata types).

    process        Download results from completed batches, parse, validate,
                   and store in generated_metadata. Records failures in
                   generation_failures table. Handles all metadata types.

    autopilot      Interleave live generations (for a specific metadata type)
                   with batch polling (for all types) until done.

Usage:
    python -m movie_ingestion.metadata_generation.run eligibility --metadata reception
    python -m movie_ingestion.metadata_generation.run submit --metadata reception
    python -m movie_ingestion.metadata_generation.run status
    python -m movie_ingestion.metadata_generation.run process
    python -m movie_ingestion.metadata_generation.run autopilot --metadata reception
"""

from __future__ import annotations

import argparse
import asyncio
import sqlite3
import time
from datetime import datetime
from pathlib import Path

from movie_ingestion.tracker import TRACKER_DB_PATH, init_db
from movie_ingestion.metadata_generation.inputs import (
    MetadataType,
    load_movie_input_data,
    parse_custom_id,
)
from movie_ingestion.metadata_generation.generator_registry import get_config
from movie_ingestion.metadata_generation.request_builder import (
    build_requests,
    DEFAULT_BATCH_SIZE,
)
from movie_ingestion.metadata_generation.openai_batch_manager import (
    BatchStatus,
    upload_and_create_batch,
    check_batch_status,
    download_results,
)
from movie_ingestion.metadata_generation.result_processor import (
    process_results,
    process_error_file,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many movies to load at once during eligibility evaluation.
# Keeps memory bounded while allowing efficient SQL batch operations.
_ELIGIBILITY_CHUNK_SIZE = 5000


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_eligibility(
    metadata_type: MetadataType,
    tracker_db_path: Path = TRACKER_DB_PATH,
) -> None:
    """Evaluate eligibility for a metadata type for all imdb_quality_passed movies.

    1. Query movie_progress for all tmdb_ids at imdb_quality_passed status.
    2. Ensure each has a row in generated_metadata (INSERT OR IGNORE).
    3. For movies where eligible_for_{type} IS NULL, load data in
       chunks and run the type's eligibility checker.
    4. Store results as 1 (eligible) or 0 (ineligible) in generated_metadata.
    """
    config = get_config(metadata_type)

    # Column name from MetadataType StrEnum — fixed values, not user input.
    eligible_col = f"eligible_for_{metadata_type}"

    db = init_db()

    # Step 1: Get all imdb_quality_passed movies
    all_tmdb_ids = _get_quality_passed_tmdb_ids(db)
    print(f"[{metadata_type}] Found {len(all_tmdb_ids)} movies at imdb_quality_passed status.")

    if not all_tmdb_ids:
        db.close()
        return

    # Step 2: Ensure every movie has a generated_metadata row
    _ensure_generated_metadata_rows(db, all_tmdb_ids)

    # Step 3: Find movies that haven't been evaluated yet for this type
    unevaluated = db.execute(
        f"""
        SELECT tmdb_id FROM generated_metadata
        WHERE {eligible_col} IS NULL
          AND tmdb_id IN (
              SELECT tmdb_id FROM movie_progress
              WHERE status = 'imdb_quality_passed'
          )
        """,
    ).fetchall()
    unevaluated_ids = [row[0] for row in unevaluated]

    if not unevaluated_ids:
        # Print current counts even if nothing new to evaluate
        _print_eligibility_summary(db, metadata_type)
        db.close()
        return

    print(f"[{metadata_type}] Evaluating eligibility for {len(unevaluated_ids)} movies...")

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

            skip_reason = config.eligibility_checker(movie)
            if skip_reason is None:
                eligible_updates.append((1, tmdb_id))
                eligible_count += 1
            else:
                eligible_updates.append((0, tmdb_id))
                ineligible_count += 1

        # Bulk update this chunk
        db.executemany(
            f"UPDATE generated_metadata SET {eligible_col} = ? WHERE tmdb_id = ?",
            eligible_updates,
        )
        db.commit()

        processed = min(chunk_start + _ELIGIBILITY_CHUNK_SIZE, len(unevaluated_ids))
        print(f"  [{metadata_type}] Processed {processed}/{len(unevaluated_ids)}...")

    print(f"\n[{metadata_type}] Newly evaluated: {eligible_count} eligible, {ineligible_count} ineligible.")
    _print_eligibility_summary(db, metadata_type)
    db.close()


def cmd_submit(
    metadata_type: MetadataType,
    tracker_db_path: Path = TRACKER_DB_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int | None = None,
) -> None:
    """Build and submit batch(es) for a metadata type to OpenAI.

    1. Build request dicts for all eligible movies without results.
    2. Upload each batch (up to batch_size requests) and create OpenAI batches.
    3. Store batch_ids in metadata_batch_ids table.

    If max_batches is set, only the first N batches are submitted.
    Remaining movies stay eligible for the next submit run.
    """
    # Ensure DB is initialized
    init_db().close()

    batches = build_requests(
        metadata_type, tracker_db_path,
        batch_size=batch_size, max_batches=max_batches,
    )
    if not batches:
        print(f"[{metadata_type}] Nothing to submit — no eligible movies without results.")
        return

    print(f"[{metadata_type}] Submitting {len(batches)} batch(es)...")

    with sqlite3.connect(str(tracker_db_path)) as db:
        for i, batch_requests in enumerate(batches, 1):
            print(f"\n  [{metadata_type}] Batch {i}/{len(batches)}: {len(batch_requests)} requests")

            # Upload and create the OpenAI batch
            batch_id = upload_and_create_batch(batch_requests)
            print(f"  [{metadata_type}] Created batch: {batch_id}")

            # Record the batch_id for each movie in this batch
            _record_batch_ids(db, batch_requests, batch_id, metadata_type)
            db.commit()

    print(f"\n[{metadata_type}] Done. Submitted {sum(len(b) for b in batches)} total requests.")


def cmd_status(tracker_db_path: Path = TRACKER_DB_PATH) -> None:
    """Check the status of all active batches across all metadata types."""
    active = _get_active_batch_ids(tracker_db_path)

    if not active:
        print("No active batches.")
        return

    print(f"{'Type':<22} {'Batch ID':<30} {'Status':<15} {'Completed':<12} {'Failed':<8} {'Total':<8}")
    print("-" * 95)

    for bid, metadata_type in active:
        status = check_batch_status(bid)
        print(
            f"{str(metadata_type):<22} {status.batch_id:<30} {status.status:<15} "
            f"{status.completed:<12} {status.failed:<8} {status.total:<8}"
        )


def cmd_process(tracker_db_path: Path = TRACKER_DB_PATH) -> None:
    """Download and process results from finished batches (all metadata types).

    Checks every active batch and routes on its OpenAI status:
      completed  — download output (+ error file if any), store results, clear IDs
      expired    — same as completed but partial; warns and clears IDs for resubmission
      failed     — log batch-level errors, clear IDs for resubmission
      cancelled  — clear IDs for resubmission
      other      — still running (validating/in_progress/finalizing), skip
    """
    active = _get_active_batch_ids(tracker_db_path)

    if not active:
        print("No active batches to process.")
        return

    for bid, metadata_type in active:
        status = check_batch_status(bid)
        label = f"[{metadata_type}]"
        print(f"\n{label} Batch {bid}: {status.status}")

        match status.status:

            # ── Fully completed ────────────────────────────────────
            case "completed":
                _download_and_process_output(status, metadata_type, label, tracker_db_path)
                _download_and_process_errors(status, metadata_type, label, tracker_db_path)
                _clear_batch_id(tracker_db_path, bid, metadata_type)
                print(f"  {label} Cleared batch_id for processed movies.")

            # ── Expired (partial results available) ────────────────
            case "expired":
                print(
                    f"  {label} EXPIRED — {status.completed}/{status.total} completed "
                    f"before the 24h window closed"
                )
                _download_and_process_output(status, metadata_type, label, tracker_db_path)
                _download_and_process_errors(status, metadata_type, label, tracker_db_path)
                # Clear IDs so unfinished movies become eligible for resubmission
                _clear_batch_id(tracker_db_path, bid, metadata_type)
                print(f"  {label} Cleared batch_id — unfinished movies available for resubmission.")

            # ── Failed (batch-level error, no output file) ────────
            case "failed":
                print(
                    f"  {label} FAILED — {status.completed}/{status.total} completed, "
                    f"{status.failed} failed"
                )
                _log_batch_errors(status, label)
                _clear_batch_id(tracker_db_path, bid, metadata_type)
                print(f"  {label} Cleared batch_id — movies available for resubmission.")

            # ── Cancelled ──────────────────────────────────────────
            case "cancelled":
                _clear_batch_id(tracker_db_path, bid, metadata_type)
                print(f"  {label} Cleared batch_id — movies available for resubmission.")

            # ── Still running (validating, in_progress, finalizing)
            case _:
                print(
                    f"  {label} Skipping — {status.completed}/{status.total} completed, "
                    f"{status.failed} failed"
                )


def cmd_autopilot(
    metadata_type: MetadataType,
    tracker_db_path: Path = TRACKER_DB_PATH,
    batch_size: int = 1000,
    max_concurrent: int = 3,
    live_batch_size: int = 25,
    live_concurrency: int = 5,
) -> None:
    """Interleave live generations with batch polling.

    The --metadata argument determines which generator to use for live
    API calls and which type to submit new batches for. Status checking
    and result processing handle ALL metadata types automatically.

    Each iteration:
    1. Run a batch of live_batch_size direct API calls (live_concurrency parallel).
    2. Check batch statuses — process any completed/failed batches (all types).
    3. Submit new batches for the specified type to fill freed slots.
    4. Repeat until no batches remain and no eligible movies exist.

    The live generation serves as a natural poll interval (~30-60s for 25
    requests at 5 concurrent), so no explicit sleep is needed.
    """
    # Ensure DB is initialized
    init_db().close()

    print(
        f"[{metadata_type}] Autopilot started: batch_size={batch_size}, "
        f"max_concurrent={max_concurrent}, live_batch_size={live_batch_size}, "
        f"live_concurrency={live_concurrency}"
    )

    while True:
        now = datetime.now().strftime("%H:%M:%S")

        # Step 1: Run live generation batch for the specified type
        live_ids = _get_live_eligible_tmdb_ids(metadata_type, tracker_db_path, limit=live_batch_size)
        if live_ids:
            print(f"[{now}] [{metadata_type}] Running live generation for {len(live_ids)} movies...")
            succeeded, failed = asyncio.run(
                _run_live_generation_batch(live_ids, live_concurrency, tracker_db_path, metadata_type)
            )
            print(f"  [{metadata_type}] Live generation: {succeeded} succeeded, {failed} failed")

        # Step 2: Check batch statuses across ALL metadata types
        all_active = _get_active_batch_ids(tracker_db_path)
        in_progress_batches = 0
        completed_batches = 0
        failed_batches = 0
        total_requests = 0
        completed_requests = 0

        for bid, _mt in all_active:
            status = check_batch_status(bid)
            total_requests += status.total
            completed_requests += status.completed
            if status.status == "completed":
                completed_batches += 1
            elif status.status == "failed":
                failed_batches += 1
            else:
                in_progress_batches += 1

        now = datetime.now().strftime("%H:%M:%S")
        print(
            f"[{now}] Batches (all types) — in progress: {in_progress_batches}, "
            f"completed: {completed_batches}, failed: {failed_batches} | "
            f"Requests — {completed_requests:,}/{total_requests:,} completed"
        )

        # Step 3: Process completed/failed batches (all types)
        if completed_batches > 0 or failed_batches > 0:
            print(f"  Processing {completed_batches + failed_batches} completed/failed batch(es)...")
            cmd_process(tracker_db_path)

        # Step 4: Submit new batches for the specified type to fill freed slots.
        # in_progress_batches was computed in Step 2 from the all_active snapshot.
        # A batch completing between Steps 2 and 4 just means we submit one fewer
        # this iteration — caught on the next loop, no correctness issue.
        slots = max_concurrent - in_progress_batches
        if slots > 0:
            print(f"  [{metadata_type}] Submitting up to {slots} new batch(es)...")
            cmd_submit(
                metadata_type=metadata_type,
                tracker_db_path=tracker_db_path,
                batch_size=batch_size,
                max_batches=slots,
            )

        # Step 5: Termination check — no active batches for this type and
        # no eligible movies remain. After Steps 3+4 we need a fresh look
        # at what's actually in the DB (process cleared IDs, submit added new ones).
        remaining_type_batches = sum(
            1 for _, mt in _get_active_batch_ids(tracker_db_path)
            if mt == metadata_type
        )
        remaining_eligible = _get_live_eligible_tmdb_ids(metadata_type, tracker_db_path, limit=1)
        if not remaining_type_batches and not remaining_eligible:
            now = datetime.now().strftime("%H:%M:%S")
            print(f"[{now}] [{metadata_type}] All batches processed and no eligible movies remain. Done.")
            return

        # If no live-eligible movies were found this iteration but batches
        # are still running, sleep briefly before re-checking.
        if not live_ids:
            time.sleep(60)


# ---------------------------------------------------------------------------
# Internal helpers — batch result processing
# ---------------------------------------------------------------------------

def _download_and_process_output(
    status: BatchStatus,
    metadata_type: MetadataType,
    label: str,
    tracker_db_path: Path,
) -> None:
    """Download the output file from a batch and store results.

    No-ops if the batch has no output_file_id (e.g. a failed batch
    that never produced results).
    """
    if not status.output_file_id:
        return

    print(f"  {label} Downloading output ({status.completed} completed)...")
    results = download_results(status.output_file_id)
    summary = process_results(results, tracker_db_path, batch_id=status.batch_id)
    print(
        f"  {label} Results: {summary.succeeded} succeeded, {summary.failed} failed"
        f" | Tokens: {summary.total_input_tokens:,} in, {summary.total_output_tokens:,} out"
    )


def _download_and_process_errors(
    status: BatchStatus,
    metadata_type: MetadataType,
    label: str,
    tracker_db_path: Path,
) -> None:
    """Download the error file from a batch and record failures.

    No-ops if the batch has no error_file_id. The error file contains
    per-request failures (distinct from batch-level errors on the
    BatchStatus.errors field).
    """
    if not status.error_file_id:
        return

    print(f"  {label} Downloading error file ({status.failed} per-request failures)...")
    errors = download_results(status.error_file_id)
    error_count = process_error_file(errors, metadata_type, tracker_db_path, batch_id=status.batch_id)
    print(f"  {label} Recorded {error_count} error(s) in generation_failures.")


def _log_batch_errors(status: BatchStatus, label: str) -> None:
    """Print batch-level errors (why the entire batch failed).

    These are structural errors (bad JSONL, schema issues) stored on
    the batch object itself, not per-request failures.
    """
    if status.errors:
        for err in status.errors:
            print(f"  {label} Error [{err.get('code', 'unknown')}]: {err.get('message', 'no message')}")
    else:
        print(f"  {label} No batch-level error details available.")


# ---------------------------------------------------------------------------
# Internal helpers — live generation and DB queries
# ---------------------------------------------------------------------------

def _get_live_eligible_tmdb_ids(
    metadata_type: MetadataType,
    tracker_db_path: Path,
    limit: int,
) -> list[int]:
    """Get tmdb_ids eligible for live generation of a specific metadata type.

    Eligible means: imdb_quality_passed status, eligible_for_{type}=1,
    no existing result, and not currently queued in a batch.
    """
    # Column names from MetadataType StrEnum — fixed values, not user input.
    eligible_col = f"eligible_for_{metadata_type}"
    result_col = str(metadata_type)
    batch_col = f"{metadata_type}_batch_id"

    with sqlite3.connect(str(tracker_db_path)) as db:
        rows = db.execute(
            f"""
            SELECT gm.tmdb_id FROM generated_metadata gm
            JOIN movie_progress mp ON gm.tmdb_id = mp.tmdb_id
            LEFT JOIN metadata_batch_ids mb ON gm.tmdb_id = mb.tmdb_id
            WHERE mp.status = 'imdb_quality_passed'
              AND gm.{eligible_col} = 1
              AND gm.{result_col} IS NULL
              AND (mb.{batch_col} IS NULL OR mb.tmdb_id IS NULL)
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [row[0] for row in rows]


async def _run_live_generation_batch(
    tmdb_ids: list[int],
    concurrency: int,
    tracker_db_path: Path,
    metadata_type: MetadataType,
) -> tuple[int, int]:
    """Run live generation for a batch of movies using the specified type's generator.

    Uses a semaphore to limit parallel calls. Stores results to
    generated_metadata on success, records failures to generation_failures.

    Returns (succeeded, failed) counts.
    """
    config = get_config(metadata_type)
    result_col = str(metadata_type)

    movies = load_movie_input_data(tmdb_ids, tracker_db_path)
    semaphore = asyncio.Semaphore(concurrency)

    # Each task returns (tmdb_id, json_content, error_msg).
    # On success: json_content is set, error_msg is None.
    # On failure: json_content is None, error_msg is set.
    async def _generate_one(tmdb_id: int) -> tuple[int, str | None, str | None]:
        movie = movies.get(tmdb_id)
        if movie is None:
            return tmdb_id, None, "Movie data could not be loaded"

        async with semaphore:
            try:
                result, _usage = await config.live_generator(movie)
                content = result.model_dump_json()
                return tmdb_id, content, None
            except (MetadataGenerationError, MetadataGenerationEmptyResponseError) as e:
                return tmdb_id, None, str(e)

    tasks = [_generate_one(tid) for tid in tmdb_ids]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Write all results to DB in one batch
    succeeded = 0
    failed = 0

    with sqlite3.connect(str(tracker_db_path)) as db:
        for i, res in enumerate(results):
            tmdb_id = tmdb_ids[i]

            # Handle unexpected exceptions from gather
            if isinstance(res, Exception):
                db.execute(
                    "INSERT INTO generation_failures (tmdb_id, metadata_type, error_message) VALUES (?, ?, ?)",
                    (tmdb_id, str(metadata_type), str(res)),
                )
                failed += 1
                continue

            tmdb_id, content, error_msg = res
            if content is not None:
                db.execute(
                    f"UPDATE generated_metadata SET {result_col} = ? WHERE tmdb_id = ?",
                    (content, tmdb_id),
                )
                succeeded += 1
            else:
                db.execute(
                    "INSERT INTO generation_failures (tmdb_id, metadata_type, error_message) VALUES (?, ?, ?)",
                    (tmdb_id, str(metadata_type), error_msg),
                )
                failed += 1

        db.commit()

    return succeeded, failed


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


def _print_eligibility_summary(
    db: sqlite3.Connection,
    metadata_type: MetadataType,
) -> None:
    """Print current eligibility counts for a metadata type."""
    # Column names from MetadataType StrEnum — fixed values, not user input.
    eligible_col = f"eligible_for_{metadata_type}"
    result_col = str(metadata_type)

    row = db.execute(
        f"""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN {eligible_col} = 1 THEN 1 ELSE 0 END) AS eligible,
            SUM(CASE WHEN {eligible_col} = 0 THEN 1 ELSE 0 END) AS ineligible,
            SUM(CASE WHEN {eligible_col} IS NULL THEN 1 ELSE 0 END) AS unevaluated,
            SUM(CASE WHEN {eligible_col} = 1 AND {result_col} IS NOT NULL THEN 1 ELSE 0 END) AS already_generated
        FROM generated_metadata
        """,
    ).fetchone()
    print(
        f"\n[{metadata_type}] Overall status:"
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
    metadata_type: MetadataType,
) -> None:
    """Record the batch_id for each movie in a submitted batch.

    Ensures a metadata_batch_ids row exists for each movie, then sets
    the {type}_batch_id column.
    """
    # Column name from MetadataType StrEnum — fixed values, not user input.
    batch_col = f"{metadata_type}_batch_id"

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
        f"UPDATE metadata_batch_ids SET {batch_col} = ? WHERE tmdb_id = ?",
        [(batch_id, tid) for tid in tmdb_ids],
    )


def _get_active_batch_ids(
    tracker_db_path: Path,
) -> list[tuple[str, MetadataType]]:
    """Get all active batch_ids across all registered metadata types.

    Returns (batch_id, MetadataType) pairs via a single UNION ALL query
    across all registered types' batch_id columns.

    Only queries columns for types in GENERATOR_REGISTRY — unregistered
    types (Wave 2 types not yet added) are skipped without touching the DB.
    """
    from movie_ingestion.metadata_generation.generator_registry import GENERATOR_REGISTRY

    registered_types = list(GENERATOR_REGISTRY.keys())
    if not registered_types:
        return []

    # Build a single UNION ALL query across all registered types.
    # Column names are from MetadataType StrEnum — fixed values, not user input.
    parts = []
    for mt in registered_types:
        batch_col = f"{mt}_batch_id"
        parts.append(
            f"SELECT DISTINCT {batch_col} AS batch_id, '{mt}' AS metadata_type "
            f"FROM metadata_batch_ids WHERE {batch_col} IS NOT NULL"
        )
    query = "\nUNION ALL\n".join(parts)

    with sqlite3.connect(str(tracker_db_path)) as db:
        rows = db.execute(query).fetchall()

    return [(row[0], MetadataType(row[1])) for row in rows]


def _clear_batch_id(
    tracker_db_path: Path,
    batch_id: str,
    metadata_type: MetadataType,
) -> None:
    """Clear the batch_id for all movies in a processed batch."""
    # Column name from MetadataType StrEnum — fixed values, not user input.
    batch_col = f"{metadata_type}_batch_id"

    with sqlite3.connect(str(tracker_db_path)) as db:
        db.execute(
            f"UPDATE metadata_batch_ids SET {batch_col} = NULL WHERE {batch_col} = ?",
            (batch_id,),
        )
        db.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

# Metadata types that have been registered and can be used with the CLI.
# Uses the registry to determine valid choices so the CLI stays in sync
# with which types actually have generator configs.
def _registered_type_choices() -> list[str]:
    """Get the list of metadata type values that have registered generators."""
    from movie_ingestion.metadata_generation.generator_registry import GENERATOR_REGISTRY
    return [str(mt) for mt in GENERATOR_REGISTRY]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Metadata generation pipeline CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- eligibility --
    eligibility_parser = subparsers.add_parser(
        "eligibility",
        help="Evaluate eligibility for a metadata type for all imdb_quality_passed movies",
    )
    eligibility_parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        choices=_registered_type_choices(),
        help="Which metadata type to evaluate eligibility for.",
    )

    # -- submit --
    submit_parser = subparsers.add_parser(
        "submit",
        help="Build and submit batch(es) for a metadata type to OpenAI",
    )
    submit_parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        choices=_registered_type_choices(),
        help="Which metadata type to submit batches for.",
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

    # -- status --
    subparsers.add_parser(
        "status",
        help="Check status of all active batches (all metadata types)",
    )

    # -- process --
    subparsers.add_parser(
        "process",
        help="Download and process results from completed batches (all metadata types)",
    )

    # -- autopilot --
    autopilot_parser = subparsers.add_parser(
        "autopilot",
        help="Interleave live generations with batch polling until done",
    )
    autopilot_parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        choices=_registered_type_choices(),
        help="Which metadata type to generate live and submit batches for. "
             "Status/process still handles all types.",
    )
    autopilot_parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Max requests per Batch API batch (default: 1000).",
    )
    autopilot_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Max Batch API batches in-flight at once (default: 3).",
    )
    autopilot_parser.add_argument(
        "--live-batch-size",
        type=int,
        default=25,
        help="Movies per live generation round (default: 25).",
    )
    autopilot_parser.add_argument(
        "--live-concurrency",
        type=int,
        default=5,
        help="Parallel live API calls (default: 5).",
    )
    args = parser.parse_args()

    if args.command == "eligibility":
        cmd_eligibility(metadata_type=MetadataType(args.metadata))
    elif args.command == "submit":
        cmd_submit(
            metadata_type=MetadataType(args.metadata),
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
    elif args.command == "status":
        cmd_status()
    elif args.command == "process":
        cmd_process()
    elif args.command == "autopilot":
        cmd_autopilot(
            metadata_type=MetadataType(args.metadata),
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            live_batch_size=args.live_batch_size,
            live_concurrency=args.live_concurrency,
        )


if __name__ == "__main__":
    main()
