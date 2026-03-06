"""
Stage 1: TMDB Export Download & Initial Filtering

Downloads the TMDB daily movie export (gzipped JSONL, ~1M entries),
stream-decompresses it line by line, and populates the SQLite tracker DB:

  - Movies passing all filters → movie_progress with status='pending'
  - Filtered-out movies → filter_log with stage='tmdb_export_filter'

Filters applied:
  1. adult == False
  2. video == False
  3. popularity > 0.0

Idempotent: re-running does INSERT OR IGNORE on movie_progress (existing
rows are untouched) and appends to the append-only filter_log.

Expected runtime: ~1-2 minutes.
"""

import gzip
import io
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Generator
from datetime import datetime, timedelta

from tqdm import tqdm

from movie_ingestion.tracker import PipelineStage, init_db

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TMDB publishes a daily export at this URL pattern. The date is yesterday's
# date formatted as MM_DD_YYYY (today's file may not be available yet).
_EXPORT_URL_TEMPLATE = "https://files.tmdb.org/p/exports/movie_ids_{date}.json.gz"

# Number of entries to accumulate before flushing to SQLite.
# 5000 balances transaction overhead (~200 commits for 1M entries) against
# crash resilience (at most 5K entries lost on crash, recovered via re-run).
_FLUSH_BATCH_SIZE = 5_000

# Stage identifier for all filter_log entries created by Stage 1.
_STAGE = PipelineStage.TMDB_EXPORT_FILTER

# Approximate total entries in the TMDB export (for tqdm progress bar).
# tqdm handles overruns gracefully if the actual count exceeds this.
_APPROX_TOTAL_ENTRIES = 1_100_000


# ---------------------------------------------------------------------------
# URL construction
# ---------------------------------------------------------------------------


def _build_export_url() -> str:
    """
    Build the TMDB daily export URL for yesterday's date.

    TMDB publishes exports for the previous day; using yesterday's date
    ensures the file is available.
    """
    yesterday = datetime.now() - timedelta(days=1)
    date_str = yesterday.strftime("%m_%d_%Y")
    return _EXPORT_URL_TEMPLATE.format(date=date_str)


# ---------------------------------------------------------------------------
# Stream decompression
# ---------------------------------------------------------------------------


def _stream_export_lines(url: str) -> Generator[dict, None, None]:
    """
    Download and stream-decompress the gzipped JSONL export, yielding one
    parsed JSON dict per line.

    Uses gzip.GzipFile wrapping the HTTP response stream so that the full
    compressed/decompressed payload is never held in memory at once. Peak
    memory usage is proportional to a single line (~200 bytes), not the
    full file (~200MB decompressed).

    Malformed lines are skipped with a warning printed to stderr — a single
    corrupt entry should not abort the entire run.

    Yields:
        Parsed JSON dict for each valid, non-empty line in the export.

    Raises:
        urllib.error.HTTPError: If the server returns an error (e.g., 404).
        urllib.error.URLError: If the connection fails entirely.
    """
    print(f"Downloading export from {url} ...")
    response = urllib.request.urlopen(url)

    # Stack: HTTP response (bytes) → GzipFile (decompressed bytes) →
    # TextIOWrapper (decoded str lines). Each layer pulls data on demand.
    gzip_stream = gzip.GzipFile(fileobj=response)
    text_stream = io.TextIOWrapper(gzip_stream, encoding="utf-8")

    line_num = 0
    for line in text_stream:
        line_num += 1
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            print(f"  Warning: malformed JSON on line {line_num}, skipping", file=sys.stderr)

    response.close()


# ---------------------------------------------------------------------------
# Entry classification
# ---------------------------------------------------------------------------


def _classify_entry(entry: dict) -> str | None:
    """
    Determine whether a TMDB export entry passes all filters.

    Check order matters for filter_log correctness: a movie that is both
    adult=True and popularity=0 should be logged with reason='adult'
    (the primary disqualifier), not 'zero_popularity'.

    Returns:
        None if the entry passes all filters (should be kept).
        A reason string ('adult', 'video', or 'zero_popularity') if filtered.
    """
    if entry.get("adult", False):
        return "adult"
    if entry.get("video", False):
        return "video"
    if entry.get("popularity", 0.0) <= 0.0:
        return "zero_popularity"
    return None


# ---------------------------------------------------------------------------
# Batch flush
# ---------------------------------------------------------------------------


def _flush_batch(db, pending_inserts: list[tuple], filter_entries: list[tuple]) -> None:
    """
    Write accumulated pending inserts and filter log entries to SQLite
    in a single transaction, then commit.

    Uses executemany for batch efficiency. INSERT OR IGNORE on movie_progress
    ensures idempotency — re-running Stage 1 silently skips tmdb_ids that
    already exist (regardless of their current status).

    Args:
        db:              Open SQLite connection.
        pending_inserts: List of (tmdb_id,) tuples for movie_progress.
        filter_entries:  List of (tmdb_id, title, year, stage, reason, details)
                         tuples for filter_log.
    """
    if pending_inserts:
        db.executemany(
            "INSERT OR IGNORE INTO movie_progress (tmdb_id) VALUES (?)",
            pending_inserts,
        )
    if filter_entries:
        db.executemany(
            """INSERT INTO filter_log (tmdb_id, title, year, stage, reason, details)
               VALUES (?, ?, ?, ?, ?, ?)""",
            filter_entries,
        )
    db.commit()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """
    Execute Stage 1: download TMDB daily export, filter, and populate tracker DB.

    Pipeline:
      1. Initialize the tracker DB (creates tables/indexes if needed)
      2. Stream-decompress the TMDB export line by line
      3. Classify each entry (pass / adult / video / zero_popularity)
      4. Batch-insert passing entries into movie_progress (INSERT OR IGNORE)
      5. Batch-insert filter entries into filter_log
      6. Print summary statistics
    """
    # --- Init ---
    db = init_db()
    url = _build_export_url()

    # --- Accumulators ---
    pending_inserts: list[tuple] = []
    filter_entries: list[tuple] = []

    total_count = 0
    inserted_count = 0
    filtered_count = 0
    filter_reasons = {"adult": 0, "video": 0, "zero_popularity": 0}
    skipped_no_id = 0

    start_time = time.monotonic()

    # --- Stream and process ---
    try:
        entry_stream = _stream_export_lines(url)
    except urllib.error.HTTPError as e:
        print(f"Error: TMDB export download failed with HTTP {e.code}.")
        print(f"  URL: {url}")
        print("  The file may not be available yet. Try again later or check the date.")
        db.close()
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Error: could not connect to TMDB export server: {e.reason}")
        db.close()
        sys.exit(1)

    for entry in tqdm(entry_stream, desc="Processing TMDB export", unit=" entries", total=_APPROX_TOTAL_ENTRIES):
        total_count += 1

        # Guard against malformed entries missing an ID field
        tmdb_id = entry.get("id")
        if tmdb_id is None:
            skipped_no_id += 1
            continue

        reason = _classify_entry(entry)

        if reason is None:
            # Entry passes all filters — queue for movie_progress insertion
            pending_inserts.append((tmdb_id,))
            inserted_count += 1
        else:
            # Entry filtered out — queue for filter_log
            # title and year are NULL at Stage 1 (no TMDB detail file exists yet)
            filter_entries.append((tmdb_id, None, None, _STAGE, reason, None))
            filtered_count += 1
            filter_reasons[reason] += 1

        # Flush when combined buffer reaches the batch threshold
        if len(pending_inserts) + len(filter_entries) >= _FLUSH_BATCH_SIZE:
            _flush_batch(db, pending_inserts, filter_entries)
            pending_inserts.clear()
            filter_entries.clear()

    # Flush any remaining entries
    if pending_inserts or filter_entries:
        _flush_batch(db, pending_inserts, filter_entries)

    elapsed = time.monotonic() - start_time
    db.close()

    # --- Summary ---
    print(f"\n{'=' * 60}")
    print("Stage 1 Complete")
    print(f"{'=' * 60}")
    print(f"  Total entries processed: {total_count:,}")
    print(f"  Inserted (pending):      {inserted_count:,}")
    print(f"  Filtered out:            {filtered_count:,}")
    print(f"    - adult:               {filter_reasons['adult']:,}")
    print(f"    - video:               {filter_reasons['video']:,}")
    print(f"    - zero_popularity:     {filter_reasons['zero_popularity']:,}")
    if skipped_no_id:
        print(f"  Skipped (no ID):         {skipped_no_id:,}")
    print(f"  Duration:                {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run()
