"""
Stage 3: TMDB Quality Funnel — Hard Filters + Quality Score Threshold

Applies five hard filters and a quality-score threshold to every movie that has
status='tmdb_fetched' in movie_progress.  Movies failing any check are marked
filtered_out and logged to filter_log; survivors retain status='tmdb_fetched'
and advanced to 'tmdb_quality_passed' for Stage 4 (IMDB scraping).

Quality scores are computed inline via compute_quality_score() and persisted to
movie_progress.quality_score for every movie, eliminating the need for a
separate Stage 3b scoring pass.

Filters (applied in priority order — first failing reason is logged):

  1. zero_vote_count          — vote_count = 0   (no audience engagement)
  2. missing_or_zero_duration — duration IS NULL OR duration = 0
  3. missing_overview         — overview_length = 0
  4. no_genres                — genre_count = 0
  5. future_release           — release_date IS NOT NULL AND release_date > today
  6. below_quality_threshold  — quality_score < QUALITY_SCORE_THRESHOLD (-0.0441)

Idempotent: already-filtered movies (status != 'tmdb_fetched') are skipped,
so the script can be re-run safely after a partial execution.

Usage:
    python -m movie_ingestion.tmdb_filter
"""

import datetime
import json
import sqlite3
from collections.abc import Callable

from movie_ingestion.tmdb_quality_scorer import compute_quality_score
from movie_ingestion.tracker import MovieStatus, PipelineStage, init_db, log_filter

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Flush to disk every N rows processed.  Every row now generates at least one
# DB write (quality score UPDATE), plus an additional INSERT for filtered movies.
# Bounds data loss on crash to at most COMMIT_EVERY rows of work.
COMMIT_EVERY: int = 1_000

# Emit a progress line every N rows processed (independent of filter outcome).
LOG_EVERY: int = 10_000

# Soft quality-score cutoff derived from the f'' local max of the survival
# curve's derivative analysis.  Marks the boundary of the "uncertainty zone"
# where TMDB scoring errors have outsized impact.  Movies scoring below this
# threshold are filtered out before IMDB scraping to avoid wasting budget on
# movies unlikely to survive the final hard cutoff (0.1356).
QUALITY_SCORE_THRESHOLD: float = -0.0441

# ---------------------------------------------------------------------------
# Hard filter predicates
# ---------------------------------------------------------------------------
# Each predicate receives a sqlite3.Row (named-column access) and today's ISO
# date string.  Returns True when the movie FAILS the filter and should be
# eliminated.  The ordered list below defines priority: when a movie fails
# multiple filters, the first matching entry supplies the logged reason.

_FilterPredicate = Callable[[sqlite3.Row, str], bool]


def _fails_vote_count(row: sqlite3.Row, _today: str) -> bool:
    """No TMDB audience engagement at all — the foundational quality gate."""
    return row["vote_count"] == 0


def _fails_duration(row: sqlite3.Row, _today: str) -> bool:
    """Runtime unknown (NULL) or recorded as zero — cannot classify as a film."""
    d = row["duration"]
    return d is None or d == 0


def _fails_overview(row: sqlite3.Row, _today: str) -> bool:
    """No overview text — movie cannot participate in vector search at all."""
    return row["overview_length"] == 0


def _fails_genre_count(row: sqlite3.Row, _today: str) -> bool:
    """No genre classification — movie is not properly catalogued in TMDB."""
    return row["genre_count"] == 0


def _fails_future_release(row: sqlite3.Row, today: str) -> bool:
    """Release date is set but lies in the future — IMDB page will be sparse.

    Null release dates are intentionally NOT filtered here (that filter was
    evaluated and rejected as too aggressive — a missing date doesn't impair
    the movie's usability in the pipeline).

    Note: release_date is stored as TEXT in SQLite (no native DATE type).
    We parse it via fromisoformat() rather than comparing strings directly so
    that malformed values (e.g. year-only "2026" or year-month "2026-03")
    are caught and skipped rather than producing a silently wrong result.
    """
    rd = row["release_date"]
    if rd is None:
        return False
    try:
        return datetime.date.fromisoformat(rd) > datetime.date.fromisoformat(today)
    except ValueError:
        # Non-parseable date format — do not filter on incomplete information.
        return False


# Priority-ordered list of (reason_string, predicate) pairs.
# The reason string matches the values written to filter_log.reason.
_HARD_FILTERS: list[tuple[str, _FilterPredicate]] = [
    ("zero_vote_count",          _fails_vote_count),
    ("missing_or_zero_duration", _fails_duration),
    ("missing_overview",         _fails_overview),
    ("no_genres",                _fails_genre_count),
    ("future_release",           _fails_future_release),
]


# ---------------------------------------------------------------------------
# Filter evaluation
# ---------------------------------------------------------------------------


def _evaluate_filters(
    row: sqlite3.Row,
    today: str,
) -> tuple[str | None, list[str]]:
    """Check all hard filters for a single movie row.

    Evaluates every predicate in _HARD_FILTERS priority order.

    Args:
        row:   Named-column sqlite3.Row containing the filter-relevant fields.
        today: Today's date as an ISO 8601 string (``"YYYY-MM-DD"``).

    Returns:
        A tuple ``(primary_reason, all_failing_reasons)`` where:

        - ``primary_reason`` is the first (highest-priority) reason string
          that fired, or ``None`` if the movie passes all filters.
        - ``all_failing_reasons`` is a list of every reason that fired.
          When the movie passes, this list is empty.
    """
    failing: list[str] = [
        reason
        for reason, predicate in _HARD_FILTERS
        if predicate(row, today)
    ]
    primary = failing[0] if failing else None
    return primary, failing


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Apply hard filters and quality-score threshold to all tmdb_fetched movies.

    For every movie with status='tmdb_fetched':
      1. Compute and persist a quality score (via compute_quality_score).
      2. Evaluate the five hard filters in priority order.
      3. Check the quality-score threshold (lowest priority).
      4. Call log_filter() for any movie that fails at least one check.

    Survivors are advanced to 'tmdb_quality_passed' via a single bulk UPDATE.

    Progress is reported every LOG_EVERY rows.  A summary table is printed
    at the end showing the per-reason breakdown and the surviving count.
    """
    # Compute today once so every row uses the same reference point.
    # Both forms are needed: datetime.date for compute_quality_score,
    # ISO string for the hard-filter predicates.
    today_date: datetime.date = datetime.date.today()
    today: str = today_date.isoformat()

    db = init_db()

    # sqlite3.Row enables name-based column access (row["vote_count"] etc.)
    # and is compatible with the positional access used inside log_filter().
    db.row_factory = sqlite3.Row

    print("Stage 3 hard filter + quality scoring: loading tmdb_fetched movies...")
    # fetchall() materialises the full result set upfront.  This is intentional:
    # the loop body mutates movie_progress (quality score UPDATEs, status changes
    # via log_filter), and lazy cursor iteration over a join that includes
    # movie_progress can produce undefined behavior in SQLite when the underlying
    # table is modified mid-scan.
    rows = db.execute("""
        SELECT
            d.tmdb_id,
            d.vote_count,
            d.duration,
            d.overview_length,
            d.genre_count,
            d.release_date,
            d.popularity,
            d.poster_url,
            d.watch_provider_keys,
            d.has_revenue,
            d.has_budget,
            d.has_production_companies,
            d.has_keywords,
            d.has_cast_and_crew
        FROM tmdb_data d
        JOIN movie_progress p ON d.tmdb_id = p.tmdb_id
        WHERE p.status = ?
    """, (MovieStatus.TMDB_FETCHED,)).fetchall()

    total = len(rows)
    print(f"  {total:,} movies to evaluate (today = {today})")

    if total == 0:
        print("No tmdb_fetched movies found. Has Stage 2 completed?")
        db.close()
        return

    # Per-reason counters for the final summary table.
    reason_counts: dict[str, int] = {reason: 0 for reason, _ in _HARD_FILTERS}
    reason_counts["below_quality_threshold"] = 0
    filtered_total: int = 0
    # Tracks writes since the last commit so we flush at COMMIT_EVERY intervals.
    # Every row generates at least one write (quality score UPDATE), so this now
    # increments unconditionally rather than only on filter hits.
    pending_commit: int = 0

    try:
        for i, row in enumerate(rows):
            # --- Quality score: compute once, persist for all movies ---
            # This eliminates the need for a separate Stage 3b scoring pass.
            score = compute_quality_score(row, today_date)

            db.execute(
                """UPDATE movie_progress
                   SET quality_score = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE tmdb_id = ?""",
                (score, row["tmdb_id"]),
            )
            pending_commit += 1

            # --- Evaluate hard filters + quality threshold ---
            primary_reason, all_failing = _evaluate_filters(row, today)

            # Quality score threshold is the lowest-priority filter: if a movie
            # also fails a hard filter (e.g. zero votes), the hard filter reason
            # remains the primary logged reason for better diagnostics.
            if score < QUALITY_SCORE_THRESHOLD:
                all_failing.append("below_quality_threshold")
                if primary_reason is None:
                    primary_reason = "below_quality_threshold"

            if primary_reason is not None:
                # When multiple filters fire, record the secondary reasons in the
                # details JSON so no diagnostic information is lost.
                details: str | None = None
                if len(all_failing) > 1:
                    details = json.dumps({"also_failed": all_failing[1:]})

                # log_filter handles both the filter_log INSERT and the
                # movie_progress status update — never write to those tables
                # directly from this module.
                log_filter(
                    db,
                    tmdb_id=row["tmdb_id"],
                    stage=PipelineStage.TMDB_QUALITY_FUNNEL,
                    reason=primary_reason,
                    details=details,
                )

                reason_counts[primary_reason] += 1
                filtered_total += 1

            # Commit periodically so progress survives a crash.
            if pending_commit >= COMMIT_EVERY:
                db.commit()
                pending_commit = 0

            if (i + 1) % LOG_EVERY == 0:
                print(
                    f"  Processed {i + 1:,}/{total:,}"
                    f" | filtered so far: {filtered_total:,}"
                )

        # Flush any remaining uncommitted writes.
        db.commit()

        # Advance all surviving movies (still at 'tmdb_fetched') to the next
        # pipeline status in a single bulk UPDATE.  Any movie whose status was
        # changed to 'filtered_out' during the loop is excluded by the WHERE.
        db.execute(
            """UPDATE movie_progress
               SET status = ?, updated_at = CURRENT_TIMESTAMP
               WHERE status = ?""",
            (MovieStatus.TMDB_QUALITY_PASSED, MovieStatus.TMDB_FETCHED),
        )
        db.commit()
    finally:
        # Guarantee the connection is released even if compute_quality_score
        # raises (e.g. corrupted BLOB in watch_provider_keys), preventing a
        # connection leak and ensuring SQLite's WAL journal is properly closed.
        db.close()

    # ---------------------------------------------------------------------------
    # Final summary
    # ---------------------------------------------------------------------------
    surviving = total - filtered_total

    print(f"\nStage 3 hard filter + quality scoring complete")
    print(f"  Total evaluated:  {total:,}")
    print(f"  Filtered out:     {filtered_total:,}")
    print(f"  Surviving:        {surviving:,}")
    print(f"\nBreakdown by primary reason:")
    for reason, count in reason_counts.items():
        pct = (count / total * 100) if total > 0 else 0.0
        print(f"  {reason:<30}  {count:>8,}  ({pct:.1f}%)")


if __name__ == "__main__":
    run()
