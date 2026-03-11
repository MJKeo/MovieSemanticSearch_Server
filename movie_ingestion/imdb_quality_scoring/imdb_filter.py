"""
Stage 5: IMDB Quality Filter — Per-Group Score Thresholds

Reads pre-computed stage_5_quality_score values from movie_progress and applies
per-group thresholds.  Movies are classified into three groups based on watch
provider availability and release recency (see scoring_utils.MovieGroup), each
with its own threshold from survival-curve analysis.

Movies scoring below their group's threshold are marked filtered_out; survivors
are advanced from 'imdb_quality_calculated' to 'imdb_quality_passed' for
Stage 6 (LLM generation).

This module does NOT compute scores — that is handled by imdb_quality_scorer.py,
which must be run first.  Separating scoring from filtering allows re-running
the threshold without re-scoring.

Idempotent: already-filtered movies (status != 'imdb_quality_calculated') are
skipped, so the script can be re-run safely after a partial execution.

Usage:
    python -m movie_ingestion.imdb_quality_scoring.imdb_filter
"""

import datetime
import sqlite3

from movie_ingestion.scoring_utils import (
    IMDB_QUALITY_THRESHOLDS,
    MovieGroup,
    classify_movie_group,
    passes_imdb_quality_threshold,
)
from movie_ingestion.tracker import MovieStatus, PipelineStage, init_db, log_filter

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Flush to disk every N rows processed.
COMMIT_EVERY: int = 1_000

# Emit a progress line every N rows processed.
LOG_EVERY: int = 10_000

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Apply per-group quality-score thresholds to all imdb_quality_calculated movies.

    For each movie:
      1. Classify into a MovieGroup using watch_provider_keys + release_date.
      2. Compare stage_5_quality_score against the group's threshold.
      3. Below threshold → log_filter (filtered_out).
      4. At or above threshold → left in place for bulk advancement.

    After all movies are evaluated, survivors (still at 'imdb_quality_calculated')
    are advanced to 'imdb_quality_passed' via a single bulk UPDATE.
    """
    today = datetime.date.today()
    db = init_db()
    db.row_factory = sqlite3.Row

    print("Stage 5 quality filter: loading imdb_quality_calculated movies...")

    # Join movie_progress with tmdb_data to get the fields needed for group
    # classification (watch_provider_keys, release_date).  Materialise upfront
    # because log_filter() mutates movie_progress in the loop.
    rows = db.execute("""
        SELECT
            mp.tmdb_id,
            mp.stage_5_quality_score,
            td.watch_provider_keys,
            td.release_date
        FROM movie_progress mp
        JOIN tmdb_data td ON td.tmdb_id = mp.tmdb_id
        WHERE mp.status = ?
    """, (MovieStatus.IMDB_QUALITY_CALCULATED,)).fetchall()

    total = len(rows)

    # Print threshold summary for each group.
    print(f"  {total:,} movies to evaluate")
    for group, threshold in IMDB_QUALITY_THRESHOLDS.items():
        print(f"    [{group.value}] threshold = {threshold}")

    if total == 0:
        print("No imdb_quality_calculated movies found. Nothing to filter.")
        db.close()
        return

    # Guard: abort if scores haven't been computed yet.  A NULL score means
    # imdb_quality_scorer.py was not run (or didn't reach this movie).
    null_count = sum(1 for r in rows if r["stage_5_quality_score"] is None)
    if null_count > 0:
        print(
            f"ERROR: {null_count:,} movies have NULL stage_5_quality_score.\n"
            f"Run imdb_quality_scorer.py first to compute scores."
        )
        db.close()
        return

    # Per-group counters for the summary.
    group_totals: dict[MovieGroup, int] = {g: 0 for g in MovieGroup}
    group_filtered: dict[MovieGroup, int] = {g: 0 for g in MovieGroup}

    filtered_total: int = 0
    pending_commit: int = 0

    try:
        for i, row in enumerate(rows):
            score = row["stage_5_quality_score"]
            group = classify_movie_group(
                provider_keys=row["watch_provider_keys"],
                release_date=row["release_date"],
                today=today,
            )
            group_totals[group] += 1

            if not passes_imdb_quality_threshold(group, score):
                log_filter(
                    db,
                    tmdb_id=row["tmdb_id"],
                    stage=PipelineStage.IMDB_QUALITY_FUNNEL,
                    reason="below_quality_threshold",
                )
                filtered_total += 1
                group_filtered[group] += 1
                pending_commit += 1

            if pending_commit >= COMMIT_EVERY:
                db.commit()
                pending_commit = 0

            if (i + 1) % LOG_EVERY == 0:
                print(
                    f"  Processed {i + 1:,}/{total:,}"
                    f" | filtered so far: {filtered_total:,}"
                )

        # Flush any remaining uncommitted filter writes.
        if pending_commit > 0:
            db.commit()

        # Advance all surviving movies (still at 'imdb_quality_calculated') to
        # the next pipeline status in a single bulk UPDATE.
        db.execute(
            """UPDATE movie_progress
               SET status = ?, updated_at = CURRENT_TIMESTAMP
               WHERE status = ?""",
            (MovieStatus.IMDB_QUALITY_PASSED, MovieStatus.IMDB_QUALITY_CALCULATED),
        )
        db.commit()
    finally:
        db.close()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    surviving = total - filtered_total

    print(f"\nStage 5 quality filter complete")
    print(f"  Total evaluated:  {total:,}")
    print(f"  Filtered out:     {filtered_total:,}")
    print(f"  Surviving:        {surviving:,}")

    # Per-group breakdown.
    print(f"\n  Per-group breakdown:")
    for group in MovieGroup:
        gt = group_totals[group]
        gf = group_filtered[group]
        gs = gt - gf
        threshold = IMDB_QUALITY_THRESHOLDS[group]
        print(
            f"    [{group.value}] (threshold {threshold}): "
            f"{gt:,} total, {gf:,} filtered, {gs:,} surviving"
        )


if __name__ == "__main__":
    run()
