"""
Stage 3: TMDB Quality Filter — Score Threshold

Reads pre-computed stage_3_quality_score values from movie_progress and applies
a threshold.  Movies scoring below the threshold are marked filtered_out;
survivors are advanced from 'tmdb_quality_calculated' to 'tmdb_quality_passed' for
Stage 4 (IMDB scraping).

This module does NOT compute scores — that is handled by tmdb_quality_scorer.py,
which must be run first.  Separating scoring from filtering follows the Stage 5
pattern and allows re-running the threshold without re-scoring.

Idempotent: already-filtered movies (status != 'tmdb_quality_calculated') are skipped,
so the script can be re-run safely after a partial execution.

Usage:
    python -m movie_ingestion.tmdb_quality_scoring.tmdb_filter
"""

import sqlite3

from movie_ingestion.tracker import MovieStatus, PipelineStage, init_db, log_filter

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Flush to disk every N rows processed.
COMMIT_EVERY: int = 1_000

# Emit a progress line every N rows processed.
LOG_EVERY: int = 10_000

# Threshold determined by survival curve derivative analysis on the
# no-provider population (~505K movies).  0.2344 is the inflection point
# (f' minimum / peak attrition) — everything below is in the densest
# concentration of low-quality movies.  Deliberately lenient: Stage 5
# performs the real quality cut with doubled data (IMDB + TMDB).
QUALITY_SCORE_THRESHOLD: float = 0.2344

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Apply quality-score threshold to all tmdb_quality_calculated movies.

    Reads stage_3_quality_score from movie_progress (populated by
    tmdb_quality_scorer.py).  Movies below QUALITY_SCORE_THRESHOLD are
    logged as filtered_out.  Survivors are advanced to 'tmdb_quality_passed'
    via a single bulk UPDATE.
    """
    db = init_db()
    db.row_factory = sqlite3.Row

    print("Stage 3 quality filter: loading tmdb_quality_calculated movies...")

    # Materialise upfront — we mutate movie_progress in the loop via
    # log_filter(), so lazy cursor iteration would be unsafe.
    rows = db.execute("""
        SELECT tmdb_id, stage_3_quality_score
        FROM movie_progress
        WHERE status = ?
    """, (MovieStatus.TMDB_QUALITY_CALCULATED,)).fetchall()

    total = len(rows)
    print(f"  {total:,} movies to evaluate (threshold = {QUALITY_SCORE_THRESHOLD})")

    if total == 0:
        print("No tmdb_quality_calculated movies found. Nothing to filter.")
        db.close()
        return

    # Guard: abort if scores haven't been computed yet.  A NULL score means
    # tmdb_quality_scorer.py was not run (or didn't reach this movie).
    null_count = sum(1 for r in rows if r["stage_3_quality_score"] is None)
    if null_count > 0:
        print(
            f"ERROR: {null_count:,} movies have NULL stage_3_quality_score.\n"
            f"Run tmdb_quality_scorer.py first to compute scores."
        )
        db.close()
        return

    filtered_total: int = 0
    pending_commit: int = 0

    try:
        for i, row in enumerate(rows):
            score = row["stage_3_quality_score"]

            if score < QUALITY_SCORE_THRESHOLD:
                log_filter(
                    db,
                    tmdb_id=row["tmdb_id"],
                    stage=PipelineStage.TMDB_QUALITY_FUNNEL,
                    reason="below_quality_threshold",
                )
                filtered_total += 1
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

        # Advance all surviving movies (still at 'tmdb_quality_calculated') to
        # the next pipeline status in a single bulk UPDATE.
        db.execute(
            """UPDATE movie_progress
               SET status = ?, updated_at = CURRENT_TIMESTAMP
               WHERE status = ?""",
            (MovieStatus.TMDB_QUALITY_PASSED, MovieStatus.TMDB_QUALITY_CALCULATED),
        )
        db.commit()
    finally:
        db.close()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    surviving = total - filtered_total

    print(f"\nStage 3 quality filter complete")
    print(f"  Total evaluated:  {total:,}")
    print(f"  Filtered out:     {filtered_total:,}")
    print(f"  Surviving:        {surviving:,}")


if __name__ == "__main__":
    run()
