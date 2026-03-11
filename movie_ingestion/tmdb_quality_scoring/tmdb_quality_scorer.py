"""
Stage 3: TMDB Quality Scorer

Computes a quality score for every movie with status='tmdb_fetched' in
movie_progress and persists it to the stage_3_quality_score column.

Scoring model (4 signals, weights sum to 1.0, output range [0, 1])
------------------------------------------------------------------
  quality_score = 0.50 * vote_count
                + 0.20 * popularity
                + 0.15 * overview_length
                + 0.15 * data_completeness

Two edge cases bypass the formula entirely (checked in this order):
  1. Unreleased movies (release_date set and in the future) → 0.0
  2. Movies with ≥1 US watch provider → 1.0

The formula only runs on the no-provider, released population (~508K movies).
This is deliberately lenient — the real quality gate is Stage 5 after IMDB
data is available.

See docs/modules/ingestion.md "Stage 3: Quality Scoring Model" for the
full design rationale, signal details, and expected score distributions.

Idempotent: re-running overwrites existing stage_3_quality_score values and
re-sets status to TMDB_QUALITY_CALCULATED.  The filter threshold is applied
separately in tmdb_filter.py.

Usage:
    python -m movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer
"""

import datetime
import sqlite3

from movie_ingestion.scoring_utils import (
    VoteCountSource,
    score_popularity,
    score_vote_count,
    unpack_provider_keys,
    validate_weights,
)
from movie_ingestion.tracker import MovieStatus, init_db

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Flush scored rows to disk every N movies.
COMMIT_EVERY: int = 1_000

# Print a progress line every N movies.
LOG_EVERY: int = 10_000

# ---------------------------------------------------------------------------
# Signal weights — must sum to 1.0
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "vote_count":        0.50,
    "popularity":        0.20,
    "overview_length":   0.15,
    "data_completeness": 0.15,
}

# Guard at module load time — see scoring_utils.validate_weights docstring.
validate_weights(WEIGHTS)

# ---------------------------------------------------------------------------
# Edge-case helpers
# ---------------------------------------------------------------------------


def _is_unreleased(release_date: str | None, today: datetime.date) -> bool:
    """Return True if the movie has a future release date.

    Null release_date → False (not treated as unreleased — a missing date
    doesn't mean the movie is upcoming, it's often just incomplete data).
    Non-parseable dates → False (conservative: don't auto-fail on bad data).
    """
    if release_date is None:
        return False
    try:
        return datetime.date.fromisoformat(release_date) > today
    except ValueError:
        return False


def _has_us_providers(watch_provider_keys_blob: bytes | None) -> bool:
    """Return True if the movie has at least one US watch provider."""
    return len(unpack_provider_keys(watch_provider_keys_blob)) > 0


# ---------------------------------------------------------------------------
# Individual signal scoring functions
# ---------------------------------------------------------------------------


def score_overview_length(length: int) -> float:
    """Tiered overview-length score in [0.0, 1.0] based on character count.

    Movies with no overview get 0.0 (they cannot participate meaningfully in
    vector search).  Full overviews (201+ chars) receive the maximum score.
    Tiers are calibrated to the no-provider population where 10.2% have no
    overview and the median overview is ~180 chars.
    """
    if length == 0:
        return 0.0
    elif length <= 50:
        return 0.2
    elif length <= 100:
        return 0.5
    elif length <= 200:
        return 0.8
    else:
        return 1.0


def score_data_completeness(row: sqlite3.Row) -> float:
    """Average of 8 binary data-presence indicators, in [0.0, 1.0].

    Consolidates many small metadata signals into a single composite.
    Each indicator is worth 1/8 = 0.125.  A fully populated movie scores
    1.0; a bare stub with only a TMDB ID scores 0.0.

    The 8 indicators:
      - has_genres:                genre_count > 0
      - has_poster:                poster_url is not None
      - has_cast_and_crew:         from tmdb_data column (0/1)
      - has_production_countries:  from tmdb_data column (0/1)
      - has_production_companies:  from tmdb_data column (0/1)
      - has_keywords:              from tmdb_data column (0/1)
      - has_budget:                from tmdb_data column (0/1)
      - has_revenue:               from tmdb_data column (0/1)
    """
    indicators = (
        row["genre_count"] > 0,
        row["poster_url"] is not None,
        bool(row["has_cast_and_crew"]),
        bool(row["has_production_countries"]),
        bool(row["has_production_companies"]),
        bool(row["has_keywords"]),
        bool(row["has_budget"]),
        bool(row["has_revenue"]),
    )
    return sum(indicators) / len(indicators)


# ---------------------------------------------------------------------------
# Public scoring function — reusable entry point
# ---------------------------------------------------------------------------


def compute_quality_score(row: sqlite3.Row, today: datetime.date) -> float:
    """Compute the Stage 3 quality score for a single movie.

    This is the public, isolated entry point for Stage 3 scoring.  It can
    be imported and called from any module that has a sqlite3.Row with the
    required columns (see run() for the SELECT statement).

    Edge cases are checked first, in order:
      1. Unreleased → 0.0  (future release date means no engagement data)
      2. Has US providers → 1.0  (streaming licensing proves commercial viability)
      3. Otherwise → weighted formula on 4 signals

    Args:
        row:   sqlite3.Row with columns: vote_count, popularity, release_date,
               watch_provider_keys, overview_length, poster_url, genre_count,
               has_cast_and_crew, has_production_countries, has_production_companies,
               has_keywords, has_budget, has_revenue.
        today: Scoring reference date (computed once by the caller).

    Returns:
        Quality score in [0.0, 1.0].
    """
    # Edge case 1: unreleased movies cannot have meaningful engagement data.
    if _is_unreleased(row["release_date"], today):
        return 0.0

    # Edge case 2: a US streaming license is definitive proof of commercial
    # viability — no need to score further.
    if _has_us_providers(row["watch_provider_keys"]):
        return 1.0

    # Weighted formula for no-provider, released/null-date movies.
    # Uses TMDB_NO_PROVIDER log cap (101) calibrated to the no-provider
    # population where p99 vote_count ≈ 72.
    vc_score = score_vote_count(
        row["vote_count"], row["release_date"], today,
        VoteCountSource.TMDB_NO_PROVIDER,
    )
    pop_score = score_popularity(row["popularity"])
    ol_score = score_overview_length(row["overview_length"])
    dc_score = score_data_completeness(row)

    return (
        WEIGHTS["vote_count"]        * vc_score
        + WEIGHTS["popularity"]      * pop_score
        + WEIGHTS["overview_length"] * ol_score
        + WEIGHTS["data_completeness"] * dc_score
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run() -> None:
    """Score all tmdb_fetched movies and persist to movie_progress.stage_3_quality_score.

    Uses executemany batch writes (COMMIT_EVERY rows per flush) instead of
    per-row UPDATEs for efficiency.  Progress is reported every LOG_EVERY rows.
    A summary with score distribution statistics is printed on completion.
    """
    today: datetime.date = datetime.date.today()

    db = init_db()
    db.row_factory = sqlite3.Row

    print("Stage 3 quality scorer: loading tmdb_fetched movies...")

    # fetchall() materialises the full result set upfront.  This is intentional:
    # the loop body mutates movie_progress (quality score UPDATEs), and lazy
    # cursor iteration over a JOIN that includes movie_progress can produce
    # undefined behavior in SQLite when the underlying table is modified mid-scan.
    rows = db.execute("""
        SELECT
            d.tmdb_id,
            d.vote_count,
            d.popularity,
            d.release_date,
            d.watch_provider_keys,
            d.overview_length,
            d.poster_url,
            d.genre_count,
            d.has_cast_and_crew,
            d.has_production_countries,
            d.has_production_companies,
            d.has_keywords,
            d.has_budget,
            d.has_revenue
        FROM tmdb_data d
        JOIN movie_progress p ON d.tmdb_id = p.tmdb_id
        WHERE p.status = ?
    """, (MovieStatus.TMDB_FETCHED,)).fetchall()

    total = len(rows)
    print(f"  {total:,} movies to score (reference date = {today})")

    if total == 0:
        print("No tmdb_fetched movies found. Has Stage 2 completed?")
        db.close()
        return

    scored: int = 0
    score_sum: float = 0.0
    score_min: float = float("inf")
    score_max: float = float("-inf")

    # Batch of (score, tmdb_id) tuples for executemany.
    batch: list[tuple[float, int]] = []

    try:
        for i, row in enumerate(rows):
            score = compute_quality_score(row, today)

            batch.append((score, row["tmdb_id"]))
            scored += 1
            score_sum += score
            score_min = min(score_min, score)
            score_max = max(score_max, score)

            # Flush batch to DB periodically.
            if len(batch) >= COMMIT_EVERY:
                db.executemany(
                    """UPDATE movie_progress
                       SET stage_3_quality_score = ?,
                           status = ?,
                           updated_at = CURRENT_TIMESTAMP
                       WHERE tmdb_id = ?""",
                    [(score, MovieStatus.TMDB_QUALITY_CALCULATED, tmdb_id) for score, tmdb_id in batch],
                )
                db.commit()
                batch.clear()

            if (i + 1) % LOG_EVERY == 0:
                print(f"  Scored {i + 1:,}/{total:,}")

        # Flush any remaining rows in the final partial batch.
        if batch:
            db.executemany(
                """UPDATE movie_progress
                   SET stage_3_quality_score = ?,
                       status = ?,
                       updated_at = CURRENT_TIMESTAMP
                   WHERE tmdb_id = ?""",
                [(score, MovieStatus.TMDB_QUALITY_CALCULATED, tmdb_id) for score, tmdb_id in batch],
            )
            db.commit()
    finally:
        db.close()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    score_mean = score_sum / scored if scored > 0 else 0.0

    print(f"\nStage 3 quality scorer complete")
    print(f"  Movies scored: {scored:,}")
    print(f"  Score min:     {score_min:.4f}")
    print(f"  Score mean:    {score_mean:.4f}")
    print(f"  Score max:     {score_max:.4f}")


if __name__ == "__main__":
    run()
