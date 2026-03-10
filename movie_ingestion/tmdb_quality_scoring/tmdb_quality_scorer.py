"""
Stage 3b: TMDB Quality Scorer

Computes a raw weighted quality score for every movie that has
status='tmdb_fetched' in movie_progress and writes it to the pre-existing
stage_3_quality_score column.  The score is intentionally NOT normalised here — a
separate pass will min-max normalise across all movies, plot the distribution,
and apply the final quality cutoff that transitions movies to
status='tmdb_quality_passed' or status='filtered_out'.

Scoring model
-------------
Ten signals contribute to a single weighted sum.  The weights sum to 1.0.
Individual signal outputs are NOT all in [0, 1] — boolean fields use a signed
scale so that common fields (present in 75–95% of movies) generate a real
penalty when absent rather than a meaningless boost when present.

  Signal                   Weight   Range    Notes
  ───────────────────────  ──────   ──────   ───────────────────────────────
  vote_count               0.38    [0,  1]  Log-scaled; cap at vc=2000.
                                            Recency multiplier (up to 2×) for
                                            films < 2 years old; classic
                                            multiplier (up to 1.5×) for films
                                            > 20 years old.
  watch_providers          0.25   [-1, +1]  Tiered; harsh -1 for no US
                                            streaming past theater window.
  popularity               0.12    [0,  1]  Log-scaled; cap at pop=10.
  has_revenue              0.05    [0, +1]  Rare (7%): bonus when present.
  poster_url               0.05   [-1,  0]  Common (95%): penalty when absent.
  overview_length          0.04  [0.2, 1]  Tiered by character count.
  has_keywords             0.04  [-0.5,+0.5]  Symmetric; ~half have it.
  has_production_companies 0.03   [-1,  0]  Common (78%): penalty when absent.
  has_budget               0.02    [0, +1]  Rare (10%): bonus when present.
  has_cast_and_crew        0.02   [-1,  0]  Common (91%): penalty when absent.

Raw score range
---------------
  Maximum ≈ +0.88   (vc=2000+, 3+ providers, pop=10+, all bonuses, no deficits)
  Minimum ≈ −0.34   (vc=1, no providers past window, all penalty fields absent)

Idempotent: re-running overwrites existing stage_3_quality_score values.  The movie's
status is not changed by this stage — that happens in the normalisation pass.

Usage:
    python -m movie_ingestion.tmdb_quality_scorer
"""

import datetime
import sqlite3

from movie_ingestion.scoring_utils import (
    THEATER_WINDOW_DAYS,
    VoteCountSource,
    score_popularity,
    score_vote_count,
    unpack_provider_keys,
    validate_weights,
)
from movie_ingestion.tracker import MovieStatus, init_db

# Commit scored rows to disk every N movies.
COMMIT_EVERY: int = 1_000

# Print a progress line every N movies (independent of commit cadence).
LOG_EVERY: int = 10_000

# ---------------------------------------------------------------------------
# Signal weights — must sum to 1.0.
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "vote_count":               0.38,
    "watch_providers":          0.25,
    "popularity":               0.12,
    "has_revenue":              0.05,
    "poster_url":               0.05,
    "overview_length":          0.04,
    "has_keywords":             0.04,
    "has_production_companies": 0.03,
    "has_budget":               0.02,
    "has_cast_and_crew":        0.02,
}

# Guard at module load time — see scoring_utils.validate_weights docstring.
validate_weights(WEIGHTS)

# ---------------------------------------------------------------------------
# Individual signal scoring functions
# ---------------------------------------------------------------------------


def _score_vote_count(
    vc: int,
    release_date: str | None,
    today: datetime.date,
) -> float:
    """Log-scaled TMDB vote_count score in [0, 1] with age-based multipliers.

    Delegates to the shared score_vote_count() in scoring_utils with the TMDB
    log cap (2001).  See scoring_utils.score_vote_count for full documentation.
    """
    return score_vote_count(vc, release_date, today, VoteCountSource.TMDB)


def _score_watch_providers(
    provider_count: int,
    release_date: str | None,
    today: datetime.date,
) -> float:
    """Tiered watch-provider score in [-1, +1] with theater-window logic.

    For a US-only app, a movie that cannot be streamed in the US after its
    theatrical window has passed is nearly worthless as a recommendation.
    That case receives the maximum penalty of -1.0.

    Within the 75-day theater window the absence of streaming is expected
    (the film may still be in cinemas), so no penalty is applied (0.0).
    Movies available on 1–2 US platforms score +0.5; 3+ platforms score +1.0.
    The jump from 0 to ≥1 is the primary signal; wider availability gives
    only a small additional boost.

    Null release_date is treated conservatively as past the theater window
    since we cannot confirm the film is still in cinemas.
    """
    # Determine whether the theater window has elapsed.
    past_theater: bool = True
    if release_date is not None:
        try:
            release = datetime.date.fromisoformat(release_date)
            past_theater = (today - release).days > THEATER_WINDOW_DAYS
        except ValueError:
            # Non-parseable date — conservatively assume post-theater.
            pass

    if provider_count >= 3:
        return 1.0
    elif provider_count >= 1:
        return 0.5
    else:
        # No US streaming: harsh penalty if theatrical window has elapsed,
        # neutral if film may still be in cinemas.
        return -1.0 if past_theater else 0.0


def _score_popularity(popularity: float) -> float:
    """Log-scaled popularity score in [0, 1], capped at popularity=10.

    Delegates to the shared score_popularity() in scoring_utils.
    """
    return score_popularity(popularity)


def _score_overview_length(length: int) -> float:
    """Tiered overview-length score in [0.2, 1.0] based on character count.

    Overviews with fewer than 50 characters are effectively empty and will
    produce weak vector embeddings in downstream stages.  Full overviews
    (200+ chars) receive the maximum score.  No additional bonus is given
    beyond 200 characters — the marginal value of 400 chars vs 201 is minor.

    Note: length=0 is eliminated by the upstream hard filter (tmdb_filter.py)
    before any movie reaches this scorer.  The ≤50 branch handles the rare
    1–50 char stub overviews that pass the hard filter.
    """
    if length <= 50:
        return 0.2
    elif length <= 100:
        return 0.6
    elif length <= 200:
        return 0.85
    else:
        return 1.0


# ---------------------------------------------------------------------------
# Composite score
# ---------------------------------------------------------------------------


def compute_quality_score(row: sqlite3.Row, today: datetime.date) -> float:
    """Compute the raw weighted quality score for a single movie row.

    Returns the un-normalised weighted sum.  Scores can be negative (down to
    roughly -0.34) for movies with multiple significant data deficiencies.
    Normalisation across the full corpus is performed in a separate step after
    all movies have been scored.

    Args:
        row:   Named-column sqlite3.Row with the fields selected by run().
        today: Scoring reference date (computed once by the caller).

    Returns:
        Raw quality score as a float.
    """
    provider_count = len(unpack_provider_keys(row["watch_provider_keys"]))
    release_date: str | None = row["release_date"]

    vc_score  = _score_vote_count(row["vote_count"], release_date, today)
    wp_score  = _score_watch_providers(provider_count, release_date, today)
    pop_score = _score_popularity(row["popularity"])
    ol_score  = _score_overview_length(row["overview_length"])

    # Signed boolean signals.  Rarity determines which direction carries
    # the weight: rare fields (has_revenue, has_budget) are pure bonuses —
    # absence is expected and generates no penalty.  Common fields
    # (poster_url, has_production_companies, has_cast_and_crew) use the
    # inverse logic: presence is expected (score 0), absence is the signal
    # (score -1).  has_keywords sits in between with a symmetric ±0.5.
    rev_score  = 1.0  if row["has_revenue"]              else 0.0
    bud_score  = 1.0  if row["has_budget"]               else 0.0
    kw_score   = 0.5  if row["has_keywords"]             else -0.5
    post_score = 0.0  if row["poster_url"] is not None   else -1.0
    pc_score   = 0.0  if row["has_production_companies"] else -1.0
    cc_score   = 0.0  if row["has_cast_and_crew"]        else -1.0

    return (
        WEIGHTS["vote_count"]               * vc_score
        + WEIGHTS["watch_providers"]        * wp_score
        + WEIGHTS["popularity"]             * pop_score
        + WEIGHTS["has_revenue"]            * rev_score
        + WEIGHTS["poster_url"]             * post_score
        + WEIGHTS["overview_length"]        * ol_score
        + WEIGHTS["has_keywords"]           * kw_score
        + WEIGHTS["has_production_companies"] * pc_score
        + WEIGHTS["has_budget"]             * bud_score
        + WEIGHTS["has_cast_and_crew"]      * cc_score
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run() -> None:
    """Score all tmdb_fetched movies and write results to movie_progress.stage_3_quality_score.

    Processes every movie with status='tmdb_fetched', computes its raw quality
    score, and stores it in the stage_3_quality_score column.  The movie's status is
    NOT changed — that happens in the separate normalisation/cutoff pass.

    Progress is reported every LOG_EVERY rows.  A summary with score
    distribution statistics is printed on completion.
    """
    today: datetime.date = datetime.date.today()

    db = init_db()
    db.row_factory = sqlite3.Row

    # Fetch the total count up front for progress reporting, without
    # materialising all rows into memory.
    print("Stage 3b quality scorer: loading tmdb_fetched movies...")
    total: int = db.execute("""
        SELECT COUNT(*)
        FROM tmdb_data d
        JOIN movie_progress p ON d.tmdb_id = p.tmdb_id
        WHERE p.status = ?
    """, (MovieStatus.TMDB_FETCHED,)).fetchone()[0]

    print(f"  {total:,} movies to score (reference date = {today})")

    if total == 0:
        print("No tmdb_fetched movies found. Has Stage 3 (hard filter) completed?")
        db.close()
        return

    scored: int = 0
    pending_commit: int = 0
    score_sum: float = 0.0
    score_min: float = float("inf")
    score_max: float = float("-inf")

    # Iterate lazily via the cursor rather than fetchall() to avoid loading the
    # entire 287K-row result set into memory at once.
    cursor = db.execute("""
        SELECT
            d.tmdb_id,
            d.vote_count,
            d.popularity,
            d.release_date,
            d.poster_url,
            d.watch_provider_keys,
            d.overview_length,
            d.has_revenue,
            d.has_budget,
            d.has_production_companies,
            d.has_keywords,
            d.has_cast_and_crew
        FROM tmdb_data d
        JOIN movie_progress p ON d.tmdb_id = p.tmdb_id
        WHERE p.status = ?
    """, (MovieStatus.TMDB_FETCHED,))

    try:
        for i, row in enumerate(cursor):
            score = compute_quality_score(row, today)

            db.execute(
                """UPDATE movie_progress
                   SET stage_3_quality_score = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE tmdb_id = ?""",
                (score, row["tmdb_id"]),
            )

            scored += 1
            pending_commit += 1
            score_sum += score
            score_min = min(score_min, score)
            score_max = max(score_max, score)

            if pending_commit >= COMMIT_EVERY:
                db.commit()
                pending_commit = 0

            if (i + 1) % LOG_EVERY == 0:
                print(f"  Scored {i + 1:,}/{total:,}")

        # Flush any writes that haven't reached the COMMIT_EVERY threshold.
        db.commit()
    finally:
        # Guarantee the connection is released even if compute_quality_score
        # raises an unexpected exception, preventing a connection leak and
        # ensuring SQLite's WAL journal is properly closed.
        db.close()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    score_mean = score_sum / scored if scored > 0 else 0.0

    print(f"\nStage 3b quality scorer complete")
    print(f"  Movies scored: {scored:,}")
    print(f"  Score min:     {score_min:.4f}")
    print(f"  Score mean:    {score_mean:.4f}")
    print(f"  Score max:     {score_max:.4f}")
    print(
        f"\nNext step: normalise stage_3_quality_score to [0, 1] across all scored movies,"
        f" plot the distribution, and select a cutoff to advance to 'quality_passed'."
    )


if __name__ == "__main__":
    run()
