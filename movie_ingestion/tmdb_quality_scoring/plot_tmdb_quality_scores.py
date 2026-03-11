"""
Gaussian-smoothed survival curves with first and second derivatives for
Stage 3 TMDB quality scores.

Produces two plots:
  1. All tmdb_quality_calculated movies (includes with-provider movies at 1.0)
  2. No-provider movies only (the population where the threshold matters)

Both plots include derivative analysis with survival counts at all points
of interest (local extrema and zero-crossings).

Usage:
    python -m movie_ingestion.tmdb_quality_scoring.plot_tmdb_quality_scores
"""

from movie_ingestion.scoring_utils import THEATER_WINDOW_DAYS
from movie_ingestion.survival_curve_utils import (
    SurvivalCurveConfig,
    plot_survival_curve_with_derivatives,
)
from movie_ingestion.tracker import MovieStatus, init_db


def _fetch_all_scores() -> list[float]:
    """Return sorted quality scores for all tmdb_quality_calculated movies."""
    db = init_db()
    try:
        scores = [
            row[0]
            for row in db.execute(
                """
                SELECT stage_3_quality_score
                FROM movie_progress
                WHERE status = ?
                  AND stage_3_quality_score IS NOT NULL
                ORDER BY stage_3_quality_score ASC
                """,
                (MovieStatus.TMDB_QUALITY_CALCULATED,),
            )
        ]
    finally:
        db.close()
    return scores


def _fetch_no_provider_scores() -> list[float]:
    """Return sorted quality scores for tmdb_quality_calculated movies
    that have no US watch providers and are outside the theater window.

    This isolates the formula-scored population — with-provider movies
    all receive 1.0 and are excluded here so they don't distort the
    survival curve used for threshold selection.
    """
    db = init_db()
    try:
        scores = [
            row[0]
            for row in db.execute(
                """
                SELECT mp.stage_3_quality_score
                FROM movie_progress mp
                JOIN tmdb_data td ON td.tmdb_id = mp.tmdb_id
                WHERE mp.status = ?
                  AND mp.stage_3_quality_score IS NOT NULL
                  AND (td.watch_provider_keys IS NULL
                       OR length(td.watch_provider_keys) = 0)
                  AND (td.release_date IS NULL
                       OR td.release_date < date('now', ?))
                ORDER BY mp.stage_3_quality_score ASC
                """,
                (MovieStatus.TMDB_QUALITY_CALCULATED, f"-{THEATER_WINDOW_DAYS} days"),
            )
        ]
    finally:
        db.close()
    return scores


if __name__ == "__main__":
    # --- Plot 1: All scored movies ---
    all_scores = _fetch_all_scores()
    all_config = SurvivalCurveConfig(
        title="Stage 3 TMDB Quality Score (All) — Survival Curve & Derivatives",
        stage_label="tmdb_quality_calculated",
        output_filename="tmdb_quality_score_all_smoothed_derivative.png",
    )
    plot_survival_curve_with_derivatives(all_scores, all_config)

    # --- Plot 2: No-provider movies only ---
    no_provider_scores = _fetch_no_provider_scores()
    no_provider_config = SurvivalCurveConfig(
        title="Stage 3 TMDB Quality Score (No Providers) — Survival Curve & Derivatives",
        stage_label="tmdb_quality_calculated_no_providers",
        output_filename="tmdb_quality_score_no_providers_smoothed_derivative.png",
    )
    plot_survival_curve_with_derivatives(no_provider_scores, no_provider_config)

    # Block here so both plot windows stay open until manually closed.
    import matplotlib.pyplot as plt
    plt.show()
