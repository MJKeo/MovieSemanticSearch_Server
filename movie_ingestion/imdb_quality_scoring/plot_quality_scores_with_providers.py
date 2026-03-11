"""
Gaussian-smoothed survival curve with first and second derivatives for
movies with a stage_5_quality_score that also have watch providers
(or were released within the last 75 days).

Identical analysis to plot_quality_scores.py, but restricted to the
subset of movies that would score positively on the watch_providers
signal — i.e., movies that are actually streamable/purchasable or
still within their theatrical window.

Usage:
    python -m movie_ingestion.imdb_quality_scoring.plot_quality_scores_with_providers
"""

from movie_ingestion.scoring_utils import THEATER_WINDOW_DAYS
from movie_ingestion.survival_curve_utils import (
    SurvivalCurveConfig,
    plot_survival_curve_with_derivatives,
)
from movie_ingestion.tracker import init_db


def _fetch_quality_scores() -> list[float]:
    """Return sorted stage_5 quality scores for movies with watch providers
    or released within the theater window."""
    db = init_db()
    try:
        scores = [
            row[0]
            for row in db.execute(
                """
                SELECT mp.stage_5_quality_score
                FROM movie_progress mp
                JOIN tmdb_data td ON td.tmdb_id = mp.tmdb_id
                WHERE mp.stage_5_quality_score IS NOT NULL
                  AND (
                      (td.watch_provider_keys IS NOT NULL
                       AND length(td.watch_provider_keys) > 0)
                      OR td.release_date >= date('now', ?)
                  )
                ORDER BY mp.stage_5_quality_score ASC
                """,
                (f"-{THEATER_WINDOW_DAYS} days",),
            )
        ]
    finally:
        db.close()
    return scores


if __name__ == "__main__":
    scores = _fetch_quality_scores()
    config = SurvivalCurveConfig(
        title="Stage 5 IMDB Quality Score (With Providers) — Survival Curve & Derivatives",
        stage_label="with_providers",
        output_filename="imdb_quality_score_with_providers_smoothed_derivative.png",
    )
    plot_survival_curve_with_derivatives(scores, config)
