"""
Gaussian-smoothed survival curve with first and second derivatives for
movies with a stage_5_quality_score (post-IMDB quality scoring).

Plots the smoothed survival curve, its derivatives, and marks all points
of interest (local extrema and zero-crossings) on the curve. Prints
these critical points to the console for threshold analysis.

Usage:
    python -m movie_ingestion.imdb_quality_scoring.plot_quality_scores
"""

from movie_ingestion.survival_curve_utils import (
    SurvivalCurveConfig,
    plot_survival_curve_with_derivatives,
)
from movie_ingestion.tracker import init_db


def _fetch_quality_scores() -> list[float]:
    """Return sorted stage_5 quality scores for all scored movies."""
    db = init_db()
    try:
        scores = [
            row[0]
            for row in db.execute(
                """
                SELECT stage_5_quality_score
                FROM movie_progress
                WHERE stage_5_quality_score IS NOT NULL
                ORDER BY stage_5_quality_score ASC
                """,
            )
        ]
    finally:
        db.close()
    return scores


if __name__ == "__main__":
    scores = _fetch_quality_scores()
    config = SurvivalCurveConfig(
        title="Stage 5 IMDB Quality Score — Survival Curve & Derivatives",
        stage_label="essential_data_passed",
        output_filename="imdb_quality_score_smoothed_derivative.png",
    )
    plot_survival_curve_with_derivatives(scores, config)
