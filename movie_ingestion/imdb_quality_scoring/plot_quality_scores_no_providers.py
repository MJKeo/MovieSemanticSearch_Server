"""
Gaussian-smoothed survival curve with first and second derivatives for
movies with a stage_5_quality_score that have NO watch providers and
were released more than 75 days ago.

This is the inverse of plot_quality_scores_with_providers.py — it
isolates movies that scored negatively on the watch_providers signal,
helping assess whether any are still worth including based on other
data quality signals.

Usage:
    python -m movie_ingestion.imdb_quality_scoring.plot_quality_scores_no_providers
"""

from movie_ingestion.scoring_utils import THEATER_WINDOW_DAYS
from movie_ingestion.survival_curve_utils import (
    SurvivalCurveConfig,
    plot_survival_curve_with_derivatives,
)
from movie_ingestion.tracker import init_db


def _fetch_quality_scores() -> list[float]:
    """Return sorted stage_5 quality scores for movies without watch
    providers that are also outside the theater window."""
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
                  AND (td.watch_provider_keys IS NULL
                       OR length(td.watch_provider_keys) = 0)
                  AND (td.release_date IS NULL
                       OR td.release_date < date('now', ?))
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
        title="Stage 5 IMDB Quality Score (No Providers) — Survival Curve & Derivatives",
        stage_label="no_providers",
        output_filename="imdb_quality_score_no_providers_smoothed_derivative.png",
    )
    plot_survival_curve_with_derivatives(scores, config)
