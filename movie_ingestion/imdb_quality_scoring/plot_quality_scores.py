"""
Gaussian-smoothed survival curves with first and second derivatives for
stage_5_quality_score, split into three non-overlapping groups:

  1. Movies WITH watch providers (any release date)
  2. Movies WITHOUT providers, released ≤ 75 days ago (theater window)
  3. Movies WITHOUT providers, released > 75 days ago

Each group gets its own plot saved to ingestion_data/.

Usage:
    python -m movie_ingestion.imdb_quality_scoring.plot_quality_scores
"""

import matplotlib.pyplot as plt

from movie_ingestion.scoring_utils import THEATER_WINDOW_DAYS
from movie_ingestion.survival_curve_utils import (
    SurvivalCurveConfig,
    plot_survival_curve_with_derivatives,
)
from movie_ingestion.tracker import init_db

# -- SQL fragments shared across queries ------------------------------------

_SCORE_NOT_NULL = "mp.stage_5_quality_score IS NOT NULL"

_HAS_PROVIDERS = (
    "td.watch_provider_keys IS NOT NULL AND length(td.watch_provider_keys) > 0"
)

_NO_PROVIDERS = (
    "(td.watch_provider_keys IS NULL OR length(td.watch_provider_keys) = 0)"
)

_THEATER_WINDOW_PARAM = f"-{THEATER_WINDOW_DAYS} days"


# -- Fetch helpers -----------------------------------------------------------


def _fetch_scores(where_clause: str, params: tuple = ()) -> list[float]:
    """Run a scored-movie query with the given WHERE extras and return
    sorted quality scores."""
    db = init_db()
    try:
        scores = [
            row[0]
            for row in db.execute(
                f"""
                SELECT mp.stage_5_quality_score
                FROM movie_progress mp
                JOIN tmdb_data td ON td.tmdb_id = mp.tmdb_id
                WHERE {_SCORE_NOT_NULL}
                  AND {where_clause}
                ORDER BY mp.stage_5_quality_score ASC
                """,
                params,
            )
        ]
    finally:
        db.close()
    return scores


def _fetch_with_providers() -> list[float]:
    """Movies that have at least one watch provider."""
    return _fetch_scores(_HAS_PROVIDERS)


def _fetch_no_providers_recent() -> list[float]:
    """Movies without providers, released within the theater window."""
    return _fetch_scores(
        f"{_NO_PROVIDERS} AND td.release_date >= date('now', ?)",
        (_THEATER_WINDOW_PARAM,),
    )


def _fetch_no_providers_old() -> list[float]:
    """Movies without providers, released outside the theater window
    (or with no release date recorded)."""
    return _fetch_scores(
        f"{_NO_PROVIDERS} AND (td.release_date IS NULL OR td.release_date < date('now', ?))",
        (_THEATER_WINDOW_PARAM,),
    )


# -- Group definitions -------------------------------------------------------

_GROUPS: list[tuple[callable, SurvivalCurveConfig]] = [
    (
        _fetch_with_providers,
        SurvivalCurveConfig(
            title="Stage 5 IMDB Quality Score (With Providers) — Survival Curve & Derivatives",
            stage_label="with_providers",
            output_filename="imdb_quality_score_with_providers.png",
        ),
    ),
    (
        _fetch_no_providers_recent,
        SurvivalCurveConfig(
            title=(
                f"Stage 5 IMDB Quality Score (No Providers, ≤{THEATER_WINDOW_DAYS} Days)"
                " — Survival Curve & Derivatives"
            ),
            stage_label="no_providers_recent",
            output_filename="imdb_quality_score_no_providers_recent.png",
        ),
    ),
    (
        _fetch_no_providers_old,
        SurvivalCurveConfig(
            title=(
                f"Stage 5 IMDB Quality Score (No Providers, >{THEATER_WINDOW_DAYS} Days)"
                " — Survival Curve & Derivatives"
            ),
            stage_label="no_providers_old",
            output_filename="imdb_quality_score_no_providers_old.png",
        ),
    ),
]


if __name__ == "__main__":
    for fetch_fn, config in _GROUPS:
        scores = fetch_fn()
        plot_survival_curve_with_derivatives(scores, config)

    # Keep plot windows open if running interactively.
    plt.show()
