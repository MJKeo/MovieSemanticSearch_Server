"""
Gaussian-smoothed survival curve with first and second derivatives for
tmdb_fetched movies.

Plots three curves on a single graph with twin y-axes:
  1. Left axis  — Gaussian-smoothed survival curve (raw movie counts)
  2. Right axis — First derivative f'(x) normalized to [0, 1]
  3. Right axis — Second derivative f''(x) normalized to [0, 1]

The survival curve is resampled onto a uniform grid before smoothing and
differentiation, since the raw curve has irregular x-spacing.

Usage:
    python -m movie_ingestion.plot_quality_scores
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

from movie_ingestion.tracker import TRACKER_DB_PATH, init_db


def _normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    span = hi - lo
    if span == 0:
        return np.zeros_like(arr)
    return (arr - lo) / span


def _find_zero_crossings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Find the x-values where y crosses zero via linear interpolation.

    For each pair of adjacent points where y changes sign, interpolate
    linearly to find the precise x at which y = 0.
    Returns an array of interpolated x-values (may be empty).
    """
    signs = np.sign(y)
    sign_changes = np.where(np.diff(signs) != 0)[0]

    if len(sign_changes) == 0:
        return np.array([])

    x_lo, x_hi = x[sign_changes], x[sign_changes + 1]
    y_lo, y_hi = y[sign_changes], y[sign_changes + 1]
    dy = y_hi - y_lo

    # Guard against division by zero — use midpoint as fallback.
    safe = np.abs(dy) > 1e-30
    zeros = np.where(
        safe,
        x_lo - y_lo * (x_hi - x_lo) / np.where(safe, dy, 1.0),
        (x_lo + x_hi) / 2,
    )
    return zeros


def _fetch_quality_scores() -> list[float]:
    """Return sorted quality scores for all tmdb_fetched movies."""
    db = init_db()
    try:
        # Single-column query — index access is cleaner and avoids the
        # overhead of sqlite3.Row wrapper objects for 287K rows.
        scores = [
            row[0]
            for row in db.execute(
                """
                SELECT quality_score
                FROM movie_progress
                WHERE status = 'tmdb_fetched'
                  AND quality_score IS NOT NULL
                ORDER BY quality_score ASC
                """
            )
        ]
    finally:
        db.close()
    return scores


def _build_survival_curve(
    scores: list[float],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Compute the survival curve from a sorted list of scores.

    Returns:
        (unique_scores, counts_gte, total) where counts_gte[i] is the
        number of movies with score >= unique_scores[i].
    """
    total = len(scores)
    arr = np.array(scores)
    unique_scores = np.unique(arr)
    # For each score x, the number of movies with score >= x is
    # total - (index of x in the sorted array). searchsorted gives the
    # insertion point, which equals the count of values strictly less than x.
    counts_gte = total - np.searchsorted(arr, unique_scores, side="left")
    return unique_scores, counts_gte, total


# ---------------------------------------------------------------------------
# Tuning parameters
# ---------------------------------------------------------------------------

# Number of evenly spaced points for the resampled grid. Smoothing and
# differentiation require uniform spacing; the raw survival curve does not
# have it.
RESAMPLE_POINTS = 2_000

# Gaussian filter width in grid points. sigma=30 on a 2000-point grid means
# each output point averages roughly ±3% of the total x-range.
GAUSSIAN_SIGMA = 30

# Neighborhood size for local extrema detection (scipy argrelextrema order).
# A point must be greater/less than its `order` neighbors on each side to
# qualify. order=5 on a 2000-point grid ≈ 0.5% neighborhood, filtering
# out micro-noise while preserving meaningful peaks/troughs.
EXTREMA_ORDER = 5


def plot_smoothed_with_derivatives(scores: list[float]) -> None:
    """
    Plot the Gaussian-smoothed survival curve (left y-axis) alongside its
    first and second derivatives (right y-axis, both normalized to [0, 1]).

    The left axis shows raw movie counts. The right axis shows both
    derivatives min-max normalized so they share a common [0, 1] scale.
    """
    total = len(scores)
    if total == 0:
        print("No quality scores found for tmdb_fetched movies.")
        return

    # --- Build survival curve and resample onto uniform grid ---

    x_orig, counts_orig, _ = _build_survival_curve(scores)

    x_grid = np.linspace(scores[0], scores[-1], RESAMPLE_POINTS)
    counts_grid = np.interp(x_grid, x_orig, counts_orig.astype(float))

    # --- Gaussian smoothing ---

    counts_smooth = gaussian_filter1d(counts_grid, sigma=GAUSSIAN_SIGMA)

    # --- First and second derivatives of the smoothed curve ---
    # Uniform grid spacing means dx is constant across the entire grid.

    dx = x_grid[1] - x_grid[0]
    f1 = np.diff(counts_smooth) / dx
    # Derivative x-values sit at the midpoints between adjacent grid points.
    x1 = (x_grid[:-1] + x_grid[1:]) / 2

    f2 = np.diff(f1) / dx
    x2 = (x1[:-1] + x1[1:]) / 2

    # Normalize both derivatives to [0, 1] so they can share the right axis.
    f1_norm = _normalize_to_01(f1)
    f2_norm = _normalize_to_01(f2)

    # --- Print diagnostics ---

    print(f"\nSmoothed survival + derivatives ({total:,} tmdb_fetched movies)")
    print(f"  Raw score range: [{scores[0]:.4f}, {scores[-1]:.4f}]")
    print(f"  Resample grid:   {RESAMPLE_POINTS} points")
    print(f"  Gaussian sigma:  {GAUSSIAN_SIGMA}")
    print(f"  f'  range: [{f1.min():.1f}, {f1.max():.1f}] movies/score-unit")
    print(f"  f'' range: [{f2.min():.1f}, {f2.max():.1f}] movies/score-unit²")

    # --- Local extrema of the normalized derivative curves ---

    f1_max_idxs = argrelextrema(f1_norm, np.greater, order=EXTREMA_ORDER)[0]
    f1_min_idxs = argrelextrema(f1_norm, np.less, order=EXTREMA_ORDER)[0]
    f2_max_idxs = argrelextrema(f2_norm, np.greater, order=EXTREMA_ORDER)[0]
    f2_min_idxs = argrelextrema(f2_norm, np.less, order=EXTREMA_ORDER)[0]

    def _print_extrema(label: str, idxs: np.ndarray, x_arr: np.ndarray,
                       y_arr: np.ndarray) -> None:
        """Print local extrema with their quality scores and normalized values."""
        if len(idxs) == 0:
            print(f"\n  {label}: none")
            return
        print(f"\n  {label} ({len(idxs)}):")
        for i in idxs:
            print(f"    score: {x_arr[i]:.4f}  |  normalized value: {y_arr[i]:.4f}")

    _print_extrema("f'(x) local maxima", f1_max_idxs, x1, f1_norm)
    _print_extrema("f'(x) local minima", f1_min_idxs, x1, f1_norm)
    _print_extrema("f''(x) local maxima", f2_max_idxs, x2, f2_norm)
    _print_extrema("f''(x) local minima", f2_min_idxs, x2, f2_norm)

    # --- Zero-crossings of the raw (unnormalized) derivatives ---
    # Normalized derivatives shift the zero point, so crossings must be
    # computed on the raw arrays to be physically meaningful.

    f1_zeros = _find_zero_crossings(x1, f1)
    f2_zeros = _find_zero_crossings(x2, f2)

    def _print_zero_crossings(label: str, zeros: np.ndarray) -> None:
        """Print zero-crossings with their quality scores and survival counts."""
        if len(zeros) == 0:
            print(f"\n  {label}: none")
            return
        # Look up the survival curve count at each crossing point.
        counts_at = np.interp(zeros, x_grid, counts_smooth)
        print(f"\n  {label} ({len(zeros)}):")
        for score, count in zip(zeros, counts_at):
            print(f"    score: {score:.4f}  |  survival count: {count:,.0f}")

    _print_zero_crossings("f'(x) zero-crossings", f1_zeros)
    _print_zero_crossings("f''(x) zero-crossings", f2_zeros)

    # --- Plot with twin y-axes ---

    fig, ax_left = plt.subplots(figsize=(14, 7))
    ax_right = ax_left.twinx()

    # Left axis: Gaussian-smoothed survival curve.
    line_surv, = ax_left.plot(
        x_grid, counts_smooth,
        color="tab:blue", linewidth=1.8,
        label=f"Smoothed survival (σ={GAUSSIAN_SIGMA})",
    )
    ax_left.set_xlabel("Quality Score")
    ax_left.set_ylabel("Movies with Score ≥ x", color="tab:blue")
    ax_left.tick_params(axis="y", labelcolor="tab:blue")

    # Right axis: first and second derivatives (both normalized to [0, 1]).
    line_f1, = ax_right.plot(
        x1, f1_norm,
        color="tab:orange", linewidth=1.4,
        label="f'(x) — First derivative",
    )
    line_f2, = ax_right.plot(
        x2, f2_norm,
        color="tab:green", linewidth=1.4,
        label="f''(x) — Second derivative",
    )
    ax_right.set_ylabel("Normalized Derivative [0, 1]")
    ax_right.tick_params(axis="y")

    # --- Mark extrema as dots on the survival curve ---
    # Map derivative x-values (midpoints) back onto the smoothed survival
    # curve via interpolation, since x1/x2 don't exactly align with x_grid.

    dot_handles = []

    if len(f1_min_idxs) > 0:
        x_pts = x1[f1_min_idxs]
        y_pts = np.interp(x_pts, x_grid, counts_smooth)
        h = ax_left.scatter(x_pts, y_pts, color="green", s=80, zorder=5,
                            label="f' local min")
        dot_handles.append(h)

    if len(f1_max_idxs) > 0:
        x_pts = x1[f1_max_idxs]
        y_pts = np.interp(x_pts, x_grid, counts_smooth)
        h = ax_left.scatter(x_pts, y_pts, color="purple", s=80, zorder=5,
                            label="f' local max")
        dot_handles.append(h)

    if len(f2_min_idxs) > 0:
        x_pts = x2[f2_min_idxs]
        y_pts = np.interp(x_pts, x_grid, counts_smooth)
        h = ax_left.scatter(x_pts, y_pts, color="red", s=80, zorder=5,
                            label="f'' local min")
        dot_handles.append(h)

    if len(f2_max_idxs) > 0:
        x_pts = x2[f2_max_idxs]
        y_pts = np.interp(x_pts, x_grid, counts_smooth)
        h = ax_left.scatter(x_pts, y_pts, color="yellow", s=80, zorder=5,
                            edgecolors="black", linewidths=0.6,
                            label="f'' local max")
        dot_handles.append(h)

    # Combined legend from both axes.
    ax_left.legend(
        handles=[line_surv, line_f1, line_f2] + dot_handles,
        loc="center right", fontsize=9,
    )

    ax_left.set_title("Gaussian-Smoothed Survival Curve & Derivatives")
    ax_left.grid(True, alpha=0.3)

    ax_left.annotate(
        f"Total: {total:,}",
        xy=(0.97, 0.55),
        xycoords="axes fraction",
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8),
    )

    fig.tight_layout()

    output_path = TRACKER_DB_PATH.parent / "quality_score_smoothed_derivative.png"
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")

    plt.show()


if __name__ == "__main__":
    scores = _fetch_quality_scores()
    plot_smoothed_with_derivatives(scores)
