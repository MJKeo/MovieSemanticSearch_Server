"""
Shared utilities for plotting Gaussian-smoothed survival curves with
first and second derivatives.

Provides reusable math helpers (normalization, zero-crossing detection,
survival curve construction) and a parameterized plotting function that
both the TMDB and IMDB quality scoring modules call with stage-specific
configuration.

Used by:
    movie_ingestion.tmdb_quality_scoring.plot_tmdb_quality_scores
    movie_ingestion.imdb_quality_scoring.plot_quality_scores
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelextrema

from movie_ingestion.tracker import INGESTION_DATA_DIR


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------


def normalize_to_01(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize an array to [0, 1]."""
    lo, hi = arr.min(), arr.max()
    span = hi - lo
    if span == 0:
        return np.zeros_like(arr)
    return (arr - lo) / span


def find_zero_crossings(x: np.ndarray, y: np.ndarray) -> np.ndarray:
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


def build_survival_curve(
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
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SurvivalCurveConfig:
    """Caller-specific parameters for the survival curve plot."""

    # Display / output
    title: str              # Plot title
    stage_label: str        # Used in console diagnostics (e.g., "tmdb_fetched")
    output_filename: str    # PNG filename saved under output_dir

    # Tuning — defaults match the original TMDB analysis
    resample_points: int = 2_000    # Uniform grid resolution
    gaussian_sigma: int = 30       # Smoothing width in grid points
    extrema_order: int = 5         # Neighborhood size for local extrema


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _print_extrema(
    label: str,
    idxs: np.ndarray,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    x_grid: np.ndarray,
    counts_smooth: np.ndarray,
) -> None:
    """Print local extrema with their quality scores, normalized values, and survival counts."""
    if len(idxs) == 0:
        print(f"\n  {label}: none")
        return
    # Interpolate the survival count at each extremum's score value.
    survival_counts = np.interp(x_arr[idxs], x_grid, counts_smooth)
    print(f"\n  {label} ({len(idxs)}):")
    for i, count in zip(idxs, survival_counts):
        print(
            f"    score: {x_arr[i]:.4f}  |  normalized value: {y_arr[i]:.4f}"
            f"  |  survival count: {count:,.0f}"
        )


def _print_zero_crossings(
    label: str,
    zeros: np.ndarray,
    x_grid: np.ndarray,
    counts_smooth: np.ndarray,
) -> None:
    """Print zero-crossings with their quality scores and survival counts."""
    if len(zeros) == 0:
        print(f"\n  {label}: none")
        return
    counts_at = np.interp(zeros, x_grid, counts_smooth)
    print(f"\n  {label} ({len(zeros)}):")
    for score, count in zip(zeros, counts_at):
        print(f"    score: {score:.4f}  |  survival count: {count:,.0f}")


def _scatter_points(
    ax: plt.Axes,
    x_pts: np.ndarray,
    x_grid: np.ndarray,
    counts_smooth: np.ndarray,
    *,
    color: str,
    label: str,
    marker: str = "o",
    edgecolors: str | None = None,
    size: int = 80,
) -> plt.Artist | None:
    """
    Plot points of interest on the survival curve and annotate each with
    its quality score value. Returns the scatter handle for the legend,
    or None if there are no points.
    """
    if len(x_pts) == 0:
        return None

    y_pts = np.interp(x_pts, x_grid, counts_smooth)

    scatter_kwargs: dict = dict(
        color=color, s=size, zorder=5, marker=marker, label=label,
    )
    if edgecolors:
        scatter_kwargs["edgecolors"] = edgecolors
        scatter_kwargs["linewidths"] = 0.6

    handle = ax.scatter(x_pts, y_pts, **scatter_kwargs)

    # Annotate each dot with its score value. Alternate above/below to
    # reduce overlap when points cluster together.
    for idx, (x, y) in enumerate(zip(x_pts, y_pts)):
        offset_y = 12 if idx % 2 == 0 else -14
        ax.annotate(
            f"{x:.4f}",
            xy=(x, y),
            textcoords="offset points",
            xytext=(0, offset_y),
            fontsize=7,
            ha="center",
            color=color if color not in ("yellow",) else "goldenrod",
        )

    return handle


def plot_survival_curve_with_derivatives(
    scores: list[float],
    config: SurvivalCurveConfig,
    output_dir: Path | None = None,
) -> None:
    """
    Plot the Gaussian-smoothed survival curve (left y-axis) alongside its
    first and second derivatives (right y-axis, both normalized to [0, 1]).

    Marks all points of interest on the survival curve:
      - Local maxima and minima of f'(x) and f''(x) as circles
      - Zero-crossings of f'(x) and f''(x) as diamonds

    All points are annotated with their quality score value and printed
    to the console with survival counts.
    """
    if output_dir is None:
        output_dir = INGESTION_DATA_DIR

    total = len(scores)
    if total == 0:
        print(f"No quality scores found for {config.stage_label} movies.")
        return

    # --- Build survival curve and resample onto uniform grid ---

    x_orig, counts_orig, _ = build_survival_curve(scores)

    x_grid = np.linspace(scores[0], scores[-1], config.resample_points)
    counts_grid = np.interp(x_grid, x_orig, counts_orig.astype(float))

    # --- Gaussian smoothing ---

    counts_smooth = gaussian_filter1d(counts_grid, sigma=config.gaussian_sigma)

    # --- First and second derivatives of the smoothed curve ---
    # Uniform grid spacing means dx is constant across the entire grid.

    dx = x_grid[1] - x_grid[0]
    f1 = np.diff(counts_smooth) / dx
    # Derivative x-values sit at the midpoints between adjacent grid points.
    x1 = (x_grid[:-1] + x_grid[1:]) / 2

    f2 = np.diff(f1) / dx
    x2 = (x1[:-1] + x1[1:]) / 2

    # Normalize both derivatives to [0, 1] so they can share the right axis.
    f1_norm = normalize_to_01(f1)
    f2_norm = normalize_to_01(f2)

    # --- Print diagnostics ---

    print(f"\nSmoothed survival + derivatives ({total:,} {config.stage_label} movies)")
    print(f"  Raw score range: [{scores[0]:.4f}, {scores[-1]:.4f}]")
    print(f"  Resample grid:   {config.resample_points} points")
    print(f"  Gaussian sigma:  {config.gaussian_sigma}")
    print(f"  f'  range: [{f1.min():.1f}, {f1.max():.1f}] movies/score-unit")
    print(f"  f'' range: [{f2.min():.1f}, {f2.max():.1f}] movies/score-unit²")

    # --- Local extrema of the normalized derivative curves ---

    f1_max_idxs = argrelextrema(f1_norm, np.greater, order=config.extrema_order)[0]
    f1_min_idxs = argrelextrema(f1_norm, np.less, order=config.extrema_order)[0]
    f2_max_idxs = argrelextrema(f2_norm, np.greater, order=config.extrema_order)[0]
    f2_min_idxs = argrelextrema(f2_norm, np.less, order=config.extrema_order)[0]

    _print_extrema("f'(x) local maxima", f1_max_idxs, x1, f1_norm, x_grid, counts_smooth)
    _print_extrema("f'(x) local minima", f1_min_idxs, x1, f1_norm, x_grid, counts_smooth)
    _print_extrema("f''(x) local maxima", f2_max_idxs, x2, f2_norm, x_grid, counts_smooth)
    _print_extrema("f''(x) local minima", f2_min_idxs, x2, f2_norm, x_grid, counts_smooth)

    # --- Zero-crossings of the raw (unnormalized) derivatives ---
    # Normalized derivatives shift the zero point, so crossings must be
    # computed on the raw arrays to be physically meaningful.

    f1_zeros = find_zero_crossings(x1, f1)
    f2_zeros = find_zero_crossings(x2, f2)

    _print_zero_crossings("f'(x) zero-crossings", f1_zeros, x_grid, counts_smooth)
    _print_zero_crossings("f''(x) zero-crossings", f2_zeros, x_grid, counts_smooth)

    # --- Plot with twin y-axes ---

    fig, ax_left = plt.subplots(figsize=(14, 7))
    ax_right = ax_left.twinx()

    # Left axis: Gaussian-smoothed survival curve.
    line_surv, = ax_left.plot(
        x_grid, counts_smooth,
        color="tab:blue", linewidth=1.8,
        label=f"Smoothed survival (σ={config.gaussian_sigma})",
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

    # --- Mark points of interest on the survival curve ---

    dot_handles = []

    # Extrema — circles
    h = _scatter_points(
        ax_left, x1[f1_max_idxs], x_grid, counts_smooth,
        color="purple", label="f' local max",
    )
    if h:
        dot_handles.append(h)

    h = _scatter_points(
        ax_left, x1[f1_min_idxs], x_grid, counts_smooth,
        color="green", label="f' local min",
    )
    if h:
        dot_handles.append(h)

    h = _scatter_points(
        ax_left, x2[f2_max_idxs], x_grid, counts_smooth,
        color="yellow", label="f'' local max",
        edgecolors="black",
    )
    if h:
        dot_handles.append(h)

    h = _scatter_points(
        ax_left, x2[f2_min_idxs], x_grid, counts_smooth,
        color="red", label="f'' local min",
    )
    if h:
        dot_handles.append(h)

    # Zero-crossings — diamonds
    h = _scatter_points(
        ax_left, f1_zeros, x_grid, counts_smooth,
        color="cyan", label="f' zero-crossing",
        marker="D", edgecolors="black",
    )
    if h:
        dot_handles.append(h)

    h = _scatter_points(
        ax_left, f2_zeros, x_grid, counts_smooth,
        color="magenta", label="f'' zero-crossing",
        marker="D", edgecolors="black",
    )
    if h:
        dot_handles.append(h)

    # Combined legend from both axes.
    ax_left.legend(
        handles=[line_surv, line_f1, line_f2] + dot_handles,
        loc="center right", fontsize=9,
    )

    ax_left.set_title(config.title)
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

    output_path = output_dir / config.output_filename
    fig.savefig(output_path, dpi=150)
    print(f"\nSaved plot to {output_path}")

    # Show the plot non-blocking so callers that produce multiple plots
    # can continue execution.  The caller is responsible for a final
    # blocking plt.show() if it wants windows to stay open.
    if plt.get_backend().lower() not in ("agg",):
        plt.show(block=False)
    else:
        plt.close(fig)
