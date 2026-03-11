"""
Unit tests for movie_ingestion.survival_curve_utils — shared plotting utilities.

Covers:
  - normalize_to_01          — min-max normalization
  - find_zero_crossings      — linear-interpolated zero crossings
  - build_survival_curve     — score → survival count conversion
  - plot_survival_curve_with_derivatives — file creation and guard clauses
"""

from pathlib import Path

import matplotlib
import numpy as np
import pytest

# Force non-interactive backend before any pyplot imports to avoid display issues.
matplotlib.use("Agg")

from movie_ingestion.survival_curve_utils import (
    SurvivalCurveConfig,
    build_survival_curve,
    find_zero_crossings,
    normalize_to_01,
    plot_survival_curve_with_derivatives,
)


# ---------------------------------------------------------------------------
# normalize_to_01
# ---------------------------------------------------------------------------


class TestNormalizeTo01:
    """Tests for min-max normalization to [0, 1]."""

    def test_normalizes_to_zero_one_range(self) -> None:
        """Output min=0.0, max=1.0."""
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        result = normalize_to_01(arr)
        assert result.min() == pytest.approx(0.0)
        assert result.max() == pytest.approx(1.0)

    def test_constant_array_returns_zeros(self) -> None:
        """All-same values → zeros (span=0 guard)."""
        arr = np.array([5.0, 5.0, 5.0])
        result = normalize_to_01(arr)
        np.testing.assert_array_equal(result, np.zeros(3))

    def test_preserves_relative_ordering(self) -> None:
        """argmax/argmin positions unchanged."""
        arr = np.array([3.0, 1.0, 4.0, 1.5, 9.0])
        result = normalize_to_01(arr)
        assert np.argmin(result) == np.argmin(arr)
        assert np.argmax(result) == np.argmax(arr)

    def test_two_element_array(self) -> None:
        """[3, 7] → [0.0, 1.0]."""
        arr = np.array([3.0, 7.0])
        result = normalize_to_01(arr)
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_negative_values(self) -> None:
        """[-5, 0, 5] → [0.0, 0.5, 1.0]."""
        arr = np.array([-5.0, 0.0, 5.0])
        result = normalize_to_01(arr)
        np.testing.assert_allclose(result, [0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# find_zero_crossings
# ---------------------------------------------------------------------------


class TestFindZeroCrossings:
    """Tests for zero-crossing detection via linear interpolation."""

    def test_single_crossing(self) -> None:
        """y=[-1, 1] at x=[0, 2] → crossing at x=1.0."""
        x = np.array([0.0, 2.0])
        y = np.array([-1.0, 1.0])
        zeros = find_zero_crossings(x, y)
        assert len(zeros) == 1
        assert zeros[0] == pytest.approx(1.0)

    def test_no_crossings(self) -> None:
        """y=[1, 2, 3] → empty array."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        zeros = find_zero_crossings(x, y)
        assert len(zeros) == 0

    def test_multiple_crossings(self) -> None:
        """y=[-1, 1, -1] → two crossings."""
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([-1.0, 1.0, -1.0])
        zeros = find_zero_crossings(x, y)
        assert len(zeros) == 2

    def test_crossing_at_midpoint(self) -> None:
        """Symmetric y around zero → crossing at exact midpoint."""
        x = np.array([0.0, 10.0])
        y = np.array([-5.0, 5.0])
        zeros = find_zero_crossings(x, y)
        assert len(zeros) == 1
        assert zeros[0] == pytest.approx(5.0)

    def test_empty_arrays(self) -> None:
        """Empty x, y → empty result."""
        x = np.array([])
        y = np.array([])
        zeros = find_zero_crossings(x, y)
        assert len(zeros) == 0

    def test_zero_dy_uses_midpoint(self) -> None:
        """y=[0, 0] change (degenerate case with sign change from positive to zero).

        When y transitions through zero with dy≈0, the midpoint fallback is used.
        We test a case where sign changes but dy is effectively zero.
        """
        # sign(-eps) = -1, sign(+eps) = +1 → sign change detected
        # dy = 2*eps ≈ 0 → below threshold → midpoint fallback
        eps = 1e-35
        x = np.array([0.0, 10.0])
        y = np.array([-eps, eps])
        zeros = find_zero_crossings(x, y)
        assert len(zeros) == 1
        assert zeros[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# build_survival_curve
# ---------------------------------------------------------------------------


class TestBuildSurvivalCurve:
    """Tests for survival curve construction from sorted scores."""

    def test_simple_curve(self) -> None:
        """[1, 2, 3] → counts_gte=[3, 2, 1]."""
        unique, counts, total = build_survival_curve([1, 2, 3])
        np.testing.assert_array_equal(unique, [1, 2, 3])
        np.testing.assert_array_equal(counts, [3, 2, 1])
        assert total == 3

    def test_duplicate_scores(self) -> None:
        """[1, 1, 2] → unique=[1, 2], counts=[3, 1]."""
        unique, counts, total = build_survival_curve([1, 1, 2])
        np.testing.assert_array_equal(unique, [1, 2])
        np.testing.assert_array_equal(counts, [3, 1])
        assert total == 3

    def test_single_score(self) -> None:
        """[5] → unique=[5], counts=[1], total=1."""
        unique, counts, total = build_survival_curve([5])
        np.testing.assert_array_equal(unique, [5])
        np.testing.assert_array_equal(counts, [1])
        assert total == 1

    def test_total_equals_input_length(self) -> None:
        """total matches len(scores)."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        _, _, total = build_survival_curve(scores)
        assert total == len(scores)

    def test_first_count_equals_total(self) -> None:
        """counts_gte[0] == total (all movies survive lowest threshold)."""
        scores = [0.1, 0.3, 0.5, 0.7]
        _, counts, total = build_survival_curve(scores)
        assert counts[0] == total


# ---------------------------------------------------------------------------
# plot_survival_curve_with_derivatives
# ---------------------------------------------------------------------------


class TestPlotSurvivalCurveWithDerivatives:
    """Tests for the plotting function's side effects and guard clauses."""

    @pytest.fixture()
    def config(self) -> SurvivalCurveConfig:
        """Minimal config for testing."""
        return SurvivalCurveConfig(
            title="Test Plot",
            stage_label="test",
            output_filename="test_plot.png",
            resample_points=100,
            gaussian_sigma=5,
            extrema_order=3,
        )

    def test_creates_output_file(self, tmp_path, config) -> None:
        """PNG file exists at output_dir/config.output_filename after call."""
        # Generate enough scores to produce meaningful derivatives.
        scores = sorted([i * 0.01 for i in range(200)])
        plot_survival_curve_with_derivatives(scores, config, output_dir=tmp_path)
        assert (tmp_path / config.output_filename).exists()

    def test_empty_scores_prints_message(self, tmp_path, config, capsys) -> None:
        """Empty list → early return, no file created, no error."""
        plot_survival_curve_with_derivatives([], config, output_dir=tmp_path)
        assert not (tmp_path / config.output_filename).exists()
        captured = capsys.readouterr()
        assert "No quality scores found" in captured.out

    def test_uses_default_output_dir(self, config, monkeypatch, tmp_path) -> None:
        """output_dir=None → saves under INGESTION_DATA_DIR."""
        # Monkeypatch INGESTION_DATA_DIR to a temp path to avoid real filesystem writes.
        monkeypatch.setattr(
            "movie_ingestion.survival_curve_utils.INGESTION_DATA_DIR", tmp_path
        )
        scores = sorted([i * 0.01 for i in range(200)])
        plot_survival_curve_with_derivatives(scores, config, output_dir=None)
        assert (tmp_path / config.output_filename).exists()
