"""
Unit tests for movie_ingestion.tmdb_quality_scoring.tmdb_data_analysis — Stage 2.5 diagnostics.

Covers:
  - _select          — element extraction by index list
  - _build_analysis  — full analysis suite orchestration and output shape
"""

import pytest

from movie_ingestion.tmdb_quality_scoring.tmdb_data_analysis import (
    _build_analysis,
    _select,
)


# ---------------------------------------------------------------------------
# _select
# ---------------------------------------------------------------------------


class TestSelect:
    """Tests for the index-based element extraction helper."""

    def test_selects_correct_elements(self) -> None:
        """_select([a, b, c, d], [0, 2]) → [a, c]."""
        assert _select(["a", "b", "c", "d"], [0, 2]) == ["a", "c"]

    def test_empty_indices_returns_empty(self) -> None:
        """_select([a, b], []) → []."""
        assert _select(["a", "b"], []) == []

    def test_preserves_order_of_indices(self) -> None:
        """_select([a, b, c], [2, 0]) → [c, a]."""
        assert _select(["a", "b", "c"], [2, 0]) == ["c", "a"]


# ---------------------------------------------------------------------------
# _build_analysis
# ---------------------------------------------------------------------------


def _make_analysis_inputs(n: int = 10) -> dict:
    """Build minimal synthetic column lists for _build_analysis.

    Returns a dict of keyword arguments matching _build_analysis's signature.
    All lists have length n with reasonable values.
    """
    return {
        "titles":                   [f"Movie {i}" for i in range(n)],
        "release_dates":            ["2020-01-15"] * n,
        "durations":                [120] * n,
        "poster_urls":              ["https://example.com/p.jpg"] * n,
        "provider_key_counts":      [2] * n,
        "vote_counts":              [100 + i * 10 for i in range(n)],
        "popularities":             [1.0 + i * 0.5 for i in range(n)],
        "vote_averages":            [6.0 + i * 0.1 for i in range(n)],
        "overview_lengths":         [150 + i for i in range(n)],
        "genre_counts":             [2] * n,
        "has_revenue":              [1] * n,
        "has_budget":               [1] * n,
        "has_production_companies": [1] * n,
        "has_production_countries": [1] * n,
        "has_keywords":             [1] * n,
        "has_cast_and_crew":        [1] * n,
    }


class TestBuildAnalysis:
    """Tests for the full analysis suite orchestrator."""

    def test_returns_total_movies_key(self) -> None:
        """Output contains 'total_movies' matching input length."""
        inputs = _make_analysis_inputs(15)
        result = _build_analysis(**inputs)
        assert result["total_movies"] == 15

    def test_computes_group_specific_percentiles(self) -> None:
        """vote_count_percentiles reflect the input group, not global data."""
        # Create a group with known vote_count range.
        inputs = _make_analysis_inputs(5)
        inputs["vote_counts"] = [10, 20, 30, 40, 50]
        result = _build_analysis(**inputs)

        # p50 should be the median of [10, 20, 30, 40, 50] = 30.
        assert result["vote_count_percentiles"]["p50"] == pytest.approx(30.0)

    def test_all_expected_analysis_sections_present(self) -> None:
        """All 14 expected keys present in output."""
        inputs = _make_analysis_inputs(10)
        result = _build_analysis(**inputs)

        expected_keys = {
            "total_movies",
            "vote_count_percentiles",
            "title",
            "release_date",
            "duration",
            "poster_url",
            "watch_providers",
            "vote_count",
            "popularity",
            "vote_average",
            "overview_length",
            "genre_count",
            "boolean_fields",
            "cross_attributes",
        }
        assert set(result.keys()) == expected_keys
