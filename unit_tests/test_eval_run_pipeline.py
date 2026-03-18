"""
Unit tests for movie_ingestion.metadata_generation.evaluations.run_evaluations_pipeline.

Covers:
  - _filter_plot_events_eligible (eligible only, rich movie, empty input, all ineligible)

No real LLM calls or DB access — tests use MovieInputData directly.
"""

from movie_ingestion.metadata_generation.evaluations.run_evaluations_pipeline import (
    _filter_plot_events_eligible,
)
from movie_ingestion.metadata_generation.evaluations.shared import (
    EVALUATION_TEST_SET_TMDB_IDS,
    HIGH_SPARSITY_TMDB_IDS,
    MEDIUM_SPARSITY_TMDB_IDS,
    ORIGINAL_SET_TMDB_IDS,
)
from movie_ingestion.metadata_generation.inputs import MovieInputData


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_movie(**overrides) -> MovieInputData:
    """Build a MovieInputData with minimal required fields + overrides."""
    defaults = {"tmdb_id": 12345, "title": "Test Movie"}
    defaults.update(overrides)
    return MovieInputData(**defaults)


# ---------------------------------------------------------------------------
# Tests: _filter_plot_events_eligible
# ---------------------------------------------------------------------------


class TestFilterPlotEventsEligible:
    def test_filter_plot_events_eligible_returns_eligible_only(self) -> None:
        """Movies failing check_plot_events are excluded."""
        eligible_movie = _make_movie(
            tmdb_id=1,
            overview="A long enough overview for plot events to pass the threshold.",
        )
        ineligible_movie = _make_movie(
            tmdb_id=2,
            overview="",
            plot_synopses=[],
            plot_summaries=[],
        )
        movie_inputs = {1: eligible_movie, 2: ineligible_movie}

        result = _filter_plot_events_eligible(movie_inputs)

        assert 1 in result
        assert 2 not in result

    def test_filter_plot_events_eligible_includes_rich_movie(self) -> None:
        """Movies with sufficient data pass through."""
        rich_movie = _make_movie(
            tmdb_id=42,
            overview="A computer hacker learns about the true nature of reality.",
            plot_synopses=["Neo discovers he is living in a simulated reality."],
            plot_summaries=["A lengthy summary of The Matrix."],
        )
        result = _filter_plot_events_eligible({42: rich_movie})

        assert 42 in result
        assert result[42] is rich_movie

    def test_filter_plot_events_eligible_empty_input(self) -> None:
        """Empty dict input returns empty dict."""
        result = _filter_plot_events_eligible({})
        assert result == {}

    def test_filter_plot_events_eligible_all_ineligible(self) -> None:
        """All movies ineligible returns empty dict (not error)."""
        sparse_1 = _make_movie(tmdb_id=1, overview="", plot_synopses=[], plot_summaries=[])
        sparse_2 = _make_movie(tmdb_id=2, overview="", plot_synopses=[], plot_summaries=[])
        movie_inputs = {1: sparse_1, 2: sparse_2}

        result = _filter_plot_events_eligible(movie_inputs)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests: temp evaluation set (run_evaluations_pipeline.main)
# ---------------------------------------------------------------------------


class TestTempEvaluationSet:
    def test_temp_evaluation_set_is_proper_subset_of_full_set(self) -> None:
        """The temporary sliced set has exactly 11 IDs, all contained in EVALUATION_TEST_SET_TMDB_IDS."""
        # Mirrors the slice expression used in main()
        temp_set = ORIGINAL_SET_TMDB_IDS[:5] + MEDIUM_SPARSITY_TMDB_IDS[:3] + HIGH_SPARSITY_TMDB_IDS[:3]

        assert len(temp_set) == 11

        full_set = set(EVALUATION_TEST_SET_TMDB_IDS)
        for tmdb_id in temp_set:
            assert tmdb_id in full_set, f"tmdb_id {tmdb_id} from temp set not in EVALUATION_TEST_SET_TMDB_IDS"
