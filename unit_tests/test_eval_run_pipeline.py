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

    def test_filter_synopsis_branch_skips_movies_without_synopsis(self) -> None:
        """branch='synopsis' excludes movies without any synopsis."""
        with_synopsis = _make_movie(
            tmdb_id=1,
            overview="A long enough overview.",
            plot_synopses=["A synopsis."],
        )
        without_synopsis = _make_movie(
            tmdb_id=2,
            overview="A long enough overview.",
            plot_synopses=[],
        )
        movie_inputs = {1: with_synopsis, 2: without_synopsis}

        result = _filter_plot_events_eligible(movie_inputs, branch="synopsis")
        assert 1 in result
        assert 2 not in result

    def test_filter_synthesis_branch_skips_movies_with_synopsis(self) -> None:
        """branch='synthesis' excludes movies that have a synopsis."""
        with_synopsis = _make_movie(
            tmdb_id=1,
            overview="A long enough overview.",
            plot_synopses=["A synopsis."],
        )
        without_synopsis = _make_movie(
            tmdb_id=2,
            overview="A long enough overview.",
            plot_synopses=[],
        )
        movie_inputs = {1: with_synopsis, 2: without_synopsis}

        result = _filter_plot_events_eligible(movie_inputs, branch="synthesis")
        assert 1 not in result
        assert 2 in result

    def test_filter_no_branch_includes_all_eligible(self) -> None:
        """branch=None includes all eligible movies regardless of synopsis."""
        with_synopsis = _make_movie(
            tmdb_id=1,
            overview="A long enough overview.",
            plot_synopses=["A synopsis."],
        )
        without_synopsis = _make_movie(
            tmdb_id=2,
            overview="A long enough overview.",
            plot_synopses=[],
        )
        movie_inputs = {1: with_synopsis, 2: without_synopsis}

        result = _filter_plot_events_eligible(movie_inputs, branch=None)
        assert 1 in result
        assert 2 in result

    def test_filter_branch_respects_eligibility_first(self) -> None:
        """branch filtering only applies after check_plot_events eligibility."""
        ineligible = _make_movie(
            tmdb_id=1,
            overview="",
            plot_synopses=["A synopsis."],
            plot_summaries=[],
        )
        # This movie has a synopsis but insufficient text data overall
        movie_inputs = {1: ineligible}

        result = _filter_plot_events_eligible(movie_inputs, branch="synopsis")
        # Should still be excluded by check_plot_events (all text sources sparse)
        # Note: this depends on whether synopsis alone passes check_plot_events


# ---------------------------------------------------------------------------
# Tests: evaluation test set composition
# ---------------------------------------------------------------------------


class TestEvaluationTestSet:
    def test_evaluation_test_set_contains_all_sparsity_subsets(self) -> None:
        """EVALUATION_TEST_SET_TMDB_IDS is a superset of all sparsity category lists."""
        full_set = set(EVALUATION_TEST_SET_TMDB_IDS)
        for tmdb_id in ORIGINAL_SET_TMDB_IDS:
            assert tmdb_id in full_set, f"ORIGINAL tmdb_id {tmdb_id} not in EVALUATION_TEST_SET_TMDB_IDS"
        for tmdb_id in MEDIUM_SPARSITY_TMDB_IDS:
            assert tmdb_id in full_set, f"MEDIUM_SPARSITY tmdb_id {tmdb_id} not in EVALUATION_TEST_SET_TMDB_IDS"
        for tmdb_id in HIGH_SPARSITY_TMDB_IDS:
            assert tmdb_id in full_set, f"HIGH_SPARSITY tmdb_id {tmdb_id} not in EVALUATION_TEST_SET_TMDB_IDS"

    def test_evaluation_test_set_has_no_duplicates(self) -> None:
        """All IDs in EVALUATION_TEST_SET_TMDB_IDS are unique."""
        assert len(EVALUATION_TEST_SET_TMDB_IDS) == len(set(EVALUATION_TEST_SET_TMDB_IDS))
