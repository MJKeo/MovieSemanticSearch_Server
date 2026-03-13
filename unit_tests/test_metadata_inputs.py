"""
Unit tests for movie_ingestion.metadata_generation.inputs.

Covers:
  - MovieInputData: batch_id(), title_with_year(), defaults
  - SkipAssessment and ConsolidatedInputs: default factories
  - build_user_prompt(): field assembly, None skipping, list joining
"""

import pytest

from movie_ingestion.metadata_generation.inputs import (
    MovieInputData,
    SkipAssessment,
    ConsolidatedInputs,
    build_user_prompt,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_movie(**overrides) -> MovieInputData:
    """Build a MovieInputData with minimal required fields + overrides."""
    defaults = {"tmdb_id": 12345, "title": "The Matrix"}
    defaults.update(overrides)
    return MovieInputData(**defaults)


# ---------------------------------------------------------------------------
# MovieInputData.batch_id()
# ---------------------------------------------------------------------------

class TestBatchId:
    def test_batch_id_format(self):
        movie = _make_movie()
        assert movie.batch_id("plot_events") == "12345-plot_events"

    def test_batch_id_different_types(self):
        movie = _make_movie()
        id_a = movie.batch_id("plot_events")
        id_b = movie.batch_id("reception")
        assert id_a != id_b
        assert id_a == "12345-plot_events"
        assert id_b == "12345-reception"


# ---------------------------------------------------------------------------
# MovieInputData.title_with_year()
# ---------------------------------------------------------------------------

class TestTitleWithYear:
    def test_title_with_year_present(self):
        movie = _make_movie(release_year=1999)
        assert movie.title_with_year() == "The Matrix (1999)"

    def test_title_with_year_none(self):
        movie = _make_movie(release_year=None)
        assert movie.title_with_year() == "The Matrix"

    def test_title_with_year_zero(self):
        # year=0 is falsy as an int but is not None — should still format
        movie = _make_movie(title="Title", release_year=0)
        assert movie.title_with_year() == "Title (0)"


# ---------------------------------------------------------------------------
# MovieInputData defaults
# ---------------------------------------------------------------------------

class TestMovieInputDataDefaults:
    def test_movie_input_data_defaults(self):
        movie = _make_movie()
        assert movie.tmdb_id == 12345
        assert movie.title == "The Matrix"
        assert movie.release_year is None
        assert movie.overview == ""
        assert movie.genres == []
        assert movie.plot_synopses == []
        assert movie.plot_summaries == []
        assert movie.plot_keywords == []
        assert movie.overall_keywords == []
        assert movie.featured_reviews == []
        assert movie.reception_summary is None
        assert movie.audience_reception_attributes == []
        assert movie.maturity_rating == ""
        assert movie.maturity_reasoning == []
        assert movie.parental_guide_items == []

    def test_movie_input_data_with_overrides(self):
        movie = _make_movie(
            release_year=1999,
            overview="A computer hacker learns about the true nature of reality.",
            genres=["Action", "Sci-Fi"],
            plot_synopses=["Neo discovers the Matrix."],
            plot_summaries=["Summary of the plot."],
            plot_keywords=["hacker", "simulation"],
            overall_keywords=["cyberpunk"],
            featured_reviews=[{"summary": "Great!", "text": "A masterpiece."}],
            reception_summary="Widely acclaimed.",
            audience_reception_attributes=[{"name": "groundbreaking", "sentiment": "positive"}],
            maturity_rating="R",
            maturity_reasoning=["Rated R for violence"],
            parental_guide_items=[{"category": "violence", "severity": "severe"}],
        )
        assert movie.release_year == 1999
        assert movie.overview == "A computer hacker learns about the true nature of reality."
        assert movie.genres == ["Action", "Sci-Fi"]
        assert len(movie.plot_synopses) == 1
        assert len(movie.featured_reviews) == 1
        assert movie.reception_summary == "Widely acclaimed."
        assert movie.maturity_rating == "R"


# ---------------------------------------------------------------------------
# SkipAssessment and ConsolidatedInputs defaults
# ---------------------------------------------------------------------------

class TestDataContainerDefaults:
    def test_skip_assessment_defaults(self):
        sa = SkipAssessment()
        assert sa.generations_to_run == set()
        assert sa.skip_reasons == {}

    def test_consolidated_inputs_defaults(self):
        movie = _make_movie()
        ci = ConsolidatedInputs(movie_input=movie)
        assert ci.movie_input is movie
        assert ci.title_with_year == ""
        assert ci.merged_keywords == []
        assert ci.maturity_summary is None
        assert isinstance(ci.skip_assessment, SkipAssessment)
        assert ci.skip_assessment.generations_to_run == set()


# ---------------------------------------------------------------------------
# build_user_prompt()
# ---------------------------------------------------------------------------

class TestBuildUserPrompt:
    def test_build_user_prompt_basic(self):
        result = build_user_prompt(title="The Matrix")
        assert result == "title: The Matrix"

    def test_build_user_prompt_multiple_fields(self):
        result = build_user_prompt(title="The Matrix", year="1999")
        assert "title: The Matrix" in result
        assert "year: 1999" in result
        # Should be newline-separated
        assert "\n" in result

    def test_build_user_prompt_skips_none(self):
        result = build_user_prompt(title="The Matrix", overview=None)
        assert "title: The Matrix" in result
        assert "overview" not in result

    def test_build_user_prompt_list_values(self):
        result = build_user_prompt(genres=["Action", "Sci-Fi"])
        assert result == "genres: Action, Sci-Fi"

    def test_build_user_prompt_empty_list(self):
        result = build_user_prompt(genres=[])
        assert result == "genres: "

    def test_build_user_prompt_no_fields(self):
        result = build_user_prompt()
        assert result == ""

    def test_build_user_prompt_mixed_types(self):
        result = build_user_prompt(
            title="The Matrix",
            genres=["Action", "Sci-Fi"],
            overview=None,
            year="1999",
        )
        lines = result.split("\n")
        # None value (overview) should be excluded
        assert len(lines) == 3
        assert "overview" not in result
        assert "title: The Matrix" in result
        assert "genres: Action, Sci-Fi" in result
        assert "year: 1999" in result
