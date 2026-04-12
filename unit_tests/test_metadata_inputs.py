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
    MetadataType,
    MultiLineList,
    SkipAssessment,
    ConsolidatedInputs,
    build_user_prompt,
    build_custom_id,
    parse_custom_id,
    WAVE1_TYPES,
    WAVE2_TYPES,
    ALL_GENERATION_TYPES,
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
        """batch_id() produces '{metadata_type}_{tmdb_id}' format."""
        movie = _make_movie()
        assert movie.batch_id(MetadataType.PLOT_EVENTS) == "plot_events_12345"

    def test_batch_id_different_types(self):
        """Different metadata types produce different batch_ids."""
        movie = _make_movie()
        id_a = movie.batch_id(MetadataType.PLOT_EVENTS)
        id_b = movie.batch_id(MetadataType.RECEPTION)
        assert id_a != id_b
        assert id_a == "plot_events_12345"
        assert id_b == "reception_12345"

    def test_batch_id_delegates_to_build_custom_id(self):
        """MovieInputData.batch_id() produces same result as build_custom_id()."""
        movie = _make_movie()
        assert movie.batch_id(MetadataType.PLOT_EVENTS) == build_custom_id(12345, MetadataType.PLOT_EVENTS)


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
        assert result == ""

    def test_build_user_prompt_no_fields(self):
        result = build_user_prompt()
        assert result == ""

    def test_build_user_prompt_multiline_list_renders_with_dash_prefix(self):
        result = build_user_prompt(items=MultiLineList(["item1", "item2"]))
        assert result == "items: \n- item1\n- item2"

    def test_build_user_prompt_regular_list_renders_comma_separated(self):
        result = build_user_prompt(items=["item1", "item2"])
        assert result == "items: item1, item2"

    def test_build_user_prompt_empty_multiline_list_skipped(self):
        result = build_user_prompt(items=MultiLineList([]))
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


# ---------------------------------------------------------------------------
# MovieInputData.merged_keywords()
# ---------------------------------------------------------------------------

class TestMergedKeywords:
    def test_merged_keywords_basic(self):
        movie = _make_movie(plot_keywords=["a", "b"], overall_keywords=["c"])
        assert movie.merged_keywords() == ["a", "b", "c"]

    def test_merged_keywords_deduplication(self):
        movie = _make_movie(plot_keywords=["action"], overall_keywords=["action"])
        assert movie.merged_keywords() == ["action"]

    def test_merged_keywords_case_normalization(self):
        movie = _make_movie(plot_keywords=["Action"], overall_keywords=["action"])
        assert movie.merged_keywords() == ["action"]

    def test_merged_keywords_strip_whitespace(self):
        movie = _make_movie(plot_keywords=["  hacker  "], overall_keywords=[])
        assert movie.merged_keywords() == ["hacker"]

    def test_merged_keywords_order_preserving(self):
        movie = _make_movie(
            plot_keywords=["beta", "alpha"],
            overall_keywords=["gamma", "delta"],
        )
        assert movie.merged_keywords() == ["beta", "alpha", "gamma", "delta"]

    def test_merged_keywords_empty_both(self):
        movie = _make_movie(plot_keywords=[], overall_keywords=[])
        assert movie.merged_keywords() == []

    def test_merged_keywords_empty_plot(self):
        movie = _make_movie(plot_keywords=[], overall_keywords=["Cyberpunk"])
        assert movie.merged_keywords() == ["cyberpunk"]

    def test_merged_keywords_empty_overall(self):
        movie = _make_movie(plot_keywords=["Hacker"], overall_keywords=[])
        assert movie.merged_keywords() == ["hacker"]

    def test_merged_keywords_all_duplicates(self):
        movie = _make_movie(plot_keywords=["a"], overall_keywords=["A"])
        assert movie.merged_keywords() == ["a"]


# ---------------------------------------------------------------------------
# MovieInputData.maturity_summary()
# ---------------------------------------------------------------------------

class TestMaturitySummary:
    def test_maturity_summary_with_reasoning(self):
        movie = _make_movie(
            maturity_rating="R",
            maturity_reasoning=["Rated R for violence"],
        )
        result = movie.maturity_summary()
        assert result == "Rated R for violence"

    def test_maturity_summary_with_parental_items(self):
        movie = _make_movie(
            maturity_rating="R",
            maturity_reasoning=[],
            parental_guide_items=[
                {"category": "violence", "severity": "severe"},
            ],
        )
        result = movie.maturity_summary()
        assert result == "R — severe violence"

    def test_maturity_summary_mpaa_rating_only(self):
        movie = _make_movie(
            maturity_rating="R",
            maturity_reasoning=[],
            parental_guide_items=[],
        )
        result = movie.maturity_summary()
        assert result == "R — Restricted"

    def test_maturity_summary_none(self):
        movie = _make_movie()
        assert movie.maturity_summary() is None

    def test_maturity_summary_delegates_to_consolidate_maturity(self):
        """Verify output matches pre_consolidation.consolidate_maturity() directly."""
        from movie_ingestion.metadata_generation.batch_generation.pre_consolidation import consolidate_maturity

        movie = _make_movie(
            maturity_rating="PG-13",
            maturity_reasoning=["Rated PG-13 for intense action"],
            parental_guide_items=[{"category": "violence", "severity": "moderate"}],
        )
        expected = consolidate_maturity(
            movie.maturity_rating,
            movie.maturity_reasoning,
            movie.parental_guide_items,
        )
        assert movie.maturity_summary() == expected


# ---------------------------------------------------------------------------
# build_user_prompt() edge cases
# ---------------------------------------------------------------------------

class TestBuildUserPromptEdgeCases:
    def test_build_user_prompt_integer_value(self):
        """Non-string scalars (integers) are formatted correctly via str()."""
        result = build_user_prompt(year=1999)
        assert result == "year: 1999"

    def test_build_user_prompt_field_ordering_preserved(self):
        """Kwargs order is preserved in output (Python 3.7+ dict ordering)."""
        result = build_user_prompt(alpha="first", beta="second", gamma="third")
        lines = result.split("\n")
        assert lines[0] == "alpha: first"
        assert lines[1] == "beta: second"
        assert lines[2] == "gamma: third"


# ---------------------------------------------------------------------------
# MovieInputData.batch_id() edge cases
# ---------------------------------------------------------------------------

class TestBatchIdEdgeCases:
    def test_batch_id_large_tmdb_id(self):
        """Large tmdb_id values are formatted correctly."""
        movie = _make_movie(tmdb_id=999999999)
        assert movie.batch_id(MetadataType.PLOT_EVENTS) == "plot_events_999999999"

    def test_batch_id_zero_tmdb_id(self):
        """tmdb_id=0 is handled correctly."""
        movie = _make_movie(tmdb_id=0)
        assert movie.batch_id(MetadataType.RECEPTION) == "reception_0"


# ---------------------------------------------------------------------------
# MultiLineList isinstance check
# ---------------------------------------------------------------------------

class TestMultiLineList:
    def test_multiline_list_is_list(self):
        """MultiLineList inherits from list — isinstance check passes."""
        ml = MultiLineList(["a", "b"])
        assert isinstance(ml, list)


# ---------------------------------------------------------------------------
# build_custom_id / parse_custom_id
# ---------------------------------------------------------------------------


class TestBuildCustomId:
    def test_build_custom_id_format(self):
        """build_custom_id produces '{metadata_type}_{tmdb_id}' format."""
        assert build_custom_id(12345, MetadataType.PLOT_EVENTS) == "plot_events_12345"

    def test_build_custom_id_underscore_in_type_name(self):
        """Multi-underscore type names are preserved in custom_id."""
        result = build_custom_id(42, MetadataType.SOURCE_OF_INSPIRATION)
        assert result == "source_of_inspiration_42"

    def test_build_custom_id_all_types(self):
        """All MetadataType values produce valid custom_ids."""
        for mt in MetadataType:
            result = build_custom_id(1, mt)
            assert result == f"{mt.value}_1"


class TestParseCustomId:
    def test_parse_custom_id_round_trip(self):
        """parse_custom_id(build_custom_id(...)) round-trips correctly."""
        for mt in MetadataType:
            custom_id = build_custom_id(42, mt)
            parsed_type, parsed_id = parse_custom_id(custom_id)
            assert parsed_type == mt
            assert parsed_id == 42

    def test_parse_custom_id_plot_events(self):
        """Parse a plot_events custom_id."""
        mt, tid = parse_custom_id("plot_events_12345")
        assert mt == MetadataType.PLOT_EVENTS
        assert tid == 12345

    def test_parse_custom_id_source_of_inspiration(self):
        """Multi-underscore type name parses correctly via rsplit."""
        mt, tid = parse_custom_id("source_of_inspiration_42")
        assert mt == MetadataType.SOURCE_OF_INSPIRATION
        assert tid == 42

    def test_parse_custom_id_narrative_techniques(self):
        """Another multi-underscore type name parses correctly."""
        mt, tid = parse_custom_id("narrative_techniques_99")
        assert mt == MetadataType.NARRATIVE_TECHNIQUES
        assert tid == 99

    def test_parse_custom_id_invalid_type_raises_valueerror(self):
        """Invalid metadata type in custom_id raises ValueError."""
        with pytest.raises(ValueError):
            parse_custom_id("bogus_type_999")

    def test_parse_custom_id_non_integer_tmdb_id_raises(self):
        """Non-integer tmdb_id portion raises ValueError."""
        with pytest.raises(ValueError):
            parse_custom_id("plot_events_abc")

    def test_parse_custom_id_returns_metadata_type_not_str(self):
        """Returned type is MetadataType instance, not plain str."""
        mt, _ = parse_custom_id("reception_1")
        assert isinstance(mt, MetadataType)


# ---------------------------------------------------------------------------
# MetadataType enum
# ---------------------------------------------------------------------------


class TestMetadataTypeEnum:
    def test_metadata_type_enum_has_12_members(self):
        """MetadataType has all current generation types."""
        assert len(MetadataType) == 12

    def test_metadata_type_is_strenum(self):
        """All MetadataType members are strings."""
        for mt in MetadataType:
            assert isinstance(mt, str)

    def test_wave1_types_contains_plot_events_and_reception(self):
        """WAVE1_TYPES contains plot_events and reception."""
        assert MetadataType.PLOT_EVENTS in WAVE1_TYPES
        assert MetadataType.RECEPTION in WAVE1_TYPES
        assert len(WAVE1_TYPES) == 2

    def test_wave2_types_contains_current_types(self):
        """WAVE2_TYPES contains all current Wave 2 generation types."""
        assert len(WAVE2_TYPES) == 8
        assert MetadataType.PLOT_ANALYSIS in WAVE2_TYPES
        assert MetadataType.VIEWER_EXPERIENCE in WAVE2_TYPES
        assert MetadataType.WATCH_CONTEXT in WAVE2_TYPES
        assert MetadataType.NARRATIVE_TECHNIQUES in WAVE2_TYPES
        assert MetadataType.PRODUCTION_KEYWORDS in WAVE2_TYPES
        assert MetadataType.PRODUCTION_TECHNIQUES in WAVE2_TYPES
        assert MetadataType.SOURCE_OF_INSPIRATION in WAVE2_TYPES
        assert MetadataType.SOURCE_MATERIAL_V2 in WAVE2_TYPES
        assert MetadataType.CONCEPT_TAGS in WAVE2_TYPES

    def test_all_generation_types_is_union(self):
        """ALL_GENERATION_TYPES is the union of WAVE1_TYPES and WAVE2_TYPES."""
        assert ALL_GENERATION_TYPES == WAVE1_TYPES | WAVE2_TYPES
