"""
Unit tests for movie_ingestion.metadata_generation.pre_consolidation.

Covers:
  - route_keywords: normalization, dedup, merge ordering
  - consolidate_maturity: 4-step priority chain
  - Per-generation eligibility checks (check_*/_ check_* functions)
  - assess_skip_conditions: Wave 1 and Wave 2 orchestration
  - run_pre_consolidation: end-to-end orchestrator
"""

import pytest

from movie_ingestion.metadata_generation.inputs import (
    MovieInputData,
    ConsolidatedInputs,
    SkipAssessment,
)
from movie_ingestion.metadata_generation.schemas import (
    PlotEventsOutput,
    ReceptionOutput,
    MajorCharacter,
)
from movie_ingestion.metadata_generation.pre_consolidation import (
    route_keywords,
    consolidate_maturity,
    check_plot_events,
    check_reception,
    _check_plot_analysis,
    _check_viewer_experience,
    _check_watch_context,
    _check_narrative_techniques,
    _check_production_keywords,
    _check_source_of_inspiration,
    _all_text_sources_sparse,
    assess_skip_conditions,
    run_pre_consolidation,
    MPAA_DEFINITIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_movie(**overrides) -> MovieInputData:
    """Build a MovieInputData with minimal required fields + overrides."""
    defaults = {"tmdb_id": 12345, "title": "The Matrix"}
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_rich_movie() -> MovieInputData:
    """Build a MovieInputData with all fields populated richly."""
    return _make_movie(
        release_year=1999,
        overview="A computer hacker learns about the true nature of reality and his role in the war against its controllers.",
        genres=["Action", "Sci-Fi"],
        plot_synopses=["Neo discovers he is living in a simulated reality and joins a rebellion to fight the machines."],
        plot_summaries=["A lengthy summary of the plot of The Matrix that covers all major events."],
        plot_keywords=["hacker", "simulation"],
        overall_keywords=["cyberpunk", "dystopia"],
        featured_reviews=[
            {"summary": "Groundbreaking", "text": "A revolutionary film that changed cinema forever with its stunning visuals."},
        ],
        reception_summary="Widely acclaimed for its visual effects and philosophical themes.",
        audience_reception_attributes=[{"name": "groundbreaking", "sentiment": "positive"}],
        maturity_rating="R",
        maturity_reasoning=["Rated R for sci-fi violence and brief language"],
        parental_guide_items=[{"category": "violence", "severity": "moderate"}],
    )


def _make_plot_events_output(**overrides) -> PlotEventsOutput:
    """Build a minimal valid PlotEventsOutput."""
    defaults = {
        "plot_summary": "Neo discovers the Matrix is a simulation.",
        "setting": "Near-future dystopia",
        "major_characters": [],
    }
    defaults.update(overrides)
    return PlotEventsOutput(**defaults)


def _make_reception_output(**overrides) -> ReceptionOutput:
    """Build a minimal valid ReceptionOutput."""
    defaults = {
        "new_reception_summary": "Widely acclaimed.",
        "praise_attributes": ["groundbreaking"],
        "complaint_attributes": [],
        "review_insights_brief": "Critics praised the visual effects and philosophical depth.",
    }
    defaults.update(overrides)
    return ReceptionOutput(**defaults)


# ---------------------------------------------------------------------------
# route_keywords
# ---------------------------------------------------------------------------

class TestRouteKeywords:
    def test_route_keywords_basic(self):
        plot_kw, overall_kw, merged = route_keywords(
            ["hacker", "simulation"], ["cyberpunk", "dystopia"]
        )
        assert plot_kw == ["hacker", "simulation"]
        assert overall_kw == ["cyberpunk", "dystopia"]
        assert merged == ["hacker", "simulation", "cyberpunk", "dystopia"]

    def test_route_keywords_normalization(self):
        plot_kw, overall_kw, merged = route_keywords(
            ["  Hacker  ", "SIMULATION"], ["CyberPunk"]
        )
        assert plot_kw == ["hacker", "simulation"]
        assert overall_kw == ["cyberpunk"]
        assert merged == ["hacker", "simulation", "cyberpunk"]

    def test_route_keywords_dedup_within_list(self):
        plot_kw, _, _ = route_keywords(["hacker", "Hacker", "HACKER"], [])
        assert plot_kw == ["hacker"]

    def test_route_keywords_dedup_across_lists(self):
        _, _, merged = route_keywords(["action"], ["Action", "comedy"])
        # "action" from overall is a duplicate of plot's "action"
        assert merged == ["action", "comedy"]

    def test_route_keywords_order_preserved(self):
        _, _, merged = route_keywords(["beta", "alpha"], ["gamma", "delta"])
        assert merged == ["beta", "alpha", "gamma", "delta"]

    def test_route_keywords_empty_inputs(self):
        plot_kw, overall_kw, merged = route_keywords([], [])
        assert plot_kw == []
        assert overall_kw == []
        assert merged == []

    def test_route_keywords_one_empty(self):
        plot_kw, overall_kw, merged = route_keywords(["action"], [])
        assert plot_kw == ["action"]
        assert overall_kw == []
        assert merged == ["action"]

    def test_route_keywords_case_dedup(self):
        plot_kw, _, _ = route_keywords(["Action", "action"], [])
        assert plot_kw == ["action"]


# ---------------------------------------------------------------------------
# consolidate_maturity
# ---------------------------------------------------------------------------

class TestConsolidateMaturity:
    def test_consolidate_maturity_reasoning_priority(self):
        result = consolidate_maturity(
            "R",
            ["Rated R for violence and language"],
            [{"category": "violence", "severity": "severe"}],
        )
        assert result == "Rated R for violence and language"

    def test_consolidate_maturity_reasoning_multiple(self):
        result = consolidate_maturity(
            "R",
            ["Rated R for violence", "Brief nudity"],
            [],
        )
        assert result == "Rated R for violence, Brief nudity"

    def test_consolidate_maturity_parental_items_with_rating(self):
        result = consolidate_maturity(
            "R",
            [],
            [
                {"category": "violence", "severity": "severe"},
                {"category": "language", "severity": "moderate"},
            ],
        )
        assert result == "R — severe violence, moderate language"

    def test_consolidate_maturity_parental_items_no_rating(self):
        result = consolidate_maturity(
            "",
            [],
            [
                {"category": "violence", "severity": "severe"},
                {"category": "language", "severity": "moderate"},
            ],
        )
        assert result == "severe violence, moderate language"

    def test_consolidate_maturity_mpaa_rating_only(self):
        result = consolidate_maturity("R", [], [])
        assert result == "R — Restricted"

    def test_consolidate_maturity_unknown_rating_only(self):
        result = consolidate_maturity("TV-MA", [], [])
        assert result is None

    def test_consolidate_maturity_nothing(self):
        result = consolidate_maturity("", [], [])
        assert result is None

    def test_consolidate_maturity_empty_rating_string(self):
        result = consolidate_maturity("", [], [])
        assert result is None


# ---------------------------------------------------------------------------
# check_plot_events
# ---------------------------------------------------------------------------

class TestCheckPlotEvents:
    def test_check_plot_events_eligible_overview(self):
        # Overview >= 10 chars → eligible
        movie = _make_movie(overview="A long enough overview for the test.")
        assert check_plot_events(movie) is None

    def test_check_plot_events_eligible_synopsis(self):
        movie = _make_movie(overview="", plot_synopses=["A synopsis that is long enough to pass the threshold easily."])
        assert check_plot_events(movie) is None

    def test_check_plot_events_eligible_summary(self):
        movie = _make_movie(overview="", plot_summaries=["A summary that is long enough to pass the threshold easily."])
        assert check_plot_events(movie) is None

    def test_check_plot_events_skip_all_missing(self):
        movie = _make_movie(overview="", plot_synopses=[], plot_summaries=[])
        reason = check_plot_events(movie)
        assert reason is not None
        assert "No overview" in reason

    def test_check_plot_events_skip_all_sparse(self):
        # overview < 10 chars, each synopsis < 50 chars, combined summaries < 50 chars
        movie = _make_movie(
            overview="Short",
            plot_synopses=["Tiny"],
            plot_summaries=["Small"],
        )
        reason = check_plot_events(movie)
        assert reason is not None
        assert "sparse" in reason

    def test_check_plot_events_sparse_overview_but_good_synopsis(self):
        # overview too short but synopsis is long enough
        movie = _make_movie(
            overview="Short",
            plot_synopses=["This is a synopsis that is definitely longer than fifty characters to pass the threshold."],
        )
        assert check_plot_events(movie) is None

    def test_check_plot_events_empty_overview_string(self):
        # overview="" is treated as missing (falsy)
        movie = _make_movie(overview="", plot_synopses=[], plot_summaries=[])
        reason = check_plot_events(movie)
        assert reason is not None


# ---------------------------------------------------------------------------
# check_reception
# ---------------------------------------------------------------------------

class TestCheckReception:
    def test_check_reception_eligible_reviews(self):
        # Reviews with combined text >= 25 chars
        movie = _make_movie(
            featured_reviews=[{"text": "A truly wonderful and groundbreaking film."}]
        )
        assert check_reception(movie) is None

    def test_check_reception_eligible_summary_only(self):
        movie = _make_movie(
            reception_summary="Widely acclaimed.",
            featured_reviews=[],
        )
        assert check_reception(movie) is None

    def test_check_reception_eligible_attributes_only(self):
        movie = _make_movie(
            audience_reception_attributes=[{"name": "visionary", "sentiment": "positive"}],
        )
        assert check_reception(movie) is None

    def test_check_reception_skip_nothing(self):
        movie = _make_movie(
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[],
        )
        reason = check_reception(movie)
        assert reason is not None

    def test_check_reception_skip_short_reviews(self):
        # Reviews with combined text < 25 chars, no other data
        movie = _make_movie(
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[{"text": "Short."}],
        )
        reason = check_reception(movie)
        assert reason is not None

    def test_check_reception_reviews_at_threshold(self):
        # Exactly 25 chars combined → eligible
        movie = _make_movie(
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[{"text": "a" * 25}],
        )
        assert check_reception(movie) is None

    def test_check_reception_reviews_missing_text_key(self):
        # Reviews without "text" key — get() returns "" → counted as 0 chars
        movie = _make_movie(
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[{"summary": "No text key here"}],
        )
        reason = check_reception(movie)
        assert reason is not None


# ---------------------------------------------------------------------------
# _check_plot_analysis
# ---------------------------------------------------------------------------

class TestCheckPlotAnalysis:
    def test_check_plot_analysis_eligible_synopsis(self):
        assert _check_plot_analysis("A plot synopsis.", None) is None

    def test_check_plot_analysis_eligible_insights(self):
        assert _check_plot_analysis(None, "Review insights.") is None

    def test_check_plot_analysis_skip_both_none(self):
        assert _check_plot_analysis(None, None) is not None

    def test_check_plot_analysis_skip_both_empty(self):
        # Empty strings are falsy
        assert _check_plot_analysis("", "") is not None


# ---------------------------------------------------------------------------
# _check_viewer_experience
# ---------------------------------------------------------------------------

class TestCheckViewerExperience:
    def test_check_viewer_experience_eligible_synopsis(self):
        assert _check_viewer_experience("synopsis", None, [], [], None) is None

    def test_check_viewer_experience_eligible_insights(self):
        assert _check_viewer_experience(None, "insights", [], [], None) is None

    def test_check_viewer_experience_eligible_all_contextual(self):
        assert _check_viewer_experience(
            None, None, ["Action"], ["keyword"], "R — Restricted"
        ) is None

    def test_check_viewer_experience_skip_partial_contextual(self):
        # genres + keywords but no maturity → skip
        reason = _check_viewer_experience(None, None, ["Action"], ["keyword"], None)
        assert reason is not None

    def test_check_viewer_experience_skip_nothing(self):
        reason = _check_viewer_experience(None, None, [], [], None)
        assert reason is not None


# ---------------------------------------------------------------------------
# _check_watch_context
# ---------------------------------------------------------------------------

class TestCheckWatchContext:
    def test_check_watch_context_eligible_insights(self):
        assert _check_watch_context("insights", [], [], None) is None

    def test_check_watch_context_eligible_all_contextual(self):
        assert _check_watch_context(None, ["Action"], ["keyword"], "R — Restricted") is None

    def test_check_watch_context_skip_partial(self):
        # Missing maturity → skip
        reason = _check_watch_context(None, ["Action"], ["keyword"], None)
        assert reason is not None

    def test_check_watch_context_skip_nothing(self):
        reason = _check_watch_context(None, [], [], None)
        assert reason is not None


# ---------------------------------------------------------------------------
# _check_narrative_techniques
# ---------------------------------------------------------------------------

class TestCheckNarrativeTechniques:
    def test_check_narrative_eligible_synopsis(self):
        assert _check_narrative_techniques("synopsis", None, [], []) is None

    def test_check_narrative_eligible_genres_keywords(self):
        assert _check_narrative_techniques(None, None, ["Action"], ["keyword"]) is None

    def test_check_narrative_skip_nothing(self):
        reason = _check_narrative_techniques(None, None, [], [])
        assert reason is not None


# ---------------------------------------------------------------------------
# _check_production_keywords and _check_source_of_inspiration
# ---------------------------------------------------------------------------

class TestCheckProductionAndInspiration:
    def test_check_production_keywords_eligible(self):
        assert _check_production_keywords(["keyword"]) is None

    def test_check_production_keywords_skip_empty(self):
        assert _check_production_keywords([]) is not None

    def test_check_source_of_inspiration_eligible_keywords(self):
        assert _check_source_of_inspiration(["keyword"], None, None) is None

    def test_check_source_of_inspiration_eligible_synopsis(self):
        assert _check_source_of_inspiration([], None, "synopsis") is None

    def test_check_source_of_inspiration_skip_nothing(self):
        assert _check_source_of_inspiration([], None, None) is not None


# ---------------------------------------------------------------------------
# assess_skip_conditions
# ---------------------------------------------------------------------------

class TestAssessSkipConditions:
    def test_assess_wave1_all_eligible(self):
        movie = _make_rich_movie()
        result = assess_skip_conditions(movie)
        assert "plot_events" in result.generations_to_run
        assert "reception" in result.generations_to_run
        assert result.skip_reasons == {}

    def test_assess_wave1_skip_plot_events(self):
        # No text data → plot_events should be skipped
        movie = _make_movie(
            overview="",
            plot_synopses=[],
            plot_summaries=[],
            featured_reviews=[{"text": "A review that is long enough for reception."}],
            reception_summary="Good movie.",
        )
        result = assess_skip_conditions(movie)
        assert "plot_events" not in result.generations_to_run
        assert "plot_events" in result.skip_reasons
        assert "reception" in result.generations_to_run

    def test_assess_wave1_skip_reception(self):
        movie = _make_movie(
            overview="A long enough overview for plot events to pass.",
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[],
        )
        result = assess_skip_conditions(movie)
        assert "reception" not in result.generations_to_run
        assert "reception" in result.skip_reasons
        assert "plot_events" in result.generations_to_run

    def test_assess_wave2_all_eligible(self):
        movie = _make_rich_movie()
        pe_output = _make_plot_events_output()
        rec_output = _make_reception_output()
        result = assess_skip_conditions(
            movie,
            plot_events_output=pe_output,
            reception_output=rec_output,
            merged_keywords=["hacker", "simulation", "cyberpunk"],
            maturity_summary="R — Restricted",
        )
        # All 6 Wave 2 types should be eligible
        expected = {
            "plot_analysis", "viewer_experience", "watch_context",
            "narrative_techniques", "production_keywords", "source_of_inspiration",
        }
        assert result.generations_to_run == expected
        assert result.skip_reasons == {}

    def test_assess_wave2_skip_some(self):
        # Provide no Wave 1 outputs → plot_synopsis and review_insights_brief
        # will both be None, and with no keywords/maturity most types skip
        movie = _make_movie(genres=[])
        result = assess_skip_conditions(
            movie,
            plot_events_output=None,
            reception_output=None,
            merged_keywords=[],
            maturity_summary=None,
        )
        # With no Wave 1 outputs provided, this is actually Wave 1 dispatch.
        # Instead, provide outputs but leave the intermediate fields absent
        # by passing None for both typed outputs individually.
        pe_output = _make_plot_events_output()  # has plot_summary (truthy)
        result = assess_skip_conditions(
            movie,
            plot_events_output=pe_output,
            reception_output=None,  # review_insights_brief will be None
            merged_keywords=[],
            maturity_summary=None,
        )
        # production_keywords should be skipped (no keywords)
        assert len(result.skip_reasons) > 0
        assert "production_keywords" in result.skip_reasons

    def test_assess_wave2_computes_merged_keywords_if_none(self):
        # merged_keywords=None → should fall back to movie_input keywords
        movie = _make_movie(
            plot_keywords=["hacker"],
            overall_keywords=["cyberpunk"],
        )
        pe_output = _make_plot_events_output()
        rec_output = _make_reception_output()
        result = assess_skip_conditions(
            movie,
            plot_events_output=pe_output,
            reception_output=rec_output,
            merged_keywords=None,
            maturity_summary="R — Restricted",
        )
        # production_keywords should be eligible because keywords were computed
        assert "production_keywords" in result.generations_to_run

    def test_assess_skip_reasons_are_strings(self):
        movie = _make_movie(
            overview="",
            plot_synopses=[],
            plot_summaries=[],
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[],
        )
        result = assess_skip_conditions(movie)
        for reason in result.skip_reasons.values():
            assert isinstance(reason, str)
            assert len(reason) > 0


# ---------------------------------------------------------------------------
# run_pre_consolidation
# ---------------------------------------------------------------------------

class TestRunPreConsolidation:
    def test_run_pre_consolidation_populates_all_fields(self):
        movie = _make_rich_movie()
        result = run_pre_consolidation(movie)
        assert isinstance(result, ConsolidatedInputs)
        assert result.title_with_year != ""
        assert len(result.merged_keywords) > 0
        assert result.maturity_summary is not None
        assert isinstance(result.skip_assessment, SkipAssessment)

    def test_run_pre_consolidation_title_format(self):
        movie = _make_rich_movie()
        result = run_pre_consolidation(movie)
        assert result.title_with_year == movie.title_with_year()

    def test_run_pre_consolidation_preserves_movie_input(self):
        movie = _make_rich_movie()
        result = run_pre_consolidation(movie)
        assert result.movie_input is movie

    def test_run_pre_consolidation_sparse_movie(self):
        # Minimal data — skip assessment should skip appropriate types
        movie = _make_movie(
            overview="",
            plot_synopses=[],
            plot_summaries=[],
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[],
        )
        result = run_pre_consolidation(movie)
        assert len(result.skip_assessment.skip_reasons) > 0
        assert "plot_events" in result.skip_assessment.skip_reasons
        assert "reception" in result.skip_assessment.skip_reasons

    def test_run_pre_consolidation_merged_keywords_normalized(self):
        """run_pre_consolidation uses route_keywords (normalized/deduped), not raw lists."""
        movie = _make_movie(
            overview="A long enough overview for the test.",
            plot_keywords=["Action", "  action  "],
            overall_keywords=["ACTION"],
        )
        result = run_pre_consolidation(movie)
        # Normalized + deduped: only one "action"
        assert result.merged_keywords == ["action"]


# ---------------------------------------------------------------------------
# _all_text_sources_sparse boundary tests
# ---------------------------------------------------------------------------

class TestAllTextSourcesSparse:
    def test_overview_exactly_10_chars_is_not_sparse(self):
        """Overview of exactly 10 chars passes the threshold."""
        movie = _make_movie(overview="0123456789")  # 10 chars
        assert _all_text_sources_sparse(movie) is False

    def test_overview_9_chars_is_sparse(self):
        """Overview of 9 chars is below threshold — sparse if no other sources."""
        movie = _make_movie(overview="012345678")  # 9 chars
        assert _all_text_sources_sparse(movie) is True

    def test_combined_summaries_exactly_50_chars_is_not_sparse(self):
        """Combined summaries at exactly 50 chars passes threshold."""
        movie = _make_movie(
            overview="",
            plot_summaries=["a" * 25, "b" * 25],  # 50 chars total
        )
        assert _all_text_sources_sparse(movie) is False

    def test_multiple_short_synopses_each_below_threshold(self):
        """Each synopsis checked individually — multiple short ones still sparse."""
        movie = _make_movie(
            overview="",
            plot_synopses=["a" * 30, "b" * 30],  # Each < 50, but sum > 50
        )
        # Function checks each synopsis individually, not their sum
        assert _all_text_sources_sparse(movie) is True


# ---------------------------------------------------------------------------
# consolidate_maturity with multiple parental_guide_items
# ---------------------------------------------------------------------------

class TestConsolidateMaturityMultipleItems:
    def test_consolidate_maturity_multiple_items_comma_separated(self):
        """Multiple parental_guide_items are comma-separated in output."""
        result = consolidate_maturity(
            "R",
            [],
            [
                {"category": "violence", "severity": "severe"},
                {"category": "language", "severity": "moderate"},
                {"category": "nudity", "severity": "mild"},
            ],
        )
        assert result == "R — severe violence, moderate language, mild nudity"


# ---------------------------------------------------------------------------
# assess_skip_conditions Wave 2: partial Wave 1 outputs
# ---------------------------------------------------------------------------

class TestAssessSkipConditionsWave2Partial:
    def test_wave2_plot_events_present_reception_none(self):
        """When reception_output is None, review_insights_brief is None."""
        movie = _make_rich_movie()
        pe_output = _make_plot_events_output()
        result = assess_skip_conditions(
            movie,
            plot_events_output=pe_output,
            reception_output=None,
            merged_keywords=["hacker"],
            maturity_summary="R — Restricted",
        )
        # plot_analysis should still be eligible (has plot_synopsis)
        assert "plot_analysis" in result.generations_to_run

    def test_wave2_reception_present_plot_events_none(self):
        """When plot_events_output is None, plot_synopsis is None."""
        movie = _make_rich_movie()
        rec_output = _make_reception_output()
        result = assess_skip_conditions(
            movie,
            plot_events_output=None,
            reception_output=rec_output,
            merged_keywords=["hacker"],
            maturity_summary="R — Restricted",
        )
        # plot_analysis should still be eligible (has review_insights_brief)
        assert "plot_analysis" in result.generations_to_run


# ---------------------------------------------------------------------------
# check_reception: reviews exactly at 25-char threshold
# ---------------------------------------------------------------------------

class TestCheckReceptionThreshold:
    def test_check_reception_reviews_exactly_25_chars(self):
        """Combined review text of exactly 25 chars passes threshold."""
        movie = _make_movie(
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[{"text": "a" * 25}],
        )
        assert check_reception(movie) is None

    def test_check_reception_reviews_24_chars_fails(self):
        """Combined review text of 24 chars is below threshold."""
        movie = _make_movie(
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[{"text": "a" * 24}],
        )
        assert check_reception(movie) is not None


# ---------------------------------------------------------------------------
# _check_source_of_inspiration: eligible via review_insights_brief alone
# ---------------------------------------------------------------------------

class TestCheckSourceOfInspirationEligibility:
    def test_eligible_via_review_insights_brief_alone(self):
        """source_of_inspiration is eligible with only review_insights_brief."""
        assert _check_source_of_inspiration(
            [], "Critics noted the source material.", None,
        ) is None


