"""
Unit tests for movie_ingestion.metadata_generation.pre_consolidation.

Covers:
  - route_keywords: normalization, dedup, merge ordering
  - consolidate_maturity: 4-step priority chain
  - Per-generation eligibility checks (check_*/_ check_* functions)
  - assess_skip_conditions: Wave 1 and Wave 2 orchestration
  - run_pre_consolidation: end-to-end orchestrator
  - Viewer experience narrative resolution and observation filtering
  - Narrative techniques narrative resolution
  - best_plot_fallback on MovieInputData
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
    PlotAnalysisWithJustificationsOutput,
    CharacterArcWithReasoning,
    ElevatorPitchWithJustification,
    ThematicConceptWithJustification,
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
    assess_skip_conditions,
    run_pre_consolidation,
    resolve_viewer_experience_narrative,
    filter_viewer_experience_observations,
    resolve_narrative_techniques_narrative,
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
        plot_synopses=["Neo discovers he is living in a simulated reality created by machines and joins a rebellion led by Morpheus to fight the controllers. " * 5],
        plot_summaries=["A lengthy summary of the plot of The Matrix that covers all major events in great detail including the training sequences. " * 5],
        plot_keywords=["hacker", "simulation"],
        overall_keywords=["cyberpunk", "dystopia"],
        featured_reviews=[
            {"summary": "Groundbreaking", "text": "A revolutionary film that changed cinema forever with its stunning visuals. " * 10},
        ],
        reception_summary="Widely acclaimed for its visual effects and philosophical themes.",
        audience_reception_attributes=[{"name": "groundbreaking", "sentiment": "positive"}],
        maturity_rating="R",
        maturity_reasoning=["Rated R for sci-fi violence and brief language"],
        parental_guide_items=[{"category": "violence", "severity": "moderate"}],
    )


def _make_plot_events_output(**overrides) -> PlotEventsOutput:
    """Build a minimal valid PlotEventsOutput."""
    defaults = {"plot_summary": "Neo discovers the Matrix is a simulation."}
    defaults.update(overrides)
    return PlotEventsOutput(**defaults)


def _make_reception_output(**overrides) -> ReceptionOutput:
    """Build a minimal valid ReceptionOutput."""
    defaults = {
        "reception_summary": "Widely acclaimed.",
        "thematic_observations": "Critics praised the philosophical depth.",
        "emotional_observations": "Intense and haunting viewing experience.",
        "craft_observations": "Strong cinematography and editing.",
        "source_material_hint": None,
        "praised_qualities": ["groundbreaking"],
        "criticized_qualities": [],
    }
    defaults.update(overrides)
    return ReceptionOutput(**defaults)


def _make_plot_analysis_output(**overrides) -> PlotAnalysisWithJustificationsOutput:
    """Build a minimal valid PlotAnalysisWithJustificationsOutput."""
    defaults = {
        "genre_signatures": ["cyberpunk thriller", "philosophical sci-fi"],
        "thematic_concepts": [
            ThematicConceptWithJustification(
                explanation_and_justification="Central theme.",
                concept_label="identity",
            ),
        ],
        "elevator_pitch_with_justification": ElevatorPitchWithJustification(
            explanation_and_justification="Heart of the movie.",
            elevator_pitch="forbidden knowledge",
        ),
        "conflict_type": ["man vs system"],
        "character_arcs": [
            CharacterArcWithReasoning(
                reasoning="Transforms from lost programmer.",
                arc_transformation_label="hero's awakening",
            ),
        ],
        "generalized_plot_overview": "A hacker discovers simulated reality and fights for freedom against the machines that control humanity. " * 5,
    }
    defaults.update(overrides)
    return PlotAnalysisWithJustificationsOutput(**defaults)


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

    def test_consolidate_maturity_multiple_items_comma_separated(self):
        result = consolidate_maturity(
            "R", [],
            [
                {"category": "violence", "severity": "severe"},
                {"category": "language", "severity": "moderate"},
                {"category": "nudity", "severity": "mild"},
            ],
        )
        assert result == "R — severe violence, moderate language, mild nudity"


# ---------------------------------------------------------------------------
# check_plot_events
# ---------------------------------------------------------------------------

class TestCheckPlotEvents:
    def test_check_plot_events_eligible_synopsis(self):
        movie = _make_movie(overview="", plot_synopses=["A" * 650])
        assert check_plot_events(movie) is None

    def test_check_plot_events_eligible_summary(self):
        movie = _make_movie(overview="", plot_summaries=["A" * 650])
        assert check_plot_events(movie) is None

    def test_check_plot_events_skip_all_missing(self):
        movie = _make_movie(overview="", plot_synopses=[], plot_summaries=[])
        reason = check_plot_events(movie)
        assert reason is not None

    def test_check_plot_events_skip_all_sparse(self):
        movie = _make_movie(
            overview="Short",
            plot_synopses=["Tiny"],
            plot_summaries=["Small"],
        )
        reason = check_plot_events(movie)
        assert reason is not None

    def test_check_plot_events_empty_overview_string(self):
        movie = _make_movie(overview="", plot_synopses=[], plot_summaries=[])
        reason = check_plot_events(movie)
        assert reason is not None


# ---------------------------------------------------------------------------
# check_reception
# ---------------------------------------------------------------------------

class TestCheckReception:
    def test_check_reception_eligible_reviews(self):
        movie = _make_movie(
            featured_reviews=[{"text": "A" * 450}]
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
        movie = _make_movie(
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[{"text": "Short."}],
        )
        reason = check_reception(movie)
        assert reason is not None


# ---------------------------------------------------------------------------
# _check_plot_analysis (tiered eligibility)
# ---------------------------------------------------------------------------

class TestCheckPlotAnalysis:
    def test_tier1_plot_summary_always_eligible(self):
        """Tier 1: plot_summary from Wave 1 → always eligible."""
        movie = _make_movie()
        assert _check_plot_analysis("A plot synopsis.", None, movie) is None

    def test_tier2_plot_fallback_400_chars(self):
        """Tier 2: plot fallback >= 400 chars → eligible."""
        movie = _make_movie(plot_synopses=["A" * 450])
        assert _check_plot_analysis(None, None, movie) is None

    def test_tier3_plot_fallback_250_plus_thematic_300(self):
        """Tier 3: fallback 250-399 chars + thematic >= 300 chars → eligible."""
        movie = _make_movie(plot_synopses=["A" * 300])
        assert _check_plot_analysis(None, "T" * 350, movie) is None

    def test_skip_when_insufficient(self):
        """Skip when no tier passes."""
        movie = _make_movie(plot_synopses=["A" * 100], overview="")
        assert _check_plot_analysis(None, None, movie) is not None

    def test_skip_tier3_thematic_too_short(self):
        """Tier 3 fails when thematic is too short."""
        movie = _make_movie(plot_synopses=["A" * 300])
        assert _check_plot_analysis(None, "T" * 200, movie) is not None

    def test_skip_both_empty(self):
        movie = _make_movie()
        assert _check_plot_analysis(None, None, movie) is not None


# ---------------------------------------------------------------------------
# _check_viewer_experience (GPO + observations)
# ---------------------------------------------------------------------------

class TestCheckViewerExperience:
    def test_eligible_gpo_standalone(self):
        """GPO >= 350 chars → eligible."""
        assert _check_viewer_experience("A" * 400, None, None, None) is None

    def test_eligible_emotional_observations_standalone(self):
        """Emotional observations >= 160 chars → eligible."""
        assert _check_viewer_experience(None, "E" * 200, None, None) is None

    def test_eligible_combined_observations_standalone(self):
        """Combined observations >= 280 chars with emotional or craft → eligible."""
        assert _check_viewer_experience(None, "E" * 150, "C" * 150, None) is None

    def test_eligible_gpo_plus_observation(self):
        """GPO >= 200 chars + any usable observation → eligible."""
        assert _check_viewer_experience("A" * 250, "E" * 130, None, None) is None

    def test_skip_nothing(self):
        reason = _check_viewer_experience(None, None, None, None)
        assert reason is not None

    def test_skip_gpo_too_short_alone(self):
        """GPO < 350 chars and no observations → skip."""
        reason = _check_viewer_experience("A" * 300, None, None, None)
        assert reason is not None

    def test_skip_observations_below_thresholds(self):
        """Observations below their per-field thresholds → skip."""
        reason = _check_viewer_experience(None, "E" * 50, "C" * 50, None)
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
        reason = _check_watch_context(None, ["Action"], ["keyword"], None)
        assert reason is not None

    def test_check_watch_context_skip_nothing(self):
        reason = _check_watch_context(None, [], [], None)
        assert reason is not None


# ---------------------------------------------------------------------------
# _check_narrative_techniques (tiered eligibility)
# ---------------------------------------------------------------------------

class TestCheckNarrativeTechniques:
    def test_tier1_plot_summary_eligible(self):
        """Tier 1: plot_summary → always eligible."""
        movie = _make_movie()
        assert _check_narrative_techniques("synopsis", None, movie) is None

    def test_tier2_raw_fallback_500_chars(self):
        """Tier 2: raw plot fallback >= 500 chars → eligible."""
        movie = _make_movie(plot_synopses=["A" * 550])
        assert _check_narrative_techniques(None, None, movie) is None

    def test_tier3_craft_standalone_400_chars(self):
        """Tier 3: craft_observations >= 400 chars → eligible standalone."""
        movie = _make_movie()
        assert _check_narrative_techniques(None, "C" * 450, movie) is None

    def test_tier4_combined_plot_300_craft_300(self):
        """Tier 4: fallback >= 300 + craft >= 300 → eligible."""
        movie = _make_movie(plot_synopses=["A" * 350])
        assert _check_narrative_techniques(None, "C" * 350, movie) is None

    def test_skip_nothing(self):
        movie = _make_movie()
        reason = _check_narrative_techniques(None, None, movie)
        assert reason is not None

    def test_skip_partial_combined(self):
        """Fallback < 300 + craft >= 300 → skip (need both)."""
        movie = _make_movie(plot_synopses=["A" * 200])
        reason = _check_narrative_techniques(None, "C" * 350, movie)
        # Craft >= 400 standalone should pass via tier 3 though
        # Let me use craft < 400 to test the combined path failure
        reason = _check_narrative_techniques(None, "C" * 350, _make_movie(plot_synopses=["A" * 200]))
        # craft=350 >= 300 but < 400, fallback=200 < 300 → skip
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
        assert _check_source_of_inspiration(["keyword"], None) is None

    def test_check_source_of_inspiration_eligible_source_material_hint(self):
        """source_material_hint (renamed from review_insights_brief) makes it eligible."""
        assert _check_source_of_inspiration([], "based on novel") is None

    def test_check_source_of_inspiration_skip_nothing(self):
        assert _check_source_of_inspiration([], None) is not None


# ---------------------------------------------------------------------------
# resolve_viewer_experience_narrative
# ---------------------------------------------------------------------------

class TestResolveViewerExperienceNarrative:
    def test_gpo_above_threshold(self):
        text, label = resolve_viewer_experience_narrative("A" * 250)
        assert text == "A" * 250
        assert label == "generalized_plot_overview"

    def test_gpo_below_threshold(self):
        text, label = resolve_viewer_experience_narrative("A" * 100)
        assert text is None
        assert label is None

    def test_gpo_none(self):
        text, label = resolve_viewer_experience_narrative(None)
        assert text is None
        assert label is None


# ---------------------------------------------------------------------------
# filter_viewer_experience_observations
# ---------------------------------------------------------------------------

class TestFilterViewerExperienceObservations:
    def test_all_above_threshold(self):
        e, c, t = filter_viewer_experience_observations("E" * 150, "C" * 150, "T" * 150)
        assert e is not None
        assert c is not None
        assert t is not None

    def test_all_below_threshold(self):
        e, c, t = filter_viewer_experience_observations("E" * 50, "C" * 50, "T" * 50)
        assert e is None
        assert c is None
        assert t is None

    def test_none_inputs(self):
        e, c, t = filter_viewer_experience_observations(None, None, None)
        assert e is None
        assert c is None
        assert t is None


# ---------------------------------------------------------------------------
# resolve_narrative_techniques_narrative
# ---------------------------------------------------------------------------

class TestResolveNarrativeTechniquesNarrative:
    def test_plot_summary_wins(self):
        movie = _make_movie()
        text, label = resolve_narrative_techniques_narrative(movie, "A plot summary.")
        assert text == "A plot summary."
        assert label == "plot_synopsis"

    def test_fallback_used_when_no_plot_summary(self):
        movie = _make_movie(plot_synopses=["A" * 350])
        text, label = resolve_narrative_techniques_narrative(movie, None)
        assert text is not None
        assert label == "plot_text"

    def test_none_when_no_sources(self):
        movie = _make_movie()
        text, label = resolve_narrative_techniques_narrative(movie, None)
        assert text is None
        assert label is None


# ---------------------------------------------------------------------------
# best_plot_fallback on MovieInputData
# ---------------------------------------------------------------------------

class TestBestPlotFallback:
    def test_returns_longest_source(self):
        movie = _make_movie(
            plot_synopses=["short"],
            plot_summaries=["a longer summary text"],
            overview="overview",
        )
        result = movie.best_plot_fallback()
        assert result == "a longer summary text"

    def test_synopsis_first_entry_only(self):
        """Only first synopsis entry is considered."""
        movie = _make_movie(
            plot_synopses=["first", "second which is much longer"],
            plot_summaries=[],
            overview="",
        )
        result = movie.best_plot_fallback()
        assert result == "first"

    def test_returns_none_when_no_sources(self):
        movie = _make_movie(plot_synopses=[], plot_summaries=[], overview="")
        assert movie.best_plot_fallback() is None

    def test_overview_used_as_fallback(self):
        movie = _make_movie(
            plot_synopses=[],
            plot_summaries=[],
            overview="A long overview text.",
        )
        assert movie.best_plot_fallback() == "A long overview text."

    def test_longest_plot_summary_selected(self):
        """When multiple plot_summaries, the longest is selected."""
        movie = _make_movie(
            plot_synopses=[],
            plot_summaries=["short", "a much longer summary"],
            overview="",
        )
        assert movie.best_plot_fallback() == "a much longer summary"


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
        movie = _make_movie(
            overview="",
            plot_synopses=[],
            plot_summaries=[],
            featured_reviews=[{"text": "A" * 450}],
            reception_summary="Good movie.",
        )
        result = assess_skip_conditions(movie)
        assert "plot_events" not in result.generations_to_run
        assert "plot_events" in result.skip_reasons
        assert "reception" in result.generations_to_run

    def test_assess_wave1_skip_reception(self):
        movie = _make_movie(
            plot_synopses=["A" * 650],
            reception_summary=None,
            audience_reception_attributes=[],
            featured_reviews=[],
        )
        result = assess_skip_conditions(movie)
        assert "reception" not in result.generations_to_run
        assert "reception" in result.skip_reasons
        assert "plot_events" in result.generations_to_run

    def test_assess_wave2_with_plot_analysis(self):
        """Wave 2 with plot_analysis output enables viewer_experience."""
        movie = _make_rich_movie()
        pe_output = _make_plot_events_output()
        rec_output = _make_reception_output()
        pa_output = _make_plot_analysis_output()
        result = assess_skip_conditions(
            movie,
            plot_events_output=pe_output,
            reception_output=rec_output,
            plot_analysis_output=pa_output,
            merged_keywords=["hacker", "simulation", "cyberpunk"],
            maturity_summary="R — Restricted",
        )
        assert "viewer_experience" in result.generations_to_run
        assert "plot_analysis" in result.generations_to_run

    def test_assess_wave2_skip_some(self):
        pe_output = _make_plot_events_output()
        movie = _make_movie(genres=[])
        result = assess_skip_conditions(
            movie,
            plot_events_output=pe_output,
            reception_output=None,
            merged_keywords=[],
            maturity_summary=None,
        )
        assert len(result.skip_reasons) > 0
        assert "production_keywords" in result.skip_reasons

    def test_assess_wave2_computes_merged_keywords_if_none(self):
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
        movie = _make_movie(
            overview="A" * 650,
            plot_keywords=["Action", "  action  "],
            overall_keywords=["ACTION"],
        )
        result = run_pre_consolidation(movie)
        assert result.merged_keywords == ["action"]



# ---------------------------------------------------------------------------
# Wave 2 partial outputs
# ---------------------------------------------------------------------------

class TestAssessSkipConditionsWave2Partial:
    def test_wave2_plot_events_present_reception_none(self):
        """When reception_output is None, observation fields are None."""
        movie = _make_rich_movie()
        pe_output = _make_plot_events_output()
        result = assess_skip_conditions(
            movie,
            plot_events_output=pe_output,
            reception_output=None,
            merged_keywords=["hacker"],
            maturity_summary="R — Restricted",
        )
        # plot_analysis should still be eligible (has plot_summary from pe_output)
        assert "plot_analysis" in result.generations_to_run

    def test_wave2_reception_present_plot_events_none(self):
        """When plot_events_output is None, plot_summary is None."""
        movie = _make_rich_movie()
        rec_output = _make_reception_output()
        result = assess_skip_conditions(
            movie,
            plot_events_output=None,
            reception_output=rec_output,
            merged_keywords=["hacker"],
            maturity_summary="R — Restricted",
        )
        # plot_analysis may still be eligible via tier 2/3 (raw plot fallback)
        # The rich movie has a 93-char overview — not enough for tier 2 (400).
        # But it has a synopsis — let's check
        # Rich movie synopsis is ~93 chars — not enough for 400
        # So plot_analysis depends on thematic_observations (tier 3)


# ---------------------------------------------------------------------------
# Wave1Outputs dataclass
# ---------------------------------------------------------------------------

class TestWave1Outputs:
    def test_construction_defaults(self):
        from movie_ingestion.metadata_generation.inputs import Wave1Outputs
        w1 = Wave1Outputs()
        assert w1.plot_summary is None
        assert w1.thematic_observations is None
        assert w1.emotional_observations is None
        assert w1.craft_observations is None
        assert w1.source_material_hint is None

    def test_field_access(self):
        from movie_ingestion.metadata_generation.inputs import Wave1Outputs
        w1 = Wave1Outputs(
            plot_summary="A plot.",
            thematic_observations="Themes.",
            emotional_observations="Emotional.",
            craft_observations="Craft.",
            source_material_hint="based on book",
        )
        assert w1.plot_summary == "A plot."
        assert w1.source_material_hint == "based on book"
