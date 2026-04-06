"""
Unit tests for vector text generation functions.

Tests the bridge between Movie data and embedding input text for each
of the 8 named vector spaces. Uses the Movie fixture directly (no DB).
"""

import pytest

from implementation.classes.enums import BudgetSize
from implementation.misc.helpers import normalize_string
from schemas.movie import Movie, TMDBData, IMDBData
from schemas.metadata import (
    ElevatorPitchWithJustification,
    CharacterArcWithReasoning,
    ThematicConceptWithJustification,
    PlotAnalysisOutput,
    PlotEventsOutput,
    ReceptionOutput,
    ViewerExperienceOutput,
    WatchContextOutput,
    NarrativeTechniquesOutput,
    ProductionKeywordsOutput,
    SourceOfInspirationOutput,
    TermsWithNegationsAndJustificationSection,
    TermsWithJustificationSection,
)
from movie_ingestion.final_ingestion.vector_text import (
    budget_size_to_vector_text,
    create_anchor_vector_text,
    create_plot_events_vector_text,
    create_plot_analysis_vector_text,
    create_narrative_techniques_vector_text,
    create_viewer_experience_vector_text,
    create_watch_context_vector_text,
    create_production_vector_text,
    create_reception_vector_text,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _neg_section(**overrides):
    defaults = {"justification": "placeholder", "terms": [], "negations": []}
    defaults.update(overrides)
    return TermsWithNegationsAndJustificationSection(**defaults)


def _just_section(**overrides):
    defaults = {"evidence_basis": "placeholder", "terms": []}
    defaults.update(overrides)
    return TermsWithJustificationSection(**defaults)


def _make_movie(**overrides) -> Movie:
    """Build a minimal valid Movie with targeted overrides."""
    tmdb_defaults = {
        "tmdb_id": 1,
        "imdb_id": "tt0000001",
        "title": "Spider-Man",
        "release_date": "2002-05-03",
        "duration": 121,
        "budget": 139_000_000,
        "maturity_rating": "PG-13",
    }
    imdb_defaults = {
        "tmdb_id": 1,
        "original_title": None,
        "overview": "A student gets spider-like abilities.",
        "imdb_rating": 7.4,
        "metacritic_rating": 73.0,
        "budget": None,
        "genres": ["Action", "Adventure"],
        "countries_of_origin": ["USA"],
        "production_companies": ["Columbia Pictures"],
        "filming_locations": ["New York"],
        "languages": ["English"],
        "overall_keywords": ["hero", "city"],
        "maturity_rating": None,
        "maturity_reasoning": [],
        "synopses": [],
        "plot_summaries": [],
        "plot_keywords": [],
    }

    tmdb_overrides = overrides.pop("tmdb_data", {})
    imdb_overrides = overrides.pop("imdb_data", {})

    tmdb_data = TMDBData(**{**tmdb_defaults, **tmdb_overrides})
    imdb_data = IMDBData(**{**imdb_defaults, **imdb_overrides})
    return Movie(tmdb_data=tmdb_data, imdb_data=imdb_data, **overrides)


def _make_plot_analysis(**overrides):
    defaults = dict(
        genre_signatures=["Drama", "Thriller"],
        thematic_concepts=[
            ThematicConceptWithJustification(
                explanation_and_justification="Core theme.",
                concept_label="Justice",
            ),
        ],
        elevator_pitch_with_justification=ElevatorPitchWithJustification(
            explanation_and_justification="Because it's the core.",
            elevator_pitch="Revenge in a small town",
        ),
        conflict_type=["man vs society"],
        character_arcs=[
            CharacterArcWithReasoning(
                reasoning="He changes.",
                arc_transformation_label="Redeemed",
            ),
        ],
        generalized_plot_overview="A tense drama unfolds.",
    )
    defaults.update(overrides)
    return PlotAnalysisOutput(**defaults)


def _make_viewer_experience(**section_overrides):
    section_names = [
        "emotional_palette", "tension_adrenaline", "tone_self_seriousness",
        "cognitive_complexity", "disturbance_profile", "sensory_load",
        "emotional_volatility", "ending_aftertaste",
    ]
    data = {}
    for name in section_names:
        data[name] = _neg_section(**section_overrides.get(name, {}))
    return ViewerExperienceOutput(**data)


def _make_watch_context(**section_overrides):
    data = {
        "identity_note": "sincere drama",
        "self_experience_motivations": _just_section(
            **section_overrides.get("self_experience_motivations", {})
        ),
        "external_motivations": _just_section(
            **section_overrides.get("external_motivations", {})
        ),
        "key_movie_feature_draws": _just_section(
            **section_overrides.get("key_movie_feature_draws", {})
        ),
        "watch_scenarios": _just_section(
            **section_overrides.get("watch_scenarios", {})
        ),
    }
    return WatchContextOutput(**data)


def _make_narrative_techniques(**section_overrides):
    section_names = [
        "narrative_archetype", "narrative_delivery", "pov_perspective",
        "characterization_methods", "character_arcs",
        "audience_character_perception", "information_control",
        "conflict_stakes_design", "additional_narrative_devices",
    ]
    data = {}
    for name in section_names:
        data[name] = _just_section(**section_overrides.get(name, {}))
    return NarrativeTechniquesOutput(**data)


# ---------------------------------------------------------------------------
# budget_size_to_vector_text
# ---------------------------------------------------------------------------

class TestBudgetSizeToVectorText:
    def test_small(self):
        assert budget_size_to_vector_text(BudgetSize.SMALL) == "small budget"

    def test_large(self):
        assert budget_size_to_vector_text(BudgetSize.LARGE) == "big budget, blockbuster"

    def test_none(self):
        assert budget_size_to_vector_text(None) == ""


# ---------------------------------------------------------------------------
# Anchor vector text
# ---------------------------------------------------------------------------

class TestAnchorVectorText:
    def test_minimal_movie(self):
        """Anchor text should be produced even with no metadata."""
        movie = _make_movie()
        result = create_anchor_vector_text(movie)
        assert "spider-man" in result
        assert len(result) > 0

    def test_full_movie(self):
        """All metadata sections should contribute to anchor text."""
        movie = _make_movie(
            plot_analysis_metadata=_make_plot_analysis(),
            viewer_experience_metadata=_make_viewer_experience(
                emotional_palette={"terms": ["thrilling"]},
            ),
            watch_context_metadata=_make_watch_context(
                key_movie_feature_draws={"terms": ["iconic villain"]},
            ),
            reception_metadata=ReceptionOutput(
                reception_summary="A landmark film.",
                praised_qualities=["great acting"],
            ),
            source_of_inspiration_metadata=SourceOfInspirationOutput(
                source_material=["based on comic"],
                franchise_lineage=[],
            ),
        )
        result = create_anchor_vector_text(movie)
        assert "spider-man" in result
        assert "revenge in a small town" in result  # elevator pitch
        assert "genres:" in result
        assert "themes:" in result
        assert "source material:" in result
        assert "emotional palette:" in result
        assert "key draws:" in result
        assert "reception:" in result
        assert "a landmark film." in result

    def test_uses_lowercase_for_prose(self):
        movie = _make_movie(
            plot_analysis_metadata=_make_plot_analysis(
                generalized_plot_overview="A TENSE Drama.",
            ),
        )
        result = create_anchor_vector_text(movie)
        assert "a tense drama." in result

    def test_uses_normalize_string_for_terms(self):
        movie = _make_movie(imdb_data={"overall_keywords": ["Ocean's Treasure"]})
        result = create_anchor_vector_text(movie)
        assert normalize_string("Ocean's Treasure") in result

    def test_falls_back_to_overview(self):
        """Without plot_analysis, should fall back to imdb overview."""
        movie = _make_movie(imdb_data={"overview": "A great adventure."})
        result = create_anchor_vector_text(movie)
        assert "a great adventure." in result

    def test_includes_source_material(self):
        movie = _make_movie(
            source_of_inspiration_metadata=SourceOfInspirationOutput(
                source_material=["based on novel"],
                franchise_lineage=["sequel"],
            ),
        )
        result = create_anchor_vector_text(movie)
        assert "source material:" in result
        assert "franchise position:" in result

    def test_includes_emotional_palette(self):
        movie = _make_movie(
            viewer_experience_metadata=_make_viewer_experience(
                emotional_palette={"terms": ["heartwarming"]},
            ),
        )
        result = create_anchor_vector_text(movie)
        assert "emotional palette:" in result

    def test_includes_key_draws(self):
        movie = _make_movie(
            watch_context_metadata=_make_watch_context(
                key_movie_feature_draws={"terms": ["iconic soundtrack"]},
            ),
        )
        result = create_anchor_vector_text(movie)
        assert "key draws:" in result

    def test_includes_maturity(self):
        movie = _make_movie(
            imdb_data={"maturity_reasoning": ["Intense violence"]},
        )
        result = create_anchor_vector_text(movie)
        assert "intense violence" in result

    def test_includes_reception(self):
        movie = _make_movie(
            reception_metadata=ReceptionOutput(
                reception_summary="Universally praised.",
            ),
        )
        result = create_anchor_vector_text(movie)
        assert "universally praised." in result
        assert "reception:" in result

    def test_all_optional_metadata_none(self):
        """Should still produce valid text from TMDB/IMDB data alone."""
        movie = _make_movie()
        result = create_anchor_vector_text(movie)
        assert "spider-man" in result
        assert "genres:" in result
        assert "keywords:" in result


# ---------------------------------------------------------------------------
# Plot events vector text
# ---------------------------------------------------------------------------

class TestPlotEventsVectorText:
    def test_prefers_synopsis(self):
        """synopses available → use longest synopsis."""
        movie = _make_movie(
            imdb_data={"synopses": ["Short.", "A much longer synopsis text here."]},
            plot_events_metadata=PlotEventsOutput(plot_summary="Generated text."),
        )
        result = create_plot_events_vector_text(movie)
        assert "a much longer synopsis text here." in result

    def test_falls_back_to_metadata(self):
        """No synopses → use plot_events_metadata."""
        movie = _make_movie(
            imdb_data={"synopses": []},
            plot_events_metadata=PlotEventsOutput(plot_summary="Generated Plot."),
        )
        result = create_plot_events_vector_text(movie)
        assert "generated plot." in result

    def test_falls_back_to_plot_summaries(self):
        """No synopses or metadata → use longest plot_summary."""
        movie = _make_movie(
            imdb_data={
                "synopses": [],
                "plot_summaries": ["Short.", "A longer plot summary text."],
            },
        )
        result = create_plot_events_vector_text(movie)
        assert "a longer plot summary text." in result

    def test_falls_back_to_overview(self):
        """Last resort → overview."""
        movie = _make_movie(
            imdb_data={
                "synopses": [],
                "plot_summaries": [],
                "overview": "Overview text.",
            },
        )
        result = create_plot_events_vector_text(movie)
        assert "overview text." in result

    def test_returns_none_when_empty(self):
        """No text sources at all → None."""
        movie = _make_movie(
            imdb_data={
                "synopses": [],
                "plot_summaries": [],
                "overview": None,
            },
        )
        assert create_plot_events_vector_text(movie) is None

    def test_uses_lower_not_normalize(self):
        """Should use .lower(), not normalize_string."""
        movie = _make_movie(
            imdb_data={"synopses": ["A Hero's Journey Through L.A."]},
        )
        result = create_plot_events_vector_text(movie)
        # .lower() preserves apostrophe and period
        assert "hero's" in result
        assert "l.a." in result

    def test_empty_string_in_synopses_falls_through(self):
        """Empty strings in synopses should fall through to next source."""
        movie = _make_movie(
            imdb_data={"synopses": [""], "overview": "Fallback overview."},
        )
        result = create_plot_events_vector_text(movie)
        assert "fallback overview." in result


# TestPlotEventsFallback removed — create_plot_events_vector_text_fallback
# was inlined as _plot_events_fallback_text (private). Fallback behavior is
# now tested via create_plot_events_vector_text auto-fallback tests.


# ---------------------------------------------------------------------------
# Plot analysis vector text
# ---------------------------------------------------------------------------

class TestPlotAnalysisVectorText:
    def test_merges_imdb_genres(self):
        """IMDB genres should be appended to genre_signatures."""
        meta = _make_plot_analysis(genre_signatures=["Drama", "Mystery"])
        movie = _make_movie(
            imdb_data={"genres": ["Thriller"]},
            plot_analysis_metadata=meta,
        )
        result = create_plot_analysis_vector_text(movie)
        assert normalize_string("Drama") in result
        assert normalize_string("Thriller") in result

    def test_deduplicates_genres(self):
        """Overlapping genres should not be duplicated."""
        meta = _make_plot_analysis(genre_signatures=["Action", "Drama"])
        movie = _make_movie(
            imdb_data={"genres": ["Action"]},
            plot_analysis_metadata=meta,
        )
        result = create_plot_analysis_vector_text(movie)
        # "Action" should appear once in the genre signatures section
        genre_line = [l for l in result.split("\n") if "genre signatures:" in l][0]
        assert genre_line.count(normalize_string("Action")) == 1

    def test_returns_none_without_metadata(self):
        movie = _make_movie()
        assert create_plot_analysis_vector_text(movie) is None

    def test_genre_mutation_does_not_corrupt_source(self):
        """Appending IMDB genres should not corrupt the metadata object
        for a second call with different IMDB genres."""
        meta = _make_plot_analysis(genre_signatures=["Drama", "Mystery"])
        movie = _make_movie(
            imdb_data={"genres": ["Comedy"]},
            plot_analysis_metadata=meta,
        )
        # First call appends "Comedy"
        create_plot_analysis_vector_text(movie)
        # genre_signatures now has ["Drama", "Mystery", "Comedy"] due to in-place mutation
        # A second call should not re-add "Comedy"
        result = create_plot_analysis_vector_text(movie)
        genre_line = [l for l in result.split("\n") if "genre signatures:" in l][0]
        assert genre_line.count(normalize_string("Comedy")) == 1


# ---------------------------------------------------------------------------
# Simple delegate functions
# ---------------------------------------------------------------------------

class TestNarrativeTechniquesReturnsNone:
    def test_returns_none_without_metadata(self):
        movie = _make_movie()
        assert create_narrative_techniques_vector_text(movie) is None

    def test_returns_text_with_metadata(self):
        meta = _make_narrative_techniques(
            narrative_archetype={"terms": ["hero journey"]},
        )
        movie = _make_movie(narrative_techniques_metadata=meta)
        result = create_narrative_techniques_vector_text(movie)
        assert result is not None
        assert normalize_string("hero journey") in result


class TestViewerExperienceReturnsNone:
    def test_returns_none_without_metadata(self):
        movie = _make_movie()
        assert create_viewer_experience_vector_text(movie) is None


class TestWatchContextReturnsNone:
    def test_returns_none_without_metadata(self):
        movie = _make_movie()
        assert create_watch_context_vector_text(movie) is None


# ---------------------------------------------------------------------------
# Production vector text
# ---------------------------------------------------------------------------

class TestProductionVectorText:
    def test_excludes_filming_locations_for_animation(self):
        movie = _make_movie(
            imdb_data={
                "genres": ["Animation", "Comedy"],
                "filming_locations": ["Los Angeles"],
            },
        )
        result = create_production_vector_text(movie)
        assert "filming locations" not in result

    def test_includes_filming_locations_for_live_action(self):
        movie = _make_movie(
            imdb_data={
                "genres": ["Action"],
                "filming_locations": ["New York"],
            },
        )
        result = create_production_vector_text(movie)
        assert "filming locations" in result

    def test_production_medium_animation(self):
        movie = _make_movie(imdb_data={"genres": ["Animation"]})
        result = create_production_vector_text(movie)
        assert "production medium: animation" in result

    def test_production_medium_live_action(self):
        movie = _make_movie(imdb_data={"genres": ["Drama"]})
        result = create_production_vector_text(movie)
        assert "production medium: live action" in result

    def test_default_original_screenplay(self):
        """No source_of_inspiration_metadata → 'original screenplay'."""
        movie = _make_movie()
        result = create_production_vector_text(movie)
        assert "source material: original screenplay" in result

    def test_uses_source_embedding_text(self):
        """Source metadata present → delegates to embedding_text()."""
        movie = _make_movie(
            source_of_inspiration_metadata=SourceOfInspirationOutput(
                source_material=["based on novel"],
                franchise_lineage=[],
            ),
        )
        result = create_production_vector_text(movie)
        assert "source material:" in result
        assert "original screenplay" not in result

    def test_includes_production_keywords(self):
        movie = _make_movie(
            production_keywords_metadata=ProductionKeywordsOutput(
                terms=["practical effects", "IMAX"],
            ),
        )
        result = create_production_vector_text(movie)
        assert normalize_string("practical effects") in result
        assert normalize_string("IMAX") in result


# ---------------------------------------------------------------------------
# Reception vector text
# ---------------------------------------------------------------------------

class TestReceptionVectorText:
    def test_returns_none_without_metadata(self):
        movie = _make_movie()
        assert create_reception_vector_text(movie) is None

    def test_includes_tier_label(self):
        movie = _make_movie(
            reception_metadata=ReceptionOutput(
                reception_summary="An instant classic.",
            ),
        )
        result = create_reception_vector_text(movie)
        # Movie has imdb=7.4, meta=73 → score ~71.8 → "Generally favorable reviews"
        assert "reception: generally favorable reviews" in result

    def test_delegates_to_embedding_text(self):
        """Should include the metadata's embedding_text() output."""
        movie = _make_movie(
            reception_metadata=ReceptionOutput(
                reception_summary="An instant classic.",
                praised_qualities=["great acting"],
            ),
        )
        result = create_reception_vector_text(movie)
        assert "an instant classic." in result
        assert "praised:" in result


# ---------------------------------------------------------------------------
# Token limit checking (_exceeds_token_limit)
# ---------------------------------------------------------------------------

from movie_ingestion.final_ingestion.vector_text import (
    _exceeds_token_limit,
    _CHAR_GATE_THRESHOLD,
    _EMBEDDING_TOKEN_LIMIT,
)


class TestExceedsTokenLimit:
    def test_short_text_below_char_gate_returns_false(self):
        """Text under the character gate threshold always returns False."""
        short_text = "a" * (_CHAR_GATE_THRESHOLD - 1)
        assert _exceeds_token_limit(short_text) is False

    def test_long_text_above_char_gate_but_under_token_limit(self):
        """Text above char gate but under token limit returns False."""
        # Each word is ~1 token. 5000 words ≈ 5000 tokens, well under 8191.
        # But text length > 15K chars.
        text = ("longword " * 2000)  # ~18K chars, ~2000 tokens
        assert _exceeds_token_limit(text) is False

    def test_text_exceeding_token_limit_returns_true(self):
        """Text exceeding the token limit should return True."""
        # Each unique word is ~1 token. Generate enough to exceed 8191.
        words = [f"word{i}" for i in range(9000)]
        text = " ".join(words)
        assert _exceeds_token_limit(text) is True


# ---------------------------------------------------------------------------
# create_plot_events_vector_text — auto-fallback on token overflow
# ---------------------------------------------------------------------------


class TestPlotEventsAutoFallback:
    def test_normal_path_returns_primary_text(self):
        """When primary text is short, should return it lowercased."""
        movie = _make_movie(
            imdb_data={"synopses": ["A hero saves the city from danger."]},
        )
        result = create_plot_events_vector_text(movie)
        assert "a hero saves the city from danger." in result

    def test_auto_fallback_on_token_overflow(self, mocker):
        """When primary text exceeds token limit, should fall back to shorter text."""
        # Create a movie with a very long synopsis and a shorter plot summary
        movie = _make_movie(
            imdb_data={
                "synopses": ["Very long synopsis text."],
                "plot_summaries": ["Short summary."],
                "overview": "Overview.",
            },
        )

        # Mock _exceeds_token_limit to return True on first call (primary),
        # False on second (fallback)
        call_count = [0]
        original_exceeds = _exceeds_token_limit

        def _mock_exceeds(text):
            call_count[0] += 1
            if call_count[0] == 1:
                return True  # Primary text "exceeds" the limit
            return False  # Fallback is fine

        mocker.patch(
            "movie_ingestion.final_ingestion.vector_text._exceeds_token_limit",
            side_effect=_mock_exceeds,
        )

        result = create_plot_events_vector_text(movie)
        # Should have fallen back — the fallback picks the longer of
        # plot_summaries vs generated metadata, then overview
        assert result is not None
        # The primary synopsis should NOT be the result since we forced fallback
        assert "very long synopsis text." not in result

    def test_returns_none_when_all_empty(self):
        """No text sources at all should return None."""
        movie = _make_movie(
            imdb_data={
                "synopses": [],
                "plot_summaries": [],
                "overview": None,
            },
        )
        assert create_plot_events_vector_text(movie) is None
