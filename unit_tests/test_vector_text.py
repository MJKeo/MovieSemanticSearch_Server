"""
Unit tests for vector text generation functions.

Tests the bridge between Movie data and embedding input text for each
of the 8 named vector spaces. Uses the Movie fixture directly (no DB).
"""

import pytest

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
    TermsWithNegationsAndJustificationSection,
    TermsWithJustificationSection,
)
from movie_ingestion.final_ingestion.vector_text import (
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


# TestBudgetSizeToVectorText removed — budget_size_to_vector_text()
# was removed from vector_text.py.


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
        )
        result = create_anchor_vector_text(movie)
        assert "spider-man" in result
        assert "revenge in a small town" in result  # elevator pitch
        assert "genre_signatures:" in result
        assert "themes:" in result
        assert "emotional_palette:" in result
        assert "key_draws:" in result
        assert "reception_summary:" in result
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
        """genre_signatures terms should be individually normalized."""
        movie = _make_movie(
            plot_analysis_metadata=_make_plot_analysis(
                genre_signatures=["Ocean's Thriller", "Film-Noir"],
            ),
        )
        result = create_anchor_vector_text(movie)
        assert normalize_string("Ocean's Thriller") in result

    def test_falls_back_to_overview(self):
        """Without plot_analysis, should fall back to imdb overview."""
        movie = _make_movie(imdb_data={"overview": "A great adventure."})
        result = create_anchor_vector_text(movie)
        assert "identity_overview:" in result
        assert "a great adventure." in result

    def test_includes_original_title_when_different(self):
        """original_title should appear when it differs from title."""
        movie = _make_movie(imdb_data={"original_title": "El Hombre Araña"})
        result = create_anchor_vector_text(movie)
        assert "original_title:" in result

    def test_excludes_original_title_when_same(self):
        """original_title should be omitted when it matches title."""
        movie = _make_movie(imdb_data={"original_title": "Spider-Man"})
        result = create_anchor_vector_text(movie)
        assert "original_title:" not in result

    def test_includes_identity_pitch(self):
        """Elevator pitch from plot_analysis should appear as identity_pitch."""
        movie = _make_movie(plot_analysis_metadata=_make_plot_analysis())
        result = create_anchor_vector_text(movie)
        assert "identity_pitch:" in result

    def test_includes_emotional_palette(self):
        movie = _make_movie(
            viewer_experience_metadata=_make_viewer_experience(
                emotional_palette={"terms": ["heartwarming"]},
            ),
        )
        result = create_anchor_vector_text(movie)
        assert "emotional_palette:" in result

    def test_includes_key_draws(self):
        movie = _make_movie(
            watch_context_metadata=_make_watch_context(
                key_movie_feature_draws={"terms": ["iconic soundtrack"]},
            ),
        )
        result = create_anchor_vector_text(movie)
        assert "key_draws:" in result

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
        assert "reception_summary:" in result

    def test_all_optional_metadata_none(self):
        """Should still produce valid text from TMDB/IMDB data alone."""
        movie = _make_movie()
        result = create_anchor_vector_text(movie)
        assert "spider-man" in result
        # Without metadata, only title and overview appear
        assert "title:" in result


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
    def test_returns_none_without_metadata(self):
        movie = _make_movie()
        assert create_plot_analysis_vector_text(movie) is None

    def test_delegates_to_embedding_text(self):
        """Result should equal the metadata's embedding_text() output exactly."""
        meta = _make_plot_analysis()
        movie = _make_movie(plot_analysis_metadata=meta)
        result = create_plot_analysis_vector_text(movie)
        assert result == meta.embedding_text()


# ---------------------------------------------------------------------------
# Simple delegate functions
# ---------------------------------------------------------------------------

class TestNarrativeTechniquesReturnsNone:
    def test_returns_none_without_metadata(self):
        movie = _make_movie()
        assert create_narrative_techniques_vector_text(movie) is None

    def test_returns_exact_labeled_embedding_text_with_metadata(self):
        meta = _make_narrative_techniques(
            narrative_archetype={"terms": ["Hero's Journey"]},
            information_control={"terms": ["Plot Twist"]},
        )
        movie = _make_movie(narrative_techniques_metadata=meta)
        result = create_narrative_techniques_vector_text(movie)
        hero = normalize_string("Hero's Journey")
        twist = normalize_string("Plot Twist")
        assert result == (
            f"narrative_archetype: {hero}\n"
            f"information_control: {twist}"
        )


class TestViewerExperienceReturnsNone:
    def test_returns_none_without_metadata(self):
        movie = _make_movie()
        assert create_viewer_experience_vector_text(movie) is None

    def test_returns_labeled_embedding_text_with_metadata(self):
        movie = _make_movie(
            viewer_experience_metadata=_make_viewer_experience(
                emotional_palette={
                    "terms": ["Warm"],
                    "negations": ["Not Bleak"],
                },
                tension_adrenaline={"terms": ["Slow Burn Suspense"]},
            ),
        )

        assert create_viewer_experience_vector_text(movie) == (
            "emotional_palette: warm\n"
            "emotional_palette_negations: not bleak\n"
            "tension_adrenaline: slow burn suspense"
        )


class TestWatchContextReturnsNone:
    def test_returns_none_without_metadata(self):
        movie = _make_movie()
        assert create_watch_context_vector_text(movie) is None

    def test_returns_labeled_embedding_text_with_metadata(self):
        movie = _make_movie(
            watch_context_metadata=_make_watch_context(
                self_experience_motivations={"terms": ["Need a Laugh"]},
                key_movie_feature_draws={"terms": ["Great Soundtrack"]},
                watch_scenarios={"terms": ["Date Night"]},
            ),
        )

        assert create_watch_context_vector_text(movie) == (
            "self_experience_motivations: need a laugh\n"
            "key_movie_feature_draws: great soundtrack\n"
            "watch_scenarios: date night"
        )


# ---------------------------------------------------------------------------
# Production vector text
# ---------------------------------------------------------------------------

class TestProductionVectorText:
    def test_excludes_filming_locations_for_animation(self):
        """Animation movies should not include filming locations."""
        movie = _make_movie(
            imdb_data={
                "genres": ["Animation", "Comedy"],
                "filming_locations": ["Los Angeles"],
            },
        )
        result = create_production_vector_text(movie)
        assert result is None or "filming_locations" not in result

    def test_includes_filming_locations_for_live_action(self):
        movie = _make_movie(
            imdb_data={
                "genres": ["Action"],
                "filming_locations": ["New York"],
            },
        )
        result = create_production_vector_text(movie)
        assert "filming_locations:" in result

    def test_returns_none_when_no_data(self):
        """Animation movie with no techniques → None."""
        movie = _make_movie(
            imdb_data={
                "genres": ["Animation"],
                "filming_locations": [],
            },
        )
        assert create_production_vector_text(movie) is None

    def test_includes_production_techniques(self):
        """production_techniques_metadata with terms → 'production_techniques:' in result."""
        from schemas.metadata import ProductionTechniquesOutput as PTOutput
        movie = _make_movie(
            production_techniques_metadata=PTOutput(
                terms=["practical effects", "IMAX"],
            ),
        )
        result = create_production_vector_text(movie)
        assert "production_techniques:" in result

    def test_production_techniques_only(self):
        """Animation movie with techniques but no locations → result has only techniques."""
        from schemas.metadata import ProductionTechniquesOutput as PTOutput
        movie = _make_movie(
            imdb_data={"genres": ["Animation"], "filming_locations": ["LA"]},
            production_techniques_metadata=PTOutput(terms=["stop motion"]),
        )
        result = create_production_vector_text(movie)
        assert "production_techniques:" in result
        assert "filming_locations:" not in result

    def test_filming_locations_only(self):
        """Live action with locations but no techniques → result has only locations."""
        movie = _make_movie(
            imdb_data={"genres": ["Drama"], "filming_locations": ["London"]},
        )
        result = create_production_vector_text(movie)
        assert "filming_locations:" in result
        assert "production_techniques:" not in result

    def test_filming_locations_limited_to_3(self):
        """Only first 3 filming locations are included."""
        movie = _make_movie(
            imdb_data={
                "genres": ["Drama"],
                "filming_locations": ["New York", "London", "Paris", "Tokyo"],
            },
        )
        result = create_production_vector_text(movie)
        assert "new york" in result
        assert "london" in result
        assert "paris" in result
        assert "tokyo" not in result


# ---------------------------------------------------------------------------
# Reception vector text
# ---------------------------------------------------------------------------

class TestReceptionVectorText:
    def test_returns_none_without_metadata(self):
        movie = _make_movie()
        assert create_reception_vector_text(movie) is None

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
# Reception award wins text
# ---------------------------------------------------------------------------

from schemas.imdb_models import AwardNomination
from schemas.enums import AwardOutcome
from movie_ingestion.final_ingestion.vector_text import _reception_award_wins_text


class TestReceptionAwardWins:
    def test_includes_award_wins_in_reception_text(self):
        """Academy Awards win → 'major_award_wins: academy awards' in reception text."""
        movie = _make_movie(
            reception_metadata=ReceptionOutput(reception_summary="Acclaimed."),
            imdb_data={
                "awards": [
                    AwardNomination(
                        ceremony="Academy Awards, USA",
                        award_name="Oscar",
                        category="Best Picture",
                        outcome=AwardOutcome.WINNER,
                        year=2020,
                    ),
                ],
            },
        )
        result = create_reception_vector_text(movie)
        assert "major_award_wins: academy awards" in result

    def test_excludes_razzie(self):
        """Razzie wins are deliberately excluded from the prestige vector."""
        movie = _make_movie(
            imdb_data={
                "awards": [
                    AwardNomination(
                        ceremony="Razzie Awards",
                        award_name="Razzie",
                        category="Worst Picture",
                        outcome=AwardOutcome.WINNER,
                        year=2020,
                    ),
                ],
            },
        )
        result = _reception_award_wins_text(movie)
        assert result is None

    def test_excludes_nominees(self):
        """Nominee-only awards → no major_award_wins line."""
        movie = _make_movie(
            imdb_data={
                "awards": [
                    AwardNomination(
                        ceremony="Academy Awards, USA",
                        award_name="Oscar",
                        category="Best Picture",
                        outcome=AwardOutcome.NOMINEE,
                        year=2020,
                    ),
                ],
            },
        )
        result = _reception_award_wins_text(movie)
        assert result is None

    def test_deduplicates_ceremonies(self):
        """Multiple wins in the same ceremony → single ceremony entry."""
        movie = _make_movie(
            imdb_data={
                "awards": [
                    AwardNomination(
                        ceremony="Academy Awards, USA",
                        award_name="Oscar",
                        category="Best Picture",
                        outcome=AwardOutcome.WINNER,
                        year=2020,
                    ),
                    AwardNomination(
                        ceremony="Academy Awards, USA",
                        award_name="Oscar",
                        category="Best Director",
                        outcome=AwardOutcome.WINNER,
                        year=2020,
                    ),
                ],
            },
        )
        result = _reception_award_wins_text(movie)
        assert result.count("academy awards") == 1

    def test_prestige_ordering(self):
        """Academy Awards should appear before Sundance."""
        movie = _make_movie(
            imdb_data={
                "awards": [
                    AwardNomination(
                        ceremony="Sundance Film Festival",
                        award_name="Grand Jury Prize",
                        outcome=AwardOutcome.WINNER,
                        year=2020,
                    ),
                    AwardNomination(
                        ceremony="Academy Awards, USA",
                        award_name="Oscar",
                        category="Best Picture",
                        outcome=AwardOutcome.WINNER,
                        year=2020,
                    ),
                ],
            },
        )
        result = _reception_award_wins_text(movie)
        assert result.index("academy awards") < result.index("sundance")

    def test_unknown_ceremony_skipped(self):
        """Unknown ceremony strings are gracefully skipped."""
        movie = _make_movie(
            imdb_data={
                "awards": [
                    AwardNomination(
                        ceremony="Nonexistent Awards",
                        award_name="Trophy",
                        outcome=AwardOutcome.WINNER,
                        year=2020,
                    ),
                ],
            },
        )
        result = _reception_award_wins_text(movie)
        assert result is None

    def test_no_awards_returns_none(self):
        """Empty awards list → None."""
        movie = _make_movie(imdb_data={"awards": []})
        result = _reception_award_wins_text(movie)
        assert result is None


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
