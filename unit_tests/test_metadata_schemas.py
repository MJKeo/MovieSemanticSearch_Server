"""
Unit tests for movie_ingestion.metadata_generation.schemas.

Covers:
  - ReceptionOutput dual-zone structure (extraction + synthesis fields)
  - ReceptionOutput.__str__() excludes extraction-zone fields
  - ReceptionOutput field constraints and validation
  - CharacterArcWithReasoning: extra="forbid", required fields, __str__()
  - WithJustifications sub-model __str__() returns label only
  - WithJustifications __str__() parity with base variants (6 pairs)
  - PlotAnalysisWithJustificationsOutput field constraints and __str__()
  - ViewerExperienceOutput simplified schema (flat sections, no skip wrappers)
"""

import pytest
from pydantic import ValidationError

from movie_ingestion.metadata_generation.schemas import (
    PlotEventsOutput,
    ReceptionOutput,
    CharacterArcWithReasoning,
    ElevatorPitchWithJustification,
    ThematicConceptWithJustification,
    PlotAnalysisWithJustificationsOutput,
    TermsSection,
    TermsWithNegationsSection,
    TermsWithJustificationSection,
    TermsWithNegationsAndJustificationSection,
    ViewerExperienceOutput,
    ViewerExperienceWithJustificationsOutput,
    WatchContextOutput,
    WatchContextWithJustificationsOutput,
    NarrativeTechniquesOutput,
    NarrativeTechniquesWithJustificationsOutput,
    ProductionKeywordsOutput,
    ProductionKeywordsWithJustificationsOutput,
    SourceOfInspirationOutput,
    SourceOfInspirationWithJustificationsOutput,
)


# ---------------------------------------------------------------------------
# ReceptionOutput
# ---------------------------------------------------------------------------

class TestReceptionOutputStr:
    def test_reception_output_str_excludes_extraction_zone_fields(self):
        """Extraction-zone observation fields must NOT appear in __str__() output."""
        output = ReceptionOutput(
            source_material_hint="based on autobiography",
            thematic_observations="Critics noted philosophical depth and innovative themes.",
            emotional_observations="Reviewers described a haunting emotional tone.",
            craft_observations="Praised for masterful cinematography.",
            reception_summary="Widely acclaimed for visual effects.",
            praised_qualities=["groundbreaking visuals"],
            criticized_qualities=["thin plot"],
        )
        result = str(output)

        # Extraction-zone text must NOT appear
        assert "philosophical depth" not in result
        assert "haunting emotional tone" not in result
        assert "masterful cinematography" not in result
        assert "autobiography" not in result

        # Synthesis-zone fields SHOULD appear (lowercased)
        assert "widely acclaimed" in result
        assert "groundbreaking visuals" in result
        assert "thin plot" in result

    def test_reception_output_str_with_all_observations_none(self):
        """__str__() works correctly when all nullable extraction fields are None."""
        output = ReceptionOutput(
            reception_summary="A decent film.",
        )
        result = str(output)
        assert "a decent film." in result

    def test_reception_output_str_uses_reception_summary(self):
        """__str__() includes reception_summary (lowercased)."""
        output = ReceptionOutput(
            reception_summary="A Brilliant Film.",
        )
        assert "a brilliant film." in str(output)


class TestReceptionOutputObservationFields:
    def test_observation_fields_are_nullable(self):
        """thematic/emotional/craft_observations and source_material_hint accept None."""
        output = ReceptionOutput(
            reception_summary="Good movie.",
            source_material_hint=None,
            thematic_observations=None,
            emotional_observations=None,
            craft_observations=None,
        )
        assert output.source_material_hint is None
        assert output.thematic_observations is None
        assert output.emotional_observations is None
        assert output.craft_observations is None

    def test_observation_fields_accept_strings(self):
        """Observation fields accept non-None string values."""
        output = ReceptionOutput(
            reception_summary="Good movie.",
            source_material_hint="based on book",
            thematic_observations="Themes of loss.",
            emotional_observations="Melancholic tone.",
            craft_observations="Strong editing.",
        )
        assert output.source_material_hint == "based on book"
        assert output.thematic_observations == "Themes of loss."


class TestReceptionOutputExtraForbid:
    def test_old_field_names_raise_validation_error(self):
        """Passing removed field names (new_reception_summary, praise_attributes, etc.) raises."""
        with pytest.raises(ValidationError):
            ReceptionOutput(
                new_reception_summary="Good movie.",
                review_insights_brief="Critics liked it.",
            )

        with pytest.raises(ValidationError):
            ReceptionOutput(
                reception_summary="Good movie.",
                praise_attributes=["good"],
            )

        with pytest.raises(ValidationError):
            ReceptionOutput(
                reception_summary="Good movie.",
                complaint_attributes=["bad"],
            )


class TestReceptionOutputConstraints:
    def test_reception_summary_is_required(self):
        """Omitting reception_summary raises ValidationError."""
        with pytest.raises(ValidationError):
            ReceptionOutput(
                praised_qualities=["good"],
            )

    def test_praised_qualities_max_length_6(self):
        """praised_qualities accepts 6 items but rejects 7."""
        # 6 items should be fine
        output = ReceptionOutput(
            reception_summary="Good.",
            praised_qualities=["a", "b", "c", "d", "e", "f"],
        )
        assert len(output.praised_qualities) == 6

        # 7 items should fail
        with pytest.raises(ValidationError):
            ReceptionOutput(
                reception_summary="Good.",
                praised_qualities=["a", "b", "c", "d", "e", "f", "g"],
            )

    def test_criticized_qualities_max_length_6(self):
        """criticized_qualities accepts 6 items but rejects 7."""
        output = ReceptionOutput(
            reception_summary="Good.",
            criticized_qualities=["a", "b", "c", "d", "e", "f"],
        )
        assert len(output.criticized_qualities) == 6

        with pytest.raises(ValidationError):
            ReceptionOutput(
                reception_summary="Good.",
                criticized_qualities=["a", "b", "c", "d", "e", "f", "g"],
            )


# ---------------------------------------------------------------------------
# CharacterArcWithReasoning
# ---------------------------------------------------------------------------

class TestCharacterArcWithReasoning:
    def test_requires_reasoning_and_label(self):
        """Both reasoning and arc_transformation_label are required."""
        with pytest.raises(ValidationError):
            CharacterArcWithReasoning(
                arc_transformation_label="hero's awakening",
            )
        with pytest.raises(ValidationError):
            CharacterArcWithReasoning(
                reasoning="Discovers true power.",
            )

    def test_extra_forbid(self):
        with pytest.raises(ValidationError):
            CharacterArcWithReasoning(
                reasoning="Discovers true power.",
                arc_transformation_label="hero's awakening",
                extra_field="not allowed",
            )

    def test_str_returns_label_only(self):
        arc = CharacterArcWithReasoning(
            reasoning="Discovers his true power and becomes The One.",
            arc_transformation_label="hero's awakening",
        )
        assert str(arc) == "hero's awakening"
        # Reasoning should NOT appear in str()
        assert "discovers" not in str(arc).lower()

    def test_old_character_arc_fields_rejected(self):
        """Old CharacterArc fields (character_name, arc_transformation_description) are rejected."""
        with pytest.raises(ValidationError):
            CharacterArcWithReasoning(
                character_name="Neo",
                reasoning="Discovers true power.",
                arc_transformation_label="hero's awakening",
            )


# ---------------------------------------------------------------------------
# WithJustifications sub-model __str__() tests
# ---------------------------------------------------------------------------

class TestElevatorPitchWithJustificationStr:
    def test_str_returns_pitch_only(self):
        obj = ElevatorPitchWithJustification(
            explanation_and_justification="This concept captures the essence.",
            elevator_pitch="forbidden knowledge",
        )
        assert str(obj) == "forbidden knowledge"
        assert "essence" not in str(obj)


class TestThematicConceptWithJustificationStr:
    def test_str_returns_label_only(self):
        obj = ThematicConceptWithJustification(
            explanation_and_justification="This theme runs through every scene.",
            concept_label="identity and transformation",
        )
        assert str(obj) == "identity and transformation"
        assert "runs through" not in str(obj)


# ---------------------------------------------------------------------------
# PlotAnalysisWithJustificationsOutput __str__() and constraints
# ---------------------------------------------------------------------------

def _make_plot_analysis_output(**overrides):
    """Build a minimal valid PlotAnalysisWithJustificationsOutput."""
    defaults = dict(
        generalized_plot_overview="A hacker discovers the truth about simulated reality.",
        genre_signatures=["cyberpunk thriller", "philosophical sci-fi"],
        conflict_type=["man vs system"],
        character_arcs=[
            CharacterArcWithReasoning(
                reasoning="Transforms from a lost programmer into a messianic figure.",
                arc_transformation_label="hero's awakening",
            ),
        ],
        elevator_pitch_with_justification=ElevatorPitchWithJustification(
            explanation_and_justification="This is the heart of the movie.",
            elevator_pitch="forbidden knowledge",
        ),
        thematic_concepts=[
            ThematicConceptWithJustification(
                explanation_and_justification="Central to the narrative.",
                concept_label="identity",
            ),
            ThematicConceptWithJustification(
                explanation_and_justification="Runs through every choice.",
                concept_label="free will",
            ),
        ],
    )
    defaults.update(overrides)
    return PlotAnalysisWithJustificationsOutput(**defaults)


class TestPlotAnalysisWithJustificationsStr:
    def test_str_includes_all_embedded_fields(self):
        """__str__() includes all label fields, lowercased."""
        output = _make_plot_analysis_output()
        result = str(output)
        assert "forbidden knowledge" in result
        assert "cyberpunk thriller" in result
        assert "man vs system" in result
        assert "hero's awakening" in result
        assert "identity" in result
        assert "free will" in result
        assert "a hacker discovers" in result

    def test_str_excludes_justification_text(self):
        output = _make_plot_analysis_output(
            elevator_pitch_with_justification=ElevatorPitchWithJustification(
                explanation_and_justification="JUSTIFICATION_MARKER_PITCH",
                elevator_pitch="forbidden knowledge",
            ),
            thematic_concepts=[
                ThematicConceptWithJustification(
                    explanation_and_justification="JUSTIFICATION_MARKER_THEME",
                    concept_label="identity",
                ),
            ],
            character_arcs=[
                CharacterArcWithReasoning(
                    reasoning="JUSTIFICATION_MARKER_ARC",
                    arc_transformation_label="hero's awakening",
                ),
            ],
        )
        result = str(output)
        assert "JUSTIFICATION_MARKER_PITCH" not in result
        assert "JUSTIFICATION_MARKER_THEME" not in result
        assert "JUSTIFICATION_MARKER_ARC" not in result

    def test_str_with_empty_optional_lists(self):
        """__str__() handles empty conflict_type, character_arcs, thematic_concepts."""
        output = _make_plot_analysis_output(
            conflict_type=[],
            character_arcs=[],
            thematic_concepts=[],
        )
        result = str(output)
        # Should still contain overview, pitch, genre_signatures
        assert "a hacker discovers" in result
        assert "forbidden knowledge" in result
        assert "cyberpunk thriller" in result
        # Empty lists should not produce empty entries
        assert "man vs system" not in result
        assert "hero's awakening" not in result

    def test_conflict_type_joined_lowercased(self):
        """conflict_type entries are comma-joined and lowercased."""
        output = _make_plot_analysis_output(
            conflict_type=["Man vs Nature", "Man vs Self"],
        )
        result = str(output)
        assert "man vs nature, man vs self" in result

    def test_conflict_type_empty_produces_no_entry(self):
        """Empty conflict_type list produces no conflict entry in __str__()."""
        output = _make_plot_analysis_output(conflict_type=[])
        result = str(output)
        assert "conflict" not in result.lower() or "man vs" not in result


class TestPlotAnalysisConstraints:
    def test_genre_signatures_min_length_2(self):
        """genre_signatures requires min_length=2."""
        with pytest.raises(ValidationError):
            _make_plot_analysis_output(genre_signatures=["only one"])

    def test_genre_signatures_max_length_6(self):
        """genre_signatures accepts up to 6."""
        output = _make_plot_analysis_output(
            genre_signatures=["a", "b", "c", "d", "e", "f"],
        )
        assert len(output.genre_signatures) == 6

        with pytest.raises(ValidationError):
            _make_plot_analysis_output(
                genre_signatures=["a", "b", "c", "d", "e", "f", "g"],
            )

    def test_conflict_type_max_length_2(self):
        """conflict_type accepts 0-2 entries."""
        # 0 is fine
        output = _make_plot_analysis_output(conflict_type=[])
        assert len(output.conflict_type) == 0
        # 2 is fine
        output = _make_plot_analysis_output(conflict_type=["a", "b"])
        assert len(output.conflict_type) == 2
        # 3 is rejected
        with pytest.raises(ValidationError):
            _make_plot_analysis_output(conflict_type=["a", "b", "c"])

    def test_character_arcs_min_0_max_3(self):
        """character_arcs accepts 0-3 entries."""
        # 0 is fine (min_length=0)
        output = _make_plot_analysis_output(character_arcs=[])
        assert len(output.character_arcs) == 0
        # 3 is fine
        arcs = [
            CharacterArcWithReasoning(reasoning=f"Reason {i}.", arc_transformation_label=f"arc{i}")
            for i in range(3)
        ]
        output = _make_plot_analysis_output(character_arcs=arcs)
        assert len(output.character_arcs) == 3
        # 4 is rejected
        arcs_4 = arcs + [CharacterArcWithReasoning(reasoning="Extra.", arc_transformation_label="extra")]
        with pytest.raises(ValidationError):
            _make_plot_analysis_output(character_arcs=arcs_4)

    def test_thematic_concepts_min_0_max_5(self):
        """thematic_concepts accepts 0-5 entries."""
        output = _make_plot_analysis_output(thematic_concepts=[])
        assert len(output.thematic_concepts) == 0

        concepts = [
            ThematicConceptWithJustification(
                explanation_and_justification=f"Reason {i}.",
                concept_label=f"concept{i}",
            )
            for i in range(5)
        ]
        output = _make_plot_analysis_output(thematic_concepts=concepts)
        assert len(output.thematic_concepts) == 5

        concepts_6 = concepts + [
            ThematicConceptWithJustification(
                explanation_and_justification="Extra.",
                concept_label="extra",
            ),
        ]
        with pytest.raises(ValidationError):
            _make_plot_analysis_output(thematic_concepts=concepts_6)

    def test_extra_forbid_rejects_old_fields(self):
        """Old field names (core_concept_label, conflict_scale, themes_primary, etc.) are rejected."""
        with pytest.raises(ValidationError):
            PlotAnalysisWithJustificationsOutput(
                core_concept_label="test",
                genre_signatures=["a", "b"],
                conflict_type=[],
                character_arcs=[],
                thematic_concepts=[],
                elevator_pitch_with_justification=ElevatorPitchWithJustification(
                    explanation_and_justification="j", elevator_pitch="p",
                ),
                generalized_plot_overview="Overview.",
            )


# ---------------------------------------------------------------------------
# ViewerExperience __str__() parity
# ---------------------------------------------------------------------------

def _make_terms_section(terms=None, negations=None):
    """Build a TermsWithNegationsSection."""
    return TermsWithNegationsSection(
        terms=terms or [],
        negations=negations or [],
    )


def _make_terms_j_section(terms=None, negations=None, justification="Because."):
    """Build a TermsWithNegationsAndJustificationSection."""
    return TermsWithNegationsAndJustificationSection(
        justification=justification,
        terms=terms or [],
        negations=negations or [],
    )


_VE_TERMS = ["tense", "thrilling"]
_VE_NEGATIONS = ["not boring"]


class TestViewerExperienceWithJustificationsStrParity:
    def test_str_parity_with_viewer_experience(self):
        """All 8 sections use flat TermsWithNegationsSection; str() matches between variants."""
        base = ViewerExperienceOutput(
            emotional_palette=_make_terms_section(_VE_TERMS, _VE_NEGATIONS),
            tension_adrenaline=_make_terms_section(["high stakes"]),
            tone_self_seriousness=_make_terms_section(["dead serious"]),
            cognitive_complexity=_make_terms_section(["mind-bending"]),
            disturbance_profile=_make_terms_section(["disturbing imagery"]),
            sensory_load=_make_terms_section(["loud"]),
            emotional_volatility=_make_terms_section(["whiplash"]),
            ending_aftertaste=_make_terms_section(["lingering dread"]),
        )
        with_j = ViewerExperienceWithJustificationsOutput(
            emotional_palette=_make_terms_j_section(_VE_TERMS, _VE_NEGATIONS),
            tension_adrenaline=_make_terms_j_section(["high stakes"]),
            tone_self_seriousness=_make_terms_j_section(["dead serious"]),
            cognitive_complexity=_make_terms_j_section(["mind-bending"]),
            disturbance_profile=_make_terms_j_section(["disturbing imagery"]),
            sensory_load=_make_terms_j_section(["loud"]),
            emotional_volatility=_make_terms_j_section(["whiplash"]),
            ending_aftertaste=_make_terms_j_section(["lingering dread"]),
        )
        assert str(base) == str(with_j)

    def test_str_excludes_justification_text(self):
        with_j = ViewerExperienceWithJustificationsOutput(
            emotional_palette=_make_terms_j_section(
                ["tense"], justification="VE_MARKER"
            ),
            tension_adrenaline=_make_terms_j_section(),
            tone_self_seriousness=_make_terms_j_section(),
            cognitive_complexity=_make_terms_j_section(),
            disturbance_profile=_make_terms_j_section(),
            sensory_load=_make_terms_j_section(),
            emotional_volatility=_make_terms_j_section(),
            ending_aftertaste=_make_terms_j_section(),
        )
        assert "VE_MARKER" not in str(with_j)

    def test_str_parity_with_empty_sections(self):
        """Empty sections produce identical str() between base and justifications variants."""
        base = ViewerExperienceOutput(
            emotional_palette=_make_terms_section(["warm"]),
            tension_adrenaline=_make_terms_section(),
            tone_self_seriousness=_make_terms_section(),
            cognitive_complexity=_make_terms_section(),
            disturbance_profile=_make_terms_section(),
            sensory_load=_make_terms_section(),
            emotional_volatility=_make_terms_section(),
            ending_aftertaste=_make_terms_section(),
        )
        with_j = ViewerExperienceWithJustificationsOutput(
            emotional_palette=_make_terms_j_section(["warm"]),
            tension_adrenaline=_make_terms_j_section(),
            tone_self_seriousness=_make_terms_j_section(),
            cognitive_complexity=_make_terms_j_section(),
            disturbance_profile=_make_terms_j_section(),
            sensory_load=_make_terms_j_section(),
            emotional_volatility=_make_terms_j_section(),
            ending_aftertaste=_make_terms_j_section(),
        )
        assert str(base) == str(with_j)


# ---------------------------------------------------------------------------
# WatchContext __str__() parity
# ---------------------------------------------------------------------------

def _make_terms_only_section(terms=None):
    return TermsSection(terms=terms or [])


def _make_terms_only_j_section(terms=None, justification="Because."):
    return TermsWithJustificationSection(
        justification=justification,
        terms=terms or [],
    )


class TestWatchContextWithJustificationsStrParity:
    def test_str_parity_with_watch_context(self):
        base = WatchContextOutput(
            self_experience_motivations=_make_terms_only_section(["escape"]),
            external_motivations=_make_terms_only_section(["date night"]),
            key_movie_feature_draws=_make_terms_only_section(["star cast"]),
            watch_scenarios=_make_terms_only_section(["rainy day"]),
        )
        with_j = WatchContextWithJustificationsOutput(
            self_experience_motivations=_make_terms_only_j_section(["escape"]),
            external_motivations=_make_terms_only_j_section(["date night"]),
            key_movie_feature_draws=_make_terms_only_j_section(["star cast"]),
            watch_scenarios=_make_terms_only_j_section(["rainy day"]),
        )
        assert str(base) == str(with_j)

    def test_str_excludes_justification_text(self):
        with_j = WatchContextWithJustificationsOutput(
            self_experience_motivations=_make_terms_only_j_section(
                ["escape"], justification="WC_MARKER"
            ),
            external_motivations=_make_terms_only_j_section(),
            key_movie_feature_draws=_make_terms_only_j_section(),
            watch_scenarios=_make_terms_only_j_section(),
        )
        assert "WC_MARKER" not in str(with_j)


# ---------------------------------------------------------------------------
# NarrativeTechniques __str__() parity
# ---------------------------------------------------------------------------

# Current field order (cognitive scaffolding): archetype, delivery,
# devices, pov, characterization, arcs, perception, info control,
# stakes, thematic, meta
_NT_SECTION_NAMES = [
    "narrative_archetype", "narrative_delivery", "additional_plot_devices",
    "pov_perspective", "characterization_methods", "character_arcs",
    "audience_character_perception", "information_control",
    "conflict_stakes_design", "thematic_delivery", "meta_techniques",
]


class TestNarrativeTechniquesWithJustificationsStrParity:
    def test_str_parity_with_narrative_techniques(self):
        sections_base = {
            name: _make_terms_only_section(["term1"])
            for name in _NT_SECTION_NAMES
        }
        sections_j = {
            name: _make_terms_only_j_section(["term1"])
            for name in _NT_SECTION_NAMES
        }
        base = NarrativeTechniquesOutput(**sections_base)
        with_j = NarrativeTechniquesWithJustificationsOutput(**sections_j)
        assert str(base) == str(with_j)

    def test_str_excludes_justification_text(self):
        sections = {
            name: _make_terms_only_j_section(
                ["term1"], justification="NT_MARKER"
            )
            for name in _NT_SECTION_NAMES
        }
        with_j = NarrativeTechniquesWithJustificationsOutput(**sections)
        assert "NT_MARKER" not in str(with_j)


# ---------------------------------------------------------------------------
# ProductionKeywords __str__() parity
# ---------------------------------------------------------------------------

class TestProductionKeywordsWithJustificationsStrParity:
    def test_str_parity_with_production_keywords(self):
        base = ProductionKeywordsOutput(terms=["CGI", "IMAX"])
        with_j = ProductionKeywordsWithJustificationsOutput(
            justification="These are production terms.",
            terms=["CGI", "IMAX"],
        )
        assert str(base) == str(with_j)

    def test_str_excludes_justification_text(self):
        with_j = ProductionKeywordsWithJustificationsOutput(
            justification="PK_MARKER",
            terms=["CGI"],
        )
        assert "PK_MARKER" not in str(with_j)


# ---------------------------------------------------------------------------
# SourceOfInspiration __str__() parity
# ---------------------------------------------------------------------------

class TestSourceOfInspirationWithJustificationsStrParity:
    def test_str_parity_with_source_of_inspiration(self):
        base = SourceOfInspirationOutput(
            sources_of_inspiration=["original screenplay"],
            production_mediums=["live-action", "CGI"],
        )
        with_j = SourceOfInspirationWithJustificationsOutput(
            justification="Based on original work.",
            sources_of_inspiration=["original screenplay"],
            production_mediums=["live-action", "CGI"],
        )
        assert str(base) == str(with_j)

    def test_str_excludes_justification_text(self):
        with_j = SourceOfInspirationWithJustificationsOutput(
            justification="SOI_MARKER",
            sources_of_inspiration=["novel"],
            production_mediums=["animation"],
        )
        assert "SOI_MARKER" not in str(with_j)


# ---------------------------------------------------------------------------
# PlotEventsOutput.__str__()
# ---------------------------------------------------------------------------

class TestPlotEventsOutputStr:
    def test_plot_events_str_returns_lowercased_summary(self):
        """PlotEventsOutput.__str__() returns lowercased plot_summary."""
        output = PlotEventsOutput(
            plot_summary="Neo DISCOVERS the Matrix.",
        )
        result = str(output)
        assert result == "neo discovers the matrix."

    def test_plot_events_output_only_has_plot_summary_field(self):
        """PlotEventsOutput has exactly one user-facing field: plot_summary."""
        fields = set(PlotEventsOutput.model_fields.keys())
        assert fields == {"plot_summary"}

    def test_plot_events_output_rejects_setting_field(self):
        """PlotEventsOutput(plot_summary=..., setting=...) raises ValidationError."""
        with pytest.raises(ValidationError):
            PlotEventsOutput(plot_summary="A plot.", setting="New York")

    def test_plot_events_output_rejects_major_characters_field(self):
        """PlotEventsOutput(plot_summary=..., major_characters=[]) raises ValidationError."""
        with pytest.raises(ValidationError):
            PlotEventsOutput(plot_summary="A plot.", major_characters=[])


# ---------------------------------------------------------------------------
# ViewerExperienceOutput.__str__() independent content correctness
# ---------------------------------------------------------------------------

class TestViewerExperienceOutputStrContent:
    def test_viewer_experience_str_comma_separated_and_lowercased(self):
        """ViewerExperienceOutput.__str__() returns comma-separated, lowercased terms."""
        output = ViewerExperienceOutput(
            emotional_palette=_make_terms_section(["Warm", "Bittersweet"]),
            tension_adrenaline=_make_terms_section(["Low Stakes"]),
            tone_self_seriousness=_make_terms_section(),
            cognitive_complexity=_make_terms_section(),
            disturbance_profile=_make_terms_section(),
            sensory_load=_make_terms_section(),
            emotional_volatility=_make_terms_section(),
            ending_aftertaste=_make_terms_section(),
        )
        result = str(output)
        assert "warm" in result
        assert "bittersweet" in result
        assert "low stakes" in result
        # Comma-separated, not newline-separated
        assert "\n" not in result
        assert ", " in result

    def test_viewer_experience_all_sections_flat(self):
        """All 8 sections are flat TermsWithNegationsSection (no OptionalTermsWithNegationsSection)."""
        # This should work without any wrapper — all sections take flat sections
        output = ViewerExperienceOutput(
            emotional_palette=_make_terms_section(["tense"]),
            tension_adrenaline=_make_terms_section(),
            tone_self_seriousness=_make_terms_section(),
            cognitive_complexity=_make_terms_section(),
            disturbance_profile=_make_terms_section(),
            sensory_load=_make_terms_section(),
            emotional_volatility=_make_terms_section(),
            ending_aftertaste=_make_terms_section(),
        )
        assert "tense" in str(output)

    def test_viewer_experience_rejects_optional_wrapper(self):
        """ViewerExperienceOutput rejects OptionalTermsWithNegationsSection for its fields."""
        from movie_ingestion.metadata_generation.schemas import OptionalTermsWithNegationsSection
        optional_section = OptionalTermsWithNegationsSection(
            should_skip=False,
            section_data=_make_terms_section(["term"]),
        )
        with pytest.raises(ValidationError):
            ViewerExperienceOutput(
                emotional_palette=_make_terms_section(),
                tension_adrenaline=_make_terms_section(),
                tone_self_seriousness=_make_terms_section(),
                cognitive_complexity=_make_terms_section(),
                disturbance_profile=optional_section,
                sensory_load=_make_terms_section(),
                emotional_volatility=_make_terms_section(),
                ending_aftertaste=_make_terms_section(),
            )


# ---------------------------------------------------------------------------
# WatchContextOutput.__str__() independent content correctness
# ---------------------------------------------------------------------------

class TestWatchContextOutputStrContent:
    def test_watch_context_str_comma_separated(self):
        """WatchContextOutput.__str__() returns comma-separated, lowercased terms."""
        output = WatchContextOutput(
            self_experience_motivations=_make_terms_only_section(["Escape"]),
            external_motivations=_make_terms_only_section(["Date Night"]),
            key_movie_feature_draws=_make_terms_only_section(),
            watch_scenarios=_make_terms_only_section(),
        )
        result = str(output)
        assert "escape, date night" == result


# ---------------------------------------------------------------------------
# NarrativeTechniquesOutput.__str__() independent content correctness
# ---------------------------------------------------------------------------

class TestNarrativeTechniquesOutputStrContent:
    def test_narrative_techniques_str_all_11_sections_contribute(self):
        """NarrativeTechniquesOutput.__str__() includes terms from all 11 sections."""
        sections = {
            name: _make_terms_only_section([f"{name}_term"])
            for name in _NT_SECTION_NAMES
        }
        output = NarrativeTechniquesOutput(**sections)
        result = str(output)
        for name in _NT_SECTION_NAMES:
            assert f"{name}_term" in result

    def test_narrative_techniques_field_order_is_cognitive_scaffolding(self):
        """Fields are ordered: archetype, delivery, devices, pov, ..., meta."""
        field_names = list(NarrativeTechniquesOutput.model_fields.keys())
        assert field_names[0] == "narrative_archetype"
        assert field_names[1] == "narrative_delivery"
        assert field_names[2] == "additional_plot_devices"
        assert field_names[-1] == "meta_techniques"


# ---------------------------------------------------------------------------
# ProductionKeywordsOutput.__str__() with empty terms
# ---------------------------------------------------------------------------

class TestProductionKeywordsOutputStrEmpty:
    def test_production_keywords_str_empty_terms(self):
        """ProductionKeywordsOutput.__str__() returns '' when terms is empty."""
        output = ProductionKeywordsOutput(terms=[])
        assert str(output) == ""


# ---------------------------------------------------------------------------
# SourceOfInspirationOutput.__str__() with empty lists
# ---------------------------------------------------------------------------

class TestSourceOfInspirationOutputStrEmpty:
    def test_source_of_inspiration_str_empty_lists(self):
        """SourceOfInspirationOutput.__str__() returns '' when both lists empty."""
        output = SourceOfInspirationOutput(
            sources_of_inspiration=[],
            production_mediums=[],
        )
        assert str(output) == ""


# ---------------------------------------------------------------------------
# TermsSection / TermsWithNegationsSection extra="forbid"
# ---------------------------------------------------------------------------

class TestSectionExtraForbid:
    def test_terms_section_extra_forbid(self):
        with pytest.raises(ValidationError):
            TermsSection(terms=["a"], extra_field="not allowed")

    def test_terms_with_negations_section_extra_forbid(self):
        with pytest.raises(ValidationError):
            TermsWithNegationsSection(
                terms=["a"], negations=[], extra_field="not allowed",
            )


# ---------------------------------------------------------------------------
# ReceptionOutput.__str__() with empty attributes
# ---------------------------------------------------------------------------

class TestReceptionOutputStrEmptyAttributes:
    def test_reception_str_empty_praised_and_criticized(self):
        """ReceptionOutput.__str__() handles empty quality lists gracefully."""
        output = ReceptionOutput(
            reception_summary="A Fine Film.",
            praised_qualities=[],
            criticized_qualities=[],
        )
        result = str(output)
        # Only the summary should appear
        assert "a fine film." in result
        # No empty comma-separated sections
        assert result.strip() == "a fine film."
