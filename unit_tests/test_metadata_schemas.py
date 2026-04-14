"""
Unit tests for movie_ingestion.metadata_generation.schemas.

Covers:
  - ReceptionOutput dual-zone structure (extraction + synthesis fields)
  - ReceptionOutput.__str__() excludes extraction-zone fields
  - ReceptionOutput field constraints and validation
  - CharacterArcWithReasoning: extra="forbid", required fields, __str__()
  - WithJustifications sub-model __str__() returns label only
  - PlotAnalysisOutput field constraints and __str__()
  - ViewerExperienceOutput __str__() excludes justification text
  - WatchContextOutput __str__() excludes identity_note and evidence_basis
  - NarrativeTechniquesOutput __str__() excludes justification text
"""

import pytest
from pydantic import ValidationError

from schemas.metadata import (
    PlotEventsOutput,
    ReceptionOutput,
    CharacterArcWithReasoning,
    ElevatorPitchWithJustification,
    ThematicConceptWithJustification,
    PlotAnalysisOutput,
    TermsSection,
    TermsWithNegationsSection,
    TermsWithJustificationSection,
    TermsWithNegationsAndJustificationSection,
    ViewerExperienceOutput,
    WatchContextOutput,
    NarrativeTechniquesOutput,
    ProductionKeywordsOutput,
    SourceOfInspirationOutput,
    SourceMaterialV2Output,
    EmbeddableOutput,
)
from schemas.enums import LineagePosition, SourceMaterialType


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
# PlotAnalysisOutput __str__() and constraints
# ---------------------------------------------------------------------------

def _make_plot_analysis_output(**overrides):
    """Build a minimal valid PlotAnalysisOutput."""
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
    return PlotAnalysisOutput(**defaults)


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
            PlotAnalysisOutput(
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
# ViewerExperienceOutput __str__() tests
# ---------------------------------------------------------------------------

def _make_terms_j_section(terms=None, negations=None, justification="Because."):
    """Build a TermsWithNegationsAndJustificationSection."""
    return TermsWithNegationsAndJustificationSection(
        justification=justification,
        terms=terms or [],
        negations=negations or [],
    )


class TestViewerExperienceOutputStr:
    def test_str_excludes_justification_text(self):
        output = ViewerExperienceOutput(
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
        assert "VE_MARKER" not in str(output)


# ---------------------------------------------------------------------------
# WatchContext __str__() parity
# ---------------------------------------------------------------------------

def _make_terms_only_section(terms=None):
    return TermsSection(terms=terms or [])


def _make_terms_only_j_section(terms=None, evidence_basis="Because."):
    return TermsWithJustificationSection(
        evidence_basis=evidence_basis,
        terms=terms or [],
    )


# TestWatchContextWithJustificationsStrParity removed —
# WatchContextWithJustificationsOutput was deleted from schemas/metadata.py.


# ---------------------------------------------------------------------------
# NarrativeTechniques __str__() parity
# ---------------------------------------------------------------------------

# Current field order (cognitive scaffolding): archetype, delivery,
# pov, characterization, arcs, perception, info control,
# stakes, additional devices (catchall last)
_NT_SECTION_NAMES = [
    "narrative_archetype", "narrative_delivery",
    "pov_perspective", "characterization_methods", "character_arcs",
    "audience_character_perception", "information_control",
    "conflict_stakes_design", "additional_narrative_devices",
]


# TestNarrativeTechniquesWithJustificationsStrParity removed —
# NarrativeTechniquesWithJustificationsOutput was deleted from schemas/metadata.py.


# ---------------------------------------------------------------------------
# ProductionKeywords __str__() parity
# ---------------------------------------------------------------------------

# TestProductionKeywordsWithJustificationsStrParity removed —
# ProductionKeywordsWithJustificationsOutput was deleted from schemas/metadata.py.


# ---------------------------------------------------------------------------
# SourceOfInspiration __str__() parity
# ---------------------------------------------------------------------------

# TestSourceOfInspirationWithJustificationsStrParity removed —
# SourceOfInspirationWithJustificationsOutput was deleted from schemas/metadata.py.


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

    # test_viewer_experience_rejects_optional_wrapper removed —
    # OptionalTermsWithNegationsSection was deleted from schemas/metadata.py.


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
    def test_narrative_techniques_str_all_9_sections_contribute(self):
        """NarrativeTechniquesOutput.__str__() includes terms from all 9 sections."""
        sections = {
            name: _make_terms_only_section([f"{name}_term"])
            for name in _NT_SECTION_NAMES
        }
        output = NarrativeTechniquesOutput(**sections)
        result = str(output)
        for name in _NT_SECTION_NAMES:
            assert f"{name}_term" in result

    def test_narrative_techniques_field_order_is_cognitive_scaffolding(self):
        """Fields are ordered: archetype, delivery, pov, ..., additional_narrative_devices (catchall last)."""
        field_names = list(NarrativeTechniquesOutput.model_fields.keys())
        assert field_names[0] == "narrative_archetype"
        assert field_names[1] == "narrative_delivery"
        assert field_names[2] == "pov_perspective"
        assert field_names[-1] == "additional_narrative_devices"


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

class TestSourceOfInspirationOldFieldNamesRejected:
    def test_source_of_inspiration_rejects_old_field_names(self):
        """Old field names (sources_of_inspiration, production_mediums) are rejected by extra='forbid'."""
        with pytest.raises(ValidationError):
            SourceOfInspirationOutput(
                sources_of_inspiration=["original screenplay"],
                production_mediums=["live-action"],
            )

    # test_source_of_inspiration_with_reasoning_rejects_old_field_names removed —
    # SourceOfInspirationWithReasoningOutput was deleted from schemas/metadata.py.


class TestSourceOfInspirationOutputStrContent:
    def test_source_of_inspiration_str_combines_both_lists(self):
        """__str__() concatenates source_material + franchise_lineage, comma-separated."""
        output = SourceOfInspirationOutput(
            source_material=["based on a novel"],
            franchise_lineage=["sequel"],
        )
        assert str(output) == "based on a novel, sequel"

    def test_source_of_inspiration_str_source_material_only(self):
        """__str__() with only source_material returns just source terms, no trailing comma."""
        output = SourceOfInspirationOutput(
            source_material=["based on a novel", "remake of a film"],
            franchise_lineage=[],
        )
        assert str(output) == "based on a novel, remake of a film"

    def test_source_of_inspiration_str_franchise_lineage_only(self):
        """__str__() with only franchise_lineage returns just lineage terms."""
        output = SourceOfInspirationOutput(
            source_material=[],
            franchise_lineage=["franchise starter", "first in trilogy"],
        )
        assert str(output) == "franchise starter, first in trilogy"


# TestSourceOfInspirationPromptAliasRemoval removed — no longer relevant.


# TestSourceOfInspirationWithReasoningEvidenceConstraints removed —
# SourceOfInspirationWithReasoningOutput was deleted from schemas/metadata.py.


class TestSourceOfInspirationOutputStrEmpty:
    def test_source_of_inspiration_str_empty_lists(self):
        """SourceOfInspirationOutput.__str__() returns '' when both lists empty."""
        output = SourceOfInspirationOutput(
            source_material=[],
            franchise_lineage=[],
        )
        assert str(output) == ""


# ---------------------------------------------------------------------------
# TermsSection / TermsWithNegationsSection extra="forbid"
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SourceMaterialV2Output
# ---------------------------------------------------------------------------

class TestSourceMaterialV2OutputValidation:
    def test_extra_forbid(self):
        """Extra fields are rejected by extra='forbid'."""
        with pytest.raises(ValidationError):
            SourceMaterialV2Output(
                source_material_types=[],
                unknown_field="not allowed",
            )

    def test_empty_list_valid(self):
        """Empty list is valid — represents original screenplays."""
        output = SourceMaterialV2Output(source_material_types=[])
        assert output.source_material_types == []

    def test_single_type_valid(self):
        """A single SourceMaterialType value is accepted."""
        output = SourceMaterialV2Output(
            source_material_types=[SourceMaterialType.NOVEL_ADAPTATION],
        )
        assert len(output.source_material_types) == 1
        assert output.source_material_types[0] == SourceMaterialType.NOVEL_ADAPTATION

    def test_multiple_types_valid(self):
        """Multiple SourceMaterialType values are accepted."""
        output = SourceMaterialV2Output(
            source_material_types=[
                SourceMaterialType.NOVEL_ADAPTATION,
                SourceMaterialType.TRUE_STORY,
            ],
        )
        assert len(output.source_material_types) == 2

    def test_rejects_invalid_enum_value(self):
        """A raw string not in SourceMaterialType raises ValidationError."""
        with pytest.raises(ValidationError):
            SourceMaterialV2Output(source_material_types=["nonexistent_type"])

    def test_string_auto_coercion(self):
        """Pydantic accepts string values that match enum members."""
        output = SourceMaterialV2Output(
            source_material_types=["novel_adaptation"],
        )
        assert output.source_material_types[0] == SourceMaterialType.NOVEL_ADAPTATION

    def test_duplicate_enum_values_accepted(self):
        """Duplicate enum values in the list are accepted by Pydantic."""
        output = SourceMaterialV2Output(
            source_material_types=[
                SourceMaterialType.NOVEL_ADAPTATION,
                SourceMaterialType.NOVEL_ADAPTATION,
            ],
        )
        assert len(output.source_material_types) == 2


class TestSourceMaterialV2OutputStr:
    def test_str_empty(self):
        """__str__() returns '' when source_material_types is empty."""
        output = SourceMaterialV2Output(source_material_types=[])
        assert str(output) == ""

    def test_str_single(self):
        """__str__() with one type returns human-readable form (underscores → spaces)."""
        output = SourceMaterialV2Output(
            source_material_types=[SourceMaterialType.NOVEL_ADAPTATION],
        )
        assert str(output) == "novel adaptation"

    def test_str_multiple(self):
        """__str__() with multiple types returns comma-separated human-readable forms."""
        output = SourceMaterialV2Output(
            source_material_types=[
                SourceMaterialType.NOVEL_ADAPTATION,
                SourceMaterialType.TRUE_STORY,
            ],
        )
        assert str(output) == "novel adaptation, true story"


class TestSourceMaterialV2OutputIsEmbeddable:
    def test_is_embeddable_subclass(self):
        """SourceMaterialV2Output is a subclass of EmbeddableOutput."""
        assert issubclass(SourceMaterialV2Output, EmbeddableOutput)

    def test_implements_embedding_text(self):
        """SourceMaterialV2Output implements embedding_text()."""
        output = SourceMaterialV2Output(source_material_types=[])
        # Should not raise — method is implemented, not abstract
        result = output.embedding_text()
        assert isinstance(result, str)


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


# ---------------------------------------------------------------------------
# FranchiseOutput.validate_and_fix()
# ---------------------------------------------------------------------------

from schemas.metadata import FranchiseOutput


def _franchise_json(**overrides) -> str:
    """Build minimal valid FranchiseOutput JSON with targeted overrides."""
    defaults = {
        "lineage_reasoning": "reasoning",
        "lineage": None,
        "shared_universe": None,
        "subgroups_reasoning": "reasoning",
        "recognized_subgroups": [],
        "launched_subgroup": False,
        "position_reasoning": "reasoning",
        "lineage_position": None,
        "crossover_reasoning": "reasoning",
        "is_crossover": False,
        "spinoff_reasoning": "reasoning",
        "is_spinoff": False,
        "launch_reasoning": "reasoning",
        "launched_franchise": False,
    }
    defaults.update(overrides)
    import json as _json
    return _json.dumps(defaults)


class TestFranchiseOutputValidateAndFix:
    def test_lineage_null_clears_shared_universe_and_subgroups(self):
        """Rule 1: lineage=null propagates to shared_universe, subgroups, launched_subgroup."""
        result = FranchiseOutput.validate_and_fix(_franchise_json(
            lineage=None,
            shared_universe="marvel cinematic universe",
            recognized_subgroups=["phase one"],
            launched_subgroup=True,
        ))
        assert result.shared_universe is None
        assert result.recognized_subgroups == []
        assert result.launched_subgroup is False

    def test_lineage_null_preserves_lineage_position(self):
        """Rule 1 exception: lineage_position is kept for pair-remakes."""
        result = FranchiseOutput.validate_and_fix(_franchise_json(
            lineage=None,
            lineage_position="sequel",
        ))
        assert result.lineage_position == LineagePosition.SEQUEL

    def test_lineage_null_preserves_is_crossover(self):
        """Rule 1 exception: is_crossover is kept when lineage=null."""
        result = FranchiseOutput.validate_and_fix(_franchise_json(
            lineage=None,
            is_crossover=True,
        ))
        assert result.is_crossover is True

    def test_lineage_null_preserves_is_spinoff(self):
        """Rule 1 exception: is_spinoff is kept when lineage=null."""
        result = FranchiseOutput.validate_and_fix(_franchise_json(
            lineage=None,
            is_spinoff=True,
        ))
        assert result.is_spinoff is True

    def test_launched_subgroup_false_when_no_recognized_subgroups(self):
        """Rule 2: launched_subgroup forced false when groups list is empty."""
        result = FranchiseOutput.validate_and_fix(_franchise_json(
            lineage="marvel",
            launched_subgroup=True,
            recognized_subgroups=[],
        ))
        assert result.launched_subgroup is False

    def test_launched_subgroup_true_with_recognized_subgroups(self):
        """Rule 2 positive: launched_subgroup stays true when groups list is populated."""
        result = FranchiseOutput.validate_and_fix(_franchise_json(
            lineage="marvel",
            launched_subgroup=True,
            recognized_subgroups=["marvel cinematic universe"],
        ))
        assert result.launched_subgroup is True

    def test_launched_franchise_false_when_lineage_null(self):
        """Rule 3a: launched_franchise forced false when lineage is null."""
        result = FranchiseOutput.validate_and_fix(_franchise_json(
            lineage=None,
            launched_franchise=True,
        ))
        assert result.launched_franchise is False

    def test_launched_franchise_false_when_lineage_position_populated(self):
        """Rule 3b: launched_franchise forced false when lineage_position is set."""
        result = FranchiseOutput.validate_and_fix(_franchise_json(
            lineage="star wars",
            lineage_position="sequel",
            launched_franchise=True,
        ))
        assert result.launched_franchise is False

    def test_launched_franchise_false_when_is_spinoff(self):
        """Rule 3c: launched_franchise forced false when is_spinoff is true."""
        result = FranchiseOutput.validate_and_fix(_franchise_json(
            lineage="shrek",
            is_spinoff=True,
            launched_franchise=True,
        ))
        assert result.launched_franchise is False

    def test_launched_franchise_true_when_all_preconditions_met(self):
        """Rule 3 positive: launched_franchise stays true when all preconditions pass."""
        result = FranchiseOutput.validate_and_fix(_franchise_json(
            lineage="shrek",
            lineage_position=None,
            is_spinoff=False,
            launched_franchise=True,
        ))
        assert result.launched_franchise is True


# ---------------------------------------------------------------------------
# ConceptTagsOutput.validate_and_fix() and all_concept_tag_ids()
# ---------------------------------------------------------------------------

from schemas.metadata import (
    ConceptTagsOutput,
    NarrativeStructureAssessment,
    PlotArchetypeAssessment,
    SettingAssessment,
    CharacterAssessment,
    EndingAssessment,
    ExperientialAssessment,
    ContentFlagAssessment,
)
from schemas.enums import (
    NarrativeStructureTag,
    PlotArchetypeTag,
    SettingTag,
    CharacterTag,
    EndingTag,
    ExperientialTag,
    ContentFlagTag,
)


def _concept_tags_output(**overrides) -> ConceptTagsOutput:
    """Build a minimal ConceptTagsOutput with targeted overrides."""
    defaults = {
        "narrative_structure": NarrativeStructureAssessment(tags=[]),
        "plot_archetypes": PlotArchetypeAssessment(tags=[]),
        "settings": SettingAssessment(tags=[]),
        "characters": CharacterAssessment(tags=[]),
        "endings": EndingAssessment(tag=EndingTag.NO_CLEAR_CHOICE),
        "experiential": ExperientialAssessment(tags=[]),
        "content_flags": ContentFlagAssessment(tags=[]),
    }
    defaults.update(overrides)
    return ConceptTagsOutput(**defaults)


class TestConceptTagsOutputValidateAndFix:
    def test_deduplicates_tags(self):
        """Duplicate tags within a category are deduplicated."""
        output = _concept_tags_output(
            narrative_structure=NarrativeStructureAssessment(
                tags=[NarrativeStructureTag.PLOT_TWIST, NarrativeStructureTag.PLOT_TWIST],
            ),
        )
        output.apply_deterministic_fixups()
        assert output.narrative_structure.tags == [NarrativeStructureTag.PLOT_TWIST]

    def test_twist_villain_implies_plot_twist(self):
        """TWIST_VILLAIN without PLOT_TWIST → PLOT_TWIST added."""
        output = _concept_tags_output(
            narrative_structure=NarrativeStructureAssessment(
                tags=[NarrativeStructureTag.TWIST_VILLAIN],
            ),
        )
        output.apply_deterministic_fixups()
        assert NarrativeStructureTag.PLOT_TWIST in output.narrative_structure.tags

    def test_twist_villain_no_duplicate_when_plot_twist_present(self):
        """Both present → no duplicate PLOT_TWIST."""
        output = _concept_tags_output(
            narrative_structure=NarrativeStructureAssessment(
                tags=[NarrativeStructureTag.TWIST_VILLAIN, NarrativeStructureTag.PLOT_TWIST],
            ),
        )
        output.apply_deterministic_fixups()
        plot_twist_count = output.narrative_structure.tags.count(NarrativeStructureTag.PLOT_TWIST)
        assert plot_twist_count == 1

    def test_validate_and_fix_roundtrip(self):
        """Full JSON → validate_and_fix → both fixups applied."""
        import json as _json
        raw = _json.dumps({
            "narrative_structure": {"tags": ["twist_villain"]},
            "plot_archetypes": {"tags": []},
            "settings": {"tags": []},
            "characters": {"tags": []},
            "endings": {"tag": "happy_ending"},
            "experiential": {"tags": []},
            "content_flags": {"tags": []},
        })
        result = ConceptTagsOutput.validate_and_fix(raw)
        assert NarrativeStructureTag.TWIST_VILLAIN in result.narrative_structure.tags
        assert NarrativeStructureTag.PLOT_TWIST in result.narrative_structure.tags

    def test_all_concept_tag_ids_filters_no_clear_choice(self):
        """NO_CLEAR_CHOICE (id=-1) is filtered out of all_concept_tag_ids()."""
        output = _concept_tags_output(
            endings=EndingAssessment(tag=EndingTag.NO_CLEAR_CHOICE),
        )
        assert -1 not in output.all_concept_tag_ids()

    def test_all_concept_tag_ids_includes_positive_ending(self):
        """Positive ending tags are included in all_concept_tag_ids()."""
        output = _concept_tags_output(
            endings=EndingAssessment(tag=EndingTag.HAPPY_ENDING),
        )
        assert 41 in output.all_concept_tag_ids()

    def test_all_concept_tag_ids_sorted_and_unique(self):
        """Result is sorted with no duplicates."""
        output = _concept_tags_output(
            narrative_structure=NarrativeStructureAssessment(
                tags=[NarrativeStructureTag.PLOT_TWIST, NarrativeStructureTag.TWIST_VILLAIN],
            ),
            plot_archetypes=PlotArchetypeAssessment(tags=[PlotArchetypeTag.REVENGE]),
            endings=EndingAssessment(tag=EndingTag.SAD_ENDING),
        )
        ids = output.all_concept_tag_ids()
        assert ids == sorted(set(ids))

    def test_all_concept_tag_ids_empty_when_no_tags(self):
        """All empty assessments + NO_CLEAR_CHOICE ending → empty list."""
        output = _concept_tags_output()
        assert output.all_concept_tag_ids() == []


# ---------------------------------------------------------------------------
# ProductionTechniquesOutput
# ---------------------------------------------------------------------------

from schemas.metadata import ProductionTechniquesOutput


class TestProductionTechniquesOutput:
    def test_extra_forbid(self):
        """Extra fields are rejected."""
        with pytest.raises(ValidationError):
            ProductionTechniquesOutput(terms=["cgi"], unknown="bad")

    def test_empty_terms_valid(self):
        """Selective classifier may return nothing — empty terms is valid."""
        output = ProductionTechniquesOutput(terms=[])
        assert output.terms == []

    def test_is_embeddable_subclass(self):
        """ProductionTechniquesOutput subclasses EmbeddableOutput."""
        assert issubclass(ProductionTechniquesOutput, EmbeddableOutput)

    def test_embedding_text_normalizes(self):
        """embedding_text() applies normalize_string to each term."""
        from implementation.misc.helpers import normalize_string
        output = ProductionTechniquesOutput(terms=["Stop-Motion", "L.A. CGI"])
        result = output.embedding_text()
        assert normalize_string("Stop-Motion") in result
        assert normalize_string("L.A. CGI") in result

    def test_str_lowercases(self):
        """__str__() returns lowercased comma-separated terms."""
        output = ProductionTechniquesOutput(terms=["Stop Motion", "CGI"])
        assert str(output) == "stop motion, cgi"
