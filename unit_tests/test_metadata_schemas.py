"""
Unit tests for movie_ingestion.metadata_generation.schemas.

Covers:
  - ReceptionOutput.__str__() excludes review_insights_brief
  - ReceptionOutput field rename (reception_summary -> new_reception_summary)
  - CharacterArc: extra="forbid", required fields, __str__()
  - WithJustifications sub-model __str__() returns label only
  - WithJustifications __str__() parity with base variants (6 pairs)
"""

import pytest
from pydantic import ValidationError

from movie_ingestion.metadata_generation.schemas import (
    ReceptionOutput,
    CharacterArc,
    CoreConceptWithJustification,
    MajorThemeWithJustification,
    MajorLessonLearnedWithJustification,
    PlotAnalysisOutput,
    PlotAnalysisWithJustificationsOutput,
    TermsSection,
    TermsWithNegationsSection,
    OptionalTermsWithNegationsSection,
    TermsWithJustificationSection,
    TermsWithNegationsAndJustificationSection,
    OptionalTermsWithNegationsAndJustificationSection,
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
    def test_reception_output_str_excludes_review_insights_brief(self):
        output = ReceptionOutput(
            new_reception_summary="Widely acclaimed for visual effects.",
            praise_attributes=["groundbreaking visuals"],
            complaint_attributes=["thin plot"],
            review_insights_brief="Critics noted philosophical depth and innovative cinematography.",
        )
        result = str(output)

        # The review_insights_brief text should NOT appear in str() output
        assert "philosophical depth" not in result
        assert "innovative cinematography" not in result
        assert "critics noted" not in result.lower()

        # But the other fields SHOULD appear (lowercased)
        assert "widely acclaimed" in result
        assert "groundbreaking visuals" in result
        assert "thin plot" in result


class TestReceptionOutputRenamed:
    def test_reception_output_uses_new_reception_summary(self):
        """Field name is new_reception_summary; old name raises ValidationError."""
        output = ReceptionOutput(
            new_reception_summary="Good movie.",
            review_insights_brief="Critics liked it.",
        )
        assert output.new_reception_summary == "Good movie."

        with pytest.raises(ValidationError):
            ReceptionOutput(
                reception_summary="Good movie.",
                review_insights_brief="Critics liked it.",
            )

    def test_reception_output_str_uses_new_reception_summary(self):
        output = ReceptionOutput(
            new_reception_summary="A Brilliant Film.",
            review_insights_brief="Insightful.",
        )
        assert "a brilliant film." in str(output)


# ---------------------------------------------------------------------------
# CharacterArc updates
# ---------------------------------------------------------------------------

class TestCharacterArcUpdated:
    def test_character_arc_requires_description(self):
        with pytest.raises(ValidationError):
            CharacterArc(
                character_name="Neo",
                arc_transformation_label="hero's awakening",
            )

    def test_character_arc_extra_forbid(self):
        with pytest.raises(ValidationError):
            CharacterArc(
                character_name="Neo",
                arc_transformation_description="Discovers his true power.",
                arc_transformation_label="hero's awakening",
                extra_field="not allowed",
            )

    def test_character_arc_str_returns_label(self):
        arc = CharacterArc(
            character_name="Neo",
            arc_transformation_description="Discovers his true power and becomes The One.",
            arc_transformation_label="hero's awakening",
        )
        assert str(arc) == "hero's awakening"
        # Description should NOT appear in str()
        assert "discovers" not in str(arc).lower()


# ---------------------------------------------------------------------------
# WithJustifications sub-model __str__() tests
# ---------------------------------------------------------------------------

class TestCoreConceptWithJustificationStr:
    def test_str_returns_label_only(self):
        obj = CoreConceptWithJustification(
            explanation_and_justification="This concept captures the essence.",
            core_concept_label="forbidden knowledge",
        )
        assert str(obj) == "forbidden knowledge"
        assert "essence" not in str(obj)


class TestMajorThemeWithJustificationStr:
    def test_str_returns_label_only(self):
        obj = MajorThemeWithJustification(
            explanation_and_justification="This theme runs through every scene.",
            theme_label="identity and transformation",
        )
        assert str(obj) == "identity and transformation"
        assert "runs through" not in str(obj)


class TestMajorLessonLearnedWithJustificationStr:
    def test_str_returns_label_only(self):
        obj = MajorLessonLearnedWithJustification(
            explanation_and_justification="The film teaches us to question reality.",
            lesson_label="question everything",
        )
        assert str(obj) == "question everything"
        assert "teaches" not in str(obj)


# ---------------------------------------------------------------------------
# PlotAnalysis __str__() parity
# ---------------------------------------------------------------------------

# Shared test data for plot analysis parity tests
_PLOT_ANALYSIS_DATA = dict(
    generalized_plot_overview="A hacker discovers the truth about simulated reality.",
    genre_signatures=["cyberpunk thriller", "philosophical sci-fi"],
    conflict_scale="global",
    character_arcs=[
        CharacterArc(
            character_name="Neo",
            arc_transformation_description="Transforms from a lost programmer into a messianic figure.",
            arc_transformation_label="hero's awakening",
        ),
    ],
)


class TestPlotAnalysisWithJustificationsStrParity:
    def test_str_parity_with_plot_analysis(self):
        """WithJustifications variant must produce identical str() to base."""
        base = PlotAnalysisOutput(
            core_concept_label="forbidden knowledge",
            themes_primary=["identity", "free will"],
            lessons_learned=["question everything"],
            **_PLOT_ANALYSIS_DATA,
        )
        with_j = PlotAnalysisWithJustificationsOutput(
            core_concept=CoreConceptWithJustification(
                explanation_and_justification="This is the heart of the movie.",
                core_concept_label="forbidden knowledge",
            ),
            themes_primary=[
                MajorThemeWithJustification(
                    explanation_and_justification="Central to the narrative.",
                    theme_label="identity",
                ),
                MajorThemeWithJustification(
                    explanation_and_justification="Runs through every choice.",
                    theme_label="free will",
                ),
            ],
            lessons_learned=[
                MajorLessonLearnedWithJustification(
                    explanation_and_justification="The film teaches skepticism.",
                    lesson_label="question everything",
                ),
            ],
            **_PLOT_ANALYSIS_DATA,
        )
        assert str(base) == str(with_j)

    def test_str_excludes_justification_text(self):
        with_j = PlotAnalysisWithJustificationsOutput(
            core_concept=CoreConceptWithJustification(
                explanation_and_justification="JUSTIFICATION_MARKER_CORE",
                core_concept_label="forbidden knowledge",
            ),
            themes_primary=[
                MajorThemeWithJustification(
                    explanation_and_justification="JUSTIFICATION_MARKER_THEME",
                    theme_label="identity",
                ),
            ],
            lessons_learned=[
                MajorLessonLearnedWithJustification(
                    explanation_and_justification="JUSTIFICATION_MARKER_LESSON",
                    lesson_label="question everything",
                ),
            ],
            **_PLOT_ANALYSIS_DATA,
        )
        result = str(with_j)
        assert "JUSTIFICATION_MARKER_CORE" not in result
        assert "JUSTIFICATION_MARKER_THEME" not in result
        assert "JUSTIFICATION_MARKER_LESSON" not in result

    def test_str_parity_empty_lessons(self):
        base = PlotAnalysisOutput(
            core_concept_label="forbidden knowledge",
            themes_primary=["identity"],
            lessons_learned=[],
            **_PLOT_ANALYSIS_DATA,
        )
        with_j = PlotAnalysisWithJustificationsOutput(
            core_concept=CoreConceptWithJustification(
                explanation_and_justification="Justification.",
                core_concept_label="forbidden knowledge",
            ),
            themes_primary=[
                MajorThemeWithJustification(
                    explanation_and_justification="Justification.",
                    theme_label="identity",
                ),
            ],
            lessons_learned=[],
            **_PLOT_ANALYSIS_DATA,
        )
        assert str(base) == str(with_j)


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
        base = ViewerExperienceOutput(
            emotional_palette=_make_terms_section(_VE_TERMS, _VE_NEGATIONS),
            tension_adrenaline=_make_terms_section(["high stakes"]),
            tone_self_seriousness=_make_terms_section(["dead serious"]),
            cognitive_complexity=_make_terms_section(["mind-bending"]),
            disturbance_profile=OptionalTermsWithNegationsSection(
                should_skip=False,
                section_data=_make_terms_section(["disturbing imagery"]),
            ),
            sensory_load=OptionalTermsWithNegationsSection(
                should_skip=False,
                section_data=_make_terms_section(["loud"]),
            ),
            emotional_volatility=OptionalTermsWithNegationsSection(
                should_skip=False,
                section_data=_make_terms_section(["whiplash"]),
            ),
            ending_aftertaste=_make_terms_section(["lingering dread"]),
        )
        with_j = ViewerExperienceWithJustificationsOutput(
            emotional_palette=_make_terms_j_section(_VE_TERMS, _VE_NEGATIONS),
            tension_adrenaline=_make_terms_j_section(["high stakes"]),
            tone_self_seriousness=_make_terms_j_section(["dead serious"]),
            cognitive_complexity=_make_terms_j_section(["mind-bending"]),
            disturbance_profile=OptionalTermsWithNegationsAndJustificationSection(
                should_skip=False,
                section_data=_make_terms_j_section(["disturbing imagery"]),
            ),
            sensory_load=OptionalTermsWithNegationsAndJustificationSection(
                should_skip=False,
                section_data=_make_terms_j_section(["loud"]),
            ),
            emotional_volatility=OptionalTermsWithNegationsAndJustificationSection(
                should_skip=False,
                section_data=_make_terms_j_section(["whiplash"]),
            ),
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
            disturbance_profile=OptionalTermsWithNegationsAndJustificationSection(
                should_skip=True,
                section_data=_make_terms_j_section(),
            ),
            sensory_load=OptionalTermsWithNegationsAndJustificationSection(
                should_skip=True,
                section_data=_make_terms_j_section(),
            ),
            emotional_volatility=OptionalTermsWithNegationsAndJustificationSection(
                should_skip=True,
                section_data=_make_terms_j_section(),
            ),
            ending_aftertaste=_make_terms_j_section(),
        )
        assert "VE_MARKER" not in str(with_j)

    def test_str_parity_with_optional_sections_skipped(self):
        base = ViewerExperienceOutput(
            emotional_palette=_make_terms_section(["warm"]),
            tension_adrenaline=_make_terms_section(),
            tone_self_seriousness=_make_terms_section(),
            cognitive_complexity=_make_terms_section(),
            disturbance_profile=OptionalTermsWithNegationsSection(
                should_skip=True,
                section_data=_make_terms_section(["should be ignored"]),
            ),
            sensory_load=OptionalTermsWithNegationsSection(
                should_skip=True,
                section_data=_make_terms_section(),
            ),
            emotional_volatility=OptionalTermsWithNegationsSection(
                should_skip=True,
                section_data=_make_terms_section(),
            ),
            ending_aftertaste=_make_terms_section(),
        )
        with_j = ViewerExperienceWithJustificationsOutput(
            emotional_palette=_make_terms_j_section(["warm"]),
            tension_adrenaline=_make_terms_j_section(),
            tone_self_seriousness=_make_terms_j_section(),
            cognitive_complexity=_make_terms_j_section(),
            disturbance_profile=OptionalTermsWithNegationsAndJustificationSection(
                should_skip=True,
                section_data=_make_terms_j_section(["should be ignored"]),
            ),
            sensory_load=OptionalTermsWithNegationsAndJustificationSection(
                should_skip=True,
                section_data=_make_terms_j_section(),
            ),
            emotional_volatility=OptionalTermsWithNegationsAndJustificationSection(
                should_skip=True,
                section_data=_make_terms_j_section(),
            ),
            ending_aftertaste=_make_terms_j_section(),
        )
        assert str(base) == str(with_j)
        # Skipped section data should NOT appear
        assert "should be ignored" not in str(base)


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

class TestNarrativeTechniquesWithJustificationsStrParity:
    def test_str_parity_with_narrative_techniques(self):
        sections_base = {
            name: _make_terms_only_section(["term1"])
            for name in [
                "pov_perspective", "narrative_delivery", "narrative_archetype",
                "information_control", "characterization_methods", "character_arcs",
                "audience_character_perception", "conflict_stakes_design",
                "thematic_delivery", "meta_techniques", "additional_plot_devices",
            ]
        }
        sections_j = {
            name: _make_terms_only_j_section(["term1"])
            for name in sections_base
        }
        base = NarrativeTechniquesOutput(**sections_base)
        with_j = NarrativeTechniquesWithJustificationsOutput(**sections_j)
        assert str(base) == str(with_j)

    def test_str_excludes_justification_text(self):
        sections = {
            name: _make_terms_only_j_section(
                ["term1"], justification="NT_MARKER"
            )
            for name in [
                "pov_perspective", "narrative_delivery", "narrative_archetype",
                "information_control", "characterization_methods", "character_arcs",
                "audience_character_perception", "conflict_stakes_design",
                "thematic_delivery", "meta_techniques", "additional_plot_devices",
            ]
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
