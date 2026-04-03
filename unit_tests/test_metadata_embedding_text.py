"""
Unit tests for embedding_text() methods on all metadata output schemas.

embedding_text() is the canonical source for vector embeddings across
all 8 vector spaces. These tests verify normalization patterns, semantic
labels, field inclusion/exclusion, and edge cases.
"""

import pytest

from implementation.misc.helpers import normalize_string
from schemas.metadata import (
    PlotEventsOutput,
    ReceptionOutput,
    PlotAnalysisOutput,
    ElevatorPitchWithJustification,
    ThematicConceptWithJustification,
    CharacterArcWithReasoning,
    ViewerExperienceOutput,
    TermsWithNegationsAndJustificationSection,
    WatchContextOutput,
    TermsWithJustificationSection,
    NarrativeTechniquesOutput,
    ProductionKeywordsOutput,
    SourceOfInspirationOutput,
)


# ---------------------------------------------------------------------------
# Helpers: minimal valid section builders
# ---------------------------------------------------------------------------

def _negation_section(**overrides):
    """Build a minimal TermsWithNegationsAndJustificationSection."""
    defaults = {"justification": "placeholder", "terms": [], "negations": []}
    defaults.update(overrides)
    return TermsWithNegationsAndJustificationSection(**defaults)


def _justification_section(**overrides):
    """Build a minimal TermsWithJustificationSection."""
    defaults = {"evidence_basis": "placeholder", "terms": []}
    defaults.update(overrides)
    return TermsWithJustificationSection(**defaults)


# ---------------------------------------------------------------------------
# PlotEventsOutput
# ---------------------------------------------------------------------------

class TestPlotEventsEmbeddingText:
    def test_lowercased(self):
        """embedding_text() should lowercase the plot_summary, not normalize_string it."""
        output = PlotEventsOutput(plot_summary="A Hero's Journey Through L.A.")
        result = output.embedding_text()
        # .lower() preserves punctuation; normalize_string would strip periods
        assert result == "a hero's journey through l.a."

    def test_matches_str(self):
        """embedding_text() and __str__() should produce the same output."""
        output = PlotEventsOutput(plot_summary="Some Plot Summary")
        assert output.embedding_text() == str(output)


# ---------------------------------------------------------------------------
# ReceptionOutput
# ---------------------------------------------------------------------------

class TestReceptionEmbeddingText:
    def test_has_labeled_sections(self):
        """embedding_text() should include 'praised:' and 'criticized:' labels."""
        output = ReceptionOutput(
            reception_summary="Well received.",
            praised_qualities=["great acting"],
            criticized_qualities=["weak script"],
        )
        result = output.embedding_text()
        assert "praised:" in result
        assert "criticized:" in result

    def test_excludes_extraction_zone(self):
        """Extraction-zone fields must NOT appear in embedding_text()."""
        output = ReceptionOutput(
            source_material_hint="based on autobiography",
            thematic_observations="Reviewers noted deep philosophical themes.",
            emotional_observations="Haunting emotional tone.",
            craft_observations="Masterful cinematography.",
            reception_summary="Well received.",
            praised_qualities=["great acting"],
            criticized_qualities=["weak script"],
        )
        result = output.embedding_text()
        assert "autobiography" not in result
        assert "philosophical" not in result
        assert "haunting" not in result
        assert "cinematography" not in result

    def test_empty_praised_omitted(self):
        """No 'praised:' line when praised_qualities is empty."""
        output = ReceptionOutput(
            reception_summary="Mixed reviews.",
            praised_qualities=[],
            criticized_qualities=["bland performances"],
        )
        result = output.embedding_text()
        assert "praised:" not in result
        assert "criticized:" in result

    def test_empty_criticized_omitted(self):
        """No 'criticized:' line when criticized_qualities is empty."""
        output = ReceptionOutput(
            reception_summary="Universal praise.",
            praised_qualities=["stunning visuals"],
            criticized_qualities=[],
        )
        result = output.embedding_text()
        assert "praised:" in result
        assert "criticized:" not in result

    def test_per_term_normalized(self):
        """Praised/criticized terms should each be individually normalize_string'd."""
        output = ReceptionOutput(
            reception_summary="Decent.",
            praised_qualities=["Ocean's Eleven Style"],
            criticized_qualities=["L.A. Clichés"],
        )
        result = output.embedding_text()
        assert normalize_string("Ocean's Eleven Style") in result
        assert normalize_string("L.A. Clichés") in result

    def test_reception_summary_lowercased_not_normalized(self):
        """reception_summary should be .lower()'d, not normalize_string'd."""
        output = ReceptionOutput(
            reception_summary="A Hero's Journey.",
        )
        result = output.embedding_text()
        # .lower() preserves apostrophe and period
        assert "a hero's journey." in result


# ---------------------------------------------------------------------------
# PlotAnalysisOutput
# ---------------------------------------------------------------------------

def _make_plot_analysis(**overrides):
    """Build a minimal PlotAnalysisOutput."""
    defaults = dict(
        genre_signatures=["Drama", "Thriller"],
        thematic_concepts=[],
        elevator_pitch_with_justification=ElevatorPitchWithJustification(
            explanation_and_justification="Because it's the core.",
            elevator_pitch="Revenge in a small town",
        ),
        conflict_type=[],
        character_arcs=[],
        generalized_plot_overview="A tense drama unfolds.",
    )
    defaults.update(overrides)
    return PlotAnalysisOutput(**defaults)


class TestPlotAnalysisEmbeddingText:
    def test_has_labels(self):
        """embedding_text() should have semantic labels for categorical fields."""
        output = _make_plot_analysis(
            conflict_type=["man vs society"],
            character_arcs=[
                CharacterArcWithReasoning(
                    reasoning="He changes.", arc_transformation_label="Redeemed"
                )
            ],
            thematic_concepts=[
                ThematicConceptWithJustification(
                    explanation_and_justification="Core theme.",
                    concept_label="Justice",
                )
            ],
        )
        result = output.embedding_text()
        assert "genre signatures:" in result
        assert "conflict:" in result
        assert "character arcs:" in result
        assert "themes:" in result

    def test_field_order_elevator_pitch_before_overview(self):
        """Elevator pitch should appear before the generalized overview."""
        output = _make_plot_analysis()
        result = output.embedding_text()
        pitch_pos = result.index("revenge in a small town")
        overview_pos = result.index("a tense drama unfolds.")
        assert pitch_pos < overview_pos

    def test_excludes_justification(self):
        """Justification/reasoning text must NOT appear in embedding_text()."""
        output = _make_plot_analysis(
            thematic_concepts=[
                ThematicConceptWithJustification(
                    explanation_and_justification="This is the core theme because reasons.",
                    concept_label="Justice",
                )
            ],
            character_arcs=[
                CharacterArcWithReasoning(
                    reasoning="The character transforms completely.",
                    arc_transformation_label="Redeemed",
                )
            ],
        )
        result = output.embedding_text()
        assert "core theme because reasons" not in result
        assert "transforms completely" not in result
        # Labels should be present
        assert normalize_string("Justice") in result
        assert normalize_string("Redeemed") in result

    def test_empty_optional_fields(self):
        """embedding_text() works when conflict_type, character_arcs, and thematic_concepts are empty."""
        output = _make_plot_analysis(
            conflict_type=[],
            character_arcs=[],
            thematic_concepts=[],
        )
        result = output.embedding_text()
        assert "conflict:" not in result
        assert "character arcs:" not in result
        assert "themes:" not in result
        # Required fields still present
        assert "genre signatures:" in result
        assert "revenge in a small town" in result

    def test_per_term_normalized(self):
        """Categorical terms should each be individually normalize_string'd."""
        output = _make_plot_analysis(
            genre_signatures=["Sci-Fi", "Film-Noir"],
        )
        result = output.embedding_text()
        assert normalize_string("Sci-Fi") in result
        assert normalize_string("Film-Noir") in result


# ---------------------------------------------------------------------------
# ViewerExperienceOutput
# ---------------------------------------------------------------------------

def _make_viewer_experience(**section_overrides):
    """Build a minimal ViewerExperienceOutput. Pass section name → kwargs."""
    section_names = [
        "emotional_palette", "tension_adrenaline", "tone_self_seriousness",
        "cognitive_complexity", "disturbance_profile", "sensory_load",
        "emotional_volatility", "ending_aftertaste",
    ]
    data = {}
    for name in section_names:
        data[name] = _negation_section(**section_overrides.get(name, {}))
    return ViewerExperienceOutput(**data)


class TestViewerExperienceEmbeddingText:
    def test_per_term_normalized(self):
        """Each term should be normalize_string'd individually."""
        output = _make_viewer_experience(
            emotional_palette={"terms": ["Ocean's Warmth", "L.A. Glow"]},
        )
        result = output.embedding_text()
        assert normalize_string("Ocean's Warmth") in result
        assert normalize_string("L.A. Glow") in result

    def test_includes_negations(self):
        """Negation terms should be included in embedding_text()."""
        output = _make_viewer_experience(
            emotional_palette={
                "terms": ["warm"],
                "negations": ["not scary"],
            },
        )
        result = output.embedding_text()
        assert normalize_string("not scary") in result

    def test_empty_sections_produce_empty_string(self):
        """All empty sections should produce an empty string."""
        output = _make_viewer_experience()
        assert output.embedding_text() == ""


# ---------------------------------------------------------------------------
# WatchContextOutput
# ---------------------------------------------------------------------------

def _make_watch_context(**section_overrides):
    """Build a minimal WatchContextOutput."""
    data = {
        "identity_note": "sincere emotional drama",
        "self_experience_motivations": _justification_section(
            **section_overrides.get("self_experience_motivations", {})
        ),
        "external_motivations": _justification_section(
            **section_overrides.get("external_motivations", {})
        ),
        "key_movie_feature_draws": _justification_section(
            **section_overrides.get("key_movie_feature_draws", {})
        ),
        "watch_scenarios": _justification_section(
            **section_overrides.get("watch_scenarios", {})
        ),
    }
    return WatchContextOutput(**data)


class TestWatchContextEmbeddingText:
    def test_excludes_identity_note(self):
        """identity_note should NOT appear in embedding_text()."""
        output = _make_watch_context()
        result = output.embedding_text()
        assert "sincere emotional drama" not in result

    def test_per_term_normalized(self):
        """Each term should be normalize_string'd individually."""
        output = _make_watch_context(
            self_experience_motivations={"terms": ["Ocean's Feel"]},
        )
        result = output.embedding_text()
        assert normalize_string("Ocean's Feel") in result

    def test_empty_sections_produce_empty_string(self):
        """All empty sections should produce an empty string."""
        output = _make_watch_context()
        assert output.embedding_text() == ""


# ---------------------------------------------------------------------------
# NarrativeTechniquesOutput
# ---------------------------------------------------------------------------

def _make_narrative_techniques(**section_overrides):
    """Build a minimal NarrativeTechniquesOutput."""
    section_names = [
        "narrative_archetype", "narrative_delivery", "pov_perspective",
        "characterization_methods", "character_arcs",
        "audience_character_perception", "information_control",
        "conflict_stakes_design", "additional_narrative_devices",
    ]
    data = {}
    for name in section_names:
        data[name] = _justification_section(**section_overrides.get(name, {}))
    return NarrativeTechniquesOutput(**data)


class TestNarrativeTechniquesEmbeddingText:
    def test_all_9_sections_contribute(self):
        """Terms from all 9 sections should appear in the output."""
        overrides = {}
        section_names = [
            "narrative_archetype", "narrative_delivery", "pov_perspective",
            "characterization_methods", "character_arcs",
            "audience_character_perception", "information_control",
            "conflict_stakes_design", "additional_narrative_devices",
        ]
        for i, name in enumerate(section_names):
            overrides[name] = {"terms": [f"term{i}"]}
        output = _make_narrative_techniques(**overrides)
        result = output.embedding_text()
        for i in range(9):
            assert f"term{i}" in result

    def test_per_term_normalized(self):
        """Each term should be normalize_string'd individually."""
        output = _make_narrative_techniques(
            narrative_archetype={"terms": ["Hero's Journey"]},
        )
        result = output.embedding_text()
        assert normalize_string("Hero's Journey") in result

    def test_empty_sections_produce_empty_string(self):
        """All empty sections should produce an empty string."""
        output = _make_narrative_techniques()
        assert output.embedding_text() == ""


# ---------------------------------------------------------------------------
# ProductionKeywordsOutput
# ---------------------------------------------------------------------------

class TestProductionKeywordsEmbeddingText:
    def test_per_term_normalized(self):
        """Each term should be normalize_string'd individually."""
        output = ProductionKeywordsOutput(terms=["L.A. Noir", "Ocean's Style"])
        result = output.embedding_text()
        assert normalize_string("L.A. Noir") in result
        assert normalize_string("Ocean's Style") in result

    def test_empty_terms(self):
        """Empty terms list should return empty string."""
        output = ProductionKeywordsOutput(terms=[])
        assert output.embedding_text() == ""

    def test_single_term(self):
        """Single term should produce output with no spurious comma."""
        output = ProductionKeywordsOutput(terms=["practical effects"])
        result = output.embedding_text()
        assert result == normalize_string("practical effects")
        assert "," not in result


# ---------------------------------------------------------------------------
# SourceOfInspirationOutput
# ---------------------------------------------------------------------------

class TestSourceOfInspirationEmbeddingText:
    def test_labeled_sections(self):
        """embedding_text() should have 'source material:' and 'franchise position:' labels."""
        output = SourceOfInspirationOutput(
            source_material=["based on novel"],
            franchise_lineage=["sequel"],
        )
        result = output.embedding_text()
        assert "source material:" in result
        assert "franchise position:" in result

    def test_original_screenplay_default(self):
        """No source_material + no/starter franchise → 'original screenplay'."""
        output = SourceOfInspirationOutput(
            source_material=[],
            franchise_lineage=[],
        )
        result = output.embedding_text()
        assert "source material: original screenplay" in result

    def test_original_screenplay_with_franchise_starter(self):
        """Franchise starter still gets 'original screenplay'."""
        output = SourceOfInspirationOutput(
            source_material=[],
            franchise_lineage=["first in franchise"],
        )
        result = output.embedding_text()
        assert "source material: original screenplay" in result
        assert "franchise position:" in result

    def test_sequel_no_original_screenplay(self):
        """Sequel without source_material should NOT get 'original screenplay'."""
        output = SourceOfInspirationOutput(
            source_material=[],
            franchise_lineage=["sequel"],
        )
        result = output.embedding_text()
        assert "original screenplay" not in result
        assert "franchise position:" in result

    def test_prequel_no_original_screenplay(self):
        """Prequel without source_material should NOT get 'original screenplay'."""
        output = SourceOfInspirationOutput(
            source_material=[],
            franchise_lineage=["prequel"],
        )
        result = output.embedding_text()
        assert "original screenplay" not in result

    def test_both_lists_empty_returns_original(self):
        """Both lists empty should produce 'source material: original screenplay'."""
        output = SourceOfInspirationOutput(
            source_material=[],
            franchise_lineage=[],
        )
        result = output.embedding_text()
        assert result == "source material: original screenplay"

    def test_per_term_normalized(self):
        """Terms should be individually normalize_string'd."""
        output = SourceOfInspirationOutput(
            source_material=["Based on Author's Novel"],
            franchise_lineage=[],
        )
        result = output.embedding_text()
        assert normalize_string("Based on Author's Novel") in result


# ---------------------------------------------------------------------------
# SourceOfInspirationOutput._is_likely_original
# ---------------------------------------------------------------------------

class TestIsLikelyOriginal:
    def test_empty_franchise(self):
        """Empty franchise_lineage means original."""
        output = SourceOfInspirationOutput(source_material=[], franchise_lineage=[])
        assert output._is_likely_original() is True

    def test_franchise_starter(self):
        """'first in franchise' is still original."""
        output = SourceOfInspirationOutput(
            source_material=[], franchise_lineage=["first in franchise"]
        )
        assert output._is_likely_original() is True

    def test_franchise_starter_mixed(self):
        """'franchise starter' contains 'start' → original."""
        output = SourceOfInspirationOutput(
            source_material=[], franchise_lineage=["franchise starter"]
        )
        assert output._is_likely_original() is True

    def test_sequel(self):
        """'sequel' is NOT original."""
        output = SourceOfInspirationOutput(
            source_material=[], franchise_lineage=["sequel"]
        )
        assert output._is_likely_original() is False

    def test_prequel(self):
        """'prequel' is NOT original."""
        output = SourceOfInspirationOutput(
            source_material=[], franchise_lineage=["prequel"]
        )
        assert output._is_likely_original() is False

    def test_mixed_first_and_sequel(self):
        """If any lineage term is NOT a starter, not original."""
        output = SourceOfInspirationOutput(
            source_material=[],
            franchise_lineage=["first in franchise", "sequel"],
        )
        assert output._is_likely_original() is False


# ---------------------------------------------------------------------------
# Cross-type: embedding_text vs __str__ divergence
# ---------------------------------------------------------------------------

class TestEmbeddingTextVsStr:
    """Verify that embedding_text() and __str__() intentionally diverge
    where the plan says they should."""

    def test_reception_diverges(self):
        """ReceptionOutput embedding_text uses normalize_string + labels; __str__ uses .lower()."""
        output = ReceptionOutput(
            reception_summary="Good Film.",
            praised_qualities=["Ocean's Acting"],
        )
        # __str__ uses .lower() for terms
        assert "ocean's acting" in str(output)
        # embedding_text uses normalize_string (removes apostrophe)
        assert normalize_string("Ocean's Acting") in output.embedding_text()

    def test_plot_analysis_diverges(self):
        """PlotAnalysisOutput embedding_text uses labels; __str__ does not."""
        output = _make_plot_analysis()
        assert "genre signatures:" in output.embedding_text()
        assert "genre signatures:" not in str(output)
