"""
Unit tests for metadata generation prompt constants.

Covers:
  - plot_events: SYSTEM_PROMPT_SYNOPSIS, SYSTEM_PROMPT_SYNTHESIS
  - source_of_inspiration: SYSTEM_PROMPT
  - production_keywords: SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_JUSTIFICATIONS
"""

from movie_ingestion.metadata_generation.prompts.plot_events import (
    SYSTEM_PROMPT_SYNOPSIS,
    SYSTEM_PROMPT_SYNTHESIS,
)
from movie_ingestion.metadata_generation.prompts.source_of_inspiration import SYSTEM_PROMPT
from movie_ingestion.metadata_generation.prompts.production_keywords import (
    SYSTEM_PROMPT as PK_SYSTEM_PROMPT,
    SYSTEM_PROMPT_WITH_JUSTIFICATIONS as PK_SYSTEM_PROMPT_WITH_JUSTIFICATIONS,
    _PREAMBLE as PK_PREAMBLE,
    _OUTPUT_NO_JUSTIFICATIONS as PK_OUTPUT_NO_JUSTIFICATIONS,
)


# ---------------------------------------------------------------------------
# Plot events prompt constants
# ---------------------------------------------------------------------------


class TestPlotEventsPromptConstants:
    """Tests for the two branch-specific plot events system prompts."""

    def test_synopsis_prompt_is_non_empty(self) -> None:
        """SYSTEM_PROMPT_SYNOPSIS is a non-empty string."""
        assert isinstance(SYSTEM_PROMPT_SYNOPSIS, str)
        assert len(SYSTEM_PROMPT_SYNOPSIS) > 0

    def test_synopsis_prompt_contains_condense(self) -> None:
        """SYSTEM_PROMPT_SYNOPSIS describes a condensation task."""
        assert "condense" in SYSTEM_PROMPT_SYNOPSIS.lower() or "condensation" in SYSTEM_PROMPT_SYNOPSIS.lower()

    def test_synopsis_prompt_mentions_primary_source(self) -> None:
        """SYSTEM_PROMPT_SYNOPSIS mentions plot_synopsis as primary source."""
        assert "primary" in SYSTEM_PROMPT_SYNOPSIS.lower()
        assert "plot_synopsis" in SYSTEM_PROMPT_SYNOPSIS.lower()

    def test_synthesis_prompt_is_non_empty(self) -> None:
        """SYSTEM_PROMPT_SYNTHESIS is a non-empty string."""
        assert isinstance(SYSTEM_PROMPT_SYNTHESIS, str)
        assert len(SYSTEM_PROMPT_SYNTHESIS) > 0

    def test_synthesis_prompt_contains_consolidat(self) -> None:
        """SYSTEM_PROMPT_SYNTHESIS describes a consolidation task."""
        lower = SYSTEM_PROMPT_SYNTHESIS.lower()
        assert "consolidat" in lower

    def test_synthesis_prompt_contains_anti_hallucination(self) -> None:
        """SYSTEM_PROMPT_SYNTHESIS contains anti-hallucination constraint."""
        lower = SYSTEM_PROMPT_SYNTHESIS.lower()
        assert "no knowledge of any film" in lower

    def test_synopsis_prompt_contains_json_output_instruction(self) -> None:
        """SYSTEM_PROMPT_SYNOPSIS contains JSON output format instruction."""
        assert "JSON with a single field: plot_summary" in SYSTEM_PROMPT_SYNOPSIS

    def test_synthesis_prompt_contains_json_output_instruction(self) -> None:
        """SYSTEM_PROMPT_SYNTHESIS contains JSON output format instruction."""
        assert "JSON with a single field: plot_summary" in SYSTEM_PROMPT_SYNTHESIS

    def test_synopsis_prompt_does_not_mention_setting(self) -> None:
        """SYSTEM_PROMPT_SYNOPSIS does not reference a 'setting' output field."""
        # Check the OUTPUT section specifically
        lower = SYSTEM_PROMPT_SYNOPSIS.lower()
        output_idx = lower.find("output")
        if output_idx >= 0:
            output_section = lower[output_idx:]
            assert "setting:" not in output_section
            assert "setting (" not in output_section

    def test_synopsis_prompt_does_not_mention_major_characters(self) -> None:
        """SYSTEM_PROMPT_SYNOPSIS does not reference 'major_characters' output field."""
        lower = SYSTEM_PROMPT_SYNOPSIS.lower()
        assert "major_characters" not in lower

    def test_synthesis_prompt_does_not_mention_setting(self) -> None:
        """SYSTEM_PROMPT_SYNTHESIS does not reference a 'setting' output field."""
        lower = SYSTEM_PROMPT_SYNTHESIS.lower()
        output_idx = lower.find("output")
        if output_idx >= 0:
            output_section = lower[output_idx:]
            assert "setting:" not in output_section
            assert "setting (" not in output_section

    def test_synthesis_prompt_does_not_mention_major_characters(self) -> None:
        """SYSTEM_PROMPT_SYNTHESIS does not reference 'major_characters' output field."""
        lower = SYSTEM_PROMPT_SYNTHESIS.lower()
        assert "major_characters" not in lower

    def test_legacy_prompt_still_mentions_plot_summary(self) -> None:
        """SYSTEM_PROMPT (legacy) still references plot_summary."""
        from movie_ingestion.metadata_generation.prompts.plot_events import SYSTEM_PROMPT
        assert "plot_summary" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Source of inspiration prompt constants
# ---------------------------------------------------------------------------


class TestSourceOfInspirationPromptConstants:
    """Tests for the source of inspiration system prompt."""

    def test_base_prompt_does_not_mention_justification_in_output(self) -> None:
        """SYSTEM_PROMPT does not mention 'justification' in the OUTPUT section."""
        # Find the OUTPUT section and check it
        lower = SYSTEM_PROMPT.lower()
        output_idx = lower.find("output")
        if output_idx >= 0:
            output_section = lower[output_idx:]
            assert "justification" not in output_section

    def test_prompt_mentions_evidence(self) -> None:
        """SYSTEM_PROMPT mentions 'evidence'."""
        lower = SYSTEM_PROMPT.lower()
        assert "evidence" in lower

    def test_prompt_contains_parametric_knowledge_allowance(self) -> None:
        """SYSTEM_PROMPT contains the parametric knowledge allowance text."""
        lower = SYSTEM_PROMPT.lower()
        assert "parametric knowledge" in lower


# ---------------------------------------------------------------------------
# Production keywords prompt constants
# ---------------------------------------------------------------------------


class TestProductionKeywordsPromptConstants:
    """Tests for the production keywords system prompt variants."""

    def test_system_prompt_is_non_empty(self) -> None:
        """SYSTEM_PROMPT is a non-empty string."""
        assert isinstance(PK_SYSTEM_PROMPT, str)
        assert len(PK_SYSTEM_PROMPT) > 0

    def test_system_prompt_with_justifications_is_non_empty(self) -> None:
        """SYSTEM_PROMPT_WITH_JUSTIFICATIONS is a non-empty string."""
        assert isinstance(PK_SYSTEM_PROMPT_WITH_JUSTIFICATIONS, str)
        assert len(PK_SYSTEM_PROMPT_WITH_JUSTIFICATIONS) > 0

    def test_system_prompt_contains_core_test(self) -> None:
        """SYSTEM_PROMPT contains the core classification test question."""
        lower = PK_SYSTEM_PROMPT.lower()
        assert "real world" in lower or "how the movie was made" in lower

    def test_system_prompt_contains_all_seven_categories(self) -> None:
        """SYSTEM_PROMPT contains all 7 production-relevant category headings."""
        lower = PK_SYSTEM_PROMPT.lower()
        assert "production medium" in lower
        assert "origin and language" in lower
        assert "source material" in lower
        assert "production process" in lower
        assert "franchise" in lower
        assert "production form" in lower
        assert "production era" in lower

    def test_system_prompt_contains_exclusion_section(self) -> None:
        """SYSTEM_PROMPT contains the 'WHAT DOES NOT COUNT' exclusion section."""
        assert "WHAT DOES NOT COUNT" in PK_SYSTEM_PROMPT

    def test_system_prompt_contains_no_invention_rule(self) -> None:
        """SYSTEM_PROMPT contains the anti-hallucination rule."""
        assert "Inventing new keywords is a catastrophic failure" in PK_SYSTEM_PROMPT

    def test_system_prompt_contains_empty_list_guidance(self) -> None:
        """SYSTEM_PROMPT contains guidance that an empty terms list is correct."""
        lower = PK_SYSTEM_PROMPT.lower()
        assert "empty" in lower and "correct" in lower

    def test_justifications_variant_mentions_justification(self) -> None:
        """SYSTEM_PROMPT_WITH_JUSTIFICATIONS mentions 'justification' in output section."""
        lower = PK_SYSTEM_PROMPT_WITH_JUSTIFICATIONS.lower()
        assert "justification" in lower

    def test_no_justifications_variant_omits_justification_in_output(self) -> None:
        """The no-justifications output section does not contain 'justification'."""
        lower = PK_OUTPUT_NO_JUSTIFICATIONS.lower()
        assert "justification" not in lower

    def test_both_variants_share_same_preamble(self) -> None:
        """Both prompt variants start with the same preamble content."""
        assert PK_SYSTEM_PROMPT.startswith(PK_PREAMBLE)
        assert PK_SYSTEM_PROMPT_WITH_JUSTIFICATIONS.startswith(PK_PREAMBLE)


# ---------------------------------------------------------------------------
# Source material V2 prompt constants
# ---------------------------------------------------------------------------

from movie_ingestion.metadata_generation.prompts.source_material_v2 import (
    SYSTEM_PROMPT as SMV2_SYSTEM_PROMPT,
)

# All 10 SourceMaterialType enum value strings that must appear in the prompt
_ALL_SMV2_TYPE_NAMES = [
    "novel_adaptation", "short_story_adaptation", "stage_adaptation",
    "true_story", "biography", "comic_adaptation", "folklore_adaptation",
    "video_game_adaptation", "remake", "tv_adaptation",
]


class TestSourceMaterialV2PromptConstants:
    """Tests for the source material V2 system prompt."""

    def test_prompt_is_non_empty(self) -> None:
        """SYSTEM_PROMPT is a non-empty string."""
        assert isinstance(SMV2_SYSTEM_PROMPT, str)
        assert len(SMV2_SYSTEM_PROMPT) > 0

    def test_prompt_contains_all_10_type_names(self) -> None:
        """Prompt contains all 10 SourceMaterialType headings."""
        for type_name in _ALL_SMV2_TYPE_NAMES:
            assert type_name in SMV2_SYSTEM_PROMPT, (
                f"Missing type name in prompt: {type_name}"
            )

    def test_prompt_contains_identification_not_fitting(self) -> None:
        """Prompt contains the 'IDENTIFICATION, not fitting' framing."""
        assert "IDENTIFICATION, not fitting" in SMV2_SYSTEM_PROMPT

    def test_prompt_contains_parametric_knowledge_allowance(self) -> None:
        """Prompt contains the 95% parametric knowledge threshold."""
        assert "95%" in SMV2_SYSTEM_PROMPT

    def test_prompt_contains_empty_list_guidance(self) -> None:
        """Prompt mentions empty list as valid output for original screenplays."""
        lower = SMV2_SYSTEM_PROMPT.lower()
        assert "empty list" in lower

    def test_prompt_contains_decision_rules(self) -> None:
        """Prompt includes DECISION RULES section."""
        assert "DECISION RULES" in SMV2_SYSTEM_PROMPT

    def test_prompt_contains_output_format(self) -> None:
        """Prompt includes OUTPUT FORMAT section."""
        assert "OUTPUT FORMAT" in SMV2_SYSTEM_PROMPT

    def test_prompt_mentions_source_material_types(self) -> None:
        """Prompt references the SourceMaterialType enum in output format."""
        assert "SourceMaterialType" in SMV2_SYSTEM_PROMPT

    def test_prompt_does_not_mention_franchise_lineage_as_output(self) -> None:
        """franchise_lineage does not appear in the OUTPUT section (removed from V2)."""
        # Find the OUTPUT FORMAT section and check it
        output_idx = SMV2_SYSTEM_PROMPT.find("OUTPUT FORMAT")
        assert output_idx >= 0
        output_section = SMV2_SYSTEM_PROMPT[output_idx:]
        assert "franchise_lineage" not in output_section

    def test_prompt_mentions_franchise_ignore(self) -> None:
        """Prompt tells the LLM to ignore franchise terms in source_material_hint."""
        lower = SMV2_SYSTEM_PROMPT.lower()
        # The prompt says franchise terms like "sequel" should be ignored
        assert "franchise" in lower
        assert "ignore" in lower
