"""
Unit tests for metadata generation prompt constants.

Covers:
  - plot_events: SYSTEM_PROMPT_SYNOPSIS, SYSTEM_PROMPT_SYNTHESIS
  - source_of_inspiration: SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_JUSTIFICATIONS
"""

from movie_ingestion.metadata_generation.prompts.plot_events import (
    SYSTEM_PROMPT_SYNOPSIS,
    SYSTEM_PROMPT_SYNTHESIS,
)
from movie_ingestion.metadata_generation.prompts.source_of_inspiration import (
    SYSTEM_PROMPT,
    SYSTEM_PROMPT_WITH_JUSTIFICATIONS,
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
    """Tests for the source of inspiration system prompt variants."""

    def test_base_prompt_does_not_mention_justification_in_output(self) -> None:
        """SYSTEM_PROMPT does not mention 'justification' in the OUTPUT section."""
        # Find the OUTPUT section and check it
        lower = SYSTEM_PROMPT.lower()
        output_idx = lower.find("output")
        if output_idx >= 0:
            output_section = lower[output_idx:]
            assert "justification" not in output_section

    def test_with_justifications_prompt_mentions_justification(self) -> None:
        """SYSTEM_PROMPT_WITH_JUSTIFICATIONS mentions 'justification'."""
        lower = SYSTEM_PROMPT_WITH_JUSTIFICATIONS.lower()
        assert "justification" in lower

    def test_both_prompts_contain_parametric_knowledge_allowance(self) -> None:
        """Both prompts contain the parametric knowledge allowance text."""
        for prompt in (SYSTEM_PROMPT, SYSTEM_PROMPT_WITH_JUSTIFICATIONS):
            lower = prompt.lower()
            assert "parametric knowledge" in lower
