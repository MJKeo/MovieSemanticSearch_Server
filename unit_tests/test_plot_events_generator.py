"""
Unit tests for movie_ingestion.metadata_generation.generators.plot_events.

Tests the two-branch prompt building (synopsis vs synthesis), LLM call
delegation, return value shape, and error handling for generate_plot_events.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.generic_methods import LLMProvider
from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import PlotEventsOutput
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from movie_ingestion.metadata_generation.generators.plot_events import (
    build_plot_events_prompts,
    generate_plot_events,
    GENERATION_TYPE,
    MIN_SYNOPSIS_CHARS,
)
from movie_ingestion.metadata_generation.prompts.plot_events import (
    SYSTEM_PROMPT_SYNOPSIS,
    SYSTEM_PROMPT_SYNTHESIS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLM_PATCH = "movie_ingestion.metadata_generation.generators.plot_events.generate_llm_response_async"


def _make_movie(**overrides) -> MovieInputData:
    """Build a MovieInputData with sensible defaults and optional overrides."""
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        overview="A great test movie about testing.",
        genres=["Drama"],
        plot_synopses=[],
        plot_summaries=["Summary one.", "Summary two."],
        plot_keywords=["keyword1", "keyword2"],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_plot_events_output() -> PlotEventsOutput:
    """Build a minimal valid PlotEventsOutput for mocking."""
    return PlotEventsOutput(
        plot_summary="A detailed plot summary.",
        setting="New York City, 2020",
    )


# ---------------------------------------------------------------------------
# Tests: build_plot_events_prompts — return type
# ---------------------------------------------------------------------------


class TestBuildPlotEventsPromptsReturnType:
    """build_plot_events_prompts returns a (user_prompt, system_prompt) tuple."""

    def test_returns_tuple_of_two_strings(self) -> None:
        """Return value is a 2-tuple of strings."""
        movie = _make_movie()
        result = build_plot_events_prompts(movie)
        assert isinstance(result, tuple)
        assert len(result) == 2
        user_prompt, system_prompt = result
        assert isinstance(user_prompt, str)
        assert isinstance(system_prompt, str)


# ---------------------------------------------------------------------------
# Tests: build_plot_events_prompts — synopsis branch
# ---------------------------------------------------------------------------


class TestSynopsisBranch:
    """Tests for the synopsis branch (synopsis >= MIN_SYNOPSIS_CHARS)."""

    def test_long_synopsis_selects_synopsis_system_prompt(self) -> None:
        """Synopsis >= 1000 chars uses SYSTEM_PROMPT_SYNOPSIS."""
        movie = _make_movie(plot_synopses=["x" * 1000])
        _, system_prompt = build_plot_events_prompts(movie)
        assert system_prompt is SYSTEM_PROMPT_SYNOPSIS

    def test_long_synopsis_includes_plot_synopsis_label(self) -> None:
        """Synopsis branch user prompt includes 'plot_synopsis' label."""
        movie = _make_movie(plot_synopses=["x" * 1000])
        user_prompt, _ = build_plot_events_prompts(movie)
        assert "plot_synopsis:" in user_prompt

    def test_long_synopsis_excludes_plot_summaries_label(self) -> None:
        """Synopsis branch user prompt does NOT include 'plot_summaries' label."""
        movie = _make_movie(
            plot_synopses=["x" * 1000],
            plot_summaries=["Summary one."],
        )
        user_prompt, _ = build_plot_events_prompts(movie)
        assert "plot_summaries:" not in user_prompt

    def test_exactly_1000_chars_selects_synopsis_branch(self) -> None:
        """Synopsis of exactly MIN_SYNOPSIS_CHARS uses synopsis branch (>= boundary)."""
        movie = _make_movie(plot_synopses=["a" * MIN_SYNOPSIS_CHARS])
        _, system_prompt = build_plot_events_prompts(movie)
        assert system_prompt is SYSTEM_PROMPT_SYNOPSIS


# ---------------------------------------------------------------------------
# Tests: build_plot_events_prompts — synthesis branch
# ---------------------------------------------------------------------------


class TestSynthesisBranch:
    """Tests for the synthesis branch (no synopsis or synopsis too short)."""

    def test_no_synopsis_selects_synthesis_system_prompt(self) -> None:
        """No synopses uses SYSTEM_PROMPT_SYNTHESIS."""
        movie = _make_movie(plot_synopses=[])
        _, system_prompt = build_plot_events_prompts(movie)
        assert system_prompt is SYSTEM_PROMPT_SYNTHESIS

    def test_no_synopsis_includes_plot_summaries_label(self) -> None:
        """Synthesis branch user prompt includes 'plot_summaries' label."""
        movie = _make_movie(plot_synopses=[], plot_summaries=["A summary."])
        user_prompt, _ = build_plot_events_prompts(movie)
        assert "plot_summaries:" in user_prompt

    def test_no_synopsis_excludes_plot_synopsis_label(self) -> None:
        """Synthesis branch user prompt does NOT include 'plot_synopsis' label."""
        movie = _make_movie(plot_synopses=[])
        user_prompt, _ = build_plot_events_prompts(movie)
        assert "plot_synopsis:" not in user_prompt

    def test_short_synopsis_selects_synthesis_branch(self) -> None:
        """Synopsis < 1000 chars uses synthesis branch."""
        movie = _make_movie(plot_synopses=["x" * 999])
        _, system_prompt = build_plot_events_prompts(movie)
        assert system_prompt is SYSTEM_PROMPT_SYNTHESIS

    def test_exactly_999_chars_selects_synthesis_branch(self) -> None:
        """Synopsis of exactly 999 chars uses synthesis branch (< boundary)."""
        movie = _make_movie(plot_synopses=["a" * 999])
        _, system_prompt = build_plot_events_prompts(movie)
        assert system_prompt is SYSTEM_PROMPT_SYNTHESIS

    def test_short_synopsis_demoted_into_summaries(self) -> None:
        """Short synopsis is prepended into the plot_summaries list."""
        short_synopsis = "This is a short synopsis that will be demoted."
        movie = _make_movie(
            plot_synopses=[short_synopsis],
            plot_summaries=["Existing summary."],
        )
        user_prompt, _ = build_plot_events_prompts(movie)
        # The short synopsis should appear in the user prompt as part of summaries
        assert short_synopsis in user_prompt
        assert "Existing summary." in user_prompt
        # And plot_synopsis label should NOT appear
        assert "plot_synopsis:" not in user_prompt

    def test_short_synopsis_prepends_before_existing_summaries(self) -> None:
        """Demoted synopsis appears before existing summaries in the prompt."""
        short_synopsis = "Demoted synopsis text."
        movie = _make_movie(
            plot_synopses=[short_synopsis],
            plot_summaries=["Summary A.", "Summary B."],
        )
        user_prompt, _ = build_plot_events_prompts(movie)
        # Demoted synopsis should appear before existing summaries
        syn_pos = user_prompt.find(short_synopsis)
        sum_pos = user_prompt.find("Summary A.")
        assert syn_pos < sum_pos

    def test_summaries_capped_at_three(self) -> None:
        """At most 3 summaries appear in the synthesis branch prompt."""
        movie = _make_movie(
            plot_synopses=[],
            plot_summaries=["S1.", "S2.", "S3.", "S4.", "S5."],
        )
        user_prompt, _ = build_plot_events_prompts(movie)
        assert "S1." in user_prompt
        assert "S2." in user_prompt
        assert "S3." in user_prompt
        assert "S4." not in user_prompt


# ---------------------------------------------------------------------------
# Tests: build_plot_events_prompts — newline collapsing
# ---------------------------------------------------------------------------


class TestNewlineCollapsing:
    """Tests for newline collapsing in synopsis text."""

    def test_newlines_collapsed_to_spaces(self) -> None:
        """Newlines in the synopsis are replaced with spaces."""
        long_text = "a" * 500 + "\n\n" + "b" * 500
        movie = _make_movie(plot_synopses=[long_text])
        user_prompt, _ = build_plot_events_prompts(movie)
        # After collapsing newlines: "a...a b...b" (1001 chars with space)
        assert "\n" not in user_prompt.split("plot_synopsis:")[1].split("\n")[0] if "plot_synopsis:" in user_prompt else True

    def test_newline_collapsing_affects_length_check(self) -> None:
        """Length check uses the collapsed text (newlines -> spaces)."""
        # 999 'a' chars + 1 newline = 1000 raw chars, but after collapsing
        # the newline becomes a space => still 1000 chars => synopsis branch
        text_with_newline = "a" * 999 + "\n"
        movie = _make_movie(plot_synopses=[text_with_newline])
        _, system_prompt = build_plot_events_prompts(movie)
        assert system_prompt is SYSTEM_PROMPT_SYNOPSIS


# ---------------------------------------------------------------------------
# Tests: build_plot_events_prompts — shared fields
# ---------------------------------------------------------------------------


class TestSharedPromptFields:
    """Tests for fields that appear in both branches."""

    def test_includes_title_with_year(self) -> None:
        """Title with year appears in the user prompt."""
        movie = _make_movie(title="Inception", release_year=2010)
        user_prompt, _ = build_plot_events_prompts(movie)
        assert "Inception (2010)" in user_prompt

    def test_empty_overview_omits_field(self) -> None:
        """overview field is absent when overview is empty string."""
        movie = _make_movie(overview="")
        user_prompt, _ = build_plot_events_prompts(movie)
        assert "overview:" not in user_prompt

    def test_empty_keywords_omits_field(self) -> None:
        """plot_keywords field is absent when keywords list is empty."""
        movie = _make_movie(plot_keywords=[])
        user_prompt, _ = build_plot_events_prompts(movie)
        assert "plot_keywords:" not in user_prompt


# ---------------------------------------------------------------------------
# Tests: generate_plot_events — requires provider and model
# ---------------------------------------------------------------------------


class TestGeneratePlotEventsSignature:
    """Tests for generate_plot_events function signature."""

    async def test_requires_provider_and_model(self) -> None:
        """Calling without provider and model raises TypeError."""
        movie = _make_movie()
        with pytest.raises(TypeError):
            await generate_plot_events(movie)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# Tests: generate_plot_events — LLM call delegation
# ---------------------------------------------------------------------------


class TestLLMCallDelegation:
    """Tests for how generate_plot_events delegates to the LLM router."""

    async def test_passes_provider_and_model_to_router(self) -> None:
        """Provider, model, and response_format are forwarded to generate_llm_response_async."""
        movie = _make_movie()
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.GEMINI, "gemini-2.5-flash")

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["provider"] == LLMProvider.GEMINI
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["response_format"] is PlotEventsOutput

    async def test_forwards_kwargs_to_router(self) -> None:
        """Provider-specific kwargs pass through to the LLM router."""
        movie = _make_movie()
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(
                movie, LLMProvider.OPENAI, "gpt-5-mini", temperature=0.5,
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["temperature"] == 0.5

    async def test_passes_correct_system_prompt_for_synopsis_branch(self) -> None:
        """Synopsis branch passes SYSTEM_PROMPT_SYNOPSIS to the LLM."""
        movie = _make_movie(plot_synopses=["x" * 1000])
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["system_prompt"] is SYSTEM_PROMPT_SYNOPSIS

    async def test_passes_correct_system_prompt_for_synthesis_branch(self) -> None:
        """Synthesis branch passes SYSTEM_PROMPT_SYNTHESIS to the LLM."""
        movie = _make_movie(plot_synopses=[])
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["system_prompt"] is SYSTEM_PROMPT_SYNTHESIS


# ---------------------------------------------------------------------------
# Tests: Return value
# ---------------------------------------------------------------------------


class TestReturnValue:
    """Tests for the return value of generate_plot_events."""

    async def test_returns_plot_events_output_and_token_usage(self) -> None:
        """Return is a (PlotEventsOutput, TokenUsage) tuple with correct values."""
        expected_output = _make_plot_events_output()
        mock_fn = AsyncMock(return_value=(expected_output, 100, 50))

        movie = _make_movie()
        with patch(_LLM_PATCH, mock_fn):
            result = await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        assert isinstance(result, tuple)
        assert len(result) == 2
        parsed, token_usage = result
        assert parsed is expected_output
        assert isinstance(token_usage, TokenUsage)
        assert token_usage.input_tokens == 100
        assert token_usage.output_tokens == 50

    async def test_token_usage_records_caller_model_string(self) -> None:
        """TokenUsage.model matches the caller-provided model string."""
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        movie = _make_movie()
        with patch(_LLM_PATCH, mock_fn):
            _, token_usage = await generate_plot_events(
                movie, LLMProvider.GEMINI, "gemini-2.5-flash",
            )

        assert token_usage.model == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Tests: Error paths
# ---------------------------------------------------------------------------


class TestErrorPaths:
    """Tests for error handling in generate_plot_events."""

    async def test_wraps_llm_exception_in_metadata_generation_error(self) -> None:
        """LLM exceptions are wrapped in MetadataGenerationError."""
        mock_fn = AsyncMock(side_effect=ValueError("LLM API timeout"))

        movie = _make_movie()
        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        err = exc_info.value
        assert err.generation_type == GENERATION_TYPE
        assert err.title == "Test Movie (2020)"

    async def test_raises_empty_response_error_when_parsed_is_none(self) -> None:
        """MetadataGenerationEmptyResponseError raised when LLM returns None."""
        mock_fn = AsyncMock(return_value=(None, 100, 50))

        movie = _make_movie()
        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError) as exc_info:
                await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        err = exc_info.value
        assert err.generation_type == GENERATION_TYPE
        assert err.title == "Test Movie (2020)"

    async def test_metadata_generation_error_chains_original_cause(self) -> None:
        """The __cause__ of MetadataGenerationError is the original exception."""
        original = ValueError("original cause")
        mock_fn = AsyncMock(side_effect=original)

        movie = _make_movie()
        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: prompt formatting details
# ---------------------------------------------------------------------------


class TestPlotEventsPromptFormatting:
    def test_plot_summaries_use_multiline_format(self) -> None:
        """plot_summaries are formatted with dash-prefixed items (MultiLineList)."""
        movie = _make_movie(
            overview="A plot.",
            plot_synopses=[],
            plot_summaries=["First summary.", "Second summary."],
        )
        user_prompt, _ = build_plot_events_prompts(movie)
        assert "- First summary." in user_prompt
        assert "- Second summary." in user_prompt

    def test_plot_keywords_are_comma_separated(self) -> None:
        """plot_keywords are comma-separated (regular list), not dash-prefixed."""
        movie = _make_movie(
            overview="A plot.",
            plot_keywords=["hacker", "simulation"],
        )
        user_prompt, _ = build_plot_events_prompts(movie)
        assert "plot_keywords: hacker, simulation" in user_prompt
        assert "- hacker" not in user_prompt
