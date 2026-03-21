"""
Unit tests for movie_ingestion.metadata_generation.generators.plot_events.

Tests prompt building, LLM call delegation, return value shape, and
error handling for the generate_plot_events function.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, MagicMock, patch

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
        plot_synopses=["First synopsis about the plot."],
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
# Tests: build_plot_events_user_prompt (direct)
# ---------------------------------------------------------------------------


class TestBuildPlotEventsUserPrompt:
    """Tests for the standalone build_plot_events_user_prompt function."""

    def test_build_plot_events_user_prompt_returns_string(self) -> None:
        """Return value is a str."""
        movie = _make_movie()
        result = build_plot_events_prompts(movie)
        assert isinstance(result, str)

    def test_build_plot_events_user_prompt_includes_title_with_year(self) -> None:
        """Title with year appears in the prompt."""
        movie = _make_movie(title="Inception", release_year=2010)
        result = build_plot_events_prompts(movie)
        assert "Inception (2010)" in result

    def test_build_plot_events_user_prompt_first_synopsis_only(self) -> None:
        """Only the first synopsis is used when multiple exist."""
        movie = _make_movie(plot_synopses=[
            "First synopsis about the story.",
            "Second synopsis with different details.",
            "Third synopsis entirely different.",
        ])
        result = build_plot_events_prompts(movie)
        assert "First synopsis about the story." in result
        assert "Second synopsis" not in result
        assert "Third synopsis" not in result

    def test_build_plot_events_user_prompt_collapses_newlines(self) -> None:
        """Newlines in the synopsis are collapsed to single spaces."""
        movie = _make_movie(plot_synopses=["Line one.\n\nLine two.\nLine three."])
        result = build_plot_events_prompts(movie)
        assert "Line one. Line two. Line three." in result

    def test_build_plot_events_user_prompt_caps_summaries_at_three(self) -> None:
        """At most 3 summaries appear in the prompt."""
        movie = _make_movie(plot_summaries=[
            "Summary 1.", "Summary 2.", "Summary 3.", "Summary 4.", "Summary 5.",
        ])
        result = build_plot_events_prompts(movie)
        assert "Summary 1." in result
        assert "Summary 2." in result
        assert "Summary 3." in result
        assert "Summary 4." not in result

    def test_build_plot_events_user_prompt_empty_synopses_omits_field(self) -> None:
        """plot_synopsis field is absent when synopses list is empty."""
        movie = _make_movie(plot_synopses=[])
        result = build_plot_events_prompts(movie)
        assert "plot_synopsis:" not in result

    def test_build_plot_events_user_prompt_empty_overview_omits_field(self) -> None:
        """overview field is absent when overview is empty string."""
        movie = _make_movie(overview="")
        result = build_plot_events_prompts(movie)
        assert "overview:" not in result

    def test_build_plot_events_user_prompt_empty_keywords_omits_field(self) -> None:
        """plot_keywords field is absent when keywords list is empty."""
        movie = _make_movie(plot_keywords=[])
        result = build_plot_events_prompts(movie)
        assert "plot_keywords:" not in result

    def test_build_plot_events_user_prompt_all_fields_present(self) -> None:
        """All fields appear when movie has complete data."""
        movie = _make_movie(
            overview="A great test movie about testing.",
            plot_synopses=["A detailed synopsis."],
            plot_summaries=["Summary one."],
            plot_keywords=["keyword1", "keyword2"],
        )
        result = build_plot_events_prompts(movie)
        assert "title:" in result
        assert "overview:" in result
        assert "plot_synopsis:" in result
        assert "plot_summaries:" in result
        assert "plot_keywords:" in result

    def test_build_plot_events_user_prompt_no_year(self) -> None:
        """Movie with release_year=None returns just the title, prompt still well-formed."""
        movie = _make_movie(release_year=None)
        result = build_plot_events_prompts(movie)
        assert "title: Test Movie" in result
        # Should not contain parenthesized year
        assert "(None)" not in result

    def test_build_plot_events_user_prompt_empty_strings_in_summaries(self) -> None:
        """Empty strings in plot_summaries are included (not filtered as None)."""
        movie = _make_movie(plot_summaries=["", "Real summary."])
        result = build_plot_events_prompts(movie)
        assert "Real summary." in result


# ---------------------------------------------------------------------------
# Tests: Prompt building (via generate_plot_events)
# ---------------------------------------------------------------------------


class TestPromptBuilding:
    """Tests for how generate_plot_events constructs the user prompt."""

    async def test_uses_first_synopsis_only(self) -> None:
        """Only the first synopsis appears in the user_prompt, even with multiple synopses."""
        movie = _make_movie(plot_synopses=["First synopsis.", "Second synopsis.", "Third synopsis."])
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        user_prompt = mock_fn.call_args[1]["user_prompt"]
        assert "First synopsis." in user_prompt
        assert "Second synopsis." not in user_prompt
        assert "Third synopsis." not in user_prompt

    async def test_collapses_newlines_in_synopsis(self) -> None:
        """Newlines in the synopsis are collapsed to single spaces."""
        movie = _make_movie(plot_synopses=["Line one.\n\nLine two.\nLine three."])
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        user_prompt = mock_fn.call_args[1]["user_prompt"]
        assert "\n\n" not in user_prompt.split("plot_synopsis:")[1] if "plot_synopsis:" in user_prompt else True
        # The synopsis portion should have spaces instead of newlines
        assert "Line one. Line two. Line three." in user_prompt

    async def test_caps_plot_summaries_at_three(self) -> None:
        """Only the first 3 plot summaries appear in the prompt."""
        movie = _make_movie(plot_summaries=[
            "Summary 1.", "Summary 2.", "Summary 3.", "Summary 4.", "Summary 5.",
        ])
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        user_prompt = mock_fn.call_args[1]["user_prompt"]
        assert "Summary 1." in user_prompt
        assert "Summary 2." in user_prompt
        assert "Summary 3." in user_prompt
        assert "Summary 4." not in user_prompt
        assert "Summary 5." not in user_prompt

    async def test_empty_synopses_passes_none(self) -> None:
        """Empty plot_synopses list results in plot_synopsis being omitted from prompt."""
        movie = _make_movie(plot_synopses=[])
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        user_prompt = mock_fn.call_args[1]["user_prompt"]
        assert "plot_synopsis:" not in user_prompt

    async def test_empty_plot_summaries_passes_none(self) -> None:
        """Empty plot_summaries list results in plot_summaries being omitted from prompt."""
        movie = _make_movie(plot_summaries=[])
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        user_prompt = mock_fn.call_args[1]["user_prompt"]
        assert "plot_summaries:" not in user_prompt

    async def test_empty_overview_passes_none(self) -> None:
        """Empty overview string is converted to None (omitted from prompt)."""
        movie = _make_movie(overview="")
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        user_prompt = mock_fn.call_args[1]["user_prompt"]
        assert "overview:" not in user_prompt

    async def test_empty_plot_keywords_passes_none(self) -> None:
        """Empty plot_keywords list results in keywords being omitted from prompt."""
        movie = _make_movie(plot_keywords=[])
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        user_prompt = mock_fn.call_args[1]["user_prompt"]
        assert "plot_keywords:" not in user_prompt


# ---------------------------------------------------------------------------
# Tests: LLM call delegation
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
        """TokenUsage.model matches the caller-provided model string, not a hardcoded value."""
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
        """LLM exceptions are wrapped in MetadataGenerationError with correct attributes."""
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
        """The __cause__ of MetadataGenerationError is the original exception (via 'from e')."""
        original = ValueError("original cause")
        mock_fn = AsyncMock(side_effect=original)

        movie = _make_movie()
        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_plot_events(movie, LLMProvider.OPENAI, "gpt-5-mini")

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: _DEFAULT_KWARGS merge behavior
# ---------------------------------------------------------------------------

class TestPlotEventsDefaultKwargsMerge:
    async def test_default_kwargs_present_when_no_override(self) -> None:
        """Default thinking_config is present when caller provides no kwargs."""
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))
        movie = _make_movie(overview="A long enough overview for plot events.")

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(movie, LLMProvider.GEMINI, "gemini-2.5-flash-lite")

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["temperature"] == 0.2
        assert call_kwargs["thinking_config"] == {"thinking_budget": 1024}

    async def test_caller_kwargs_override_defaults(self) -> None:
        """Caller-provided kwargs override _DEFAULT_KWARGS values."""
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), 100, 50))
        movie = _make_movie(overview="A long enough overview for plot events.")

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_events(
                movie, LLMProvider.GEMINI, "gemini-2.5-flash-lite",
                temperature=0.5,
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        # thinking_config default should still be present
        assert call_kwargs["thinking_config"] == {"thinking_budget": 1024}


# ---------------------------------------------------------------------------
# Tests: prompt formatting details
# ---------------------------------------------------------------------------

class TestPlotEventsPromptFormatting:
    def test_plot_summaries_use_multiline_format(self) -> None:
        """plot_summaries are formatted with dash-prefixed items (MultiLineList)."""
        movie = _make_movie(
            overview="A plot.",
            plot_summaries=["First summary.", "Second summary."],
        )
        prompt = build_plot_events_prompts(movie)
        # MultiLineList renders as "key:\n- item1\n- item2"
        assert "- First summary." in prompt
        assert "- Second summary." in prompt

    def test_plot_keywords_are_comma_separated(self) -> None:
        """plot_keywords are comma-separated (regular list), not dash-prefixed."""
        movie = _make_movie(
            overview="A plot.",
            plot_keywords=["hacker", "simulation"],
        )
        prompt = build_plot_events_prompts(movie)
        assert "plot_keywords: hacker, simulation" in prompt
        # Should NOT use MultiLineList dash format
        assert "- hacker" not in prompt
