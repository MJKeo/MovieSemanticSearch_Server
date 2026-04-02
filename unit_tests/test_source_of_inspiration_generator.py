"""
Unit tests for movie_ingestion.metadata_generation.generators.source_of_inspiration.

Tests prompt building, LLM call delegation, return value shape, and
error handling for the generate_source_of_inspiration function.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.generic_methods import LLMProvider
from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import SourceOfInspirationOutput
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from movie_ingestion.metadata_generation.generators.source_of_inspiration import (
    build_source_of_inspiration_user_prompt,
    generate_source_of_inspiration,
    GENERATION_TYPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLM_PATCH = "movie_ingestion.metadata_generation.generators.source_of_inspiration.generate_llm_response_async"


def _make_movie(**overrides) -> MovieInputData:
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        plot_keywords=["hacker"],
        overall_keywords=["cyberpunk"],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_soi_output() -> SourceOfInspirationOutput:
    return SourceOfInspirationOutput(
        source_material=["based on a novel"],
        franchise_lineage=["sequel"],
    )


# ---------------------------------------------------------------------------
# Tests: build_source_of_inspiration_user_prompt
# ---------------------------------------------------------------------------

class TestBuildSourceOfInspirationUserPrompt:
    def test_includes_title_with_year(self):
        movie = _make_movie(title="Inception", release_year=2010)
        result = build_source_of_inspiration_user_prompt(movie, None)
        assert "Inception (2010)" in result

    def test_includes_merged_keywords(self):
        movie = _make_movie(plot_keywords=["hacker"], overall_keywords=["cyberpunk"])
        result = build_source_of_inspiration_user_prompt(movie, None)
        assert "merged_keywords:" in result
        assert "hacker" in result
        assert "cyberpunk" in result

    def test_includes_source_material_hint(self):
        movie = _make_movie()
        result = build_source_of_inspiration_user_prompt(movie, "Critics noted source material.")
        assert "source_material_hint: Critics noted source material." in result

    def test_absent_fields_signal_not_available(self):
        """Empty keywords and absent hint are signaled as 'not available'."""
        movie = _make_movie(plot_keywords=[], overall_keywords=[])
        result = build_source_of_inspiration_user_prompt(movie, None)
        assert "merged_keywords: not available" in result
        assert "source_material_hint: not available" in result

    def test_does_not_accept_plot_synopsis_argument(self):
        """plot_synopsis was removed per ADR-033 — passing it should raise TypeError."""
        movie = _make_movie()
        with pytest.raises(TypeError):
            build_source_of_inspiration_user_prompt(movie, "A synopsis.", "insights")  # type: ignore[call-arg]

    def test_prompt_does_not_contain_plot_synopsis_section(self):
        """User prompt should not contain a plot_synopsis section."""
        movie = _make_movie()
        result = build_source_of_inspiration_user_prompt(movie, None)
        assert "plot_synopsis" not in result


# ---------------------------------------------------------------------------
# Tests: generate_source_of_inspiration — LLM delegation
# ---------------------------------------------------------------------------

class TestGenerateSourceOfInspiration:
    async def test_returns_output_and_token_usage(self):
        expected = _make_soi_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_source_of_inspiration(movie)

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)

    async def test_hardcoded_reasoning_effort(self):
        """Generator always passes reasoning_effort='low' and verbosity='low'."""
        mock_fn = AsyncMock(return_value=(_make_soi_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_source_of_inspiration(movie)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["reasoning_effort"] == "low"
        assert call_kwargs["verbosity"] == "low"


# ---------------------------------------------------------------------------
# Tests: generate_source_of_inspiration — error paths
# ---------------------------------------------------------------------------

class TestGenerateSourceOfInspirationErrors:
    async def test_wraps_llm_exception(self):
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_source_of_inspiration(movie)

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_raises_empty_response_error(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError):
                await generate_source_of_inspiration(movie)

    async def test_error_chains_original_cause(self):
        original = ValueError("original")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_source_of_inspiration(movie)

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: signature lockdown — removed parameters
# ---------------------------------------------------------------------------

class TestSourceOfInspirationSignatureLockdown:
    def test_generate_does_not_accept_provider_kwarg(self):
        """provider parameter was removed — passing it should raise TypeError."""
        import inspect
        sig = inspect.signature(generate_source_of_inspiration)
        assert "provider" not in sig.parameters

    def test_generate_does_not_accept_model_kwarg(self):
        """model parameter was removed — passing it should raise TypeError."""
        import inspect
        sig = inspect.signature(generate_source_of_inspiration)
        assert "model" not in sig.parameters

    async def test_hardcoded_llm_params(self):
        """All 6 hardcoded params are passed to the LLM call."""
        from movie_ingestion.metadata_generation.prompts.source_of_inspiration import SYSTEM_PROMPT

        mock_fn = AsyncMock(return_value=(_make_soi_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_source_of_inspiration(movie)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["provider"] == LLMProvider.OPENAI
        assert call_kwargs["model"] == "gpt-5-mini"
        assert call_kwargs["system_prompt"] == SYSTEM_PROMPT
        assert call_kwargs["response_format"] is SourceOfInspirationOutput
        assert call_kwargs["reasoning_effort"] == "low"
        assert call_kwargs["verbosity"] == "low"


# ---------------------------------------------------------------------------
# Tests: no genres field in prompt
# ---------------------------------------------------------------------------

class TestSourceOfInspirationPromptFields:
    def test_no_genres_in_prompt(self):
        """source_of_inspiration does not include genres in the prompt."""
        movie_with_genres = MovieInputData(
            tmdb_id=12345, title="Test Movie", release_year=2020,
            genres=["Action", "Sci-Fi"],
            plot_keywords=["hacker"], overall_keywords=["cyberpunk"],
        )
        result = build_source_of_inspiration_user_prompt(movie_with_genres, None)
        assert "genres" not in result

    def test_generate_does_not_accept_plot_synopsis(self):
        """generate_source_of_inspiration does not accept plot_synopsis parameter."""
        # The old signature had plot_synopsis — verify it's gone
        import inspect
        sig = inspect.signature(generate_source_of_inspiration)
        assert "plot_synopsis" not in sig.parameters
