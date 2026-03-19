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
        sources_of_inspiration=["original screenplay"],
        production_mediums=["live-action"],
    )


# ---------------------------------------------------------------------------
# Tests: build_source_of_inspiration_user_prompt
# ---------------------------------------------------------------------------

class TestBuildSourceOfInspirationUserPrompt:
    def test_includes_title_with_year(self):
        movie = _make_movie(title="Inception", release_year=2010)
        result = build_source_of_inspiration_user_prompt(movie, None, None)
        assert "Inception (2010)" in result

    def test_includes_plot_synopsis(self):
        movie = _make_movie()
        result = build_source_of_inspiration_user_prompt(movie, "A synopsis.", None)
        assert "plot_synopsis: A synopsis." in result

    def test_includes_merged_keywords(self):
        movie = _make_movie(plot_keywords=["hacker"], overall_keywords=["cyberpunk"])
        result = build_source_of_inspiration_user_prompt(movie, None, None)
        assert "merged_keywords:" in result
        assert "hacker" in result
        assert "cyberpunk" in result

    def test_includes_review_insights_brief(self):
        movie = _make_movie()
        result = build_source_of_inspiration_user_prompt(movie, None, "Critics noted source material.")
        assert "review_insights_brief: Critics noted source material." in result

    def test_omits_none_fields(self):
        movie = _make_movie(plot_keywords=[], overall_keywords=[])
        result = build_source_of_inspiration_user_prompt(movie, None, None)
        assert "plot_synopsis" not in result
        assert "merged_keywords" not in result
        assert "review_insights_brief" not in result


# ---------------------------------------------------------------------------
# Tests: generate_source_of_inspiration — LLM delegation
# ---------------------------------------------------------------------------

class TestGenerateSourceOfInspiration:
    async def test_returns_output_and_token_usage(self):
        expected = _make_soi_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_source_of_inspiration(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)

    async def test_merges_default_kwargs(self):
        mock_fn = AsyncMock(return_value=(_make_soi_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_source_of_inspiration(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["reasoning_effort"] == "low"


# ---------------------------------------------------------------------------
# Tests: generate_source_of_inspiration — error paths
# ---------------------------------------------------------------------------

class TestGenerateSourceOfInspirationErrors:
    async def test_wraps_llm_exception(self):
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_source_of_inspiration(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_raises_empty_response_error(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError):
                await generate_source_of_inspiration(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

    async def test_error_chains_original_cause(self):
        original = ValueError("original")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_source_of_inspiration(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: system_prompt and response_format override
# ---------------------------------------------------------------------------

class TestSourceOfInspirationOverrides:
    async def test_custom_system_prompt_forwarded(self):
        """A custom system_prompt is forwarded to the LLM call."""
        mock_fn = AsyncMock(return_value=(_make_soi_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_source_of_inspiration(
                movie, system_prompt="CUSTOM_PROMPT",
                provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["system_prompt"] == "CUSTOM_PROMPT"

    async def test_custom_response_format_forwarded(self):
        """A custom response_format is forwarded to the LLM call."""
        from movie_ingestion.metadata_generation.schemas import SourceOfInspirationWithJustificationsOutput

        mock_fn = AsyncMock(return_value=(_make_soi_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_source_of_inspiration(
                movie,
                response_format=SourceOfInspirationWithJustificationsOutput,
                provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["response_format"] is SourceOfInspirationWithJustificationsOutput


# ---------------------------------------------------------------------------
# Tests: no genres field in prompt
# ---------------------------------------------------------------------------

class TestSourceOfInspirationPromptFields:
    def test_no_genres_in_prompt(self):
        """source_of_inspiration does not include genres in the prompt."""
        movie = _make_movie(genres=["Action", "Sci-Fi"])
        # Need to add genres to _make_movie to test it's excluded
        movie_with_genres = MovieInputData(
            tmdb_id=12345, title="Test Movie", release_year=2020,
            genres=["Action", "Sci-Fi"],
            plot_keywords=["hacker"], overall_keywords=["cyberpunk"],
        )
        result = build_source_of_inspiration_user_prompt(movie_with_genres, None, None)
        assert "genres" not in result
