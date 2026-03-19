"""
Unit tests for movie_ingestion.metadata_generation.generators.production_keywords.

Tests prompt building, LLM call delegation, return value shape, and
error handling for the generate_production_keywords function.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.generic_methods import LLMProvider
from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import ProductionKeywordsOutput
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from movie_ingestion.metadata_generation.generators.production_keywords import (
    build_production_keywords_user_prompt,
    generate_production_keywords,
    GENERATION_TYPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLM_PATCH = "movie_ingestion.metadata_generation.generators.production_keywords.generate_llm_response_async"


def _make_movie(**overrides) -> MovieInputData:
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        plot_keywords=["CGI", "practical effects"],
        overall_keywords=["IMAX"],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_pk_output() -> ProductionKeywordsOutput:
    return ProductionKeywordsOutput(terms=["CGI", "IMAX"])


# ---------------------------------------------------------------------------
# Tests: build_production_keywords_user_prompt
# ---------------------------------------------------------------------------

class TestBuildProductionKeywordsUserPrompt:
    def test_includes_title_with_year(self):
        movie = _make_movie(title="Inception", release_year=2010)
        result = build_production_keywords_user_prompt(movie)
        assert "Inception (2010)" in result

    def test_includes_merged_keywords(self):
        movie = _make_movie(
            plot_keywords=["CGI"],
            overall_keywords=["IMAX"],
        )
        result = build_production_keywords_user_prompt(movie)
        assert "merged_keywords:" in result
        # merged_keywords normalizes to lowercase
        assert "cgi" in result
        assert "imax" in result

    def test_no_other_fields(self):
        """Only title and merged_keywords in prompt — no genres, no synopsis."""
        movie = _make_movie(genres=["Action"])
        result = build_production_keywords_user_prompt(movie)
        assert "genres" not in result
        assert "plot_synopsis" not in result
        assert "review_insights_brief" not in result

    def test_omits_merged_keywords_when_empty(self):
        movie = _make_movie(plot_keywords=[], overall_keywords=[])
        result = build_production_keywords_user_prompt(movie)
        assert "merged_keywords" not in result


# ---------------------------------------------------------------------------
# Tests: generate_production_keywords — LLM delegation
# ---------------------------------------------------------------------------

class TestGenerateProductionKeywords:
    async def test_returns_output_and_token_usage(self):
        expected = _make_pk_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_production_keywords(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)

    async def test_merges_default_kwargs(self):
        mock_fn = AsyncMock(return_value=(_make_pk_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_production_keywords(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["reasoning_effort"] == "low"


# ---------------------------------------------------------------------------
# Tests: generate_production_keywords — error paths
# ---------------------------------------------------------------------------

class TestGenerateProductionKeywordsErrors:
    async def test_wraps_llm_exception(self):
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_production_keywords(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_raises_empty_response_error(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError):
                await generate_production_keywords(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

    async def test_error_chains_original_cause(self):
        original = ValueError("original")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_production_keywords(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: prompt contains only title and merged_keywords
# ---------------------------------------------------------------------------

class TestProductionKeywordsPromptFields:
    def test_prompt_includes_only_title_and_merged_keywords(self):
        """Prompt has only title and merged_keywords — no other fields."""
        movie = _make_movie(genres=["Action"])
        result = build_production_keywords_user_prompt(movie)
        assert "title:" in result
        assert "merged_keywords:" in result
        assert "genres" not in result
        assert "plot_synopsis" not in result
        assert "review_insights_brief" not in result

    def test_prompt_omits_merged_keywords_when_empty(self):
        """merged_keywords is excluded when both keyword lists are empty."""
        movie = _make_movie(plot_keywords=[], overall_keywords=[])
        result = build_production_keywords_user_prompt(movie)
        assert "merged_keywords" not in result
