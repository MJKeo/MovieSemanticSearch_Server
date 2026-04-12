"""
Unit tests for movie_ingestion.metadata_generation.generators.production_techniques.

Tests prompt building, schema validation shape, LLM call delegation,
error handling, and signature lockdown for generate_production_techniques.
"""

import inspect
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from implementation.llms.generic_methods import LLMProvider
from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationEmptyResponseError,
    MetadataGenerationError,
)
from movie_ingestion.metadata_generation.generators.production_techniques import (
    GENERATION_TYPE,
    build_production_techniques_user_prompt,
    generate_production_techniques,
)
from movie_ingestion.metadata_generation.inputs import MovieInputData
from schemas.metadata import ProductionTechniquesOutput


_LLM_PATCH = (
    "movie_ingestion.metadata_generation.generators.production_techniques."
    "generate_llm_response_async"
)


def _make_movie(**overrides) -> MovieInputData:
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        plot_keywords=["single-take"],
        overall_keywords=["anthology", "mockumentary", "drama"],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_output() -> ProductionTechniquesOutput:
    return ProductionTechniquesOutput(terms=["single-take", "mockumentary"])


class TestProductionTechniquesSchema:
    def test_validates_terms_payload(self):
        output = ProductionTechniquesOutput(terms=["IMAX", "single-take"])
        assert output.terms == ["IMAX", "single-take"]

    def test_rejects_extra_fields(self):
        with pytest.raises(ValidationError):
            ProductionTechniquesOutput(terms=["IMAX"], extra_field="nope")


class TestBuildProductionTechniquesUserPrompt:
    def test_includes_title_with_year(self):
        movie = _make_movie(title="Birdman", release_year=2014)
        result = build_production_techniques_user_prompt(movie)
        assert "Birdman (2014)" in result

    def test_includes_separate_keyword_fields(self):
        movie = _make_movie(
            plot_keywords=["single-take"],
            overall_keywords=["anthology", "mockumentary", "drama"],
        )
        result = build_production_techniques_user_prompt(movie)
        assert "plot_keywords:" in result
        assert "overall_keywords:" in result
        assert "single-take" in result
        assert "anthology" in result
        assert "mockumentary" in result

    def test_omits_empty_keyword_fields(self):
        movie = _make_movie(plot_keywords=[], overall_keywords=[])
        result = build_production_techniques_user_prompt(movie)
        assert "plot_keywords:" not in result
        assert "overall_keywords:" not in result

    def test_no_unrelated_fields(self):
        movie = _make_movie(genres=["Drama"], overview="Ignored.")
        result = build_production_techniques_user_prompt(movie)
        assert "genres:" not in result
        assert "overview:" not in result
        assert "merged_keywords:" not in result


class TestGenerateProductionTechniques:
    async def test_returns_output_and_token_usage(self):
        expected = _make_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_production_techniques(_make_movie())

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)

    async def test_hardcoded_llm_params(self):
        from movie_ingestion.metadata_generation.prompts.production_techniques import (
            SYSTEM_PROMPT,
        )

        mock_fn = AsyncMock(return_value=(_make_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_production_techniques(movie)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["provider"] == LLMProvider.OPENAI
        assert call_kwargs["model"] == "gpt-5-mini"
        assert call_kwargs["system_prompt"] == SYSTEM_PROMPT
        assert call_kwargs["response_format"] is ProductionTechniquesOutput
        assert call_kwargs["reasoning_effort"] == "low"
        assert call_kwargs["verbosity"] == "low"

    async def test_wraps_llm_exception(self):
        movie = _make_movie()
        mock_fn = AsyncMock(side_effect=ValueError("API error"))

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_production_techniques(movie)

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_raises_empty_response_error(self):
        movie = _make_movie()
        mock_fn = AsyncMock(return_value=(None, 100, 50))

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError):
                await generate_production_techniques(movie)


class TestProductionTechniquesSignatureLockdown:
    def test_generate_does_not_accept_provider_kwarg(self):
        sig = inspect.signature(generate_production_techniques)
        assert "provider" not in sig.parameters

    def test_generate_does_not_accept_model_kwarg(self):
        sig = inspect.signature(generate_production_techniques)
        assert "model" not in sig.parameters

    def test_generate_does_not_accept_kwargs(self):
        sig = inspect.signature(generate_production_techniques)
        for param in sig.parameters.values():
            assert param.kind != inspect.Parameter.VAR_KEYWORD
