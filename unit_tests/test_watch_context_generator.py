"""
Unit tests for movie_ingestion.metadata_generation.generators.watch_context.

Tests prompt building, LLM call delegation, return value shape, and
error handling for the generate_watch_context function.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.generic_methods import LLMProvider
from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import (
    WatchContextOutput,
    TermsSection,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from movie_ingestion.metadata_generation.generators.watch_context import (
    build_watch_context_user_prompt,
    generate_watch_context,
    GENERATION_TYPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLM_PATCH = "movie_ingestion.metadata_generation.generators.watch_context.generate_llm_response_async"


def _make_movie(**overrides) -> MovieInputData:
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        genres=["Drama"],
        plot_keywords=["suspense"],
        overall_keywords=["dark"],
        maturity_rating="R",
        maturity_reasoning=["Rated R for violence"],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_wc_output() -> WatchContextOutput:
    section = TermsSection(terms=["date night"])
    return WatchContextOutput(
        self_experience_motivations=section,
        external_motivations=section,
        key_movie_feature_draws=section,
        watch_scenarios=section,
    )


# ---------------------------------------------------------------------------
# Tests: build_watch_context_user_prompt
# ---------------------------------------------------------------------------

class TestBuildWatchContextUserPrompt:
    def test_includes_title_with_year(self):
        movie = _make_movie(title="Inception", release_year=2010)
        result = build_watch_context_user_prompt(movie, None)
        assert "Inception (2010)" in result

    def test_includes_genres(self):
        movie = _make_movie(genres=["Action", "Sci-Fi"])
        result = build_watch_context_user_prompt(movie, None)
        assert "genres: Action, Sci-Fi" in result

    def test_includes_merged_keywords(self):
        movie = _make_movie(plot_keywords=["suspense"], overall_keywords=["dark"])
        result = build_watch_context_user_prompt(movie, None)
        assert "merged_keywords:" in result
        assert "suspense" in result
        assert "dark" in result

    def test_includes_maturity_summary(self):
        movie = _make_movie(
            maturity_rating="R",
            maturity_reasoning=["Rated R for violence"],
        )
        result = build_watch_context_user_prompt(movie, None)
        assert "maturity_summary: Rated R for violence" in result

    def test_includes_review_insights_brief(self):
        movie = _make_movie()
        result = build_watch_context_user_prompt(movie, "Critics loved it.")
        assert "review_insights_brief: Critics loved it." in result

    def test_no_plot_synopsis_field(self):
        """CRITICAL: No plot information in watch context prompts."""
        movie = _make_movie()
        result = build_watch_context_user_prompt(movie, "review insights")
        assert "plot_synopsis" not in result
        assert "plot_summary" not in result

    def test_omits_none_fields(self):
        movie = _make_movie(
            genres=[],
            plot_keywords=[],
            overall_keywords=[],
            maturity_rating="",
            maturity_reasoning=[],
        )
        result = build_watch_context_user_prompt(movie, None)
        assert "genres" not in result
        assert "merged_keywords" not in result
        assert "review_insights_brief" not in result


# ---------------------------------------------------------------------------
# Tests: generate_watch_context — LLM delegation
# ---------------------------------------------------------------------------

class TestGenerateWatchContext:
    async def test_returns_output_and_token_usage(self):
        expected = _make_wc_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_watch_context(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)

    async def test_default_reasoning_effort_is_medium(self):
        """Watch context uses reasoning_effort='medium' by default."""
        mock_fn = AsyncMock(return_value=(_make_wc_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_watch_context(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["reasoning_effort"] == "medium"


# ---------------------------------------------------------------------------
# Tests: generate_watch_context — error paths
# ---------------------------------------------------------------------------

class TestGenerateWatchContextErrors:
    async def test_wraps_llm_exception(self):
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_watch_context(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_raises_empty_response_error(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError):
                await generate_watch_context(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

    async def test_error_chains_original_cause(self):
        original = ValueError("original")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_watch_context(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.__cause__ is original
