"""
Unit tests for movie_ingestion.metadata_generation.generators.viewer_experience.

Tests prompt building, LLM call delegation, return value shape, and
error handling for the generate_viewer_experience function.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.generic_methods import LLMProvider
from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import (
    ViewerExperienceOutput,
    TermsWithNegationsSection,
    OptionalTermsWithNegationsSection,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from movie_ingestion.metadata_generation.generators.viewer_experience import (
    build_viewer_experience_user_prompt,
    generate_viewer_experience,
    GENERATION_TYPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLM_PATCH = "movie_ingestion.metadata_generation.generators.viewer_experience.generate_llm_response_async"


def _make_movie(**overrides) -> MovieInputData:
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        genres=["Drama", "Thriller"],
        plot_keywords=["suspense", "mystery"],
        overall_keywords=["dark", "atmospheric"],
        maturity_rating="R",
        maturity_reasoning=["Rated R for violence"],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_ve_output() -> ViewerExperienceOutput:
    section = TermsWithNegationsSection(terms=["tense"], negations=[])
    optional = OptionalTermsWithNegationsSection(
        should_skip=True, section_data=section,
    )
    return ViewerExperienceOutput(
        emotional_palette=section,
        tension_adrenaline=section,
        tone_self_seriousness=section,
        cognitive_complexity=section,
        disturbance_profile=optional,
        sensory_load=optional,
        emotional_volatility=optional,
        ending_aftertaste=section,
    )


# ---------------------------------------------------------------------------
# Tests: build_viewer_experience_user_prompt
# ---------------------------------------------------------------------------

class TestBuildViewerExperienceUserPrompt:
    def test_includes_title_with_year(self):
        movie = _make_movie(title="Inception", release_year=2010)
        result = build_viewer_experience_user_prompt(movie, None, None)
        assert "Inception (2010)" in result

    def test_includes_genres(self):
        movie = _make_movie(genres=["Action", "Sci-Fi"])
        result = build_viewer_experience_user_prompt(movie, None, None)
        assert "genres: Action, Sci-Fi" in result

    def test_includes_plot_synopsis(self):
        movie = _make_movie()
        result = build_viewer_experience_user_prompt(movie, "A synopsis.", None)
        assert "plot_synopsis: A synopsis." in result

    def test_includes_merged_keywords(self):
        """Prompt contains merged keywords from movie.merged_keywords()."""
        movie = _make_movie(
            plot_keywords=["suspense"],
            overall_keywords=["dark"],
        )
        result = build_viewer_experience_user_prompt(movie, None, None)
        assert "merged_keywords:" in result
        assert "suspense" in result
        assert "dark" in result

    def test_includes_maturity_summary(self):
        movie = _make_movie(
            maturity_rating="R",
            maturity_reasoning=["Rated R for violence"],
        )
        result = build_viewer_experience_user_prompt(movie, None, None)
        assert "maturity_summary: Rated R for violence" in result

    def test_includes_review_insights_brief(self):
        movie = _make_movie()
        result = build_viewer_experience_user_prompt(movie, None, "Critics loved it.")
        assert "review_insights_brief: Critics loved it." in result

    def test_omits_none_fields(self):
        movie = _make_movie(
            genres=[],
            plot_keywords=[],
            overall_keywords=[],
            maturity_rating="",
            maturity_reasoning=[],
        )
        result = build_viewer_experience_user_prompt(movie, None, None)
        assert "genres" not in result
        assert "plot_synopsis" not in result
        assert "merged_keywords" not in result
        assert "review_insights_brief" not in result

    def test_omits_empty_genres(self):
        movie = _make_movie(genres=[])
        result = build_viewer_experience_user_prompt(movie, None, None)
        assert "genres" not in result

    def test_omits_empty_merged_keywords(self):
        movie = _make_movie(plot_keywords=[], overall_keywords=[])
        result = build_viewer_experience_user_prompt(movie, None, None)
        assert "merged_keywords" not in result


# ---------------------------------------------------------------------------
# Tests: generate_viewer_experience — LLM delegation
# ---------------------------------------------------------------------------

class TestGenerateViewerExperience:
    async def test_returns_output_and_token_usage(self):
        expected = _make_ve_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_viewer_experience(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)

    async def test_passes_provider_and_model(self):
        mock_fn = AsyncMock(return_value=(_make_ve_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_viewer_experience(
                movie, provider=LLMProvider.GEMINI, model="gemini-2.5-flash",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["provider"] == LLMProvider.GEMINI
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["response_format"] is ViewerExperienceOutput

    async def test_no_default_reasoning_effort_injected(self):
        """No default reasoning_effort is injected when caller doesn't provide one."""
        mock_fn = AsyncMock(return_value=(_make_ve_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_viewer_experience(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert "reasoning_effort" not in call_kwargs


# ---------------------------------------------------------------------------
# Tests: generate_viewer_experience — error paths
# ---------------------------------------------------------------------------

class TestGenerateViewerExperienceErrors:
    async def test_wraps_llm_exception(self):
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_viewer_experience(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_raises_empty_response_error(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError):
                await generate_viewer_experience(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

    async def test_error_chains_original_cause(self):
        original = ValueError("original")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_viewer_experience(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: maturity_summary in prompt
# ---------------------------------------------------------------------------

class TestViewerExperienceMaturityInPrompt:
    def test_prompt_includes_maturity_summary_when_available(self):
        """maturity_summary appears in prompt when maturity data exists."""
        movie = _make_movie(
            maturity_rating="R",
            maturity_reasoning=["Rated R for violence"],
        )
        result = build_viewer_experience_user_prompt(movie, None, None)
        assert "maturity_summary: Rated R for violence" in result

    def test_prompt_omits_maturity_summary_when_none(self):
        """maturity_summary is excluded when there's no maturity data."""
        movie = _make_movie(
            maturity_rating="",
            maturity_reasoning=[],
        )
        result = build_viewer_experience_user_prompt(movie, None, None)
        assert "maturity_summary" not in result


# ---------------------------------------------------------------------------
# Tests: system_prompt and response_format override
# ---------------------------------------------------------------------------

class TestViewerExperienceOverrides:
    async def test_custom_system_prompt_forwarded(self):
        """A custom system_prompt is forwarded to the LLM call."""
        mock_fn = AsyncMock(return_value=(_make_ve_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_viewer_experience(
                movie, system_prompt="CUSTOM_PROMPT",
                provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["system_prompt"] == "CUSTOM_PROMPT"

    async def test_custom_response_format_forwarded(self):
        """A custom response_format is forwarded to the LLM call."""
        from movie_ingestion.metadata_generation.schemas import ViewerExperienceWithJustificationsOutput

        mock_fn = AsyncMock(return_value=(_make_ve_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_viewer_experience(
                movie,
                response_format=ViewerExperienceWithJustificationsOutput,
                provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["response_format"] is ViewerExperienceWithJustificationsOutput
