"""
Unit tests for movie_ingestion.metadata_generation.generators.plot_analysis.

Tests prompt building, LLM call delegation, return value shape, and
error handling for the generate_plot_analysis function.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.generic_methods import LLMProvider
from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import (
    PlotAnalysisOutput,
    CharacterArc,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from movie_ingestion.metadata_generation.generators.plot_analysis import (
    build_plot_analysis_user_prompt,
    generate_plot_analysis,
    GENERATION_TYPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLM_PATCH = "movie_ingestion.metadata_generation.generators.plot_analysis.generate_llm_response_async"


def _make_movie(**overrides) -> MovieInputData:
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        genres=["Drama"],
        plot_keywords=["keyword1", "keyword2"],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_plot_analysis_output() -> PlotAnalysisOutput:
    return PlotAnalysisOutput(
        core_concept_label="forbidden knowledge",
        genre_signatures=["cyberpunk thriller", "philosophical sci-fi"],
        conflict_scale="global",
        character_arcs=[
            CharacterArc(
                character_name="Neo",
                arc_transformation_description="Discovers his true power.",
                arc_transformation_label="hero's awakening",
            ),
        ],
        themes_primary=["identity"],
        lessons_learned=[],
        generalized_plot_overview="A hacker discovers the truth.",
    )


# ---------------------------------------------------------------------------
# Tests: build_plot_analysis_user_prompt
# ---------------------------------------------------------------------------

class TestBuildPlotAnalysisUserPrompt:
    def test_includes_title_with_year(self):
        movie = _make_movie(title="Inception", release_year=2010)
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "Inception (2010)" in result

    def test_includes_genres(self):
        movie = _make_movie(genres=["Action", "Sci-Fi"])
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "genres: Action, Sci-Fi" in result

    def test_includes_plot_synopsis(self):
        movie = _make_movie()
        result = build_plot_analysis_user_prompt(movie, "A detailed synopsis.", None)
        assert "plot_synopsis: A detailed synopsis." in result

    def test_includes_merged_keywords(self):
        movie = _make_movie(plot_keywords=["hacker", "simulation"])
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "merged_keywords: hacker, simulation" in result

    def test_includes_review_insights_brief(self):
        movie = _make_movie()
        result = build_plot_analysis_user_prompt(movie, None, "Critics praised depth.")
        assert "review_insights_brief: Critics praised depth." in result

    def test_omits_none_plot_synopsis(self):
        movie = _make_movie()
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "plot_synopsis" not in result

    def test_omits_none_review_insights(self):
        movie = _make_movie()
        result = build_plot_analysis_user_prompt(movie, "synopsis", None)
        assert "review_insights_brief" not in result

    def test_omits_empty_genres(self):
        movie = _make_movie(genres=[])
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "genres" not in result

    def test_omits_empty_merged_keywords(self):
        movie = _make_movie(plot_keywords=[], overall_keywords=[])
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "merged_keywords" not in result

    def test_minimal_inputs(self):
        """Only title present; all other fields None/empty."""
        movie = _make_movie(genres=[], plot_keywords=[])
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "title:" in result
        lines = [l for l in result.split("\n") if l.strip()]
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Tests: generate_plot_analysis — LLM delegation
# ---------------------------------------------------------------------------

class TestGeneratePlotAnalysis:
    async def test_returns_output_and_token_usage(self):
        expected = _make_plot_analysis_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_plot_analysis(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)
        assert token_usage.input_tokens == 100
        assert token_usage.output_tokens == 50

    async def test_passes_provider_and_model_to_router(self):
        mock_fn = AsyncMock(return_value=(_make_plot_analysis_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_analysis(
                movie, provider=LLMProvider.GEMINI, model="gemini-2.5-flash",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["provider"] == LLMProvider.GEMINI
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["response_format"] is PlotAnalysisOutput

    async def test_no_default_reasoning_effort_injected(self):
        """No default reasoning_effort is injected when caller doesn't provide one."""
        mock_fn = AsyncMock(return_value=(_make_plot_analysis_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_analysis(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert "reasoning_effort" not in call_kwargs

    async def test_override_replaces_default_kwarg(self):
        mock_fn = AsyncMock(return_value=(_make_plot_analysis_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_analysis(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                reasoning_effort="medium",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["reasoning_effort"] == "medium"

    async def test_token_usage_records_caller_model_string(self):
        mock_fn = AsyncMock(return_value=(_make_plot_analysis_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            _, token_usage = await generate_plot_analysis(
                movie, provider=LLMProvider.GEMINI, model="gemini-2.5-flash",
            )

        assert token_usage.model == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Tests: generate_plot_analysis — error paths
# ---------------------------------------------------------------------------

class TestGeneratePlotAnalysisErrors:
    async def test_wraps_llm_exception(self):
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_plot_analysis(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.generation_type == GENERATION_TYPE
        assert exc_info.value.title == "Test Movie (2020)"

    async def test_raises_empty_response_error(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError) as exc_info:
                await generate_plot_analysis(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_error_chains_original_cause(self):
        original = ValueError("original")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_plot_analysis(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: prompt uses merged_keywords (not plot_keywords)
# ---------------------------------------------------------------------------

class TestPlotAnalysisPromptUsesCorrectKeywordLabel:
    def test_prompt_uses_merged_keywords_label(self):
        """The prompt label for keywords is 'merged_keywords', not 'plot_keywords'.

        build_plot_analysis_user_prompt passes `merged_keywords=movie.merged_keywords()`
        to build_user_prompt, so the label in the output is 'merged_keywords'.
        """
        movie = _make_movie(plot_keywords=["hacker", "simulation"])
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "merged_keywords:" in result
        assert "hacker" in result
        assert "simulation" in result


# ---------------------------------------------------------------------------
# Tests: system_prompt and response_format override
# ---------------------------------------------------------------------------

class TestPlotAnalysisOverrides:
    async def test_custom_system_prompt_forwarded(self):
        """A custom system_prompt is forwarded to the LLM call."""
        mock_fn = AsyncMock(return_value=(_make_plot_analysis_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_analysis(
                movie, system_prompt="CUSTOM_PROMPT",
                provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["system_prompt"] == "CUSTOM_PROMPT"

    async def test_custom_response_format_forwarded(self):
        """A custom response_format is forwarded to the LLM call."""
        from movie_ingestion.metadata_generation.schemas import PlotAnalysisWithJustificationsOutput

        mock_fn = AsyncMock(return_value=(_make_plot_analysis_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_analysis(
                movie,
                response_format=PlotAnalysisWithJustificationsOutput,
                provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["response_format"] is PlotAnalysisWithJustificationsOutput
