"""
Unit tests for movie_ingestion.metadata_generation.generators.plot_analysis.

Tests prompt building, LLM call delegation, return value shape, and
error handling for the generate_plot_analysis function.

The generator now has hardcoded production config (gpt-5-mini, minimal
reasoning, justification schema) — provider/model/system_prompt/response_format
are no longer caller-configurable.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import (
    PlotAnalysisOutput,
    CharacterArcWithReasoning,
    ElevatorPitchWithJustification,
    ThematicConceptWithJustification,
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
        genre_signatures=["cyberpunk thriller", "philosophical sci-fi"],
        thematic_concepts=[
            ThematicConceptWithJustification(
                explanation_and_justification="Central to the narrative.",
                concept_label="identity",
            ),
        ],
        elevator_pitch_with_justification=ElevatorPitchWithJustification(
            explanation_and_justification="Heart of the movie.",
            elevator_pitch="forbidden knowledge",
        ),
        conflict_type=["man vs system"],
        character_arcs=[
            CharacterArcWithReasoning(
                reasoning="Transforms from lost programmer to messianic figure.",
                arc_transformation_label="hero's awakening",
            ),
        ],
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

    def test_includes_plot_summary_as_plot_synopsis(self):
        """When plot_summary is provided, it's labeled as 'plot_synopsis'."""
        movie = _make_movie()
        result = build_plot_analysis_user_prompt(movie, "A detailed synopsis.", None)
        assert "plot_synopsis: A detailed synopsis." in result

    def test_includes_merged_keywords(self):
        movie = _make_movie(plot_keywords=["hacker", "simulation"])
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "merged_keywords: hacker, simulation" in result

    def test_includes_thematic_observations(self):
        """Third parameter is thematic_observations (renamed from review_insights_brief)."""
        movie = _make_movie()
        result = build_plot_analysis_user_prompt(movie, None, "Critics praised depth.")
        assert "thematic_observations:" in result
        assert "Critics praised depth." in result

    def test_absent_plot_signals_not_available(self):
        """When no plot data at all, prompt includes 'plot_synopsis: not available'."""
        movie = _make_movie(plot_synopses=[], plot_summaries=[], overview="")
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "plot_synopsis: not available" in result

    def test_absent_thematic_observations_signals_not_available(self):
        """When thematic_observations is None, prompt includes 'not available'."""
        movie = _make_movie()
        result = build_plot_analysis_user_prompt(movie, "synopsis", None)
        assert "thematic_observations: not available" in result

    def test_uses_plot_fallback_when_no_plot_summary(self):
        """When plot_summary is None, uses best_plot_fallback with 'plot_text' label."""
        movie = _make_movie(
            plot_synopses=["A long synopsis that is definitely more than a few words for fallback."],
        )
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "plot_text:" in result
        assert "A long synopsis" in result

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
        movie = _make_movie(genres=[], plot_keywords=[], plot_synopses=[], plot_summaries=[], overview="")
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "title:" in result


# ---------------------------------------------------------------------------
# Tests: generate_plot_analysis — LLM delegation
# ---------------------------------------------------------------------------

class TestGeneratePlotAnalysis:
    async def test_returns_output_and_token_usage(self):
        expected = _make_plot_analysis_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_plot_analysis(movie)

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)
        assert token_usage.input_tokens == 100
        assert token_usage.output_tokens == 50

    async def test_uses_hardcoded_production_config(self):
        """Generator uses hardcoded gpt-5-mini with PlotAnalysisOutput."""
        mock_fn = AsyncMock(return_value=(_make_plot_analysis_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_analysis(movie)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["model"] == "gpt-5-mini"
        assert call_kwargs["response_format"] is PlotAnalysisOutput
        assert call_kwargs["reasoning_effort"] == "minimal"

    async def test_token_usage_records_model(self):
        mock_fn = AsyncMock(return_value=(_make_plot_analysis_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            _, token_usage = await generate_plot_analysis(movie)

        assert token_usage.model == "gpt-5-mini"

    async def test_passes_plot_summary_and_thematic_observations(self):
        """Optional params are forwarded to the prompt builder."""
        mock_fn = AsyncMock(return_value=(_make_plot_analysis_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_plot_analysis(
                movie,
                plot_summary="A plot summary.",
                thematic_observations="Themes of identity.",
            )

        # The prompt should contain these values
        call_kwargs = mock_fn.call_args[1]
        assert "A plot summary." in call_kwargs["user_prompt"]
        assert "Themes of identity." in call_kwargs["user_prompt"]


# ---------------------------------------------------------------------------
# Tests: generate_plot_analysis — error paths
# ---------------------------------------------------------------------------

class TestGeneratePlotAnalysisErrors:
    async def test_wraps_llm_exception(self):
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_plot_analysis(movie)

        assert exc_info.value.generation_type == GENERATION_TYPE
        assert exc_info.value.title == "Test Movie (2020)"

    async def test_raises_empty_response_error(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError) as exc_info:
                await generate_plot_analysis(movie)

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_error_chains_original_cause(self):
        original = ValueError("original")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_plot_analysis(movie)

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: prompt uses merged_keywords (not plot_keywords)
# ---------------------------------------------------------------------------

class TestPlotAnalysisPromptUsesCorrectKeywordLabel:
    def test_prompt_uses_merged_keywords_label(self):
        """The prompt label for keywords is 'merged_keywords', not 'plot_keywords'."""
        movie = _make_movie(plot_keywords=["hacker", "simulation"])
        result = build_plot_analysis_user_prompt(movie, None, None)
        assert "merged_keywords:" in result
        assert "hacker" in result
        assert "simulation" in result

    def test_no_provider_model_params(self):
        """generate_plot_analysis does not accept provider or model parameters."""
        import inspect
        sig = inspect.signature(generate_plot_analysis)
        assert "provider" not in sig.parameters
        assert "model" not in sig.parameters
        assert "system_prompt" not in sig.parameters
        assert "response_format" not in sig.parameters
