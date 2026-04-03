"""
Unit tests for movie_ingestion.metadata_generation.generators.viewer_experience.

Tests prompt building, LLM call delegation, return value shape, and
error handling for the generate_viewer_experience function.

The generator now uses GPO-only narrative (no plot_summary fallback chain),
has removed merged_keywords and character_arcs inputs, and uses hardcoded
production config (gpt-5-mini, minimal reasoning, justification schema).

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import (
    ViewerExperienceOutput,
    TermsWithNegationsAndJustificationSection,
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
    section = TermsWithNegationsAndJustificationSection(
        justification="Based on evidence.",
        terms=["tense"],
        negations=[],
    )
    return ViewerExperienceOutput(
        emotional_palette=section,
        tension_adrenaline=section,
        tone_self_seriousness=section,
        cognitive_complexity=section,
        disturbance_profile=section,
        sensory_load=section,
        emotional_volatility=section,
        ending_aftertaste=section,
    )


# ---------------------------------------------------------------------------
# Tests: build_viewer_experience_user_prompt
# ---------------------------------------------------------------------------

class TestBuildViewerExperienceUserPrompt:
    def test_includes_title_with_year(self):
        movie = _make_movie(title="Inception", release_year=2010)
        result = build_viewer_experience_user_prompt(movie)
        assert "Inception (2010)" in result

    def test_includes_genre_context(self):
        """Raw genres appear as genre_context when genre_signatures not provided."""
        movie = _make_movie(genres=["Action", "Sci-Fi"])
        result = build_viewer_experience_user_prompt(movie)
        assert "genre_context:" in result
        assert "Action" in result

    def test_genre_signatures_override_raw_genres(self):
        """When genre_signatures has >= 2 entries, it overrides raw genres."""
        movie = _make_movie(genres=["Drama"])
        result = build_viewer_experience_user_prompt(
            movie, genre_signatures=["cyberpunk thriller", "philosophical sci-fi"],
        )
        assert "cyberpunk thriller" in result
        # Raw genre "Drama" should not appear as genre_context
        # (genre_signatures take priority)

    def test_includes_generalized_plot_overview(self):
        """GPO is the sole narrative source — labeled as narrative_input."""
        movie = _make_movie()
        # GPO must be >= 200 chars to pass inclusion threshold
        gpo = "A" * 250
        result = build_viewer_experience_user_prompt(movie, generalized_plot_overview=gpo)
        assert gpo in result

    def test_includes_emotional_observations(self):
        movie = _make_movie()
        # Must be >= 120 chars to pass inclusion threshold
        obs = "A" * 150
        result = build_viewer_experience_user_prompt(movie, emotional_observations=obs)
        assert obs in result

    def test_includes_craft_observations(self):
        movie = _make_movie()
        obs = "B" * 150
        result = build_viewer_experience_user_prompt(movie, craft_observations=obs)
        assert obs in result

    def test_includes_thematic_observations(self):
        movie = _make_movie()
        obs = "C" * 150
        result = build_viewer_experience_user_prompt(movie, thematic_observations=obs)
        assert obs in result

    def test_includes_maturity_summary(self):
        movie = _make_movie(
            maturity_rating="R",
            maturity_reasoning=["Rated R for violence"],
        )
        result = build_viewer_experience_user_prompt(movie)
        assert "maturity_summary: Rated R for violence" in result

    def test_absent_inputs_signal_not_available(self):
        """Absent primary inputs are explicitly signaled as 'not available'."""
        movie = _make_movie(genres=[], maturity_rating="", maturity_reasoning=[])
        result = build_viewer_experience_user_prompt(movie)
        assert "not available" in result

    def test_does_not_include_merged_keywords(self):
        """merged_keywords was removed from viewer_experience inputs."""
        movie = _make_movie(plot_keywords=["suspense"], overall_keywords=["dark"])
        result = build_viewer_experience_user_prompt(movie)
        assert "merged_keywords" not in result

    def test_does_not_accept_plot_synopsis_or_character_arcs(self):
        """plot_synopsis and character_arcs parameters were removed."""
        import inspect
        sig = inspect.signature(build_viewer_experience_user_prompt)
        assert "plot_synopsis" not in sig.parameters
        assert "character_arcs" not in sig.parameters
        assert "plot_summary" not in sig.parameters


# ---------------------------------------------------------------------------
# Tests: generate_viewer_experience — LLM delegation
# ---------------------------------------------------------------------------

class TestGenerateViewerExperience:
    async def test_returns_output_and_token_usage(self):
        expected = _make_ve_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_viewer_experience(movie)

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)

    async def test_uses_production_config(self):
        """Generator uses gpt-5-mini with justification schema."""
        mock_fn = AsyncMock(return_value=(_make_ve_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_viewer_experience(movie)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["model"] == "gpt-5-mini"
        assert call_kwargs["response_format"] is ViewerExperienceOutput
        assert call_kwargs["reasoning_effort"] == "minimal"


# ---------------------------------------------------------------------------
# Tests: generate_viewer_experience — error paths
# ---------------------------------------------------------------------------

class TestGenerateViewerExperienceErrors:
    async def test_wraps_llm_exception(self):
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_viewer_experience(movie)

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_raises_empty_response_error(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError):
                await generate_viewer_experience(movie)

    async def test_error_chains_original_cause(self):
        original = ValueError("original")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_viewer_experience(movie)

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
        result = build_viewer_experience_user_prompt(movie)
        assert "maturity_summary: Rated R for violence" in result

    def test_prompt_omits_maturity_summary_when_none(self):
        """maturity_summary is excluded when there's no maturity data."""
        movie = _make_movie(
            maturity_rating="",
            maturity_reasoning=[],
        )
        result = build_viewer_experience_user_prompt(movie)
        assert "maturity_summary" not in result or "not available" not in result.split("maturity_summary")[0]


# ---------------------------------------------------------------------------
# Tests: no provider/model params
# ---------------------------------------------------------------------------

class TestViewerExperienceSignature:
    def test_no_provider_model_params(self):
        """generate_viewer_experience does not accept provider or model parameters."""
        import inspect
        sig = inspect.signature(generate_viewer_experience)
        assert "provider" not in sig.parameters
        assert "model" not in sig.parameters
        assert "system_prompt" not in sig.parameters
        assert "response_format" not in sig.parameters
