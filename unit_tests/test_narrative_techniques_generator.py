"""
Unit tests for movie_ingestion.metadata_generation.generators.narrative_techniques.

Tests prompt building, LLM call delegation, return value shape, and
error handling for the generate_narrative_techniques function.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.generic_methods import LLMProvider
from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import (
    NarrativeTechniquesOutput,
    TermsSection,
)
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from movie_ingestion.metadata_generation.generators.narrative_techniques import (
    build_narrative_techniques_user_prompt,
    generate_narrative_techniques,
    GENERATION_TYPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLM_PATCH = "movie_ingestion.metadata_generation.generators.narrative_techniques.generate_llm_response_async"


def _make_movie(**overrides) -> MovieInputData:
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        genres=["Drama"],
        plot_keywords=["suspense"],
        overall_keywords=["nonlinear timeline", "unreliable narrator"],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_nt_output() -> NarrativeTechniquesOutput:
    section = TermsSection(terms=["unreliable narrator"])
    return NarrativeTechniquesOutput(
        pov_perspective=section,
        narrative_delivery=section,
        narrative_archetype=section,
        information_control=section,
        characterization_methods=section,
        character_arcs=section,
        audience_character_perception=section,
        conflict_stakes_design=section,
        thematic_delivery=section,
        meta_techniques=section,
        additional_plot_devices=section,
    )


# ---------------------------------------------------------------------------
# Tests: build_narrative_techniques_user_prompt
# ---------------------------------------------------------------------------

class TestBuildNarrativeTechniquesUserPrompt:
    def test_includes_title_with_year(self):
        movie = _make_movie(title="Inception", release_year=2010)
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "Inception (2010)" in result

    def test_includes_genres(self):
        movie = _make_movie(genres=["Action", "Sci-Fi"])
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "genres: Action, Sci-Fi" in result

    def test_includes_plot_synopsis(self):
        movie = _make_movie()
        result = build_narrative_techniques_user_prompt(movie, "A synopsis.", None)
        assert "plot_synopsis: A synopsis." in result

    def test_includes_overall_keywords(self):
        """Uses movie.overall_keywords (NOT merged_keywords)."""
        movie = _make_movie(overall_keywords=["nonlinear timeline"])
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "overall_keywords: nonlinear timeline" in result

    def test_includes_review_insights_brief(self):
        movie = _make_movie()
        result = build_narrative_techniques_user_prompt(movie, None, "Critics noted structure.")
        assert "review_insights_brief: Critics noted structure." in result

    def test_does_not_include_plot_keywords(self):
        """plot_keywords should NOT appear in narrative techniques prompt."""
        movie = _make_movie(plot_keywords=["suspense", "mystery"])
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "plot_keywords" not in result

    def test_does_not_include_merged_keywords(self):
        """merged_keywords should NOT appear in narrative techniques prompt."""
        movie = _make_movie()
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "merged_keywords" not in result

    def test_omits_none_fields(self):
        movie = _make_movie(genres=[], overall_keywords=[])
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "genres" not in result
        assert "overall_keywords" not in result
        assert "plot_synopsis" not in result
        assert "review_insights_brief" not in result


# ---------------------------------------------------------------------------
# Tests: generate_narrative_techniques — LLM delegation
# ---------------------------------------------------------------------------

class TestGenerateNarrativeTechniques:
    async def test_returns_output_and_token_usage(self):
        expected = _make_nt_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_narrative_techniques(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)

    async def test_default_reasoning_effort_is_medium(self):
        mock_fn = AsyncMock(return_value=(_make_nt_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_narrative_techniques(
                movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["reasoning_effort"] == "medium"


# ---------------------------------------------------------------------------
# Tests: generate_narrative_techniques — error paths
# ---------------------------------------------------------------------------

class TestGenerateNarrativeTechniquesErrors:
    async def test_wraps_llm_exception(self):
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_narrative_techniques(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_raises_empty_response_error(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError):
                await generate_narrative_techniques(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

    async def test_error_chains_original_cause(self):
        original = ValueError("original")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_narrative_techniques(
                    movie, provider=LLMProvider.OPENAI, model="gpt-5-mini",
                )

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: prompt uses overall_keywords (not merged_keywords or plot_keywords)
# ---------------------------------------------------------------------------

class TestNarrativeTechniquesPromptKeywords:
    def test_prompt_uses_overall_keywords(self):
        """Prompt uses overall_keywords label, not merged_keywords or plot_keywords."""
        movie = _make_movie(overall_keywords=["nonlinear timeline", "unreliable narrator"])
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "overall_keywords:" in result
        assert "nonlinear timeline" in result
        assert "unreliable narrator" in result


# ---------------------------------------------------------------------------
# Tests: system_prompt and response_format override
# ---------------------------------------------------------------------------

class TestNarrativeTechniquesOverrides:
    async def test_custom_system_prompt_forwarded(self):
        """A custom system_prompt is forwarded to the LLM call."""
        mock_fn = AsyncMock(return_value=(_make_nt_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_narrative_techniques(
                movie, system_prompt="CUSTOM_PROMPT",
                provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["system_prompt"] == "CUSTOM_PROMPT"

    async def test_custom_response_format_forwarded(self):
        """A custom response_format is forwarded to the LLM call."""
        from movie_ingestion.metadata_generation.schemas import NarrativeTechniquesWithJustificationsOutput

        mock_fn = AsyncMock(return_value=(_make_nt_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_narrative_techniques(
                movie,
                response_format=NarrativeTechniquesWithJustificationsOutput,
                provider=LLMProvider.OPENAI, model="gpt-5-mini",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["response_format"] is NarrativeTechniquesWithJustificationsOutput
