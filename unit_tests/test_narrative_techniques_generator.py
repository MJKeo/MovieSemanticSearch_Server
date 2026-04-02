"""
Unit tests for movie_ingestion.metadata_generation.generators.narrative_techniques.

Tests prompt building, LLM call delegation, return value shape, and
error handling for the generate_narrative_techniques function.

The generator now uses merged_keywords (not overall_keywords), takes
craft_observations (not review_insights_brief), and resolves narrative
input via a shared fallback ladder in pre_consolidation.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import (
    NarrativeTechniquesWithJustificationsOutput,
    TermsWithJustificationSection,
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


def _make_nt_output() -> NarrativeTechniquesWithJustificationsOutput:
    section = TermsWithJustificationSection(
        evidence_basis="Based on evidence.",
        terms=["unreliable narrator"],
    )
    return NarrativeTechniquesWithJustificationsOutput(
        narrative_archetype=section,
        narrative_delivery=section,
        pov_perspective=section,
        characterization_methods=section,
        character_arcs=section,
        audience_character_perception=section,
        information_control=section,
        conflict_stakes_design=section,
        additional_narrative_devices=section,
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

    def test_includes_plot_summary_as_plot_synopsis(self):
        """When plot_summary is provided, it's labeled as 'plot_synopsis'."""
        movie = _make_movie()
        result = build_narrative_techniques_user_prompt(movie, "A synopsis.", None)
        assert "plot_synopsis: A synopsis." in result

    def test_includes_merged_keywords(self):
        """Now uses merged_keywords (plot + overall), not just overall_keywords."""
        movie = _make_movie(
            plot_keywords=["suspense"],
            overall_keywords=["nonlinear timeline"],
        )
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "keywords:" in result
        assert "suspense" in result
        assert "nonlinear timeline" in result

    def test_includes_craft_observations(self):
        """Second parameter is craft_observations (renamed from review_insights_brief).
        Must meet minimum length threshold."""
        movie = _make_movie()
        # Craft observations must be >= 300 chars (combined threshold)
        craft = "C" * 350
        result = build_narrative_techniques_user_prompt(movie, None, craft)
        assert craft in result

    def test_short_craft_observations_filtered_out(self):
        """Craft observations below threshold are filtered and shown as 'not available'."""
        movie = _make_movie()
        result = build_narrative_techniques_user_prompt(movie, None, "Short craft.")
        assert "craft_observations: not available" in result

    def test_absent_plot_signals_not_available(self):
        """When no plot data, prompt includes 'plot_synopsis: not available'."""
        movie = _make_movie(plot_synopses=[], plot_summaries=[], overview="")
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "plot_synopsis: not available" in result

    def test_omits_none_fields(self):
        movie = _make_movie(genres=[], plot_keywords=[], overall_keywords=[])
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "genres" not in result

    def test_does_not_include_review_insights_brief(self):
        """review_insights_brief label should not appear in the prompt."""
        movie = _make_movie()
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "review_insights_brief" not in result

    def test_does_not_include_overall_keywords_label(self):
        """overall_keywords label should not appear — it's now 'keywords'."""
        movie = _make_movie()
        result = build_narrative_techniques_user_prompt(movie, None, None)
        assert "overall_keywords:" not in result


# ---------------------------------------------------------------------------
# Tests: generate_narrative_techniques — LLM delegation
# ---------------------------------------------------------------------------

class TestGenerateNarrativeTechniques:
    async def test_returns_output_and_token_usage(self):
        expected = _make_nt_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_narrative_techniques(movie)

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)

    async def test_uses_hardcoded_production_config(self):
        """Generator uses hardcoded gpt-5-mini with NarrativeTechniquesWithJustificationsOutput."""
        mock_fn = AsyncMock(return_value=(_make_nt_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_narrative_techniques(movie)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["model"] == "gpt-5-mini"
        assert call_kwargs["response_format"] is NarrativeTechniquesWithJustificationsOutput
        assert call_kwargs["reasoning_effort"] == "minimal"


# ---------------------------------------------------------------------------
# Tests: generate_narrative_techniques — error paths
# ---------------------------------------------------------------------------

class TestGenerateNarrativeTechniquesErrors:
    async def test_wraps_llm_exception(self):
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_narrative_techniques(movie)

        assert exc_info.value.generation_type == GENERATION_TYPE

    async def test_raises_empty_response_error(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError):
                await generate_narrative_techniques(movie)

    async def test_error_chains_original_cause(self):
        original = ValueError("original")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_narrative_techniques(movie)

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: no provider/model params (hardcoded production config)
# ---------------------------------------------------------------------------

class TestNarrativeTechniquesSignature:
    def test_no_provider_model_params(self):
        """generate_narrative_techniques does not accept provider or model parameters."""
        import inspect
        sig = inspect.signature(generate_narrative_techniques)
        assert "provider" not in sig.parameters
        assert "model" not in sig.parameters
        assert "system_prompt" not in sig.parameters
        assert "response_format" not in sig.parameters
