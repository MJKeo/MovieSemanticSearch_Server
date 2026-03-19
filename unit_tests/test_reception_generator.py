"""
Unit tests for movie_ingestion.metadata_generation.generators.reception.

Tests prompt building helpers, prompt construction, LLM call delegation,
return value shape, and error handling for the generate_reception function.

All LLM calls are mocked — no real API traffic.
"""

from unittest.mock import AsyncMock, patch

import pytest

from implementation.llms.generic_methods import LLMProvider
from implementation.llms.vector_metadata_generation_methods import TokenUsage
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import ReceptionOutput
from movie_ingestion.metadata_generation.errors import (
    MetadataGenerationError,
    MetadataGenerationEmptyResponseError,
)
from movie_ingestion.metadata_generation.generators.reception import (
    _truncate_reviews,
    _format_attributes,
    build_reception_user_prompt,
    generate_reception,
    GENERATION_TYPE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_LLM_PATCH = "movie_ingestion.metadata_generation.generators.reception.generate_llm_response_async"


def _make_movie(**overrides) -> MovieInputData:
    """Build a MovieInputData with sensible defaults and optional overrides."""
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        reception_summary="Widely acclaimed.",
        audience_reception_attributes=[
            {"name": "visionary", "sentiment": "positive"},
        ],
        featured_reviews=[
            {"summary": "Brilliant", "text": "A brilliant film that redefines cinema."},
        ],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_reception_output() -> ReceptionOutput:
    """Build a minimal valid ReceptionOutput for mocking."""
    return ReceptionOutput(
        new_reception_summary="Widely acclaimed.",
        praise_attributes=["visionary"],
        complaint_attributes=[],
        review_insights_brief="Critics praised the visual effects.",
    )


# ---------------------------------------------------------------------------
# Tests: _truncate_reviews
# ---------------------------------------------------------------------------

class TestTruncateReviews:
    def test_under_count_limit(self):
        reviews = [{"text": f"Review {i}"} for i in range(3)]
        assert len(_truncate_reviews(reviews, max_count=5)) == 3

    def test_at_count_limit(self):
        reviews = [{"text": f"Review {i}"} for i in range(5)]
        assert len(_truncate_reviews(reviews, max_count=5)) == 5

    def test_over_count_limit(self):
        reviews = [{"text": f"Review {i}"} for i in range(7)]
        assert len(_truncate_reviews(reviews, max_count=5)) == 5

    def test_char_limit_crossing_review_included(self):
        """The review that crosses the char threshold IS included."""
        reviews = [
            {"text": "a" * 3000},
            {"text": "b" * 3000},  # This crosses 5000 — should be included
            {"text": "c" * 1000},  # Should NOT be included
        ]
        result = _truncate_reviews(reviews, max_count=10, max_chars=5000)
        assert len(result) == 2

    def test_empty_input(self):
        assert _truncate_reviews([]) == []

    def test_missing_text_key(self):
        """Reviews without 'text' key count as 0 chars."""
        reviews = [{"summary": "No text"} for _ in range(3)]
        result = _truncate_reviews(reviews, max_count=5, max_chars=5000)
        assert len(result) == 3

    def test_both_limits_count_first(self):
        """Count limit hit before char limit."""
        reviews = [{"text": "short"} for _ in range(7)]
        result = _truncate_reviews(reviews, max_count=3, max_chars=50000)
        assert len(result) == 3

    def test_both_limits_char_first(self):
        """Char limit hit before count limit (crossing review included)."""
        reviews = [{"text": "a" * 2000} for _ in range(7)]
        # 3 reviews = 6000 chars, crossing 5000 at review 3
        result = _truncate_reviews(reviews, max_count=10, max_chars=5000)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: _format_attributes
# ---------------------------------------------------------------------------

class TestFormatAttributes:
    def test_basic(self):
        attrs = [{"name": "visionary", "sentiment": "positive"}]
        assert _format_attributes(attrs) == "visionary (positive)"

    def test_multiple(self):
        attrs = [
            {"name": "a", "sentiment": "positive"},
            {"name": "b", "sentiment": "negative"},
            {"name": "c", "sentiment": "mixed"},
        ]
        assert _format_attributes(attrs) == "a (positive), b (negative), c (mixed)"

    def test_missing_keys(self):
        attrs = [{}]
        assert _format_attributes(attrs) == " ()"


# ---------------------------------------------------------------------------
# Tests: build_reception_user_prompt
# ---------------------------------------------------------------------------

class TestBuildReceptionUserPrompt:
    def test_includes_title_with_year(self):
        movie = _make_movie(title="Inception", release_year=2010)
        result = build_reception_user_prompt(movie)
        assert "Inception (2010)" in result

    def test_includes_reception_summary(self):
        movie = _make_movie(reception_summary="Acclaimed.")
        result = build_reception_user_prompt(movie)
        assert "reception_summary: Acclaimed." in result

    def test_includes_formatted_attributes(self):
        movie = _make_movie(
            audience_reception_attributes=[
                {"name": "visionary", "sentiment": "positive"},
            ]
        )
        result = build_reception_user_prompt(movie)
        assert "audience_reception_attributes: visionary (positive)" in result

    def test_includes_featured_reviews_as_multiline(self):
        movie = _make_movie(
            featured_reviews=[
                {"summary": "Great", "text": "A great film."},
            ]
        )
        result = build_reception_user_prompt(movie)
        assert "featured_reviews:" in result
        assert "- Great: A great film." in result

    def test_omits_none_reception_summary(self):
        movie = _make_movie(reception_summary=None)
        result = build_reception_user_prompt(movie)
        assert "reception_summary" not in result

    def test_omits_empty_attributes(self):
        movie = _make_movie(audience_reception_attributes=[])
        result = build_reception_user_prompt(movie)
        assert "audience_reception_attributes" not in result

    def test_omits_empty_reviews(self):
        movie = _make_movie(featured_reviews=[])
        result = build_reception_user_prompt(movie)
        assert "featured_reviews" not in result

    def test_collapses_review_newlines(self):
        movie = _make_movie(
            featured_reviews=[
                {"summary": "Good", "text": "Line one.\n\nLine two."},
            ]
        )
        result = build_reception_user_prompt(movie)
        assert "Line one. Line two." in result
        assert "\n\n" not in result.split("featured_reviews:")[1]

    def test_reviews_truncated_to_five(self):
        movie = _make_movie(
            featured_reviews=[
                {"summary": f"R{i}", "text": f"Review text {i}."}
                for i in range(7)
            ]
        )
        result = build_reception_user_prompt(movie)
        assert "R4:" in result   # 5th review (0-indexed)
        assert "R5:" not in result  # 6th review should be truncated


# ---------------------------------------------------------------------------
# Tests: generate_reception — LLM delegation and return value
# ---------------------------------------------------------------------------

class TestGenerateReception:
    async def test_returns_reception_output_and_token_usage(self):
        expected = _make_reception_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_reception(
                movie, LLMProvider.OPENAI, "gpt-5-mini",
            )

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)
        assert token_usage.input_tokens == 100
        assert token_usage.output_tokens == 50

    async def test_passes_provider_and_model_to_router(self):
        mock_fn = AsyncMock(return_value=(_make_reception_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_reception(movie, LLMProvider.GEMINI, "gemini-2.5-flash")

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["provider"] == LLMProvider.GEMINI
        assert call_kwargs["model"] == "gemini-2.5-flash"
        assert call_kwargs["response_format"] is ReceptionOutput

    async def test_forwards_kwargs_directly(self):
        """Reception does NOT merge _DEFAULT_KWARGS — kwargs pass through as-is."""
        mock_fn = AsyncMock(return_value=(_make_reception_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_reception(
                movie, LLMProvider.OPENAI, "gpt-5-mini",
                reasoning_effort="high",
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["reasoning_effort"] == "high"

    async def test_token_usage_records_caller_model_string(self):
        mock_fn = AsyncMock(return_value=(_make_reception_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            _, token_usage = await generate_reception(
                movie, LLMProvider.GEMINI, "gemini-2.5-flash",
            )

        assert token_usage.model == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# Tests: generate_reception — error paths
# ---------------------------------------------------------------------------

class TestGenerateReceptionErrors:
    async def test_wraps_llm_exception_in_metadata_generation_error(self):
        mock_fn = AsyncMock(side_effect=ValueError("LLM API timeout"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_reception(movie, LLMProvider.OPENAI, "gpt-5-mini")

        err = exc_info.value
        assert err.generation_type == GENERATION_TYPE
        assert err.title == "Test Movie (2020)"

    async def test_raises_empty_response_error_when_parsed_is_none(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError) as exc_info:
                await generate_reception(movie, LLMProvider.OPENAI, "gpt-5-mini")

        err = exc_info.value
        assert err.generation_type == GENERATION_TYPE
        assert err.title == "Test Movie (2020)"

    async def test_metadata_generation_error_chains_original_cause(self):
        original = ValueError("original cause")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_reception(movie, LLMProvider.OPENAI, "gpt-5-mini")

        assert exc_info.value.__cause__ is original
