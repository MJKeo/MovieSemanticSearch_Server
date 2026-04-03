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
from schemas.metadata import ReceptionOutput
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
        source_material_hint=None,
        thematic_observations="Explores themes of identity.",
        emotional_observations="Haunting and intense.",
        craft_observations="Masterful cinematography.",
        reception_summary="Widely acclaimed.",
        praised_qualities=["visionary"],
        criticized_qualities=[],
    )


# ---------------------------------------------------------------------------
# Tests: _truncate_reviews
# ---------------------------------------------------------------------------

class TestTruncateReviews:
    def test_char_limit_crossing_review_included(self):
        """The review that crosses the char threshold IS included."""
        # After ascending sort: 1000, 2000, 3000. Accumulated: 1000, 3000, 6000.
        # 3rd review crosses 5000 — should be included, but no more after it.
        reviews = [
            {"text": "a" * 3000},
            {"text": "b" * 2000},
            {"text": "c" * 1000},
            {"text": "d" * 2000},  # Would push to 8000, should NOT be included
        ]
        result = _truncate_reviews(reviews, max_chars=5000)
        assert len(result) == 3

    def test_empty_input(self):
        """Empty list returns empty list."""
        assert _truncate_reviews([]) == []

    def test_missing_text_key(self):
        """Reviews without 'text' key count as 0 chars."""
        reviews = [{"summary": "No text"} for _ in range(3)]
        result = _truncate_reviews(reviews, max_chars=5000)
        assert len(result) == 3

    def test_char_limit_stops_accumulation(self):
        """Char limit stops after the crossing review (no count cap)."""
        reviews = [{"text": "a" * 2000} for _ in range(7)]
        # 3 reviews = 6000 chars, crossing 5000 at review 3
        result = _truncate_reviews(reviews, max_chars=5000)
        assert len(result) == 3

    def test_ascending_sort_by_length(self):
        """Reviews are sorted ascending by text length for diversity."""
        reviews = [
            {"text": "a" * 3000, "id": "long"},
            {"text": "b" * 500, "id": "medium"},
            {"text": "c" * 200, "id": "short"},
            {"text": "d" * 800, "id": "medium2"},
        ]
        result = _truncate_reviews(reviews, max_chars=6000)
        # Should be sorted: 200, 500, 800, 3000 = 4500 total, all included
        assert len(result) == 4
        assert result[0]["id"] == "short"
        assert result[1]["id"] == "medium"
        assert result[2]["id"] == "medium2"
        assert result[3]["id"] == "long"

    def test_no_count_cap_many_short_reviews(self):
        """20 short reviews all fit within the char budget."""
        reviews = [{"text": "x" * 100} for _ in range(20)]
        result = _truncate_reviews(reviews, max_chars=6000)
        # 20 × 100 = 2000 chars, well within 6000
        assert len(result) == 20

    def test_truncation_fallback_single_oversized_review(self):
        """A single review exceeding the budget is truncated to fit."""
        review = {"text": "x" * 15000}
        result = _truncate_reviews([review], max_chars=6000)
        assert len(result) == 1
        assert len(result[0]["text"]) == 6000

    def test_truncation_fallback_all_reviews_exceed_budget(self):
        """When all reviews exceed budget individually, shortest is truncated."""
        reviews = [
            {"text": "a" * 10000},
            {"text": "b" * 8000},  # shortest after sort
            {"text": "c" * 12000},
        ]
        result = _truncate_reviews(reviews, max_chars=6000)
        assert len(result) == 1
        assert len(result[0]["text"]) == 6000
        # Should be the shortest one (b) that got truncated
        assert result[0]["text"].startswith("b")


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

    def test_reviews_truncated_by_char_budget(self):
        """Reviews exceeding the char budget are dropped (no count cap)."""
        movie = _make_movie(
            featured_reviews=[
                {"summary": f"R{i}", "text": "x" * 2000}
                for i in range(7)
            ]
        )
        result = build_reception_user_prompt(movie)
        # 6000 char budget ÷ 2000 per review = 3 reviews fit (3rd crosses at 6000)
        assert "R0:" in result or "R1:" in result  # Some reviews present
        # At most 3-4 reviews should appear (sorted ascending, all same length)
        review_count = result.count("- R")
        assert review_count <= 4


# ---------------------------------------------------------------------------
# Tests: generate_reception — LLM delegation and return value
# ---------------------------------------------------------------------------

class TestGenerateReception:
    async def test_returns_reception_output_and_token_usage(self):
        """generate_reception returns (ReceptionOutput, TokenUsage) tuple."""
        expected = _make_reception_output()
        mock_fn = AsyncMock(return_value=(expected, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            parsed, token_usage = await generate_reception(movie)

        assert parsed is expected
        assert isinstance(token_usage, TokenUsage)
        assert token_usage.input_tokens == 100
        assert token_usage.output_tokens == 50

    async def test_uses_fixed_provider_and_model(self):
        """generate_reception uses fixed OpenAI gpt-5-mini with minimal reasoning."""
        mock_fn = AsyncMock(return_value=(_make_reception_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            await generate_reception(movie)

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["provider"] == LLMProvider.OPENAI
        assert call_kwargs["model"] == "gpt-5-mini"
        assert call_kwargs["response_format"] is ReceptionOutput
        assert call_kwargs["reasoning_effort"] == "minimal"
        assert call_kwargs["verbosity"] == "low"

    async def test_token_usage_records_fixed_model_string(self):
        """TokenUsage.model is the fixed gpt-5-mini constant."""
        mock_fn = AsyncMock(return_value=(_make_reception_output(), 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            _, token_usage = await generate_reception(movie)

        assert token_usage.model == "gpt-5-mini"

    def test_signature_has_only_movie_param(self):
        """generate_reception accepts only a single 'movie' positional parameter."""
        import inspect
        sig = inspect.signature(generate_reception)
        params = [
            name for name, p in sig.parameters.items()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        assert params == ["movie"]


# ---------------------------------------------------------------------------
# Tests: generate_reception — error paths
# ---------------------------------------------------------------------------

class TestGenerateReceptionErrors:
    async def test_wraps_llm_exception_in_metadata_generation_error(self):
        mock_fn = AsyncMock(side_effect=ValueError("LLM API timeout"))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_reception(movie)

        err = exc_info.value
        assert err.generation_type == GENERATION_TYPE
        assert err.title == "Test Movie (2020)"

    async def test_raises_empty_response_error_when_parsed_is_none(self):
        mock_fn = AsyncMock(return_value=(None, 100, 50))
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationEmptyResponseError) as exc_info:
                await generate_reception(movie)

        err = exc_info.value
        assert err.generation_type == GENERATION_TYPE
        assert err.title == "Test Movie (2020)"

    async def test_metadata_generation_error_chains_original_cause(self):
        original = ValueError("original cause")
        mock_fn = AsyncMock(side_effect=original)
        movie = _make_movie()

        with patch(_LLM_PATCH, mock_fn):
            with pytest.raises(MetadataGenerationError) as exc_info:
                await generate_reception(movie)

        assert exc_info.value.__cause__ is original


# ---------------------------------------------------------------------------
# Tests: _truncate_reviews edge case
# ---------------------------------------------------------------------------

class TestTruncateReviewsEdge:
    def test_single_review_under_char_limit(self):
        """A single review under _MAX_REVIEW_CHARS is included unchanged."""
        review = {"text": "x" * 5000}
        result = _truncate_reviews([review])
        assert len(result) == 1
        assert result[0] is review


# ---------------------------------------------------------------------------
# Tests: build_reception_user_prompt edge case
# ---------------------------------------------------------------------------

class TestBuildReceptionUserPromptEdge:
    def test_empty_reception_summary_treated_as_none(self):
        """An empty string '' for reception_summary is treated as None (omitted)."""
        movie = _make_movie(
            reception_summary="",
            featured_reviews=[{"summary": "Fine", "text": "A fine and decent movie overall."}],
        )
        prompt = build_reception_user_prompt(movie)
        assert "reception_summary" not in prompt

    def test_generate_reception_uses_hardcoded_system_prompt(self):
        """generate_reception does NOT accept system_prompt, response_format, or kwargs overrides."""
        import inspect
        sig = inspect.signature(generate_reception)
        assert "system_prompt" not in sig.parameters
        assert "response_format" not in sig.parameters
        assert "provider" not in sig.parameters
        assert "model" not in sig.parameters

    def test_prompt_does_not_include_overview(self):
        """build_reception_user_prompt does NOT include the movie overview."""
        movie = _make_movie(overview="This is the overview text.")
        result = build_reception_user_prompt(movie)
        assert "overview" not in result.lower()
        assert "This is the overview text." not in result

    def test_prompt_includes_genres(self):
        """build_reception_user_prompt includes genres when present."""
        movie = _make_movie(genres=["Drama", "Thriller"])
        result = build_reception_user_prompt(movie)
        assert "genres:" in result.lower() or "Drama" in result
