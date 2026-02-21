"""
Unit tests for db.lexical_search methods.

This module provides exhaustive coverage of the lexical_search function and its
supporting helpers. Tests are organized into logical groups covering:
- Private helper functions (_dedupe_preserve_order, etc.)
- Term ID resolution functions
- Posting search functions (people, characters, studios, titles)
- The main lexical_search orchestration function

The lexical_search function is complex, combining multiple entity buckets
(people, characters, studios, titles, franchises) with INCLUDE/EXCLUDE
semantics, metadata filtering, and normalized scoring. These tests verify
every code path, edge case, and integration scenario.
"""

from unittest.mock import AsyncMock, MagicMock, call, patch
from typing import Any

import pytest

from db import lexical_search
from implementation.misc.sql_like import escape_like
from implementation.classes.enums import EntityCategory, Genre
from implementation.classes.schemas import (
    ExtractedEntitiesResponse,
    ExtractedEntityData,
    LexicalCandidate,
    MetadataFilters,
)


@pytest.fixture(autouse=True)
def _mock_excluded_movie_id_resolution(mocker):
    """Default cross-bucket exclusion resolution to empty sets.

    Every orchestration-level test for lexical_search needs this mock because
    fetch_movie_ids_by_term_ids is called inside the orchestrator whenever
    exclusion term IDs exist.  Individual tests that need to control the
    return value can re-patch it.
    """
    mocker.patch(
        "db.lexical_search.fetch_movie_ids_by_term_ids",
        new=AsyncMock(return_value=set()),
    )


# =============================================================================
#                           HELPER FUNCTION TESTS
# =============================================================================


class TestEscapeLike:
    """Tests for the shared SQL LIKE escape helper function."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # Basic strings without metacharacters
            ("plain", "plain"),
            ("hello world", "hello world"),
            ("", ""),
            ("a", "a"),
            # Percent sign escaping
            ("100%", r"100\%"),
            ("%start", r"\%start"),
            ("mid%dle", r"mid\%dle"),
            ("100% complete", r"100\% complete"),
            ("%%", r"\%\%"),
            # Underscore escaping
            ("under_score", r"under\_score"),
            ("_start", r"\_start"),
            ("end_", r"end\_"),
            ("__double__", r"\_\_double\_\_"),
            # Backslash escaping
            (r"path\name", r"path\\name"),
            (r"\start", r"\\start"),
            ("end\\", r"end\\"),
            (r"\\double", r"\\\\double"),
            # Combined metacharacters
            ("100%_done", r"100\%\_done"),
            (r"path\100%", r"path\\100\%"),
            (r"_\%_", r"\_\\\%\_"),
            (r"a%b_c\d", r"a\%b\_c\\d"),
            # Real-world examples
            ("50%_off_sale", r"50\%\_off\_sale"),
            (r"C:\Users\name", r"C:\\Users\\name"),
            ("file_name_v2%", r"file\_name\_v2\%"),
        ],
    )
    def test_escape_like_escapes_metacharacters(self, value: str, expected: str) -> None:
        """escape_like should escape percent, underscore, and backslash characters."""
        assert escape_like(value) == expected

    def test_escape_like_preserves_other_special_chars(self) -> None:
        """escape_like should not escape characters that are not LIKE metacharacters."""
        special_chars = "!@#$^&*()+=[]{}|;:',.<>?/~`"
        assert escape_like(special_chars) == special_chars

    def test_escape_like_handles_unicode(self) -> None:
        """escape_like should handle unicode strings correctly."""
        assert escape_like("café%") == r"café\%"
        assert escape_like("日本語_test") == r"日本語\_test"


class TestDedupePreserveOrder:
    """Tests for the _dedupe_preserve_order helper function."""

    @pytest.mark.parametrize(
        ("values", "expected"),
        [
            # Empty list
            ([], []),
            # Single element
            (["a"], ["a"]),
            # No duplicates
            (["a", "b", "c"], ["a", "b", "c"]),
            # Simple duplicates
            (["a", "b", "a", "c", "b"], ["a", "b", "c"]),
            # All duplicates
            (["x", "x", "x", "x"], ["x"]),
            # Duplicates at start
            (["a", "a", "b", "c"], ["a", "b", "c"]),
            # Duplicates at end
            (["a", "b", "c", "c"], ["a", "b", "c"]),
            # Alternating duplicates
            (["a", "b", "a", "b", "a", "b"], ["a", "b"]),
            # Case sensitivity preserved
            (["A", "a", "A"], ["A", "a"]),
            # Whitespace strings
            (["", " ", "", " "], ["", " "]),
            # Long list with scattered duplicates
            (["a", "b", "c", "d", "a", "e", "b", "f", "c"], ["a", "b", "c", "d", "e", "f"]),
        ],
    )
    def test_dedupe_preserve_order(self, values: list[str], expected: list[str]) -> None:
        """_dedupe_preserve_order should keep first occurrence ordering."""
        source_copy = list(values)
        assert lexical_search._dedupe_preserve_order(values) == expected
        # Verify original list is not mutated
        assert values == source_copy

    def test_dedupe_preserve_order_returns_new_list(self) -> None:
        """_dedupe_preserve_order should return a new list, not modify in place."""
        original = ["a", "b", "a"]
        result = lexical_search._dedupe_preserve_order(original)
        assert result is not original
        assert result == ["a", "b"]


# =============================================================================
#                       TERM ID RESOLUTION TESTS
# =============================================================================


class TestResolveAllTitleTokens:
    """Tests for the _resolve_all_title_tokens helper function."""

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits(self, mocker) -> None:
        """Should return empty list when no title searches exist."""
        fetch_ids = mocker.patch("db.lexical_search.fetch_title_token_ids", new=AsyncMock())
        result = await lexical_search._resolve_all_title_tokens([])
        assert result == []
        fetch_ids.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_title_single_token(self, mocker) -> None:
        """Should resolve a single token for a single title search."""
        fetch_ids = mocker.patch(
            "db.lexical_search.fetch_title_token_ids",
            new=AsyncMock(return_value={0: [101]}),
        )
        result = await lexical_search._resolve_all_title_tokens([["matrix"]])
        assert result == [{0: [101]}]
        fetch_ids.assert_awaited_once_with(tokens=["matrix"], max_df=lexical_search.MAX_DF)

    @pytest.mark.asyncio
    async def test_single_title_multiple_tokens(self, mocker) -> None:
        """Should resolve multiple tokens for a single title search."""
        fetch_ids = mocker.patch(
            "db.lexical_search.fetch_title_token_ids",
            new=AsyncMock(return_value={0: [101], 1: [102], 2: [103]}),
        )
        result = await lexical_search._resolve_all_title_tokens([["the", "dark", "knight"]])
        assert result == [{0: [101], 1: [102], 2: [103]}]
        fetch_ids.assert_awaited_once_with(
            tokens=["the", "dark", "knight"],
            max_df=lexical_search.MAX_DF,
        )

    @pytest.mark.asyncio
    async def test_multiple_titles_with_shared_tokens(self, mocker) -> None:
        """Should deduplicate tokens across title searches and distribute results."""
        fetch_ids = mocker.patch(
            "db.lexical_search.fetch_title_token_ids",
            new=AsyncMock(return_value={0: [11], 1: [22]}),
        )
        result = await lexical_search._resolve_all_title_tokens(
            [["star", "wars"], ["star"]],
        )
        # "star" appears twice but should only be resolved once
        fetch_ids.assert_awaited_once_with(tokens=["star", "wars"], max_df=lexical_search.MAX_DF)
        assert result == [{0: [11], 1: [22]}, {0: [11]}]

    @pytest.mark.asyncio
    async def test_omits_unresolved_tokens(self, mocker) -> None:
        """Should omit token indexes for tokens that don't resolve to any IDs."""
        mocker.patch(
            "db.lexical_search.fetch_title_token_ids",
            new=AsyncMock(return_value={0: [], 1: [44]}),
        )
        result = await lexical_search._resolve_all_title_tokens([["alpha", "beta"]])
        assert result == [{1: [44]}]

    @pytest.mark.asyncio
    async def test_all_tokens_unresolved(self, mocker) -> None:
        """Should return empty map when all tokens fail to resolve."""
        mocker.patch(
            "db.lexical_search.fetch_title_token_ids",
            new=AsyncMock(return_value={}),
        )
        result = await lexical_search._resolve_all_title_tokens([["unknown", "tokens"]])
        assert result == [{}]

    @pytest.mark.asyncio
    async def test_multiple_title_searches_independent_resolution(self, mocker) -> None:
        """Should resolve each title search independently."""
        mocker.patch(
            "db.lexical_search.fetch_title_token_ids",
            new=AsyncMock(return_value={0: [1], 1: [2], 2: [3], 3: [4]}),
        )
        result = await lexical_search._resolve_all_title_tokens([
            ["foo", "bar"],
            ["baz", "qux"],
        ])
        assert result == [{0: [1], 1: [2]}, {0: [3], 1: [4]}]

    @pytest.mark.asyncio
    async def test_empty_title_search_in_list(self, mocker) -> None:
        """Should handle empty title searches within the list."""
        mocker.patch(
            "db.lexical_search.fetch_title_token_ids",
            new=AsyncMock(return_value={0: [99]}),
        )
        result = await lexical_search._resolve_all_title_tokens([[], ["token"]])
        assert result == [{}, {0: [99]}]

    @pytest.mark.asyncio
    async def test_multiple_term_ids_per_token(self, mocker) -> None:
        """Should handle tokens that resolve to multiple term IDs."""
        mocker.patch(
            "db.lexical_search.fetch_title_token_ids",
            new=AsyncMock(return_value={0: [101, 102, 103]}),
        )
        result = await lexical_search._resolve_all_title_tokens([["fuzzy"]])
        assert result == [{0: [101, 102, 103]}]


class TestResolveCharacterTermIds:
    """Tests for the _resolve_character_term_ids helper function."""

    @pytest.mark.asyncio
    async def test_empty_input_short_circuits(self, mocker) -> None:
        """Should return empty map and skip DB helper for empty input."""
        fetch_term_ids = mocker.patch("db.lexical_search.fetch_character_term_ids", new=AsyncMock())
        result = await lexical_search._resolve_character_term_ids([])
        assert result == {}
        fetch_term_ids.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_phrase(self, mocker) -> None:
        """Should resolve a single character phrase."""
        fetch_term_ids = mocker.patch(
            "db.lexical_search.fetch_character_term_ids",
            new=AsyncMock(return_value={0: [100]}),
        )
        result = await lexical_search._resolve_character_term_ids(["batman"])
        fetch_term_ids.assert_awaited_once()
        args = fetch_term_ids.await_args.args
        assert args[0] == [0]
        assert args[1] == ["%batman%"]
        assert result == {0: [100]}

    @pytest.mark.asyncio
    async def test_multiple_phrases(self, mocker) -> None:
        """Should resolve multiple character phrases in one batch."""
        fetch_term_ids = mocker.patch(
            "db.lexical_search.fetch_character_term_ids",
            new=AsyncMock(return_value={0: [1], 1: [2], 2: [3]}),
        )
        result = await lexical_search._resolve_character_term_ids(["batman", "joker", "robin"])
        args = fetch_term_ids.await_args.args
        assert args[0] == [0, 1, 2]
        assert args[1] == ["%batman%", "%joker%", "%robin%"]
        assert result == {0: [1], 1: [2], 2: [3]}

    @pytest.mark.asyncio
    async def test_escapes_like_metacharacters(self, mocker) -> None:
        """Should escape LIKE metacharacters in character phrases."""
        fetch_term_ids = mocker.patch(
            "db.lexical_search.fetch_character_term_ids",
            new=AsyncMock(return_value={0: [1], 1: [2]}),
        )
        result = await lexical_search._resolve_character_term_ids(["100%_hero", r"path\name"])
        args = fetch_term_ids.await_args.args
        assert args[1] == [r"%100\%\_hero%", r"%path\\name%"]
        assert result == {0: [1], 1: [2]}

    @pytest.mark.asyncio
    async def test_partial_resolution(self, mocker) -> None:
        """Should handle partial resolution where some phrases don't match."""
        fetch_term_ids = mocker.patch(
            "db.lexical_search.fetch_character_term_ids",
            new=AsyncMock(return_value={0: [1]}),  # Only first phrase resolves
        )
        result = await lexical_search._resolve_character_term_ids(["batman", "unknown"])
        assert result == {0: [1]}


class TestResolveExactExcludeTitleTermIds:
    """Tests for the _resolve_exact_exclude_title_term_ids helper function."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self, mocker) -> None:
        """Should return empty list for empty input."""
        fetch_exact = mocker.patch("db.lexical_search.fetch_title_token_ids_exact", new=AsyncMock())
        result = await lexical_search._resolve_exact_exclude_title_term_ids([])
        assert result == []
        fetch_exact.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_token_resolution(self, mocker) -> None:
        """Should resolve a single token with exact lookup."""
        fetch_exact = mocker.patch(
            "db.lexical_search.fetch_title_token_ids_exact",
            new=AsyncMock(return_value={0: [101]}),
        )
        result = await lexical_search._resolve_exact_exclude_title_term_ids(["matrix"])
        assert result == [101]
        fetch_exact.assert_awaited_once_with(["matrix"], max_df=lexical_search.MAX_DF)

    @pytest.mark.asyncio
    async def test_deduplicates_resolved_ids(self, mocker) -> None:
        """Should deduplicate IDs across multiple tokens in order."""
        fetch_exact = mocker.patch(
            "db.lexical_search.fetch_title_token_ids_exact",
            new=AsyncMock(return_value={0: [11], 1: [11, 22], 2: []}),
        )
        result = await lexical_search._resolve_exact_exclude_title_term_ids(["a", "b", "c"])
        assert result == [11, 22]
        fetch_exact.assert_awaited_once_with(["a", "b", "c"], max_df=lexical_search.MAX_DF)

    @pytest.mark.asyncio
    async def test_preserves_first_seen_order(self, mocker) -> None:
        """Should preserve first-seen order when deduplicating."""
        mocker.patch(
            "db.lexical_search.fetch_title_token_ids_exact",
            new=AsyncMock(return_value={0: [1, 2], 1: [3, 1], 2: [2, 4]}),
        )
        result = await lexical_search._resolve_exact_exclude_title_term_ids(["x", "y", "z"])
        assert result == [1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_uses_max_df_constant(self, mocker) -> None:
        """Should pass MAX_DF constant to exact lookup."""
        fetch_exact = mocker.patch(
            "db.lexical_search.fetch_title_token_ids_exact",
            new=AsyncMock(return_value={}),
        )
        await lexical_search._resolve_exact_exclude_title_term_ids(["token"])
        assert fetch_exact.await_args.kwargs["max_df"] == lexical_search.MAX_DF

    @pytest.mark.asyncio
    async def test_all_tokens_unresolved(self, mocker) -> None:
        """Should return empty list when no tokens resolve."""
        mocker.patch(
            "db.lexical_search.fetch_title_token_ids_exact",
            new=AsyncMock(return_value={}),
        )
        result = await lexical_search._resolve_exact_exclude_title_term_ids(["a", "b", "c"])
        assert result == []


# =============================================================================
#                       POSTING SEARCH TESTS
# =============================================================================


class TestSearchPeoplePostings:
    """Tests for the search_people_postings function."""

    @pytest.mark.asyncio
    async def test_delegates_to_phrase_match_counts(self, mocker) -> None:
        """Should delegate to fetch_phrase_postings_match_counts with person table."""
        fetch_counts = mocker.patch(
            "db.lexical_search.fetch_phrase_postings_match_counts",
            new=AsyncMock(return_value={10: 2}),
        )
        filters = MetadataFilters(min_runtime=80)
        result = await lexical_search.search_people_postings([1, 2], filters=filters, exclude_movie_ids={99})
        fetch_counts.assert_awaited_once_with(
            lexical_search.PostingTable.PERSON,
            [1, 2],
            filters,
            {99},
        )
        assert result == {10: 2}

    @pytest.mark.asyncio
    async def test_with_no_filters(self, mocker) -> None:
        """Should work with no filters or exclude IDs."""
        fetch_counts = mocker.patch(
            "db.lexical_search.fetch_phrase_postings_match_counts",
            new=AsyncMock(return_value={1: 1, 2: 2}),
        )
        result = await lexical_search.search_people_postings([100])
        fetch_counts.assert_awaited_once_with(
            lexical_search.PostingTable.PERSON,
            [100],
            None,
            None,
        )
        assert result == {1: 1, 2: 2}

    @pytest.mark.asyncio
    async def test_empty_term_ids(self, mocker) -> None:
        """Should handle empty term IDs list."""
        fetch_counts = mocker.patch(
            "db.lexical_search.fetch_phrase_postings_match_counts",
            new=AsyncMock(return_value={}),
        )
        result = await lexical_search.search_people_postings([])
        assert result == {}


class TestSearchStudioPostings:
    """Tests for the search_studio_postings function."""

    @pytest.mark.asyncio
    async def test_delegates_to_phrase_match_counts(self, mocker) -> None:
        """Should delegate to fetch_phrase_postings_match_counts with studio table."""
        fetch_counts = mocker.patch(
            "db.lexical_search.fetch_phrase_postings_match_counts",
            new=AsyncMock(return_value={8: 1}),
        )
        result = await lexical_search.search_studio_postings([7], filters=None, exclude_movie_ids=None)
        fetch_counts.assert_awaited_once_with(lexical_search.PostingTable.STUDIO, [7], None, None)
        assert result == {8: 1}

    @pytest.mark.asyncio
    async def test_with_filters_and_excludes(self, mocker) -> None:
        """Should pass filters and exclude IDs correctly."""
        fetch_counts = mocker.patch(
            "db.lexical_search.fetch_phrase_postings_match_counts",
            new=AsyncMock(return_value={100: 3}),
        )
        filters = MetadataFilters(max_runtime=180, genres=[Genre.ACTION, Genre.ADVENTURE])
        result = await lexical_search.search_studio_postings(
            [1, 2, 3],
            filters=filters,
            exclude_movie_ids={50, 51},
        )
        fetch_counts.assert_awaited_once_with(
            lexical_search.PostingTable.STUDIO,
            [1, 2, 3],
            filters,
            {50, 51},
        )
        assert result == {100: 3}


class TestSearchCharacterPostings:
    """Tests for the search_character_postings function."""

    @pytest.mark.asyncio
    async def test_empty_phrases_short_circuits(self, mocker) -> None:
        """Should return empty map for empty phrase list."""
        resolver = mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock())
        fetch_counts = mocker.patch("db.lexical_search.fetch_character_match_counts", new=AsyncMock())
        result = await lexical_search.search_character_postings([])
        assert result == {}
        resolver.assert_not_awaited()
        fetch_counts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_resolved_terms_short_circuits(self, mocker) -> None:
        """Should return empty map if character terms do not resolve."""
        resolver = mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value={}),
        )
        fetch_counts = mocker.patch("db.lexical_search.fetch_character_match_counts", new=AsyncMock())
        result = await lexical_search.search_character_postings(["batman"])
        assert result == {}
        resolver.assert_awaited_once_with(["batman"])
        fetch_counts.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delegates_with_flattened_pairs(self, mocker) -> None:
        """Should flatten query_idx/term_id pairs and pass use_eligible flag."""
        mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value={0: [11, 12], 2: [33]}),
        )
        fetch_counts = mocker.patch(
            "db.lexical_search.fetch_character_match_counts",
            new=AsyncMock(return_value={1: 2}),
        )
        filters = MetadataFilters(min_release_ts=1)
        result = await lexical_search.search_character_postings(
            ["a", "b", "c"],
            filters=filters,
            exclude_movie_ids={99},
        )
        fetch_counts.assert_awaited_once_with([0, 0, 2], [11, 12, 33], True, filters, {99})
        assert result == {1: 2}

    @pytest.mark.asyncio
    async def test_use_eligible_false_when_no_filters(self, mocker) -> None:
        """Should set use_eligible to False when filters is None."""
        mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value={0: [1]}),
        )
        fetch_counts = mocker.patch(
            "db.lexical_search.fetch_character_match_counts",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_character_postings(["test"], filters=None)
        args = fetch_counts.await_args.args
        assert args[2] is False  # use_eligible

    @pytest.mark.asyncio
    async def test_use_eligible_false_when_filters_inactive(self, mocker) -> None:
        """Should set use_eligible to False when filters exist but are inactive."""
        mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value={0: [1]}),
        )
        fetch_counts = mocker.patch(
            "db.lexical_search.fetch_character_match_counts",
            new=AsyncMock(return_value={}),
        )
        # MetadataFilters with all None values is inactive
        inactive_filters = MetadataFilters()
        await lexical_search.search_character_postings(["test"], filters=inactive_filters)
        args = fetch_counts.await_args.args
        assert args[2] is False  # use_eligible


class TestSearchCharacterPostingsByQuery:
    """Tests for the search_character_postings_by_query function."""

    @pytest.mark.asyncio
    async def test_empty_phrases_short_circuits(self, mocker) -> None:
        """Should return [] and skip DB helpers for empty input."""
        resolver = mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock())
        fetch_by_query = mocker.patch("db.lexical_search.fetch_character_match_counts_by_query", new=AsyncMock())
        result = await lexical_search.search_character_postings_by_query([])
        assert result == []
        resolver.assert_not_awaited()
        fetch_by_query.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("resolved_map", "expected"),
        [
            ({}, [{}, {}, {}]),
            ({0: [], 1: []}, [{}, {}, {}]),
        ],
    )
    async def test_returns_empty_maps_when_no_term_ids(
        self,
        mocker,
        resolved_map: dict[int, list[int]],
        expected: list[dict[int, int]],
    ) -> None:
        """When phrase resolution yields no usable term IDs, output should preserve query count as empty maps."""
        mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value=resolved_map),
        )
        fetch_by_query = mocker.patch("db.lexical_search.fetch_character_match_counts_by_query", new=AsyncMock())
        result = await lexical_search.search_character_postings_by_query(["a", "b", "c"])
        assert result == expected
        fetch_by_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_delegates_and_aligns_result_indexes(self, mocker) -> None:
        """Should flatten pairs once and align sparse DB output to phrase order."""
        resolver = mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value={0: [11], 2: [31, 32]}),
        )
        fetch_by_query = mocker.patch(
            "db.lexical_search.fetch_character_match_counts_by_query",
            new=AsyncMock(return_value={0: {100: 1}, 2: {200: 2}}),
        )
        filters = MetadataFilters(max_runtime=120)
        result = await lexical_search.search_character_postings_by_query(
            ["p0", "p1", "p2"],
            filters=filters,
            exclude_movie_ids={999},
        )
        resolver.assert_awaited_once_with(["p0", "p1", "p2"])
        fetch_by_query.assert_awaited_once_with(
            [0, 2, 2],
            [11, 31, 32],
            True,
            filters,
            {999},
        )
        assert result == [{100: 1}, {}, {200: 2}]

    @pytest.mark.asyncio
    async def test_all_phrases_resolve_to_results(self, mocker) -> None:
        """Should handle case where all phrases resolve and have results."""
        mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value={0: [1], 1: [2], 2: [3]}),
        )
        mocker.patch(
            "db.lexical_search.fetch_character_match_counts_by_query",
            new=AsyncMock(return_value={0: {10: 1}, 1: {20: 2}, 2: {30: 3}}),
        )
        result = await lexical_search.search_character_postings_by_query(["a", "b", "c"])
        assert result == [{10: 1}, {20: 2}, {30: 3}]


class TestSearchTitlePostings:
    """
    Comprehensive tests for the search_title_postings function.
    
    This function is central to title-based lexical search and involves:
    - Input validation and short-circuit logic
    - Flattening token_term_id_map into parallel arrays
    - Computing F-score coefficients from constants
    - Counting non-empty token positions (k)
    - Determining eligible-set usage based on filter activity
    - Delegating to fetch_title_token_match_scores with computed parameters
    
    The F-score formula used is:
        title_score = (1+β²)·(coverage·specificity) / (β²·specificity + coverage)
    where:
        coverage = m/k (matched tokens / query tokens)
        specificity = m/L (matched tokens / movie title tokens)
        β = TITLE_SCORE_BETA (default 2.0)
    """

    # =========================================================================
    #                    SHORT-CIRCUIT / EMPTY INPUT TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_empty_map_short_circuits(self, mocker) -> None:
        """Should return empty map for empty token map without calling DB."""
        fetch_scores = mocker.patch("db.lexical_search.fetch_title_token_match_scores", new=AsyncMock())
        result = await lexical_search.search_title_postings({})
        assert result == {}
        fetch_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_term_ids_short_circuits(self, mocker) -> None:
        """Should return empty map when flattened term list is empty."""
        fetch_scores = mocker.patch("db.lexical_search.fetch_title_token_match_scores", new=AsyncMock())
        result = await lexical_search.search_title_postings({0: [], 1: []})
        assert result == {}
        fetch_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_all_empty_lists_short_circuits(self, mocker) -> None:
        """Should short-circuit when all token positions have empty term ID lists."""
        fetch_scores = mocker.patch("db.lexical_search.fetch_title_token_match_scores", new=AsyncMock())
        result = await lexical_search.search_title_postings({0: [], 1: [], 2: []})
        assert result == {}
        fetch_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_single_empty_list_short_circuits(self, mocker) -> None:
        """Should short-circuit when single token position has empty list."""
        fetch_scores = mocker.patch("db.lexical_search.fetch_title_token_match_scores", new=AsyncMock())
        result = await lexical_search.search_title_postings({0: []})
        assert result == {}
        fetch_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_many_empty_lists_short_circuits(self, mocker) -> None:
        """Should short-circuit with many token positions all having empty lists."""
        fetch_scores = mocker.patch("db.lexical_search.fetch_title_token_match_scores", new=AsyncMock())
        token_map = {i: [] for i in range(100)}
        result = await lexical_search.search_title_postings(token_map)
        assert result == {}
        fetch_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_short_circuit_with_filters_present(self, mocker) -> None:
        """Should short-circuit even when filters are provided if no term IDs."""
        fetch_scores = mocker.patch("db.lexical_search.fetch_title_token_match_scores", new=AsyncMock())
        filters = MetadataFilters(min_runtime=90, max_runtime=180)
        result = await lexical_search.search_title_postings({0: [], 1: []}, filters=filters)
        assert result == {}
        fetch_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_short_circuit_with_exclude_movie_ids_present(self, mocker) -> None:
        """Should short-circuit even when exclude_movie_ids are provided if no term IDs."""
        fetch_scores = mocker.patch("db.lexical_search.fetch_title_token_match_scores", new=AsyncMock())
        result = await lexical_search.search_title_postings({}, exclude_movie_ids={1, 2, 3})
        assert result == {}
        fetch_scores.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_short_circuit_with_both_filters_and_excludes(self, mocker) -> None:
        """Should short-circuit with both filters and excludes if no term IDs."""
        fetch_scores = mocker.patch("db.lexical_search.fetch_title_token_match_scores", new=AsyncMock())
        filters = MetadataFilters(genres=[Genre.ACTION, Genre.ADVENTURE])
        result = await lexical_search.search_title_postings(
            {0: [], 1: []},
            filters=filters,
            exclude_movie_ids={99, 100},
        )
        assert result == {}
        fetch_scores.assert_not_awaited()

    # =========================================================================
    #                    MAP FLATTENING / PARALLEL ARRAYS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_single_token_single_term_id_flattening(self, mocker) -> None:
        """Should correctly flatten single token with single term ID."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [100]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["token_idxs"] == [0]
        assert kwargs["term_ids"] == [100]

    @pytest.mark.asyncio
    async def test_single_token_multiple_term_ids_flattening(self, mocker) -> None:
        """Should flatten single token with multiple term IDs (fuzzy expansion)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [100, 101, 102]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["token_idxs"] == [0, 0, 0]
        assert kwargs["term_ids"] == [100, 101, 102]

    @pytest.mark.asyncio
    async def test_multiple_tokens_single_term_id_each_flattening(self, mocker) -> None:
        """Should flatten multiple tokens each with single term ID."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [10], 1: [20], 2: [30]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["token_idxs"] == [0, 1, 2]
        assert kwargs["term_ids"] == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_multiple_tokens_multiple_term_ids_flattening(self, mocker) -> None:
        """Should flatten multiple tokens each with multiple term IDs."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [10, 11], 1: [20, 21, 22]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["token_idxs"] == [0, 0, 1, 1, 1]
        assert kwargs["term_ids"] == [10, 11, 20, 21, 22]

    @pytest.mark.asyncio
    async def test_mixed_empty_and_populated_token_positions(self, mocker) -> None:
        """Should skip empty positions but include populated ones in flattening."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [], 1: [20, 21], 2: []})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["token_idxs"] == [1, 1]
        assert kwargs["term_ids"] == [20, 21]

    @pytest.mark.asyncio
    async def test_non_contiguous_token_indices(self, mocker) -> None:
        """Should handle non-contiguous token indices correctly."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [10], 5: [50], 10: [100]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["token_idxs"] == [0, 5, 10]
        assert kwargs["term_ids"] == [10, 50, 100]

    @pytest.mark.asyncio
    async def test_large_token_map_flattening(self, mocker) -> None:
        """Should handle large token maps with many positions."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        token_map = {i: [i * 100, i * 100 + 1] for i in range(50)}
        await lexical_search.search_title_postings(token_map)
        kwargs = fetch_scores.await_args.kwargs
        assert len(kwargs["token_idxs"]) == 100
        assert len(kwargs["term_ids"]) == 100

    @pytest.mark.asyncio
    async def test_preserves_term_id_order_within_position(self, mocker) -> None:
        """Should preserve order of term IDs within each token position."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [999, 1, 500, 2]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["term_ids"] == [999, 1, 500, 2]

    @pytest.mark.asyncio
    async def test_negative_token_indices(self, mocker) -> None:
        """Should handle negative token indices (unusual but valid dict keys)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({-1: [10], 0: [20], 1: [30]})
        kwargs = fetch_scores.await_args.kwargs
        # Dict iteration order is insertion order in Python 3.7+
        assert -1 in kwargs["token_idxs"]
        assert 0 in kwargs["token_idxs"]
        assert 1 in kwargs["token_idxs"]

    @pytest.mark.asyncio
    async def test_large_term_ids(self, mocker) -> None:
        """Should handle large term ID values (BIGINT range)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        large_ids = [2**62, 2**62 + 1, 2**62 + 2]
        await lexical_search.search_title_postings({0: large_ids})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["term_ids"] == large_ids

    # =========================================================================
    #                    K CALCULATION (NON-EMPTY TOKEN COUNT)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_k_equals_one_for_single_non_empty_position(self, mocker) -> None:
        """Should set k=1 when only one token position has term IDs."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 1

    @pytest.mark.asyncio
    async def test_k_equals_count_of_non_empty_positions(self, mocker) -> None:
        """Should set k to count of positions with non-empty term ID lists."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1], 1: [2], 2: [3]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 3

    @pytest.mark.asyncio
    async def test_k_excludes_empty_positions(self, mocker) -> None:
        """Should not count empty positions toward k."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [], 1: [20, 21], 2: []})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 1

    @pytest.mark.asyncio
    async def test_k_with_many_term_ids_per_position(self, mocker) -> None:
        """k should count positions, not total term IDs."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        # 2 positions, but many term IDs total
        await lexical_search.search_title_postings({0: [1, 2, 3, 4, 5], 1: [10, 20, 30]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 2

    @pytest.mark.asyncio
    async def test_k_with_mixed_empty_and_populated(self, mocker) -> None:
        """Should correctly count k with interleaved empty and populated positions."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({
            0: [],
            1: [10],
            2: [],
            3: [30],
            4: [],
            5: [50],
            6: [],
        })
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 3

    @pytest.mark.asyncio
    async def test_k_with_all_positions_having_single_term_id(self, mocker) -> None:
        """k should equal number of positions when each has exactly one term ID."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1], 1: [2], 2: [3], 3: [4], 4: [5]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 5

    @pytest.mark.asyncio
    async def test_k_zero_causes_short_circuit(self, mocker) -> None:
        """Should short-circuit when k would be zero (all positions empty)."""
        fetch_scores = mocker.patch("db.lexical_search.fetch_title_token_match_scores", new=AsyncMock())
        result = await lexical_search.search_title_postings({0: [], 1: [], 2: []})
        assert result == {}
        fetch_scores.assert_not_awaited()

    # =========================================================================
    #                    F-SCORE COEFFICIENT CALCULATION
    # =========================================================================

    @pytest.mark.asyncio
    async def test_f_coeff_calculation_with_default_beta(self, mocker) -> None:
        """Should calculate f_coeff as 1 + β² using TITLE_SCORE_BETA."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        expected_f_coeff = 1.0 + (lexical_search.TITLE_SCORE_BETA ** 2)
        assert kwargs["f_coeff"] == expected_f_coeff

    @pytest.mark.asyncio
    async def test_f_coeff_is_5_for_beta_2(self, mocker) -> None:
        """With β=2.0, f_coeff should be 1 + 4 = 5.0."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        # Verify the constant is what we expect
        assert lexical_search.TITLE_SCORE_BETA == 2.0
        assert kwargs["f_coeff"] == 5.0

    @pytest.mark.asyncio
    async def test_f_coeff_consistent_across_calls(self, mocker) -> None:
        """f_coeff should be consistent across multiple calls."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        first_f_coeff = fetch_scores.await_args.kwargs["f_coeff"]
        
        await lexical_search.search_title_postings({0: [2], 1: [3]})
        second_f_coeff = fetch_scores.await_args.kwargs["f_coeff"]
        
        assert first_f_coeff == second_f_coeff

    # =========================================================================
    #                    SCORING CONSTANTS PASSTHROUGH
    # =========================================================================

    @pytest.mark.asyncio
    async def test_passes_beta_constant(self, mocker) -> None:
        """Should pass TITLE_SCORE_BETA to DB helper."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["beta"] == lexical_search.TITLE_SCORE_BETA

    @pytest.mark.asyncio
    async def test_passes_title_score_threshold(self, mocker) -> None:
        """Should pass TITLE_SCORE_THRESHOLD to DB helper."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["title_score_threshold"] == lexical_search.TITLE_SCORE_THRESHOLD

    @pytest.mark.asyncio
    async def test_passes_title_max_candidates(self, mocker) -> None:
        """Should pass TITLE_MAX_CANDIDATES to DB helper."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["title_max_candidates"] == lexical_search.TITLE_MAX_CANDIDATES

    @pytest.mark.asyncio
    async def test_all_scoring_constants_passed_together(self, mocker) -> None:
        """Should pass all scoring constants in a single call."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1], 1: [2]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["beta"] == lexical_search.TITLE_SCORE_BETA
        assert kwargs["title_score_threshold"] == lexical_search.TITLE_SCORE_THRESHOLD
        assert kwargs["title_max_candidates"] == lexical_search.TITLE_MAX_CANDIDATES
        assert kwargs["f_coeff"] == 1.0 + (lexical_search.TITLE_SCORE_BETA ** 2)

    @pytest.mark.asyncio
    async def test_scoring_constants_have_expected_values(self, mocker) -> None:
        """Verify the module-level constants have expected values per spec."""
        assert lexical_search.TITLE_SCORE_BETA == 2.0
        assert lexical_search.TITLE_SCORE_THRESHOLD == 0.15
        assert lexical_search.TITLE_MAX_CANDIDATES == 10_000
        assert lexical_search.MAX_DF == 10_000

    # =========================================================================
    #                    USE_ELIGIBLE FLAG (FILTER ACTIVITY)
    # =========================================================================

    @pytest.mark.asyncio
    async def test_use_eligible_false_when_filters_none(self, mocker) -> None:
        """Should set use_eligible=False when filters is None."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]}, filters=None)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is False

    @pytest.mark.asyncio
    async def test_use_eligible_false_when_filters_empty(self, mocker) -> None:
        """Should set use_eligible=False when filters has no active fields."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        empty_filters = MetadataFilters()
        await lexical_search.search_title_postings({0: [1]}, filters=empty_filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is False

    @pytest.mark.asyncio
    async def test_use_eligible_true_with_min_release_ts(self, mocker) -> None:
        """Should set use_eligible=True when min_release_ts is set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(min_release_ts=1609459200)
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_use_eligible_true_with_max_release_ts(self, mocker) -> None:
        """Should set use_eligible=True when max_release_ts is set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(max_release_ts=1640995200)
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_use_eligible_true_with_min_runtime(self, mocker) -> None:
        """Should set use_eligible=True when min_runtime is set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(min_runtime=90)
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_use_eligible_true_with_max_runtime(self, mocker) -> None:
        """Should set use_eligible=True when max_runtime is set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(max_runtime=180)
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_use_eligible_true_with_min_maturity_rank(self, mocker) -> None:
        """Should set use_eligible=True when min_maturity_rank is set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(min_maturity_rank=1)
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_use_eligible_true_with_max_maturity_rank(self, mocker) -> None:
        """Should set use_eligible=True when max_maturity_rank is set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(max_maturity_rank=3)
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_use_eligible_true_with_genres(self, mocker) -> None:
        """Should set use_eligible=True when genres is set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(genres=[Genre.ACTION, Genre.ADVENTURE, Genre.SCI_FI])
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_use_eligible_true_with_watch_offer_keys(self, mocker) -> None:
        """Should set use_eligible=True when watch_offer_keys is set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(watch_offer_keys=[8, 337, 15])
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_use_eligible_true_with_multiple_filters(self, mocker) -> None:
        """Should set use_eligible=True when multiple filters are set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(
            min_runtime=90,
            max_runtime=180,
            genres=[Genre.ACTION],
            min_release_ts=946684800,
        )
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_use_eligible_true_with_all_filters(self, mocker) -> None:
        """Should set use_eligible=True when all filter fields are set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(
            min_release_ts=946684800,
            max_release_ts=1640995200,
            min_runtime=60,
            max_runtime=240,
            min_maturity_rank=1,
            max_maturity_rank=5,
            genres=[Genre.ACTION, Genre.ADVENTURE],
            watch_offer_keys=[8, 337],
        )
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    # =========================================================================
    #                    FILTERS PASSTHROUGH
    # =========================================================================

    @pytest.mark.asyncio
    async def test_passes_filters_object_to_db_helper(self, mocker) -> None:
        """Should pass the filters object directly to DB helper."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(min_runtime=90, max_runtime=180)
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["filters"] is filters

    @pytest.mark.asyncio
    async def test_passes_none_filters_to_db_helper(self, mocker) -> None:
        """Should pass None filters to DB helper when not provided."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["filters"] is None

    @pytest.mark.asyncio
    async def test_passes_empty_filters_to_db_helper(self, mocker) -> None:
        """Should pass empty MetadataFilters object to DB helper."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        empty_filters = MetadataFilters()
        await lexical_search.search_title_postings({0: [1]}, filters=empty_filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["filters"] is empty_filters

    # =========================================================================
    #                    EXCLUDE_TERM_IDS PASSTHROUGH
    # =========================================================================

    @pytest.mark.asyncio
    async def test_passes_exclude_movie_ids_to_db_helper(self, mocker) -> None:
        """Should pass exclude_movie_ids set to DB helper."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        exclude_ids = [50, 51, 52]
        await lexical_search.search_title_postings({0: [1]}, exclude_movie_ids=set(exclude_ids))
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["exclude_movie_ids"] == set(exclude_ids)

    @pytest.mark.asyncio
    async def test_passes_none_exclude_movie_ids_to_db_helper(self, mocker) -> None:
        """Should pass None for exclude_movie_ids when not provided."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["exclude_movie_ids"] is None

    @pytest.mark.asyncio
    async def test_passes_empty_exclude_movie_ids_set(self, mocker) -> None:
        """Should pass empty set for exclude_movie_ids when provided empty."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]}, exclude_movie_ids=set())
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["exclude_movie_ids"] == set()

    @pytest.mark.asyncio
    async def test_passes_single_exclude_term_id(self, mocker) -> None:
        """Should pass single-element exclude_movie_ids set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]}, exclude_movie_ids={99})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["exclude_movie_ids"] == {99}

    @pytest.mark.asyncio
    async def test_passes_many_exclude_movie_ids(self, mocker) -> None:
        """Should pass large exclude_movie_ids set."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        exclude_ids = list(range(1000))
        await lexical_search.search_title_postings({0: [1]}, exclude_movie_ids=set(exclude_ids))
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["exclude_movie_ids"] == set(exclude_ids)

    @pytest.mark.asyncio
    async def test_passes_large_exclude_movie_ids(self, mocker) -> None:
        """Should handle large term ID values in exclude list."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        large_ids = [2**62, 2**62 + 1]
        await lexical_search.search_title_postings({0: [1]}, exclude_movie_ids=set(large_ids))
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["exclude_movie_ids"] == set(large_ids)

    # =========================================================================
    #                    COMBINED ARGUMENTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_all_arguments_passed_correctly(self, mocker) -> None:
        """Should pass all arguments correctly in a comprehensive call."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={1: 0.9, 2: 0.75}),
        )
        filters = MetadataFilters(min_runtime=90, genres=[Genre.ACTION, Genre.ADVENTURE])
        exclude_ids = [100, 101]
        token_map = {0: [10, 11], 1: [20], 2: [30, 31, 32]}
        
        result = await lexical_search.search_title_postings(
            token_map,
            filters=filters,
            exclude_movie_ids=set(exclude_ids),
        )
        
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["token_idxs"] == [0, 0, 1, 2, 2, 2]
        assert kwargs["term_ids"] == [10, 11, 20, 30, 31, 32]
        assert kwargs["use_eligible"] is True
        assert kwargs["f_coeff"] == 5.0
        assert kwargs["k"] == 3
        assert kwargs["beta"] == 2.0
        assert kwargs["title_score_threshold"] == 0.15
        assert kwargs["title_max_candidates"] == 10_000
        assert kwargs["filters"] is filters
        assert kwargs["exclude_movie_ids"] == set(exclude_ids)
        assert result == {1: 0.9, 2: 0.75}

    @pytest.mark.asyncio
    async def test_minimal_valid_call(self, mocker) -> None:
        """Should work with minimal valid input (single token, single term ID)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={42: 0.5}),
        )
        result = await lexical_search.search_title_postings({0: [1]})
        
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["token_idxs"] == [0]
        assert kwargs["term_ids"] == [1]
        assert kwargs["k"] == 1
        assert kwargs["use_eligible"] is False
        assert kwargs["filters"] is None
        assert kwargs["exclude_movie_ids"] is None
        assert result == {42: 0.5}

    # =========================================================================
    #                    RETURN VALUE HANDLING
    # =========================================================================

    @pytest.mark.asyncio
    async def test_returns_db_helper_result_directly(self, mocker) -> None:
        """Should return the result from DB helper unchanged."""
        expected_result = {1: 0.9, 2: 0.8, 3: 0.7}
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value=expected_result),
        )
        result = await lexical_search.search_title_postings({0: [1]})
        assert result == expected_result

    @pytest.mark.asyncio
    async def test_returns_empty_dict_from_db_helper(self, mocker) -> None:
        """Should return empty dict when DB helper returns empty dict."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        result = await lexical_search.search_title_postings({0: [1]})
        assert result == {}

    @pytest.mark.asyncio
    async def test_returns_single_result_from_db_helper(self, mocker) -> None:
        """Should return single-entry dict from DB helper."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={12345: 0.95}),
        )
        result = await lexical_search.search_title_postings({0: [1]})
        assert result == {12345: 0.95}

    @pytest.mark.asyncio
    async def test_returns_many_results_from_db_helper(self, mocker) -> None:
        """Should return large result dict from DB helper."""
        expected_result = {i: 0.5 + (i / 20000) for i in range(10000)}
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value=expected_result),
        )
        result = await lexical_search.search_title_postings({0: [1]})
        assert result == expected_result
        assert len(result) == 10000

    @pytest.mark.asyncio
    async def test_returns_scores_at_threshold_boundary(self, mocker) -> None:
        """Should return scores exactly at the threshold (0.15)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={1: 0.15, 2: 0.150001}),
        )
        result = await lexical_search.search_title_postings({0: [1]})
        assert result == {1: 0.15, 2: 0.150001}

    @pytest.mark.asyncio
    async def test_returns_perfect_score(self, mocker) -> None:
        """Should return perfect score (1.0) when returned by DB helper."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={1: 1.0}),
        )
        result = await lexical_search.search_title_postings({0: [1]})
        assert result == {1: 1.0}

    @pytest.mark.asyncio
    async def test_returns_various_score_values(self, mocker) -> None:
        """Should return various score values correctly."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={
                1: 0.15,      # At threshold
                2: 0.5,       # Mid-range
                3: 0.999999,  # Near perfect
                4: 1.0,       # Perfect
                5: 0.151,     # Just above threshold
            }),
        )
        result = await lexical_search.search_title_postings({0: [1]})
        assert len(result) == 5
        assert result[1] == 0.15
        assert result[4] == 1.0

    @pytest.mark.asyncio
    async def test_returns_large_movie_ids(self, mocker) -> None:
        """Should handle large movie IDs (BIGINT range) in results."""
        large_id = 2**62
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={large_id: 0.8}),
        )
        result = await lexical_search.search_title_postings({0: [1]})
        assert result == {large_id: 0.8}

    # =========================================================================
    #                    EDGE CASES AND BOUNDARY CONDITIONS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_single_position_many_term_ids(self, mocker) -> None:
        """Should handle single position with many fuzzy-expanded term IDs."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        many_term_ids = list(range(1, 501))  # 500 fuzzy expansions
        await lexical_search.search_title_postings({0: many_term_ids})
        kwargs = fetch_scores.await_args.kwargs
        assert len(kwargs["term_ids"]) == 500
        assert kwargs["k"] == 1  # Still only one token position

    @pytest.mark.asyncio
    async def test_many_positions_single_term_id_each(self, mocker) -> None:
        """Should handle many positions each with single term ID."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        token_map = {i: [i * 10] for i in range(100)}
        await lexical_search.search_title_postings(token_map)
        kwargs = fetch_scores.await_args.kwargs
        assert len(kwargs["term_ids"]) == 100
        assert kwargs["k"] == 100

    @pytest.mark.asyncio
    async def test_sparse_token_indices(self, mocker) -> None:
        """Should handle sparse (non-sequential) token indices."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1], 100: [2], 1000: [3]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 3
        assert 0 in kwargs["token_idxs"]
        assert 100 in kwargs["token_idxs"]
        assert 1000 in kwargs["token_idxs"]

    @pytest.mark.asyncio
    async def test_duplicate_term_ids_across_positions(self, mocker) -> None:
        """Should not deduplicate term IDs that appear in multiple positions."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        # Same term ID 99 appears in two different positions
        await lexical_search.search_title_postings({0: [99], 1: [99]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["term_ids"] == [99, 99]
        assert kwargs["token_idxs"] == [0, 1]

    @pytest.mark.asyncio
    async def test_duplicate_term_ids_within_position(self, mocker) -> None:
        """Should pass duplicate term IDs within a position (caller's responsibility)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        # Duplicates within same position (unusual but possible)
        await lexical_search.search_title_postings({0: [1, 1, 1]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["term_ids"] == [1, 1, 1]

    @pytest.mark.asyncio
    async def test_zero_term_id(self, mocker) -> None:
        """Should handle zero as a valid term ID."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [0]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["term_ids"] == [0]

    @pytest.mark.asyncio
    async def test_zero_token_index(self, mocker) -> None:
        """Should handle zero as a valid token index."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["token_idxs"] == [0]

    @pytest.mark.asyncio
    async def test_filters_with_zero_values(self, mocker) -> None:
        """Should handle filters with zero values (still active)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(min_runtime=0)
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_filters_with_empty_genres_list(self, mocker) -> None:
        """Should handle filters with empty genres list (still active)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        filters = MetadataFilters(genres=[])
        await lexical_search.search_title_postings({0: [1]}, filters=filters)
        kwargs = fetch_scores.await_args.kwargs
        # Empty list is still "not None", so is_active should be True
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_iteration_order_preserved(self, mocker) -> None:
        """Should preserve dict iteration order when flattening."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        # Python 3.7+ guarantees dict insertion order
        token_map = {2: [20], 0: [10], 1: [15]}
        await lexical_search.search_title_postings(token_map)
        kwargs = fetch_scores.await_args.kwargs
        # Should follow insertion order: 2, 0, 1
        assert kwargs["token_idxs"] == [2, 0, 1]
        assert kwargs["term_ids"] == [20, 10, 15]

    # =========================================================================
    #                    ASYNC BEHAVIOR
    # =========================================================================

    @pytest.mark.asyncio
    async def test_awaits_db_helper_once(self, mocker) -> None:
        """Should await the DB helper exactly once per call."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1], 1: [2]})
        assert fetch_scores.await_count == 1

    @pytest.mark.asyncio
    async def test_does_not_await_on_short_circuit(self, mocker) -> None:
        """Should not await DB helper when short-circuiting."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({})
        assert fetch_scores.await_count == 0

    @pytest.mark.asyncio
    async def test_multiple_sequential_calls(self, mocker) -> None:
        """Should handle multiple sequential calls correctly."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(side_effect=[{1: 0.5}, {2: 0.6}, {3: 0.7}]),
        )
        
        result1 = await lexical_search.search_title_postings({0: [10]})
        result2 = await lexical_search.search_title_postings({0: [20]})
        result3 = await lexical_search.search_title_postings({0: [30]})
        
        assert result1 == {1: 0.5}
        assert result2 == {2: 0.6}
        assert result3 == {3: 0.7}
        assert fetch_scores.await_count == 3

    # =========================================================================
    #                    REAL-WORLD SCENARIO TESTS
    # =========================================================================

    @pytest.mark.asyncio
    async def test_realistic_single_word_title_search(self, mocker) -> None:
        """Simulate searching for a single-word title like 'Matrix'."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={
                603: 1.0,    # The Matrix (exact match)
                604: 0.8,    # Matrix Reloaded
                605: 0.8,    # Matrix Revolutions
            }),
        )
        # Single token "matrix" with fuzzy expansions
        result = await lexical_search.search_title_postings({0: [101, 102, 103]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 1
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_realistic_multi_word_title_search(self, mocker) -> None:
        """Simulate searching for 'The Dark Knight'."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={
                155: 1.0,    # The Dark Knight (perfect match)
                49026: 0.9,  # The Dark Knight Rises
            }),
        )
        # Three tokens: "the", "dark", "knight"
        result = await lexical_search.search_title_postings({
            0: [1],      # "the" - single exact match
            1: [50, 51], # "dark" - fuzzy expanded
            2: [100],    # "knight" - single exact match
        })
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 3
        assert len(kwargs["term_ids"]) == 4

    @pytest.mark.asyncio
    async def test_realistic_partial_match_scenario(self, mocker) -> None:
        """Simulate partial match where some tokens don't resolve."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={12345: 0.6}),
        )
        # "star wars" where "wars" didn't resolve (empty list)
        result = await lexical_search.search_title_postings({
            0: [200, 201],  # "star" resolved
            1: [],          # "wars" didn't resolve (max_df exceeded?)
        })
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 1  # Only "star" counted
        assert kwargs["term_ids"] == [200, 201]

    @pytest.mark.asyncio
    async def test_realistic_with_genre_filter(self, mocker) -> None:
        """Simulate title search with genre filter (Action movies only)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={155: 0.95}),
        )
        filters = MetadataFilters(genres=[Genre.ACTION])
        result = await lexical_search.search_title_postings(
            {0: [100], 1: [200]},
            filters=filters,
        )
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True
        assert kwargs["filters"].genres == [Genre.ACTION]

    @pytest.mark.asyncio
    async def test_realistic_with_streaming_filter(self, mocker) -> None:
        """Simulate title search filtered to Netflix streaming."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={100: 0.7, 200: 0.65}),
        )
        # Netflix streaming: provider_id=8, method=stream(1), key = (8 << 2) | 1 = 33
        filters = MetadataFilters(watch_offer_keys=[33])
        result = await lexical_search.search_title_postings(
            {0: [10]},
            filters=filters,
        )
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["use_eligible"] is True

    @pytest.mark.asyncio
    async def test_realistic_exclude_sequel_titles(self, mocker) -> None:
        """Simulate excluding sequel-related title tokens."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={603: 0.9}),
        )
        # Exclude tokens like "reloaded", "revolutions", "2", "3"
        exclude_ids = [500, 501, 502, 503]
        result = await lexical_search.search_title_postings(
            {0: [100]},
            exclude_movie_ids=set(exclude_ids),
        )
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["exclude_movie_ids"] == set(exclude_ids)

    @pytest.mark.asyncio
    async def test_realistic_complex_query(self, mocker) -> None:
        """Simulate complex query with filters, excludes, and multiple tokens."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={
                155: 0.95,
                49026: 0.85,
                27205: 0.75,
            }),
        )
        filters = MetadataFilters(
            min_release_ts=1199145600,  # 2008-01-01
            max_release_ts=1356998400,  # 2013-01-01
            genres=[Genre.ACTION, Genre.CRIME],
            min_runtime=120,
        )
        result = await lexical_search.search_title_postings(
            {
                0: [1, 2],      # "the" with variants
                1: [50, 51],    # "dark" with variants
                2: [100, 101],  # "knight" with variants
            },
            filters=filters,
            exclude_movie_ids={999},  # Exclude some movie ID
        )
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["k"] == 3
        assert kwargs["use_eligible"] is True
        assert len(result) == 3

    # =========================================================================
    #                    F-SCORE FORMULA VERIFICATION
    # =========================================================================

    @pytest.mark.asyncio
    async def test_f_score_formula_components_correct(self, mocker) -> None:
        """Verify all F-score formula components are passed correctly."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1], 1: [2], 2: [3]})
        kwargs = fetch_scores.await_args.kwargs
        
        # F-score formula: (1+β²)·(coverage·specificity) / (β²·specificity + coverage)
        # where coverage = m/k, specificity = m/L
        # With β=2.0: f_coeff = 1 + 4 = 5
        assert kwargs["f_coeff"] == 5.0
        assert kwargs["beta"] == 2.0
        assert kwargs["k"] == 3
        # The DB helper will compute m (matched tokens) and L (title_token_count)

    @pytest.mark.asyncio
    async def test_threshold_value_matches_spec(self, mocker) -> None:
        """Verify threshold matches spec (0.15 per lexical search guide)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["title_score_threshold"] == 0.15

    @pytest.mark.asyncio
    async def test_max_candidates_value_matches_spec(self, mocker) -> None:
        """Verify max candidates matches spec (10,000 per lexical search guide)."""
        fetch_scores = mocker.patch(
            "db.lexical_search.fetch_title_token_match_scores",
            new=AsyncMock(return_value={}),
        )
        await lexical_search.search_title_postings({0: [1]})
        kwargs = fetch_scores.await_args.kwargs
        assert kwargs["title_max_candidates"] == 10_000


# =============================================================================
#                       ENTITY HELPER FOR TESTS
# =============================================================================


def _entity(
    phrase: str,
    category: EntityCategory,
    *,
    exclude: bool = False,
    corrected: str | None = None,
) -> ExtractedEntityData:
    """Build an entity object used in lexical_search tests."""
    return ExtractedEntityData(
        candidate_entity_phrase=phrase,
        most_likely_category=category.value,
        exclude_from_results=exclude,
        corrected_and_normalized_entity=corrected if corrected is not None else phrase,
    )


def _entities(*entity_list: ExtractedEntityData) -> ExtractedEntitiesResponse:
    """Build an ExtractedEntitiesResponse from a list of entities."""
    return ExtractedEntitiesResponse(entity_candidates=list(entity_list))


# =============================================================================
#                    MAIN LEXICAL_SEARCH FUNCTION TESTS
# =============================================================================


class TestLexicalSearchEmptyAndNoInclude:
    """Tests for lexical_search when there are no include entities."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_entities(self, mocker) -> None:
        """Should return empty list when entity_candidates is empty."""
        entities = ExtractedEntitiesResponse(entity_candidates=[])
        result = await lexical_search.lexical_search(entities)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_include_entities_after_normalization(self, mocker) -> None:
        """Should return empty list when no include entities survive normalization."""
        entities = _entities(_entity("###", EntityCategory.PERSON))
        mocker.patch("db.lexical_search.normalize_string", return_value="")
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        result = await lexical_search.lexical_search(entities)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_only_exclude_entities(self, mocker) -> None:
        """Should return empty list when all entities are EXCLUDE."""
        entities = _entities(
            _entity("Tom Hanks", EntityCategory.PERSON, exclude=True),
            _entity("Star Wars", EntityCategory.MOVIE_TITLE, exclude=True),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split())
        result = await lexical_search.lexical_search(entities)
        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_when_all_titles_normalize_to_empty(self, mocker) -> None:
        """Should return empty when all title entities normalize to empty tokens."""
        entities = _entities(_entity("...", EntityCategory.MOVIE_TITLE))
        mocker.patch("db.lexical_search.normalize_string", return_value="")
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        result = await lexical_search.lexical_search(entities)
        assert result == []


class TestLexicalSearchEntityCategorization:
    """Tests for how lexical_search categorizes different entity types."""

    @pytest.fixture
    def mock_all_dependencies(self, mocker):
        """Mock all external dependencies for lexical_search."""
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split() if x else [])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))

    @pytest.mark.asyncio
    async def test_person_entity_include(self, mocker, mock_all_dependencies) -> None:
        """Should categorize PERSON entities correctly for INCLUDE."""
        entities = _entities(_entity("Tom Hanks", EntityCategory.PERSON))
        fetch_phrase = mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(return_value={"tom hanks": 101}),
        )
        search_people = mocker.patch(
            "db.lexical_search.search_people_postings",
            new=AsyncMock(return_value={1: 1}),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        # Verify people search was called with the resolved term ID
        search_people.assert_awaited_once()
        assert len(result) == 1
        assert result[0].matched_people_count == 1

    @pytest.mark.asyncio
    async def test_character_entity_include(self, mocker, mock_all_dependencies) -> None:
        """Should categorize CHARACTER entities correctly for INCLUDE."""
        entities = _entities(_entity("Luke Skywalker", EntityCategory.CHARACTER))
        search_chars = mocker.patch(
            "db.lexical_search.search_character_postings",
            new=AsyncMock(return_value={1: 1}),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        search_chars.assert_awaited_once()
        assert len(result) == 1
        assert result[0].matched_character_count == 1

    @pytest.mark.asyncio
    async def test_studio_entity_include(self, mocker, mock_all_dependencies) -> None:
        """Should categorize STUDIO entities correctly for INCLUDE."""
        entities = _entities(_entity("Warner Bros", EntityCategory.STUDIO))
        mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(return_value={"warner bros": 201}),
        )
        search_studios = mocker.patch(
            "db.lexical_search.search_studio_postings",
            new=AsyncMock(return_value={1: 1}),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        search_studios.assert_awaited_once()
        assert len(result) == 1
        assert result[0].matched_studio_count == 1

    @pytest.mark.asyncio
    async def test_movie_title_entity_include(self, mocker, mock_all_dependencies) -> None:
        """Should categorize MOVIE_TITLE entities correctly for INCLUDE."""
        entities = _entities(_entity("The Matrix", EntityCategory.MOVIE_TITLE))
        mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [1], 1: [2]}]),
        )
        search_titles = mocker.patch(
            "db.lexical_search.search_title_postings",
            new=AsyncMock(return_value={1: 0.8}),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        search_titles.assert_awaited_once()
        assert len(result) == 1
        assert result[0].title_score_sum == 0.8

    @pytest.mark.asyncio
    async def test_franchise_entity_include(self, mocker, mock_all_dependencies) -> None:
        """Should categorize FRANCHISE entities correctly for INCLUDE (searches both title and character)."""
        entities = _entities(_entity("Star Wars", EntityCategory.FRANCHISE))
        mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [1], 1: [2]}]),
        )
        search_titles = mocker.patch(
            "db.lexical_search.search_title_postings",
            new=AsyncMock(return_value={1: 0.7}),
        )
        search_chars_by_query = mocker.patch(
            "db.lexical_search.search_character_postings_by_query",
            new=AsyncMock(return_value=[{1: 1}]),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        search_titles.assert_awaited()
        search_chars_by_query.assert_awaited_once()
        assert len(result) == 1
        # Franchise contributes max(title_score, character_score)
        assert result[0].franchise_score_sum > 0


class TestLexicalSearchExcludeHandling:
    """Tests for EXCLUDE entity handling in lexical_search.

    The key invariant: excluded entities from *any* category produce a single
    unified ``excluded_movie_ids`` set that is passed to *every* include
    search, not just the search for the same category.
    """

    @pytest.fixture
    def mock_base_dependencies(self, mocker):
        """Mock base dependencies for exclude tests."""
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split() if x else [])

    @pytest.mark.asyncio
    async def test_exclude_person_propagates_to_all_searches(self, mocker, mock_base_dependencies) -> None:
        """Excluding a person should propagate movie IDs to all include searches."""
        entities = _entities(
            _entity("Tom Hanks", EntityCategory.PERSON),
            _entity("Bad Actor", EntityCategory.PERSON, exclude=True),
            _entity("Good Movie", EntityCategory.MOVIE_TITLE),
        )
        mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(return_value={"tom hanks": 101, "bad actor": 999}),
        )
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [10]}]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch(
            "db.lexical_search.fetch_movie_ids_by_term_ids",
            new=AsyncMock(return_value={50, 51}),
        )
        search_people = mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={1: 1}))
        search_chars = mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        search_studios = mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        search_titles = mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))

        await lexical_search.lexical_search(entities)

        expected_excluded = {50, 51}
        for search_mock, name in [
            (search_people, "people"),
            (search_chars, "characters"),
            (search_studios, "studios"),
            (search_titles, "titles"),
        ]:
            passed = search_mock.await_args.args[2] if len(search_mock.await_args.args) > 2 else search_mock.await_args.kwargs.get("exclude_movie_ids")
            assert passed == expected_excluded, f"{name} search did not receive excluded movie IDs"

    @pytest.mark.asyncio
    async def test_exclude_title_uses_exact_lookup_and_propagates_globally(self, mocker, mock_base_dependencies) -> None:
        """Title exclusions should use exact lookup, then propagate movie IDs to all searches."""
        entities = _entities(
            _entity("Good Movie", EntityCategory.MOVIE_TITLE),
            _entity("Bad Movie", EntityCategory.MOVIE_TITLE, exclude=True),
            _entity("Actor", EntityCategory.PERSON),
        )
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={"actor": 1}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [1], 1: [2]}]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        resolve_exact = mocker.patch(
            "db.lexical_search._resolve_exact_exclude_title_term_ids",
            new=AsyncMock(return_value=[888, 889]),
        )
        mocker.patch(
            "db.lexical_search.fetch_movie_ids_by_term_ids",
            new=AsyncMock(return_value={60, 61}),
        )
        search_people = mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        search_titles = mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))

        await lexical_search.lexical_search(entities)

        resolve_exact.assert_awaited_once()
        expected_excluded = {60, 61}
        people_excluded = search_people.await_args.args[2] if len(search_people.await_args.args) > 2 else search_people.await_args.kwargs.get("exclude_movie_ids")
        title_excluded = search_titles.await_args.args[2] if len(search_titles.await_args.args) > 2 else search_titles.await_args.kwargs.get("exclude_movie_ids")
        assert people_excluded == expected_excluded, "People search should receive title-derived excluded movie IDs"
        assert title_excluded == expected_excluded, "Title search should receive title-derived excluded movie IDs"

    @pytest.mark.asyncio
    async def test_exclude_franchise_affects_all_searches(self, mocker, mock_base_dependencies) -> None:
        """Excluding a franchise should propagate exclusion movie IDs to all include searches."""
        entities = _entities(
            _entity("Star Wars", EntityCategory.FRANCHISE),
            _entity("Bad Franchise", EntityCategory.FRANCHISE, exclude=True),
        )
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [1], 1: [2]}]))
        resolve_char = mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value={0: [777]}),
        )
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[888]))
        mocker.patch(
            "db.lexical_search.fetch_movie_ids_by_term_ids",
            new=AsyncMock(return_value={70, 71}),
        )
        search_people = mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        search_chars = mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[{}]))
        search_titles = mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))

        await lexical_search.lexical_search(entities)

        resolve_char.assert_awaited_once()
        assert "bad franchise" in resolve_char.await_args.args[0]
        expected_excluded = {70, 71}
        for search_mock in [search_people, search_chars, search_titles]:
            passed = search_mock.await_args.args[2] if len(search_mock.await_args.args) > 2 else search_mock.await_args.kwargs.get("exclude_movie_ids")
            assert passed == expected_excluded

    @pytest.mark.asyncio
    async def test_multiple_exclude_categories_union_into_single_set(self, mocker, mock_base_dependencies) -> None:
        """Excluding entities across multiple categories should produce a unioned movie ID set."""
        entities = _entities(
            _entity("Good Actor", EntityCategory.PERSON),
            _entity("Bad Actor", EntityCategory.PERSON, exclude=True),
            _entity("Bad Movie", EntityCategory.MOVIE_TITLE, exclude=True),
        )
        mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(return_value={"good actor": 1, "bad actor": 999}),
        )
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[888]))

        call_count = 0
        async def _fake_resolve(table, term_ids):
            nonlocal call_count
            call_count += 1
            if table.value == "lex.inv_person_postings":
                return {50, 51}
            if table.value == "lex.inv_title_token_postings":
                return {51, 52}
            return set()

        mocker.patch("db.lexical_search.fetch_movie_ids_by_term_ids", side_effect=_fake_resolve)
        search_people = mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))

        await lexical_search.lexical_search(entities)

        assert call_count == 2
        passed = search_people.await_args.args[2] if len(search_people.await_args.args) > 2 else search_people.await_args.kwargs.get("exclude_movie_ids")
        assert passed == {50, 51, 52}

    @pytest.mark.asyncio
    async def test_no_exclusion_resolution_when_no_excludes(self, mocker, mock_base_dependencies) -> None:
        """Should not call fetch_movie_ids_by_term_ids when there are no exclude entities."""
        entities = _entities(
            _entity("Tom Hanks", EntityCategory.PERSON),
        )
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={"tom hanks": 1}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        resolve_movies = mocker.patch(
            "db.lexical_search.fetch_movie_ids_by_term_ids",
            new=AsyncMock(return_value=set()),
        )
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))

        await lexical_search.lexical_search(entities)

        resolve_movies.assert_not_awaited()


class TestLexicalSearchDeduplication:
    """Tests for deduplication logic in lexical_search."""

    @pytest.fixture
    def mock_base_dependencies(self, mocker):
        """Mock base dependencies."""
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))

    @pytest.mark.asyncio
    async def test_deduplicates_people_entities(self, mocker, mock_base_dependencies) -> None:
        """Should deduplicate identical PERSON entities."""
        entities = _entities(
            _entity("Tom Hanks", EntityCategory.PERSON),
            _entity("TOM HANKS", EntityCategory.PERSON),  # Same after normalization
            _entity("Tom Hanks", EntityCategory.PERSON),  # Exact duplicate
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        fetch_phrase = mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(return_value={"tom hanks": 101}),
        )
        
        await lexical_search.lexical_search(entities)
        
        # Should only have one unique person in the lookup
        call_args = fetch_phrase.await_args.args[0]
        assert call_args.count("tom hanks") == 1

    @pytest.mark.asyncio
    async def test_deduplicates_title_searches(self, mocker, mock_base_dependencies) -> None:
        """Should deduplicate identical title searches."""
        entities = _entities(
            _entity("Star Wars", EntityCategory.MOVIE_TITLE),
            _entity("STAR WARS", EntityCategory.MOVIE_TITLE),  # Same after tokenization
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split())
        resolve_titles = mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [1], 1: [2]}]),
        )
        
        await lexical_search.lexical_search(entities)
        
        # Should only resolve one title search
        call_args = resolve_titles.await_args.args[0]
        assert len(call_args) == 1

    @pytest.mark.asyncio
    async def test_deduplicates_franchise_phrases(self, mocker, mock_base_dependencies) -> None:
        """Should deduplicate identical franchise phrases."""
        entities = _entities(
            _entity("Batman", EntityCategory.FRANCHISE),
            _entity("BATMAN", EntityCategory.FRANCHISE),
        )
        normalize = mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        tokenize = mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.split())
        resolve_titles = mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [11]}]),
        )
        
        await lexical_search.lexical_search(entities)
        
        # Tokenize should only be called once for the deduped franchise phrase
        tokenize.assert_called_once_with("batman")
        # Only one franchise title search should be resolved
        resolve_titles.assert_awaited_once_with([["batman"]])


class TestLexicalSearchScoring:
    """Tests for scoring logic in lexical_search."""

    @pytest.fixture
    def mock_all_dependencies(self, mocker):
        """Mock all dependencies with controllable return values."""
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split() if x else [])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))

    @pytest.mark.asyncio
    async def test_max_possible_score_calculation(self, mocker, mock_all_dependencies) -> None:
        """Should calculate max_possible correctly from all include entity types."""
        entities = _entities(
            _entity("Person1", EntityCategory.PERSON),
            _entity("Person2", EntityCategory.PERSON),
            _entity("Character1", EntityCategory.CHARACTER),
            _entity("Studio1", EntityCategory.STUDIO),
            _entity("Title1", EntityCategory.MOVIE_TITLE),
            _entity("Franchise1", EntityCategory.FRANCHISE),
        )
        mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(return_value={"person1": 1, "person2": 2, "studio1": 3}),
        )
        mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [10]}, {0: [20]}]),  # 1 title + 1 franchise title
        )
        mocker.patch(
            "db.lexical_search.search_people_postings",
            new=AsyncMock(return_value={1: 2}),  # Movie 1 matches 2 people
        )
        
        result = await lexical_search.lexical_search(entities)
        
        # max_possible = 2 people + 1 character + 1 studio + 1 title + 1 franchise = 6
        # Movie 1 has raw_score = 2 (people)
        # normalized = 2 / 6 = 0.333...
        assert len(result) == 1
        assert result[0].normalized_lexical_score == pytest.approx(2.0 / 6.0)

    @pytest.mark.asyncio
    async def test_raw_score_aggregation(self, mocker, mock_all_dependencies) -> None:
        """Should aggregate raw scores from all buckets correctly."""
        entities = _entities(
            _entity("Tom Hanks", EntityCategory.PERSON),
            _entity("Star Wars", EntityCategory.MOVIE_TITLE),
            _entity("Batman", EntityCategory.FRANCHISE),
        )
        mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(return_value={"tom hanks": 101}),
        )
        mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [11], 1: [12]}, {0: [21]}]),
        )
        mocker.patch(
            "db.lexical_search.search_people_postings",
            new=AsyncMock(return_value={1: 1}),  # Movie 1: 1 person match
        )
        mocker.patch(
            "db.lexical_search.search_title_postings",
            new=AsyncMock(side_effect=[{1: 0.8}, {1: 0.5, 2: 0.6}]),  # Title and franchise title scores
        )
        mocker.patch(
            "db.lexical_search.search_character_postings_by_query",
            new=AsyncMock(return_value=[{1: 2}]),  # Franchise character match
        )
        
        result = await lexical_search.lexical_search(entities)
        
        # max_possible = 1 person + 1 title + 1 franchise = 3
        # Movie 1: people=1, title=0.8, franchise=max(0.5, 2)=2 → raw=3.8
        # Movie 2: franchise=0.6 → raw=0.6
        assert len(result) == 2
        movie1 = next(c for c in result if c.movie_id == 1)
        movie2 = next(c for c in result if c.movie_id == 2)
        assert movie1.normalized_lexical_score == pytest.approx((1 + 0.8 + 2) / 3.0)
        assert movie2.normalized_lexical_score == pytest.approx(0.6 / 3.0)

    @pytest.mark.asyncio
    async def test_franchise_uses_max_of_title_and_character(self, mocker, mock_all_dependencies) -> None:
        """Should use max(title_score, character_score) for franchise scoring."""
        entities = _entities(_entity("Marvel", EntityCategory.FRANCHISE))
        mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [1]}]),
        )
        # Movie 1: title=0.3, character=0.8 → max=0.8
        # Movie 2: title=0.9, character=0.2 → max=0.9
        # Movie 3: title=0.5, character=0.5 → max=0.5
        mocker.patch(
            "db.lexical_search.search_title_postings",
            new=AsyncMock(return_value={1: 0.3, 2: 0.9, 3: 0.5}),
        )
        mocker.patch(
            "db.lexical_search.search_character_postings_by_query",
            new=AsyncMock(return_value=[{1: 0.8, 2: 0.2, 3: 0.5}]),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        movie1 = next(c for c in result if c.movie_id == 1)
        movie2 = next(c for c in result if c.movie_id == 2)
        movie3 = next(c for c in result if c.movie_id == 3)
        assert movie1.franchise_score_sum == pytest.approx(0.8)
        assert movie2.franchise_score_sum == pytest.approx(0.9)
        assert movie3.franchise_score_sum == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_title_scores_sum_across_multiple_searches(self, mocker, mock_all_dependencies) -> None:
        """Should sum title scores across multiple title searches."""
        entities = _entities(
            _entity("Star Wars", EntityCategory.MOVIE_TITLE),
            _entity("Empire Strikes", EntityCategory.MOVIE_TITLE),
        )
        mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [1], 1: [2]}, {0: [3], 1: [4]}]),
        )
        # Movie 1 matches both title searches
        mocker.patch(
            "db.lexical_search.search_title_postings",
            new=AsyncMock(side_effect=[{1: 0.6}, {1: 0.4}]),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        assert len(result) == 1
        assert result[0].title_score_sum == pytest.approx(1.0)  # 0.6 + 0.4

    @pytest.mark.asyncio
    async def test_results_sorted_by_normalized_score_descending(self, mocker, mock_all_dependencies) -> None:
        """Should sort results by normalized_lexical_score in descending order."""
        entities = _entities(_entity("Actor", EntityCategory.PERSON))
        mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(return_value={"actor": 1}),
        )
        # Return movies with different match counts
        mocker.patch(
            "db.lexical_search.search_people_postings",
            new=AsyncMock(return_value={1: 1, 2: 1, 3: 1}),
        )
        # Add title scores to differentiate
        mocker.patch(
            "db.lexical_search.search_character_postings",
            new=AsyncMock(return_value={1: 0, 2: 2, 3: 1}),
        )
        
        # Need to add a character entity to make character scores count
        entities = _entities(
            _entity("Actor", EntityCategory.PERSON),
            _entity("Hero", EntityCategory.CHARACTER),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        # Verify descending order
        scores = [c.normalized_lexical_score for c in result]
        assert scores == sorted(scores, reverse=True)


class TestLexicalSearchTitleMaxCandidates:
    """Tests for TITLE_MAX_CANDIDATES cap in lexical_search."""

    @pytest.mark.asyncio
    async def test_title_candidates_capped_at_max(self, mocker) -> None:
        """Should cap title candidates at TITLE_MAX_CANDIDATES."""
        entities = _entities(_entity("Common Title", EntityCategory.MOVIE_TITLE))
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split())
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [1]}]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        
        # Return more than TITLE_MAX_CANDIDATES results
        many_results = {i: 0.5 - (i * 0.00001) for i in range(15000)}
        mocker.patch(
            "db.lexical_search.search_title_postings",
            new=AsyncMock(return_value=many_results),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        # Should be capped at TITLE_MAX_CANDIDATES
        assert len(result) <= lexical_search.TITLE_MAX_CANDIDATES

    @pytest.mark.asyncio
    async def test_title_cap_keeps_highest_scores(self, mocker) -> None:
        """Should keep highest scoring title candidates when capping."""
        entities = _entities(_entity("Title", EntityCategory.MOVIE_TITLE))
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split())
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [1]}]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        
        # Create results where movie_id 99999 has highest score
        many_results = {i: 0.1 for i in range(15000)}
        many_results[99999] = 1.0  # Highest score
        mocker.patch(
            "db.lexical_search.search_title_postings",
            new=AsyncMock(return_value=many_results),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        # Movie 99999 should be in results (highest score)
        movie_ids = [c.movie_id for c in result]
        assert 99999 in movie_ids


class TestLexicalSearchMetadataFilters:
    """Tests for metadata filter handling in lexical_search."""

    @pytest.mark.asyncio
    async def test_filters_passed_to_all_search_functions(self, mocker) -> None:
        """Should pass metadata filters to all posting search functions."""
        entities = _entities(
            _entity("Actor", EntityCategory.PERSON),
            _entity("Character", EntityCategory.CHARACTER),
            _entity("Studio", EntityCategory.STUDIO),
            _entity("Title", EntityCategory.MOVIE_TITLE),
            _entity("Franchise", EntityCategory.FRANCHISE),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split())
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={"actor": 1, "studio": 2}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [10]}, {0: [20]}]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        
        search_people = mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        search_chars = mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        search_studios = mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        search_chars_by_query = mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[{}]))
        search_titles = mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))
        
        filters = MetadataFilters(min_runtime=90, max_runtime=180, genres=[Genre.ACTION])
        
        await lexical_search.lexical_search(entities, filters=filters)
        
        # Verify filters passed to all search functions
        assert search_people.await_args.args[1] == filters
        assert search_chars.await_args.args[1] == filters
        assert search_studios.await_args.args[1] == filters
        assert search_chars_by_query.await_args.args[1] == filters
        # Title search is called twice (regular + franchise)
        for call in search_titles.await_args_list:
            assert call.args[1] == filters


class TestLexicalSearchCorrectedEntity:
    """Tests for corrected_and_normalized_entity handling."""

    @pytest.mark.asyncio
    async def test_uses_corrected_entity_when_available(self, mocker) -> None:
        """Should use corrected_and_normalized_entity over candidate_entity_phrase."""
        entities = _entities(
            _entity("Tome Hanks", EntityCategory.PERSON, corrected="Tom Hanks"),  # Typo corrected
        )
        normalize = mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={"tom hanks": 1}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        
        await lexical_search.lexical_search(entities)
        
        # Should normalize "Tom Hanks" (corrected), not "Tome Hanks" (original)
        normalize.assert_any_call("Tom Hanks")


class TestLexicalSearchCandidateConstruction:
    """Tests for LexicalCandidate construction in lexical_search."""

    @pytest.mark.asyncio
    async def test_candidate_fields_populated_correctly(self, mocker) -> None:
        """Should populate all LexicalCandidate fields correctly."""
        entities = _entities(
            _entity("Actor", EntityCategory.PERSON),
            _entity("Character", EntityCategory.CHARACTER),
            _entity("Studio", EntityCategory.STUDIO),
            _entity("Title", EntityCategory.MOVIE_TITLE),
            _entity("Franchise", EntityCategory.FRANCHISE),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split())
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={"actor": 1, "studio": 2}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [10]}, {0: [20]}]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        
        # All searches return movie_id=1 with various scores
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={1: 2}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(side_effect=[{1: 0.7}, {1: 0.3}]))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[{1: 1}]))
        
        result = await lexical_search.lexical_search(entities)
        
        assert len(result) == 1
        candidate = result[0]
        assert candidate.movie_id == 1
        assert candidate.matched_people_count == 1
        assert candidate.matched_character_count == 2
        assert candidate.matched_studio_count == 1
        assert candidate.title_score_sum == pytest.approx(0.7)
        # Franchise: max(title=0.3, character=1) = 1
        assert candidate.franchise_score_sum == pytest.approx(1.0)
        # raw = 1 + 2 + 1 + 0.7 + 1 = 5.7, max_possible = 5
        assert candidate.normalized_lexical_score == pytest.approx(5.7 / 5.0)

    @pytest.mark.asyncio
    async def test_candidate_with_zero_scores_in_some_buckets(self, mocker) -> None:
        """Should handle candidates that only match some buckets."""
        entities = _entities(
            _entity("Actor", EntityCategory.PERSON),
            _entity("Character", EntityCategory.CHARACTER),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={"actor": 1}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        
        # Movie 1 only matches people, Movie 2 only matches characters
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={2: 1}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        
        result = await lexical_search.lexical_search(entities)
        
        assert len(result) == 2
        movie1 = next(c for c in result if c.movie_id == 1)
        movie2 = next(c for c in result if c.movie_id == 2)
        
        assert movie1.matched_people_count == 1
        assert movie1.matched_character_count == 0
        assert movie2.matched_people_count == 0
        assert movie2.matched_character_count == 1


class TestLexicalSearchIntegration:
    """Integration-style tests for lexical_search covering complex scenarios."""

    @pytest.mark.asyncio
    async def test_full_orchestration_with_all_entity_types(self, mocker) -> None:
        """Should correctly orchestrate search across all entity types."""
        entities = _entities(
            _entity("Tom Hanks", EntityCategory.PERSON),
            _entity("Star Wars", EntityCategory.MOVIE_TITLE),
            _entity("Batman", EntityCategory.FRANCHISE),
            _entity("Bad Guy", EntityCategory.CHARACTER, exclude=True),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split())
        mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(return_value={"tom hanks": 101}),
        )
        mocker.patch(
            "db.lexical_search._resolve_exact_exclude_title_term_ids",
            new=AsyncMock(return_value=[]),
        )
        mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [11], 1: [12]}, {0: [21]}]),
        )
        mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value={0: [900]}),
        )
        mocker.patch(
            "db.lexical_search.search_people_postings",
            new=AsyncMock(return_value={1: 1}),
        )
        search_characters = mocker.patch(
            "db.lexical_search.search_character_postings",
            new=AsyncMock(return_value={}),
        )
        search_characters_by_query = mocker.patch(
            "db.lexical_search.search_character_postings_by_query",
            new=AsyncMock(return_value=[{1: 2}]),
        )
        mocker.patch(
            "db.lexical_search.search_studio_postings",
            new=AsyncMock(return_value={}),
        )
        search_titles = mocker.patch(
            "db.lexical_search.search_title_postings",
            new=AsyncMock(side_effect=[{1: 0.8}, {1: 0.5, 2: 0.6}]),
        )

        result = await lexical_search.lexical_search(entities)

        assert len(result) == 2
        assert result[0].movie_id == 1
        assert result[0].normalized_lexical_score == pytest.approx((1 + 0.8 + 2) / 3.0)
        assert result[1].movie_id == 2
        assert result[1].normalized_lexical_score == pytest.approx(0.6 / 3.0)
        assert search_titles.await_count == 2
        search_characters.assert_awaited_once()
        search_characters_by_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_franchise_title_searches_derived_from_deduped_phrases(self, mocker) -> None:
        """Franchise title searches should be derived once from deduped normalized phrases."""
        entities = _entities(
            _entity("Batman", EntityCategory.FRANCHISE),
            _entity("BATMAN", EntityCategory.FRANCHISE),
        )
        normalize = mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        tokenize = mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.split())
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        resolve_titles = mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [11]}]),
        )
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[{}]))
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))

        await lexical_search.lexical_search(entities)

        assert normalize.call_count == 2
        tokenize.assert_called_once_with("batman")
        resolve_titles.assert_awaited_once_with([["batman"]])

    @pytest.mark.asyncio
    async def test_multiple_franchises_each_contribute_separately(self, mocker) -> None:
        """Multiple franchises should each contribute to max_possible and scoring."""
        entities = _entities(
            _entity("Marvel", EntityCategory.FRANCHISE),
            _entity("DC", EntityCategory.FRANCHISE),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: [x])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [1]}, {0: [2]}]),  # Two franchise title searches
        )
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        # Movie 1 matches both franchises via character
        mocker.patch(
            "db.lexical_search.search_character_postings_by_query",
            new=AsyncMock(return_value=[{1: 1}, {1: 1}]),
        )
        mocker.patch(
            "db.lexical_search.search_title_postings",
            new=AsyncMock(side_effect=[{1: 0.5}, {1: 0.6}]),
        )
        
        result = await lexical_search.lexical_search(entities)
        
        # max_possible = 2 franchises
        # Movie 1: franchise1=max(0.5, 1)=1, franchise2=max(0.6, 1)=1 → raw=2
        assert len(result) == 1
        assert result[0].franchise_score_sum == pytest.approx(2.0)
        assert result[0].normalized_lexical_score == pytest.approx(1.0)


class TestLexicalSearchEdgeCases:
    """Edge case tests for lexical_search."""

    @pytest.mark.asyncio
    async def test_entity_with_empty_corrected_uses_candidate_phrase(self, mocker) -> None:
        """Should fall back to candidate_entity_phrase when corrected is empty string."""
        entity = ExtractedEntityData(
            candidate_entity_phrase="Original Phrase",
            most_likely_category=EntityCategory.PERSON.value,
            exclude_from_results=False,
            corrected_and_normalized_entity="",  # Empty string triggers fallback
        )
        entities = ExtractedEntitiesResponse(entity_candidates=[entity])
        
        normalize = mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower() if x else "")
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        
        await lexical_search.lexical_search(entities)
        
        # Should use candidate_entity_phrase since corrected is empty
        normalize.assert_any_call("Original Phrase")

    @pytest.mark.asyncio
    async def test_empty_string_after_normalization_skipped(self, mocker) -> None:
        """Should skip entities that normalize to empty strings."""
        entities = _entities(
            _entity("...", EntityCategory.PERSON),  # Normalizes to empty
            _entity("Valid Person", EntityCategory.PERSON),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: "" if x == "..." else x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        fetch_phrase = mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(return_value={"valid person": 1}),
        )
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        
        await lexical_search.lexical_search(entities)
        
        # Only "valid person" should be in the phrase lookup
        call_args = fetch_phrase.await_args.args[0]
        assert "valid person" in call_args
        assert "" not in call_args

    @pytest.mark.asyncio
    async def test_title_with_empty_tokens_after_tokenization_skipped(self, mocker) -> None:
        """Should skip title searches that produce empty token lists."""
        entities = _entities(
            _entity("...", EntityCategory.MOVIE_TITLE),  # Tokenizes to empty
            _entity("Valid Title", EntityCategory.MOVIE_TITLE),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: [] if x == "..." else x.lower().split())
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        resolve_titles = mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [1], 1: [2]}]),
        )
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))
        
        await lexical_search.lexical_search(entities)
        
        # Only one title search should be resolved (the valid one)
        resolve_titles.assert_awaited_once()
        call_args = resolve_titles.await_args.args[0]
        assert len(call_args) == 1
        assert call_args[0] == ["valid", "title"]

    @pytest.mark.asyncio
    async def test_franchise_with_empty_normalized_phrase_skipped(self, mocker) -> None:
        """Should skip franchises that normalize to empty strings."""
        entities = _entities(
            _entity("...", EntityCategory.FRANCHISE),
            _entity("Valid Franchise", EntityCategory.FRANCHISE),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: "" if x == "..." else x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.split() if x else [])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        resolve_titles = mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [1], 1: [2]}]),
        )
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[{}]))
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))
        
        await lexical_search.lexical_search(entities)
        
        # Only one franchise title search should be resolved
        resolve_titles.assert_awaited_once()
        call_args = resolve_titles.await_args.args[0]
        assert len(call_args) == 1

    @pytest.mark.asyncio
    async def test_movie_with_zero_franchise_score_not_included(self, mocker) -> None:
        """Movies with zero franchise score should not contribute to franchise_score_sums."""
        entities = _entities(_entity("Franchise", EntityCategory.FRANCHISE))
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: [x])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [1]}]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        # Movie 1 has title=0, character=0 → best_score=0 → not added
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[{1: 0}]))
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={1: 0.0}))
        
        result = await lexical_search.lexical_search(entities)
        
        # Movie should not appear since both scores are 0
        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_handles_very_large_number_of_entities(self, mocker) -> None:
        """Should handle a large number of entities without issues."""
        # Create 100 person entities
        many_entities = [_entity(f"Person{i}", EntityCategory.PERSON) for i in range(100)]
        entities = ExtractedEntitiesResponse(entity_candidates=many_entities)
        
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        # Return term IDs for all 100 people
        term_map = {f"person{i}": i for i in range(100)}
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value=term_map))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        # Movie 1 matches all 100 people
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={1: 100}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        
        result = await lexical_search.lexical_search(entities)
        
        assert len(result) == 1
        assert result[0].matched_people_count == 100
        assert result[0].normalized_lexical_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_handles_unicode_entities(self, mocker) -> None:
        """Should handle unicode characters in entity names."""
        entities = _entities(
            _entity("Amélie Poulain", EntityCategory.CHARACTER),
            _entity("François Truffaut", EntityCategory.PERSON),
            _entity("日本映画", EntityCategory.STUDIO),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={
            "amélie poulain": 1,
            "françois truffaut": 2,
            "日本映画": 3,
        }))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        
        result = await lexical_search.lexical_search(entities)
        
        assert len(result) == 1
        assert result[0].matched_people_count == 1
        assert result[0].matched_character_count == 1
        assert result[0].matched_studio_count == 1


class TestLexicalSearchConstants:
    """Tests verifying the constants used in lexical_search."""

    def test_max_df_constant(self) -> None:
        """MAX_DF should be 10,000 as per the guide."""
        assert lexical_search.MAX_DF == 10_000

    def test_title_score_beta_constant(self) -> None:
        """TITLE_SCORE_BETA should be 2.0 as per the guide."""
        assert lexical_search.TITLE_SCORE_BETA == 2.0

    def test_title_score_threshold_constant(self) -> None:
        """TITLE_SCORE_THRESHOLD should be 0.15 as per the guide."""
        assert lexical_search.TITLE_SCORE_THRESHOLD == 0.15

    def test_title_max_candidates_constant(self) -> None:
        """TITLE_MAX_CANDIDATES should be 10,000 as per the guide."""
        assert lexical_search.TITLE_MAX_CANDIDATES == 10_000


class TestMetadataFiltersIsActive:
    """Tests for MetadataFilters.is_active property."""

    def test_is_active_when_all_none(self) -> None:
        """Should return False when all fields are None."""
        filters = MetadataFilters()
        assert filters.is_active is False

    def test_is_active_when_min_release_ts_set(self) -> None:
        """Should return True when min_release_ts is set."""
        filters = MetadataFilters(min_release_ts=1000000)
        assert filters.is_active is True

    def test_is_active_when_max_release_ts_set(self) -> None:
        """Should return True when max_release_ts is set."""
        filters = MetadataFilters(max_release_ts=2000000)
        assert filters.is_active is True

    def test_is_active_when_min_runtime_set(self) -> None:
        """Should return True when min_runtime is set."""
        filters = MetadataFilters(min_runtime=90)
        assert filters.is_active is True

    def test_is_active_when_max_runtime_set(self) -> None:
        """Should return True when max_runtime is set."""
        filters = MetadataFilters(max_runtime=180)
        assert filters.is_active is True

    def test_is_active_when_min_maturity_rank_set(self) -> None:
        """Should return True when min_maturity_rank is set."""
        filters = MetadataFilters(min_maturity_rank=1)
        assert filters.is_active is True

    def test_is_active_when_max_maturity_rank_set(self) -> None:
        """Should return True when max_maturity_rank is set."""
        filters = MetadataFilters(max_maturity_rank=4)
        assert filters.is_active is True

    def test_is_active_when_genres_set(self) -> None:
        """Should return True when genres is set."""
        filters = MetadataFilters(genres=[Genre.ACTION, Genre.ADVENTURE])
        assert filters.is_active is True

    def test_is_active_when_watch_offer_keys_set(self) -> None:
        """Should return True when watch_offer_keys is set."""
        filters = MetadataFilters(watch_offer_keys=[101, 102])
        assert filters.is_active is True

    def test_is_active_when_multiple_fields_set(self) -> None:
        """Should return True when multiple fields are set."""
        filters = MetadataFilters(
            min_runtime=90,
            max_runtime=180,
            genres=[Genre.ACTION],
        )
        assert filters.is_active is True


class TestLexicalSearchConcurrency:
    """Tests for concurrent execution patterns in lexical_search."""

    @pytest.mark.asyncio
    async def test_term_resolution_runs_concurrently(self, mocker) -> None:
        """Should resolve phrase IDs, title tokens, character terms, and exclude titles concurrently."""
        entities = _entities(
            _entity("Person", EntityCategory.PERSON),
            _entity("Title", EntityCategory.MOVIE_TITLE),
            _entity("Franchise", EntityCategory.FRANCHISE),
            _entity("Bad Title", EntityCategory.MOVIE_TITLE, exclude=True),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split())
        
        # Track call order
        call_order = []
        
        async def track_phrase(*args):
            call_order.append("phrase")
            return {"person": 1}
        
        async def track_titles(*args):
            call_order.append("titles")
            return [{0: [1]}, {0: [2]}]
        
        async def track_chars(*args):
            call_order.append("chars")
            return {}
        
        async def track_exclude(*args):
            call_order.append("exclude")
            return []
        
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(side_effect=track_phrase))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(side_effect=track_titles))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(side_effect=track_chars))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(side_effect=track_exclude))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[{}]))
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))
        
        await lexical_search.lexical_search(entities)
        
        # All four resolution functions should have been called
        assert "phrase" in call_order
        assert "titles" in call_order
        assert "chars" in call_order
        assert "exclude" in call_order

    @pytest.mark.asyncio
    async def test_posting_searches_run_concurrently(self, mocker) -> None:
        """Should run all posting searches concurrently via asyncio.gather."""
        entities = _entities(
            _entity("Person", EntityCategory.PERSON),
            _entity("Character", EntityCategory.CHARACTER),
            _entity("Studio", EntityCategory.STUDIO),
            _entity("Title", EntityCategory.MOVIE_TITLE),
            _entity("Franchise", EntityCategory.FRANCHISE),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split())
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={"person": 1, "studio": 2}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [10]}, {0: [20]}]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        
        search_people = mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        search_chars = mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        search_studios = mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        search_chars_by_query = mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[{}]))
        search_titles = mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))
        
        await lexical_search.lexical_search(entities)
        
        # All search functions should have been called
        search_people.assert_awaited_once()
        search_chars.assert_awaited_once()
        search_studios.assert_awaited_once()
        search_chars_by_query.assert_awaited_once()
        # Title search called twice: once for regular title, once for franchise title
        assert search_titles.await_count == 2


class TestLexicalSearchRegressions:
    """Regression tests for specific bugs or edge cases discovered."""

    @pytest.mark.asyncio
    async def test_franchise_only_in_character_space(self, mocker) -> None:
        """Franchise that only matches in character space should still contribute."""
        entities = _entities(_entity("Marvel", EntityCategory.FRANCHISE))
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: [x])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [1]}]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        # Title search returns nothing, but character search returns a match
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[{1: 1}]))
        
        result = await lexical_search.lexical_search(entities)
        
        assert len(result) == 1
        assert result[0].franchise_score_sum == 1.0

    @pytest.mark.asyncio
    async def test_franchise_only_in_title_space(self, mocker) -> None:
        """Franchise that only matches in title space should still contribute."""
        entities = _entities(_entity("Marvel", EntityCategory.FRANCHISE))
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: [x])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[{0: [1]}]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        # Title search returns a match, character search returns nothing
        mocker.patch("db.lexical_search.search_title_postings", new=AsyncMock(return_value={1: 0.8}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[{}]))
        
        result = await lexical_search.lexical_search(entities)
        
        assert len(result) == 1
        assert result[0].franchise_score_sum == pytest.approx(0.8)

    @pytest.mark.asyncio
    async def test_same_movie_from_multiple_buckets(self, mocker) -> None:
        """Movie appearing in multiple buckets should aggregate scores correctly."""
        entities = _entities(
            _entity("Actor", EntityCategory.PERSON),
            _entity("Character", EntityCategory.CHARACTER),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={"actor": 1}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search._resolve_character_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        # Same movie matches both people and characters
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        
        result = await lexical_search.lexical_search(entities)
        
        assert len(result) == 1
        assert result[0].matched_people_count == 1
        assert result[0].matched_character_count == 1
        # raw = 1 + 1 = 2, max = 2
        assert result[0].normalized_lexical_score == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_exclude_character_combined_with_exclude_franchise(self, mocker) -> None:
        """Exclude characters and exclude franchises should be combined for character term resolution."""
        entities = _entities(
            _entity("Hero", EntityCategory.CHARACTER),
            _entity("Villain", EntityCategory.CHARACTER, exclude=True),
            _entity("Bad Franchise", EntityCategory.FRANCHISE, exclude=True),
        )
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda x: x.lower())
        mocker.patch("db.lexical_search.tokenize_title_phrase", side_effect=lambda x: x.lower().split())
        mocker.patch("db.lexical_search.fetch_phrase_term_ids", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search._resolve_all_title_tokens", new=AsyncMock(return_value=[]))
        resolve_chars = mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value={0: [100], 1: [200]}),
        )
        mocker.patch("db.lexical_search._resolve_exact_exclude_title_term_ids", new=AsyncMock(return_value=[]))
        mocker.patch("db.lexical_search.search_people_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings", new=AsyncMock(return_value={1: 1}))
        mocker.patch("db.lexical_search.search_studio_postings", new=AsyncMock(return_value={}))
        mocker.patch("db.lexical_search.search_character_postings_by_query", new=AsyncMock(return_value=[]))
        
        await lexical_search.lexical_search(entities)
        
        # Should resolve both exclude character and exclude franchise phrases
        resolve_chars.assert_awaited_once()
        call_args = resolve_chars.await_args.args[0]
        assert "villain" in call_args
        assert "bad franchise" in call_args
