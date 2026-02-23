"""Unit tests for db.lexical_search."""

from unittest.mock import AsyncMock

import pytest

from db import lexical_search
from db.postgres import CompoundLexicalResult
from implementation.classes.enums import EntityCategory
from implementation.classes.schemas import (
    ExtractedEntitiesResponse,
    ExtractedEntityData,
)


def _entity(
    phrase: str,
    category: EntityCategory,
    *,
    exclude: bool = False,
    corrected: str | None = None,
) -> ExtractedEntityData:
    """Build one ExtractedEntityData for lexical_search tests."""
    return ExtractedEntityData(
        candidate_entity_phrase=phrase,
        most_likely_category=category.value,
        exclude_from_results=exclude,
        corrected_and_normalized_entity=corrected if corrected is not None else phrase,
    )


def _entities(*entity_list: ExtractedEntityData) -> ExtractedEntitiesResponse:
    """Build an ExtractedEntitiesResponse from entities."""
    return ExtractedEntitiesResponse(entity_candidates=list(entity_list))


@pytest.fixture(autouse=True)
def _mock_excluded_movie_id_resolution(mocker):
    """Default exclusion-term to movie-id resolution to empty sets."""
    mocker.patch(
        "db.lexical_search.fetch_movie_ids_by_term_ids",
        new=AsyncMock(return_value=set()),
    )


class TestHelpers:
    """Tests for compound-query input helper functions."""

    def test_dedupe_preserve_order(self) -> None:
        """Should keep the first-seen order while removing duplicates."""
        assert lexical_search._dedupe_preserve_order(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_resolve_all_title_tokens_deduplicates_tokens(self, mocker) -> None:
        """Should resolve shared tokens once and remap to each title search."""
        fetch = mocker.patch(
            "db.lexical_search.fetch_title_token_ids",
            new=AsyncMock(return_value={0: [11], 1: [22]}),
        )
        result = await lexical_search._resolve_all_title_tokens([["star", "wars"], ["star"]])
        fetch.assert_awaited_once_with(tokens=["star", "wars"], max_df=lexical_search.MAX_DF)
        assert result == [{0: [11], 1: [22]}, {0: [11]}]

    def test_flatten_character_term_map_with_offset(self) -> None:
        """Should flatten term-map to query and term arrays with offset support."""
        query_idxs, term_ids = lexical_search._flatten_character_term_map(
            {0: [10, 11], 2: [20]},
            query_idx_offset=3,
        )
        assert query_idxs == [3, 3, 5]
        assert term_ids == [10, 11, 20]

    def test_build_title_search_input_uses_expected_constants(self) -> None:
        """Should build TitleSearchInput with expected F-score parameters."""
        payload = lexical_search._build_title_search_input({0: [100], 1: [200, 201]})
        assert payload.token_idxs == [0, 1, 1]
        assert payload.term_ids == [100, 200, 201]
        assert payload.f_coeff == 5.0
        assert payload.beta_sq == 4.0
        assert payload.k == 2
        assert payload.score_threshold == lexical_search.TITLE_SCORE_THRESHOLD
        assert payload.max_candidates == lexical_search.TITLE_MAX_CANDIDATES


class TestLexicalSearch:
    """Tests for the lexical_search orchestrator with compound query execution."""

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_include_entities(self, mocker) -> None:
        """Should return an empty candidate list when include set is empty."""
        mocker.patch("db.lexical_search.normalize_string", return_value="")
        mocker.patch("db.lexical_search.tokenize_title_phrase", return_value=[])
        result = await lexical_search.lexical_search(_entities(_entity("...", EntityCategory.PERSON)))
        assert result == []

    @pytest.mark.asyncio
    async def test_step6_uses_single_compound_query(self, mocker) -> None:
        """Should execute one compound query and preserve scoring semantics."""
        mocker.patch("db.lexical_search.normalize_string", side_effect=lambda value: value.lower())
        mocker.patch(
            "db.lexical_search.tokenize_title_phrase",
            side_effect=lambda value: value.lower().split() if value else [],
        )
        mocker.patch(
            "db.lexical_search.fetch_phrase_term_ids",
            new=AsyncMock(
                return_value={
                    "tom hanks": 1,
                    "pixar": 2,
                }
            ),
        )
        mocker.patch(
            "db.lexical_search._resolve_all_title_tokens",
            new=AsyncMock(return_value=[{0: [101]}, {0: [201]}]),
        )
        mocker.patch(
            "db.lexical_search._resolve_exact_exclude_title_term_ids",
            new=AsyncMock(return_value=[]),
        )
        mock_resolve_character = mocker.patch(
            "db.lexical_search._resolve_character_term_ids",
            new=AsyncMock(return_value={0: [301], 1: [401]}),
        )
        mock_compound = mocker.patch(
            "db.lexical_search.execute_compound_lexical_search",
            new=AsyncMock(
                return_value=CompoundLexicalResult(
                    people_scores={7: 1},
                    studio_scores={7: 1},
                    character_by_query={0: {7: 1}, 1: {7: 1}},
                    title_scores_by_search={0: {7: 0.6}, 1: {7: 0.8}},
                )
            ),
        )

        entities = _entities(
            _entity("Tom Hanks", EntityCategory.PERSON),
            _entity("Pixar", EntityCategory.STUDIO),
            _entity("Batman", EntityCategory.CHARACTER),
            _entity("Toy Story", EntityCategory.MOVIE_TITLE),
            _entity("Marvel", EntityCategory.FRANCHISE),
        )
        result = await lexical_search.lexical_search(entities)

        assert mock_compound.await_count == 1
        assert mock_resolve_character.await_count == 1
        mock_resolve_character.assert_awaited_once_with(["batman", "marvel"])
        kwargs = mock_compound.await_args.kwargs
        assert kwargs["people_term_ids"] == [1]
        assert kwargs["studio_term_ids"] == [2]
        assert kwargs["character_query_idxs"] == [0, 1]
        assert kwargs["character_term_ids"] == [301, 401]
        assert len(kwargs["title_searches"]) == 2

        assert len(result) == 1
        candidate = result[0]
        assert candidate.movie_id == 7
        assert candidate.matched_people_count == 1
        assert candidate.matched_character_count == 1
        assert candidate.matched_studio_count == 1
        assert candidate.title_score_sum == 0.6
        assert candidate.franchise_score_sum == 1.0
        assert candidate.normalized_lexical_score == pytest.approx((1.0 + 1.0 + 1.0 + 0.6 + 1.0) / 5.0)

