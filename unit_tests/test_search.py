"""Unit tests for db.search (unified search orchestrator)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from db.search import search, SearchCandidate, SearchDebug, SearchResult
from db.lexical_search import LexicalSearchDebug, LexicalSearchResult
from db.vector_search import VectorSearchResult, VectorSearchDebug, CandidateVectorScores
from db.vector_scoring import VectorScoringResult
from implementation.classes.schemas import (
    MetadataFilters,
    ExtractedEntitiesResponse,
    LexicalCandidate,
    VectorWeights,
    VectorSubqueries,
)
from implementation.classes.enums import RelevanceSize, VectorName


DUMMY_QUERY = "find me a fun movie"
DUMMY_FILTERS = MetadataFilters()


def _lexical_debug() -> LexicalSearchDebug:
    return LexicalSearchDebug(
        latency_ms=50.0,
        llm_generation_time_ms=30.0,
        extracted_entities=ExtractedEntitiesResponse(entity_candidates=[]),
        candidates_returned=0,
    )


def _vector_debug() -> VectorSearchDebug:
    return VectorSearchDebug(
        total_jobs_executed=1,
        total_candidates=0,
        per_job_stats=[],
        wall_clock_ms=100.0,
        errors=[],
    )


def _vector_weights() -> VectorWeights:
    return VectorWeights(
        plot_events_weight=RelevanceSize.NOT_RELEVANT,
        plot_analysis_weight=RelevanceSize.NOT_RELEVANT,
        viewer_experience_weight=RelevanceSize.NOT_RELEVANT,
        watch_context_weight=RelevanceSize.NOT_RELEVANT,
        narrative_techniques_weight=RelevanceSize.NOT_RELEVANT,
        production_weight=RelevanceSize.NOT_RELEVANT,
        reception_weight=RelevanceSize.NOT_RELEVANT,
    )


def _vector_subqueries() -> VectorSubqueries:
    return VectorSubqueries(
        plot_events_subquery=None,
        plot_analysis_subquery=None,
        viewer_experience_subquery=None,
        watch_context_subquery=None,
        narrative_techniques_subquery=None,
        production_subquery=None,
        reception_subquery=None,
    )


def _lexical_candidate(movie_id: int, score: float) -> LexicalCandidate:
    return LexicalCandidate(
        movie_id=movie_id,
        normalized_lexical_score=score,
    )


@pytest.mark.asyncio
async def test_search_calls_both_with_correct_args():
    """Both lexical_search and run_vector_search are called with the right args."""
    lexical_res = LexicalSearchResult(candidates=[], debug=_lexical_debug())
    vector_res = VectorSearchResult(
        candidates={},
        vector_weights=_vector_weights(),
        vector_subqueries=_vector_subqueries(),
        debug=_vector_debug(),
    )
    scoring_res = VectorScoringResult(
        final_scores={},
        space_contexts=[],
        per_space_normalized={},
    )

    mock_client = AsyncMock()

    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)) as mock_lex,
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)) as mock_vec,
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
    ):
        result = await search(DUMMY_QUERY, DUMMY_FILTERS, mock_client)

        mock_lex.assert_awaited_once_with(DUMMY_QUERY, DUMMY_FILTERS)
        mock_vec.assert_awaited_once_with(
            query=DUMMY_QUERY,
            metadata_filters=DUMMY_FILTERS,
            qdrant_client=mock_client,
            candidate_limit_original=2000,
            candidate_limit_subquery=2000,
            candidate_limit_anchor=2000,
        )

    assert isinstance(result, SearchResult)
    assert result.candidates == []


@pytest.mark.asyncio
async def test_search_merges_overlapping_candidates():
    """Movie IDs present in both lexical and vector results get both scores."""
    lexical_res = LexicalSearchResult(
        candidates=[
            _lexical_candidate(1, 0.9),
            _lexical_candidate(2, 0.5),
        ],
        debug=_lexical_debug(),
    )
    vector_res = VectorSearchResult(
        candidates={},
        vector_weights=_vector_weights(),
        vector_subqueries=_vector_subqueries(),
        debug=_vector_debug(),
    )
    # movie 1 appears in both, movie 3 only in vector
    scoring_res = VectorScoringResult(
        final_scores={1: 0.8, 3: 0.6},
        space_contexts=[],
        per_space_normalized={},
    )

    mock_client = AsyncMock()

    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)),
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)),
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
    ):
        result = await search(DUMMY_QUERY, DUMMY_FILTERS, mock_client)

    by_id = {c.movie_id: c for c in result.candidates}
    assert len(by_id) == 3

    # movie 1: both sources
    assert by_id[1].vector_score == 0.8
    assert by_id[1].lexical_score == 0.9

    # movie 2: lexical only
    assert by_id[2].vector_score == 0.0
    assert by_id[2].lexical_score == 0.5

    # movie 3: vector only
    assert by_id[3].vector_score == 0.6
    assert by_id[3].lexical_score == 0.0


@pytest.mark.asyncio
async def test_search_non_overlapping_candidates():
    """When lexical and vector return disjoint movie sets, all are present."""
    lexical_res = LexicalSearchResult(
        candidates=[_lexical_candidate(10, 0.7)],
        debug=_lexical_debug(),
    )
    vector_res = VectorSearchResult(
        candidates={},
        vector_weights=_vector_weights(),
        vector_subqueries=_vector_subqueries(),
        debug=_vector_debug(),
    )
    scoring_res = VectorScoringResult(
        final_scores={20: 0.4},
        space_contexts=[],
        per_space_normalized={},
    )

    mock_client = AsyncMock()

    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)),
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)),
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
    ):
        result = await search(DUMMY_QUERY, DUMMY_FILTERS, mock_client)

    by_id = {c.movie_id: c for c in result.candidates}
    assert len(by_id) == 2
    assert by_id[10].lexical_score == 0.7
    assert by_id[10].vector_score == 0.0
    assert by_id[20].vector_score == 0.4
    assert by_id[20].lexical_score == 0.0


@pytest.mark.asyncio
async def test_search_debug_fields_populated():
    """Debug fields are populated correctly."""
    lex_debug = _lexical_debug()
    vec_debug = _vector_debug()

    lexical_res = LexicalSearchResult(candidates=[], debug=lex_debug)
    vector_res = VectorSearchResult(
        candidates={},
        vector_weights=_vector_weights(),
        vector_subqueries=_vector_subqueries(),
        debug=vec_debug,
    )
    scoring_res = VectorScoringResult(
        final_scores={},
        space_contexts=[],
        per_space_normalized={},
    )

    mock_client = AsyncMock()

    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)),
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)),
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
    ):
        result = await search(DUMMY_QUERY, DUMMY_FILTERS, mock_client)

    assert result.debug.lexical_debug is lex_debug
    assert result.debug.vector_debug is vec_debug
    assert result.debug.total_candidates == 0
    assert result.debug.total_latency_ms > 0


@pytest.mark.asyncio
async def test_search_empty_results():
    """Both searches returning empty results produces an empty candidate list."""
    lexical_res = LexicalSearchResult(candidates=[], debug=_lexical_debug())
    vector_res = VectorSearchResult(
        candidates={},
        vector_weights=_vector_weights(),
        vector_subqueries=_vector_subqueries(),
        debug=_vector_debug(),
    )
    scoring_res = VectorScoringResult(
        final_scores={},
        space_contexts=[],
        per_space_normalized={},
    )

    mock_client = AsyncMock()

    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)),
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)),
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
    ):
        result = await search(DUMMY_QUERY, DUMMY_FILTERS, mock_client)

    assert result.candidates == []
    assert result.debug.total_candidates == 0
