"""Unit tests for db.search (unified search orchestrator)."""

import importlib
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

try:
    importlib.import_module("qdrant_client")
except ModuleNotFoundError:
    qdrant_module = ModuleType("qdrant_client")
    qdrant_models_module = ModuleType("qdrant_client.models")

    class _StubAsyncQdrantClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

    class _StubQdrantModel:
        def __init__(self, *args, **kwargs) -> None:
            pass

    qdrant_module.AsyncQdrantClient = _StubAsyncQdrantClient
    qdrant_models_module.Filter = _StubQdrantModel
    qdrant_models_module.FieldCondition = _StubQdrantModel
    qdrant_models_module.MatchAny = _StubQdrantModel
    qdrant_models_module.Range = _StubQdrantModel
    qdrant_models_module.SearchParams = _StubQdrantModel
    qdrant_models_module.QuantizationSearchParams = _StubQdrantModel
    sys.modules["qdrant_client"] = qdrant_module
    sys.modules["qdrant_client.models"] = qdrant_models_module

try:
    importlib.import_module("implementation.llms.generic_methods")
except ModuleNotFoundError:
    generic_methods_module = ModuleType("implementation.llms.generic_methods")

    async def _stub_generate_vector_embedding(*args, **kwargs):
        return []

    generic_methods_module.generate_vector_embedding = _stub_generate_vector_embedding
    sys.modules["implementation.llms.generic_methods"] = generic_methods_module

try:
    importlib.import_module("implementation.llms.query_understanding_methods")
except ImportError:
    query_methods_module = ModuleType("implementation.llms.query_understanding_methods")

    async def _stub_async_method(*args, **kwargs):
        return None

    query_methods_module.create_single_vector_weight_async = _stub_async_method
    query_methods_module.create_single_vector_subquery_async = _stub_async_method
    query_methods_module.create_channel_weights_async = _stub_async_method
    query_methods_module.extract_all_metadata_preferences_async = _stub_async_method
    query_methods_module.extract_lexical_entities_async = _stub_async_method
    sys.modules["implementation.llms.query_understanding_methods"] = query_methods_module

from db.search import search, SearchCandidate, SearchDebug, SearchResult
from db.lexical_search import LexicalSearchDebug, LexicalSearchResult
from db.vector_search import VectorSearchResult, VectorSearchDebug, CandidateVectorScores
from db.vector_scoring import VectorScoringResult
from implementation.classes.schemas import (
    BudgetSizePreference,
    ChannelWeightsResponse,
    MetadataFilters,
    MetadataPreferencesResponse,
    ExtractedEntitiesResponse,
    LexicalCandidate,
    VectorWeights,
    VectorSubqueries,
    DatePreference,
    NumericalPreference,
    GenreListPreference,
    LanguageListPreference,
    WatchProvidersPreference,
    MaturityPreference,
    PopularTrendingPreference,
    ReceptionPreference,
)
from implementation.classes.enums import BudgetSize, RelevanceSize, ReceptionType, VectorName


DUMMY_QUERY = "find me a fun movie"
DUMMY_FILTERS = MetadataFilters()


def _default_metadata_preferences() -> MetadataPreferencesResponse:
    return MetadataPreferencesResponse(
        release_date_preference=DatePreference(result=None),
        duration_preference=NumericalPreference(result=None),
        genres_preference=GenreListPreference(result=None),
        audio_languages_preference=LanguageListPreference(result=None),
        watch_providers_preference=WatchProvidersPreference(result=None),
        maturity_rating_preference=MaturityPreference(result=None),
        popular_trending_preference=PopularTrendingPreference(
            prefers_trending_movies=False, prefers_popular_movies=False,
        ),
        reception_preference=ReceptionPreference(reception_type=ReceptionType.NO_PREFERENCE),
        budget_size_preference=BudgetSizePreference(
            budget_size=BudgetSize.NO_PREFERENCE,
        ),
    )


def _default_channel_weights() -> ChannelWeightsResponse:
    return ChannelWeightsResponse(
        lexical_relevance=RelevanceSize.MEDIUM,
        metadata_relevance=RelevanceSize.MEDIUM,
        vector_relevance=RelevanceSize.MEDIUM,
    )


def _mock_new_dependencies():
    """Return a context manager that patches the four new async dependencies."""
    metadata_prefs = _default_metadata_preferences()
    channel_weights = _default_channel_weights()

    return (
        patch(
            "db.search.extract_all_metadata_preferences_async",
            new=AsyncMock(return_value=metadata_prefs),
        ),
        patch(
            "db.search.create_channel_weights_async",
            new=AsyncMock(return_value=channel_weights),
        ),
        patch(
            "db.metadata_scoring.create_metadata_scores",
            new=AsyncMock(side_effect=lambda prefs, candidates, reception_scores_out=None: candidates),
        ),
        patch(
            "db.search.fetch_reception_scores",
            new=AsyncMock(return_value={}),
        ),
    )


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

    p1, p2, p3, p4 = _mock_new_dependencies()
    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)) as mock_lex,
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)) as mock_vec,
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
        p1, p2, p3, p4,
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

    p1, p2, p3, p4 = _mock_new_dependencies()
    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)),
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)),
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
        p1, p2, p3, p4,
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

    p1, p2, p3, p4 = _mock_new_dependencies()
    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)),
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)),
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
        p1, p2, p3, p4,
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

    p1, p2, p3, p4 = _mock_new_dependencies()
    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)),
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)),
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
        p1, p2, p3, p4,
    ):
        result = await search(DUMMY_QUERY, DUMMY_FILTERS, mock_client)

    assert result.debug.lexical_debug is lex_debug
    assert result.debug.vector_debug is vec_debug
    assert result.debug.total_candidates == 0
    assert result.debug.total_latency_ms > 0
    assert result.debug.metadata_preferences_debug is not None
    assert result.debug.channel_weights_debug is not None


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

    p1, p2, p3, p4 = _mock_new_dependencies()
    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)),
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)),
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
        p1, p2, p3, p4,
    ):
        result = await search(DUMMY_QUERY, DUMMY_FILTERS, mock_client)

    assert result.candidates == []
    assert result.debug.total_candidates == 0


@pytest.mark.asyncio
async def test_search_reranking_applies_quality_prior():
    """Candidates with the same bucketed relevance sort by reception score."""
    # Two lexical-only candidates with nearly identical scores
    # (both round to 0.33 at 2 decimal places under equal channel weights)
    lexical_res = LexicalSearchResult(
        candidates=[
            _lexical_candidate(1, 0.999),
            _lexical_candidate(2, 0.998),
        ],
        debug=_lexical_debug(),
    )
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

    # Movie 2 has much higher reception than movie 1
    reception_scores = {1: 35.0, 2: 85.0}

    mock_client = AsyncMock()

    p1, p2, p3, p4 = _mock_new_dependencies()
    # Override the default empty reception scores with our test data
    p4_override = patch(
        "db.search.fetch_reception_scores",
        new=AsyncMock(return_value=reception_scores),
    )
    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)),
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)),
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
        p1, p2, p3, p4_override,
    ):
        result = await search(DUMMY_QUERY, DUMMY_FILTERS, mock_client)

    assert len(result.candidates) == 2
    # Both candidates land in the same relevance bucket (both have equal final_score
    # since channel weight correction zeroes lexical weight with no entities).
    # Movie 2 has higher reception, so it should be ranked first via quality prior.
    assert result.candidates[0].movie_id == 2
    assert result.candidates[1].movie_id == 1
    # Verify the new fields are populated on the candidates
    assert result.candidates[0].quality_prior > result.candidates[1].quality_prior


@pytest.mark.asyncio
async def test_search_reranking_skips_fetch_when_metadata_scoring_provides_scores():
    """When create_metadata_scores populates reception_scores_out, fetch_reception_scores is not called."""
    lexical_res = LexicalSearchResult(
        candidates=[_lexical_candidate(1, 0.9)],
        debug=_lexical_debug(),
    )
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

    # Mock create_metadata_scores to populate the reception_scores_out dict
    async def _mock_metadata_scoring(prefs, candidates, reception_scores_out=None):
        if reception_scores_out is not None:
            reception_scores_out[1] = 75.0
        return candidates

    mock_client = AsyncMock()
    mock_fetch_reception = AsyncMock(return_value={})

    p1, p2, _, _ = _mock_new_dependencies()
    with (
        patch("db.search.lexical_search", new=AsyncMock(return_value=lexical_res)),
        patch("db.search.run_vector_search", new=AsyncMock(return_value=vector_res)),
        patch("db.search.calculate_vector_scores", return_value=scoring_res),
        p1, p2,
        patch("db.metadata_scoring.create_metadata_scores", new=AsyncMock(side_effect=_mock_metadata_scoring)),
        patch("db.search.fetch_reception_scores", new=mock_fetch_reception),
    ):
        result = await search(DUMMY_QUERY, DUMMY_FILTERS, mock_client)

    # fetch_reception_scores should NOT have been called since metadata scoring
    # already populated the reception scores dict
    mock_fetch_reception.assert_not_called()
    assert len(result.candidates) == 1
    assert result.candidates[0].quality_prior > 0
