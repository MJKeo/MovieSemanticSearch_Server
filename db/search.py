"""
search.py — Unified search orchestrator.

Runs lexical search (Postgres entity matching), vector search (Qdrant
similarity), metadata preference extraction, and channel weight inference
in parallel, then merges candidates into a single result set scored by a
weighted combination of all three channels.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from qdrant_client import AsyncQdrantClient

from db.lexical_search import lexical_search, LexicalSearchDebug
from db.vector_search import run_vector_search, VectorSearchDebug
from db.vector_scoring import calculate_vector_scores, RELEVANCE_RAW_WEIGHTS
from implementation.llms.query_understanding_methods import (
    create_channel_weights_async,
    extract_all_metadata_preferences_async,
)
from implementation.classes.enums import RelevanceSize
from implementation.classes.schemas import (
    ChannelWeightsResponse,
    MetadataPreferencesResponse,
    MetadataFilters,
)


@dataclass(slots=True)
class SearchCandidate:
    movie_id: int
    vector_score: float          # final weighted sum from calculate_vector_scores [0,1]
    lexical_score: float         # normalized_lexical_score from LexicalCandidate
    metadata_score: float = 0.0  # normalized metadata score calculated AFTER search completes
    final_score: float = 0.0     # weighted combination of all three channel scores


@dataclass(slots=True)
class MetadataPreferencesDebug:
    llm_generation_time_ms: float
    preferences: MetadataPreferencesResponse


@dataclass(slots=True)
class ChannelWeightsDebug:
    llm_generation_time_ms: float
    channel_weights: Optional[ChannelWeightsResponse]


@dataclass(slots=True)
class SearchDebug:
    total_candidates: int
    total_latency_ms: float
    lexical_debug: LexicalSearchDebug
    vector_debug: Optional[VectorSearchDebug]
    metadata_preferences_debug: Optional[MetadataPreferencesDebug] = None
    channel_weights_debug: Optional[ChannelWeightsDebug] = None


@dataclass(slots=True)
class SearchResult:
    candidates: list[SearchCandidate]
    debug: SearchDebug


# ---------------------------------------------------------------------------
# Small async timing helpers
# ---------------------------------------------------------------------------

async def _timed_channel_weights(query: str):
    """Run create_channel_weights_async with monotonic timing."""
    t0 = time.monotonic()
    result = await create_channel_weights_async(query)
    elapsed_ms = (time.monotonic() - t0) * 1000
    return result, elapsed_ms


async def _timed_metadata_preferences(query: str):
    """Run extract_all_metadata_preferences_async with monotonic timing."""
    t0 = time.monotonic()
    result = await extract_all_metadata_preferences_async(query)
    elapsed_ms = (time.monotonic() - t0) * 1000
    return result, elapsed_ms


# ---------------------------------------------------------------------------
# Main search orchestrator
# ---------------------------------------------------------------------------

async def search(
    query: str,
    metadata_filters: MetadataFilters,
    qdrant_client: AsyncQdrantClient,
    vector_candidate_limit_original: int = 2000,
    vector_candidate_limit_subquery: int = 2000,
    vector_candidate_limit_anchor: int = 2000,
) -> SearchResult:
    """
    Unified search entry point.

    Runs lexical search, vector search, metadata preference extraction, and
    channel weight inference in parallel, scores and merges all candidates,
    then produces a final weighted score per candidate.
    """
    # Deferred import to avoid circular dependency (metadata_scoring imports SearchCandidate)
    from db.metadata_scoring import create_metadata_scores

    start = time.perf_counter()

    # Phase A — Launch all 4 tasks in parallel
    channel_weights_task = asyncio.create_task(_timed_channel_weights(query))

    lexical_result, vector_result, metadata_pref_timed = await asyncio.gather(
        lexical_search(query, metadata_filters),
        run_vector_search(
            query=query,
            metadata_filters=metadata_filters,
            qdrant_client=qdrant_client,
            candidate_limit_original=vector_candidate_limit_original,
            candidate_limit_subquery=vector_candidate_limit_subquery,
            candidate_limit_anchor=vector_candidate_limit_anchor,
        ),
        _timed_metadata_preferences(query),
    )

    metadata_preferences, metadata_pref_ms = metadata_pref_timed

    # Phase B — Score vector candidates and merge with lexical (unchanged logic)
    scoring_result = calculate_vector_scores(vector_result)
    final_scores: dict[int, float] = scoring_result.final_scores

    merged: dict[int, SearchCandidate] = {}
    for movie_id, v_score in final_scores.items():
        merged[movie_id] = SearchCandidate(
            movie_id=movie_id,
            vector_score=v_score,
            lexical_score=0.0,
        )

    for lc in lexical_result.candidates:
        if lc.movie_id in merged:
            merged[lc.movie_id].lexical_score = lc.normalized_lexical_score
        else:
            merged[lc.movie_id] = SearchCandidate(
                movie_id=lc.movie_id,
                vector_score=0.0,
                lexical_score=lc.normalized_lexical_score,
            )

    # Phase C — Run metadata scoring
    candidates_list = list(merged.values())
    candidates_list = await create_metadata_scores(metadata_preferences, candidates_list)

    # Phase D — Await channel weights and compute final scores
    channel_weights_result, channel_weights_ms = await channel_weights_task

    if channel_weights_result is not None:
        raw_vector = RELEVANCE_RAW_WEIGHTS[RelevanceSize(channel_weights_result.vector_relevance)]
        raw_lexical = RELEVANCE_RAW_WEIGHTS[RelevanceSize(channel_weights_result.lexical_relevance)]
        raw_metadata = RELEVANCE_RAW_WEIGHTS[RelevanceSize(channel_weights_result.metadata_relevance)]
        total = raw_vector + raw_lexical + raw_metadata

        if total > 0:
            w_vector = raw_vector / total
            w_lexical = raw_lexical / total
            w_metadata = raw_metadata / total
        else:
            w_vector = w_lexical = w_metadata = 1.0 / 3.0
    else:
        # Fallback: equal weights when channel weights LLM returns None
        w_vector = w_lexical = w_metadata = 1.0 / 3.0

    for c in candidates_list:
        c.final_score = (
            w_vector * c.vector_score
            + w_lexical * c.lexical_score
            + w_metadata * c.metadata_score
        )

    # Phase E — Sort descending by final_score
    candidates_list.sort(key=lambda c: c.final_score, reverse=True)

    # Phase F — Build debug and return
    end = time.perf_counter()

    debug = SearchDebug(
        total_candidates=len(candidates_list),
        total_latency_ms=(end - start) * 1000,
        lexical_debug=lexical_result.debug,
        vector_debug=vector_result.debug,
        metadata_preferences_debug=MetadataPreferencesDebug(
            llm_generation_time_ms=metadata_pref_ms,
            preferences=metadata_preferences,
        ),
        channel_weights_debug=ChannelWeightsDebug(
            llm_generation_time_ms=channel_weights_ms,
            channel_weights=channel_weights_result,
        ),
    )

    return SearchResult(
        candidates=candidates_list,
        debug=debug,
    )
