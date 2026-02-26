"""
search.py — Unified search orchestrator.

Runs lexical search (Postgres entity matching) and vector search (Qdrant
similarity) in parallel, then merges candidates into a single result set.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Optional

from qdrant_client import AsyncQdrantClient

from db.lexical_search import lexical_search, LexicalSearchDebug, LexicalSearchResult
from db.vector_search import run_vector_search, VectorSearchResult, VectorSearchDebug
from db.vector_scoring import calculate_vector_scores
from implementation.classes.schemas import MetadataFilters


@dataclass(slots=True)
class SearchCandidate:
    movie_id: int
    vector_score: float        # final weighted sum from calculate_vector_scores [0,1]
    lexical_score: float       # normalized_lexical_score from LexicalCandidate


@dataclass(slots=True)
class SearchDebug:
    total_candidates: int
    total_latency_ms: float
    lexical_debug: LexicalSearchDebug
    vector_debug: Optional[VectorSearchDebug]


@dataclass(slots=True)
class SearchResult:
    candidates: list[SearchCandidate]
    debug: SearchDebug


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

    Runs lexical and vector searches in parallel, scores vector results,
    and merges all candidates into a single list.
    """
    start = time.perf_counter()

    lexical_result, vector_result = await asyncio.gather(
        lexical_search(query, metadata_filters),
        run_vector_search(
            query=query,
            metadata_filters=metadata_filters,
            qdrant_client=qdrant_client,
            candidate_limit_original=vector_candidate_limit_original,
            candidate_limit_subquery=vector_candidate_limit_subquery,
            candidate_limit_anchor=vector_candidate_limit_anchor,
        ),
    )

    # Score vector candidates
    scoring_result = calculate_vector_scores(vector_result)
    final_scores: dict[int, float] = scoring_result.final_scores

    # Merge candidates: start with vector scores
    merged: dict[int, SearchCandidate] = {}
    for movie_id, v_score in final_scores.items():
        merged[movie_id] = SearchCandidate(
            movie_id=movie_id,
            vector_score=v_score,
            lexical_score=0.0,
        )

    # Layer in lexical candidates
    for lc in lexical_result.candidates:
        if lc.movie_id in merged:
            merged[lc.movie_id].lexical_score = lc.normalized_lexical_score
        else:
            merged[lc.movie_id] = SearchCandidate(
                movie_id=lc.movie_id,
                vector_score=0.0,
                lexical_score=lc.normalized_lexical_score,
            )

    end = time.perf_counter()

    debug = SearchDebug(
        total_candidates=len(merged),
        total_latency_ms=(end - start) * 1000,
        lexical_debug=lexical_result.debug,
        vector_debug=vector_result.debug,
    )

    return SearchResult(
        candidates=list(merged.values()),
        debug=debug,
    )
