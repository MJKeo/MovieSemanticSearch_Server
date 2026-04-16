# Search V2 — Stage 3 Trending Endpoint: Query Execution
#
# Reads precomputed trending scores from Redis and emits an
# EndpointResult of (movie_id, score ∈ [0, 1]) pairs. The concave-decay
# rank→score curve lives in the refresh job (db/trending_movies.py);
# this module is strict pass-through.
#
# Unlike the other stage-3 endpoints, trending needs no LLM translation
# — step 2 flags the intent and execution just reads the Redis hash.
# As a result there is no TrendingQuerySpec and no query-generation
# sibling module; this file is the whole endpoint.
#
# Two modes on one function (matches the sibling endpoints):
#   - Dealbreaker mode (restrict_to_movie_ids=None): return one
#     ScoredCandidate per movie in the trending hash. The returned
#     set also doubles as this endpoint's contribution to Phase 4a
#     candidate pool assembly.
#   - Preference mode (restrict_to_movie_ids is a set of ids): return
#     exactly one ScoredCandidate per supplied id, with 0.0 for ids
#     absent from the trending hash.
#
# Direction-agnostic and scoring-policy-agnostic: inclusion/exclusion
# framing and preference weighting are orchestrator concerns handled
# in step 4. See search_improvement_planning/finalized_search_proposal.md
# §Endpoint 7 for the scoring contract.

from __future__ import annotations

from db.redis import read_trending_scores
from schemas.endpoint_result import EndpointResult, ScoredCandidate


def _build_endpoint_result(
    scores_by_movie: dict[int, float],
    restrict_movie_ids: set[int] | None,
) -> EndpointResult:
    """Convert the raw score map into an EndpointResult.

    Dealbreaker path (restrict is None): one ScoredCandidate per
    trending movie, at its precomputed score.

    Preference path (restrict provided): one ScoredCandidate per
    supplied ID, using the trending score or 0.0 for non-matches. An
    empty restrict set yields an empty EndpointResult.
    """
    if restrict_movie_ids is None:
        return EndpointResult(
            scores=[
                ScoredCandidate(movie_id=mid, score=score)
                for mid, score in scores_by_movie.items()
            ]
        )

    return EndpointResult(
        scores=[
            ScoredCandidate(movie_id=mid, score=scores_by_movie.get(mid, 0.0))
            for mid in restrict_movie_ids
        ]
    )


async def execute_trending_query(
    *,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Read the precomputed trending hash from Redis and emit an EndpointResult.

    Single entry point for both dealbreakers and preferences. The
    restrict_to_movie_ids parameter controls output shape:
      - None (dealbreaker path) → one ScoredCandidate per movie in the
        trending hash, non-trending movies omitted.
      - set[int] (preference path) → exactly one ScoredCandidate per
        supplied ID, with 0.0 for IDs absent from the trending hash.

    Returns an empty EndpointResult when the Redis hash is absent or
    empty — graceful degradation per the endpoint-failure contract in
    search_improvement_planning/open_questions.md.

    Args:
        restrict_to_movie_ids: Optional candidate-pool restriction.
            Pass the preference's candidate pool to get one entry per
            ID; omit to get the full trending set for dealbreakers.

    Returns:
        EndpointResult with scores ∈ [0, 1] per movie.
    """
    # Safety net: if an empty candidate pool reaches preference execution,
    # skip the Redis round-trip — the orchestrator should already be
    # short-circuiting before this point, but returning an empty result
    # here costs nothing and avoids a wasted HGETALL on the edge case.
    if restrict_to_movie_ids is not None and not restrict_to_movie_ids:
        return EndpointResult(scores=[])

    scores_by_movie = await read_trending_scores()
    return _build_endpoint_result(scores_by_movie, restrict_to_movie_ids)
