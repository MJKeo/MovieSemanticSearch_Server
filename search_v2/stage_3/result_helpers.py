# Search V2 — Stage 3 shared result assembly helper.
#
# All stage-3 execution modules produce EndpointResult objects with the
# same dual-mode shape (dealbreaker vs. preference). This module holds
# the single shared implementation so each executor imports rather than
# duplicating.

from __future__ import annotations

from schemas.endpoint_result import EndpointResult, ScoredCandidate


def build_endpoint_result(
    scores_by_movie: dict[int, float],
    restrict_movie_ids: set[int] | None,
) -> EndpointResult:
    """Convert a raw score map into an EndpointResult.

    Dealbreaker path (restrict is None): one ScoredCandidate per matched
    movie at its computed score. Non-matches are omitted — the dealbreaker
    is a set-membership gate; step 4 handles inclusion/exclusion framing.

    Preference path (restrict provided): one ScoredCandidate per supplied
    ID, using the matched score or 0.0 for non-matches. The preference
    orchestrator expects a score for every candidate in the pool.

    Args:
        scores_by_movie: Mapping of movie_id → score ∈ [0, 1] for movies
            that satisfied the query. Binary-scoring endpoints (franchise)
            should pass {mid: 1.0 for mid in matched_ids}.
        restrict_movie_ids: Optional candidate-pool restriction. None for
            the dealbreaker path; a set of IDs for the preference path.

    Returns:
        EndpointResult with one ScoredCandidate per relevant movie.
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
