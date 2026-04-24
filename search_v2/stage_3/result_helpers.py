# Search V2 — Stage 3 shared result assembly helper.
#
# All stage-3 execution modules produce EndpointResult objects with the
# same dual-mode shape (dealbreaker vs. preference). This module holds
# the single shared implementation so each executor imports rather than
# duplicating.

from __future__ import annotations

from schemas.endpoint_result import EndpointResult, ScoredCandidate


def compress_to_dealbreaker_floor(raw: float) -> float:
    """Affine-map a raw [0, 1] score into the dealbreaker band [0.5, 1.0].

    Formula: ``0.5 + 0.5 * raw``. Every stage-3 endpoint that endorses a
    movie via a dealbreaker must emit a score of at least 0.5 so the
    candidate pool's downstream aggregation can rely on a uniform floor.
    Endpoints compute their natural [0, 1] score, drop zeros where
    appropriate, then pass the surviving values through this helper.

    The outer clamp defends against floating-point drift at the
    endpoints so ScoredCandidate range validation never rejects a row
    that should have landed at exactly 0.5 or 1.0.
    """
    return max(0.5, min(1.0, 0.5 + 0.5 * raw))


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
