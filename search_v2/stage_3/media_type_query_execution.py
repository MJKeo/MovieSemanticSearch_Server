# Search V2 — Stage 3 Media Type Endpoint: Query Execution
#
# Takes the LLM's MediaTypeQuerySpec output and runs a single
# structured-attribute lookup against public.movie_card.release_format,
# producing an EndpointResult with [0, 1] scores per matched movie_id.
# Works uniformly for both dealbreakers (no restrict set — return
# naturally matched movies) and preferences (restrict set provided —
# return one entry per supplied ID, with 0.0 for non-matches).
#
# Single path, flat 1.0 scoring per match. There is no prominence
# signal on a movie's media type — either it is a TV movie or it
# isn't. The trait category (schemas/trait_category.py: MEDIA_TYPE)
# fires only on explicit non-default requests ("TV movies", "shorts",
# "direct-to-video"), so the matched set is intentionally narrow.
#
# Closed-enum input: the LLM emits ReleaseFormat members directly via
# the Literal subset on MediaTypeQuerySpec.formats. The executor maps
# each member to its `release_format_id` int and hands a SMALLINT list
# to the Postgres helper — no string normalization layer.

from __future__ import annotations

from db.postgres import fetch_movie_ids_by_release_format
from schemas.endpoint_result import EndpointResult
from schemas.media_type_translation import MediaTypeQuerySpec
from search_v2.stage_3.result_helpers import build_endpoint_result


async def execute_media_type_query(
    spec: MediaTypeQuerySpec,
    *,
    restrict_to_movie_ids: set[int] | None = None,
) -> EndpointResult:
    """Execute one MediaTypeQuerySpec against public.movie_card.release_format.

    Single entry point for both dealbreakers and preferences. The
    restrict_to_movie_ids parameter controls output shape:
      - None (dealbreaker path) → one ScoredCandidate per naturally
        matched movie at flat 1.0, non-matches omitted.
      - set[int] (preference path) → exactly one ScoredCandidate per
        supplied ID, with 1.0 for matches and 0.0 for non-matches.

    Args:
        spec: Validated MediaTypeQuerySpec from the step 3 media-type LLM.
            spec.formats is a non-empty list of ReleaseFormat members.
        restrict_to_movie_ids: Optional candidate-pool restriction.
            Pass the preference's candidate pool to get one entry per
            ID; omit to get the natural match set for dealbreakers.

    Returns:
        EndpointResult with scores ∈ {0.0, 1.0} per movie.
    """
    # ReleaseFormat members carry an int `release_format_id` attribute
    # (set in __new__) that keys the SMALLINT column on movie_card.
    # Hand the int list to the SQL-layer helper so it stays pure.
    release_format_ids = [fmt.release_format_id for fmt in spec.formats]

    matched_ids = await fetch_movie_ids_by_release_format(
        release_format_ids, restrict_to_movie_ids
    )
    scores_by_movie = {mid: 1.0 for mid in matched_ids}

    return build_endpoint_result(scores_by_movie, restrict_to_movie_ids)
