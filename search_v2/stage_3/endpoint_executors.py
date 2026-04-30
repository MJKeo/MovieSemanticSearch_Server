# Shared route → executor dispatch for stage-3 endpoint parameters.
#
# Two callers need the same logic: handler.py (which fires whichever
# endpoints the per-category LLM elected) and orchestrator.py (which
# fires deferred preference_specs against the consolidated candidate
# pool, and also runs preferences against the full corpus when
# inclusion produces nothing). Both want the same mapping from
# EndpointRoute to its execute_*_query coroutine and the same way of
# unpacking the EndpointParameters wrapper.
#
# TRENDING is intentionally absent — it has no LLM codepath and the
# trending executor is invoked directly elsewhere. Reaching this
# dispatch with a TRENDING route is a programmer error.

from __future__ import annotations

from typing import Any, Coroutine

from qdrant_client import AsyncQdrantClient

from schemas.endpoint_parameters import EndpointParameters
from schemas.endpoint_result import EndpointResult
from schemas.enums import EndpointRoute
from schemas.semantic_translation import SemanticEndpointParameters
from search_v2.stage_3.award_query_execution import execute_award_query
from search_v2.stage_3.category_handlers.endpoint_registry import (
    ROUTE_TO_WRAPPER,
)
from search_v2.stage_3.entity_query_execution import execute_entity_query
from search_v2.stage_3.franchise_query_execution import execute_franchise_query
from search_v2.stage_3.keyword_query_execution import execute_keyword_query
from search_v2.stage_3.media_type_query_execution import execute_media_type_query
from search_v2.stage_3.metadata_query_execution import execute_metadata_query
from search_v2.stage_3.semantic_query_execution import execute_semantic_query
from search_v2.stage_3.studio_query_execution import execute_studio_query


def build_endpoint_coroutine(
    route: EndpointRoute,
    wrapper: EndpointParameters,
    *,
    qdrant_client: AsyncQdrantClient,
    restrict_to_movie_ids: set[int] | None,
) -> Coroutine[Any, Any, EndpointResult]:
    """Build the awaitable that runs `wrapper`'s endpoint executor.

    `restrict_to_movie_ids` is forwarded to the executor verbatim:
    None means "search the full corpus" (handler-time candidate
    generation; preference-against-corpus fallback). A non-None set
    restricts the search to the given pool (orchestrator-time
    deferred preference scoring).
    """
    params = wrapper.parameters

    if route == EndpointRoute.ENTITY:
        return execute_entity_query(
            params, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if route == EndpointRoute.STUDIO:
        return execute_studio_query(
            params, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if route == EndpointRoute.FRANCHISE_STRUCTURE:
        return execute_franchise_query(
            params, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if route == EndpointRoute.KEYWORD:
        return execute_keyword_query(
            params, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if route == EndpointRoute.METADATA:
        return execute_metadata_query(
            params, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if route == EndpointRoute.AWARDS:
        return execute_award_query(
            params, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if route == EndpointRoute.MEDIA_TYPE:
        return execute_media_type_query(
            params, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if route == EndpointRoute.SEMANTIC:
        # Semantic executor branches on match_mode internally — pass
        # the wrapper's match_mode through and let it pick between
        # the primary_vector-only FILTER path and the all-spaces
        # TRAIT path.
        assert isinstance(wrapper, SemanticEndpointParameters)
        return execute_semantic_query(
            params,
            match_mode=wrapper.match_mode,
            restrict_to_movie_ids=restrict_to_movie_ids,
            qdrant_client=qdrant_client,
        )

    raise ValueError(f"Unsupported route for handler execution: {route!r}")


# Reverse of ROUTE_TO_WRAPPER: maps an EndpointParameters subclass
# back to its EndpointRoute. Used by the orchestrator to dispatch
# deferred preference_specs (it sees the wrapper instances but
# needs the route to call build_endpoint_coroutine).
#
# Built once at import time. TRENDING is excluded because it has no
# wrapper; ROUTE_TO_WRAPPER's None values are filtered out here.
_WRAPPER_TO_ROUTE: dict[type[EndpointParameters], EndpointRoute] = {
    wrapper_cls: route
    for route, wrapper_cls in ROUTE_TO_WRAPPER.items()
    if wrapper_cls is not None
}


def route_for_wrapper(wrapper: EndpointParameters) -> EndpointRoute:
    """Return the EndpointRoute for a concrete EndpointParameters instance.

    Inverts ROUTE_TO_WRAPPER. Each wrapper class is unique to a
    single route so a direct type lookup suffices — no isinstance
    chain needed.
    """
    route = _WRAPPER_TO_ROUTE.get(type(wrapper))
    if route is None:
        raise ValueError(
            f"No EndpointRoute registered for wrapper type "
            f"{type(wrapper).__name__!r}"
        )
    return route
