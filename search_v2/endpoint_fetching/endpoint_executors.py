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
from schemas.entity_translation import (
    CharacterQuerySpec,
    PersonQuerySpec,
    TitlePatternQuerySpec,
)
from schemas.enums import EndpointRoute
from schemas.semantic_translation import (
    CarverSemanticEndpointParameters,
    QualifierSemanticEndpointParameters,
)
from search_v2.endpoint_fetching.award_query_execution import execute_award_query
from search_v2.endpoint_fetching.category_handlers.endpoint_registry import (
    ROUTE_TO_WRAPPER,
)
from search_v2.endpoint_fetching.entity_query_execution import execute_entity_query
from search_v2.endpoint_fetching.franchise_query_execution import execute_franchise_query
from search_v2.endpoint_fetching.keyword_query_execution import execute_keyword_query
from search_v2.endpoint_fetching.media_type_query_execution import execute_media_type_query
from search_v2.endpoint_fetching.metadata_query_execution import execute_metadata_query
from search_v2.endpoint_fetching.semantic_query_execution import execute_semantic_query
from search_v2.endpoint_fetching.studio_query_execution import execute_studio_query


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
    # ENTITY is the one route whose `wrapper` IS the spec (the three
    # per-category specs inherit from EndpointParameters directly,
    # without an intermediate Entity wrapper). Every other route still
    # uses the wrapper.parameters indirection.
    if route == EndpointRoute.ENTITY:
        return execute_entity_query(
            wrapper, restrict_to_movie_ids=restrict_to_movie_ids
        )

    params = wrapper.parameters

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
        # Semantic executor branches on role internally — pass
        # the wrapper's role through and let it pick between
        # the primary_vector-only CARVER path and the all-spaces
        # QUALIFIER path.
        assert isinstance(
            wrapper,
            (CarverSemanticEndpointParameters, QualifierSemanticEndpointParameters),
        )
        return execute_semantic_query(
            params,
            role=wrapper.role,
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
# wrapper; ROUTE_TO_WRAPPER's None values are filtered out here. ENTITY
# is also absent from ROUTE_TO_WRAPPER (it dispatches per category via
# endpoint_registry._ENTITY_DISPATCH), so its three concrete specs are
# registered explicitly below — the orchestrator only sees the spec
# instances, never a wrapper class, so all three need an entry.
_WRAPPER_TO_ROUTE: dict[type[EndpointParameters], EndpointRoute] = {
    wrapper_cls: route
    for route, wrapper_cls in ROUTE_TO_WRAPPER.items()
    if wrapper_cls is not None
}
_WRAPPER_TO_ROUTE.update({
    PersonQuerySpec: EndpointRoute.ENTITY,
    CharacterQuerySpec: EndpointRoute.ENTITY,
    TitlePatternQuerySpec: EndpointRoute.ENTITY,
})


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
