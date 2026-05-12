# Shared GeneratedEndpointSpec → executor dispatch for stage-3.
#
# Two callers need the same logic: handler.py (which fires whichever
# endpoints the per-category LLM elected) and orchestrator.py (which
# fires deferred preference_specs against the consolidated candidate
# pool, and also runs preferences against the full corpus when
# inclusion produces nothing). Both hand a GeneratedEndpointSpec to
# build_endpoint_coroutine and want the same operation-type-aware
# gating, the same EndpointRoute → execute_*_query mapping, and the
# same way of unpacking the EndpointParameters wrapper.
#
# Pre-dispatch gates (applied uniformly to every route):
#   1. POOL_RERANKER + falsy restrict_to_movie_ids → empty result.
#      A reranker with no pool has nothing to rerank.
#   2. spec.params is None → log a warning + empty result. Upstream
#      should never produce a None-params spec for an executable
#      route; surfacing it here keeps stage-4 robust to upstream
#      regressions instead of AttributeError-ing inside an executor.
#   3. CANDIDATE_GENERATOR → restrict_to_movie_ids is locally rebound
#      to None so finders never accidentally narrow to a pool. The
#      caller's set is never mutated.
#
# TRENDING is intentionally absent from the dispatch — it has no LLM
# codepath and the trending executor is invoked directly elsewhere.
# It's also the one route where params=None is a legitimate output of
# run_query_generation (see handler._generate_deterministic_specs);
# routing TRENDING through this dispatcher would fire a spurious
# None-params warning on every call. Callers must special-case
# TRENDING before calling build_endpoint_coroutine — reaching this
# dispatch with a TRENDING route is a programmer error.

from __future__ import annotations

import logging
from typing import Any, Coroutine

from qdrant_client import AsyncQdrantClient

from schemas.chronological_translation import ChronologicalQuerySpec
from schemas.endpoint_parameters import EndpointParameters
from schemas.endpoint_result import EndpointResult
from schemas.entity_translation import (
    CharacterQuerySpec,
    PersonQuerySpec,
    TitlePatternQuerySpec,
)
from schemas.enums import EndpointRoute, OperationType
from schemas.semantic_translation import (
    SemanticEndpointParameters,
    SemanticEndpointSubintentParameters,
)
from search_v2.endpoint_fetching.award_query_execution import execute_award_query
from search_v2.endpoint_fetching.category_handlers.endpoint_registry import (
    ROUTE_TO_WRAPPER,
    ROUTE_TO_SUBINTENT_WRAPPER,
)
from search_v2.endpoint_fetching.chronological_query_execution import (
    execute_chronological_query,
)
from search_v2.endpoint_fetching.category_handlers.generated_endpoint_spec import (
    GeneratedEndpointSpec,
)
from search_v2.endpoint_fetching.entity_query_execution import execute_entity_query
from search_v2.endpoint_fetching.franchise_query_execution import execute_franchise_query
from search_v2.endpoint_fetching.keyword_query_execution import execute_keyword_query
from search_v2.endpoint_fetching.media_type_query_execution import execute_media_type_query
from search_v2.endpoint_fetching.metadata_query_execution import execute_metadata_query
from search_v2.endpoint_fetching.result_helpers import build_endpoint_result
from search_v2.endpoint_fetching.semantic_query_execution import execute_semantic_query
from search_v2.endpoint_fetching.studio_query_execution import execute_studio_query


logger = logging.getLogger(__name__)


async def _empty_endpoint_result(
    restrict_to_movie_ids: set[int] | None,
) -> EndpointResult:
    """Resolve to an empty EndpointResult shaped to the requested
    restrict-mode (omit non-matches in dealbreaker mode; one zero-scored
    entry per pool ID in preference mode)."""
    return build_endpoint_result({}, restrict_to_movie_ids)


def build_endpoint_coroutine(
    spec: GeneratedEndpointSpec,
    *,
    qdrant_client: AsyncQdrantClient,
    restrict_to_movie_ids: set[int] | None,
) -> Coroutine[Any, Any, EndpointResult]:
    """Build the awaitable that runs `spec`'s endpoint executor.

    Operation-type-aware gating is applied here so every route benefits
    uniformly:

      - POOL_RERANKER with no candidate pool (None or empty set) →
        empty-result coroutine. A reranker with nothing to rerank has
        no work to do; running the underlying executor would either
        fall back to a full-corpus scan (wrong semantics for a
        reranker) or just emit an empty result anyway.

      - Missing `spec.params` → log a warning and return an empty
        result. Upstream (handler / orchestrator) should never produce
        a None-params spec for an executable route, but surfacing it
        here keeps stage-4 robust to upstream regressions instead of
        AttributeError-ing inside an executor.

      - CANDIDATE_GENERATOR specs always search the full corpus —
        passing a `restrict_to_movie_ids` to a finder is a category
        error (it would silently constrain candidate generation to a
        pre-existing pool). We override to None locally so callers
        can't accidentally narrow a finder; the caller's set is never
        mutated.

    For SEMANTIC the carver-vs-qualifier decision is committed by the
    LLM inside the unified semantic schema's `role` field; the
    executor reads it from `params` at runtime. Other endpoints have
    no role-shaped dispatch.
    """
    if (
        spec.operation_type == OperationType.POOL_RERANKER
        and not restrict_to_movie_ids
    ):
        return _empty_endpoint_result(restrict_to_movie_ids)

    if spec.params is None:
        logger.warning(
            "build_endpoint_coroutine: spec.params is None for "
            "route=%s; returning empty result.",
            spec.route.value,
        )
        return _empty_endpoint_result(restrict_to_movie_ids)

    # Candidate generators always run against the full corpus. Local
    # rebind only — the caller's restrict set is left untouched.
    if spec.operation_type == OperationType.CANDIDATE_GENERATOR:
        restrict_to_movie_ids = None

    route = spec.route
    wrapper = spec.params

    # ENTITY and CHRONOLOGICAL are routes whose `wrapper` IS the spec
    # — the spec class inherits from EndpointParameters directly with
    # no intermediate `parameters` wrapper. ENTITY does this because
    # the three per-category specs each have a different shape and a
    # union-typed wrapper offered no schema-level narrowing;
    # CHRONOLOGICAL does it because the spec carries a single
    # `direction` field and a wrapper indirection would be empty
    # overhead. Every other route still uses the wrapper.parameters
    # indirection.
    if route == EndpointRoute.ENTITY:
        return execute_entity_query(
            wrapper, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if route == EndpointRoute.CHRONOLOGICAL:
        assert isinstance(wrapper, ChronologicalQuerySpec)
        return execute_chronological_query(
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
        # Semantic executor branches on the LLM-committed `role` field
        # inside `params` (carver vs qualifier).
        assert isinstance(
            wrapper,
            (SemanticEndpointParameters, SemanticEndpointSubintentParameters),
        )
        return execute_semantic_query(
            params,
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
    wrapper_cls: route
    for route, wrapper_cls in ROUTE_TO_SUBINTENT_WRAPPER.items()
    if wrapper_cls is not None
})
_WRAPPER_TO_ROUTE.update({
    PersonQuerySpec: EndpointRoute.ENTITY,
    CharacterQuerySpec: EndpointRoute.ENTITY,
    TitlePatternQuerySpec: EndpointRoute.ENTITY,
    SemanticEndpointParameters: EndpointRoute.SEMANTIC,
    SemanticEndpointSubintentParameters: EndpointRoute.SEMANTIC,
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
