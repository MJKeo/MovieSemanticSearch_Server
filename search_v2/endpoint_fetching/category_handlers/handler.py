# Query-generation and temporary execution surface for one category
# handler on one (CategoryCall, Trait) pair.
#
# run_query_generation owns the per-call translation from a routed
# CategoryCall into zero or more GeneratedEndpointSpec objects. It
# handles explicit no-op categories, deterministic no-LLM categories,
# handler-LLM calls, fired-endpoint extraction, and initial operation
# type assignment.
#
# run_query_execution is intentionally temporary. It executes only the
# deterministic TRENDING and MEDIA_TYPE routes for now and no-ops for
# every other route until the stage-4 execution contract is finalized.

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)
from schemas.enums import (
    EndpointRoute,
    HandlerBucket,
    OperationType,
    Polarity,
    TraitCombineMode,
)
from schemas.media_type_translation import MediaTypeEndpointParameters
from schemas.step_2 import Trait
from schemas.step_3 import CategoryCall
from schemas.trait_category import CategoryName
from search_v2.endpoint_fetching.category_handlers.generated_endpoint_spec import (
    GeneratedEndpointSpec,
)
from search_v2.endpoint_fetching.category_handlers.media_type_router import (
    build_media_type_query_spec,
)
from search_v2.endpoint_fetching.category_handlers.output_extractor import (
    extract_fired_endpoints,
)
from search_v2.endpoint_fetching.category_handlers.prompt_builder import (
    build_system_prompt,
    build_user_message,
)
from search_v2.endpoint_fetching.category_handlers.schema_factories import (
    get_output_schema,
)
from search_v2.endpoint_fetching.endpoint_executors import (
    build_endpoint_coroutine,
)
from search_v2.endpoint_fetching.trending_query_execution import (
    execute_trending_query,
)

logger = logging.getLogger(__name__)


# Timeout / retry / jitter for the handler-LLM call live one layer
# below us, inside `generate_llm_response_async`. The constants here
# remain only because the deterministic-endpoint execution surface
# (`run_query_execution`) still uses TIMEOUT_SECONDS to bound the
# Qdrant / Postgres dispatch — that path is not an LLM call and
# manages its own timeout.
TIMEOUT_SECONDS = 25.0

# Exported (no underscore prefix) so other Step-3-translator callers
# can use the same model + reasoning settings without a copy-paste.
# Today's other caller is the non-character franchise executor at
# search_v2/non_character_franchise_search.py, which reuses the
# franchise translator for alias expansion and must stay in lockstep
# with handler dispatch on model choice.
HANDLER_LLM_PROVIDER = LLMProvider.OPENAI
HANDLER_LLM_MODEL = "gpt-5.4-mini"
HANDLER_LLM_KWARGS: dict = {
    "reasoning_effort": "none",
    "verbosity": "low",
}


async def run_query_generation(
    *,
    category_call: CategoryCall,
    trait: Trait,
    sibling_calls: list[CategoryCall] | None = None,
    combine_mode: TraitCombineMode | None = None,
) -> list[GeneratedEndpointSpec]:
    """Generate endpoint specs for one CategoryCall in its Trait context.

    `sibling_calls` carries the OTHER CategoryCalls Step 3 committed
    for the same trait (self-category excluded). `combine_mode` is
    the trait-level fold rule. Both feed the user-message
    `<sibling_categories>` block so handlers can coordinate
    commit-vs-abstain decisions against parallel sibling tasks
    without violating per-call isolation. Both default to None so
    legacy callsites (and deterministic / no-op buckets) keep
    working unchanged — None routes through to the empty-wrapper
    ``combine_mode="single"`` shape in `build_user_message`.
    """
    category = category_call.category

    if category.bucket is HandlerBucket.EXPLICIT_NO_OP:
        return []

    if category.bucket is HandlerBucket.NO_LLM_PURE_CODE:
        return _generate_deterministic_specs(category_call, trait=trait)

    output = await _run_handler_llm(
        category=category,
        category_call=category_call,
        sibling_calls=sibling_calls,
        combine_mode=combine_mode,
    )
    if output is None:
        return []

    fired = extract_fired_endpoints(category, output)
    return [
        GeneratedEndpointSpec(
            route=route,
            params=params,
            operation_type=determine_operation_type(
                category, route, trait.polarity
            ),
        )
        for route, params in fired
    ]


async def run_query_execution(
    spec: GeneratedEndpointSpec,
    *,
    qdrant_client: AsyncQdrantClient,
    restrict_to_movie_ids: set[int] | None = None,
) -> None:
    """Execute one generated endpoint spec, discarding the result.

    TRENDING bypasses `build_endpoint_coroutine` (it has no LLM
    codepath and the trending executor is invoked directly elsewhere
    in the dispatcher's contract). Every other route is routed through
    `build_endpoint_coroutine`, which centralizes operation-type-aware
    gating: pool rerankers with no pool short-circuit to empty,
    candidate generators ignore `restrict_to_movie_ids`, and
    None-params specs log a warning instead of crashing.

    Results are intentionally discarded — this surface only preserves
    the deterministic endpoint execution hook while stage-4 execution
    is still being reshaped.
    """
    if spec.route is EndpointRoute.TRENDING:
        try:
            await asyncio.wait_for(
                execute_trending_query(
                    restrict_to_movie_ids=None if spec.operation_type == OperationType.CANDIDATE_GENERATOR else restrict_to_movie_ids
                ),
                timeout=TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001 - temporary soft-fail path
            logger.warning(
                "trending query execution failed; skipping (%r)",
                exc,
            )
        return

    try:
        await asyncio.wait_for(
            build_endpoint_coroutine(
                spec,
                qdrant_client=qdrant_client,
                restrict_to_movie_ids=restrict_to_movie_ids,
            ),
            timeout=TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 - temporary soft-fail path
        logger.warning(
            "%s query execution failed; skipping (%r)",
            spec.route.value,
            exc,
        )


def _generate_deterministic_specs(
    category_call: CategoryCall,
    *,
    trait: Trait,
) -> list[GeneratedEndpointSpec]:
    category = category_call.category

    if category is CategoryName.TRENDING:
        return [
            GeneratedEndpointSpec(
                route=EndpointRoute.TRENDING,
                params=None,
                operation_type=determine_operation_type(
                    category,
                    EndpointRoute.TRENDING,
                    trait.polarity,
                ),
            )
        ]

    if category is CategoryName.MEDIA_TYPE:
        spec = build_media_type_query_spec(category_call)
        if spec is None:
            return []

        return [
            GeneratedEndpointSpec(
                route=EndpointRoute.MEDIA_TYPE,
                params=MediaTypeEndpointParameters(parameters=spec),
                operation_type=determine_operation_type(
                    category,
                    EndpointRoute.MEDIA_TYPE,
                    trait.polarity,
                ),
            )
        ]

    raise ValueError(
        f"Unhandled NO_LLM_PURE_CODE category: CategoryName.{category.name}. "
        f"Add a deterministic handler before routing this category."
    )


async def _run_handler_llm(
    *,
    category: CategoryName,
    category_call: CategoryCall,
    sibling_calls: list[CategoryCall] | None = None,
    combine_mode: TraitCombineMode | None = None,
) -> BaseModel | None:
    system_prompt = build_system_prompt(category)
    user_message = build_user_message(
        category_call,
        sibling_calls=sibling_calls,
        combine_mode=combine_mode,
    )
    response_format = get_output_schema(category)

    try:
        # Timeout, retry, and jittered backoff are all applied
        # inside generate_llm_response_async — see
        # `LLM_PER_ATTEMPT_TIMEOUT_SECONDS` / `LLM_MAX_ATTEMPTS` /
        # `_LLM_RETRY_BACKOFF_*` constants in generic_methods.py.
        response, _, _ = await generate_llm_response_async(
            provider=HANDLER_LLM_PROVIDER,
            user_prompt=user_message,
            system_prompt=system_prompt,
            response_format=response_format,
            model=HANDLER_LLM_MODEL,
            **HANDLER_LLM_KWARGS,
        )
        return response
    except Exception as exc:  # noqa: BLE001 — soft-fail by design
        logger.error(
            "handler LLM call exhausted retries; returning empty specs "
            "(category=%s, error=%r)",
            category.name,
            exc,
        )
        return None


def determine_operation_type(
    category: CategoryName,
    route: EndpointRoute,
    polarity: Polarity,
) -> OperationType:
    """Return the initial operation type for a generated endpoint spec."""
    if polarity is Polarity.NEGATIVE:
        return OperationType.POOL_RERANKER

    if route in (
        EndpointRoute.ENTITY,
        EndpointRoute.STUDIO,
        EndpointRoute.AWARDS,
        EndpointRoute.FRANCHISE_STRUCTURE,
        EndpointRoute.KEYWORD,
        EndpointRoute.TRENDING,
        EndpointRoute.MEDIA_TYPE,
        EndpointRoute.NEUTRAL_SEED,
    ):
        return OperationType.CANDIDATE_GENERATOR

    if route is EndpointRoute.SEMANTIC:
        return OperationType.POOL_RERANKER

    if route is EndpointRoute.CHRONOLOGICAL:
        return OperationType.POOL_RERANKER

    if route is EndpointRoute.METADATA:
        if category in (
            CategoryName.GENERAL_APPEAL,
            CategoryName.CULTURAL_STATUS,
        ):
            return OperationType.POOL_RERANKER
        return OperationType.CANDIDATE_GENERATOR

    raise ValueError(f"Unhandled endpoint route for operation_type: {route}")
