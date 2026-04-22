# Search V2 — Stage 4 dispatch layer.
#
# One abstraction over seven heterogeneous endpoints. Responsibilities:
#
#   translate_item(item, ...)
#     → wrap the endpoint's generator in a 20s timeout; return the
#       spec plus timing / status metadata. Trending has no generator
#       and returns a sentinel (None spec, ok status) immediately so
#       the caller can fan out uniformly.
#
#   execute_item(item, spec, restrict_to_movie_ids, ...)
#     → wrap the endpoint's executor in a 20s timeout; on success
#       return the EndpointResult; on timeout or error return an
#       empty EndpointResult with the failure stamped onto the
#       outcome's status.
#
# Both helpers never raise. Soft-fail is the whole point — the branch
# should keep going when one endpoint stalls or blows up. The
# existing executors already retry transient errors internally and
# collapse to empty results on the second failure, so in practice
# almost all failures surface here as asyncio.TimeoutError.

from __future__ import annotations

import asyncio
import time
from datetime import date
from typing import Any

from qdrant_client import AsyncQdrantClient

from implementation.llms.generic_methods import LLMProvider
from schemas.endpoint_result import EndpointResult
from schemas.enums import EndpointRoute
from search_v2.stage_3.award_query_execution import execute_award_query
from search_v2.stage_3.award_query_generation import generate_award_query
from search_v2.stage_3.entity_query_execution import execute_entity_query
from search_v2.stage_3.entity_query_generation import generate_entity_query
from search_v2.stage_3.franchise_query_execution import execute_franchise_query
from search_v2.stage_3.franchise_query_generation import generate_franchise_query
from search_v2.stage_3.keyword_query_execution import execute_keyword_query
from search_v2.stage_3.keyword_query_generation import generate_keyword_query
from search_v2.stage_3.metadata_query_execution import execute_metadata_query
from search_v2.stage_3.metadata_query_generation import generate_metadata_query
from search_v2.stage_3.semantic_query_execution import (
    execute_semantic_dealbreaker_query,
    execute_semantic_preference_query,
)
from search_v2.stage_3.semantic_query_generation import (
    generate_semantic_dealbreaker_query,
    generate_semantic_preference_query,
)
from search_v2.stage_3.studio_query_execution import execute_studio_query
from search_v2.stage_3.studio_query_generation import generate_studio_query
from search_v2.stage_3.trending_query_execution import execute_trending_query
from search_v2.stage_4.types import OutcomeStatus, TaggedItem


# Independent per-call budget. See step_4_planning.md §"Timeout model":
# the branch has no overarching budget; each LLM and each execution
# is its own island.
TIMEOUT_SECONDS = 20.0


# ---------------------------------------------------------------------------
# Translation (Step-3 LLM call)
# ---------------------------------------------------------------------------


async def translate_item(
    item: TaggedItem,
    *,
    intent_rewrite: str,
    today: date,
    provider: LLMProvider,
    model: str,
) -> tuple[Any | None, float | None, OutcomeStatus, str | None]:
    """Run one item's step-3 LLM translation under a 20s timeout.

    Returns (spec, llm_ms, status, error_message). Spec is whatever
    the endpoint's generator emits (EntityQuerySpec, AwardQuerySpec,
    etc.); callers hand it straight into execute_item without needing
    to care about the concrete type.

    Trending short-circuits: no LLM step exists, so the function
    returns (None, None, "ok", None) synchronously-fast so trending
    items can fan out through the same code path as everything else.
    """
    if item.endpoint == EndpointRoute.TRENDING:
        return (None, None, "ok", None)

    start = time.perf_counter()
    try:
        coro = _build_generator_call(
            item,
            intent_rewrite=intent_rewrite,
            today=today,
            provider=provider,
            model=model,
        )
        # Each generator returns (spec, input_tokens, output_tokens);
        # only the spec matters here (token accounting lives upstream).
        spec, _, _ = await asyncio.wait_for(coro, timeout=TIMEOUT_SECONDS)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (spec, elapsed_ms, "ok", None)
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (
            None,
            elapsed_ms,
            "timeout",
            f"translation exceeded {TIMEOUT_SECONDS}s",
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail by design
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (None, elapsed_ms, "error", repr(exc))


def _build_generator_call(
    item: TaggedItem,
    *,
    intent_rewrite: str,
    today: date,
    provider: LLMProvider,
    model: str,
):
    # Returns the awaitable for the endpoint's generator. Kept
    # separate from translate_item so the timing block stays tight
    # around the actual await.
    description = item.source.description
    route_rationale = item.source.route_rationale

    endpoint = item.endpoint
    if endpoint == EndpointRoute.ENTITY:
        return generate_entity_query(
            intent_rewrite, description, route_rationale, provider, model
        )
    if endpoint == EndpointRoute.STUDIO:
        return generate_studio_query(
            intent_rewrite, description, route_rationale, provider, model
        )
    if endpoint == EndpointRoute.FRANCHISE_STRUCTURE:
        return generate_franchise_query(
            intent_rewrite, description, route_rationale, provider, model
        )
    if endpoint == EndpointRoute.KEYWORD:
        return generate_keyword_query(
            intent_rewrite, description, route_rationale, provider, model
        )
    if endpoint == EndpointRoute.METADATA:
        return generate_metadata_query(
            intent_rewrite,
            description,
            route_rationale,
            today,
            provider,
            model,
        )
    if endpoint == EndpointRoute.AWARDS:
        return generate_award_query(
            intent_rewrite,
            description,
            route_rationale,
            today,
            provider,
            model,
        )
    if endpoint == EndpointRoute.SEMANTIC:
        # Preferences and dealbreakers use different generators; semantic
        # exclusions share the dealbreaker generator (positive-presence
        # framing is identical; orchestrator applies the E_MULT penalty).
        if item.role == "preference":
            return generate_semantic_preference_query(
                intent_rewrite,
                description,
                route_rationale,
                provider,
                model,
            )
        return generate_semantic_dealbreaker_query(
            intent_rewrite, description, route_rationale, provider, model
        )
    raise ValueError(f"Unsupported endpoint for translation: {endpoint}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


async def execute_item(
    item: TaggedItem,
    spec: Any | None,
    restrict_to_movie_ids: set[int] | None,
    *,
    qdrant_client: AsyncQdrantClient,
) -> tuple[EndpointResult, float | None, OutcomeStatus, str | None]:
    """Run one item's step-3 executor under a 20s timeout.

    `spec` is whatever translate_item returned. For trending it is
    None (no translation needed). For every other endpoint a None
    spec means translation failed upstream, so execution is skipped
    with status "skipped".
    """
    # Skip if translation already failed for an LLM-backed endpoint.
    if spec is None and item.endpoint != EndpointRoute.TRENDING:
        return (
            EndpointResult(),
            None,
            "skipped",
            "spec unavailable (translation failed)",
        )

    start = time.perf_counter()
    try:
        coro = _build_executor_call(
            item,
            spec=spec,
            restrict_to_movie_ids=restrict_to_movie_ids,
            qdrant_client=qdrant_client,
        )
        result: EndpointResult = await asyncio.wait_for(
            coro, timeout=TIMEOUT_SECONDS
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (result, elapsed_ms, "ok", None)
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (
            EndpointResult(),
            elapsed_ms,
            "timeout",
            f"execution exceeded {TIMEOUT_SECONDS}s",
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return (EndpointResult(), elapsed_ms, "error", repr(exc))


def _build_executor_call(
    item: TaggedItem,
    *,
    spec: Any | None,
    restrict_to_movie_ids: set[int] | None,
    qdrant_client: AsyncQdrantClient,
):
    endpoint = item.endpoint
    # Semantic needs the Qdrant client; nothing else does.
    if endpoint == EndpointRoute.ENTITY:
        return execute_entity_query(
            spec, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if endpoint == EndpointRoute.STUDIO:
        return execute_studio_query(
            spec, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if endpoint == EndpointRoute.FRANCHISE_STRUCTURE:
        return execute_franchise_query(
            spec, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if endpoint == EndpointRoute.KEYWORD:
        return execute_keyword_query(
            spec, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if endpoint == EndpointRoute.METADATA:
        # metadata executor takes the arg positionally (no keyword-only
        # marker), but passing by name is allowed and keeps this
        # dispatcher uniform.
        return execute_metadata_query(
            spec, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if endpoint == EndpointRoute.AWARDS:
        return execute_award_query(
            spec, restrict_to_movie_ids=restrict_to_movie_ids
        )
    if endpoint == EndpointRoute.SEMANTIC:
        if item.role == "preference":
            return execute_semantic_preference_query(
                spec,
                restrict_to_movie_ids=restrict_to_movie_ids,
                qdrant_client=qdrant_client,
            )
        return execute_semantic_dealbreaker_query(
            spec,
            restrict_to_movie_ids=restrict_to_movie_ids,
            qdrant_client=qdrant_client,
        )
    if endpoint == EndpointRoute.TRENDING:
        return execute_trending_query(
            restrict_to_movie_ids=restrict_to_movie_ids
        )
    raise ValueError(f"Unsupported endpoint for execution: {endpoint}")
