# Runtime driver for a single category handler on a single
# coverage_evidence entry.
#
# Responsibilities:
#   1. Short-circuit TRENDING to its deterministic executor (no LLM).
#   2. Short-circuit fit_quality=no_fit to an empty HandlerResult.
#   3. Build the per-category system prompt + user message, call the
#      handler LLM with the per-category output schema, retry once on
#      failure.
#   4. Extract fired (route, EndpointParameters) pairs from the LLM
#      output, bucketed by the category's handler shape
#      (SINGLE / MUTEX / TIERED / COMBO).
#   5. Classify each fired wrapper by (match_mode, polarity) into one
#      of the four return buckets, executing non-preference endpoints
#      in parallel and deferring preference specs as-is.
#   6. Consolidate scores additively across endpoints inside
#      inclusion_candidates / downrank_candidates and unify tmdb_ids
#      inside exclusion_ids.
#
# Scoped to a single category per invocation — fan-out across
# coverage_evidence entries is the orchestrator's job, one layer up.
#
# See search_improvement_planning/category_handler_planning.md for the
# design rationale (§"Handler return contract" and §"From LLM output
# to return buckets").

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)
from schemas.endpoint_parameters import EndpointParameters
from schemas.endpoint_result import EndpointResult
from schemas.enums import (
    CategoryName,
    EndpointRoute,
    FitQuality,
    HandlerBucket,
    MatchMode,
    Polarity,
)
from schemas.semantic_translation import SemanticEndpointParameters
from schemas.step_2 import CoverageEvidence, RequirementFragment
from search_v2.stage_3.category_handlers.handler_result import HandlerResult
from search_v2.stage_3.category_handlers.prompt_builder import (
    build_system_prompt,
    build_user_message,
)
from search_v2.stage_3.category_handlers.schema_factories import get_output_schema
from search_v2.stage_3.endpoint_executors import build_endpoint_coroutine
from search_v2.stage_3.trending_query_execution import execute_trending_query

logger = logging.getLogger(__name__)


# Per-call timeout budget for both the handler LLM call and each
# endpoint execution. Mirrors search_v2/stage_4/dispatch.py's
# TIMEOUT_SECONDS (currently 20.0) — kept as a sibling constant rather
# than imported because stage_4/dispatch pulls in modules that still
# reference the pre-unification semantic generators, and we don't want
# this file's import graph to depend on that cleanup.
TIMEOUT_SECONDS = 20.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_handler(
    *,
    category: CategoryName,
    target_entry: CoverageEvidence,
    raw_query: str,
    overall_query_intention_exploration: str,
    parent_fragment: RequirementFragment,
    sibling_fragments: list[RequirementFragment],
    qdrant_client: AsyncQdrantClient,
) -> HandlerResult:
    """Run one category handler on one coverage_evidence entry.

    Never raises. LLM double-failures and per-endpoint execution
    failures are soft-failed to empty buckets so a single bad handler
    cannot tank the whole query.
    """
    # Step 0a — TRENDING short-circuit. TRENDING has no LLM codepath
    # and the only SINGLE-bucket category that resolves to it is cat 9.
    # Must run even when fit_quality == no_fit because dispatch has
    # already validated the routing before reaching this layer.
    if category == CategoryName.TRENDING:
        result = await _run_trending()
        result.category = category
        return result

    # Step 0b — no_fit entries carry no actionable signal. Dispatch is
    # expected to filter these out before they reach this module, but
    # defense-in-depth keeps the handler honest if it ever slips past.
    if target_entry.fit_quality == FitQuality.NO_FIT:
        return HandlerResult(category=category)

    # Step 1 — build prompt + run LLM with a single retry.
    output = await _run_handler_llm(
        category=category,
        target_entry=target_entry,
        raw_query=raw_query,
        overall_query_intention_exploration=overall_query_intention_exploration,
        parent_fragment=parent_fragment,
        sibling_fragments=sibling_fragments,
    )
    if output is None:
        return HandlerResult(category=category)

    # Step 2 — extract the list of (route, wrapper) pairs the LLM
    # elected to fire. Zero fired endpoints is a valid, non-error
    # outcome (the LLM judged nothing to be a good fit).
    fired = _extract_fired_endpoints(category, output)
    if not fired:
        return HandlerResult(category=category)

    # Steps 3-5 — classify, execute in parallel, consolidate. The
    # final result also carries the category and the full fired
    # list (including preferences) for notebook / debug inspection.
    result = await _assemble_result(fired, qdrant_client=qdrant_client)
    result.category = category
    result.fired_endpoints = list(fired)
    return result


# ---------------------------------------------------------------------------
# Step 0a — trending
# ---------------------------------------------------------------------------


async def _run_trending() -> HandlerResult:
    # Trending has no candidate pool to restrict against here — the
    # handler is candidate-generating, so restrict_to_movie_ids=None.
    try:
        result: EndpointResult = await asyncio.wait_for(
            execute_trending_query(restrict_to_movie_ids=None),
            timeout=TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail by design
        logger.warning(
            "trending handler execution failed; returning empty result (%r)",
            exc,
        )
        return HandlerResult()

    inclusion = {sc.movie_id: sc.score for sc in result.scores}
    return HandlerResult(inclusion_candidates=inclusion)


# ---------------------------------------------------------------------------
# Step 1 — LLM call with single retry
# ---------------------------------------------------------------------------


async def _run_handler_llm(
    *,
    category: CategoryName,
    target_entry: CoverageEvidence,
    raw_query: str,
    overall_query_intention_exploration: str,
    parent_fragment: RequirementFragment,
    sibling_fragments: list[RequirementFragment],
) -> BaseModel | None:
    # Prompt build is cheap and deterministic — do it outside the retry
    # loop so a retry only re-attempts the network call.
    system_prompt = build_system_prompt(category)
    user_message = build_user_message(
        raw_query=raw_query,
        overall_query_intention_exploration=overall_query_intention_exploration,
        target_entry=target_entry,
        parent_fragment=parent_fragment,
        sibling_fragments=sibling_fragments,
    )
    response_format = get_output_schema(category)

    # Single retry covers transient failures: provider errors, timeout,
    # invalid structured output. Second failure returns None so the
    # caller emits an empty HandlerResult — a broken handler must not
    # propagate as an exception that tanks the whole query branch.
    for attempt in range(2):
        try:
            response, _, _ = await asyncio.wait_for(
                generate_llm_response_async(
                    provider=LLMProvider.OPENAI,
                    user_prompt=user_message,
                    system_prompt=system_prompt,
                    response_format=response_format,
                    model="gpt-5.4-mini",
                    reasoning_effort="none",
                    verbosity="low",
                ),
                timeout=TIMEOUT_SECONDS,
            )
            return response
        except Exception as exc:  # noqa: BLE001
            if attempt == 0:
                logger.warning(
                    "handler LLM call failed on first attempt; retrying "
                    "(category=%s, error=%r)",
                    category.name,
                    exc,
                )
                continue
            logger.error(
                "handler LLM call failed on retry; returning empty result "
                "(category=%s, error=%r)",
                category.name,
                exc,
            )
    return None


# ---------------------------------------------------------------------------
# Step 2 — extract fired (route, wrapper) pairs
# ---------------------------------------------------------------------------


def _extract_fired_endpoints(
    category: CategoryName,
    output: BaseModel,
) -> list[tuple[EndpointRoute, EndpointParameters]]:
    # Route extraction is keyed on the category's bucket. The output
    # schemas built in schema_factories follow a predictable shape per
    # bucket, so we can reach for named attributes without a Union
    # dispatch.
    bucket = category.bucket

    if bucket == HandlerBucket.SINGLE:
        if output.should_run_endpoint and output.endpoint_parameters is not None:
            # SINGLE-bucket category with a TRENDING endpoint was
            # already short-circuited in run_handler, so by here the
            # sole endpoint is guaranteed to be an LLM endpoint.
            route = category.endpoints[0]
            return [(route, output.endpoint_parameters)]
        return []

    if bucket in (HandlerBucket.MUTEX, HandlerBucket.TIERED):
        # endpoint_to_run is a Literal over route values plus the
        # sentinel "None" — see schema_factories._build_mutex_or_tiered.
        picked = output.endpoint_to_run
        if picked == "None" or output.endpoint_parameters is None:
            return []
        return [(EndpointRoute(picked), output.endpoint_parameters)]

    if bucket == HandlerBucket.COMBO:
        # per_endpoint_breakdown is a dynamically-built sub-model whose
        # field names are the route values (route.value). Every candidate
        # endpoint is present as a field; we iterate and collect the
        # ones the LLM elected to fire.
        breakdown = output.per_endpoint_breakdown
        fired: list[tuple[EndpointRoute, EndpointParameters]] = []
        for route_value in type(breakdown).model_fields.keys():
            entry = getattr(breakdown, route_value)
            if entry.should_run_endpoint and entry.endpoint_parameters is not None:
                fired.append((EndpointRoute(route_value), entry.endpoint_parameters))
        return fired

    # Should never reach — bucket is an exhaustive enum and every
    # variant is handled above. Raise loudly if a new bucket is ever
    # added without updating this dispatch.
    raise ValueError(f"Unhandled handler bucket: {bucket!r}")


# ---------------------------------------------------------------------------
# Steps 3-5 — classify, execute, consolidate
# ---------------------------------------------------------------------------


# Target-bucket sentinels for the _ExecutionPlan below. Keeping them as
# plain string tags (not an Enum) so the classification dispatch stays
# compact — the choices are stable and internal.
_INCLUSION = "inclusion_candidates"
_DOWNRANK = "downrank_candidates"
_EXCLUSION = "exclusion_ids"


async def _assemble_result(
    fired: list[tuple[EndpointRoute, EndpointParameters]],
    *,
    qdrant_client: AsyncQdrantClient,
) -> HandlerResult:
    result = HandlerResult()
    execution_plans: list[tuple[EndpointRoute, str]] = []
    coroutines: list[Coroutine[Any, Any, EndpointResult]] = []

    # Step 3 — classify each fired wrapper into a target bucket. Trait
    # positives are deferred (no execution), everything else goes into
    # the parallel execution batch.
    for route, wrapper in fired:
        target_bucket = _classify_wrapper(wrapper)

        if target_bucket is None:
            # TRAIT + POSITIVE → preference spec, deferred to the
            # orchestrator. The wrapper itself is appended raw so the
            # orchestrator can isinstance-route it to the right endpoint
            # once a candidate pool has been assembled.
            result.preference_specs.append(wrapper)
            continue

        execution_plans.append((route, target_bucket))
        coroutines.append(
            asyncio.wait_for(
                build_endpoint_coroutine(
                    route,
                    wrapper,
                    qdrant_client=qdrant_client,
                    restrict_to_movie_ids=None,
                ),
                timeout=TIMEOUT_SECONDS,
            )
        )

    if not coroutines:
        return result

    # Step 4 — fire every execution in parallel. return_exceptions=True
    # ensures a single endpoint timeout or error doesn't cancel its
    # siblings; we inspect each slot below.
    outcomes = await asyncio.gather(*coroutines, return_exceptions=True)

    # Step 5 — fold outcomes into the HandlerResult buckets with
    # additive score consolidation for score-bearing buckets and set
    # union for exclusion_ids.
    for (route, target_bucket), outcome in zip(execution_plans, outcomes):
        if isinstance(outcome, BaseException):
            logger.warning(
                "handler endpoint execution failed; skipping "
                "(route=%s, error=%r)",
                route.value,
                outcome,
            )
            continue

        _land_outcome(result, target_bucket, outcome)

    return result


def _classify_wrapper(wrapper: EndpointParameters) -> str | None:
    # Returns the target-bucket tag for a fired wrapper, or None when
    # the wrapper should be deferred as a preference spec (not executed
    # in-handler).
    #
    # The base 2x2 is:
    #   FILTER + POSITIVE → inclusion_candidates (execute)
    #   FILTER + NEGATIVE → exclusion_ids       (execute, drop scores)
    #   TRAIT  + POSITIVE → preference_specs    (defer — no execute)
    #   TRAIT  + NEGATIVE → downrank_candidates (execute)
    #
    # Semantic override: a semantic FILTER+NEGATIVE is *not* a hard
    # exclude. Similarity scores are soft by nature, so a "not scary"
    # match is a gradient downrank, not a crisp set removal. Route it
    # to downrank_candidates instead.
    match_mode = wrapper.match_mode
    polarity = wrapper.polarity

    if match_mode == MatchMode.FILTER and polarity == Polarity.POSITIVE:
        return _INCLUSION
    if match_mode == MatchMode.FILTER and polarity == Polarity.NEGATIVE:
        if isinstance(wrapper, SemanticEndpointParameters):
            return _DOWNRANK
        return _EXCLUSION
    if match_mode == MatchMode.TRAIT and polarity == Polarity.POSITIVE:
        return None  # preference spec — deferred
    if match_mode == MatchMode.TRAIT and polarity == Polarity.NEGATIVE:
        return _DOWNRANK

    # Exhaustive check — both enums are 2-valued, so reaching here means
    # an unexpected enum state. Surface the programmer error rather than
    # silently dropping the finding.
    raise ValueError(
        f"Unhandled match_mode/polarity combination: "
        f"match_mode={match_mode!r}, polarity={polarity!r}"
    )


def _land_outcome(
    result: HandlerResult,
    target_bucket: str,
    outcome: EndpointResult,
) -> None:
    # Mutates `result` in place. Score consolidation is additive on
    # tmdb_id collisions for inclusion / downrank; exclusion_ids is a
    # set so membership alone drives removal downstream.
    if target_bucket == _EXCLUSION:
        for sc in outcome.scores:
            result.exclusion_ids.add(sc.movie_id)
        return

    # Only _INCLUSION and _DOWNRANK reach here. Exhaustiveness is
    # enforced by _classify_wrapper — any unexpected tag would be a
    # programmer error upstream, not a runtime hazard worth catching.
    bucket = (
        result.downrank_candidates
        if target_bucket == _DOWNRANK
        else result.inclusion_candidates
    )
    for sc in outcome.scores:
        bucket[sc.movie_id] = bucket.get(sc.movie_id, 0.0) + sc.score


