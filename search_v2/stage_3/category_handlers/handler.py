# Runtime driver for a single category handler on a single
# (CategoryCall, Trait) pair.
#
# Responsibilities:
#   1. Short-circuit TRENDING and MEDIA_TYPE to deterministic
#      codepaths (no LLM).
#   2. Build the per-category system prompt + user message from the
#      CategoryCall (retrieval_intent + expressions are the entire
#      LLM input), call the handler LLM with the per-category output
#      schema, retry once on failure.
#   3. Extract fired (route, EndpointParameters) pairs from the LLM
#      output, bucketed by the category's handler shape
#      (SINGLE / MUTEX / TIERED / COMBO).
#   4. Stamp match_mode and polarity onto each fired wrapper from
#      the parent Trait's pre-committed role/polarity. The LLM does
#      not infer these — they are upstream commitments.
#   5. Classify each fired wrapper by (match_mode, polarity) into one
#      of the four return buckets, executing non-preference endpoints
#      in parallel and deferring preference specs as-is.
#   6. Consolidate scores additively across endpoints inside
#      inclusion_candidates / downrank_candidates and unify tmdb_ids
#      inside exclusion_ids.
#
# Scoped to one CategoryCall per invocation — fan-out across calls
# (a trait may produce multiple) and across traits is the
# orchestrator's job, one layer up.
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
    EndpointRoute,
    HandlerBucket,
    MatchMode,
    Polarity,
)
from schemas.trait_category import CategoryName
from schemas.semantic_translation import SemanticEndpointParameters
from schemas.step_2 import Trait
from schemas.step_3 import CategoryCall
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
    category_call: CategoryCall,
    trait: Trait,
    qdrant_client: AsyncQdrantClient,
) -> HandlerResult:
    """Run one category handler on one CategoryCall in the context of
    its parent Trait.

    The CategoryCall supplies the routing (``category``) and the LLM
    inputs (``retrieval_intent`` + ``expressions``). The Trait
    supplies the pre-committed ``role`` and ``polarity`` that get
    stamped onto every fired wrapper — the LLM does not infer either.
    Salience is intentionally not read; it only affects cross-trait
    reranking, which is out of scope for this stage.

    Never raises. LLM double-failures and per-endpoint execution
    failures are soft-failed to empty buckets so a single bad handler
    cannot tank the whole query.
    """
    category = category_call.category

    # Step 0a — TRENDING short-circuit. TRENDING has no LLM codepath
    # and the only SINGLE-bucket category that resolves to it is cat 9.
    if category == CategoryName.TRENDING:
        result = await _run_trending()
        result.category = category
        return result

    # Step 0a.2 — MEDIA_TYPE short-circuit. MEDIA_TYPE will be routed
    # deterministically by code (matching surface phrases against the
    # ReleaseFormat enum) rather than through the LLM handler. The
    # deterministic routing path is not yet wired up; until it lands,
    # soft-fail to an empty result. Reaching the LLM codepath would
    # crash because MEDIA_TYPE is in prompt_builder._ENDPOINT_PROMPTLESS
    # and the MEDIA_TYPE category routes only to the MEDIA_TYPE
    # endpoint, tripping the "no LLM-wrapper endpoints" raise in
    # build_system_prompt.
    if category == CategoryName.MEDIA_TYPE:
        return HandlerResult(category=category)

    # Step 1 — build prompt + run LLM with a single retry.
    output = await _run_handler_llm(
        category=category,
        category_call=category_call,
    )
    if output is None:
        return HandlerResult(category=category)

    # Step 2 — extract the list of (route, wrapper) pairs the LLM
    # elected to fire. Zero fired endpoints is a valid, non-error
    # outcome (the LLM judged nothing to be a good fit).
    fired = _extract_fired_endpoints(category, output)
    if not fired:
        return HandlerResult(category=category)

    # Step 3 — stamp the trait's pre-committed role/polarity onto
    # each fired wrapper. The LLM emits the rest of the parameters;
    # match_mode and polarity are upstream commitments.
    fired = _stamp_role_and_polarity(fired, trait)

    # Steps 4-6 — classify, execute in parallel, consolidate. The
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
    category_call: CategoryCall,
) -> BaseModel | None:
    # Prompt build is cheap and deterministic — do it outside the retry
    # loop so a retry only re-attempts the network call.
    system_prompt = build_system_prompt(category)
    user_message = build_user_message(category_call)
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
# Step 3 — stamp role + polarity from the parent Trait
# ---------------------------------------------------------------------------


# Trait.role and Trait.polarity are pre-committed string literals; the
# wrappers expect the corresponding enum values. These mappings are
# the single point of translation so the rest of the handler can
# treat the wrapper fields as authoritative.
_ROLE_TO_MATCH_MODE: dict[str, MatchMode] = {
    "carver": MatchMode.FILTER,
    "qualifier": MatchMode.TRAIT,
}
_POLARITY_TO_ENUM: dict[str, Polarity] = {
    "positive": Polarity.POSITIVE,
    "negative": Polarity.NEGATIVE,
}


def _stamp_role_and_polarity(
    fired: list[tuple[EndpointRoute, EndpointParameters]],
    trait: Trait,
) -> list[tuple[EndpointRoute, EndpointParameters]]:
    # Overwrite whatever match_mode / polarity the LLM emitted with
    # the Trait's committed values. Pydantic model_copy returns a new
    # wrapper rather than mutating in place, so the LLM-emitted
    # objects stay untouched (useful when fired_endpoints is exposed
    # for debug inspection).
    match_mode = _ROLE_TO_MATCH_MODE[trait.role]
    polarity = _POLARITY_TO_ENUM[trait.polarity]
    return [
        (
            route,
            wrapper.model_copy(
                update={"match_mode": match_mode, "polarity": polarity}
            ),
        )
        for route, wrapper in fired
    ]


# ---------------------------------------------------------------------------
# Steps 4-6 — classify, execute, consolidate
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

