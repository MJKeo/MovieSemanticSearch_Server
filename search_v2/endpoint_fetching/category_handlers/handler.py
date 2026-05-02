# Runtime driver for a single category handler on a single
# (CategoryCall, Trait) pair.
#
# Responsibilities:
#   1. Short-circuit no-LLM buckets: deterministic categories run
#      their codepaths, explicit no-op categories return empty.
#   2. Build the per-category system prompt + user message from the
#      CategoryCall (retrieval_intent + expressions are the entire
#      LLM input), call the handler LLM with the per-category output
#      schema, retry once on failure.
#   3. Extract fired (route, EndpointParameters) pairs from the LLM
#      output, bucketed by the category's handler shape.
#   4. Classify each fired wrapper by the parent Trait's committed
#      role/polarity into one of
#      the four return buckets, executing non-preference endpoints
#      in parallel and deferring preference specs as-is.
#   5. Consolidate scores additively across endpoints inside
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
from schemas.entity_translation import (
    CharacterProminenceMode,
    CharacterQuerySpec,
    CharacterTarget,
)
from schemas.enums import (
    EndpointRoute,
    HandlerBucket,
    Polarity,
    Role,
)
from schemas.franchise_translation import (
    FranchiseEndpointParameters,
    FranchiseQuerySpec,
)
from schemas.media_type_translation import MediaTypeEndpointParameters
from schemas.trait_category import CategoryName
from schemas.semantic_translation import (
    CarverSemanticEndpointParameters,
    CarverSemanticEndpointSubintentParameters,
    QualifierSemanticEndpointParameters,
    QualifierSemanticEndpointSubintentParameters,
)
from schemas.step_2 import Trait
from schemas.step_3 import CategoryCall
from search_v2.endpoint_fetching.category_handlers.handler_result import HandlerResult
from search_v2.endpoint_fetching.category_handlers.media_type_router import (
    build_media_type_query_spec,
)
from search_v2.endpoint_fetching.category_handlers.prompt_builder import (
    build_system_prompt,
    build_user_message,
)
from search_v2.endpoint_fetching.category_handlers.schema_factories import get_output_schema
from search_v2.endpoint_fetching.endpoint_executors import build_endpoint_coroutine
from search_v2.endpoint_fetching.trending_query_execution import execute_trending_query

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
    supplies the pre-committed ``role`` and ``polarity`` used for
    runtime bucketing. The LLM does not infer either. Salience is
    intentionally not read; it only affects cross-trait reranking,
    which is out of scope for this stage.

    LLM double-failures and per-endpoint execution failures are
    soft-failed to empty buckets so a single bad handler cannot tank
    the whole query. Configuration / programmer errors still raise.
    """
    category = category_call.category

    # Step 0a — explicit no-op. These categories are valid routing
    # sinks but have no backing endpoint yet, so they intentionally
    # produce no candidates rather than reaching prompt/schema lookup.
    if category.bucket is HandlerBucket.EXPLICIT_NO_OP:
        return HandlerResult(category=category)

    # Step 0b — deterministic no-LLM codepaths. Keep this bucket-level
    # guard explicit so any future no-LLM category must register a
    # deterministic handler here rather than accidentally falling into
    # the prompt builder.
    if category.bucket is HandlerBucket.NO_LLM_PURE_CODE:
        if category is CategoryName.TRENDING:
            result = await _run_trending()
            result.category = category
            return result
        if category is CategoryName.MEDIA_TYPE:
            result = await _run_media_type(category_call, trait, qdrant_client)
            result.category = category
            return result
        raise ValueError(
            f"Unhandled no-LLM category: CategoryName.{category.name}. "
            f"Add a deterministic handler before routing this category."
        )

    # Step 1 — build prompt + run LLM with a single retry.
    output = await _run_handler_llm(
        category=category,
        category_call=category_call,
        role=trait.role,
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
    result = await _assemble_result(
        fired,
        trait=trait,
        qdrant_client=qdrant_client,
    )
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
# Step 0a.2 — media type
# ---------------------------------------------------------------------------


async def _run_media_type(
    category_call: CategoryCall,
    trait: Trait,
    qdrant_client: AsyncQdrantClient,
) -> HandlerResult:
    spec = build_media_type_query_spec(category_call)
    if spec is None:
        return HandlerResult()

    wrapper = MediaTypeEndpointParameters(
        parameters=spec,
    )
    fired = [(EndpointRoute.MEDIA_TYPE, wrapper)]
    result = await _assemble_result(
        fired,
        trait=trait,
        qdrant_client=qdrant_client,
    )
    result.fired_endpoints = list(fired)
    return result


# ---------------------------------------------------------------------------
# Step 1 — LLM call with single retry
# ---------------------------------------------------------------------------


async def _run_handler_llm(
    *,
    category: CategoryName,
    category_call: CategoryCall,
    role: Role,
) -> BaseModel | None:
    # Prompt build is cheap and deterministic — do it outside the retry
    # loop so a retry only re-attempts the network call.
    system_prompt = build_system_prompt(category)
    user_message = build_user_message(category_call)
    response_format = get_output_schema(category, role)

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


# Single-endpoint buckets share the same output schema shape: a
# `should_run_endpoint` gate plus an `endpoint_parameters` slot for
# the one wrapper. See schema_factories._build_single.
_SINGLE_ENDPOINT_BUCKETS: frozenset[HandlerBucket] = frozenset({
    HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
    HandlerBucket.SINGLE_METADATA_ENDPOINT,
})


# Buckets whose output schema carries one Optional `<route>_parameters`
# field per candidate endpoint. The fired set is whichever fields are
# non-null. Reasoning fields (intent prose, opportunity lists) live
# alongside but do not need to be inspected here. See schema_factories
# _build_preferred_fallback / _build_semantic_with_augmentation /
# _build_suitability_combo.
_PER_ROUTE_PARAMETER_BUCKETS: frozenset[HandlerBucket] = frozenset({
    HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK,
    HandlerBucket.SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT,
    HandlerBucket.AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST,
})


def _extract_fired_endpoints(
    category: CategoryName,
    output: BaseModel,
) -> list[tuple[EndpointRoute, EndpointParameters]]:
    # Route extraction is keyed on the category's bucket. The output
    # schemas built in schema_factories follow a predictable shape per
    # bucket family, so we can dispatch on bucket without unioning.
    bucket = category.bucket

    if bucket in _SINGLE_ENDPOINT_BUCKETS:
        if output.should_run_endpoint and output.endpoint_parameters is not None:
            # Single-endpoint categories with a TRENDING endpoint are
            # short-circuited in run_handler, so the sole endpoint is
            # guaranteed to be an LLM endpoint by the time we reach here.
            route = category.endpoints[0]
            return [(route, output.endpoint_parameters)]
        return []

    if bucket in _PER_ROUTE_PARAMETER_BUCKETS:
        # Walk fields named '<route>_parameters' and collect the ones
        # the LLM filled. Field name → EndpointRoute via EndpointRoute(
        # route_value); the `_parameters` suffix is dropped first.
        fired: list[tuple[EndpointRoute, EndpointParameters]] = []
        for field_name in type(output).model_fields.keys():
            if not field_name.endswith("_parameters"):
                continue
            value = getattr(output, field_name)
            if value is None:
                continue
            route_value = field_name.removesuffix("_parameters")
            fired.append((EndpointRoute(route_value), value))
        return fired

    if bucket is HandlerBucket.CHARACTER_FRANCHISE_FANOUT:
        # The fanout schema (CharacterFranchiseFanoutSchema) carries
        # one shared referent identification plus two parallel form
        # lists rather than per-route parameter wrappers — the bucket's
        # design intent is "identify the referent once, fan out to two
        # retrievals." Translate each non-empty form list into the
        # ordinary per-endpoint payload the rest of the pipeline
        # consumes, then let the standard classify/execute/consolidate
        # path handle scoring exactly as it does for any other multi-
        # route bucket. Either form list may be empty (the LLM judged
        # that path not applicable for this referent); both empty is a
        # valid zero-fired outcome.
        return _fanout_to_fired_endpoints(output)

    # NO_LLM_PURE_CODE / EXPLICIT_NO_OP buckets do not invoke the
    # handler LLM and therefore should never reach extraction. Any
    # other bucket appearing here is a programmer error.
    raise ValueError(f"Unhandled handler bucket: {bucket!r}")


# Stub strings for the per-target / per-spec exploration prose that
# CharacterQuerySpec and FranchiseQuerySpec require. The fanout schema
# replaces those LLM-authored reasoning slots with a single shared
# `referent_form_exploration`, so when we synthesize the downstream
# specs the only thing we have to fill the remaining exploration slots
# is the same referent prose (used for query_exploration /
# character_exploration / request_overview) plus a fixed sentinel for
# prominence_exploration (the fanout schema does not commit a
# centrality reading — DEFAULT prominence is the safe fallback).
# Executors do not read these strings; they exist purely as LLM
# scaffolding on the spec models, so stubs preserve schema validity
# without affecting retrieval behavior.
_FANOUT_PROMINENCE_EXPLORATION_STUB = (
    "no centrality signal — fanout retrieval does not commit a "
    "separate prominence reading."
)


def _fanout_to_fired_endpoints(
    output: BaseModel,
) -> list[tuple[EndpointRoute, EndpointParameters]]:
    # Adapter for HandlerBucket.CHARACTER_FRANCHISE_FANOUT. Reads the
    # shared CharacterFranchiseFanoutSchema and emits up to two ordinary
    # (route, wrapper) pairs so the caller can hand them to the same
    # classify / execute / consolidate path every other multi-route
    # bucket uses. See the call site in _extract_fired_endpoints for
    # rationale.
    referent: str = output.referent_form_exploration
    fired: list[tuple[EndpointRoute, EndpointParameters]] = []

    # Character path — collapses every variant into a single target
    # because the fanout schema treats the referent as one entity. Were
    # multiple distinct characters meant, the routing layer would have
    # produced multiple traits / category calls upstream rather than one
    # shared form list here.
    character_forms = list(output.character_forms)
    if character_forms:
        character_spec = CharacterQuerySpec(
            query_exploration=referent,
            targets=[
                CharacterTarget(
                    character_exploration=referent,
                    forms=character_forms,
                    prominence_exploration=_FANOUT_PROMINENCE_EXPLORATION_STUB,
                    prominence_mode=CharacterProminenceMode.DEFAULT,
                )
            ],
        )
        fired.append((EndpointRoute.ENTITY, character_spec))

    # Franchise path — the franchise endpoint is name-axis-driven for
    # the fanout case. lineage_position / structural_flags / launch_
    # scope / prefer_lineage are deliberately left unset; the fanout
    # schema commits no narrative-position or structural reading, so
    # franchise_names alone is the correct projection.
    franchise_forms = list(output.franchise_forms)
    if franchise_forms:
        franchise_wrapper = FranchiseEndpointParameters(
            parameters=FranchiseQuerySpec(
                request_overview=referent,
                franchise_names=franchise_forms,
            ),
        )
        fired.append((EndpointRoute.FRANCHISE_STRUCTURE, franchise_wrapper))

    return fired


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
    trait: Trait,
    qdrant_client: AsyncQdrantClient,
) -> HandlerResult:
    result = HandlerResult()
    execution_plans: list[tuple[EndpointRoute, str]] = []
    coroutines: list[Coroutine[Any, Any, EndpointResult]] = []

    # Step 3 — classify each fired wrapper into a target bucket. Role
    # and polarity come from the parent Trait, not the endpoint
    # wrapper. Qualifier positives are deferred (no execution),
    # everything else goes into
    # the parallel execution batch.
    for route, wrapper in fired:
        target_bucket = _classify_wrapper(wrapper, trait)

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
                    role=trait.role,
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


def _classify_wrapper(wrapper: EndpointParameters, trait: Trait) -> str | None:
    # Returns the target-bucket tag for a fired wrapper, or None when
    # the wrapper should be deferred as a preference spec (not executed
    # in-handler).
    #
    # The base 2x2 is:
    #   CARVER    + POSITIVE → inclusion_candidates (execute)
    #   CARVER    + NEGATIVE → exclusion_ids       (execute, drop scores)
    #   QUALIFIER + POSITIVE → preference_specs    (defer — no execute)
    #   QUALIFIER + NEGATIVE → downrank_candidates (execute)
    #
    # Semantic override: a semantic CARVER+NEGATIVE is *not* a hard
    # exclude. Similarity scores are soft by nature, so a "not scary"
    # match is a gradient downrank, not a crisp set removal. Route it
    # to downrank_candidates instead.
    role = trait.role
    polarity = trait.polarity

    if role == Role.CARVER and polarity == Polarity.POSITIVE:
        return _INCLUSION
    if role == Role.CARVER and polarity == Polarity.NEGATIVE:
        if isinstance(
            wrapper,
            (
                CarverSemanticEndpointParameters,
                QualifierSemanticEndpointParameters,
                CarverSemanticEndpointSubintentParameters,
                QualifierSemanticEndpointSubintentParameters,
            ),
        ):
            return _DOWNRANK
        return _EXCLUSION
    if role == Role.QUALIFIER and polarity == Polarity.POSITIVE:
        return None  # preference spec — deferred
    if role == Role.QUALIFIER and polarity == Polarity.NEGATIVE:
        return _DOWNRANK

    # Exhaustive check — both enums are 2-valued, so reaching here means
    # an unexpected enum state. Surface the programmer error rather than
    # silently dropping the finding.
    raise ValueError(
        f"Unhandled role/polarity combination: "
        f"role={role!r}, polarity={polarity!r}"
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
