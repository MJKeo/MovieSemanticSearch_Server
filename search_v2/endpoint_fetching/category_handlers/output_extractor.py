# Per-bucket extraction of fired (route, EndpointParameters) pairs from
# a category-handler LLM output.
#
# Responsibilities:
#   - Walk the per-category output schema produced by the handler LLM
#     and surface the (EndpointRoute, EndpointParameters) pairs the
#     LLM elected to fire.
#   - Adapt the CHARACTER_FRANCHISE_FANOUT shared schema into the
#     same per-route payload shape every other multi-route bucket
#     uses, so callers can route uniformly.
#
# Called by the orchestrator after a successful handler-LLM call. Pure
# function: no side effects, no I/O.

from __future__ import annotations

from pydantic import BaseModel

from schemas.endpoint_parameters import EndpointParameters
from schemas.entity_translation import (
    CharacterProminenceMode,
    CharacterQuerySpec,
    CharacterTarget,
)
from schemas.enums import EndpointRoute, HandlerBucket
from schemas.franchise_translation import (
    FranchiseEndpointParameters,
    FranchiseQuerySpec,
)
from schemas.trait_category import CategoryName


# Single-endpoint buckets share the same output schema shape: a
# `should_run_endpoint` gate plus an `endpoint_parameters` slot for
# the one wrapper. See schema_factories._build_single.
_SINGLE_ENDPOINT_BUCKETS: frozenset[HandlerBucket] = frozenset({
    HandlerBucket.SINGLE_NON_METADATA_ENDPOINT,
    HandlerBucket.SINGLE_METADATA_ENDPOINT,
})


# Buckets whose output schema carries one Optional `<route>_parameters`
# field per candidate endpoint. The fired set is whichever fields are
# non-null. Reasoning fields (`<route>_walk` blocks, coverage_exploration,
# coverage_assignments) live alongside but do not need to be inspected
# here — the suffix filter below picks up only `*_parameters`. See
# schema_factories._build_walk_then_commit.
_PER_ROUTE_PARAMETER_BUCKETS: frozenset[HandlerBucket] = frozenset({
    HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK,
    HandlerBucket.SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT,
    HandlerBucket.AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST,
})


# The fanout schema carries two parallel walks rather than the
# per-target / per-spec exploration prose the downstream specs were
# designed around. When we synthesize CharacterQuerySpec and
# FranchiseQuerySpec we wire each side's walk into the equivalent
# exploration slot on the downstream spec; query_exploration on
# CharacterQuerySpec restates the character-side context (single
# target by design) and prominence_exploration is filled with a fixed
# sentinel because the fanout schema does not commit a centrality
# reading — DEFAULT prominence is the safe fallback.
# Executors do not read these strings; they exist purely as LLM
# scaffolding on the spec models, so a stub or a copy preserves
# schema validity without affecting retrieval behavior.
_FANOUT_PROMINENCE_EXPLORATION_STUB = (
    "no centrality signal — fanout retrieval does not commit a "
    "separate prominence reading."
)

_FANOUT_CHARACTER_QUERY_EXPLORATION_STUB = (
    "single referent — fanout retrieval emits one CharacterTarget "
    "per call by design."
)


def extract_fired_endpoints(
    category: CategoryName,
    output: BaseModel,
) -> list[tuple[EndpointRoute, EndpointParameters]]:
    """Pull the (route, wrapper) pairs the LLM elected to fire.

    Dispatches by `category.bucket`. The bucket determines the shape
    of `output` (built in schema_factories) and therefore where the
    fired endpoints live on it.

    Returns an empty list when the LLM judged nothing to fire — a
    valid, non-error outcome.
    """
    bucket = category.bucket

    if bucket in _SINGLE_ENDPOINT_BUCKETS:
        if output.should_run_endpoint and output.endpoint_parameters is not None:
            # Single-endpoint categories with a TRENDING endpoint are
            # short-circuited upstream of the handler LLM, so the sole
            # endpoint here is guaranteed to be an LLM endpoint.
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
        # consumes. Either form list may be empty (the LLM judged that
        # path not applicable for this referent); both empty is a
        # valid zero-fired outcome.
        return _fanout_to_fired_endpoints(output)

    # NO_LLM_PURE_CODE / EXPLICIT_NO_OP buckets do not invoke the
    # handler LLM and therefore should never reach extraction. Any
    # other bucket appearing here is a programmer error.
    raise ValueError(f"Unhandled handler bucket: {bucket!r}")


def _fanout_to_fired_endpoints(
    output: BaseModel,
) -> list[tuple[EndpointRoute, EndpointParameters]]:
    # Adapter for HandlerBucket.CHARACTER_FRANCHISE_FANOUT. Reads the
    # shared CharacterFranchiseFanoutSchema and emits up to two ordinary
    # (route, wrapper) pairs so callers can route them through the same
    # pipeline as any other multi-route bucket.
    fired: list[tuple[EndpointRoute, EndpointParameters]] = []

    # Character path — collapses every variant into a single target
    # because the fanout schema treats the referent as one entity. Were
    # multiple distinct characters meant, the routing layer would have
    # produced multiple traits / category calls upstream rather than
    # one shared form list here.
    character_forms = list(output.character_forms)
    if character_forms:
        character_spec = CharacterQuerySpec(
            query_exploration=_FANOUT_CHARACTER_QUERY_EXPLORATION_STUB,
            targets=[
                CharacterTarget(
                    character_exploration=output.character_form_exploration,
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
                request_overview=output.franchise_form_exploration,
                franchise_names=franchise_forms,
            ),
        )
        fired.append((EndpointRoute.FRANCHISE_STRUCTURE, franchise_wrapper))

    return fired
