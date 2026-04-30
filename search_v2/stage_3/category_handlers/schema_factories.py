# Per-category handler output schemas.
#
# One Pydantic class per CategoryName, built from the category's
# bucket (single / mutex / tiered / combo) and endpoint tuple. The
# schemas are what the step-3 category-handler LLM produces via
# structured output; the shape is bucket-level and the endpoint-
# specific atoms come from the sibling endpoint_registry module.
#
# Schemas are eagerly built at module import so any misconfiguration
# (missing wrapper, invalid field name, JSON-schema size issue) fails
# loudly at startup instead of on first request. Access via
# get_output_schema(category).
#
# See search_improvement_planning/category_handler_planning.md
# §"Handler LLM output schema per bucket" and §"Building output
# schemas dynamically" for the design rationale.

from __future__ import annotations

from typing import Callable, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, create_model

from schemas.endpoint_parameters import EndpointParameters
from search_v2.stage_3.category_handlers.endpoint_registry import ROUTE_TO_WRAPPER
from schemas.enums import EndpointRoute, HandlerBucket
from schemas.trait_category import CategoryName


# Every dynamically-generated class inherits this so OpenAI structured
# output gets additionalProperties: false on every sub-object.
class _HandlerOutputBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Shared Field descriptions ─────────────────────────────────────
# Module-level constants prevent wording drift across the four bucket
# factories. Tuned for small / instruction-tuned models: phrasal cues,
# anti-failure-mode framing, concrete direction over abstract framing.

_REQUIREMENT_ASPECTS_DESC = (
    "Break the fragment into discrete sub-requirements before deciding "
    "anything else. One entry per distinguishable ask. If the fragment "
    "is simple, a single aspect is fine — do not invent sub-parts that "
    "are not actually there."
)

_ASPECT_DESCRIPTION_DESC = (
    "One concrete thing the user is asking for, stated in their own "
    "terms. Not a summary of the whole fragment, not a generalization."
)

_RELATION_TO_ENDPOINT_DESC = (
    "What this endpoint can concretely do toward satisfying the aspect "
    "— specific vocabulary it covers, vector space it embeds, metadata "
    "column it predicates on. Avoid vague 'it should work'."
)

_COVERAGE_GAPS_DESC = (
    "What the aspect needs that the endpoint cannot provide, or null "
    "when the endpoint fully covers it. Be specific about the gap."
)

_ENDPOINT_COVERAGE_DESC = (
    "For each candidate endpoint, a short note on how it could cover "
    "this aspect. Address every candidate — even briefly."
)

_BEST_ENDPOINT_DESC = (
    "Which candidate endpoint best covers this aspect. One pick per "
    "aspect — the one whose capabilities most directly fit the ask."
)

_BEST_ENDPOINT_GAPS_DESC = (
    "What the chosen endpoint still cannot cover for this aspect, or "
    "null. If every candidate leaves meaningful gaps, say so here."
)

_SHOULD_RUN_ENDPOINT_DESC = (
    "True when this endpoint should fire. False is a valid and "
    "preferred answer when the fragment does not cleanly fit — "
    "do not invent parameters for a bad match."
)

_ENDPOINT_PARAMETERS_SINGLE_DESC = (
    "The endpoint's parameter payload, carrying role and polarity. "
    "Leave null when should_run_endpoint is false."
)

_ENDPOINT_PARAMETERS_MUTEX_DESC = (
    "The chosen endpoint's parameter payload, carrying role and "
    "polarity. Leave null when endpoint_to_run is 'None'."
)

_ENDPOINT_TO_RUN_DESC = (
    "The single endpoint that best fits — or 'None' when no candidate "
    "is a genuine fit. 'None' is valid; do not force a bad match."
)

_PERFORMANCE_VS_BIAS_DESC = (
    "Short analysis of how the candidate endpoints stack up against "
    "the preference bias (earlier entries in the endpoint tuple are "
    "biased-toward). State whether one is clearly better on its own "
    "merits, or whether the bias is what breaks a close call."
)

_OVERALL_ENDPOINT_FITS_DESC = (
    "Brief synthesis of which endpoints fit, why, and how they "
    "complement each other. One or two sentences."
)

_COMBO_PER_ENDPOINT_DESC = (
    "Address every candidate endpoint explicitly. For each one, "
    "decide whether it should fire and fill its parameter payload "
    "if so. Do not skip an endpoint."
)


# ── Naming helper ─────────────────────────────────────────────────


def _pascal(name: str) -> str:
    # CategoryName.name is UPPER_SNAKE_CASE (e.g. "CREDIT_TITLE"); the
    # codebase convention for Pydantic models is PascalCase.
    return "".join(part.capitalize() for part in name.split("_"))


# ── Per-aspect sub-model factories ────────────────────────────────
# Each bucket has a slightly different requirement_aspects entry
# shape. Built per-category because the multi-endpoint variants carry
# a Literal scoped to that category's endpoints.


def _build_single_aspect_model(category_name: str) -> type[BaseModel]:
    return create_model(
        f"{_pascal(category_name)}RequirementAspect",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        aspect_description=(str, Field(..., description=_ASPECT_DESCRIPTION_DESC)),
        relation_to_endpoint=(str, Field(..., description=_RELATION_TO_ENDPOINT_DESC)),
        coverage_gaps=(Optional[str], Field(default=None, description=_COVERAGE_GAPS_DESC)),
    )


def _build_multi_aspect_model(
    category_name: str,
    endpoint_values: tuple[str, ...],
) -> type[BaseModel]:
    # Mutex/Tiered requirement-aspect: includes per-endpoint coverage
    # and a best_endpoint pick scoped to the category's candidates.
    best_endpoint_type = Literal[endpoint_values]
    return create_model(
        f"{_pascal(category_name)}RequirementAspect",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        aspect_description=(str, Field(..., description=_ASPECT_DESCRIPTION_DESC)),
        endpoint_coverage=(str, Field(..., description=_ENDPOINT_COVERAGE_DESC)),
        best_endpoint=(best_endpoint_type, Field(..., description=_BEST_ENDPOINT_DESC)),
        best_endpoint_gaps=(Optional[str], Field(default=None, description=_BEST_ENDPOINT_GAPS_DESC)),
    )


def _build_combo_aspect_model(category_name: str) -> type[BaseModel]:
    return create_model(
        f"{_pascal(category_name)}RequirementAspect",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        aspect_description=(str, Field(..., description=_ASPECT_DESCRIPTION_DESC)),
        endpoint_coverage=(str, Field(..., description=_ENDPOINT_COVERAGE_DESC)),
    )


# ── Helpers ───────────────────────────────────────────────────────


def _resolve_wrappers(
    endpoints: tuple[EndpointRoute, ...],
) -> tuple[tuple[EndpointRoute, type[EndpointParameters]], ...]:
    # Returns (route, wrapper) pairs, skipping routes that map to None
    # in the registry (e.g. TRENDING). Preserves priority order.
    pairs: list[tuple[EndpointRoute, type[EndpointParameters]]] = []
    for route in endpoints:
        wrapper = ROUTE_TO_WRAPPER[route]
        if wrapper is not None:
            pairs.append((route, wrapper))
    return tuple(pairs)


# ── Bucket factories ──────────────────────────────────────────────


def _build_single(
    endpoints: tuple[EndpointRoute, ...],
    name: str,
) -> type[BaseModel] | None:
    # Categories whose only endpoint has no LLM wrapper (TRENDING) get
    # no schema — they run through a deterministic code path instead.
    pairs = _resolve_wrappers(endpoints)
    if not pairs:
        return None
    wrapper = pairs[0][1]

    aspect_model = _build_single_aspect_model(name)
    return create_model(
        f"{_pascal(name)}Output",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        requirement_aspects=(list[aspect_model], Field(..., description=_REQUIREMENT_ASPECTS_DESC)),
        should_run_endpoint=(bool, Field(..., description=_SHOULD_RUN_ENDPOINT_DESC)),
        endpoint_parameters=(Optional[wrapper], Field(default=None, description=_ENDPOINT_PARAMETERS_SINGLE_DESC)),
    )


def _build_mutex_or_tiered(
    endpoints: tuple[EndpointRoute, ...],
    name: str,
    *,
    include_bias_analysis: bool,
) -> type[BaseModel] | None:
    pairs = _resolve_wrappers(endpoints)
    if len(pairs) < 2:
        raise ValueError(
            f"Category {name} is declared mutex/tiered but resolves to "
            f"{len(pairs)} candidate wrapper(s). Needs at least 2."
        )

    endpoint_values = tuple(route.value for route, _ in pairs)
    wrappers = tuple(wrapper for _, wrapper in pairs)

    endpoint_to_run_type = Literal[endpoint_values + ("None",)]
    aspect_model = _build_multi_aspect_model(name, endpoint_values)
    wrapper_union = Union[wrappers]  # type: ignore[valid-type]

    fields: dict[str, tuple] = {
        "requirement_aspects": (
            list[aspect_model],
            Field(..., description=_REQUIREMENT_ASPECTS_DESC),
        ),
    }
    if include_bias_analysis:
        # Tiered-only: retrospective field sitting between aspects and
        # the final pick. Placed here so the model reasons about the
        # bias BEFORE committing to endpoint_to_run.
        fields["performance_vs_bias_analysis"] = (
            str,
            Field(..., description=_PERFORMANCE_VS_BIAS_DESC),
        )
    fields["endpoint_to_run"] = (
        endpoint_to_run_type,
        Field(..., description=_ENDPOINT_TO_RUN_DESC),
    )
    fields["endpoint_parameters"] = (
        Optional[wrapper_union],
        Field(default=None, description=_ENDPOINT_PARAMETERS_MUTEX_DESC),
    )

    return create_model(
        f"{_pascal(name)}Output",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        **fields,
    )


def _build_mutex(
    endpoints: tuple[EndpointRoute, ...], name: str,
) -> type[BaseModel] | None:
    return _build_mutex_or_tiered(endpoints, name, include_bias_analysis=False)


def _build_tiered(
    endpoints: tuple[EndpointRoute, ...], name: str,
) -> type[BaseModel] | None:
    return _build_mutex_or_tiered(endpoints, name, include_bias_analysis=True)


def _build_combo(
    endpoints: tuple[EndpointRoute, ...],
    name: str,
) -> type[BaseModel] | None:
    pairs = _resolve_wrappers(endpoints)
    if not pairs:
        return None

    # One nested breakdown entry per endpoint, keyed by endpoint
    # value. The enumerated-not-freeform shape is the planning doc's
    # hard requirement — forces the LLM to address every endpoint
    # explicitly instead of silently omitting.
    breakdown_fields: dict[str, tuple] = {}
    for route, wrapper in pairs:
        entry_model = create_model(
            f"{_pascal(name)}{_pascal(route.value)}Breakdown",
            __base__=_HandlerOutputBase,
            __module__=__name__,
            should_run_endpoint=(bool, Field(..., description=_SHOULD_RUN_ENDPOINT_DESC)),
            endpoint_parameters=(
                Optional[wrapper],
                Field(default=None, description=_ENDPOINT_PARAMETERS_SINGLE_DESC),
            ),
        )
        breakdown_fields[route.value] = (
            entry_model,
            Field(
                ...,
                description=(
                    f"Decision and (if firing) parameter payload for "
                    f"the {route.value} endpoint."
                ),
            ),
        )

    breakdown_model = create_model(
        f"{_pascal(name)}PerEndpointBreakdown",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        **breakdown_fields,
    )

    aspect_model = _build_combo_aspect_model(name)
    return create_model(
        f"{_pascal(name)}Output",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        requirement_aspects=(list[aspect_model], Field(..., description=_REQUIREMENT_ASPECTS_DESC)),
        overall_endpoint_fits=(str, Field(..., description=_OVERALL_ENDPOINT_FITS_DESC)),
        per_endpoint_breakdown=(breakdown_model, Field(..., description=_COMBO_PER_ENDPOINT_DESC)),
    )


_BUCKET_FACTORIES: dict[
    HandlerBucket,
    Callable[[tuple[EndpointRoute, ...], str], Optional[type[BaseModel]]],
] = {
    HandlerBucket.SINGLE: _build_single,
    HandlerBucket.MUTEX: _build_mutex,
    HandlerBucket.TIERED: _build_tiered,
    HandlerBucket.COMBO: _build_combo,
}


# ── Eager build + public accessor ─────────────────────────────────

# Populated at module import. Categories whose endpoint set resolves
# to no LLM wrapper (TRENDING-only) are deliberately absent.
OUTPUT_SCHEMAS: dict[CategoryName, type[BaseModel]] = {}


def _build_all() -> None:
    for category in CategoryName:
        factory = _BUCKET_FACTORIES[category.bucket]
        schema = factory(category.endpoints, category.name)
        if schema is not None:
            OUTPUT_SCHEMAS[category] = schema


_build_all()


def get_output_schema(category: CategoryName) -> type[BaseModel]:
    # Raises KeyError for categories with no LLM schema (e.g.
    # CategoryName.TRENDING — handled by a deterministic code path,
    # not a handler LLM). Callers that legitimately handle those
    # categories should special-case them upstream of this lookup.
    return OUTPUT_SCHEMAS[category]
