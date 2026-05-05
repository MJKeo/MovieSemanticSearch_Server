# Per-category handler output schemas.
#
# One Pydantic class per CategoryName, built from the category's
# HandlerBucket and endpoint tuple. The schemas are what the step-3
# category-handler LLM produces via structured output; the bucket-level
# reasoning fields are declared here, and the endpoint-specific
# parameter payloads come from the sibling endpoint_registry module via
# get_output_wrapper(endpoint, bucket, category=...). The `category`
# kwarg disambiguates ENTITY (Person / Character / Title share the
# route but emit different spec classes per category).
#
# Buckets (see search_improvement_planning/query_buckets.md):
#   1. NO_LLM_PURE_CODE                          — no schema (deterministic codepath)
#   2. EXPLICIT_NO_OP                            — no schema
#   3. SINGLE_NON_METADATA_ENDPOINT              — single-endpoint, includes requirement_aspects
#   4. SINGLE_METADATA_ENDPOINT                  — single-endpoint, includes requirement_aspects
#   5. PREFERRED_REPRESENTATION_FALLBACK         — walk-then-commit (preferred + fallback)
#   6. SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT  — walk-then-commit (semantic + deterministic support)
#   7. CHARACTER_FRANCHISE_FANOUT                — single shared schema (no per-endpoint payloads)
#   8. AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST  — walk-then-commit (every candidate endpoint)
#
# Buckets 5/6/8 share one shape: per-endpoint grounded walks → coverage
# commitment (assignments + intentionally_uncovered) → thin per-endpoint
# params. See _build_walk_then_commit.
#
# Schemas are eagerly built at module import so any misconfiguration
# (missing wrapper, invalid field name, JSON-schema size issue) fails
# loudly at startup instead of on first request. Access via
# get_output_schema(category).
#
# See search_improvement_planning/query_buckets.md for the bucket
# taxonomy and per-bucket reasoning shape.

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from schemas.enums import EndpointRoute, HandlerBucket
from schemas.trait_category import CategoryName
from search_v2.endpoint_fetching.category_handlers.endpoint_registry import (
    get_output_wrapper,
    get_walk_class,
)


# Every dynamically-generated class inherits this so OpenAI structured
# output gets additionalProperties: false on every sub-object.
class _HandlerOutputBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ── Shared Field descriptions ─────────────────────────────────────
# Module-level constants prevent wording drift across the bucket
# factories. Tuned for small / instruction-tuned models: phrasal cues,
# anti-failure-mode framing, concrete direction over abstract framing.

# Single-endpoint buckets (3, 4)
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

_SHOULD_RUN_ENDPOINT_DESC = (
    "True when this endpoint should fire. False is a valid and "
    "preferred answer when the fragment does not cleanly fit — "
    "do not invent parameters for a bad match."
)

_ENDPOINT_PARAMETERS_SINGLE_DESC = (
    "The endpoint's parameter payload, carrying role and polarity. "
    "Leave null when should_run_endpoint is false."
)

# Buckets 5/6/8 — walk-then-commit shape (one shared factory).
#
# Phase 1: per-endpoint grounded walks. One `{route}_walk` field per
# declared endpoint, holding the registry/space/column-grounded
# analysis lifted out of the per-endpoint subintent params. Emitted
# BEFORE the commitment phase so the LLM walks concrete candidates
# before deciding who fires.
#
# Phase 2: coverage commitment. `coverage_assignments` delegates
# slices to specific endpoints; `intentionally_uncovered` names what
# the call asks for that no declared endpoint can cleanly serve.
# Both grounded in the walks above.
#
# Phase 3: per-endpoint thin params. One Optional `{route}_parameters`
# slot per declared endpoint, populated iff coverage_assignments
# contains an entry for that endpoint.

_WALK_DESC_TEMPLATE = (
    "Grounded walk for the {route} endpoint. Read the call's "
    "retrieval_intent + expressions (in the user message) and surface "
    "what {route} could concretely cover, with explicit "
    "covers/misses prose grounded in the {grounded_in}. This is "
    "the analysis layer the commitment phase below draws from — "
    "abstract optimism about the endpoint's general fitness is not "
    "useful here, only concrete candidates. An empty / no-match walk "
    "is a valid signal that {route} has nothing to offer for this "
    "call; the commitment phase is allowed to leave the call unowned "
    "by {route}."
)

_WALK_GROUNDING: dict[EndpointRoute, str] = {
    EndpointRoute.KEYWORD: "UnifiedClassification registry",
    EndpointRoute.SEMANTIC: "7 vector spaces",
    EndpointRoute.METADATA: "10 structured-attribute columns",
}


def _walk_desc_for(route: EndpointRoute) -> str:
    grounded_in = _WALK_GROUNDING.get(route, "endpoint's domain")
    return _WALK_DESC_TEMPLATE.format(
        route=route.value, grounded_in=grounded_in
    )


_COVERAGE_ASSIGNMENT_ENDPOINT_KIND_DESC = (
    "Which declared endpoint this assignment delegates a slice to. "
    "Must be one of the routes whose grounded walk appears above on "
    "this bucket schema."
)

_COVERAGE_ASSIGNMENT_SLICE_DESCRIPTION_DESC = (
    "The slice of the call's intent this endpoint owns, written "
    "specifically enough that the per-endpoint parameters below can "
    "translate it without re-reading the upstream retrieval_intent. "
    "Pulled from the matching endpoint's grounded walk above — name "
    "what the walk concretely surfaced (registry members / vector "
    "spaces / columns) and what aspect(s) of the call's intent they "
    "address. This string flows to the wrapper's "
    "<endpoint>_retrieval_intent field as the per-endpoint "
    "commitment record."
)

_COVERAGE_ASSIGNMENTS_DESC = (
    "Coverage commitment phase. One entry per declared endpoint "
    "that should fire, naming the slice of the call's intent it "
    "owns. Read every endpoint walk above and decide who fires on "
    "what. Multiple assignments are fine when the call is genuinely "
    "compound and several endpoints catch distinct facets; a single "
    "assignment is the natural shape when one endpoint cleanly owns "
    "the call.\n"
    "\n"
    "EMPTY is a valid outcome when no declared endpoint cleanly fits "
    "and the call is better unowned than served by a poor fit.\n"
    "\n"
    "Priority order (the order endpoints appear in the bucket "
    "schema, top to bottom) is a tiebreaker only — when two "
    "endpoints could cover the same slice equally well, prefer the "
    "earlier one. The walks themselves are order-independent.\n"
    "\n"
    "NEVER:\n"
    "- DELEGATE TO AN ENDPOINT WHOSE WALK SHOWS NO CLEAN FIT. The "
    "walks are the audit; assignments must be readable off them.\n"
    "- DUPLICATE AN ENDPOINT (one assignment per endpoint kind).\n"
    "- SPLIT ONE SLICE ACROSS ENDPOINTS to look thorough; let the "
    "endpoint that owns the slice own it cleanly."
)

_INTENTIONALLY_UNCOVERED_DESC = (
    "Aspects of the call's intent that no declared endpoint can "
    "cleanly serve, named explicitly. Forces the commitment phase to "
    "be honest about gaps rather than fabricating shallow coverage. "
    "Empty list is correct when coverage_assignments cleanly handle "
    "the whole call.\n"
    "\n"
    "An entry here is a deliberate trade-off — better to leave "
    "an aspect unserved than to ship a poor-fit assignment. Reviewers "
    "read this list to understand what the bucket consciously walked "
    "away from.\n"
    "\n"
    "NEVER:\n"
    "- HEDGE. Either an aspect is owned (in coverage_assignments) or "
    "intentionally not (here).\n"
    "- LIST AN ASPECT THAT'S ALREADY OWNED by some assignment."
)

_THIN_PARAMETERS_DESC_TEMPLATE = (
    "Thin commitment payload for the {route} endpoint. Fill it iff "
    "coverage_assignments above contains an entry whose endpoint_kind "
    "is {route!r}; null otherwise. The wrapper's "
    "`{route}_retrieval_intent` mirrors the matching assignment's "
    "slice_description; the inner parameters draw on that intent and "
    "the upstream `{route}_walk` analysis to commit the route-specific "
    "translation."
)


# ── Naming helper ─────────────────────────────────────────────────


def _pascal(name: str) -> str:
    # CategoryName.name is UPPER_SNAKE_CASE (e.g. "CREDIT_TITLE"); the
    # codebase convention for Pydantic models is PascalCase.
    return "".join(part.capitalize() for part in name.split("_"))


# ── Per-aspect sub-model factory (single-endpoint buckets only) ────


def _build_single_aspect_model(category_name: str) -> type[BaseModel]:
    return create_model(
        f"{_pascal(category_name)}RequirementAspect",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        aspect_description=(str, Field(..., description=_ASPECT_DESCRIPTION_DESC)),
        relation_to_endpoint=(str, Field(..., description=_RELATION_TO_ENDPOINT_DESC)),
        coverage_gaps=(Optional[str], Field(default=None, description=_COVERAGE_GAPS_DESC)),
    )


# ── Wrapper resolution ────────────────────────────────────────────


def _resolve_wrappers_for_bucket(
    category: CategoryName,
    bucket: HandlerBucket,
) -> tuple[tuple[EndpointRoute, Any], ...]:
    # Returns (route, wrapper) pairs, dropping any route whose wrapper
    # resolves to None (e.g. TRENDING — no LLM codepath). Preserves the
    # category's declared endpoint order so position-sensitive buckets
    # (Bucket 5: preferred = position 0, fallback = position 1) can
    # rely on it.
    #
    # ENTITY needs `category` to disambiguate Person / Character /
    # Title specs (see endpoint_registry._ENTITY_DISPATCH).
    pairs: list[tuple[EndpointRoute, Any]] = []
    for route in category.endpoints:
        if route is EndpointRoute.ENTITY:
            wrapper = get_output_wrapper(route, bucket, category=category)
        else:
            wrapper = get_output_wrapper(route, bucket)
        if wrapper is not None:
            pairs.append((route, wrapper))
    return tuple(pairs)


def _output_class_name(name: str) -> str:
    return f"{_pascal(name)}Output"


# ── Bucket factories ──────────────────────────────────────────────


def _no_schema(
    category: CategoryName,
    bucket: HandlerBucket,
) -> type[BaseModel] | None:
    # Buckets 1 & 2 (NO_LLM_PURE_CODE, EXPLICIT_NO_OP) never invoke the
    # handler LLM — no schema is needed. Returning None excludes the
    # category from OUTPUT_SCHEMAS.
    return None


def _build_single(
    category: CategoryName,
    bucket: HandlerBucket,
) -> type[BaseModel] | None:
    # Buckets 3 & 4 — one endpoint owns the whole call. Categories
    # whose only endpoint has no LLM wrapper (e.g. TRENDING-only) get
    # no schema; they run via a deterministic code path elsewhere.
    pairs = _resolve_wrappers_for_bucket(category, bucket)
    if not pairs:
        return None
    wrapper = pairs[0][1]

    aspect_model = _build_single_aspect_model(category.name)
    return create_model(
        _output_class_name(category.name),
        __base__=_HandlerOutputBase,
        __module__=__name__,
        requirement_aspects=(list[aspect_model], Field(..., description=_REQUIREMENT_ASPECTS_DESC)),
        should_run_endpoint=(bool, Field(..., description=_SHOULD_RUN_ENDPOINT_DESC)),
        endpoint_parameters=(Optional[wrapper], Field(default=None, description=_ENDPOINT_PARAMETERS_SINGLE_DESC)),
    )


def _build_walk_then_commit(
    category: CategoryName,
    bucket: HandlerBucket,
) -> type[BaseModel] | None:
    # Buckets 5/6/8 share one shape — three sequential phases at the
    # bucket level that together replace the bucket-specific
    # coverage-reasoning fields used previously:
    #
    #   Phase 1: per-endpoint walks. For each declared endpoint, a
    #   `{route}_walk` field holds the registry/space/column-grounded
    #   analysis (KeywordWalk / SemanticWalk / MetadataWalk).
    #   Surfaces concrete candidates BEFORE any commitment.
    #
    #   Phase 2: coverage commitment. `coverage_assignments` (a list
    #   of CoverageAssignment with Literal-bounded endpoint_kind)
    #   delegates slices to specific endpoints; `intentionally_uncovered`
    #   names aspects no declared endpoint can cleanly serve.
    #
    #   Phase 3: per-endpoint thin params. One Optional
    #   `{route}_parameters` per declared endpoint, populated iff a
    #   matching coverage_assignments entry exists.
    #
    # Field declaration order matches that phase ordering — which
    # matters because Pydantic structured output emits top-down, so
    # the LLM walks all endpoints concretely before committing to who
    # fires. This is the structural fix for the prior design's
    # "abstract commitment before grounded walk" failure mode.
    pairs = _resolve_wrappers_for_bucket(category, bucket)
    if not pairs:
        return None

    # Build the per-category CoverageAssignment with a Literal of the
    # declared route values. Same dynamic-Literal pattern that the
    # prior augmentation/coverage opportunity models used; lifted up
    # to the new shared shape.
    declared_route_values = tuple(r.value for r, _ in pairs)
    endpoint_kind_type = Literal[declared_route_values]  # type: ignore[valid-type]

    coverage_assignment_model = create_model(
        f"{_pascal(category.name)}CoverageAssignment",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        endpoint_kind=(
            endpoint_kind_type,
            Field(..., description=_COVERAGE_ASSIGNMENT_ENDPOINT_KIND_DESC),
        ),
        slice_description=(
            str,
            Field(..., description=_COVERAGE_ASSIGNMENT_SLICE_DESCRIPTION_DESC),
        ),
    )

    fields: dict[str, tuple] = {}

    # Phase 1: per-endpoint grounded walks. Every declared multi-
    # endpoint route must have a walk class registered — without it
    # the bucket cannot be assembled. Failing loud at import time
    # surfaces the missing walk before it ever reaches a request.
    for route, _ in pairs:
        walk_cls = get_walk_class(route)
        if walk_cls is None:
            raise ValueError(
                f"Category {category.name} (bucket {bucket.value!r}) "
                f"declares route {route.value!r}, but no walk class is "
                f"registered in endpoint_registry.ROUTE_TO_WALK. Author "
                f"a {route.value.title()}Walk class in the matching "
                f"translation file before routing this combination to "
                f"a multi-endpoint bucket."
            )
        fields[f"{route.value}_walk"] = (
            walk_cls,
            Field(..., description=_walk_desc_for(route)),
        )

    # Phase 2: coverage commitment.
    fields["coverage_assignments"] = (
        list[coverage_assignment_model],
        Field(..., description=_COVERAGE_ASSIGNMENTS_DESC),
    )
    fields["intentionally_uncovered"] = (
        list[str],
        Field(..., description=_INTENTIONALLY_UNCOVERED_DESC),
    )

    # Phase 3: per-endpoint thin params, one Optional per declared
    # route. Names match the existing `{route}_parameters` pattern so
    # output_extractor's per-route field walk picks them up unchanged.
    for route, wrapper in pairs:
        fields[f"{route.value}_parameters"] = (
            Optional[wrapper],
            Field(
                default=None,
                description=_THIN_PARAMETERS_DESC_TEMPLATE.format(
                    route=route.value
                ),
            ),
        )

    return create_model(
        _output_class_name(category.name),
        __base__=_HandlerOutputBase,
        __module__=__name__,
        **fields,
    )


def _build_character_franchise_fanout(
    category: CategoryName,
    bucket: HandlerBucket,
) -> type[BaseModel] | None:
    # Bucket 7 — special case. The bucket emits a single shared schema
    # (referent identification + character_forms + franchise_forms)
    # that drives both retrieval paths from one named referent.
    # get_output_wrapper handles the bucket-level dispatch and returns
    # CharacterFranchiseFanoutSchema for any endpoint asked under this
    # bucket; we just hand it back. Category narrowing is irrelevant —
    # the schema has no SEMANTIC slot and no per-category dispatch.
    if not category.endpoints:
        return None
    schema = get_output_wrapper(category.endpoints[0], bucket)
    assert isinstance(schema, type) and issubclass(schema, BaseModel), (
        f"Bucket 7 dispatch returned non-BaseModel for {category.name}: {schema!r}"
    )
    return schema


# ── Dispatch table ────────────────────────────────────────────────


_BucketFactory = Callable[
    [CategoryName, HandlerBucket],
    Optional[type[BaseModel]],
]


_BUCKET_FACTORIES: dict[HandlerBucket, _BucketFactory] = {
    HandlerBucket.NO_LLM_PURE_CODE: _no_schema,
    HandlerBucket.EXPLICIT_NO_OP: _no_schema,
    HandlerBucket.SINGLE_NON_METADATA_ENDPOINT: _build_single,
    HandlerBucket.SINGLE_METADATA_ENDPOINT: _build_single,
    HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK: _build_walk_then_commit,
    HandlerBucket.SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT: _build_walk_then_commit,
    HandlerBucket.CHARACTER_FRANCHISE_FANOUT: _build_character_franchise_fanout,
    HandlerBucket.AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST: _build_walk_then_commit,
}


# ── Eager build + public accessor ─────────────────────────────────

# Populated at module import. One schema per category. Categories
# whose endpoint set resolves to no LLM wrapper (TRENDING-only) and
# categories in the no-LLM / no-op buckets are deliberately absent.
OUTPUT_SCHEMAS: dict[CategoryName, type[BaseModel]] = {}


def _build_all() -> None:
    for category in CategoryName:
        factory = _BUCKET_FACTORIES[category.bucket]
        schema = factory(category, category.bucket)
        if schema is not None:
            OUTPUT_SCHEMAS[category] = schema


_build_all()


def get_output_schema(category: CategoryName) -> type[BaseModel]:
    # Raises KeyError for categories with no LLM schema (TRENDING,
    # MEDIA_TYPE, BELOW_THE_LINE_CREATOR — handled by deterministic
    # code paths or as no-ops, not handler LLMs). Callers that
    # legitimately handle those categories should special-case them
    # upstream of this lookup.
    return OUTPUT_SCHEMAS[category]
