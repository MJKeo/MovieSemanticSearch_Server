# Per-category handler output schemas.
#
# One Pydantic class per CategoryName, built from the category's
# HandlerBucket and endpoint tuple. The schemas are what the step-3
# category-handler LLM produces via structured output; the bucket-level
# reasoning fields are declared here, and the endpoint-specific
# parameter payloads come from the sibling endpoint_registry module via
# get_output_wrapper(endpoint, bucket, role=..., category=...). The
# `role` kwarg disambiguates SEMANTIC; the `category` kwarg
# disambiguates ENTITY (Person / Character / Title share the route
# but emit different spec classes per category).
#
# Buckets (see search_improvement_planning/query_buckets.md):
#   1. NO_LLM_PURE_CODE                          — no schema (deterministic codepath)
#   2. EXPLICIT_NO_OP                            — no schema
#   3. SINGLE_NON_METADATA_ENDPOINT              — single-endpoint, includes requirement_aspects
#   4. SINGLE_METADATA_ENDPOINT                  — single-endpoint, includes requirement_aspects
#   5. PREFERRED_REPRESENTATION_FALLBACK         — preferred + (optional) fallback split
#   6. SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT  — semantic always + per-deterministic augmentation
#   7. CHARACTER_FRANCHISE_FANOUT                — single shared schema (no per-endpoint payloads)
#   8. AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST  — every candidate endpoint with worth_running gate
#
# Schemas are eagerly built at module import so any misconfiguration
# (missing wrapper, invalid field name, JSON-schema size issue) fails
# loudly at startup instead of on first request. Access via
# get_output_schema(category, role).
#
# See search_improvement_planning/query_buckets.md for the bucket
# taxonomy and per-bucket reasoning shape.

from __future__ import annotations

from typing import Any, Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model

from schemas.enums import EndpointRoute, HandlerBucket, Role
from schemas.trait_category import CategoryName
from search_v2.endpoint_fetching.category_handlers.endpoint_registry import (
    get_output_wrapper,
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

# Bucket 5 — preferred / fallback
_PREFERRED_COVERAGE_EXPLORATION_DESC = (
    "Walk through how the preferred representation could cover this "
    "requirement before committing. Name what it cleanly handles, what "
    "it half-handles, and what it cannot reach. Stay in prose — do not "
    "jump to parameters yet."
)

_PREFERRED_INTENT_DESC = (
    "What the preferred endpoint should retrieve, in plain language. "
    "Name the endpoint and describe its slice. This is the official "
    "commitment to the preferred half of the split, written before "
    "the parameter payload below is filled."
)

_FALLBACK_INTENT_DESC = (
    "What the fallback endpoint should retrieve, or null when the "
    "preferred representation fully covers and the fallback is not "
    "needed. When set, name the endpoint and describe only the part "
    "the preferred representation cannot reach — do not duplicate "
    "what the preferred call already handles."
)

_PREFERRED_PARAMETERS_DESC = (
    "Parameter payload for the preferred endpoint. Fill it whenever "
    "preferred_intent commits to firing the preferred representation; "
    "leave null only when the preferred representation has nothing to "
    "contribute and the requirement falls entirely on the fallback."
)

_FALLBACK_PARAMETERS_DESC = (
    "Parameter payload for the fallback endpoint. Fill it whenever "
    "fallback_intent is non-null; leave null otherwise."
)

# Bucket 6 — semantic + deterministic augmentation
_SEMANTIC_INTENT_DESC = (
    "What the semantic endpoint should retrieve, in plain language. "
    "The semantic call always fires for this bucket; this field "
    "commits the prose query and gives the deterministic decisions "
    "below a stable reference for what the semantic read covers."
)

_AUGMENTATION_OPPORTUNITIES_DESC = (
    "For each non-semantic candidate endpoint, decide whether it "
    "carries a binary or canonical signal that semantic retrieval "
    "blurs across. List one entry per candidate — addressing every "
    "one explicitly is the failure mode this bucket guards against. "
    "Overlap with the semantic read is welcome; skip only when no "
    "clean deterministic signal is implied."
)

_AUGMENTATION_ENDPOINT_KIND_DESC = (
    "Which deterministic candidate this entry is about. Every "
    "category-declared deterministic endpoint should appear in the "
    "list once — do not omit any."
)

_AUGMENTATION_SIGNAL_DESCRIPTION_DESC = (
    "What this endpoint could deterministically catch (a named tag, "
    "a pinned number, a popularity prior). Be specific — vague "
    "descriptions are a sign no clean signal exists."
)

_AUGMENTATION_WORTH_RUNNING_DESC = (
    "True when this endpoint should fire alongside the semantic call. "
    "Skip only when no clean deterministic signal is implied — not "
    "because semantic 'already covers it.'"
)

_SEMANTIC_PARAMETERS_DESC = (
    "Parameter payload for the semantic endpoint. Filled by default — "
    "leave null only when the entire bucket abstains because the "
    "requirement is too ambiguous or out of scope."
)

_AUGMENTATION_PARAMETERS_DESC = (
    "Parameter payload for this deterministic endpoint. Fill it when "
    "its augmentation_opportunities entry has worth_running=True; "
    "leave null otherwise."
)

# Bucket 8 — audience-suitability combo
_SUITABILITY_OVERVIEW_DESC = (
    "High-level scoping pass: enumerate every angle the suitability "
    "concept exposes — hard ceilings, content categories the user "
    "wants more of, content categories to avoid, tone, watch-context. "
    "This is the opportunity inventory the per-endpoint decisions "
    "below draw from."
)

_COVERAGE_OPPORTUNITIES_DESC = (
    "For each candidate endpoint, decide whether it carries a real "
    "complementary slice of the suitability inventory. List one entry "
    "per candidate — silent skipping is the failure mode this bucket "
    "exists to prevent. Overlap with another endpoint's slice is "
    "welcome; skip only when this endpoint has nothing distinct to add."
)

_COVERAGE_ENDPOINT_KIND_DESC = (
    "Which candidate endpoint this entry is about. Every "
    "category-declared candidate should appear in the list once."
)

_COVERAGE_OPPORTUNITY_DESCRIPTION_DESC = (
    "What this endpoint could contribute to the suitability "
    "requirement — a hard maturity ceiling, a tag for content to "
    "include or exclude, a tone or watch-context query. Be specific."
)

_COVERAGE_WORTH_RUNNING_DESC = (
    "True when this endpoint should fire. Overlap with another "
    "endpoint's slice is welcome — skip only when this endpoint has "
    "nothing distinct to add."
)

_SUITABILITY_PARAMETERS_DESC = (
    "Parameter payload for this endpoint. Fill it when its "
    "coverage_opportunities entry has worth_running=True; leave null "
    "otherwise."
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
    role: Role | None,
) -> tuple[tuple[EndpointRoute, Any], ...]:
    # Returns (route, wrapper) pairs, dropping any route whose wrapper
    # resolves to None (e.g. TRENDING — no LLM codepath). Preserves the
    # category's declared endpoint order so position-sensitive buckets
    # (Bucket 5: preferred = position 0, fallback = position 1) can
    # rely on it.
    #
    # Two routes need extra context beyond (route, bucket):
    #   - SEMANTIC: `role` disambiguates Carver vs Qualifier wrappers.
    #   - ENTITY:   `category` disambiguates Person / Character / Title
    #               specs (see endpoint_registry._ENTITY_DISPATCH).
    # All other routes ignore both kwargs.
    pairs: list[tuple[EndpointRoute, Any]] = []
    for route in category.endpoints:
        if route is EndpointRoute.SEMANTIC:
            wrapper = get_output_wrapper(route, bucket, role=role)
        elif route is EndpointRoute.ENTITY:
            wrapper = get_output_wrapper(route, bucket, category=category)
        else:
            wrapper = get_output_wrapper(route, bucket)
        if wrapper is not None:
            pairs.append((route, wrapper))
    return tuple(pairs)


def _output_class_name(name: str, role: Role | None) -> str:
    # Role-dependent schemas (any category whose endpoint tuple contains
    # SEMANTIC) get a Carver / Qualifier suffix so the two variants have
    # distinct class names in the JSON schema $defs section. Categories
    # without SEMANTIC produce one role-independent schema and use the
    # bare `<Name>Output` form.
    suffix = role.name.capitalize() if role is not None else ""
    return f"{_pascal(name)}{suffix}Output"


# ── Bucket factories ──────────────────────────────────────────────


def _no_schema(
    category: CategoryName,
    bucket: HandlerBucket,
    role: Role | None,
) -> type[BaseModel] | None:
    # Buckets 1 & 2 (NO_LLM_PURE_CODE, EXPLICIT_NO_OP) never invoke the
    # handler LLM — no schema is needed. Returning None excludes the
    # category from OUTPUT_SCHEMAS.
    return None


def _build_single(
    category: CategoryName,
    bucket: HandlerBucket,
    role: Role | None,
) -> type[BaseModel] | None:
    # Buckets 3 & 4 — one endpoint owns the whole call. Categories
    # whose only endpoint has no LLM wrapper (e.g. TRENDING-only) get
    # no schema; they run via a deterministic code path elsewhere.
    pairs = _resolve_wrappers_for_bucket(category, bucket, role)
    if not pairs:
        return None
    wrapper = pairs[0][1]

    aspect_model = _build_single_aspect_model(category.name)
    return create_model(
        _output_class_name(category.name, role),
        __base__=_HandlerOutputBase,
        __module__=__name__,
        requirement_aspects=(list[aspect_model], Field(..., description=_REQUIREMENT_ASPECTS_DESC)),
        should_run_endpoint=(bool, Field(..., description=_SHOULD_RUN_ENDPOINT_DESC)),
        endpoint_parameters=(Optional[wrapper], Field(default=None, description=_ENDPOINT_PARAMETERS_SINGLE_DESC)),
    )


def _build_preferred_fallback(
    category: CategoryName,
    bucket: HandlerBucket,
    role: Role | None,
) -> type[BaseModel] | None:
    # Bucket 5 — first endpoint is the preferred representation,
    # second is the fallback. The reasoning fields commit to the split
    # in order (exploration → preferred intent → preferred params →
    # fallback intent → fallback params) so each endpoint payload sits
    # right after the intent that commits to firing it.
    pairs = _resolve_wrappers_for_bucket(category, bucket, role)
    if len(pairs) < 2:
        raise ValueError(
            f"Category {category.name} (PREFERRED_REPRESENTATION_FALLBACK) "
            f"needs ≥2 candidate wrappers, resolved to {len(pairs)}."
        )

    preferred_route, preferred_wrapper = pairs[0]
    fallback_route, fallback_wrapper = pairs[1]

    fields: dict[str, tuple] = {
        "preferred_coverage_exploration": (
            str,
            Field(..., description=_PREFERRED_COVERAGE_EXPLORATION_DESC),
        ),
        "preferred_intent": (str, Field(..., description=_PREFERRED_INTENT_DESC)),
        f"{preferred_route.value}_parameters": (
            Optional[preferred_wrapper],
            Field(default=None, description=_PREFERRED_PARAMETERS_DESC),
        ),
        "fallback_intent": (
            Optional[str],
            Field(default=None, description=_FALLBACK_INTENT_DESC),
        ),
        f"{fallback_route.value}_parameters": (
            Optional[fallback_wrapper],
            Field(default=None, description=_FALLBACK_PARAMETERS_DESC),
        ),
    }

    return create_model(
        _output_class_name(category.name, role),
        __base__=_HandlerOutputBase,
        __module__=__name__,
        **fields,
    )


def _build_semantic_with_augmentation(
    category: CategoryName,
    bucket: HandlerBucket,
    role: Role | None,
) -> type[BaseModel] | None:
    # Bucket 6 — semantic always fires; each non-semantic candidate is
    # a deterministic augmentation that may also fire when it carries
    # a binary or canonical signal semantic blurs across. The schema
    # forces the LLM to enumerate every deterministic candidate via
    # the augmentation_opportunities list (Literal-bounded endpoint_kind).
    pairs = _resolve_wrappers_for_bucket(category, bucket, role)

    semantic_pairs = [(r, w) for r, w in pairs if r is EndpointRoute.SEMANTIC]
    deterministic_pairs = [(r, w) for r, w in pairs if r is not EndpointRoute.SEMANTIC]

    if not semantic_pairs:
        raise ValueError(
            f"Category {category.name} (SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT) "
            f"requires SEMANTIC in its endpoint tuple."
        )
    if not deterministic_pairs:
        raise ValueError(
            f"Category {category.name} (SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT) "
            f"has no deterministic augmentation candidates — at least "
            f"one non-SEMANTIC endpoint is required."
        )

    semantic_route, semantic_wrapper = semantic_pairs[0]

    deterministic_values = tuple(r.value for r, _ in deterministic_pairs)
    endpoint_kind_type = Literal[deterministic_values]  # type: ignore[valid-type]

    opportunity_model = create_model(
        f"{_pascal(category.name)}AugmentationOpportunity",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        endpoint_kind=(endpoint_kind_type, Field(..., description=_AUGMENTATION_ENDPOINT_KIND_DESC)),
        signal_description=(str, Field(..., description=_AUGMENTATION_SIGNAL_DESCRIPTION_DESC)),
        worth_running=(bool, Field(..., description=_AUGMENTATION_WORTH_RUNNING_DESC)),
    )

    fields: dict[str, tuple] = {
        "semantic_intent": (str, Field(..., description=_SEMANTIC_INTENT_DESC)),
        "augmentation_opportunities": (
            list[opportunity_model],
            Field(..., description=_AUGMENTATION_OPPORTUNITIES_DESC),
        ),
        f"{semantic_route.value}_parameters": (
            Optional[semantic_wrapper],
            Field(default=None, description=_SEMANTIC_PARAMETERS_DESC),
        ),
    }
    for det_route, det_wrapper in deterministic_pairs:
        fields[f"{det_route.value}_parameters"] = (
            Optional[det_wrapper],
            Field(default=None, description=_AUGMENTATION_PARAMETERS_DESC),
        )

    return create_model(
        _output_class_name(category.name, role),
        __base__=_HandlerOutputBase,
        __module__=__name__,
        **fields,
    )


def _build_character_franchise_fanout(
    category: CategoryName,
    bucket: HandlerBucket,
    role: Role | None,
) -> type[BaseModel] | None:
    # Bucket 7 — special case. The bucket emits a single shared schema
    # (referent identification + character_forms + franchise_forms)
    # that drives both retrieval paths from one named referent.
    # get_output_wrapper handles the bucket-level dispatch and returns
    # CharacterFranchiseFanoutSchema for any endpoint asked under this
    # bucket; we just hand it back. Role and category are irrelevant —
    # the schema has no SEMANTIC slot and no per-category narrowing.
    if not category.endpoints:
        return None
    schema = get_output_wrapper(category.endpoints[0], bucket)
    assert isinstance(schema, type) and issubclass(schema, BaseModel), (
        f"Bucket 7 dispatch returned non-BaseModel for {category.name}: {schema!r}"
    )
    return schema


def _build_suitability_combo(
    category: CategoryName,
    bucket: HandlerBucket,
    role: Role | None,
) -> type[BaseModel] | None:
    # Bucket 8 — every candidate endpoint contributes one Optional
    # parameter slot plus one entry in the coverage_opportunities list
    # (Literal-bounded endpoint_kind forces the LLM to address every
    # candidate). No always-fires endpoint here; each fires only when
    # its worth_running flag is True.
    pairs = _resolve_wrappers_for_bucket(category, bucket, role)
    if not pairs:
        return None

    endpoint_values = tuple(r.value for r, _ in pairs)
    endpoint_kind_type = Literal[endpoint_values]  # type: ignore[valid-type]

    opportunity_model = create_model(
        f"{_pascal(category.name)}CoverageOpportunity",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        endpoint_kind=(endpoint_kind_type, Field(..., description=_COVERAGE_ENDPOINT_KIND_DESC)),
        opportunity_description=(str, Field(..., description=_COVERAGE_OPPORTUNITY_DESCRIPTION_DESC)),
        worth_running=(bool, Field(..., description=_COVERAGE_WORTH_RUNNING_DESC)),
    )

    fields: dict[str, tuple] = {
        "suitability_overview": (str, Field(..., description=_SUITABILITY_OVERVIEW_DESC)),
        "coverage_opportunities": (
            list[opportunity_model],
            Field(..., description=_COVERAGE_OPPORTUNITIES_DESC),
        ),
    }
    for route, wrapper in pairs:
        fields[f"{route.value}_parameters"] = (
            Optional[wrapper],
            Field(default=None, description=_SUITABILITY_PARAMETERS_DESC),
        )

    return create_model(
        _output_class_name(category.name, role),
        __base__=_HandlerOutputBase,
        __module__=__name__,
        **fields,
    )


# ── Dispatch table ────────────────────────────────────────────────


_BucketFactory = Callable[
    [CategoryName, HandlerBucket, Optional[Role]],
    Optional[type[BaseModel]],
]


_BUCKET_FACTORIES: dict[HandlerBucket, _BucketFactory] = {
    HandlerBucket.NO_LLM_PURE_CODE: _no_schema,
    HandlerBucket.EXPLICIT_NO_OP: _no_schema,
    HandlerBucket.SINGLE_NON_METADATA_ENDPOINT: _build_single,
    HandlerBucket.SINGLE_METADATA_ENDPOINT: _build_single,
    HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK: _build_preferred_fallback,
    HandlerBucket.SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT: _build_semantic_with_augmentation,
    HandlerBucket.CHARACTER_FRANCHISE_FANOUT: _build_character_franchise_fanout,
    HandlerBucket.AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST: _build_suitability_combo,
}


# ── Eager build + public accessor ─────────────────────────────────

# Populated at module import. Keyed by (category, role) — every
# category gets one entry per Role so callers can always look up by
# the upstream Trait's role without knowing whether the schema is
# actually role-dependent. Categories with SEMANTIC in their endpoint
# tuple get two distinct schemas (Carver / Qualifier wrappers differ);
# all other categories store the same schema object under both Role
# keys. Categories whose endpoint set resolves to no LLM wrapper
# (TRENDING-only) and categories in the no-LLM / no-op buckets are
# deliberately absent.
OUTPUT_SCHEMAS: dict[tuple[CategoryName, Role], type[BaseModel]] = {}


def _build_all() -> None:
    for category in CategoryName:
        factory = _BUCKET_FACTORIES[category.bucket]
        if EndpointRoute.SEMANTIC in category.endpoints:
            # Role-dependent schema: build one per role.
            for role in Role:
                schema = factory(category, category.bucket, role)
                if schema is not None:
                    OUTPUT_SCHEMAS[(category, role)] = schema
        else:
            # Role-independent schema: build once, store under both
            # role keys for uniform lookup.
            schema = factory(category, category.bucket, None)
            if schema is not None:
                for role in Role:
                    OUTPUT_SCHEMAS[(category, role)] = schema


_build_all()


def get_output_schema(category: CategoryName, role: Role) -> type[BaseModel]:
    # Raises KeyError for categories with no LLM schema (TRENDING,
    # MEDIA_TYPE, BELOW_THE_LINE_CREATOR — handled by deterministic
    # code paths or as no-ops, not handler LLMs). Callers that
    # legitimately handle those categories should special-case them
    # upstream of this lookup.
    #
    # `role` is the parent Trait's committed role. It selects between
    # Carver / Qualifier semantic wrappers for categories whose schema
    # is role-dependent (any category with SEMANTIC in its endpoints);
    # for other categories the same schema is returned regardless of
    # role.
    return OUTPUT_SCHEMAS[(category, role)]
