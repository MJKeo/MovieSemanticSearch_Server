"""Schema-factories shape tests, designed from planning docs.

Expectations come from `search_improvement_planning/query_buckets.md`
(bucket assignment + bucket-level reasoning shape) and
`search_improvement_planning/query_categories.md` (endpoint identity
per category, including which endpoint is the preferred / fallback
side for bucket 5 categories — derived from each category's
mechanism description).

If schema_factories disagrees with the planning docs, these tests
fail — that's the point.

Schemas are role-dependent for any category whose endpoint tuple
contains SEMANTIC: Carver and Qualifier semantic wrappers carry
structurally different `parameters`, so two distinct schemas exist.
For role-independent categories (no SEMANTIC slot) the same schema
is returned regardless of role; we still parametrize over both roles
so the uniform-lookup contract is exercised.
"""

from __future__ import annotations

import types
import typing

import pytest
from pydantic import BaseModel

from schemas.award_translation import AwardEndpointParameters
from schemas.entity_translation import EntityEndpointParameters
from schemas.enums import EndpointRoute, HandlerBucket, Role
from schemas.franchise_translation import FranchiseEndpointParameters
from schemas.keyword_translation import (
    KeywordEndpointParameters,
    KeywordEndpointSubintentParameters,
)
from schemas.metadata_translation import (
    MetadataEndpointParameters,
    MetadataEndpointSubintentParameters,
)
from schemas.semantic_translation import (
    CarverSemanticEndpointParameters,
    CarverSemanticEndpointSubintentParameters,
    QualifierSemanticEndpointParameters,
    QualifierSemanticEndpointSubintentParameters,
)
from schemas.studio_translation import StudioEndpointParameters
from schemas.trait_category import CategoryName

from search_v2.endpoint_fetching.category_handlers.endpoint_registry import (
    CharacterFranchiseFanoutSchema,
)
from search_v2.endpoint_fetching.category_handlers.schema_factories import (
    OUTPUT_SCHEMAS,
    get_output_schema,
)


# ───────────────────────────────────────────────────────────────────────
# Planning-doc-driven category lists (transcribed from
# search_improvement_planning/query_buckets.md).
# Endpoint mappings come from query_categories.md per-category
# "Endpoints:" lines, collapsing semantic sub-spaces (P-EVT, P-ANA,
# VWX, CTX, NRT, PRD, RCP) to SEMANTIC at the bucket level (per
# query_buckets.md principle: "A category can query one or more
# semantic spaces while still belonging to the same bucket if the
# handler instruction is the same.").
# ───────────────────────────────────────────────────────────────────────

NO_LLM_PURE_CODE_CATEGORIES = ["TRENDING", "MEDIA_TYPE"]
EXPLICIT_NO_OP_CATEGORIES = ["BELOW_THE_LINE_CREATOR"]

# (enum name, single endpoint per query_categories.md)
SINGLE_NON_METADATA_CATEGORIES = [
    ("PERSON_CREDIT",          EndpointRoute.ENTITY),               # Cat 1
    ("TITLE_TEXT",             EndpointRoute.ENTITY),               # Cat 2
    ("NAMED_CHARACTER",        EndpointRoute.ENTITY),               # Cat 3
    ("STUDIO_BRAND",           EndpointRoute.STUDIO),               # Cat 4
    ("FRANCHISE_LINEAGE",      EndpointRoute.FRANCHISE_STRUCTURE),  # Cat 5
    ("ADAPTATION_SOURCE",      EndpointRoute.KEYWORD),              # Cat 7
    ("AWARDS",                 EndpointRoute.AWARDS),               # Cat 11
    ("FILMING_LOCATION",       EndpointRoute.SEMANTIC),             # Cat 24 (PRD)
    ("PLOT_EVENTS",            EndpointRoute.SEMANTIC),             # Cat 30 (P-EVT)
    ("NARRATIVE_SETTING",      EndpointRoute.SEMANTIC),             # Cat 31 (P-EVT)
    ("VIEWING_OCCASION",       EndpointRoute.SEMANTIC),             # Cat 34 (CTX)
    ("VISUAL_CRAFT_ACCLAIM",   EndpointRoute.SEMANTIC),             # Cat 35 (RCP+PRD → semantic family)
    ("MUSIC_SCORE_ACCLAIM",    EndpointRoute.SEMANTIC),             # Cat 36 (RCP)
    ("DIALOGUE_CRAFT_ACCLAIM", EndpointRoute.SEMANTIC),             # Cat 37 (RCP+NRT → semantic family)
    ("NAMED_SOURCE_CREATOR",   EndpointRoute.SEMANTIC),             # Cat 42 (P-EVT+RCP → semantic family)
]

SINGLE_METADATA_CATEGORIES = [
    "RELEASE_DATE",            # Cat 13
    "RUNTIME",                 # Cat 14
    "MATURITY_RATING",         # Cat 15
    "AUDIO_LANGUAGE",          # Cat 16
    "STREAMING",               # Cat 17
    "FINANCIAL_SCALE",         # Cat 18
    "NUMERIC_RECEPTION_SCORE", # Cat 19
    "COUNTRY_OF_ORIGIN",       # Cat 20
    "GENERAL_APPEAL",          # Cat 38
    "CHRONOLOGICAL",           # Cat 44
]

# Bucket 5: (enum name, preferred endpoint, fallback endpoint).
# Preferred / fallback identity comes from the mechanism description
# in query_categories.md — "keyword-first" tiers put KW preferred,
# Cat 40's "RCP carries the aspect-level like/dislike; KW can fire
# when..." puts SEMANTIC preferred and KW fallback.
PREFERRED_FALLBACK_CATEGORIES = [
    # Cat 8  CENTRAL_TOPIC: "P-EVT + KW (tiered, keyword-first)"
    ("CENTRAL_TOPIC",             EndpointRoute.KEYWORD,  EndpointRoute.SEMANTIC),
    # Cat 9  ELEMENT_PRESENCE: "P-EVT + KW (tiered, keyword-first)"
    ("ELEMENT_PRESENCE",          EndpointRoute.KEYWORD,  EndpointRoute.SEMANTIC),
    # Cat 10 CHARACTER_ARCHETYPE: "KW + NRT (tiered, keyword-first)"
    ("CHARACTER_ARCHETYPE",       EndpointRoute.KEYWORD,  EndpointRoute.SEMANTIC),
    # Cat 22 GENRE: "KW + P-ANA (mutex)" — KW first
    ("GENRE",                     EndpointRoute.KEYWORD,  EndpointRoute.SEMANTIC),
    # Cat 23 CULTURAL_TRADITION: "KW + META (mutex, keyword-first)"
    ("CULTURAL_TRADITION",        EndpointRoute.KEYWORD,  EndpointRoute.METADATA),
    # Cat 25 FORMAT_VISUAL: "KW + PRD (tiered, keyword-first)"
    ("FORMAT_VISUAL",             EndpointRoute.KEYWORD,  EndpointRoute.SEMANTIC),
    # Cat 26 NARRATIVE_DEVICES: "KW + NRT (tiered, keyword-first)"
    ("NARRATIVE_DEVICES",         EndpointRoute.KEYWORD,  EndpointRoute.SEMANTIC),
    # Cat 29 SEASONAL_HOLIDAY: KW proxy chains primary + CTX/P-EVT
    # semantic for spectrum-framed asks → KW preferred at the bucket
    # level per query_buckets.md classification.
    ("SEASONAL_HOLIDAY",          EndpointRoute.KEYWORD,  EndpointRoute.SEMANTIC),
    # Cat 32 STORY_THEMATIC_ARCHETYPE: "KW + P-ANA (tiered, keyword-first)"
    ("STORY_THEMATIC_ARCHETYPE",  EndpointRoute.KEYWORD,  EndpointRoute.SEMANTIC),
    # Cat 40 SPECIFIC_PRAISE_CRITICISM: "RCP + KW" — mechanism gives
    # RCP primary ("RCP `praised_qualities`/`criticized_qualities`
    # prose carries the aspect-level like/dislike"), KW supplemental.
    ("SPECIFIC_PRAISE_CRITICISM", EndpointRoute.SEMANTIC, EndpointRoute.KEYWORD),
]

# Bucket 6: semantic always + per-deterministic augmentation.
# (enum name, semantic_route, deterministic_routes_in_order)
SEMANTIC_AUGMENTATION_CATEGORIES = [
    # Cat 33 EMOTIONAL_EXPERIENTIAL: "VWX + CTX + RCP + KW (additive
    # combo, handler-driven field selection)" → SEMANTIC + KW.
    ("EMOTIONAL_EXPERIENTIAL", EndpointRoute.SEMANTIC, [EndpointRoute.KEYWORD]),
    # Cat 39 CULTURAL_STATUS: "RCP + META.reception_score +
    # META.popularity_score (additive combo)" → SEMANTIC + METADATA.
    ("CULTURAL_STATUS",        EndpointRoute.SEMANTIC, [EndpointRoute.METADATA]),
]

# Bucket 8: every candidate fires when worth_running.
# (enum name, endpoints in tuple-order from query_categories.md)
SUITABILITY_CATEGORIES = [
    # Cat 27 TARGET_AUDIENCE: "KW + META + CTX (gate + inclusion)"
    ("TARGET_AUDIENCE",   [EndpointRoute.KEYWORD, EndpointRoute.METADATA, EndpointRoute.SEMANTIC]),
    # Cat 28 SENSITIVE_CONTENT: "KW + META + VWX (gate + inclusion)"
    ("SENSITIVE_CONTENT", [EndpointRoute.KEYWORD, EndpointRoute.METADATA, EndpointRoute.SEMANTIC]),
]


# ───────────────────────────────────────────────────────────────────────
# Wrapper-class lookup helpers.
# Single-endpoint buckets use the regular wrapper. Multi-endpoint
# buckets use the subintent wrapper. SEMANTIC additionally splits on
# role (Carver / Qualifier).
# ───────────────────────────────────────────────────────────────────────

_REGULAR_NON_SEMANTIC: dict[EndpointRoute, type[BaseModel]] = {
    EndpointRoute.ENTITY: EntityEndpointParameters,
    EndpointRoute.STUDIO: StudioEndpointParameters,
    EndpointRoute.METADATA: MetadataEndpointParameters,
    EndpointRoute.AWARDS: AwardEndpointParameters,
    EndpointRoute.FRANCHISE_STRUCTURE: FranchiseEndpointParameters,
    EndpointRoute.KEYWORD: KeywordEndpointParameters,
}

_SUBINTENT_NON_SEMANTIC: dict[EndpointRoute, type[BaseModel]] = {
    EndpointRoute.KEYWORD: KeywordEndpointSubintentParameters,
    EndpointRoute.METADATA: MetadataEndpointSubintentParameters,
}


def _expected_wrapper(
    route: EndpointRoute, is_multi: bool, role: Role,
) -> type[BaseModel]:
    """The Pydantic class the bucket should plug into the
    `<route>_parameters` slot under (is_multi, role)."""
    if route is EndpointRoute.SEMANTIC:
        if is_multi:
            return (
                CarverSemanticEndpointSubintentParameters
                if role is Role.CARVER
                else QualifierSemanticEndpointSubintentParameters
            )
        return (
            CarverSemanticEndpointParameters
            if role is Role.CARVER
            else QualifierSemanticEndpointParameters
        )
    if is_multi:
        wrapper = _SUBINTENT_NON_SEMANTIC.get(route)
        assert wrapper is not None, (
            f"No subintent wrapper authored for {route} — registry "
            f"would raise; test list is wrong."
        )
        return wrapper
    return _REGULAR_NON_SEMANTIC[route]


def _unwrap_optional(annotation: object) -> object:
    """`Optional[X]` is `Union[X, None]`. Return X (or the rebuilt
    union of non-None args), raising if not Optional."""
    origin = typing.get_origin(annotation)
    if origin not in (typing.Union, types.UnionType):
        raise AssertionError(f"Expected Optional, got: {annotation!r}")
    non_none = [a for a in typing.get_args(annotation) if a is not type(None)]
    if len(non_none) == 1:
        return non_none[0]
    out = non_none[0]
    for a in non_none[1:]:
        out = out | a
    return out


# ───────────────────────────────────────────────────────────────────────
# Bucket 1 & 2: no schema
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", NO_LLM_PURE_CODE_CATEGORIES)
@pytest.mark.parametrize("role", list(Role))
def test_bucket_1_no_llm_pure_code_has_no_schema(name: str, role: Role) -> None:
    """Bucket 1 categories never invoke the handler LLM, so they must
    not appear in OUTPUT_SCHEMAS for any role."""
    cat = CategoryName[name]
    assert (cat, role) not in OUTPUT_SCHEMAS, (
        f"{name} ({role.name}) should be excluded from OUTPUT_SCHEMAS"
    )
    with pytest.raises(KeyError):
        get_output_schema(cat, role)


@pytest.mark.parametrize("name", EXPLICIT_NO_OP_CATEGORIES)
@pytest.mark.parametrize("role", list(Role))
def test_bucket_2_explicit_noop_has_no_schema(name: str, role: Role) -> None:
    """Bucket 2 (explicit no-op) categories must not have an LLM
    schema."""
    cat = CategoryName[name]
    assert (cat, role) not in OUTPUT_SCHEMAS
    with pytest.raises(KeyError):
        get_output_schema(cat, role)


# ───────────────────────────────────────────────────────────────────────
# Bucket 3 & 4: single endpoint shape
# ───────────────────────────────────────────────────────────────────────

_SINGLE_ENDPOINT_FIELDS = [
    "requirement_aspects",
    "should_run_endpoint",
    "endpoint_parameters",
]


def _assert_single_shape(
    cat: CategoryName, route: EndpointRoute, role: Role,
) -> None:
    schema = get_output_schema(cat, role)
    fields = list(schema.model_fields.keys())
    assert fields == _SINGLE_ENDPOINT_FIELDS, (
        f"{cat.name}/{role.name}: bucket 3/4 field order — got {fields}"
    )
    inner = _unwrap_optional(schema.model_fields["endpoint_parameters"].annotation)
    expected = _expected_wrapper(route, is_multi=False, role=role)
    assert inner is expected, (
        f"{cat.name}/{role.name}: endpoint_parameters wraps {inner}, "
        f"expected {expected}"
    )


@pytest.mark.parametrize(
    "name,route",
    SINGLE_NON_METADATA_CATEGORIES,
    ids=[n for n, _ in SINGLE_NON_METADATA_CATEGORIES],
)
@pytest.mark.parametrize("role", list(Role))
def test_bucket_3_single_non_metadata_shape(
    name: str, route: EndpointRoute, role: Role,
) -> None:
    _assert_single_shape(CategoryName[name], route, role)


@pytest.mark.parametrize("name", SINGLE_METADATA_CATEGORIES)
@pytest.mark.parametrize("role", list(Role))
def test_bucket_4_single_metadata_shape(name: str, role: Role) -> None:
    _assert_single_shape(CategoryName[name], EndpointRoute.METADATA, role)


# ───────────────────────────────────────────────────────────────────────
# Bucket 5: preferred + fallback
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,preferred,fallback",
    PREFERRED_FALLBACK_CATEGORIES,
    ids=[t[0] for t in PREFERRED_FALLBACK_CATEGORIES],
)
@pytest.mark.parametrize("role", list(Role))
def test_bucket_5_preferred_fallback_shape(
    name: str,
    preferred: EndpointRoute,
    fallback: EndpointRoute,
    role: Role,
) -> None:
    """Bucket 5 shape per query_buckets.md:
       preferred_coverage_exploration ->
       preferred_intent ->
       <preferred>_parameters ->
       fallback_intent ->
       <fallback>_parameters

    Each `<endpoint>_parameters` slot wraps the subintent variant
    (KEYWORD/METADATA) or the role-narrowed semantic subintent.
    """
    schema = get_output_schema(CategoryName[name], role)
    fields = list(schema.model_fields.keys())

    expected_fields = [
        "preferred_coverage_exploration",
        "preferred_intent",
        f"{preferred.value}_parameters",
        "fallback_intent",
        f"{fallback.value}_parameters",
    ]
    assert fields == expected_fields, (
        f"{name}/{role.name}: order mismatch — got {fields}"
    )

    pref_inner = _unwrap_optional(
        schema.model_fields[f"{preferred.value}_parameters"].annotation
    )
    expected_pref = _expected_wrapper(preferred, is_multi=True, role=role)
    assert pref_inner is expected_pref, (
        f"{name}/{role.name}: preferred ({preferred.value}) wraps "
        f"{pref_inner}, expected {expected_pref}"
    )

    fall_inner = _unwrap_optional(
        schema.model_fields[f"{fallback.value}_parameters"].annotation
    )
    expected_fall = _expected_wrapper(fallback, is_multi=True, role=role)
    assert fall_inner is expected_fall, (
        f"{name}/{role.name}: fallback ({fallback.value}) wraps "
        f"{fall_inner}, expected {expected_fall}"
    )


# ───────────────────────────────────────────────────────────────────────
# Bucket 6: semantic always + augmentation
# ───────────────────────────────────────────────────────────────────────


def _opportunity_item(schema: type[BaseModel], list_field: str) -> type[BaseModel]:
    field = schema.model_fields[list_field]
    item = typing.get_args(field.annotation)[0]
    assert isinstance(item, type) and issubclass(item, BaseModel)
    return item


@pytest.mark.parametrize(
    "name,semantic_route,deterministic_routes",
    SEMANTIC_AUGMENTATION_CATEGORIES,
    ids=[t[0] for t in SEMANTIC_AUGMENTATION_CATEGORIES],
)
@pytest.mark.parametrize("role", list(Role))
def test_bucket_6_semantic_with_augmentation_shape(
    name: str,
    semantic_route: EndpointRoute,
    deterministic_routes: list[EndpointRoute],
    role: Role,
) -> None:
    """Bucket 6 shape per query_buckets.md:
       semantic_intent ->
       augmentation_opportunities (list of {endpoint_kind ∈ deterministic, signal_description, worth_running}) ->
       <semantic>_parameters (always-fires) ->
       <each deterministic>_parameters (one per non-semantic candidate)
    """
    schema = get_output_schema(CategoryName[name], role)
    fields = list(schema.model_fields.keys())

    expected_fields = [
        "semantic_intent",
        "augmentation_opportunities",
        f"{semantic_route.value}_parameters",
    ]
    expected_fields.extend(
        f"{r.value}_parameters" for r in deterministic_routes
    )
    assert fields == expected_fields, (
        f"{name}/{role.name}: order mismatch — got {fields}"
    )

    # Semantic slot wraps the role-correct subintent.
    sem_inner = _unwrap_optional(
        schema.model_fields[f"{semantic_route.value}_parameters"].annotation
    )
    expected_sem = _expected_wrapper(semantic_route, is_multi=True, role=role)
    assert sem_inner is expected_sem, (
        f"{name}/{role.name}: semantic_parameters wraps {sem_inner}, "
        f"expected {expected_sem}"
    )

    # Each deterministic slot wraps its subintent variant.
    for det in deterministic_routes:
        inner = _unwrap_optional(
            schema.model_fields[f"{det.value}_parameters"].annotation
        )
        expected = _expected_wrapper(det, is_multi=True, role=role)
        assert inner is expected, (
            f"{name}/{role.name}: {det.value}_parameters wraps "
            f"{inner}, expected {expected}"
        )

    # augmentation_opportunities item shape + Literal coverage.
    item = _opportunity_item(schema, "augmentation_opportunities")
    item_fields = list(item.model_fields.keys())
    assert item_fields == [
        "endpoint_kind", "signal_description", "worth_running",
    ], f"{name}/{role.name}: opportunity item order — {item_fields}"
    literal_values = set(typing.get_args(item.model_fields["endpoint_kind"].annotation))
    expected_values = {r.value for r in deterministic_routes}
    assert literal_values == expected_values, (
        f"{name}/{role.name}: endpoint_kind Literal = {literal_values}, "
        f"expected {expected_values}"
    )


# ───────────────────────────────────────────────────────────────────────
# Bucket 7: character-franchise fan-out
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("role", list(Role))
def test_bucket_7_character_franchise_returns_shared_schema(role: Role) -> None:
    """Bucket 7 per query_buckets.md emits a single shared schema with
    referent_form_exploration + character_forms + franchise_forms; no
    per-endpoint payloads, no role dependency.
    """
    schema = get_output_schema(CategoryName.CHARACTER_FRANCHISE, role)
    assert schema is CharacterFranchiseFanoutSchema, (
        f"CHARACTER_FRANCHISE/{role.name}: expected "
        f"CharacterFranchiseFanoutSchema, got {schema!r}"
    )
    fields = list(schema.model_fields.keys())
    assert fields == [
        "referent_form_exploration", "character_forms", "franchise_forms",
    ], f"field order mismatch — got {fields}"
    assert schema.model_fields["character_forms"].annotation == list[str]
    assert schema.model_fields["franchise_forms"].annotation == list[str]


def test_bucket_7_schema_is_role_independent() -> None:
    """Both Role keys must point at the same class object — bucket 7
    has no SEMANTIC slot, so role cannot affect the schema."""
    carver = get_output_schema(CategoryName.CHARACTER_FRANCHISE, Role.CARVER)
    qualifier = get_output_schema(CategoryName.CHARACTER_FRANCHISE, Role.QUALIFIER)
    assert carver is qualifier


# ───────────────────────────────────────────────────────────────────────
# Bucket 8: audience-suitability combo
# ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "name,endpoints_in_order",
    SUITABILITY_CATEGORIES,
    ids=[t[0] for t in SUITABILITY_CATEGORIES],
)
@pytest.mark.parametrize("role", list(Role))
def test_bucket_8_suitability_shape(
    name: str, endpoints_in_order: list[EndpointRoute], role: Role,
) -> None:
    """Bucket 8 shape per query_buckets.md:
       suitability_overview ->
       coverage_opportunities (list of {endpoint_kind ∈ candidates, opportunity_description, worth_running}) ->
       <each endpoint>_parameters (one per candidate)
    """
    schema = get_output_schema(CategoryName[name], role)
    fields = list(schema.model_fields.keys())

    expected_fields = ["suitability_overview", "coverage_opportunities"]
    expected_fields.extend(f"{r.value}_parameters" for r in endpoints_in_order)
    assert fields == expected_fields, (
        f"{name}/{role.name}: order mismatch — got {fields}"
    )

    for r in endpoints_in_order:
        inner = _unwrap_optional(
            schema.model_fields[f"{r.value}_parameters"].annotation
        )
        expected = _expected_wrapper(r, is_multi=True, role=role)
        assert inner is expected, (
            f"{name}/{role.name}: {r.value}_parameters wraps {inner}, "
            f"expected {expected}"
        )

    # coverage_opportunities item shape + Literal coverage.
    item = _opportunity_item(schema, "coverage_opportunities")
    item_fields = list(item.model_fields.keys())
    assert item_fields == [
        "endpoint_kind", "opportunity_description", "worth_running",
    ], f"{name}/{role.name}: opportunity item order — {item_fields}"
    literal_values = set(typing.get_args(item.model_fields["endpoint_kind"].annotation))
    expected_values = {r.value for r in endpoints_in_order}
    assert literal_values == expected_values, (
        f"{name}/{role.name}: endpoint_kind Literal = {literal_values}, "
        f"expected {expected_values}"
    )


# ───────────────────────────────────────────────────────────────────────
# Cross-cutting: every CategoryName is covered by some bucket list
# ───────────────────────────────────────────────────────────────────────


def test_planning_doc_coverage_matches_enum() -> None:
    """The hand-listed bucket categories above must collectively cover
    every CategoryName member exactly once. If a new category is added
    to the enum, it needs a row in one of the bucket lists; if a name
    is renamed, this test catches the drift."""
    listed: set[str] = set()
    listed.update(NO_LLM_PURE_CODE_CATEGORIES)
    listed.update(EXPLICIT_NO_OP_CATEGORIES)
    listed.update(n for n, _ in SINGLE_NON_METADATA_CATEGORIES)
    listed.update(SINGLE_METADATA_CATEGORIES)
    listed.update(t[0] for t in PREFERRED_FALLBACK_CATEGORIES)
    listed.update(t[0] for t in SEMANTIC_AUGMENTATION_CATEGORIES)
    listed.add("CHARACTER_FRANCHISE")
    listed.update(t[0] for t in SUITABILITY_CATEGORIES)

    enum_names = {c.name for c in CategoryName}
    missing_from_tests = enum_names - listed
    extra_in_tests = listed - enum_names
    assert not missing_from_tests, (
        f"CategoryName members not covered by any bucket test list: "
        f"{sorted(missing_from_tests)}"
    )
    assert not extra_in_tests, (
        f"Test list mentions categories not in CategoryName: "
        f"{sorted(extra_in_tests)}"
    )


def test_bucket_assignment_matches_planning_doc() -> None:
    """Verify each test-listed category maps to the bucket the
    planning doc says it should be in. Catches drift between
    query_buckets.md and the code's CategoryName.bucket."""
    expected: dict[str, HandlerBucket] = {}
    for n in NO_LLM_PURE_CODE_CATEGORIES:
        expected[n] = HandlerBucket.NO_LLM_PURE_CODE
    for n in EXPLICIT_NO_OP_CATEGORIES:
        expected[n] = HandlerBucket.EXPLICIT_NO_OP
    for n, _ in SINGLE_NON_METADATA_CATEGORIES:
        expected[n] = HandlerBucket.SINGLE_NON_METADATA_ENDPOINT
    for n in SINGLE_METADATA_CATEGORIES:
        expected[n] = HandlerBucket.SINGLE_METADATA_ENDPOINT
    for t in PREFERRED_FALLBACK_CATEGORIES:
        expected[t[0]] = HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK
    for t in SEMANTIC_AUGMENTATION_CATEGORIES:
        expected[t[0]] = HandlerBucket.SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT
    expected["CHARACTER_FRANCHISE"] = HandlerBucket.CHARACTER_FRANCHISE_FANOUT
    for t in SUITABILITY_CATEGORIES:
        expected[t[0]] = HandlerBucket.AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST

    mismatches = []
    for name, expected_bucket in expected.items():
        actual = CategoryName[name].bucket
        if actual is not expected_bucket:
            mismatches.append((name, actual.name, expected_bucket.name))
    assert not mismatches, (
        f"Bucket assignment drift: {mismatches}"
    )
