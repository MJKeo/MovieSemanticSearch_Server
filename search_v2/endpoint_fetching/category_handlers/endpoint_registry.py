# Maps each (endpoint, bucket) — and for SEMANTIC, role — to the
# concrete Pydantic class the step-3 handler LLM emits as its
# response_format. Used by the sibling schema_factories module to
# dynamically build per-category output schemas from a category's
# bucket + endpoint tuple.
#
# Important: role + polarity are NOT in any wrapper schema. The
# upstream Trait (committed in step-3) owns both, and the handler
# stamps them onto the wrapper after the LLM call. Wrapper classes
# expose only `parameters` (plus `<endpoint>_retrieval_intent` on
# the subintent variants). See schemas/endpoint_parameters.py.
#
# Three lookup structures, all consulted by the single public
# accessor get_output_wrapper(endpoint, bucket, *, role=None):
#
#   ROUTE_TO_WRAPPER             — single-endpoint buckets. The
#                                   wrapper owns the entire call
#                                   and reads parameters directly
#                                   from retrieval_intent +
#                                   expressions.
#   ROUTE_TO_SUBINTENT_WRAPPER   — multi-endpoint buckets. Each
#                                   wrapper carries an
#                                   <endpoint>_retrieval_intent
#                                   field and every parameter
#                                   descriptor reads from that
#                                   field rather than from raw
#                                   inputs.
#   _SEMANTIC_DISPATCH           — semantic only: (role, is_multi)
#                                   selects between Carver/Qualifier
#                                   × regular/subintent. Semantic is
#                                   the only endpoint whose schema
#                                   shape depends on the Trait's
#                                   role (Carver and Qualifier have
#                                   structurally different
#                                   `parameters`), so it is resolved
#                                   here at lookup time rather than
#                                   exposed as a runtime Union that
#                                   OpenAI's structured-output
#                                   parse() cannot consume.
#
# Callers should always go through get_output_wrapper. The maps are
# importable for inspection / tests but should not be consulted
# directly — they are intentionally incomplete (no SEMANTIC entry,
# CHARACTER_FRANCHISE_FANOUT short-circuited inside the function).
#
# CHARACTER_FRANCHISE_FANOUT is a special case: it does not split
# into per-endpoint payloads. A single named referent drives both
# the character and the franchise retrieval, so the bucket emits one
# shared schema (CharacterFranchiseFanoutSchema below) for either
# ENTITY or FRANCHISE_STRUCTURE under that bucket.
#
# TRENDING is intentionally mapped to None: the trending endpoint
# does not run through a handler LLM (no translation wrapper exists)
# and is handled by a deterministic code path elsewhere.
#
# Lives in its own module (rather than on the EndpointRoute enum
# itself) because the wrapper classes already import from
# schemas.enums — attaching them to the enum would create a
# circular import.

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from schemas.award_translation import AwardEndpointParameters
from schemas.entity_translation import EntityEndpointParameters
from schemas.enums import EndpointRoute, HandlerBucket, Role
from schemas.franchise_translation import FranchiseEndpointParameters
from schemas.keyword_translation import (
    KeywordEndpointParameters,
    KeywordEndpointSubintentParameters,
)
from schemas.media_type_translation import MediaTypeEndpointParameters
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


# CHARACTER_FRANCHISE_FANOUT does not split into per-endpoint payloads.
# A single named referent drives both the character and the franchise
# retrieval, so the bucket emits one shared schema (referent
# identification + the form-name lists each path consumes) rather than
# two endpoint-specific wrappers. get_output_wrapper returns this class
# for either ENTITY or FRANCHISE_STRUCTURE under this bucket.
class CharacterFranchiseFanoutSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    referent_form_exploration: str = Field(
        ...,
        description=(
            "Identify the named referent and walk through the surface "
            "forms it appears under — alternate spellings, localizations, "
            "in-universe variants, named subgroups. This is the single "
            "shared identification step both retrieval paths consume. "
            "Use the user's own terms; do not invent forms not implied "
            "by the expression or general knowledge of the referent."
        ),
    )
    character_forms: list[str] = Field(
        ...,
        description=(
            "Every name, alias, or spelling the character is known by — "
            "feeds the character-presence retrieval. Include "
            "localizations and aliases a viewer might use; exclude "
            "franchise/universe titles that do not name the character "
            "directly."
        ),
    )
    franchise_forms: list[str] = Field(
        ...,
        description=(
            "Every name the franchise / universe / series is known by — "
            "feeds the franchise-lineage retrieval. Include alternate "
            "titles and umbrella series names; exclude character aliases "
            "that are not also franchise titles."
        ),
    )


# Type alias: every wrapper resolved by get_output_wrapper is a
# concrete BaseModel subclass (an EndpointParameters subclass or the
# bucket-7 fanout schema). None covers the TRENDING-no-LLM case.
WrapperRef = type[BaseModel] | None


# Single-endpoint-bucket map. The wrapper here owns the entire call
# and reads parameters directly from retrieval_intent + expressions.
# SEMANTIC is intentionally absent — handled via _SEMANTIC_DISPATCH
# inside get_output_wrapper because its schema depends on role.
ROUTE_TO_WRAPPER: dict[EndpointRoute, WrapperRef] = {
    EndpointRoute.ENTITY: EntityEndpointParameters,
    EndpointRoute.STUDIO: StudioEndpointParameters,
    EndpointRoute.METADATA: MetadataEndpointParameters,
    EndpointRoute.AWARDS: AwardEndpointParameters,
    EndpointRoute.FRANCHISE_STRUCTURE: FranchiseEndpointParameters,
    EndpointRoute.KEYWORD: KeywordEndpointParameters,
    EndpointRoute.TRENDING: None,
    EndpointRoute.MEDIA_TYPE: MediaTypeEndpointParameters,
}


# Multi-endpoint-bucket map. Each wrapper carries an
# `<endpoint>_retrieval_intent` field declared earlier in the schema
# (responsibility-splitting reasoning), and every parameter descriptor
# reads from that field rather than from the raw call inputs.
#
# SEMANTIC is intentionally absent (handled via _SEMANTIC_DISPATCH).
# Endpoints absent from this map have no subintent variant authored
# yet (entity, studio, franchise, awards, media_type); requesting one
# under a multi-endpoint bucket is treated as a routing bug and
# raises in get_output_wrapper rather than silently falling back to
# the regular wrapper, whose descriptors point at raw inputs and
# would violate the bucket's slice-of-intent contract.
ROUTE_TO_SUBINTENT_WRAPPER: dict[EndpointRoute, WrapperRef] = {
    EndpointRoute.KEYWORD: KeywordEndpointSubintentParameters,
    EndpointRoute.METADATA: MetadataEndpointSubintentParameters,
}


# (role, is_multi_endpoint_bucket) -> concrete semantic wrapper.
# Semantic is the only endpoint whose schema shape depends on the
# parent Trait's role: Carver and Qualifier have structurally
# different `parameters`. Resolved at lookup time rather than via
# a runtime Union (which OpenAI's parse() rejects as response_format).
_SEMANTIC_DISPATCH: dict[tuple[Role, bool], type[BaseModel]] = {
    (Role.CARVER,    False): CarverSemanticEndpointParameters,
    (Role.QUALIFIER, False): QualifierSemanticEndpointParameters,
    (Role.CARVER,    True):  CarverSemanticEndpointSubintentParameters,
    (Role.QUALIFIER, True):  QualifierSemanticEndpointSubintentParameters,
}


# Buckets where multiple endpoints fan out for the same category call,
# each consuming a distinct slice of intent. See
# search_improvement_planning/query_buckets.md for the full taxonomy.
# CHARACTER_FRANCHISE_FANOUT is intentionally absent — its design is the
# opposite of per-endpoint fan-out (one shared schema covers both
# retrieval paths) and it is short-circuited above this dispatch.
_MULTI_ENDPOINT_BUCKETS: frozenset[HandlerBucket] = frozenset({
    HandlerBucket.PREFERRED_REPRESENTATION_FALLBACK,
    HandlerBucket.SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT,
    HandlerBucket.AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST,
})


def get_output_wrapper(
    endpoint: EndpointRoute,
    bucket: HandlerBucket,
    *,
    role: Role | None = None,
) -> WrapperRef:
    """Return the LLM output wrapper for `endpoint` under `bucket`.

    CHARACTER_FRANCHISE_FANOUT is a special case: it does not split
    into per-endpoint payloads, so this function returns the shared
    CharacterFranchiseFanoutSchema for any endpoint asked under that
    bucket (today: ENTITY or FRANCHISE_STRUCTURE).

    Single-endpoint buckets (NO_LLM_PURE_CODE, EXPLICIT_NO_OP,
    SINGLE_NON_METADATA_ENDPOINT, SINGLE_METADATA_ENDPOINT) return the
    regular `<Endpoint>EndpointParameters` wrapper, or None for
    TRENDING (no LLM codepath).

    Multi-endpoint buckets (PREFERRED_REPRESENTATION_FALLBACK,
    SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT,
    AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST) return the
    `<Endpoint>EndpointSubintentParameters` variant. Endpoints
    without an authored subintent variant raise — the regular
    wrapper's descriptors read raw retrieval_intent + expressions
    and would violate the bucket's slice-of-intent contract.

    SEMANTIC requires `role` to disambiguate Carver vs Qualifier
    (their `parameters` shapes differ). `role` is ignored for every
    other endpoint.
    """
    # Bucket 7: shared concrete schema, irrespective of which
    # endpoint the dispatch came in on.
    if bucket is HandlerBucket.CHARACTER_FRANCHISE_FANOUT:
        return CharacterFranchiseFanoutSchema

    is_multi = bucket in _MULTI_ENDPOINT_BUCKETS

    if endpoint is EndpointRoute.SEMANTIC:
        if role is None:
            raise ValueError(
                "SEMANTIC requires `role` to pick Carver vs Qualifier; "
                "pass the parent Trait's role."
            )
        return _SEMANTIC_DISPATCH[(role, is_multi)]

    if is_multi:
        sub = ROUTE_TO_SUBINTENT_WRAPPER.get(endpoint)
        if sub is None:
            raise ValueError(
                f"No subintent wrapper authored for {endpoint.value!r} "
                f"in multi-endpoint bucket {bucket.value!r}. Author "
                f"{endpoint.value.title()}EndpointSubintentParameters "
                f"in the matching translation file before routing this "
                f"combination."
            )
        return sub

    return ROUTE_TO_WRAPPER.get(endpoint)
