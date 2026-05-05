# Maps each (endpoint, bucket) to the concrete Pydantic class the
# step-3 handler LLM emits as its response_format. Used by the
# sibling schema_factories module to dynamically build per-category
# output schemas from a category's bucket + endpoint tuple.
#
# Important: polarity is NOT in any wrapper schema. The upstream
# Trait owns it, and the handler stamps it onto the wrapper after
# the LLM call. Wrapper classes expose only `parameters` (plus
# `<endpoint>_retrieval_intent` on the subintent variants). The
# semantic carver-vs-qualifier decision is committed by the LLM
# inside the unified `SemanticParameters.role` field rather than
# being inherited from the parent Trait. See
# schemas/endpoint_parameters.py + schemas/semantic_translation.py.
#
# Four lookup structures, all consulted by the single public
# accessor get_output_wrapper(endpoint, bucket, *, category=None):
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
#   _SEMANTIC_DISPATCH           — semantic only: is_multi selects
#                                   between regular and subintent
#                                   variants. The carver-vs-qualifier
#                                   split is no longer schema-level —
#                                   the LLM commits the retrieval
#                                   shape inside the unified schema's
#                                   `role` field, and the executor
#                                   branches on it at runtime.
#   _ENTITY_DISPATCH             — entity only: CategoryName selects
#                                   between PersonQuerySpec /
#                                   CharacterQuerySpec /
#                                   TitlePatternQuerySpec. Same
#                                   motivation as _SEMANTIC_DISPATCH:
#                                   per-category narrowing avoids the
#                                   union-typed wrapper and lets each
#                                   category receive a tighter schema.
#                                   Per-category dispatch on a route
#                                   is now a recognized pattern; new
#                                   per-category fan-outs should
#                                   follow this shape.
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
from schemas.entity_translation import (
    CharacterQuerySpec,
    PersonQuerySpec,
    TitlePatternQuerySpec,
)
from schemas.enums import EndpointRoute, HandlerBucket
from schemas.franchise_translation import FranchiseEndpointParameters
from schemas.keyword_translation import (
    KeywordEndpointParameters,
    KeywordEndpointSubintentParameters,
    KeywordWalk,
)
from schemas.media_type_translation import MediaTypeEndpointParameters
from schemas.metadata_translation import (
    MetadataEndpointParameters,
    MetadataEndpointSubintentParameters,
    MetadataWalk,
)
from schemas.semantic_translation import (
    SemanticEndpointParameters,
    SemanticEndpointSubintentParameters,
    SemanticWalk,
)
from schemas.studio_translation import StudioEndpointParameters
from schemas.trait_category import CategoryName


# CHARACTER_FRANCHISE_FANOUT does not split into per-endpoint payloads.
# A single named referent drives both the character and the franchise
# retrieval, so the bucket emits one shared schema with two parallel
# walks — one for the character side (per-film cast credits), one for
# the franchise side (series / universe / umbrella titles). The two
# walks are independent because their target indexes are different
# (cast-list strings vs. franchise titles), but they share the same
# referent. get_output_wrapper returns this class for either ENTITY or
# FRANCHISE_STRUCTURE under this bucket.
class CharacterFranchiseFanoutSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Mirrors CharacterTarget.character_exploration in
    # schemas/entity_translation.py — same film-walk template, same
    # spartan tone, same "queried form is often not the dominant
    # credit" warning. Kept in sync so the bucket-7 and bucket-3 paths
    # produce equivalent character-side reasoning.
    character_form_exploration: str = Field(
        ...,
        description=(
            "Fill in this template exactly. No commentary outside it.\n"
            "\n"
            "Films: <the most popular / relevant films featuring this "
            "character, with year — walk enough to surface every "
            "distinct credited form>\n"
            "Credit per film:\n"
            "  - <film>: <every form this character is credited under "
            "in that film's cast block, comma-separated, one identity "
            "per entry>\n"
            "  - ...\n"
            "Distinct forms: <deduped union of all credit-per-film "
            "entries, comma-separated, most common form first>\n"
            "\n"
            "A 'form' is a single atomic identity — one name per form. "
            "If a literal cast-list credit bundles multiple identities "
            "(slash-combined names, or alternate identities listed "
            "together), split it into its underlying single-identity "
            "components when listing the film's credited forms. "
            "Retrieval is exact string match against atomic credit "
            "entries; bundled strings match nothing.\n"
            "\n"
            "Skip a film if you can't recall the credit — absence "
            "beats fabrication. Skip scene quotes, fan nicknames, "
            "and descriptive phrases; only credited names.\n"
            "\n"
            "A character may be credited under several distinct forms "
            "within a single film, and under different forms across "
            "reboots or incarnations. Walk widely enough to surface "
            "every distinct form; emitting only the queried form "
            "silently drops every film credited differently."
        ),
    )
    character_forms: list[str] = Field(
        ...,
        description=(
            "Every atomic name from character_form_exploration's "
            "'Distinct forms' line. Most common form first; rest are "
            "aliases of the SAME character. Retrieval is exact-string "
            "MAX across forms — extras cost ~0, omissions silently "
            "drop films. Bias toward inclusion of any atomic form "
            "grounded in a specific film credit.\n"
            "\n"
            "Skip generic role labels and descriptive phrases — only "
            "identifiable names. Skip diacritic / casing / "
            "punctuation / hyphenation variants — normalization "
            "handles those. Empty list is valid when the referent "
            "has no real character-side presence (franchise-only)."
        ),
    )
    franchise_form_exploration: str = Field(
        ...,
        description=(
            "Fill in this template exactly. No commentary outside it.\n"
            "\n"
            "Series: <named film series / franchise titles for this "
            "referent, comma-separated>\n"
            "Umbrella: <broader shared-universe or umbrella label if "
            "applicable, else 'none'>\n"
            "Subgroups: <named phases / sagas / trilogies the referent "
            "anchors, comma-separated, else 'none'>\n"
            "Distinct forms: <deduped franchise / series / universe / "
            "umbrella titles, comma-separated>\n"
            "\n"
            "Spartan: terse, no prose. Use only widely-recognized "
            "labels (studio terminology, mainstream criticism, "
            "established fan vocabulary). Do not invent series names. "
            "Do not list character aliases here — those go on the "
            "character side."
        ),
    )
    franchise_forms: list[str] = Field(
        ...,
        description=(
            "Every distinct franchise / series / universe / umbrella "
            "title from franchise_form_exploration's Distinct forms "
            "line. Includes alternate titles and umbrella series "
            "names; excludes character aliases.\n"
            "\n"
            "Empty list is valid when the referent is character-only "
            "(no anchored film series). Avoid orthographic variants — "
            "the franchise tokenizer collapses casing / punctuation / "
            "hyphenation."
        ),
    )


# Type alias: every wrapper resolved by get_output_wrapper is a
# concrete BaseModel subclass (an EndpointParameters subclass or the
# bucket-7 fanout schema). None covers the TRENDING-no-LLM case.
WrapperRef = type[BaseModel] | None


# Single-endpoint-bucket map. The wrapper here owns the entire call
# and reads parameters directly from retrieval_intent + expressions.
# SEMANTIC is intentionally absent — handled via _SEMANTIC_DISPATCH
# inside get_output_wrapper because its schema also has a subintent
# variant for multi-endpoint buckets.
# ENTITY is intentionally absent — handled via _ENTITY_DISPATCH
# because its schema depends on the parent CategoryName.
ROUTE_TO_WRAPPER: dict[EndpointRoute, WrapperRef] = {
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


# is_multi_endpoint_bucket -> concrete semantic wrapper. Both
# variants embed a unified SemanticParameters whose `role` field
# (carver / qualifier) is committed by the LLM per call; the
# executor reads that field at runtime. The subintent variant adds
# `semantic_retrieval_intent` on the wrapper so every parameter
# descriptor reads from the slice-of-intent assigned to this
# endpoint by upstream responsibility-splitting reasoning.
_SEMANTIC_DISPATCH: dict[bool, type[BaseModel]] = {
    False: SemanticEndpointParameters,
    True:  SemanticEndpointSubintentParameters,
}


# Per-endpoint walk classes — the registry/space/column-grounded
# analysis layer that lives at the bucket level in multi-endpoint
# buckets (5/6/8). Emitted BEFORE the bucket-level
# coverage_assignments commitment so the LLM walks concrete
# candidates (registry members, vector spaces, structured columns)
# before committing whether and how to fire each endpoint.
#
# Endpoints absent from this map have no walk class authored — they
# are not currently routed to multi-endpoint buckets, or their
# routing is deterministic / shape-special (TRENDING, MEDIA_TYPE,
# CHARACTER_FRANCHISE_FANOUT). Adding a new multi-endpoint route to a
# bucket requires authoring a walk class first; the schema factory
# raises if it asks for one and finds nothing.
ROUTE_TO_WALK: dict[EndpointRoute, type[BaseModel]] = {
    EndpointRoute.KEYWORD: KeywordWalk,
    EndpointRoute.SEMANTIC: SemanticWalk,
    EndpointRoute.METADATA: MetadataWalk,
}


# CategoryName -> concrete entity spec. Entity is the second endpoint
# whose schema shape depends on something other than (route, bucket):
# the three entity-family categories each receive a different spec
# class so the LLM gets a tighter schema than a Union would yield.
# Resolved at lookup time for the same reason as _SEMANTIC_DISPATCH —
# the previous approach (a single EntityEndpointParameters whose
# `parameters` was a Union of the three specs) was steered entirely
# by the per-category prompt and offered no schema-level narrowing.
#
# CHARACTER_FRANCHISE_FANOUT also routes through ENTITY but is
# short-circuited at the top of get_output_wrapper (its bucket emits
# one shared schema) so it does not need an entry here.
_ENTITY_DISPATCH: dict[CategoryName, type[BaseModel]] = {
    CategoryName.PERSON_CREDIT: PersonQuerySpec,
    CategoryName.NAMED_CHARACTER: CharacterQuerySpec,
    CategoryName.TITLE_TEXT: TitlePatternQuerySpec,
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
    category: CategoryName | None = None,
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

    SEMANTIC returns the unified wrapper (or its subintent variant);
    the LLM commits carver vs qualifier inside the schema's `role`
    field. ENTITY requires `category` to disambiguate Person /
    Character / Title (each category gets a different spec class);
    the `category` kwarg is ignored for every non-ENTITY endpoint.
    """
    # Bucket 7: shared concrete schema, irrespective of which
    # endpoint the dispatch came in on. Short-circuited above the
    # ENTITY / SEMANTIC branches so the per-category dispatch does
    # not run for this bucket.
    if bucket is HandlerBucket.CHARACTER_FRANCHISE_FANOUT:
        return CharacterFranchiseFanoutSchema

    is_multi = bucket in _MULTI_ENDPOINT_BUCKETS

    if endpoint is EndpointRoute.SEMANTIC:
        return _SEMANTIC_DISPATCH[is_multi]

    if endpoint is EndpointRoute.ENTITY:
        # ENTITY has no subintent variants — the three specs are
        # already category-narrowed and there is no multi-endpoint
        # bucket today that pairs ENTITY with another entity-family
        # endpoint requiring a slice-of-intent split.
        if is_multi:
            raise ValueError(
                f"ENTITY has no subintent wrapper authored for "
                f"multi-endpoint bucket {bucket.value!r}. Per-category "
                f"specs do not yet expose an `entity_retrieval_intent` "
                f"field; author one before routing this combination."
            )
        if category is None:
            raise ValueError(
                "ENTITY requires `category` to pick Person / Character "
                "/ Title spec; pass the parent CategoryName."
            )
        spec = _ENTITY_DISPATCH.get(category)
        if spec is None:
            raise ValueError(
                f"No entity spec registered for category "
                f"{category.name!r}. Add an entry to _ENTITY_DISPATCH."
            )
        return spec

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


def get_walk_class(endpoint: EndpointRoute) -> type[BaseModel] | None:
    """Return the bucket-level walk class for `endpoint`, or None.

    Walks live at the bucket level in multi-endpoint buckets (5/6/8)
    and carry the registry/space/column-grounded analysis the LLM
    runs before the coverage_assignments commitment phase. Endpoints
    that don't appear in ROUTE_TO_WALK have no walk authored — they
    aren't currently routed to multi-endpoint buckets, or their
    routing is deterministic.

    The schema factory uses this to assemble the per-endpoint walk
    fields on a multi-endpoint bucket schema. Raises (via the caller)
    when a multi-endpoint bucket declares a route with no walk class
    — the bucket can't be assembled until one is authored.
    """
    return ROUTE_TO_WALK.get(endpoint)
