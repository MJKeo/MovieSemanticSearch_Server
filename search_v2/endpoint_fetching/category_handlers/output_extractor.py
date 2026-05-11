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
# reading — CHARACTER_FRANCHISE forces CENTRAL prominence downstream
# regardless of what the schema does or doesn't say here, since a
# CHARACTER_FRANCHISE referent (Batman, Bond, Spider-Man) by definition
# anchors its films.
# Executors do not read these strings; they exist purely as LLM
# scaffolding on the spec models, so a stub or a copy preserves
# schema validity without affecting retrieval behavior.
_FANOUT_PROMINENCE_EXPLORATION_STUB = (
    "centrality forced to CENTRAL — CHARACTER_FRANCHISE referents "
    "anchor their films by definition."
)

_FANOUT_CHARACTER_QUERY_EXPLORATION_STUB = (
    "single referent — fanout retrieval emits one CharacterTarget "
    "per call by design."
)


# Metadata's MetadataTranslationOutputSubintent has every column
# nested under .column_spec; "all 10 columns null" is the structural
# vacuum signal. Listed here as the canonical 10 columns documented
# on the wrapper's metadata_retrieval_intent description.
_METADATA_COLUMN_FIELDS: tuple[str, ...] = (
    "release_date",
    "runtime",
    "maturity_rating",
    "streaming",
    "audio_language",
    "country_of_origin",
    "budget_scale",
    "box_office",
    "popularity",
    "reception",
)


def _is_vacuous_spec(
    route: EndpointRoute,
    wrapper: object,
) -> bool:
    """True when the wrapper's params are structurally empty.

    Symmetric with `coverage_commitments.{route}.verdict == "abstain"`
    at the bucket level — covers the case where the LLM committed to
    fire {route} but produced no actual content. Iter 8 Q5 surfaced
    this on GENRE keyword: coverage_commitments.keyword.verdict=commit
    paired with every PotentialKeyword.verdict=abstain, leaving the
    server-derived finalized_keywords=[]. Without this filter, stage
    4 runs the empty query and scores 0.0 across the board, which
    multiplies through within-category ADDITIVE and across-category
    FACETS folds into trait death (Phase 7's floor mitigates the
    final-trait-score outcome but does not prevent the empty endpoint
    from contributing 0.0 in the first place).

    Routes covered:
    - KEYWORD: parameters.finalized_keywords empty (post Iter 9 the
      schema enforces min_length=1 directly, but the check stays as
      a defense-in-depth guard).
    - SEMANTIC: parameters.space_queries empty (schema-enforced
      min_length=1 already; check stays as defense-in-depth).
    - METADATA: every column field on parameters.column_spec is None.
      Schema does not enforce non-emptiness on the column_spec, so
      this is a load-bearing check.

    Other routes (entity, franchise, studio, awards) have no
    structurally-vacuous emission path through this extractor — they
    either emit non-empty wrapper content or omit the field entirely.
    """
    inner = getattr(wrapper, "parameters", None)
    if inner is None:
        return True
    if route is EndpointRoute.KEYWORD:
        return not getattr(inner, "finalized_keywords", None)
    if route is EndpointRoute.SEMANTIC:
        return not getattr(inner, "space_queries", None)
    if route is EndpointRoute.METADATA:
        column_spec = getattr(inner, "column_spec", None)
        if column_spec is None:
            return True
        return all(
            getattr(column_spec, field, None) is None
            for field in _METADATA_COLUMN_FIELDS
        )
    # Other routes carry no vacuous-emission path; the wrapper's
    # presence is itself the fired signal.
    return False


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
            route = EndpointRoute(route_value)
            # Iter 9 fix #1: a wrapper with structurally vacuous params
            # is equivalent to coverage_commitments.{route}.verdict ==
            # "abstain" — the LLM committed at the bucket level but the
            # per-column / per-keyword choices it owed at the wrapper
            # level all came up empty. Treat as not-fired so stage 4
            # doesn't run a content-less query that scores 0.0 across
            # every candidate (which would multiply through ADDITIVE
            # within-category and FACETS across-category folds into
            # trait death). Iter 8 Q5 GENRE was the surfacing case:
            # coverage_commitments.keyword.verdict=commit but every
            # PotentialKeyword.verdict abstained, leaving server-derived
            # finalized_keywords=[]. Phase 7's floor masked the trait-
            # death; this filter closes the loop.
            if _is_vacuous_spec(route, value):
                continue
            fired.append((route, value))
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
    # prominence_mode is hard-forced to CENTRAL: a CHARACTER_FRANCHISE
    # referent (Batman, Bond, Spider-Man) by definition anchors its
    # films, so we always bias toward titles where the character is
    # central — the LLM is not given the choice to weaken this.
    character_forms = list(output.character_forms)
    if character_forms:
        character_spec = CharacterQuerySpec(
            query_exploration=_FANOUT_CHARACTER_QUERY_EXPLORATION_STUB,
            targets=[
                CharacterTarget(
                    character_exploration=output.character_form_exploration,
                    forms=character_forms,
                    prominence_exploration=_FANOUT_PROMINENCE_EXPLORATION_STUB,
                    prominence_mode=CharacterProminenceMode.CENTRAL,
                )
            ],
        )
        fired.append((EndpointRoute.ENTITY, character_spec))

    # Franchise path — the franchise endpoint is name-axis-driven for
    # the fanout case. lineage_position / structural_flags / launch_
    # scope are deliberately left unset; the fanout schema commits no
    # narrative-position or structural reading. prefer_lineage is hard-
    # forced to True: CHARACTER_FRANCHISE names a specific character-
    # anchored franchise, so main-line films should outrank shared-
    # universe-only matches. FranchiseQuerySpec's validator will coerce
    # back to False if the franchise_names projection is mechanically
    # incompatible (multi-name list, etc.), so this is safe to force.
    franchise_forms = list(output.franchise_forms)
    if franchise_forms:
        franchise_wrapper = FranchiseEndpointParameters(
            parameters=FranchiseQuerySpec(
                request_overview=output.franchise_form_exploration,
                franchise_names=franchise_forms,
                prefer_lineage=True,
            ),
        )
        fired.append((EndpointRoute.FRANCHISE_STRUCTURE, franchise_wrapper))

    return fired
