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
# Buckets 5/6/8 share one shape: per-endpoint grounded walks (each
# candidate carrying strengths + weaknesses) → coverage_exploration →
# coverage_assignments → thin per-endpoint params. See
# _build_walk_then_commit.
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

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

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


# Walk-then-commit buckets (5/6/8) inherit from this so the keyword-
# walk-driven derivation of `finalized_keywords` runs once on every
# bucket output. Phase 5 design: the LLM populates `verdict` on every
# PotentialKeyword in the walk, and `finalized_keywords` on the
# downstream KeywordQuerySpecSubintent is overwritten with the deduped
# union of verdict-commits. The LLM is instructed to emit `[]` there;
# this validator is what turns the verdict choice into the executor-
# visible commit list.
class _WalkThenCommitOutputBase(_HandlerOutputBase):
    @model_validator(mode="after")
    def _derive_keyword_finalized_from_verdicts(self):
        # The bucket may or may not declare a keyword endpoint; only
        # walk-then-commit buckets that DO declare keyword carry these
        # two fields. Gracefully no-op when keyword isn't declared.
        keyword_walk = getattr(self, "keyword_walk", None)
        keyword_params_wrapper = getattr(self, "keyword_parameters", None)
        if keyword_walk is None or keyword_params_wrapper is None:
            return self

        # Walk every potential_keyword across every attribute and
        # collect the verdict-committed members. Dedupe in declaration
        # order so the executor sees the same shape as the prior
        # finalized_keywords field validator produced.
        derived: list[str] = []
        seen: set[str] = set()
        for attribute in keyword_walk.attributes:
            for candidate in attribute.potential_keywords:
                if candidate.verdict != "commit":
                    continue
                # use_enum_values=True on PotentialKeyword means the
                # field carries the string value already; coerce
                # defensively in case the upstream config drifts.
                member = candidate.keyword
                if hasattr(member, "value"):
                    member = member.value
                if member in seen:
                    continue
                seen.add(member)
                derived.append(member)

        inner = getattr(keyword_params_wrapper, "parameters", None)
        if inner is not None:
            # Overwrite whatever the LLM emitted (should be []) with
            # the derived list. Never trust LLM-emitted finalized_keywords
            # in multi-endpoint contexts post-Phase-5.
            inner.finalized_keywords = derived

        return self


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
# analysis lifted out of the per-endpoint subintent params. Each
# candidate carries strengths + weaknesses so the commitment phase
# can compose endpoints by reading off real signals, not optimism.
#
# Phase 2: coverage exploration + commitment. `coverage_exploration`
# argues which endpoints contribute distinct strengths or fill each
# other's weaknesses, BEFORE the structural commitment.
# `coverage_assignments` is then the mechanical commit, one entry
# per endpoint that should fire. Overlap is the design — multiple
# assignments catching distinct facets is expected.
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


_COVERAGE_EXPLORATION_DESC = (
    "Argue which endpoints should fire to compose coverage of the "
    "call's intent, BEFORE the structural commit below. Read off the "
    "strengths + weaknesses already written on each walk's candidates "
    "above; do not re-derive. Frame the endpoints as puzzle pieces — "
    "they may overlap partially, and that is the design when one adds "
    "specificity another lacks or fills a gap another leaves.\n"
    "\n"
    "TEST per endpoint considered: 'does this contribute a strength "
    "the others don't, OR fill a weakness another has?' Yes → fire it. "
    "No → drop it.\n"
    "TEST for dropping: 'does another endpoint dominate this one's "
    "strengths AND weaknesses (capture the same content strictly "
    "better)?' Yes → drop the dominated one.\n"
    "\n"
    "NEVER:\n"
    "- RE-NARRATE THE WALKS. Argue endpoint composition; the walks "
    "themselves carry the candidate detail.\n"
    "- HEDGE on which endpoints fire. Pick one stance per endpoint "
    "with the local test above.\n"
    "- DECLARE A SLICE UNSERVABLE HERE. If no endpoint can carry an "
    "aspect, that's a routing problem upstream, not something to "
    "memorialize."
)


# ── coverage_commitments — fixed-shape per declared endpoint ───────
#
# Phase 5: replaces the variable-length `coverage_assignments` list
# (where the LLM abstained on an endpoint by OMITTING it) with a
# fixed-shape object that has one required slot per declared endpoint.
# Each slot carries `verdict_reason` (prose, written FIRST so the
# reasoning lands before the commit) → `verdict` (commit/abstain) →
# `slice_description` (required iff verdict == "commit"). Abstention
# is now an active choice with required reasoning, not a passive
# omission. Default omission bias becomes default explicit-choice bias.

_ENDPOINT_COMMITMENT_VERDICT_REASON_DESC = (
    "One short sentence justifying the verdict below, citing the "
    "{route} walk's strengths or weaknesses already written above. "
    "Single-claim only — do NOT write OR-disjunctions ('contributes "
    "X OR fills Y'); pick one claim and commit.\n"
    "\n"
    "For 'commit': name the single distinct strength the {route} "
    "walk surfaced that the other endpoints don't carry, OR the "
    "specific weakness on a sibling endpoint that {route} fills. "
    "Cite the walk's candidate detail.\n"
    "\n"
    "For 'abstain': name exactly ONE failure mode:\n"
    "- no-walk-candidate: the {route} walk surfaced nothing useful "
    "(empty candidates or all-weakness candidates).\n"
    "- dominated-by-sibling: another declared endpoint covers "
    "{route}'s strengths AND weaknesses strictly better.\n"
    "- commitment-criteria-fail: {route}'s walk has candidates but "
    "they fail {route}'s own commitment criteria (e.g., keyword "
    "candidates that all verdict-abstain).\n"
    "\n"
    "Cite the walk's text — do NOT generate fresh reasoning here."
)

_ENDPOINT_COMMITMENT_VERDICT_DESC = (
    "Active commit choice for the {route} endpoint, read off the "
    "verdict_reason just written.\n"
    "\n"
    "- 'commit' → fire this endpoint. Fill `{route}_parameters` "
    "below with the thin commitment payload. The slice_description "
    "below names which aspect(s) of the call's intent {route} owns.\n"
    "- 'abstain' → do NOT fire this endpoint. Leave `{route}_"
    "parameters` null. Abstention here is sanctioned and frequent — "
    "the schema requires a verdict for every declared endpoint so "
    "the choice is rendered explicitly rather than buried in "
    "omission.\n"
    "\n"
    "Default to 'abstain' when the analysis is ambiguous. 'commit' "
    "requires a single named contribution-or-gap-fill claim."
)

_ENDPOINT_COMMITMENT_SLICE_DESCRIPTION_DESC = (
    "The slice of the call's intent this endpoint owns, written "
    "specifically enough that the per-endpoint parameters below can "
    "translate it without re-reading the upstream retrieval_intent. "
    "Pulled from the matching endpoint's grounded walk above — name "
    "what the walk concretely surfaced (registry members / vector "
    "spaces / columns) and what aspect(s) of the call's intent they "
    "address. This string flows to the wrapper's "
    "<endpoint>_retrieval_intent field as the per-endpoint "
    "commitment record.\n"
    "\n"
    "Required when verdict is 'commit'. When verdict is 'abstain', "
    "leave this null — the endpoint is not firing, so no slice is "
    "owned."
)

_COVERAGE_COMMITMENTS_DESC = (
    "Per-endpoint commitment record. One required field per "
    "declared endpoint, each carrying verdict_reason (prose) → "
    "verdict (commit/abstain) → slice_description (required iff "
    "commit). Mechanical commit of the composition argued in "
    "coverage_exploration above.\n"
    "\n"
    "Every declared endpoint MUST receive an explicit verdict here. "
    "Abstention is a sanctioned choice with required reasoning — "
    "you cannot abstain on an endpoint by silence. Read off the "
    "strengths + weaknesses already written on each walk; the "
    "verdict reflects what the walk concluded, it does not generate "
    "fresh reasoning.\n"
    "\n"
    "All-endpoint abstain is valid when no walk surfaced a candidate "
    "that passes both the local fire/drop tests and the endpoint's "
    "own commitment criteria — in that case the whole call abstains."
)

_THIN_PARAMETERS_DESC_TEMPLATE = (
    "Thin commitment payload for the {route} endpoint. Fill it iff "
    "`coverage_commitments.{route}.verdict == \"commit\"`; null "
    "otherwise. The wrapper's `{route}_retrieval_intent` mirrors the "
    "matching commitment's slice_description; the inner parameters "
    "draw on that intent and the upstream `{route}_walk` analysis to "
    "commit the route-specific translation."
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
    # bucket level:
    #
    #   Phase 1: per-endpoint walks. For each declared endpoint, a
    #   `{route}_walk` field holds the registry/space/column-grounded
    #   analysis (KeywordWalk / SemanticWalk / MetadataWalk). Each
    #   candidate carries strengths + weaknesses; for keyword each
    #   also carries verdict_reason → verdict (Phase 5).
    #
    #   Phase 2: coverage exploration + commitment.
    #   `coverage_exploration` argues which endpoints contribute
    #   distinct strengths or fill each other's weaknesses, BEFORE
    #   the structural commit. `coverage_commitments` (Phase 5;
    #   replaces the prior `coverage_assignments` list) is a
    #   fixed-shape object with one required EndpointCommitment slot
    #   per declared endpoint. Abstention is now an active
    #   verdict=abstain choice with required reasoning, not a
    #   passive omission.
    #
    #   Phase 3: per-endpoint thin params. One Optional
    #   `{route}_parameters` per declared endpoint, populated iff
    #   `coverage_commitments.{route}.verdict == "commit"`.
    #
    # Field declaration order matches that phase ordering — Pydantic
    # structured output emits top-down, so the LLM walks all endpoints
    # concretely before committing to who fires. coverage_exploration
    # sits between walks and commitments so the LLM reasons about
    # composition before structurally committing.
    #
    # Inheritance from _WalkThenCommitOutputBase wires the keyword-
    # walk-driven derivation of `finalized_keywords` (Phase 5).
    pairs = _resolve_wrappers_for_bucket(category, bucket)
    if not pairs:
        return None

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

    # Phase 2: coverage exploration (argue composition) →
    # coverage_commitments (mechanical commit, fixed-shape).
    fields["coverage_exploration"] = (
        str,
        Field(..., description=_COVERAGE_EXPLORATION_DESC),
    )

    # Build the per-bucket CoverageCommitments object: one required
    # EndpointCommitment field per declared endpoint, named after the
    # route value. The LLM cannot abstain by omission; every declared
    # endpoint requires an explicit verdict.
    coverage_commitments_model = _build_coverage_commitments_model(
        category=category,
        declared_routes=tuple(r for r, _ in pairs),
    )
    fields["coverage_commitments"] = (
        coverage_commitments_model,
        Field(..., description=_COVERAGE_COMMITMENTS_DESC),
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
        __base__=_WalkThenCommitOutputBase,
        __module__=__name__,
        **fields,
    )


def _build_coverage_commitments_model(
    category: CategoryName,
    declared_routes: tuple[EndpointRoute, ...],
) -> type[BaseModel]:
    """Build the per-bucket CoverageCommitments shape.

    Phase 5: replaces the variable-length `coverage_assignments` list
    with a fixed-shape object whose fields are exactly the bucket's
    declared endpoints, each required and typed as a per-bucket
    EndpointCommitment. The EndpointCommitment is also built per
    bucket (not per endpoint) so the {route} placeholders in the
    field descriptions can be specialized per route — a keyword
    commitment description references registry candidates, a
    semantic commitment description references vector spaces, and
    so on.

    All endpoint commitments follow the same field order:
    verdict_reason → verdict → slice_description. Reasoning lands
    BEFORE the verdict so the prose is generated as fresh evidence,
    not post-hoc justification (per the Iteration 6 design lesson:
    Pydantic emits fields in declaration order under structured
    output).
    """
    commitment_fields: dict[str, tuple] = {}
    for route in declared_routes:
        # Per-route EndpointCommitment with route-specialized field
        # descriptions. Built per route so the prose is concrete.
        endpoint_commitment_model = create_model(
            f"{_pascal(category.name)}{_pascal(route.value)}Commitment",
            __base__=_HandlerOutputBase,
            __module__=__name__,
            verdict_reason=(
                str,
                Field(
                    ...,
                    description=_ENDPOINT_COMMITMENT_VERDICT_REASON_DESC.format(
                        route=route.value
                    ),
                ),
            ),
            verdict=(
                Literal["commit", "abstain"],
                Field(
                    ...,
                    description=_ENDPOINT_COMMITMENT_VERDICT_DESC.format(
                        route=route.value
                    ),
                ),
            ),
            slice_description=(
                Optional[str],
                Field(
                    default=None,
                    description=_ENDPOINT_COMMITMENT_SLICE_DESCRIPTION_DESC,
                ),
            ),
        )
        commitment_fields[route.value] = (
            endpoint_commitment_model,
            Field(
                ...,
                description=(
                    f"Commitment record for the {route.value} endpoint. "
                    f"Required slot — render an explicit verdict, do not "
                    f"abstain by silence."
                ),
            ),
        )

    return create_model(
        f"{_pascal(category.name)}CoverageCommitments",
        __base__=_HandlerOutputBase,
        __module__=__name__,
        **commitment_fields,
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
