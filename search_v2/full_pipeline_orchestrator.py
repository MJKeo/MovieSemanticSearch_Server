# Search V2 — Full front-half orchestrator.
#
# Drives the complete query-understanding stack for one raw user query:
#
#   1. Steps 0 + 1 in parallel on the raw query (Step 0 routing, Step 1
#      speculative spin generation). Optionally bypassed.
#   2. Step 2 fan-out: one branch per [original, spin_1, spin_2] slot
#      according to Step 0's flow budget. Branches run in parallel.
#   3. Step 3 fan-out per branch: one trait-decomposition call per
#      committed Trait, all in parallel.
#   4. Per-CategoryCall handler-LLM (or deterministic) endpoint-spec
#      generation. Fires immediately when its parent Step-3 call
#      returns — does NOT wait for sibling Step-3 calls to finish.
#
# Stops short of execution. The output is a per-branch list of traits
# with their polarity, commitment, and a fully-prepared list of
# (route, EndpointParameters | None) specs ready for stage-4 to fire.
#
# Error discipline:
#   - Step 0 failure → fatal (routing required).
#   - Step 1 failure → captured; standard flow falls back to original-
#     only.
#   - Step 2 failure (per-branch) → branch surfaces with branch_error,
#     no traits decomposed for that branch.
#   - Step 3 failure (per-trait) → trait surfaces with step_3_error
#     and an empty category_calls list.
#   - Handler query-generation failure (per-CategoryCall) → expected
#     handler-LLM double-failures soft-fail inside the handler to an
#     empty generated_specs list; unexpected generation errors surface
#     with handler_error. Neither tanks sibling CategoryCalls or traits.
#
# LLM-call discipline: every individual LLM call (Steps 0, 1, 2, 3,
# and the per-CategoryCall handler call) is wrapped with a 25-second
# timeout and exactly one retry on failure. Two attempts max per call;
# a second failure re-raises so the caller can soft-fail at the
# appropriate boundary.

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal, TypeVar

from db.postgres import fetch_quality_popularity_signals
from schemas.enums import (
    EndpointRoute,
    OperationType,
    PopularityMode,
    Polarity,
    ReceptionMode,
    ReleaseFormat,
    TraitCombineMode,
)
from schemas.implicit_expectations import ImplicitExpectationsResult
from schemas.media_type_translation import (
    MediaTypeEndpointParameters,
    MediaTypeQuerySpec,
)
from schemas.step_0_flow_routing import Step0Response
from schemas.step_1 import Step1Response
from schemas.step_2 import QueryAnalysis, Trait
from schemas.step_3 import CategoryCall
from schemas.trait_category import CategoryName

from search_v2.endpoint_fetching.category_handlers.generated_endpoint_spec import (
    GeneratedEndpointSpec,
)
from search_v2.endpoint_fetching.category_handlers.handler import (
    run_query_generation,
)
from search_v2.endpoint_fetching.metadata_query_execution import (
    score_popularity_prior,
    score_reception_prior,
)
from search_v2.stage_4_execution import (
    BranchRankedResults,
    execute_branches,
)
from search_v2.similar_movies import (
    SimilarMoviesSearchResult,
    run_similarity_search,
)
from search_v2.step_0 import run_step_0
from search_v2.step_1 import run_step_1
from search_v2.step_2 import run_step_2
from search_v2.step_3 import run_step_3
from search_v2.implicit_expectations import run_implicit_expectations

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Per-attempt timeout for any individual LLM call. Each call gets up to
# 2 attempts (initial + 1 retry), each independently bounded by this.
TIMEOUT_SECONDS = 25.0

# Total attempts per LLM call (initial + retries). The user spec is
# "1 retry", which means 2 attempts total.
LLM_MAX_ATTEMPTS = 2

QUALITY_PRIOR_BOOSTS: dict[str, float] = {
    "none": 0.0,
    "light": 0.025,
    "normal": 0.06,
    "strong": 0.10,
}

POPULARITY_PRIOR_BOOSTS: dict[str, float] = {
    "none": 0.0,
    "light": 0.05,
    "normal": 0.12,
    "strong": 0.20,
}


# Step 2 branch identifiers. Match the labels the upstream UI uses so
# downstream consumers can surface them directly.
BranchKind = Literal["original", "spin_1", "spin_2"]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CategoryCallWithEndpoints:
    """One Step-3 CategoryCall paired with its generated endpoint specs.

    A single CategoryCall can produce multiple fired endpoints when
    its bucket fans out (e.g. SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT
    fires semantic + augmentation; CHARACTER_FRANCHISE_FANOUT fires
    entity + franchise). `generated_specs` is empty when the handler
    judged nothing to fire, the call's category was EXPLICIT_NO_OP, or
    the handler LLM soft-failed. `handler_error` is reserved for
    unexpected generation errors that escape the handler.
    """

    category: CategoryName
    expressions: list[str]
    retrieval_intent: str
    generated_specs: list[GeneratedEndpointSpec]
    handler_error: str | None = None


@dataclass
class TraitWithEndpoints:
    """One committed Trait paired with its decomposed endpoint specs.

    `surface_text` is included for traceability in downstream logs and
    debug tooling — not strictly required by consumers but cheap and
    useful. `combine_mode` carries Step 3's commit for how stage-4
    folds per-category scores into a trait_score (FRAMINGS → MAX,
    FACETS → PRODUCT). Default is FRAMINGS so failure paths and test
    constructors that don't pass it land on V3-equivalent behavior.
    `step_3_error` populated only when Step 3 failed for this trait;
    `category_calls` is empty in that case.
    """

    surface_text: str
    polarity: Polarity
    commitment: Literal[
        "required", "elevated", "neutral", "supporting", "diminished"
    ]
    category_calls: list[CategoryCallWithEndpoints]
    combine_mode: TraitCombineMode = TraitCombineMode.FRAMINGS
    step_3_error: str | None = None


@dataclass
class Step2BranchResult:
    """One Step-2 branch outcome with its per-trait decompositions.

    `branch_error` populated only when Step 2 failed for this branch;
    `traits` is empty in that case. Otherwise traits carries one
    TraitWithEndpoints per committed trait, in trait-list order.
    """

    kind: BranchKind
    query: str
    ui_label: str
    traits: list[TraitWithEndpoints]
    implicit_expectations: ImplicitExpectationsResult | None = None
    implicit_expectations_error: str | None = None
    branch_error: str | None = None


@dataclass
class FullPipelineResult:
    """Top-level output of run_full_pipeline.

    Steps 0 and 1 fields are None when the orchestrator was invoked
    with skip_bypass_steps_0_1=True. The non-standard flow flags
    (exact_title / similarity) surface Step 0's routing decisions for
    callers that want to log what would have run, but those flows are
    not dispatched here.
    """

    query: str
    skipped_steps_0_1: bool
    step0_response: Step0Response | None
    step1_response: Step1Response | None
    step1_error: str | None
    exact_title_flow_executed: bool
    similarity_flow_executed: bool
    similarity_result: SimilarMoviesSearchResult | None = None
    similarity_error: str | None = None
    branches: list[Step2BranchResult] = field(default_factory=list)
    # Endpoint specs not attached to any trait — global fetches that
    # apply to the full result set rather than scoring a single trait.
    # Today the only entry is the default shorts-exclusion MEDIA_TYPE
    # fetch injected when no branch/trait emitted a MEDIA_TYPE call.
    auxiliary_endpoint_specs: list[GeneratedEndpointSpec] = field(
        default_factory=list
    )
    # Stage-4 output: per-branch ranked candidate lists. Empty list
    # when no branch executed (non-standard flows / hard Step-0
    # failure paths cannot reach this populated).
    branch_results: list[BranchRankedResults] = field(default_factory=list)
    total_elapsed: float = 0.0


class PromotionTier(IntEnum):
    """Fallback-promotion tier for endpoint specs.

    Lower positive values promote first. NEVER_PROMOTE is reserved for
    calls that must only rerank an already-existing or neutral fallback
    pool.
    """

    NEVER_PROMOTE = -1
    CONCRETE_FACT_OR_IDENTIFIER = 1
    CONCRETE_ELEMENT_OR_STRUCTURE = 2
    ABSTRACT_TYPE_OR_ARCHETYPE = 3
    RECEPTION_OR_PRAISE_PROSE = 4
    AUDIENCE_SENSITIVITY_OR_SEASONAL = 5
    VIBES_OR_CONTEXT_FIT = 6
    GLOBAL_METADATA_PRIOR_OR_ORDINAL = 7


# ---------------------------------------------------------------------------
# LLM retry / timeout helper
# ---------------------------------------------------------------------------

T = TypeVar("T")


async def _call_with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    label: str,
) -> T:
    """Run `coro_factory()` with a per-attempt timeout and one retry.

    The factory is invoked fresh on each attempt because awaitables
    are single-shot — retrying must build a new awaitable rather than
    re-await the prior one. Re-raises the last exception when both
    attempts fail; the caller decides whether to soft-fail or
    propagate.
    """
    last_exc: BaseException | None = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        attempt_start = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                coro_factory(), timeout=TIMEOUT_SECONDS
            )
        except Exception as exc:  # noqa: BLE001 — broad catch is intentional
            last_exc = exc
            if attempt < LLM_MAX_ATTEMPTS - 1:
                logger.warning(
                    "LLM call %s failed on attempt %d; retrying (%r)",
                    label,
                    attempt + 1,
                    exc,
                )
                continue
            logger.error(
                "LLM call %s failed on final attempt %d (%r)",
                label,
                attempt + 1,
                exc,
            )
            # Loop is about to end — fall through to the assert/raise
            # below rather than continue-to-no-iteration.
            break
        # Emit a per-call completion event so runners can surface
        # progress in real time. Uses INFO so default-verbose runners
        # see it; quiet callers can filter at the handler level.
        logger.info(
            "step %s completed in %.2fs (attempt %d)",
            label,
            time.perf_counter() - attempt_start,
            attempt + 1,
        )
        return result
    # Both attempts failed — last_exc is guaranteed to be set.
    assert last_exc is not None
    raise last_exc


# ---------------------------------------------------------------------------
# Step 2 branch planning. Standard flow gets a budget of 3 minus the
# count of non-standard flows that also fire, so the overall result
# UI shows three lists across all flows.
# ---------------------------------------------------------------------------


def _plan_step2_branches(
    step0: Step0Response,
    step1: Step1Response | None,
    raw_query: str,
) -> list[tuple[BranchKind, str, str]]:
    if not step0.enable_primary_flow:
        return []

    non_standard_firing = int(
        step0.exact_title_flow_data.should_be_searched
    ) + int(step0.similarity_flow_data.should_be_searched)
    budget = 3 - non_standard_firing  # 3, 2, or 1.

    branches: list[tuple[BranchKind, str, str]] = []

    # Slot 1 — original-query branch always comes first when the
    # standard flow fires. Label falls back when Step 1 failed.
    original_label = (
        step1.original_query_label if step1 is not None else "Original Query"
    )
    branches.append(("original", raw_query, original_label))

    if step1 is None:
        return branches[:budget]

    if budget >= 2 and len(step1.spins) >= 1:
        spin = step1.spins[0]
        branches.append(("spin_1", spin.query, spin.ui_label))
    if budget >= 3 and len(step1.spins) >= 2:
        spin = step1.spins[1]
        branches.append(("spin_2", spin.query, spin.ui_label))

    return branches


# ---------------------------------------------------------------------------
# Per-Step-2 branch — runs Step 2 on the branch's query, then fans out
# Step 3 + handler-LLM for every committed trait.
# ---------------------------------------------------------------------------


async def _run_branch(
    kind: BranchKind,
    query: str,
    ui_label: str,
) -> Step2BranchResult:
    try:
        qa, _, _, _ = await _call_with_retry(
            lambda: run_step_2(query),
            label=f"step_2[{kind}]",
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail per branch
        return Step2BranchResult(
            kind=kind,
            query=query,
            ui_label=ui_label,
            traits=[],
            branch_error=repr(exc),
        )

    implicit_task = _run_implicit_expectations_for_branch(query, qa, kind)
    # Per-trait fan-out runs alongside implicit-prior policy. The
    # implicit step consumes only Step-2 output, so it does not need to
    # wait for Step 3 endpoint decomposition.
    #
    # Each trait's Step 3 call receives its sibling traits' structural
    # fields (relationship_role + axis bookkeeping) so positioning
    # references can drop replaced axes without violating per-trait
    # isolation. Sibling list is computed here once per trait.
    trait_task = asyncio.gather(
        *(
            _decompose_and_generate(
                trait,
                siblings=[s for s in qa.traits if s is not trait],
                branch_label=kind,
            )
            for trait in qa.traits
        )
    )
    trait_results, (implicit, implicit_error) = await asyncio.gather(
        trait_task,
        implicit_task,
    )

    return Step2BranchResult(
        kind=kind,
        query=query,
        ui_label=ui_label,
        traits=trait_results,
        implicit_expectations=implicit,
        implicit_expectations_error=implicit_error,
    )


async def _run_implicit_expectations_for_branch(
    query: str,
    qa: QueryAnalysis,
    branch_label: str,
) -> tuple[ImplicitExpectationsResult | None, str | None]:
    """Run implicit-prior policy for one Step-2 branch.

    This is soft-fail by design: a prior-policy miss should not block
    endpoint generation or result assembly. The branch keeps the error
    for diagnostics and proceeds with no implicit policy attached.
    """
    try:
        response, _, _, _ = await _call_with_retry(
            lambda: run_implicit_expectations(query, qa),
            label=f"implicit_expectations[{branch_label}]",
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail per branch
        return None, repr(exc)
    return response, None


# ---------------------------------------------------------------------------
# Per-trait — runs Step 3, then immediately fans out per-CategoryCall
# handler-LLM calls. Does NOT wait for sibling traits' Step 3 calls
# to finish first; the gather one level up handles cross-trait
# parallelism.
# ---------------------------------------------------------------------------


async def _decompose_and_generate(
    trait: Trait,
    siblings: list[Trait],
    *,
    branch_label: str,
) -> TraitWithEndpoints:
    try:
        decomposition, _, _, _ = await _call_with_retry(
            lambda: run_step_3(trait, siblings),
            label=f"step_3[{branch_label}/{trait.surface_text!r}]",
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail per trait
        # Failure path: the trait will contribute zero in stage-4
        # regardless, so combine_mode default (FRAMINGS) is harmless.
        return TraitWithEndpoints(
            surface_text=trait.surface_text,
            polarity=trait.polarity,
            commitment=trait.commitment,
            category_calls=[],
            step_3_error=repr(exc),
        )

    # Fan out per CategoryCall in parallel as soon as Step 3 returns
    # for this trait. Each per-call coroutine soft-fails internally,
    # so default gather (no return_exceptions) is safe.
    #
    # Phase 6 — sibling-task context. Each handler receives the OTHER
    # CategoryCalls Step 3 committed for this trait (self-category
    # excluded) plus the trait-level combine_mode, so it can
    # coordinate commit-vs-abstain decisions against the parallel
    # siblings under the trait's fold rule. Identity-based filter is
    # safe because each CategoryCall is a distinct object inside this
    # decomposition.
    all_calls = list(decomposition.category_calls)
    combine_mode = decomposition.combine_mode

    # SOLO contract: exactly one category commit. The prompt instructs
    # the LLM to emit only the clean-fit primary, but if extras slip
    # through (or the model committed SOLO inconsistently with a
    # multi-call list), trim deterministically to the first listed
    # entry here so dropped categories never fan out to handler-LLM
    # calls or endpoint fetches. List ordering is the LLM's commit
    # surface — we trust the first entry is the intended primary.
    if combine_mode is TraitCombineMode.SOLO and len(all_calls) > 1:
        logger.info(
            "SOLO trim: trait %r committed SOLO with %d category_calls; "
            "keeping only the first (%s) and dropping the rest before "
            "retrieval.",
            trait.surface_text,
            len(all_calls),
            all_calls[0].category.name,
        )
        all_calls = all_calls[:1]

    category_call_results = await asyncio.gather(
        *(
            _process_category_call(
                cc,
                trait,
                branch_label=branch_label,
                sibling_calls=[s for s in all_calls if s is not cc],
                combine_mode=combine_mode,
            )
            for cc in all_calls
        )
    )

    return TraitWithEndpoints(
        surface_text=trait.surface_text,
        polarity=trait.polarity,
        commitment=trait.commitment,
        category_calls=category_call_results,
        combine_mode=combine_mode,
    )


# ---------------------------------------------------------------------------
# Per-CategoryCall — delegates category-specific endpoint-spec
# generation to category_handlers.handler. Always returns a
# CategoryCallWithEndpoints. Expected handler-LLM double-failures
# return empty specs from the handler; unexpected generation errors
# populate handler_error here.
# ---------------------------------------------------------------------------


async def _process_category_call(
    category_call: CategoryCall,
    trait: Trait,
    *,
    branch_label: str,
    sibling_calls: list[CategoryCall],
    combine_mode: TraitCombineMode,
) -> CategoryCallWithEndpoints:
    category = category_call.category
    try:
        generated_specs = await run_query_generation(
            category_call=category_call,
            trait=trait,
            sibling_calls=sibling_calls,
            combine_mode=combine_mode,
        )
    except Exception as exc:  # noqa: BLE001 — soft-fail per call
        logger.error(
            "handler query generation failed; returning empty specs "
            "(branch=%s, category=%s, error=%r)",
            branch_label,
            category.name,
            exc,
        )
        return CategoryCallWithEndpoints(
            category=category,
            expressions=list(category_call.expressions),
            retrieval_intent=category_call.retrieval_intent,
            generated_specs=[],
            handler_error=repr(exc),
        )

    return CategoryCallWithEndpoints(
        category=category,
        expressions=list(category_call.expressions),
        retrieval_intent=category_call.retrieval_intent,
        generated_specs=generated_specs,
    )


# ---------------------------------------------------------------------------
# Auxiliary (untraited) endpoint specs. Stage-4 will treat these as
# global fetches rather than per-trait scorers — today the only one
# is the default shorts-exclusion MEDIA_TYPE fetch that runs when no
# committed trait emitted a MEDIA_TYPE call. Without it, shorts could
# leak into the final result set; with it, stage-4 can intersect-out
# the entire SHORT release format from the candidate pool.
# ---------------------------------------------------------------------------


def _has_media_type_call(branches: list[Step2BranchResult]) -> bool:
    # True iff any branch has any trait with at least one CategoryCall
    # whose category is MEDIA_TYPE. We check at the category level
    # only — a category call is enough to mean the user's media-type
    # preference is already represented, regardless of whether the
    # deterministic translator produced any concrete formats.
    for branch in branches:
        for trait in branch.traits:
            for cc in trait.category_calls:
                if cc.category is CategoryName.MEDIA_TYPE:
                    return True
    return False


def _build_shorts_exclusion_spec() -> GeneratedEndpointSpec:
    # MEDIA_TYPE fetch that targets ReleaseFormat.SHORT. Stage-4 will
    # use this to fully exclude shorts from the result set when the
    # user did not otherwise express a media-type preference.
    #
    # IMPORTANT: this is the ONLY true categorical exclusion in the
    # entire pipeline. Per the design principle in
    # search_method_deterministic_logic.md ("everything is soft
    # scoring — no trait ever filters categorically"), every
    # user-expressed negative trait orchestrates as a soft downranker
    # via negative-polarity reranking. Shorts are the one exception:
    # when the user has not asked for them, they are fully removed
    # from the candidate pool rather than penalized in scoring. The
    # spec is therefore hardcoded as CANDIDATE_GENERATOR — if a
    # second auxiliary spec ever lands and isn't a categorical
    # exclusion, this assumption needs to be revisited.
    spec = MediaTypeQuerySpec(
        thinking=(
            "Default shorts exclusion: no trait emitted a MEDIA_TYPE "
            "call, so fetch all SHORT-format movies for global exclusion."
        ),
        formats=[ReleaseFormat.SHORT],
    )
    return GeneratedEndpointSpec(
        route=EndpointRoute.MEDIA_TYPE,
        params=MediaTypeEndpointParameters(parameters=spec),
        operation_type=OperationType.CANDIDATE_GENERATOR,
    )


def _build_neutral_seed_spec() -> GeneratedEndpointSpec:
    # Marker spec for the reranker-only fallback. Stage 4 executes
    # this route via db.postgres.fetch_neutral_reranker_seed_ids(),
    # which owns the seed size and weighted popularity/reception
    # formula.
    return GeneratedEndpointSpec(
        route=EndpointRoute.NEUTRAL_SEED,
        params=None,
        operation_type=OperationType.CANDIDATE_GENERATOR,
    )


def _build_auxiliary_specs(
    branches: list[Step2BranchResult],
) -> list[GeneratedEndpointSpec]:
    if _has_media_type_call(branches):
        return []
    return [_build_shorts_exclusion_spec()]


def _apply_reranker_only_candidate_fallback(
    branches: list[Step2BranchResult],
) -> list[GeneratedEndpointSpec]:
    """Promote minimum-tier rerankers or emit the neutral seed spec.

    Runs only when every trait-derived endpoint spec is currently a
    reranker. This function deliberately ignores auxiliary specs
    (shorts exclusion, neutral seed) so only user-derived calls decide
    whether fallback promotion is needed.
    """
    endpoint_refs: list[
        tuple[CategoryName, GeneratedEndpointSpec, Polarity]
    ] = []
    for branch in branches:
        for trait in branch.traits:
            for category_call in trait.category_calls:
                for spec in category_call.generated_specs:
                    endpoint_refs.append(
                        (category_call.category, spec, trait.polarity)
                    )

    if not endpoint_refs:
        return []

    if any(
        spec.operation_type is OperationType.CANDIDATE_GENERATOR
        for _, spec, _ in endpoint_refs
    ):
        return []

    tiered_refs = [
        (determine_promotion_tier(category, spec, polarity), spec)
        for category, spec, polarity in endpoint_refs
    ]
    promotable_tiers = [
        tier for tier, _ in tiered_refs
        if tier is not PromotionTier.NEVER_PROMOTE
    ]

    if not promotable_tiers:
        return [_build_neutral_seed_spec()]

    lowest_tier = min(promotable_tiers)
    for tier, spec in tiered_refs:
        if tier is lowest_tier:
            spec.operation_type = OperationType.CANDIDATE_GENERATOR
            # Mark so stage-4 rarity uses the semantic-promoted rule
            # (count of post-elbow 1.0 movies) instead of the regular
            # finder rule (all matched candidates).
            spec.was_promoted = True

    return []


_SEMANTIC_PROMOTION_TIERS: dict[CategoryName, PromotionTier] = {
    # Tier 1 — concrete fact / specific identifier.
    CategoryName.CENTRAL_TOPIC: PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
    CategoryName.PLOT_EVENTS: PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
    CategoryName.NARRATIVE_SETTING: PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
    CategoryName.FILMING_LOCATION: PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
    CategoryName.NAMED_SOURCE_CREATOR: PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
    # Tier 2 — concrete element / structural feature.
    CategoryName.ELEMENT_PRESENCE: PromotionTier.CONCRETE_ELEMENT_OR_STRUCTURE,
    CategoryName.GENRE: PromotionTier.CONCRETE_ELEMENT_OR_STRUCTURE,
    CategoryName.FORMAT_VISUAL: PromotionTier.CONCRETE_ELEMENT_OR_STRUCTURE,
    CategoryName.NARRATIVE_DEVICES: PromotionTier.CONCRETE_ELEMENT_OR_STRUCTURE,
    # Tier 3 — abstract type / archetype.
    CategoryName.CHARACTER_ARCHETYPE: PromotionTier.ABSTRACT_TYPE_OR_ARCHETYPE,
    CategoryName.STORY_THEMATIC_ARCHETYPE: PromotionTier.ABSTRACT_TYPE_OR_ARCHETYPE,
    # Tier 4 — reception / praise prose.
    CategoryName.VISUAL_CRAFT_ACCLAIM: PromotionTier.RECEPTION_OR_PRAISE_PROSE,
    CategoryName.MUSIC_SCORE_ACCLAIM: PromotionTier.RECEPTION_OR_PRAISE_PROSE,
    CategoryName.DIALOGUE_CRAFT_ACCLAIM: PromotionTier.RECEPTION_OR_PRAISE_PROSE,
    CategoryName.CULTURAL_STATUS: PromotionTier.RECEPTION_OR_PRAISE_PROSE,
    CategoryName.SPECIFIC_PRAISE_CRITICISM: PromotionTier.RECEPTION_OR_PRAISE_PROSE,
    # Tier 5 — audience / sensitivity / seasonal.
    CategoryName.TARGET_AUDIENCE: PromotionTier.AUDIENCE_SENSITIVITY_OR_SEASONAL,
    CategoryName.SENSITIVE_CONTENT: PromotionTier.AUDIENCE_SENSITIVITY_OR_SEASONAL,
    CategoryName.SEASONAL_HOLIDAY: PromotionTier.AUDIENCE_SENSITIVITY_OR_SEASONAL,
    # Tier 6 — vibes / context fit.
    CategoryName.EMOTIONAL_EXPERIENTIAL: PromotionTier.VIBES_OR_CONTEXT_FIT,
    CategoryName.VIEWING_OCCASION: PromotionTier.VIBES_OR_CONTEXT_FIT,
}

_METADATA_PROMOTION_TIERS: dict[CategoryName, PromotionTier] = {
    CategoryName.GENERAL_APPEAL: PromotionTier.GLOBAL_METADATA_PRIOR_OR_ORDINAL,
    CategoryName.CULTURAL_STATUS: PromotionTier.GLOBAL_METADATA_PRIOR_OR_ORDINAL,
    CategoryName.CHRONOLOGICAL: PromotionTier.GLOBAL_METADATA_PRIOR_OR_ORDINAL,
}


def determine_promotion_tier(
    category: CategoryName,
    endpoint_spec: GeneratedEndpointSpec,
    polarity: Polarity,
) -> PromotionTier:
    """Return the fallback-promotion tier for one endpoint spec.

    This helper is only for the reranker-only fallback path. Positive
    semantic rerankers and positive metadata-prior rerankers receive a
    promotable tier. Negative-polarity calls and already-candidate-
    generating routes are never promoted.
    """
    if polarity is Polarity.NEGATIVE:
        return PromotionTier.NEVER_PROMOTE

    if endpoint_spec.operation_type is OperationType.CANDIDATE_GENERATOR:
        return PromotionTier.NEVER_PROMOTE

    route = endpoint_spec.route
    if route is EndpointRoute.SEMANTIC:
        return _SEMANTIC_PROMOTION_TIERS.get(
            category, PromotionTier.NEVER_PROMOTE
        )

    if route is EndpointRoute.METADATA:
        return _METADATA_PROMOTION_TIERS.get(
            category, PromotionTier.NEVER_PROMOTE
        )

    return PromotionTier.NEVER_PROMOTE


# ---------------------------------------------------------------------------
# Implicit-prior post-reranking
# ---------------------------------------------------------------------------


async def _apply_implicit_prior_rerank(
    branches: list[Step2BranchResult],
    branch_results: list[BranchRankedResults],
) -> list[BranchRankedResults]:
    """Apply the implicit popularity boost, falling back to quality.

    Stage 4 owns base relevance scoring. This pass applies a single-axis
    post-score boost:

        boosted_score = base_score + prior_base * boost

    Popularity is the primary axis. The quality axis only fires when
    popularity is inactive (direction=none) — typically because the
    query already commits to popularity explicitly and the implicit
    policy turned it off. Treating popularity as the implicit-prior
    default keeps it from competing with quality when both are on; in
    saturated-popularity pools (e.g. tentpole franchise queries) the
    quality axis used to dominate by accident.

    `prior_base` is the movie's positive relevance contribution, or
    1.0 when no positive contribution exists. Missing axis data
    contributes 0.0 so absence of data has no effect.
    """
    if not branches or not branch_results:
        return branch_results

    branches_by_kind = {branch.kind: branch for branch in branches}
    return list(
        await asyncio.gather(
            *(
                _apply_implicit_prior_rerank_for_branch(
                    branches_by_kind.get(result.kind),
                    result,
                )
                for result in branch_results
            )
        )
    )


async def _apply_implicit_prior_rerank_for_branch(
    branch: Step2BranchResult | None,
    result: BranchRankedResults,
) -> BranchRankedResults:
    if (
        branch is None
        or branch.implicit_expectations is None
        or result.branch_error is not None
        or not result.ranked
    ):
        return result

    policy = branch.implicit_expectations
    quality_cap = QUALITY_PRIOR_BOOSTS[policy.quality_prior.strength]
    popularity_cap = POPULARITY_PRIOR_BOOSTS[policy.popularity_prior.strength]

    # Popularity is the primary axis. Quality only activates when the
    # implicit policy has set popularity_prior.direction = "none" —
    # typically because explicit query coverage already owns the
    # popularity axis. This avoids the saturated-popularity-pool case
    # where quality used to silently dominate the boost.
    popularity_active = (
        policy.popularity_prior.direction != "none" and popularity_cap > 0.0
    )
    quality_active = (
        not popularity_active
        and policy.quality_prior.direction != "none"
        and quality_cap > 0.0
    )
    if not popularity_active and not quality_active:
        return result

    movie_ids = [movie_id for movie_id, _ in result.ranked]
    signals = await fetch_quality_popularity_signals(movie_ids)

    reranked: list[tuple[int, float]] = []
    for movie_id, base_score in result.ranked:
        popularity_raw, reception_raw = signals.get(movie_id, (None, None))
        if popularity_active:
            popularity_signal = _popularity_signal(
                popularity_raw,
                direction=policy.popularity_prior.direction,
            )
            boost = popularity_cap * popularity_signal
        else:
            quality_signal = _quality_signal(
                reception_raw,
                direction=policy.quality_prior.direction,
            )
            boost = quality_cap * quality_signal
        breakdown = result.score_breakdowns.get(movie_id)
        prior_base = (
            breakdown.positive_total
            if breakdown is not None and breakdown.positive_total > 0.0
            else 1.0
        )
        reranked.append((movie_id, base_score + (prior_base * boost)))
        if breakdown is not None:
            breakdown.implicit_prior_boost = boost

    reranked.sort(key=lambda mid_score: mid_score[1], reverse=True)
    result.ranked = reranked
    return result


def _quality_signal(
    reception_score: float | None,
    *,
    direction: str,
) -> float:
    if direction == "none" or reception_score is None:
        return 0.0
    # Keep implicit-prior shape aligned with explicit metadata-prior
    # scoring. The metadata endpoint owns these sigmoid parameters.
    if direction == "inverse":
        return score_reception_prior(
            reception_score, ReceptionMode.POORLY_RECEIVED
        )
    return score_reception_prior(reception_score, ReceptionMode.WELL_RECEIVED)


def _popularity_signal(
    popularity_score: float | None,
    *,
    direction: str,
) -> float:
    if direction == "none" or popularity_score is None:
        return 0.0
    # Keep implicit-prior shape aligned with explicit metadata-prior
    # scoring. The metadata endpoint owns these sigmoid parameters.
    if direction == "inverse":
        return score_popularity_prior(popularity_score, PopularityMode.NICHE)
    return score_popularity_prior(popularity_score, PopularityMode.POPULAR)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_full_pipeline(
    query: str,
    *,
    skip_bypass_steps_0_1: bool = False,
) -> FullPipelineResult:
    """Run the full front-half query-understanding pipeline.

    Args:
        query: raw user query (non-empty after stripping).
        skip_bypass_steps_0_1: when True, skip Steps 0 + 1 entirely
            and feed the raw query straight into Step 2 as a single
            "original" branch. Use for testing / replay flows that
            already know they want the standard flow.

    Returns:
        FullPipelineResult with per-branch results. Each trait carries
        its polarity, commitment, and a list of CategoryCallWithEndpoints
        whose generated_specs are ready to hand to stage-4 execution.

    Raises:
        ValueError: if `query` is empty after stripping.
        Exception: if Step 0 fails on both attempts (routing is
            mandatory). Every other failure mode is captured on the
            result with a populated *_error field.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must be a non-empty string.")

    total_start = time.perf_counter()

    if skip_bypass_steps_0_1:
        # Bypass path — feed the raw query into Step 2 as a single
        # "original" branch. No flow routing, no spin generation.
        # No fan-out needed for a single branch — skip gather's
        # task-scheduling overhead.
        branch_list = [await _run_branch("original", query, "Original Query")]
        auxiliary = (
            _apply_reranker_only_candidate_fallback(branch_list)
            + _build_auxiliary_specs(branch_list)
        )
        branch_results = await execute_branches(branch_list, auxiliary)
        branch_results = await _apply_implicit_prior_rerank(
            branch_list,
            branch_results,
        )
        return FullPipelineResult(
            query=query,
            skipped_steps_0_1=True,
            step0_response=None,
            step1_response=None,
            step1_error=None,
            exact_title_flow_executed=False,
            similarity_flow_executed=False,
            similarity_result=None,
            similarity_error=None,
            branches=branch_list,
            auxiliary_endpoint_specs=auxiliary,
            branch_results=branch_results,
            total_elapsed=time.perf_counter() - total_start,
        )

    # Steps 0 and 1 in parallel. return_exceptions=True so a Step 1
    # failure does not cancel Step 0 (or vice versa) before we can
    # decide what to do with each.
    step0_result, step1_result = await asyncio.gather(
        _call_with_retry(lambda: run_step_0(query), label="step_0"),
        _call_with_retry(lambda: run_step_1(query), label="step_1"),
        return_exceptions=True,
    )

    # Step 0 failure is fatal — without a routing decision we have
    # nothing to dispatch. Re-raise so the caller knows.
    if isinstance(step0_result, BaseException):
        raise step0_result
    step0_response: Step0Response = step0_result[0]

    # Step 1 may have failed independently — capture the error and
    # keep going. The standard flow falls back to the original-query
    # branch only.
    if isinstance(step1_result, BaseException):
        step1_response: Step1Response | None = None
        step1_error: str | None = repr(step1_result)
    else:
        step1_response = step1_result[0]
        step1_error = None

    # Non-standard flows are TODO placeholders today; surface their
    # routing bits so callers can log what WOULD have run.
    exact_title_flow_executed = (
        step0_response.exact_title_flow_data.should_be_searched
    )
    similarity_flow_executed = (
        step0_response.similarity_flow_data.should_be_searched
    )

    similarity_result: SimilarMoviesSearchResult | None = None
    similarity_error: str | None = None
    if similarity_flow_executed:
        try:
            similarity_result = await run_similarity_search(
                step0_response.similarity_flow_data
            )
        except Exception as exc:  # noqa: BLE001 — non-standard side flow soft-fail
            logger.error(
                "similarity flow execution failed; continuing standard flow "
                "when present (error=%r)",
                exc,
            )
            similarity_error = repr(exc)

    # Standard flow — plan branches per the budget rule and run them
    # in parallel with per-branch error isolation.
    branch_plan = _plan_step2_branches(step0_response, step1_response, query)
    branches: list[Step2BranchResult] = []
    if branch_plan:
        branches = await asyncio.gather(
            *(_run_branch(kind, q, label) for kind, q, label in branch_plan)
        )

    auxiliary = (
        _apply_reranker_only_candidate_fallback(branches)
        + _build_auxiliary_specs(branches)
    )
    branch_results = await execute_branches(branches, auxiliary)
    branch_results = await _apply_implicit_prior_rerank(
        branches,
        branch_results,
    )
    return FullPipelineResult(
        query=query,
        skipped_steps_0_1=False,
        step0_response=step0_response,
        step1_response=step1_response,
        step1_error=step1_error,
        exact_title_flow_executed=exact_title_flow_executed,
        similarity_flow_executed=similarity_flow_executed,
        similarity_result=similarity_result,
        similarity_error=similarity_error,
        branches=branches,
        auxiliary_endpoint_specs=auxiliary,
        branch_results=branch_results,
        total_elapsed=time.perf_counter() - total_start,
    )
