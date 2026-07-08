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
from schemas.step_1 import Step1ClarificationResponse, Step1Response
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
from search_v2.promotion_tiers import (
    PromotionTier,
    determine_promotion_tier,
)
from search_v2.stage_4_execution import (
    BranchRankedResults,
    execute_branches,
)
from search_v2.character_franchise_search import (
    CharacterFranchiseSearchResult,
    run_character_franchise_search,
)
from search_v2.non_character_franchise_search import (
    NonCharacterFranchiseSearchResult,
    run_non_character_franchise_search,
)
from search_v2.similar_movies import (
    SimilarMoviesSearchResult,
    run_similarity_search,
)
from search_v2.query_input_validation import clean_clarification, clean_query
from search_v2.step_0 import run_step_0
from search_v2.step_1 import run_step_1
from search_v2.studio_search import (
    StudioSearchResult,
    run_studio_search,
)
from search_v2.person_search import (
    PersonSearchResult,
    run_person_search,
)
from search_v2.step_2 import run_step_2
from search_v2.step_3 import run_step_3
from search_v2.implicit_expectations import run_implicit_expectations

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from observability.names import (
    QUERY_SEARCH_STEP_2,
    QUERY_SEARCH_STEP_2_TRAIT_COUNT,
    QUERY_SEARCH_STEP_2_CONTEXTUALIZED_PHRASES,
    QUERY_SEARCH_TRAIT,
    QUERY_SEARCH_TRAIT_PHRASE,
    QUERY_SEARCH_TRAIT_POLARITY,
    QUERY_SEARCH_TRAIT_COMMITMENT,
    QUERY_SEARCH_TRAIT_STEP_3_ERROR,
    QUERY_SEARCH_STEP_3,
    QUERY_SEARCH_STEP_3_COMBINE_MODE,
    QUERY_SEARCH_STEP_3_CATEGORIES,
)

logger = logging.getLogger(__name__)

# Per-module tracer. A no-op ProxyTracer when `setup_tracing` hasn't run
# (offline ingestion/eval imports), so the manual spans below are cheap
# no-ops there. The Step 2/3 spans nest under whatever span is current —
# in the live pipeline that is the `query_search.branch` span, since each
# branch task is activated under it in the streaming orchestrator.
tracer = trace.get_tracer(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Timeout / retry / jitter discipline now lives at the lowest layer
# inside `implementation.llms.generic_methods.generate_llm_response_async`.
# This module no longer wraps step calls with its own retry — see
# `_call_with_retry` below, which is now a transparent passthrough so
# callers and tests keep their existing call shape.

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
    step1_response: Step1Response | Step1ClarificationResponse | None
    step1_error: str | None
    exact_title_flow_executed: bool
    similarity_flow_executed: bool
    non_character_franchise_flow_executed: bool = False
    character_franchise_flow_executed: bool = False
    studio_flow_executed: bool = False
    person_flow_executed: bool = False
    similarity_result: SimilarMoviesSearchResult | None = None
    similarity_error: str | None = None
    non_character_franchise_result: NonCharacterFranchiseSearchResult | None = None
    non_character_franchise_error: str | None = None
    character_franchise_result: CharacterFranchiseSearchResult | None = None
    character_franchise_error: str | None = None
    studio_result: StudioSearchResult | None = None
    studio_error: str | None = None
    person_result: PersonSearchResult | None = None
    person_error: str | None = None
    branches: list[Step2BranchResult] = field(default_factory=list)
    # Per-branch auxiliary endpoint specs — parallel to `branches`.
    # Each inner list is the auxiliary specs that were applied to the
    # branch at the same index (shorts-exclusion MEDIA_TYPE fetch when
    # that branch didn't already emit a MEDIA_TYPE call; neutral seed
    # when the branch's specs were all rerankers and nothing was
    # promotable). Per-branch rather than global so each branch can
    # dispatch Stage 4 independently as soon as its Step 3 finishes.
    auxiliary_endpoint_specs_per_branch: list[list[GeneratedEndpointSpec]] = field(
        default_factory=list
    )
    # Stage-4 output: per-branch ranked candidate lists. Empty list
    # when no branch executed (non-standard flows / hard Step-0
    # failure paths cannot reach this populated).
    branch_results: list[BranchRankedResults] = field(default_factory=list)
    total_elapsed: float = 0.0


# ---------------------------------------------------------------------------
# LLM retry / timeout helper
# ---------------------------------------------------------------------------

T = TypeVar("T")


async def _call_with_retry(
    coro_factory: Callable[[], Awaitable[T]],
    *,
    label: str,
) -> T:
    """Run `coro_factory()` and time it.

    Retry / timeout / jitter live one layer below us, inside
    `generate_llm_response_async`. This wrapper still exists so
    callers (including the streaming orchestrator) keep their
    existing call shape, and so the per-step "completed in Xs"
    INFO log keeps surfacing for live runners.

    Exceptions from the inner call propagate untouched.
    """
    started = time.perf_counter()
    result = await coro_factory()
    logger.info(
        "step %s completed in %.2fs",
        label,
        time.perf_counter() - started,
    )
    return result


# ---------------------------------------------------------------------------
# Step 2 branch planning. Standard flow gets a budget of 3 minus the
# count of non-standard flows that also fire, so the overall result
# UI shows three lists across all flows.
# ---------------------------------------------------------------------------


def _standard_branch_count(step0: Step0Response) -> int:
    """Number of standard-flow branches Step 0's routing budgets for.

    The standard flow shares a fixed 3-branch UI budget with the
    non-standard flows: it gets whatever slots the entity flows leave
    (budget = 3 - non-standard flows firing), and 0 when it doesn't
    fire at all. A pure function of Step 0, so it is the single source
    of truth for the branch budget — `_step1_needed` and
    `_plan_step2_branches` both read it, and it doubles as the
    `query_search.step_0_standard_branch_count` telemetry value.
    """
    if not step0.fire_standard_flow:
        return 0
    return 3 - _non_standard_firing_count(step0)


def _step1_needed(step0: Step0Response) -> bool:
    """True iff Step 0's routing decision could consume Step 1 spins.

    Spins are consumed only when the standard flow runs AND its
    budget leaves room for at least one spin (branch 1 is the main
    query/rewrite; spins fill branches 2-3). So spins fire exactly
    when the standard budget is >= 2. Anything else means Step 1's
    output is throw-away work the orchestrator can cancel as soon as
    Step 0 returns.
    """
    return _standard_branch_count(step0) >= 2


def _non_standard_firing_count(step0: Step0Response) -> int:
    """How many non-standard executor flows would run for this step 0 routing.

    Each entity flow with an executor (exact_title, similarity,
    non_character_franchise, character_franchise, studio, person)
    consumes one slot of the standard-flow budget.
    """
    return (
        int(step0.to_exact_title_flow_data() is not None)
        + int(step0.to_similarity_flow_data() is not None)
        + int(step0.to_non_character_franchise_flow_data() is not None)
        + int(step0.to_character_franchise_flow_data() is not None)
        + int(step0.to_studio_flow_data() is not None)
        + int(step0.to_person_flow_data() is not None)
    )


def _plan_step2_branches(
    step0: Step0Response,
    step1: Step1Response | Step1ClarificationResponse | None,
    raw_query: str,
    *,
    clarification: str | None = None,
) -> list[tuple[BranchKind, str, str]]:
    if not step0.fire_standard_flow:
        return []

    budget = _standard_branch_count(step0)  # 3, 2, or 1 (see helper).

    branches: list[tuple[BranchKind, str, str]] = []

    # Slot 1 — the "main" branch. Three shapes:
    #   - Clarification mode, Step 1 succeeded: use the faithful merge
    #     produced in step1.main_rewrite. UI shows what we're actually
    #     searching.
    #   - Clarification mode, Step 1 failed: Step 1's rewrite is
    #     missing, but the user's correction signal must not be
    #     silently dropped — searching the verbatim original would
    #     return the same results that already missed the mark. Fall
    #     back to a crude concatenation so Step 2 sees both inputs.
    #   - No clarification (Step 1 succeeded or failed): no rewrite to
    #     use, so slot 1 carries the raw user query in both fields.
    if isinstance(step1, Step1ClarificationResponse):
        branches.append(
            (
                "original",
                step1.main_rewrite.query,
                step1.main_rewrite.ui_label,
            )
        )
    elif step1 is None and clarification:
        merged = f"{raw_query}. {clarification}"
        branches.append(("original", merged, merged))
    else:
        branches.append(("original", raw_query, raw_query))

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
    # Composition of the two phases below — kept as a one-liner so
    # existing callers (run_full_pipeline) see no behavior change.
    partial, qa = await _run_step2_for_branch(kind, query, ui_label)
    if qa is None:
        # Step 2 failed; partial already carries branch_error.
        return partial
    return await _finish_branch_after_step2(partial, qa, kind)


async def _run_step2_for_branch(
    kind: BranchKind,
    query: str,
    ui_label: str,
) -> tuple[Step2BranchResult, "QueryAnalysis | None"]:
    """Run Step 2 and return (partial branch, QueryAnalysis or None).

    On Step 2 failure: returns a Step2BranchResult with branch_error
    populated, and QueryAnalysis=None. On success: returns a partial
    Step2BranchResult (traits=[] until Step 3 fills them in) and the
    QueryAnalysis the caller needs to drive Step 3.

    The streaming orchestrator uses this two-phase split so it can emit
    a `branch_traits` event between Step 2 and Step 3 fan-out.
    """
    # The step_2 span closes at the Step-2 LLM return (the trait spans start
    # after), so its duration is just this call. It nests under the branch span
    # (current context) and parents the router's `llm.generate` child.
    with tracer.start_as_current_span(QUERY_SEARCH_STEP_2) as step2_span:
        try:
            qa, _, _, _ = await _call_with_retry(
                lambda: run_step_2(query),
                label=f"step_2[{kind}]",
            )
        except Exception as exc:  # noqa: BLE001 — soft-fail per branch
            # A per-branch soft-fail: mark the failing span ERROR (the LLM step
            # genuinely failed) but let the branch degrade via branch_error —
            # the request verdict is untouched.
            step2_span.record_exception(exc)
            step2_span.set_status(Status(StatusCode.ERROR))
            return (
                Step2BranchResult(
                    kind=kind,
                    query=query,
                    ui_label=ui_label,
                    traits=[],
                    branch_error=repr(exc),
                ),
                None,
            )

        step2_span.set_attribute(
            QUERY_SEARCH_STEP_2_TRAIT_COUNT, len(qa.traits)
        )
        step2_span.set_attribute(
            QUERY_SEARCH_STEP_2_CONTEXTUALIZED_PHRASES,
            [t.contextualized_phrase for t in qa.traits],
        )

    partial = Step2BranchResult(
        kind=kind,
        query=query,
        ui_label=ui_label,
        traits=[],
    )
    return partial, qa


async def _finish_branch_after_step2(
    partial: Step2BranchResult,
    qa: "QueryAnalysis",
    kind: BranchKind,
) -> Step2BranchResult:
    """Run Step 3 + handler fan-out alongside implicit policy.

    Mirrors the post-Step-2 half of the original `_run_branch`: implicit
    expectations and per-trait Step 3/handler fan-out run in parallel,
    then both fold into the returned Step2BranchResult. Constructing
    the implicit-expectations coroutine here (rather than passing one
    in) keeps cancellation correct — `asyncio.gather` cancels its inner
    coroutines when the outer task is cancelled.
    """
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
        _run_implicit_expectations_for_branch(partial.query, qa, kind),
    )
    partial.traits = trait_results
    partial.implicit_expectations = implicit
    partial.implicit_expectations_error = implicit_error
    return partial


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
    # One trait span brackets this trait's whole Step 3 -> query-generation
    # lifecycle, so the nested step_3 and query_generation spans (and their
    # `llm.generate` children) render as one trait's work. It nests under the
    # branch span; the gather one level up runs each trait in its own task, so
    # each trait span scopes correctly via the copied OTel context.
    with tracer.start_as_current_span(QUERY_SEARCH_TRAIT) as trait_span:
        trait_span.set_attribute(QUERY_SEARCH_TRAIT_PHRASE, trait.contextualized_phrase)
        trait_span.set_attribute(QUERY_SEARCH_TRAIT_POLARITY, trait.polarity.value)
        # commitment is a Literal[str], already a plain str — set directly.
        trait_span.set_attribute(QUERY_SEARCH_TRAIT_COMMITMENT, trait.commitment)

        # solo_drop_count > 0 records that the SOLO trim fired; captured inside
        # the step_3 span and emitted as a trait-span event once it closes.
        solo_drop_count = 0
        kept_category: str | None = None

        # The step_3 span wraps just the Step-3 LLM call plus the deterministic
        # SOLO trim, so it records the POST-trim combine_mode / categories.
        with tracer.start_as_current_span(QUERY_SEARCH_STEP_3) as step3_span:
            try:
                decomposition, _, _, _ = await _call_with_retry(
                    lambda: run_step_3(trait, siblings),
                    label=f"step_3[{branch_label}/{trait.surface_text!r}]",
                )
            except Exception as exc:  # noqa: BLE001 — soft-fail per trait
                # Per-trait soft-fail: mark the failing step_3 span ERROR, and
                # record the degradation as an attribute on the trait span (which
                # stays UNSET — the request verdict is untouched). The trait
                # contributes zero in stage-4; combine_mode default is harmless.
                step3_span.record_exception(exc)
                step3_span.set_status(Status(StatusCode.ERROR))
                trait_span.set_attribute(
                    QUERY_SEARCH_TRAIT_STEP_3_ERROR, repr(exc)
                )
                return TraitWithEndpoints(
                    surface_text=trait.surface_text,
                    polarity=trait.polarity,
                    commitment=trait.commitment,
                    category_calls=[],
                    step_3_error=repr(exc),
                )

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
                solo_drop_count = len(all_calls) - 1
                kept_category = all_calls[0].category.name
                logger.info(
                    "SOLO trim: trait %r committed SOLO with %d category_calls; "
                    "keeping only the first (%s) and dropping the rest before "
                    "retrieval.",
                    trait.surface_text,
                    len(all_calls),
                    kept_category,
                )
                all_calls = all_calls[:1]

            # POST-trim state — the calls that actually reach retrieval.
            step3_span.set_attribute(
                QUERY_SEARCH_STEP_3_COMBINE_MODE, combine_mode.value
            )
            step3_span.set_attribute(
                QUERY_SEARCH_STEP_3_CATEGORIES,
                [c.category.name for c in all_calls],
            )

        if solo_drop_count > 0:
            trait_span.add_event(
                "solo trim",
                {"kept_category": kept_category, "dropped_count": solo_drop_count},
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


def _branch_has_media_type_call(branch: Step2BranchResult) -> bool:
    # True iff the branch has any trait with at least one CategoryCall
    # whose category is MEDIA_TYPE. We check at the category level
    # only — a category call is enough to mean the user's media-type
    # preference is already represented in this branch, regardless of
    # whether the deterministic translator produced any concrete
    # formats. Decisions are per-branch now: each branch independently
    # decides whether it needs the shorts-exclusion auxiliary spec,
    # which is what lets the streaming orchestrator dispatch Stage 4
    # for a branch the moment its Step 3 completes — no cross-branch
    # gate.
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
    branch: Step2BranchResult,
) -> list[GeneratedEndpointSpec]:
    """Per-branch auxiliary specs.

    Returns the shorts-exclusion MEDIA_TYPE spec when *this* branch
    didn't emit its own MEDIA_TYPE call. Previously this was a global
    decision across all branches; making it per-branch lets the
    streaming pipeline dispatch each branch's Stage 4 as soon as its
    Step 3 completes (no cross-branch gate). Semantic note: a branch
    that doesn't mention media type now gets shorts-excluded
    independently of what its sibling branches did — which is the
    right call, because each branch represents an isolated intent.
    """
    if _branch_has_media_type_call(branch):
        return []
    return [_build_shorts_exclusion_spec()]


def _apply_reranker_only_candidate_fallback(
    branch: Step2BranchResult,
) -> list[GeneratedEndpointSpec]:
    """Per-branch reranker-only fallback.

    If *this* branch's trait-derived endpoint specs are all rerankers
    (no candidate generator anywhere in the branch), promote the
    lowest-tier reranker to a CANDIDATE_GENERATOR or, if nothing is
    promotable, emit a neutral seed spec. Previously this decision
    spanned every branch — but in practice each branch needs its own
    candidate generator to produce a non-empty pool in Stage 4 Phase
    B, so per-branch fallback is strictly more correct: a branch that
    happened to be all-reranker would silently produce no results
    just because a sibling had a candidate generator.

    Auxiliary specs (shorts exclusion, neutral seed) are deliberately
    ignored when deciding whether to promote — only user-derived
    calls drive fallback.
    """
    endpoint_refs: list[
        tuple[CategoryName, GeneratedEndpointSpec, Polarity]
    ] = []
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


def _compute_branch_auxiliary(
    branch: Step2BranchResult,
) -> list[GeneratedEndpointSpec]:
    """Convenience: full per-branch auxiliary list.

    Concatenates the reranker-only fallback (which may mutate the
    branch's own specs in place) and the shorts-exclusion check.
    Order matches the previous global call sites so behavior stays
    stable for a single-branch case.
    """
    return (
        _apply_reranker_only_candidate_fallback(branch)
        + _build_auxiliary_specs(branch)
    )


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
    clarification: str | None = None,
    skip_bypass_steps_0_1: bool = False,
) -> FullPipelineResult:
    """Run the full front-half query-understanding pipeline.

    Args:
        query: raw user query (non-empty after stripping).
        clarification: optional follow-up correction the user supplied
            to refine the original query. When present, threads into
            Steps 0 and 1, which use clarification-mode prompts.
            Step 1's clarification response carries a main_rewrite
            (faithful merge of original + clarification) that replaces
            the raw-query slot in the branch plan.
        skip_bypass_steps_0_1: when True, skip Steps 0 + 1 entirely
            and feed the raw query straight into Step 2 as a single
            "original" branch. Use for testing / replay flows that
            already know they want the standard flow. Ignores
            `clarification` (bypass is a debug-only path).

    Returns:
        FullPipelineResult with per-branch results. Each trait carries
        its polarity, commitment, and a list of CategoryCallWithEndpoints
        whose generated_specs are ready to hand to stage-4 execution.

    Raises:
        QueryInputError (ValueError): if `query` is empty after
            stripping or either field exceeds its length cap.
        Exception: if Step 0 fails on both attempts (routing is
            mandatory). Every other failure mode is captured on the
            result with a populated *_error field.
    """
    # Validate/normalize at the boundary: strip, enforce non-empty +
    # length cap. Shared rules live in query_input_validation.
    query = clean_query(query)
    clarification = clean_clarification(clarification)

    total_start = time.perf_counter()

    if skip_bypass_steps_0_1:
        # Bypass path — feed the raw query into Step 2 as a single
        # "original" branch. No flow routing, no spin generation.
        # No fan-out needed for a single branch — skip gather's
        # task-scheduling overhead.
        branch_list = [await _run_branch("original", query, query)]
        # One branch → one auxiliary list. Same helper as the
        # multi-branch standard path uses, just trivially scalar.
        auxiliary_per_branch = [_compute_branch_auxiliary(branch_list[0])]
        branch_results = await execute_branches(
            branch_list, auxiliary_per_branch
        )
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
            auxiliary_endpoint_specs_per_branch=auxiliary_per_branch,
            branch_results=branch_results,
            total_elapsed=time.perf_counter() - total_start,
        )

    # Steps 0 and 1 in parallel, but launched as tasks so we can
    # cancel Step 1 the moment Step 0's routing decision proves
    # spins are not needed (no primary flow at all, or budget=1
    # because two non-standard flows are firing). This shaves a
    # full Step-1 wall-clock window off those queries without
    # affecting the cache-warm Step 1 path.
    step0_task = asyncio.create_task(
        _call_with_retry(
            lambda: run_step_0(query, clarification=clarification),
            label="step_0",
        )
    )
    step1_task = asyncio.create_task(
        _call_with_retry(
            lambda: run_step_1(query, clarification=clarification),
            label="step_1",
        )
    )

    try:
        step0_result = await step0_task
    except BaseException:
        # Step 0 failed — Step 1's output cannot be used either way.
        step1_task.cancel()
        try:
            await step1_task
        except BaseException:
            pass
        raise
    step0_response = step0_result[0]

    step1_response: Step1Response | Step1ClarificationResponse | None
    step1_error: str | None
    if _step1_needed(step0_response):
        try:
            step1_result = await step1_task
        except BaseException as exc:
            step1_response = None
            step1_error = repr(exc)
        else:
            step1_response = step1_result[0]
            step1_error = None
    else:
        # Step 1 would not feed any branch — cancel it.
        step1_task.cancel()
        try:
            await step1_task
        except BaseException:
            pass
        step1_response = None
        step1_error = None

    # Non-standard flow dispatch. exact_title still has no orchestrator-
    # owned executor call here (the runner / streaming orchestrator drive
    # it instead); similarity, non_character_franchise,
    # character_franchise, and studio execute inline.
    exact_title_flow_data = step0_response.to_exact_title_flow_data()
    similarity_flow_data = step0_response.to_similarity_flow_data()
    non_character_franchise_flow_data = (
        step0_response.to_non_character_franchise_flow_data()
    )
    character_franchise_flow_data = (
        step0_response.to_character_franchise_flow_data()
    )
    studio_flow_data = step0_response.to_studio_flow_data()
    person_flow_data = step0_response.to_person_flow_data()
    exact_title_flow_executed = exact_title_flow_data is not None
    similarity_flow_executed = similarity_flow_data is not None
    non_character_franchise_flow_executed = (
        non_character_franchise_flow_data is not None
    )
    character_franchise_flow_executed = (
        character_franchise_flow_data is not None
    )
    studio_flow_executed = studio_flow_data is not None
    person_flow_executed = person_flow_data is not None

    similarity_result: SimilarMoviesSearchResult | None = None
    similarity_error: str | None = None
    if similarity_flow_data is not None:
        try:
            similarity_result = await run_similarity_search(similarity_flow_data)
        except Exception as exc:  # noqa: BLE001 — non-standard side flow soft-fail
            logger.error(
                "similarity flow execution failed; continuing standard flow "
                "when present (error=%r)",
                exc,
            )
            similarity_error = repr(exc)

    non_character_franchise_result: NonCharacterFranchiseSearchResult | None = None
    non_character_franchise_error: str | None = None
    if non_character_franchise_flow_data is not None:
        try:
            non_character_franchise_result = (
                await run_non_character_franchise_search(
                    non_character_franchise_flow_data
                )
            )
        except Exception as exc:  # noqa: BLE001 — non-standard side flow soft-fail
            logger.error(
                "non_character_franchise flow execution failed; continuing "
                "standard flow when present (error=%r)",
                exc,
            )
            non_character_franchise_error = repr(exc)

    character_franchise_result: CharacterFranchiseSearchResult | None = None
    character_franchise_error: str | None = None
    if character_franchise_flow_data is not None:
        try:
            character_franchise_result = await run_character_franchise_search(
                character_franchise_flow_data
            )
        except Exception as exc:  # noqa: BLE001 — non-standard side flow soft-fail
            logger.error(
                "character_franchise flow execution failed; continuing "
                "standard flow when present (error=%r)",
                exc,
            )
            character_franchise_error = repr(exc)

    studio_result: StudioSearchResult | None = None
    studio_error: str | None = None
    if studio_flow_data is not None:
        try:
            studio_result = await run_studio_search(studio_flow_data)
        except Exception as exc:  # noqa: BLE001 — non-standard side flow soft-fail
            logger.error(
                "studio flow execution failed; continuing standard flow "
                "when present (error=%r)",
                exc,
            )
            studio_error = repr(exc)

    person_result: PersonSearchResult | None = None
    person_error: str | None = None
    if person_flow_data is not None:
        try:
            person_result = await run_person_search(person_flow_data)
        except Exception as exc:  # noqa: BLE001 — non-standard side flow soft-fail
            logger.error(
                "person flow execution failed; continuing standard flow "
                "when present (error=%r)",
                exc,
            )
            person_error = repr(exc)

    # Standard flow — plan branches per the budget rule and run them
    # in parallel with per-branch error isolation.
    branch_plan = _plan_step2_branches(
        step0_response,
        step1_response,
        query,
        clarification=clarification,
    )
    branches: list[Step2BranchResult] = []
    if branch_plan:
        branches = await asyncio.gather(
            *(_run_branch(kind, q, label) for kind, q, label in branch_plan)
        )

    # Per-branch auxiliary specs — each branch decides independently
    # whether it needs shorts-exclusion / reranker-only fallback. The
    # fallback helper may mutate the branch's own specs in place (to
    # promote a reranker), so order matters: compute aux before
    # executing Stage 4.
    auxiliary_per_branch: list[list[GeneratedEndpointSpec]] = [
        _compute_branch_auxiliary(branch) for branch in branches
    ]
    # Each branch runs its own Stage 4 in parallel — no cross-branch
    # gate, since auxiliary is now per-branch.
    branch_results = await execute_branches(branches, auxiliary_per_branch)
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
        non_character_franchise_flow_executed=non_character_franchise_flow_executed,
        character_franchise_flow_executed=character_franchise_flow_executed,
        studio_flow_executed=studio_flow_executed,
        person_flow_executed=person_flow_executed,
        similarity_result=similarity_result,
        similarity_error=similarity_error,
        non_character_franchise_result=non_character_franchise_result,
        non_character_franchise_error=non_character_franchise_error,
        character_franchise_result=character_franchise_result,
        character_franchise_error=character_franchise_error,
        studio_result=studio_result,
        studio_error=studio_error,
        person_result=person_result,
        person_error=person_error,
        branches=branches,
        auxiliary_endpoint_specs_per_branch=auxiliary_per_branch,
        branch_results=branch_results,
        total_elapsed=time.perf_counter() - total_start,
    )
