# Stage-3 orchestrator.
#
# Top of the step-3 stack. Takes the Step-2 response, fans out one
# run_handler call per non-NO_FIT coverage_evidence atom in parallel
# with run_implicit_expectations, consolidates the four-bucket
# HandlerResults, decides a candidate-pool path, runs deferred
# preferences against the pool, and produces a final ranked list with
# per-movie score breakdowns.
#
# Pool-path policy (no-inclusion fallback hierarchy):
#   1. inclusion_aggregated non-empty                  → use it.
#   2. else preference_specs non-empty                 → run all
#      preferences against the FULL CORPUS, sum raw scores additively
#      as the pool. Preferences are then "consumed" (no preference
#      contribution stage on top — would double-count).
#   3. else exclusion_set non-empty                    → top-2K seed
#      ordered by popularity * reception (a "well-known and
#      well-received" pool that exclusion can prune from).
#   4. else                                            → empty result.
#
# Exclusion is applied after pool selection in every path. Implicit
# priors apply to every non-empty pool, including the fallbacks.
#
# Soft-fail throughout: a single handler / implicit / preference
# failure must never tank the whole orchestration. Failures surface
# as empty HandlerResults / None implicit / zero-score preferences.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Literal

from qdrant_client import AsyncQdrantClient

from db.postgres import (
    fetch_quality_popularity_seed,
    fetch_quality_popularity_signals,
)
from db.reranking import normalize_reception
from schemas.endpoint_parameters import EndpointParameters
from schemas.endpoint_result import EndpointResult
from schemas.enums import FitQuality
from schemas.implicit_expectations import ImplicitExpectationsResult
from schemas.step_2 import Step2Response
from search_v2.implicit_expectations import run_implicit_expectations
from search_v2.endpoint_fetching.category_handlers.handler import run_handler
from search_v2.endpoint_fetching.category_handlers.handler_result import HandlerResult
from search_v2.endpoint_fetching.endpoint_executors import (
    build_endpoint_coroutine,
    route_for_wrapper,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

# Per-call timeout for deferred preference executions and the
# preferences-as-candidates fallback. Mirrors handler.py's value so
# the whole stage-3 stack uses one timeout budget.
TIMEOUT_SECONDS = 20.0

# Cap on the combined preference contribution to a movie's final
# score. Distributed equally across firing preferences.
PREFERENCE_CAP = 0.49

# Cap on the combined implicit-prior contribution. When only one
# axis is active it claims the full cap (0.25). When both are active
# the cap is split 80/20 popularity/reception — popularity is the
# stronger implicit signal for "well-known and well-liked" intent,
# so notability (popularity) gets 0.20 and quality (reception) gets 0.05.
IMPLICIT_PRIOR_CAP = 0.25
IMPLICIT_POPULARITY_SHARE_BOTH_ACTIVE = 0.8
IMPLICIT_RECEPTION_SHARE_BOTH_ACTIVE = 0.2

# Size of the popularity*reception seed pool used when only
# exclusion fired (no inclusion candidates and no preferences to
# treat as candidates).
FALLBACK_SEED_LIMIT = 2000

# Default number of movies returned at the top of the orchestrator.
DEFAULT_TOP_K = 100

# Tags that surface in Stage3Result.used_fallback so callers can tell
# which pool-path the orchestrator took.
FallbackPath = Literal[
    "none",
    "preferences_as_candidates",
    "popularity_quality_seed",
    "empty",
]


# ---------------------------------------------------------------------------
# Result shapes
# ---------------------------------------------------------------------------


@dataclass
class ScoreBreakdown:
    """Per-movie score components. Sum equals final_score."""

    movie_id: int
    inclusion_sum: float
    downrank_sum: float
    preference_contribution: float  # ≤ PREFERENCE_CAP
    implicit_prior_contribution: float  # ≤ IMPLICIT_PRIOR_CAP
    final_score: float


@dataclass
class Stage3Result:
    """Final output of the stage-3 orchestrator."""

    movie_ids: list[int]
    breakdowns: dict[int, ScoreBreakdown]
    handler_results: list[HandlerResult] = field(default_factory=list)
    implicit_expectations: ImplicitExpectationsResult | None = None
    used_fallback: FallbackPath = "none"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_stage_3(
    raw_query: str,
    step_2_response: Step2Response,
    *,
    qdrant_client: AsyncQdrantClient,
    top_k: int = DEFAULT_TOP_K,
) -> Stage3Result:
    """Run the full stage-3 pipeline for one query.

    Returns a Stage3Result with the top_k ranked movie_ids and a
    per-movie ScoreBreakdown. Never raises: every failure mode
    inside the orchestrator (handler errors, implicit-expectations
    failure, preference timeouts, empty pool) surfaces as either an
    empty result or a degraded one with the relevant signal absent.
    """
    # --- Phase 0: fan-out handlers + implicit_expectations in parallel ---
    handler_results, implicit_result = await _fan_out(
        raw_query=raw_query,
        step_2_response=step_2_response,
        qdrant_client=qdrant_client,
    )

    # --- Phase 1: consolidate the four buckets across handlers ----------
    (
        inclusion_aggregated,
        downrank_aggregated,
        exclusion_set,
        preference_specs_pool,
    ) = _consolidate(handler_results)

    # --- Phase 2: choose pool path --------------------------------------
    pool_inclusion_scores, deferred_preferences, used_fallback = (
        await _select_pool(
            inclusion_aggregated=inclusion_aggregated,
            preference_specs_pool=preference_specs_pool,
            exclusion_set=exclusion_set,
            qdrant_client=qdrant_client,
        )
    )

    if used_fallback == "empty":
        return Stage3Result(
            movie_ids=[],
            breakdowns={},
            handler_results=handler_results,
            implicit_expectations=implicit_result,
            used_fallback="empty",
        )

    # --- Phase 3: apply exclusion subtraction ---------------------------
    for mid in exclusion_set:
        pool_inclusion_scores.pop(mid, None)
    if not pool_inclusion_scores:
        return Stage3Result(
            movie_ids=[],
            breakdowns={},
            handler_results=handler_results,
            implicit_expectations=implicit_result,
            used_fallback=used_fallback,
        )

    # --- Phase 4: run deferred preferences against the pool -------------
    pool_set = set(pool_inclusion_scores)
    deferred_preference_outcomes = await _run_deferred_preferences(
        deferred_preferences,
        pool=pool_set,
        qdrant_client=qdrant_client,
    )

    # --- Phase 5: fetch implicit-prior signals if either axis is active --
    quality_active, notability_active = _resolve_active_priors(implicit_result)
    prior_signals: dict[int, tuple[float | None, float | None]] = {}
    if quality_active or notability_active:
        prior_signals = await fetch_quality_popularity_signals(
            list(pool_inclusion_scores)
        )

    # --- Phase 6: compose final per-movie scores ------------------------
    breakdowns = _score_pool(
        pool_inclusion_scores=pool_inclusion_scores,
        downrank_aggregated=downrank_aggregated,
        deferred_preference_outcomes=deferred_preference_outcomes,
        quality_active=quality_active,
        notability_active=notability_active,
        prior_signals=prior_signals,
    )

    # Stable sort by final_score desc; tie-breakers fall back to
    # arrival order in pool_inclusion_scores (insertion order).
    breakdowns.sort(key=lambda b: b.final_score, reverse=True)
    top = breakdowns[:top_k]

    return Stage3Result(
        movie_ids=[b.movie_id for b in top],
        breakdowns={b.movie_id: b for b in top},
        handler_results=handler_results,
        implicit_expectations=implicit_result,
        used_fallback=used_fallback,
    )


# ---------------------------------------------------------------------------
# Phase 0 — fan-out
# ---------------------------------------------------------------------------


async def _fan_out(
    *,
    raw_query: str,
    step_2_response: Step2Response,
    qdrant_client: AsyncQdrantClient,
) -> tuple[list[HandlerResult], ImplicitExpectationsResult | None]:
    """Fire one run_handler per non-NO_FIT atom + run_implicit_expectations.

    All calls run in a single asyncio.gather with return_exceptions=
    True. Handler failures land as empty HandlerResult; implicit
    failure lands as None so the orchestrator treats both priors as
    inactive.
    """
    requirements = step_2_response.requirements
    intent = step_2_response.overall_query_intention_exploration

    # Collect (parent_fragment, target_entry) pairs for every
    # category-bound atom worth dispatching. NO_FIT atoms are filtered
    # here too — handler.py defends against them, but skipping
    # upstream avoids the wasted fan-out slot.
    handler_specs = []
    for parent in requirements:
        siblings = [r for r in requirements if r is not parent]
        for entry in parent.coverage_evidence:
            if entry.fit_quality == FitQuality.NO_FIT:
                continue
            handler_specs.append((parent, siblings, entry))

    handler_coros = [
        run_handler(
            category=entry.category_name,
            target_entry=entry,
            raw_query=raw_query,
            overall_query_intention_exploration=intent,
            parent_fragment=parent,
            sibling_fragments=siblings,
            qdrant_client=qdrant_client,
        )
        for (parent, siblings, entry) in handler_specs
    ]

    # run_implicit_expectations returns a 4-tuple
    # (response, input_tokens, output_tokens, elapsed). We only need
    # the response object here — wrap the call so gather sees a
    # single awaitable.
    async def _run_implicit():
        try:
            response, _in, _out, _elapsed = await run_implicit_expectations(
                raw_query, step_2_response
            )
            return response
        except Exception as exc:  # noqa: BLE001 — soft-fail by design
            logger.warning(
                "implicit_expectations failed; both priors inactive (%r)",
                exc,
            )
            return None

    # Position the implicit slot last so handler_results stays
    # aligned with handler_specs after we strip exceptions.
    outcomes = await asyncio.gather(
        *handler_coros, _run_implicit(), return_exceptions=True
    )

    *handler_outcomes, implicit_outcome = outcomes

    handler_results: list[HandlerResult] = []
    for outcome, (parent, _siblings, entry) in zip(
        handler_outcomes, handler_specs
    ):
        if isinstance(outcome, BaseException):
            logger.warning(
                "handler raised; substituting empty HandlerResult "
                "(category=%s, fragment=%r, error=%r)",
                entry.category_name.name,
                parent.query_text,
                outcome,
            )
            # Preserve the category on the soft-fail substitute so
            # downstream introspection (notebook display, debug
            # tooling) can still attribute the empty result to the
            # right category.
            handler_results.append(HandlerResult(category=entry.category_name))
        else:
            handler_results.append(outcome)

    if isinstance(implicit_outcome, BaseException):
        # _run_implicit already swallows exceptions internally, so
        # reaching here would be a programmer error in the wrapper.
        logger.warning(
            "implicit wrapper escaped (%r); both priors inactive",
            implicit_outcome,
        )
        implicit_result = None
    else:
        implicit_result = implicit_outcome

    return handler_results, implicit_result


# ---------------------------------------------------------------------------
# Phase 1 — consolidate
# ---------------------------------------------------------------------------


def _consolidate(
    handler_results: list[HandlerResult],
) -> tuple[
    dict[int, float],
    dict[int, float],
    set[int],
    list[EndpointParameters],
]:
    """Merge the four buckets across handlers.

    inclusion / downrank are summed additively per tmdb_id.
    exclusion is unioned. preference_specs are flattened in order.
    """
    inclusion: dict[int, float] = {}
    downrank: dict[int, float] = {}
    exclusion: set[int] = set()
    preferences: list[EndpointParameters] = []

    for hr in handler_results:
        for mid, score in hr.inclusion_candidates.items():
            inclusion[mid] = inclusion.get(mid, 0.0) + score
        for mid, score in hr.downrank_candidates.items():
            downrank[mid] = downrank.get(mid, 0.0) + score
        exclusion.update(hr.exclusion_ids)
        preferences.extend(hr.preference_specs)

    return inclusion, downrank, exclusion, preferences


# ---------------------------------------------------------------------------
# Phase 2 — pool-path selection (with two fallback paths)
# ---------------------------------------------------------------------------


async def _select_pool(
    *,
    inclusion_aggregated: dict[int, float],
    preference_specs_pool: list[EndpointParameters],
    exclusion_set: set[int],
    qdrant_client: AsyncQdrantClient,
) -> tuple[dict[int, float], list[EndpointParameters], FallbackPath]:
    """Pick which pool to use and which preferences (if any) to defer.

    Returns (pool_inclusion_scores, deferred_preferences, fallback_tag).
    """
    if inclusion_aggregated:
        return inclusion_aggregated, preference_specs_pool, "none"

    if preference_specs_pool:
        # No inclusion at all — promote preferences to candidate
        # generators by running them against the full corpus and
        # summing raw scores additively. They are "consumed" here:
        # deferred_preferences is empty so the rerank stage doesn't
        # double-count their contribution.
        pool = await _run_preferences_as_candidates(
            preference_specs_pool, qdrant_client=qdrant_client
        )
        return pool, [], "preferences_as_candidates"

    if exclusion_set:
        # Pure-exclusion query (e.g. "PG-13 max" with nothing positive
        # to anchor on). Seed with a quality+popularity pool that
        # exclusion will prune, then let implicit priors reorder.
        seed_ids = await fetch_quality_popularity_seed(
            limit=FALLBACK_SEED_LIMIT
        )
        return {mid: 0.0 for mid in seed_ids}, [], "popularity_quality_seed"

    return {}, [], "empty"


async def _run_preferences_as_candidates(
    preference_specs: list[EndpointParameters],
    *,
    qdrant_client: AsyncQdrantClient,
) -> dict[int, float]:
    """Run all preferences against the full corpus, sum scores additively.

    Used by the no-inclusion fallback: when handlers emitted only
    preferences, we still need a candidate pool. Each preference
    runs with restrict_to_movie_ids=None (corpus-wide). Failed
    preferences contribute nothing — soft-fail per the same policy
    as the deferred-preference path.
    """
    coros = [
        asyncio.wait_for(
            build_endpoint_coroutine(
                route_for_wrapper(spec),
                spec,
                qdrant_client=qdrant_client,
                restrict_to_movie_ids=None,
            ),
            timeout=TIMEOUT_SECONDS,
        )
        for spec in preference_specs
    ]
    outcomes = await asyncio.gather(*coros, return_exceptions=True)

    pool: dict[int, float] = {}
    for spec, outcome in zip(preference_specs, outcomes):
        if isinstance(outcome, BaseException):
            logger.warning(
                "preference-as-candidate execution failed; skipping "
                "(wrapper=%s, error=%r)",
                type(spec).__name__,
                outcome,
            )
            continue
        for sc in outcome.scores:
            pool[sc.movie_id] = pool.get(sc.movie_id, 0.0) + sc.score
    return pool


# ---------------------------------------------------------------------------
# Phase 4 — deferred preferences against the established pool
# ---------------------------------------------------------------------------


async def _run_deferred_preferences(
    preference_specs: list[EndpointParameters],
    *,
    pool: set[int],
    qdrant_client: AsyncQdrantClient,
) -> list[dict[int, float]]:
    """Run each preference against the pool. Returns one score-dict per spec.

    Failures collapse to an empty score-dict so the per-preference
    weight slot still counts toward PREFERENCE_CAP / N_pref —
    failing a single preference shouldn't redistribute its share to
    its siblings (that would silently amplify the surviving
    preferences). An empty dict naturally contributes 0 to every
    candidate.
    """
    if not preference_specs:
        return []

    coros = [
        asyncio.wait_for(
            build_endpoint_coroutine(
                route_for_wrapper(spec),
                spec,
                qdrant_client=qdrant_client,
                restrict_to_movie_ids=pool,
            ),
            timeout=TIMEOUT_SECONDS,
        )
        for spec in preference_specs
    ]
    outcomes = await asyncio.gather(*coros, return_exceptions=True)

    results: list[dict[int, float]] = []
    for spec, outcome in zip(preference_specs, outcomes):
        if isinstance(outcome, BaseException):
            logger.warning(
                "deferred preference execution failed; contributes 0 "
                "(wrapper=%s, error=%r)",
                type(spec).__name__,
                outcome,
            )
            results.append({})
            continue
        results.append(_scores_to_dict(outcome))
    return results


def _scores_to_dict(result: EndpointResult) -> dict[int, float]:
    return {sc.movie_id: sc.score for sc in result.scores}


# ---------------------------------------------------------------------------
# Phase 5 helper — resolve which implicit priors are active
# ---------------------------------------------------------------------------


def _resolve_active_priors(
    implicit_result: ImplicitExpectationsResult | None,
) -> tuple[bool, bool]:
    if implicit_result is None:
        return False, False
    return (
        implicit_result.should_apply_quality_prior,
        implicit_result.should_apply_notability_prior,
    )


# ---------------------------------------------------------------------------
# Phase 6 — final score composition
# ---------------------------------------------------------------------------


def _score_pool(
    *,
    pool_inclusion_scores: dict[int, float],
    downrank_aggregated: dict[int, float],
    deferred_preference_outcomes: list[dict[int, float]],
    quality_active: bool,
    notability_active: bool,
    prior_signals: dict[int, tuple[float | None, float | None]],
) -> list[ScoreBreakdown]:
    """Compose a ScoreBreakdown for every movie in the pool.

    Score formula (per movie):
        final = inclusion_sum
              − downrank_sum
              + preference_contribution     (capped at PREFERENCE_CAP)
              + implicit_prior_contribution (capped at IMPLICIT_PRIOR_CAP)
    """
    n_pref = len(deferred_preference_outcomes)
    per_pref_weight = PREFERENCE_CAP / n_pref if n_pref else 0.0

    # Resolve per-axis weights. With both axes active we split the
    # cap 80/20 toward popularity, since notability is the dominant
    # implicit cue for default "well-known and well-liked" ranking.
    # With a single active axis it claims the full cap on its own.
    if quality_active and notability_active:
        quality_weight = IMPLICIT_PRIOR_CAP * IMPLICIT_RECEPTION_SHARE_BOTH_ACTIVE
        notability_weight = IMPLICIT_PRIOR_CAP * IMPLICIT_POPULARITY_SHARE_BOTH_ACTIVE
    else:
        quality_weight = IMPLICIT_PRIOR_CAP if quality_active else 0.0
        notability_weight = IMPLICIT_PRIOR_CAP if notability_active else 0.0
    any_prior_active = quality_active or notability_active

    breakdowns: list[ScoreBreakdown] = []
    for mid, inc_score in pool_inclusion_scores.items():
        dn = downrank_aggregated.get(mid, 0.0)

        pref = 0.0
        if n_pref:
            pref = sum(
                per_pref_weight * scores.get(mid, 0.0)
                for scores in deferred_preference_outcomes
            )

        prior = 0.0
        if any_prior_active:
            popularity, reception = prior_signals.get(mid, (None, None))
            if quality_active:
                # normalize_reception treats None as 0.5 neutral so
                # missing reception data lands at the midpoint rather
                # than the floor.
                prior += quality_weight * normalize_reception(reception)
            if notability_active:
                # popularity_score is already sigmoid-normalized to
                # [0, 1]; missing → 0.0 (no notability signal).
                prior += notability_weight * (popularity if popularity is not None else 0.0)

        final = inc_score - dn + pref + prior
        breakdowns.append(
            ScoreBreakdown(
                movie_id=mid,
                inclusion_sum=inc_score,
                downrank_sum=dn,
                preference_contribution=pref,
                implicit_prior_contribution=prior,
                final_score=final,
            )
        )
    return breakdowns
