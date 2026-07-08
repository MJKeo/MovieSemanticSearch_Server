# Search V2 — Implicit-prior post-reranking.
#
# Stage 4 owns base relevance scoring. This module applies a single-axis
# post-score boost (popularity primary, quality fallback) on top of the ranked
# candidate list, per ADR-087. It lives in its own module (rather than in the
# orchestrator) so `stage_4_execution._run_branch` can apply it INSIDE the
# `query_search.scoring` span — the boost is part of turning the scored pool
# into the final ranked list, and nesting the span there keeps the branch trace
# collapsible into one scoring group.
#
# Import discipline: the two dataclasses this touches (`Step2BranchResult`,
# `BranchRankedResults`) are referenced only in annotations, so they are imported
# under TYPE_CHECKING to avoid a runtime cycle (stage_4_execution imports this
# module; full_pipeline_orchestrator defines Step2BranchResult and imports
# stage_4_execution). The runtime accesses (`branch.implicit_expectations`,
# `result.ranked`, …) need no imported type.

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from opentelemetry import trace

from db.postgres import fetch_quality_popularity_signals
from schemas.enums import PopularityMode, ReceptionMode
from schemas.implicit_expectations import ImplicitExpectationsResult
from search_v2.endpoint_fetching.metadata_query_execution import (
    score_popularity_prior,
    score_reception_prior,
)

from observability.names import (
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_BOOST_AXIS,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_INVERSE_APPLIED,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_NOOP_REASON,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_ACTIVE,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_CAP,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_DIRECTION,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_STRENGTH,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_ACTIVE,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_CAP,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_DIRECTION,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_STRENGTH,
    QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_SIGNAL_MISSING_COUNT,
)

if TYPE_CHECKING:
    from search_v2.full_pipeline_orchestrator import Step2BranchResult
    from search_v2.stage_4_execution import BranchRankedResults

# Per-module tracer. A no-op ProxyTracer when `setup_tracing` hasn't run
# (offline ingestion/eval imports), so the manual spans below are cheap no-ops
# there. The application span nests under whatever span is current — in the live
# pipeline that is `query_search.scoring`, since `_run_branch` calls this inside
# the scoring span.
tracer = trace.get_tracer(__name__)


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


# --- Implicit-prior telemetry value enums (rule E: closed value sets live with
# their owner, not in observability/names.py) ---


class BoostAxis(str, Enum):
    """Which single axis the implicit-prior rerank actually boosted on.

    Per ADR-087 the rerank is single-axis (popularity primary, quality
    fallback), so at most one of these fires. `none` = the prior no-oped
    for one of the PriorNoopReason causes.
    """

    POPULARITY = "popularity"
    QUALITY = "quality"
    NONE = "none"


class PriorNoopReason(str, Enum):
    """Why the implicit-prior rerank produced no boost (boost_axis=none).

    Disambiguates the four distinct no-op causes; the popularity_active /
    quality_active flags alone can't, because the first three return from
    the hard gate before the caps / active flags are ever computed.
    """

    POLICY_UNAVAILABLE = "policy_unavailable"  # generation soft-failed / no branch
    BRANCH_ERROR = "branch_error"              # Stage 4 already degraded the branch
    EMPTY_POOL = "empty_pool"                  # no candidates to rerank
    BOTH_AXES_OFF = "both_axes_off"            # policy proposed nothing that clears the gates


def _implicit_prior_gate_skip_reason(
    branch: "Step2BranchResult | None",
    result: "BranchRankedResults",
) -> PriorNoopReason | None:
    """Return the no-op reason if the prior can't run at all, else None.

    Mirrors the original hard-gate condition but names WHICH clause tripped,
    in the same short-circuit order (absent/failed policy, then branch error,
    then empty pool), so a boost_axis=none span is self-explanatory.
    """
    if branch is None or branch.implicit_expectations is None:
        return PriorNoopReason.POLICY_UNAVAILABLE
    if result.branch_error is not None:
        return PriorNoopReason.BRANCH_ERROR
    if not result.ranked:
        return PriorNoopReason.EMPTY_POOL
    return None


def _stamp_implicit_prior_noop(
    span: trace.Span,
    reason: PriorNoopReason,
) -> None:
    """Record the no-op verdict on the application span (boost_axis=none)."""
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_BOOST_AXIS, BoostAxis.NONE.value
    )
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_NOOP_REASON, reason.value
    )
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_INVERSE_APPLIED, False
    )


def _record_prior_policy_and_selection(
    span: trace.Span,
    policy: ImplicitExpectationsResult,
    *,
    popularity_cap: float,
    quality_cap: float,
    popularity_active: bool,
    quality_active: bool,
) -> None:
    """Record the policy OUTPUT beside the code's axis-selection on the span.

    Both priors' direction/strength (what the LLM proposed) sit next to the
    resolved caps and active flags (how the code gated them), so a reader sees
    proposal vs. application in one place.
    """
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_DIRECTION,
        policy.popularity_prior.direction,
    )
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_STRENGTH,
        policy.popularity_prior.strength,
    )
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_DIRECTION,
        policy.quality_prior.direction,
    )
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_STRENGTH,
        policy.quality_prior.strength,
    )
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_CAP, popularity_cap
    )
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_CAP, quality_cap
    )
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_POPULARITY_ACTIVE, popularity_active
    )
    span.set_attribute(
        QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_QUALITY_ACTIVE, quality_active
    )


async def apply_implicit_prior_rerank_for_branch(
    branch: "Step2BranchResult | None",
    result: "BranchRankedResults",
) -> "BranchRankedResults":
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

    Called by `stage_4_execution._run_branch` inside the `query_search.scoring`
    span, so the `query_search.implicit_prior_rerank` application span nests
    under scoring.
    """
    # Application span (the SECOND location, distinct from generation):
    # brackets gate -> single-axis selection -> Postgres signal fetch (which
    # nests under it) -> per-movie boost -> resort. Started before the gate so a
    # skipped branch still yields a legible span (boost_axis=none + reason).
    with tracer.start_as_current_span(QUERY_SEARCH_IMPLICIT_PRIOR_RERANK) as span:
        # --- Hard gate: the prior cannot run at all. Name why, then skip. ---
        gate_reason = _implicit_prior_gate_skip_reason(branch, result)
        if gate_reason is not None:
            _stamp_implicit_prior_noop(span, gate_reason)
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

        # Record the policy output beside the selection so "what the LLM
        # proposed" and "which axis fired" read off one span.
        _record_prior_policy_and_selection(
            span,
            policy,
            popularity_cap=popularity_cap,
            quality_cap=quality_cap,
            popularity_active=popularity_active,
            quality_active=quality_active,
        )

        # --- Soft no-op: policy proposed nothing that clears the gates. ---
        if not popularity_active and not quality_active:
            _stamp_implicit_prior_noop(span, PriorNoopReason.BOTH_AXES_OFF)
            return result

        # --- Exactly one axis fires (ADR-087 single-axis selection). ---
        if popularity_active:
            fired_axis = BoostAxis.POPULARITY
            fired_direction = policy.popularity_prior.direction
        else:
            fired_axis = BoostAxis.QUALITY
            fired_direction = policy.quality_prior.direction
        span.set_attribute(
            QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_BOOST_AXIS, fired_axis.value
        )
        span.set_attribute(
            QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_INVERSE_APPLIED,
            fired_direction == "inverse",
        )

        try:
            movie_ids = [movie_id for movie_id, _ in result.ranked]
            signals = await fetch_quality_popularity_signals(movie_ids)

            # Count candidates whose FIRED-axis signal was NULL (-> 0 boost):
            # the data-coverage risk this axis carries.
            signal_missing_count = 0
            reranked: list[tuple[int, float]] = []
            for movie_id, base_score in result.ranked:
                popularity_raw, reception_raw = signals.get(movie_id, (None, None))
                if popularity_active:
                    axis_raw = popularity_raw
                    boost = popularity_cap * _popularity_signal(
                        popularity_raw,
                        direction=policy.popularity_prior.direction,
                    )
                else:
                    axis_raw = reception_raw
                    boost = quality_cap * _quality_signal(
                        reception_raw,
                        direction=policy.quality_prior.direction,
                    )
                if axis_raw is None:
                    signal_missing_count += 1
                breakdown = result.score_breakdowns.get(movie_id)
                prior_base = (
                    breakdown.positive_total
                    if breakdown is not None and breakdown.positive_total > 0.0
                    else 1.0
                )
                reranked.append((movie_id, base_score + (prior_base * boost)))
                if breakdown is not None:
                    breakdown.implicit_prior_boost = boost
        except Exception as exc:  # noqa: BLE001 — annotate then re-raise
            # The signal-fetch Postgres call is the only thing that can throw
            # here; the failure still propagates to the branch soft-fail exactly
            # as before. Add a named event so it's queryable; the context
            # manager records the exception + ERROR status on the way out.
            span.add_event(
                "implicit_prior_apply_failed",
                {"error.type": type(exc).__name__},
            )
            raise

        span.set_attribute(
            QUERY_SEARCH_IMPLICIT_PRIOR_RERANK_SIGNAL_MISSING_COUNT,
            signal_missing_count,
        )
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
