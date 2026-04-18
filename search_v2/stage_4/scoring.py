# Search V2 — Stage 4 final score composition.
#
# Phase 6 of the orchestrator. Walks the assembled, exclusion-pruned
# pool and emits one ScoreBreakdown per candidate using the closed-
# form formula from step_4_planning.md §"Final score calculation":
#
#   final_score = dealbreaker_sum
#               + preference_contribution
#               - exclusion_penalties
#
# Where:
#   dealbreaker_sum         — Σ per-candidate score across inclusion-
#                             dealbreaker outcomes. Default-zero rule
#                             gives 0 for candidates not returned by
#                             a given outcome.
#   preference_contribution — P_CAP × weighted average across every
#                             preference plus the two system priors
#                             with their mode-dependent weights.
#                             Denominator 0 → contribution 0.
#   exclusion_penalties     — Σ (E_MULT × match_score) across
#                             semantic-exclusion outcomes. Deterministic
#                             exclusions are not here — they were
#                             already enforced by pool subtraction.

from __future__ import annotations

from schemas.endpoint_result import EndpointResult
from schemas.enums import SystemPrior
from search_v2.stage_4.priors import notability_score, quality_score
from search_v2.stage_4.types import EndpointOutcome, ScoreBreakdown


# Tunable constants called out in the planning doc. Both start at
# proposal-level defaults and are the first dial to turn during
# empirical tuning.
P_CAP = 0.9
E_MULT = 2.0

# Preference weights — regular vs primary.
_WEIGHT_REGULAR_PREF = 1.0
_WEIGHT_PRIMARY_PREF = 3.0

# Prior weights by mode. SUPPRESSED is handled by simply not adding
# an entry to the weighted average (treating it as weight 0 would
# work but adds a zero term that clutters debug).
_PRIOR_WEIGHT_EMPHASIZED = 1.5  # ENHANCED or INVERTED
_PRIOR_WEIGHT_STANDARD = 0.75


def score_pool(
    pool: list[int],
    *,
    inclusion_outcomes: list[EndpointOutcome],
    preference_outcomes: list[EndpointOutcome],
    semantic_exclusion_outcomes: list[EndpointOutcome],
    prior_inputs: dict[int, tuple[float | None, float | None]],
    quality_prior_mode: SystemPrior,
    notability_prior_mode: SystemPrior,
) -> list[ScoreBreakdown]:
    """Compose the final score for every movie in the pool.

    Preserves input order in the returned list so callers get a
    deterministic arrival-order stream for tie-breaking in Phase 7.

    prior_inputs maps movie_id → (reception_score, popularity_score).
    Missing ids are treated as (None, None) — the priors collapse to
    0 in that case (see priors._reception_high / _pop_high).
    """
    # Convert EndpointResult lists into dicts once; subsequent lookups
    # are O(1) per candidate per outcome. Matches the "safe extraction
    # with default zero" rule: the dict.get(mid, 0.0) pattern never
    # assumes presence, which is critical for the default-zero rule.
    inc_by_item = [
        (o, _scores_as_dict(o.result)) for o in inclusion_outcomes
    ]
    pref_by_item = [
        (o, _scores_as_dict(o.result)) for o in preference_outcomes
    ]
    excl_by_item = [
        (o, _scores_as_dict(o.result))
        for o in semantic_exclusion_outcomes
    ]

    # Prior-side contribution: weight is fixed across the pool, only
    # score varies per candidate. Compute the weights once.
    quality_prior_weight = _prior_weight(quality_prior_mode)
    notability_prior_weight = _prior_weight(notability_prior_mode)

    breakdowns: list[ScoreBreakdown] = []

    for mid in pool:
        per_item: dict[str, float] = {}

        # --- dealbreaker_sum -------------------------------------------------
        dealbreaker_sum = 0.0
        for outcome, scores in inc_by_item:
            s = scores.get(mid, 0.0)
            if s != 0.0:
                per_item[outcome.item.debug_key] = s
            dealbreaker_sum += s

        # --- preference_contribution (weighted average × P_CAP) -------------
        weighted_sum = 0.0
        weight_sum = 0.0

        for outcome, scores in pref_by_item:
            weight = (
                _WEIGHT_PRIMARY_PREF
                if outcome.item.is_primary_preference
                else _WEIGHT_REGULAR_PREF
            )
            s = scores.get(mid, 0.0)
            weighted_sum += weight * s
            weight_sum += weight
            if s != 0.0:
                per_item[outcome.item.debug_key] = s

        reception_score, popularity_score = prior_inputs.get(
            mid, (None, None)
        )

        if quality_prior_weight > 0.0:
            q = quality_score(
                reception_score, popularity_score, quality_prior_mode
            )
            weighted_sum += quality_prior_weight * q
            weight_sum += quality_prior_weight
            if q != 0.0:
                per_item["prior:quality"] = q

        if notability_prior_weight > 0.0:
            n = notability_score(popularity_score, notability_prior_mode)
            weighted_sum += notability_prior_weight * n
            weight_sum += notability_prior_weight
            if n != 0.0:
                per_item["prior:notability"] = n

        preference_contribution = (
            P_CAP * (weighted_sum / weight_sum) if weight_sum > 0.0 else 0.0
        )

        # --- exclusion_penalties (semantic exclusions only) -----------------
        exclusion_penalties = 0.0
        for outcome, scores in excl_by_item:
            match = scores.get(mid, 0.0)
            if match != 0.0:
                per_item[outcome.item.debug_key] = match
            exclusion_penalties += E_MULT * match

        final = (
            dealbreaker_sum
            + preference_contribution
            - exclusion_penalties
        )

        breakdowns.append(
            ScoreBreakdown(
                movie_id=mid,
                dealbreaker_sum=dealbreaker_sum,
                preference_contribution=preference_contribution,
                exclusion_penalties=exclusion_penalties,
                final_score=final,
                per_item_scores=per_item,
            )
        )

    return breakdowns


def _scores_as_dict(result: EndpointResult) -> dict[int, float]:
    # Safe extraction: a missing id means 0.0 on lookup, per the
    # default-zero rule. Empty EndpointResult (soft-fail) simply
    # produces an empty dict.
    return {s.movie_id: s.score for s in result.scores}


def _prior_weight(mode: SystemPrior) -> float:
    if mode in (SystemPrior.ENHANCED, SystemPrior.INVERTED):
        return _PRIOR_WEIGHT_EMPHASIZED
    if mode == SystemPrior.STANDARD:
        return _PRIOR_WEIGHT_STANDARD
    # SUPPRESSED — drop out of the weighted average entirely.
    return 0.0
