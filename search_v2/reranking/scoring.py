# Search V2 — Stage 4 final score composition.
#
# Phase 6 of the orchestrator. Walks the assembled, exclusion-pruned
# pool and emits one ScoreBreakdown per candidate using the formula:
#
#   final_score = dealbreaker_sum
#               + preference_contribution
#               - exclusion_penalties
#
# Dealbreakers aggregate at the concept level: sibling inclusion
# dealbreakers under the same concept contribute one max-scored concept
# signal, not multiple stacked hits. Semantic exclusions likewise
# apply one max-based penalty per concept.

from __future__ import annotations

from collections import defaultdict

from schemas.endpoint_result import EndpointResult
from search_v2.stage_4.types import EndpointOutcome, ScoreBreakdown


P_CAP = 0.9
E_MULT = 2.0

_WEIGHT_REGULAR_PREF = 1.0
_WEIGHT_PRIMARY_PREF = 3.0


def score_pool(
    pool: list[int],
    *,
    inclusion_outcomes: list[EndpointOutcome],
    preference_outcomes: list[EndpointOutcome],
    semantic_exclusion_outcomes: list[EndpointOutcome],
) -> list[ScoreBreakdown]:
    """Compose the final score for every movie in the pool."""
    inc_by_concept = _group_outcomes_by_concept(inclusion_outcomes)
    pref_by_item = [(o, _scores_as_dict(o.result)) for o in preference_outcomes]
    excl_by_concept = _group_outcomes_by_concept(semantic_exclusion_outcomes)

    breakdowns: list[ScoreBreakdown] = []

    for mid in pool:
        per_item: dict[str, float] = {}

        dealbreaker_sum = 0.0
        for grouped in inc_by_concept.values():
            concept_max = 0.0
            for outcome, scores in grouped:
                score = scores.get(mid, 0.0)
                if score != 0.0:
                    per_item[outcome.item.debug_key] = score
                if score > concept_max:
                    concept_max = score
            dealbreaker_sum += concept_max

        weighted_sum = 0.0
        weight_sum = 0.0
        for outcome, scores in pref_by_item:
            weight = (
                _WEIGHT_PRIMARY_PREF
                if outcome.item.is_primary_preference
                else _WEIGHT_REGULAR_PREF
            )
            score = scores.get(mid, 0.0)
            weighted_sum += weight * score
            weight_sum += weight
            if score != 0.0:
                per_item[outcome.item.debug_key] = score

        preference_contribution = (
            P_CAP * (weighted_sum / weight_sum) if weight_sum > 0.0 else 0.0
        )

        exclusion_penalties = 0.0
        for grouped in excl_by_concept.values():
            concept_max = 0.0
            for outcome, scores in grouped:
                score = scores.get(mid, 0.0)
                if score != 0.0:
                    per_item[outcome.item.debug_key] = score
                if score > concept_max:
                    concept_max = score
            exclusion_penalties += E_MULT * concept_max

        final = dealbreaker_sum + preference_contribution - exclusion_penalties
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


def _group_outcomes_by_concept(
    outcomes: list[EndpointOutcome],
) -> dict[str, list[tuple[EndpointOutcome, dict[int, float]]]]:
    grouped: dict[str, list[tuple[EndpointOutcome, dict[int, float]]]] = defaultdict(list)
    for outcome in outcomes:
        grouped[outcome.item.concept_debug_key].append(
            (outcome, _scores_as_dict(outcome.result))
        )
    return grouped


def _scores_as_dict(result: EndpointResult) -> dict[int, float]:
    return {score.movie_id: score.score for score in result.scores}
