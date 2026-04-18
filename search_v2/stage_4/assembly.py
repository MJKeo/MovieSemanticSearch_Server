# Search V2 — Stage 4 pool assembly and exclusion subtraction.
#
# Phase 3 (union candidate sets) and Phase 4 (subtract deterministic
# exclusions) of the orchestrator. The output is an ordered list of
# candidate movie_ids that flows into pool-dependent scoring.
#
# Order matters: ties in the final score fall back to arrival order
# per step_4_planning.md §"Edge cases for scoring — Ties". Using a
# list (not a set) as the pool ensures Python's stable sort in Phase
# 7 preserves that arrival order for tied candidates.

from __future__ import annotations

from schemas.enums import EndpointRoute
from search_v2.stage_4.types import EndpointOutcome


def assemble_pool(
    candidate_outcomes: list[EndpointOutcome],
    *,
    browse_seed_ids: list[int] | None = None,
) -> list[int]:
    """Union the ID sets from candidate-generating outcomes.

    Candidates arrive in a stable order — the order the caller passes
    the outcomes, and within each outcome the order of
    `EndpointResult.scores`. That ordering becomes the tiebreaker for
    equal final scores.

    browse_seed_ids, when provided, is the sole source of candidates
    (BROWSE flow). In every other flow it is None.
    """
    seen: set[int] = set()
    ordered: list[int] = []

    if browse_seed_ids is not None:
        for mid in browse_seed_ids:
            if mid not in seen:
                seen.add(mid)
                ordered.append(mid)
        return ordered

    for outcome in candidate_outcomes:
        for scored in outcome.result.scores:
            mid = scored.movie_id
            if mid not in seen:
                seen.add(mid)
                ordered.append(mid)
    return ordered


def apply_deterministic_exclusions(
    pool: list[int],
    exclusion_outcomes: list[EndpointOutcome],
) -> list[int]:
    """Subtract every deterministic exclusion outcome's matched ids.

    Only non-semantic exclusions hard-filter. Semantic exclusions are
    filtered out here by the caller (they use the match-then-penalize
    path in scoring) — this function trusts its inputs: pass only the
    deterministic exclusion outcomes.

    A soft-failed exclusion (timeout / error) has an empty result
    set, so it subtracts nothing. That is the correct behavior — we
    should not hard-remove candidates based on a failed signal.
    """
    if not exclusion_outcomes:
        return pool

    to_remove: set[int] = set()
    for outcome in exclusion_outcomes:
        # Defensive: never subtract from a semantic exclusion even if
        # one slips in. Callers filter upstream but the guard keeps
        # this helper safe to call with a heterogeneous list.
        if outcome.item.endpoint == EndpointRoute.SEMANTIC:
            continue
        for scored in outcome.result.scores:
            to_remove.add(scored.movie_id)

    if not to_remove:
        return pool
    return [mid for mid in pool if mid not in to_remove]
