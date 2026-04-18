# Search V2 — Stage 4 flow detection and item tagging.
#
# Phase 0 of the orchestrator. Runs synchronously the moment step 2
# returns: inspects the QueryUnderstandingResponse, picks one of four
# flows (STANDARD / D2 / P2 / BROWSE), and tags every step-2 item
# with the role, endpoint, and a `generates_candidates` flag that
# downstream dispatch uses to decide whether the item's execution
# should fire pool-independent or wait for the assembly barrier.
#
# Flow selection rules mirror step_4_planning.md §"Execution ordering"
# Step A:
#   - any inclusion dealbreaker with route != SEMANTIC → STANDARD
#   - else any inclusion dealbreaker (all semantic)    → D2
#   - else any preference                              → P2
#   - else                                             → BROWSE
#
# Candidate-generator tagging per flow:
#   STANDARD — non-semantic inclusion dealbreakers yes, rest no
#   D2       — every inclusion dealbreaker yes, rest no
#   P2       — every preference yes, rest no
#   BROWSE   — nothing generates via endpoints (seed comes from
#              movie_card ordering)
#
# Deterministic exclusions are never "candidate generators" — they
# run pool-independent only so their id set is ready when the
# assembly barrier releases, and the orchestrator subtracts them
# explicitly.

from __future__ import annotations

from schemas.enums import DealbreakDirection, EndpointRoute
from schemas.query_understanding import (
    Dealbreaker,
    Preference,
    QueryUnderstandingResponse,
)
from search_v2.stage_4.types import Stage4Flow, TaggedItem


def detect_flow(qu: QueryUnderstandingResponse) -> Stage4Flow:
    """Select the stage-4 flow scenario for this branch."""
    inclusion_dbs = [
        d for d in qu.dealbreakers
        if d.direction == DealbreakDirection.INCLUSION
    ]
    non_semantic_inclusion = [
        d for d in inclusion_dbs if d.route != EndpointRoute.SEMANTIC
    ]

    if non_semantic_inclusion:
        return Stage4Flow.STANDARD
    if inclusion_dbs:
        # At least one inclusion dealbreaker exists and none are
        # non-semantic, so every inclusion dealbreaker is semantic.
        return Stage4Flow.D2
    if qu.preferences:
        return Stage4Flow.P2
    return Stage4Flow.BROWSE


def tag_items(
    qu: QueryUnderstandingResponse, flow: Stage4Flow
) -> list[TaggedItem]:
    """Build the full list of TaggedItems for this branch.

    Order: inclusion dealbreakers (preserving source index), then
    exclusion dealbreakers (preserving source index), then preferences.
    The orchestrator iterates this list for both dispatch and scoring
    so the order is also the stable "arrival order" used as tiebreaker.
    """
    tagged: list[TaggedItem] = []

    # Dealbreakers — one pass, routed by direction to the appropriate
    # role. Preserving source-list index in the debug key keeps the
    # debug shape legible even when exclusions are interleaved with
    # inclusions in the step-2 output.
    for idx, db in enumerate(qu.dealbreakers):
        if db.direction == DealbreakDirection.INCLUSION:
            role = "inclusion_dealbreaker"
            gen = _inclusion_generates(flow, db)
        else:
            role = "exclusion_dealbreaker"
            gen = False  # exclusions never "generate"; they subtract
        tagged.append(
            TaggedItem(
                source=db,
                role=role,
                endpoint=db.route,
                generates_candidates=gen,
                is_primary_preference=False,
                debug_key=f"{role}[{idx}]",
            )
        )

    # Preferences.
    for idx, pref in enumerate(qu.preferences):
        tagged.append(
            TaggedItem(
                source=pref,
                role="preference",
                endpoint=pref.route,
                generates_candidates=_preference_generates(flow),
                is_primary_preference=pref.is_primary_preference,
                debug_key=f"preference[{idx}]",
            )
        )

    return tagged


def _inclusion_generates(flow: Stage4Flow, db: Dealbreaker) -> bool:
    # Inclusion dealbreakers generate candidates in STANDARD (only the
    # non-semantic ones) and in D2 (all of them — they're all semantic
    # by definition of D2). They never generate in P2 or BROWSE because
    # those flows have no inclusion dealbreakers.
    if flow == Stage4Flow.STANDARD:
        return db.route != EndpointRoute.SEMANTIC
    if flow == Stage4Flow.D2:
        return True
    return False


def _preference_generates(flow: Stage4Flow) -> bool:
    # Preferences only generate candidates in P2. In every other flow
    # they run against the pool that inclusion dealbreakers / the
    # browse seed assembled.
    return flow == Stage4Flow.P2
