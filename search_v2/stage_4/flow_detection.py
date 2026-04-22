# Search V2 — Stage 4 flow detection and item tagging.
#
# Phase 0 of the orchestrator. Flattens slot actions into TaggedItems,
# selects one of four flow scenarios, and stamps each action with its
# runtime role and candidate-generation behavior.

from __future__ import annotations

from schemas.enums import EndpointRoute
from schemas.query_understanding import (
    ActionRole,
    PreferenceStrength,
    QueryUnderstandingResponse,
)
from search_v2.stage_4.types import Stage4Flow, TaggedItem


def detect_flow(qu: QueryUnderstandingResponse) -> Stage4Flow:
    """Select the stage-4 flow scenario for this branch."""
    items = flatten_items(qu)
    inclusion_dbs = [i for i in items if i.role == "inclusion_dealbreaker"]
    non_semantic_inclusion = [
        i for i in inclusion_dbs if i.endpoint != EndpointRoute.SEMANTIC
    ]

    if non_semantic_inclusion:
        return Stage4Flow.STANDARD
    if inclusion_dbs:
        return Stage4Flow.D2
    if any(i.role == "preference" for i in items):
        return Stage4Flow.P2
    return Stage4Flow.BROWSE


def tag_items(
    qu: QueryUnderstandingResponse, flow: Stage4Flow
) -> list[TaggedItem]:
    """Build the full list of TaggedItems for this branch."""
    tagged = flatten_items(qu)
    return [
        TaggedItem(
            source=item.source,
            role=item.role,
            concept_index=item.concept_index,
            concept_text=item.concept_text,
            expression_index=item.expression_index,
            endpoint=item.endpoint,
            generates_candidates=_generates_candidates(flow, item.role, item.endpoint),
            is_primary_preference=item.is_primary_preference,
            concept_debug_key=item.concept_debug_key,
            debug_key=item.debug_key,
        )
        for item in tagged
    ]


# Map each Stage 2B ActionRole to the internal Stage 4 role string.
# The internal strings stay ("inclusion_dealbreaker",
# "exclusion_dealbreaker", "preference") because they are referenced
# across scoring, assembly, and the flow detector — renaming them is
# out of scope for the Step 2B rewrite.
_ROLE_MAP: dict[ActionRole, str] = {
    ActionRole.INCLUSION: "inclusion_dealbreaker",
    ActionRole.EXCLUSION: "exclusion_dealbreaker",
    ActionRole.PREFERENCE: "preference",
}


def flatten_items(qu: QueryUnderstandingResponse) -> list[TaggedItem]:
    """Flatten completed-slot actions into stable TaggedItems.

    One slot becomes one concept group in Stage 4's concept-level
    aggregation. The debug key format preserves both slot position
    (for stability) and slot handle (for readability). Slots whose
    Stage 2B call produced no actions (slot skipped) contribute
    nothing here — correct behavior, since a skipped slot has no
    retrieval intent to execute.
    """
    tagged: list[TaggedItem] = []

    for slot_index, completed in enumerate(qu.completed_slots):
        # Handles aren't validated unique across slots, so the
        # positional prefix guarantees uniqueness; the handle makes
        # the key human-readable in debug output.
        concept_key = f"slot[{slot_index}]::{completed.slot.handle}"
        for action_index, action in enumerate(completed.response.actions):
            role = _ROLE_MAP[action.role]
            is_primary_preference = (
                action.role == ActionRole.PREFERENCE
                and action.preference_strength == PreferenceStrength.CORE
            )
            tagged.append(
                TaggedItem(
                    source=action,
                    role=role,
                    concept_index=slot_index,
                    concept_text=completed.slot.handle,
                    expression_index=action_index,
                    endpoint=action.route,
                    generates_candidates=False,
                    is_primary_preference=is_primary_preference,
                    concept_debug_key=concept_key,
                    debug_key=f"{concept_key}.action[{action_index}]",
                )
            )

    return tagged


def _generates_candidates(
    flow: Stage4Flow,
    role: str,
    endpoint: EndpointRoute,
) -> bool:
    if role == "exclusion_dealbreaker":
        return False
    if role == "preference":
        return flow == Stage4Flow.P2
    if flow == Stage4Flow.STANDARD:
        return endpoint != EndpointRoute.SEMANTIC
    if flow == Stage4Flow.D2:
        return True
    return False
