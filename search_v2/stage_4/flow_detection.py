# Search V2 — Stage 4 flow detection and item tagging.
#
# Phase 0 of the orchestrator. Flattens concept expressions into
# TaggedItems, selects one of four flow scenarios, and stamps each
# expression with its runtime role and candidate-generation behavior.

from __future__ import annotations

from schemas.enums import EndpointRoute
from schemas.query_understanding import (
    DealbreakerMode,
    ExpressionKind,
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


def flatten_items(qu: QueryUnderstandingResponse) -> list[TaggedItem]:
    """Flatten concept expressions into stable TaggedItems."""
    tagged: list[TaggedItem] = []

    for concept_index, concept in enumerate(qu.concepts):
        concept_key = f"concept[{concept_index}]"
        for expression_index, expression in enumerate(concept.expressions):
            if expression.kind == ExpressionKind.DEALBREAKER:
                assert expression.dealbreaker_mode is not None
                role = (
                    "inclusion_dealbreaker"
                    if expression.dealbreaker_mode == DealbreakerMode.INCLUDE
                    else "exclusion_dealbreaker"
                )
                is_primary_preference = False
            else:
                role = "preference"
                is_primary_preference = (
                    expression.preference_strength == PreferenceStrength.CORE
                )

            tagged.append(
                TaggedItem(
                    source=expression,
                    role=role,
                    concept_index=concept_index,
                    concept_text=concept.concept,
                    expression_index=expression_index,
                    endpoint=expression.route,
                    generates_candidates=False,
                    is_primary_preference=is_primary_preference,
                    concept_debug_key=concept_key,
                    debug_key=f"{concept_key}.expression[{expression_index}]",
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
