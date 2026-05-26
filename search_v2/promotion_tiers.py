# Search V2 — Promotion tier taxonomy for the reranker→generator
# fallback.
#
# Extracted from `full_pipeline_orchestrator` so Stage 4's per-branch
# tiered-promotion loop can consume the same tier data without creating
# an import cycle (the orchestrator already imports from
# `stage_4_execution`). All semantics — values, ordering, category
# membership — are preserved verbatim.

from __future__ import annotations

from enum import IntEnum

from schemas.enums import EndpointRoute, OperationType, Polarity
from schemas.trait_category import CategoryName
from search_v2.endpoint_fetching.category_handlers.generated_endpoint_spec import (
    GeneratedEndpointSpec,
)


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


_SEMANTIC_PROMOTION_TIERS: dict[CategoryName, PromotionTier] = {
    # Tier 1 — concrete fact / specific identifier.
    CategoryName.CENTRAL_TOPIC: PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
    CategoryName.PLOT_EVENTS: PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
    CategoryName.NARRATIVE_SETTING: PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
    CategoryName.FILMING_LOCATION: PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
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
