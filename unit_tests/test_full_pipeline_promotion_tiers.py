import pytest

from schemas.enums import EndpointRoute, OperationType, Polarity
from schemas.trait_category import CategoryName
from search_v2.full_pipeline_orchestrator import (
    CategoryCallWithEndpoints,
    GeneratedEndpointSpec,
    PromotionTier,
    Step2BranchResult,
    TraitWithEndpoints,
    _apply_reranker_only_candidate_fallback,
    determine_promotion_tier,
)


def _spec(
    route: EndpointRoute,
    operation_type: OperationType = OperationType.POOL_RERANKER,
) -> GeneratedEndpointSpec:
    return GeneratedEndpointSpec(
        route=route,
        params=None,
        operation_type=operation_type,
    )


def _branch(
    *calls: tuple[CategoryName, Polarity, GeneratedEndpointSpec],
) -> Step2BranchResult:
    traits: list[TraitWithEndpoints] = []
    for idx, (category, polarity, spec) in enumerate(calls):
        traits.append(
            TraitWithEndpoints(
                surface_text=f"trait {idx}",
                polarity=polarity,
                commitment="neutral",
                category_calls=[
                    CategoryCallWithEndpoints(
                        category=category,
                        expressions=[f"expr {idx}"],
                        retrieval_intent=f"intent {idx}",
                        generated_specs=[spec],
                    )
                ],
            )
        )
    return Step2BranchResult(
        kind="original",
        query="query",
        ui_label="Original Query",
        traits=traits,
    )


@pytest.mark.parametrize(
    ("category", "route", "expected"),
    [
        (
            CategoryName.CENTRAL_TOPIC,
            EndpointRoute.SEMANTIC,
            PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
        ),
        (
            CategoryName.PLOT_EVENTS,
            EndpointRoute.SEMANTIC,
            PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
        ),
        (
            CategoryName.NARRATIVE_SETTING,
            EndpointRoute.SEMANTIC,
            PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
        ),
        (
            CategoryName.FILMING_LOCATION,
            EndpointRoute.SEMANTIC,
            PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
        ),
        (
            CategoryName.NAMED_SOURCE_CREATOR,
            EndpointRoute.SEMANTIC,
            PromotionTier.CONCRETE_FACT_OR_IDENTIFIER,
        ),
    ],
)
def test_tier_1_concrete_fact_or_identifier(
    category: CategoryName,
    route: EndpointRoute,
    expected: PromotionTier,
) -> None:
    assert (
        determine_promotion_tier(category, _spec(route), Polarity.POSITIVE)
        is expected
    )


@pytest.mark.parametrize(
    ("category", "route", "expected"),
    [
        (
            CategoryName.ELEMENT_PRESENCE,
            EndpointRoute.SEMANTIC,
            PromotionTier.CONCRETE_ELEMENT_OR_STRUCTURE,
        ),
        (
            CategoryName.GENRE,
            EndpointRoute.SEMANTIC,
            PromotionTier.CONCRETE_ELEMENT_OR_STRUCTURE,
        ),
        (
            CategoryName.FORMAT_VISUAL,
            EndpointRoute.SEMANTIC,
            PromotionTier.CONCRETE_ELEMENT_OR_STRUCTURE,
        ),
        (
            CategoryName.NARRATIVE_DEVICES,
            EndpointRoute.SEMANTIC,
            PromotionTier.CONCRETE_ELEMENT_OR_STRUCTURE,
        ),
    ],
)
def test_tier_2_concrete_element_or_structure(
    category: CategoryName,
    route: EndpointRoute,
    expected: PromotionTier,
) -> None:
    assert (
        determine_promotion_tier(category, _spec(route), Polarity.POSITIVE)
        is expected
    )


@pytest.mark.parametrize(
    ("category", "route", "expected"),
    [
        (
            CategoryName.CHARACTER_ARCHETYPE,
            EndpointRoute.SEMANTIC,
            PromotionTier.ABSTRACT_TYPE_OR_ARCHETYPE,
        ),
        (
            CategoryName.STORY_THEMATIC_ARCHETYPE,
            EndpointRoute.SEMANTIC,
            PromotionTier.ABSTRACT_TYPE_OR_ARCHETYPE,
        ),
    ],
)
def test_tier_3_abstract_type_or_archetype(
    category: CategoryName,
    route: EndpointRoute,
    expected: PromotionTier,
) -> None:
    assert (
        determine_promotion_tier(category, _spec(route), Polarity.POSITIVE)
        is expected
    )


@pytest.mark.parametrize(
    ("category", "route", "expected"),
    [
        (
            CategoryName.VISUAL_CRAFT_ACCLAIM,
            EndpointRoute.SEMANTIC,
            PromotionTier.RECEPTION_OR_PRAISE_PROSE,
        ),
        (
            CategoryName.MUSIC_SCORE_ACCLAIM,
            EndpointRoute.SEMANTIC,
            PromotionTier.RECEPTION_OR_PRAISE_PROSE,
        ),
        (
            CategoryName.DIALOGUE_CRAFT_ACCLAIM,
            EndpointRoute.SEMANTIC,
            PromotionTier.RECEPTION_OR_PRAISE_PROSE,
        ),
        (
            CategoryName.CULTURAL_STATUS,
            EndpointRoute.SEMANTIC,
            PromotionTier.RECEPTION_OR_PRAISE_PROSE,
        ),
        (
            CategoryName.SPECIFIC_PRAISE_CRITICISM,
            EndpointRoute.SEMANTIC,
            PromotionTier.RECEPTION_OR_PRAISE_PROSE,
        ),
    ],
)
def test_tier_4_reception_or_praise_prose(
    category: CategoryName,
    route: EndpointRoute,
    expected: PromotionTier,
) -> None:
    assert (
        determine_promotion_tier(category, _spec(route), Polarity.POSITIVE)
        is expected
    )


@pytest.mark.parametrize(
    ("category", "route", "expected"),
    [
        (
            CategoryName.TARGET_AUDIENCE,
            EndpointRoute.SEMANTIC,
            PromotionTier.AUDIENCE_SENSITIVITY_OR_SEASONAL,
        ),
        (
            CategoryName.SENSITIVE_CONTENT,
            EndpointRoute.SEMANTIC,
            PromotionTier.AUDIENCE_SENSITIVITY_OR_SEASONAL,
        ),
        (
            CategoryName.SEASONAL_HOLIDAY,
            EndpointRoute.SEMANTIC,
            PromotionTier.AUDIENCE_SENSITIVITY_OR_SEASONAL,
        ),
    ],
)
def test_tier_5_audience_sensitivity_or_seasonal(
    category: CategoryName,
    route: EndpointRoute,
    expected: PromotionTier,
) -> None:
    assert (
        determine_promotion_tier(category, _spec(route), Polarity.POSITIVE)
        is expected
    )


@pytest.mark.parametrize(
    ("category", "route", "expected"),
    [
        (
            CategoryName.EMOTIONAL_EXPERIENTIAL,
            EndpointRoute.SEMANTIC,
            PromotionTier.VIBES_OR_CONTEXT_FIT,
        ),
        (
            CategoryName.VIEWING_OCCASION,
            EndpointRoute.SEMANTIC,
            PromotionTier.VIBES_OR_CONTEXT_FIT,
        ),
    ],
)
def test_tier_6_vibes_or_context_fit(
    category: CategoryName,
    route: EndpointRoute,
    expected: PromotionTier,
) -> None:
    assert (
        determine_promotion_tier(category, _spec(route), Polarity.POSITIVE)
        is expected
    )


@pytest.mark.parametrize(
    ("category", "route", "expected"),
    [
        (
            CategoryName.GENERAL_APPEAL,
            EndpointRoute.METADATA,
            PromotionTier.GLOBAL_METADATA_PRIOR_OR_ORDINAL,
        ),
        (
            CategoryName.CULTURAL_STATUS,
            EndpointRoute.METADATA,
            PromotionTier.GLOBAL_METADATA_PRIOR_OR_ORDINAL,
        ),
        (
            CategoryName.CHRONOLOGICAL,
            EndpointRoute.METADATA,
            PromotionTier.GLOBAL_METADATA_PRIOR_OR_ORDINAL,
        ),
    ],
)
def test_tier_7_global_metadata_prior_or_ordinal(
    category: CategoryName,
    route: EndpointRoute,
    expected: PromotionTier,
) -> None:
    assert (
        determine_promotion_tier(category, _spec(route), Polarity.POSITIVE)
        is expected
    )


@pytest.mark.parametrize(
    ("category", "route"),
    [
        (CategoryName.GENRE, EndpointRoute.KEYWORD),
        (CategoryName.RELEASE_DATE, EndpointRoute.METADATA),
        (CategoryName.PERSON_CREDIT, EndpointRoute.ENTITY),
        (CategoryName.TRENDING, EndpointRoute.TRENDING),
        (CategoryName.MEDIA_TYPE, EndpointRoute.MEDIA_TYPE),
    ],
)
def test_candidate_generators_never_promote(
    category: CategoryName,
    route: EndpointRoute,
) -> None:
    assert (
        determine_promotion_tier(
            category,
            _spec(route, OperationType.CANDIDATE_GENERATOR),
            Polarity.POSITIVE,
        )
        is PromotionTier.NEVER_PROMOTE
    )


@pytest.mark.parametrize(
    ("category", "route"),
    [
        (CategoryName.GENRE, EndpointRoute.KEYWORD),
        (CategoryName.PLOT_EVENTS, EndpointRoute.SEMANTIC),
        (CategoryName.GENERAL_APPEAL, EndpointRoute.METADATA),
    ],
)
def test_negative_polarity_never_promotes(
    category: CategoryName,
    route: EndpointRoute,
) -> None:
    assert (
        determine_promotion_tier(category, _spec(route), Polarity.NEGATIVE)
        is PromotionTier.NEVER_PROMOTE
    )


def test_positive_unmapped_metadata_reranker_never_promotes() -> None:
    assert (
        determine_promotion_tier(
            CategoryName.RELEASE_DATE,
            _spec(EndpointRoute.METADATA),
            Polarity.POSITIVE,
        )
        is PromotionTier.NEVER_PROMOTE
    )


def test_positive_unmapped_semantic_reranker_never_promotes() -> None:
    assert (
        determine_promotion_tier(
            CategoryName.CULTURAL_TRADITION,
            _spec(EndpointRoute.SEMANTIC),
            Polarity.POSITIVE,
        )
        is PromotionTier.NEVER_PROMOTE
    )


def test_promotion_tier_values_are_ordered() -> None:
    assert PromotionTier.NEVER_PROMOTE == -1
    assert PromotionTier.CONCRETE_FACT_OR_IDENTIFIER == 1
    assert PromotionTier.CONCRETE_ELEMENT_OR_STRUCTURE == 2
    assert PromotionTier.ABSTRACT_TYPE_OR_ARCHETYPE == 3
    assert PromotionTier.RECEPTION_OR_PRAISE_PROSE == 4
    assert PromotionTier.AUDIENCE_SENSITIVITY_OR_SEASONAL == 5
    assert PromotionTier.VIBES_OR_CONTEXT_FIT == 6
    assert PromotionTier.GLOBAL_METADATA_PRIOR_OR_ORDINAL == 7


def test_reranker_only_fallback_promotes_lowest_available_tier() -> None:
    tier_1 = _spec(EndpointRoute.SEMANTIC)
    tier_4 = _spec(EndpointRoute.SEMANTIC)

    auxiliary = _apply_reranker_only_candidate_fallback([
        _branch(
            (CategoryName.PLOT_EVENTS, Polarity.POSITIVE, tier_1),
            (CategoryName.MUSIC_SCORE_ACCLAIM, Polarity.POSITIVE, tier_4),
        )
    ])

    assert auxiliary == []
    assert tier_1.operation_type is OperationType.CANDIDATE_GENERATOR
    assert tier_4.operation_type is OperationType.POOL_RERANKER


def test_reranker_only_fallback_promotes_all_specs_at_lowest_tier() -> None:
    plot = _spec(EndpointRoute.SEMANTIC)
    setting = _spec(EndpointRoute.SEMANTIC)
    vibe = _spec(EndpointRoute.SEMANTIC)

    auxiliary = _apply_reranker_only_candidate_fallback([
        _branch(
            (CategoryName.PLOT_EVENTS, Polarity.POSITIVE, plot),
            (CategoryName.NARRATIVE_SETTING, Polarity.POSITIVE, setting),
            (CategoryName.EMOTIONAL_EXPERIENTIAL, Polarity.POSITIVE, vibe),
        )
    ])

    assert auxiliary == []
    assert plot.operation_type is OperationType.CANDIDATE_GENERATOR
    assert setting.operation_type is OperationType.CANDIDATE_GENERATOR
    assert vibe.operation_type is OperationType.POOL_RERANKER


def test_reranker_only_fallback_ignores_never_promote_when_positive_tier_exists() -> None:
    negative = _spec(EndpointRoute.KEYWORD)
    positive = _spec(EndpointRoute.SEMANTIC)

    auxiliary = _apply_reranker_only_candidate_fallback([
        _branch(
            (CategoryName.GENRE, Polarity.NEGATIVE, negative),
            (CategoryName.PLOT_EVENTS, Polarity.POSITIVE, positive),
        )
    ])

    assert auxiliary == []
    assert negative.operation_type is OperationType.POOL_RERANKER
    assert positive.operation_type is OperationType.CANDIDATE_GENERATOR


def test_reranker_only_fallback_emits_neutral_seed_when_nothing_promotable() -> None:
    negative = _spec(EndpointRoute.KEYWORD)

    auxiliary = _apply_reranker_only_candidate_fallback([
        _branch((CategoryName.GENRE, Polarity.NEGATIVE, negative))
    ])

    assert negative.operation_type is OperationType.POOL_RERANKER
    assert len(auxiliary) == 1
    assert auxiliary[0].route is EndpointRoute.NEUTRAL_SEED
    assert auxiliary[0].params is None
    assert auxiliary[0].operation_type is OperationType.CANDIDATE_GENERATOR


def test_reranker_only_fallback_skips_when_candidate_generator_exists() -> None:
    candidate = _spec(
        EndpointRoute.KEYWORD,
        OperationType.CANDIDATE_GENERATOR,
    )
    reranker = _spec(EndpointRoute.SEMANTIC)

    auxiliary = _apply_reranker_only_candidate_fallback([
        _branch(
            (CategoryName.GENRE, Polarity.POSITIVE, candidate),
            (CategoryName.PLOT_EVENTS, Polarity.POSITIVE, reranker),
        )
    ])

    assert auxiliary == []
    assert candidate.operation_type is OperationType.CANDIDATE_GENERATOR
    assert reranker.operation_type is OperationType.POOL_RERANKER


def test_reranker_only_fallback_skips_when_no_endpoint_specs_exist() -> None:
    auxiliary = _apply_reranker_only_candidate_fallback([
        Step2BranchResult(
            kind="original",
            query="query",
            ui_label="Original Query",
            traits=[],
        )
    ])

    assert auxiliary == []
