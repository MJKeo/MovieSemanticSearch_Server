"""Tests for Search V2 Stage 4 flow detection, scoring, and browse seeding."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from schemas.endpoint_result import EndpointResult, ScoredCandidate
from schemas.enums import EndpointRoute
from schemas.query_understanding import (
    DealbreakerMode,
    ExpressionKind,
    PreferenceStrength,
    QueryConcept,
    QueryUnderstandingResponse,
    RetrievalExpression,
)
from search_v2.stage_4.flow_detection import detect_flow, tag_items
from search_v2.stage_4.scoring import E_MULT, P_CAP, score_pool
from search_v2.stage_4.types import EndpointOutcome, TaggedItem
from db.postgres import fetch_browse_seed_ids


def _expression(
    *,
    route: EndpointRoute,
    kind: ExpressionKind,
    description: str,
    coverage_ingredients: list[str] | None = None,
    dealbreaker_mode: DealbreakerMode | None = None,
    preference_strength: PreferenceStrength | None = None,
) -> RetrievalExpression:
    return RetrievalExpression(
        coverage_ingredients=coverage_ingredients or [description],
        routing_rationale="test rationale",
        route=route,
        kind=kind,
        description=description,
        dealbreaker_mode=dealbreaker_mode,
        preference_strength=preference_strength,
    )


def _concept(name: str, expressions: list[RetrievalExpression]) -> QueryConcept:
    required_ingredients = sorted(
        {ingredient for expression in expressions for ingredient in expression.coverage_ingredients}
    ) or [name]
    return QueryConcept(
        boundary_note="test boundary note",
        concept=name,
        required_ingredients=required_ingredients,
        expression_plan_analysis="test analysis",
        expressions=expressions,
    )


def _query_understanding(concepts: list[QueryConcept]) -> QueryUnderstandingResponse:
    ingredient_inventory = [
        ingredient
        for concept in concepts
        for ingredient in concept.required_ingredients
    ]
    return QueryUnderstandingResponse(
        ingredient_inventory=ingredient_inventory,
        concept_inventory_analysis="test inventory",
        concepts=concepts,
    )


def _tagged_item(
    *,
    concept_index: int,
    expression_index: int,
    role: str,
    endpoint: EndpointRoute,
    is_primary_preference: bool = False,
) -> TaggedItem:
    expression = _expression(
        route=endpoint,
        kind=ExpressionKind.PREFERENCE if role == "preference" else ExpressionKind.DEALBREAKER,
        description=f"{role} description",
        dealbreaker_mode=(
            DealbreakerMode.INCLUDE
            if role == "inclusion_dealbreaker"
            else DealbreakerMode.EXCLUDE
            if role == "exclusion_dealbreaker"
            else None
        ),
        preference_strength=(
            PreferenceStrength.CORE if is_primary_preference else PreferenceStrength.SUPPORTING
        )
        if role == "preference"
        else None,
    )
    concept_key = f"concept[{concept_index}]"
    return TaggedItem(
        source=expression,
        role=role,  # type: ignore[arg-type]
        concept_index=concept_index,
        concept_text=f"concept {concept_index}",
        expression_index=expression_index,
        endpoint=endpoint,
        generates_candidates=False,
        is_primary_preference=is_primary_preference,
        concept_debug_key=concept_key,
        debug_key=f"{concept_key}.expression[{expression_index}]",
    )


def _outcome(item: TaggedItem, scores: dict[int, float]) -> EndpointOutcome:
    return EndpointOutcome(
        item=item,
        result=EndpointResult(
            scores=[
                ScoredCandidate(movie_id=movie_id, score=score)
                for movie_id, score in scores.items()
            ]
        ),
        status="ok",
        llm_ms=1.0,
        exec_ms=1.0,
        error_message=None,
    )


@pytest.mark.parametrize(
    ("concepts", "expected_flow"),
    [
        (
            [
                _concept(
                    "Christmas movie",
                    [
                        _expression(
                            route=EndpointRoute.KEYWORD,
                            kind=ExpressionKind.DEALBREAKER,
                            description="is a Christmas movie",
                            dealbreaker_mode=DealbreakerMode.INCLUDE,
                        )
                    ],
                )
            ],
            "standard",
        ),
        (
            [
                _concept(
                    "clown movie",
                    [
                        _expression(
                            route=EndpointRoute.SEMANTIC,
                            kind=ExpressionKind.DEALBREAKER,
                            description="centers around clowns",
                            dealbreaker_mode=DealbreakerMode.INCLUDE,
                        )
                    ],
                )
            ],
            "d2_semantic_only",
        ),
        (
            [
                _concept(
                    "cozy tone",
                    [
                        _expression(
                            route=EndpointRoute.SEMANTIC,
                            kind=ExpressionKind.PREFERENCE,
                            description="cozy and comforting tone",
                            preference_strength=PreferenceStrength.CORE,
                        )
                    ],
                )
            ],
            "p2_preference_driven",
        ),
        ([], "exclusion_only_browse"),
    ],
)
def test_detect_flow_handles_concept_based_query_understanding(
    concepts: list[QueryConcept],
    expected_flow: str,
) -> None:
    qu = _query_understanding(concepts)
    assert detect_flow(qu).value == expected_flow


def test_tag_items_keeps_concept_identity_and_candidate_flags() -> None:
    qu = _query_understanding(
        [
            _concept(
                "Christmas movie",
                [
                    _expression(
                        route=EndpointRoute.KEYWORD,
                        kind=ExpressionKind.DEALBREAKER,
                        description="is a Christmas movie",
                        dealbreaker_mode=DealbreakerMode.INCLUDE,
                    ),
                    _expression(
                        route=EndpointRoute.SEMANTIC,
                        kind=ExpressionKind.PREFERENCE,
                        description="Christmas is central to the story",
                        preference_strength=PreferenceStrength.CORE,
                    ),
                ],
            ),
            _concept(
                "animated movie",
                [
                    _expression(
                        route=EndpointRoute.KEYWORD,
                        kind=ExpressionKind.DEALBREAKER,
                        description="is animated",
                        dealbreaker_mode=DealbreakerMode.EXCLUDE,
                    )
                ],
            ),
        ]
    )

    items = tag_items(qu, detect_flow(qu))

    assert [item.role for item in items] == [
        "inclusion_dealbreaker",
        "preference",
        "exclusion_dealbreaker",
    ]
    assert items[0].concept_debug_key == "concept[0]"
    assert items[0].debug_key == "concept[0].expression[0]"
    assert items[0].generates_candidates is True
    assert items[1].generates_candidates is False
    assert items[2].generates_candidates is False


def test_score_pool_uses_concept_max_for_sibling_inclusion_dealbreakers() -> None:
    pool = [101]
    inclusion_outcomes = [
        _outcome(
            _tagged_item(
                concept_index=0,
                expression_index=0,
                role="inclusion_dealbreaker",
                endpoint=EndpointRoute.KEYWORD,
            ),
            {101: 0.4},
        ),
        _outcome(
            _tagged_item(
                concept_index=0,
                expression_index=1,
                role="inclusion_dealbreaker",
                endpoint=EndpointRoute.SEMANTIC,
            ),
            {101: 0.9},
        ),
        _outcome(
            _tagged_item(
                concept_index=1,
                expression_index=0,
                role="inclusion_dealbreaker",
                endpoint=EndpointRoute.AWARDS,
            ),
            {101: 0.2},
        ),
    ]

    result = score_pool(
        pool,
        inclusion_outcomes=inclusion_outcomes,
        preference_outcomes=[],
        semantic_exclusion_outcomes=[],
    )

    assert result[0].dealbreaker_sum == pytest.approx(1.1)


def test_score_pool_keeps_mixed_concept_dealbreaker_and_preference_separate() -> None:
    pool = [101]
    inclusion_outcomes = [
        _outcome(
            _tagged_item(
                concept_index=0,
                expression_index=0,
                role="inclusion_dealbreaker",
                endpoint=EndpointRoute.KEYWORD,
            ),
            {101: 0.8},
        )
    ]
    preference_outcomes = [
        _outcome(
            _tagged_item(
                concept_index=0,
                expression_index=1,
                role="preference",
                endpoint=EndpointRoute.SEMANTIC,
                is_primary_preference=True,
            ),
            {101: 0.6},
        )
    ]

    result = score_pool(
        pool,
        inclusion_outcomes=inclusion_outcomes,
        preference_outcomes=preference_outcomes,
        semantic_exclusion_outcomes=[],
    )

    assert result[0].dealbreaker_sum == pytest.approx(0.8)
    assert result[0].preference_contribution == pytest.approx(P_CAP * 0.6)
    assert "prior:quality" not in result[0].per_item_scores


def test_score_pool_uses_concept_max_for_semantic_exclusion_penalties() -> None:
    pool = [101]
    exclusion_outcomes = [
        _outcome(
            _tagged_item(
                concept_index=0,
                expression_index=0,
                role="exclusion_dealbreaker",
                endpoint=EndpointRoute.SEMANTIC,
            ),
            {101: 0.2},
        ),
        _outcome(
            _tagged_item(
                concept_index=0,
                expression_index=1,
                role="exclusion_dealbreaker",
                endpoint=EndpointRoute.SEMANTIC,
            ),
            {101: 0.7},
        ),
    ]

    result = score_pool(
        pool,
        inclusion_outcomes=[],
        preference_outcomes=[],
        semantic_exclusion_outcomes=exclusion_outcomes,
    )

    assert result[0].exclusion_penalties == pytest.approx(E_MULT * 0.7)


def test_score_pool_weights_preferences_normally_across_concepts() -> None:
    pool = [101]
    preference_outcomes = [
        _outcome(
            _tagged_item(
                concept_index=0,
                expression_index=0,
                role="preference",
                endpoint=EndpointRoute.SEMANTIC,
                is_primary_preference=True,
            ),
            {101: 1.0},
        ),
        _outcome(
            _tagged_item(
                concept_index=1,
                expression_index=0,
                role="preference",
                endpoint=EndpointRoute.SEMANTIC,
                is_primary_preference=False,
            ),
            {101: 0.5},
        ),
    ]

    result = score_pool(
        pool,
        inclusion_outcomes=[],
        preference_outcomes=preference_outcomes,
        semantic_exclusion_outcomes=[],
    )

    expected = P_CAP * ((3.0 * 1.0 + 1.0 * 0.5) / 4.0)
    assert result[0].preference_contribution == pytest.approx(expected)


@pytest.mark.asyncio
async def test_fetch_browse_seed_ids_uses_popularity_ordering() -> None:
    mock_execute = AsyncMock(return_value=[(7,), (5,), (3,)])

    with patch("db.postgres._execute_read", new=mock_execute):
        result = await fetch_browse_seed_ids(limit=3)

    assert result == [7, 5, 3]
    query, params = mock_execute.await_args.args
    assert "ORDER BY popularity_score DESC NULLS LAST, movie_id DESC" in query
    assert params == (3,)
