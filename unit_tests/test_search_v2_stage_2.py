"""Tests for the revamped Search V2 Stage 2 pipeline and schema."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

try:
    importlib.import_module("implementation.llms.generic_methods")
except ModuleNotFoundError:
    generic_methods_module = ModuleType("implementation.llms.generic_methods")

    class _StubLLMProvider:
        pass

    async def _stub_generate_llm_response_async(*args, **kwargs):
        raise NotImplementedError

    generic_methods_module.LLMProvider = _StubLLMProvider
    generic_methods_module.generate_llm_response_async = _stub_generate_llm_response_async
    sys.modules["implementation.llms.generic_methods"] = generic_methods_module

from schemas.enums import EndpointRoute
from schemas.query_understanding import (
    DealbreakerMode,
    ExpressionKind,
    ExtractedConcept,
    PreferenceStrength,
    QueryConcept,
    RetrievalExpression,
    Step2AResponse,
    Step2BResponse,
)
from search_v2.stage_2 import run_stage_2


def _expr(
    *,
    coverage_ingredients: list[str],
    kind: ExpressionKind,
    route: EndpointRoute = EndpointRoute.KEYWORD,
    description: str = "is a Christmas movie",
) -> RetrievalExpression:
    if kind == ExpressionKind.DEALBREAKER:
        return RetrievalExpression(
            coverage_ingredients=coverage_ingredients,
            routing_rationale="deterministic classification",
            route=route,
            kind=kind,
            description=description,
            dealbreaker_mode=DealbreakerMode.INCLUDE,
        )
    return RetrievalExpression(
        coverage_ingredients=coverage_ingredients,
        routing_rationale="ranking refinement",
        route=route,
        kind=kind,
        description=description,
        preference_strength=PreferenceStrength.CORE,
    )


def _concept(
    *,
    boundary_note: str,
    concept: str,
    required_ingredients: list[str],
    expressions: list[RetrievalExpression],
) -> QueryConcept:
    return QueryConcept(
        boundary_note=boundary_note,
        concept=concept,
        required_ingredients=required_ingredients,
        expression_plan_analysis="test analysis",
        expressions=expressions,
    )


def _step_2b_response(expressions: list[RetrievalExpression]) -> Step2BResponse:
    return Step2BResponse(
        expression_plan_analysis="test planning analysis",
        expressions=expressions,
    )


def test_retrieval_expression_requires_preference_strength() -> None:
    with pytest.raises(ValidationError):
        RetrievalExpression(
            coverage_ingredients=["cozy"],
            routing_rationale="ranking refinement",
            route=EndpointRoute.SEMANTIC,
            kind=ExpressionKind.PREFERENCE,
            description="funny and cozy",
        )


def test_retrieval_expression_requires_dealbreaker_mode() -> None:
    with pytest.raises(ValidationError):
        RetrievalExpression(
            coverage_ingredients=["animated"],
            routing_rationale="deterministic classification",
            route=EndpointRoute.KEYWORD,
            kind=ExpressionKind.DEALBREAKER,
            description="is animated",
        )


def test_positive_presence_descriptions_are_valid_for_include_and_exclude() -> None:
    include_expr = RetrievalExpression(
        coverage_ingredients=["animated"],
        routing_rationale="deterministic classification",
        route=EndpointRoute.KEYWORD,
        kind=ExpressionKind.DEALBREAKER,
        description="is animated",
        dealbreaker_mode=DealbreakerMode.INCLUDE,
    )
    exclude_expr = RetrievalExpression(
        coverage_ingredients=["animated"],
        routing_rationale="deterministic classification",
        route=EndpointRoute.KEYWORD,
        kind=ExpressionKind.DEALBREAKER,
        description="is animated",
        dealbreaker_mode=DealbreakerMode.EXCLUDE,
    )

    assert include_expr.description == "is animated"
    assert exclude_expr.description == "is animated"


def test_step_2a_response_accepts_disney_classics_split() -> None:
    response = Step2AResponse(
        ingredient_inventory=["Disney", "historically significant / classic"],
        concept_inventory_analysis="Split the scope anchor from the evaluative trait.",
        concepts=[
            ExtractedConcept(
                boundary_note="Disney sets the candidate domain.",
                concept="Disney movies",
                required_ingredients=["Disney"],
            ),
            ExtractedConcept(
                boundary_note="Classic/historically significant is a ranking trait.",
                concept="historically significant / classic",
                required_ingredients=["historically significant / classic"],
            ),
        ],
    )

    assert [concept.concept for concept in response.concepts] == [
        "Disney movies",
        "historically significant / classic",
    ]
    assert "historically significant Disney movies" not in {
        concept.concept for concept in response.concepts
    }


def test_step_2a_response_accepts_a24_horror_split() -> None:
    response = Step2AResponse(
        ingredient_inventory=["A24", "horror"],
        concept_inventory_analysis="Split the studio anchor from the genre trait.",
        concepts=[
            ExtractedConcept(
                boundary_note="A24 defines the candidate domain.",
                concept="A24 movies",
                required_ingredients=["A24"],
            ),
            ExtractedConcept(
                boundary_note="Horror is an independent content constraint.",
                concept="horror",
                required_ingredients=["horror"],
            ),
        ],
    )

    assert [concept.required_ingredients for concept in response.concepts] == [
        ["A24"],
        ["horror"],
    ]


def test_step_2a_response_accepts_spiderman_as_one_unified_concept() -> None:
    response = Step2AResponse(
        ingredient_inventory=["Spider-Man"],
        concept_inventory_analysis="Spider-Man is one fused identity concept.",
        concepts=[
            ExtractedConcept(
                boundary_note="Franchise and character views are alternate retrieval "
                "paths within one unified idea.",
                concept="Spider-Man movies",
                required_ingredients=["Spider-Man"],
            )
        ],
    )

    assert [concept.concept for concept in response.concepts] == ["Spider-Man movies"]


def test_step_2a_response_accepts_christmas_as_one_unified_concept() -> None:
    response = Step2AResponse(
        ingredient_inventory=["Christmas"],
        concept_inventory_analysis="Christmas is one fused holiday concept.",
        concepts=[
            ExtractedConcept(
                boundary_note="Deterministic membership and Christmas centrality are "
                "two retrieval views of one concept.",
                concept="Christmas movies",
                required_ingredients=["Christmas"],
            )
        ],
    )

    assert [concept.concept for concept in response.concepts] == ["Christmas movies"]


def test_step_2a_response_rejects_required_ingredient_not_in_inventory() -> None:
    with pytest.raises(
        ValidationError,
        match="must exactly match an item in ingredient_inventory",
    ):
        Step2AResponse(
            ingredient_inventory=["Disney"],
            concept_inventory_analysis="Invalid ingredient mapping.",
            concepts=[
                ExtractedConcept(
                    boundary_note="This concept references an ingredient not inventoried.",
                    concept="classic Disney movies",
                    required_ingredients=["historically significant / classic"],
                )
            ],
        )


def test_query_concept_rejects_missing_coverage_ingredient() -> None:
    with pytest.raises(
        ValidationError,
        match="must be covered by at least one expression",
    ):
        _concept(
            boundary_note="Keep scope and preference inside one concept.",
            concept="Disney movies from the 90s",
            required_ingredients=["Disney", "90s"],
            expressions=[
                _expr(
                    coverage_ingredients=["Disney"],
                    kind=ExpressionKind.DEALBREAKER,
                    route=EndpointRoute.STUDIO,
                    description="movies produced by Disney",
                )
            ],
        )


def test_query_concept_accepts_disney_not_animated_prefer_90s_coverage() -> None:
    concept = _concept(
        boundary_note="All three ingredients must be preserved for this concept.",
        concept="Disney movies not animated, preferably from the 90s",
        required_ingredients=["Disney", "animated", "90s"],
        expressions=[
            _expr(
                coverage_ingredients=["Disney"],
                kind=ExpressionKind.DEALBREAKER,
                route=EndpointRoute.STUDIO,
                description="movies produced by Disney",
            ),
            RetrievalExpression(
                coverage_ingredients=["animated"],
                routing_rationale="Animation is a deterministic exclusion attribute.",
                route=EndpointRoute.KEYWORD,
                kind=ExpressionKind.DEALBREAKER,
                description="is animated",
                dealbreaker_mode=DealbreakerMode.EXCLUDE,
            ),
            _expr(
                coverage_ingredients=["90s"],
                kind=ExpressionKind.PREFERENCE,
                route=EndpointRoute.METADATA,
                description="movies released in the 1990s",
            ),
        ],
    )

    assert [expr.coverage_ingredients for expr in concept.expressions] == [
        ["Disney"],
        ["animated"],
        ["90s"],
    ]


@pytest.mark.asyncio
async def test_run_stage_2_assembles_step_2a_and_step_2b_outputs() -> None:
    step_2a = Step2AResponse(
        ingredient_inventory=["Christmas", "cozy"],
        concept_inventory_analysis="Split holiday membership from tonal preference.",
        concepts=[
            ExtractedConcept(
                boundary_note="Christmas defines the holiday concept.",
                concept="Christmas movie",
                required_ingredients=["Christmas"],
            ),
            ExtractedConcept(
                boundary_note="Cozy is an independent tonal trait.",
                concept="cozy tone",
                required_ingredients=["cozy"],
            ),
        ],
    )

    christmas_plan = _step_2b_response(
        [
            _expr(
                coverage_ingredients=["Christmas"],
                kind=ExpressionKind.DEALBREAKER,
            ),
            _expr(
                coverage_ingredients=["Christmas"],
                kind=ExpressionKind.PREFERENCE,
                route=EndpointRoute.SEMANTIC,
                description="Christmas is central to the story",
            ),
        ]
    )
    cozy_plan = _step_2b_response(
        [
            _expr(
                coverage_ingredients=["cozy"],
                kind=ExpressionKind.PREFERENCE,
                route=EndpointRoute.SEMANTIC,
                description="cozy and comforting tone",
            )
        ]
    )

    with patch(
        "search_v2.stage_2.generate_llm_response_async",
        new=AsyncMock(return_value=(step_2a, 10, 5)),
    ), patch(
        "search_v2.stage_2._run_step_2b_for_concept",
        new=AsyncMock(side_effect=[(christmas_plan, 11, 7), (cozy_plan, 13, 9)]),
    ):
        response, input_tokens, output_tokens = await run_stage_2(
            "Christmas movies that feel cozy",
            provider=object(),
            model="fake-model",
        )

    assert response.ingredient_inventory == ["Christmas", "cozy"]
    assert [concept.concept for concept in response.concepts] == [
        "Christmas movie",
        "cozy tone",
    ]
    assert response.concepts[0].required_ingredients == ["Christmas"]
    assert response.concepts[1].required_ingredients == ["cozy"]
    assert input_tokens == 34
    assert output_tokens == 21


@pytest.mark.asyncio
async def test_run_stage_2_drops_only_failed_step_2b_concept() -> None:
    step_2a = Step2AResponse(
        ingredient_inventory=["Spider-Man", "funny"],
        concept_inventory_analysis="Split the franchise anchor from the tone trait.",
        concepts=[
            ExtractedConcept(
                boundary_note="Spider-Man is the scope anchor.",
                concept="Spider-Man movies",
                required_ingredients=["Spider-Man"],
            ),
            ExtractedConcept(
                boundary_note="Funny is a separate ranking trait.",
                concept="funny tone",
                required_ingredients=["funny"],
            ),
        ],
    )
    spiderman_plan = _step_2b_response(
        [
            _expr(
                coverage_ingredients=["Spider-Man"],
                kind=ExpressionKind.DEALBREAKER,
                route=EndpointRoute.FRANCHISE_STRUCTURE,
                description="is a Spider-Man movie",
            ),
            _expr(
                coverage_ingredients=["Spider-Man"],
                kind=ExpressionKind.DEALBREAKER,
                route=EndpointRoute.ENTITY,
                description="centers on Spider-Man as a character",
            ),
        ]
    )

    with patch(
        "search_v2.stage_2.generate_llm_response_async",
        new=AsyncMock(return_value=(step_2a, 8, 4)),
    ), patch(
        "search_v2.stage_2._run_step_2b_for_concept",
        new=AsyncMock(side_effect=[(spiderman_plan, 9, 6), RuntimeError("boom")]),
    ):
        response, input_tokens, output_tokens = await run_stage_2(
            "funny Spider-Man movies",
            provider=object(),
            model="fake-model",
        )

    assert [concept.concept for concept in response.concepts] == ["Spider-Man movies"]
    assert input_tokens == 17
    assert output_tokens == 10


@pytest.mark.asyncio
async def test_run_stage_2_drops_concept_when_step_2b_coverage_is_invalid() -> None:
    step_2a = Step2AResponse(
        ingredient_inventory=["Disney", "historically significant / classic"],
        concept_inventory_analysis="Split anchor and evaluative trait.",
        concepts=[
            ExtractedConcept(
                boundary_note="Disney is the domain anchor.",
                concept="Disney movies",
                required_ingredients=["Disney"],
            ),
            ExtractedConcept(
                boundary_note="Classic is a separate ranking trait.",
                concept="historically significant / classic",
                required_ingredients=["historically significant / classic"],
            ),
        ],
    )
    valid_plan = _step_2b_response(
        [
            _expr(
                coverage_ingredients=["Disney"],
                kind=ExpressionKind.DEALBREAKER,
                route=EndpointRoute.STUDIO,
                description="movies produced by Disney",
            )
        ]
    )
    invalid_plan = _step_2b_response(
        [
            _expr(
                coverage_ingredients=["Disney"],
                kind=ExpressionKind.PREFERENCE,
                route=EndpointRoute.SEMANTIC,
                description="movies that feel historically significant",
            )
        ]
    )

    with patch(
        "search_v2.stage_2.generate_llm_response_async",
        new=AsyncMock(return_value=(step_2a, 8, 4)),
    ), patch(
        "search_v2.stage_2._run_step_2b_for_concept",
        new=AsyncMock(side_effect=[(valid_plan, 9, 6), (invalid_plan, 7, 5)]),
    ):
        response, _, _ = await run_stage_2(
            "Disney classics",
            provider=object(),
            model="fake-model",
        )

    assert [concept.concept for concept in response.concepts] == ["Disney movies"]


@pytest.mark.asyncio
async def test_run_stage_2_raises_if_all_step_2b_concepts_are_dropped() -> None:
    step_2a = Step2AResponse(
        ingredient_inventory=["Christmas"],
        concept_inventory_analysis="One concept is present.",
        concepts=[
            ExtractedConcept(
                boundary_note="Christmas is a unified holiday concept.",
                concept="Christmas movie",
                required_ingredients=["Christmas"],
            )
        ],
    )
    invalid_plan = _step_2b_response(
        [
            _expr(
                coverage_ingredients=["cozy"],
                kind=ExpressionKind.PREFERENCE,
                route=EndpointRoute.SEMANTIC,
                description="Christmas is central to the story",
            )
        ]
    )

    with patch(
        "search_v2.stage_2.generate_llm_response_async",
        new=AsyncMock(return_value=(step_2a, 5, 3)),
    ), patch(
        "search_v2.stage_2._run_step_2b_for_concept",
        new=AsyncMock(side_effect=[(invalid_plan, 6, 4)]),
    ):
        with pytest.raises(
            RuntimeError,
            match="Step 2A extracted concepts, but every Step 2B concept was dropped.",
        ):
            await run_stage_2(
                "Christmas movies",
                provider=object(),
                model="fake-model",
            )
