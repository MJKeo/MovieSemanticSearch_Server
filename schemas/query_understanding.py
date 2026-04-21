# Step 2 (query understanding) LLM structured output models.
#
# The revamped Step 2 pipeline first extracts concept boundaries
# (Step 2A), then plans one or more retrieval expressions per concept
# (Step 2B). The final public response preserves concept identity so
# Stage 4 can aggregate sibling dealbreakers at the concept level.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt(s) in `search_v2/stage_2.py`.
# Developer notes live in comments above the class.

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.enums import EndpointRoute


class ExpressionKind(StrEnum):
    DEALBREAKER = "dealbreaker"
    PREFERENCE = "preference"


class DealbreakerMode(StrEnum):
    INCLUDE = "include"
    EXCLUDE = "exclude"


class PreferenceStrength(StrEnum):
    CORE = "core"
    SUPPORTING = "supporting"


class ExtractedConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boundary_note: constr(strip_whitespace=True, min_length=1) = Field(...)
    concept: constr(strip_whitespace=True, min_length=1) = Field(...)
    required_ingredients: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
    ) = Field(...)


class Step2AResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ingredient_inventory: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=0,
    ) = Field(...)
    concept_inventory_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    concepts: conlist(ExtractedConcept, min_length=0) = Field(...)

    @model_validator(mode="after")
    def validate_concept_ingredients_against_inventory(self) -> "Step2AResponse":
        inventory = set(self.ingredient_inventory)
        for concept in self.concepts:
            for ingredient in concept.required_ingredients:
                if ingredient not in inventory:
                    raise ValueError(
                        "Each concept required_ingredients entry must exactly match "
                        "an item in ingredient_inventory."
                    )
        return self


class RetrievalExpression(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coverage_ingredients: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
    ) = Field(...)
    routing_rationale: constr(strip_whitespace=True, min_length=1) = Field(...)
    route: EndpointRoute = Field(...)
    kind: ExpressionKind = Field(...)
    description: constr(strip_whitespace=True, min_length=1) = Field(...)
    dealbreaker_mode: DealbreakerMode | None = Field(default=None)
    preference_strength: PreferenceStrength | None = Field(default=None)

    @model_validator(mode="after")
    def validate_kind_specific_fields(self) -> "RetrievalExpression":
        if self.kind == ExpressionKind.DEALBREAKER:
            if self.dealbreaker_mode is None:
                raise ValueError(
                    "dealbreaker_mode is required when kind == 'dealbreaker'."
                )
            if self.preference_strength is not None:
                raise ValueError(
                    "preference_strength must be omitted when kind == 'dealbreaker'."
                )
        else:
            if self.preference_strength is None:
                raise ValueError(
                    "preference_strength is required when kind == 'preference'."
                )
            if self.dealbreaker_mode is not None:
                raise ValueError(
                    "dealbreaker_mode must be omitted when kind == 'preference'."
                )
        return self


class Step2BResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expression_plan_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    expressions: conlist(RetrievalExpression, min_length=1) = Field(...)


class QueryConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boundary_note: constr(strip_whitespace=True, min_length=1) = Field(...)
    concept: constr(strip_whitespace=True, min_length=1) = Field(...)
    required_ingredients: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
    ) = Field(...)
    expression_plan_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    expressions: conlist(RetrievalExpression, min_length=1) = Field(...)

    @model_validator(mode="after")
    def validate_expression_coverage(self) -> "QueryConcept":
        required_ingredients = set(self.required_ingredients)
        covered_ingredients: set[str] = set()

        for expression in self.expressions:
            for ingredient in expression.coverage_ingredients:
                if ingredient not in required_ingredients:
                    raise ValueError(
                        "Each coverage_ingredients entry must exactly match an item "
                        "in the concept required_ingredients list."
                    )
                covered_ingredients.add(ingredient)

        missing_ingredients = required_ingredients - covered_ingredients
        if missing_ingredients:
            raise ValueError(
                "Every concept required_ingredients entry must be covered by at "
                "least one expression."
            )

        return self


class QueryUnderstandingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ingredient_inventory: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=0,
    ) = Field(...)
    concept_inventory_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    concepts: conlist(QueryConcept, min_length=0) = Field(...)
