# Step 2 (query understanding) LLM structured output models.
#
# The revamped Step 2 pipeline first partitions the rewrite into
# planning slots (Step 2A), then plans one or more retrieval
# expressions per slot (Step 2B). The final public response preserves
# concept identity so Stage 4 can aggregate sibling dealbreakers at the
# concept level.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt(s) in `search_v2/stage_2a.py`
# and `search_v2/stage_2.py`.
# Developer notes live in comments above the class.

from __future__ import annotations

from enum import StrEnum
from typing import Literal

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


# Legacy shape consumed by Step 2B's per-concept loop. Kept alive until
# Step 2B is reworked to consume PlanningSlot directly; at that point
# ExtractedConcept should be removed.
class ExtractedConcept(BaseModel):
    model_config = ConfigDict(extra="forbid")

    boundary_note: constr(strip_whitespace=True, min_length=1) = Field(...)
    concept: constr(strip_whitespace=True, min_length=1) = Field(...)
    required_ingredients: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
    ) = Field(...)


# Step 2A output unit. Each slot represents one user-intent partition
# cell: a scoped, named, coherent chunk of the Step 1 rewrite that
# Step 2B can plan retrieval expressions for independently of siblings.
#
# Fields appear in the order small models should commit them:
#   - handle: a short label the model names first to anchor the slot
#   - scope: which committed inventory phrases live in this slot
#   - retrieval_shape: a ≤8-word phantom-slot sanity check
#   - cohesion: ≤15-word boundary justification
#   - confidence: literal if every scope member came from a literal
#     unit-analysis verdict; inferred if any came from a best_guess
class PlanningSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    handle: constr(strip_whitespace=True, min_length=1) = Field(...)
    scope: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
    ) = Field(...)
    retrieval_shape: constr(strip_whitespace=True, min_length=1) = Field(...)
    cohesion: constr(strip_whitespace=True, min_length=1) = Field(...)
    confidence: Literal["literal", "inferred"] = Field(...)


# Step 2A top-level response. Field order is the cognitive scaffold
# for small-model generation:
#   1. unit_analysis — per-rewrite-phrase verdict trace (literal /
#      best_guess / filler / fold_into). Commits per-item verdicts
#      BEFORE any inventory exists.
#   2. inventory — the committed actionable phrase set derived from
#      unit_analysis. Validated to back every slot scope entry.
#   3. slot_analysis — per-candidate-slot verdict trace (emit /
#      fuse_with). Commits fuse-vs-split decisions BEFORE slots are
#      written.
#   4. slots — the final partition handed off to Step 2B.
class Step2AResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    unit_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    inventory: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=0,
    ) = Field(...)
    slot_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    slots: conlist(PlanningSlot, min_length=0) = Field(...)

    @model_validator(mode="after")
    def validate_partition_completeness(self) -> "Step2AResponse":
        # Enforce the partition invariants that make Step 2B's parallel
        # planning safe:
        #   1. Every slot scope entry must appear in inventory (slots
        #      only reference phrases that were committed retrievable).
        #   2. Every inventory entry must appear in some slot scope —
        #      otherwise a "retrievable" phrase falls on the floor and
        #      the pipeline silently drops user intent.
        # The confidence flag itself is trusted from the model
        # (verifying it would require parsing the free-form trace).
        inventory = set(self.inventory)
        covered: set[str] = set()
        for slot in self.slots:
            for scope_entry in slot.scope:
                if scope_entry not in inventory:
                    raise ValueError(
                        "Each PlanningSlot scope entry must exactly match an "
                        "item in inventory."
                    )
                covered.add(scope_entry)
        orphaned = inventory - covered
        if orphaned:
            raise ValueError(
                "Every inventory entry must appear in some slot scope. "
                f"Orphaned: {sorted(orphaned)}"
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
