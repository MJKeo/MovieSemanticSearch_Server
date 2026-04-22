# Step 2 (query understanding) LLM structured output models.
#
# The revamped Step 2 pipeline first partitions the rewrite into
# planning slots (Step 2A), then plans one or more retrieval actions
# per slot (Step 2B). Each 2B call handles exactly one slot, produces
# a sibling group of RetrievalActions (or skips the slot), and the
# orchestrator assembles per-slot outputs into CompletedSlot records
# that downstream stages consume.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompts in `search_v2/stage_2a.py`
# and `search_v2/stage_2b.py`. Developer notes live in comments above
# the class.

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.enums import EndpointRoute


class PreferenceStrength(StrEnum):
    CORE = "core"
    SUPPORTING = "supporting"


class ActionRole(StrEnum):
    # Hard filter, candidate generator + concept-MAX bonus in Stage 4.
    INCLUSION = "inclusion"
    # Hard filter, subtract-from-pool (non-semantic routes) OR
    # match-then-penalize (semantic route). The route decides which.
    EXCLUSION = "exclusion"
    # Soft scorer, weighted-sum contribution in Stage 4.
    PREFERENCE = "preference"


# Step 2A output unit. Each slot represents one user-intent partition
# cell: a scoped, named, coherent chunk of the Step 1 rewrite that
# Step 2B can plan retrieval actions for independently of siblings.
#
# Fields appear in the order small models should commit them:
#   - handle: a short label the model names first to anchor the slot
#   - scope: which committed inventory phrases live in this slot
#   - retrieval_shape: a ≤8-word phantom-slot sanity check
#   - cohesion: ≤15-word boundary justification
#   - confidence: literal if every scope member came from a literal
#     unit-analysis verdict; inferred if any came from an interpret
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


# Step 2B output unit: one concrete retrieval action within a slot.
# Field order encodes the decision chain small models should commit
# in sequence:
#   1. coverage_atoms  — grounding: which atoms of the slot does this
#      cover? (Must be verbatim atoms from focal_slot.scope.)
#   2. description     — positive-framed "what we want to match,"
#      always written in the positive regardless of role. Stage 3
#      must be able to build a correct endpoint query from this plus
#      the intent_rewrite alone.
#   3. route_rationale — why this family's capability fits the
#      description. Cites the route's capability, not the atom's
#      wording.
#   4. route           — committed endpoint choice.
#   5. role            — how Stage 4 uses this action: inclusion hard
#      filter, exclusion hard filter, or preference scorer.
#   6. preference_strength — nuance, only when role == PREFERENCE.
class RetrievalAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    coverage_atoms: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
    ) = Field(...)
    description: constr(strip_whitespace=True, min_length=1) = Field(...)
    route_rationale: constr(strip_whitespace=True, min_length=1) = Field(...)
    route: EndpointRoute = Field(...)
    role: ActionRole = Field(...)
    preference_strength: PreferenceStrength | None = Field(default=None)

    @model_validator(mode="after")
    def validate_strength_pairing(self) -> "RetrievalAction":
        # preference_strength is required iff role is PREFERENCE. Hard
        # filters have no strength tier — direction is carried by role.
        if self.role == ActionRole.PREFERENCE:
            if self.preference_strength is None:
                raise ValueError(
                    "preference_strength is required when role == 'preference'."
                )
        else:
            if self.preference_strength is not None:
                raise ValueError(
                    "preference_strength must be omitted when role is not "
                    "'preference'."
                )
        return self


# Step 2B top-level response (one per slot call). Field order encodes
# the per-slot decision chain:
#   1. atom_analysis  — per-atom verdict trace. Commits the coverage /
#      expansion / role / route decisions BEFORE any action is written,
#      so structured-output generation order aligns the action list
#      with the committed verdicts.
#   2. skip_rationale — commit point for slot-level skip. Placed
#      before `actions` so that if the model skips the slot, the
#      reason is locked in before the (empty) action list is written.
#      Nullable: populated ONLY when the slot is being skipped.
#   3. actions        — the committed sibling group of retrieval
#      actions. Empty iff skip_rationale is set.
class Step2BResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    atom_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    skip_rationale: str | None = Field(default=None)
    actions: conlist(RetrievalAction, min_length=0) = Field(...)

    @model_validator(mode="after")
    def validate_skip_xor_actions(self) -> "Step2BResponse":
        # Skip-XOR-actions invariant (shape level only; scope coverage
        # against the focal slot is verified by the orchestrator which
        # has the slot in hand).
        has_skip = bool(self.skip_rationale and self.skip_rationale.strip())
        has_actions = len(self.actions) > 0
        if has_skip and has_actions:
            raise ValueError(
                "skip_rationale must be null when actions are non-empty."
            )
        if not has_skip and not has_actions:
            raise ValueError(
                "Empty actions require a non-empty skip_rationale."
            )
        return self


# Orchestrator-assembled wire record (not produced by any single LLM
# call). Pairs each Stage 2A slot with its Stage 2B resolution —
# whether that was an action plan or a slot-level skip. Downstream
# stages consume `completed_slots` from QueryUnderstandingResponse
# rather than re-deriving concept identity from anywhere else.
class CompletedSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slot: PlanningSlot
    response: Step2BResponse


# Top-level Stage 2 output consumed by Stages 3 and 4. Each slot is
# its own concept group (slot = concept by design — see the "no
# intersection combination mode" principle in the Step 2B redesign).
# Stage 4 groups sibling inclusion dealbreakers and semantic
# exclusions by slot identity (MAX within slot, additive across slots).
class QueryUnderstandingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    completed_slots: conlist(CompletedSlot, min_length=0) = Field(...)
