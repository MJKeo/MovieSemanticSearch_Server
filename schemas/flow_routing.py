# Step 1 (flow routing) LLM structured output models.
#
# Produces one required primary intent plus up to two optional
# alternatives. Step 1 decides both the major search flow and whether
# additional searches would improve browsing value under ambiguity.
# See search_improvement_planning/finalized_search_proposal.md
# (Step 1: Flow Routing) for the full design rationale.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt (search_v2/stage_1.py).
# Developer notes live in comments above the class.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.enums import QueryAmbiguityLevel, SearchFlow


def _validate_title_for_flow(flow: SearchFlow, title: str | None) -> None:
    """Enforce the flow/title invariant shared by both intent models."""
    if flow == SearchFlow.STANDARD and title is not None:
        raise ValueError("standard intents must set title to null.")

    if flow in {SearchFlow.EXACT_TITLE, SearchFlow.SIMILARITY}:
        if title is None or not title.strip():
            raise ValueError("exact_title and similarity intents require a non-empty title.")


# Primary intent output. Field order follows the model's decision
# chain: cite routing evidence → rewrite intent → classify flow →
# generate display label → extract title.
class PrimaryIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routing_signals: constr(strip_whitespace=True, min_length=1) = Field(...)
    intent_rewrite: constr(strip_whitespace=True, min_length=1) = Field(...)
    flow: SearchFlow = Field(...)
    display_phrase: constr(strip_whitespace=True, min_length=1) = Field(...)
    title: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_flow_title_pairing(self) -> "PrimaryIntent":
        _validate_title_for_flow(self.flow, self.title)
        return self


# Alternative intent output. Same core chain as primary intent, with an
# extra reasoning field immediately after the evidence inventory to force
# meaningful differentiation rather than paraphrasing.
class AlternativeIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routing_signals: constr(strip_whitespace=True, min_length=1) = Field(...)
    difference_rationale: constr(strip_whitespace=True, min_length=1) = Field(...)
    intent_rewrite: constr(strip_whitespace=True, min_length=1) = Field(...)
    flow: SearchFlow = Field(...)
    display_phrase: constr(strip_whitespace=True, min_length=1) = Field(...)
    title: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_flow_title_pairing(self) -> "AlternativeIntent":
        _validate_title_for_flow(self.flow, self.title)
        return self


# Top-level response from the step 1 flow routing LLM.
#
# Field order follows the preprocessing chain:
# 1. ambiguity_analysis — brief evidence inventory about whether more
#    than one search would add browsing value
# 2. ambiguity_level — compact classification of branching pressure
# 3. hard_constraints — fixed traits that must survive every emitted
#    branch
# 4. ambiguity_sources — the clause(s) or concept(s) that are open to
#    interpretation
# 5. primary_intent — the default search path
# 6. alternative_intents — up to two materially distinct alternates
class FlowRoutingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ambiguity_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    ambiguity_level: QueryAmbiguityLevel = Field(...)
    hard_constraints: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=0,
    ) = Field(...)
    ambiguity_sources: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=0,
    ) = Field(...)
    primary_intent: PrimaryIntent = Field(...)
    alternative_intents: conlist(
        AlternativeIntent,
        min_length=0,
        max_length=2,
    ) = Field(...)
