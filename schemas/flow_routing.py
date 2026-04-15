# Step 1 (flow routing) LLM structured output models.
#
# Classifies a user query into one of three search flows and may
# produce multiple interpretations when the query is genuinely
# ambiguous. See search_improvement_planning/finalized_search_proposal.md
# (Step 1: Flow Routing) for the full design rationale.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt (search_v2/stage_1.py).
# Developer notes live in comments above the class.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr

from schemas.enums import SearchFlow


# Per-interpretation output. Field order follows the model's decision
# chain: cite routing evidence → rewrite intent → classify flow →
# generate display label → extract title.
class QueryInterpretation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    routing_signals: constr(strip_whitespace=True, min_length=1) = Field(...)
    intent_rewrite: constr(strip_whitespace=True, min_length=1) = Field(...)
    flow: SearchFlow = Field(...)
    display_phrase: constr(strip_whitespace=True, min_length=1) = Field(...)
    title: str | None = Field(default=None)


# Top-level response from the step 1 flow routing LLM.
class FlowRoutingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interpretation_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    interpretations: conlist(
        QueryInterpretation,
        min_length=1,
        max_length=3,
    ) = Field(...)
