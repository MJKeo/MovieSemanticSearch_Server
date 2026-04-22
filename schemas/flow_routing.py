# Step 1 (flow routing) LLM structured output models.
#
# Produces one required primary intent plus up to one optional
# alternative. Step 1 decides both the major search flow and whether
# additional searches would improve browsing value under ambiguity.
# See search_improvement_planning/finalized_search_proposal.md
# (Step 1: Flow Routing) for the full design rationale.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt (search_v2/stage_1.py).
# Developer notes live in comments above the class.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.enums import SearchFlow


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


# Creative spin output. Semantically distinct from AlternativeIntent:
# alternatives capture different readings of what the user asked for,
# while spins propose productive angles on the primary's intent. A
# spin preserves at least one trait from the query and may drop or
# swap the others — covering both narrowings (preserve all traits,
# add specificity) and tangential directions (preserve one trait,
# swap another). Spins are always standard flow, and never carry a
# title — the flow/title pairing validator is reused to keep this
# invariant enforced consistently with the other intent classes.
class CreativeSpin(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spin_angle: constr(strip_whitespace=True, min_length=1) = Field(...)
    intent_rewrite: constr(strip_whitespace=True, min_length=1) = Field(...)
    flow: SearchFlow = Field(...)
    display_phrase: constr(strip_whitespace=True, min_length=1) = Field(...)
    title: str | None = Field(default=None)

    @model_validator(mode="after")
    def validate_flow_title_pairing(self) -> "CreativeSpin":
        _validate_title_for_flow(self.flow, self.title)
        return self


# Top-level response from the step 1 flow routing LLM.
#
# Field order follows the preprocessing chain:
# 1. ambiguity_analysis — brief evidence inventory describing the
#    plausible readings and which one is most likely
# 2. query_traits — single-line list of concrete traits the query
#    carries. Placed before primary_intent so the model has named
#    what the query is made of before committing to a rewrite; this
#    supports both the primary rewrite (understanding composition
#    before writing) and the downstream spin reasoning (which traits
#    to preserve or swap). Structured-output generation runs in
#    field order, so earlier reasoning is frozen by the time later
#    fields are written.
# 3. primary_intent — the default search path
# 4. alternative_intents — up to one materially distinct alternate
#    reading of the user's words (rarely are three interpretations
#    all genuinely useful, and more branches inflate downstream cost)
# 5. creative_spin_analysis — separate trace evaluating spin
#    opportunities grounded in query_traits. Placed after the
#    alternative_intents block so the earlier reasoning is already
#    committed before the model considers spins.
# 6. creative_alternatives — up to two productive angle spins on the
#    primary's intent. Each spin preserves at least one query trait
#    and may drop or swap the others; this covers both narrowings
#    (preserve all, add specificity) and tangential directions
#    (preserve one, swap another). Conceptually distinct from
#    alternative_intents, which capture competing readings.
class FlowRoutingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ambiguity_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    query_traits: constr(strip_whitespace=True, min_length=1) = Field(...)
    primary_intent: PrimaryIntent = Field(...)
    alternative_intents: conlist(
        AlternativeIntent,
        min_length=0,
        max_length=1,
    ) = Field(...)
    creative_spin_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    creative_alternatives: conlist(
        CreativeSpin,
        min_length=0,
        max_length=2,
    ) = Field(...)
