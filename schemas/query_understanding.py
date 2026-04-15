# Step 2 (query understanding) LLM structured output models.
#
# Decomposes a standard-flow query into dealbreakers, preferences,
# and system-level priors. Receives the intent_rewrite from step 1
# as input and produces the full structured decomposition that step 3
# endpoint LLMs consume.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Step 2: Query Understanding) for the full design rationale.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt (search_v2/stage_2.py).
# Developer notes live in comments above the class.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr

from schemas.enums import DealbreakDirection, EndpointRoute, SystemPrior


# Per-dealbreaker output. Field order follows the model's cognitive
# chain: what the requirement is → which way (include/exclude) →
# why this endpoint (concept-type label) → which endpoint.
class Dealbreaker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: constr(strip_whitespace=True, min_length=1) = Field(...)
    direction: DealbreakDirection = Field(...)
    routing_rationale: constr(strip_whitespace=True, min_length=1) = Field(...)
    route: EndpointRoute = Field(...)


# Per-preference output. Same chain as Dealbreaker minus direction
# (all preferences are positive — negative intent is reframed as
# a positive preference for the opposite quality). is_primary_preference
# comes last as a meta-judgment about importance relative to others.
class Preference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: constr(strip_whitespace=True, min_length=1) = Field(...)
    routing_rationale: constr(strip_whitespace=True, min_length=1) = Field(...)
    route: EndpointRoute = Field(...)
    is_primary_preference: bool = Field(...)


# Top-level response from the step 2 query understanding LLM.
#
# Field order follows the preprocessing chain:
# 1. decomposition_analysis — inventory and classify before emitting
#    structured items
# 2. dealbreakers — hard requirements, before preferences because
#    thematic centrality in preferences depends on knowing which
#    keyword dealbreakers were emitted
# 3. preferences — soft ranking qualities
# 4. prior_assessment — evidence inventory for priors, must come
#    after dealbreakers/preferences because suppressed is a
#    second-order inference
# 5-6. quality_prior / notability_prior — constrained enums
#    scaffolded by the assessment
class QueryUnderstandingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decomposition_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)
    dealbreakers: conlist(Dealbreaker, min_length=0) = Field(...)
    preferences: conlist(Preference, min_length=0) = Field(...)
    prior_assessment: constr(strip_whitespace=True, min_length=1) = Field(...)
    quality_prior: SystemPrior = Field(...)
    notability_prior: SystemPrior = Field(...)
