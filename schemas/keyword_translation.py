# Step 3 keyword endpoint structured output model.
#
# Translates a keyword dealbreaker or preference description from
# step 2 into a single UnifiedClassification registry selection that
# step 4 can execute as a GIN `&&` overlap query against the backing
# movie_card array column (keyword_ids / source_material_type_ids /
# concept_tag_ids), with entry_for(member) resolving source + id.
#
# All LLM-facing guidance (how to pick, when to prefer broader vs
# narrower, near-collision disambiguation, reasoning-field framing)
# lives in search_v2/stage_3/keyword_query_generation.py:SYSTEM_PROMPT.
# This module defines shape only.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 5: Keywords & Concept Tags) for the full design
# rationale. The unified classification registry lives in
# schemas/unified_classification.py.
#
# Direction-agnostic: the LLM never sees inclusion/exclusion — step
# 4 code applies that. No abstention: routing already committed the
# item to this endpoint, so the LLM must pick one best fit.
#
# No class-level docstrings or Field descriptions — per the
# "No docstrings on Pydantic classes used as LLM response_format"
# convention, both propagate into the JSON schema sent on every
# API call.

from pydantic import BaseModel, ConfigDict, Field, constr

from schemas.endpoint_parameters import EndpointParameters
from schemas.unified_classification import UnifiedClassification


# Field order (cognitive scaffolding; each reasoning field adjacent
# to the decision it grounds):
#   concept_analysis     — evidence inventory
#   candidate_shortlist  — comparative evaluation
#   classification       — single UnifiedClassification member
class KeywordQuerySpec(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    concept_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)

    candidate_shortlist: constr(strip_whitespace=True, min_length=1) = Field(...)

    classification: UnifiedClassification = Field(...)


# Category-handler wrapper. Direction (inclusion vs exclusion vs
# preference vs downrank) is supplied by action_role + polarity on
# the wrapper; KeywordQuerySpec itself stays direction-agnostic.
class KeywordEndpointParameters(EndpointParameters):
    parameters: KeywordQuerySpec = Field(
        ...,
        description=(
            "Keyword endpoint payload. Pick the single "
            "UnifiedClassification member (keyword, concept tag, or "
            "source-material type) whose concept definition most "
            "directly covers the requirement. One member per call — "
            "do NOT stack picks. If no member fits cleanly, pick the "
            "closest partial match rather than forcing a bad fit."
        ),
    )
