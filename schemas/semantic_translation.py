# Step 3 semantic endpoint structured output models.
#
# Two top-level response_format classes cover the semantic endpoint:
#
#   - SemanticDealbreakerSpec — one LLM call per semantic dealbreaker
#     (also used for semantic exclusions). Covers scenarios D1 and D2
#     from the finalized proposal. Dealbreakers select exactly one
#     space from the 7 non-anchor spaces.
#
#   - SemanticPreferenceSpec — one LLM call per grouped semantic
#     preference. Covers scenarios P1 and P2. Preferences select 1+
#     spaces from all 8 (anchor allowed), each with a two-level
#     categorical weight (central / supporting).
#
# Both specs emit CONCRETE per-space objects (AnchorBody,
# PlotEventsBody, etc.) rather than free-form query strings, so
# query-side vectors embed into the same structured-label format as
# document-side vectors. The discriminator pattern (Literal[space] on
# each wrapper) forces the LLM to commit to a space before filling
# the matching body shape — JSON schema validation rejects any
# mismatch between declared space and body fields.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 6: Semantic → Execution Scenarios) for the full design.
# All LLM-facing guidance (how to pick a space, how to decompose a
# grouped preference, central-vs-supporting rules) lives in the
# system prompt authored in search_v2/stage_3/semantic_query_generation.py.
#
# No class-level docstrings or Field(description=...) — those
# propagate into the JSON schema sent on every API call, per the
# convention established in schemas/keyword_translation.py and
# schemas/award_translation.py.

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.semantic_bodies import (
    AnchorBody,
    NarrativeTechniquesBody,
    PlotAnalysisBody,
    PlotEventsBody,
    ProductionBody,
    ReceptionBody,
    ViewerExperienceBody,
    WatchContextBody,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
#
# DealbreakerSpace and VectorSpace share 7 members and differ only on
# the presence of ANCHOR. They are declared separately rather than
# subclassed because OpenAI structured output compiles each enum into
# a concrete JSON-schema enum restriction; declaring the
# 7-vs-8-member sets independently keeps the schema surface explicit
# and avoids framework-specific inheritance gotchas.


class DealbreakerSpace(str, Enum):
    PLOT_EVENTS = "plot_events"
    PLOT_ANALYSIS = "plot_analysis"
    VIEWER_EXPERIENCE = "viewer_experience"
    WATCH_CONTEXT = "watch_context"
    NARRATIVE_TECHNIQUES = "narrative_techniques"
    PRODUCTION = "production"
    RECEPTION = "reception"


class VectorSpace(str, Enum):
    ANCHOR = "anchor"
    PLOT_EVENTS = "plot_events"
    PLOT_ANALYSIS = "plot_analysis"
    VIEWER_EXPERIENCE = "viewer_experience"
    WATCH_CONTEXT = "watch_context"
    NARRATIVE_TECHNIQUES = "narrative_techniques"
    PRODUCTION = "production"
    RECEPTION = "reception"


class PreferenceSpaceWeight(str, Enum):
    CENTRAL = "central"
    SUPPORTING = "supporting"


# ---------------------------------------------------------------------------
# Dealbreaker discriminator wrappers
# ---------------------------------------------------------------------------
#
# One wrapper per non-anchor space. Each pins a Literal[...] on the
# matching DealbreakerSpace value and carries exactly one Body field.
# The Literal + extra="forbid" forces every body to match exactly one
# wrapper, so even without an explicit discriminator the union resolves
# unambiguously. Mixing space="viewer_experience" with a
# NarrativeTechniquesBody-shaped payload is a schema-level error.


class PlotEventsDealbreaker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[DealbreakerSpace.PLOT_EVENTS]
    content: PlotEventsBody


class PlotAnalysisDealbreaker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[DealbreakerSpace.PLOT_ANALYSIS]
    content: PlotAnalysisBody


class ViewerExperienceDealbreaker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[DealbreakerSpace.VIEWER_EXPERIENCE]
    content: ViewerExperienceBody


class WatchContextDealbreaker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[DealbreakerSpace.WATCH_CONTEXT]
    content: WatchContextBody


class NarrativeTechniquesDealbreaker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[DealbreakerSpace.NARRATIVE_TECHNIQUES]
    content: NarrativeTechniquesBody


class ProductionDealbreaker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[DealbreakerSpace.PRODUCTION]
    content: ProductionBody


class ReceptionDealbreaker(BaseModel):
    model_config = ConfigDict(extra="forbid")

    space: Literal[DealbreakerSpace.RECEPTION]
    content: ReceptionBody


# Plain Union (no Field(discriminator=...)) so Pydantic emits an
# `anyOf` JSON-schema node rather than `oneOf`. OpenAI's Structured
# Outputs rejects `oneOf`. The Literal[DealbreakerSpace.X] tag on each
# wrapper combined with extra="forbid" still guarantees that any given
# body matches exactly one branch, so `anyOf` is functionally
# equivalent to `oneOf` here.
DealbreakerBody = Union[
    PlotEventsDealbreaker,
    PlotAnalysisDealbreaker,
    ViewerExperienceDealbreaker,
    WatchContextDealbreaker,
    NarrativeTechniquesDealbreaker,
    ProductionDealbreaker,
    ReceptionDealbreaker,
]


# ---------------------------------------------------------------------------
# Top-level dealbreaker spec.
# Field order: signal_inventory → target_fields_label → body.
# (body is a discriminated union; its `.space` is the final commit,
# `.content` holds the per-space Body.)
# LLM-facing guidance on each reasoning field lives in
# search_v2/stage_3/semantic_query_generation.py.
# ---------------------------------------------------------------------------
class SemanticDealbreakerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    signal_inventory: constr(strip_whitespace=True, min_length=1)
    target_fields_label: constr(strip_whitespace=True, min_length=1)
    body: DealbreakerBody


# ---------------------------------------------------------------------------
# Preference discriminator wrappers.
# One wrapper per space (anchor included).
# Field order inside each entry: carries_qualifiers → space → weight → content.
# LLM-facing guidance on carries_qualifiers lives in
# search_v2/stage_3/semantic_query_generation.py.
# ---------------------------------------------------------------------------


class AnchorPreferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1)
    space: Literal[VectorSpace.ANCHOR]
    weight: PreferenceSpaceWeight
    content: AnchorBody


class PlotEventsPreferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1)
    space: Literal[VectorSpace.PLOT_EVENTS]
    weight: PreferenceSpaceWeight
    content: PlotEventsBody


class PlotAnalysisPreferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1)
    space: Literal[VectorSpace.PLOT_ANALYSIS]
    weight: PreferenceSpaceWeight
    content: PlotAnalysisBody


class ViewerExperiencePreferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1)
    space: Literal[VectorSpace.VIEWER_EXPERIENCE]
    weight: PreferenceSpaceWeight
    content: ViewerExperienceBody


class WatchContextPreferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1)
    space: Literal[VectorSpace.WATCH_CONTEXT]
    weight: PreferenceSpaceWeight
    content: WatchContextBody


class NarrativeTechniquesPreferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1)
    space: Literal[VectorSpace.NARRATIVE_TECHNIQUES]
    weight: PreferenceSpaceWeight
    content: NarrativeTechniquesBody


class ProductionPreferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1)
    space: Literal[VectorSpace.PRODUCTION]
    weight: PreferenceSpaceWeight
    content: ProductionBody


class ReceptionPreferenceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1)
    space: Literal[VectorSpace.RECEPTION]
    weight: PreferenceSpaceWeight
    content: ReceptionBody


# Plain Union for the same reason as DealbreakerBody above —
# OpenAI Structured Outputs rejects `oneOf` (which
# Field(discriminator=...) produces). The Literal[VectorSpace.X] tag
# on each wrapper plus extra="forbid" keeps dispatch unambiguous.
PreferenceSpaceEntry = Union[
    AnchorPreferenceEntry,
    PlotEventsPreferenceEntry,
    PlotAnalysisPreferenceEntry,
    ViewerExperiencePreferenceEntry,
    WatchContextPreferenceEntry,
    NarrativeTechniquesPreferenceEntry,
    ProductionPreferenceEntry,
    ReceptionPreferenceEntry,
]


# ---------------------------------------------------------------------------
# Top-level preference spec.
# Field order: qualifier_inventory → space_queries.
# space_queries holds 1..8 per-space entries, at most one per space;
# the _no_duplicate_spaces validator guards the Σ(w × cos) / Σw
# downstream sum from double-counting a space.
# LLM-facing guidance on each reasoning field lives in
# search_v2/stage_3/semantic_query_generation.py.
# ---------------------------------------------------------------------------
class SemanticPreferenceSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qualifier_inventory: constr(strip_whitespace=True, min_length=1)
    space_queries: conlist(PreferenceSpaceEntry, min_length=1, max_length=8)

    @model_validator(mode="after")
    def _no_duplicate_spaces(self) -> "SemanticPreferenceSpec":
        # VectorSpace is a str-Enum, so enum members and their string
        # values compare equal and share a hash. One set works for both.
        seen: set = set()
        for entry in self.space_queries:
            if entry.space in seen:
                raise ValueError(
                    f"duplicate space in space_queries: {entry.space}"
                )
            seen.add(entry.space)
        return self
