# Category-handler semantic endpoint structured output model.
#
# One unified shape (SemanticEndpointParameters) covers both the
# dealbreaker and preference paths. Which path runs is determined by
# match_mode on the enclosing EndpointParameters wrapper, not by a
# schema-level split:
#
#   match_mode == FILTER (dealbreaker)
#       Use only the space_queries entry whose .space matches
#       primary_vector. Ignore weight.
#
#   match_mode == TRAIT (preference)
#       Use all space_queries entries with their weights
#       (central/supporting). Ignore primary_vector.
#
# Anchor is skipped in both paths, so the space enum narrows to the
# 7 non-anchor spaces for everyone. This replaces the two previous
# top-level shapes (SemanticDealbreakerSpec + SemanticPreferenceSpec)
# that differed on the anchor-space inclusion and whether a single
# space was selected.
#
# Field generation order inside SemanticParameters is deliberate:
#   qualifier_inventory -> space_queries -> primary_vector
#
# Evidence inventory primes the list; the populated list anchors the
# retrospective single-space pick. Placing primary_vector BEFORE
# space_queries would pressure small models to collapse the list to
# one entry (why list three when you already picked one) — placing
# it LAST reframes it as a retrospective summary judgment over an
# already-populated inventory. See category_handler_planning.md
# ("Unified semantic schema") for the full rationale.
#
# Field(description=...) IS used on every LLM-facing field in this
# module — the category-handler design puts per-field semantics in
# the schema (attention-anchored, single source of truth, token-
# efficient) rather than restating them in the system prompt. This
# is a deliberate departure from the older convention in
# schemas/keyword_translation.py and schemas/award_translation.py
# where guidance lived exclusively in the prompt.

from __future__ import annotations

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.endpoint_parameters import (
    MATCH_MODE_DESCRIPTION,
    POLARITY_DESCRIPTION,
    EndpointParameters,
)
from schemas.enums import MatchMode, Polarity
from schemas.semantic_bodies import (
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
# Seven non-anchor spaces. Anchor (dense_anchor_vectors) is excluded
# from every category-handler semantic call — handlers decompose the
# requirement into per-space reasoning, and the anchor space doesn't
# carry a single dimension a handler could argue for.


class SemanticSpace(str, Enum):
    PLOT_EVENTS = "plot_events"
    PLOT_ANALYSIS = "plot_analysis"
    VIEWER_EXPERIENCE = "viewer_experience"
    WATCH_CONTEXT = "watch_context"
    NARRATIVE_TECHNIQUES = "narrative_techniques"
    PRODUCTION = "production"
    RECEPTION = "reception"


# Per-space weight in the preference path. Ignored in the
# dealbreaker path. Kept as a two-level categorical so the LLM
# commits to a qualitative judgment rather than a free-form score.
class SpaceWeight(str, Enum):
    CENTRAL = "central"
    SUPPORTING = "supporting"


# ---------------------------------------------------------------------------
# Space-entry discriminator wrappers.
#
# One wrapper per space. The Literal[SemanticSpace.X] tag combined
# with ConfigDict(extra="forbid") forces every body to match exactly
# one branch — mixing a NarrativeTechniquesBody-shaped payload with
# space="viewer_experience" is a schema-level error.
#
# Field order inside every entry: carries_qualifiers -> space ->
# weight -> content. Reasoning first (per-space evidence),
# discriminator next, weight next, body last.
#
# Field descriptions are shared module constants below so all 7 entry
# classes stay in lockstep — drifting guidance across spaces would
# bias the model toward whichever space had the freshest wording.
# ---------------------------------------------------------------------------


_CARRIES_QUALIFIERS_DESC = (
    "Per-space reasoning: which specific signals from the requirement "
    "THIS space is capturing. One short sentence grounded in concrete "
    "phrases from the input — not a restatement of what the space is "
    "for in general. When the same atom appears across two entries "
    "(e.g. 'tense' in both plot_events and viewer_experience), name "
    "the distinct contribution each space draws from it — plot_events "
    "carries the stakes/setups, viewer_experience carries the felt "
    "tension."
)

_SPACE_DESC = (
    "The vector space this entry targets. Must match the Body type on "
    "'content'. If a signal honestly spans two spaces, emit two "
    "separate entries rather than picking the 'closest' one."
)

_WEIGHT_DESC = (
    "'central' when this space carries the dominant signal for the "
    "requirement — the single dimension where a non-match should hurt "
    "the score the most. 'supporting' when this space captures a "
    "secondary or compositional aspect that shapes the match but is "
    "not the core ask. A requirement can have multiple central spaces "
    "if two dimensions are equally load-bearing; don't artificially "
    "demote one to supporting just to pick a single central. Only "
    "consulted in the preference (trait) path — ignored when the "
    "enclosing wrapper's match_mode is filter."
)

_CONTENT_DESC = (
    "The structured payload for this space. Populate its fields with "
    "the concrete labels (concept tags, descriptors, or free-form "
    "phrases depending on the Body type) that ground the signals you "
    "named in carries_qualifiers above. This is what gets embedded "
    "and compared against movie-side vectors — include only labels "
    "genuinely supported by the input, never speculative ones just "
    "to fill the field."
)


class PlotEventsEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description=_CARRIES_QUALIFIERS_DESC
    )
    space: Literal[SemanticSpace.PLOT_EVENTS] = Field(..., description=_SPACE_DESC)
    weight: SpaceWeight = Field(..., description=_WEIGHT_DESC)
    content: PlotEventsBody = Field(..., description=_CONTENT_DESC)


class PlotAnalysisEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description=_CARRIES_QUALIFIERS_DESC
    )
    space: Literal[SemanticSpace.PLOT_ANALYSIS] = Field(..., description=_SPACE_DESC)
    weight: SpaceWeight = Field(..., description=_WEIGHT_DESC)
    content: PlotAnalysisBody = Field(..., description=_CONTENT_DESC)


class ViewerExperienceEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description=_CARRIES_QUALIFIERS_DESC
    )
    space: Literal[SemanticSpace.VIEWER_EXPERIENCE] = Field(..., description=_SPACE_DESC)
    weight: SpaceWeight = Field(..., description=_WEIGHT_DESC)
    content: ViewerExperienceBody = Field(..., description=_CONTENT_DESC)


class WatchContextEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description=_CARRIES_QUALIFIERS_DESC
    )
    space: Literal[SemanticSpace.WATCH_CONTEXT] = Field(..., description=_SPACE_DESC)
    weight: SpaceWeight = Field(..., description=_WEIGHT_DESC)
    content: WatchContextBody = Field(..., description=_CONTENT_DESC)


class NarrativeTechniquesEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description=_CARRIES_QUALIFIERS_DESC
    )
    space: Literal[SemanticSpace.NARRATIVE_TECHNIQUES] = Field(..., description=_SPACE_DESC)
    weight: SpaceWeight = Field(..., description=_WEIGHT_DESC)
    content: NarrativeTechniquesBody = Field(..., description=_CONTENT_DESC)


class ProductionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description=_CARRIES_QUALIFIERS_DESC
    )
    space: Literal[SemanticSpace.PRODUCTION] = Field(..., description=_SPACE_DESC)
    weight: SpaceWeight = Field(..., description=_WEIGHT_DESC)
    content: ProductionBody = Field(..., description=_CONTENT_DESC)


class ReceptionEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    carries_qualifiers: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description=_CARRIES_QUALIFIERS_DESC
    )
    space: Literal[SemanticSpace.RECEPTION] = Field(..., description=_SPACE_DESC)
    weight: SpaceWeight = Field(..., description=_WEIGHT_DESC)
    content: ReceptionBody = Field(..., description=_CONTENT_DESC)


# Plain Union (no Field(discriminator=...)) so Pydantic emits an
# `anyOf` JSON-schema node rather than `oneOf`. OpenAI structured
# outputs rejects `oneOf`. The Literal[SemanticSpace.X] tag on each
# wrapper plus extra="forbid" still guarantees that any given body
# matches exactly one branch, so `anyOf` is functionally equivalent.
SemanticSpaceEntry = Union[
    PlotEventsEntry,
    PlotAnalysisEntry,
    ViewerExperienceEntry,
    WatchContextEntry,
    NarrativeTechniquesEntry,
    ProductionEntry,
    ReceptionEntry,
]


# ---------------------------------------------------------------------------
# Merge helpers for duplicate-space recovery.
#
# Both validators below prefer recovering from a malformed LLM emission
# over raising — the alternative is a failed call that costs a retry
# for a problem we can fix deterministically. The helpers walk the
# body schema generically so they stay correct as space bodies evolve.
# ---------------------------------------------------------------------------


def _merge_bodies(bodies: list[BaseModel]) -> BaseModel:
    # Generic merge across any of the seven space body types. List
    # fields are concatenated with first-occurrence dedupe; prose
    # str | None fields are joined with ". "; nested BaseModel fields
    # (TermsSection / TermsWithNegationsSection) recurse so their
    # inner term lists also concat-dedupe.
    merged = bodies[0].model_copy()
    for field_name in type(merged).model_fields:
        values = [getattr(b, field_name) for b in bodies]
        sample = next((v for v in values if v is not None), None)

        if isinstance(sample, list):
            combined: list = []
            seen: set = set()
            for v in values:
                for item in v:
                    if item not in seen:
                        seen.add(item)
                        combined.append(item)
            setattr(merged, field_name, combined)
        elif isinstance(sample, BaseModel):
            non_none = [v for v in values if v is not None]
            setattr(merged, field_name, _merge_bodies(non_none))
        else:
            # Prose str | None — concat non-empty values, preserving
            # the order entries arrived in.
            non_empty = [v for v in values if v]
            setattr(
                merged,
                field_name,
                ". ".join(non_empty) if non_empty else None,
            )
    return merged


def _merge_entries(entries: list) -> "SemanticSpaceEntry":
    # Collapse N entries that share a space into one. carries_qualifiers
    # joins with " | " to keep each duplicate's per-space reasoning
    # visible; weight resolves to CENTRAL if any duplicate is central
    # (the stronger signal wins so the merged entry isn't quietly
    # demoted); content delegates to _merge_bodies.
    if len(entries) == 1:
        return entries[0]

    merged = entries[0].model_copy()
    qualifiers = [e.carries_qualifiers for e in entries if e.carries_qualifiers]
    merged.carries_qualifiers = " | ".join(qualifiers)
    merged.weight = (
        SpaceWeight.CENTRAL
        if any(e.weight == SpaceWeight.CENTRAL for e in entries)
        else SpaceWeight.SUPPORTING
    )
    merged.content = _merge_bodies([e.content for e in entries])
    return merged


# ---------------------------------------------------------------------------
# Inner payload.
#
# Field order: qualifier_inventory -> space_queries -> primary_vector.
# See the top-of-file comment for why primary_vector is last. The
# _deduplicate_spaces validator collapses duplicate-space entries by
# merging their content rather than failing, so the downstream
# Σ(w × cos) / Σw sum doesn't double-count a space; the
# _primary_vector_in_space_queries validator falls back to the entry
# with the most embedding text when the model picks a space it didn't
# populate.
# ---------------------------------------------------------------------------


class SemanticParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qualifier_inventory: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Inventory of the signals the requirement carries, written "
            "BEFORE committing to any specific space. 1-3 short "
            "sentences enumerating the distinguishable atoms in the "
            "input (e.g. 'heist plot premise; tense pacing; morally "
            "grey protagonist') without yet mapping them to vector "
            "spaces. This primes honest decomposition — naming the "
            "atoms first prevents collapsing a multi-axis requirement "
            "into whichever space jumps to mind."
        ),
    )
    space_queries: conlist(SemanticSpaceEntry, min_length=1, max_length=7) = Field(
        ...,
        description=(
            "One entry per vector space whose definition GENUINELY "
            "covers part of the requirement. Populate multiple entries "
            "when multiple spaces apply — this is not a 'pick the best "
            "one' field. A plot-driven preference with tonal "
            "qualifiers should populate both plot_events AND "
            "viewer_experience rather than squeezing tone into the "
            "plot entry. Under-listing is the more common failure mode "
            "than over-listing; err toward including a space when you "
            "can name a specific atom it captures."
        ),
    )
    primary_vector: SemanticSpace = Field(
        ...,
        description=(
            "Among the spaces you populated in space_queries above, "
            "identify the single most effective one — the space whose "
            "match or non-match would most strongly determine whether "
            "a movie satisfies the requirement. Listing multiple "
            "genuinely-applicable spaces above is always correct when "
            "multiple apply; this field collapses to one ONLY for "
            "execution paths that require a single target (hard-gate "
            "dealbreakers). Must be one of the spaces you already "
            "populated above — do NOT introduce a new space here."
        ),
    )

    @model_validator(mode="after")
    def _deduplicate_spaces(self) -> "SemanticParameters":
        # If the LLM emits multiple entries for the same space, merge
        # them into one rather than failing — duplicates are
        # recoverable (concat content, OR the weights, join the
        # qualifier reasoning) and a hard fail would just trigger a
        # retry for a benign collision.
        grouped: dict = {}
        order: list = []
        for entry in self.space_queries:
            if entry.space not in grouped:
                grouped[entry.space] = [entry]
                order.append(entry.space)
            else:
                grouped[entry.space].append(entry)

        if len(order) == len(self.space_queries):
            return self  # no duplicates, nothing to merge

        self.space_queries = [_merge_entries(grouped[s]) for s in order]
        return self

    @model_validator(mode="after")
    def _primary_vector_in_space_queries(self) -> "SemanticParameters":
        # primary_vector must reference a populated space — the
        # dealbreaker path needs an entry to execute. If the model
        # picked a space it didn't populate, fall back to the entry
        # whose content carries the most embedding text (a reasonable
        # proxy for which space the model invested most signal in).
        populated = {entry.space for entry in self.space_queries}
        if self.primary_vector in populated:
            return self

        best = max(
            self.space_queries,
            key=lambda e: len(e.content.embedding_text()),
        )
        self.primary_vector = best.space
        return self


# ---------------------------------------------------------------------------
# EndpointParameters wrapper.
# Same shape as every other endpoint's wrapper.
# ---------------------------------------------------------------------------


# Fields are declared in the order match_mode → parameters →
# polarity so polarity is emitted last. See endpoint_parameters.py
# for the rationale.
class SemanticEndpointParameters(EndpointParameters):
    match_mode: MatchMode = Field(..., description=MATCH_MODE_DESCRIPTION)
    parameters: SemanticParameters = Field(
        ...,
        description=(
            "Semantic endpoint payload. Emit one entry for EVERY "
            "vector space that genuinely covers part of the "
            "requirement — multi-space is the norm, not the exception. "
            "Then retrospectively pick the single most effective space "
            "as primary_vector. Do NOT collapse a multi-axis "
            "requirement into one entry just because the dealbreaker "
            "path will only consume one; populate honestly, collapse "
            "only at the end. Describe the target concept directly "
            "regardless of polarity — negation is handled on the "
            "wrapper's polarity field, never inside these parameters."
        ),
    )
    polarity: Polarity = Field(..., description=POLARITY_DESCRIPTION)
