from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, constr


PriorAxis = Literal["quality", "popularity", "both", "neither"]
PriorDirection = Literal["none", "positive", "inverse"]
PriorStrength = Literal["none", "light", "normal", "strong"]
ExplicitCoverage = Literal["none", "partial", "direct"]
PriorRoom = Literal["none", "light", "normal", "high"]
TraitPressure = Literal["low", "medium", "high"]
OrderingAxis = Literal[
    "none",
    "quality",
    "popularity",
    "trending",
    "chronology",
    "semantic_extremeness",
    "obscurity",
    "other",
]


class ExplicitPriorSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_span: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Exact Step-2 trait surface_text this row analyzes. Copy the trait "
            "surface_text verbatim and preserve Step-2 trait order. Do not merge "
            "multiple traits into one row or invent rows from the raw query."
        ),
    )
    normalized_description: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Short phrase describing what this committed trait asks for after "
            "its evaluative_intent, qualifier_relation, polarity, and commitment "
            "are considered. This field names the request before classification."
        ),
    )
    explicit_axis: PriorAxis = Field(
        ...,
        description=(
            "Which implicit-prior axis this trait explicitly addresses. Use "
            "quality for goodness, badness, reception, acclaim, taste/value "
            "judgment, or watch-worthiness. Use popularity for fame, mainstream "
            "reach, obscurity, hiddenness, cultural familiarity, or how well-known "
            "the results should be. Use both only when this trait itself bundles "
            "both axes. Use neither for ordinary content, tone, era, entity, "
            "availability, occasion, format, or style traits."
        ),
    )
    direction: PriorDirection = Field(
        ...,
        description=(
            "Direction of the explicit signal on its axis. positive means the "
            "trait wants more of the named axis. inverse means the trait wants "
            "less of the named axis, such as lower mainstream popularity or "
            "lower conventional quality/status. none when explicit_axis is "
            "neither."
        ),
    )
    strength_evidence: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Evidence for how strongly this trait covers the prior axis. Ground "
            "the explanation in Step-2 fields: surface_text, evaluative_intent, "
            "qualifier_relation, polarity, commitment, and commitment_evidence. "
            "Say whether coverage is direct, partial, adjacent, or absent."
        ),
    )
    coverage: ExplicitCoverage = Field(
        ...,
        description=(
            "direct when the trait plainly names the prior axis and should fully "
            "replace an implicit prior on that axis; partial when it shades into "
            "the axis but leaves room for a default expectation; none when the "
            "trait does not cover quality or popularity."
        ),
    )


class OrderingAxisAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    primary_ordering_axis: OrderingAxis = Field(
        ...,
        description=(
            "The main explicit ranking axis requested by the committed traits, "
            "if one exists. none means the query gives constraints or preferences "
            "but no explicit ordering axis."
        ),
    )
    axis_description: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Plain-language explanation of the ordering axis and which Step-2 "
            "trait(s) establish it. If no axis exists, state that no committed "
            "trait asks for a separate ordering objective."
        ),
    )
    suppresses_quality_prior: bool = Field(
        ...,
        description=(
            "Whether the ordering axis should prevent generic quality from "
            "meaningfully influencing rank. True only when quality would fight "
            "the requested ordering objective rather than merely break ties."
        ),
    )
    suppresses_popularity_prior: bool = Field(
        ...,
        description=(
            "Whether the ordering axis should prevent generic popularity from "
            "meaningfully influencing rank. True only when popularity would fight "
            "the requested ordering objective rather than merely break ties."
        ),
    )


class QuerySpecificityAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    specificity_summary: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Short explanation of how broad or constrained the query is after "
            "Step-2 committed traits are considered. Discuss the number of "
            "traits, their specificity, and their commitment levels."
        ),
    )
    explicit_trait_pressure: TraitPressure = Field(
        ...,
        description=(
            "How much the explicit traits should dominate ranking. Use high when "
            "multiple required/elevated or highly specific traits leave little "
            "room for defaults; low when the query is broad and underspecified."
        ),
    )
    prior_room: PriorRoom = Field(
        ...,
        description=(
            "How much room remains for implicit quality/popularity priors after "
            "explicit traits and ordering axes have spoken. This is not a trait "
            "weight; it is the allowed strength envelope for hidden priors."
        ),
    )


class PriorDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    direction: PriorDirection = Field(
        ...,
        description=(
            "Final direction for this prior axis. none means no prior should be "
            "applied. positive means boost higher values. inverse means prefer "
            "lower values on the axis."
        ),
    )
    strength: PriorStrength = Field(
        ...,
        description=(
            "Final strength consumed by scoring. Must be consistent with explicit "
            "coverage, ordering-axis suppression, and prior_room."
        ),
    )
    rationale: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Concise explanation tying the decision to explicit_signals, "
            "ordering_axis_analysis, and query_specificity_analysis. Do not cite "
            "raw-query details that are not represented in Step-2 traits."
        ),
    )


class ImplicitExpectationsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_intent_summary: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One or two sentences summarizing the query's ranking intent from "
            "Step-2 intent_exploration and committed traits. Mention whether it "
            "is broad discovery, constrained matching, explicit ranking, or an "
            "obscurity/status request."
        ),
    )
    explicit_signals: list[ExplicitPriorSignal] = Field(
        ...,
        description=(
            "Exactly one row per Step-2 committed trait, in source order. This "
            "is fragment-level evidence before final prior decisions."
        ),
    )
    ordering_axis_analysis: OrderingAxisAnalysis = Field(
        ...,
        description=(
            "Whole-query analysis of explicit ranking axes that may compete with "
            "generic quality/popularity priors."
        ),
    )
    query_specificity_analysis: QuerySpecificityAnalysis = Field(
        ...,
        description=(
            "Whole-query analysis of how much room remains for implicit priors."
        ),
    )
    quality_prior: PriorDecision = Field(
        ...,
        description="Final quality-prior decision.",
    )
    popularity_prior: PriorDecision = Field(
        ...,
        description="Final popularity-prior decision.",
    )
