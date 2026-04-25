from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, constr


class ExplicitExpectationSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_span: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "The exact Step-2 requirement fragment this row is about. Copy the "
            "fragment's query_text verbatim rather than paraphrasing or merging "
            "multiple fragments into one row."
        ),
    )
    normalized_description: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One short phrase restating what this fragment is asking for before "
            "classification. Ground it in the fragment's meaning and modifiers. "
            "Use this field to name the request first, then classify it below."
        ),
    )
    explicit_axis: Literal["quality", "notability", "both", "neither"] = Field(
        ...,
        description=(
            "'quality' only when the fragment explicitly talks about goodness, "
            "badness, acclaim, reception, or a quality direction. "
            "'notability' only when the fragment explicitly talks about fame, "
            "obscurity, mainstreamness, hiddenness, or how well-known the "
            "results should be. 'both' when one fragment explicitly carries "
            "both axes at once (for example, language like 'classics', 'best', "
            "or 'hidden gems'). 'neither' for every remaining fragment, "
            "including genre, tone, entity constraints, and explicit ordering "
            "language like 'trending', 'most recent', or 'scariest' that should "
            "instead be discussed in the top-level ordering-analysis field."
        ),
    )


class ImplicitExpectationsResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_intent_summary: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One or two sentences summarizing what the query is trying to rank "
            "for overall. Mention any explicit ordering language when present."
        ),
    )
    explicit_signals: list[ExplicitExpectationSignal] = Field(
        ...,
        description=(
            "Exactly one row per Step-2 requirement fragment, in the same order "
            "the fragments were provided. Do not drop fragments just because they "
            "do not speak to quality or notability; those rows should be "
            "classified as 'neither'."
        ),
    )
    explicit_ordering_axis_analysis: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "A short analysis of whether the query explicitly asks to rank or "
            "order results by something other than quality/notability. Name the "
            "axis plainly when present (for example: trending, chronology, or a "
            "semantic extremeness axis like scariest/funniest/most disturbing). "
            "If no such axis is present, say that directly."
        ),
    )
    explicitly_addresses_quality: bool = Field(
        ...,
        description=(
            "True iff at least one explicit_signals row is 'quality' or 'both'."
        ),
    )
    explicitly_addresses_notability: bool = Field(
        ...,
        description=(
            "True iff at least one explicit_signals row is 'notability' or "
            "'both'."
        ),
    )
    should_apply_quality_prior: bool = Field(
        ...,
        description=(
            "Whether the system should still apply an implicit quality prior. "
            "This must be false whenever quality is already explicit, and should "
            "also be false when some other explicit ordering axis should take "
            "precedence over the default quality prior."
        ),
    )
    should_apply_notability_prior: bool = Field(
        ...,
        description=(
            "Whether the system should still apply an implicit notability prior. "
            "This must be false whenever notability is already explicit, and "
            "should also be false when some other explicit ordering axis should "
            "take precedence over the default notability prior."
        ),
    )

