# Step 3 metadata endpoint LLM structured output models.
#
# Translates natural-language metadata descriptions from step 2 into
# structured query parameters. The field matching target_attribute
# should carry the primary executable payload. Other attribute fields
# usually remain null, but execution ignores them if they are
# populated.
#
# This is a pure translation layer — the LLM converts descriptions
# into structured parameters as faithfully as possible. No softening,
# no gradient logic. Deterministic code in the execution layer handles
# all softening (generous gates for candidate generation, gradient
# decay for scoring). The existing db/metadata_scoring.py gradient
# shapes serve as the primary reference for those decay functions.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 2: Movie Attributes) for the full design rationale.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt. Developer notes live in
# comments above the class.

from __future__ import annotations

from datetime import date
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, conlist, model_validator

from implementation.classes.countries import Country
from implementation.classes.enums import (
    DateMatchOperation,
    MaturityRating,
    NumericalMatchOperation,
    RatingMatchOperation,
    StreamingAccessType,
)
from implementation.classes.languages import Language
from implementation.classes.watch_providers import StreamingService
from schemas.endpoint_parameters import EndpointParameters
from schemas.enums import (
    BoxOfficeStatus,
    BudgetSize,
    MetadataAttribute,
    PopularityMode,
    ReceptionMode,
)


# ── Sub-objects for complex attributes ────────────────────────────


# Release date: literal translation of a temporal constraint.
# "80s movies" → BETWEEN 1980-01-01 / 1989-12-31.
# "Recent movies" → AFTER <concrete date>. The LLM has today's date
# injected into its prompt for resolving relative terms.
class ReleaseDateTranslation(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    first_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    match_operation: DateMatchOperation = Field(...)
    second_date: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")

    @model_validator(mode="after")
    def _validate_dates(self) -> "ReleaseDateTranslation":
        # Validate first_date is a parseable ISO date.
        try:
            first = date.fromisoformat(self.first_date)
        except (TypeError, ValueError):
            raise ValueError(f"first_date is not a valid ISO date: {self.first_date}")

        is_between = self.match_operation in {
            DateMatchOperation.BETWEEN,
            DateMatchOperation.BETWEEN.value,
        }

        if is_between:
            if self.second_date is None:
                raise ValueError("second_date is required when match_operation is BETWEEN")
            try:
                second = date.fromisoformat(self.second_date)
            except (TypeError, ValueError):
                raise ValueError(f"second_date is not a valid ISO date: {self.second_date}")
            # Ensure chronological order.
            if first > second:
                self.first_date, self.second_date = self.second_date, self.first_date
        else:
            # Discard second_date when not BETWEEN.
            self.second_date = None

        return self


# Runtime: literal translation of a duration constraint (in minutes).
class RuntimeTranslation(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    first_value: float = Field(...)
    match_operation: NumericalMatchOperation = Field(...)
    second_value: Optional[float] = Field(default=None)

    @model_validator(mode="after")
    def _validate_values(self) -> "RuntimeTranslation":
        is_between = self.match_operation in {
            NumericalMatchOperation.BETWEEN,
            NumericalMatchOperation.BETWEEN.value,
        }

        if is_between:
            if self.second_value is None:
                raise ValueError("second_value is required when match_operation is BETWEEN")
            # Ensure ascending order.
            if self.first_value > self.second_value:
                self.first_value, self.second_value = self.second_value, self.first_value
        else:
            self.second_value = None

        return self


# Maturity rating: target rating + directional comparison.
# Any query targeting a rated value (anything other than EXACT UNRATED)
# excludes UNRATED movies entirely — execution code enforces this.
class MaturityRatingTranslation(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    rating: MaturityRating = Field(...)
    match_operation: RatingMatchOperation = Field(...)


# Streaming availability: services and/or access method.
# Inclusion-only — no exclusion list. Exclusion handling is supplied
# by the enclosing EndpointParameters wrapper's polarity field;
# execution code applies downrank/exclusion scoring downstream.
# At least one of services or preferred_access_type must be populated.
class StreamingTranslation(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    services: list[StreamingService] = Field(default=[])
    preferred_access_type: Optional[StreamingAccessType] = Field(default=None)

    @model_validator(mode="after")
    def _validate_has_constraint(self) -> "StreamingTranslation":
        if not self.services and self.preferred_access_type is None:
            raise ValueError(
                "At least one of services or preferred_access_type must be populated"
            )
        return self


# Audio language: explicit audio-track constraint only.
# ONLY used when the user explicitly mentions audio/language/dubbed.
# "French films" → country_of_origin, NOT audio_language.
# "Foreign films" → country_of_origin, NOT audio_language.
class AudioLanguageTranslation(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    languages: conlist(Language, min_length=1) = Field(...)


# Country of origin: one or more target countries.
# The LLM uses parametric knowledge to expand region-level terms
# ("European movies") into concrete country lists. Execution code
# applies a position-based gradient on the country_of_origin_ids
# array (position 1 = 1.0, position 2 = ~0.7-0.8, position 3+ =
# rapid decay). When multiple countries are requested, the movie's
# score is the max across all requested countries.
class CountryOfOriginTranslation(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    countries: conlist(Country, min_length=1) = Field(...)


# ── Top-level output ──────────────────────────────────────────────


# Complete step 3 metadata endpoint output. Each field corresponds
# to one searchable metadata attribute. The LLM identifies one
# primary target_attribute and should populate that field first.
# Other fields usually remain null, but execution ignores them if
# they are present.
#
# Field ordering follows cognitive-scaffolding progression:
#   1. constraint_phrases  — evidence inventory (grounds routing)
#   2. target_attribute    — single-column routing decision
#   3. value_intent_label  — brief literal-meaning label (primes sub-object)
#   4. attribute sub-objects (target one first; extras ignored)
#
# Execution code queries ONLY the column identified by
# target_attribute for candidate generation (dealbreakers) and
# scoring (preferences). The LLM may still populate other attribute
# fields for context, but only the target drives the query. This
# guarantees one metadata item = one column query = one [0, 1] score.
#
# The dealbreaker/preference classification is not repeated here —
# it flows through action_role + polarity on the enclosing
# EndpointParameters wrapper (MetadataEndpointParameters). This
# class itself stays direction-agnostic.
class MetadataTranslationOutput(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # Evidence inventory. Populated BEFORE target_attribute so that
    # routing is grounded in actual input text rather than pattern
    # matching. Distinguishes near-collisions like "French films"
    # (country signal) vs "French audio" (language signal).
    constraint_phrases: list[str] = Field(default=[])

    target_attribute: MetadataAttribute = Field(...)

    # Brief literal-meaning label. Populated BETWEEN target_attribute
    # and the sub-objects to prime match_operation direction, numeric
    # boundary choice, and list-expansion completeness for the sub-
    # object that follows. Label form only — not a sentence, not a
    # justification, not a restatement of sub-object values.
    value_intent_label: str = Field(..., max_length=80)

    release_date: Optional[ReleaseDateTranslation] = Field(default=None)
    runtime: Optional[RuntimeTranslation] = Field(default=None)
    maturity_rating: Optional[MaturityRatingTranslation] = Field(default=None)
    streaming: Optional[StreamingTranslation] = Field(default=None)
    audio_language: Optional[AudioLanguageTranslation] = Field(default=None)
    country_of_origin: Optional[CountryOfOriginTranslation] = Field(default=None)
    budget_scale: Optional[BudgetSize] = Field(default=None)
    box_office: Optional[BoxOfficeStatus] = Field(default=None)
    popularity: Optional[PopularityMode] = Field(default=None)
    reception: Optional[ReceptionMode] = Field(default=None)


# Category-handler wrapper. Direction flows through action_role +
# polarity on the wrapper; MetadataTranslationOutput itself stays
# direction-agnostic.
class MetadataEndpointParameters(EndpointParameters):
    parameters: MetadataTranslationOutput = Field(
        ...,
        description=(
            "Metadata endpoint payload. Pick the single "
            "target_attribute the requirement targets (release_date, "
            "runtime, maturity_rating, streaming, audio_language, "
            "country_of_origin, budget_scale, box_office, popularity, "
            "or reception), then populate ONLY that attribute's sub-"
            "object. Do NOT split a single requirement across two "
            "attribute fields — if it seems to span two columns, the "
            "requirement should have been decomposed upstream, not "
            "merged here."
        ),
    )
