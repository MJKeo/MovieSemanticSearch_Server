# Metadata endpoint LLM structured output models.
#
# Whole-call translation: the LLM looks at retrieval_intent and every
# expression on a single CategoryCall as one coherent picture, then
# commits to a single ColumnSpec where each of the ten attribute
# columns is either populated with a literal sub-object or explicit
# null. Multi-column composition is committed via scoring_method
# (ANY | ALL) — the executor reads the populated fields, scores
# each independently, then folds per-column scores into one per-movie
# call score using scoring_method.
#
# Field ordering encodes the cognitive scaffold:
#   1. search_picture     — holistic intent restatement
#   2. column_candidates  — honest per-column audit (covers / misses)
#   3. scoring_method_reasoning — how the eligible columns relate
#   4. column_spec        — literal commitment, every column explicit
#   5. scoring_method     — mechanical mapping of scoring_method_reasoning
#
# Schema = micro-prompts; the system prompt is procedural and does
# not duplicate field-shape rules. Per-sub-object literal-translation
# rules (date math, runtime comparators, rating-scale direction,
# streaming-services tracked set, etc.) live on the unchanged
# sub-object types and on the system prompt.

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
    ScoringMethod,
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
        # first_date is load-bearing — there is no graceful recovery
        # if it fails to parse, so this remains a hard error.
        try:
            first = date.fromisoformat(self.first_date)
        except (TypeError, ValueError):
            raise ValueError(f"first_date is not a valid ISO date: {self.first_date}")

        is_between = self.match_operation in {
            DateMatchOperation.BETWEEN,
            DateMatchOperation.BETWEEN.value,
        }

        if not is_between:
            # Non-BETWEEN ops only consume first_date.
            self.second_date = None
            return self

        # BETWEEN: if second_date is missing or unparseable, downgrade
        # match_operation to EXACT against first_date alone rather than
        # failing the whole metadata call. The user gets a single-day
        # match instead of a range, which is a strictly weaker signal
        # but preserves the rest of the query.
        if self.second_date is None:
            self.match_operation = DateMatchOperation.EXACT.value
            return self

        try:
            second = date.fromisoformat(self.second_date)
        except (TypeError, ValueError):
            self.match_operation = DateMatchOperation.EXACT.value
            self.second_date = None
            return self

        # Both dates valid — preserve chronological order.
        if first > second:
            self.first_date, self.second_date = self.second_date, self.first_date
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

        if not is_between:
            self.second_value = None
            return self

        # BETWEEN with a missing second_value: downgrade to EXACT
        # against first_value alone rather than failing the whole
        # metadata call. The user gets a single-value match instead of
        # a range, which is a strictly weaker signal but preserves the
        # rest of the query.
        if self.second_value is None:
            self.match_operation = NumericalMatchOperation.EXACT.value
            return self

        # Ensure ascending order when both are present.
        if self.first_value > self.second_value:
            self.first_value, self.second_value = self.second_value, self.first_value
        return self


# Maturity rating: target rating + directional comparison.
# Any query targeting a rated value (anything other than EXACT UNRATED)
# excludes UNRATED movies entirely — execution code enforces this.
class MaturityRatingTranslation(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    rating: MaturityRating = Field(...)
    match_operation: RatingMatchOperation = Field(...)


# Streaming availability: services and/or access method.
# Inclusion-only. At least one of services or preferred_access_type
# must be populated.
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
class AudioLanguageTranslation(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    languages: conlist(Language, min_length=1) = Field(...)


# Country of origin: one or more target countries.
# The LLM uses parametric knowledge to expand region-level terms
# ("European movies") into concrete country lists. Execution code
# applies a position-based gradient on the country_of_origin_ids
# array.
class CountryOfOriginTranslation(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    countries: conlist(Country, min_length=1) = Field(...)


# ── Audit layer ───────────────────────────────────────────────────


# Per-column coverage analysis. Surfaces adjacency honestly so the
# commit step is grounded rather than back-rationalized.
class ColumnCandidate(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    column: MetadataAttribute = Field(
        ...,
        description="The structured-attribute column under audit.",
    )
    what_this_covers: str = Field(
        ...,
        description=(
            "One sentence naming the aspect of search_picture this "
            "column genuinely owns. Cite the specific intent fragment.\n"
            "NEVER back-rationalize. NEVER generalize from the "
            "column's default purpose."
        ),
    )
    what_this_misses: str = Field(
        ...,
        description=(
            "Aspect of search_picture this column does NOT cover, "
            "naming the column that owns the gap. \"Nothing\" when fit "
            "is clean and no adjacent column competes.\n"
            "NEVER hedge without naming. NEVER invent gaps."
        ),
    )


# ── Commit layer ──────────────────────────────────────────────────


# Literal sub-object per attribute column. Each field is required-but-
# nullable so the LLM commits an explicit decision per column rather
# than skipping. Population rules (cross-reference to search_picture
# and column_candidates, minimum-span discipline) live on the parent
# MetadataTranslationOutput.column_spec field — read that description
# before populating any field here.
class ColumnSpec(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    release_date: Optional[ReleaseDateTranslation] = Field(
        ...,
        description="Release-date sub-object or null.",
    )
    runtime: Optional[RuntimeTranslation] = Field(
        ...,
        description="Runtime sub-object or null.",
    )
    maturity_rating: Optional[MaturityRatingTranslation] = Field(
        ...,
        description="Maturity-rating sub-object or null.",
    )
    streaming: Optional[StreamingTranslation] = Field(
        ...,
        description="Streaming-availability sub-object or null.",
    )
    audio_language: Optional[AudioLanguageTranslation] = Field(
        ...,
        description=(
            "Audio-track sub-object or null. NEVER infer from country "
            "or cultural identity — that signal routes to "
            "country_of_origin."
        ),
    )
    country_of_origin: Optional[CountryOfOriginTranslation] = Field(
        ...,
        description="Country-of-origin sub-object or null.",
    )
    budget_scale: Optional[BudgetSize] = Field(
        ...,
        description="Budget-scale enum or null.",
    )
    box_office: Optional[BoxOfficeStatus] = Field(
        ...,
        description="Box-office enum or null.",
    )
    popularity: Optional[PopularityMode] = Field(
        ...,
        description="Popularity enum or null.",
    )
    reception: Optional[ReceptionMode] = Field(
        ...,
        description="Reception enum or null.",
    )


# ── Top-level output ──────────────────────────────────────────────


class MetadataTranslationOutput(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    search_picture: str = Field(
        ...,
        description=(
            "1-2 sentences. Restate retrieval_intent and every "
            "expression on this CategoryCall as ONE coherent picture "
            "— what kind of movie the call wants, taken as a whole.\n"
            "NEVER paraphrase retrieval_intent verbatim. NEVER "
            "enumerate expressions literally. NEVER name columns."
        ),
    )
    column_candidates: list[ColumnCandidate] = Field(
        ...,
        description=(
            "Honest audit of columns plausible for search_picture, "
            "with per-column what_this_covers / what_this_misses. "
            "Surface adjacency where it genuinely competes; drop "
            "columns whose only contribution is being adjacent.\n"
            "Local test: \"if I removed this candidate, would the "
            "commit step lose a real option?\" Padding → drop.\n"
            "NEVER list every column out of habit. NEVER duplicate "
            "columns."
        ),
    )
    scoring_method_reasoning: str = Field(
        ...,
        description=(
            "1 sentence. Project forward from column_candidates: of "
            "the columns you intend to populate in column_spec, do "
            "they read as SUBSTITUTABLE signals of one concept "
            "(any-one-matching qualifies) or REINFORCING facets "
            "(every populated column contributes)? Write \"single "
            "column\" when only one column will be populated.\n"
            "Justifies scoring_method below."
        ),
    )
    column_spec: ColumnSpec = Field(
        ...,
        description=(
            "Literal commitment. Populate ONLY columns surfaced in "
            "column_candidates with substantive what_this_covers; "
            "explicit null elsewhere. Apply minimum span — null a "
            "column whose search_picture aspect is fully covered by "
            "another populated column. Same-column expressions merge "
            "into ONE populated sub-object (country lists union, "
            "runtime ranges reconcile, streaming services + access "
            "pair).\n"
            "Local test per column: \"if I null this, does "
            "search_picture lose real intent?\" If no, null it.\n"
            "NEVER populate a column absent from column_candidates. "
            "NEVER split same-column expressions across multiple "
            "fields."
        ),
    )
    scoring_method: ScoringMethod = Field(
        ...,
        description=(
            "Mechanical commit of scoring_method_reasoning above. ANY when "
            "reasoning says substitutable: we only care if the movie "
            "has at least one populated column match, and movies score "
            "equally high for matching 1+ values. ALL when reasoning "
            "says \"single column\" or reinforcing: we care how many "
            "populated columns the movie matches, and movies score "
            "higher depending on how many values they match.\n"
            "NEVER re-derive from search_picture or column_spec — "
            "read off scoring_method_reasoning."
        ),
    )


# Category-handler wrapper. Direction flows through role + polarity
# on the wrapper; MetadataTranslationOutput stays direction-agnostic.
# Open question: whether this wrapper survives the migration to the
# new Step 3 contract — see the metadata endpoint generator/executor
# notes.
class MetadataEndpointParameters(EndpointParameters):
    parameters: MetadataTranslationOutput = Field(
        ...,
        description=(
            "Metadata endpoint payload. Whole-call translation: "
            "retrieval_intent + expressions resolved into one "
            "ColumnSpec where each of the ten attribute columns is "
            "either a populated sub-object or explicit null, plus a "
            "scoring_method controlling multi-column composition. "
            "Describe the target concept directly regardless of "
            "polarity — negation is handled on the wrapper's polarity "
            "field, never inside these parameters."
        ),
    )


# ── Subintent variant ─────────────────────────────────────────────
#
# Used when a single category routes to multiple endpoints and the
# upstream responsibility-splitting reasoning has already carved out
# this endpoint's slice of intent into a dedicated field. Same
# structure as MetadataTranslationOutput, but `search_picture` (the
# only field that originally read the raw call inputs) reads from
# `metadata_retrieval_intent` instead. Every other field already
# reads off downstream artifacts (search_picture / column_candidates
# / scoring_method_reasoning) and so survives unchanged.


class MetadataTranslationOutputSubintent(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # `metadata_retrieval_intent` is declared on the parent
    # MetadataEndpointSubintentParameters wrapper — it is generated
    # before this spec and every field below reads from it.

    search_picture: str = Field(
        ...,
        description=(
            "1-2 sentences. Restate `metadata_retrieval_intent` as ONE "
            "coherent picture — what kind of movie the intent wants, "
            "taken as a whole.\n"
            "NEVER paraphrase `metadata_retrieval_intent` verbatim. "
            "NEVER name columns."
        ),
    )
    column_candidates: list[ColumnCandidate] = Field(
        ...,
        description=(
            "Honest audit of columns plausible for search_picture, "
            "with per-column what_this_covers / what_this_misses. "
            "Surface adjacency where it genuinely competes; drop "
            "columns whose only contribution is being adjacent.\n"
            "Local test: \"if I removed this candidate, would the "
            "commit step lose a real option?\" Padding → drop.\n"
            "NEVER list every column out of habit. NEVER duplicate "
            "columns."
        ),
    )
    scoring_method_reasoning: str = Field(
        ...,
        description=(
            "1 sentence. Project forward from column_candidates: of "
            "the columns you intend to populate in column_spec, do "
            "they read as SUBSTITUTABLE signals of one concept "
            "(any-one-matching qualifies) or REINFORCING facets "
            "(every populated column contributes)? Write \"single "
            "column\" when only one column will be populated.\n"
            "Justifies scoring_method below."
        ),
    )
    column_spec: ColumnSpec = Field(
        ...,
        description=(
            "Literal commitment. Populate ONLY columns surfaced in "
            "column_candidates with substantive what_this_covers; "
            "explicit null elsewhere. Apply minimum span — null a "
            "column whose search_picture aspect is fully covered by "
            "another populated column. Same-column intent merges "
            "into ONE populated sub-object (country lists union, "
            "runtime ranges reconcile, streaming services + access "
            "pair).\n"
            "Local test per column: \"if I null this, does "
            "search_picture lose real intent?\" If no, null it.\n"
            "NEVER populate a column absent from column_candidates. "
            "NEVER split same-column intent across multiple fields."
        ),
    )
    scoring_method: ScoringMethod = Field(
        ...,
        description=(
            "Mechanical commit of scoring_method_reasoning above. ANY when "
            "reasoning says substitutable: we only care if the movie "
            "has at least one populated column match, and movies score "
            "equally high for matching 1+ values. ALL when reasoning "
            "says \"single column\" or reinforcing: we care how many "
            "populated columns the movie matches, and movies score "
            "higher depending on how many values they match.\n"
            "NEVER re-derive from search_picture or column_spec — "
            "read off scoring_method_reasoning."
        ),
    )


class MetadataEndpointSubintentParameters(EndpointParameters):
    metadata_retrieval_intent: str = Field(
        ...,
        description=(
            "What this endpoint specifically needs to be responsible "
            "for fetching. Read off the prior reasoning fields above "
            "that split up responsibilities across endpoints, capturing "
            "only the slice of intent assigned to the metadata endpoint "
            "(the ten structured-attribute columns: release_date, "
            "runtime, maturity_rating, streaming, audio_language, "
            "country_of_origin, budget_scale, box_office, popularity, "
            "reception). Leave categorical classification, named "
            "entities, free-form thematic qualifiers, awards, and "
            "franchise structure to their respective endpoints. Every "
            "field on this endpoint's `parameters` reads from this "
            "intent rather than from any other upstream input."
        ),
    )
    parameters: MetadataTranslationOutputSubintent = Field(
        ...,
        description=(
            "Metadata endpoint payload, sourced from "
            "`metadata_retrieval_intent` rather than from the raw "
            "call inputs."
        ),
    )
