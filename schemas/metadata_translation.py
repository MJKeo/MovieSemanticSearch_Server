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
# Two shapes coexist in this file:
#
# 1. SINGLE-ENDPOINT shape (MetadataTranslationOutput /
#    MetadataEndpointParameters) — used by buckets 3/4 when metadata
#    owns the entire call. Field ordering encodes the cognitive
#    scaffold: search_picture → column_candidates →
#    scoring_method_reasoning → column_spec → scoring_method.
#    Untouched by the walk-then-commit refactor.
#
# 2. MULTI-ENDPOINT shape (MetadataWalk +
#    MetadataTranslationOutputSubintent /
#    MetadataEndpointSubintentParameters) — used by buckets 5/6/8.
#    Split across two classes that live in different positions of the
#    bucket schema:
#       a. MetadataWalk — the column-grounded analysis layer
#          (`column_candidates`). Lives at the bucket level, emitted
#          BEFORE the coverage_assignments commitment so the LLM walks
#          the columns concretely before committing.
#       b. MetadataTranslationOutputSubintent — thin commitment-only
#          spec (`scoring_method_reasoning` + `column_spec` +
#          `scoring_method`). Lives inside the per-endpoint
#          metadata_parameters slot, populated only when
#          coverage_assignments delegates a slice to this endpoint.
#    `search_picture` does NOT survive into the multi-endpoint shape:
#    the bucket-level `coverage_assignments[kind=metadata].slice_description`
#    plus `metadata_retrieval_intent` already supply the holistic
#    restatement of intent.
#
# Schema = micro-prompts; the system prompt is procedural and does
# not duplicate field-shape rules. Per-sub-object literal-translation
# rules (date math, runtime comparators, rating-scale direction,
# streaming-services tracked set, etc.) live on the unchanged
# sub-object types and on the system prompt.

from __future__ import annotations

from datetime import date
from typing import Annotated, Callable, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    conlist,
    field_validator,
    model_validator,
)

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


# ── ISO-code field types and validation helpers ───────────────────
#
# The LLM-facing schema speaks ISO codes for languages and countries
# (see AudioLanguageTranslation / CountryOfOriginTranslation below for
# the rationale — OpenAI's 1000-enum-per-schema cap forced us off the
# full 334+262 enum surface). The Annotated types pin a strict
# canonical-case `pattern` into the JSON schema sent to OpenAI, so the
# structured-output decoder is constrained to emit lowercase 2-letter
# language codes and uppercase 2-letter country codes. The validators
# below then only need to verify each pattern-conformant code resolves
# to a real registry member.

_LangISOCode = Annotated[str, StringConstraints(pattern=r"^[a-z]{2}$")]
_CountryISOCode = Annotated[str, StringConstraints(pattern=r"^[A-Z]{2}$")]


def _validate_iso_codes(
    codes: list[str],
    from_iso: Callable[[str], Optional[object]],
    kind: str,
) -> list[str]:
    # Validates that each ISO code resolves to a real registry member.
    # Pattern-level format validation already ran upstream via
    # StringConstraints, so unknown-code rejection is the only job here.
    # Unknown codes raise ValueError → ValidationError, which the
    # handler retry path treats as a parse failure and reissues the
    # LLM call.
    for code in codes:
        if from_iso(code) is None:
            raise ValueError(
                f"Unknown ISO {kind} code: {code!r}"
            )
    return codes


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
#
# The LLM emits ISO 639-1 codes (lowercase 2-letter, e.g. "en", "fr",
# "ja"). Codes are stored as-is on the model and resolved to Language
# enum members at the executor edge via Language.from_iso(). We use
# ISO codes rather than the full enum-as-Literal because OpenAI
# structured outputs cap a schema at 1000 enum values; the 334-member
# Language enum was eating ~half the budget on its own. Long-tail
# languages without an ISO 639-1 code are unreachable here by design;
# omit the column when the requested audio language has no code rather
# than picking a near-match (audio language is an explicit constraint,
# so a wrong match is worse than no match).
class AudioLanguageTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    languages: conlist(_LangISOCode, min_length=1) = Field(
        ...,
        description=(
            "ISO 639-1 language codes (lowercase, 2 letters). Use 'en' "
            "not 'eng' or 'English', 'fr' not 'fre' or 'French'. When "
            "the requested audio language has no ISO 639-1 code, omit "
            "the column entirely — do not substitute a near-match, "
            "since audio language is an explicit user constraint."
        ),
    )

    @field_validator("languages", mode="after")
    @classmethod
    def _check_iso_codes(cls, value: list[str]) -> list[str]:
        return _validate_iso_codes(value, Language.from_iso, "639-1 language")


# Country of origin: one or more target countries.
# The LLM uses parametric knowledge to expand region-level terms
# ("European movies") into concrete country lists. Execution code
# applies a position-based gradient on the country_of_origin_ids
# array.
#
# Same ISO-code contract as AudioLanguageTranslation: LLM emits ISO
# 3166-1 alpha-2 codes (uppercase, 2 letters), resolved to Country
# enum at the executor edge via Country.from_iso().
class CountryOfOriginTranslation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    countries: conlist(_CountryISOCode, min_length=1) = Field(
        ...,
        description=(
            "ISO 3166-1 alpha-2 country codes (uppercase, 2 letters). "
            "Use 'US' not 'USA' or 'United States', 'GB' not 'UK', "
            "'KR' for South Korea, 'KP' for North Korea. Retired entities "
            "(Yugoslavia, USSR, Czechoslovakia) are unreachable via this "
            "field — pick the closest current country or omit the column."
        ),
    )

    @field_validator("countries", mode="after")
    @classmethod
    def _check_iso_codes(cls, value: list[str]) -> list[str]:
        return _validate_iso_codes(value, Country.from_iso, "3166-1 alpha-2 country")


# ── Audit layer ───────────────────────────────────────────────────


# Per-column coverage analysis. Surfaces adjacency honestly so the
# commit step is grounded rather than back-rationalized. Reused by
# both the single-endpoint MetadataTranslationOutput.column_candidates
# (where the prior search_picture field is the holistic intent
# restatement) and the multi-endpoint MetadataWalk.column_candidates
# (where the call's retrieval_intent in the user message plays that
# role) — descriptions reference "the call's intent" generically so
# the same class works in both positions.
class ColumnCandidate(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    column: MetadataAttribute = Field(
        ...,
        description="The structured-attribute column under audit.",
    )
    strengths: str = Field(
        ...,
        description=(
            "What aspect of the call's intent this column genuinely "
            "OWNS at retrieval time. Cite the specific intent fragment.\n"
            "NEVER back-rationalize. NEVER generalize from the "
            "column's default purpose."
        ),
    )
    weaknesses: str = Field(
        ...,
        description=(
            "What this column MISSES or OVER-PULLS relative to the "
            "call's intent. Two failure modes — both belong here:\n"
            "- under-coverage: aspects of the intent this column does "
            "NOT carry, naming the column or endpoint that owns the "
            "gap.\n"
            "- over-coverage: rows this column would ALSO match beyond "
            "the slice (e.g. a country list whose region pull is "
            "broader than the cultural-tradition slice).\n"
            "\n"
            "Suggested vocabulary (not enforced): prefix lines with "
            "'under-coverage:' and 'over-coverage:'. Use 'none' only "
            "when the column is a clean fit on both axes.\n"
            "NEVER hedge without naming. NEVER invent weaknesses."
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
            "with per-column strengths + weaknesses. Surface adjacency "
            "where it genuinely competes; drop columns whose only "
            "contribution is being adjacent.\n"
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
            "column_candidates with substantive strengths; explicit "
            "null elsewhere. Apply minimum span — null a column whose "
            "search_picture aspect is fully covered by another "
            "populated column. Same-column expressions merge into ONE "
            "populated sub-object (country lists union, runtime ranges "
            "reconcile, streaming services + access pair).\n"
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


# ── Multi-endpoint walk + thin commitment ─────────────────────────
#
# Used when the metadata endpoint is one of several contending for
# the call's intent (buckets 5/6/8). Two classes that live in
# DIFFERENT positions of the bucket schema:
#
#   - MetadataWalk lives at the bucket level, emitted BEFORE the
#     coverage_assignments commitment phase. It carries the
#     column-grounded audit of how the metadata endpoint could cover
#     the call's retrieval_intent. Forces concrete column awareness
#     before any commitment.
#   - MetadataTranslationOutputSubintent lives inside the per-endpoint
#     metadata_parameters slot, populated only when
#     coverage_assignments delegates a slice to this endpoint.
#     Carries just the commitment layer — scoring_method_reasoning +
#     column_spec + scoring_method.
#
# `search_picture` is dropped: in the multi-endpoint flow,
# `coverage_assignments[kind=metadata].slice_description` and the
# wrapper's `metadata_retrieval_intent` already supply the holistic
# restatement of what this endpoint is asked to retrieve.


class MetadataWalk(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # Reuses the shared ColumnCandidate class: its descriptors talk
    # about "the call's intent" generically so the same class fits
    # both the single-endpoint context (where search_picture is the
    # restatement) and the walk context (where the user-message
    # retrieval_intent + expressions play that role).
    column_candidates: list[ColumnCandidate] = Field(
        ...,
        description=(
            "Column-grounded audit of how the metadata endpoint could "
            "cover the call's intent. One entry per column whose "
            "definition plausibly carries part of the call (read the "
            "call's `retrieval_intent` + `expressions` in the user "
            "message), with concrete strengths + weaknesses.\n"
            "\n"
            "This is the GROUNDED walk that precedes the bucket-level "
            "coverage_exploration / coverage_assignments commitment. "
            "Surface every column with substantive strengths so the "
            "commitment phase reads off real candidates rather than "
            "abstract optimism. An empty list (no column meaningfully "
            "fits) is a valid signal that the metadata endpoint has "
            "nothing useful — the commitment phase is allowed to leave "
            "the call unowned by metadata.\n"
            "\n"
            "TEST per candidate: 'if I dropped this, would the commit "
            "step lose a real option?' Padding → drop.\n"
            "\n"
            "NEVER list every column out of habit. NEVER duplicate "
            "columns."
        ),
    )


class MetadataTranslationOutputSubintent(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # Thin commitment-only spec for multi-endpoint contexts. The
    # column-grounded analysis (`column_candidates`) has been lifted up
    # to MetadataWalk at the bucket level — populated BEFORE
    # coverage_assignments so the commitment phase is grounded in
    # concrete column candidates. `search_picture` is dropped because
    # `metadata_retrieval_intent` (on the wrapper) plus
    # `coverage_assignments[kind=metadata].slice_description` already
    # capture the holistic restatement.

    scoring_method_reasoning: str = Field(
        ...,
        description=(
            "1 sentence. Project forward to column_spec: of the "
            "columns you intend to populate, do they read as "
            "SUBSTITUTABLE signals of one concept (any-one-matching "
            "qualifies) or REINFORCING facets (every populated column "
            "contributes)? Write \"single column\" when only one "
            "column will be populated.\n"
            "Justifies scoring_method below."
        ),
    )
    column_spec: ColumnSpec = Field(
        ...,
        description=(
            "Literal commitment. Populate ONLY columns surfaced in "
            "the bucket-level `metadata_walk.column_candidates` above "
            "with substantive strengths; explicit null elsewhere. "
            "Apply minimum span — null a column whose intent fragment "
            "is fully covered by another populated column. Same-column "
            "intent merges into ONE populated sub-object (country lists "
            "union, runtime ranges reconcile, streaming services + "
            "access pair).\n"
            "Local test per column: \"if I null this, does the "
            "assigned slice lose real intent?\" If no, null it.\n"
            "NEVER populate a column absent from "
            "metadata_walk.column_candidates. NEVER split same-column "
            "intent across multiple fields."
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
            "NEVER re-derive from column_spec — read off "
            "scoring_method_reasoning."
        ),
    )


class MetadataEndpointSubintentParameters(EndpointParameters):
    metadata_retrieval_intent: str = Field(
        ...,
        description=(
            "The slice of the call's intent this endpoint owns, "
            "committed by the bucket-level coverage_assignments above "
            "(the entry whose endpoint_kind matches metadata). Restate "
            "the assigned slice_description here so the commitment "
            "below has a single, self-contained pointer.\n"
            "\n"
            "Scope: the ten structured-attribute columns (release_date, "
            "runtime, maturity_rating, streaming, audio_language, "
            "country_of_origin, budget_scale, box_office, popularity, "
            "reception). Leave categorical classification, named "
            "entities, free-form thematic qualifiers, awards, and "
            "franchise structure to their respective endpoints. Every "
            "field on this endpoint's `parameters` reads from this "
            "intent (and from the upstream metadata_walk) rather than "
            "from any other input."
        ),
    )
    parameters: MetadataTranslationOutputSubintent = Field(
        ...,
        description=(
            "Metadata endpoint thin commitment payload. Reads off "
            "`metadata_retrieval_intent` (the assigned slice) and the "
            "bucket-level `metadata_walk.column_candidates` (the "
            "grounded audit above) to commit column_spec + "
            "scoring_method."
        ),
    )
