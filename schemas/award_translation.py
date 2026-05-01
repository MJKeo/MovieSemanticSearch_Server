# Step 3 awards endpoint structured output model.
#
# Translates a category-handler award call into one or more concrete
# searches that the executor can run against movie_awards (or the
# award_ceremony_win_ids fast path on movie_card).
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 3: Awards) for the full design rationale.
#
# Data sources:
#   - Fast path: award_ceremony_win_ids GIN array on movie_card.
#     Used only when: all filter fields are null/empty, outcome is
#     WINNER, scoring.mode=FLOOR, scoring.mark=1.
#     Razzie id stripped unless AwardCeremony.RAZZIE is in ceremonies.
#   - Standard path: COUNT(*) on movie_awards with active filters.
#     Used for all other specs.
#
# Direction-agnostic: always expressed as positive presence.
# Exclusion is supplied by the wrapper's polarity field.

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.award_category_tags import CategoryTag
from schemas.endpoint_parameters import EndpointParameters
from schemas.enums import (
    AwardCeremony,
    AwardCombineMode,
    AwardOutcome,
    AwardScoringMode,
)


# Year range sub-model. Single year: year_from == year_to.
# If the LLM emits year_from > year_to, the validator swaps them
# rather than raising — graceful handling of transposition errors.
class AwardYearFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    year_from: int
    year_to: int

    @model_validator(mode="after")
    def _normalize(self) -> "AwardYearFilter":
        if self.year_from > self.year_to:
            self.year_from, self.year_to = self.year_to, self.year_from
        return self


# Filters for one executable award search. Null/empty means no
# restriction on that axis.
class AwardFilters(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # AwardCeremony string values. Null and empty list are equivalent —
    # execution treats both as "all non-Razzie ceremonies."
    ceremonies: list[AwardCeremony] | None = None

    # Prize names as canonical base forms. Resolved at query time via
    # normalize + tokenize + stoplist drop + posting-list intersection
    # against lex.award_name_token; NOT a raw-string comparison against
    # movie_awards.award_name. See
    # search_improvement_planning/v2_search_data_improvements.md
    # § Award Name Resolution. Null when not specified.
    award_names: list[str] | None = None

    # Category-concept tags. Members may come from any of the three tag
    # levels (leaf / mid / group) — the LLM picks at whatever specificity
    # the requirement implies. Execution converts members to their integer
    # tag ids and runs `category_tag_ids && ARRAY[...]` against the
    # GIN-indexed column. Null when not specified.
    category_tags: list[CategoryTag] | None = None

    # WINNER | NOMINEE | None (both outcomes count).
    outcome: AwardOutcome | None = None

    # Year range. Null = any year. Single year: year_from == year_to.
    years: AwardYearFilter | None = None


# Scoring formula for one executable award search.
class AwardScoring(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # FLOOR:     1.0 if has_count >= mark else 0.0
    # THRESHOLD: min(has_count, mark) / mark
    mode: AwardScoringMode
    mark: int = Field(..., ge=1)


# One executable COUNT(*)-style award search. Multiple filters inside
# one search are ANDed across axes and ORed within each list axis.
class AwardSearch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filters: AwardFilters
    scoring: AwardScoring


# Step 3 awards endpoint output. A category call may become one search
# when its expressions are parts of one structured award query, or
# multiple searches when its expressions represent separate award asks.
class AwardQueryPlan(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    combine: AwardCombineMode
    searches: list[AwardSearch] = Field(..., min_length=1)


# Backwards-compatible name for older imports while the endpoint prompt
# surface is migrated to the query-plan terminology.
AwardQuerySpec = AwardQueryPlan


# Category-handler wrapper. role + polarity are stamped post-LLM-
# call from the parent Trait, not declared on this schema (see
# endpoint_parameters.py).
class AwardEndpointParameters(EndpointParameters):
    parameters: AwardQueryPlan = Field(
        ...,
        description=(
            "Award endpoint payload. Emit one or more executable award "
            "searches plus a combine mode. Use one search when the "
            "expressions are parts of a single structured award query; "
            "use multiple searches when they represent separate award "
            "asks. Filters are ANDed across axes inside each search and "
            "ORed within list axes. combine=any takes the best search "
            "score per movie; combine=average gives partial credit by "
            "averaging all searches, with missing searches counting as "
            "0.0. Describe the target concept directly regardless of "
            "polarity — negation is handled on the wrapper's polarity "
            "field, never inside these parameters."
        ),
    )
