# Step 3 awards endpoint structured output model.
#
# Translates an award dealbreaker or preference description from step 2
# into a concrete query specification that step 4 can execute against
# movie_awards (or the award_ceremony_win_ids fast path on movie_card).
#
# Receives: intent_rewrite (step 1) + one item's description and
# routing hint from step 2.
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 3: Awards) for the full design rationale.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt. Developer notes live in
# comments above the class.
#
# Data sources:
#   - Fast path: award_ceremony_win_ids GIN array on movie_card.
#     Used only when: all filter fields are null/empty, outcome is
#     WINNER, scoring_mode=FLOOR, scoring_mark=1.
#     Razzie id stripped unless AwardCeremony.RAZZIE is in ceremonies.
#   - Standard path: COUNT(*) on movie_awards with active filters.
#     Used for all other specs.
#
# Direction-agnostic: always expressed as positive presence.
# Exclusion is a step 4 concern.

from pydantic import BaseModel, ConfigDict, Field, model_validator

from schemas.award_category_tags import CategoryTag
from schemas.enums import AwardCeremony, AwardOutcome, AwardScoringMode


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


# Step 3 awards endpoint output.
#
# All filter fields are optional — null/empty means no restriction on
# that axis. Generic "award-winning" is currently represented as
# THRESHOLD / 3 with all filters null. FLOOR / 1 with all filters null
# is the generic binary "has at least one non-Razzie win" shape.
#
# Field ordering:
#   concept_analysis    — filter-axis evidence inventory, emitted first
#   scoring_shape_label — brief intensity-pattern classification (primes mode + mark)
#   scoring_mode        — FLOOR | THRESHOLD
#   scoring_mark        — the count value that determines scoring shape
#   ceremonies          — filter to specific ceremonies (enum strings)
#   award_names         — filter to specific prize names
#   category_tags       — filter to specific award categories via the
#                         3-level tag taxonomy (leaf / mid / group)
#   outcome             — WINNER | NOMINEE | None (both)
#   years               — optional year range or single year
#
# Razzie handling (execution concern, not schema):
#   When ceremonies is null/empty, Razzie is excluded from all counts
#   and filters. When AwardCeremony.RAZZIE is explicitly present in
#   ceremonies, it is included — the user intentionally asked for it.
#
# Scoring:
#   has_count = COUNT(*) on movie_awards rows matching active filters
#   FLOOR:     1.0 if has_count >= scoring_mark else 0.0
#   THRESHOLD: min(has_count, scoring_mark) / scoring_mark
class AwardQuerySpec(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # Filter-axis evidence inventory (see system prompt).
    concept_analysis: str

    # Brief intensity-pattern classification (see system prompt).
    # Placed before scoring_mode to prime mode + mark decisions.
    scoring_shape_label: str

    # --- Scoring shape ---

    scoring_mode: AwardScoringMode

    # The count that determines the scoring shape:
    #   FLOOR:     1.0 if has_count >= scoring_mark else 0.0
    #   THRESHOLD: min(has_count, scoring_mark) / scoring_mark
    scoring_mark: int = Field(..., ge=1)

    # --- Filters (null/empty = no restriction on that axis) ---

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
