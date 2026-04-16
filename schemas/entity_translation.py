# Step 3 entity endpoint structured output model.
#
# Translates an entity dealbreaker or preference description from
# step 2 into a concrete query specification that step 4 can execute
# against the lexical posting tables.
#
# Receives: intent_rewrite (step 1) + one item's description and
# routing_rationale (step 2).
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 1: Entity Lookup) for the full design rationale.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt. Developer notes live in
# comments above the class.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.enums import (
    ActorProminenceMode,
    EntityType,
    PersonCategory,
    TitlePatternMatchType,
)


# Step 3 entity endpoint output.
#
# Flat model with nullable type-specific fields. The LLM populates
# entity_name and entity_type for all lookups, then fills only the
# fields relevant to that entity type (others null).
#
# Field ordering:
#   entity_name — the primary search key, always required
#   entity_type — classifies which sub-type logic applies
#   person_category — narrows person searches to specific role tables
#   primary_category — cross-posting anchor for broad_person searches
#   actor_prominence_mode — billing-position scoring for actor results
#   title_pattern_match_type — contains vs starts_with for title searches
#   character_alternative_names — additional credited name variations
class EntityQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # The corrected, normalized name or pattern to search for.
    # Person: "Christopher Nolan". Character: "The Joker".
    # Studio: "Pixar". Title pattern: "love".
    entity_name: constr(strip_whitespace=True, min_length=1) = Field(...)

    entity_type: EntityType = Field(...)

    # --- Person fields (entity_type == "person") ---

    # Which role table(s) to search. Specific role = single table.
    # broad_person = all 5 tables, scored via primary_category.
    person_category: PersonCategory | None = Field(default=None)

    # Cross-posting anchor for broad_person searches. Set to the
    # specific role the person is predominantly known for — that
    # table gets full credit, others get 0.5x. Null = all tables
    # contribute equally. Ignored when person_category is a specific
    # role (single-table search, no cross-posting). Must be a specific
    # role value — broad_person is silently coerced to null by the
    # validator below.
    primary_category: PersonCategory | None = Field(default=None)

    # How to score actor billing position. Only meaningful when
    # person_category is actor or broad_person (determines how the
    # actor table's results are scored). Null defaults to DEFAULT
    # in execution code.
    actor_prominence_mode: ActorProminenceMode | None = Field(default=None)

    # --- Title pattern fields (entity_type == "title_pattern") ---

    # Whether entity_name appears anywhere in the title (contains)
    # or must be at the start (starts_with).
    title_pattern_match_type: TitlePatternMatchType | None = Field(default=None)

    # --- Character fields (entity_type == "character") ---

    # Additional credited name variations beyond entity_name for
    # exact matching. Empty list when entity_name is the only known
    # form. Each variation is searched independently; a match on any
    # (including entity_name itself) scores 1.0.
    character_alternative_names: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=0
    ) | None = Field(default=None)

    # --- Validators ---

    # broad_person is not a valid anchor — it means "search all
    # tables", which is the person_category's job. Coerce to null
    # so execution code doesn't need to handle this case.
    @model_validator(mode="after")
    def coerce_broad_person_primary(self) -> "EntityQuerySpec":
        if self.primary_category == PersonCategory.BROAD_PERSON:
            self.primary_category = None
        return self
