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
    SpecificPersonCategory,
    TitlePatternMatchType,
)

# Which person_category values cause the actor posting table to
# participate in the search. broad_person unions all five role tables
# (including actor); actor targets the actor table directly.
_ACTOR_TABLE_CATEGORIES: frozenset[PersonCategory] = frozenset(
    {PersonCategory.ACTOR, PersonCategory.BROAD_PERSON}
)


# Step 3 entity endpoint output.
#
# Flat model with nullable type-specific fields. The LLM populates
# lookup_text and entity_type for all lookups, then fills only the
# fields relevant to that entity type (others null).
#
# Three scoped reasoning fields precede the decisions they scaffold:
#   entity_type_evidence → scaffolds entity_type and person_category
#   name_resolution_notes → scaffolds lookup_text
#   prominence_evidence  → scaffolds actor_prominence_mode
#
# Field ordering:
#   entity_type_evidence — evidence inventory for lookup type + role signal
#   name_resolution_notes — brief canonicalization / literal-pattern note
#   lookup_text — the primary search key, always required
#   entity_type — classifies which sub-type logic applies
#   person_category — narrows person searches to specific role tables
#   primary_category — cross-posting anchor for broad_person searches
#   prominence_evidence — evidence inventory for prominence language
#   actor_prominence_mode — billing-position scoring for actor results
#   title_pattern_match_type — contains vs starts_with for title searches
#   character_alternative_names — additional credited name variations
class EntityQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Evidence-inventory reasoning that scaffolds entity_type and
    # (for persons) person_category. Must be emitted BEFORE those
    # fields. Guidance lives in the system prompt.
    entity_type_evidence: constr(strip_whitespace=True, min_length=1) = Field(...)

    # Brief canonicalization / literal-pattern note emitted before the
    # search key. Guidance lives in the system prompt.
    name_resolution_notes: constr(strip_whitespace=True, min_length=1) = Field(...)

    # The corrected, normalized name or pattern to search for.
    # Person: "Christopher Nolan". Character: "The Joker".
    # Studio: "Pixar". Title pattern: "love".
    lookup_text: constr(strip_whitespace=True, min_length=1) = Field(...)

    entity_type: EntityType = Field(...)

    # --- Person fields (entity_type == "person") ---

    # Which role table(s) to search. Specific role = single table.
    # broad_person = all 5 tables, scored via primary_category.
    person_category: PersonCategory | None = Field(default=None)

    # Cross-posting anchor for broad_person searches. Set to the
    # specific role the person is predominantly known for — that
    # table gets full credit, others get 0.5x. Null = all tables
    # contribute equally. Ignored when person_category is a specific
    # role (single-table search, no cross-posting).
    primary_category: SpecificPersonCategory | None = Field(default=None)

    # Evidence-inventory reasoning that scaffolds actor_prominence_mode.
    # Only populated when the entity search touches the actor table
    # (entity_type == person AND person_category in {actor,
    # broad_person}); otherwise null. Must be emitted BEFORE
    # actor_prominence_mode.
    prominence_evidence: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
    )

    # How to score actor billing position. Only meaningful when
    # person_category is actor or broad_person (determines how the
    # actor table's results are scored). Null defaults to DEFAULT
    # in execution code.
    actor_prominence_mode: ActorProminenceMode | None = Field(default=None)

    # --- Title pattern fields (entity_type == "title_pattern") ---

    # Whether lookup_text appears anywhere in the title (contains)
    # or must be at the start (starts_with).
    title_pattern_match_type: TitlePatternMatchType | None = Field(default=None)

    # --- Character fields (entity_type == "character") ---

    # Additional credited name variations beyond lookup_text for
    # exact matching. Empty list when lookup_text is the only known
    # form. Each variation is searched independently; a match on any
    # (including lookup_text itself) scores 1.0.
    character_alternative_names: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=0
    ) | None = Field(default=None)

    # --- Validators ---

    # Structural coercions applied post-parse so execution code
    # does not need to re-implement the same defaults:
    #   (1) the abstention sentinel "not applicable" on
    #       prominence_evidence is normalized back to null.
    #   (2) prominence_evidence defaults to "no prominence signal"
    #       whenever the actor table participates and the LLM left it
    #       null, so actor-applicable specs don't silently lose that
    #       field.
    #   (3) actor_prominence_mode defaults to DEFAULT whenever the
    #       actor posting table is part of the search (entity_type is
    #       person AND person_category is actor or broad_person) and
    #       the LLM left it null. Matches the execution-layer default
    #       described in the finalized proposal.
    @model_validator(mode="after")
    def _normalize_person_fields(self) -> "EntityQuerySpec":
        if (
            self.prominence_evidence is not None
            and self.prominence_evidence.strip().lower() == "not applicable"
        ):
            self.prominence_evidence = None

        if (
            self.entity_type == EntityType.PERSON
            and self.person_category in _ACTOR_TABLE_CATEGORIES
        ):
            if self.prominence_evidence is None:
                self.prominence_evidence = "no prominence signal"

            if self.actor_prominence_mode is None:
                self.actor_prominence_mode = ActorProminenceMode.DEFAULT

        return self
