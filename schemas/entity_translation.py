# Step 3 entity endpoint structured output model.
#
# Translates an entity dealbreaker or preference description from
# step 2 into a concrete query specification that step 4 can execute
# against the lexical posting tables.
#
# Receives: intent_rewrite (step 1) + one item's description and
# route_rationale (step 2).
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 1: Entity Lookup) for the full design rationale.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt. Developer notes live in
# comments above the class.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.enums import (
    EntityType,
    PersonCategory,
    ProminenceMode,
    SpecificPersonCategory,
    TitlePatternMatchType,
)

# Which person_category values cause the actor posting table to
# participate in the search. broad_person unions all five role tables
# (including actor); actor targets the actor table directly.
_ACTOR_TABLE_CATEGORIES: frozenset[PersonCategory] = frozenset(
    {PersonCategory.ACTOR, PersonCategory.BROAD_PERSON}
)

# Remap table for when the LLM picks a prominence mode whose semantics
# don't apply to the entity being searched. Rather than reject the
# output — which would force a retry for what is really a
# misclassification, not a malformed response — we translate to the
# nearest valid mode for the target entity. See ProminenceMode
# docstring in schemas/enums.py for the rationale behind each pairing.
_CHARACTER_MODE_REMAP: dict[ProminenceMode, ProminenceMode] = {
    ProminenceMode.LEAD: ProminenceMode.CENTRAL,
    ProminenceMode.SUPPORTING: ProminenceMode.DEFAULT,
    ProminenceMode.MINOR: ProminenceMode.DEFAULT,
}
_ACTOR_MODE_REMAP: dict[ProminenceMode, ProminenceMode] = {
    ProminenceMode.CENTRAL: ProminenceMode.LEAD,
}


# Step 3 entity endpoint output.
#
# Flat model with nullable type-specific fields. The LLM populates
# primary_form and entity_type for all lookups, then fills only the
# fields relevant to that entity type (others null).
#
# Reasoning fields precede the decisions they scaffold:
#   entity_type_evidence        → scaffolds entity_type + person_category
#   name_resolution_notes       → scaffolds primary_form
#   alternative_forms_evidence  → scaffolds alternative_forms
#   prominence_evidence         → scaffolds prominence_mode
#
# Field ordering:
#   entity_type_evidence
#   name_resolution_notes
#   primary_form
#   entity_type
#   person_category
#   primary_category
#   alternative_forms_evidence
#   alternative_forms
#   prominence_evidence
#   prominence_mode
#   title_pattern_match_type
class EntityQuerySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Evidence-inventory reasoning that scaffolds entity_type and
    # (for persons) person_category. Must be emitted BEFORE those
    # fields. Guidance lives in the system prompt.
    entity_type_evidence: constr(strip_whitespace=True, min_length=1) = Field(...)

    # Brief canonicalization note scoped to resolving the single
    # primary_form string — NOT the place for alias reasoning.
    # Guidance lives in the system prompt.
    name_resolution_notes: constr(strip_whitespace=True, min_length=1) = Field(...)

    # The primary credited form of the entity name, or the literal
    # pattern for title_pattern lookups. Always populated. For
    # person/character entities, additional credited forms go in
    # alternative_forms (see below).
    #
    # Person: "Christopher Nolan". Character: "The Joker".
    # Title pattern: "love".
    primary_form: constr(strip_whitespace=True, min_length=1) = Field(...)

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

    # --- Identity-alias fields (person + character only) ---

    # Evidence-inventory reasoning that scaffolds alternative_forms.
    # Prompts the model to actively consider whether the entity has
    # additional credited forms (re-castings of the same character
    # under different actors, secret-identity aliases, stage-vs-legal
    # name for persons). Only populated when entity_type is person or
    # character; null for title_pattern.
    alternative_forms_evidence: constr(
        strip_whitespace=True, min_length=1
    ) | None = Field(default=None)

    # Additional credited forms beyond primary_form for exact matching.
    # Each form is searched independently against the lexical /
    # character string dictionaries; per-movie score is the max across
    # all variant rows (handles the "peter parker OR spider-man wins"
    # case). Empty list is valid when primary_form is the only known
    # form. Null for title_pattern entities — literal substring
    # patterns don't have aliases.
    alternative_forms: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=0
    ) | None = Field(default=None)

    # --- Prominence fields (actor-table persons + characters) ---

    # Evidence-inventory reasoning that scaffolds prominence_mode.
    # Populated only when billing-position scoring is meaningful:
    #   entity_type == person AND person_category in {actor, broad_person}
    #   OR entity_type == character
    # Null for everyone else (director/writer/producer/composer-only
    # persons, title_pattern).
    prominence_evidence: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
    )

    # How to score billing prominence. Valid modes depend on the
    # entity:
    #   actor-table → {DEFAULT, LEAD, SUPPORTING, MINOR}
    #   character   → {DEFAULT, CENTRAL}
    # When the LLM emits an out-of-scope value, the validator remaps
    # to the nearest in-scope mode (see _CHARACTER_MODE_REMAP /
    # _ACTOR_MODE_REMAP above). Null when prominence scoring does not
    # apply to the entity.
    prominence_mode: ProminenceMode | None = Field(default=None)

    # --- Title pattern fields (entity_type == "title_pattern") ---

    # Whether primary_form appears anywhere in the title (contains)
    # or must be at the start (starts_with).
    title_pattern_match_type: TitlePatternMatchType | None = Field(default=None)

    # --- Validators ---

    # Structural coercions applied post-parse so execution code does
    # not need to re-implement the same defaults:
    #   (1) the abstention sentinel "not applicable" on any evidence
    #       field is normalized back to null.
    #   (2) prominence_evidence defaults to "no prominence signal"
    #       whenever prominence scoring applies and the LLM left it
    #       null (actor-table persons, characters).
    #   (3) prominence_mode defaults to DEFAULT whenever prominence
    #       scoring applies and the LLM left it null. When the LLM
    #       emitted an out-of-scope mode, remap to the nearest valid
    #       mode per the remap tables at module scope.
    #   (4) prominence_* fields are forced to null when the entity is
    #       not prominence-eligible (director-only persons, etc.).
    #   (5) alternative_forms_evidence defaults to "no additional
    #       forms considered" when the entity is person/character and
    #       the LLM left it null. Forced to null for title_pattern.
    #   (6) alternative_forms is forced to null for title_pattern.
    @model_validator(mode="after")
    def _normalize_fields(self) -> "EntityQuerySpec":
        if (
            self.prominence_evidence is not None
            and self.prominence_evidence.strip().lower() == "not applicable"
        ):
            self.prominence_evidence = None

        if (
            self.alternative_forms_evidence is not None
            and self.alternative_forms_evidence.strip().lower() == "not applicable"
        ):
            self.alternative_forms_evidence = None

        is_actor_table_person = (
            self.entity_type == EntityType.PERSON
            and self.person_category in _ACTOR_TABLE_CATEGORIES
        )
        is_character = self.entity_type == EntityType.CHARACTER
        prominence_applies = is_actor_table_person or is_character
        aliases_apply = (
            self.entity_type == EntityType.PERSON
            or is_character
        )

        # (2) + (3) + (4): prominence field coercion.
        if prominence_applies:
            if self.prominence_evidence is None:
                self.prominence_evidence = "no prominence signal"
            if self.prominence_mode is None:
                self.prominence_mode = ProminenceMode.DEFAULT
            elif is_character and self.prominence_mode in _CHARACTER_MODE_REMAP:
                self.prominence_mode = _CHARACTER_MODE_REMAP[self.prominence_mode]
            elif is_actor_table_person and self.prominence_mode in _ACTOR_MODE_REMAP:
                self.prominence_mode = _ACTOR_MODE_REMAP[self.prominence_mode]
        else:
            # Non-prominence-eligible entities: force both fields null
            # so downstream code can trust the invariant.
            self.prominence_evidence = None
            self.prominence_mode = None

        # (5) + (6): alternative-forms field coercion. For
        # prominence-eligible entities the prompt requires the LLM to
        # walk a three-question procedure; if it omits the field we
        # use a NEUTRAL placeholder rather than something that reads
        # as a canonical "null answer" ("no additional forms known"
        # acted as a tractor-beam default that the model raced toward
        # before walking the questions). "walkthrough skipped" is
        # obviously a process failure, not a valid answer — it does
        # not prime the model in the same way.
        if aliases_apply:
            if self.alternative_forms_evidence is None:
                self.alternative_forms_evidence = "walkthrough skipped"
            # alternative_forms left as emitted (None or a list). Execution
            # code treats None as empty.
        else:
            self.alternative_forms_evidence = None
            self.alternative_forms = None

        return self
