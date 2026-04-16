# Step 3 franchise_structure endpoint structured output model.
#
# Translates a franchise dealbreaker or preference description from
# step 2 into a concrete query specification that step 4 can execute
# against `movie_franchise_metadata` and `lex.inv_franchise_postings`.
#
# Receives: intent_rewrite (step 1) + one item's description and
# routing_rationale (step 2).
#
# See search_improvement_planning/finalized_search_proposal.md
# (Endpoint 4: Franchise Structure) for the full design rationale.
#
# No class-level docstrings or Field descriptions — all LLM-facing
# guidance lives in the system prompt. Developer notes live in
# comments above the class. The prompt must import the same
# canonical-naming, subgroup, spinoff, crossover, and
# launched_franchise definitions used by the ingest-side franchise
# generator (movie_ingestion/metadata_generation/prompts/franchise.py)
# so the two LLMs agree on what to write into each slot.
#
# Searchable axes (see finalized_search_proposal.md Endpoint 4):
#   1. lineage_or_universe_names — fuzzy match via
#      lex.inv_franchise_postings against both `lineage` and
#      `shared_universe` columns (always searched together because
#      the ingest LLM can legitimately place the same brand in either
#      slot — e.g., Shrek vs. Puss in Boots).
#   2. recognized_subgroups — trigram match applied post-lookup on
#      the 3-30 movies returned by the franchise name search.
#   3. lineage_position — SEQUEL / PREQUEL / REMAKE / REBOOT.
#   4. is_spinoff, is_crossover, launched_franchise, launched_subgroup
#      — boolean filters on movie_franchise_metadata.
#
# Direction-agnostic: always expressed as positive presence.
# Exclusion is a step 4 concern.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.enums import LineagePosition


# Step 3 franchise endpoint output.
#
# Flat model with nullable per-axis fields. Step 2 sends one
# distinct concept per call; the LLM populates only the axis (or
# axes) that concept targets. If more than one field populates
# (e.g., "Marvel spinoffs" → name + is_spinoff), execution treats
# them as AND, not OR — both conditions must hold.
#
# Field ordering (cognitive scaffolding — each reasoning field
# immediately precedes the decisions it grounds, per the
# "cognitive-scaffolding field ordering" convention):
#   concept_analysis            — axis-signal evidence inventory
#   name_resolution_notes       — canonical-name expansion inventory
#   lineage_or_universe_names   — up to 3 canonical name variations
#   recognized_subgroups        — up to 3 subgroup-name variations
#   lineage_position            — narrative position enum
#   is_spinoff / is_crossover   — structural booleans
#   launched_franchise / launched_subgroup — launcher booleans
#
# Booleans: True means "this constraint is active"; None means
# "not asserted." False should never appear — direction-agnostic
# framing means we never search for "is NOT a spinoff."
class FranchiseQuerySpec(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # Evidence-inventory reasoning. Emitted FIRST, before any axis
    # field — it scaffolds the presence/absence decision for every
    # axis that follows. The procedure the system prompt teaches:
    #
    #   1. Reread `description` and `intent_rewrite`.
    #   2. For each signal phrase found, quote it verbatim and
    #      pair it with the franchise axis it implicates:
    #        - franchise / IP / "cinematic universe" name
    #              → lineage_or_universe_names
    #        - named sub-series / phase / saga / trilogy label
    #              → recognized_subgroups
    #        - sequel / prequel / reboot (REMAKE rarely used —
    #          generic remakes route to the keyword endpoint)
    #              → lineage_position
    #        - spinoff / side-story-with-side-character lead
    #              → is_spinoff
    #        - crossover / team-up / characters-from-separate-
    #          stories-interact
    #              → is_crossover
    #        - "started / launched / first / kicked off a
    #          franchise"
    #              → launched_franchise
    #        - "started / launched a subgroup / phase / saga"
    #              → launched_subgroup
    #   3. If no phrase signals a given axis, say so explicitly
    #      ("no signal for lineage_position") rather than leaving
    #      the inventory silent. Explicit absence calibrates the
    #      model against over-assignment.
    #   4. For ambiguous phrasing (e.g., "started the MCU" — does
    #      this mean launcher of the MCU franchise, or launcher
    #      of a subgroup inside some larger brand?), surface the
    #      ambiguity here so the boolean choice below is grounded
    #      rather than guessed.
    #
    # Evidence-inventory framing (quote the input, don't justify
    # the output) is deliberate — it constrains over-inference.
    # The model cannot assign an axis that has no cited phrase
    # supporting it. Empty-evidence paths are explicit: "no
    # franchise name phrase present" is a valid trace, not a
    # signal to fabricate a name.
    concept_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)

    # --- Name axis ---

    # Parametric-knowledge inventory of canonical name variations
    # for the franchise/IP identified above. Emitted immediately
    # BEFORE `lineage_or_universe_names` so the model consciously
    # enumerates alternate canonical forms before committing to
    # the list length (1 vs. 2 vs. 3). Also scaffolds
    # `recognized_subgroups` further down, since the same
    # canonical-form reasoning applies to subgroup names.
    #
    # The procedure the system prompt teaches:
    #
    #   1. If `concept_analysis` cites no franchise-name phrase
    #      (purely structural query like "spinoffs" or "movies
    #      that started a franchise"), emit the sentinel
    #      "not applicable — purely structural" and stop. No
    #      further expansion is needed.
    #   2. Otherwise, starting from the name phrase cited in
    #      `concept_analysis`, consult parametric knowledge for
    #      alternate CANONICAL forms of the SAME IP that are in
    #      common use — forms the ingest-side franchise generator
    #      might legitimately write into either `lineage` or
    #      `shared_universe`. Examples:
    #        - "Marvel Cinematic Universe; Marvel" — two distinct
    #          canonical forms of the same brand.
    #        - "The Lord of the Rings; Middle-earth" — distinct
    #          world label vs. series title.
    #        - "Star Wars" — single canonical form, no alternatives.
    #   3. EXCLUDE spelling / punctuation / casing / stylistic
    #      variants ("Lord of the Rings" vs. "The Lord of the
    #      Rings"; "Spider-Man" vs. "Spiderman"). Trigram fuzzy
    #      match on `lex.inv_franchise_postings` already handles
    #      those — listing them wastes slots and adds noise.
    #   4. Repeat the check for the subgroup name (if any) cited
    #      in `concept_analysis`. Include subgroup alternates in
    #      the same trace. Subgroups typically have fewer
    #      alternate canonical forms than top-level franchise
    #      names, and "single canonical form" is the common case.
    #
    # Telegraphic form: semicolon-separated phrases, or the
    # sentinel "not applicable — purely structural". Not a
    # reasoning paragraph — short labels prime the values below
    # without templating them (per the "brief pre-generation
    # fields, no consistency coupling" convention).
    #
    # Nullable: if the LLM abstains on a malformed description,
    # this field may be null; execution treats that as "no
    # expansion guidance" and trusts `lineage_or_universe_names`
    # as-is.
    name_resolution_notes: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
    )

    # Up to 3 canonical, fully-expanded franchise/IP name variations.
    # Position 1 = the most canonical form (follows the same
    # "most common, fully expanded, no abbreviations" rule as the
    # ingest-side franchise generator). Additional entries are added
    # ONLY when a genuinely different canonical name is in common
    # use (e.g., "Marvel Cinematic Universe" vs. "Marvel";
    # "The Lord of the Rings" vs. "Middle-earth"). Do not add
    # spelling variants — trigram fuzzy match on the posting table
    # already handles punctuation and minor spelling drift.
    #
    # Searched against both `lineage` and `shared_universe` columns
    # via lex.inv_franchise_postings; any variation matching either
    # column counts as a hit.
    #
    # Null when the concept is purely structural (e.g., "spinoff
    # movies", "movies that launched a franchise") with no named
    # franchise.
    lineage_or_universe_names: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=1, max_length=3
    ) | None = Field(default=None)

    # --- Subgroup axis ---

    # Up to 3 canonical subgroup-name variations (e.g., "MCU Phase 3"
    # vs. "Phase Three" vs. "Infinity Saga Part 2"). Applied as
    # trigram similarity on the normalized subgroup labels of movies
    # returned by the franchise name lookup. Any variation matching
    # any subgroup on a candidate movie counts as a hit.
    #
    # Must be null when lineage_or_universe_names is null — subgroups
    # are only meaningful within a resolved franchise result set.
    recognized_subgroups: conlist(
        constr(strip_whitespace=True, min_length=1), min_length=1, max_length=3
    ) | None = Field(default=None)

    # --- Narrative position ---

    # SEQUEL / PREQUEL / REMAKE / REBOOT. Null when the concept does
    # not reference narrative position.
    #
    # Note on REMAKE: the enum value is retained for ingest-side
    # classification fidelity but is NOT consumed at search time —
    # film-to-film retellings route through source_of_inspiration /
    # source_material instead. Step 2 should route "remake"-style
    # queries to the keyword endpoint; this field exists for the
    # rare case where a franchise-specific remake concept still
    # lands here. See schemas/enums.py LineagePosition.REMAKE.
    lineage_position: LineagePosition | None = Field(default=None)

    # --- Structural booleans ---
    # True = filter enabled (movie must match). None = not asserted.
    # False is never valid — direction-agnostic framing means we
    # always search for positive presence; step 4 applies exclusion.

    is_spinoff: bool | None = Field(default=None)
    is_crossover: bool | None = Field(default=None)
    launched_franchise: bool | None = Field(default=None)
    launched_subgroup: bool | None = Field(default=None)

    # --- Validators ---

    @model_validator(mode="after")
    def _validate(self) -> "FranchiseQuerySpec":
        # Subgroups only make sense within a resolved franchise.
        if self.recognized_subgroups is not None and self.lineage_or_universe_names is None:
            raise ValueError(
                "recognized_subgroups requires lineage_or_universe_names to be populated"
            )

        # Direction-agnostic framing: only True or None are meaningful.
        # Coerce stray False values to None so execution code doesn't
        # have to distinguish "not asserted" from "asserted false."
        for name in ("is_spinoff", "is_crossover", "launched_franchise", "launched_subgroup"):
            if getattr(self, name) is False:
                setattr(self, name, None)

        # At least one axis must be populated — otherwise the
        # endpoint has nothing to search for.
        has_any_axis = any([
            self.lineage_or_universe_names is not None,
            self.recognized_subgroups is not None,
            self.lineage_position is not None,
            self.is_spinoff is not None,
            self.is_crossover is not None,
            self.launched_franchise is not None,
            self.launched_subgroup is not None,
        ])
        if not has_any_axis:
            raise ValueError(
                "FranchiseQuerySpec must populate at least one axis "
                "(name, subgroup, lineage_position, or a structural boolean)"
            )

        return self
