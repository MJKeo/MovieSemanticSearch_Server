# Step 3 franchise_structure endpoint structured output model.
#
# Translates a franchise dealbreaker or preference description from
# step 2 into a concrete query specification that step 4 can execute
# against `movie_franchise_metadata`.
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
# Searchable axes (see v2_search_data_improvements.md §Franchise
# Resolution):
#   1. franchise_or_universe_names — up to 3 canonical surface forms
#      resolved through the franchise token inverted index
#      (`lex.franchise_token`) and matched against
#      `movie_card.franchise_name_entry_ids` (the per-movie UNION of
#      lineage + shared_universe entry ids). Ingest may have placed
#      the franchise name in either the `lineage` or `shared_universe`
#      stored column; the union-at-ingest representation means the
#      query side does not need to predict which column it landed in.
#   2. recognized_subgroups — up to 3 canonical subgroup surface forms
#      resolved through the same token index and matched against
#      `movie_card.subgroup_entry_ids`.
#   3. lineage_position — SEQUEL / PREQUEL / REMAKE / REBOOT.
#   4. structural_flags — SPINOFF and/or CROSSOVER.
#   5. launch_scope — FRANCHISE or SUBGROUP.
#
# Direction-agnostic: always expressed as positive presence.
# Exclusion is a step 4 concern.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.enums import (
    FranchiseLaunchScope,
    FranchiseStructuralFlag,
    LineagePosition,
)


# Step 3 franchise endpoint output.
#
# Flat model with nullable per-axis fields. Step 2 sends one
# distinct concept per call; the LLM populates only the axis (or
# axes) that concept targets. If more than one field populates
# (e.g., "Marvel spinoffs" → name + structural_flags=[SPINOFF]),
# execution treats them as AND, not OR — both conditions must hold.
#
# Field ordering (cognitive scaffolding — each reasoning field
# immediately precedes the decisions it grounds, per the
# "cognitive-scaffolding field ordering" convention):
#   concept_analysis              — axis-signal evidence inventory
#   franchise_or_universe_names   — up to 3 canonical surface forms
#   recognized_subgroups          — up to 3 subgroup surface forms
#   lineage_position              — narrative position enum
#   structural_flags              — spinoff / crossover list
#   launch_scope                  — franchise vs subgroup launcher
class FranchiseQuerySpec(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    # Evidence-inventory reasoning. Emitted FIRST, before any axis
    # field — it scaffolds the presence/absence decision for every
    # axis that follows. The system prompt teaches the model to
    # extract signals from `description` first, then use
    # `intent_rewrite` only to disambiguate vague references from
    # the description. `routing_rationale` is a hint, not evidence.
    #
    # Evidence-inventory framing (quote the input, don't justify
    # the output) is deliberate — it constrains over-inference.
    # The model cannot assign an axis that has no cited phrase
    # supporting it. Empty-evidence paths are explicit: "no
    # franchise name phrase present" is a valid trace, not a
    # signal to fabricate a name.
    concept_analysis: constr(strip_whitespace=True, min_length=1) = Field(...)

    # --- Name axis ---
    #
    # Up to 3 canonical surface forms for the franchise / IP /
    # shared universe. Each entry is tokenized (whitespace + hyphen
    # split after shared normalization) and resolved through
    # `lex.franchise_token` — intra-name token intersection produces
    # one `franchise_entry_id` set per name, and the across-name
    # union sweeps the umbrella. Example: emitting
    # `["marvel cinematic universe", "marvel"]` intersects to the MCU
    # entry AND unions in every `marvel`-tagged entry (Marvel Comics,
    # Marvel Knights, etc.) for an umbrella sweep.
    #
    # The ingest side unions lineage + shared_universe entry ids onto
    # `movie_card.franchise_name_entry_ids`, so either stored column
    # is searchable from this single field — the LLM does NOT pick
    # between them.
    #
    # Position 1 = the most canonical form for the query's apparent
    # specificity (broadest for umbrella queries, narrowest for
    # specific-lineage queries). Add 2-3 entries ONLY when genuinely
    # different canonical forms are in common use (e.g., "Marvel
    # Cinematic Universe" + "Marvel"; "The Lord of the Rings" +
    # "Middle-earth"). Do NOT pad with spelling, punctuation,
    # hyphenation, diacritic, or digit-vs-word variants — shared
    # normalization collapses those symmetrically at ingest and
    # query time.
    #
    # Null when the concept is purely structural (e.g., "spinoff
    # movies", "movies that launched a franchise") with no named
    # franchise.
    franchise_or_universe_names: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
        max_length=3,
    ) | None = Field(default=None)

    # --- Subgroup axis ---
    #
    # Up to 3 canonical subgroup surface forms. Same tokenization +
    # token-index resolution as the top-level name axis, matched
    # against `movie_card.subgroup_entry_ids`.
    #
    # Independent of `franchise_or_universe_names` — a subgroup-only
    # spec ("trilogies", "phase one movies") is a valid, complete
    # query and populates only this field. Execution AND-composes
    # whichever axes are populated.
    recognized_subgroups: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
        max_length=3,
    ) | None = Field(default=None)

    # --- Narrative position ---
    #
    # SEQUEL / PREQUEL / REMAKE / REBOOT. Null when the concept does
    # not reference narrative position.
    #
    # Note on REMAKE: the enum value is retained for ingest-side
    # classification fidelity but is NOT commonly consumed at search
    # time — broad remake queries route to source material. This field
    # remains available for the narrow franchise-structural cases that
    # still belong here.
    lineage_position: LineagePosition | None = Field(default=None)

    # --- Structural flags ---
    #
    # Optional list of orthogonal structural traits. A single concept
    # can legitimately request both, though that is rare. Null means
    # neither trait is asserted.
    structural_flags: conlist(
        FranchiseStructuralFlag,
        min_length=1,
        max_length=2,
    ) | None = Field(default=None)

    # --- Launch scope ---
    #
    # Distinguishes "launched a franchise" from "launched a subgroup."
    # Null when launch behavior is not part of the concept.
    launch_scope: FranchiseLaunchScope | None = Field(default=None)

    # --- Validators ---

    @model_validator(mode="after")
    def _validate(self) -> "FranchiseQuerySpec":
        # Deduplicate structural flags while preserving first-seen order.
        # Small data-quality fix, not a semantic constraint.
        if self.structural_flags is not None:
            deduped_flags = list(dict.fromkeys(self.structural_flags))
            self.structural_flags = deduped_flags or None

        # At least one axis must be populated — otherwise the
        # endpoint has nothing to search for. All axes are mutually
        # independent: a subgroup-only spec ("trilogies") is valid,
        # a structural-only spec ("spinoffs") is valid, etc.
        has_any_axis = any([
            self.franchise_or_universe_names is not None,
            self.recognized_subgroups is not None,
            self.lineage_position is not None,
            self.structural_flags is not None,
            self.launch_scope is not None,
        ])
        if not has_any_axis:
            raise ValueError(
                "FranchiseQuerySpec must populate at least one axis "
                "(name, subgroup, lineage_position, structural_flags, or launch_scope)"
            )

        return self
