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
# Searchable axes (see finalized_search_proposal.md Endpoint 4):
#   1. lineage_or_universe_names — up to 3 alternate exact stored-form
#      attempts searched against both `lineage` and `shared_universe`
#      after shared normalization.
#   2. recognized_subgroups — up to 3 alternate exact stored-form
#      attempts searched against normalized subgroup labels.
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
#   concept_analysis            — axis-signal evidence inventory
#   lineage_or_universe_names   — up to 3 canonical name variations
#   recognized_subgroups        — up to 3 subgroup-name variations
#   lineage_position            — narrative position enum
#   structural_flags            — spinoff / crossover list
#   launch_scope                — franchise vs subgroup launcher
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
    # Up to 3 canonical franchise/IP name variations. These are not
    # fuzzy expansions; they are alternate exact stored-form attempts
    # for the same underlying brand after shared normalization.
    #
    # Position 1 = the most canonical form. Additional entries are
    # added ONLY when a genuinely different canonical name is in
    # common use and might plausibly be the one stored at ingest
    # time (e.g., "Marvel Cinematic Universe" vs. "Marvel";
    # "The Lord of the Rings" vs. "Middle-earth"). Do not add
    # orthographic variants — casing and punctuation are normalized
    # separately, but the remaining string match is exact.
    #
    # Null when the concept is purely structural (e.g., "spinoff
    # movies", "movies that launched a franchise") with no named
    # franchise.
    lineage_or_universe_names: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
        max_length=3,
    ) | None = Field(default=None)

    # --- Subgroup axis ---
    #
    # Up to 3 canonical subgroup-name variations. Like the top-level
    # name axis, these are alternate exact stored-form attempts after
    # shared normalization, not fuzzy variants.
    #
    # Must be null when lineage_or_universe_names is null — subgroup
    # labels are only meaningful inside a resolved franchise scope.
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
        # Subgroups only make sense within a resolved franchise.
        if self.recognized_subgroups is not None and self.lineage_or_universe_names is None:
            raise ValueError(
                "recognized_subgroups requires lineage_or_universe_names to be populated"
            )

        # Deduplicate structural flags while preserving first-seen order.
        if self.structural_flags is not None:
            deduped_flags = list(dict.fromkeys(self.structural_flags))
            self.structural_flags = deduped_flags or None

        # At least one axis must be populated — otherwise the
        # endpoint has nothing to search for.
        has_any_axis = any([
            self.lineage_or_universe_names is not None,
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
