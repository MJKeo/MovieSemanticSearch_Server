# Step 3 franchise_structure endpoint structured output model.
#
# Receives (from a step-3 CategoryCall): `expressions` — list of
# short searchable phrases — plus `retrieval_intent` — 1-3 sentence
# operational context. Wrapper carries role + polarity.
#
# Cognitive scaffolding: `request_overview` is the single
# interpretation point. Every axis field below READS from it
# rather than re-interpreting raw inputs. Without that pivot, axis
# fields drift independently and contradict each other.
#
# No class-level docstrings. Schema = micro-prompts (field-shape
# rules); the system prompt carries procedural workflow. Canonical-
# naming, subgroup, spinoff, crossover, and launched_franchise
# definitions must stay aligned with the ingest-side franchise
# generator (movie_ingestion/metadata_generation/prompts/franchise.py)
# so the two LLMs agree on what to write into each slot.
#
# Searchable axes (see v2_search_data_improvements.md §Franchise
# Resolution):
#   1. franchise_names — canonical surface forms resolved through the
#      franchise token inverted index (`lex.franchise_token`) and
#      matched against `movie_card.lineage_entry_ids` OR
#      `movie_card.shared_universe_entry_ids`. Pack alt forms of one
#      franchise AND distinct franchises with shared axes into one
#      list — the index OR-unions both cases identically.
#   2. subgroup_names — canonical subgroup surface forms resolved
#      through the same token index, matched against
#      `movie_card.subgroup_entry_ids`.
#   3. lineage_position — SEQUEL / PREQUEL / REMAKE / REBOOT.
#   4. structural_flags — SPINOFF and/or CROSSOVER (AND semantics).
#   5. launch_scope — FRANCHISE or SUBGROUP.
#   6. prefer_lineage — bool. Biases scoring toward lineage matches
#      over shared-universe-only matches when the request commits to
#      one specific franchise's main line. Does not restrict the
#      match set.
#
# Direction-agnostic: always expressed as positive presence.
# Exclusion is supplied by the wrapper's polarity field.

from pydantic import BaseModel, ConfigDict, Field, conlist, constr, model_validator

from schemas.endpoint_parameters import (
    POLARITY_DESCRIPTION,
    ROLE_DESCRIPTION,
    EndpointParameters,
)
from schemas.enums import (
    FranchiseLaunchScope,
    FranchiseStructuralFlag,
    LineagePosition,
    Polarity,
    Role,
)


# Field order is cognitive scaffolding — earlier fields ground later
# ones, and pointer dependencies flow forward only:
#   request_overview  — interpretation, source-of-truth for axes
#   franchise_names   — primary lookup
#   subgroup_names    — independent lookup
#   lineage_position  — narrative position enum
#   structural_flags  — structural traits list
#   launch_scope      — references subgroup_names
#   prefer_lineage    — references request_overview + retrieval_intent
class FranchiseQuerySpec(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    request_overview: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "1-2 compact sentences. Holistic reading of "
            "retrieval_intent + expressions: what is being requested, "
            "how many distinct franchises are involved, their common "
            "aliases / canonical surface forms, and the handling "
            "posture (umbrella sweep, single specific lineage, "
            "structural-only, subgroup-only, position-only).\n"
            "\n"
            "Source-of-truth for every axis below — axis fields READ "
            "from this prose, they do not re-derive from raw inputs. "
            "Decisive commitments only; hedged framings produce "
            "drifting axes.\n"
            "\n"
            "NEVER:\n"
            "- COPY retrieval_intent verbatim — restate as an "
            "executable interpretation.\n"
            "- ENUMERATE the axis values here — that is the next "
            "step's job. Stay at the interpretation layer."
        ),
    )

    # --- Name axis ---
    franchise_names: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
    ) | None = Field(
        default=None,
        description=(
            "Canonical franchise / IP / shared-universe surface forms "
            "drawn from the franchises and aliases enumerated in "
            "request_overview. Resolved via shared tokenizer + "
            "lex.franchise_token, OR-unioned across entries, matched "
            "against movie_card.lineage_entry_ids OR "
            "shared_universe_entry_ids.\n"
            "\n"
            "Pack every name whose franchise_entry_ids should be "
            "OR-unioned into one list — alt forms of one franchise, "
            "distinct franchises with shared axes, or both. The index "
            "treats them identically.\n"
            "\n"
            "Tokenizer (matched at ingest, do NOT pad against): "
            "lowercase, diacritic fold, punct strip, whitespace + "
            "hyphen split, ordinal/cardinal digit-to-word, stopword "
            "drop. Spelling / casing / hyphenation / digit-word "
            "variants collide automatically.\n"
            "\n"
            "Null when request_overview commits to a purely "
            "structural / subgroup-only / position-only request.\n"
            "\n"
            "NEVER:\n"
            "- INVENT names not present in request_overview.\n"
            "- PAD with orthographic variants — the tokenizer "
            "collapses those.\n"
            "- ADD an umbrella term beside a narrow lineage already "
            "subsumed by it."
        ),
    )

    # --- Subgroup axis ---
    subgroup_names: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1,
    ) | None = Field(
        default=None,
        description=(
            "Canonical subgroup surface forms — phases, sagas, "
            "trilogies, timelines, director-eras — drawn from "
            "request_overview. Same tokenizer + token-index "
            "resolution as franchise_names; matched against "
            "movie_card.subgroup_entry_ids. Independent of "
            "franchise_names: subgroup-only requests are valid.\n"
            "\n"
            "Only widely-used labels (studio terminology, mainstream "
            "criticism, established fan vocabulary). Null when "
            "request_overview commits to no subgroup.\n"
            "\n"
            "NEVER:\n"
            "- INVENT a label.\n"
            "- RESTATE the parent franchise as a subgroup."
        ),
    )

    # --- Narrative position ---
    lineage_position: LineagePosition | None = Field(
        default=None,
        description=(
            "Narrative-position commitment drawn from "
            "request_overview. Null when no narrative position is "
            "signaled. REMAKE belongs here only for franchise-"
            "specific remake concepts; broad remake queries route "
            "elsewhere upstream."
        ),
    )

    # --- Structural flags ---
    structural_flags: conlist(
        FranchiseStructuralFlag,
        min_length=1,
    ) | None = Field(
        default=None,
        description=(
            "Structural traits drawn from request_overview. AND "
            "semantics — every flag listed must hold on the matched "
            "movie. Null when request_overview commits to no "
            "structural trait."
        ),
    )

    # --- Launch scope ---
    launch_scope: FranchiseLaunchScope | None = Field(
        default=None,
        description=(
            "What the movie launched, drawn from request_overview. "
            "Null when launch behavior is not part of the request.\n"
            "\n"
            "Composes with subgroup_names. SUBGROUP paired with a "
            "populated subgroup_names narrows to the launcher of "
            "THAT subgroup; SUBGROUP with subgroup_names null "
            "matches launchers of any subgroup. FRANCHISE matches "
            "franchise launchers regardless of subgroup_names."
        ),
    )

    # --- Lineage preference ---
    prefer_lineage: bool = Field(
        default=False,
        description=(
            "Scoring bias: lineage matches score 1.0, shared-"
            "universe-only matches score 0.75. Match set unchanged. "
            "Default False.\n"
            "\n"
            "Read primarily from request_overview's posture — does "
            "the overview commit to a single specific franchise's "
            "main line, vs umbrella sweep, vs multi-franchise, vs "
            "spinoff-affirmative? retrieval_intent is the secondary "
            "source when the overview is silent on main-line vs "
            "umbrella posture.\n"
            "\n"
            "Set True only when request_overview commits to one "
            "specific franchise with a clear main line and the "
            "request does not invite spinoffs, umbrella content, or "
            "named-subgroup content. Mechanical incompatibilities "
            "(no franchise_names, multi-name list, SPINOFF flag, "
            "populated subgroup_names) are coerced to False by the "
            "validator — commit freely from the overview."
        ),
    )

    # --- Validators ---

    @model_validator(mode="after")
    def _validate(self) -> "FranchiseQuerySpec":
        # Deduplicate structural flags while preserving first-seen order.
        # Small data-quality fix, not a semantic constraint.
        if self.structural_flags is not None:
            deduped_flags = list(dict.fromkeys(self.structural_flags))
            self.structural_flags = deduped_flags or None

        # Soft coercion for prefer_lineage. The LLM commits the flag
        # from request_overview's posture; mechanical incompatibilities
        # are silently flipped to False rather than raised, so the LLM
        # is not asked to track these in-prompt. Cases:
        #   - no franchise_names (nothing to bias)
        #   - franchise_names has >1 entry (umbrella / multi-franchise;
        #     main-line bias is meaningless)
        #   - SPINOFF in structural_flags (user invited spinoffs;
        #     biasing lineage would invert the intent)
        #   - subgroup_names populated (subgroup already disambiguates)
        # StrEnum members compare equal to their string values, so the
        # SPINOFF membership check handles both enum and post-
        # use_enum_values string representations.
        if self.prefer_lineage:
            no_name = not self.franchise_names
            multi_name = (
                self.franchise_names is not None
                and len(self.franchise_names) > 1
            )
            spinoff_invited = (
                self.structural_flags is not None
                and FranchiseStructuralFlag.SPINOFF in self.structural_flags
            )
            subgroup_present = bool(self.subgroup_names)
            if no_name or multi_name or spinoff_invited or subgroup_present:
                self.prefer_lineage = False

        # At least one axis must be populated — otherwise the endpoint
        # has nothing to search for. All axes are mutually independent:
        # a subgroup-only spec ("trilogies") is valid, a structural-
        # only spec ("spinoffs") is valid, etc.
        has_any_axis = any([
            self.franchise_names is not None,
            self.subgroup_names is not None,
            self.lineage_position is not None,
            self.structural_flags is not None,
            self.launch_scope is not None,
        ])
        if not has_any_axis:
            raise ValueError(
                "FranchiseQuerySpec must populate at least one axis "
                "(franchise_names, subgroup_names, lineage_position, "
                "structural_flags, or launch_scope)"
            )

        return self


# Category-handler wrapper. Direction flows through role + polarity
# on the wrapper. Fields are declared in the order role → parameters
# → polarity so polarity is emitted last. See endpoint_parameters.py
# for the rationale.
class FranchiseEndpointParameters(EndpointParameters):
    role: Role = Field(..., description=ROLE_DESCRIPTION)
    parameters: FranchiseQuerySpec = Field(
        ...,
        description=(
            "Franchise endpoint payload. request_overview commits the "
            "interpretation of retrieval_intent + expressions; axis "
            "fields (franchise_names, subgroup_names, "
            "lineage_position, structural_flags, launch_scope, "
            "prefer_lineage) read from it. Multiple populated axes "
            "are ANDed at execution. At least one axis must be "
            "populated. Describe positively regardless of polarity — "
            "negation is handled on the wrapper's polarity field, "
            "never inside these parameters."
        ),
    )
    polarity: Polarity = Field(..., description=POLARITY_DESCRIPTION)
