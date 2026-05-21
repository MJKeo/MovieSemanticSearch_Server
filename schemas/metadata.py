"""
Pydantic response schemas for LLM structured output.

These are the generation-side schemas used as response_format in batch
requests. One output schema per generation type:

    - PlotEventsOutput (Wave 1)
    - ReceptionOutput (Wave 1, dual-zone: extraction + synthesis)
    - PlotAnalysisOutput (Wave 2, justification sub-models for CoT)
    - ViewerExperienceOutput (Wave 2, per-section justifications)
    - WatchContextOutput (Wave 2, identity_note + evidence_basis)
    - NarrativeTechniquesOutput (Wave 2, per-section justifications)
    - ProductionKeywordsOutput (Wave 2, classification only)
    - ProductionTechniquesOutput (Wave 2, classification only)
    - FranchiseOutput (independent, non-embeddable classification)
    - SourceOfInspirationOutput (Wave 2, parametric knowledge allowed)

The existing schemas in implementation/classes/schemas.py remain
unchanged -- they're consumed by the search pipeline for reading
metadata from Qdrant. These generation-side schemas can evolve
independently. When deploying, align the search-side schemas.

Each schema class subclasses EmbeddableOutput and implements
embedding_text() which returns normalize_string()-processed text
for vector embedding. Legacy __str__() methods are retained for
backward compatibility but embedding_text() is the canonical source.

Pydantic's type_to_response_format_param() is used in generators
to convert these schemas into the json_schema format required by
the Batch API's response_format field.
"""

from abc import abstractmethod

from pydantic import BaseModel, Field, ConfigDict, constr, conlist, model_validator

from implementation.misc.helpers import normalize_string
from schemas.enums import (
    LineagePosition,
    SourceMaterialType,
    NarrativeStructureTag,
    PlotArchetypeTag,
    SettingTag,
    CharacterTag,
    EndingTag,
    ExperientialTag,
    ContentFlagTag,
)


# ---------------------------------------------------------------------------
# Base class for all metadata output schemas
# ---------------------------------------------------------------------------

# Base class for metadata output schemas. Provides two extension points:
#
# 1. validate_and_fix(content) — classmethod entry point used by the
#    batch result processor to validate raw LLM JSON and apply any
#    deterministic post-generation fixups. Default: pure validation.
#    Subclasses override to add type-specific fixup logic.
#
# 2. embedding_text() — returns the normalized string used for vector
#    embedding. Every embeddable *Output schema must implement this.
#    Non-embeddable schemas (e.g., ConceptTagsOutput) do not subclass
#    EmbeddableOutput and do not need to implement this.
class EmbeddableOutput(BaseModel):

    @classmethod
    def validate_and_fix(cls, content: str) -> "EmbeddableOutput":
        """Validate raw JSON content and apply deterministic fixups.

        Default: pure validation with no fixups. Subclasses that need
        post-generation transformations (e.g., implied tag insertion)
        override this method.
        """
        return cls.model_validate_json(content)

    @abstractmethod
    def embedding_text(self) -> str:
        """Return the normalized text to be embedded for this metadata type."""
        ...


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

# A section containing search-query-like term phrases.
# Used by WatchContext, NarrativeTechniques, and ProductionKeywords
# where each section is a flat list of terms without negations.
# No justification field — removed per spec Decision 5.
class TermsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description="Search-query-like phrases.",
    )


# A section containing both positive terms and negation phrases.
# Used by ViewerExperience where each section captures what the movie
# IS and what it is NOT. No justification field — removed per spec
# Decision 5.
class TermsWithNegationsSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "Search-query-like phrases representing prominent "
            "characteristics of the movie relevant to this section."
        ),
    )
    negations: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            'Search-query-like "avoidance" phrases for what the movie '
            'does NOT have or is NOT like. ALWAYS has "not" or "no" in it.'
        ),
    )



# ---------------------------------------------------------------------------
# Franchise (v9 — exact-match franchise reference anchors plus
# expanded worked examples, with independent crossover/spinoff
# boolean tests, NOT embedded into vectors)
# ---------------------------------------------------------------------------
#
# Three orthogonal blocks: IDENTITY (lineage, shared_universe,
# recognized_subgroups, launched_subgroup), NARRATIVE POSITION
# (lineage_position enum plus independent is_crossover and is_spinoff
# booleans, each with their own reasoning trace), and FRANCHISE LAUNCH
# (launched_franchise, which answers "did THIS film kick off a
# cinematic franchise that people recognize as a multi-film franchise
# today?"). The crossover and spinoff tests were previously bundled
# into a single special_attributes enum list with one shared reasoning
# field; splitting them prevents the longer spinoff analysis from
# crowding out crossover and lets each test run on its own scaffold.
# v9 also adds a prompt-side FRANCHISE REFERENCE section for known
# exact-match failure cases (Creed, Logan, Space Jam, Detective
# Pikachu, Venom, etc.) plus extra worked examples for modern sequel /
# reboot / source-boundary edge cases.
# See docs/modules/ingestion.md (franchise section) for design rationale
# and prompts/franchise.py for the procedure, definitions, and examples
# driving generation.
#
# Field order is load-bearing: each scoped reasoning field comes before
# the decision block it informs (chain-of-thought via schema order).
# Field descriptions are intentionally compact — the system prompt
# carries the main definitional weight. validate_and_fix() enforces
# internal consistency (partial null-propagation + launched_subgroup
# coupling + launched_franchise coherence) as cheap post-parse fixup.

class FranchiseOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Identity block
    lineage_reasoning: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Required reasoning trace for lineage and shared_universe. "
            "Follow the procedure defined in the system prompt. Must be "
            "emitted BEFORE lineage and shared_universe."
        ),
    )
    lineage: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
        description=(
            "See the system prompt for the definition, normalization "
            "rules, and null conditions."
        ),
    )
    shared_universe: constr(strip_whitespace=True, min_length=1) | None = Field(
        default=None,
        description=(
            "See the system prompt for the definition, valid shapes, "
            "normalization rules, and null conditions."
        ),
    )

    # Subgroup block
    subgroups_reasoning: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Required reasoning trace for recognized_subgroups and "
            "launched_subgroup. Follow the procedure defined in the "
            "system prompt. Must be emitted BEFORE recognized_subgroups "
            "and launched_subgroup."
        ),
    )
    recognized_subgroups: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "See the system prompt for the definition, filters, and "
            "normalization rules."
        ),
    )
    launched_subgroup: bool = Field(
        default=False,
        description="See the system prompt for the definition.",
    )

    # Narrative position block
    position_reasoning: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Required reasoning trace for lineage_position. Follow the "
            "procedure defined in the system prompt. Must be emitted "
            "BEFORE lineage_position."
        ),
    )
    lineage_position: LineagePosition | None = Field(
        default=None,
        description=(
            "See the system prompt for the definition of each enum "
            "value and the null conditions."
        ),
    )

    # Crossover test (runs first — short, often short-circuits)
    crossover_reasoning: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Required reasoning trace for is_crossover. Follow the "
            "procedure defined in the system prompt. Must be emitted "
            "BEFORE is_crossover."
        ),
    )
    is_crossover: bool = Field(
        default=False,
        description="See the system prompt for the definition.",
    )

    # Spinoff test (runs second — parametric recall, then structural
    # situating, then conditional character disambiguation)
    spinoff_reasoning: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Required reasoning trace for is_spinoff. Follow the "
            "procedure defined in the system prompt. Must be emitted "
            "BEFORE is_spinoff."
        ),
    )
    is_spinoff: bool = Field(
        default=False,
        description="See the system prompt for the definition.",
    )

    # Franchise launch flag
    launch_reasoning: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Required reasoning trace for launched_franchise. Follow "
            "the procedure defined in the system prompt. Must be "
            "emitted BEFORE launched_franchise."
        ),
    )
    launched_franchise: bool = Field(
        default=False,
        description=(
            "See the system prompt for the definition and the "
            "distinction from launched_subgroup."
        ),
    )

    @classmethod
    def validate_and_fix(cls, content: str) -> "FranchiseOutput":
        """Validate raw LLM JSON and apply deterministic fixups.

        (1) Partial null-propagation: if lineage is null, clear
        shared_universe / recognized_subgroups / launched_subgroup.
        lineage_position, is_crossover, and is_spinoff are deliberately
        preserved — pair-remakes and standalone spinoff-flavored films
        are legitimate with lineage=null.
        (2) launched_subgroup ⇄ recognized_subgroups coupling.
        (3) launched_franchise coherence: forcibly false when any
        structural precondition fails (lineage null, lineage_position
        populated, or is_spinoff true). Keeps the flag from drifting
        out of sync with the rest of the record if the LLM's reasoning
        slips.
        """
        instance = cls.model_validate_json(content)

        if instance.lineage is None:
            instance.shared_universe = None
            instance.recognized_subgroups = []
            instance.launched_subgroup = False

        if instance.launched_subgroup and not instance.recognized_subgroups:
            instance.launched_subgroup = False

        # launched_franchise coherence — enforce the hard preconditions
        # from the prompt's FIELD 7 test so a slipped reasoning trace
        # cannot leave the record internally inconsistent.
        if instance.launched_franchise:
            if (
                instance.lineage is None
                or instance.lineage_position is not None
                or instance.is_spinoff
            ):
                instance.launched_franchise = False

        return instance


# ---------------------------------------------------------------------------
# Wave 1: Plot Events
# ---------------------------------------------------------------------------

# Structured output from the plot_events generation (Wave 1).
#
# Produces a single chronological plot summary. The plot_summary
# field is passed directly to all downstream Wave 2 consumers.
#
# Setting and major_characters fields were removed after evaluation
# showed setting is redundant (already in the summary) and structured
# character extraction adds analytical burden better handled by the
# downstream plot_analysis generator. Character names appear naturally
# in the plot_summary text.
#
# Model: gpt-5-mini, reasoning_effort: minimal
class PlotEventsOutput(EmbeddableOutput):
    model_config = ConfigDict(extra="forbid")

    plot_summary: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description="Chronological plot summary.",
    )

    def __str__(self) -> str:
        return self.plot_summary.lower()

    def embedding_text(self) -> str:
        return self.plot_summary.lower()


# ---------------------------------------------------------------------------
# Wave 1: Reception
# ---------------------------------------------------------------------------

# Structured output from the reception generation (Wave 1).
#
# Dual-zone output structure:
#
# Extraction zone (NOT embedded, consumed by Wave 2 generators):
#     source_material_hint — short classifying phrase for adaptations/remakes
#     thematic_observations — themes, meaning, messages from reviews
#     emotional_observations — emotional tone, mood, atmosphere
#     craft_observations — narrative structure, pacing, performances as craft
#
# Synthesis zone (embedded into reception_vectors):
#     reception_summary — evaluative summary of audience opinion
#     praised_qualities — tag phrases for what audiences liked
#     criticized_qualities — tag phrases for what audiences disliked
#
# Fields are ordered extraction-first so that structured output
# models produce observations before synthesizing evaluative content.
#
# Model: gpt-5-mini, reasoning_effort: low
class ReceptionOutput(EmbeddableOutput):
    model_config = ConfigDict(extra="forbid")

    # -- Extraction zone: observations for downstream generators (NOT embedded) --

    source_material_hint: str | None = Field(
        default=None,
        description=(
            "Short classifying phrase for source material, adaptation, "
            "or remake status. Only if explicitly supported by input "
            "data. Examples: 'based on autobiography', 'remake', "
            "'based on book, sequel'. Null for originals or when unsure."
        ),
    )
    thematic_observations: str | None = Field(
        default=None,
        description=(
            "1-4 sentences: what did reviewers observe about themes, "
            "meaning, and messages? Descriptive, not evaluative. "
            "Null when input data has nothing for this dimension."
        ),
    )
    emotional_observations: str | None = Field(
        default=None,
        description=(
            "1-4 sentences: what did reviewers observe about emotional "
            "tone, mood, atmosphere, and viewing experience? "
            "Null when input data has nothing for this dimension."
        ),
    )
    craft_observations: str | None = Field(
        default=None,
        description=(
            "1-4 sentences: what did reviewers observe about narrative "
            "structure, pacing, and performances as craft? "
            "Null when input data has nothing for this dimension."
        ),
    )

    # -- Synthesis zone: evaluative content for embedding --

    reception_summary: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "2-3 sentence evaluative summary of how the movie was "
            "received: 'what did people think?'"
        ),
    )
    praised_qualities: conlist(str, max_length=6) = Field(
        default_factory=list,
        description="0-6 tag-like phrases for what audiences liked.",
    )
    criticized_qualities: conlist(str, max_length=6) = Field(
        default_factory=list,
        description="0-6 tag-like phrases for what audiences disliked.",
    )

    def __str__(self) -> str:
        parts = []
        if self.reception_summary:
            parts.append(self.reception_summary.lower())
        if self.praised_qualities:
            parts.append(", ".join(self.praised_qualities).lower())
        if self.criticized_qualities:
            parts.append(", ".join(self.criticized_qualities).lower())
        # Extraction-zone fields intentionally excluded from embedding text
        return "\n".join(parts)

    def embedding_text(self) -> str:
        # Synthesis zone only — extraction-zone fields excluded
        parts = []
        parts.append(f"reception_summary: {self.reception_summary.lower()}")
        if self.praised_qualities:
            praised = ", ".join(normalize_string(q) for q in self.praised_qualities)
            parts.append(f"praised: {praised}")
        if self.criticized_qualities:
            criticized = ", ".join(normalize_string(q) for q in self.criticized_qualities)
            parts.append(f"criticized: {criticized}")
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Wave 2: Plot Analysis
# ---------------------------------------------------------------------------


# Character arc with a reasoning field for chain-of-thought quality.
# Used only in the justification/evaluation variant. The reasoning
# field is generated first to scaffold a better label. Only the
# label is embedded — reasoning is never included in embedding text.
class CharacterArcWithReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One sentence explaining the character's arc and why it's "
            "central to the thematic concepts of the movie."
        ),
    )
    arc_transformation_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "1-4 word generic label for the character's final state or "
            "outcome of transformation. Search-query-like phrasing."
        ),
    )

    def __str__(self) -> str:
        # Only the label is embedded — reasoning aids generation quality
        return self.arc_transformation_label


# -- Sub-models with justification fields (used by production schema) --
# Justification text scaffolds better labels via chain-of-thought.
# NEVER embedded — __str__() methods return only the label.


# Elevator pitch with an explanation field for chain-of-thought quality.
class ElevatorPitchWithJustification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation_and_justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One sentence explaining why this elevator pitch is the best "
            "representation of the heart of this movie. Remove "
            "meta framing ('the story/movie'), articles, and filler."
        ),
    )
    elevator_pitch: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "The movie's elevator pitch — what you'd say in response to "
            "'what is this movie about?' Simple, concrete terms."
        ),
    )

    def __str__(self) -> str:
        # Only the pitch is embedded — justification aids generation quality
        return self.elevator_pitch


# A thematic concept (theme or lesson) with a justification field.
# Replaces both MajorThemeWithJustification and
# MajorLessonLearnedWithJustification — the theme/lesson distinction
# was ambiguous for small LLMs and irrelevant for vector search.
# The justification text is NEVER embedded — only the concept_label
# is used in embedding text.
class ThematicConceptWithJustification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    explanation_and_justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One sentence explaining why this concept captures a key "
            "theme or lesson of the movie."
        ),
    )
    concept_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "High-signal label summarizing the thematic concept for "
            "vector embedding. Use simple, generic, human-world-friendly terms."
        ),
    )

    def __str__(self) -> str:
        # Only the label is embedded — justification aids generation quality
        return self.concept_label


# Output schema for plot_analysis generation (Wave 2).
#
# Uses sub-models with explanation_and_justification / reasoning fields
# that scaffold better labels via chain-of-thought. Only the labels
# are embedded — justification text is never included in embedding text.
#
# Field order is optimized for autoregressive generation: genre →
# themes → elevator pitch → conflict → arcs → overview.
#
# Model: gpt-5-mini, reasoning_effort: minimal, verbosity: low
class PlotAnalysisOutput(EmbeddableOutput):
    model_config = ConfigDict(extra="forbid")

    genre_signatures: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=2, max_length=6,
    ) = Field(
        ...,
        description="2-6 search-query-like genre phrases.",
    )
    thematic_concepts: conlist(
        ThematicConceptWithJustification,
        min_length=0, max_length=5,
    ) = Field(
        ...,
        description=(
            "0-5 high-signal labels capturing both the central themes "
            "the story explores AND any moral messages or lessons it "
            "conveys. Use simple, generic, human-world-friendly terms. "
            "Empty list when input data is too sparse for confident extraction."
        ),
    )
    elevator_pitch_with_justification: ElevatorPitchWithJustification = Field(
        ...,
        description=(
            "The movie's elevator pitch — what you'd say in response to "
            "'what is this movie about?' 6 words or less, log-line style, "
            "simple concrete terms."
        ),
    )
    conflict_type: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=0, max_length=2,
    ) = Field(
        ...,
        description=(
            "0-2 search-query-like phrases describing the fundamental "
            "dramatic tension in the story. Empty list when no clear "
            "conflict is identifiable from the input."
        ),
    )
    character_arcs: conlist(CharacterArcWithReasoning, min_length=0, max_length=3) = Field(
        ...,
        description=(
            "0-3 key character transformations. Empty list when input "
            "data lacks character or plot information."
        ),
    )
    generalized_plot_overview: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="1-3 sentence thematic overview of the plot.",
    )

    def __str__(self) -> str:
        # Delegates to embedding_text() so the two stay in lockstep.
        return self.embedding_text()

    def embedding_text(self) -> str:
        """Build labeled embedding text for the plot_analysis vector space.

        V2 structured-label format: every field is emitted with an explicit
        snake_case label matching its Pydantic field name. This preserves
        per-attribute semantic context for the embedding model and lets the
        search-side subquery generator template queries into the exact same
        shape (prerequisite for cross-space rescoring — see
        search_improvement_planning/new_system_brainstorm.md "Embedding
        Format: Structured Labels").

        Field order:
          1. elevator_pitch  — shortest, highest-signal capsule first
          2. plot_overview   — longer prose thematic summary
          3. genre_signatures — enumerated categorical slots follow
          4. conflict
          5. themes          — thematic_concepts and character_arcs are
          6. character_arcs    adjacent because plot_analysis character arcs
                               are *thematic* arcs (e.g. "mentor's sacrificial
                               legacy"), semantically closest to themes.
                               Distinct from narrative_techniques'
                               film-language arc labels ("coming-of-age").
        """
        parts = []

        # Prose fields — lowercased only; punctuation is meaningful here.
        if self.elevator_pitch_with_justification:
            parts.append(
                "elevator_pitch: "
                + self.elevator_pitch_with_justification.elevator_pitch.lower()
            )
        if self.generalized_plot_overview:
            parts.append("plot_overview: " + self.generalized_plot_overview.lower())

        # Enumerated categorical fields — each term individually normalized.
        if self.genre_signatures:
            parts.append("genre_signatures: " + ", ".join(
                normalize_string(g) for g in self.genre_signatures
            ))
        if self.conflict_type:
            parts.append("conflict: " + ", ".join(
                normalize_string(c) for c in self.conflict_type
            ))
        if self.thematic_concepts:
            parts.append("themes: " + ", ".join(
                normalize_string(t.concept_label) for t in self.thematic_concepts
            ))
        if self.character_arcs:
            parts.append("character_arcs: " + ", ".join(
                normalize_string(arc.arc_transformation_label) for arc in self.character_arcs
            ))

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Wave 2: Viewer Experience
# ---------------------------------------------------------------------------

# Section with terms, negations, and a justification for chain-of-thought.
# The justification field provides chain-of-thought that improves
# output quality. It is NEVER embedded — only terms and negations
# are used in embedding text.
class TermsWithNegationsAndJustificationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "1 sentence. Why you chose these terms and negations "
            "for this section, or why the section is empty. "
            "Not used for embeddings."
        ),
    )
    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "Search-query-like phrases representing prominent "
            "characteristics of the movie relevant to this section."
        ),
    )
    negations: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            'Search-query-like "avoidance" phrases for what the movie '
            'does NOT have or is NOT like. ALWAYS has "not" or "no" in it.'
        ),
    )


# Output schema for viewer_experience generation (Wave 2).
#
# 8 sections capturing the emotional/sensory viewing experience.
# Each section includes a justification field that provides
# chain-of-thought improving specificity and term diversity.
# Justifications are discarded before embedding — no retrieval impact.
#
# Model: gpt-5-mini, reasoning_effort: minimal, verbosity: low
class ViewerExperienceOutput(EmbeddableOutput):
    model_config = ConfigDict(extra="forbid")

    emotional_palette: TermsWithNegationsAndJustificationSection
    tension_adrenaline: TermsWithNegationsAndJustificationSection
    tone_self_seriousness: TermsWithNegationsAndJustificationSection
    cognitive_complexity: TermsWithNegationsAndJustificationSection
    disturbance_profile: TermsWithNegationsAndJustificationSection
    sensory_load: TermsWithNegationsAndJustificationSection
    emotional_volatility: TermsWithNegationsAndJustificationSection
    ending_aftertaste: TermsWithNegationsAndJustificationSection

    def __str__(self) -> str:
        combined_terms: list[str] = []
        for section in (
            self.emotional_palette,
            self.tension_adrenaline,
            self.tone_self_seriousness,
            self.cognitive_complexity,
            self.disturbance_profile,
            self.sensory_load,
            self.emotional_volatility,
            self.ending_aftertaste,
        ):
            combined_terms.extend(section.terms)
            combined_terms.extend(section.negations)
        return ", ".join(t.lower() for t in combined_terms)

    def embedding_text(self) -> str:
        parts: list[str] = []
        labeled_sections = (
            ("emotional_palette", self.emotional_palette),
            ("tension_adrenaline", self.tension_adrenaline),
            ("tone_self_seriousness", self.tone_self_seriousness),
            ("cognitive_complexity", self.cognitive_complexity),
            ("disturbance_profile", self.disturbance_profile),
            ("sensory_load", self.sensory_load),
            ("emotional_volatility", self.emotional_volatility),
            ("ending_aftertaste", self.ending_aftertaste),
        )

        for label, section in labeled_sections:
            if section.terms:
                normalized_terms = ", ".join(
                    normalize_string(term) for term in section.terms
                )
                parts.append(f"{label}: {normalized_terms}")
            if section.negations:
                normalized_negations = ", ".join(
                    normalize_string(term) for term in section.negations
                )
                parts.append(f"{label}_negations: {normalized_negations}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Wave 2: Watch Context
# ---------------------------------------------------------------------------

# TermsSection with upstream evidence assessment.
#
# The evidence_basis field forces the model to inventory which specific
# input phrases support this section BEFORE generating terms. This is
# an upstream constraint — the model must identify concrete evidence
# first, then generate only terms that follow from that evidence.
#
# If no specific input phrases can be cited, terms should be empty.
#
# The evidence_basis text is NEVER embedded — only terms are used in
# embedding text.
class TermsWithJustificationSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_basis: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "1 concise sentence. Quote or closely paraphrase the specific "
            "input phrases that support terms in this section. Verify "
            "cited evidence supports the experiential conclusion, not "
            "just the topic. If no specific phrases can be cited, write "
            "'No direct evidence' and leave terms empty. Not embedded."
        ),
    )
    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description="Search-query-like phrases.",
    )


# Output schema for watch_context generation (Wave 2).
#
# Includes a brief identity_note pre-classification (2-8 words)
# that primes tone calibration before section generation, plus
# evidence_basis per section as an upstream constraint.
#
# 4 sections capturing when/why/how to watch the movie. Deliberately
# receives NO plot information — focuses on experiential attributes.
# All sections allow 0 terms — sparse inputs should produce fewer
# terms, not fabricated ones.
#
# The identity_note is NOT embedded — only section terms are used
# in embedding text.
#
# Model: gpt-5-mini, reasoning_effort: low
class WatchContextOutput(EmbeddableOutput):
    model_config = ConfigDict(extra="forbid")

    identity_note: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "2-8 words classifying this movie's viewing appeal type. "
            "E.g., 'sincere emotional drama', 'ironic camp classic', "
            "'visceral fun horror', 'intellectually challenging arthouse', "
            "'so-bad-it's-good guilty pleasure'. Not embedded."
        ),
    )
    self_experience_motivations: TermsWithJustificationSection = Field(
        ...,
        description=(
            "0-8 search-query-like phrases capturing the self-focused "
            "experiential reason someone would seek out this movie. "
            "Empty when inputs are too sparse for confident generation."
        ),
    )
    external_motivations: TermsWithJustificationSection = Field(
        ...,
        description=(
            "0-4 search-query-like phrases capturing value beyond the "
            "viewing experience: cultural significance, social currency, "
            "conversation starters. Empty when not supported by inputs."
        ),
    )
    key_movie_feature_draws: TermsWithJustificationSection = Field(
        ...,
        description=(
            "0-4 search-query-like phrases for standout movie attributes "
            "that function as draws (positive or negative). "
            "Empty when not supported by inputs."
        ),
    )
    watch_scenarios: TermsWithJustificationSection = Field(
        ...,
        description=(
            "0-6 search-query-like phrases for real-world occasions, "
            "contexts, and social settings for watching this movie. "
            "Empty when not supported by inputs."
        ),
    )

    def __str__(self) -> str:
        # identity_note intentionally excluded from embedding text
        combined_terms = (
            self.self_experience_motivations.terms
            + self.external_motivations.terms
            + self.key_movie_feature_draws.terms
            + self.watch_scenarios.terms
        )
        return ", ".join(t.lower() for t in combined_terms)

    def embedding_text(self) -> str:
        # identity_note intentionally excluded from embedding text
        parts: list[str] = []

        sections = (
            ("self_experience_motivations", self.self_experience_motivations.terms),
            ("external_motivations", self.external_motivations.terms),
            ("key_movie_feature_draws", self.key_movie_feature_draws.terms),
            ("watch_scenarios", self.watch_scenarios.terms),
        )
        for label, terms in sections:
            if not terms:
                continue
            normalized_terms = [normalize_string(term) for term in terms]
            parts.append(f"{label}: {', '.join(normalized_terms)}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Wave 2: Narrative Techniques
# ---------------------------------------------------------------------------

# Output schema for narrative_techniques generation (Wave 2).
#
# 9 sections capturing storytelling structure, POV, delivery mechanism,
# and narrative devices. Each section includes a justification field
# for chain-of-thought quality. Justification text is NEVER embedded.
#
# Field order is optimized for autoregressive generation: specific
# sections first, catchall last.
#
# Model: gpt-5-mini, reasoning_effort: minimal
class NarrativeTechniquesOutput(EmbeddableOutput):
    model_config = ConfigDict(extra="forbid")

    # Easiest to identify from any input type
    narrative_archetype: TermsWithJustificationSection
    narrative_delivery: TermsWithJustificationSection
    # Moderate — often evidenced in craft observations
    pov_perspective: TermsWithJustificationSection
    characterization_methods: TermsWithJustificationSection
    character_arcs: TermsWithJustificationSection
    audience_character_perception: TermsWithJustificationSection
    # Hardest — require plot knowledge or synthesis
    information_control: TermsWithJustificationSection
    conflict_stakes_design: TermsWithJustificationSection
    # Catchall — placed last so specific sections are filled first
    additional_narrative_devices: TermsWithJustificationSection

    def __str__(self) -> str:
        combined_terms: list[str] = []
        for section in (
            self.narrative_archetype,
            self.narrative_delivery,
            self.pov_perspective,
            self.characterization_methods,
            self.character_arcs,
            self.audience_character_perception,
            self.information_control,
            self.conflict_stakes_design,
            self.additional_narrative_devices,
        ):
            combined_terms.extend(section.terms)
        return ", ".join(t.lower() for t in combined_terms)

    def embedding_text(self) -> str:
        parts: list[str] = []

        sections = (
            ("narrative_archetype", self.narrative_archetype.terms),
            ("narrative_delivery", self.narrative_delivery.terms),
            ("pov_perspective", self.pov_perspective.terms),
            ("characterization_methods", self.characterization_methods.terms),
            ("character_arcs", self.character_arcs.terms),
            (
                "audience_character_perception",
                self.audience_character_perception.terms,
            ),
            ("information_control", self.information_control.terms),
            ("conflict_stakes_design", self.conflict_stakes_design.terms),
            (
                "additional_narrative_devices",
                self.additional_narrative_devices.terms,
            ),
        )
        for label, terms in sections:
            if not terms:
                continue
            normalized_terms = [normalize_string(term) for term in terms]
            parts.append(f"{label}: {', '.join(normalized_terms)}")

        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Wave 2: Production Keywords (separate LLM call)
# ---------------------------------------------------------------------------

# Structured output from the production_keywords generation (Wave 2).
#
# Classification task: the LLM filters merged_keywords to keep only
# production-relevant ones. Not generative — purely selective.
#
# Model: gpt-5-mini, reasoning_effort: low
class ProductionKeywordsOutput(EmbeddableOutput):
    model_config = ConfigDict(extra="forbid")

    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description="Production-relevant keywords filtered from the input list.",
    )

    def __str__(self) -> str:
        return ", ".join(t.lower() for t in self.terms)

    def embedding_text(self) -> str:
        normalized_terms = [normalize_string(term) for term in self.terms]
        return ", ".join(normalized_terms)


# ---------------------------------------------------------------------------
# Wave 2: Production Techniques (separate LLM call)
# ---------------------------------------------------------------------------

# Structured output from the production_techniques generation (Wave 2).
#
# Classification task: the LLM filters plot_keywords and overall_keywords
# to keep only production-technique terms: concrete making/rendering/capture
# methods, including animation modalities/sub-techniques plus the explicit
# found-footage exception. Not generative — purely selective.
#
# Model: gpt-5-mini, reasoning_effort: low
class ProductionTechniquesOutput(EmbeddableOutput):
    model_config = ConfigDict(extra="forbid")

    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description=(
            "Production-technique keywords filtered from the input lists, "
            "covering concrete making/rendering/capture methods."
        ),
    )

    def __str__(self) -> str:
        return ", ".join(t.lower() for t in self.terms)

    def embedding_text(self) -> str:
        normalized_terms = [normalize_string(term) for term in self.terms]
        return ", ".join(normalized_terms)


# ---------------------------------------------------------------------------
# Wave 2: Source of Inspiration (separate LLM call)
# ---------------------------------------------------------------------------

# Source of inspiration classification output.
#
# Two independent lists from the same inputs:
# - source_material: what existing media the film draws from
#   (adaptations, remakes, reboots, reimaginings, spinoffs)
# - franchise_lineage: where the film sits in a franchise timeline
#   (sequel, prequel, trilogy position, franchise starter)
#
# Parametric knowledge allowed at 95%+ confidence. Leaf-node
# classification — errors don't cascade to other generations.
class SourceOfInspirationOutput(EmbeddableOutput):
    model_config = ConfigDict(extra="forbid")

    source_material: list[str] = Field(
        default_factory=list,
        description=(
            "What existing media this film draws from. "
            "Adaptations: based on a novel, comic, true story, etc. "
            "Retellings/branches: remake, reboot, reimagining, spinoff. "
            "Empty when original or evidence insufficient."
        ),
    )
    franchise_lineage: list[str] = Field(
        default_factory=list,
        description=(
            "Where this film sits in a franchise timeline. "
            "E.g. franchise starter, first in franchise, sequel, prequel, "
            "first in trilogy, trilogy finale, series entry. "
            "NOT for remakes/reboots/spinoffs (those go in source_material). "
            "Empty when standalone or evidence insufficient."
        ),
    )

    def __str__(self) -> str:
        all_terms = self.source_material + self.franchise_lineage
        return ", ".join(t.lower() for t in all_terms)

    def embedding_text(self) -> str:
        parts: list[str] = []

        if self.source_material:
            normalized = [normalize_string(t) for t in self.source_material]
            parts.append(f"source material: {', '.join(normalized)}")
        elif self._is_likely_original():
            # No source material and not a continuation → likely original work
            parts.append("source material: original screenplay")

        if self.franchise_lineage:
            normalized = [normalize_string(t) for t in self.franchise_lineage]
            parts.append(f"franchise position: {', '.join(normalized)}")

        return "\n".join(parts)

    def _is_likely_original(self) -> bool:
        """True when franchise_lineage is empty or only has first/starter terms.

        Movies that are franchise starters (e.g. "first in franchise") are still
        original screenplays — they just also launched a franchise. Movies that
        are sequels, prequels, or continuations are NOT original even if
        source_material is empty (they derive from the prior film).
        """
        if not self.franchise_lineage:
            return True
        return all(
            "first" in t.lower() or "start" in t.lower()
            for t in self.franchise_lineage
        )


# ---------------------------------------------------------------------------
# Wave 2: Source Material V2 (enum-constrained re-generation)
# ---------------------------------------------------------------------------

# Enum-constrained source material classification output.
#
# Replaces the free-text source_material field from SourceOfInspirationOutput
# with a fixed set of SourceMaterialType enum values. franchise_lineage is
# removed entirely (handled by a separate franchise generation task).
#
# The LLM identifies which source material types are genuinely present —
# either directly evidenced in the inputs or known with 95%+ parametric
# confidence. This is identification, not fitting: only types that
# actually apply are included.
#
# An empty list means original screenplay — the movie has no external
# source material. This is handled at the search layer, not by the LLM.
#
# Parametric knowledge allowed at 95%+ confidence. Leaf-node
# classification — errors don't cascade to other generations.
#
# Model: gpt-5-mini, reasoning_effort: low
class SourceMaterialV2Output(EmbeddableOutput):
    model_config = ConfigDict(extra="forbid")

    source_material_types: list[SourceMaterialType] = Field(
        default_factory=list,
        description=(
            "All SourceMaterialType values that genuinely apply to this "
            "movie. Assign every type that is directly evidenced in the "
            "inputs or known with 95%+ confidence from parametric "
            "knowledge. Empty list when the movie is an original "
            "screenplay with no external source material. May contain "
            "multiple values (e.g. a film can be both novel_adaptation "
            "and true_story)."
        ),
    )

    def __str__(self) -> str:
        return ", ".join(t.value.replace("_", " ") for t in self.source_material_types)

    def embedding_text(self) -> str:
        if not self.source_material_types:
            return ""
        normalized = [
            normalize_string(t.value.replace("_", " "))
            for t in self.source_material_types
        ]
        return f"sources of inspiration: {', '.join(normalized)}"


# ---------------------------------------------------------------------------
# Concept Tags (binary classification — NOT embedded into vectors)
# ---------------------------------------------------------------------------

# Multi-label binary classification of 25 concept tags across 7 categories.
# Each tag answers a yes/no question: "does this movie have X?"
#
# Unlike all other output schemas, ConceptTagsOutput does NOT subclass
# EmbeddableOutput because concept tags are stored as integer IDs in a
# Postgres INT[] column, not embedded into vector spaces.
#
# Model: gpt-5-mini, reasoning_effort: minimal

# Per-category concept tag evidence classes.
#
# Each category has its own evidence class with a typed tag enum field,
# making the JSON schema self-enforcing — the model cannot produce a
# tag in the wrong category field. This eliminates the need for a
# runtime model_validator.
#
# Each category is a single assessment object containing a `reasoning`
# field (always emitted FIRST in the schema so the model walks the
# evidence before concluding) and the selected tags (or a single tag
# for endings). The reasoning field's description names the primary
# evidence sources for that category and enforces evidence-before-
# conclusion ordering, which combats post-hoc justification — the
# failure mode where the model decides a tag and then writes the
# evidence that justifies it.


class NarrativeStructureAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        ...,
        description=(
            "Walk through the evidence BEFORE deciding which tags apply. "
            "Primary sources for narrative structure: craft_observations "
            "(reviewer descriptions of how the story is told — twists, "
            "nonlinearity, narration style, fourth-wall, ending shape) "
            "and the narrative_technique_terms sections narrative_delivery, "
            "information_control, additional_narrative_devices, and "
            "pov_perspective. NOTE: those NT terms may use vocabulary "
            "that overlaps with concept-tag names (e.g. \"intercut "
            "flashback structure\", \"evidence-driven reversal\") — treat "
            "them as descriptive shorthand to investigate, not as direct "
            "labels. For cliffhanger_ending, additionally consult the "
            "closing scene in plot_summary and emotional_observations "
            "language about unresolved endings. Quote or name the "
            "specific signals you observe, then state what each implies "
            "for the candidate tags. EVIDENCE FIRST, conclusion second — "
            "never state the conclusion and then write the evidence that "
            "justifies it."
        ),
    )

    tags: list[NarrativeStructureTag] = Field(
        default_factory=list,
        description=(
            "Supported tags, derived from the conclusions in the reasoning "
            "field above. The reasoning field is the SOURCE OF TRUTH for "
            "this output — only emit a tag when the reasoning's conclusions "
            "support it, and do not omit a tag the reasoning argues for. "
            "You may include a tag when the reasoning's conclusions "
            "logically imply it (the reasoning does not need to literally "
            "name the tag), but the inclusion must follow from your "
            "reasoning, not from re-examined evidence. Empty list is "
            "correct when the reasoning concludes no tags apply."
        ),
    )


class PlotArchetypeAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        ...,
        description=(
            "Walk through the evidence BEFORE deciding which tags apply. "
            "Plot archetype tags answer 'is THIS the central premise of "
            "the movie?' — not 'does the movie contain this element'. "
            "Primary sources: plot_keywords, plot_summary or plot_text "
            "(the narrative itself), conflict_type (the central "
            "conflict), and character_arc_labels (the protagonist's "
            "thematic transformation). For each candidate tag, ask: "
            "would removing this concept collapse the plot? Cite the "
            "specific signal from the inputs, then state what it implies. "
            "EVIDENCE FIRST, conclusion second — never state the "
            "conclusion and then write the evidence that justifies it."
        ),
    )

    tags: list[PlotArchetypeTag] = Field(
        default_factory=list,
        description=(
            "Supported tags, derived from the conclusions in the reasoning "
            "field above. The reasoning field is the SOURCE OF TRUTH for "
            "this output — only emit a tag when the reasoning's conclusions "
            "support it, and do not omit a tag the reasoning argues for. "
            "You may include a tag when the reasoning's conclusions "
            "logically imply it (the reasoning does not need to literally "
            "name the tag), but the inclusion must follow from your "
            "reasoning, not from re-examined evidence. Empty list is "
            "correct when the reasoning concludes no tags apply."
        ),
    )


class SettingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        ...,
        description=(
            "Walk through the evidence BEFORE deciding which tags apply. "
            "A setting tag fires only when the setting is a DEFINING "
            "characteristic of the movie's identity, not incidental "
            "backdrop. Primary sources: plot_keywords, plot_summary or "
            "plot_text (where the action occurs and how central the "
            "setting is), and title/release_year for temporal context. "
            "Cite the specific signal that names or strongly implies the "
            "setting, then state whether it is defining or merely "
            "incidental. EVIDENCE FIRST, conclusion second — never state "
            "the conclusion and then write the evidence that justifies it."
        ),
    )

    tags: list[SettingTag] = Field(
        default_factory=list,
        description=(
            "Supported tags, derived from the conclusions in the reasoning "
            "field above. The reasoning field is the SOURCE OF TRUTH for "
            "this output — only emit a tag when the reasoning's conclusions "
            "support it, and do not omit a tag the reasoning argues for. "
            "You may include a tag when the reasoning's conclusions "
            "logically imply it (the reasoning does not need to literally "
            "name the tag), but the inclusion must follow from your "
            "reasoning, not from re-examined evidence. Empty list is "
            "correct when the reasoning concludes no tags apply."
        ),
    )


class CharacterAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        ...,
        description=(
            "Walk through the evidence BEFORE deciding which tags apply. "
            "Character tags classify the lead role(s) or cast structure. "
            "Primary sources: plot_summary or plot_text (who drives the "
            "story, how many leads there are, and how each protagonist "
            "actually behaves across the runtime), top_billed_cast "
            "(prominence ranking — corroborating evidence for who fills "
            "the lead role(s)), character_arc_labels from plot_analysis "
            "(thematic arc transformations such as \"impostor to "
            "reconciled contributor\" — these describe what the arc IS, "
            "but a redemption arc does NOT by itself disqualify any "
            "character tag), and conflict_type (whether the "
            "protagonist's stance is structurally outside conventional "
            "morality). DO NOT derive ANTI_HERO from upstream-labeled "
            "terms — derive it from what the protagonist actually does "
            "in plot_summary. For FEMALE_LEAD identify EVERY lead role "
            "and check whether every one is female. For ENSEMBLE_CAST "
            "apply the removal test: would removing any one storyline "
            "collapse the movie? For ANTI_HERO ask: is morally "
            "compromised behavior a substantive operating mode for a "
            "meaningful portion of the runtime, or one extreme choice "
            "under duress? Cite the specific cross-referenced signals "
            "you see, then state what they imply. EVIDENCE FIRST, "
            "conclusion second — never state the conclusion and then "
            "write the evidence that justifies it."
        ),
    )

    tags: list[CharacterTag] = Field(
        default_factory=list,
        description=(
            "Supported tags, derived from the conclusions in the reasoning "
            "field above. The reasoning field is the SOURCE OF TRUTH for "
            "this output — only emit a tag when the reasoning's conclusions "
            "support it, and do not omit a tag the reasoning argues for. "
            "You may include a tag when the reasoning's conclusions "
            "logically imply it (the reasoning does not need to literally "
            "name the tag), but the inclusion must follow from your "
            "reasoning, not from re-examined evidence. Empty list is "
            "correct when the reasoning concludes no tags apply."
        ),
    )


class EndingAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        ...,
        description=(
            "Walk through the evidence BEFORE deciding the tag. NO "
            "pre-labeled ending field is provided — you must derive the "
            "audience's ending feeling from raw evidence. Process: "
            "(1) CLOSING SCENE TEST (PRIMARY) — identify the literal "
            "final scene before credits from plot_summary. A "
            "celebration/reunion/kiss/cheer/platform-raise beat → HAPPY "
            "regardless of what was lost during the runtime. A "
            "funeral/destruction/loss beat with no recuperative upswing "
            "→ SAD. A quiet, contemplative beat (long look, unspoken "
            "moment, what-might-have-been montage, protagonist staring "
            "into the distance with both achievement and loss visible) → "
            "BITTERSWEET candidate, but verify with step 2. An existential / "
            "cosmic-indifference beat without clear valence → "
            "NO_CLEAR_CHOICE candidate. (2) emotional_observations — "
            "filter for language describing how audiences felt LEAVING "
            "the theater: celebration/triumph language ('uplifting', "
            "'satisfying', 'triumphant', 'earned', 'hard-won', "
            "'achievement at a cost') are HAPPY signals, NOT bittersweet; "
            "explicit unresolved language ('mixed feelings the audience "
            "cannot resolve', 'knot despite the win', 'unable to "
            "celebrate fully', 'genuinely torn') are BITTERSWEET signals; "
            "'devastating', 'crushing', 'heartbreaking' are SAD signals; "
            "'ambiguous', 'philosophical', 'contemplative' without "
            "valence are NO_CLEAR_CHOICE signals. Filter out journey-"
            "level emotions ('tense', 'frightening', 'dark') that "
            "describe the runtime, not the ending. (3) Final state of "
            "affairs from plot_summary — what is gained, lost, or "
            "unresolved? Reconcile with the closing-scene beat. "
            "(4) BASE RATES — HAPPY is empirically dominant; BITTERSWEET "
            "is uncommon. When ambiguous between HAPPY and BITTERSWEET "
            "→ HAPPY. When ambiguous between SAD and BITTERSWEET → SAD. "
            "BITTERSWEET requires that NEITHER HAPPY nor SAD would be a "
            "defensible alternative. EVIDENCE FIRST, conclusion second — "
            "never state the tag and then write the evidence that "
            "justifies it."
        ),
    )

    tag: EndingTag = Field(
        ...,
        description=(
            "Exactly one ending classification, derived from the "
            "conclusion in the reasoning field above. The reasoning field "
            "is the SOURCE OF TRUTH — the chosen tag must follow from "
            "what the reasoning concludes. Use no_clear_choice when the "
            "reasoning concludes the evidence is ambiguous, insufficient, "
            "or the ending's emotion does not fit happy/sad/bittersweet."
        ),
    )


class ExperientialAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        ...,
        description=(
            "Walk through the evidence BEFORE deciding which tags apply. "
            "Experiential tags (feel_good, tearjerker) describe how "
            "audiences actually FELT, not what happened in the plot. "
            "PRIMARY and AUTHORITATIVE source: emotional_observations "
            "(reviewer reports of audience emotional response). "
            "plot_keywords is a weak supporting signal only. Quote the "
            "specific emotional language from emotional_observations, "
            "then state what it implies for each candidate tag. A movie "
            "can be heavy or sad without being a tearjerker; a movie can "
            "be pleasant without being feel_good. EVIDENCE FIRST, "
            "conclusion second — never state the conclusion and then "
            "write the evidence that justifies it."
        ),
    )

    tags: list[ExperientialTag] = Field(
        default_factory=list,
        description=(
            "Supported tags, derived from the conclusions in the reasoning "
            "field above. The reasoning field is the SOURCE OF TRUTH for "
            "this output — only emit a tag when the reasoning's conclusions "
            "support it, and do not omit a tag the reasoning argues for. "
            "You may include a tag when the reasoning's conclusions "
            "logically imply it (the reasoning does not need to literally "
            "name the tag), but the inclusion must follow from your "
            "reasoning, not from re-examined evidence. Empty list is "
            "correct when the reasoning concludes no tags apply."
        ),
    )


class ContentFlagAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        ...,
        description=(
            "Walk through the evidence BEFORE deciding which tags apply. "
            "Content flag tags identify content users specifically search "
            "to AVOID (severe violence, sexual content, animal death, "
            "suicide depiction). PRIMARY source: parental_guide_items "
            "(IMDB content-advisory categories with severity ratings "
            "like 'Violence Against Animals (severe)' or 'Suicide "
            "(mild)'). Supporting: plot_keywords and plot_summary or "
            "plot_text. A tag fires only when the content is genuinely "
            "present at meaningful severity — not when a similar word "
            "merely appears in a plot summary. Quote the specific "
            "parental_guide_items line items you see and their severity, "
            "then state what each implies for the candidate tags. "
            "EVIDENCE FIRST, conclusion second — never state the "
            "conclusion and then write the evidence that justifies it."
        ),
    )

    tags: list[ContentFlagTag] = Field(
        default_factory=list,
        description=(
            "Supported tags, derived from the conclusions in the reasoning "
            "field above. The reasoning field is the SOURCE OF TRUTH for "
            "this output — only emit a tag when the reasoning's conclusions "
            "support it, and do not omit a tag the reasoning argues for. "
            "You may include a tag when the reasoning's conclusions "
            "logically imply it (the reasoning does not need to literally "
            "name the tag), but the inclusion must follow from your "
            "reasoning, not from re-examined evidence. Empty list is "
            "correct when the reasoning concludes no tags apply."
        ),
    )


# Multi-label binary classification of concept tags by category.
#
# Each category is a single assessment object containing a `reasoning`
# field (emitted FIRST in the schema, forcing evidence-before-
# conclusion ordering) and the selected tags (or a single tag for
# endings). Each Assessment's reasoning field description names the
# primary evidence sources the model should examine for that category.
#
# The per-category typed enum lists are JSON-schema self-enforcing
# for category membership. The category structure forces the model
# to consider each domain independently.
#
# Not an EmbeddableOutput — concept tags become integer IDs in
# movie_card.concept_tag_ids, not embedding text.

class ConceptTagsOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narrative_structure: NarrativeStructureAssessment = Field(
        ...,
        description=(
            "Structural choices in how the story is told. Include "
            "only tags supported by input evidence."
        ),
    )
    plot_archetypes: PlotArchetypeAssessment = Field(
        ...,
        description=(
            "The central premise or driving force of the movie. "
            "Tag applies when the concept IS the movie, not just "
            "an element in the plot."
        ),
    )
    settings: SettingAssessment = Field(
        ...,
        description=(
            "Settings that define the movie's identity. The setting "
            "must be a defining characteristic, not incidental."
        ),
    )
    characters: CharacterAssessment = Field(
        ...,
        description=(
            "Character-level classifications about the protagonist "
            "or cast structure."
        ),
    )
    endings: EndingAssessment = Field(
        ...,
        description=(
            "How the audience feels when the credits roll. Exactly "
            "one classification (including no_clear_choice). "
            "Independent of open/cliffhanger structure tags."
        ),
    )
    experiential: ExperientialAssessment = Field(
        ...,
        description=(
            "Binary experiential qualities that users treat as "
            "deal-breakers when choosing what to watch."
        ),
    )
    content_flags: ContentFlagAssessment = Field(
        ...,
        description=(
            "Content that users specifically search to avoid. "
            "Strong avoidance deal-breakers."
        ),
    )

    def _deduplicate_tags(self) -> "ConceptTagsOutput":
        """Remove duplicate tags from all category tag lists.

        LLMs occasionally emit the same enum value twice in an array.
        Preserves first-occurrence order. Skips endings (single value,
        not a list).

        Mutates in place and returns self for chaining.
        """
        for field_name in (
            "narrative_structure", "plot_archetypes", "settings",
            "characters", "experiential", "content_flags",
        ):
            assessment = getattr(self, field_name)
            seen: set = set()
            deduped: list = []
            for tag in assessment.tags:
                if tag not in seen:
                    seen.add(tag)
                    deduped.append(tag)
            assessment.tags = deduped
        return self

    def apply_deterministic_fixups(self) -> "ConceptTagsOutput":
        """Apply deterministic post-generation fixups.

        Two rules applied in order:
        1. Deduplicate all tag lists (LLMs occasionally repeat tags).
        2. TWIST_VILLAIN implies PLOT_TWIST: if narrative_structure
           contains TWIST_VILLAIN but not PLOT_TWIST, add PLOT_TWIST.
           TWIST_VILLAIN is definitionally a subset of PLOT_TWIST, so
           this is handled deterministically rather than adding
           cognitive load to the LLM prompt.

        Mutates in place and returns self for chaining.
        """
        # Step 1: deduplicate before any implication logic
        self._deduplicate_tags()

        # Step 2: TWIST_VILLAIN → PLOT_TWIST implication
        ns_tags = self.narrative_structure.tags
        has_twist_villain = NarrativeStructureTag.TWIST_VILLAIN in ns_tags
        has_plot_twist = NarrativeStructureTag.PLOT_TWIST in ns_tags

        if has_twist_villain and not has_plot_twist:
            ns_tags.append(NarrativeStructureTag.PLOT_TWIST)

        return self

    @classmethod
    def validate_and_fix(cls, content: str) -> "ConceptTagsOutput":
        """Validate raw JSON, deduplicate tags, and apply fixups."""
        instance = cls.model_validate_json(content)
        return instance.apply_deterministic_fixups()

    def all_concept_tag_ids(self) -> list[int]:
        """Extract all concept_tag_ids as a sorted deduplicated int list.

        This is the value stored in movie_card.concept_tag_ids (planned).
        Filters out classification-only values (NO_CLEAR_ENDING, id=-1).
        """
        ids: set[int] = set()
        # Categories with tag lists
        for field_name in (
            "narrative_structure", "plot_archetypes", "settings",
            "characters", "experiential", "content_flags",
        ):
            assessment = getattr(self, field_name)
            for tag in assessment.tags:
                ids.add(tag.concept_tag_id)
        # Endings: single tag field, skip classification-only values
        ending_tag = self.endings.tag
        if ending_tag.concept_tag_id >= 0:
            ids.add(ending_tag.concept_tag_id)
        return sorted(ids)


# ---------------------------------------------------------------------------
# Legacy payload normalization (shared by every reader of stored metadata)
# ---------------------------------------------------------------------------

# Maps stored metadata JSON written under older schema shapes onto the
# current shape. The DB still holds rows written before two known schema
# changes — both readers in the codebase (the Movie loader for final
# ingestion, and inputs.load_narrative_techniques_output for the concept
# tags generator) must apply this before model_validate, or the legacy
# rows fail to parse and silently appear absent.
#
# Known migrations handled here:
#   1. NarrativeTechniquesOutput / WatchContextOutput: per-section
#      "justification" was renamed to "evidence_basis" when the prompt
#      shifted from "explain why" to "cite specific input phrases."
#      Old rows still carry "justification"; today's schema requires
#      "evidence_basis" with extra=forbid, so the bare key has to be
#      renamed in-place.
#   2. SourceOfInspirationOutput: top-level "source_evidence" and
#      "lineage_evidence" reasoning fields were dropped from the schema.
#      Old rows still carry them; today's schema rejects them under
#      extra=forbid, so they have to be stripped.
#
# Add new migrations by appending more branches here — keep them
# defensive (no-op if the legacy key isn't present) so re-applying to
# already-normalized payloads is safe.
def normalize_legacy_metadata_payload(
    payload: object,
    schema_class: type[BaseModel],
) -> object:
    """Normalize known legacy metadata shapes into the current schema.

    Returns the payload unchanged when it's not a dict or when no
    legacy shape applies to ``schema_class``. Safe to call on
    already-normalized payloads — each branch checks for the legacy
    key before touching anything.
    """
    if not isinstance(payload, dict):
        return payload

    if schema_class in {NarrativeTechniquesOutput, WatchContextOutput}:
        normalized_payload = dict(payload)
        for key, value in normalized_payload.items():
            if not isinstance(value, dict):
                continue
            if "justification" in value and "evidence_basis" not in value:
                section = dict(value)
                section["evidence_basis"] = section.pop("justification")
                normalized_payload[key] = section
        payload = normalized_payload

    if schema_class is SourceOfInspirationOutput:
        normalized_payload = dict(payload)
        normalized_payload.pop("source_evidence", None)
        normalized_payload.pop("lineage_evidence", None)
        payload = normalized_payload

    return payload
