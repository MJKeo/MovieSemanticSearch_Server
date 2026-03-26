"""
Pydantic response schemas for LLM structured output.

These are the generation-side schemas used as response_format in batch
requests. They're based on the existing schemas in
implementation/classes/schemas.py but with key modifications from the
redesigned flow (docs/llm_metadata_generation_new_flow.md):

Changes from existing schemas:
    - Justification fields REMOVED from all section models (pending
      empirical validation -- easy to add back). This includes:
      * GenericTermsSection.justification
      * ViewerExperienceSection.justification
      * SourceOfInspirationSection.justification
      * CoreConcept.explanation_and_justification
      * MajorTheme / MajorLessonLearned merged into ThematicConcept

    - ReceptionOutput uses a dual-zone structure:
      Extraction zone (4 nullable observation fields consumed by Wave 2
      generators, NOT embedded): source_material_hint, thematic_observations,
      emotional_observations, craft_observations.
      Synthesis zone (embedded into reception_vectors): reception_summary,
      praised_qualities, criticized_qualities.

    - ProductionKeywordsOutput and SourceOfInspirationOutput are
      SEPARATE schemas (they're separate LLM calls). The existing
      ProductionMetadata merges them awkwardly into one model.

The existing schemas in implementation/classes/schemas.py remain
unchanged -- they're consumed by the search pipeline for reading
metadata from Qdrant. These generation-side schemas can evolve
independently. When deploying, align the search-side schemas.

Each schema class implements __str__() for vector text generation
(lowercased, concatenated terms) matching the existing pattern.

Pydantic's type_to_response_format_param() is used in generators
to convert these schemas into the json_schema format required by
the Batch API's response_format field.
"""

from pydantic import BaseModel, Field, ConfigDict, constr, conlist


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class TermsSection(BaseModel):
    """A section containing search-query-like term phrases.

    Used by WatchContext, NarrativeTechniques, and ProductionKeywords
    where each section is a flat list of terms without negations.
    No justification field — removed per spec Decision 5.
    """
    model_config = ConfigDict(extra="forbid")

    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description="Search-query-like phrases.",
    )


class TermsWithNegationsSection(BaseModel):
    """A section containing both positive terms and negation phrases.

    Used by ViewerExperience where each section captures what the movie
    IS and what it is NOT. No justification field — removed per spec
    Decision 5.
    """
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


class OptionalTermsWithNegationsSection(BaseModel):
    """Wrapper for viewer experience sections that may not apply.

    The LLM sets should_skip=True when the section is irrelevant
    for a given movie (e.g. disturbance_profile for a children's film).
    """
    model_config = ConfigDict(extra="forbid")

    should_skip: bool = Field(
        False,
        description=(
            "Set to true only if this section is not applicable "
            "to this movie. All data in section_data will be ignored."
        ),
    )
    section_data: TermsWithNegationsSection


# ---------------------------------------------------------------------------
# Wave 1: Plot Events
# ---------------------------------------------------------------------------

class PlotEventsOutput(BaseModel):
    """Structured output from the plot_events generation (Wave 1).

    Produces a single chronological plot summary. The plot_summary
    field is passed directly to all downstream Wave 2 consumers.

    Setting and major_characters fields were removed after evaluation
    showed setting is redundant (already in the summary) and structured
    character extraction adds analytical burden better handled by the
    downstream plot_analysis generator. Character names appear naturally
    in the plot_summary text.

    Model: gpt-5-mini, reasoning_effort: minimal
    """
    model_config = ConfigDict(extra="forbid")

    plot_summary: constr(strip_whitespace=True, min_length=1) = Field(
        ..., description="Chronological plot summary.",
    )

    def __str__(self) -> str:
        return self.plot_summary.lower()


# ---------------------------------------------------------------------------
# Wave 1: Reception
# ---------------------------------------------------------------------------

class ReceptionOutput(BaseModel):
    """Structured output from the reception generation (Wave 1).

    Dual-zone output structure:

    Extraction zone (NOT embedded, consumed by Wave 2 generators):
        source_material_hint — short classifying phrase for adaptations/remakes
        thematic_observations — themes, meaning, messages from reviews
        emotional_observations — emotional tone, mood, atmosphere
        craft_observations — narrative structure, pacing, performances as craft

    Synthesis zone (embedded into reception_vectors):
        reception_summary — evaluative summary of audience opinion
        praised_qualities — tag phrases for what audiences liked
        criticized_qualities — tag phrases for what audiences disliked

    Fields are ordered extraction-first so that structured output
    models produce observations before synthesizing evaluative content.

    Model: gpt-5-mini, reasoning_effort: low
    """
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


# ---------------------------------------------------------------------------
# Wave 2: Plot Analysis
# ---------------------------------------------------------------------------


class CharacterArcWithReasoning(BaseModel):
    """Character arc with a reasoning field for chain-of-thought quality.

    Used only in the justification/evaluation variant. The reasoning
    field is generated first to scaffold a better label. Only the
    label is embedded — reasoning is never included in embedding text.
    """
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


class ElevatorPitchWithJustification(BaseModel):
    """Elevator pitch with an explanation field for chain-of-thought quality."""
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


class ThematicConceptWithJustification(BaseModel):
    """A thematic concept (theme or lesson) with a justification field.

    Replaces both MajorThemeWithJustification and
    MajorLessonLearnedWithJustification — the theme/lesson distinction
    was ambiguous for small LLMs and irrelevant for vector search.
    The justification text is NEVER embedded — only the concept_label
    is used in embedding text.
    """
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


class PlotAnalysisWithJustificationsOutput(BaseModel):
    """Production output schema for plot_analysis generation (Wave 2).

    Uses sub-models with explanation_and_justification / reasoning fields
    that scaffold better labels via chain-of-thought. Only the labels
    are embedded — justification text is never included in embedding text.

    The __str__() method produces IDENTICAL embedding text to
    PlotAnalysisOutput (labels only, no justifications).

    Field order is optimized for autoregressive generation: genre →
    themes → elevator pitch → conflict → arcs → overview.

    Model: gpt-5-mini, reasoning_effort: minimal, verbosity: low
    """
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
        # Must produce identical embedding text to PlotAnalysisOutput.__str__()
        parts = []
        if self.generalized_plot_overview:
            parts.append(self.generalized_plot_overview.lower())
        if self.elevator_pitch_with_justification:
            # ElevatorPitchWithJustification.__str__ returns only the pitch
            parts.append(str(self.elevator_pitch_with_justification).lower())
        if self.genre_signatures:
            parts.append(", ".join(self.genre_signatures).lower())
        if self.conflict_type:
            parts.append(", ".join(self.conflict_type).lower())
        if self.character_arcs:
            # CharacterArcWithReasoning.__str__ returns only the label
            parts.extend(str(arc).lower() for arc in self.character_arcs)
        if self.thematic_concepts:
            # ThematicConceptWithJustification.__str__ returns only concept_label
            parts.extend(str(t).lower() for t in self.thematic_concepts)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Wave 2: Viewer Experience
# ---------------------------------------------------------------------------

class ViewerExperienceOutput(BaseModel):
    """Structured output from the viewer_experience generation (Wave 2).

    8 sections capturing the emotional/sensory viewing experience.
    All sections use the same flat TermsWithNegationsSection with
    min_length=0 on both terms and negations. Sections that don't
    apply to a movie (e.g. disturbance_profile for a children's film)
    produce empty lists — no should_skip wrapper needed.

    Non-production variant (no justifications). Kept for backward
    compatibility and evaluation comparison. Production uses
    ViewerExperienceWithJustificationsOutput.

    Model: gpt-5-mini, reasoning_effort: minimal
    """
    model_config = ConfigDict(extra="forbid")

    emotional_palette: TermsWithNegationsSection
    tension_adrenaline: TermsWithNegationsSection
    tone_self_seriousness: TermsWithNegationsSection
    cognitive_complexity: TermsWithNegationsSection
    disturbance_profile: TermsWithNegationsSection
    sensory_load: TermsWithNegationsSection
    emotional_volatility: TermsWithNegationsSection
    ending_aftertaste: TermsWithNegationsSection

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


# -- With-justifications variant for evaluation comparison --
# These section models add a justification field that the LLM fills via
# structured output. The justification text is NEVER embedded —
# __str__() produces identical output to ViewerExperienceOutput.
# This lets us compare output quality with vs. without justifications
# using the same prompt.


class TermsWithNegationsAndJustificationSection(BaseModel):
    """TermsWithNegationsSection + justification for evaluation comparison.

    The justification field provides chain-of-thought that may improve
    output quality. It is NEVER embedded — only terms and negations
    are used in embedding text.
    """
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


class OptionalTermsWithNegationsAndJustificationSection(BaseModel):
    """Wrapper for viewer experience sections with justification that may not apply.

    The LLM sets should_skip=True when the section is irrelevant
    for a given movie (e.g. disturbance_profile for a children's film).
    """
    model_config = ConfigDict(extra="forbid")

    should_skip: bool = Field(
        False,
        description=(
            "Set to true only if this section is not applicable "
            "to this movie. All data in section_data will be ignored."
        ),
    )
    section_data: TermsWithNegationsAndJustificationSection


class ViewerExperienceWithJustificationsOutput(BaseModel):
    """Production output schema for viewer_experience generation (Wave 2).

    Identical output structure to ViewerExperienceOutput but uses section
    models that include a justification field. Justifications provide
    chain-of-thought that improves specificity (+0.44) and term diversity
    (+0.38) per Round 2 evaluation. Justifications are discarded before
    embedding — no retrieval impact.

    The __str__() method produces IDENTICAL embedding text to
    ViewerExperienceOutput — justification text is never embedded.

    Model: gpt-5-mini, reasoning_effort: minimal, verbosity: low
    """
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
        # Must produce identical embedding text to ViewerExperienceOutput.__str__()
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


# ---------------------------------------------------------------------------
# Wave 2: Watch Context
# ---------------------------------------------------------------------------

class WatchContextOutput(BaseModel):
    """Structured output from the watch_context generation (Wave 2).

    4 sections capturing when/why/how to watch the movie. Deliberately
    receives NO plot information (Decision 2) — focuses on experiential
    attributes.

    Model: gpt-5-mini, reasoning_effort: medium
    """
    model_config = ConfigDict(extra="forbid")

    self_experience_motivations: TermsSection
    external_motivations: TermsSection
    key_movie_feature_draws: TermsSection
    watch_scenarios: TermsSection

    def __str__(self) -> str:
        combined_terms = (
            self.self_experience_motivations.terms
            + self.external_motivations.terms
            + self.key_movie_feature_draws.terms
            + self.watch_scenarios.terms
        )
        return ", ".join(t.lower() for t in combined_terms)


# -- With-justifications variant for evaluation comparison --
# Adds a justification field to TermsSection for chain-of-thought quality.
# The justification text is NEVER embedded — __str__() produces identical
# output to WatchContextOutput. This lets us compare output quality with
# vs. without justifications using the same prompt.


class TermsWithJustificationSection(BaseModel):
    """TermsSection + justification for evaluation comparison.

    The justification field provides chain-of-thought that may improve
    output quality. It is NEVER embedded — only terms are used in
    embedding text. Mirrors the search-side GenericTermsSection structure.
    """
    model_config = ConfigDict(extra="forbid")

    justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "1 sentence. Why you chose these terms for this section. "
            "Not used for embeddings."
        ),
    )
    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description="Search-query-like phrases.",
    )


class WatchContextWithJustificationsOutput(BaseModel):
    """Watch context variant WITH justification fields.

    Identical output structure to WatchContextOutput but uses section
    models that include a justification field. Used during evaluation to
    compare output quality with vs. without justifications.

    The __str__() method produces IDENTICAL embedding text to
    WatchContextOutput — justification text is never embedded.

    Model: gpt-5-mini, reasoning_effort: medium
    """
    model_config = ConfigDict(extra="forbid")

    self_experience_motivations: TermsWithJustificationSection
    external_motivations: TermsWithJustificationSection
    key_movie_feature_draws: TermsWithJustificationSection
    watch_scenarios: TermsWithJustificationSection

    def __str__(self) -> str:
        # Must produce identical embedding text to WatchContextOutput.__str__()
        combined_terms = (
            self.self_experience_motivations.terms
            + self.external_motivations.terms
            + self.key_movie_feature_draws.terms
            + self.watch_scenarios.terms
        )
        return ", ".join(t.lower() for t in combined_terms)


# ---------------------------------------------------------------------------
# Wave 2: Narrative Techniques
# ---------------------------------------------------------------------------

class NarrativeTechniquesOutput(BaseModel):
    """Structured output from the narrative_techniques generation (Wave 2).

    11 sections capturing storytelling structure, POV, delivery
    mechanism, and narrative devices. Field order is optimized for
    autoregressive generation: easiest/most-concrete sections first,
    building toward harder/more-abstract ones.

    Model: gpt-5-mini, reasoning_effort: medium
    """
    model_config = ConfigDict(extra="forbid")

    # Easiest to identify from any input type
    narrative_archetype: TermsSection
    narrative_delivery: TermsSection
    additional_plot_devices: TermsSection
    # Moderate — often evidenced in craft observations
    pov_perspective: TermsSection
    characterization_methods: TermsSection
    character_arcs: TermsSection
    audience_character_perception: TermsSection
    # Hardest — require plot knowledge or synthesis
    information_control: TermsSection
    conflict_stakes_design: TermsSection
    thematic_delivery: TermsSection
    # Rare — already allows 0 entries
    meta_techniques: TermsSection

    def __str__(self) -> str:
        combined_terms: list[str] = []
        for section in (
            self.narrative_archetype,
            self.narrative_delivery,
            self.additional_plot_devices,
            self.pov_perspective,
            self.characterization_methods,
            self.character_arcs,
            self.audience_character_perception,
            self.information_control,
            self.conflict_stakes_design,
            self.thematic_delivery,
            self.meta_techniques,
        ):
            combined_terms.extend(section.terms)
        return ", ".join(t.lower() for t in combined_terms)


# -- With-justifications variant for evaluation comparison --
# Adds a justification field to TermsSection for chain-of-thought quality.
# The justification text is NEVER embedded — __str__() produces identical
# output to NarrativeTechniquesOutput. This lets us compare output quality
# with vs. without justifications using the same prompt.


class NarrativeTechniquesWithJustificationsOutput(BaseModel):
    """Narrative techniques variant WITH justification fields.

    Identical output structure to NarrativeTechniquesOutput but uses section
    models that include a justification field. Used during evaluation to
    compare output quality with vs. without justifications.

    The __str__() method produces IDENTICAL embedding text to
    NarrativeTechniquesOutput — justification text is never embedded.

    Model: gpt-5-mini, reasoning_effort: medium
    """
    model_config = ConfigDict(extra="forbid")

    # Field order matches NarrativeTechniquesOutput (cognitive scaffolding)
    narrative_archetype: TermsWithJustificationSection
    narrative_delivery: TermsWithJustificationSection
    additional_plot_devices: TermsWithJustificationSection
    pov_perspective: TermsWithJustificationSection
    characterization_methods: TermsWithJustificationSection
    character_arcs: TermsWithJustificationSection
    audience_character_perception: TermsWithJustificationSection
    information_control: TermsWithJustificationSection
    conflict_stakes_design: TermsWithJustificationSection
    thematic_delivery: TermsWithJustificationSection
    meta_techniques: TermsWithJustificationSection

    def __str__(self) -> str:
        # Must produce identical embedding text to NarrativeTechniquesOutput.__str__()
        combined_terms: list[str] = []
        for section in (
            self.narrative_archetype,
            self.narrative_delivery,
            self.additional_plot_devices,
            self.pov_perspective,
            self.characterization_methods,
            self.character_arcs,
            self.audience_character_perception,
            self.information_control,
            self.conflict_stakes_design,
            self.thematic_delivery,
            self.meta_techniques,
        ):
            combined_terms.extend(section.terms)
        return ", ".join(t.lower() for t in combined_terms)


# ---------------------------------------------------------------------------
# Wave 2: Production Keywords (separate LLM call)
# ---------------------------------------------------------------------------

class ProductionKeywordsOutput(BaseModel):
    """Structured output from the production_keywords generation (Wave 2).

    Classification task: the LLM filters merged_keywords to keep only
    production-relevant ones. Not generative — purely selective.

    Model: gpt-5-mini, reasoning_effort: low
    """
    model_config = ConfigDict(extra="forbid")

    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description="Production-relevant keywords filtered from the input list.",
    )

    def __str__(self) -> str:
        return ", ".join(t.lower() for t in self.terms)


# -- With-justifications variant for evaluation comparison --
# Adds a justification field for chain-of-thought quality. The justification
# text is NEVER embedded — __str__() produces identical output to
# ProductionKeywordsOutput. Mirrors the search-side GenericTermsSection
# which has a justification field.


class ProductionKeywordsWithJustificationsOutput(BaseModel):
    """Production keywords variant WITH justification field.

    Identical output structure to ProductionKeywordsOutput but adds a
    justification field. Used during evaluation to compare output quality
    with vs. without justifications.

    The __str__() method produces IDENTICAL embedding text to
    ProductionKeywordsOutput — justification text is never embedded.

    Model: gpt-5-mini, reasoning_effort: low
    """
    model_config = ConfigDict(extra="forbid")

    justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "1 sentence. Why you chose these production-relevant keywords. "
            "Not used for embeddings."
        ),
    )
    terms: list[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description="Production-relevant keywords filtered from the input list.",
    )

    def __str__(self) -> str:
        # Must produce identical embedding text to ProductionKeywordsOutput.__str__()
        return ", ".join(t.lower() for t in self.terms)


# ---------------------------------------------------------------------------
# Wave 2: Source of Inspiration (separate LLM call)
# ---------------------------------------------------------------------------

class SourceOfInspirationOutput(BaseModel):
    """Structured output from the source_of_inspiration generation (Wave 2).

    Identifies source material and production medium. ONLY generation
    where parametric knowledge is explicitly allowed — the LLM may use
    its training data for source material facts.

    Model: gpt-5-mini, reasoning_effort: low
    """
    model_config = ConfigDict(extra="forbid")

    sources_of_inspiration: list[str] = Field(
        default_factory=list,
        description=(
            "Source material: novel, true story, original screenplay, "
            "remake of [film], etc."
        ),
    )
    production_mediums: list[str] = Field(
        default_factory=list,
        description=(
            "Production medium: live-action, animation, CGI, "
            "practical effects, stop-motion, etc."
        ),
    )

    def __str__(self) -> str:
        all_terms = self.sources_of_inspiration + self.production_mediums
        return ", ".join(t.lower() for t in all_terms)


# -- With-justifications variant for evaluation comparison --
# Adds a single justification field covering both sources_of_inspiration
# and production_mediums. Mirrors the search-side SourceOfInspirationSection
# which has a justification field. The justification text is NEVER embedded.


class SourceOfInspirationWithJustificationsOutput(BaseModel):
    """Source of inspiration variant WITH justification field.

    Identical output structure to SourceOfInspirationOutput but adds a
    single justification field covering both lists. Used during evaluation
    to compare output quality with vs. without justifications.

    The __str__() method produces IDENTICAL embedding text to
    SourceOfInspirationOutput — justification text is never embedded.

    Model: gpt-5-mini, reasoning_effort: low
    """
    model_config = ConfigDict(extra="forbid")

    justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "1-2 sentences. Why you chose these sources and production "
            "mediums, citing concrete evidence. Not used for embeddings."
        ),
    )
    sources_of_inspiration: list[str] = Field(
        default_factory=list,
        description=(
            "Source material: novel, true story, original screenplay, "
            "remake of [film], etc."
        ),
    )
    production_mediums: list[str] = Field(
        default_factory=list,
        description=(
            "Production medium: live-action, animation, CGI, "
            "practical effects, stop-motion, etc."
        ),
    )

    def __str__(self) -> str:
        # Must produce identical embedding text to SourceOfInspirationOutput.__str__()
        all_terms = self.sources_of_inspiration + self.production_mediums
        return ", ".join(t.lower() for t in all_terms)
