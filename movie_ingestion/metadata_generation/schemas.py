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
      * MajorTheme.explanation_and_justification
      * MajorLessonLearned.explanation_and_justification

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
    field becomes plot_synopsis for all downstream Wave 2 consumers.

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

class CharacterArc(BaseModel):
    """A key character transformation identified in plot analysis."""
    model_config = ConfigDict(extra="forbid")

    character_name: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="The name of the character who undergoes the arc.",
    )
    arc_transformation_description: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One-sentence description of the character's arc transformation "
            "and why it's central to the themes and lessons of the movie."
        ),
    )
    arc_transformation_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Generic, search-query-like phrase classifying the "
            "character's arc transformation."
        ),
    )

    def __str__(self) -> str:
        return self.arc_transformation_label


class PlotAnalysisOutput(BaseModel):
    """Structured output from the plot_analysis generation (Wave 2).

    Extracts thematic meaning, character arcs, and genre signatures
    from the plot synopsis. Justification/explanation fields removed
    per spec Decision 5 (core_concept, themes, lessons are flat
    strings). arc_transformation_description retained on CharacterArc
    — it aids label quality and is not a justification.

    Model: gpt-5-mini, reasoning_effort: low
    """
    model_config = ConfigDict(extra="forbid")

    core_concept_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "The single dominant story concept representing the heart "
            "of the movie. 6 words or less, simple concrete terms."
        ),
    )
    genre_signatures: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=2, max_length=6,
    ) = Field(
        ...,
        description="2-6 search-query-like genre phrases.",
    )
    conflict_scale: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Scale of consequences in the story's conflict.",
    )
    character_arcs: conlist(CharacterArc, min_length=1, max_length=3) = Field(
        ...,
        description="1-3 key character transformations.",
    )
    themes_primary: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=1, max_length=3,
    ) = Field(
        ...,
        description=(
            "1-3 core thematic concepts. High-signal labels using "
            "simple, generic, human-world-friendly terms."
        ),
    )
    lessons_learned: conlist(
        constr(strip_whitespace=True, min_length=1),
        max_length=3,
    ) = Field(
        default_factory=list,
        description=(
            "0-3 key takeaways / lessons. High-signal labels using "
            "simple, generic, human-world-friendly terms."
        ),
    )
    generalized_plot_overview: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="1-3 sentence thematic overview of the plot.",
    )

    def __str__(self) -> str:
        parts = []
        if self.generalized_plot_overview:
            parts.append(self.generalized_plot_overview.lower())
        if self.core_concept_label:
            parts.append(self.core_concept_label.lower())
        if self.genre_signatures:
            parts.append(", ".join(self.genre_signatures).lower())
        if self.conflict_scale:
            parts.append(f"{self.conflict_scale.lower()} conflict")
        if self.character_arcs:
            parts.extend(str(arc).lower() for arc in self.character_arcs)
        if self.themes_primary:
            parts.extend(t.lower() for t in self.themes_primary)
        if self.lessons_learned:
            parts.extend(l.lower() for l in self.lessons_learned)
        return "\n".join(parts)


# -- With-justifications variant for evaluation comparison --
# These sub-models add explanation_and_justification fields that the LLM
# fills via structured output. The justification text is NEVER embedded —
# __str__() methods return only the label. This lets us compare output
# quality with vs. without justifications using the same prompt.


class CoreConceptWithJustification(BaseModel):
    """Core concept with an explanation field for chain-of-thought quality."""
    model_config = ConfigDict(extra="forbid")

    explanation_and_justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "One sentence explaining why this core concept is the best "
            "representation of the heart / core of this movie. Remove "
            "meta framing ('the story/movie'), articles, and filler."
        ),
    )
    core_concept_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "The single dominant story concept representing the heart "
            "of the movie. Simple, concrete terms."
        ),
    )

    def __str__(self) -> str:
        # Only the label is embedded — justification aids generation quality
        return self.core_concept_label


class MajorThemeWithJustification(BaseModel):
    """A primary theme with an explanation field for chain-of-thought quality."""
    model_config = ConfigDict(extra="forbid")

    explanation_and_justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "A one-sentence explanation of the theme and why it's one "
            "of the most important central themes of the movie."
        ),
    )
    theme_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "High-signal label summarizing the theme for vector embedding. "
            "Use simple, generic, human-world-friendly terms."
        ),
    )

    def __str__(self) -> str:
        return self.theme_label


class MajorLessonLearnedWithJustification(BaseModel):
    """A lesson learned with an explanation field for chain-of-thought quality."""
    model_config = ConfigDict(extra="forbid")

    explanation_and_justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "A one-sentence explanation of the lesson and why it's one "
            "of the most important central lessons of the movie."
        ),
    )
    lesson_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "High-signal label summarizing the lesson for vector embedding. "
            "Use simple, generic, human-world-friendly terms."
        ),
    )

    def __str__(self) -> str:
        return self.lesson_label


class PlotAnalysisWithJustificationsOutput(BaseModel):
    """Plot analysis variant WITH justification/explanation fields.

    Identical output structure to PlotAnalysisOutput but uses sub-models
    that include explanation_and_justification fields. Used during
    evaluation to compare output quality with vs. without justifications.

    The __str__() method produces IDENTICAL embedding text to
    PlotAnalysisOutput — justification text is never embedded.

    Model: gpt-5-mini, reasoning_effort: low
    """
    model_config = ConfigDict(extra="forbid")

    core_concept: CoreConceptWithJustification = Field(
        ...,
        description=(
            "The single dominant story concept representing the heart "
            "of the movie. 6 words or less, simple concrete terms."
        ),
    )
    genre_signatures: conlist(
        constr(strip_whitespace=True, min_length=1),
        min_length=2, max_length=6,
    ) = Field(
        ...,
        description="2-6 search-query-like genre phrases.",
    )
    conflict_scale: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Scale of consequences in the story's conflict.",
    )
    character_arcs: conlist(CharacterArc, min_length=1, max_length=3) = Field(
        ...,
        description="1-3 key character transformations.",
    )
    themes_primary: conlist(
        MajorThemeWithJustification,
        min_length=1, max_length=3,
    ) = Field(
        ...,
        description=(
            "1-3 core thematic concepts. High-signal labels using "
            "simple, generic, human-world-friendly terms."
        ),
    )
    lessons_learned: conlist(
        MajorLessonLearnedWithJustification,
        max_length=3,
    ) = Field(
        default_factory=list,
        description=(
            "0-3 key takeaways / lessons. High-signal labels using "
            "simple, generic, human-world-friendly terms."
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
        if self.core_concept:
            # CoreConceptWithJustification.__str__ returns only the label
            parts.append(str(self.core_concept).lower())
        if self.genre_signatures:
            parts.append(", ".join(self.genre_signatures).lower())
        if self.conflict_scale:
            parts.append(f"{self.conflict_scale.lower()} conflict")
        if self.character_arcs:
            parts.extend(str(arc).lower() for arc in self.character_arcs)
        if self.themes_primary:
            # MajorThemeWithJustification.__str__ returns only theme_label
            parts.extend(str(t).lower() for t in self.themes_primary)
        if self.lessons_learned:
            # MajorLessonLearnedWithJustification.__str__ returns only lesson_label
            parts.extend(str(lesson).lower() for lesson in self.lessons_learned)
        return "\n".join(parts)


# ---------------------------------------------------------------------------
# Wave 2: Viewer Experience
# ---------------------------------------------------------------------------

class ViewerExperienceOutput(BaseModel):
    """Structured output from the viewer_experience generation (Wave 2).

    8 sections capturing the emotional/sensory viewing experience.
    3 sections (disturbance, sensory, volatility) are optional — the
    LLM sets should_skip=True when not applicable.

    Model: gpt-5-mini, reasoning_effort: low
    """
    model_config = ConfigDict(extra="forbid")

    emotional_palette: TermsWithNegationsSection
    tension_adrenaline: TermsWithNegationsSection
    tone_self_seriousness: TermsWithNegationsSection
    cognitive_complexity: TermsWithNegationsSection
    disturbance_profile: OptionalTermsWithNegationsSection
    sensory_load: OptionalTermsWithNegationsSection
    emotional_volatility: OptionalTermsWithNegationsSection
    ending_aftertaste: TermsWithNegationsSection

    def __str__(self) -> str:
        combined_terms: list[str] = []

        # Required sections — always included
        for section in (
            self.emotional_palette,
            self.tension_adrenaline,
            self.tone_self_seriousness,
            self.cognitive_complexity,
            self.ending_aftertaste,
        ):
            combined_terms.extend(section.terms)
            combined_terms.extend(section.negations)

        # Optional sections — included only when not skipped
        for optional in (
            self.disturbance_profile,
            self.sensory_load,
            self.emotional_volatility,
        ):
            if not optional.should_skip:
                combined_terms.extend(optional.section_data.terms)
                combined_terms.extend(optional.section_data.negations)

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
            "for this section. Not used for embeddings."
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
    """Viewer experience variant WITH justification fields.

    Identical output structure to ViewerExperienceOutput but uses section
    models that include a justification field. Used during evaluation to
    compare output quality with vs. without justifications.

    The __str__() method produces IDENTICAL embedding text to
    ViewerExperienceOutput — justification text is never embedded.

    Model: gpt-5-mini, reasoning_effort: low
    """
    model_config = ConfigDict(extra="forbid")

    emotional_palette: TermsWithNegationsAndJustificationSection
    tension_adrenaline: TermsWithNegationsAndJustificationSection
    tone_self_seriousness: TermsWithNegationsAndJustificationSection
    cognitive_complexity: TermsWithNegationsAndJustificationSection
    disturbance_profile: OptionalTermsWithNegationsAndJustificationSection
    sensory_load: OptionalTermsWithNegationsAndJustificationSection
    emotional_volatility: OptionalTermsWithNegationsAndJustificationSection
    ending_aftertaste: TermsWithNegationsAndJustificationSection

    def __str__(self) -> str:
        # Must produce identical embedding text to ViewerExperienceOutput.__str__()
        combined_terms: list[str] = []

        # Required sections — always included
        for section in (
            self.emotional_palette,
            self.tension_adrenaline,
            self.tone_self_seriousness,
            self.cognitive_complexity,
            self.ending_aftertaste,
        ):
            combined_terms.extend(section.terms)
            combined_terms.extend(section.negations)

        # Optional sections — included only when not skipped
        for optional in (
            self.disturbance_profile,
            self.sensory_load,
            self.emotional_volatility,
        ):
            if not optional.should_skip:
                combined_terms.extend(optional.section_data.terms)
                combined_terms.extend(optional.section_data.negations)

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
    mechanism, and narrative devices.

    Model: gpt-5-mini, reasoning_effort: medium
    """
    model_config = ConfigDict(extra="forbid")

    pov_perspective: TermsSection
    narrative_delivery: TermsSection
    narrative_archetype: TermsSection
    information_control: TermsSection
    characterization_methods: TermsSection
    character_arcs: TermsSection
    audience_character_perception: TermsSection
    conflict_stakes_design: TermsSection
    thematic_delivery: TermsSection
    meta_techniques: TermsSection
    additional_plot_devices: TermsSection

    def __str__(self) -> str:
        combined_terms: list[str] = []
        for section in (
            self.pov_perspective,
            self.narrative_delivery,
            self.narrative_archetype,
            self.information_control,
            self.characterization_methods,
            self.character_arcs,
            self.audience_character_perception,
            self.conflict_stakes_design,
            self.thematic_delivery,
            self.meta_techniques,
            self.additional_plot_devices,
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

    pov_perspective: TermsWithJustificationSection
    narrative_delivery: TermsWithJustificationSection
    narrative_archetype: TermsWithJustificationSection
    information_control: TermsWithJustificationSection
    characterization_methods: TermsWithJustificationSection
    character_arcs: TermsWithJustificationSection
    audience_character_perception: TermsWithJustificationSection
    conflict_stakes_design: TermsWithJustificationSection
    thematic_delivery: TermsWithJustificationSection
    meta_techniques: TermsWithJustificationSection
    additional_plot_devices: TermsWithJustificationSection

    def __str__(self) -> str:
        # Must produce identical embedding text to NarrativeTechniquesOutput.__str__()
        combined_terms: list[str] = []
        for section in (
            self.pov_perspective,
            self.narrative_delivery,
            self.narrative_archetype,
            self.information_control,
            self.characterization_methods,
            self.character_arcs,
            self.audience_character_perception,
            self.conflict_stakes_design,
            self.thematic_delivery,
            self.meta_techniques,
            self.additional_plot_devices,
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
