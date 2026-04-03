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

from pydantic import BaseModel, Field, ConfigDict, constr, conlist

from implementation.misc.helpers import normalize_string


# ---------------------------------------------------------------------------
# Base class for all embeddable output schemas
# ---------------------------------------------------------------------------

class EmbeddableOutput(BaseModel):
    """Base class for metadata output schemas that produce embedding text.

    Every *Output schema must subclass this and implement embedding_text(),
    which returns the normalized string used for vector embedding.
    This replaces the previous __str__() convention.
    """

    @abstractmethod
    def embedding_text(self) -> str:
        """Return the normalized text to be embedded for this metadata type."""
        ...


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



# ---------------------------------------------------------------------------
# Wave 1: Plot Events
# ---------------------------------------------------------------------------

class PlotEventsOutput(EmbeddableOutput):
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

    def embedding_text(self) -> str:
        return normalize_string(self.plot_summary)


# ---------------------------------------------------------------------------
# Wave 1: Reception
# ---------------------------------------------------------------------------

class ReceptionOutput(EmbeddableOutput):
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

    def embedding_text(self) -> str:
        # Synthesis zone only — extraction-zone fields excluded
        parts = []
        if self.reception_summary:
            parts.append(self.reception_summary)
        if self.praised_qualities:
            parts.append(", ".join(self.praised_qualities))
        if self.criticized_qualities:
            parts.append(", ".join(self.criticized_qualities))
        return normalize_string("\n".join(parts))


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


class PlotAnalysisOutput(EmbeddableOutput):
    """Output schema for plot_analysis generation (Wave 2).

    Uses sub-models with explanation_and_justification / reasoning fields
    that scaffold better labels via chain-of-thought. Only the labels
    are embedded — justification text is never included in embedding text.

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

    def embedding_text(self) -> str:
        parts = []
        if self.generalized_plot_overview:
            parts.append(self.generalized_plot_overview)
        if self.elevator_pitch_with_justification:
            parts.append(self.elevator_pitch_with_justification.elevator_pitch)
        if self.genre_signatures:
            parts.append(", ".join(self.genre_signatures))
        if self.conflict_type:
            parts.append(", ".join(self.conflict_type))
        if self.character_arcs:
            parts.extend(arc.arc_transformation_label for arc in self.character_arcs)
        if self.thematic_concepts:
            parts.extend(t.concept_label for t in self.thematic_concepts)
        return normalize_string("\n".join(parts))


# ---------------------------------------------------------------------------
# Wave 2: Viewer Experience
# ---------------------------------------------------------------------------

class TermsWithNegationsAndJustificationSection(BaseModel):
    """Section with terms, negations, and a justification for chain-of-thought.

    The justification field provides chain-of-thought that improves
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


class ViewerExperienceOutput(EmbeddableOutput):
    """Output schema for viewer_experience generation (Wave 2).

    8 sections capturing the emotional/sensory viewing experience.
    Each section includes a justification field that provides
    chain-of-thought improving specificity and term diversity.
    Justifications are discarded before embedding — no retrieval impact.

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
        return normalize_string(", ".join(combined_terms))


# ---------------------------------------------------------------------------
# Wave 2: Watch Context
# ---------------------------------------------------------------------------

class TermsWithJustificationSection(BaseModel):
    """TermsSection with upstream evidence assessment.

    The evidence_basis field forces the model to inventory which specific
    input phrases support this section BEFORE generating terms. This is
    an upstream constraint — the model must identify concrete evidence
    first, then generate only terms that follow from that evidence.

    If no specific input phrases can be cited, terms should be empty.

    The evidence_basis text is NEVER embedded — only terms are used in
    embedding text.
    """
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


class WatchContextOutput(EmbeddableOutput):
    """Output schema for watch_context generation (Wave 2).

    Includes a brief identity_note pre-classification (2-8 words)
    that primes tone calibration before section generation, plus
    evidence_basis per section as an upstream constraint.

    4 sections capturing when/why/how to watch the movie. Deliberately
    receives NO plot information — focuses on experiential attributes.
    All sections allow 0 terms — sparse inputs should produce fewer
    terms, not fabricated ones.

    The identity_note is NOT embedded — only section terms are used
    in embedding text.

    Model: gpt-5-mini, reasoning_effort: low
    """
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
        combined_terms = (
            self.self_experience_motivations.terms
            + self.external_motivations.terms
            + self.key_movie_feature_draws.terms
            + self.watch_scenarios.terms
        )
        return normalize_string(", ".join(combined_terms))


# ---------------------------------------------------------------------------
# Wave 2: Narrative Techniques
# ---------------------------------------------------------------------------

class NarrativeTechniquesOutput(EmbeddableOutput):
    """Output schema for narrative_techniques generation (Wave 2).

    9 sections capturing storytelling structure, POV, delivery mechanism,
    and narrative devices. Each section includes a justification field
    for chain-of-thought quality. Justification text is NEVER embedded.

    Field order is optimized for autoregressive generation: specific
    sections first, catchall last.

    Model: gpt-5-mini, reasoning_effort: minimal
    """
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
        return normalize_string(", ".join(combined_terms))


# ---------------------------------------------------------------------------
# Wave 2: Production Keywords (separate LLM call)
# ---------------------------------------------------------------------------

class ProductionKeywordsOutput(EmbeddableOutput):
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

    def embedding_text(self) -> str:
        return normalize_string(", ".join(self.terms))


# ---------------------------------------------------------------------------
# Wave 2: Source of Inspiration (separate LLM call)
# ---------------------------------------------------------------------------

class SourceOfInspirationOutput(EmbeddableOutput):
    """Source of inspiration classification output.

    Two independent lists from the same inputs:
    - source_material: what existing media the film draws from
      (adaptations, remakes, reboots, reimaginings, spinoffs)
    - franchise_lineage: where the film sits in a franchise timeline
      (sequel, prequel, trilogy position, franchise starter)

    Parametric knowledge allowed at 95%+ confidence. Leaf-node
    classification — errors don't cascade to other generations.
    """
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
        all_terms = self.source_material + self.franchise_lineage
        return normalize_string(", ".join(all_terms))
