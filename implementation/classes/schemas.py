"""
Pydantic schemas for data models and LLM response structures.

This module contains Pydantic models used for structured outputs from LLM API calls
and other data model classes used throughout the project.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, conlist, constr, ConfigDict, field_validator
from enum import Enum
from .enums import (
    WatchMethodType, 
    DateMatchOperation, 
    NumericalMatchOperation, 
    RatingMatchOperation, 
    ReceptionType, 
    StreamingAccessType, 
    Genre,
    EntityCategory,
    RelevanceSize,
)


# -----------------------------
#         PLOT EVENTS
# -----------------------------

class MajorCharacter(BaseModel):
    name: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Character name as it appears in the film."
    )
    description: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Who they are in plain terms (e.g., 'a disillusioned detective', "
            "'a widowed single parent'). Keep it short and plot-relevant."
        )
    )
    role: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Narrative role label describing their function in the story. "
            "Examples: 'protagonist', 'co-protagonist', 'antagonist', 'love interest', 'mentor', 'foil',"
        )
    )
    primary_motivations: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "1 concise sentence stating what the character aims to achieve overall and why (high-level)."
        )
    )

    def __str__(self) -> str:
        return f"{self.name}: {self.description} Motivations: {self.primary_motivations}"


class PlotEventsMetadata(BaseModel):
    plot_summary: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Detailed chronological, spoiler-containing plot summary of the entire movie preserving character names and locations."
        )
    )
    setting: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description=(
            "Short phrase (10 words or less). Specific where/when the story takes place (include proper nouns if meaningful). "
            "Focus on setting details that shape what happens."
        )
    )
    major_characters: List[MajorCharacter] = Field(
        default_factory=list,
        description="Additional details about the ABSOLUTELY ESSENTIAL characters in the story."
    )

    def __str__(self) -> str:
        parts = []
        if self.plot_summary:
            parts.append(f"{self.plot_summary.lower()}")
        if self.setting:
            parts.append(f"{self.setting.lower()}")
        if self.major_characters:
            stringified_characters = [str(character).lower() for character in self.major_characters]
            parts.append(f"{"\n".join(stringified_characters)}")
        return "\n".join(parts)


# -----------------------------
#       PLOT ANALYSIS
# -----------------------------

class CoreConcept(BaseModel):
    explanation_and_justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="One sentence explaining why this core concept is the best representation of the heart / core of this movie. Remove meta framing ('the story/movie'), articles, and filler."
    )
    core_concept_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="The single dominant story concept representing the heart of the movie. Simple, concrete terms."
    )

    def __str__(self) -> str:
        return f"{self.core_concept_label}: {self.explanation_and_justification}"

class CharacterArc(BaseModel):
    character_name: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="The name of the character who undergoes the arc transformation."
    )
    arc_transformation_description: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="One-sentence description of the character's arc transformation and why it's central to the themes and lessons of the movie."
    )
    arc_transformation_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="Generic, search-query-like phrase classifying the character's arc transformation."
    )

    def __str__(self) -> str:
        return self.arc_transformation_label

class MajorTheme(BaseModel):
    explanation_and_justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="A one-sentence explanation of the theme and why it's one of the most important central themes of the movie."
    )
    theme_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="High-signal label summarizing the theme for vector embedding. Use simple, generic, human-world-friendly terms."
    )

    def __str__(self) -> str:
        return self.theme_label

class MajorLessonLearned(BaseModel):
    explanation_and_justification: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="A one-sentence explanation of the lesson and why it's one of the most important central lessons of the movie."
    )
    lesson_label: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
        description="High-signal label summarizing the lesson for vector embedding. Use simple, generic, human-world-friendly terms."
    )

    def __str__(self) -> str:
        return self.lesson_label


class PlotAnalysisMetadata(BaseModel):
    core_concept: CoreConcept = Field(
        ...,
    )
    genre_signatures: conlist(constr(strip_whitespace=True, min_length=1), min_length=2, max_length=6) = Field(
        ...,
    )
    conflict_scale: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
    )
    character_arcs: conlist(CharacterArc, min_length=1, max_length=3) = Field(
        ...,
    )
    themes_primary: conlist(MajorTheme, min_length=1, max_length=3) = Field(
        ...,
    )
    lessons_learned: conlist(MajorLessonLearned, min_length=1, max_length=3) = Field(
        default_factory=list,
    )
    generalized_plot_overview: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
    )

    def __str__(self) -> str:
        parts = []
        if self.generalized_plot_overview:
            parts.append(f"{self.generalized_plot_overview.lower()}")
        if self.core_concept:
            parts.append(f"{str(self.core_concept).lower()}")
        if self.genre_signatures:
            parts.append(f"{", ".join(self.genre_signatures).lower()}")
        if self.conflict_scale:
            parts.append(f"{self.conflict_scale.lower()} conflict")
        if self.character_arcs:
            stringified_character_arcs = [str(character_arc).lower() for character_arc in self.character_arcs]
            parts.append(f"{"\n".join(stringified_character_arcs)}")
        if self.themes_primary:
            stringified_themes = [str(theme).lower() for theme in self.themes_primary]
            parts.append(f"{"\n".join(stringified_themes)}")
        if self.lessons_learned:
            stringified_lessons = [str(lesson).lower() for lesson in self.lessons_learned]
            parts.append(f"{"\n".join(stringified_lessons)}")
        return "\n".join(parts)


# -----------------------------
#      VIEWER EXPERIENCE
# -----------------------------

class ViewerExperienceSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    justification: str = Field(
        ...,
        description="1 sentence. Why you chose these terms and negations for this trait. Not used for embeddings."
    )
    terms: List[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description="Search-query-like phrases representing prominent characteristics of the movie that are relevant to this section."
    )
    negations: List[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description='Search-query-like "avoidance" phrases versions of what the movie does NOT have or is NOT like. ALWAYS has "not" or "no" in it.'
    )

class OptionalViewerExperienceSection(BaseModel):
    """
    Base shape for each viewer-experience trait.
    """
    model_config = ConfigDict(extra="forbid")

    should_skip: bool = Field(
        False,
        description="Set to true only if this section is not applicable to this movie. All data generated within this section will be ignored."
    )

    section_data: ViewerExperienceSection

class ViewerExperienceMetadata(BaseModel):
    """
    Structured output for the Viewer Experience vector.

    Embedding text should be derived only from:
      - section.terms
      - section.negations
    ...and only from sections where should_skip is False.
    """
    model_config = ConfigDict(extra="forbid")

    emotional_palette: ViewerExperienceSection
    tension_adrenaline: ViewerExperienceSection
    tone_self_seriousness: ViewerExperienceSection
    cognitive_complexity: ViewerExperienceSection
    disturbance_profile: OptionalViewerExperienceSection
    sensory_load: OptionalViewerExperienceSection
    emotional_volatility: OptionalViewerExperienceSection
    ending_aftertaste: ViewerExperienceSection

    def __str__(self) -> str:
        combined_terms = []
        # emotional palette
        combined_terms.extend(self.emotional_palette.terms)
        combined_terms.extend(self.emotional_palette.negations)
        # tension adrenaline
        combined_terms.extend(self.tension_adrenaline.terms)
        combined_terms.extend(self.tension_adrenaline.negations)
        # tone self seriousness
        combined_terms.extend(self.tone_self_seriousness.terms)
        combined_terms.extend(self.tone_self_seriousness.negations)
        # mind bend clarity
        combined_terms.extend(self.cognitive_complexity.terms)
        combined_terms.extend(self.cognitive_complexity.negations)
        # disturbance profile
        if not self.disturbance_profile.should_skip:
            combined_terms.extend(self.disturbance_profile.section_data.terms)
            combined_terms.extend(self.disturbance_profile.section_data.negations)
        # sensory intensity
        if not self.sensory_load.should_skip:
            combined_terms.extend(self.sensory_load.section_data.terms)
            combined_terms.extend(self.sensory_load.section_data.negations)
        # emotional volatility
        if not self.emotional_volatility.should_skip:
            combined_terms.extend(self.emotional_volatility.section_data.terms)
            combined_terms.extend(self.emotional_volatility.section_data.negations)
        # ending aftertaste
        combined_terms.extend(self.ending_aftertaste.terms)
        combined_terms.extend(self.ending_aftertaste.negations)
        return ", ".join(combined_terms)
    

# -----------------------------
#        WATCH CONTEXT
# -----------------------------

class GenericTermsSection(BaseModel):
    """
    Base shape for each watch context section.
    """
    model_config = ConfigDict(extra="forbid")

    justification: str = Field(
        ...,
        description="1 sentence. Why you chose these terms for this trait. Not used for embeddings."
    )
    terms: List[constr(strip_whitespace=True, min_length=1)] = Field(
        default_factory=list,
        description="Search-query-like phrases."
    )

class WatchContextMetadata(BaseModel):
    """
    Structured output for the Viewer Experience vector.

    Embedding text should be derived only from:
      - section.terms
      - section.negations
    ...and only from sections where should_skip is False.
    """
    model_config = ConfigDict(extra="forbid")

    self_experience_motivations: GenericTermsSection
    external_motivations: GenericTermsSection
    key_movie_feature_draws: GenericTermsSection
    watch_scenarios: GenericTermsSection

    def __str__(self) -> str:
        combined_terms = (
            self.self_experience_motivations.terms 
            + self.external_motivations.terms 
            + self.key_movie_feature_draws.terms 
            + self.watch_scenarios.terms
        )
        return ", ".join(combined_terms)


# -----------------------------
#      NARRATIVE TECHNIQUES
# -----------------------------

class NarrativeTechniquesMetadata(BaseModel):
    """
    Structured output for the Viewer Experience vector.

    Embedding text should be derived only from:
      - section.terms
      - section.negations
    ...and only from sections where should_skip is False.
    """
    model_config = ConfigDict(extra="forbid")

    pov_perspective: GenericTermsSection
    narrative_delivery: GenericTermsSection
    narrative_archetype: GenericTermsSection
    information_control: GenericTermsSection
    characterization_methods: GenericTermsSection
    character_arcs: GenericTermsSection
    audience_character_perception: GenericTermsSection
    conflict_stakes_design: GenericTermsSection
    thematic_delivery: GenericTermsSection
    meta_techniques: GenericTermsSection
    additional_plot_devices: GenericTermsSection

    def __str__(self) -> str:
        combined_terms = (
            self.pov_perspective.terms
            + self.narrative_delivery.terms
            + self.narrative_archetype.terms
            + self.information_control.terms
            + self.characterization_methods.terms
            + self.character_arcs.terms
            + self.audience_character_perception.terms
            + self.conflict_stakes_design.terms
            + self.thematic_delivery.terms
            + self.meta_techniques.terms
            + self.additional_plot_devices.terms
        )
        return ", ".join(combined_terms)


# -----------------------------
#         PRODUCTION
# -----------------------------

class SourceOfInspirationSection(BaseModel):
    """
    LLM Generated Metadata for Source of Inspiration Vector
    """
    model_config = ConfigDict(extra="forbid")

    justification: str = Field(
        ...,
        description="Two sentence justification for your decisions citing concrete evidence."
    )
    sources_of_inspiration: List[str]
    production_mediums: List[str]

class ProductionMetadata(BaseModel):
    """
    LLM Generated Metadata for Production Vector
    """
    model_config = ConfigDict(extra="forbid")

    production_keywords: GenericTermsSection
    sources_of_inspiration: SourceOfInspirationSection

    def __str__(self) -> str:
        combined_terms = (
            self.production_keywords.terms
            + self.sources_of_inspiration.sources_of_inspiration
            + self.sources_of_inspiration.production_mediums
        )
        return ", ".join(combined_terms)


# -----------------------------
#          RECEPTION
# -----------------------------

class ReceptionMetadata(BaseModel):
    """
    LLM Generated Metadata for Reception Vector
    """
    model_config = ConfigDict(extra="forbid")

    new_reception_summary: str
    praise_attributes: List[str]
    complaint_attributes: List[str]


# -----------------------------
#     METADATA PREFERENCES
# -----------------------------

class DatePreferenceResult(BaseModel):
    """Inner result object containing date preference fields."""
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    first_date: str = Field(
        ...,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Either the first date in the range or the exact date to match. ISO 8601 date: YYYY-MM-DD",
    )
    match_operation: DateMatchOperation = Field(..., description="Whether we want the date to be before, after, or exactly at the first date, or between the two provided dates.")
    second_date: Optional[str] = Field(
        default=None,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="Optional second date in the range only if match_operation is BETWEEN. ISO 8601 date: YYYY-MM-DD"
    )


class DatePreference(BaseModel):
    """Date preference wrapper with optional result."""
    model_config = ConfigDict(extra="forbid")
    result: Optional[DatePreferenceResult]


class NumericalPreferenceResult(BaseModel):
    """Inner result object containing numerical preference fields."""
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    first_value: float = Field(..., description="Either the first value in the range or the exact value to match.")
    match_operation: NumericalMatchOperation = Field(..., description="How we should evaluate the provided first and (maybe) second values.")
    second_value: Optional[float] = Field(default=None, description="Optional second value in the range only if match_operation is BETWEEN.")


class NumericalPreference(BaseModel):
    """Numerical preference wrapper with optional result."""
    model_config = ConfigDict(extra="forbid")
    result: Optional[NumericalPreferenceResult]


class ListPreferenceResult(BaseModel):
    """Inner result object containing list preference fields."""
    model_config = ConfigDict(extra="forbid")
    should_include: List[str] = Field(default=[], description="List of items that should be included in the movie's metadata.")
    should_exclude: List[str] = Field(default=[], description="List of items that should be excluded from the movie's metadata.")


class ListPreference(BaseModel):
    """List preference wrapper with optional result."""
    model_config = ConfigDict(extra="forbid")
    result: Optional[ListPreferenceResult]


class GenreListPreferenceResult(BaseModel):
    """Inner result object containing genre list preference fields."""
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    should_include: List[Genre] = Field(default=[], description="List of genres that the user's query wants the movie to fall under.")
    should_exclude: List[Genre] = Field(default=[], description="List of genres that the user's query wants to avoid in the movie.")


class GenreListPreference(BaseModel):
    """Genre list preference wrapper with optional result."""
    model_config = ConfigDict(extra="forbid")
    result: Optional[GenreListPreferenceResult]


class MaturityPreferenceResult(BaseModel):
    """Inner result object containing maturity preference fields."""
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    rating: str = Field(..., description="Standard USA ratings: G, PG, PG-13, R, NC-17")
    match_operation: RatingMatchOperation = Field(..., description="Whether we prefer movies with this rating, greater (more mature), or less (less mature).")


class MaturityPreference(BaseModel):
    """Maturity preference wrapper with optional result."""
    model_config = ConfigDict(extra="forbid")
    result: Optional[MaturityPreferenceResult]


class WatchProvidersPreferenceResult(BaseModel):
    """Inner result object containing watch providers preference fields."""
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    should_include: List[str]
    should_exclude: List[str]
    preferred_access_type: Optional[StreamingAccessType]


class WatchProvidersPreference(BaseModel):
    """Watch providers preference wrapper with optional result."""
    model_config = ConfigDict(extra="forbid")
    result: Optional[WatchProvidersPreferenceResult]

class PopularTrendingPreference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prefers_trending_movies: bool
    prefers_popular_movies: bool

class ReceptionPreference(BaseModel):
    model_config = ConfigDict(use_enum_values=True, extra="forbid")
    reception_type: ReceptionType

class MetadataPreferencesResponse(BaseModel):
    release_date_preference: DatePreference
    duration_preference: NumericalPreference
    genres_preference: GenreListPreference
    audio_languages_preference: ListPreference
    watch_providers_preference: WatchProvidersPreference
    maturity_rating_preference: MaturityPreference
    popular_trending_preference: PopularTrendingPreference
    reception_preference: ReceptionPreference


# -----------------------------
#   LEXICAL ENTITY EXTRACTION
# -----------------------------

class ExtractedEntityData(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    candidate_entity_phrase: str
    most_likely_category: EntityCategory
    exclude_from_results: bool
    corrected_and_normalized_entity: str

class ExtractedEntitiesResponse(BaseModel):
    entity_candidates: List[ExtractedEntityData]


# -----------------------------
#       VECTOR ROUTING
# -----------------------------

class VectorCollectionSubqueryData(BaseModel):
    justification: str
    relevant_subquery_text: Optional[str]

class VectorSubqueriesResponse(BaseModel):
    plot_events_data: VectorCollectionSubqueryData
    plot_analysis_data: VectorCollectionSubqueryData
    viewer_experience_data: VectorCollectionSubqueryData
    watch_context_data: VectorCollectionSubqueryData
    narrative_techniques_data: VectorCollectionSubqueryData
    production_data: VectorCollectionSubqueryData
    reception_data: VectorCollectionSubqueryData


# -----------------------------
#   VECTOR COLLECTION WEIGHTS
# -----------------------------

class VectorCollectionWeightData(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    relevance: RelevanceSize
    justification: str

class VectorWeightsResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    plot_events_data: VectorCollectionWeightData
    plot_analysis_data: VectorCollectionWeightData
    viewer_experience_data: VectorCollectionWeightData
    watch_context_data: VectorCollectionWeightData
    narrative_techniques_data: VectorCollectionWeightData
    production_data: VectorCollectionWeightData
    reception_data: VectorCollectionWeightData

# -----------------------------
#       CHANNEL WEIGHTS
# -----------------------------

class ChannelWeightsResponse(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    lexical_relevance: RelevanceSize
    metadata_relevance: RelevanceSize
    vector_relevance: RelevanceSize


# -----------------------------
#  OVERALL QUERY UNDERSTANDING
# -----------------------------

class QueryUnderstandingResponse(BaseModel):
    channel_weights: ChannelWeightsResponse
    lexical_entities: ExtractedEntitiesResponse
    metadata_preferences: MetadataPreferencesResponse
    vector_subqueries: VectorSubqueriesResponse
    vector_weights: VectorWeightsResponse


# -----------------------------
#        MISC CLASSES
# -----------------------------

class WatchProvider(BaseModel):
    """
    Represents a watch provider (streaming service, rental platform, etc.).
    
    Configured to serialize enums as their values for JSON compatibility.
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: int
    name: str
    logo_path: str
    display_priority: int
    types: list[WatchMethodType]

    @field_validator("types", mode="before")
    @classmethod
    def parse_types(cls, v: list) -> list[WatchMethodType]:
        """
        Convert string values to WatchMethodType enum values.
        Filters out any invalid types that don't match known enum values.
        """
        if not isinstance(v, list):
            return []
        
        result = []
        for item in v:
            # If already a WatchMethodType, keep it as-is.
            if isinstance(item, WatchMethodType):
                result.append(item)
            # If it's a string, attempt to convert via from_string.
            elif isinstance(item, str):
                parsed = WatchMethodType.from_string(item)
                if parsed is not None:
                    result.append(parsed)
            # If it's an int, attempt to convert directly to WatchMethodType.
            elif isinstance(item, int):
                try:
                    result.append(WatchMethodType(item))
                except ValueError:
                    pass
        return result


class ParentalGuideItem(BaseModel):
    """
    Represents a parental guide item with category and severity information.
    """
    category: str
    severity: str


class ChromaVectorCollection(BaseModel):
    """
    Represents a collection of vectors fetched from ChromaDB.
    
    This class encapsulates all data returned from a ChromaDB collection,
    including vector IDs, embeddings, original documents, and metadata.
    Distances are optional and only included when returned from similarity searches.
    """
    ids: list[str]
    embeddings: list[list[float]]
    documents: list[str]
    metadatas: list[dict[str, str | int | float]]
    distances: list[float] | None = None


class IMDBFeaturedReview(BaseModel):
    """
    Represents a featured review from the reviews page.
    """
    summary: str
    text: str

class IMDBReviewTheme(BaseModel):
    """
    Represents a review theme from the user reception summary.
    """

    name: str
    sentiment: str

    def __str__(self) -> str:
        return f"Attribute: {self.name}; audience reception: {self.sentiment}"