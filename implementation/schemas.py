"""
Pydantic schemas for data models and LLM response structures.

This module contains Pydantic models used for structured outputs from LLM API calls
and other data model classes used throughout the project.
"""

from typing import List, Literal
from pydantic import BaseModel, Field, conlist, constr, ConfigDict
from enum import Enum
from .enums import WatchProviderType

# CLASS DEFINITIONS

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
            "1 short sentence stating what the character aims to achieve overall and why (high-level)."
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
            parts.append(f"{self.plot_summary}")
        if self.setting:
            parts.append(f"{self.setting}")
        if self.major_characters:
            stringified_characters = [str(character) for character in self.major_characters]
            parts.append(f"{"\n".join(stringified_characters)}")
        return "\n".join(parts)


# -----------------------------
# Section B — What Type
# -----------------------------


class PlotAnalysisMetadata(BaseModel):
    core_engine: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
    )
    genre_signature: conlist(constr(strip_whitespace=True, min_length=1), min_length=1, max_length=3) = Field(
        ...,
    )
    generalized_plot_overview: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
    )
    conflict_scale: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
    )
    narrative_delivery: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
    )
    character_arc_shapes: conlist(constr(strip_whitespace=True, min_length=1), min_length=1, max_length=2) = Field(
        default_factory=list,
    )
    narrative_archetype: constr(strip_whitespace=True, min_length=1) = Field(
        ...,
    )
    themes_primary: conlist(constr(strip_whitespace=True, min_length=1), min_length=2, max_length=2) = Field(
        ...,
    )
    lessons_learned: conlist(constr(strip_whitespace=True, min_length=1), min_length=0, max_length=2) = Field(
        default_factory=list,
    )

    def __str__(self) -> str:
        parts = []
        if self.generalized_plot_overview:
            parts.append(f"{self.generalized_plot_overview}")
        if self.core_engine:
            parts.append(f"{self.core_engine}")
        if self.genre_signature:
            parts.append(f"{", ".join(self.genre_signature)}")
        if self.conflict_scale:
            parts.append(f"{self.conflict_scale} conflict")
        if self.narrative_delivery:
            parts.append(f"{self.narrative_delivery} narrative")
        if self.character_arc_shapes:
            parts.append(f"{", ".join(self.character_arc_shapes)}")
        if self.narrative_archetype:
            parts.append(f"{self.narrative_archetype}")
        if self.themes_primary:
            parts.append(f"{", ".join(self.themes_primary)}")
        if self.lessons_learned:
            parts.append(f"{", ".join(self.lessons_learned)}")
        return "\n".join(parts)

class VibeMetadata(BaseModel):
    """
    Structured response schema for DenseVibe generation.
    
    Used for LLM-generated vibe metadata as specified in the movie search guide (section 5.3).
    Contains viewer experience descriptors for semantic search matching.
    
    Each field is a short, descriptive string capturing a distinct aspect of the watch experience:
    - dominant_mood: The prevailing atmosphere and emotional tone during viewing
    - movie_energy: How the movie flows in terms of pacing, momentum, and energy
    - intensity_and_driver: Overall intensity level and what primarily creates it
    - romance_humor_sexuality: The presence of romance, humor, and sexuality in the movie
    - final_viewer_impression: What the movie leaves you feeling like
    - viewing_context: The ideal viewing context or what the movie asks of the viewer
    """
    dominant_mood: str
    movie_energy: str
    intensity: str
    romance_humor_sexuality: str
    final_viewer_impression: str
    viewing_context: str


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
    types: list[WatchProviderType]


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

