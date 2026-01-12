"""
Pydantic schemas for LLM response structures.

This module contains Pydantic models used for structured outputs from LLM API calls.
"""

from typing import Optional
from pydantic import BaseModel, ConfigDict
from vibe_enums import (
    MoodAtmosphere,
    TonalValence,
    PacingMomentum,
    KineticIntensity,
    TensionPressure,
    UnpredictabilityTwistiness,
    ScarinessLevel,
    FearMode,
    HumorLevel,
    HumorFlavor,
    ViolenceIntensity,
    GoreBodyGrossness,
    RomanceProminence,
    RomanceTone,
    SexualExplicitness,
    EroticCharge,
    SexualTone,
    EmotionalHeaviness,
    EmotionalVolatility,
    WeirdnessSurrealism,
    AttentionDemand,
    NarrativeComplexity,
    AmbiguityInterpretiveNess,
    SenseOfScale,
)


class PlotMetadata(BaseModel):
    """
    Structured response schema for plot synopsis and keyphrases generation.
    
    Used for LLM-generated plot metadata as specified in the movie search guide (section 4.8).
    """
    plot_synopsis: str
    plot_keyphrases: list[str]


class VibeMetadata(BaseModel):
    """
    Structured response schema for DenseVibe generation.
    
    Used for LLM-generated vibe metadata as specified in the movie search guide (section 5.3).
    Contains viewer experience descriptors for semantic search matching.
    
    Each field corresponds to one enum from vibe_enums.py and can be None if that
    attribute is not applicable or cannot be determined for the movie.
    """
    model_config = ConfigDict(use_enum_values=True)
    
    # Mood / atmosphere
    mood_atmosphere: Optional[MoodAtmosphere] = None
    
    # Tonal valence (warmth ↔ cynicism/bleakness)
    tonal_valence: Optional[TonalValence] = None
    
    # Pacing / momentum
    pacing_momentum: Optional[PacingMomentum] = None
    
    # Kinetic intensity (energy level)
    kinetic_intensity: Optional[KineticIntensity] = None
    
    # Tension pressure
    tension_pressure: Optional[TensionPressure] = None
    
    # Unpredictability / twistiness
    unpredictability_twistiness: Optional[UnpredictabilityTwistiness] = None
    
    # Scariness level
    scariness_level: Optional[ScarinessLevel] = None
    
    # Fear mode
    fear_mode: Optional[FearMode] = None
    
    # Humor level
    humor_level: Optional[HumorLevel] = None
    
    # Humor flavor
    humor_flavor: Optional[HumorFlavor] = None
    
    # Violence intensity
    violence_intensity: Optional[ViolenceIntensity] = None
    
    # Gore / body grossness
    gore_body_grossness: Optional[GoreBodyGrossness] = None
    
    # Romance prominence
    romance_prominence: Optional[RomanceProminence] = None
    
    # Romance tone
    romance_tone: Optional[RomanceTone] = None
    
    # Sexual explicitness
    sexual_explicitness: Optional[SexualExplicitness] = None
    
    # Erotic charge
    erotic_charge: Optional[EroticCharge] = None
    
    # Sexual tone
    sexual_tone: Optional[SexualTone] = None
    
    # Emotional heaviness
    emotional_heaviness: Optional[EmotionalHeaviness] = None
    
    # Emotional volatility
    emotional_volatility: Optional[EmotionalVolatility] = None
    
    # Weirdness / surrealism
    weirdness_surrealism: Optional[WeirdnessSurrealism] = None
    
    # Attention demand
    attention_demand: Optional[AttentionDemand] = None
    
    # Narrative complexity
    narrative_complexity: Optional[NarrativeComplexity] = None
    
    # Ambiguity / interpretive-ness
    ambiguity_interpretive_ness: Optional[AmbiguityInterpretiveNess] = None
    
    # Sense of scale
    sense_of_scale: Optional[SenseOfScale] = None

