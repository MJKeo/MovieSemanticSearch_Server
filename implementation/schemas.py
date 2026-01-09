"""
Pydantic schemas for LLM response structures.

This module contains Pydantic models used for structured outputs from LLM API calls.
"""

from pydantic import BaseModel


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
    """
    vibe_summary: str  # 1 short sentence describing how it feels to watch (mood + pacing/energy + intensity + style). No plot retell, no themes.
    vibe_keywords: list[str]  # 12–20 short phrases (1–3 words) capturing viewer-experience descriptors (mood, pacing, intensity, humor/scare/gross style, sensory/aesthetic feel).
    watch_context_tags: list[str]  # 4–10 broad tags that together answer: what kind of night, how social, how demanding, what emotional payoff, and audience fit.

