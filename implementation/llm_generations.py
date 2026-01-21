"""
LLM generation utilities for creating plot metadata and vibe metadata from movie data.

This module contains functions to generate LLM-derived plot synopsis/keyphrases and
DenseVibe fields for vector search, as specified in the movie search guide (sections 4.8 and 5.3).
"""

import os
import json
from openai import OpenAI
from .prompts import DENSE_VIBE_SYSTEM_PROMPT, PLOT_EVENTS_SYSTEM_PROMPT, PLOT_ANALYSIS_SYSTEM_PROMPT
from .schemas import VibeMetadata, ParentalGuideItem, PlotEventsMetadata, PlotAnalysisMetadata
from dotenv import load_dotenv
from typing import Optional, Union
import time

# Load environment variables (for API key)
load_dotenv()

# Get OpenAI API key from environment and initialize client once at module load
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "OPENAI_API_KEY environment variable not set. "
        "Please set it before importing this module."
    )

# Initialize OpenAI client - created once when module is loaded
client = OpenAI(api_key=api_key)

# ================================
#      METADATA GENERATION
# ================================

def generate_plot_events_metadata(
    title: str,
    overview: str,
    plot_keywords: list[str],
    plot_summaries: list[str],
    plot_synopses: list[str]
) -> PlotEventsMetadata:
    # Build user prompt with all plot-related information
    # Combine all inputs into a structured prompt for the LLM
    plot_info_parts = []
    if title:
        plot_info_parts.append(f"title: {title}")
    if overview:
        plot_info_parts.append(f"overview: {overview}")
    if plot_summaries:
        plot_info_parts.append(f"plot_summaries: \n-{'\n-'.join(plot_summaries)}")
    if plot_synopses:
        plot_info_parts.append(f"plot_synopses: \n-{'\n-'.join(plot_synopses)}")
    if plot_keywords:
        plot_info_parts.append(f"plot_keywords: {', '.join(plot_keywords)}")
    
    user_prompt = "\n".join(plot_info_parts)
    
    # Generate response using gpt-5-nano with lowest thinking setting
    # Using .parse() for structured output - automatically validates and parses response
    # Note: reasoning_effort="minimal" uses the minimum reasoning capacity for faster responses
    response = client.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": PLOT_EVENTS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=PlotEventsMetadata,
        reasoning_effort="minimal",
        verbosity="low"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches PlotMetadata
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model failed to generate plot events metadata: {message.refusal}")


def generate_plot_analysis_metadata(
    title: str,
    overview: str,
    plot_synopsis: str,
    plot_keywords: list[str],
    reception_summary: Optional[str] = None
) -> PlotAnalysisMetadata:
    # Build user prompt with all plot-related information
    # Combine all inputs into a structured prompt for the LLM
    plot_info_parts = []
    if title:
        plot_info_parts.append(f"title: {title}")
    if overview:
        plot_info_parts.append(f"overview: {overview}")
    if plot_synopsis:
        plot_info_parts.append(f"plot_synopsis: {plot_synopsis}")
    if plot_keywords:
        plot_info_parts.append(f"plot_keywords: {', '.join(plot_keywords)}")
    if reception_summary:
        plot_info_parts.append(f"reception_summary: {reception_summary}")
    
    user_prompt = "\n".join(plot_info_parts)
    
    # Generate response using gpt-5-nano with lowest thinking setting
    # Using .parse() for structured output - automatically validates and parses response
    # Note: reasoning_effort="minimal" uses the minimum reasoning capacity for faster responses
    response = client.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": PLOT_ANALYSIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=PlotAnalysisMetadata,
        reasoning_effort="medium",
        verbosity="low"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches PlotMetadata
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model failed to generate plot analysis metadata: {message.refusal}")


def generate_vibe_summary(
    overview: str,
    genres: list[str],
    overall_keywords: list[str],
    plot_keywords: list[str],
    synopsis: Optional[list[str]] = None,
    plot_summaries: Optional[list[str]] = None,
    maturity_rating: Optional[str] = None,
    maturity_reasoning: Optional[list[str]] = None,
    parental_guide_items: Optional[list[Union[dict, ParentalGuideItem]]] = None,
    reception_summary: Optional[str] = None
) -> VibeMetadata:
    """
    Generates DenseVibe metadata using gpt-5-nano.
    
    This function generates LLM-derived vibe metadata for vector search (section 5.3 of the
    movie search guide). It creates viewer experience descriptors that capture how it feels
    to watch the movie, enabling semantic matching for queries like "cozy date night movies"
    or "edge-of-your-seat thrillers".
    
    The inputs are derived from the movie's various fields:
    - overview: High-level movie overview (hint only)
    - genres: List of genres (light prior, not main signal)
    - overall_keywords: High-level keywords (hint only)
    - plot_keywords: Plot-related keywords (hint only)
    - synopsis: Optional list of synopsis text segments
    - plot_summaries: Optional list of plot summary text segments
    - maturity_rating: Rating (G, PG, PG-13, R, NC-17, Unrated)
    - maturity_reasoning: List of rating reasons
    - parental_guide_items: List of {category, severity} items
    - reception_summary: Optional review summary (can hint pacing, crowd-pleaser vs bleak)
    
    Args:
        overview: High-level movie overview text
        genres: List of genre strings
        overall_keywords: List of high-level keywords
        plot_keywords: List of plot-related keywords
        synopsis: Optional list of synopsis text segments
        plot_summaries: Optional list of plot summary text segments
        maturity_rating: Optional maturity rating string
        maturity_reasoning: Optional list of maturity reasoning strings
        parental_guide_items: Optional list of dicts or ParentalGuideItem objects with 'category' and 'severity' attributes
        reception_summary: Optional reception summary text
        
    Returns:
        VibeMetadata instance containing one nullable field per enum from vibe_enums.py.
        Each field can be None if that attribute is not applicable or cannot be determined
        for the movie. Enum values are automatically validated by Pydantic.
        
    Raises:
        ValueError: If the model refuses to generate output
        Exception: If the API call fails
    """
    
    # Determine imdb_story_text according to guide section 5.3.2:
    # If synopsis[0] exists and is non-empty → use it
    # Else if plot_summaries[0] exists and is non-empty → use it
    # Else omit imdb_story_text
    imdb_story_text = None
    if synopsis and len(synopsis) > 0 and synopsis[0]:
        imdb_story_text = synopsis[0]
    elif plot_summaries and len(plot_summaries) > 0 and plot_summaries[0]:
        imdb_story_text = plot_summaries[0]
    
    # Build user prompt with all vibe-related information
    # Combine all inputs into a structured prompt for the LLM
    vibe_info_parts = []
    
    # Core plot/story inputs (hints only)
    if overview:
        vibe_info_parts.append(f"Overview: {overview}")
    if genres:
        vibe_info_parts.append(f"Genres: {', '.join(genres)}")
    if overall_keywords:
        vibe_info_parts.append(f"Overall keywords: {', '.join(overall_keywords)}")
    if plot_keywords:
        vibe_info_parts.append(f"Plot keywords: {', '.join(plot_keywords)}")
    
    # Optional IMDB story text (use whole string if provided)
    if imdb_story_text:
        vibe_info_parts.append(f"Story text: {imdb_story_text}")
    
    # Maturity information (for suitability hints)
    if maturity_rating:
        vibe_info_parts.append(f"Maturity rating: {maturity_rating}")
    if maturity_reasoning:
        vibe_info_parts.append(f"Maturity reasoning: {', '.join(maturity_reasoning)}")
    if parental_guide_items:
        # Format parental guide items as "severity category"
        # Handle both dicts and ParentalGuideItem objects
        guide_strings = []
        for item in parental_guide_items:
            if isinstance(item, dict):
                severity = item.get('severity', '')
                category = item.get('category', '')
            else:
                # ParentalGuideItem object with attributes
                severity = getattr(item, 'severity', '')
                category = getattr(item, 'category', '')
            
            if severity and category:
                guide_strings.append(f"{severity} {category}")
        
        if guide_strings:
            vibe_info_parts.append(f"Parental guide: {', '.join(guide_strings)}")
    
    # Optional reception summary (can hint pacing, crowd-pleaser vs bleak)
    if reception_summary:
        vibe_info_parts.append(f"Reception summary: {reception_summary}")
    
    user_prompt = "\n".join(vibe_info_parts) if vibe_info_parts else "No input data provided."
    
    # Generate response using gpt-5-nano with lowest thinking setting
    # Using .parse() for structured output - automatically validates and parses response
    # Note: reasoning_effort="none" uses the minimum reasoning capacity for faster responses
    response = client.chat.completions.parse(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": DENSE_VIBE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=VibeMetadata,
        reasoning_effort="minimal"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches VibeMetadata
    # Enum values are automatically validated by Pydantic, so no post-processing is needed
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model refused to generate output: {message.refusal}")


# ================================
#    PARALLEL EXECUTION HELPERS
# ================================


def generate_vibe_metadata(
    overview: str,
    genres: list[str],
    overall_keywords: list[str],
    plot_keywords: list[str],
    synopsis: Optional[list[str]],
    plot_summaries: Optional[list[str]],
    maturity_rating: str,
    maturity_reasoning: list[str],
    parental_guide_items: list[dict],
    reception_summary: Optional[str]
) -> tuple[str, Optional[VibeMetadata]]:
    """
    Generate DenseVibe metadata using LLM with error handling.
    
    This is a wrapper around generate_vibe_summary that handles errors gracefully
    and returns a tuple format suitable for parallel execution.
    
    Args:
        overview: High-level movie overview text
        genres: List of genre strings
        overall_keywords: List of high-level keywords
        plot_keywords: List of plot-related keywords
        synopsis: Optional list of synopsis text segments
        plot_summaries: Optional list of plot summary text segments
        maturity_rating: Maturity rating string
        maturity_reasoning: List of maturity reasoning strings
        parental_guide_items: List of dicts with 'category' and 'severity' keys
        reception_summary: Optional reception summary text
        
    Returns:
        Tuple of (result_type, result_data) where result_type is "vibe" and
        result_data is VibeMetadata object or None on error.
    """
    try:
        vibe_metadata = generate_vibe_summary(
            overview=overview,
            genres=genres,
            overall_keywords=overall_keywords,
            plot_keywords=plot_keywords,
            synopsis=synopsis,
            plot_summaries=plot_summaries,
            maturity_rating=maturity_rating,
            maturity_reasoning=maturity_reasoning,
            parental_guide_items=parental_guide_items,
            reception_summary=reception_summary
        )
        return ("vibe", vibe_metadata)
    except Exception as e:
        print(f"Error generating DenseVibe metadata: {e}")
        return ("vibe", None)


def generate_plot_metadata(
    title: str,
    overview: str,
    plot_summaries: list[str],
    plot_synopses: list[str],
    plot_keywords: list[str],
    reception_summary: Optional[str] = None
) -> tuple[str, Optional[tuple[PlotEventsMetadata, PlotAnalysisMetadata]]]:
    try:
        print(f"Generating plot metadata for {title}")

        # Time the plot events metadata generation
        start_time_events = time.perf_counter()
        plot_events_metadata = generate_plot_events_metadata(
            title=title,
            overview=overview,
            plot_keywords=plot_keywords,
            plot_summaries=plot_summaries,
            plot_synopses=plot_synopses
        )
        end_time_events = time.perf_counter()
        events_duration = end_time_events - start_time_events

        print(f"Plot events metadata for {title} (completed in {events_duration:.2f} seconds):")
        print(plot_events_metadata)

        # Time the plot analysis metadata generation
        start_time_analysis = time.perf_counter()
        plot_analysis_metadata = generate_plot_analysis_metadata(
            title=title,
            overview=overview,
            plot_synopsis=plot_events_metadata.plot_summary,
            plot_keywords=plot_keywords,
            reception_summary=reception_summary
        )
        end_time_analysis = time.perf_counter()
        analysis_duration = end_time_analysis - start_time_analysis

        print(f"\nPlot analysis metadata for {title} (completed in {analysis_duration:.2f} seconds):")

        return ("plot", (plot_events_metadata, plot_analysis_metadata))
    except Exception as e:
        print(f"Error generating plot metadata: {e}")
        return ("plot", None)
