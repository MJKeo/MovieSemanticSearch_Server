"""
LLM generation utilities for creating plot metadata and vibe metadata from movie data.

This module contains functions to generate LLM-derived plot synopsis/keyphrases and
DenseVibe fields for vector search, as specified in the movie search guide (sections 4.8 and 5.3).
"""

import os
from openai import OpenAI
from prompts import PLOT_SUMMARY_SYSTEM_PROMPT, DENSE_VIBE_SYSTEM_PROMPT
from schemas import PlotMetadata, VibeMetadata
from dotenv import load_dotenv
from typing import Optional, Union
from classes import ParentalGuideItem

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


def generate_plot_summary(
    overview: str,
    plot_keywords: list[str],
    plot_summaries: list[str],
    synopsis: list[str]
) -> PlotMetadata:
    """
    Generates plot synopsis and plot keyphrases using gpt-5-nano with the lowest thinking setting.
    
    This function generates LLM-derived plot metadata for vector search (section 4.8 of the
    movie search guide). It creates a brief but complete plot synopsis (spoilers allowed) and
    a list of key terms and phrases related to the plot.
    
    The inputs are derived from the movie's plot-related fields:
    - plot_keywords: High-level keywords about the plot
    - plot_summaries: Detailed plot summaries
    - synopsis: Extended synopsis text
    - overview: High-level movie overview
    
    Args:
        overview: High-level movie overview text
        plot_keywords: List of plot-related keywords
        plot_summaries: List of detailed plot summaries
        synopsis: List of synopsis text segments
        
    Returns:
        PlotMetadata instance containing:
        - plot_synopsis: Brief but complete summary (spoilers allowed)
        - plot_keyphrases: List of key terms and phrases related to the plot
        
    Raises:
        ValueError: If the model refuses to generate output
        Exception: If the API call fails
    """
    
    # Build user prompt with all plot-related information
    # Combine all inputs into a structured prompt for the LLM
    plot_info_parts = [
        f"Overview: {overview}",
        f"Plot keywords: {', '.join(plot_keywords)}",
        f"Plot summaries: \n-{'\n-'.join(plot_summaries)}",
        f"Synopsis: \n-{'\n-'.join(synopsis)}",
    ]
    
    user_prompt = "\n".join(plot_info_parts)
    
    # Generate response using gpt-5-nano with lowest thinking setting
    # Using .parse() for structured output - automatically validates and parses response
    # Note: reasoning_effort="minimal" uses the minimum reasoning capacity for faster responses
    response = client.chat.completions.parse(
        model="gpt-5-nano",
        messages=[
            {"role": "system", "content": PLOT_SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=PlotMetadata,
        reasoning_effort="minimal"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches PlotMetadata
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model refused to generate output: {message.refusal}")


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
    Generates DenseVibe metadata (vibe_summary, vibe_keywords, watch_context_tags) using gpt-5-nano.
    
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
        VibeMetadata instance containing:
        - vibe_summary: 1 short sentence describing how it feels to watch (mood + pacing/energy + intensity + style). No plot retell, no themes.
        - vibe_keywords: 12–20 short phrases (1–3 words) capturing viewer-experience descriptors (mood, pacing, intensity, humor/scare/gross style, sensory/aesthetic feel).
        - watch_context_tags: 4–10 broad tags that together answer: what kind of night, how social, how demanding, what emotional payoff, and audience fit.
        
    Raises:
        ValueError: If the model refuses to generate output
        Exception: If the API call fails
        
    Post-processing:
        All fields are lowercased and stripped as specified in section 5.3.3 of the guide.
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
        vibe_info_parts.append(f"IMDB story text: {imdb_story_text}")
    
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
    message = response.choices[0].message
    if message.parsed:
        vibe_metadata = message.parsed
        
        # Post-processing: lowercase and strip all fields (section 5.3.3)
        vibe_metadata.vibe_summary = vibe_metadata.vibe_summary.strip().lower()
        vibe_metadata.vibe_keywords = [kw.strip().lower() for kw in vibe_metadata.vibe_keywords]
        vibe_metadata.watch_context_tags = [tag.strip().lower() for tag in vibe_metadata.watch_context_tags]
        
        return vibe_metadata
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model refused to generate output: {message.refusal}")


# ================================
#         PARALLEL EXECUTION HELPERS
# ================================

def generate_plot_metadata(
    overview: str,
    plot_keywords: list[str],
    plot_summaries: list[str],
    synopsis: list[str]
) -> tuple[str, dict]:
    """
    Generate plot metadata using LLM with error handling.
    
    This is a wrapper around generate_plot_summary that handles errors gracefully
    and returns a tuple format suitable for parallel execution.
    
    Args:
        overview: High-level movie overview text
        plot_keywords: List of plot-related keywords
        plot_summaries: List of detailed plot summaries
        synopsis: List of synopsis text segments
        
    Returns:
        Tuple of (result_type, result_data) where result_type is "plot" and
        result_data contains "plot_synopsis" and "plot_keyphrases" keys.
    """
    try:
        plot_metadata = generate_plot_summary(
            overview=overview,
            plot_keywords=plot_keywords,
            plot_summaries=plot_summaries,
            synopsis=synopsis
        )
        
        # Check if result is empty or invalid
        if plot_metadata and plot_metadata.plot_synopsis:
            return ("plot", {
                "plot_synopsis": plot_metadata.plot_synopsis,
                "plot_keyphrases": plot_metadata.plot_keyphrases if plot_metadata.plot_keyphrases else []
            })
        else:
            raise ValueError("Plot metadata generation failed")
    except Exception as e:
        print(f"Error generating plot metadata: {e}")
        # Exception occurred - use defaults
        if plot_summaries:
            return ("plot", {
                "plot_synopsis": plot_summaries[0],
                "plot_keyphrases": []
            })
        return ("plot", {"plot_synopsis": None, "plot_keyphrases": []})


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

