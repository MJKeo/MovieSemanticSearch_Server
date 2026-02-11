"""
LLM generation utilities for creating plot metadata and vibe metadata from movie data.

This module contains functions to generate LLM-derived plot synopsis/keyphrases and
DenseVibe fields for vector search, as specified in the movie search guide (sections 4.8 and 5.3).
"""

import os
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from .prompts.metadata_generation_prompts import (
    PLOT_EVENTS_SYSTEM_PROMPT, 
    PLOT_ANALYSIS_SYSTEM_PROMPT,
    VIEWER_EXPERIENCE_SYSTEM_PROMPT,
    WATCH_CONTEXT_SYSTEM_PROMPT,
    NARRATIVE_TECHNIQUES_SYSTEM_PROMPT,
    PRODUCTION_KEYWORDS_SYSTEM_PROMPT,
    SOURCE_OF_INSPIRATION_SYSTEM_PROMPT,
    RECEPTION_SYSTEM_PROMPT,
)
from .schemas import (
    ParentalGuideItem, 
    PlotEventsMetadata, 
    PlotAnalysisMetadata, 
    IMDBFeaturedReview,
    ViewerExperienceMetadata,
    WatchContextMetadata,
    NarrativeTechniquesMetadata,
    ProductionMetadata,
    GenericTermsSection,
    SourceOfInspirationSection,
    ReceptionMetadata,
    IMDBReviewTheme,
)
from dotenv import load_dotenv
from typing import Optional, List
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
#         Plot Events
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


# ================================
#        Plot Analysis
# ================================

def generate_plot_analysis_metadata(
    title: str,
    genres: list[str],
    overview: str,
    plot_synopsis: str,
    plot_keywords: list[str],
    reception_summary: Optional[str] = None,
    featured_reviews: list[IMDBFeaturedReview] = []
) -> PlotAnalysisMetadata:
    # Build user prompt with all plot-related information
    # Combine all inputs into a structured prompt for the LLM
    plot_info_parts = []
    if title:
        plot_info_parts.append(f"title: {title}")
    if genres:
        plot_info_parts.append(f"genres: {', '.join(genres)}")
    if overview:
        plot_info_parts.append(f"overview: {overview}")
    if plot_synopsis:
        plot_info_parts.append(f"plot_synopsis: {plot_synopsis}")
    if plot_keywords:
        plot_info_parts.append(f"plot_keywords: {', '.join(plot_keywords)}")
    if reception_summary:
        plot_info_parts.append(f"reception_summary: {reception_summary}")
    if featured_reviews:
        formatted_reviews = [f"{review.summary}: {review.text}" for review in featured_reviews[:5]]
        plot_info_parts.append(f"featured_reviews: \n -{"\n -".join(formatted_reviews)}")
    
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
        reasoning_effort="low",
        verbosity="low"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches PlotMetadata
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model failed to generate plot analysis metadata: {message.refusal}")


# ================================
#        Viewer Experience
# ================================

def generate_viewer_experience_metadata(
    title: str,
    genres: List[str],
    plot_synopsis: str,
    plot_keywords: List[str],
    overall_keywords: List[str],
    maturity_rating: str,
    maturity_reasoning: List[str],
    parental_guide_items: List[ParentalGuideItem],
    reception_summary: str,
    audience_reception_attributes: List[IMDBReviewTheme],
    featured_reviews: List[IMDBFeaturedReview],
) -> ViewerExperienceMetadata:
    """
    Generate viewer experience metadata for a movie.
    """
    plot_info_parts = []
    if title:
        plot_info_parts.append(f"title: {title}")
    if genres:
        plot_info_parts.append(f"genres: {', '.join(genres)}")
    if plot_synopsis:
        plot_info_parts.append(f"plot_synopsis: {plot_synopsis}")
    if plot_keywords:
        plot_info_parts.append(f"plot_keywords: {', '.join(plot_keywords)}")
    if overall_keywords:
        plot_info_parts.append(f"overall_keywords: {', '.join(overall_keywords)}")
    if maturity_rating:
        plot_info_parts.append(f"maturity_rating: {maturity_rating}")
    if maturity_reasoning:
        plot_info_parts.append(f"maturity_reasoning: {', '.join(maturity_reasoning)}")
    if parental_guide_items:
        plot_info_parts.append(f"parental_guide_items: {', '.join([f"{item.category}: {item.severity}" for item in parental_guide_items])}")
    if reception_summary:
        plot_info_parts.append(f"reception_summary: {reception_summary}")
    if audience_reception_attributes:
        converted_receptions = [str(attribute) for attribute in audience_reception_attributes]
        plot_info_parts.append(f"audience_reception_attributes: \n -{". ".join(converted_receptions)}")
    if featured_reviews:
        formatted_reviews = [f"{review.summary}: {review.text}" for review in featured_reviews[:5]]
        plot_info_parts.append(f"featured_reviews: \n -{"\n -".join(formatted_reviews)}")
    
    user_prompt = "\n".join(plot_info_parts)
    
    # Generate response using gpt-5-nano with lowest thinking setting
    # Using .parse() for structured output - automatically validates and parses response
    # Note: reasoning_effort="minimal" uses the minimum reasoning capacity for faster responses
    response = client.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": VIEWER_EXPERIENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=ViewerExperienceMetadata,
        reasoning_effort="low",
        verbosity="low"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches PlotMetadata
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model failed to generate viewer experience metadata: {message.refusal}")


# ================================
#        Watch Context
# ================================

def generate_watch_context_metadata(
    title: str,
    genres: List[str],
    overview: str,
    plot_keywords: List[str],
    overall_keywords: List[str],
    audience_reception_attributes: List[IMDBReviewTheme],
    reception_summary: str,
    featured_reviews: List[IMDBFeaturedReview]
) -> WatchContextMetadata:
    """
    Generate watch context metadata for a movie.
    """
    plot_info_parts = []
    if title:
        plot_info_parts.append(f"title: {title}")
    if genres:
        plot_info_parts.append(f"genres: {', '.join(genres)}")
    if overview:
        plot_info_parts.append(f"overview: {overview}")
    if plot_keywords:
        plot_info_parts.append(f"plot_keywords: {', '.join(plot_keywords)}")
    if overall_keywords:
        plot_info_parts.append(f"overall_keywords: {', '.join(overall_keywords)}")
    if reception_summary:
        plot_info_parts.append(f"reception_summary: {reception_summary}")
    if audience_reception_attributes:
        converted_receptions = [str(attribute) for attribute in audience_reception_attributes]
        plot_info_parts.append(f"audience_reception_attributes: \n -{". ".join(converted_receptions)}")
    if featured_reviews:
        formatted_reviews = [f"{review.summary}: {review.text}" for review in featured_reviews[:5]]
        plot_info_parts.append(f"featured_reviews: \n -{"\n -".join(formatted_reviews)}")
    
    user_prompt = "\n".join(plot_info_parts)
    
    # Generate response using gpt-5-nano with lowest thinking setting
    # Using .parse() for structured output - automatically validates and parses response
    # Note: reasoning_effort="minimal" uses the minimum reasoning capacity for faster responses
    response = client.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": WATCH_CONTEXT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=WatchContextMetadata,
        reasoning_effort="medium",
        verbosity="low"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches PlotMetadata
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model failed to generate watch context metadata: {message.refusal}")


# ================================
#      Narrative Techniques
# ================================

def generate_narrative_techniques_metadata(
    title: str,
    plot_synopsis: str,
    plot_keywords: List[str],
    overall_keywords: List[str],
    featured_reviews: List[IMDBFeaturedReview],
    reception_summary: str,
) -> NarrativeTechniquesMetadata:
    """
    Generate narrative techniques metadata for a movie.
    """
    plot_info_parts = []
    if title:
        plot_info_parts.append(f"title: {title}")
    if plot_synopsis:
        plot_info_parts.append(f"plot_synopsis: {plot_synopsis}")
    if plot_keywords:
        plot_info_parts.append(f"plot_keywords: {', '.join(plot_keywords)}")
    if overall_keywords:
        plot_info_parts.append(f"overall_keywords: {', '.join(overall_keywords)}")
    if reception_summary:
        plot_info_parts.append(f"reception_summary: {reception_summary}")
    if featured_reviews:
        formatted_reviews = [f"{review.summary}: {review.text}" for review in featured_reviews[:5]]
        plot_info_parts.append(f"featured_reviews: \n -{"\n -".join(formatted_reviews)}")
    
    user_prompt = "\n".join(plot_info_parts)
    
    # Generate response using gpt-5-nano with lowest thinking setting
    # Using .parse() for structured output - automatically validates and parses response
    # Note: reasoning_effort="minimal" uses the minimum reasoning capacity for faster responses
    response = client.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": NARRATIVE_TECHNIQUES_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=NarrativeTechniquesMetadata,
        reasoning_effort="medium",
        verbosity="low"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches PlotMetadata
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model failed to generate narrative techniques metadata: {message.refusal}")


# ================================
#           Production
# ================================

def generate_production_metadata(
    title: str,
    plot_synopsis: str,
    plot_keywords: List[str],
    overall_keywords: List[str],
    featured_reviews: List[IMDBFeaturedReview]
) -> ProductionMetadata:
    """
    Generate production metadata by calling generate_production_keywords and 
    generate_source_of_inspiration in parallel, then combining the results.
    
    Raises an error if either function fails to return a value or raises an error.
    """
    # Execute both functions in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both tasks
        future_production_keywords = executor.submit(
            generate_production_keywords,
            title,
            overall_keywords
        )
        future_source_of_inspiration = executor.submit(
            generate_source_of_inspiration,
            title,
            plot_synopsis,
            plot_keywords,
            overall_keywords,
            featured_reviews
        )
        
        # Wait for both to complete and collect results
        production_keywords_result = None
        source_of_inspiration_result = None
        production_keywords_error = None
        source_of_inspiration_error = None
        
        # Get results, catching any exceptions
        try:
            production_keywords_result = future_production_keywords.result()
        except Exception as e:
            production_keywords_error = e
        
        try:
            source_of_inspiration_result = future_source_of_inspiration.result()
        except Exception as e:
            source_of_inspiration_error = e
        
        # Check if either failed or returned None
        if production_keywords_error is not None:
            raise RuntimeError(
                f"generate_production_keywords failed: {production_keywords_error}"
            ) from production_keywords_error
        
        if source_of_inspiration_error is not None:
            raise RuntimeError(
                f"generate_source_of_inspiration failed: {source_of_inspiration_error}"
            ) from source_of_inspiration_error
        
        if production_keywords_result is None:
            raise RuntimeError("generate_production_keywords returned None")
        
        if source_of_inspiration_result is None:
            raise RuntimeError("generate_source_of_inspiration returned None")
        
        # Both succeeded, combine into ProductionMetadata
        return ProductionMetadata(
            production_keywords=production_keywords_result,
            sources_of_inspiration=source_of_inspiration_result
        )

def generate_production_keywords(
    title: str,
    overall_keywords: List[str]
) -> GenericTermsSection:
    plot_info_parts = []
    if title:
        plot_info_parts.append(f"title: {title}")
    if overall_keywords:
        plot_info_parts.append(f"keywords: {', '.join(overall_keywords)}")
    
    user_prompt = "\n".join(plot_info_parts)
    
    response = client.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": PRODUCTION_KEYWORDS_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=GenericTermsSection,
        reasoning_effort="low",
        verbosity="low"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches PlotMetadata
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model failed to generate production keywords: {message.refusal}")

def generate_source_of_inspiration(
    title: str,
    plot_synopsis: str,
    plot_keywords: List[str],
    overall_keywords: List[str],
    featured_reviews: List[IMDBFeaturedReview],
) -> SourceOfInspirationSection:
    """
    Generate source of inspiration metadata for a movie.
    """
    keywords = plot_keywords + overall_keywords

    plot_info_parts = []
    if title:
        plot_info_parts.append(f"title: {title}")
    if keywords:
        plot_info_parts.append(f"keywords: {', '.join(keywords)}")
    if plot_synopsis:
        plot_info_parts.append(f"plot_synopsis: {plot_synopsis}")
    if featured_reviews:
        formatted_reviews = [f"{review.summary}: {review.text}" for review in featured_reviews[:5]]
        plot_info_parts.append(f"featured_reviews: \n -{"\n -".join(formatted_reviews)}")
    
    user_prompt = "\n".join(plot_info_parts)
    
    response = client.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": SOURCE_OF_INSPIRATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=SourceOfInspirationSection,
        reasoning_effort="low",
        verbosity="low"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches PlotMetadata
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model failed to generate sources of inspiration: {message.refusal}")


# ================================
#          Reception
# ================================

def generate_reception_metadata(
    title: str,
    reception_summary: str,
    audience_reception_attributes: List[IMDBFeaturedReview],
    featured_reviews: List[IMDBFeaturedReview],
) -> ReceptionMetadata:
    """
    Generate reception metadata for a movie.
    """
    plot_info_parts = []
    if title:
        plot_info_parts.append(f"title: {title}")
    if reception_summary:
        plot_info_parts.append(f"reception_summary: {reception_summary}")
    if audience_reception_attributes:
        converted_receptions = [str(attribute) for attribute in audience_reception_attributes]
        plot_info_parts.append(f"audience_reception_attributes: \n -{". ".join(converted_receptions)}")
    if featured_reviews:
        formatted_reviews = [f"{review.summary}: {review.text}" for review in featured_reviews[:5]]
        plot_info_parts.append(f"featured_reviews: \n -{"\n -".join(formatted_reviews)}")
    
    user_prompt = "\n".join(plot_info_parts)
    
    # Generate response using gpt-5-nano with lowest thinking setting
    # Using .parse() for structured output - automatically validates and parses response
    # Note: reasoning_effort="minimal" uses the minimum reasoning capacity for faster responses
    response = client.chat.completions.parse(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": RECEPTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        response_format=ReceptionMetadata,
        reasoning_effort="low",
        verbosity="low"
    )
    
    # Extract the parsed response - OpenAI automatically validates structure matches PlotMetadata
    message = response.choices[0].message
    if message.parsed:
        return message.parsed
    else:
        # Handle case where model refuses to generate output
        raise ValueError(f"Model failed to generate reception metadata: {message.refusal}")


# ================================
#    PARALLEL EXECUTION HELPERS
# ================================

def generate_llm_metadata(
    title: str,
    overview: str,
    plot_summaries: list[str],
    plot_synopses: list[str],
    plot_keywords: list[str],
    featured_reviews: list[IMDBFeaturedReview] = [],
    # Additional parameters for functions that need them
    genres: list[str] = [],
    overall_keywords: list[str] = [],
    reception_summary: Optional[str] = None,
    audience_reception_attributes: list[IMDBReviewTheme] = [],
    maturity_rating: str = "",
    maturity_reasoning: list[str] = [],
    parental_guide_items: list[ParentalGuideItem] = []
) -> dict:
    """
    Generate all LLM metadata for a movie using parallel execution.
    
    Execution flow:
    1. Calls generate_plot_events_metadata, generate_watch_context_metadata, and 
       generate_reception_metadata in parallel
    2. As soon as generate_plot_events_metadata completes successfully, calls
       generate_plot_analysis_metadata, generate_viewer_experience_metadata,
       generate_narrative_techniques_metadata, and generate_production_metadata in parallel
    3. Returns a dictionary where each key maps to one of the metadata objects generated
    
    If generate_plot_events_metadata fails, the whole method raises an error.
    If any other method fails, that key is set to None in the returned dictionary.
    
    Returns:
        dict: A dictionary with keys: 'plot_events', 'watch_context', 'reception',
              'plot_analysis', 'viewer_experience', 'narrative_techniques', 'production'
    """
    print(f"Generating llm metadata for {title}")
    
    # Step 1: Execute plot_events, watch_context, and reception in parallel
    # These don't depend on plot_events completing
    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit the three independent tasks
        plot_events_future = executor.submit(
            generate_plot_events_metadata,
            title=title,
            overview=overview,
            plot_keywords=plot_keywords,
            plot_summaries=plot_summaries,
            plot_synopses=plot_synopses
        )
        watch_context_future = executor.submit(
            generate_watch_context_metadata,
            title=title,
            genres=genres,
            overview=overview,
            plot_keywords=plot_keywords,
            overall_keywords=overall_keywords,
            audience_reception_attributes=audience_reception_attributes,
            reception_summary=reception_summary or "",
            featured_reviews=featured_reviews
        )
        reception_future = executor.submit(
            generate_reception_metadata,
            title=title,
            reception_summary=reception_summary or "",
            audience_reception_attributes=audience_reception_attributes,
            featured_reviews=featured_reviews
        )
        
        # Wait for plot_events to complete first (critical - must succeed)
        # If it fails, the whole method should error
        try:
            plot_events_metadata = plot_events_future.result()
            plot_synopsis = plot_events_metadata.plot_summary
            if plot_events_metadata is not None:
                print(f"✓ plot_events_metadata for {title}: SUCCESS")
            else:
                print(f"✗ plot_events_metadata for {title}: FAILED (result is None)")
                raise RuntimeError("generate_plot_events_metadata returned None")
        except Exception as e:
            print(f"✗ plot_events_metadata for {title}: FAILED ({e})")
            raise RuntimeError(f"generate_plot_events_metadata failed: {e}") from e
        
        # Collect results from watch_context and reception (these can fail gracefully)
        watch_context_metadata = None
        reception_metadata = None
        
        try:
            watch_context_metadata = watch_context_future.result()
            if watch_context_metadata is not None:
                print(f"✓ watch_context_metadata for {title}: SUCCESS")
            else:
                print(f"✗ watch_context_metadata for {title}: FAILED (result is None)")
        except Exception as e:
            print(f"✗ watch_context_metadata for {title}: FAILED ({e})")
        
        try:
            reception_metadata = reception_future.result()
            if reception_metadata is not None:
                print(f"✓ reception_metadata for {title}: SUCCESS")
            else:
                print(f"✗ reception_metadata for {title}: FAILED (result is None)")
        except Exception as e:
            print(f"✗ reception_metadata for {title}: FAILED ({e})")
    
    # Step 2: Now that plot_events is complete, execute the dependent functions in parallel
    # These all depend on plot_synopsis from plot_events_metadata
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit the four dependent tasks
        plot_analysis_future = executor.submit(
            generate_plot_analysis_metadata,
            title=title,
            genres=genres,
            overview=overview,
            plot_synopsis=plot_synopsis,
            plot_keywords=plot_keywords,
            reception_summary=reception_summary,
            featured_reviews=featured_reviews
        )
        viewer_experience_future = executor.submit(
            generate_viewer_experience_metadata,
            title=title,
            genres=genres,
            plot_synopsis=plot_synopsis,
            plot_keywords=plot_keywords,
            overall_keywords=overall_keywords,
            maturity_rating=maturity_rating,
            maturity_reasoning=maturity_reasoning,
            parental_guide_items=parental_guide_items,
            reception_summary=reception_summary or "",
            audience_reception_attributes=audience_reception_attributes,
            featured_reviews=featured_reviews
        )
        narrative_techniques_future = executor.submit(
            generate_narrative_techniques_metadata,
            title=title,
            plot_synopsis=plot_synopsis,
            plot_keywords=plot_keywords,
            overall_keywords=overall_keywords,
            featured_reviews=featured_reviews,
            reception_summary=reception_summary or ""
        )
        production_future = executor.submit(
            generate_production_metadata,
            title=title,
            plot_synopsis=plot_synopsis,
            plot_keywords=plot_keywords,
            overall_keywords=overall_keywords,
            featured_reviews=featured_reviews
        )
        
        # Collect results from all four tasks (these can fail gracefully)
        plot_analysis_metadata = None
        viewer_experience_metadata = None
        narrative_techniques_metadata = None
        production_metadata = None
        
        try:
            plot_analysis_metadata = plot_analysis_future.result()
            if plot_analysis_metadata is not None:
                print(f"✓ plot_analysis_metadata for {title}: SUCCESS")
            else:
                print(f"✗ plot_analysis_metadata for {title}: FAILED (result is None)")
        except Exception as e:
            print(f"✗ plot_analysis_metadata for {title}: FAILED ({e})")
        
        try:
            viewer_experience_metadata = viewer_experience_future.result()
            if viewer_experience_metadata is not None:
                print(f"✓ viewer_experience_metadata for {title}: SUCCESS")
            else:
                print(f"✗ viewer_experience_metadata for {title}: FAILED (result is None)")
        except Exception as e:
            print(f"✗ viewer_experience_metadata for {title}: FAILED ({e})")
        
        try:
            narrative_techniques_metadata = narrative_techniques_future.result()
            if narrative_techniques_metadata is not None:
                print(f"✓ narrative_techniques_metadata for {title}: SUCCESS")
            else:
                print(f"✗ narrative_techniques_metadata for {title}: FAILED (result is None)")
        except Exception as e:
            print(f"✗ narrative_techniques_metadata for {title}: FAILED ({e})")
        
        try:
            production_metadata = production_future.result()
            if production_metadata is not None:
                print(f"✓ production_metadata for {title}: SUCCESS")
            else:
                print(f"✗ production_metadata for {title}: FAILED (result is None)")
        except Exception as e:
            print(f"✗ production_metadata for {title}: FAILED ({e})")
    
    # Return a dictionary with all results
    return {
        "plot_events_metadata": plot_events_metadata,
        "watch_context_metadata": watch_context_metadata,
        "reception_metadata": reception_metadata,
        "plot_analysis_metadata": plot_analysis_metadata,
        "viewer_experience_metadata": viewer_experience_metadata,
        "narrative_techniques_metadata": narrative_techniques_metadata,
        "production_metadata": production_metadata
    }
