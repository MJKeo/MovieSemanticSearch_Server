"""
Vector text generation functions for each of the 8 named vector spaces.

This module is the single source of truth for converting a BaseMovie into
the text representation that will be embedded for each vector space. The
same functions are used at ingestion time (batch and single-movie) and
must stay in sync with the vector space definitions in Qdrant.

Vector spaces:
    1. anchor            — full "movie card" identity
    2. plot_events       — chronological plot details
    3. plot_analysis     — themes, arcs, concepts
    4. narrative_techniques — storytelling style/tone
    5. viewer_experience — emotional tone, pacing
    6. watch_context     — when/how to watch
    7. production        — budget, locations, technical
    8. reception         — critical reception, awards
"""

from implementation.classes.movie import BaseMovie
from implementation.classes.enums import BudgetSize
from implementation.misc.helpers import normalize_string
from schemas.metadata import (
    NarrativeTechniquesOutput,
    ViewerExperienceOutput,
    WatchContextOutput,
)
from schemas.movie import Movie


# ===============================
#         Normalization
# ===============================

def budget_size_to_vector_text(budget_size: BudgetSize | None) -> str:
    """Map budget size enums to the semantic phrases used in vector text."""
    if budget_size == BudgetSize.SMALL:
        return "small budget"
    if budget_size == BudgetSize.LARGE:
        return "big budget, blockbuster"
    return ""


# ===============================
#         Vector Text
# ===============================

def create_anchor_vector_text(movie: BaseMovie) -> str:
    """
    Creates the text representation for DenseAnchor vector embedding.

    DenseAnchor captures the full "movie card" identity to provide good recall
    for most queries. It includes comprehensive information about the movie's
    identity, content, production, cast, and reception.

    Args:
        movie: IMDBMovie instance containing all movie metadata

    Returns:
        Formatted text string ready for embedding as DenseAnchor vector
    """
    # Build the text components according to the DenseAnchor template (section 5.1)
    parts = []

    parts.append("# Overview:")

    # Title string
    parts.append(movie.title_string())

    if movie.plot_analysis_metadata:
        parts.append(movie.plot_analysis_metadata.generalized_plot_overview)

    # Genres as comma-separated values
    genres_csv = ", ".join(movie.genres_subset()) if movie.genres_subset() else ""
    if genres_csv:
        parts.append(f"Genres: {genres_csv}")

    # Keywords as comma-separated values
    combined_keywords = movie.overall_keywords + movie.plot_keywords
    keywords_csv = ", ".join(combined_keywords) if combined_keywords else ""
    if keywords_csv:
        parts.append(keywords_csv)


    parts.append("\n# Production:")

    # Production information
    production_text = movie.production_text()
    if production_text:
        parts.append(production_text)

    # Languages information
    languages_text = movie.languages_text()
    if languages_text:
        parts.append(languages_text)

    # Release decade bucket
    decade_bucket = movie.release_decade_bucket()
    if decade_bucket:
        parts.append(decade_bucket)

    # Duration bucket
    duration_bucket = movie.duration_bucket()
    if duration_bucket:
        parts.append(f"Duration: {duration_bucket}")

    # Budget scale for era
    budget_bucket = budget_size_to_vector_text(movie.budget_bucket_for_era())
    if budget_bucket:
        parts.append(budget_bucket)

    if movie.production_metadata:
        if movie.production_metadata.production_keywords.terms:
            parts.append(", ".join(movie.production_metadata.production_keywords.terms))
        if movie.production_metadata.sources_of_inspiration.production_mediums:
            parts.append(", ".join(movie.production_metadata.sources_of_inspiration.production_mediums))
        if movie.production_metadata.sources_of_inspiration.sources_of_inspiration:
            parts.append(", ".join(movie.production_metadata.sources_of_inspiration.sources_of_inspiration))

    parts.append("\n# Cast and Characters:")

    cast_text = movie.cast_text()
    if cast_text:
        parts.append(cast_text)

    # Characters information
    characters_text = movie.characters_text()
    if characters_text:
        parts.append(characters_text)

    parts.append("\n# Themes and Lessons:")

    if movie.plot_analysis_metadata:
        parts.append(f"Core concept: {movie.plot_analysis_metadata.core_concept.core_concept_label.lower()}")
        themes = [theme.theme_label.lower() for theme in movie.plot_analysis_metadata.themes_primary]
        parts.append(f"Themes: {", ".join(themes)}")
        lessons = [lesson.lesson_label.lower() for lesson in movie.plot_analysis_metadata.lessons_learned]
        parts.append(f"Lessons: {", ".join(lessons)}")

    parts.append("\n# Audience Reception:")

    if movie.viewer_experience_metadata:
        parts.append(f"Emotional palette: {", ".join(movie.viewer_experience_metadata.emotional_palette.terms)}")

    if movie.watch_context_metadata:
        parts.append(f"Key draws: {", ".join(movie.watch_context_metadata.key_movie_feature_draws.terms)}")

    # Maturity guidance
    maturity_guidance = movie.maturity_guidance_text()
    if maturity_guidance:
        parts.append(maturity_guidance)

    # Reception information
    reception_tier = movie.reception_tier()
    if reception_tier:
        parts.append(f"Reception: {reception_tier}")

    if movie.reception_metadata:
        parts.append(f"Praises: {", ".join(movie.reception_metadata.praise_attributes)}")
        parts.append(f"Complaints: {", ".join(movie.reception_metadata.complaint_attributes)}")

    # Join all parts with newlines
    return "\n".join(parts)


def create_plot_events_vector_text(movie: Movie) -> str | None:
    """
    Creates the text representation for the plot_events vector embedding.

    Uses a fallback hierarchy to select the richest available plot text:
      1. Full original scraped synopsis (longest from IMDB synopses)
      2. Generated plot_summary via plot_events metadata
      3. Longest scraped plot_summary entry
      4. IMDB overview
    """
    text = ""

    if movie.imdb_data.synopses and max(movie.imdb_data.synopses, key=len):
        # 1. Full original scraped synopsis — richest source of plot detail
        text = max(movie.imdb_data.synopses, key=len)
    elif movie.plot_events_metadata:
        # 2. LLM-generated plot summary from plot_events metadata
        text = movie.plot_events_metadata.embedding_text()
    elif movie.imdb_data.plot_summaries and max(movie.imdb_data.plot_summaries, key=len):
        # 3. Longest scraped plot_summary entry
        text = max(movie.imdb_data.plot_summaries, key=len)
    elif movie.imdb_data.overview:
        # 4. Overview as last resort
        text = movie.imdb_data.overview

    return normalize_string(text) if text else None


def create_plot_events_vector_text_fallback(movie: Movie) -> str | None:
    """
    Fallback text for plot_events when the primary text exceeds the
    embedding model's token limit (8,191 tokens for text-embedding-3-small).

    Uses the longer of:
      1. Longest scraped plot_summary entry
      2. Generated plot_summary from plot_events metadata
    Then falls back to overview if neither is available.
    """
    text = ""

    # Collect the two candidates and pick the longer one
    longest_plot_summary = ""
    if movie.imdb_data.plot_summaries:
        longest_plot_summary = max(movie.imdb_data.plot_summaries, key=len)

    generated_plot_summary = ""
    if movie.plot_events_metadata:
        generated_plot_summary = movie.plot_events_metadata.embedding_text()

    # Pick whichever candidate is longer
    best = max(longest_plot_summary, generated_plot_summary, key=len)

    if best:
        text = best
    elif movie.imdb_data.overview:
        text = movie.imdb_data.overview

    return normalize_string(text) if text else None


def create_plot_analysis_vector_text(movie: BaseMovie) -> str:
    parts = []

    if movie.plot_analysis_metadata:
        parts.append(str(movie.plot_analysis_metadata))
    if movie.genres_subset():
        parts.append(", ".join(movie.genres_subset()))
    if movie.plot_keywords:
        parts.append(", ".join(movie.plot_keywords))


    return "\n".join(parts)


def create_narrative_techniques_vector_text(narrative_techniques: NarrativeTechniquesOutput) -> str:
    return narrative_techniques.embedding_text()


def create_viewer_experience_vector_text(viewer_experience: ViewerExperienceOutput) -> str:
    return viewer_experience.embedding_text()


def create_watch_context_vector_text(watch_context: WatchContextOutput) -> str:
    return watch_context.embedding_text()


def create_production_vector_text(movie: BaseMovie) -> str:
    parts = []

    parts.append("\n# Production:")

    # Production information
    production_text = movie.production_text()
    if production_text:
        parts.append(production_text.lower())

    # Languages information
    languages_text = movie.languages_text()
    if languages_text:
        parts.append(languages_text.lower())

    # Release decade bucket
    decade_bucket = movie.release_decade_bucket()
    if decade_bucket:
        parts.append(decade_bucket.lower())

    # Budget scale for era
    budget_bucket = budget_size_to_vector_text(movie.budget_bucket_for_era())
    if budget_bucket:
        parts.append(budget_bucket.lower())

    if movie.production_metadata:
        if movie.production_metadata.production_keywords.terms:
            parts.append(", ".join(movie.production_metadata.production_keywords.terms))
        if movie.production_metadata.sources_of_inspiration.production_mediums:
            parts.append(", ".join(movie.production_metadata.sources_of_inspiration.production_mediums))
        if movie.production_metadata.sources_of_inspiration.sources_of_inspiration:
            parts.append(", ".join(movie.production_metadata.sources_of_inspiration.sources_of_inspiration))

    parts.append("\n# Cast and Characters:")

    cast_text = movie.cast_text()
    if cast_text:
        parts.append(cast_text)

    # Characters information
    characters_text = movie.characters_text()
    if characters_text:
        parts.append(characters_text)

    # Maturity rating
    parts.append(f"{movie.maturity_rating.lower()} maturity rating")

    return "\n".join(parts)


def create_reception_vector_text(movie: BaseMovie) -> str:
    parts = []

    if movie.reception_tier():
        parts.append(movie.reception_tier().lower())

    if movie.reception_metadata:
        parts.append(f"{movie.reception_metadata.new_reception_summary.lower()}")
        parts.append(f"Praises: {", ".join(movie.reception_metadata.praise_attributes)}")
        parts.append(f"Complaints: {", ".join(movie.reception_metadata.complaint_attributes)}")

    return "\n".join(parts)
