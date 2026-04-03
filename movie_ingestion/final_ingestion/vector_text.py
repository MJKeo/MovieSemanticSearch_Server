"""
Vector text generation functions for each of the 8 named vector spaces.

This module is the single source of truth for converting a Movie into
the text representation that will be embedded for each vector space. The
same functions are used at ingestion time (batch and single-movie) and
must stay in sync with the vector space definitions in Qdrant.

Vector spaces:
    1. anchor            — broad-recall identity fingerprint
    2. plot_events       — chronological plot details
    3. plot_analysis     — themes, arcs, concepts
    4. narrative_techniques — storytelling style/tone
    5. viewer_experience — emotional tone, pacing
    6. watch_context     — when/how to watch
    7. production        — budget, locations, technical
    8. reception         — critical reception, awards
"""

from implementation.classes.enums import BudgetSize
from implementation.misc.helpers import normalize_string
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

def create_anchor_vector_text(movie: Movie) -> str:
    """Create the text representation for the anchor (dense_anchor) vector.

    The anchor is always queried and weighted as a broad-recall safety net
    (0.8× mean of active non-anchor weights). Its job is to capture the
    movie's core semantic identity without duplicating signals that
    specialized spaces or lexical/metadata channels handle better.

    Formatting rules:
        - Prose fields use .lower() — self-contextualizing, no label needed
        - Categorical term lists use normalize_string() with a semantic
          label to disambiguate their role for the embedding model

    Deliberately excluded (handled better elsewhere):
        - Cast/crew/character names → lexical search
        - Production companies/filming locations → production vector
        - Detailed maturity reasoning → metadata filters
        - Reception praise/complaint lists → reception vector
    """
    parts: list[str] = []

    # -- Identity: what is this movie? --

    # Title with original title for international films
    title = movie.title_with_original()
    if title:
        parts.append(title.lower())

    # Elevator pitch — single most concise "what is this movie about" (~6 words)
    if movie.plot_analysis_metadata:
        pitch = movie.plot_analysis_metadata.elevator_pitch_with_justification.elevator_pitch
        parts.append(pitch.lower())

    # Generalized overview gives thematic context in 1-3 sentences.
    # Fall back to IMDB overview when plot_analysis wasn't generated.
    if movie.plot_analysis_metadata:
        parts.append(movie.plot_analysis_metadata.generalized_plot_overview.lower())
    elif movie.imdb_data.overview:
        parts.append(movie.imdb_data.overview.lower())

    # -- Classification: what type of movie is this? --

    # Deduplicated merge of LLM genre_signatures + IMDB genres
    genres = movie.deduplicated_genres()
    if genres:
        genres = [normalize_string(g) for g in genres]
        parts.append("genres: " + ", ".join(genres))

    # Overall keywords — broad topical descriptors. Plot keywords excluded
    # to avoid over-emphasizing minor content details (e.g. "male nudity").
    if movie.imdb_data.overall_keywords:
        keywords = [normalize_string(kw) for kw in movie.imdb_data.overall_keywords]
        parts.append("keywords: " + ", ".join(keywords))

    # Thematic concepts from plot analysis
    if movie.plot_analysis_metadata and movie.plot_analysis_metadata.thematic_concepts:
        themes = [
            normalize_string(t.concept_label)
            for t in movie.plot_analysis_metadata.thematic_concepts
        ]
        parts.append("themes: " + ", ".join(themes))

    # Source material and franchise position — helps match queries like
    # "book adaptations", "sequels", "remakes"
    if movie.source_of_inspiration_metadata:
        src = movie.source_of_inspiration_metadata
        if src.source_material:
            normalized = [normalize_string(t) for t in src.source_material]
            parts.append("source material: " + ", ".join(normalized))
        if src.franchise_lineage:
            normalized = [normalize_string(t) for t in src.franchise_lineage]
            parts.append("franchise position: " + ", ".join(normalized))

    # -- Experience: what does watching it feel like? --

    # Cherry-picked experiential signals give anchor a taste of the
    # viewer-experience and watch-context dimensions without full duplication
    if movie.viewer_experience_metadata:
        palette_terms = movie.viewer_experience_metadata.emotional_palette.terms
        if palette_terms:
            normalized = [normalize_string(t) for t in palette_terms]
            parts.append("emotional palette: " + ", ".join(normalized))

    if movie.watch_context_metadata:
        draw_terms = movie.watch_context_metadata.key_movie_feature_draws.terms
        if draw_terms:
            normalized = [normalize_string(t) for t in draw_terms]
            parts.append("key draws: " + ", ".join(normalized))

    # -- Context: when, where, how? --

    # Release decade with semantic era label
    decade_bucket = movie.release_decade_bucket()
    if decade_bucket:
        parts.append(decade_bucket.lower())

    # All languages together as a flat list
    if movie.imdb_data.languages:
        languages = [normalize_string(lang) for lang in movie.imdb_data.languages]
        parts.append("languages: " + ", ".join(languages))

    # Budget scale relative to era (only when notable)
    budget_text = budget_size_to_vector_text(movie.budget_bucket_for_era())
    if budget_text:
        parts.append(budget_text)

    # Maturity — prose reasoning when available, MPA description as fallback
    maturity = movie.maturity_text_short()
    if maturity:
        parts.append(maturity.lower())

    # -- Reception: how was it received? --

    # Prose reception summary — evaluative "what did people think?"
    if movie.reception_metadata:
        parts.append(movie.reception_metadata.reception_summary.lower())

    # Tier label as a compact quality signal
    reception_tier = movie.reception_tier()
    if reception_tier:
        parts.append("reception: " + reception_tier.lower())

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

    return text.lower() if text else None


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

    return text.lower() if text else None


def create_plot_analysis_vector_text(movie: Movie) -> str | None:
    """
    Creates the text representation for the plot_analysis vector embedding.

    Merges TMDB genres into the metadata's genre_signatures so they are
    embedded as a single labeled field, then delegates to embedding_text().
    """
    if not movie.plot_analysis_metadata:
        return None

    # Append TMDB genres to the LLM-generated genre signatures for grounding
    meta = movie.plot_analysis_metadata
    if movie.imdb_data.genres:
        existing = set(g.lower() for g in meta.genre_signatures)
        for genre in movie.imdb_data.genres:
            if genre.lower() not in existing:
                meta.genre_signatures.append(genre)

    return meta.embedding_text()


def create_narrative_techniques_vector_text(movie: Movie) -> str | None:
    if not movie.narrative_techniques_metadata:
        return None
    return movie.narrative_techniques_metadata.embedding_text()


def create_viewer_experience_vector_text(movie: Movie) -> str | None:
    if not movie.viewer_experience_metadata:
        return None
    return movie.viewer_experience_metadata.embedding_text()


def create_watch_context_vector_text(movie: Movie) -> str | None:
    if not movie.watch_context_metadata:
        return None
    return movie.watch_context_metadata.embedding_text()


def create_production_vector_text(movie: Movie) -> str:
    """Build production vector text focused on how/where the film was made.

    Excludes cast, characters, and maturity rating — those are handled
    by lexical search (names) or other vector spaces (content classification).
    """
    parts = []

    # Origin, companies, filming locations (labeled format).
    # Filming locations excluded for animation — they're irrelevant
    # (voice actors record in studios, not on location).
    production_text = movie.production_text(include_filming_locations=not movie.is_animation())
    if production_text:
        parts.append(production_text.lower())

    # Primary and additional languages
    languages_text = movie.languages_text()
    if languages_text:
        parts.append(languages_text.lower())

    # Semantic era label (e.g. "Release date: 1940s, golden age of hollywood")
    decade_bucket = movie.release_decade_bucket()
    if decade_bucket:
        parts.append(decade_bucket.lower())

    # Budget scale relative to era
    budget_bucket = budget_size_to_vector_text(movie.budget_bucket_for_era())
    if budget_bucket:
        parts.append(f"budget: {budget_bucket.lower()}")

    # Production medium — binary: "animation" or "live action"
    medium = "animation" if movie.is_animation() else "live action"
    parts.append(f"production medium: {medium}")

    # Source material and franchise lineage (labeled via embedding_text).
    # When metadata is None (not generated), default to "original screenplay"
    # since most movies are original works.
    if movie.source_of_inspiration_metadata:
        source_text = movie.source_of_inspiration_metadata.embedding_text()
        if source_text:
            parts.append(source_text)
    else:
        parts.append("source material: original screenplay")

    # LLM-generated production keywords (unlabeled — intentionally broad grab-bag)
    if movie.production_keywords_metadata:
        keywords_text = movie.production_keywords_metadata.embedding_text()
        if keywords_text:
            parts.append(keywords_text)

    return "\n".join(parts)


def create_reception_vector_text(movie: Movie) -> str | None:
    if not movie.reception_metadata:
        return None

    parts = []

    tier = movie.reception_tier()
    if tier:
        parts.append(f"reception: {tier.lower()}")

    parts.append(movie.reception_metadata.embedding_text())

    return "\n".join(parts)
