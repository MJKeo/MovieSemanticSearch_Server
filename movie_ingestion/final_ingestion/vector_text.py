"""
Vector text generation functions for each of the 8 named vector spaces.

This module is the single source of truth for converting a Movie into
the text representation that will be embedded for each vector space. The
same functions are used at ingestion time (batch and single-movie) and
must stay in sync with the vector space definitions in Qdrant.

Vector spaces:
    1. anchor            — lean holistic movie fingerprint
    2. plot_events       — chronological plot details
    3. plot_analysis     — themes, arcs, concepts
    4. narrative_techniques — storytelling style/tone
    5. viewer_experience — emotional tone, pacing
    6. watch_context     — when/how to watch
    7. production        — budget, locations, technical
    8. reception         — critical reception, awards
"""

import tiktoken

from implementation.classes.enums import BudgetSize
from implementation.misc.helpers import normalize_string
from schemas.movie import Movie

# Token limit for text-embedding-3-large. Texts exceeding this cause
# embedding API errors and need to fall back to shorter alternatives.
# (3-small and 3-large share the same 8191-token limit and the same
# cl100k_base tokenizer.)
_EMBEDDING_TOKEN_LIMIT = 8_191

# Cheap character-length gate to avoid running tiktoken on every movie.
# Only texts above this threshold get tokenized (~561 of ~109K movies).
# At ~3.7 chars/token, 15K chars ≈ 4K tokens — well under the limit,
# so anything below this is guaranteed safe.
_CHAR_GATE_THRESHOLD = 15_000

_TIKTOKEN_ENC = tiktoken.encoding_for_model("text-embedding-3-large")

_RECEPTION_AWARD_CEREMONY_ORDER: tuple[tuple[str, str], ...] = (
    ("Academy Awards, USA", "academy awards"),
    ("Golden Globes, USA", "golden globes"),
    ("BAFTA Awards", "bafta"),
    ("Cannes Film Festival", "cannes"),
    ("Venice Film Festival", "venice"),
    ("Berlin International Film Festival", "berlin"),
    ("Actor Awards", "sag"),
    ("Critics Choice Awards", "critics choice"),
    ("Sundance Film Festival", "sundance"),
    # Deliberately excluded: Razzie Awards. Negative-award queries should
    # route to structured award data, not prestige-oriented vector text.
    ("Film Independent Spirit Awards", "spirit awards"),
    ("Gotham Awards", "gotham awards"),
)
_RECEPTION_AWARD_CEREMONY_DISPLAY = dict(_RECEPTION_AWARD_CEREMONY_ORDER)


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

    The anchor is a lean, holistic fingerprint for "movie as a whole"
    similarity. It keeps only high-level identity, thematic, experiential,
    maturity, and reception summary signals, while leaving structured or
    filterable facts to specialized spaces and deterministic retrieval.

    Deliberately excluded (handled better elsewhere):
        - Keywords, source material, franchise position
        - Languages, decade, budget / box office
        - Awards, reception tiers, detailed praise / complaint tags
        - Watch-context scenarios / motivations and viewer-experience negations
    """
    parts: list[str] = []

    title = movie.tmdb_data.title
    if title:
        parts.append(f"title: {title.lower()}")

    original_title = movie.imdb_data.original_title
    if original_title and original_title != title:
        parts.append(f"original_title: {original_title.lower()}")

    if movie.plot_analysis_metadata:
        pitch = movie.plot_analysis_metadata.elevator_pitch_with_justification.elevator_pitch
        parts.append(f"identity_pitch: {pitch.lower()}")

    overview = ""
    if movie.plot_analysis_metadata:
        overview = movie.plot_analysis_metadata.generalized_plot_overview
    elif movie.imdb_data.overview:
        overview = movie.imdb_data.overview
    if overview:
        parts.append(f"identity_overview: {overview.lower()}")

    if movie.plot_analysis_metadata and movie.plot_analysis_metadata.genre_signatures:
        genre_signatures = [
            normalize_string(signature)
            for signature in movie.plot_analysis_metadata.genre_signatures
        ]
        parts.append("genre_signatures: " + ", ".join(genre_signatures))

    if movie.plot_analysis_metadata and movie.plot_analysis_metadata.thematic_concepts:
        themes = [
            normalize_string(t.concept_label)
            for t in movie.plot_analysis_metadata.thematic_concepts
        ]
        parts.append("themes: " + ", ".join(themes))

    if movie.viewer_experience_metadata:
        palette_terms = movie.viewer_experience_metadata.emotional_palette.terms
        if palette_terms:
            normalized = [normalize_string(t) for t in palette_terms]
            parts.append("emotional_palette: " + ", ".join(normalized))

    if movie.watch_context_metadata:
        draw_terms = movie.watch_context_metadata.key_movie_feature_draws.terms
        if draw_terms:
            normalized = [normalize_string(t) for t in draw_terms]
            parts.append("key_draws: " + ", ".join(normalized))

    maturity = movie.maturity_text_short()
    if maturity:
        parts.append(f"maturity_summary: {maturity.lower()}")

    if movie.reception_metadata and movie.reception_metadata.reception_summary:
        parts.append(
            f"reception_summary: {movie.reception_metadata.reception_summary.lower()}"
        )

    return "\n".join(parts)


def _exceeds_token_limit(text: str) -> bool:
    """Check if text exceeds the embedding model's token limit.

    Uses a cheap character-length gate first — only texts above
    _CHAR_GATE_THRESHOLD are tokenized with tiktoken.
    """
    if len(text) <= _CHAR_GATE_THRESHOLD:
        return False
    print(f"   Character gate threshold tripped: {len(text)}")
    token_count = len(_TIKTOKEN_ENC.encode(text))
    print(f"   Token count: {token_count}")
    return token_count >= _EMBEDDING_TOKEN_LIMIT


def _plot_events_primary_text(movie: Movie) -> str:
    """Select the richest available plot text via fallback hierarchy.

    Priority:
      1. Full original scraped synopsis (longest from IMDB synopses)
      2. Generated plot_summary via plot_events metadata
      3. Longest scraped plot_summary entry
      4. IMDB overview
    """
    if movie.imdb_data.synopses and max(movie.imdb_data.synopses, key=len):
        # 1. Full original scraped synopsis — richest source of plot detail
        return max(movie.imdb_data.synopses, key=len)
    if movie.plot_events_metadata:
        # 2. LLM-generated plot summary from plot_events metadata
        return movie.plot_events_metadata.embedding_text()
    if movie.imdb_data.plot_summaries and max(movie.imdb_data.plot_summaries, key=len):
        # 3. Longest scraped plot_summary entry
        return max(movie.imdb_data.plot_summaries, key=len)
    if movie.imdb_data.overview:
        # 4. Overview as last resort
        return movie.imdb_data.overview
    return ""


def _plot_events_fallback_text(movie: Movie) -> str:
    """Shorter alternative text for when primary text exceeds the token limit.

    Uses the longer of:
      1. Longest scraped plot_summary entry
      2. Generated plot_summary from plot_events metadata
    Then falls back to overview if neither is available.
    """
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
        return best
    if movie.imdb_data.overview:
        return movie.imdb_data.overview
    return ""


def create_plot_events_vector_text(movie: Movie) -> str | None:
    """Create the text representation for the plot_events vector embedding.

    Selects the richest available plot text, but if the primary text
    exceeds the embedding model's token limit (8,191 tokens for
    text-embedding-3-large), falls back to a shorter alternative that
    skips the full synopsis in favor of plot summaries or metadata.
    """
    text = _plot_events_primary_text(movie)
    if text:
        text = text.lower()

    if text and _exceeds_token_limit(text):
        text = _plot_events_fallback_text(movie)
        if text:
            text = text.lower()
        print(f"   Fallback events being used with length {len(text)}")

    return text if text else None


def create_plot_analysis_vector_text(movie: Movie) -> str | None:
    """
    Creates the text representation for the plot_analysis vector embedding.

    Thin wrapper over PlotAnalysisOutput.embedding_text(). TMDB genres are
    no longer merged in — under V2 they are a deterministic hard filter
    via movie_card.genre_ids, and the LLM-generated genre_signatures
    already carry the thematic phrasing this space is responsible for.
    """
    if not movie.plot_analysis_metadata:
        return None
    return movie.plot_analysis_metadata.embedding_text()


def create_narrative_techniques_vector_text(movie: Movie) -> str | None:
    if not movie.narrative_techniques_metadata:
        return None
    return movie.narrative_techniques_metadata.embedding_text()


def create_viewer_experience_vector_text(movie: Movie) -> str | None:
    if not movie.viewer_experience_metadata:
        return None
    return movie.viewer_experience_metadata.embedding_text()


def create_watch_context_vector_text(movie: Movie) -> str | None:
    """Return watch-context embedding text in labeled multiline schema format."""
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

    parts = [movie.reception_metadata.embedding_text()]

    award_wins_text = _reception_award_wins_text(movie)
    if award_wins_text:
        parts.append(award_wins_text)

    return "\n".join(parts)


def _reception_award_wins_text(movie: Movie) -> str | None:
    """Summarize prestige-oriented award wins for the reception vector.

    Uses only winning rows from tracked major ceremonies, collapses
    multiple wins in the same ceremony to a single ceremony label, and
    omits Razzie entirely so "award-winning" semantics stay positive.
    """
    if not movie.imdb_data.awards:
        return None

    winning_ceremonies = {
        award.ceremony
        for award in movie.imdb_data.awards
        if award.did_win() and award.ceremony in _RECEPTION_AWARD_CEREMONY_DISPLAY
    }
    if not winning_ceremonies:
        return None

    ordered_ceremonies = [
        display_name
        for ceremony, display_name in _RECEPTION_AWARD_CEREMONY_ORDER
        if ceremony in winning_ceremonies
    ]
    if not ordered_ceremonies:
        return None

    return "major_award_wins: " + ", ".join(ordered_ceremonies)
