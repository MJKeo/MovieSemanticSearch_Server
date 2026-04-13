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
    7. production        — filming locations, production techniques
    8. reception         — critical reception, awards
"""

import tiktoken

from implementation.misc.helpers import normalize_string
from schemas.enums import AwardCeremony
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

# Ordered mapping of prestige award ceremonies to short display names
# for the reception vector text. Insertion order defines the output
# ordering (most prestigious first). Also serves as the membership
# check — ceremonies not in this dict are excluded from vector text.
# Deliberately excludes Razzie Awards — negative-award queries should
# route to structured award data, not prestige-oriented vector text.
_RECEPTION_AWARD_CEREMONY_DISPLAY: dict[AwardCeremony, str] = {
    AwardCeremony.ACADEMY_AWARDS: "academy awards",
    AwardCeremony.GOLDEN_GLOBES:  "golden globes",
    AwardCeremony.BAFTA:          "bafta",
    AwardCeremony.CANNES:         "cannes",
    AwardCeremony.VENICE:         "venice",
    AwardCeremony.BERLIN:         "berlin",
    AwardCeremony.SAG:            "sag",
    AwardCeremony.CRITICS_CHOICE: "critics choice",
    AwardCeremony.SUNDANCE:       "sundance",
    AwardCeremony.SPIRIT_AWARDS:  "spirit awards",
    AwardCeremony.GOTHAM:         "gotham awards",
}


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


def create_production_vector_text(movie: Movie) -> str | None:
    """Build lean production vector text for where/how the film was made.

    The production vector now carries only scraped filming locations plus
    finalized production-technique terms. Structured/filterable facts such
    as country, company, language, decade, budget, source material, and
    franchise position are handled elsewhere and are deliberately excluded.
    """
    parts: list[str] = []

    if not movie.is_animation() and movie.imdb_data.filming_locations:
        locations = ", ".join(movie.imdb_data.filming_locations[:3]).lower()
        parts.append(f"filming_locations: {locations}")

    if movie.production_techniques_metadata:
        techniques_text = movie.production_techniques_metadata.embedding_text()
        if techniques_text:
            parts.append(f"production_techniques: {techniques_text}")

    return "\n".join(parts) if parts else None


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
    Output order follows _RECEPTION_AWARD_CEREMONY_DISPLAY insertion
    order (most prestigious first).
    """
    if not movie.imdb_data.awards:
        return None

    # Collect AwardCeremony enums for winning, known ceremonies.
    # ceremony_id is None for unknown ceremonies — skip those.
    from schemas.enums import CEREMONY_BY_EVENT_TEXT
    winning_ceremonies: set[AwardCeremony] = set()
    for award in movie.imdb_data.awards:
        if award.did_win():
            ceremony_enum = CEREMONY_BY_EVENT_TEXT.get(award.ceremony)
            if ceremony_enum is not None and ceremony_enum in _RECEPTION_AWARD_CEREMONY_DISPLAY:
                winning_ceremonies.add(ceremony_enum)

    if not winning_ceremonies:
        return None

    # Iterate the display dict to preserve prestige ordering.
    ordered_names = [
        display_name
        for ceremony, display_name in _RECEPTION_AWARD_CEREMONY_DISPLAY.items()
        if ceremony in winning_ceremonies
    ]
    if not ordered_names:
        return None

    return "major_award_wins: " + ", ".join(ordered_names)
