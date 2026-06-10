"""
Enum classes for movie data models.

This module contains all Enum classes used for movie data representation
across the project.
"""

import logging
from enum import Enum
from typing import Iterable

from implementation.misc.helpers import normalize_string

logger = logging.getLogger(__name__)


class MaturityRating(Enum):
    maturity_rank: int
    value: str
    aliases: frozenset

    def __new__(
        cls,
        maturity_rank: int,
        value: str,
        aliases: Iterable[str] = (),
    ) -> "MaturityRating":
        """Create a MaturityRating member with rank, canonical value, and aliases.

        ``aliases`` holds the raw, human-readable strings (TV / legacy / foreign
        certificates) that should resolve to this rating. They are normalized
        once at module load into the reverse-lookup map (see
        ``_build_maturity_alias_map``) — never normalize them by hand here.
        """
        obj = object.__new__(cls)
        obj._value_ = value
        obj.maturity_rank = maturity_rank
        obj.aliases = frozenset(aliases)
        return obj

    @classmethod
    def _missing_(cls, value: object) -> "MaturityRating | None":
        """Resolve a non-canonical input string to a supported rating.

        Enum calls this only when ``value`` doesn't match a member's canonical
        ``_value_``. We normalize the input (so callers needn't pre-normalize)
        and consult the alias map. Returning ``None`` lets Enum raise the usual
        ``ValueError``, preserving the fallback contract of callers. Kept pure
        (no logging) since it fires on every non-canonical lookup codebase-wide.
        """
        if not isinstance(value, str):
            return None
        return _MATURITY_ALIAS_TO_MEMBER.get(normalize_string(value))

    @classmethod
    def from_string_with_default(cls, value: str | None) -> "MaturityRating":
        """Resolve a raw maturity string to a member, defaulting to UNRATED.

        Empty/None input means "no rating" and silently maps to UNRATED. A
        non-empty value we fail to resolve is logged: it's a real certificate
        we don't yet recognize (e.g. a video-game rating leaking in), worth
        surfacing rather than swallowing.
        """
        # Normalize outside the try so the value is available for logging even
        # if resolution fails, and to guard None before normalize_string runs.
        normalized_value = normalize_string(value) if value else ""
        try:
            return cls(normalized_value)
        except ValueError:
            if normalized_value:
                logger.warning(
                    "Unrecognized maturity rating %r; defaulting to UNRATED", value
                )
            return cls.UNRATED

    # Each member: (rank, canonical value, alternative-form aliases). Aliases
    # are the *non-canonical* strings (TV / legacy / foreign certs) that should
    # resolve to this rating; the canonical value is matched natively by Enum,
    # so it is intentionally not repeated here. Aliases are raw strings,
    # normalized identically to ingest/query input at load time.
    G = (1, "g", ("tv-g", "tv-y", "approved", "passed"))
    PG = (2, "pg", ("tv-pg", "tv-y7", "tv-y7-fv", "gp", "m", "m/pg"))
    PG_13 = (3, "pg-13", ("tv-14", "13+", "12"))
    R = (4, "r", ("tv-ma", "16+", "18+", "18"))
    NC_17 = (5, "nc-17", ("x",))
    UNRATED = (999, "unrated", ("not rated", "nr"))


def _build_maturity_alias_map() -> dict[str, MaturityRating]:
    """Build the normalized {alias -> member} reverse map for MaturityRating.

    Only the *non-canonical* aliases are mapped here — canonical values are
    resolved natively by Enum value-lookup, so ``_missing_`` (the sole consumer
    of this map) never sees them. Every alias is run through ``normalize_string``
    so lookups match ingest/query-normalized input exactly. Raises at import if
    a normalized alias resolves to two *different* members (catches typos early);
    duplicate keys pointing to the same member are a harmless dedup — e.g.
    "18+" and "18" both normalize to "18" under R.
    """
    mapping: dict[str, MaturityRating] = {}
    for member in MaturityRating:
        for raw in member.aliases:
            normalized = normalize_string(raw)
            existing = mapping.get(normalized)
            if existing is not None and existing is not member:
                raise ValueError(
                    f"Maturity alias collision: {normalized!r} maps to both "
                    f"{existing.name} and {member.name}"
                )
            mapping[normalized] = member
    return mapping


# Built once at module load; consulted by MaturityRating._missing_.
_MATURITY_ALIAS_TO_MEMBER: dict[str, MaturityRating] = _build_maturity_alias_map()


class StreamingAccessType(Enum):
    """Enum representing types of watch provider services."""
    type_id: int
    value: str

    def __new__(cls, type_id: int, value: str) -> "StreamingAccessType":
        """Create a StreamingAccessType enum member with rank and display value."""
        obj = object.__new__(cls)
        obj._value_ = value
        obj.type_id = type_id
        return obj

    @classmethod
    def from_string(cls, name: str) -> "StreamingAccessType | None":
        """Resolve a string label to its StreamingAccessType enum member."""
        normalized = normalize_string(name)
        try:
            return cls(normalized)
        except ValueError:
            return None

    @classmethod
    def from_type_id(cls, type_id: int) -> "StreamingAccessType | None":
        """Resolve a numeric type_id to its StreamingAccessType enum member."""
        for member in cls:
            if member.type_id == type_id:
                return member
        return None

    SUBSCRIPTION = (1, "subscription")
    BUY = (2, "buy")
    RENT = (3, "rent")


class VectorCollectionName(Enum):
    """Enum representing Qdrant vector collection names."""
    DENSE_ANCHOR_VECTORS = "dense_anchor_vectors"
    PLOT_EVENTS_VECTORS = "plot_events_vectors"
    PLOT_ANALYSIS_VECTORS = "plot_analysis_vectors"
    NARRATIVE_TECHNIQUES_VECTORS = "narrative_techniques_vectors"
    VIEWER_EXPERIENCE_VECTORS = "viewer_experience_vectors"
    WATCH_CONTEXT_VECTORS = "watch_context_vectors"
    PRODUCTION_VECTORS = "production_vectors"
    RECEPTION_VECTORS = "reception_vectors"

class VectorName(Enum):
    ANCHOR = "anchor"
    PLOT_EVENTS = "plot_events"
    PLOT_ANALYSIS = "plot_analysis"
    VIEWER_EXPERIENCE = "viewer_experience"
    WATCH_CONTEXT = "watch_context"
    NARRATIVE_TECHNIQUES = "narrative_techniques"
    PRODUCTION = "production"
    RECEPTION = "reception"


class RelevanceSize(Enum):
    NOT_RELEVANT = "not_relevant"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"

# ============================
#  Metadata Preferences Enums
# ============================

class MetadataPreferenceName(Enum):
    RELEASE_DATE = "release_date"
    DURATION = "duration"
    GENRES = "genres"
    AUDIO_LANGUAGES = "audio_languages"
    WATCH_PROVIDERS = "watch_providers"
    MATURITY_RATING = "maturity_rating"
    POPULARITY = "popularity"
    RECEPTION = "reception"
    BUDGET_SIZE = "budget_size"

class Genre(Enum):
    genre_id: int
    value: str
    normalized_name: str

    def __new__(cls, genre_id: int, value: str, normalized_name: str) -> "Genre":
        """Create a Genre enum member with a stable numeric ID, display name, and normalized name.

        Args:
            genre_id: Stable 1-based integer identifier assigned in alphabetical order.
            value: Human-readable display name (e.g. "Sci-Fi").
            normalized_name: Lowercased/normalized form used for indexing and matching.
        """
        obj = object.__new__(cls)
        obj._value_ = value
        obj.genre_id = genre_id
        obj.normalized_name = normalized_name
        return obj

    ACTION      = (1,  "Action",     "action")
    ADVENTURE   = (2,  "Adventure",  "adventure")
    ANIMATION   = (3,  "Animation",  "animation")
    BIOGRAPHY   = (4,  "Biography",  "biography")
    COMEDY      = (5,  "Comedy",     "comedy")
    CRIME       = (6,  "Crime",      "crime")
    DOCUMENTARY = (7,  "Documentary","documentary")
    DRAMA       = (8,  "Drama",      "drama")
    FAMILY      = (9,  "Family",     "family")
    FANTASY     = (10, "Fantasy",    "fantasy")
    FILM_NOIR   = (11, "Film-Noir",  "film-noir")
    GAME_SHOW   = (12, "Game-Show",  "game-show")
    HISTORY     = (13, "History",    "history")
    HORROR      = (14, "Horror",     "horror")
    MUSIC       = (15, "Music",      "music")
    MUSICAL     = (16, "Musical",    "musical")
    MYSTERY     = (17, "Mystery",    "mystery")
    NEWS        = (18, "News",       "news")
    REALITY_TV  = (19, "Reality-TV", "reality-tv")
    ROMANCE     = (20, "Romance",    "romance")
    SCI_FI      = (21, "Sci-Fi",     "sci-fi")
    SHORT       = (22, "Short",      "short")
    SPORT       = (23, "Sport",      "sport")
    TALK_SHOW   = (24, "Talk-Show",  "talk-show")
    THRILLER    = (25, "Thriller",   "thriller")
    WAR         = (26, "War",        "war")
    WESTERN     = (27, "Western",    "western")

    @classmethod
    def from_string(cls, name: str) -> "Genre | None":
        """Resolve a genre name string to its Genre enum member.

        Uses normalize_string for case/whitespace tolerance, then matches
        against each member's normalized_name.

        Args:
            name: Raw genre name (e.g. "Sci-Fi", "sci-fi", " Action ").

        Returns:
            The matching Genre member, or None if unrecognized.
        """
        normalized = normalize_string(name)
        if not normalized:
            return None
        for member in cls:
            if member.normalized_name == normalized:
                return member
        return None

class DateMatchOperation(Enum):
    EXACT = "exact"
    BEFORE = "before"
    AFTER = "after"
    BETWEEN = "between"

class NumericalMatchOperation(Enum):
    EXACT = "exact"
    BETWEEN = "between"
    LESS_THAN = "less_than"
    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"

class RatingMatchOperation(Enum):
    EXACT = "exact"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"

class ReceptionType(Enum):
    CRITICALLY_ACCLAIMED = "critically_acclaimed"
    POORLY_RECEIVED = "poorly_received"
    NO_PREFERENCE = "no_preference"

class BudgetSize(Enum):
    SMALL = "small"
    LARGE = "large"
    NO_PREFERENCE = "no_preference"

# ============================
#     Lexical Entity Enums
# ============================

class EntityCategory(Enum):
    """Enum representing the various categories of lexical entities."""
    CHARACTER = "character"
    FRANCHISE = "franchise"
    MOVIE_TITLE = "movie_title"
    PERSON = "person"
    STUDIO = "studio"
