"""
Enum classes for movie data models.

This module contains all Enum classes used for movie data representation
across the project.
"""

from enum import Enum
from implementation.misc.helpers import normalize_string


class MaturityRating(Enum):
    maturity_rank: int
    value: str

    def __new__(cls, maturity_rank: int, value: str) -> "MaturityRating":
        """Create a MaturityRating enum member with rank and display value."""
        obj = object.__new__(cls)
        obj._value_ = value
        obj.maturity_rank = maturity_rank
        return obj

    @classmethod
    def from_string_with_default(cls, value: str) -> "MaturityRating":
        """Create a MaturityRating enum member with rank and display value."""
        try:
            normalized_value = normalize_string(value)
            return cls(normalized_value)
        except ValueError:
            return cls.UNRATED

    G = (1, "g")
    PG = (2, "pg")
    PG_13 = (3, "pg-13")
    R = (4, "r")
    NC_17 = (5, "nc-17")
    UNRATED = (999, "unrated")


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
    """Enum representing ChromaDB vector collection names."""
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
