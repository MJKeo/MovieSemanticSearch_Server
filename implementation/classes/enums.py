"""
Enum classes for movie data models.

This module contains all Enum classes used for movie data representation
across the project.
"""

from enum import Enum, IntEnum
from implementation.misc.helpers import normalize_string


class MaturityRating(IntEnum):
    G = 1
    PG = 2
    PG_13 = 3
    R = 4
    NC_17 = 5
    UNRATED = 999

    @classmethod
    def from_string(cls, rating: str) -> "MaturityRating":
        normalized_rating = normalize_string(rating)
        _map = {
            normalize_string("G"): cls.G,
            normalize_string("PG"): cls.PG,
            normalize_string("PG-13"): cls.PG_13,
            normalize_string("R"): cls.R,
            normalize_string("NC-17"): cls.NC_17,
            normalize_string("Unrated"): cls.UNRATED,
        }
        if normalized_rating not in _map:
            return cls.UNRATED
        return _map[normalized_rating]

    def __str__(self) -> str:
        _labels = {
            MaturityRating.G: "G",
            MaturityRating.PG: "PG",
            MaturityRating.PG_13: "PG-13",
            MaturityRating.R: "R",
            MaturityRating.NC_17: "NC-17",
            MaturityRating.UNRATED: "Unrated",
        }
        return _labels[self]


class WatchMethodType(IntEnum):
    """Enum representing types of watch provider services."""
    SUBSCRIPTION = 1
    PURCHASE = 2
    RENT = 3

    @classmethod
    def from_string(cls, provider_type: str) -> "WatchMethodType | None":
        """
        Convert a string to a WatchMethodType enum value.
        Returns None if the string doesn't match any valid provider type.
        """
        normalized_type = normalize_string(provider_type)
        _map = {
            normalize_string("subscription"): cls.SUBSCRIPTION,
            normalize_string("purchase"): cls.PURCHASE,
            normalize_string("rent"): cls.RENT,
        }
        return _map.get(normalized_type, None)

    def __str__(self) -> str:
        """Return the human-readable string representation of the provider type."""
        _labels = {
            WatchMethodType.SUBSCRIPTION: "subscription",
            WatchMethodType.PURCHASE: "purchase",
            WatchMethodType.RENT: "rent",
        }
        return _labels[self]


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

class StreamingAccessType(Enum):
    SUBSCRIPTION = "subscription"
    RENT = "rent"
    BUY = "buy"

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