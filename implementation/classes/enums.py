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


class WatchProviderType(IntEnum):
    """Enum representing types of watch provider services."""
    SUBSCRIPTION = 1
    PURCHASE = 2
    RENT = 3

    @classmethod
    def from_string(cls, provider_type: str) -> "WatchProviderType | None":
        """
        Convert a string to a WatchProviderType enum value.
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
            WatchProviderType.SUBSCRIPTION: "subscription",
            WatchProviderType.PURCHASE: "purchase",
            WatchProviderType.RENT: "rent",
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
    ACTION = "Action"
    ADVENTURE = "Adventure"
    ANIMATION = "Animation"
    BIOGRAPHY = "Biography"
    COMEDY = "Comedy"
    CRIME = "Crime"
    DOCUMENTARY = "Documentary"
    DRAMA = "Drama"
    FAMILY = "Family"
    FANTASY = "Fantasy"
    FILM_NOIR = "Film-Noir"
    GAME_SHOW = "Game-Show"
    HISTORY = "History"
    HORROR = "Horror"
    MUSIC = "Music"
    MUSICAL = "Musical"
    MYSTERY = "Mystery"
    NEWS = "News"
    REALITY_TV = "Reality-TV"
    ROMANCE = "Romance"
    SCI_FI = "Sci-Fi"
    SHORT = "Short"
    SPORT = "Sport"
    TALK_SHOW = "Talk-Show"
    THRILLER = "Thriller"
    WAR = "War"
    WESTERN = "Western"

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