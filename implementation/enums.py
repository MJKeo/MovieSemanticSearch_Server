"""
Enum classes for movie data models.

This module contains all Enum classes used for movie data representation
across the project.
"""

from enum import Enum


class MaturityRating(Enum):
    """Enum representing movie maturity/content ratings."""
    G = "G"
    PG = "PG"
    PG_13 = "PG-13"
    R = "R"
    NC_17 = "NC-17"
    UNRATED = "Unrated"

    @classmethod
    def from_string(cls, string: str):
        """
        Match by enum value (e.g., "PG") rather than key (e.g., "PG").
        
        Args:
            string: Maturity rating string to match
            
        Returns:
            Matching MaturityRating enum or UNRATED as default
        """
        for rating in cls:
            if rating.value == string:
                return rating
        return cls.UNRATED


class WatchProviderType(Enum):
    """Enum representing types of watch provider services."""
    SUBSCRIPTION = "subscription"
    PURCHASE = "purchase"
    RENT = "rent"


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
