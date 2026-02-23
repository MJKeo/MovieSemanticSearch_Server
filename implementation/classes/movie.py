from typing import Optional
from .schemas import (
    PlotAnalysisMetadata, 
    PlotEventsMetadata,
    WatchProvider,
    ParentalGuideItem,
    IMDBFeaturedReview,
    ViewerExperienceMetadata,
    WatchContextMetadata,
    IMDBReviewTheme,
    NarrativeTechniquesMetadata,
    ReceptionMetadata,
    ProductionMetadata
)
from datetime import datetime, timezone
from pydantic import BaseModel, ConfigDict
from .enums import MaturityRating, WatchMethodType, Genre
from .watch_providers import FILTERABLE_WATCH_PROVIDER_IDS
from implementation.misc.helpers import normalize_string, tokenize_title_phrase, create_watch_provider_offering_key
import re


class BaseMovie(BaseModel):
    """
    Represents a complete IMDb movie with all metadata.
    
    Configured to serialize enums as their values for JSON compatibility.
    """
    model_config = ConfigDict(use_enum_values=True)
    
    id: str
    tmdb_id: int
    # Base stats
    title: str
    original_title: Optional[str] = None
    overall_keywords: list[str]
    release_date: str
    duration: int
    genres: list[str]
    countries_of_origin: list[str]
    languages: list[str]
    filming_locations: list[str] = []
    budget: Optional[int] = None
    watch_providers: list[WatchProvider]
    poster_url: Optional[str] = None
    # Maturity
    maturity_rating: str
    maturity_reasoning: list[str] = []
    parental_guide_items: list[ParentalGuideItem] = []
    # Plot
    overview: str
    plot_keywords: list[str]
    # Cast
    directors: list[str]
    writers: list[str]
    producers: list[str]
    composers: list[str]
    actors: list[str]
    characters: list[str]
    production_companies: list[str]
    # Popularity
    imdb_rating: Optional[float] = None
    metacritic_rating: Optional[float] = None
    reception_summary: Optional[str] = None
    featured_reviews: list[IMDBFeaturedReview] = []
    review_themes: list[IMDBReviewTheme] = []
    # METADATA
    plot_events_metadata: Optional[PlotEventsMetadata] = None
    plot_analysis_metadata: Optional[PlotAnalysisMetadata] = None
    viewer_experience_metadata: Optional[ViewerExperienceMetadata] = None
    watch_context_metadata: Optional[WatchContextMetadata] = None
    narrative_techniques_metadata: Optional[NarrativeTechniquesMetadata] = None
    reception_metadata: Optional[ReceptionMetadata] = None
    production_metadata: Optional[ProductionMetadata] = None
    # ONLY FOR DEBUGGING
    debug_synopses: list[str] = []
    debug_plot_summaries: list[str] = []

    # Budget bucket thresholds by decade (era-aware budget classification)
    _DECADE_THRESHOLDS: dict[int, tuple[int, int]] = {
        1920: (100_000, 1_000_000),
        1930: (150_000, 2_000_000),
        1940: (250_000, 3_000_000),
        1950: (750_000, 10_000_000),
        1960: (1_000_000, 15_000_000),
        1970: (2_000_000, 20_000_000),
        1980: (7_000_000, 40_000_000),
        1990: (20_000_000, 100_000_000),
        2000: (25_000_000, 150_000_000),
        2010: (25_000_000, 200_000_000),
        2020: (25_000_000, 250_000_000),
    }

    # Maturity rating to semantic description mapping (no numbers)
    _MATURITY_DESCRIPTIONS: dict[str, str] = {
        "G": "General audiences. Family friendly, safe for kids, children, and all ages. Wholesome entertainment",
        "PG": "Parental guidance suggested. Good for families and most children, but may contain some mild material",
        "PG-13": "Parents strongly cautioned. Best for teens and young adults. May contain material inappropriate for young children",
        "R": "Restricted. Mature audiences only. Contains adult themes, violence, or strong language. Not for kids",
        "NC-17": "Adults only. Explicit content. Strictly for mature audiences. Not suitable for children or teens",
    }

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text by lowercasing and removing punctuation.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text string
        """
        # Lowercase
        normalized = text.lower()
        # Remove punctuation (keep alphanumeric and spaces)
        normalized = re.sub(r'[^\w\s]', '', normalized)
        # Collapse multiple spaces to single space
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()

    def title_string(self) -> str:
        """
        Returns formatted title string for vector search (4.1).
        
        Format: "Movie: <title> (<original_title>)" if original_title exists,
                otherwise "Movie: <title>"
        
        Returns:
            Formatted title string
        """
        if self.original_title:
            return f"Movie: {self.title} ({self.original_title})"
        return f"Movie: {self.title}"

    def release_ts(self) -> Optional[int]:
        """
        Returns the release timestamp of the movie.
        """
        if not self.release_date:
            return None
        parsed_release = datetime.strptime(self.release_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(parsed_release.timestamp())

    def normalized_title_tokens(self) -> list[str]:
        """Build normalized title tokens including hyphen expansions."""
        return tokenize_title_phrase(self.title)

    def maturity_rating_and_rank(self) -> tuple[str, int]:
        """
        Returns the normalized maturity rating label and its ordinal rank.

        Resolves the movie's raw maturity_rating string to a MaturityRating enum
        member (case-insensitive, whitespace-tolerant), then returns:
          - The normalized label (e.g. "pg-13", "nc-17", "unrated")
          - The ordinal rank integer (G=1, PG=2, PG-13=3, R=4, NC-17=5, Unrated=999)

        Unrecognized ratings fall back to UNRATED (rank 999).

        Returns:
            Tuple of (normalized_label, rank).
        """
        rating = MaturityRating.from_string(self.maturity_rating)
        rank = rating.value
        # str(rating) calls MaturityRating.__str__ which returns the human label
        return normalize_string(str(rating)), rank

    def genres_subset(self, limit: int = None) -> list[str]:
        """
        Returns genres as-is for vector search (4.4).
        
        Returns:
            List of genres unchanged
        """
        # Access attribute directly to avoid method/attribute name conflict
        if limit:
            return self.genres[:limit]
        return self.genres

    def genre_ids(self) -> list[int]:
        """
        Returns the list of genre IDs for the movie.
        """
        raw_genres = self.genres
        if not raw_genres:
            return []
        
        genre_ids: list[int] = []
        for genre_name in raw_genres:
            genre_enum = Genre.from_string(str(genre_name))
            if genre_enum is None:
                continue
            genre_ids.append(genre_enum.genre_id)
        return genre_ids

    def watch_offer_keys(self) -> list[int]:
        """
        Returns the list of watch offer keys for the movie.
        """
        raw_providers = self.watch_providers
        if not isinstance(raw_providers, list):
            raw_providers = []

        watch_offer_key_set: set[int] = set()
        for provider in raw_providers:
            provider_id = getattr(provider, "id", None)
            if provider_id is None or provider_id not in FILTERABLE_WATCH_PROVIDER_IDS:
                continue

            provider_name = str(getattr(provider, "name", "") or "")

            watch_method_types = getattr(provider, "types", [])
            if not isinstance(watch_method_types, list):
                watch_method_types = []

            for watch_method_type in watch_method_types:
                watch_method_id = int(watch_method_type)

                watch_offer_key = create_watch_provider_offering_key(provider_id, watch_method_id)
                watch_offer_key_set.add(watch_offer_key)

        return sorted(watch_offer_key_set)

    def release_decade_bucket(self) -> str:
        """
        Converts release_date into a semantic decade label for vector search (4.5).
        
        Examples: 
        - 1925 -> "Silent Era & Early Cinema 1920s"
        - 1942 -> "Golden Age of Hollywood 1940s"
        - 1985 -> "80s"
        - 2005 -> "2000s"
        
        Returns:
            Semantic decade bucket string or empty string if parsing fails
        """
        try:
            # Extract year from "YYYY-MM-DD" format
            year = int(self.release_date[:4])
            decade = (year // 10) * 10
            date_string = f"{decade}s"
            
            if year < 1930:
                date_string += ", silent era & early cinema"
            elif 1930 <= year < 1950:
                date_string += ", golden age of hollywood"
            else:
                # 1950s - 1990s: "50s", "60s", etc.
                date_string += f", {str(decade)[-2:]}s"

            if date_string:
                return f"Release date: {date_string}"
            else:
                raise ValueError(f"Unknown release date: {self.release_date}")
        except (ValueError, IndexError):
            print(f"Error occurred getting release decade bucket: {self.release_date}")
            return ""

    def duration_bucket(self) -> str:
        """
        Buckets duration into descriptive categories for vector search (4.6).
        
        Buckets (inclusive lower bound, exclusive upper bound):
        - < 102 minutes: "short, quick watch"
        - 102-118 minutes: "Standard length"
        - 118-144 minutes: "Long"
        - >= 144 minutes: "Very long"
        
        Returns:
            Duration bucket category string
        """
        if self.duration < 102:
            return "short, quick watch"
        elif 102 <= self.duration < 118:
            return "standard length"
        elif 118 <= self.duration < 144:
            return "long"
        else:  # >= 144
            return "very long"

    def budget_bucket_for_era(self) -> str:
        """
        Classifies budget as small/blockbuster/noteworthy based on era thresholds (4.7).
        
        Uses decade-specific thresholds to determine if budget is exceptionally
        small or large for the movie's release era.
        
        Returns:
            Budget classification string:
            - "small budget"
            - "big budget, blockbuster"
            - "" (budget unknown or not of note)
        """
        if self.budget is None:
            return ""
        
        try:
            # Extract year from release_date
            year = int(self.release_date[:4])
            # Determine decade for threshold lookup
            decade = (year // 10) * 10
            
            # Use 1920s thresholds for movies before 1920
            if decade < 1920:
                decade = 1920
            # Use 2020s thresholds for movies 2030 or later
            elif decade >= 2030:
                decade = 2020
            
            # Get thresholds for this decade
            min_threshold, max_threshold = self._DECADE_THRESHOLDS[decade]
            
            # Classify budget
            if self.budget < min_threshold:
                return "small budget"
            elif self.budget > max_threshold:
                return "big budget, blockbuster"
            else:
                return ""
        except (ValueError, IndexError, KeyError):
            print(f"Error occurred getting budget bucket: {self.budget}")
            return ""

    def maturity_guidance_text(self) -> str:
        """
        Formats maturity guidance text for vector search (4.9).
        
        For "Unrated" movies: joins parental guide items as "<severity> <category>".
        For rated movies: maps rating to semantic description and appends reasoning.
        
        Returns:
            Formatted maturity guidance string
        """
        if self.maturity_rating == "Unrated" or not self.maturity_reasoning:
            # Generate one string per parental guide item: "<severity> <category>"
            guide_strings = [
                f"{item.severity} {item.category}"
                for item in self.parental_guide_items
            ]
            joined_guide_strings = ", ".join(guide_strings)
            if self.maturity_rating == "Unrated":
                return joined_guide_strings
            else:
                return f"Rated {self.maturity_rating} for {joined_guide_strings}"
        else:
            # Map rating to semantic description
            base_description = self._MATURITY_DESCRIPTIONS.get(
                self.maturity_rating.upper(),
                f"Rated {self.maturity_rating}"
            )
            
            # Append maturity reasoning if available
            if self.maturity_reasoning:
                reasons = ". ".join(self.maturity_reasoning)
                return f"{base_description}. {reasons}"
            else:
                return base_description

    def production_text(self) -> str:
        """
        Formats production information as three concatenated sentences (4.10).
        
        Each sentence becomes empty string if its associated data is empty,
        preventing useless words from polluting the vector.
        
        Format:
        - "Produced in <countries> by <companies>."
        - "Filming happened in <locations>."
        
        Returns:
            Concatenated production text (may be empty if all data missing)
        """
        sentences = []
        
        # Sentence 1: Countries and production companies
        if self.countries_of_origin or self.production_companies:
            parts = []
            if self.countries_of_origin:
                countries_str = ", ".join(self.countries_of_origin)
                parts.append(f"Produced in {countries_str}")
            if self.production_companies:
                companies_str = ", ".join(self.production_companies)
                if parts:
                    parts.append(f"by {companies_str}")
                else:
                    parts.append(f"Produced by {companies_str}")
            if parts:
                sentences.append(" ".join(parts) + ".")
        
        # Sentence 2: Filming locations
        if self.filming_locations:
            locations_str = ", ".join(self.filming_locations)
            sentences.append(f"Filming happened in {locations_str}.")
        
        return " ".join(sentences)

    def languages_text(self) -> str:
        """
        Formats language information for vector search (4.11).
        
        Format: "Primary language: <languages[0]>. Audio also available for <languages[1:]>"
        Returns empty string if no languages available.
        
        Returns:
            Formatted languages string or empty string if no languages
        """
        if not self.languages:
            return ""
        
        primary = self.languages[0]
        additional = self.languages[1:]
        
        if additional:
            additional_str = ", ".join(additional)
            return f"Primary language: {primary}. Audio also available for {additional_str}"
        else:
            return f"Primary language: {primary}"

    def cast_text(self) -> str:
        """
        Formats cast and crew information (truncated) for vector search (4.12).
        
        Format: "Directed by <directors>. Written by <writers>. Produced by <producers[:4]>.
                 Music composed by <composers>. Main actors: <actors[:8]>"
        
        Returns:
            Formatted cast text string
        """
        parts = []
        
        if self.directors:
            directors_str = ", ".join(self.directors)
            parts.append(f"Directed by {directors_str}")
        
        if self.writers:
            writers_str = ", ".join(self.writers)
            parts.append(f"Written by {writers_str}")
        
        if self.producers:
            producers_str = ", ".join(self.producers[:4])
            parts.append(f"Produced by {producers_str}")
        
        if self.composers:
            composers_str = ", ".join(self.composers)
            parts.append(f"Music composed by {composers_str}")
        
        if self.actors:
            actors_str = ", ".join(self.actors[:5])
            parts.append(f"Main actors: {actors_str}")
        
        return ".\n".join(parts) + "." if parts else ""

    def characters_text(self) -> str:
        """
        Formats main characters (truncated) for vector search (4.13).
        
        Format: "Main characters: <characters[:8]>"
        
        Returns:
            Formatted characters string or empty string if no characters
        """
        if self.plot_events_metadata.major_characters:
            characters_str = ", ".join([character.name.lower() for character in self.plot_events_metadata.major_characters])
            return f"Main characters: {characters_str}"

        if not self.characters:
            return ""
        
        characters_str = ", ".join(self.characters[:5])
        return f"Main characters: {characters_str}"

    def reception_score(self) -> Optional[float]:
        """
        Computes numeric reception score for ranking (4.14).
        
        Formula: (0.4 * 10 * imdb_rating) + (0.6 * metacritic_rating)
        Note: imdb_rating is 0..10, metacritic_rating is 0..100
        
        Returns:
            Numeric reception score
        """
        if self.imdb_rating and self.metacritic_rating:
            return (0.4 * 10 * self.imdb_rating) + (0.6 * self.metacritic_rating)
        elif self.imdb_rating:
            return 0.4 * 10 * self.imdb_rating
        elif self.metacritic_rating:
            return 0.6 * self.metacritic_rating
        else:
            return None

    def reception_tier(self) -> str:
        """
        Maps reception score to tier label for vector search (4.14).
        
        Tier mappings:
        - >= 81: "Universally acclaimed"
        - >= 61: "Generally favorable reviews"
        - >= 41: "Mixed or average reviews"
        - >= 21: "Generally unfavorable reviews"
        - < 21: "Overwhelming dislike"
        
        Returns:
            Reception tier label string
        """
        score = self.reception_score()
        if score is None:
            return None
        if score >= 81:
            return "Universally acclaimed"
        elif score >= 61:
            return "Generally favorable reviews"
        elif score >= 41:
            return "Mixed or average reviews"
        elif score >= 21:
            return "Generally unfavorable reviews"
        else:
            return "Overwhelming dislike"

    def reception_summary_text(self) -> str:
        """
        Formats reception summary for vector search (4.15).
        
        Format: "Review summary: <reception_summary>"
        Returns empty string if reception_summary is None.
        
        Returns:
            Formatted reception summary string or empty string
        """
        if self.reception_summary:
            return f"Review summary: {self.reception_summary}"
        return ""

    def watch_providers_text(self) -> str:
        """
        Formats watch providers for vector search (4.16).
        
        Format: "Available on <providers>"
        Returns empty string if no watch providers available.
        
        Returns:
            Formatted watch providers string or empty string
        """
        if not self.watch_providers:
            return ""
        
        providers_str = ", ".join([provider.name for provider in self.watch_providers])
        return f"Watch on {providers_str}"

