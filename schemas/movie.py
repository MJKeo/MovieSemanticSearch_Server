"""
Tracker-backed movie schema for ingestion-time data access.

`Movie` is a pure validated data object backed by rows from the
ingestion tracker SQLite database. Use `Movie.from_tmdb_id()` to load:

- non-null `tmdb_data`
- non-null `imdb_data`
- optional parsed metadata objects for all generated metadata columns
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from datetime import datetime, timezone

from implementation.classes.countries import country_from_string
from implementation.classes.enums import BudgetSize, Genre, MaturityRating
from implementation.classes.languages import LANGUAGE_BY_NORMALIZED_NAME
from implementation.misc.helpers import normalize_string, tokenize_title_phrase

from movie_ingestion.imdb_scraping.models import (
    AwardNomination,
    FeaturedReview,
    ParentalGuideItem,
    ReviewTheme,
)
from movie_ingestion.scoring_utils import unpack_provider_keys
from movie_ingestion.tracker import IMDB_DATA_COLUMNS, IMDB_JSON_COLUMNS
from schemas.metadata import (
    NarrativeTechniquesOutput,
    PlotAnalysisOutput,
    PlotEventsOutput,
    ProductionKeywordsOutput,
    ReceptionOutput,
    SourceMaterialV2Output,
    SourceOfInspirationOutput,
    ViewerExperienceOutput,
    WatchContextOutput,
)


_DEFAULT_TRACKER_DB = (
    Path(__file__).resolve().parent.parent / "ingestion_data" / "tracker.db"
)

_TMDB_DATA_COLUMNS: tuple[str, ...] = (
    "tmdb_id",
    "imdb_id",
    "title",
    "release_date",
    "duration",
    "poster_url",
    "watch_provider_keys",
    "vote_count",
    "popularity",
    "vote_average",
    "overview_length",
    "genre_count",
    "has_revenue",
    "has_budget",
    "has_production_companies",
    "has_production_countries",
    "has_keywords",
    "has_cast_and_crew",
    "budget",
    "maturity_rating",
    "reviews",
    "revenue",
)

_METADATA_COLUMNS: tuple[str, ...] = (
    "plot_events",
    "reception",
    "plot_analysis",
    "viewer_experience",
    "watch_context",
    "narrative_techniques",
    "production_keywords",
    "source_of_inspiration",
    "source_material_v2",
)

_METADATA_FIELD_TO_MODEL: dict[str, type[BaseModel]] = {
    "plot_events_metadata": PlotEventsOutput,
    "reception_metadata": ReceptionOutput,
    "plot_analysis_metadata": PlotAnalysisOutput,
    "viewer_experience_metadata": ViewerExperienceOutput,
    "watch_context_metadata": WatchContextOutput,
    "narrative_techniques_metadata": NarrativeTechniquesOutput,
    "production_keywords_metadata": ProductionKeywordsOutput,
    "source_of_inspiration_metadata": SourceOfInspirationOutput,
    "source_material_v2_metadata": SourceMaterialV2Output,
}

_METADATA_FIELD_TO_COLUMN: dict[str, str] = {
    field_name: field_name.removesuffix("_metadata")
    for field_name in _METADATA_FIELD_TO_MODEL
}


class TMDBData(BaseModel):
    """Typed view of one `tmdb_data` tracker row."""

    model_config = ConfigDict(extra="forbid")

    tmdb_id: int
    imdb_id: str | None = None
    title: str | None = None
    release_date: str | None = None
    duration: int | None = None
    poster_url: str | None = None
    watch_provider_keys: list[int] = Field(default_factory=list)
    vote_count: int | None = None
    popularity: float | None = None
    vote_average: float | None = None
    overview_length: int | None = None
    genre_count: int | None = None
    has_revenue: int | None = None
    has_budget: int | None = None
    has_production_companies: int | None = None
    has_production_countries: int | None = None
    has_keywords: int | None = None
    has_cast_and_crew: int | None = None
    budget: int | None = None
    maturity_rating: str | None = None
    reviews: list[str] = Field(default_factory=list)
    revenue: int | None = None


class IMDBData(BaseModel):
    """Typed view of one `imdb_data` tracker row."""

    model_config = ConfigDict(extra="forbid")

    tmdb_id: int
    original_title: str | None = None
    maturity_rating: str | None = None
    overview: str | None = None
    imdb_rating: float | None = None
    imdb_vote_count: int | None = None
    metacritic_rating: float | None = None
    reception_summary: str | None = None
    budget: int | None = None
    overall_keywords: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    countries_of_origin: list[str] = Field(default_factory=list)
    production_companies: list[str] = Field(default_factory=list)
    filming_locations: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    synopses: list[str] = Field(default_factory=list)
    plot_summaries: list[str] = Field(default_factory=list)
    plot_keywords: list[str] = Field(default_factory=list)
    maturity_reasoning: list[str] = Field(default_factory=list)
    directors: list[str] = Field(default_factory=list)
    writers: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    producers: list[str] = Field(default_factory=list)
    composers: list[str] = Field(default_factory=list)
    review_themes: list[ReviewTheme] = Field(default_factory=list)
    parental_guide_items: list[ParentalGuideItem] = Field(default_factory=list)
    featured_reviews: list[FeaturedReview] = Field(default_factory=list)
    awards: list[AwardNomination] = Field(default_factory=list)
    box_office_worldwide: int | None = None
    imdb_title_type: str | None = None


class Movie(BaseModel):
    """Full tracker-backed movie object with source rows and metadata."""

    model_config = ConfigDict(extra="forbid")

    tmdb_data: TMDBData
    imdb_data: IMDBData
    plot_events_metadata: PlotEventsOutput | None = None
    reception_metadata: ReceptionOutput | None = None
    plot_analysis_metadata: PlotAnalysisOutput | None = None
    viewer_experience_metadata: ViewerExperienceOutput | None = None
    watch_context_metadata: WatchContextOutput | None = None
    narrative_techniques_metadata: NarrativeTechniquesOutput | None = None
    production_keywords_metadata: ProductionKeywordsOutput | None = None
    source_of_inspiration_metadata: SourceOfInspirationOutput | None = None
    source_material_v2_metadata: SourceMaterialV2Output | None = None

    # Era-adjusted budget thresholds: decade → (small_ceiling, large_floor)
    _DECADE_THRESHOLDS: ClassVar[dict[int, tuple[int, int]]] = {
        1920: (    100_000,   1_000_000),
        1930: (    150_000,   2_000_000),
        1940: (    250_000,   3_000_000),
        1950: (    750_000,  12_000_000),
        1960: (  1_000_000,  20_000_000),
        1970: (  2_000_000,  25_000_000),
        1980: (  5_000_000,  45_000_000),
        1990: (  9_000_000,  80_000_000),
        2000: ( 12_000_000, 110_000_000),
        2010: ( 15_000_000, 150_000_000),
        2020: ( 18_000_000, 185_000_000),
    }

    # Shared SELECT + FROM + JOIN clause used by both the single-movie
    # and batch loaders. Only the WHERE clause differs between them.
    _SELECT_FROM: ClassVar[str] = f"""
        SELECT
            {", ".join(f"t.{col} AS tmdb__{col}" for col in _TMDB_DATA_COLUMNS)},
            {", ".join(
                f"i.{col} AS imdb__{col}"
                for col in ("tmdb_id", *IMDB_DATA_COLUMNS)
            )},
            {", ".join(f"g.{col} AS meta__{col}" for col in _METADATA_COLUMNS)}
        FROM tmdb_data t
        JOIN imdb_data i ON i.tmdb_id = t.tmdb_id
        LEFT JOIN generated_metadata g ON g.tmdb_id = t.tmdb_id
    """

    _QUERY: ClassVar[str] = f"{_SELECT_FROM} WHERE t.tmdb_id = ?"

    # Batch query uses json_each() to expand a JSON array parameter into
    # a virtual table. This avoids building placeholder strings and also
    # sidesteps SQLite's bound-parameter count limit.
    _BATCH_QUERY: ClassVar[str] = (
        f"{_SELECT_FROM} WHERE t.tmdb_id IN (SELECT value FROM json_each(?))"
    )

    @classmethod
    def from_tmdb_id(
        cls,
        tmdb_id: int,
        tracker_db_path: Path = _DEFAULT_TRACKER_DB,
    ) -> Movie:
        """Load and parse one movie from the ingestion tracker DB."""
        if not tracker_db_path.exists():
            raise FileNotFoundError(
                f"Tracker DB not found at {tracker_db_path}."
            )

        with sqlite3.connect(str(tracker_db_path)) as tracker:
            tracker.row_factory = sqlite3.Row
            row = tracker.execute(cls._QUERY, (tmdb_id,)).fetchone()

        if row is None:
            raise LookupError(
                f"TMDB movie {tmdb_id} was not found in both tmdb_data and imdb_data."
            )

        row_dict = dict(row)
        return cls(
            tmdb_data=_build_tmdb_data(row_dict),
            imdb_data=_build_imdb_data(row_dict),
            **_build_metadata_fields(row_dict),
        )

    @classmethod
    def from_tmdb_ids(
        cls,
        tmdb_ids: list[int],
        tracker_db_path: Path = _DEFAULT_TRACKER_DB,
    ) -> dict[int, "Movie"]:
        """Batch-load multiple movies from the ingestion tracker DB.

        Executes a single SQLite query for all requested IDs. Movies
        that are missing from the joined tmdb_data/imdb_data tables or
        that fail to parse are silently skipped (logged to stdout).

        Args:
            tmdb_ids: TMDB movie IDs to load.
            tracker_db_path: Path to the ingestion tracker SQLite DB.

        Returns:
            Mapping of tmdb_id → Movie for every successfully loaded movie.
        """
        if not tmdb_ids:
            return {}

        if not tracker_db_path.exists():
            raise FileNotFoundError(
                f"Tracker DB not found at {tracker_db_path}."
            )

        # Pass the ID list as a JSON array string; json_each() expands it
        # in-SQL so we bind exactly one parameter regardless of list size.
        id_list_json = json.dumps(tmdb_ids)

        with sqlite3.connect(str(tracker_db_path)) as tracker:
            tracker.row_factory = sqlite3.Row
            rows = tracker.execute(cls._BATCH_QUERY, (id_list_json,)).fetchall()

        result: dict[int, Movie] = {}
        for row in rows:
            row_dict = dict(row)
            try:
                movie = cls(
                    tmdb_data=_build_tmdb_data(row_dict),
                    imdb_data=_build_imdb_data(row_dict),
                    **_build_metadata_fields(row_dict),
                )
                result[movie.tmdb_data.tmdb_id] = movie
            except Exception as exc:
                # Extract tmdb_id from raw row for logging even when parsing fails
                raw_id = row_dict.get("tmdb__tmdb_id", "unknown")
                print(f"Failed to parse movie {raw_id}: {exc}")

        return result

    # Maturity rating to semantic description mapping (no numbers).
    # Used by maturity_text_short() when maturity_reasoning is unavailable.
    _MATURITY_DESCRIPTIONS: ClassVar[dict[str, str]] = {
        "G": "general audiences, family friendly, safe for kids and all ages",
        "PG": "parental guidance suggested, good for families and most children",
        "PG-13": "parents strongly cautioned, best for teens and young adults",
        "R": "restricted, mature audiences only, contains adult themes or violence or strong language",
        "NC-17": "adults only, explicit content, strictly for mature audiences",
    }

    def title_with_original(self) -> str:
        """Format title for vector text, including original title when different."""
        title = self.tmdb_data.title or ""
        original = self.imdb_data.original_title
        if original and original != title:
            return f"{title} ({original})"
        return title

    def maturity_text_short(self) -> str:
        """Compact maturity signal for vector embedding.

        Prefers IMDB maturity_reasoning (joined prose) when available,
        otherwise maps the resolved MPA rating to a semantic description.
        Returns empty string when no maturity info exists at all.
        """
        # Prose reasoning is the richest signal when available
        if self.imdb_data.maturity_reasoning:
            return ". ".join(self.imdb_data.maturity_reasoning)

        # Fall back to MPA rating → semantic description
        rating = self.resolved_maturity_rating()
        if not rating:
            return ""
        description = self._MATURITY_DESCRIPTIONS.get(rating.upper())
        if description:
            return description

        # Non-standard values like "Not Rated" / "Unrated" carry no useful
        # maturity signal — return empty so anchor text stays clean.
        return ""

    def resolved_budget(self) -> int | None:
        """Prefer IMDB budget when present; fall back to TMDB.

        Returns None when budget is missing or zero (zero typically
        means 'unknown' in TMDB/IMDB data, not actually free).
        """
        if self.imdb_data.budget:
            return self.imdb_data.budget
        if self.tmdb_data.budget:
            return self.tmdb_data.budget
        return None

    def resolved_box_office_revenue(self) -> int | None:
        """Prefer IMDB worldwide box office when positive; fall back to TMDB revenue.

        Returns None when neither source has a valid (positive) value.
        Zero and negative values are treated as missing data.
        """
        imdb_box_office = self.imdb_data.box_office_worldwide
        if imdb_box_office and imdb_box_office > 0:
            return imdb_box_office
        tmdb_revenue = self.tmdb_data.revenue
        if tmdb_revenue and tmdb_revenue > 0:
            return tmdb_revenue
        return None

    def resolved_maturity_rating(self) -> str | None:
        """Prefer IMDB maturity rating when present; fall back to TMDB."""
        if self.imdb_data.maturity_rating:
            return self.imdb_data.maturity_rating
        return self.tmdb_data.maturity_rating

    def reception_score(self) -> float | None:
        """Compute a 0-100 reception score from IMDB and Metacritic ratings.

        When both ratings are available, returns a weighted blend
        (40% IMDB scaled to 100, 60% Metacritic). Falls back to
        whichever single source is present, or None if neither exists.
        """
        imdb = self.imdb_data.imdb_rating
        meta = self.imdb_data.metacritic_rating
        if imdb and meta:
            return (0.4 * 10 * imdb) + (0.6 * meta)
        elif imdb:
            return 10 * imdb
        elif meta:
            return meta
        return None

    def release_ts(self) -> int | None:
        """Convert the TMDB release_date string to a UTC Unix timestamp."""
        release_date = self.tmdb_data.release_date
        if not release_date:
            return None
        parsed = datetime.strptime(release_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        return int(parsed.timestamp())

    def maturity_rating_and_rank(self) -> tuple[str, int]:
        """Resolve the maturity rating to a normalized label and ordinal rank.

        Uses the IMDB-preferred maturity rating (via resolved_maturity_rating),
        falling back to UNRATED for missing or unrecognized values.
        """
        rating = MaturityRating.from_string_with_default(self.resolved_maturity_rating())
        return rating.value, rating.maturity_rank

    def normalized_title_tokens(self) -> list[str]:
        """Build normalized title tokens including hyphen expansions.

        Merges tokens from the primary title and original title (if different),
        deduplicating while preserving first-seen order from the primary title.
        """
        tokens = tokenize_title_phrase(self.tmdb_data.title or "")
        original = self.imdb_data.original_title
        if original and original != (self.tmdb_data.title or ""):
            seen = set(tokens)
            for token in tokenize_title_phrase(original):
                if token not in seen:
                    seen.add(token)
                    tokens.append(token)
        return tokens

    def genre_ids(self) -> list[int]:
        """Map IMDB genre strings to their integer genre IDs."""
        genre_ids: list[int] = []
        for genre_name in self.imdb_data.genres:
            genre_enum = Genre.from_string(str(genre_name))
            if genre_enum is not None:
                genre_ids.append(genre_enum.genre_id)
        return genre_ids

    def watch_offer_keys(self) -> list[int]:
        """Return pre-decoded watch provider offering keys from TMDB data."""
        return self.tmdb_data.watch_provider_keys

    def audio_language_ids(self) -> list[int]:
        """Map IMDB language strings to their integer language IDs."""
        language_ids: list[int] = []
        for language in self.imdb_data.languages:
            normalized = normalize_string(str(language))
            if not normalized:
                continue
            language_enum = LANGUAGE_BY_NORMALIZED_NAME.get(normalized)
            if language_enum is not None:
                language_ids.append(language_enum.language_id)
        return language_ids

    def country_ids(self) -> list[int]:
        """Map IMDB country-of-origin strings to their integer country IDs."""
        country_ids: list[int] = []
        seen_ids: set[int] = set()
        for country_name in self.imdb_data.countries_of_origin:
            country = country_from_string(str(country_name))
            if country is None or country.country_id in seen_ids:
                continue
            seen_ids.add(country.country_id)
            country_ids.append(country.country_id)
        return country_ids

    def production_text(self, include_filming_locations: bool = True) -> str:
        """Format production info as labeled lines for vector embedding.

        Uses concise labels so the embedding captures geographic and
        company signals without filler words diluting the vector.

        Args:
            include_filming_locations: When False, filming locations are
                omitted entirely (e.g. for animation where locations are
                irrelevant to the production).
        """
        lines: list[str] = []

        if self.imdb_data.countries_of_origin:
            countries = ", ".join(self.imdb_data.countries_of_origin)
            lines.append(f"countries of origin: {countries}")

        if self.imdb_data.production_companies:
            companies = ", ".join(self.imdb_data.production_companies)
            lines.append(f"production companies: {companies}")

        # Limit to first 3 locations to keep vector text focused
        if include_filming_locations and self.imdb_data.filming_locations:
            locations = ", ".join(self.imdb_data.filming_locations[:3])
            lines.append(f"filming locations: {locations}")

        return "\n".join(lines)

    def languages_text(self) -> str:
        """Format language info as labeled lines for vector embedding."""
        if not self.imdb_data.languages:
            return ""

        lines: list[str] = [f"primary language: {self.imdb_data.languages[0]}"]

        additional = self.imdb_data.languages[1:]
        if additional:
            lines.append(f"additional languages: {', '.join(additional)}")

        return "\n".join(lines)

    def release_decade_bucket(self) -> str:
        """Convert release_date into a semantic decade label for vector search.

        Examples:
            1925 → "Release date: 1920s, silent era & early cinema"
            1942 → "Release date: 1940s, golden age of hollywood"
            1985 → "Release date: 1980s, 80s"
        """
        release_date = self.tmdb_data.release_date
        if not release_date:
            return ""

        try:
            year = int(release_date[:4])
            decade = (year // 10) * 10
            date_string = f"{decade}s"

            if year < 1930:
                date_string += ", silent era & early cinema"
            elif 1930 <= year < 1950:
                date_string += ", golden age of hollywood"
            else:
                date_string += f", {str(decade)[-2:]}s"

            if date_string:
                return f"Release date: {date_string}"
            else:
                raise ValueError(f"Unknown release date: {release_date}")
        except (ValueError, IndexError):
            print(f"Error occurred getting release decade bucket: {release_date}")
            return ""

    def budget_bucket_for_era(self) -> BudgetSize | None:
        """Classify budget as small/large relative to era thresholds.

        Linearly interpolates between adjacent decade anchor points
        so that boundary years don't experience threshold cliffs.

        Returns:
            BudgetSize.SMALL when notably low for the era,
            BudgetSize.LARGE when notably high for the era,
            or None when budget is unknown, typical, or unparseable.
        """
        budget = self.resolved_budget()
        release_date = self.tmdb_data.release_date
        if budget is None or not release_date:
            return None

        try:
            year = int(release_date[:4])
            clamped_year = max(1920, min(year, 2029))
            small_threshold, large_threshold = self._interpolated_thresholds(clamped_year)

            if budget < small_threshold:
                return BudgetSize.SMALL
            elif budget > large_threshold:
                return BudgetSize.LARGE
            else:
                return None
        except (ValueError, IndexError, KeyError):
            print(f"Error occurred getting budget bucket: {budget}")
            return None

    def _interpolated_thresholds(self, year: int) -> tuple[float, float]:
        """Linearly interpolate small/large thresholds for an exact year.

        Finds the two bounding decade anchors and blends their
        thresholds proportionally so mid-decade years get smooth values.
        """
        thresholds = self._DECADE_THRESHOLDS
        decades = sorted(thresholds.keys())

        if year <= decades[0]:
            return thresholds[decades[0]]
        if year >= decades[-1]:
            return thresholds[decades[-1]]

        # Find the two bounding decades
        lo_decade = decades[0]
        for d in decades:
            if d <= year:
                lo_decade = d
            else:
                break
        hi_decade = lo_decade + 10

        lo_small, lo_large = thresholds[lo_decade]
        hi_small, hi_large = thresholds[hi_decade]

        # Fractional position within the decade (0.0 at lo, 1.0 at hi)
        t = (year - lo_decade) / 10.0

        small_threshold = lo_small + t * (hi_small - lo_small)
        large_threshold = lo_large + t * (hi_large - lo_large)

        return small_threshold, large_threshold

    def deduplicated_genres(self) -> list[str]:
        """Merge LLM genre_signatures with IMDB genres, exact-match deduped.

        Combines both sources into a set (lowercased) so exact duplicates
        are removed but near-overlaps like "thriller" and "psychological
        thriller" are both kept — minor duplication is fine for embedding.
        """
        combined: set[str] = set()

        if self.plot_analysis_metadata:
            for sig in self.plot_analysis_metadata.genre_signatures:
                combined.add(sig.lower())

        for genre in self.imdb_data.genres:
            combined.add(genre.lower())

        return sorted(combined)

    def is_animation(self) -> bool:
        """True when the movie has the Animation genre."""
        return any(g.lower() == "animation" for g in self.imdb_data.genres)

    def reception_tier(self) -> str | None:
        """Map reception score to a human-readable tier label.

        Tier thresholds (Metacritic-style boundaries):
            >= 81 → "Universally acclaimed"
            >= 61 → "Generally favorable reviews"
            >= 41 → "Mixed or average reviews"
            >= 21 → "Generally unfavorable reviews"
             < 21 → "Overwhelming dislike"
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


def _build_tmdb_data(row: dict[str, object]) -> TMDBData:
    """Parse TMDB row data into the typed tracker model."""
    data: dict[str, object] = {}

    for column in _TMDB_DATA_COLUMNS:
        value = row[f"tmdb__{column}"]
        if column == "watch_provider_keys":
            data[column] = _parse_watch_provider_keys(value)
            continue
        if column == "reviews":
            data[column] = _parse_json_list(value, "tmdb__reviews")
            continue
        data[column] = value

    return TMDBData.model_validate(data)


def _build_imdb_data(row: dict[str, object]) -> IMDBData:
    """Parse IMDB row data into the typed tracker model."""
    data: dict[str, object] = {}

    for column in ("tmdb_id", *IMDB_DATA_COLUMNS):
        value = row[f"imdb__{column}"]
        if column not in IMDB_JSON_COLUMNS:
            data[column] = value
            continue

        parsed = _parse_json_list(value, f"imdb__{column}")
        if column == "review_themes":
            data[column] = [ReviewTheme.model_validate(item) for item in parsed]
        elif column == "parental_guide_items":
            data[column] = [ParentalGuideItem.model_validate(item) for item in parsed]
        elif column == "featured_reviews":
            data[column] = [FeaturedReview.model_validate(item) for item in parsed]
        elif column == "awards":
            data[column] = [AwardNomination.model_validate(item) for item in parsed]
        else:
            data[column] = parsed

    return IMDBData.model_validate(data)


def _build_metadata_fields(row: dict[str, object]) -> dict[str, BaseModel | None]:
    """Parse all metadata JSON columns into their schema objects."""
    parsed: dict[str, BaseModel | None] = {}

    for field_name, schema_class in _METADATA_FIELD_TO_MODEL.items():
        column = _METADATA_FIELD_TO_COLUMN[field_name]
        raw_value = row[f"meta__{column}"]
        parsed[field_name] = _parse_metadata_json(
            raw_value,
            schema_class,
            f"meta__{column}",
        )

    return parsed


def _parse_watch_provider_keys(raw: object) -> list[int]:
    """Decode the TMDB watch-provider BLOB into integer offering keys."""
    if raw is None:
        return []

    if not isinstance(raw, bytes):
        raise ValueError(
            f"Expected tmdb__watch_provider_keys to be bytes or None, got {type(raw).__name__}."
        )

    try:
        return unpack_provider_keys(raw)
    except Exception as exc:
        raise ValueError("Malformed tmdb__watch_provider_keys BLOB.") from exc


def _parse_json_list(raw: object, label: str) -> list:
    """Parse a JSON-encoded SQLite TEXT column expected to contain a list."""
    if raw is None:
        return []

    if not isinstance(raw, str):
        raise ValueError(
            f"Expected {label} to be a JSON string or NULL, got {type(raw).__name__}."
        )

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {label}.") from exc

    if not isinstance(parsed, list):
        raise ValueError(f"Expected {label} to decode to a list.")

    return parsed


def _parse_metadata_json(
    raw: object,
    schema_class: type[BaseModel],
    label: str,
) -> BaseModel | None:
    """Parse one metadata JSON column into its Pydantic schema object."""
    if raw is None:
        return None

    if not isinstance(raw, str):
        raise ValueError(
            f"Expected {label} to be a JSON string or NULL, got {type(raw).__name__}."
        )

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed metadata JSON in {label}.") from exc

    try:
        normalized = _normalize_metadata_payload(decoded, schema_class)
        return schema_class.model_validate(normalized)
    except Exception as exc:
        raise ValueError(f"Malformed metadata JSON in {label}.") from exc


def _normalize_metadata_payload(
    payload: object,
    schema_class: type[BaseModel],
) -> object:
    """Normalize known legacy metadata shapes into the current schema."""
    if not isinstance(payload, dict):
        return payload

    if schema_class in {NarrativeTechniquesOutput, WatchContextOutput}:
        normalized_payload = dict(payload)
        for key, value in normalized_payload.items():
            if not isinstance(value, dict):
                continue
            if "justification" in value and "evidence_basis" not in value:
                section = dict(value)
                section["evidence_basis"] = section.pop("justification")
                normalized_payload[key] = section
        payload = normalized_payload

    if schema_class is SourceOfInspirationOutput:
        normalized_payload = dict(payload)
        normalized_payload.pop("source_evidence", None)
        normalized_payload.pop("lineage_evidence", None)
        payload = normalized_payload

    return payload


__all__ = ["IMDBData", "Movie", "TMDBData"]
