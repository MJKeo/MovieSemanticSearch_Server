"""
Tracker-backed movie schema for ingestion-time data access.

`Movie` is a pure validated data object backed by rows from the
ingestion tracker SQLite database. Use `Movie.from_tmdb_id()` to load:

- non-null `tmdb_data`
- non-null `imdb_data`
- optional parsed metadata objects for all 8 generated metadata columns
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field

from movie_ingestion.imdb_scraping.models import (
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

    _QUERY: ClassVar[str] = f"""
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
        WHERE t.tmdb_id = ?
    """

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

    def resolved_budget(self) -> int | None:
        """Prefer IMDB budget when present; fall back to TMDB."""
        if self.imdb_data.budget is not None:
            return self.imdb_data.budget
        return self.tmdb_data.budget

    def resolved_maturity_rating(self) -> str | None:
        """Prefer IMDB maturity rating when present; fall back to TMDB."""
        if self.imdb_data.maturity_rating:
            return self.imdb_data.maturity_rating
        return self.tmdb_data.maturity_rating


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
