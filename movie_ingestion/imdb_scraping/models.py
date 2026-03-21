"""
Pydantic models for Stage 4 IMDB scraping.

Per-page models represent the structured output of each HTML parser.
IMDBScrapedMovie is the merged output saved to disk as the stage's
final artifact at ./ingestion_data/imdb/{tmdb_id}.json.

These models are structurally compatible with, but independent from,
the downstream models in implementation/classes/schemas.py. This
avoids coupling the scraper package to the implementation layer.
"""

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models used across multiple page-level models
# ---------------------------------------------------------------------------


class ReviewTheme(BaseModel):
    """A single review theme with name and sentiment direction."""
    name: str
    sentiment: str


class ParentalGuideItem(BaseModel):
    """A single content advisory entry with category and severity level."""
    category: str
    severity: str


class FeaturedReview(BaseModel):
    """A single featured IMDB user review with title and body text."""
    summary: str
    text: str


# ---------------------------------------------------------------------------
# Per-page parser output models
# ---------------------------------------------------------------------------


class MainPageData(BaseModel):
    """Parsed output from the IMDB main page (/title/{imdb_id}/)."""
    original_title: Optional[str] = None
    maturity_rating: Optional[str] = None
    overview: Optional[str] = None
    overall_keywords: list[str] = Field(default_factory=list)
    imdb_rating: Optional[float] = None
    imdb_vote_count: int = 0
    metacritic_rating: Optional[float] = None
    reception_summary: Optional[str] = None
    genres: list[str] = Field(default_factory=list)
    countries_of_origin: list[str] = Field(default_factory=list)
    production_companies: list[str] = Field(default_factory=list)
    filming_locations: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    budget: Optional[int] = None
    review_themes: list[ReviewTheme] = Field(default_factory=list)


class SummaryPageData(BaseModel):
    """Parsed output from the plot summary page (/title/{imdb_id}/plotsummary/)."""
    synopses: list[str] = Field(default_factory=list)
    plot_summaries: list[str] = Field(default_factory=list)


class KeywordsPageData(BaseModel):
    """Parsed output from the keywords page (/title/{imdb_id}/keywords/)."""
    plot_keywords: list[str] = Field(default_factory=list)


class ParentalGuidePageData(BaseModel):
    """Parsed output from the parental guide page (/title/{imdb_id}/parentalguide/)."""
    maturity_reasoning: list[str] = Field(default_factory=list)
    parental_guide_items: list[ParentalGuideItem] = Field(default_factory=list)


class CreditsPageData(BaseModel):
    """Parsed output from the full credits page (/title/{imdb_id}/fullcredits/)."""
    directors: list[str] = Field(default_factory=list)
    writers: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    producers: list[str] = Field(default_factory=list)
    composers: list[str] = Field(default_factory=list)


class ReviewsPageData(BaseModel):
    """Parsed output from the reviews page (/title/{imdb_id}/reviews/)."""
    featured_reviews: list[FeaturedReview] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Merged output model — the final artifact written to disk per movie
# ---------------------------------------------------------------------------


class IMDBScrapedMovie(BaseModel):
    """
    Merged output from all 6 IMDB pages for a single movie.

    This is the final artifact saved to ./ingestion_data/imdb/{tmdb_id}.json.
    Field names use the final downstream names (matching BaseMovie) from the
    start, so no rename step is needed when constructing BaseMovie later.
    """
    # IMDB title type (e.g. "movie", "tvSeries", "videoGame")
    imdb_title_type: Optional[str] = None

    # From main page
    original_title: Optional[str] = None
    maturity_rating: Optional[str] = None
    overview: Optional[str] = None
    overall_keywords: list[str] = Field(default_factory=list)
    imdb_rating: Optional[float] = None
    imdb_vote_count: int = 0
    metacritic_rating: Optional[float] = None
    reception_summary: Optional[str] = None
    genres: list[str] = Field(default_factory=list)
    countries_of_origin: list[str] = Field(default_factory=list)
    production_companies: list[str] = Field(default_factory=list)
    filming_locations: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    budget: Optional[int] = None
    review_themes: list[ReviewTheme] = Field(default_factory=list)

    # From summary page
    synopses: list[str] = Field(default_factory=list)
    plot_summaries: list[str] = Field(default_factory=list)

    # From keywords page
    plot_keywords: list[str] = Field(default_factory=list)

    # From parental guide page
    maturity_reasoning: list[str] = Field(default_factory=list)
    parental_guide_items: list[ParentalGuideItem] = Field(default_factory=list)

    # From credits page
    directors: list[str] = Field(default_factory=list)
    writers: list[str] = Field(default_factory=list)
    actors: list[str] = Field(default_factory=list)
    characters: list[str] = Field(default_factory=list)
    producers: list[str] = Field(default_factory=list)
    composers: list[str] = Field(default_factory=list)

    # From reviews page
    featured_reviews: list[FeaturedReview] = Field(default_factory=list)
