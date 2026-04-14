"""
Shared Pydantic sub-models for IMDB-sourced data.

These models originated in the IMDB scraping pipeline but are referenced
by db/, schemas/movie.py, and other modules outside movie_ingestion/.
Keeping them in the shared schemas layer avoids coupling those consumers
to the ingestion package.
"""

from typing import Optional

from pydantic import BaseModel

from schemas.enums import AwardOutcome, CEREMONY_BY_EVENT_TEXT


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


class AwardNomination(BaseModel):
    """A single award nomination with ceremony, award name, category, outcome, and year."""
    ceremony: str
    award_name: str                  # specific prize name (e.g., "Oscar", "Palme d'Or", "Golden Lion")
    category: Optional[str] = None   # null for festival grand prizes (Palme d'Or, etc.)
    outcome: AwardOutcome
    year: int

    @property
    def ceremony_id(self) -> int | None:
        """Stable integer ID for this award's ceremony, for Postgres storage.

        Returns None when the ceremony string doesn't match any known
        AwardCeremony member — callers must handle the None case
        (e.g. skip the award row during ingestion).
        """
        ceremony = CEREMONY_BY_EVENT_TEXT.get(self.ceremony)
        return ceremony.ceremony_id if ceremony is not None else None

    def did_win(self) -> bool:
        return self.outcome == AwardOutcome.WINNER
