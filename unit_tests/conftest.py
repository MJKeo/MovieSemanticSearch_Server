"""Shared pytest fixtures for unit tests."""

from typing import Any, Callable
from pathlib import Path
import sys

import pytest

# Ensure project root is importable when tests run from repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from implementation.classes.movie import BaseMovie
from implementation.classes.schemas import ParentalGuideItem, WatchProvider


def _default_watch_provider() -> WatchProvider:
    """Build a default watch provider object used by movie fixtures."""
    return WatchProvider(
        id=101,
        name="Netflix",
        logo_path="/logo.png",
        display_priority=1,
        types=["subscription"],
    )


@pytest.fixture
def base_movie_factory() -> Callable[..., BaseMovie]:
    """Return a factory that builds a valid BaseMovie with optional overrides."""

    def _factory(**overrides: Any) -> BaseMovie:
        """Construct a complete BaseMovie while allowing targeted field overrides."""
        base_data: dict[str, Any] = {
            "id": "tt0000001",
            "tmdb_id": 1,
            "title": "Spider-Man",
            "original_title": None,
            "overall_keywords": ["hero", "city"],
            "release_date": "2002-05-03",
            "duration": 121,
            "genres": ["Action", "Adventure"],
            "countries_of_origin": ["USA"],
            "languages": ["English", "Spanish"],
            "filming_locations": ["New York"],
            "budget": 139_000_000,
            "watch_providers": [_default_watch_provider()],
            "poster_url": "https://image.test/poster.jpg",
            "maturity_rating": "PG-13",
            "maturity_reasoning": ["Intense action violence", "Brief language"],
            "parental_guide_items": [
                ParentalGuideItem(category="violence", severity="moderate"),
                ParentalGuideItem(category="language", severity="mild"),
            ],
            "overview": "A student gets spider-like abilities.",
            "plot_keywords": ["superhero", "identity"],
            "directors": ["Sam Raimi"],
            "writers": ["David Koepp"],
            "producers": ["Laura Ziskin", "Ian Bryce", "Avi Arad", "Grant Curtis", "Extra Producer"],
            "composers": ["Danny Elfman"],
            "actors": ["Tobey Maguire", "Kirsten Dunst", "Willem Dafoe", "James Franco", "Rosemary Harris", "J.K. Simmons"],
            "characters": ["Peter Parker", "Mary Jane Watson", "Norman Osborn"],
            "production_companies": ["Columbia Pictures", "Marvel Enterprises"],
            "imdb_rating": 7.4,
            "metacritic_rating": 73.0,
            "reception_summary": "Energetic and emotionally sincere.",
            "featured_reviews": [],
            "review_themes": [],
            "plot_events_metadata": None,
            "plot_analysis_metadata": None,
            "viewer_experience_metadata": None,
            "watch_context_metadata": None,
            "narrative_techniques_metadata": None,
            "reception_metadata": None,
            "production_metadata": None,
            "debug_synopses": [],
            "debug_plot_summaries": [],
        }

        # Apply caller-specific overrides to target scenario-specific behavior.
        base_data.update(overrides)
        return BaseMovie(**base_data)

    return _factory
