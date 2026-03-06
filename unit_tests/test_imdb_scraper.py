"""
Unit tests for movie_ingestion.imdb_scraping.scraper.

Tests the per-movie process_movie orchestration (failure routing for
404, fetch failure, transform errors, and the success path).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from movie_ingestion.imdb_scraping.http_client import FetchResult
from movie_ingestion.imdb_scraping.scraper import process_movie
from movie_ingestion.imdb_scraping.models import IMDBScrapedMovie
from movie_ingestion.tracker import MovieStatus


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid GraphQL title data that transform_graphql_response can process
_SAMPLE_TITLE_DATA = {
    "originalTitleText": {"text": "Test Movie"},
    "certificate": {"rating": "PG-13", "ratingReason": None},
    "plot": {"plotText": {"plainText": "A test plot."}},
    "ratingsSummary": {"aggregateRating": 7.0, "voteCount": 5000},
    "metacritic": None,
    "titleGenres": {"genres": [{"genre": {"text": "Drama"}}]},
    "interests": None,
    "countriesOfOrigin": None,
    "filmingLocations": None,
    "spokenLanguages": None,
    "productionBudget": None,
    "reviewSummary": None,
    "companyCredits": None,
    "plots": None,
    "keywords": None,
    "parentsGuide": None,
    "directors": None,
    "writers": None,
    "cast": None,
    "producers": None,
    "composers": None,
    "reviews": None,
}


def _make_counters() -> dict:
    """Return a fresh counters dict matching the run module's structure."""
    return {"scraped": 0, "filtered": 0, "errors": 0}


def _success(data: dict = _SAMPLE_TITLE_DATA) -> tuple[FetchResult, dict]:
    """Return a SUCCESS fetch result tuple."""
    return (FetchResult.SUCCESS, data)


def _failed() -> tuple[FetchResult, None]:
    """Return a FAILED fetch result tuple."""
    return (FetchResult.FAILED, None)


def _http_404() -> tuple[FetchResult, None]:
    """Return an HTTP_404 fetch result tuple."""
    return (FetchResult.HTTP_404, None)


# ---------------------------------------------------------------------------
# Tests: process_movie
# ---------------------------------------------------------------------------


class TestProcessMovie:
    """Tests for the per-movie async orchestration."""

    def _patch_fetch(self, return_value):
        """Return a patch context for fetch_movie with the given return_value."""
        return patch(
            "movie_ingestion.imdb_scraping.scraper.fetch_movie",
            new_callable=AsyncMock,
            return_value=return_value,
        )

    @pytest.mark.asyncio
    async def test_happy_path_success(self) -> None:
        """Successful fetch + transform: JSON saved, status updated, scraped counter incremented."""
        counters = _make_counters()
        db = MagicMock()

        with (
            self._patch_fetch(_success()),
            patch("movie_ingestion.imdb_scraping.scraper.save_json") as mock_save,
        ):
            await process_movie(
                AsyncMock(), asyncio.Semaphore(30), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100", db=db, counters=counters,
            )

        assert counters["scraped"] == 1
        mock_save.assert_called_once()
        db.execute.assert_called()

    @pytest.mark.asyncio
    async def test_http_404_filters_movie(self) -> None:
        """HTTP_404 result: log_filter called with reason='imdb_404', filtered counter incremented."""
        counters = _make_counters()
        db = MagicMock()

        with (
            self._patch_fetch(_http_404()),
            patch("movie_ingestion.imdb_scraping.scraper.log_filter") as mock_log,
            patch("movie_ingestion.imdb_scraping.scraper.save_json") as mock_save,
        ):
            await process_movie(
                AsyncMock(), asyncio.Semaphore(30), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100", db=db, counters=counters,
            )

        assert counters["filtered"] == 1
        mock_log.assert_called_once()
        # Verify reason is "imdb_404" (passed as keyword arg)
        assert mock_log.call_args[1]["reason"] == "imdb_404"
        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_fetch_failed_filters_movie(self) -> None:
        """FAILED result: log_filter called with reason='fetch_failed', filtered counter incremented."""
        counters = _make_counters()
        db = MagicMock()

        with (
            self._patch_fetch(_failed()),
            patch("movie_ingestion.imdb_scraping.scraper.log_filter") as mock_log,
            patch("movie_ingestion.imdb_scraping.scraper.save_json") as mock_save,
        ):
            await process_movie(
                AsyncMock(), asyncio.Semaphore(30), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100", db=db, counters=counters,
            )

        assert counters["filtered"] == 1
        mock_log.assert_called_once()
        # Verify reason is "fetch_failed" (passed as keyword arg)
        assert mock_log.call_args[1]["reason"] == "fetch_failed"
        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_transform_exception_increments_errors(self) -> None:
        """Transform raises an exception: errors counter incremented, movie skipped."""
        counters = _make_counters()
        db = MagicMock()

        with (
            self._patch_fetch(_success()),
            patch(
                "movie_ingestion.imdb_scraping.scraper.transform_graphql_response",
                side_effect=ValueError("bad data"),
            ),
            patch("movie_ingestion.imdb_scraping.scraper.save_json") as mock_save,
        ):
            await process_movie(
                AsyncMock(), asyncio.Semaphore(30), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100", db=db, counters=counters,
            )

        assert counters["errors"] == 1
        assert counters["scraped"] == 0
        mock_save.assert_not_called()

    @pytest.mark.asyncio
    async def test_db_status_update_on_success(self) -> None:
        """db.execute called with correct UPDATE SQL and status on success."""
        counters = _make_counters()
        db = MagicMock()

        with (
            self._patch_fetch(_success()),
            patch("movie_ingestion.imdb_scraping.scraper.save_json"),
        ):
            await process_movie(
                AsyncMock(), asyncio.Semaphore(30), MagicMock(),
                tmdb_id=777, imdb_id="tt0000777", db=db, counters=counters,
            )

        # Find the status update call among db.execute calls
        update_calls = [
            c for c in db.execute.call_args_list
            if "UPDATE movie_progress" in str(c)
        ]
        assert len(update_calls) == 1
        args = update_calls[0][0]
        assert args[1] == (MovieStatus.IMDB_SCRAPED, 777)

    @pytest.mark.asyncio
    async def test_save_json_called_with_correct_path(self) -> None:
        """save_json is called with the tmdb_id-based path and model dump."""
        counters = _make_counters()
        db = MagicMock()

        with (
            self._patch_fetch(_success()),
            patch("movie_ingestion.imdb_scraping.scraper.save_json") as mock_save,
        ):
            await process_movie(
                AsyncMock(), asyncio.Semaphore(30), MagicMock(),
                tmdb_id=42, imdb_id="tt0000042", db=db, counters=counters,
            )

        mock_save.assert_called_once()
        path_arg = mock_save.call_args[0][0]
        assert str(path_arg).endswith("42.json")
        # Second arg should be a dict (model_dump output)
        data_arg = mock_save.call_args[0][1]
        assert isinstance(data_arg, dict)
