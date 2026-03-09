"""
Unit tests for movie_ingestion.imdb_scraping.scraper.

Tests the per-movie process_movie orchestration (failure routing for
404, fetch failure, transform errors, and the success path) and the
MovieResult NamedTuple.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from movie_ingestion.imdb_scraping.http_client import FetchResult
from movie_ingestion.imdb_scraping.scraper import MovieResult, process_movie
from movie_ingestion.imdb_scraping.models import IMDBScrapedMovie


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
# Tests: MovieResult
# ---------------------------------------------------------------------------


class TestMovieResult:
    """Tests for the MovieResult NamedTuple."""

    def test_movie_result_fields(self) -> None:
        """MovieResult has tmdb_id (int), status (str), reason (str|None) fields."""
        result = MovieResult(tmdb_id=42, status="scraped", reason=None)

        assert result.tmdb_id == 42
        assert result.status == "scraped"
        assert result.reason is None

    def test_movie_result_is_namedtuple(self) -> None:
        """MovieResult supports indexing and unpacking like a NamedTuple."""
        result = MovieResult(tmdb_id=1, status="filtered", reason="imdb_404")

        # Supports indexing
        assert result[0] == 1
        assert result[1] == "filtered"
        assert result[2] == "imdb_404"

        # Supports unpacking
        tid, status, reason = result
        assert tid == 1
        assert status == "filtered"
        assert reason == "imdb_404"


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

    async def test_success_returns_scraped_result(self) -> None:
        """Successful fetch+transform returns MovieResult with status='scraped', reason=None."""
        with (
            self._patch_fetch(_success()),
            patch("movie_ingestion.imdb_scraping.scraper.save_json"),
        ):
            result = await process_movie(
                AsyncMock(), MagicMock(), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100",
            )

        assert isinstance(result, MovieResult)
        assert result.status == "scraped"
        assert result.reason is None
        assert result.tmdb_id == 100

    async def test_success_saves_json_file(self) -> None:
        """Successful path: save_json called with correct path and model dump."""
        with (
            self._patch_fetch(_success()),
            patch("movie_ingestion.imdb_scraping.scraper.save_json") as mock_save,
        ):
            await process_movie(
                AsyncMock(), MagicMock(), MagicMock(),
                tmdb_id=42, imdb_id="tt0000042",
            )

        mock_save.assert_called_once()
        path_arg = mock_save.call_args[0][0]
        assert str(path_arg).endswith("42.json")
        # Second arg should be a dict (model_dump output)
        data_arg = mock_save.call_args[0][1]
        assert isinstance(data_arg, dict)

    async def test_http_404_returns_filtered_result(self) -> None:
        """HTTP_404 fetch returns MovieResult(status='filtered', reason='imdb_404')."""
        with self._patch_fetch(_http_404()):
            result = await process_movie(
                AsyncMock(), MagicMock(), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100",
            )

        assert result.status == "filtered"
        assert result.reason == "imdb_404"

    async def test_http_404_does_not_save_json(self) -> None:
        """HTTP_404 path: save_json is NOT called."""
        with (
            self._patch_fetch(_http_404()),
            patch("movie_ingestion.imdb_scraping.scraper.save_json") as mock_save,
        ):
            await process_movie(
                AsyncMock(), MagicMock(), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100",
            )

        mock_save.assert_not_called()

    async def test_fetch_failed_returns_filtered_result(self) -> None:
        """FAILED fetch returns MovieResult(status='filtered', reason='fetch_failed')."""
        with self._patch_fetch(_failed()):
            result = await process_movie(
                AsyncMock(), MagicMock(), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100",
            )

        assert result.status == "filtered"
        assert result.reason == "fetch_failed"

    async def test_fetch_failed_does_not_save_json(self) -> None:
        """FAILED path: save_json is NOT called."""
        with (
            self._patch_fetch(_failed()),
            patch("movie_ingestion.imdb_scraping.scraper.save_json") as mock_save,
        ):
            await process_movie(
                AsyncMock(), MagicMock(), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100",
            )

        mock_save.assert_not_called()

    async def test_transform_exception_returns_error_result(self) -> None:
        """Transform raises exception: returns MovieResult(status='error', reason=None)."""
        with (
            self._patch_fetch(_success()),
            patch(
                "movie_ingestion.imdb_scraping.scraper.transform_graphql_response",
                side_effect=ValueError("bad data"),
            ),
        ):
            result = await process_movie(
                AsyncMock(), MagicMock(), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100",
            )

        assert result.status == "error"
        assert result.reason is None

    async def test_transform_exception_does_not_save_json(self) -> None:
        """Transform error path: save_json is NOT called."""
        with (
            self._patch_fetch(_success()),
            patch(
                "movie_ingestion.imdb_scraping.scraper.transform_graphql_response",
                side_effect=ValueError("bad data"),
            ),
            patch("movie_ingestion.imdb_scraping.scraper.save_json") as mock_save,
        ):
            await process_movie(
                AsyncMock(), MagicMock(), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100",
            )

        mock_save.assert_not_called()

    async def test_no_db_interaction(self) -> None:
        """process_movie never touches a database — it only returns a result."""
        with (
            self._patch_fetch(_success()),
            patch("movie_ingestion.imdb_scraping.scraper.save_json"),
        ):
            # process_movie takes no db parameter — this test verifies the
            # API contract: it returns a result, not writes to a database.
            result = await process_movie(
                AsyncMock(), MagicMock(), MagicMock(),
                tmdb_id=100, imdb_id="tt0000100",
            )

        # The function signature has no db parameter — if it did, this
        # test would fail at the call site. The result is a plain NamedTuple.
        assert isinstance(result, MovieResult)
