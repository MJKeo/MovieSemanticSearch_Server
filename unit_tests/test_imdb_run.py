"""
Unit tests for movie_ingestion.imdb_scraping.run.

Tests the _log_unexpected_error helper, _scrape_all batch orchestration,
and the run() entry point.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from movie_ingestion.imdb_scraping.run import _log_unexpected_error, _scrape_all, run
from movie_ingestion.imdb_scraping.scraper import MovieResult
from movie_ingestion.tracker import MovieStatus


# ---------------------------------------------------------------------------
# Tests: _log_unexpected_error
# ---------------------------------------------------------------------------


class TestLogUnexpectedError:
    """Tests for the structured error logging helper."""

    def test_writes_to_file(self, tmp_path: Path) -> None:
        """Error entry written to file with tmdb_id and exception info."""
        log_file = tmp_path / "errors.log"
        exc = ValueError("something broke")

        with patch("movie_ingestion.imdb_scraping.run._ERROR_LOG_PATH", log_file):
            _log_unexpected_error(12345, exc)

        content = log_file.read_text()
        assert "tmdb_id=12345" in content
        assert "ValueError" in content
        assert "something broke" in content

    def test_appends_not_overwrites(self, tmp_path: Path) -> None:
        """Calling twice produces 2 separate entries in the file."""
        log_file = tmp_path / "errors.log"

        with patch("movie_ingestion.imdb_scraping.run._ERROR_LOG_PATH", log_file):
            _log_unexpected_error(1, ValueError("first"))
            _log_unexpected_error(2, RuntimeError("second"))

        content = log_file.read_text()
        assert "tmdb_id=1" in content
        assert "tmdb_id=2" in content
        assert "first" in content
        assert "second" in content

    def test_includes_traceback(self, tmp_path: Path) -> None:
        """Entry includes a Python traceback."""
        log_file = tmp_path / "errors.log"

        try:
            raise TypeError("traceback test")
        except TypeError as exc:
            with patch("movie_ingestion.imdb_scraping.run._ERROR_LOG_PATH", log_file):
                _log_unexpected_error(999, exc)

        content = log_file.read_text()
        assert "Traceback" in content
        assert "traceback test" in content

    def test_includes_timestamp(self, tmp_path: Path) -> None:
        """Entry includes a UTC timestamp."""
        log_file = tmp_path / "errors.log"
        exc = ValueError("ts test")

        with patch("movie_ingestion.imdb_scraping.run._ERROR_LOG_PATH", log_file):
            _log_unexpected_error(1, exc)

        content = log_file.read_text()
        assert "UTC" in content

    def test_includes_exception_type_and_message(self, tmp_path: Path) -> None:
        """Entry includes the exception class name and message string."""
        log_file = tmp_path / "errors.log"
        exc = RuntimeError("custom error message")

        with patch("movie_ingestion.imdb_scraping.run._ERROR_LOG_PATH", log_file):
            _log_unexpected_error(42, exc)

        content = log_file.read_text()
        assert "RuntimeError" in content
        assert "custom error message" in content


# ---------------------------------------------------------------------------
# Tests: _scrape_all
# ---------------------------------------------------------------------------


class TestScrapeAll:
    """Tests for the batch orchestration of IMDB scraping."""

    def _patch_client(self):
        """Patch create_client to return an async context manager with a MagicMock."""
        mock_client = AsyncMock()
        mock_cm = AsyncMock()
        mock_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_cm.__aexit__ = AsyncMock(return_value=False)
        return patch(
            "movie_ingestion.imdb_scraping.run.create_client",
            return_value=mock_cm,
        )

    def _patch_ua(self):
        """Patch create_ua_generator to return a MagicMock."""
        return patch(
            "movie_ingestion.imdb_scraping.run.create_ua_generator",
            return_value=MagicMock(),
        )

    async def test_processes_in_batches_with_commits(self, mocker) -> None:
        """1200 candidates produce 3 batches (500+500+200) with 3 db.commit calls."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.imdb_scraping.run.process_movie",
            new_callable=AsyncMock,
            return_value=MovieResult(1, "scraped", None, {"original_title": "Test"}),
        )
        mocker.patch("movie_ingestion.imdb_scraping.run.batch_log_filter")

        candidates = [(i, f"tt{i:07d}") for i in range(1200)]

        with self._patch_client(), self._patch_ua():
            counters = await _scrape_all(mock_db, candidates)

        assert mock_db.commit.call_count == 3

    async def test_scraped_movies_get_status_update(self, mocker) -> None:
        """MovieResult(status='scraped') causes db.executemany with IMDB_SCRAPED status."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.imdb_scraping.run.process_movie",
            new_callable=AsyncMock,
            return_value=MovieResult(100, "scraped", None, {"original_title": "Test"}),
        )
        mocker.patch("movie_ingestion.imdb_scraping.run.batch_log_filter")

        candidates = [(100, "tt0000100")]

        with self._patch_client(), self._patch_ua():
            await _scrape_all(mock_db, candidates)

        # Find executemany calls that update to IMDB_SCRAPED
        executemany_calls = mock_db.executemany.call_args_list
        status_updates = [
            c for c in executemany_calls
            if MovieStatus.IMDB_SCRAPED in str(c)
        ]
        assert len(status_updates) == 1

    async def test_filtered_movies_get_batch_log_filter(self, mocker) -> None:
        """MovieResult(status='filtered') causes batch_log_filter call with correct entries."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.imdb_scraping.run.process_movie",
            new_callable=AsyncMock,
            return_value=MovieResult(100, "filtered", "imdb_404", None),
        )
        blf_mock = mocker.patch("movie_ingestion.imdb_scraping.run.batch_log_filter")

        candidates = [(100, "tt0000100")]

        with self._patch_client(), self._patch_ua():
            await _scrape_all(mock_db, candidates)

        blf_mock.assert_called_once()
        entries = blf_mock.call_args[0][1]
        assert len(entries) == 1
        assert entries[0][0] == 100  # tmdb_id
        assert entries[0][2] == "imdb_404"  # reason

    async def test_error_movies_increment_counter(self, mocker) -> None:
        """MovieResult(status='error') increments counters['errors'], no DB write."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.imdb_scraping.run.process_movie",
            new_callable=AsyncMock,
            return_value=MovieResult(100, "error", None, None),
        )
        mocker.patch("movie_ingestion.imdb_scraping.run.batch_log_filter")

        candidates = [(100, "tt0000100")]

        with self._patch_client(), self._patch_ua():
            counters = await _scrape_all(mock_db, candidates)

        assert counters["errors"] == 1

    async def test_gather_exception_logged_and_counted(self, mocker) -> None:
        """asyncio.gather captures Exception — _log_unexpected_error called, errors incremented."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.imdb_scraping.run.process_movie",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        )
        mocker.patch("movie_ingestion.imdb_scraping.run.batch_log_filter")
        log_mock = mocker.patch(
            "movie_ingestion.imdb_scraping.run._log_unexpected_error"
        )

        candidates = [(100, "tt0000100")]

        with self._patch_client(), self._patch_ua():
            counters = await _scrape_all(mock_db, candidates)

        assert counters["errors"] == 1
        log_mock.assert_called_once()

    async def test_empty_candidates_no_ops(self, mocker) -> None:
        """Empty list produces no batches, no commits, zero counters."""
        mock_db = MagicMock()
        mocker.patch("movie_ingestion.imdb_scraping.run.batch_log_filter")

        with self._patch_client(), self._patch_ua():
            counters = await _scrape_all(mock_db, [])

        assert counters == {"scraped": 0, "filtered": 0, "errors": 0}
        mock_db.commit.assert_not_called()

    async def test_returns_counters_dict(self, mocker) -> None:
        """Returns dict with scraped/filtered/errors keys."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.imdb_scraping.run.process_movie",
            new_callable=AsyncMock,
            return_value=MovieResult(1, "scraped", None, {"original_title": "Test"}),
        )
        mocker.patch("movie_ingestion.imdb_scraping.run.batch_log_filter")

        candidates = [(1, "tt0000001")]

        with self._patch_client(), self._patch_ua():
            counters = await _scrape_all(mock_db, candidates)

        assert "scraped" in counters
        assert "filtered" in counters
        assert "errors" in counters

    async def test_single_batch_commit(self, mocker) -> None:
        """100 candidates (< 500) produce exactly 1 commit."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.imdb_scraping.run.process_movie",
            new_callable=AsyncMock,
            return_value=MovieResult(1, "scraped", None, {"original_title": "Test"}),
        )
        mocker.patch("movie_ingestion.imdb_scraping.run.batch_log_filter")

        candidates = [(i, f"tt{i:07d}") for i in range(100)]

        with self._patch_client(), self._patch_ua():
            await _scrape_all(mock_db, candidates)

        assert mock_db.commit.call_count == 1


# ---------------------------------------------------------------------------
# Tests: run
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for the Stage 4 entry point."""

    def test_no_candidates_exits_early(self, mocker, capsys) -> None:
        """No tmdb_quality_passed movies prints 'Nothing to do', does not call _scrape_all."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        mocker.patch("movie_ingestion.imdb_scraping.run.init_db", return_value=mock_db)
        scrape_mock = mocker.patch("movie_ingestion.imdb_scraping.run._scrape_all")

        run()

        output = capsys.readouterr().out
        assert "Nothing to do" in output
        scrape_mock.assert_not_called()

    def test_closes_db_on_success(self, mocker) -> None:
        """db.close() called after successful run."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        mocker.patch("movie_ingestion.imdb_scraping.run.init_db", return_value=mock_db)

        run()

        mock_db.close.assert_called_once()

    def test_closes_db_on_exception(self, mocker) -> None:
        """db.close() called even when _scrape_all raises."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = [(1, "tt001")]
        mocker.patch("movie_ingestion.imdb_scraping.run.init_db", return_value=mock_db)
        mocker.patch(
            "movie_ingestion.imdb_scraping.run.asyncio.run",
            side_effect=RuntimeError("crash"),
        )

        with pytest.raises(RuntimeError, match="crash"):
            run()

        mock_db.close.assert_called_once()
