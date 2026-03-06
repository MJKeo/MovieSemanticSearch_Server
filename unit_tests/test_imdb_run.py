"""
Unit tests for movie_ingestion.imdb_scraping.run.

Tests the _log_unexpected_error helper that writes structured error entries
to the Stage 4 debug log file.
"""

from pathlib import Path
from unittest.mock import patch

from movie_ingestion.imdb_scraping.run import _log_unexpected_error


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
