"""
Unit tests for movie_ingestion.tmdb_quality_scoring.tmdb_filter — Stage 3 threshold filter.

Covers:
  - run() — threshold-only filter that reads pre-computed scores and applies
    QUALITY_SCORE_THRESHOLD to advance survivors to tmdb_quality_passed.
"""

import sqlite3
from typing import Any
from unittest.mock import patch

import pytest

from movie_ingestion.tmdb_quality_scoring.tmdb_filter import (
    QUALITY_SCORE_THRESHOLD,
    run,
)
from movie_ingestion.tracker import MovieStatus, PipelineStage, _SCHEMA_SQL


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------


@pytest.fixture()
def filter_db(tmp_path) -> Any:
    """File-based SQLite DB seeded with the tracker schema.

    Returns the db Path so each test can open fresh connections.
    """
    db_path = tmp_path / "tracker.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    conn.close()
    return db_path


def _seed_calculated_movie(
    db_path: Any, tmdb_id: int, score: float | None = 0.5
) -> None:
    """Insert a movie_progress row with status='tmdb_quality_calculated' and a score."""
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """INSERT OR REPLACE INTO movie_progress
           (tmdb_id, status, stage_3_quality_score)
           VALUES (?, ?, ?)""",
        (tmdb_id, MovieStatus.TMDB_QUALITY_CALCULATED, score),
    )
    conn.commit()
    conn.close()


def _read_status(db_path: Any, tmdb_id: int) -> str | None:
    """Return the status for tmdb_id."""
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT status FROM movie_progress WHERE tmdb_id = ?", (tmdb_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _filter_log_count(db_path: Any, tmdb_id: int) -> int:
    """Return the number of filter_log entries for tmdb_id."""
    conn = sqlite3.connect(str(db_path))
    count = conn.execute(
        "SELECT COUNT(*) FROM filter_log WHERE tmdb_id = ?", (tmdb_id,)
    ).fetchone()[0]
    conn.close()
    return count


def _make_run_conn(db_path: Any) -> sqlite3.Connection:
    """Open a fresh connection for run() to use."""
    return sqlite3.connect(str(db_path))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTmdbFilterRun:
    """Tests for the threshold-based quality filter."""

    def test_filters_below_threshold(self, filter_db) -> None:
        """Movie with score < QUALITY_SCORE_THRESHOLD → filtered_out."""
        _seed_calculated_movie(filter_db, 1, score=QUALITY_SCORE_THRESHOLD - 0.01)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()

        assert _read_status(filter_db, 1) == "filtered_out"

    def test_passes_above_threshold(self, filter_db) -> None:
        """Movie with score > QUALITY_SCORE_THRESHOLD → tmdb_quality_passed."""
        _seed_calculated_movie(filter_db, 2, score=QUALITY_SCORE_THRESHOLD + 0.01)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()

        assert _read_status(filter_db, 2) == "tmdb_quality_passed"

    def test_exact_threshold_passes(self, filter_db) -> None:
        """Movie with score == QUALITY_SCORE_THRESHOLD → passes (not strictly less)."""
        _seed_calculated_movie(filter_db, 3, score=QUALITY_SCORE_THRESHOLD)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()

        assert _read_status(filter_db, 3) == "tmdb_quality_passed"

    def test_filter_log_entry_created(self, filter_db) -> None:
        """Filtered movie gets a filter_log row with correct stage and reason."""
        _seed_calculated_movie(filter_db, 4, score=0.01)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()

        conn = sqlite3.connect(str(filter_db))
        row = conn.execute(
            "SELECT stage, reason FROM filter_log WHERE tmdb_id = ?", (4,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == PipelineStage.TMDB_QUALITY_FUNNEL
        assert row[1] == "below_quality_threshold"

    def test_survivors_advanced_to_tmdb_quality_passed(self, filter_db) -> None:
        """Bulk UPDATE sets status to tmdb_quality_passed for survivors."""
        _seed_calculated_movie(filter_db, 5, score=0.9)
        _seed_calculated_movie(filter_db, 6, score=0.8)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()

        assert _read_status(filter_db, 5) == "tmdb_quality_passed"
        assert _read_status(filter_db, 6) == "tmdb_quality_passed"

    def test_null_score_guard_aborts(self, filter_db) -> None:
        """Movie with NULL score → prints error, exits without filtering."""
        _seed_calculated_movie(filter_db, 7, score=None)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()

        # Movie should still be in tmdb_quality_calculated — run() aborted.
        assert _read_status(filter_db, 7) == "tmdb_quality_calculated"
        assert _filter_log_count(filter_db, 7) == 0

    def test_skips_non_calculated_movies(self, filter_db) -> None:
        """Movies in other statuses are not evaluated."""
        # Insert a movie with status 'tmdb_fetched' — should be ignored.
        conn = sqlite3.connect(str(filter_db))
        conn.execute(
            "INSERT INTO movie_progress (tmdb_id, status, stage_3_quality_score) VALUES (?, ?, ?)",
            (8, "tmdb_fetched", 0.01),
        )
        conn.commit()
        conn.close()

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()

        # Status should remain unchanged.
        assert _read_status(filter_db, 8) == "tmdb_fetched"

    def test_empty_db_exits_gracefully(self, filter_db) -> None:
        """No tmdb_quality_calculated movies → no error."""
        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()  # must not raise

    def test_idempotent(self, filter_db) -> None:
        """Second run with no calculated movies left → no-op."""
        _seed_calculated_movie(filter_db, 9, score=0.9)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()

        assert _read_status(filter_db, 9) == "tmdb_quality_passed"

        # Second run — no tmdb_quality_calculated movies remain.
        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()

        # Status should still be tmdb_quality_passed (not re-filtered).
        assert _read_status(filter_db, 9) == "tmdb_quality_passed"

    def test_mixed_pass_and_fail(self, filter_db) -> None:
        """Multiple movies: some filtered, some passed, counts correct."""
        # 3 movies below threshold, 2 above.
        for tmdb_id in [10, 11, 12]:
            _seed_calculated_movie(filter_db, tmdb_id, score=0.05)
        for tmdb_id in [13, 14]:
            _seed_calculated_movie(filter_db, tmdb_id, score=0.9)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_filter.init_db",
                   side_effect=lambda: _make_run_conn(filter_db)):
            run()

        conn = sqlite3.connect(str(filter_db))
        filtered_count = conn.execute(
            "SELECT COUNT(*) FROM movie_progress WHERE status = 'filtered_out'"
        ).fetchone()[0]
        passed_count = conn.execute(
            "SELECT COUNT(*) FROM movie_progress WHERE status = 'tmdb_quality_passed'"
        ).fetchone()[0]
        conn.close()

        assert filtered_count == 3
        assert passed_count == 2
