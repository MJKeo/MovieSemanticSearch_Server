"""
Unit tests for internal helper functions in movie_ingestion.metadata_generation.run.

Tests the DB query helpers using real SQLite databases (no API mocking needed).

Covers:
  - _get_quality_passed_tmdb_ids
  - _ensure_generated_metadata_rows
  - _record_batch_ids
  - _get_active_batch_ids
  - _clear_batch_id
  - _get_live_eligible_tmdb_ids
  - _print_eligibility_summary
"""

import sqlite3

import pytest

from movie_ingestion.tracker import _SCHEMA_SQL, MovieStatus
from movie_ingestion.metadata_generation.run import (
    _get_quality_passed_tmdb_ids,
    _ensure_generated_metadata_rows,
    _record_batch_ids,
    _get_active_batch_ids,
    _clear_batch_id,
    _get_live_eligible_tmdb_ids,
    _print_eligibility_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def in_memory_db() -> sqlite3.Connection:
    """In-memory SQLite database with the full pipeline schema applied."""
    db = sqlite3.connect(":memory:")
    db.executescript(_SCHEMA_SQL)
    db.commit()
    return db


@pytest.fixture()
def tracker_db(tmp_path):
    """Tracker DB at a tmp_path for functions that accept a path param."""
    db_path = tmp_path / "tracker.db"
    db = sqlite3.connect(str(db_path))
    db.executescript(_SCHEMA_SQL)
    db.commit()
    return db, db_path


# ---------------------------------------------------------------------------
# Tests: _get_quality_passed_tmdb_ids
# ---------------------------------------------------------------------------


class TestGetQualityPassedTmdbIds:
    """Tests for _get_quality_passed_tmdb_ids."""

    def test_returns_imdb_quality_passed_movies(self, in_memory_db) -> None:
        """Only movies at imdb_quality_passed status are returned."""
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (1, 'imdb_quality_passed')"
        )
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (2, 'pending')"
        )
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (3, 'imdb_quality_passed')"
        )
        in_memory_db.commit()

        result = _get_quality_passed_tmdb_ids(in_memory_db)
        assert set(result) == {1, 3}

    def test_empty_db_returns_empty(self, in_memory_db) -> None:
        """No movies returns empty list."""
        result = _get_quality_passed_tmdb_ids(in_memory_db)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: _ensure_generated_metadata_rows
# ---------------------------------------------------------------------------


class TestEnsureGeneratedMetadataRows:
    """Tests for _ensure_generated_metadata_rows."""

    def test_creates_rows(self, in_memory_db) -> None:
        """INSERT OR IGNORE creates rows for new tmdb_ids."""
        _ensure_generated_metadata_rows(in_memory_db, [1, 2, 3])

        rows = in_memory_db.execute(
            "SELECT tmdb_id FROM generated_metadata ORDER BY tmdb_id"
        ).fetchall()
        assert [r[0] for r in rows] == [1, 2, 3]

    def test_idempotent(self, in_memory_db) -> None:
        """Calling twice with same IDs doesn't duplicate or error."""
        _ensure_generated_metadata_rows(in_memory_db, [1, 2])
        _ensure_generated_metadata_rows(in_memory_db, [1, 2])

        count = in_memory_db.execute(
            "SELECT COUNT(*) FROM generated_metadata"
        ).fetchone()[0]
        assert count == 2

    def test_preserves_existing_data(self, in_memory_db) -> None:
        """Existing row with plot_events data is not overwritten."""
        in_memory_db.execute(
            "INSERT INTO generated_metadata (tmdb_id, plot_events) VALUES (1, '{\"plot_summary\": \"test\"}')"
        )
        in_memory_db.commit()

        _ensure_generated_metadata_rows(in_memory_db, [1])

        row = in_memory_db.execute(
            "SELECT plot_events FROM generated_metadata WHERE tmdb_id = 1"
        ).fetchone()
        assert row[0] is not None
        assert "test" in row[0]


# ---------------------------------------------------------------------------
# Tests: _record_batch_ids
# ---------------------------------------------------------------------------


class TestRecordBatchIds:
    """Tests for _record_batch_ids."""

    def test_creates_and_sets_batch_id(self, in_memory_db) -> None:
        """metadata_batch_ids rows are created with correct batch_id."""
        batch_requests = [
            {"custom_id": "plot_events_1"},
            {"custom_id": "plot_events_2"},
        ]
        _record_batch_ids(in_memory_db, batch_requests, "batch_abc")
        in_memory_db.commit()

        rows = in_memory_db.execute(
            "SELECT tmdb_id, plot_events_batch_id FROM metadata_batch_ids ORDER BY tmdb_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == (1, "batch_abc")
        assert rows[1] == (2, "batch_abc")

    def test_parses_custom_ids(self, in_memory_db) -> None:
        """tmdb_ids are correctly extracted from request custom_ids."""
        batch_requests = [
            {"custom_id": "plot_events_42"},
            {"custom_id": "plot_events_99"},
        ]
        _record_batch_ids(in_memory_db, batch_requests, "batch_xyz")
        in_memory_db.commit()

        tmdb_ids = [
            r[0] for r in in_memory_db.execute(
                "SELECT tmdb_id FROM metadata_batch_ids ORDER BY tmdb_id"
            ).fetchall()
        ]
        assert tmdb_ids == [42, 99]


# ---------------------------------------------------------------------------
# Tests: _get_active_batch_ids
# ---------------------------------------------------------------------------


class TestGetActiveBatchIds:
    """Tests for _get_active_batch_ids."""

    def test_returns_distinct(self, tracker_db) -> None:
        """Multiple movies with same batch_id returns one entry."""
        db, db_path = tracker_db
        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (1, 'batch_1')"
        )
        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (2, 'batch_1')"
        )
        db.commit()
        db.close()

        result = _get_active_batch_ids(db_path)
        assert result == ["batch_1"]

    def test_excludes_null(self, tracker_db) -> None:
        """Movies with NULL batch_id are excluded."""
        db, db_path = tracker_db
        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (1, NULL)"
        )
        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (2, 'batch_2')"
        )
        db.commit()
        db.close()

        result = _get_active_batch_ids(db_path)
        assert result == ["batch_2"]


# ---------------------------------------------------------------------------
# Tests: _clear_batch_id
# ---------------------------------------------------------------------------


class TestClearBatchId:
    """Tests for _clear_batch_id."""

    def test_sets_null(self, tracker_db) -> None:
        """After clearing, plot_events_batch_id is NULL for all movies in that batch."""
        db, db_path = tracker_db
        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (1, 'batch_1')"
        )
        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (2, 'batch_1')"
        )
        db.commit()
        db.close()

        _clear_batch_id(db_path, "batch_1")

        with sqlite3.connect(str(db_path)) as check_db:
            rows = check_db.execute(
                "SELECT plot_events_batch_id FROM metadata_batch_ids ORDER BY tmdb_id"
            ).fetchall()
        assert all(r[0] is None for r in rows)

    def test_only_affects_target_batch(self, tracker_db) -> None:
        """Other batches are not affected by clearing."""
        db, db_path = tracker_db
        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (1, 'batch_1')"
        )
        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (2, 'batch_2')"
        )
        db.commit()
        db.close()

        _clear_batch_id(db_path, "batch_1")

        with sqlite3.connect(str(db_path)) as check_db:
            bid = check_db.execute(
                "SELECT plot_events_batch_id FROM metadata_batch_ids WHERE tmdb_id = 2"
            ).fetchone()[0]
        assert bid == "batch_2"


# ---------------------------------------------------------------------------
# Tests: _get_live_eligible_tmdb_ids
# ---------------------------------------------------------------------------


class TestGetLiveEligibleTmdbIds:
    """Tests for _get_live_eligible_tmdb_ids."""

    def _setup_eligible_movie(self, db: sqlite3.Connection, tmdb_id: int) -> None:
        """Insert a movie that is eligible for live generation."""
        db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (?, 'imdb_quality_passed')",
            (tmdb_id,),
        )
        db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (?, 1)",
            (tmdb_id,),
        )

    def test_basic(self, tracker_db) -> None:
        """Eligible movies with no batch and no result are returned."""
        db, db_path = tracker_db
        self._setup_eligible_movie(db, 1)
        db.commit()
        db.close()

        result = _get_live_eligible_tmdb_ids(db_path, limit=10)
        assert 1 in result

    def test_excludes_batched(self, tracker_db) -> None:
        """Movies with active batch_id are excluded."""
        db, db_path = tracker_db
        self._setup_eligible_movie(db, 1)
        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (1, 'batch_x')"
        )
        db.commit()
        db.close()

        result = _get_live_eligible_tmdb_ids(db_path, limit=10)
        assert 1 not in result

    def test_excludes_already_generated(self, tracker_db) -> None:
        """Movies with plot_events result are excluded."""
        db, db_path = tracker_db
        self._setup_eligible_movie(db, 1)
        db.execute(
            "UPDATE generated_metadata SET plot_events = '{}' WHERE tmdb_id = 1"
        )
        db.commit()
        db.close()

        result = _get_live_eligible_tmdb_ids(db_path, limit=10)
        assert 1 not in result

    def test_excludes_ineligible(self, tracker_db) -> None:
        """Movies with eligible_for_plot_events=0 are excluded."""
        db, db_path = tracker_db
        db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (1, 'imdb_quality_passed')"
        )
        db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (1, 0)"
        )
        db.commit()
        db.close()

        result = _get_live_eligible_tmdb_ids(db_path, limit=10)
        assert 1 not in result

    def test_respects_limit(self, tracker_db) -> None:
        """Limit parameter caps returned count."""
        db, db_path = tracker_db
        for tid in range(1, 11):
            self._setup_eligible_movie(db, tid)
        db.commit()
        db.close()

        result = _get_live_eligible_tmdb_ids(db_path, limit=3)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: _print_eligibility_summary
# ---------------------------------------------------------------------------


class TestPrintEligibilitySummary:
    """Tests for _print_eligibility_summary."""

    def test_no_exception(self, in_memory_db) -> None:
        """Summary query doesn't crash with empty or populated data."""
        # Empty table
        _print_eligibility_summary(in_memory_db)

        # With some data
        in_memory_db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (1, 1)"
        )
        in_memory_db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (2, 0)"
        )
        in_memory_db.commit()

        # Should not raise
        _print_eligibility_summary(in_memory_db)
