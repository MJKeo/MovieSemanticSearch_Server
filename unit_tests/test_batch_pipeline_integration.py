"""
Integration-level tests for the batch pipeline using real SQLite DBs.

Tests the cmd_ functions with mocked OpenAI/LLM calls. Verifies the
end-to-end flow: eligibility → submit → process.

Covers:
  - cmd_eligibility: evaluates, skips already-evaluated, handles empty DB
  - cmd_submit: creates batches, respects max_batches
  - cmd_process: stores results, handles failed batches, skips in-progress
"""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from movie_ingestion.tracker import _SCHEMA_SQL, MovieStatus
from movie_ingestion.metadata_generation.inputs import MetadataType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline_db(tmp_path, mocker):
    """Create a fully-initialized tracker DB for pipeline integration tests.

    Patches init_db to use tmp_path and returns (db_path, db_connection).
    """
    db_path = tmp_path / "ingestion_data" / "tracker.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    db = sqlite3.connect(str(db_path))
    db.executescript(_SCHEMA_SQL)
    db.commit()

    # Patch tracker paths so init_db() points to our temp DB
    mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", db_path.parent)
    mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", db_path)

    return db_path, db


def _insert_quality_passed_movie(
    db: sqlite3.Connection,
    tmdb_id: int,
    title: str = "Test Movie",
) -> None:
    """Insert a movie at imdb_quality_passed with tmdb_data and imdb_data."""
    db.execute(
        "INSERT INTO movie_progress (tmdb_id, status) VALUES (?, ?)",
        (tmdb_id, MovieStatus.IMDB_QUALITY_PASSED),
    )
    db.execute(
        "INSERT INTO tmdb_data (tmdb_id, title, release_date) VALUES (?, ?, ?)",
        (tmdb_id, title, "2020-01-01"),
    )
    db.execute(
        "INSERT INTO imdb_data (tmdb_id, overview, synopses, plot_summaries) VALUES (?, ?, ?, ?)",
        (tmdb_id, f"Overview for {title}", '["A long synopsis text."]', '["Summary."]'),
    )


# ---------------------------------------------------------------------------
# Tests: cmd_eligibility
# ---------------------------------------------------------------------------


class TestCmdEligibility:
    """Tests for cmd_eligibility."""

    def test_evaluates_unevaluated_movies(self, pipeline_db) -> None:
        """Inserts imdb_quality_passed movies, runs eligibility, verifies flags are set."""
        db_path, db = pipeline_db
        _insert_quality_passed_movie(db, 1)
        _insert_quality_passed_movie(db, 2)
        db.commit()

        from movie_ingestion.metadata_generation.run import cmd_eligibility
        cmd_eligibility(metadata_type=MetadataType.PLOT_EVENTS, tracker_db_path=db_path)

        # Check that eligibility flags were set (either 0 or 1, not NULL)
        rows = db.execute(
            "SELECT tmdb_id, eligible_for_plot_events FROM generated_metadata ORDER BY tmdb_id"
        ).fetchall()

        assert len(rows) == 2
        for _, eligible in rows:
            assert eligible is not None  # Was evaluated (0 or 1)

    def test_skips_already_evaluated(self, pipeline_db) -> None:
        """Movies with existing eligibility flags are not re-evaluated."""
        db_path, db = pipeline_db
        _insert_quality_passed_movie(db, 1)
        db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (1, 1)"
        )
        db.commit()

        from movie_ingestion.metadata_generation.run import cmd_eligibility
        cmd_eligibility(metadata_type=MetadataType.PLOT_EVENTS, tracker_db_path=db_path)

        # Value should still be 1 (not re-evaluated to 0 or anything else)
        row = db.execute(
            "SELECT eligible_for_plot_events FROM generated_metadata WHERE tmdb_id = 1"
        ).fetchone()
        assert row[0] == 1

    def test_handles_empty_db(self, pipeline_db) -> None:
        """No movies returns early without error."""
        db_path, db = pipeline_db
        db.commit()

        from movie_ingestion.metadata_generation.run import cmd_eligibility
        # Should not raise
        cmd_eligibility(metadata_type=MetadataType.PLOT_EVENTS, tracker_db_path=db_path)

    def test_reception_eligibility(self, pipeline_db) -> None:
        """cmd_eligibility for reception type sets eligible_for_reception flags."""
        db_path, db = pipeline_db
        _insert_quality_passed_movie(db, 1)
        db.commit()

        from movie_ingestion.metadata_generation.run import cmd_eligibility
        cmd_eligibility(metadata_type=MetadataType.RECEPTION, tracker_db_path=db_path)

        row = db.execute(
            "SELECT eligible_for_reception FROM generated_metadata WHERE tmdb_id = 1"
        ).fetchone()
        # Should be evaluated (0 or 1, not NULL)
        assert row is not None
        assert row[0] is not None


# ---------------------------------------------------------------------------
# Tests: cmd_submit
# ---------------------------------------------------------------------------


class TestCmdSubmit:
    """Tests for cmd_submit with mocked OpenAI batch API."""

    @patch("movie_ingestion.metadata_generation.run.upload_and_create_batch")
    def test_creates_batches(self, mock_upload, pipeline_db) -> None:
        """Mock upload_and_create_batch, verify batch_ids are recorded."""
        db_path, db = pipeline_db
        _insert_quality_passed_movie(db, 1)
        db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (1, 1)"
        )
        db.commit()

        mock_upload.return_value = "batch_test_123"

        from movie_ingestion.metadata_generation.run import cmd_submit
        cmd_submit(metadata_type=MetadataType.PLOT_EVENTS, tracker_db_path=db_path, batch_size=100)

        # Verify batch_id was recorded in metadata_batch_ids
        row = db.execute(
            "SELECT plot_events_batch_id FROM metadata_batch_ids WHERE tmdb_id = 1"
        ).fetchone()
        assert row is not None
        assert row[0] == "batch_test_123"

    @patch("movie_ingestion.metadata_generation.run.upload_and_create_batch")
    def test_respects_max_batches(self, mock_upload, pipeline_db) -> None:
        """With max_batches=1 and enough movies for 2 batches, only 1 is submitted."""
        db_path, db = pipeline_db
        for tid in range(1, 6):
            _insert_quality_passed_movie(db, tid)
            db.execute(
                "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (?, 1)",
                (tid,),
            )
        db.commit()

        mock_upload.return_value = "batch_limited"

        from movie_ingestion.metadata_generation.run import cmd_submit
        cmd_submit(
            metadata_type=MetadataType.PLOT_EVENTS,
            tracker_db_path=db_path,
            batch_size=3,
            max_batches=1,
        )

        # Only 1 batch should have been submitted (3 movies in it)
        assert mock_upload.call_count == 1


# ---------------------------------------------------------------------------
# Tests: cmd_process
# ---------------------------------------------------------------------------


class TestCmdProcess:
    """Tests for cmd_process with mocked OpenAI batch API."""

    @patch("movie_ingestion.metadata_generation.run.download_results")
    @patch("movie_ingestion.metadata_generation.run.check_batch_status")
    def test_stores_results(self, mock_status, mock_download, pipeline_db) -> None:
        """Mock completed batch, verify results stored in generated_metadata."""
        db_path, db = pipeline_db

        # Set up a movie with an active batch
        db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (1, 1)"
        )
        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (1, 'batch_done')"
        )
        db.commit()

        # Mock the batch as completed
        from movie_ingestion.metadata_generation.openai_batch_manager import BatchStatus
        mock_status.return_value = BatchStatus(
            batch_id="batch_done",
            status="completed",
            total=1,
            completed=1,
            failed=0,
            output_file_id="file-out-123",
            error_file_id=None,
        )

        # Mock the download to return a valid result
        content = json.dumps({"plot_summary": "Test result."})
        mock_download.return_value = [
            {
                "custom_id": "plot_events_1",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{"message": {"content": content}}],
                        "usage": {"prompt_tokens": 50, "completion_tokens": 25},
                    },
                },
            },
        ]

        from movie_ingestion.metadata_generation.run import cmd_process
        cmd_process(tracker_db_path=db_path)

        # Verify result was stored
        row = db.execute(
            "SELECT plot_events FROM generated_metadata WHERE tmdb_id = 1"
        ).fetchone()
        assert row[0] is not None
        assert "Test result." in row[0]

    @patch("movie_ingestion.metadata_generation.run.check_batch_status")
    def test_handles_failed_batch(self, mock_status, pipeline_db) -> None:
        """Mock failed batch, verify batch_ids are cleared for resubmission."""
        db_path, db = pipeline_db

        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (1, 'batch_fail')"
        )
        db.commit()

        from movie_ingestion.metadata_generation.openai_batch_manager import BatchStatus
        mock_status.return_value = BatchStatus(
            batch_id="batch_fail",
            status="failed",
            total=1,
            completed=0,
            failed=1,
            output_file_id=None,
            error_file_id=None,
            errors=[{"code": "token_limit_exceeded", "message": "Too many tokens"}],
        )

        from movie_ingestion.metadata_generation.run import cmd_process
        cmd_process(tracker_db_path=db_path)

        # Verify batch_id was cleared
        row = db.execute(
            "SELECT plot_events_batch_id FROM metadata_batch_ids WHERE tmdb_id = 1"
        ).fetchone()
        assert row[0] is None

    @patch("movie_ingestion.metadata_generation.run.check_batch_status")
    def test_skips_in_progress(self, mock_status, pipeline_db) -> None:
        """Mock in-progress batch, verify it's skipped without processing."""
        db_path, db = pipeline_db

        db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (1, 'batch_wip')"
        )
        db.commit()

        from movie_ingestion.metadata_generation.openai_batch_manager import BatchStatus
        mock_status.return_value = BatchStatus(
            batch_id="batch_wip",
            status="in_progress",
            total=10,
            completed=5,
            failed=0,
            output_file_id=None,
            error_file_id=None,
        )

        from movie_ingestion.metadata_generation.run import cmd_process
        cmd_process(tracker_db_path=db_path)

        # Batch_id should still be present (not cleared)
        row = db.execute(
            "SELECT plot_events_batch_id FROM metadata_batch_ids WHERE tmdb_id = 1"
        ).fetchone()
        assert row[0] == "batch_wip"
