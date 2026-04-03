"""
Unit tests for movie_ingestion.metadata_generation.batch_generation.result_processor.

Covers:
  - process_results: success, HTTP error, invalid JSON, empty choices,
    empty content, token accumulation, mixed results, invalid custom_id
  - process_results with reception type custom_ids
  - SCHEMA_BY_TYPE registry coverage
  - process_error_file: records failures, extracts error messages, skips
    invalid custom_ids, empty list
  - ProcessingSummary defaults
"""

import json
import sqlite3

import pytest

from movie_ingestion.metadata_generation.batch_generation.result_processor import (
    process_results,
    process_error_file,
    ProcessingSummary,
    SCHEMA_BY_TYPE,
)
from movie_ingestion.metadata_generation.inputs import MetadataType
from movie_ingestion.tracker import _SCHEMA_SQL


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tracker_db(tmp_path):
    """Create a tracker DB with schema and return (db, db_path) tuple."""
    db_path = tmp_path / "tracker.db"
    db = sqlite3.connect(str(db_path))
    db.executescript(_SCHEMA_SQL)
    db.commit()
    return db, db_path


def _seed_movie(db: sqlite3.Connection, tmdb_id: int) -> None:
    """Insert a movie into generated_metadata so result processing can UPDATE it."""
    db.execute(
        "INSERT OR IGNORE INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (?, 1)",
        (tmdb_id,),
    )
    db.commit()


def _make_success_result(tmdb_id: int, plot_summary: str = "A plot summary.") -> dict:
    """Build a valid batch result dict for a successful plot_events response."""
    content = json.dumps({"plot_summary": plot_summary})
    return {
        "custom_id": f"plot_events_{tmdb_id}",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {"message": {"content": content}}
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                },
            },
        },
    }


def _make_reception_success_result(tmdb_id: int) -> dict:
    """Build a valid batch result dict for a successful reception response."""
    content = json.dumps({
        "source_material_hint": None,
        "thematic_observations": "Themes of identity.",
        "emotional_observations": "Intense and haunting.",
        "craft_observations": "Strong cinematography.",
        "reception_summary": "Widely acclaimed.",
        "praised_qualities": ["visionary"],
        "criticized_qualities": [],
    })
    return {
        "custom_id": f"reception_{tmdb_id}",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {"message": {"content": content}}
                ],
                "usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 80,
                },
            },
        },
    }


def _make_error_result(tmdb_id: int, status_code: int = 500) -> dict:
    """Build a batch result dict for an HTTP error response."""
    return {
        "custom_id": f"plot_events_{tmdb_id}",
        "response": {
            "status_code": status_code,
            "body": {
                "error": {"message": f"Server error {status_code}"},
            },
        },
    }


# ---------------------------------------------------------------------------
# Tests: ProcessingSummary
# ---------------------------------------------------------------------------


class TestProcessingSummary:
    """Tests for ProcessingSummary dataclass defaults."""

    def test_defaults(self) -> None:
        """ProcessingSummary() has all zeros."""
        s = ProcessingSummary()
        assert s.total == 0
        assert s.succeeded == 0
        assert s.failed == 0
        assert s.total_input_tokens == 0
        assert s.total_output_tokens == 0


# ---------------------------------------------------------------------------
# Tests: SCHEMA_BY_TYPE
# ---------------------------------------------------------------------------


class TestSchemaByType:
    """Tests for SCHEMA_BY_TYPE registry."""

    def test_includes_all_registered_types(self) -> None:
        """SCHEMA_BY_TYPE has entries for all 7 registered types."""
        assert MetadataType.PLOT_EVENTS in SCHEMA_BY_TYPE
        assert MetadataType.RECEPTION in SCHEMA_BY_TYPE
        assert MetadataType.PLOT_ANALYSIS in SCHEMA_BY_TYPE
        assert MetadataType.PRODUCTION_KEYWORDS in SCHEMA_BY_TYPE
        assert MetadataType.VIEWER_EXPERIENCE in SCHEMA_BY_TYPE
        assert MetadataType.WATCH_CONTEXT in SCHEMA_BY_TYPE
        assert MetadataType.NARRATIVE_TECHNIQUES in SCHEMA_BY_TYPE


# ---------------------------------------------------------------------------
# Tests: process_results — plot_events
# ---------------------------------------------------------------------------


class TestProcessResults:
    """Tests for process_results (generic, replaces process_plot_events_results)."""

    def test_success_stores_content(self, tracker_db) -> None:
        """Valid result with HTTP 200 stores content in generated_metadata."""
        db, db_path = tracker_db
        _seed_movie(db, 1)
        db.close()

        results = [_make_success_result(1, "Neo discovers the Matrix.")]
        summary = process_results(results, db_path)

        assert summary.succeeded == 1
        assert summary.failed == 0

        # Verify the result was stored
        with sqlite3.connect(str(db_path)) as check_db:
            row = check_db.execute(
                "SELECT plot_events FROM generated_metadata WHERE tmdb_id = 1"
            ).fetchone()
        assert row[0] is not None
        assert "Neo discovers the Matrix." in row[0]

    def test_extracts_custom_id_correctly(self, tracker_db) -> None:
        """tmdb_id is correctly parsed from custom_id in result."""
        db, db_path = tracker_db
        _seed_movie(db, 42)
        db.close()

        results = [_make_success_result(42)]
        summary = process_results(results, db_path)
        assert summary.succeeded == 1

    def test_http_error_records_failure(self, tracker_db) -> None:
        """Result with status_code != 200 records failure in generation_failures."""
        db, db_path = tracker_db
        _seed_movie(db, 1)
        db.close()

        results = [_make_error_result(1, 500)]
        summary = process_results(results, db_path)

        assert summary.failed == 1
        assert summary.succeeded == 0

        # Verify failure was recorded
        with sqlite3.connect(str(db_path)) as check_db:
            row = check_db.execute(
                "SELECT error_message FROM generation_failures WHERE tmdb_id = 1"
            ).fetchone()
        assert row is not None
        assert "500" in row[0]

    def test_invalid_json_records_failure(self, tracker_db) -> None:
        """Result with content that's not valid PlotEventsOutput records failure."""
        db, db_path = tracker_db
        _seed_movie(db, 1)
        db.close()

        result = {
            "custom_id": "plot_events_1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {"message": {"content": '{"invalid_field": "value"}'}}
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            },
        }
        summary = process_results([result], db_path)
        assert summary.failed == 1

    def test_empty_choices_records_failure(self, tracker_db) -> None:
        """Result with empty choices array records failure."""
        db, db_path = tracker_db
        _seed_movie(db, 1)
        db.close()

        result = {
            "custom_id": "plot_events_1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            },
        }
        summary = process_results([result], db_path)
        assert summary.failed == 1

    def test_empty_content_records_failure(self, tracker_db) -> None:
        """Result with None content records failure."""
        db, db_path = tracker_db
        _seed_movie(db, 1)
        db.close()

        result = {
            "custom_id": "plot_events_1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": None}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            },
        }
        summary = process_results([result], db_path)
        assert summary.failed == 1

    def test_token_accumulation(self, tracker_db) -> None:
        """ProcessingSummary accumulates input/output tokens correctly."""
        db, db_path = tracker_db
        _seed_movie(db, 1)
        _seed_movie(db, 2)
        db.close()

        results = [
            _make_success_result(1),
            _make_success_result(2),
        ]
        summary = process_results(results, db_path)

        assert summary.total_input_tokens == 200  # 100 + 100
        assert summary.total_output_tokens == 100  # 50 + 50

    def test_mixed_success_failure(self, tracker_db) -> None:
        """Batch with some successes and some failures, verify counts match."""
        db, db_path = tracker_db
        _seed_movie(db, 1)
        _seed_movie(db, 2)
        db.close()

        results = [
            _make_success_result(1),
            _make_error_result(2, 429),
        ]
        summary = process_results(results, db_path)

        assert summary.succeeded == 1
        assert summary.failed == 1
        assert summary.total == 2

    def test_invalid_custom_id_increments_failed(self, tracker_db) -> None:
        """Result with unparseable custom_id increments failed count but doesn't crash."""
        db, db_path = tracker_db
        db.close()

        result = {
            "custom_id": "garbage_no_underscore",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": '{"plot_summary": "test"}'}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            },
        }
        summary = process_results([result], db_path)
        assert summary.failed == 1

    def test_empty_list(self, tracker_db) -> None:
        """Empty results list returns ProcessingSummary with all zeros."""
        _, db_path = tracker_db
        summary = process_results([], db_path)
        assert summary.total == 0
        assert summary.succeeded == 0
        assert summary.failed == 0


# ---------------------------------------------------------------------------
# Tests: process_results — reception type
# ---------------------------------------------------------------------------


class TestProcessResultsReception:
    """Tests for process_results handling reception custom_ids."""

    def test_reception_success_stores_in_reception_column(self, tracker_db) -> None:
        """Valid reception result is stored in the 'reception' column."""
        db, db_path = tracker_db
        db.execute(
            "INSERT OR IGNORE INTO generated_metadata (tmdb_id, eligible_for_reception) VALUES (1, 1)"
        )
        db.commit()
        db.close()

        results = [_make_reception_success_result(1)]
        summary = process_results(results, db_path)

        assert summary.succeeded == 1

        with sqlite3.connect(str(db_path)) as check_db:
            row = check_db.execute(
                "SELECT reception FROM generated_metadata WHERE tmdb_id = 1"
            ).fetchone()
        assert row[0] is not None
        assert "Widely acclaimed." in row[0]

    def test_reception_validates_against_reception_schema(self, tracker_db) -> None:
        """Reception custom_id with PlotEventsOutput JSON content fails validation."""
        db, db_path = tracker_db
        db.execute(
            "INSERT OR IGNORE INTO generated_metadata (tmdb_id, eligible_for_reception) VALUES (1, 1)"
        )
        db.commit()
        db.close()

        # PlotEventsOutput content under a reception custom_id
        content = json.dumps({"plot_summary": "A plot summary."})
        result = {
            "custom_id": "reception_1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 50},
                },
            },
        }
        summary = process_results([result], db_path)
        assert summary.failed == 1

    def test_mixed_types_in_single_batch(self, tracker_db) -> None:
        """Process results containing both plot_events and reception custom_ids."""
        db, db_path = tracker_db
        _seed_movie(db, 1)
        db.execute(
            "INSERT OR IGNORE INTO generated_metadata (tmdb_id, eligible_for_reception) VALUES (2, 1)"
        )
        db.commit()
        db.close()

        results = [
            _make_success_result(1),
            _make_reception_success_result(2),
        ]
        summary = process_results(results, db_path)

        assert summary.succeeded == 2
        assert summary.failed == 0

        with sqlite3.connect(str(db_path)) as check_db:
            pe_row = check_db.execute(
                "SELECT plot_events FROM generated_metadata WHERE tmdb_id = 1"
            ).fetchone()
            rc_row = check_db.execute(
                "SELECT reception FROM generated_metadata WHERE tmdb_id = 2"
            ).fetchone()
        assert pe_row[0] is not None
        assert rc_row[0] is not None

    def test_unknown_metadata_type_records_failure(self, tracker_db) -> None:
        """Result with unregistered type prefix records failure."""
        db, db_path = tracker_db
        db.close()

        content = json.dumps({"some_field": "value"})
        result = {
            "custom_id": "unknown_type_1",
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [{"message": {"content": content}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
            },
        }
        summary = process_results([result], db_path)
        assert summary.failed == 1


# ---------------------------------------------------------------------------
# Tests: process_error_file
# ---------------------------------------------------------------------------


class TestProcessErrorFile:
    """Tests for process_error_file."""

    def test_records_failures(self, tracker_db) -> None:
        """Error file entries create rows in generation_failures."""
        db, db_path = tracker_db
        db.close()

        errors = [
            {
                "custom_id": "plot_events_1",
                "response": {
                    "body": {
                        "error": {"message": "Rate limit exceeded"},
                    },
                },
            },
        ]
        count = process_error_file(errors, MetadataType.PLOT_EVENTS, db_path)
        assert count == 1

        # Verify the failure was recorded
        with sqlite3.connect(str(db_path)) as check_db:
            row = check_db.execute(
                "SELECT error_message FROM generation_failures WHERE tmdb_id = 1"
            ).fetchone()
        assert row is not None
        assert "Rate limit exceeded" in row[0]

    def test_extracts_error_message(self, tracker_db) -> None:
        """Error message is extracted from response.body.error.message."""
        db, db_path = tracker_db
        db.close()

        errors = [
            {
                "custom_id": "plot_events_42",
                "response": {
                    "body": {
                        "error": {"message": "Token limit exceeded for request"},
                    },
                },
            },
        ]
        process_error_file(errors, MetadataType.PLOT_EVENTS, db_path)

        with sqlite3.connect(str(db_path)) as check_db:
            row = check_db.execute(
                "SELECT error_message FROM generation_failures WHERE tmdb_id = 42"
            ).fetchone()
        assert "Token limit exceeded" in row[0]

    def test_skips_invalid_custom_ids(self, tracker_db) -> None:
        """Entries with unparseable custom_ids are skipped (not counted)."""
        db, db_path = tracker_db
        db.close()

        errors = [
            {
                "custom_id": "bad_format",
                "response": {
                    "body": {"error": {"message": "some error"}},
                },
            },
        ]
        count = process_error_file(errors, MetadataType.PLOT_EVENTS, db_path)
        assert count == 0

    def test_empty_returns_zero(self, tracker_db) -> None:
        """Empty error list returns 0."""
        _, db_path = tracker_db
        count = process_error_file([], MetadataType.PLOT_EVENTS, db_path)
        assert count == 0

    def test_returns_actual_recorded_count(self, tracker_db) -> None:
        """Returned count matches number of successfully recorded errors."""
        db, db_path = tracker_db
        db.close()

        errors = [
            {
                "custom_id": "plot_events_1",
                "response": {"body": {"error": {"message": "err1"}}},
            },
            {
                "custom_id": "bad_id",
                "response": {"body": {"error": {"message": "err2"}}},
            },
            {
                "custom_id": "plot_events_3",
                "response": {"body": {"error": {"message": "err3"}}},
            },
        ]
        count = process_error_file(errors, MetadataType.PLOT_EVENTS, db_path)
        # Only 2 have valid custom_ids (plot_events_1, plot_events_3)
        assert count == 2
