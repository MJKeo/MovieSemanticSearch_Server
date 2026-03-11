"""
Unit tests for movie_ingestion.tracker — SQLite checkpoint tracker.

Covers PipelineStage enum, init_db schema creation, log_filter helper,
and the JSON file I/O utilities (load_json, save_json).
"""

import json
import sqlite3

import pytest

from movie_ingestion.tracker import (
    INGESTION_DATA_DIR,
    TRACKER_DB_PATH,
    MovieStatus,
    PipelineStage,
    _SCHEMA_SQL,
    batch_log_filter,
    init_db,
    load_json,
    log_filter,
    save_json,
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


# ---------------------------------------------------------------------------
# PipelineStage
# ---------------------------------------------------------------------------


class TestPipelineStage:
    """Tests for the PipelineStage StrEnum."""

    def test_all_members_equal_string_values(self) -> None:
        """PipelineStage StrEnum members equal their string values."""
        for member in PipelineStage:
            assert member == member.value
            assert isinstance(member, str)

    def test_expected_members_exist(self) -> None:
        """PipelineStage contains all 9 expected pipeline stages."""
        expected = {
            "tmdb_export_filter",
            "tmdb_fetch",
            "tmdb_quality_funnel",
            "imdb_scrape",
            "essential_data_check",
            "llm_phase1",
            "llm_phase2",
            "embedding",
            "ingestion",
        }
        actual = {m.value for m in PipelineStage}
        assert actual == expected

    def test_usable_as_string_in_sql(self) -> None:
        """PipelineStage members interpolate cleanly into SQL text."""
        assert f"stage = '{PipelineStage.TMDB_FETCH}'" == "stage = 'tmdb_fetch'"


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    """Tests for init_db schema creation and configuration."""

    def test_creates_directory_and_database(self, mocker, tmp_path) -> None:
        """init_db creates the ingestion_data directory and tracker.db file."""
        data_dir = tmp_path / "ingestion_data"
        db_path = data_dir / "tracker.db"

        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", db_path)

        db = init_db()
        db.close()

        assert data_dir.is_dir()
        assert db_path.is_file()

    def test_creates_movie_progress_table(self, mocker, tmp_path) -> None:
        """init_db creates the movie_progress table with expected columns."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(movie_progress)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        expected = {
            "tmdb_id", "imdb_id", "status", "stage_3_quality_score",
            "stage_5_quality_score", "batch1_custom_id", "batch2_custom_id",
            "updated_at",
        }
        assert col_names == expected

    def test_creates_filter_log_table(self, mocker, tmp_path) -> None:
        """init_db creates the filter_log table with expected columns."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(filter_log)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        expected = {
            "id", "tmdb_id", "stage",
            "reason", "details", "created_at",
        }
        assert col_names == expected

    def test_creates_tmdb_data_table(self, mocker, tmp_path) -> None:
        """init_db creates the tmdb_data table with all 21 columns."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(tmdb_data)").fetchall()
        db.close()

        assert len(cols) == 21

    def test_creates_indexes(self, mocker, tmp_path) -> None:
        """init_db creates all three expected indexes."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name LIKE 'idx_%'"
        ).fetchall()
        db.close()

        index_names = {row[0] for row in rows}
        assert {"idx_progress_status", "idx_filter_log_stage", "idx_filter_log_tmdb"} <= index_names

    def test_enables_wal_mode(self, mocker, tmp_path) -> None:
        """init_db enables WAL journal mode."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        mode = db.execute("PRAGMA journal_mode").fetchone()[0]
        db.close()

        assert mode == "wal"

    def test_idempotent(self, mocker, tmp_path) -> None:
        """init_db can be called multiple times without error."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db1 = init_db()
        db1.close()
        # Second call should not raise — IF NOT EXISTS guards all DDL.
        db2 = init_db()
        db2.close()

    def test_returns_connection(self, mocker, tmp_path) -> None:
        """init_db returns an open sqlite3.Connection."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        assert isinstance(db, sqlite3.Connection)
        db.close()

    def test_enables_synchronous_full(self, mocker, tmp_path) -> None:
        """init_db sets PRAGMA synchronous to FULL (value 2)."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        sync_mode = db.execute("PRAGMA synchronous").fetchone()[0]
        db.close()

        assert sync_mode == 2

    def test_creates_tmdb_data_budget_column(self, mocker, tmp_path) -> None:
        """init_db creates the tmdb_data table with a budget column."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(tmdb_data)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        assert "budget" in col_names

    def test_creates_tmdb_data_maturity_rating_column(self, mocker, tmp_path) -> None:
        """init_db creates the tmdb_data table with a maturity_rating column."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(tmdb_data)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        assert "maturity_rating" in col_names

    def test_creates_tmdb_data_reviews_column(self, mocker, tmp_path) -> None:
        """init_db creates the tmdb_data table with a reviews column."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(tmdb_data)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        assert "reviews" in col_names

    def test_migration_drops_filter_log_title_year(self, mocker, tmp_path) -> None:
        """init_db migrates old filter_log schema by dropping title and year columns."""
        data_dir = tmp_path / "ingestion_data"
        db_path = data_dir / "tracker.db"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", db_path)

        # Create old schema with title/year columns
        data_dir.mkdir(parents=True, exist_ok=True)
        old_db = sqlite3.connect(str(db_path))
        old_db.execute("""
            CREATE TABLE IF NOT EXISTS filter_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tmdb_id INTEGER NOT NULL,
                title TEXT,
                year INTEGER,
                stage TEXT NOT NULL,
                reason TEXT NOT NULL,
                details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        old_db.commit()
        old_db.close()

        # Run init_db which should drop title/year via migrations
        db = init_db()
        cols = db.execute("PRAGMA table_info(filter_log)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        assert "title" not in col_names
        assert "year" not in col_names


# ---------------------------------------------------------------------------
# log_filter
# ---------------------------------------------------------------------------


class TestLogFilter:
    """Tests for the log_filter helper that records filtered-out movies."""

    def test_inserts_filter_log_row(self, in_memory_db) -> None:
        """log_filter inserts a row into filter_log with all provided fields."""
        log_filter(in_memory_db, tmdb_id=100, stage="tmdb_fetch", reason="tmdb_404")
        in_memory_db.commit()

        row = in_memory_db.execute("SELECT tmdb_id, stage, reason FROM filter_log").fetchone()
        assert row == (100, "tmdb_fetch", "tmdb_404")

    def test_updates_movie_progress_to_filtered_out(self, in_memory_db) -> None:
        """log_filter updates movie_progress status to 'filtered_out'."""
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (200, 'pending')"
        )

        log_filter(in_memory_db, tmdb_id=200, stage="tmdb_fetch", reason="missing_imdb_id")
        in_memory_db.commit()

        status = in_memory_db.execute(
            "SELECT status FROM movie_progress WHERE tmdb_id = 200"
        ).fetchone()[0]
        assert status == "filtered_out"

    def test_tolerates_missing_movie_progress_row(self, in_memory_db) -> None:
        """log_filter does not raise when movie_progress row does not exist."""
        # No movie_progress row inserted — UPDATE matches zero rows.
        log_filter(in_memory_db, tmdb_id=999, stage="tmdb_export_filter", reason="adult")
        in_memory_db.commit()

        # filter_log row still created
        count = in_memory_db.execute("SELECT COUNT(*) FROM filter_log").fetchone()[0]
        assert count == 1

    def test_does_not_commit(self, in_memory_db) -> None:
        """log_filter does not call db.commit() — caller is responsible."""
        # Insert a pending movie, commit, then call log_filter WITHOUT committing.
        # Open a second connection and verify the filter_log row is NOT visible
        # (proving log_filter did not commit on its own).
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (400, 'pending')"
        )
        in_memory_db.commit()

        log_filter(in_memory_db, tmdb_id=400, stage="tmdb_fetch", reason="test")

        # The row should be visible within the same connection (uncommitted)
        count_same = in_memory_db.execute(
            "SELECT COUNT(*) FROM filter_log WHERE tmdb_id = 400"
        ).fetchone()[0]
        assert count_same == 1

        # Roll back to prove it was never committed
        in_memory_db.rollback()
        count_after = in_memory_db.execute(
            "SELECT COUNT(*) FROM filter_log WHERE tmdb_id = 400"
        ).fetchone()[0]
        assert count_after == 0

    def test_details_stored_when_provided(self, in_memory_db) -> None:
        """log_filter stores details field when provided."""
        log_filter(
            in_memory_db, tmdb_id=307, stage="tmdb_fetch", reason="test",
            details='{"http_status": 500}',
        )
        in_memory_db.commit()

        details = in_memory_db.execute(
            "SELECT details FROM filter_log WHERE tmdb_id = 307"
        ).fetchone()[0]
        assert details == '{"http_status": 500}'

    def test_details_null_when_omitted(self, in_memory_db) -> None:
        """log_filter stores NULL for details when not provided."""
        log_filter(in_memory_db, tmdb_id=308, stage="tmdb_fetch", reason="test")
        in_memory_db.commit()

        details = in_memory_db.execute(
            "SELECT details FROM filter_log WHERE tmdb_id = 308"
        ).fetchone()[0]
        assert details is None

    def test_multiple_entries_for_same_movie(self, in_memory_db) -> None:
        """log_filter allows multiple filter_log rows for the same tmdb_id."""
        log_filter(in_memory_db, tmdb_id=309, stage="tmdb_fetch", reason="reason_a")
        log_filter(in_memory_db, tmdb_id=309, stage="tmdb_fetch", reason="reason_b")
        in_memory_db.commit()

        count = in_memory_db.execute(
            "SELECT COUNT(*) FROM filter_log WHERE tmdb_id = 309"
        ).fetchone()[0]
        assert count == 2

    def test_autoincrement_id(self, in_memory_db) -> None:
        """log_filter auto-increments filter_log id."""
        log_filter(in_memory_db, tmdb_id=310, stage="tmdb_fetch", reason="a")
        log_filter(in_memory_db, tmdb_id=311, stage="tmdb_fetch", reason="b")
        in_memory_db.commit()

        ids = [
            row[0]
            for row in in_memory_db.execute("SELECT id FROM filter_log ORDER BY id").fetchall()
        ]
        assert ids == [1, 2]


# ---------------------------------------------------------------------------
# batch_log_filter
# ---------------------------------------------------------------------------


class TestBatchLogFilter:
    """Tests for the bulk version of log_filter."""

    def test_batch_inserts_all_filter_log_rows(self, in_memory_db) -> None:
        """batch_log_filter inserts one filter_log row per entry."""
        entries = [
            (100, "tmdb_fetch", "missing_imdb_id", None),
            (200, "tmdb_fetch", "tmdb_404", None),
            (300, "imdb_scrape", "fetch_failed", '{"attempt": 3}'),
        ]

        batch_log_filter(in_memory_db, entries)
        in_memory_db.commit()

        rows = in_memory_db.execute(
            "SELECT tmdb_id, stage, reason, details FROM filter_log ORDER BY tmdb_id"
        ).fetchall()

        assert len(rows) == 3
        assert rows[0] == (100, "tmdb_fetch", "missing_imdb_id", None)
        assert rows[1] == (200, "tmdb_fetch", "tmdb_404", None)
        assert rows[2] == (300, "imdb_scrape", "fetch_failed", '{"attempt": 3}')

    def test_batch_updates_all_movie_progress_statuses(self, in_memory_db) -> None:
        """batch_log_filter updates all matching movie_progress rows to filtered_out."""
        for tid in (100, 200, 300):
            in_memory_db.execute(
                "INSERT INTO movie_progress (tmdb_id, status) VALUES (?, 'pending')",
                (tid,),
            )
        in_memory_db.commit()

        entries = [
            (100, "tmdb_fetch", "reason_a", None),
            (200, "tmdb_fetch", "reason_b", None),
            (300, "tmdb_fetch", "reason_c", None),
        ]
        batch_log_filter(in_memory_db, entries)
        in_memory_db.commit()

        statuses = in_memory_db.execute(
            "SELECT status FROM movie_progress ORDER BY tmdb_id"
        ).fetchall()
        assert all(s[0] == "filtered_out" for s in statuses)

    def test_batch_empty_list_is_noop(self, in_memory_db) -> None:
        """Empty entry list does not insert rows and does not raise."""
        batch_log_filter(in_memory_db, [])
        in_memory_db.commit()

        count = in_memory_db.execute("SELECT COUNT(*) FROM filter_log").fetchone()[0]
        assert count == 0

    def test_batch_tolerates_missing_progress_rows(self, in_memory_db) -> None:
        """batch_log_filter creates filter_log rows even without movie_progress entries."""
        entries = [
            (999, "tmdb_export_filter", "adult", None),
        ]
        batch_log_filter(in_memory_db, entries)
        in_memory_db.commit()

        count = in_memory_db.execute("SELECT COUNT(*) FROM filter_log").fetchone()[0]
        assert count == 1

    def test_batch_does_not_commit(self, in_memory_db) -> None:
        """batch_log_filter does not call db.commit() — caller is responsible."""
        entries = [(100, "tmdb_fetch", "reason", None)]
        batch_log_filter(in_memory_db, entries)

        # Row visible in same connection (uncommitted transaction)
        count = in_memory_db.execute(
            "SELECT COUNT(*) FROM filter_log WHERE tmdb_id = 100"
        ).fetchone()[0]
        assert count == 1

        # Roll back proves batch_log_filter did not commit
        in_memory_db.rollback()
        count_after = in_memory_db.execute(
            "SELECT COUNT(*) FROM filter_log WHERE tmdb_id = 100"
        ).fetchone()[0]
        assert count_after == 0

    def test_batch_single_entry(self, in_memory_db) -> None:
        """batch_log_filter handles a single-entry list correctly."""
        entries = [(42, "imdb_scrape", "imdb_404", None)]
        batch_log_filter(in_memory_db, entries)
        in_memory_db.commit()

        row = in_memory_db.execute(
            "SELECT tmdb_id, stage, reason FROM filter_log"
        ).fetchone()
        assert row == (42, "imdb_scrape", "imdb_404")

    def test_batch_preserves_details_field(self, in_memory_db) -> None:
        """batch_log_filter stores details JSON string correctly."""
        details_json = '{"http_status": 500, "retries": 3}'
        entries = [(100, "tmdb_fetch", "fetch_error", details_json)]
        batch_log_filter(in_memory_db, entries)
        in_memory_db.commit()

        details = in_memory_db.execute(
            "SELECT details FROM filter_log WHERE tmdb_id = 100"
        ).fetchone()[0]
        assert details == details_json

    def test_batch_mixed_details_null_and_present(self, in_memory_db) -> None:
        """batch_log_filter handles mix of entries with and without details."""
        entries = [
            (100, "tmdb_fetch", "reason_a", '{"key": "val"}'),
            (200, "tmdb_fetch", "reason_b", None),
            (300, "tmdb_fetch", "reason_c", '{"other": 1}'),
        ]
        batch_log_filter(in_memory_db, entries)
        in_memory_db.commit()

        rows = in_memory_db.execute(
            "SELECT tmdb_id, details FROM filter_log ORDER BY tmdb_id"
        ).fetchall()

        assert rows[0] == (100, '{"key": "val"}')
        assert rows[1] == (200, None)
        assert rows[2] == (300, '{"other": 1}')


# ---------------------------------------------------------------------------
# load_json / save_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    """Tests for the load_json file reader."""

    def test_reads_valid_file(self, tmp_path) -> None:
        """load_json reads and parses a valid JSON file."""
        path = tmp_path / "data.json"
        path.write_text('{"key": "value"}', encoding="utf-8")

        result = load_json(path)
        assert result == {"key": "value"}

    def test_raises_on_missing_file(self, tmp_path) -> None:
        """load_json raises FileNotFoundError for non-existent path."""
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nope.json")

    def test_raises_on_invalid_json(self, tmp_path) -> None:
        """load_json raises JSONDecodeError for malformed JSON."""
        path = tmp_path / "bad.json"
        path.write_text("not json {{", encoding="utf-8")

        with pytest.raises(json.JSONDecodeError):
            load_json(path)


class TestSaveJson:
    """Tests for the save_json atomic file writer."""

    def test_writes_file(self, tmp_path) -> None:
        """save_json writes JSON content to the specified path."""
        path = tmp_path / "out.json"
        save_json(path, {"a": 1})

        assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1}

    def test_creates_parent_directories(self, tmp_path) -> None:
        """save_json creates parent directories if they do not exist."""
        path = tmp_path / "a" / "b" / "file.json"
        save_json(path, {"nested": True})

        assert path.is_file()

    def test_atomic_write_no_tmp_left(self, tmp_path) -> None:
        """save_json uses write-then-rename, leaving no .tmp file behind."""
        path = tmp_path / "out.json"
        save_json(path, {"x": 1})

        assert not path.with_suffix(".tmp").exists()
        assert path.exists()

    def test_overwrites_existing_file(self, tmp_path) -> None:
        """save_json replaces existing file content."""
        path = tmp_path / "out.json"
        save_json(path, {"version": 1})
        save_json(path, {"version": 2})

        assert json.loads(path.read_text(encoding="utf-8")) == {"version": 2}

    def test_handles_list_data(self, tmp_path) -> None:
        """save_json accepts list data in addition to dict."""
        path = tmp_path / "list.json"
        save_json(path, [1, 2, 3])

        assert json.loads(path.read_text(encoding="utf-8")) == [1, 2, 3]

    def test_preserves_unicode(self, tmp_path) -> None:
        """save_json preserves non-ASCII characters."""
        path = tmp_path / "unicode.json"
        save_json(path, {"name": "Am\u00e9lie"})

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["name"] == "Am\u00e9lie"


# ---------------------------------------------------------------------------
# MovieStatus — new ESSENTIAL_DATA_PASSED member
# ---------------------------------------------------------------------------


class TestMovieStatus:
    """Tests for MovieStatus enum additions."""

    def test_movie_status_essential_data_passed_exists(self) -> None:
        """MovieStatus.ESSENTIAL_DATA_PASSED == 'essential_data_passed'."""
        assert MovieStatus.ESSENTIAL_DATA_PASSED == "essential_data_passed"

    def test_tmdb_quality_calculated_value(self) -> None:
        """MovieStatus.TMDB_QUALITY_CALCULATED == 'tmdb_quality_calculated'."""
        assert MovieStatus.TMDB_QUALITY_CALCULATED == "tmdb_quality_calculated"

    def test_tmdb_quality_calculated_in_status_progression(self) -> None:
        """TMDB_QUALITY_CALCULATED exists between TMDB_FETCHED and TMDB_QUALITY_PASSED.

        The enum members are defined in progression order; verify the new status
        sits between its neighbors in the member list.
        """
        members = list(MovieStatus)
        fetched_idx = members.index(MovieStatus.TMDB_FETCHED)
        calculated_idx = members.index(MovieStatus.TMDB_QUALITY_CALCULATED)
        passed_idx = members.index(MovieStatus.TMDB_QUALITY_PASSED)
        assert fetched_idx < calculated_idx < passed_idx


# ---------------------------------------------------------------------------
# Schema migrations — quality_score rename and stage_5_quality_score addition
# ---------------------------------------------------------------------------


class TestQualityScoreMigrations:
    """Tests for the quality_score → stage_3_quality_score rename
    and stage_5_quality_score column addition."""

    def test_movie_progress_has_stage_3_quality_score_column(self, mocker, tmp_path) -> None:
        """stage_3_quality_score column present after init_db."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(movie_progress)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        assert "stage_3_quality_score" in col_names

    def test_movie_progress_has_stage_5_quality_score_column(self, mocker, tmp_path) -> None:
        """stage_5_quality_score column present after init_db."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(movie_progress)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        assert "stage_5_quality_score" in col_names

    def test_migration_renames_quality_score_to_stage_3(self, mocker, tmp_path) -> None:
        """DB created with old schema → init_db migrates column name."""
        data_dir = tmp_path / "ingestion_data"
        db_path = data_dir / "tracker.db"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", db_path)

        # Create old schema with quality_score column
        data_dir.mkdir(parents=True, exist_ok=True)
        old_db = sqlite3.connect(str(db_path))
        old_db.execute("""
            CREATE TABLE IF NOT EXISTS movie_progress (
                tmdb_id          INTEGER PRIMARY KEY,
                imdb_id          TEXT,
                status           TEXT NOT NULL DEFAULT 'pending',
                quality_score    REAL,
                batch1_custom_id TEXT,
                batch2_custom_id TEXT,
                updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Insert a test row with a score to verify data survives migration
        old_db.execute(
            "INSERT INTO movie_progress (tmdb_id, quality_score) VALUES (1, 0.75)"
        )
        old_db.commit()
        old_db.close()

        # Run init_db which should apply migrations
        db = init_db()
        cols = db.execute("PRAGMA table_info(movie_progress)").fetchall()
        col_names = {row[1] for row in cols}

        # Old column should be gone, new column present
        assert "quality_score" not in col_names
        assert "stage_3_quality_score" in col_names

        # Verify data was preserved through the rename
        score = db.execute(
            "SELECT stage_3_quality_score FROM movie_progress WHERE tmdb_id = 1"
        ).fetchone()[0]
        db.close()

        assert score == pytest.approx(0.75)

    def test_migration_adds_stage_5_quality_score(self, mocker, tmp_path) -> None:
        """DB created without stage_5_quality_score → init_db adds it."""
        data_dir = tmp_path / "ingestion_data"
        db_path = data_dir / "tracker.db"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", db_path)

        # Create old schema without stage_5_quality_score
        data_dir.mkdir(parents=True, exist_ok=True)
        old_db = sqlite3.connect(str(db_path))
        old_db.execute("""
            CREATE TABLE IF NOT EXISTS movie_progress (
                tmdb_id               INTEGER PRIMARY KEY,
                imdb_id               TEXT,
                status                TEXT NOT NULL DEFAULT 'pending',
                stage_3_quality_score REAL,
                batch1_custom_id      TEXT,
                batch2_custom_id      TEXT,
                updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        old_db.commit()
        old_db.close()

        # Run init_db which should add the missing column
        db = init_db()
        cols = db.execute("PRAGMA table_info(movie_progress)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        assert "stage_5_quality_score" in col_names
