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
    PipelineStage,
    _SCHEMA_SQL,
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
            "tmdb_id", "imdb_id", "status", "quality_score",
            "batch1_custom_id", "batch2_custom_id", "updated_at",
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
            "id", "tmdb_id", "title", "year", "stage",
            "reason", "details", "created_at",
        }
        assert col_names == expected

    def test_creates_tmdb_data_table(self, mocker, tmp_path) -> None:
        """init_db creates the tmdb_data table with all 18 columns."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(tmdb_data)").fetchall()
        db.close()

        assert len(cols) == 18

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

    def test_pulls_title_from_tmdb_data(self, in_memory_db) -> None:
        """log_filter populates title from tmdb_data table when available."""
        in_memory_db.execute(
            "INSERT INTO tmdb_data (tmdb_id, title, release_date) VALUES (300, 'Inception', '2010-07-16')"
        )

        log_filter(in_memory_db, tmdb_id=300, stage="tmdb_fetch", reason="test")
        in_memory_db.commit()

        title = in_memory_db.execute("SELECT title FROM filter_log WHERE tmdb_id = 300").fetchone()[0]
        assert title == "Inception"

    def test_parses_year_from_release_date(self, in_memory_db) -> None:
        """log_filter parses year from first 4 characters of release_date."""
        in_memory_db.execute(
            "INSERT INTO tmdb_data (tmdb_id, title, release_date) VALUES (301, 'Test', '2010-07-16')"
        )

        log_filter(in_memory_db, tmdb_id=301, stage="tmdb_fetch", reason="test")
        in_memory_db.commit()

        year = in_memory_db.execute("SELECT year FROM filter_log WHERE tmdb_id = 301").fetchone()[0]
        assert year == 2010

    def test_short_release_date_leaves_year_null(self, in_memory_db) -> None:
        """log_filter leaves year as NULL when release_date is shorter than 4 chars."""
        in_memory_db.execute(
            "INSERT INTO tmdb_data (tmdb_id, title, release_date) VALUES (302, 'Test', '20')"
        )

        log_filter(in_memory_db, tmdb_id=302, stage="tmdb_fetch", reason="test")
        in_memory_db.commit()

        year = in_memory_db.execute("SELECT year FROM filter_log WHERE tmdb_id = 302").fetchone()[0]
        assert year is None

    def test_empty_release_date_leaves_year_null(self, in_memory_db) -> None:
        """log_filter leaves year as NULL when release_date is empty string."""
        in_memory_db.execute(
            "INSERT INTO tmdb_data (tmdb_id, title, release_date) VALUES (303, 'Test', '')"
        )

        log_filter(in_memory_db, tmdb_id=303, stage="tmdb_fetch", reason="test")
        in_memory_db.commit()

        year = in_memory_db.execute("SELECT year FROM filter_log WHERE tmdb_id = 303").fetchone()[0]
        assert year is None

    def test_null_release_date_leaves_year_null(self, in_memory_db) -> None:
        """log_filter leaves year as NULL when release_date is NULL."""
        in_memory_db.execute(
            "INSERT INTO tmdb_data (tmdb_id, title, release_date) VALUES (304, 'Test', NULL)"
        )

        log_filter(in_memory_db, tmdb_id=304, stage="tmdb_fetch", reason="test")
        in_memory_db.commit()

        year = in_memory_db.execute("SELECT year FROM filter_log WHERE tmdb_id = 304").fetchone()[0]
        assert year is None

    def test_unparseable_year_leaves_year_null(self, in_memory_db) -> None:
        """log_filter leaves year as NULL when release_date starts with non-numeric chars."""
        in_memory_db.execute(
            "INSERT INTO tmdb_data (tmdb_id, title, release_date) VALUES (305, 'Test', 'abcd-01-01')"
        )

        log_filter(in_memory_db, tmdb_id=305, stage="tmdb_fetch", reason="test")
        in_memory_db.commit()

        year = in_memory_db.execute("SELECT year FROM filter_log WHERE tmdb_id = 305").fetchone()[0]
        assert year is None

    def test_no_tmdb_data_leaves_title_year_null(self, in_memory_db) -> None:
        """log_filter sets title and year to NULL when tmdb_data row does not exist."""
        log_filter(in_memory_db, tmdb_id=306, stage="tmdb_export_filter", reason="adult")
        in_memory_db.commit()

        row = in_memory_db.execute(
            "SELECT title, year FROM filter_log WHERE tmdb_id = 306"
        ).fetchone()
        assert row == (None, None)

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
