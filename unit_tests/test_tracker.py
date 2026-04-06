"""
Unit tests for movie_ingestion.tracker — SQLite checkpoint tracker.

Covers PipelineStage enum, init_db schema creation, log_filter helper,
and the JSON file I/O utilities (load_json, save_json).
"""

import json
import sqlite3

import pytest

from movie_ingestion.tracker import (
    IMDB_DATA_COLUMNS,
    IMDB_JSON_COLUMNS,
    INGESTION_DATA_DIR,
    TRACKER_DB_PATH,
    MovieStatus,
    PipelineStage,
    _SCHEMA_SQL,
    batch_log_filter,
    deserialize_imdb_row,
    init_db,
    load_json,
    log_filter,
    log_ingestion_failures,
    save_json,
    serialize_imdb_movie,
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
            "imdb_quality_funnel",
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

        # batch1_custom_id and batch2_custom_id were removed by migration
        expected = {
            "tmdb_id", "imdb_id", "status", "stage_3_quality_score",
            "stage_5_quality_score", "updated_at",
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

    def test_creates_metadata_batch_ids_table(self, mocker, tmp_path) -> None:
        """init_db creates the metadata_batch_ids table with expected columns."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(metadata_batch_ids)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        expected = {
            "tmdb_id",
            "plot_events_batch_id", "reception_batch_id",
            "plot_analysis_batch_id", "viewer_experience_batch_id",
            "watch_context_batch_id", "narrative_techniques_batch_id",
            "production_keywords_batch_id", "source_of_inspiration_batch_id",
        }
        assert col_names == expected

    def test_creates_generated_metadata_table(self, mocker, tmp_path) -> None:
        """init_db creates generated_metadata with 8 JSON cols + 8 eligible_for_ cols."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(generated_metadata)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        # 1 PK + 8 JSON result cols + 8 eligible_for_ cols = 17
        assert "tmdb_id" in col_names
        for mt in [
            "plot_events", "reception", "plot_analysis", "viewer_experience",
            "watch_context", "narrative_techniques", "production_keywords",
            "source_of_inspiration",
        ]:
            assert mt in col_names, f"Missing JSON column: {mt}"
            assert f"eligible_for_{mt}" in col_names, f"Missing eligible column: eligible_for_{mt}"

    def test_creates_generation_failures_table(self, mocker, tmp_path) -> None:
        """init_db creates generation_failures table with expected columns."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        cols = db.execute("PRAGMA table_info(generation_failures)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        expected = {"id", "tmdb_id", "metadata_type", "error_message", "created_at"}
        assert col_names == expected

    def test_creates_generation_failures_index(self, mocker, tmp_path) -> None:
        """init_db creates idx_gen_failures_type index."""
        data_dir = tmp_path / "ingestion_data"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", data_dir / "tracker.db")

        db = init_db()
        rows = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_gen_failures_type'"
        ).fetchall()
        db.close()

        assert len(rows) == 1

    def test_migration_drops_batch_custom_id_columns(self, mocker, tmp_path) -> None:
        """init_db migrates old schema by dropping batch1/batch2_custom_id columns."""
        data_dir = tmp_path / "ingestion_data"
        db_path = data_dir / "tracker.db"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", db_path)

        # Create old schema with batch columns
        data_dir.mkdir(parents=True, exist_ok=True)
        old_db = sqlite3.connect(str(db_path))
        old_db.execute("""
            CREATE TABLE IF NOT EXISTS movie_progress (
                tmdb_id          INTEGER PRIMARY KEY,
                imdb_id          TEXT,
                status           TEXT NOT NULL DEFAULT 'pending',
                stage_3_quality_score REAL,
                stage_5_quality_score REAL,
                batch1_custom_id TEXT,
                batch2_custom_id TEXT,
                updated_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        old_db.commit()
        old_db.close()

        # Run init_db which should drop batch columns via migrations
        db = init_db()
        cols = db.execute("PRAGMA table_info(movie_progress)").fetchall()
        col_names = {row[1] for row in cols}
        db.close()

        assert "batch1_custom_id" not in col_names
        assert "batch2_custom_id" not in col_names

    def test_migration_drops_wave1_results_table(self, mocker, tmp_path) -> None:
        """init_db migrates old schema by dropping wave1_results table."""
        data_dir = tmp_path / "ingestion_data"
        db_path = data_dir / "tracker.db"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", db_path)

        # Create old wave1_results table
        data_dir.mkdir(parents=True, exist_ok=True)
        old_db = sqlite3.connect(str(db_path))
        old_db.execute("""
            CREATE TABLE IF NOT EXISTS wave1_results (
                tmdb_id INTEGER PRIMARY KEY,
                plot_events TEXT,
                reception TEXT
            )
        """)
        old_db.commit()
        old_db.close()

        # Run init_db which should drop wave1_results
        db = init_db()
        tables = db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'wave1_results'"
        ).fetchall()
        db.close()

        assert len(tables) == 0


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
# MovieStatus
# ---------------------------------------------------------------------------


class TestMovieStatus:
    """Tests for MovieStatus enum."""

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


# ---------------------------------------------------------------------------
# IMDB_DATA_COLUMNS / IMDB_JSON_COLUMNS — imdb_title_type
# ---------------------------------------------------------------------------


class TestImdbDataColumns:
    """Tests for the IMDB data column definitions."""

    def test_imdb_title_type_is_first_column(self) -> None:
        """imdb_title_type is the first entry in IMDB_DATA_COLUMNS."""
        assert IMDB_DATA_COLUMNS[0] == "imdb_title_type"

    def test_imdb_title_type_not_in_json_columns(self) -> None:
        """imdb_title_type is a scalar TEXT field, not JSON."""
        assert "imdb_title_type" not in IMDB_JSON_COLUMNS


# ---------------------------------------------------------------------------
# serialize_imdb_movie / deserialize_imdb_row — imdb_title_type round-trip
# ---------------------------------------------------------------------------


class TestSerializeImdbMovie:
    """Tests for serialize_imdb_movie and deserialize_imdb_row."""

    def test_serialize_includes_imdb_title_type(self) -> None:
        """imdb_title_type appears in the serialized tuple at position 1 (after tmdb_id)."""
        data = {"imdb_title_type": "movie"}
        result = serialize_imdb_movie(12345, data)
        # result[0] is tmdb_id, result[1] is the first IMDB_DATA_COLUMN
        assert result[0] == 12345
        assert result[1] == "movie"

    def test_serialize_imdb_title_type_passthrough(self) -> None:
        """imdb_title_type is a scalar — passed through directly (not JSON-serialized)."""
        data = {"imdb_title_type": "tvMovie"}
        result = serialize_imdb_movie(42, data)
        assert result[1] == "tvMovie"

    def test_round_trip_imdb_title_type(self, in_memory_db) -> None:
        """serialize → INSERT → SELECT → deserialize round-trips imdb_title_type."""
        data = {
            "imdb_title_type": "short",
            "original_title": "Test",
            "maturity_rating": "PG",
            "overview": "A test movie.",
            "imdb_rating": 7.0,
            "imdb_vote_count": 100,
            "metacritic_rating": None,
            "reception_summary": None,
            "budget": None,
            "overall_keywords": ["kw"],
            "genres": ["Drama"],
            "countries_of_origin": [],
            "production_companies": [],
            "filming_locations": [],
            "languages": ["English"],
            "synopses": [],
            "plot_summaries": [],
            "plot_keywords": [],
            "maturity_reasoning": [],
            "directors": ["Director"],
            "writers": [],
            "actors": [],
            "characters": [],
            "producers": [],
            "composers": [],
            "review_themes": [],
            "parental_guide_items": [],
            "featured_reviews": [],
        }
        row_tuple = serialize_imdb_movie(999, data)

        from movie_ingestion.tracker import IMDB_INSERT_SQL
        in_memory_db.execute(IMDB_INSERT_SQL, row_tuple)
        in_memory_db.commit()

        prev_factory = in_memory_db.row_factory
        in_memory_db.row_factory = sqlite3.Row
        row = in_memory_db.execute(
            "SELECT * FROM imdb_data WHERE tmdb_id = 999"
        ).fetchone()
        in_memory_db.row_factory = prev_factory

        result = deserialize_imdb_row(row)
        assert result["imdb_title_type"] == "short"
        assert result["original_title"] == "Test"
        assert result["genres"] == ["Drama"]


# ---------------------------------------------------------------------------
# MovieStatus enum updates
# ---------------------------------------------------------------------------


class TestMovieStatusUpdates:
    """Tests for new and removed MovieStatus enum members."""

    def test_metadata_generated_exists(self) -> None:
        """METADATA_GENERATED should be a valid member."""
        assert MovieStatus.METADATA_GENERATED == "metadata_generated"

    def test_ingestion_failed_exists(self) -> None:
        """INGESTION_FAILED should be a valid member."""
        assert MovieStatus.INGESTION_FAILED == "ingestion_failed"

    def test_phase1_complete_does_not_exist(self) -> None:
        """PHASE1_COMPLETE should no longer be a member."""
        assert not hasattr(MovieStatus, "PHASE1_COMPLETE")

    def test_phase2_complete_does_not_exist(self) -> None:
        """PHASE2_COMPLETE should no longer be a member."""
        assert not hasattr(MovieStatus, "PHASE2_COMPLETE")


# ---------------------------------------------------------------------------
# ingestion_failures table
# ---------------------------------------------------------------------------


class TestIngestionFailuresTable:
    """Tests for the ingestion_failures table schema created by _SCHEMA_SQL."""

    def test_table_exists(self, in_memory_db) -> None:
        """ingestion_failures table should be created by the schema."""
        tables = in_memory_db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'ingestion_failures'"
        ).fetchall()
        assert len(tables) == 1

    def test_expected_columns(self, in_memory_db) -> None:
        """ingestion_failures should have id, tmdb_id, error_message, created_at columns."""
        cols = in_memory_db.execute("PRAGMA table_info(ingestion_failures)").fetchall()
        col_names = {row[1] for row in cols}
        assert col_names == {"id", "tmdb_id", "error_message", "created_at"}

    def test_index_exists(self, in_memory_db) -> None:
        """idx_ingest_failures_tmdb index should exist."""
        rows = in_memory_db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' AND name = 'idx_ingest_failures_tmdb'"
        ).fetchall()
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# log_ingestion_failures
# ---------------------------------------------------------------------------


class TestLogIngestionFailures:
    """Tests for the log_ingestion_failures bulk-insert function."""

    def test_happy_path_inserts_rows_and_updates_status(self, in_memory_db) -> None:
        """Should insert failure rows and update movie_progress to ingestion_failed."""
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (100, 'embedded')"
        )
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (200, 'embedded')"
        )
        in_memory_db.commit()

        log_ingestion_failures(in_memory_db, [
            (100, "Postgres movie card: connection timeout"),
            (200, "Qdrant upsert: vector dimension mismatch"),
        ])
        in_memory_db.commit()

        # Verify failure rows
        rows = in_memory_db.execute(
            "SELECT tmdb_id, error_message FROM ingestion_failures ORDER BY tmdb_id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == (100, "Postgres movie card: connection timeout")
        assert rows[1] == (200, "Qdrant upsert: vector dimension mismatch")

        # Verify status updates
        for tid in (100, 200):
            status = in_memory_db.execute(
                "SELECT status FROM movie_progress WHERE tmdb_id = ?", (tid,)
            ).fetchone()[0]
            assert status == "ingestion_failed"

    def test_empty_list_is_noop(self, in_memory_db) -> None:
        """Empty failures list should not insert anything."""
        log_ingestion_failures(in_memory_db, [])
        count = in_memory_db.execute("SELECT COUNT(*) FROM ingestion_failures").fetchone()[0]
        assert count == 0

    def test_deduplicates_status_update(self, in_memory_db) -> None:
        """Same tmdb_id with multiple errors should only update status once."""
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (300, 'embedded')"
        )
        in_memory_db.commit()

        log_ingestion_failures(in_memory_db, [
            (300, "Postgres movie card: error A"),
            (300, "Postgres lexical: error B"),
        ])
        in_memory_db.commit()

        # Should have 2 failure rows
        failure_count = in_memory_db.execute(
            "SELECT COUNT(*) FROM ingestion_failures WHERE tmdb_id = 300"
        ).fetchone()[0]
        assert failure_count == 2

        # But only 1 movie_progress row
        status = in_memory_db.execute(
            "SELECT status FROM movie_progress WHERE tmdb_id = 300"
        ).fetchone()[0]
        assert status == "ingestion_failed"

    def test_does_not_commit(self, in_memory_db) -> None:
        """log_ingestion_failures should not commit — caller is responsible."""
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (400, 'embedded')"
        )
        in_memory_db.commit()

        log_ingestion_failures(in_memory_db, [(400, "test error")])

        # Row visible within same connection
        count = in_memory_db.execute(
            "SELECT COUNT(*) FROM ingestion_failures WHERE tmdb_id = 400"
        ).fetchone()[0]
        assert count == 1

        # Rollback proves it was not committed
        in_memory_db.rollback()
        count_after = in_memory_db.execute(
            "SELECT COUNT(*) FROM ingestion_failures WHERE tmdb_id = 400"
        ).fetchone()[0]
        assert count_after == 0


# ---------------------------------------------------------------------------
# Migration: phase1/phase2 → metadata_generated
# ---------------------------------------------------------------------------


class TestPhaseStatusMigration:
    """Tests for the migration that collapses phase1/phase2 into metadata_generated."""

    def test_migration_updates_phase1_to_metadata_generated(self, mocker, tmp_path) -> None:
        """Movies with phase1_complete status should become metadata_generated after init_db."""
        data_dir = tmp_path / "ingestion_data"
        db_path = data_dir / "tracker.db"
        mocker.patch("movie_ingestion.tracker.INGESTION_DATA_DIR", data_dir)
        mocker.patch("movie_ingestion.tracker.TRACKER_DB_PATH", db_path)

        # Create old schema with phase1_complete status
        data_dir.mkdir(parents=True, exist_ok=True)
        old_db = sqlite3.connect(str(db_path))
        old_db.execute("""
            CREATE TABLE IF NOT EXISTS movie_progress (
                tmdb_id INTEGER PRIMARY KEY,
                imdb_id TEXT,
                status TEXT NOT NULL DEFAULT 'pending',
                stage_3_quality_score REAL,
                stage_5_quality_score REAL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        old_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (100, 'phase1_complete')"
        )
        old_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (200, 'phase2_complete')"
        )
        old_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (300, 'embedded')"
        )
        old_db.commit()
        old_db.close()

        # Run init_db which should apply the migration
        db = init_db()
        statuses = {
            row[0]: row[1]
            for row in db.execute("SELECT tmdb_id, status FROM movie_progress").fetchall()
        }
        db.close()

        assert statuses[100] == "metadata_generated"
        assert statuses[200] == "metadata_generated"
        assert statuses[300] == "embedded"  # Unaffected
