"""
Unit tests for movie_ingestion.metadata_generation.evaluations.shared.

Covers:
  - EvaluationCandidate dataclass (frozen, defaults)
  - get_eval_connection (directory creation, WAL mode, row_factory, str/Path)
  - create_candidates_table (idempotent)
  - store_candidate (insert, upsert, serialization)
  - compute_score_summary (aggregation, filtering, sorting, edge cases)
  - EVALUATION_TEST_SET_TMDB_IDS (count, uniqueness)
  - load_movie_input_data (missing tracker)

All DB tests use tmp_path for ephemeral SQLite databases — no real eval DB.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from implementation.llms.generic_methods import LLMProvider
from movie_ingestion.metadata_generation.evaluations.shared import (
    EVALUATION_TEST_SET_TMDB_IDS,
    EvaluationCandidate,
    compute_score_summary,
    create_candidates_table,
    get_eval_connection,
    load_movie_input_data,
    store_candidate,
)
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummySchema(BaseModel):
    """Minimal Pydantic model used as response_format in tests."""
    value: str


def _make_candidate(**overrides) -> EvaluationCandidate:
    """Build an EvaluationCandidate with sensible defaults and optional overrides."""
    defaults = dict(
        candidate_id="test-candidate",
        provider=LLMProvider.OPENAI,
        model="gpt-5-mini",
        system_prompt="You are a helpful assistant.",
        response_format=_DummySchema,
    )
    defaults.update(overrides)
    return EvaluationCandidate(**defaults)


def _create_eval_table_with_scores(conn: sqlite3.Connection) -> None:
    """Create a minimal evaluation results table for testing compute_score_summary."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS test_evaluations (
            movie_id       INTEGER NOT NULL,
            candidate_id   TEXT NOT NULL,
            dim_a_score    INTEGER,
            dim_b_score    INTEGER,
            PRIMARY KEY (movie_id, candidate_id)
        )
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Tests: EvaluationCandidate
# ---------------------------------------------------------------------------


class TestEvaluationCandidate:
    def test_evaluation_candidate_is_frozen_dataclass(self) -> None:
        """Instances are immutable (frozen=True)."""
        candidate = _make_candidate()
        # Frozen: mutation should raise
        with pytest.raises(AttributeError):
            candidate.candidate_id = "mutated"

    def test_evaluation_candidate_default_kwargs(self) -> None:
        """kwargs defaults to empty dict when not provided."""
        candidate = _make_candidate()
        assert candidate.kwargs == {}


# ---------------------------------------------------------------------------
# Tests: get_eval_connection
# ---------------------------------------------------------------------------


class TestGetEvalConnection:
    def test_get_eval_connection_creates_directory(self, tmp_path: Path) -> None:
        """Creates parent directories if they don't exist."""
        nested_path = tmp_path / "a" / "b" / "eval.db"
        conn = get_eval_connection(nested_path)
        assert nested_path.exists()
        conn.close()

    def test_get_eval_connection_sets_wal_mode(self, tmp_path: Path) -> None:
        """PRAGMA journal_mode=WAL is set on connection."""
        db_path = tmp_path / "eval.db"
        conn = get_eval_connection(db_path)
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_get_eval_connection_sets_row_factory(self, tmp_path: Path) -> None:
        """row_factory is sqlite3.Row."""
        db_path = tmp_path / "eval.db"
        conn = get_eval_connection(db_path)
        assert conn.row_factory is sqlite3.Row
        conn.close()

    def test_get_eval_connection_accepts_str_or_path(self, tmp_path: Path) -> None:
        """Both str and Path arguments work."""
        db_path = tmp_path / "eval.db"
        # Path
        conn1 = get_eval_connection(db_path)
        conn1.close()
        # str
        conn2 = get_eval_connection(str(db_path))
        conn2.close()
        assert db_path.exists()


# ---------------------------------------------------------------------------
# Tests: create_candidates_table
# ---------------------------------------------------------------------------


class TestCreateCandidatesTable:
    def test_create_candidates_table_idempotent(self, tmp_path: Path) -> None:
        """Calling twice doesn't raise."""
        conn = get_eval_connection(tmp_path / "eval.db")
        create_candidates_table(conn)
        create_candidates_table(conn)  # Should not raise
        conn.close()


# ---------------------------------------------------------------------------
# Tests: store_candidate
# ---------------------------------------------------------------------------


class TestStoreCandidate:
    def test_store_candidate_inserts_record(self, tmp_path: Path) -> None:
        """Store a candidate and verify it can be read back with correct fields."""
        conn = get_eval_connection(tmp_path / "eval.db")
        create_candidates_table(conn)
        candidate = _make_candidate(candidate_id="my-candidate", model="gpt-5-mini")
        store_candidate(conn, candidate, "plot_events")

        row = conn.execute(
            "SELECT * FROM candidates WHERE candidate_id = ? AND metadata_type = ?",
            ("my-candidate", "plot_events"),
        ).fetchone()

        assert row is not None
        assert row["candidate_id"] == "my-candidate"
        assert row["model"] == "gpt-5-mini"
        assert row["metadata_type"] == "plot_events"
        conn.close()

    def test_store_candidate_upsert_replaces(self, tmp_path: Path) -> None:
        """Store same candidate_id twice with different model, verify latest wins."""
        conn = get_eval_connection(tmp_path / "eval.db")
        create_candidates_table(conn)

        candidate_v1 = _make_candidate(candidate_id="upsert-test", model="gpt-5-mini")
        store_candidate(conn, candidate_v1, "plot_events")

        candidate_v2 = _make_candidate(candidate_id="upsert-test", model="gpt-5-nano")
        store_candidate(conn, candidate_v2, "plot_events")

        row = conn.execute(
            "SELECT model FROM candidates WHERE candidate_id = ? AND metadata_type = ?",
            ("upsert-test", "plot_events"),
        ).fetchone()
        assert row["model"] == "gpt-5-nano"
        conn.close()

    def test_store_candidate_serializes_kwargs_as_json(self, tmp_path: Path) -> None:
        """kwargs dict is stored as JSON string."""
        conn = get_eval_connection(tmp_path / "eval.db")
        create_candidates_table(conn)
        candidate = _make_candidate(kwargs={"temperature": 0.5, "max_tokens": 2048})
        store_candidate(conn, candidate, "plot_events")

        row = conn.execute(
            "SELECT parameters FROM candidates WHERE candidate_id = ?",
            (candidate.candidate_id,),
        ).fetchone()
        parsed = json.loads(row["parameters"])
        assert parsed == {"temperature": 0.5, "max_tokens": 2048}
        conn.close()

    def test_store_candidate_records_provider_value(self, tmp_path: Path) -> None:
        """provider.value (string) is stored, not the enum."""
        conn = get_eval_connection(tmp_path / "eval.db")
        create_candidates_table(conn)
        candidate = _make_candidate(provider=LLMProvider.GEMINI)
        store_candidate(conn, candidate, "plot_events")

        row = conn.execute(
            "SELECT provider FROM candidates WHERE candidate_id = ?",
            (candidate.candidate_id,),
        ).fetchone()
        assert row["provider"] == "gemini"
        conn.close()

    def test_store_candidate_empty_kwargs(self, tmp_path: Path) -> None:
        """Empty kwargs dict is serialized correctly as '{}'."""
        conn = get_eval_connection(tmp_path / "eval.db")
        create_candidates_table(conn)
        candidate = _make_candidate(kwargs={})
        store_candidate(conn, candidate, "plot_events")

        row = conn.execute(
            "SELECT parameters FROM candidates WHERE candidate_id = ?",
            (candidate.candidate_id,),
        ).fetchone()
        assert json.loads(row["parameters"]) == {}
        conn.close()


# ---------------------------------------------------------------------------
# Tests: compute_score_summary
# ---------------------------------------------------------------------------


class TestComputeScoreSummary:
    def _setup_table_with_data(self, tmp_path: Path) -> sqlite3.Connection:
        """Create a test DB with evaluation data and return the connection."""
        conn = get_eval_connection(tmp_path / "eval.db")
        _create_eval_table_with_scores(conn)

        # Insert test data: 2 candidates, 3 movies each
        test_data = [
            # candidate A: dim_a scores [2, 3, 4], dim_b scores [3, 3, 3]
            (1, "candidate-a", 2, 3),
            (2, "candidate-a", 3, 3),
            (3, "candidate-a", 4, 3),
            # candidate B: dim_a scores [1, 1, 1], dim_b scores [4, 4, 4]
            (1, "candidate-b", 1, 4),
            (2, "candidate-b", 1, 4),
            (3, "candidate-b", 1, 4),
        ]
        conn.executemany(
            "INSERT INTO test_evaluations VALUES (?, ?, ?, ?)",
            test_data,
        )
        conn.commit()
        return conn

    def test_compute_score_summary_basic(self, tmp_path: Path) -> None:
        """Mean and median computed correctly per candidate."""
        conn = self._setup_table_with_data(tmp_path)
        result = compute_score_summary(
            conn, "test_evaluations", ["dim_a_score", "dim_b_score"],
        )

        # candidate-a: dim_a mean=3.0, dim_b mean=3.0
        assert result.loc["candidate-a", "dim_a_mean"] == 3.0
        assert result.loc["candidate-a", "dim_b_mean"] == 3.0
        # candidate-a: dim_a median=3.0, dim_b median=3.0
        assert result.loc["candidate-a", "dim_a_median"] == 3.0
        assert result.loc["candidate-a", "dim_b_median"] == 3.0
        # candidate-b: dim_a mean=1.0, dim_b mean=4.0
        assert result.loc["candidate-b", "dim_a_mean"] == 1.0
        assert result.loc["candidate-b", "dim_b_mean"] == 4.0
        conn.close()

    def test_compute_score_summary_overall_mean(self, tmp_path: Path) -> None:
        """overall_mean is the mean of all dimension means."""
        conn = self._setup_table_with_data(tmp_path)
        result = compute_score_summary(
            conn, "test_evaluations", ["dim_a_score", "dim_b_score"],
        )

        # candidate-a: mean of (3.0, 3.0) = 3.0
        assert result.loc["candidate-a", "overall_mean"] == 3.0
        # candidate-b: mean of (1.0, 4.0) = 2.5
        assert result.loc["candidate-b", "overall_mean"] == 2.5
        conn.close()

    def test_compute_score_summary_filters_by_candidate_ids(self, tmp_path: Path) -> None:
        """candidate_ids parameter filters correctly."""
        conn = self._setup_table_with_data(tmp_path)
        result = compute_score_summary(
            conn, "test_evaluations", ["dim_a_score", "dim_b_score"],
            candidate_ids=["candidate-a"],
        )

        assert "candidate-a" in result.index
        assert "candidate-b" not in result.index
        conn.close()

    def test_compute_score_summary_returns_empty_df_when_no_data(self, tmp_path: Path) -> None:
        """Empty DataFrame returned for empty table."""
        conn = get_eval_connection(tmp_path / "eval.db")
        _create_eval_table_with_scores(conn)

        result = compute_score_summary(
            conn, "test_evaluations", ["dim_a_score", "dim_b_score"],
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty
        conn.close()

    def test_compute_score_summary_sorted_by_overall_mean_desc(self, tmp_path: Path) -> None:
        """Results are sorted descending by overall_mean."""
        conn = self._setup_table_with_data(tmp_path)
        result = compute_score_summary(
            conn, "test_evaluations", ["dim_a_score", "dim_b_score"],
        )

        # candidate-a has overall_mean=3.0, candidate-b has 2.5
        # So candidate-a should come first
        assert result.index[0] == "candidate-a"
        assert result.index[1] == "candidate-b"
        conn.close()

    def test_compute_score_summary_identical_scores(self, tmp_path: Path) -> None:
        """When all scores are identical, mean equals median."""
        conn = get_eval_connection(tmp_path / "eval.db")
        _create_eval_table_with_scores(conn)
        conn.executemany(
            "INSERT INTO test_evaluations VALUES (?, ?, ?, ?)",
            [(1, "same", 3, 3), (2, "same", 3, 3), (3, "same", 3, 3)],
        )
        conn.commit()

        result = compute_score_summary(
            conn, "test_evaluations", ["dim_a_score", "dim_b_score"],
        )
        assert result.loc["same", "dim_a_mean"] == result.loc["same", "dim_a_median"]
        assert result.loc["same", "dim_b_mean"] == result.loc["same", "dim_b_median"]
        conn.close()


# ---------------------------------------------------------------------------
# Tests: load_movie_input_data
# ---------------------------------------------------------------------------


class TestLoadMovieInputData:
    def test_load_movie_input_data_raises_on_missing_tracker(self, tmp_path: Path) -> None:
        """FileNotFoundError when tracker DB doesn't exist."""
        nonexistent = tmp_path / "nonexistent.db"
        with pytest.raises(FileNotFoundError):
            load_movie_input_data([1, 2, 3], tracker_db_path=nonexistent)


# ---------------------------------------------------------------------------
# Tests: EVALUATION_TEST_SET_TMDB_IDS
# ---------------------------------------------------------------------------


class TestEvaluationTestSet:
    def test_evaluation_test_set_has_70_movies(self) -> None:
        """Combined test set contains exactly 70 movies."""
        assert len(EVALUATION_TEST_SET_TMDB_IDS) == 70

    def test_evaluation_test_set_no_duplicates(self) -> None:
        """No duplicate TMDB IDs in the combined list."""
        assert len(set(EVALUATION_TEST_SET_TMDB_IDS)) == len(EVALUATION_TEST_SET_TMDB_IDS)
