"""
Unit tests for movie_ingestion.metadata_generation.wave1_runner.

Covers:
  - init_wave1_table: creation and idempotency
  - generate_and_store_plot_events: idempotent re-runs, JSON storage,
    LLM failure handling, skip ineligible, empty input
  - generate_and_store_reception: same pattern as plot_events
  - get_wave1_results: deserialization, NULL handling, missing movies,
    empty input

All LLM calls are mocked — no real API traffic. Uses a temp SQLite DB
per test to avoid cross-test contamination.
"""

import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import (
    PlotEventsOutput,
    ReceptionOutput,
)
from movie_ingestion.metadata_generation.wave1_runner import (
    init_wave1_table,
    generate_and_store_plot_events,
    generate_and_store_reception,
    get_wave1_results,
    _open_connection,
)
from implementation.llms.vector_metadata_generation_methods import TokenUsage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLOT_EVENTS_PATCH = "movie_ingestion.metadata_generation.wave1_runner.generate_plot_events"
_RECEPTION_PATCH = "movie_ingestion.metadata_generation.wave1_runner.generate_reception"


def _make_movie(**overrides) -> MovieInputData:
    """Build a MovieInputData with fields sufficient for eligibility."""
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        overview="A long enough overview for plot events to pass the eligibility check.",
        featured_reviews=[{"summary": "Great", "text": "A wonderful and detailed movie review."}],
        reception_summary="Widely acclaimed.",
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


def _make_plot_events_output() -> PlotEventsOutput:
    return PlotEventsOutput(
        plot_summary="A detailed plot summary.",
        setting="Modern day New York",
        major_characters=[],
    )


def _make_reception_output() -> ReceptionOutput:
    return ReceptionOutput(
        new_reception_summary="Widely acclaimed.",
        praise_attributes=["groundbreaking"],
        complaint_attributes=[],
        review_insights_brief="Critics praised the visual effects.",
    )


@pytest.fixture
def tmp_db_path(tmp_path):
    """Provide a fresh temp DB path for each test."""
    return tmp_path / "test_tracker.db"


# ---------------------------------------------------------------------------
# init_wave1_table
# ---------------------------------------------------------------------------

class TestInitWave1Table:
    def test_creates_table(self, tmp_db_path):
        """init_wave1_table creates the wave1_results table."""
        conn = _open_connection(tmp_db_path)
        init_wave1_table(conn)

        # Verify table exists by querying it
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='wave1_results'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_idempotent(self, tmp_db_path):
        """Calling init_wave1_table twice does not raise."""
        conn = _open_connection(tmp_db_path)
        init_wave1_table(conn)
        init_wave1_table(conn)  # Should not raise
        conn.close()


# ---------------------------------------------------------------------------
# generate_and_store_plot_events
# ---------------------------------------------------------------------------

class TestGenerateAndStorePlotEvents:
    async def test_stores_json_result_in_db(self, tmp_db_path):
        """Successful generation stores JSON in the plot_events column."""
        pe_output = _make_plot_events_output()
        token_usage = TokenUsage(100, 50, "test-model")
        mock_fn = AsyncMock(return_value=(pe_output, token_usage))
        movie = _make_movie(tmdb_id=1)

        with patch(_PLOT_EVENTS_PATCH, mock_fn):
            await generate_and_store_plot_events({1: movie}, db_path=tmp_db_path)

        # Verify stored in DB
        conn = _open_connection(tmp_db_path)
        row = conn.execute("SELECT plot_events FROM wave1_results WHERE tmdb_id=1").fetchone()
        conn.close()

        assert row is not None
        stored_data = json.loads(row["plot_events"])
        assert stored_data["plot_summary"] == "A detailed plot summary."

    async def test_skips_existing_movies(self, tmp_db_path):
        """Movies already in DB with non-NULL plot_events are skipped."""
        # Pre-populate the DB
        conn = _open_connection(tmp_db_path)
        init_wave1_table(conn)
        conn.execute(
            "INSERT INTO wave1_results (tmdb_id, plot_events) VALUES (?, ?)",
            (1, '{"plot_summary":"existing","setting":"here","major_characters":[]}'),
        )
        conn.commit()
        conn.close()

        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), TokenUsage(100, 50, "m")))
        movie = _make_movie(tmdb_id=1)

        with patch(_PLOT_EVENTS_PATCH, mock_fn):
            await generate_and_store_plot_events({1: movie}, db_path=tmp_db_path)

        # LLM should not have been called
        mock_fn.assert_not_called()

    async def test_handles_llm_failure_gracefully(self, tmp_db_path):
        """Failed movies are not stored in the DB."""
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie(tmdb_id=1)

        with patch(_PLOT_EVENTS_PATCH, mock_fn):
            await generate_and_store_plot_events({1: movie}, db_path=tmp_db_path)

        # No row should be stored for the failed movie
        conn = _open_connection(tmp_db_path)
        row = conn.execute("SELECT plot_events FROM wave1_results WHERE tmdb_id=1").fetchone()
        conn.close()
        assert row is None

    async def test_skips_ineligible_movies(self, tmp_db_path):
        """Movies that fail check_plot_events are skipped without LLM call."""
        mock_fn = AsyncMock(return_value=(_make_plot_events_output(), TokenUsage(100, 50, "m")))
        # Movie with no text data — ineligible
        movie = _make_movie(
            tmdb_id=1, overview="", plot_synopses=[], plot_summaries=[],
        )

        with patch(_PLOT_EVENTS_PATCH, mock_fn):
            await generate_and_store_plot_events({1: movie}, db_path=tmp_db_path)

        mock_fn.assert_not_called()

    async def test_empty_input_returns_early(self, tmp_db_path):
        """Empty movie_inputs dict returns early without DB writes."""
        mock_fn = AsyncMock()

        with patch(_PLOT_EVENTS_PATCH, mock_fn):
            await generate_and_store_plot_events({}, db_path=tmp_db_path)

        mock_fn.assert_not_called()


# ---------------------------------------------------------------------------
# generate_and_store_reception
# ---------------------------------------------------------------------------

class TestGenerateAndStoreReception:
    async def test_stores_json_result_in_db(self, tmp_db_path):
        """Successful generation stores JSON in the reception column."""
        rec_output = _make_reception_output()
        token_usage = TokenUsage(100, 50, "test-model")
        mock_fn = AsyncMock(return_value=(rec_output, token_usage))
        movie = _make_movie(tmdb_id=1)

        with patch(_RECEPTION_PATCH, mock_fn):
            await generate_and_store_reception({1: movie}, db_path=tmp_db_path)

        conn = _open_connection(tmp_db_path)
        row = conn.execute("SELECT reception FROM wave1_results WHERE tmdb_id=1").fetchone()
        conn.close()

        assert row is not None
        stored_data = json.loads(row["reception"])
        assert stored_data["new_reception_summary"] == "Widely acclaimed."

    async def test_skips_existing_movies(self, tmp_db_path):
        """Movies already in DB with non-NULL reception are skipped."""
        conn = _open_connection(tmp_db_path)
        init_wave1_table(conn)
        conn.execute(
            "INSERT INTO wave1_results (tmdb_id, reception) VALUES (?, ?)",
            (1, '{"new_reception_summary":"existing","praise_attributes":[],"complaint_attributes":[],"review_insights_brief":"existing"}'),
        )
        conn.commit()
        conn.close()

        mock_fn = AsyncMock(return_value=(_make_reception_output(), TokenUsage(100, 50, "m")))
        movie = _make_movie(tmdb_id=1)

        with patch(_RECEPTION_PATCH, mock_fn):
            await generate_and_store_reception({1: movie}, db_path=tmp_db_path)

        mock_fn.assert_not_called()

    async def test_handles_llm_failure_gracefully(self, tmp_db_path):
        """Failed movies are not stored in the DB."""
        mock_fn = AsyncMock(side_effect=ValueError("API error"))
        movie = _make_movie(tmdb_id=1)

        with patch(_RECEPTION_PATCH, mock_fn):
            await generate_and_store_reception({1: movie}, db_path=tmp_db_path)

        conn = _open_connection(tmp_db_path)
        row = conn.execute("SELECT reception FROM wave1_results WHERE tmdb_id=1").fetchone()
        conn.close()
        assert row is None

    async def test_skips_ineligible_movies(self, tmp_db_path):
        """Movies that fail check_reception are skipped without LLM call."""
        mock_fn = AsyncMock(return_value=(_make_reception_output(), TokenUsage(100, 50, "m")))
        # Movie with no reception data — ineligible
        movie = _make_movie(
            tmdb_id=1, reception_summary=None,
            audience_reception_attributes=[], featured_reviews=[],
        )

        with patch(_RECEPTION_PATCH, mock_fn):
            await generate_and_store_reception({1: movie}, db_path=tmp_db_path)

        mock_fn.assert_not_called()

    async def test_empty_input_returns_early(self, tmp_db_path):
        """Empty movie_inputs dict returns early without DB writes."""
        mock_fn = AsyncMock()

        with patch(_RECEPTION_PATCH, mock_fn):
            await generate_and_store_reception({}, db_path=tmp_db_path)

        mock_fn.assert_not_called()


# ---------------------------------------------------------------------------
# get_wave1_results
# ---------------------------------------------------------------------------

class TestGetWave1Results:
    def test_returns_deserialized_outputs(self, tmp_db_path):
        """Stored JSON is deserialized back to Pydantic models."""
        pe_output = _make_plot_events_output()
        rec_output = _make_reception_output()

        conn = _open_connection(tmp_db_path)
        init_wave1_table(conn)
        conn.execute(
            "INSERT INTO wave1_results (tmdb_id, plot_events, reception) VALUES (?, ?, ?)",
            (1, json.dumps(pe_output.model_dump()), json.dumps(rec_output.model_dump())),
        )
        conn.commit()
        conn.close()

        results = get_wave1_results([1], db_path=tmp_db_path)
        assert 1 in results
        assert isinstance(results[1]["plot_events"], PlotEventsOutput)
        assert isinstance(results[1]["reception"], ReceptionOutput)
        assert results[1]["plot_events"].plot_summary == "A detailed plot summary."

    def test_handles_null_columns(self, tmp_db_path):
        """NULL columns are returned as None in the result dict."""
        conn = _open_connection(tmp_db_path)
        init_wave1_table(conn)
        conn.execute(
            "INSERT INTO wave1_results (tmdb_id, plot_events) VALUES (?, ?)",
            (1, json.dumps(_make_plot_events_output().model_dump())),
        )
        conn.commit()
        conn.close()

        results = get_wave1_results([1], db_path=tmp_db_path)
        assert results[1]["plot_events"] is not None
        assert results[1]["reception"] is None

    def test_omits_movies_not_in_table(self, tmp_db_path):
        """Movies not present in the table are omitted from results."""
        conn = _open_connection(tmp_db_path)
        init_wave1_table(conn)
        conn.commit()
        conn.close()

        results = get_wave1_results([999], db_path=tmp_db_path)
        assert 999 not in results

    def test_empty_tmdb_ids_returns_empty_dict(self, tmp_db_path):
        """Empty tmdb_ids list returns an empty dict."""
        conn = _open_connection(tmp_db_path)
        init_wave1_table(conn)
        conn.commit()
        conn.close()

        results = get_wave1_results([], db_path=tmp_db_path)
        assert results == {}
