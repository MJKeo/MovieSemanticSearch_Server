"""
Unit tests for movie_ingestion.metadata_generation.request_builder.

Covers:
  - _build_single_request structure, model, kwargs, messages, response_format
  - Branch selection (synopsis vs synthesis) in built requests
  - _chunk utility function
  - build_plot_events_requests DB query logic (requires tracker DB fixture)
"""

import sqlite3

import pytest

from movie_ingestion.metadata_generation.inputs import (
    MovieInputData,
    MetadataType,
    build_custom_id,
)
from movie_ingestion.metadata_generation.request_builder import (
    _build_single_request,
    _chunk,
    _get_pending_tmdb_ids,
    build_plot_events_requests,
    DEFAULT_BATCH_SIZE,
    _MODEL,
    _MODEL_KWARGS,
)
from movie_ingestion.metadata_generation.prompts.plot_events import (
    SYSTEM_PROMPT_SYNOPSIS,
    SYSTEM_PROMPT_SYNTHESIS,
)
from movie_ingestion.metadata_generation.generators.plot_events import (
    MIN_SYNOPSIS_CHARS,
    GENERATION_TYPE,
)
from movie_ingestion.tracker import _SCHEMA_SQL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_movie(**overrides) -> MovieInputData:
    """Build a MovieInputData with sensible defaults and optional overrides."""
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2020,
        overview="A great test movie about testing.",
        genres=["Drama"],
        plot_synopses=[],
        plot_summaries=["Summary one.", "Summary two."],
        plot_keywords=["keyword1"],
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


@pytest.fixture()
def tracker_db(tmp_path) -> sqlite3.Connection:
    """In-memory-like tracker DB at a tmp_path for request builder tests."""
    db_path = tmp_path / "tracker.db"
    db = sqlite3.connect(str(db_path))
    db.executescript(_SCHEMA_SQL)
    db.commit()
    return db


# ---------------------------------------------------------------------------
# Tests: _build_single_request — structure
# ---------------------------------------------------------------------------


class TestBuildSingleRequestStructure:
    """Tests for the structure of a single Batch API request dict."""

    def test_request_has_required_keys(self) -> None:
        """Returned dict has custom_id, method, url, and body keys."""
        movie = _make_movie()
        request = _build_single_request(movie)
        assert "custom_id" in request
        assert "method" in request
        assert "url" in request
        assert "body" in request

    def test_method_is_post(self) -> None:
        """Request method is POST."""
        movie = _make_movie()
        request = _build_single_request(movie)
        assert request["method"] == "POST"

    def test_url_is_chat_completions(self) -> None:
        """Request URL is /v1/chat/completions."""
        movie = _make_movie()
        request = _build_single_request(movie)
        assert request["url"] == "/v1/chat/completions"

    def test_custom_id_format(self) -> None:
        """custom_id matches build_custom_id(tmdb_id, GENERATION_TYPE)."""
        movie = _make_movie(tmdb_id=42)
        request = _build_single_request(movie)
        expected = build_custom_id(42, GENERATION_TYPE)
        assert request["custom_id"] == expected


class TestBuildSingleRequestBody:
    """Tests for the body contents of a Batch API request."""

    def test_model(self) -> None:
        """body.model matches the module-level _MODEL constant."""
        movie = _make_movie()
        request = _build_single_request(movie)
        assert request["body"]["model"] == _MODEL

    def test_model_kwargs(self) -> None:
        """body contains reasoning_effort and verbosity from _MODEL_KWARGS."""
        movie = _make_movie()
        request = _build_single_request(movie)
        body = request["body"]
        for key, value in _MODEL_KWARGS.items():
            assert body[key] == value

    def test_messages_has_system_and_user(self) -> None:
        """body.messages has exactly 2 messages: system and user."""
        movie = _make_movie()
        request = _build_single_request(movie)
        messages = request["body"]["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_response_format_structure(self) -> None:
        """body.response_format has json_schema type with PlotEventsOutput schema."""
        movie = _make_movie()
        request = _build_single_request(movie)
        rf = request["body"]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "PlotEventsOutput"
        assert rf["json_schema"]["strict"] is True

    def test_synopsis_branch_system_prompt(self) -> None:
        """Movie with long synopsis produces SYSTEM_PROMPT_SYNOPSIS in system message."""
        movie = _make_movie(plot_synopses=["x" * MIN_SYNOPSIS_CHARS])
        request = _build_single_request(movie)
        system_content = request["body"]["messages"][0]["content"]
        assert system_content == SYSTEM_PROMPT_SYNOPSIS

    def test_synthesis_branch_system_prompt(self) -> None:
        """Movie with no synopsis produces SYSTEM_PROMPT_SYNTHESIS in system message."""
        movie = _make_movie(plot_synopses=[])
        request = _build_single_request(movie)
        system_content = request["body"]["messages"][0]["content"]
        assert system_content == SYSTEM_PROMPT_SYNTHESIS


# ---------------------------------------------------------------------------
# Tests: _chunk utility
# ---------------------------------------------------------------------------


class TestChunk:
    """Tests for the _chunk list-splitting utility."""

    def test_chunk_basic(self) -> None:
        """Basic chunking of 5 items into chunks of 2."""
        assert _chunk([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]

    def test_chunk_exact_multiple(self) -> None:
        """Items evenly divisible by chunk size."""
        assert _chunk([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_chunk_single_item(self) -> None:
        """Single item in a larger chunk."""
        assert _chunk([1], 5) == [[1]]

    def test_chunk_empty(self) -> None:
        """Empty list returns empty list of chunks."""
        assert _chunk([], 5) == []


# ---------------------------------------------------------------------------
# Tests: _get_pending_tmdb_ids — DB query logic
# ---------------------------------------------------------------------------


class TestGetPendingTmdbIds:
    """Tests for _get_pending_tmdb_ids DB query."""

    def test_returns_eligible_movies_without_results(self, tracker_db, tmp_path) -> None:
        """Movies with eligible=1 and no result are returned."""
        db_path = tmp_path / "tracker.db"
        tracker_db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (1, 1)"
        )
        tracker_db.commit()
        tracker_db.close()

        result = _get_pending_tmdb_ids(db_path)
        assert 1 in result

    def test_excludes_movies_with_results(self, tracker_db, tmp_path) -> None:
        """Movies that already have plot_events results are excluded."""
        db_path = tmp_path / "tracker.db"
        tracker_db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events, plot_events) VALUES (1, 1, '{}')"
        )
        tracker_db.commit()
        tracker_db.close()

        result = _get_pending_tmdb_ids(db_path)
        assert 1 not in result

    def test_excludes_movies_with_active_batch(self, tracker_db, tmp_path) -> None:
        """Movies with an active batch_id are excluded."""
        db_path = tmp_path / "tracker.db"
        tracker_db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (1, 1)"
        )
        tracker_db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (1, 'batch_abc')"
        )
        tracker_db.commit()
        tracker_db.close()

        result = _get_pending_tmdb_ids(db_path)
        assert 1 not in result

    def test_includes_movies_without_batch_id(self, tracker_db, tmp_path) -> None:
        """Movies with NULL batch_id are included."""
        db_path = tmp_path / "tracker.db"
        tracker_db.execute(
            "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (1, 1)"
        )
        tracker_db.execute(
            "INSERT INTO metadata_batch_ids (tmdb_id, plot_events_batch_id) VALUES (1, NULL)"
        )
        tracker_db.commit()
        tracker_db.close()

        result = _get_pending_tmdb_ids(db_path)
        assert 1 in result

    def test_empty_db_returns_empty(self, tracker_db, tmp_path) -> None:
        """No eligible movies returns empty list."""
        db_path = tmp_path / "tracker.db"
        tracker_db.close()

        result = _get_pending_tmdb_ids(db_path)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: build_plot_events_requests — integration
# ---------------------------------------------------------------------------


class TestBuildPlotEventsRequests:
    """Tests for build_plot_events_requests end-to-end with a real DB."""

    def test_empty_when_no_pending(self, tracker_db, tmp_path) -> None:
        """Returns empty list when no eligible movies exist."""
        db_path = tmp_path / "tracker.db"
        tracker_db.close()

        result = build_plot_events_requests(db_path)
        assert result == []

    def test_respects_batch_size(self, tracker_db, tmp_path) -> None:
        """With 3 eligible movies and batch_size=2, returns 2 batches (2+1)."""
        db_path = tmp_path / "tracker.db"

        # Insert tmdb_data and imdb_data rows needed by load_movie_input_data
        for tid in (1, 2, 3):
            tracker_db.execute(
                "INSERT INTO generated_metadata (tmdb_id, eligible_for_plot_events) VALUES (?, 1)",
                (tid,),
            )
            tracker_db.execute(
                "INSERT INTO tmdb_data (tmdb_id, title, release_date) VALUES (?, ?, ?)",
                (tid, f"Movie {tid}", "2020-01-01"),
            )
            tracker_db.execute(
                "INSERT INTO imdb_data (tmdb_id, overview) VALUES (?, ?)",
                (tid, f"Overview for movie {tid}"),
            )
        tracker_db.commit()
        tracker_db.close()

        result = build_plot_events_requests(db_path, batch_size=2)
        assert len(result) == 2
        assert len(result[0]) == 2
        assert len(result[1]) == 1
