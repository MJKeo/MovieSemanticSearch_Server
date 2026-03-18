"""
Unit tests for movie_ingestion.metadata_generation.evaluations.plot_events.

Covers:
  - create_plot_events_tables (idempotent)
  - _serialize_output / _deserialize_output round-trip
  - _format_characters_for_prompt (non-empty and empty)
  - _build_judge_user_prompt (sections, content)
  - PlotEventsJudgeOutput validation (score range, rejection)
  - PLOT_EVENTS_CANDIDATES (unique IDs, response_format)
  - SCORE_COLUMNS matches evaluation table DDL

All DB tests use tmp_path for ephemeral SQLite databases — no real eval DB.
"""

import json
import sqlite3
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel, ValidationError

from implementation.llms.generic_methods import LLMProvider
from movie_ingestion.metadata_generation.evaluations.plot_events import (
    PLOT_EVENTS_CANDIDATES,
    SCORE_COLUMNS,
    PlotEventsJudgeOutput,
    _build_judge_user_prompt,
    _deserialize_output,
    _format_characters_for_prompt,
    _serialize_output,
    create_plot_events_tables,
    generate_reference_responses,
    run_evaluation,
)
from movie_ingestion.metadata_generation.evaluations.shared import (
    EvaluationCandidate,
    get_eval_connection,
)
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.schemas import (
    MajorCharacter,
    PlotEventsOutput,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_character(**overrides) -> MajorCharacter:
    """Build a MajorCharacter with sensible defaults and optional overrides."""
    defaults = dict(
        name="Neo",
        description="a computer hacker who discovers the truth about reality",
        role="protagonist",
        primary_motivations="To free humanity from the Matrix.",
    )
    defaults.update(overrides)
    return MajorCharacter(**defaults)


def _make_plot_events_output(**overrides) -> PlotEventsOutput:
    """Build a PlotEventsOutput with sensible defaults and optional overrides."""
    defaults = dict(
        plot_summary="Neo discovers the Matrix is a simulation and fights Agent Smith.",
        setting="Near-future dystopia, simulated reality",
        major_characters=[_make_character()],
    )
    defaults.update(overrides)
    return PlotEventsOutput(**defaults)


def _make_judge_output() -> PlotEventsJudgeOutput:
    """Build a valid PlotEventsJudgeOutput with high scores for use in mocks."""
    return PlotEventsJudgeOutput(
        groundedness_reasoning="All details grounded in inputs.",
        plot_summary_reasoning="Comprehensive chronological coverage.",
        character_quality_reasoning="Essential characters correctly identified.",
        setting_reasoning="Specific location and time period included.",
        groundedness_score=4,
        plot_summary_score=4,
        character_quality_score=3,
        setting_score=4,
    )


def _make_eval_movie(**overrides) -> MovieInputData:
    """Build a MovieInputData suitable for evaluation pipeline tests."""
    defaults = dict(
        tmdb_id=12345,
        title="Test Movie",
        release_year=2000,
        overview="A test movie with sufficient overview text for generation.",
    )
    defaults.update(overrides)
    return MovieInputData(**defaults)


# ---------------------------------------------------------------------------
# Tests: create_plot_events_tables
# ---------------------------------------------------------------------------


class TestCreatePlotEventsTables:
    def test_create_plot_events_tables_idempotent(self, tmp_path: Path) -> None:
        """Calling twice doesn't raise, all 3 tables created."""
        conn = get_eval_connection(tmp_path / "eval.db")
        create_plot_events_tables(conn)
        create_plot_events_tables(conn)  # Should not raise

        # Verify all three tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {row["name"] for row in tables}
        assert "plot_events_references" in table_names
        assert "plot_events_candidate_outputs" in table_names
        assert "plot_events_evaluations" in table_names
        conn.close()


# ---------------------------------------------------------------------------
# Tests: _serialize_output / _deserialize_output
# ---------------------------------------------------------------------------


class TestSerializeDeserialize:
    def test_serialize_output_round_trip(self) -> None:
        """Serialize a PlotEventsOutput, then deserialize it, verify equality."""
        original = _make_plot_events_output()
        plot_summary, setting, chars_json = _serialize_output(original)

        # Simulate a DB row using a dict-like object
        row = {
            "plot_summary": plot_summary,
            "setting": setting,
            "major_characters": chars_json,
        }
        reconstructed = _deserialize_output(row)

        assert reconstructed.plot_summary == original.plot_summary
        assert reconstructed.setting == original.setting
        assert len(reconstructed.major_characters) == len(original.major_characters)
        assert reconstructed.major_characters[0].name == original.major_characters[0].name

    def test_serialize_output_handles_characters(self) -> None:
        """major_characters are serialized as JSON array of dicts."""
        output = _make_plot_events_output(major_characters=[
            _make_character(name="Neo"),
            _make_character(name="Trinity", role="love interest"),
        ])
        _, _, chars_json = _serialize_output(output)
        parsed = json.loads(chars_json)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["name"] == "Neo"
        assert parsed[1]["name"] == "Trinity"

    def test_deserialize_output_reconstructs_major_characters(self) -> None:
        """MajorCharacter objects are reconstructed from JSON."""
        chars_json = json.dumps([{
            "name": "Morpheus",
            "description": "captain of the Nebuchadnezzar",
            "role": "mentor",
            "primary_motivations": "To find The One.",
        }])
        row = {
            "plot_summary": "Test summary.",
            "setting": "Dystopian future",
            "major_characters": chars_json,
        }
        result = _deserialize_output(row)
        assert len(result.major_characters) == 1
        assert isinstance(result.major_characters[0], MajorCharacter)
        assert result.major_characters[0].name == "Morpheus"

    def test_deserialize_output_empty_characters(self) -> None:
        """Empty major_characters JSON '[]' returns empty list."""
        row = {
            "plot_summary": "Test summary.",
            "setting": "A setting.",
            "major_characters": "[]",
        }
        result = _deserialize_output(row)
        assert result.major_characters == []


# ---------------------------------------------------------------------------
# Tests: _format_characters_for_prompt
# ---------------------------------------------------------------------------


class TestFormatCharactersForPrompt:
    def test_format_characters_for_prompt_non_empty(self) -> None:
        """Formatted string includes name, role, description, motivations."""
        output = _make_plot_events_output()
        result = _format_characters_for_prompt(output)
        assert "Neo" in result
        assert "protagonist" in result
        assert "computer hacker" in result
        assert "Motivations:" in result

    def test_format_characters_for_prompt_empty(self) -> None:
        """Returns '  (none)' for output with no characters."""
        output = _make_plot_events_output(major_characters=[])
        result = _format_characters_for_prompt(output)
        assert result == "  (none)"


# ---------------------------------------------------------------------------
# Tests: _build_judge_user_prompt
# ---------------------------------------------------------------------------


class TestBuildJudgeUserPrompt:
    def test_build_judge_user_prompt_contains_all_sections(self) -> None:
        """Output contains 'GENERATION PROMPT', 'REFERENCE RESPONSE', 'CANDIDATE OUTPUT' sections."""
        reference = _make_plot_events_output()
        candidate = _make_plot_events_output(plot_summary="Different summary.")
        result = _build_judge_user_prompt("Test generation prompt", reference, candidate)

        assert "GENERATION PROMPT:" in result
        assert "REFERENCE RESPONSE:" in result
        assert "CANDIDATE OUTPUT:" in result

    def test_build_judge_user_prompt_includes_generation_prompt(self) -> None:
        """The generation_user_prompt text appears in the output."""
        reference = _make_plot_events_output()
        candidate = _make_plot_events_output()
        gen_prompt = "title: The Matrix (1999)\noverview: A computer hacker learns about reality."
        result = _build_judge_user_prompt(gen_prompt, reference, candidate)

        assert gen_prompt in result


# ---------------------------------------------------------------------------
# Tests: PlotEventsJudgeOutput
# ---------------------------------------------------------------------------


class TestPlotEventsJudgeOutput:
    def test_plot_events_judge_output_score_range(self) -> None:
        """PlotEventsJudgeOutput only accepts scores 1-4."""
        output = PlotEventsJudgeOutput(
            groundedness_reasoning="Good.",
            plot_summary_reasoning="Solid.",
            character_quality_reasoning="Accurate.",
            setting_reasoning="Specific.",
            groundedness_score=4,
            plot_summary_score=3,
            character_quality_score=2,
            setting_score=1,
        )
        assert output.groundedness_score == 4
        assert output.setting_score == 1

    def test_plot_events_judge_output_rejects_invalid_score(self) -> None:
        """Scores of 0 or 5 are rejected by Pydantic validation."""
        with pytest.raises(ValidationError):
            PlotEventsJudgeOutput(
                groundedness_reasoning="Bad.",
                plot_summary_reasoning="Bad.",
                character_quality_reasoning="Bad.",
                setting_reasoning="Bad.",
                groundedness_score=0,
                plot_summary_score=5,
                character_quality_score=3,
                setting_score=3,
            )

    def test_plot_events_judge_output_reasoning_before_scores(self) -> None:
        """Model fields maintain declared order: reasoning fields before score fields in schema."""
        field_names = list(PlotEventsJudgeOutput.model_fields.keys())
        # All reasoning fields should come before all score fields
        reasoning_indices = [
            field_names.index("groundedness_reasoning"),
            field_names.index("plot_summary_reasoning"),
            field_names.index("character_quality_reasoning"),
            field_names.index("setting_reasoning"),
        ]
        score_indices = [
            field_names.index("groundedness_score"),
            field_names.index("plot_summary_score"),
            field_names.index("character_quality_score"),
            field_names.index("setting_score"),
        ]
        assert max(reasoning_indices) < min(score_indices)


# ---------------------------------------------------------------------------
# Tests: PLOT_EVENTS_CANDIDATES
# ---------------------------------------------------------------------------


class TestPlotEventsCandidates:
    def test_plot_events_candidates_have_unique_ids(self) -> None:
        """All candidate_ids in PLOT_EVENTS_CANDIDATES are unique."""
        ids = [c.candidate_id for c in PLOT_EVENTS_CANDIDATES]
        assert len(ids) == len(set(ids))

    def test_plot_events_candidates_all_use_plot_events_output(self) -> None:
        """All candidates have response_format=PlotEventsOutput."""
        for candidate in PLOT_EVENTS_CANDIDATES:
            assert candidate.response_format is PlotEventsOutput


# ---------------------------------------------------------------------------
# Tests: SCORE_COLUMNS
# ---------------------------------------------------------------------------


class TestScoreColumns:
    def test_score_columns_match_evaluation_table(self, tmp_path: Path) -> None:
        """SCORE_COLUMNS names match columns in the evaluations DDL."""
        conn = get_eval_connection(tmp_path / "eval.db")
        create_plot_events_tables(conn)

        # Get actual column names from the evaluations table
        cursor = conn.execute("PRAGMA table_info(plot_events_evaluations)")
        actual_columns = {row["name"] for row in cursor.fetchall()}

        for col in SCORE_COLUMNS:
            assert col in actual_columns, f"SCORE_COLUMNS entry '{col}' not found in evaluations table"
        conn.close()


# ---------------------------------------------------------------------------
# Tests: generate_reference_responses
# ---------------------------------------------------------------------------

# Patch targets for generate_reference_responses
_LLM_PATCH = "movie_ingestion.metadata_generation.evaluations.plot_events.generate_llm_response_async"
_PROMPT_PATCH = "movie_ingestion.metadata_generation.evaluations.plot_events.build_plot_events_user_prompt"
_SLEEP_PATCH = "movie_ingestion.metadata_generation.evaluations.plot_events.asyncio.sleep"
_CONN_PATCH = "movie_ingestion.metadata_generation.evaluations.plot_events.get_eval_connection"


class TestGenerateReferenceResponses:
    """Tests for generate_reference_responses (Phase 0 of the eval pipeline).

    All tests use tmp_path for ephemeral SQLite databases and mock the LLM.
    The real get_eval_connection is used via db_path= for most tests;
    a patched wrapper is used for the commit-ordering test.
    """

    async def test_all_pending_movies_processed_in_series(self, tmp_path: Path) -> None:
        """LLM is called once per pending movie and all rows are inserted."""
        # Arrange
        movie_inputs = {
            101: _make_eval_movie(tmdb_id=101, title="Movie A"),
            102: _make_eval_movie(tmdb_id=102, title="Movie B"),
            103: _make_eval_movie(tmdb_id=103, title="Movie C"),
        }
        mock_output = _make_plot_events_output()
        mock_llm = AsyncMock(return_value=(mock_output, 10, 5))

        # Act
        with patch(_LLM_PATCH, mock_llm), patch(_PROMPT_PATCH, return_value="test prompt"):
            await generate_reference_responses(movie_inputs, db_path=tmp_path / "eval.db")

        # Assert: LLM called once per movie
        assert mock_llm.call_count == 3

        # Assert: all 3 rows in the DB
        conn = get_eval_connection(tmp_path / "eval.db")
        row_count = conn.execute(
            "SELECT COUNT(*) FROM plot_events_references"
        ).fetchone()[0]
        conn.close()
        assert row_count == 3

    async def test_all_pending_movies_processed_with_zero_pending(self, tmp_path: Path) -> None:
        """When all movies already have references, LLM is never called."""
        # Pre-insert a reference so there's nothing pending
        movie = _make_eval_movie(tmdb_id=201, title="Already Done")
        conn = get_eval_connection(tmp_path / "eval.db")
        create_plot_events_tables(conn)
        output = _make_plot_events_output()
        plot_summary, setting, chars_json = _serialize_output(output)
        conn.execute(
            "INSERT INTO plot_events_references "
            "(movie_id, plot_summary, setting, major_characters, reference_model, input_tokens, output_tokens, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (201, plot_summary, setting, chars_json, "claude-opus-4-6", 10, 5, "2026-01-01T00:00:00+00:00"),
        )
        conn.commit()
        conn.close()

        mock_llm = AsyncMock()

        with patch(_LLM_PATCH, mock_llm), patch(_PROMPT_PATCH, return_value="test prompt"):
            await generate_reference_responses({201: movie}, db_path=tmp_path / "eval.db")

        mock_llm.assert_not_called()

    async def test_429_error_triggers_sleep_and_retry(self, tmp_path: Path) -> None:
        """A ValueError containing '429' causes a 60s sleep and one retry; retry success stores the row."""
        # Arrange: first call raises 429, second call succeeds
        mock_output = _make_plot_events_output()
        mock_llm = AsyncMock(
            side_effect=[ValueError("rate limited: 429 too many requests"), (mock_output, 10, 5)]
        )
        mock_sleep = AsyncMock()
        movie_inputs = {301: _make_eval_movie(tmdb_id=301)}

        # Act
        with patch(_LLM_PATCH, mock_llm), \
             patch(_PROMPT_PATCH, return_value="test prompt"), \
             patch(_SLEEP_PATCH, mock_sleep):
            await generate_reference_responses(movie_inputs, db_path=tmp_path / "eval.db")

        # Assert: sleep called with 60 seconds
        mock_sleep.assert_called_once_with(60)
        # Assert: LLM called exactly twice (original + retry)
        assert mock_llm.call_count == 2
        # Assert: row was successfully stored after retry
        conn = get_eval_connection(tmp_path / "eval.db")
        row = conn.execute(
            "SELECT movie_id FROM plot_events_references WHERE movie_id = 301"
        ).fetchone()
        conn.close()
        assert row is not None

    async def test_429_retry_failure_caught_without_inserting_row(self, tmp_path: Path) -> None:
        """When the retry after 429 also fails, the outer except catches it and no row is inserted."""
        mock_sleep = AsyncMock()
        # Both calls fail: first with 429, retry with a different error
        mock_llm = AsyncMock(
            side_effect=[
                ValueError("429 rate limit"),
                ValueError("Anthropic async failed: server error"),
            ]
        )
        movie_inputs = {302: _make_eval_movie(tmdb_id=302)}

        with patch(_LLM_PATCH, mock_llm), \
             patch(_PROMPT_PATCH, return_value="test prompt"), \
             patch(_SLEEP_PATCH, mock_sleep):
            await generate_reference_responses(movie_inputs, db_path=tmp_path / "eval.db")

        # Sleep was called (we did try to retry)
        mock_sleep.assert_called_once_with(60)
        # No row inserted since retry also failed
        conn = get_eval_connection(tmp_path / "eval.db")
        row_count = conn.execute(
            "SELECT COUNT(*) FROM plot_events_references WHERE movie_id = 302"
        ).fetchone()[0]
        conn.close()
        assert row_count == 0

    async def test_non_429_value_error_no_retry(self, tmp_path: Path) -> None:
        """A ValueError not containing '429' is caught by the outer except; no sleep, no retry, no row."""
        mock_sleep = AsyncMock()
        mock_llm = AsyncMock(side_effect=ValueError("Anthropic async failed: auth error"))
        movie_inputs = {401: _make_eval_movie(tmdb_id=401)}

        with patch(_LLM_PATCH, mock_llm), \
             patch(_PROMPT_PATCH, return_value="test prompt"), \
             patch(_SLEEP_PATCH, mock_sleep):
            await generate_reference_responses(movie_inputs, db_path=tmp_path / "eval.db")

        # No sleep — not a 429
        mock_sleep.assert_not_called()
        # LLM called exactly once (no retry)
        assert mock_llm.call_count == 1
        # No row inserted
        conn = get_eval_connection(tmp_path / "eval.db")
        row_count = conn.execute(
            "SELECT COUNT(*) FROM plot_events_references WHERE movie_id = 401"
        ).fetchone()[0]
        conn.close()
        assert row_count == 0

    async def test_idempotency_existing_references_skipped(self, tmp_path: Path) -> None:
        """Movies with pre-existing references are skipped; LLM only called for the new movie."""
        db_path = tmp_path / "eval.db"

        # Pre-insert references for tmdb_ids 501 and 502
        conn = get_eval_connection(db_path)
        create_plot_events_tables(conn)
        output = _make_plot_events_output()
        plot_summary, setting, chars_json = _serialize_output(output)
        for tmdb_id in (501, 502):
            conn.execute(
                "INSERT INTO plot_events_references "
                "(movie_id, plot_summary, setting, major_characters, reference_model, input_tokens, output_tokens, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (tmdb_id, plot_summary, setting, chars_json, "claude-opus-4-6", 10, 5, "2026-01-01T00:00:00+00:00"),
            )
        conn.commit()
        conn.close()

        # movie_inputs has the 2 existing movies + 1 new one
        movie_inputs = {
            501: _make_eval_movie(tmdb_id=501),
            502: _make_eval_movie(tmdb_id=502),
            503: _make_eval_movie(tmdb_id=503),
        }
        mock_output = _make_plot_events_output()
        mock_llm = AsyncMock(return_value=(mock_output, 10, 5))

        with patch(_LLM_PATCH, mock_llm), patch(_PROMPT_PATCH, return_value="test prompt"):
            await generate_reference_responses(movie_inputs, db_path=db_path)

        # Only the new movie triggered an LLM call
        assert mock_llm.call_count == 1

    async def test_each_successful_response_committed_immediately(self, tmp_path: Path) -> None:
        """Row for movie 1 is committed before the LLM call for movie 2 begins."""
        db_path = tmp_path / "eval.db"
        movie_inputs = {
            601: _make_eval_movie(tmdb_id=601, title="First"),
            602: _make_eval_movie(tmdb_id=602, title="Second"),
        }
        mock_output = _make_plot_events_output()
        # Track whether the first row was committed by the time the second call starts
        first_row_committed_before_second_call = [False]
        call_count = [0]

        async def mock_llm(**kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                # Open a fresh read connection — only sees committed data in WAL mode
                check_conn = get_eval_connection(db_path)
                row_count = check_conn.execute(
                    "SELECT COUNT(*) FROM plot_events_references"
                ).fetchone()[0]
                check_conn.close()
                first_row_committed_before_second_call[0] = row_count >= 1
            return (mock_output, 10, 5)

        with patch(_LLM_PATCH, side_effect=mock_llm), \
             patch(_PROMPT_PATCH, return_value="test prompt"):
            await generate_reference_responses(movie_inputs, db_path=db_path)

        assert first_row_committed_before_second_call[0], (
            "First movie's row was not committed to the DB before the second LLM call"
        )

    async def test_max_tokens_4096_in_reference_generation_calls(self, tmp_path: Path) -> None:
        """Both the primary LLM call and the retry call explicitly use max_tokens=4096."""
        mock_output = _make_plot_events_output()
        # First call raises 429 to trigger retry, both calls' kwargs are captured
        captured_kwargs: list[dict] = []

        async def mock_llm(**kwargs):
            captured_kwargs.append(dict(kwargs))
            if len(captured_kwargs) == 1:
                raise ValueError("429 rate limit exceeded")
            return (mock_output, 10, 5)

        mock_sleep = AsyncMock()
        movie_inputs = {701: _make_eval_movie(tmdb_id=701)}

        with patch(_LLM_PATCH, side_effect=mock_llm), \
             patch(_PROMPT_PATCH, return_value="test prompt"), \
             patch(_SLEEP_PATCH, mock_sleep):
            await generate_reference_responses(movie_inputs, db_path=tmp_path / "eval.db")

        assert len(captured_kwargs) == 2, "Expected both the primary call and the retry"
        assert captured_kwargs[0]["max_tokens"] == 4096
        assert captured_kwargs[1]["max_tokens"] == 4096


# ---------------------------------------------------------------------------
# Tests: run_evaluation — judge call parameters
# ---------------------------------------------------------------------------

# Helper: one generic non-Anthropic candidate for judge call tests
_TEST_CANDIDATE = EvaluationCandidate(
    candidate_id="test_openai_candidate",
    provider=LLMProvider.OPENAI,
    model="gpt-5-mini",
    system_prompt="Generate plot events metadata.",
    response_format=PlotEventsOutput,
    kwargs={"reasoning_effort": "low", "verbosity": "low"},
)


def _insert_reference(conn, tmdb_id: int) -> None:
    """Insert a pre-built reference row for the given tmdb_id."""
    output = _make_plot_events_output()
    plot_summary, setting, chars_json = _serialize_output(output)
    conn.execute(
        "INSERT INTO plot_events_references "
        "(movie_id, plot_summary, setting, major_characters, reference_model, input_tokens, output_tokens, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (tmdb_id, plot_summary, setting, chars_json, "claude-opus-4-6", 100, 50, "2026-01-01T00:00:00+00:00"),
    )
    conn.commit()


class TestJudgeCallParameters:
    """Tests for the judge LLM call parameters inside run_evaluation._evaluate_one."""

    async def test_judge_call_uses_max_tokens_4096(self, tmp_path: Path) -> None:
        """The judge generate_llm_response_async call passes max_tokens=4096."""
        db_path = tmp_path / "eval.db"
        tmdb_id = 800
        conn = get_eval_connection(db_path)
        create_plot_events_tables(conn)
        _insert_reference(conn, tmdb_id)
        conn.close()

        candidate_output = _make_plot_events_output()
        judge_output = _make_judge_output()
        judge_call_kwargs: list[dict] = []

        async def mock_llm(**kwargs):
            if kwargs.get("response_format") is PlotEventsJudgeOutput:
                judge_call_kwargs.append(dict(kwargs))
                return (judge_output, 10, 5)
            return (candidate_output, 10, 5)

        with patch(_LLM_PATCH, side_effect=mock_llm), \
             patch(_PROMPT_PATCH, return_value="test user prompt"):
            await run_evaluation(
                candidates=[_TEST_CANDIDATE],
                movie_inputs={tmdb_id: _make_eval_movie(tmdb_id=tmdb_id)},
                db_path=db_path,
            )

        assert len(judge_call_kwargs) == 1, "Expected exactly one judge call"
        assert judge_call_kwargs[0]["max_tokens"] == 4096

    async def test_judge_call_uses_temperature_0_2(self, tmp_path: Path) -> None:
        """The judge generate_llm_response_async call passes temperature=0.2."""
        db_path = tmp_path / "eval.db"
        tmdb_id = 801
        conn = get_eval_connection(db_path)
        create_plot_events_tables(conn)
        _insert_reference(conn, tmdb_id)
        conn.close()

        candidate_output = _make_plot_events_output()
        judge_output = _make_judge_output()
        judge_call_kwargs: list[dict] = []

        async def mock_llm(**kwargs):
            if kwargs.get("response_format") is PlotEventsJudgeOutput:
                judge_call_kwargs.append(dict(kwargs))
                return (judge_output, 10, 5)
            return (candidate_output, 10, 5)

        with patch(_LLM_PATCH, side_effect=mock_llm), \
             patch(_PROMPT_PATCH, return_value="test user prompt"):
            await run_evaluation(
                candidates=[_TEST_CANDIDATE],
                movie_inputs={tmdb_id: _make_eval_movie(tmdb_id=tmdb_id)},
                db_path=db_path,
            )

        assert len(judge_call_kwargs) == 1, "Expected exactly one judge call"
        assert judge_call_kwargs[0]["temperature"] == 0.2

    async def test_candidate_generation_call_does_not_receive_temperature_from_judge(
        self, tmp_path: Path
    ) -> None:
        """temperature=0.2 is passed to the judge call only, not to the candidate generation call."""
        db_path = tmp_path / "eval.db"
        tmdb_id = 802
        conn = get_eval_connection(db_path)
        create_plot_events_tables(conn)
        _insert_reference(conn, tmdb_id)
        conn.close()

        candidate_output = _make_plot_events_output()
        judge_output = _make_judge_output()
        candidate_call_kwargs: list[dict] = []

        async def mock_llm(**kwargs):
            if kwargs.get("response_format") is PlotEventsJudgeOutput:
                return (judge_output, 10, 5)
            candidate_call_kwargs.append(dict(kwargs))
            return (candidate_output, 10, 5)

        with patch(_LLM_PATCH, side_effect=mock_llm), \
             patch(_PROMPT_PATCH, return_value="test user prompt"):
            await run_evaluation(
                candidates=[_TEST_CANDIDATE],
                movie_inputs={tmdb_id: _make_eval_movie(tmdb_id=tmdb_id)},
                db_path=db_path,
            )

        assert len(candidate_call_kwargs) == 1, "Expected exactly one candidate generation call"
        # temperature=0.2 belongs to the judge call, not the candidate call
        assert "temperature" not in candidate_call_kwargs[0]
