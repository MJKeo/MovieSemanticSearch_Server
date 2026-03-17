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

import pytest
from pydantic import BaseModel, ValidationError

from movie_ingestion.metadata_generation.evaluations.plot_events import (
    PLOT_EVENTS_CANDIDATES,
    SCORE_COLUMNS,
    PlotEventsJudgeOutput,
    _build_judge_user_prompt,
    _deserialize_output,
    _format_characters_for_prompt,
    _serialize_output,
    create_plot_events_tables,
)
from movie_ingestion.metadata_generation.evaluations.shared import get_eval_connection
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
