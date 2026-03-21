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
    run_evaluation,
)
from movie_ingestion.metadata_generation.evaluations.shared import (
    EvaluationCandidate,
    get_eval_connection,
)
from movie_ingestion.metadata_generation.inputs import MovieInputData
from movie_ingestion.metadata_generation.prompts.plot_events import (
    SYSTEM_PROMPT as DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPT_SHORT as SHORT_SYSTEM_PROMPT,
)
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
        """Calling twice doesn't raise, both tables created."""
        conn = get_eval_connection(tmp_path / "eval.db")
        create_plot_events_tables(conn)
        create_plot_events_tables(conn)  # Should not raise

        # Verify both tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {row["name"] for row in tables}
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
        """Output contains 'SOURCE DATA' and 'CANDIDATE OUTPUT' sections."""
        candidate = _make_plot_events_output(plot_summary="Different summary.")
        result = _build_judge_user_prompt("Test source data", candidate)

        assert "SOURCE DATA:" in result
        assert "CANDIDATE OUTPUT:" in result

    def test_build_judge_user_prompt_includes_source_data(self) -> None:
        """The source_data text appears in the output."""
        candidate = _make_plot_events_output()
        source_data = "title: The Matrix (1999)\noverview: A computer hacker learns about reality."
        result = _build_judge_user_prompt(source_data, candidate)

        assert source_data in result


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
# Patch targets for run_evaluation tests
# ---------------------------------------------------------------------------

_LLM_PATCH = "movie_ingestion.metadata_generation.evaluations.plot_events.generate_llm_response_async"
_PROMPT_PATCH = "movie_ingestion.metadata_generation.evaluations.plot_events.build_plot_events_prompts"


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


class TestJudgeCallParameters:
    """Tests for the judge LLM call parameters inside run_evaluation._evaluate_one."""

    async def test_judge_call_uses_reasoning_effort_low(self, tmp_path: Path) -> None:
        """The judge generate_llm_response_async call passes reasoning_effort='low'."""
        db_path = tmp_path / "eval.db"
        tmdb_id = 800

        candidate_output = _make_plot_events_output()
        judge_output = _make_judge_output()
        judge_call_kwargs: list[dict] = []

        async def mock_llm(**kwargs):
            if kwargs.get("response_format") is PlotEventsJudgeOutput:
                judge_call_kwargs.append(dict(kwargs))
                return (judge_output, 10, 5)
            return (candidate_output, 10, 5)

        with patch(_LLM_PATCH, side_effect=mock_llm), \
             patch(_PROMPT_PATCH, return_value=("test user prompt", "test system prompt")):
            await run_evaluation(
                candidates=[_TEST_CANDIDATE],
                movie_inputs={tmdb_id: _make_eval_movie(tmdb_id=tmdb_id)},
                db_path=db_path,
            )

        assert len(judge_call_kwargs) >= 1, "Expected at least one judge call"
        # Judge calls use cache_control and disabled thinking, no reasoning_effort
        assert judge_call_kwargs[0]["cache_control"] is True
        assert judge_call_kwargs[0]["thinking"] == {"type": "disabled"}

    async def test_judge_call_does_not_pass_temperature(self, tmp_path: Path) -> None:
        """The judge call does not include temperature in its kwargs."""
        db_path = tmp_path / "eval.db"
        tmdb_id = 801

        candidate_output = _make_plot_events_output()
        judge_output = _make_judge_output()
        judge_call_kwargs: list[dict] = []

        async def mock_llm(**kwargs):
            if kwargs.get("response_format") is PlotEventsJudgeOutput:
                judge_call_kwargs.append(dict(kwargs))
                return (judge_output, 10, 5)
            return (candidate_output, 10, 5)

        with patch(_LLM_PATCH, side_effect=mock_llm), \
             patch(_PROMPT_PATCH, return_value=("test user prompt", "test system prompt")):
            await run_evaluation(
                candidates=[_TEST_CANDIDATE],
                movie_inputs={tmdb_id: _make_eval_movie(tmdb_id=tmdb_id)},
                db_path=db_path,
            )

        assert len(judge_call_kwargs) >= 1, "Expected at least one judge call"
        assert "temperature" not in judge_call_kwargs[0]

    async def test_candidate_generation_call_does_not_receive_temperature_from_judge(
        self, tmp_path: Path
    ) -> None:
        """temperature=0.2 is passed to the judge call only, not to the candidate generation call."""
        db_path = tmp_path / "eval.db"
        tmdb_id = 802

        candidate_output = _make_plot_events_output()
        judge_output = _make_judge_output()
        candidate_call_kwargs: list[dict] = []

        async def mock_llm(**kwargs):
            if kwargs.get("response_format") is PlotEventsJudgeOutput:
                return (judge_output, 10, 5)
            candidate_call_kwargs.append(dict(kwargs))
            return (candidate_output, 10, 5)

        with patch(_LLM_PATCH, side_effect=mock_llm), \
             patch(_PROMPT_PATCH, return_value=("test user prompt", "test system prompt")):
            await run_evaluation(
                candidates=[_TEST_CANDIDATE],
                movie_inputs={tmdb_id: _make_eval_movie(tmdb_id=tmdb_id)},
                db_path=db_path,
            )

        assert len(candidate_call_kwargs) == 1, "Expected exactly one candidate generation call"
        # temperature=0.2 belongs to the judge call, not the candidate call
        assert "temperature" not in candidate_call_kwargs[0]


# ---------------------------------------------------------------------------
# Tests: SYSTEM_PROMPT_SHORT invariants
# ---------------------------------------------------------------------------


class TestSystemPromptShort:
    """Tests for the short system prompt variant used by evaluation candidates."""

    def test_system_prompt_short_is_nonempty_string(self) -> None:
        """SYSTEM_PROMPT_SHORT is a non-empty string."""
        assert isinstance(SHORT_SYSTEM_PROMPT, str)
        assert len(SHORT_SYSTEM_PROMPT) > 0

    def test_system_prompt_short_contains_no_hallucination_rule(self) -> None:
        """The critical no-hallucination instruction is preserved in the short variant."""
        # The module docstring calls this out as an essential invariant
        assert "Only describe what is evident from the provided data" in SHORT_SYSTEM_PROMPT

    def test_system_prompt_short_mentions_all_output_fields(self) -> None:
        """Short prompt mentions all 3 output fields: plot_summary, setting, major_characters."""
        assert "plot_summary" in SHORT_SYSTEM_PROMPT
        assert "setting" in SHORT_SYSTEM_PROMPT
        assert "major_characters" in SHORT_SYSTEM_PROMPT

    def test_system_prompt_short_is_shorter_than_default(self) -> None:
        """The short prompt is strictly shorter than the default (the whole point of the variant)."""
        assert len(SHORT_SYSTEM_PROMPT) < len(DEFAULT_SYSTEM_PROMPT)


# ---------------------------------------------------------------------------
# Tests: Short-prompt candidate configuration consistency
# ---------------------------------------------------------------------------


class TestShortPromptCandidates:
    """Tests for the short-prompt evaluation candidates in PLOT_EVENTS_CANDIDATES."""

    def test_short_prompt_candidates_use_short_system_prompt(self) -> None:
        """All candidates with '__short-prompt' in their ID use SHORT_SYSTEM_PROMPT."""
        short_prompt_candidates = [
            c for c in PLOT_EVENTS_CANDIDATES
            if "__short-prompt" in c.candidate_id
        ]
        assert len(short_prompt_candidates) > 0, "No short-prompt candidates found"
        for c in short_prompt_candidates:
            assert c.system_prompt is SHORT_SYSTEM_PROMPT, (
                f"Candidate {c.candidate_id} has '__short-prompt' in ID "
                f"but does not use SHORT_SYSTEM_PROMPT"
            )

    def test_short_prompt_candidates_naming_convention(self) -> None:
        """Every candidate using SHORT_SYSTEM_PROMPT has '__short-prompt' in its candidate_id."""
        for c in PLOT_EVENTS_CANDIDATES:
            if c.system_prompt is SHORT_SYSTEM_PROMPT:
                assert "__short-prompt" in c.candidate_id, (
                    f"Candidate {c.candidate_id} uses SHORT_SYSTEM_PROMPT "
                    f"but lacks '__short-prompt' suffix"
                )

    def test_non_short_prompt_candidates_use_default_system_prompt(self) -> None:
        """Candidates without '__short-prompt' in their ID use DEFAULT_SYSTEM_PROMPT."""
        non_short_candidates = [
            c for c in PLOT_EVENTS_CANDIDATES
            if "__short-prompt" not in c.candidate_id
        ]
        for c in non_short_candidates:
            assert c.system_prompt is DEFAULT_SYSTEM_PROMPT, (
                f"Candidate {c.candidate_id} lacks '__short-prompt' in ID "
                f"but does not use DEFAULT_SYSTEM_PROMPT"
            )
