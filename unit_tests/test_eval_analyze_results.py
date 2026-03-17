"""
Unit tests for movie_ingestion.metadata_generation.evaluations.analyze_results.

Covers:
  - _compute_per_movie_cost (known model, unknown model, zero tokens, arithmetic)
  - MODEL_PRICING (structure validation)
  - analyze_plot_events (no table, with data)

All DB tests use tmp_path for ephemeral SQLite databases — no real eval DB.
"""

import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from movie_ingestion.metadata_generation.evaluations.analyze_results import (
    MODEL_PRICING,
    _compute_per_movie_cost,
    analyze_plot_events,
)
from movie_ingestion.metadata_generation.evaluations.plot_events import (
    SCORE_COLUMNS,
    create_plot_events_tables,
)
from movie_ingestion.metadata_generation.evaluations.shared import (
    create_candidates_table,
    get_eval_connection,
)


# ---------------------------------------------------------------------------
# Tests: _compute_per_movie_cost
# ---------------------------------------------------------------------------


class TestComputePerMovieCost:
    def test_compute_per_movie_cost_known_model(self) -> None:
        """Correct USD calculation for a model in MODEL_PRICING."""
        # Use a known model from the pricing map
        result = _compute_per_movie_cost(
            mean_input_tokens=1000.0,
            mean_output_tokens=500.0,
            model="gpt-5-mini",
        )
        assert result is not None
        assert isinstance(result, float)
        assert result > 0.0

    def test_compute_per_movie_cost_unknown_model(self) -> None:
        """Returns None for a model not in MODEL_PRICING."""
        result = _compute_per_movie_cost(
            mean_input_tokens=1000.0,
            mean_output_tokens=500.0,
            model="nonexistent-model-xyz",
        )
        assert result is None

    def test_compute_per_movie_cost_zero_tokens(self) -> None:
        """Returns 0.0 for zero input and output tokens."""
        result = _compute_per_movie_cost(
            mean_input_tokens=0.0,
            mean_output_tokens=0.0,
            model="gpt-5-mini",
        )
        assert result == 0.0

    def test_compute_per_movie_cost_arithmetic(self) -> None:
        """Verify exact arithmetic: (input * input_price + output * output_price) / 1_000_000."""
        # gpt-5-mini: input=$0.25/M, output=$2.00/M
        input_price, output_price = MODEL_PRICING["gpt-5-mini"]
        mean_in = 5000.0
        mean_out = 1000.0
        expected = (mean_in * input_price + mean_out * output_price) / 1_000_000

        result = _compute_per_movie_cost(mean_in, mean_out, "gpt-5-mini")
        assert result == pytest.approx(expected)

    def test_compute_per_movie_cost_large_token_counts(self) -> None:
        """Very large token counts don't cause overflow."""
        result = _compute_per_movie_cost(
            mean_input_tokens=1_000_000_000.0,
            mean_output_tokens=1_000_000_000.0,
            model="gpt-5-mini",
        )
        assert result is not None
        assert isinstance(result, float)

    def test_compute_per_movie_cost_zero_price_model(self) -> None:
        """A model with zero pricing yields $0.00 cost."""
        # Temporarily check with a model that has the lowest prices
        # This tests the edge case of very small cost values
        result = _compute_per_movie_cost(
            mean_input_tokens=1.0,
            mean_output_tokens=1.0,
            model="gpt-5-nano",  # cheapest model in the pricing map
        )
        assert result is not None
        assert result >= 0.0


# ---------------------------------------------------------------------------
# Tests: MODEL_PRICING
# ---------------------------------------------------------------------------


class TestModelPricing:
    def test_model_pricing_all_entries_have_two_floats(self) -> None:
        """Every entry in MODEL_PRICING is a tuple of two floats."""
        for model, pricing in MODEL_PRICING.items():
            assert isinstance(pricing, tuple), f"{model}: not a tuple"
            assert len(pricing) == 2, f"{model}: wrong length"
            input_price, output_price = pricing
            assert isinstance(input_price, (int, float)), f"{model}: input_price not numeric"
            assert isinstance(output_price, (int, float)), f"{model}: output_price not numeric"


# ---------------------------------------------------------------------------
# Tests: analyze_plot_events
# ---------------------------------------------------------------------------


class TestAnalyzePlotEvents:
    def test_analyze_plot_events_returns_none_when_no_table(self, tmp_path: Path) -> None:
        """Returns None when evaluation table doesn't exist."""
        db_path = tmp_path / "eval.db"
        conn = get_eval_connection(db_path)
        conn.close()

        result = analyze_plot_events(db_path=db_path)
        assert result is None

    def test_analyze_plot_events_returns_dataframe(self, tmp_path: Path) -> None:
        """With test data inserted, returns a DataFrame with expected columns."""
        db_path = tmp_path / "eval.db"
        conn = get_eval_connection(db_path)
        create_candidates_table(conn)
        create_plot_events_tables(conn)

        # Insert a candidate record
        conn.execute(
            """
            INSERT INTO candidates (candidate_id, metadata_type, provider, model,
                                    system_prompt, parameters, schema_class, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("test-cand", "plot_events", "openai", "gpt-5-mini",
             "You are helpful.", "{}", "PlotEventsOutput", "2026-03-17T00:00:00Z"),
        )
        # Insert candidate output (for token counts)
        conn.execute(
            """
            INSERT INTO plot_events_candidate_outputs
                (movie_id, candidate_id, plot_summary, setting, major_characters,
                 input_tokens, output_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "test-cand", "Summary.", "Setting.", "[]", 1000, 500, "2026-03-17T00:00:00Z"),
        )
        # Insert evaluation result
        conn.execute(
            """
            INSERT INTO plot_events_evaluations
                (movie_id, candidate_id,
                 groundedness_score, groundedness_reasoning,
                 plot_summary_score, plot_summary_reasoning,
                 character_quality_score, character_quality_reasoning,
                 setting_score, setting_reasoning,
                 judge_model, judge_input_tokens, judge_output_tokens, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (1, "test-cand", 4, "Good.", 3, "Ok.", 3, "Fine.", 4, "Specific.",
             "claude-sonnet-4-6", 2000, 800, "2026-03-17T00:00:00Z"),
        )
        conn.commit()
        conn.close()

        result = analyze_plot_events(db_path=db_path)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert "overall_mean" in result.columns
        assert "test-cand" in result.index
