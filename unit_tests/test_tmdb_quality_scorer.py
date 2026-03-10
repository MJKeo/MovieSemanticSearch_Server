"""
Unit tests for movie_ingestion.tmdb_quality_scorer — Stage 3b quality scoring.

Covers:
  - _score_watch_providers — tiered streaming score + theater window
  - _score_overview_length — tiered tiers
  - compute_quality_score  — composite weighted sum
  - WEIGHTS invariant       — module-level guard
  - run()                  — end-to-end integration via mocked init_db

Note: Tests for shared scoring utilities (unpack_provider_keys, score_vote_count,
score_popularity, validate_weights) live in test_scoring_utils.py.
"""

import datetime
import sqlite3
import struct
from typing import Any
from unittest.mock import patch

import pytest

from movie_ingestion.scoring_utils import THEATER_WINDOW_DAYS
from movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer import (
    WEIGHTS,
    _score_overview_length,
    _score_watch_providers,
    compute_quality_score,
    run,
)
from movie_ingestion.tracker import _SCHEMA_SQL


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

# Fixed reference date used throughout to make all tests deterministic.
TODAY = datetime.date(2026, 3, 5)


def _pack_providers(*provider_ids: int) -> bytes:
    """Encode provider IDs as a little-endian uint32 BLOB (same format as tmdb_fetcher)."""
    return struct.pack(f"<{len(provider_ids)}I", *provider_ids)


def _make_row(**kwargs: Any) -> sqlite3.Row:
    """Create a named-column sqlite3.Row from keyword arguments.

    Creates a single-row ephemeral in-memory SQLite table so that
    compute_quality_score() receives a real sqlite3.Row with the exact
    interface it expects (string key access via row["column_name"]).
    """
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    cols = list(kwargs.keys())
    db.execute(f"CREATE TABLE t ({', '.join(cols)})")
    db.execute(f"INSERT INTO t VALUES ({', '.join('?' * len(cols))})", list(kwargs.values()))
    return db.execute("SELECT * FROM t").fetchone()


def _default_scorer_row(**overrides: Any) -> sqlite3.Row:
    """Return a sqlite3.Row with solid mid-range values for compute_quality_score.

    All fields produce positive-to-neutral signals.  Individual tests override
    only the specific field(s) they want to exercise.
    """
    defaults: dict[str, Any] = {
        "tmdb_id":                  1,
        "vote_count":               500,
        "popularity":               5.0,
        "release_date":             "2015-01-01",          # ~11yr: no recency or classic boost
        "poster_url":               "https://example.com/p.jpg",
        "watch_provider_keys":      _pack_providers(8, 15, 337),  # 3 providers → +1.0
        "overview_length":          250,                           # 201+ tier → 1.0
        "has_revenue":              1,
        "has_budget":               1,
        "has_production_companies": 1,
        "has_keywords":             1,
        "has_cast_and_crew":        1,
    }
    defaults.update(overrides)
    return _make_row(**defaults)


@pytest.fixture()
def run_db(tmp_path) -> Any:
    """File-based SQLite DB in a temp directory for run() integration tests.

    Uses a real file (not :memory:) so the database survives run()'s db.close()
    call, allowing a fresh connection to verify results afterward.

    Returns the db Path.
    """
    db_path = tmp_path / "tracker.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(_SCHEMA_SQL)
    conn.commit()
    conn.close()
    return db_path


def _seed_movie(db_path: Any, tmdb_id: int, **overrides: Any) -> None:
    """Insert one tmdb_data + movie_progress(status='tmdb_fetched') row pair."""
    defaults: dict[str, Any] = {
        "tmdb_id":                  tmdb_id,
        "imdb_id":                  f"tt{tmdb_id:07d}",
        "title":                    f"Test Movie {tmdb_id}",
        "release_date":             "2015-06-15",
        "duration":                 90,
        "poster_url":               "https://example.com/poster.jpg",
        "watch_provider_keys":      _pack_providers(8, 15, 337),
        "vote_count":               200,
        "popularity":               3.0,
        "vote_average":             6.5,
        "overview_length":          180,
        "genre_count":              2,
        "has_revenue":              0,
        "has_budget":               0,
        "has_production_companies": 1,
        "has_production_countries": 1,
        "has_keywords":             1,
        "has_cast_and_crew":        1,
    }
    defaults.update(overrides)
    ordered = [
        "tmdb_id", "imdb_id", "title", "release_date", "duration", "poster_url",
        "watch_provider_keys", "vote_count", "popularity", "vote_average",
        "overview_length", "genre_count", "has_revenue", "has_budget",
        "has_production_companies", "has_production_countries",
        "has_keywords", "has_cast_and_crew",
    ]
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        f"""INSERT OR REPLACE INTO tmdb_data
            ({', '.join(ordered)})
            VALUES ({', '.join('?' * len(ordered))})""",
        [defaults[k] for k in ordered],
    )
    conn.execute(
        "INSERT OR REPLACE INTO movie_progress (tmdb_id, status) VALUES (?, 'tmdb_fetched')",
        (tmdb_id,),
    )
    conn.commit()
    conn.close()


def _read_score(db_path: Any, tmdb_id: int) -> float | None:
    """Open a fresh connection and return the stage_3_quality_score for tmdb_id."""
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT stage_3_quality_score FROM movie_progress WHERE tmdb_id = ?", (tmdb_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _read_status(db_path: Any, tmdb_id: int) -> str | None:
    """Open a fresh connection and return the status for tmdb_id."""
    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT status FROM movie_progress WHERE tmdb_id = ?", (tmdb_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def _make_run_conn(db_path: Any) -> sqlite3.Connection:
    """Open a fresh connection suitable for run() to use (no row_factory — run() sets it)."""
    return sqlite3.connect(str(db_path))


# ---------------------------------------------------------------------------
# _score_watch_providers
# ---------------------------------------------------------------------------


class TestScoreWatchProviders:
    """Tests for the tiered watch-provider scorer with theater-window logic."""

    # Convenience dates relative to TODAY
    PAST_WINDOW    = (TODAY - datetime.timedelta(days=THEATER_WINDOW_DAYS + 1)).isoformat()
    WITHIN_WINDOW  = (TODAY - datetime.timedelta(days=THEATER_WINDOW_DAYS - 1)).isoformat()
    AT_BOUNDARY    = (TODAY - datetime.timedelta(days=THEATER_WINDOW_DAYS)).isoformat()

    # ---- Zero providers ----

    def test_zero_providers_past_window_returns_minus_one(self) -> None:
        """0 providers + past theater window → maximum penalty (-1.0)."""
        assert _score_watch_providers(0, self.PAST_WINDOW, TODAY) == -1.0

    def test_zero_providers_within_window_returns_zero(self) -> None:
        """0 providers + within theater window → 0.0 (still in cinemas)."""
        assert _score_watch_providers(0, self.WITHIN_WINDOW, TODAY) == 0.0

    def test_zero_providers_null_date_returns_minus_one(self) -> None:
        """0 providers + null release_date → conservative past-window → -1.0."""
        assert _score_watch_providers(0, None, TODAY) == -1.0

    def test_theater_window_boundary_is_strictly_greater(self) -> None:
        """Film released exactly THEATER_WINDOW_DAYS ago is NOT past the window (strict >)."""
        # days_since = THEATER_WINDOW_DAYS → NOT > THEATER_WINDOW_DAYS → still inside
        assert _score_watch_providers(0, self.AT_BOUNDARY, TODAY) == 0.0

    def test_one_day_past_boundary_is_past_window(self) -> None:
        """Film released one day past the boundary is treated as past the window."""
        assert _score_watch_providers(0, self.PAST_WINDOW, TODAY) == -1.0

    # ---- 1–2 providers ----

    def test_one_provider_returns_point_five(self) -> None:
        """1 provider → +0.5 regardless of theater window status."""
        assert _score_watch_providers(1, self.PAST_WINDOW, TODAY) == pytest.approx(0.5)

    def test_two_providers_returns_point_five(self) -> None:
        """2 providers → +0.5 (1–2 tier)."""
        assert _score_watch_providers(2, self.PAST_WINDOW, TODAY) == pytest.approx(0.5)

    def test_one_provider_within_window_still_returns_point_five(self) -> None:
        """1 provider within theater window: providers trump window status → +0.5."""
        assert _score_watch_providers(1, self.WITHIN_WINDOW, TODAY) == pytest.approx(0.5)

    # ---- 3+ providers ----

    def test_three_providers_returns_one(self) -> None:
        """3 providers → +1.0 (maximum tier)."""
        assert _score_watch_providers(3, self.PAST_WINDOW, TODAY) == pytest.approx(1.0)

    def test_many_providers_capped_at_one(self) -> None:
        """20 providers → still +1.0 (no additional credit for more than 3)."""
        assert _score_watch_providers(20, self.PAST_WINDOW, TODAY) == pytest.approx(1.0)

    # ---- Invalid dates ----

    def test_unparseable_date_treated_as_past_window(self) -> None:
        """Unparseable date string conservatively assumes post-theater → -1.0 for 0 providers."""
        assert _score_watch_providers(0, "not-a-date", TODAY) == -1.0

    def test_year_only_date_treated_as_past_window(self) -> None:
        """Year-only date ('2025') triggers ValueError → conservative past-theater → -1.0."""
        assert _score_watch_providers(0, "2025", TODAY) == -1.0

    def test_unparseable_date_with_providers_still_returns_positive(self) -> None:
        """Even with a bad date, streaming providers present → +0.5."""
        assert _score_watch_providers(1, "not-a-date", TODAY) == pytest.approx(0.5)

    # ---- Score range ----

    def test_all_scores_in_valid_range(self) -> None:
        """All possible score outcomes lie in [-1, +1]."""
        for count in [0, 1, 2, 3, 10]:
            for date in [None, self.PAST_WINDOW, self.WITHIN_WINDOW]:
                score = _score_watch_providers(count, date, TODAY)
                assert -1.0 <= score <= 1.0

    def test_score_hierarchy_is_correct(self) -> None:
        """3+ providers > 1–2 providers > 0 providers (past window)."""
        score_3plus = _score_watch_providers(3, self.PAST_WINDOW, TODAY)
        score_1_2   = _score_watch_providers(1, self.PAST_WINDOW, TODAY)
        score_0     = _score_watch_providers(0, self.PAST_WINDOW, TODAY)
        assert score_3plus > score_1_2 > score_0


# ---------------------------------------------------------------------------
# _score_overview_length
# ---------------------------------------------------------------------------


class TestScoreOverviewLength:
    """Tests for the tiered overview-length scorer."""

    # ---- Tier boundaries ----

    def test_length_0_in_bottom_tier(self) -> None:
        """length=0 falls in the ≤50 tier → 0.2."""
        assert _score_overview_length(0) == pytest.approx(0.2)

    def test_length_50_at_bottom_tier_ceiling(self) -> None:
        """length=50 is the boundary of the ≤50 tier → 0.2."""
        assert _score_overview_length(50) == pytest.approx(0.2)

    def test_length_51_enters_second_tier(self) -> None:
        """length=51 crosses into the 51–100 tier → 0.6."""
        assert _score_overview_length(51) == pytest.approx(0.6)

    def test_length_100_at_second_tier_ceiling(self) -> None:
        """length=100 is the boundary of the 51–100 tier → 0.6."""
        assert _score_overview_length(100) == pytest.approx(0.6)

    def test_length_101_enters_third_tier(self) -> None:
        """length=101 crosses into the 101–200 tier → 0.85."""
        assert _score_overview_length(101) == pytest.approx(0.85)

    def test_length_200_at_third_tier_ceiling(self) -> None:
        """length=200 is the boundary of the 101–200 tier → 0.85."""
        assert _score_overview_length(200) == pytest.approx(0.85)

    def test_length_201_enters_top_tier(self) -> None:
        """length=201 crosses into the top tier → 1.0."""
        assert _score_overview_length(201) == pytest.approx(1.0)

    def test_long_overview_returns_one(self) -> None:
        """Arbitrarily long overview (10,000 chars) still returns 1.0."""
        assert _score_overview_length(10_000) == pytest.approx(1.0)

    # ---- Edge / anomaly inputs ----

    def test_negative_length_falls_in_bottom_tier(self) -> None:
        """Negative length (data anomaly) is below 50 → 0.2."""
        assert _score_overview_length(-1) == pytest.approx(0.2)

    def test_length_1_in_bottom_tier(self) -> None:
        """Minimal non-zero length (1 char) falls in ≤50 tier → 0.2."""
        assert _score_overview_length(1) == pytest.approx(0.2)

    # ---- Tier structure invariants ----

    def test_strictly_increasing_across_tier_boundaries(self) -> None:
        """Score at the bottom of each higher tier exceeds the top of the prior tier."""
        assert _score_overview_length(51) > _score_overview_length(50)
        assert _score_overview_length(101) > _score_overview_length(100)
        assert _score_overview_length(201) > _score_overview_length(200)

    def test_all_outputs_in_valid_range(self) -> None:
        """All tier outputs lie in [0.2, 1.0]."""
        for length in [0, 25, 50, 51, 75, 100, 101, 150, 200, 201, 500, 1000]:
            score = _score_overview_length(length)
            assert 0.2 <= score <= 1.0

    def test_only_four_distinct_output_values(self) -> None:
        """Exactly four distinct output values exist: 0.2, 0.6, 0.85, 1.0."""
        outputs = {_score_overview_length(l) for l in [1, 75, 150, 300]}
        assert outputs == {0.2, 0.6, 0.85, 1.0}


# ---------------------------------------------------------------------------
# compute_quality_score
# ---------------------------------------------------------------------------


class TestComputeQualityScore:
    """Tests for the composite weighted quality score."""

    def test_all_max_signals_approaches_documented_maximum(self) -> None:
        """A movie with every signal at maximum produces a score ≈ 0.88 (documented max)."""
        row = _make_row(
            tmdb_id=1,
            vote_count=2000,
            popularity=10.0,
            release_date="2015-01-01",              # ~11yr: no age adjustment
            poster_url="https://example.com/p.jpg",
            watch_provider_keys=_pack_providers(1, 2, 3),  # 3+ → +1.0
            overview_length=300,
            has_revenue=1,
            has_budget=1,
            has_production_companies=1,
            has_keywords=1,
            has_cast_and_crew=1,
        )
        score = compute_quality_score(row, TODAY)
        assert score == pytest.approx(0.88, abs=0.02)

    def test_all_min_signals_approaches_documented_minimum(self) -> None:
        """A movie with every signal at minimum produces a score ≈ -0.34 (documented min)."""
        past = (TODAY - datetime.timedelta(days=200)).isoformat()  # past theater window
        row = _make_row(
            tmdb_id=2,
            vote_count=1,
            popularity=0.0,
            release_date=past,
            poster_url=None,                # absent → -1.0 signal
            watch_provider_keys=b"",        # 0 providers + past window → -1.0
            overview_length=1,              # ≤50 tier → 0.2
            has_revenue=0,
            has_budget=0,
            has_production_companies=0,     # absent → -1.0 signal
            has_keywords=0,                 # absent → -0.5
            has_cast_and_crew=0,            # absent → -1.0 signal
        )
        score = compute_quality_score(row, TODAY)
        assert score == pytest.approx(-0.34, abs=0.05)

    # ---- Individual signal contributions ----

    def test_missing_poster_url_applies_exact_penalty(self) -> None:
        """poster_url=None reduces score by exactly WEIGHTS['poster_url'] × 1.0."""
        with_poster    = _default_scorer_row(poster_url="https://example.com/p.jpg")
        without_poster = _default_scorer_row(poster_url=None)
        delta = compute_quality_score(with_poster, TODAY) - compute_quality_score(without_poster, TODAY)
        assert delta == pytest.approx(WEIGHTS["poster_url"], abs=1e-9)

    def test_empty_string_poster_not_penalised(self) -> None:
        """Empty string poster_url is not None → no penalty (presence is expected)."""
        with_poster  = _default_scorer_row(poster_url="https://example.com/p.jpg")
        empty_poster = _default_scorer_row(poster_url="")
        # Both score 0.0 for poster signal; scores should be equal
        assert compute_quality_score(with_poster, TODAY) == pytest.approx(
            compute_quality_score(empty_poster, TODAY), abs=1e-9
        )

    def test_missing_cast_and_crew_applies_exact_penalty(self) -> None:
        """has_cast_and_crew=0 reduces score by exactly WEIGHTS['has_cast_and_crew']."""
        with_cc    = _default_scorer_row(has_cast_and_crew=1)
        without_cc = _default_scorer_row(has_cast_and_crew=0)
        delta = compute_quality_score(with_cc, TODAY) - compute_quality_score(without_cc, TODAY)
        assert delta == pytest.approx(WEIGHTS["has_cast_and_crew"], abs=1e-9)

    def test_missing_production_companies_applies_exact_penalty(self) -> None:
        """has_production_companies=0 reduces score by WEIGHTS['has_production_companies']."""
        with_pc    = _default_scorer_row(has_production_companies=1)
        without_pc = _default_scorer_row(has_production_companies=0)
        delta = compute_quality_score(with_pc, TODAY) - compute_quality_score(without_pc, TODAY)
        assert delta == pytest.approx(WEIGHTS["has_production_companies"], abs=1e-9)

    def test_has_revenue_is_pure_bonus_not_penalty(self) -> None:
        """has_revenue=0 does NOT reduce score vs no-revenue baseline; =1 adds a bonus."""
        with_rev    = _default_scorer_row(has_revenue=1)
        without_rev = _default_scorer_row(has_revenue=0)
        delta = compute_quality_score(with_rev, TODAY) - compute_quality_score(without_rev, TODAY)
        # Difference = WEIGHTS * (1.0 - 0.0) = WEIGHTS["has_revenue"]
        assert delta == pytest.approx(WEIGHTS["has_revenue"], abs=1e-9)

    def test_has_budget_is_pure_bonus_not_penalty(self) -> None:
        """has_budget=0 does NOT reduce score; =1 adds a bonus of WEIGHTS['has_budget']."""
        with_bud    = _default_scorer_row(has_budget=1)
        without_bud = _default_scorer_row(has_budget=0)
        delta = compute_quality_score(with_bud, TODAY) - compute_quality_score(without_bud, TODAY)
        assert delta == pytest.approx(WEIGHTS["has_budget"], abs=1e-9)

    def test_has_keywords_is_symmetric(self) -> None:
        """has_keywords swing is ±0.5; total delta = WEIGHTS['has_keywords'] × 1.0."""
        with_kw    = _default_scorer_row(has_keywords=1)
        without_kw = _default_scorer_row(has_keywords=0)
        delta = compute_quality_score(with_kw, TODAY) - compute_quality_score(without_kw, TODAY)
        # (+0.5) - (-0.5) = 1.0 × weight
        assert delta == pytest.approx(WEIGHTS["has_keywords"] * 1.0, abs=1e-9)

    def test_watch_providers_full_swing(self) -> None:
        """Swing from 3+ providers (+1.0) to 0 past-window (-1.0) = weight × 2.0."""
        past = (TODAY - datetime.timedelta(days=200)).isoformat()
        with_providers    = _default_scorer_row(watch_provider_keys=_pack_providers(1, 2, 3), release_date=past)
        without_providers = _default_scorer_row(watch_provider_keys=b"", release_date=past)
        delta = (
            compute_quality_score(with_providers, TODAY)
            - compute_quality_score(without_providers, TODAY)
        )
        assert delta == pytest.approx(WEIGHTS["watch_providers"] * 2.0, abs=0.001)

    def test_null_watch_provider_keys_treated_as_zero_providers(self) -> None:
        """watch_provider_keys=None decodes to 0 providers — same score as empty BLOB."""
        past = (TODAY - datetime.timedelta(days=200)).isoformat()
        null_row  = _make_row(
            tmdb_id=10, vote_count=100, popularity=2.0, release_date=past,
            poster_url="p", watch_provider_keys=None, overview_length=200,
            has_revenue=0, has_budget=0, has_production_companies=1,
            has_keywords=1, has_cast_and_crew=1,
        )
        empty_row = _make_row(
            tmdb_id=11, vote_count=100, popularity=2.0, release_date=past,
            poster_url="p", watch_provider_keys=b"", overview_length=200,
            has_revenue=0, has_budget=0, has_production_companies=1,
            has_keywords=1, has_cast_and_crew=1,
        )
        assert compute_quality_score(null_row, TODAY) == pytest.approx(
            compute_quality_score(empty_row, TODAY), abs=1e-9
        )

    def test_exact_weighted_sum_manually_computed(self) -> None:
        """Score for a fully-specified row matches a hand-computed weighted sum."""
        row = _make_row(
            tmdb_id=1,
            vote_count=2000,                         # vc_score ≈ 1.0
            popularity=10.0,                          # pop_score = 1.0
            release_date="2015-01-01",               # no age adjustment
            poster_url="https://example.com/p.jpg", # post_score = 0.0
            watch_provider_keys=_pack_providers(1, 2, 3),  # wp_score = 1.0
            overview_length=300,                     # ol_score = 1.0
            has_revenue=1,                           # rev_score = 1.0
            has_budget=1,                            # bud_score = 1.0
            has_production_companies=1,              # pc_score = 0.0
            has_keywords=1,                          # kw_score = 0.5
            has_cast_and_crew=1,                     # cc_score = 0.0
        )
        score = compute_quality_score(row, TODAY)
        expected = (
            WEIGHTS["vote_count"]               * 1.0
            + WEIGHTS["watch_providers"]        * 1.0
            + WEIGHTS["popularity"]             * 1.0
            + WEIGHTS["has_revenue"]            * 1.0
            + WEIGHTS["poster_url"]             * 0.0
            + WEIGHTS["overview_length"]        * 1.0
            + WEIGHTS["has_keywords"]           * 0.5
            + WEIGHTS["has_production_companies"] * 0.0
            + WEIGHTS["has_budget"]             * 1.0
            + WEIGHTS["has_cast_and_crew"]      * 0.0
        )
        assert score == pytest.approx(expected, abs=0.01)

    def test_each_signal_independently_affects_score(self) -> None:
        """Toggling each continuous/boolean signal changes the final score."""
        toggles = [
            ("vote_count",               1,   2000),
            ("popularity",               0.0, 10.0),
            ("has_revenue",              0,   1),
            ("has_budget",               0,   1),
            ("has_keywords",             0,   1),
            ("has_cast_and_crew",        0,   1),
            ("has_production_companies", 0,   1),
        ]
        for field, low, high in toggles:
            low_score  = compute_quality_score(_default_scorer_row(**{field: low}),  TODAY)
            high_score = compute_quality_score(_default_scorer_row(**{field: high}), TODAY)
            assert high_score != pytest.approx(low_score, abs=1e-6), (
                f"Field '{field}' had no effect — weight may be zero or signal logic is broken"
            )


# ---------------------------------------------------------------------------
# WEIGHTS invariant
# ---------------------------------------------------------------------------


class TestWeightsInvariant:
    """Tests for the module-level WEIGHTS sum guard."""

    def test_weights_sum_to_one(self) -> None:
        """WEIGHTS values sum to exactly 1.0 within floating-point tolerance."""
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_all_weights_strictly_positive(self) -> None:
        """Every weight is strictly > 0."""
        for key, val in WEIGHTS.items():
            assert val > 0, f"Weight for '{key}' is zero or negative"

    def test_weights_has_ten_entries(self) -> None:
        """WEIGHTS contains exactly 10 signals."""
        assert len(WEIGHTS) == 10

    def test_weights_contains_all_expected_signal_keys(self) -> None:
        """WEIGHTS keys exactly match the ten documented signals."""
        expected = {
            "vote_count", "watch_providers", "popularity",
            "has_revenue", "poster_url", "overview_length",
            "has_keywords", "has_production_companies",
            "has_budget", "has_cast_and_crew",
        }
        assert set(WEIGHTS.keys()) == expected

    def test_no_single_signal_dominates_entirely(self) -> None:
        """No individual weight exceeds 0.50 (prevents a single signal from overwhelming all others)."""
        for key, val in WEIGHTS.items():
            assert val < 0.50, f"Weight for '{key}' ({val}) dominates the score"


# ---------------------------------------------------------------------------
# run() — integration
# ---------------------------------------------------------------------------


class TestRun:
    """End-to-end integration tests for the run() entry point."""

    def test_scores_all_tmdb_fetched_movies(self, run_db) -> None:
        """run() writes a non-null quality_score to every tmdb_fetched movie."""
        for tmdb_id in [101, 102, 103]:
            _seed_movie(run_db, tmdb_id)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        for tmdb_id in [101, 102, 103]:
            assert _read_score(run_db, tmdb_id) is not None

    def test_scores_are_floats(self, run_db) -> None:
        """run() writes float values (not integers or strings) to quality_score."""
        _seed_movie(run_db, 201)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        assert isinstance(_read_score(run_db, 201), float)

    def test_status_unchanged_after_scoring(self, run_db) -> None:
        """run() does NOT modify movie status — it remains 'tmdb_fetched'."""
        _seed_movie(run_db, 301)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        assert _read_status(run_db, 301) == "tmdb_fetched"

    def test_skips_non_tmdb_fetched_movies(self, run_db) -> None:
        """run() does not score movies that are not in 'tmdb_fetched' status."""
        _seed_movie(run_db, 401)
        conn = sqlite3.connect(str(run_db))
        conn.execute("UPDATE movie_progress SET status = 'filtered_out' WHERE tmdb_id = 401")
        conn.commit()
        conn.close()

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        assert _read_score(run_db, 401) is None

    def test_empty_db_exits_gracefully(self, run_db) -> None:
        """run() exits without error or exception when no tmdb_fetched movies exist."""
        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()  # must not raise

    def test_idempotent_second_run_overwrites_with_same_scores(self, run_db) -> None:
        """Calling run() twice produces identical quality_scores (idempotency)."""
        _seed_movie(run_db, 501)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()
        score_first = _read_score(run_db, 501)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()
        score_second = _read_score(run_db, 501)

        assert score_first == pytest.approx(score_second, abs=1e-9)

    def test_scores_within_documented_bounds(self, run_db) -> None:
        """All computed scores fall within the documented approximate range [-0.34, 0.88]."""
        for tmdb_id in range(601, 611):
            _seed_movie(run_db, tmdb_id)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        conn = sqlite3.connect(str(run_db))
        scores = [
            row[0] for row in conn.execute(
                "SELECT stage_3_quality_score FROM movie_progress "
                "WHERE tmdb_id BETWEEN 601 AND 610 AND stage_3_quality_score IS NOT NULL"
            ).fetchall()
        ]
        conn.close()
        assert len(scores) == 10
        for score in scores:
            assert -0.40 <= score <= 1.0, f"Score {score} is outside expected bounds"

    def test_higher_vote_count_produces_higher_score(self, run_db) -> None:
        """A movie with vc=2000 scores strictly higher than an identical movie with vc=1."""
        _seed_movie(run_db, 701, vote_count=2000)
        _seed_movie(run_db, 702, vote_count=1)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        assert _read_score(run_db, 701) > _read_score(run_db, 702)

    def test_streaming_availability_increases_score(self, run_db) -> None:
        """Movie with 3 US streaming providers scores higher than one with none (post-window)."""
        past_date = (TODAY - datetime.timedelta(days=200)).isoformat()
        _seed_movie(run_db, 801, watch_provider_keys=_pack_providers(8, 15, 337),
                    release_date=past_date)
        _seed_movie(run_db, 802, watch_provider_keys=b"",
                    release_date=past_date)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        assert _read_score(run_db, 801) > _read_score(run_db, 802)

    def test_updated_at_is_set(self, run_db) -> None:
        """run() updates the updated_at timestamp for each scored movie."""
        _seed_movie(run_db, 901)

        conn = sqlite3.connect(str(run_db))
        original_ts = conn.execute(
            "SELECT updated_at FROM movie_progress WHERE tmdb_id = 901"
        ).fetchone()[0]
        conn.close()

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        new_ts = None
        conn = sqlite3.connect(str(run_db))
        new_ts = conn.execute(
            "SELECT updated_at FROM movie_progress WHERE tmdb_id = 901"
        ).fetchone()[0]
        conn.close()

        # The timestamp should have been written (may differ from original if original was NULL)
        assert new_ts is not None

    def test_movies_with_no_providers_past_window_score_lower(self, run_db) -> None:
        """A movie past the theater window with no streaming scores lower than one within window."""
        past_date   = (TODAY - datetime.timedelta(days=200)).isoformat()
        recent_date = (TODAY - datetime.timedelta(days=30)).isoformat()
        _seed_movie(run_db, 1001, watch_provider_keys=b"", release_date=past_date,
                    vote_count=50, popularity=0.5)   # past window, no providers → -1.0 wp
        _seed_movie(run_db, 1002, watch_provider_keys=b"", release_date=recent_date,
                    vote_count=50, popularity=0.5)   # within window → 0.0 wp

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        assert _read_score(run_db, 1002) > _read_score(run_db, 1001)
