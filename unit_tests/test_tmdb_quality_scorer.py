"""
Unit tests for movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer — Stage 3.

Covers:
  - _is_unreleased        — future-date detection with None/bad-date handling
  - _has_us_providers     — BLOB → boolean provider presence
  - score_overview_length — 5-tier character-count scoring (0.0/0.2/0.5/0.8/1.0)
  - score_data_completeness — 8-indicator average
  - compute_quality_score — edge cases + 4-signal weighted formula
  - WEIGHTS invariant     — 4 entries summing to 1.0
  - run()                 — end-to-end integration via mocked init_db

Note: Tests for shared scoring utilities (unpack_provider_keys, score_vote_count,
score_popularity, validate_weights) live in test_scoring_utils.py.
"""

import datetime
import sqlite3
import struct
from typing import Any
from unittest.mock import patch

import pytest

from movie_ingestion.scoring_utils import (
    VoteCountSource,
    score_popularity,
    score_vote_count,
)
from movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer import (
    COMMIT_EVERY,
    WEIGHTS,
    _has_us_providers,
    _is_unreleased,
    compute_quality_score,
    run,
    score_data_completeness,
    score_overview_length,
)
from movie_ingestion.tracker import MovieStatus, _SCHEMA_SQL


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

    Default row has NO providers and a past release date so the weighted
    formula path is exercised (not the edge cases).
    """
    defaults: dict[str, Any] = {
        "tmdb_id":                  1,
        "vote_count":               50,
        "popularity":               3.0,
        "release_date":             "2015-01-01",
        "poster_url":               "https://example.com/p.jpg",
        "watch_provider_keys":      None,       # No providers → formula path
        "overview_length":          250,
        "genre_count":              2,
        "has_revenue":              1,
        "has_budget":               1,
        "has_production_companies": 1,
        "has_production_countries": 1,
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
        "watch_provider_keys":      None,       # No providers by default
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
# _is_unreleased
# ---------------------------------------------------------------------------


class TestIsUnreleased:
    """Tests for the future-date detection helper."""

    def test_future_date_returns_true(self) -> None:
        """Date after today → True."""
        future = (TODAY + datetime.timedelta(days=30)).isoformat()
        assert _is_unreleased(future, TODAY) is True

    def test_past_date_returns_false(self) -> None:
        """Date before today → False."""
        past = (TODAY - datetime.timedelta(days=30)).isoformat()
        assert _is_unreleased(past, TODAY) is False

    def test_today_returns_false(self) -> None:
        """Exact today → False (not strictly future)."""
        assert _is_unreleased(TODAY.isoformat(), TODAY) is False

    def test_none_returns_false(self) -> None:
        """None → False (missing date ≠ unreleased)."""
        assert _is_unreleased(None, TODAY) is False

    def test_unparseable_date_returns_false(self) -> None:
        """'not-a-date' → False (conservative: don't auto-fail on bad data)."""
        assert _is_unreleased("not-a-date", TODAY) is False

    def test_partial_date_returns_false(self) -> None:
        """'2027' → False (ValueError path — partial date is not parseable)."""
        assert _is_unreleased("2027", TODAY) is False


# ---------------------------------------------------------------------------
# _has_us_providers
# ---------------------------------------------------------------------------


class TestHasUsProviders:
    """Tests for the BLOB → boolean provider presence check."""

    def test_none_blob_returns_false(self) -> None:
        """None → False."""
        assert _has_us_providers(None) is False

    def test_empty_blob_returns_false(self) -> None:
        """b'' → False."""
        assert _has_us_providers(b"") is False

    def test_single_provider_returns_true(self) -> None:
        """One packed uint32 → True."""
        assert _has_us_providers(_pack_providers(8)) is True

    def test_multiple_providers_returns_true(self) -> None:
        """Three packed uint32s → True."""
        assert _has_us_providers(_pack_providers(8, 15, 337)) is True


# ---------------------------------------------------------------------------
# score_overview_length
# ---------------------------------------------------------------------------


class TestScoreOverviewLength:
    """Tests for the 5-tier overview-length scorer (rewritten tiers)."""

    def test_length_0_returns_zero(self) -> None:
        """length=0 → 0.0 (no overview)."""
        assert score_overview_length(0) == pytest.approx(0.0)

    def test_length_1_in_first_tier(self) -> None:
        """length=1 → 0.2."""
        assert score_overview_length(1) == pytest.approx(0.2)

    def test_length_50_at_first_tier_ceiling(self) -> None:
        """length=50 → 0.2."""
        assert score_overview_length(50) == pytest.approx(0.2)

    def test_length_51_enters_second_tier(self) -> None:
        """length=51 → 0.5."""
        assert score_overview_length(51) == pytest.approx(0.5)

    def test_length_100_at_second_tier_ceiling(self) -> None:
        """length=100 → 0.5."""
        assert score_overview_length(100) == pytest.approx(0.5)

    def test_length_101_enters_third_tier(self) -> None:
        """length=101 → 0.8."""
        assert score_overview_length(101) == pytest.approx(0.8)

    def test_length_200_at_third_tier_ceiling(self) -> None:
        """length=200 → 0.8."""
        assert score_overview_length(200) == pytest.approx(0.8)

    def test_length_201_enters_top_tier(self) -> None:
        """length=201 → 1.0."""
        assert score_overview_length(201) == pytest.approx(1.0)

    def test_long_overview_returns_one(self) -> None:
        """Arbitrarily long overview (10,000 chars) still returns 1.0."""
        assert score_overview_length(10_000) == pytest.approx(1.0)

    def test_strictly_increasing_across_boundaries(self) -> None:
        """Each tier boundary produces a higher score."""
        scores = [
            score_overview_length(0),
            score_overview_length(1),
            score_overview_length(51),
            score_overview_length(101),
            score_overview_length(201),
        ]
        for i in range(1, len(scores)):
            assert scores[i] > scores[i - 1]

    def test_five_distinct_output_values(self) -> None:
        """Exactly {0.0, 0.2, 0.5, 0.8, 1.0} are the possible outputs."""
        outputs = {score_overview_length(l) for l in [0, 1, 75, 150, 300]}
        assert outputs == {0.0, 0.2, 0.5, 0.8, 1.0}


# ---------------------------------------------------------------------------
# score_data_completeness
# ---------------------------------------------------------------------------


class TestScoreDataCompleteness:
    """Tests for the 8-indicator data-completeness scorer."""

    def test_all_present_returns_one(self) -> None:
        """All 8 indicators True → 1.0."""
        row = _make_row(
            genre_count=3, poster_url="url", has_cast_and_crew=1,
            has_production_countries=1, has_production_companies=1,
            has_keywords=1, has_budget=1, has_revenue=1,
        )
        assert score_data_completeness(row) == pytest.approx(1.0)

    def test_all_absent_returns_zero(self) -> None:
        """All 8 indicators False/0/None → 0.0."""
        row = _make_row(
            genre_count=0, poster_url=None, has_cast_and_crew=0,
            has_production_countries=0, has_production_companies=0,
            has_keywords=0, has_budget=0, has_revenue=0,
        )
        assert score_data_completeness(row) == pytest.approx(0.0)

    def test_half_present_returns_half(self) -> None:
        """4 of 8 → 0.5."""
        row = _make_row(
            genre_count=2, poster_url="url", has_cast_and_crew=1,
            has_production_countries=1, has_production_companies=0,
            has_keywords=0, has_budget=0, has_revenue=0,
        )
        assert score_data_completeness(row) == pytest.approx(0.5)

    def test_single_indicator_returns_one_eighth(self) -> None:
        """1 of 8 → 0.125."""
        row = _make_row(
            genre_count=1, poster_url=None, has_cast_and_crew=0,
            has_production_countries=0, has_production_companies=0,
            has_keywords=0, has_budget=0, has_revenue=0,
        )
        assert score_data_completeness(row) == pytest.approx(0.125)

    def test_genre_count_zero_is_false(self) -> None:
        """genre_count=0 → not counted as present."""
        row = _make_row(
            genre_count=0, poster_url=None, has_cast_and_crew=0,
            has_production_countries=0, has_production_companies=0,
            has_keywords=0, has_budget=0, has_revenue=0,
        )
        assert score_data_completeness(row) == pytest.approx(0.0)

    def test_genre_count_positive_is_true(self) -> None:
        """genre_count=3 → counted as present (contributes 0.125)."""
        row_with = _make_row(
            genre_count=3, poster_url=None, has_cast_and_crew=0,
            has_production_countries=0, has_production_companies=0,
            has_keywords=0, has_budget=0, has_revenue=0,
        )
        row_without = _make_row(
            genre_count=0, poster_url=None, has_cast_and_crew=0,
            has_production_countries=0, has_production_companies=0,
            has_keywords=0, has_budget=0, has_revenue=0,
        )
        assert score_data_completeness(row_with) - score_data_completeness(row_without) == pytest.approx(0.125)

    def test_poster_url_none_is_false(self) -> None:
        """poster_url=None → not counted."""
        row = _make_row(
            genre_count=0, poster_url=None, has_cast_and_crew=0,
            has_production_countries=0, has_production_companies=0,
            has_keywords=0, has_budget=0, has_revenue=0,
        )
        assert score_data_completeness(row) == pytest.approx(0.0)

    def test_poster_url_present_is_true(self) -> None:
        """poster_url='url' → counted (contributes 0.125)."""
        row = _make_row(
            genre_count=0, poster_url="url", has_cast_and_crew=0,
            has_production_countries=0, has_production_companies=0,
            has_keywords=0, has_budget=0, has_revenue=0,
        )
        assert score_data_completeness(row) == pytest.approx(0.125)


# ---------------------------------------------------------------------------
# compute_quality_score
# ---------------------------------------------------------------------------


class TestComputeQualityScore:
    """Tests for the edge-case + 4-signal weighted quality score."""

    def test_unreleased_movie_returns_zero(self) -> None:
        """Future release_date → 0.0 regardless of other signals."""
        future = (TODAY + datetime.timedelta(days=60)).isoformat()
        row = _default_scorer_row(release_date=future, watch_provider_keys=None)
        assert compute_quality_score(row, TODAY) == pytest.approx(0.0)

    def test_unreleased_checked_before_providers(self) -> None:
        """Future date + has providers → still 0.0 (unreleased takes precedence)."""
        future = (TODAY + datetime.timedelta(days=60)).isoformat()
        row = _default_scorer_row(
            release_date=future,
            watch_provider_keys=_pack_providers(8, 15, 337),
        )
        assert compute_quality_score(row, TODAY) == pytest.approx(0.0)

    def test_has_providers_returns_one(self) -> None:
        """Released + has providers → 1.0 regardless of other signals."""
        row = _default_scorer_row(watch_provider_keys=_pack_providers(8))
        assert compute_quality_score(row, TODAY) == pytest.approx(1.0)

    def test_has_providers_trumps_weak_signals(self) -> None:
        """Has providers + zero votes + no overview → still 1.0."""
        row = _default_scorer_row(
            vote_count=0,
            popularity=0.0,
            overview_length=0,
            watch_provider_keys=_pack_providers(42),
            genre_count=0,
            has_revenue=0,
            has_budget=0,
            has_production_companies=0,
            has_production_countries=0,
            has_keywords=0,
            has_cast_and_crew=0,
        )
        assert compute_quality_score(row, TODAY) == pytest.approx(1.0)

    def test_no_provider_formula_all_max(self) -> None:
        """No providers, all max signals → score near 1.0."""
        row = _default_scorer_row(
            vote_count=100,       # Saturates at log cap 101
            popularity=10.0,      # Saturates at POP_LOG_CAP
            overview_length=300,  # Top tier → 1.0
            # All completeness indicators present (default)
        )
        score = compute_quality_score(row, TODAY)
        assert score == pytest.approx(1.0, abs=0.05)

    def test_no_provider_formula_all_min(self) -> None:
        """No providers, all min signals → score near 0.0."""
        row = _default_scorer_row(
            vote_count=0,
            popularity=0.0,
            overview_length=0,
            genre_count=0,
            poster_url=None,
            has_revenue=0,
            has_budget=0,
            has_production_companies=0,
            has_production_countries=0,
            has_keywords=0,
            has_cast_and_crew=0,
        )
        score = compute_quality_score(row, TODAY)
        assert score == pytest.approx(0.0, abs=0.01)

    def test_no_provider_uses_tmdb_no_provider_source(self) -> None:
        """Verify score matches manual calc with TMDB_NO_PROVIDER log cap (101)."""
        row = _default_scorer_row(
            vote_count=50,
            popularity=3.0,
            overview_length=150,  # → 0.8
            release_date="2015-01-01",
        )
        score = compute_quality_score(row, TODAY)

        # Manual calculation using the same signal functions.
        vc_score = score_vote_count(50, "2015-01-01", TODAY, VoteCountSource.TMDB_NO_PROVIDER)
        pop_score = score_popularity(3.0)
        ol_score = 0.8   # 101-200 tier
        dc_score = 1.0   # All 8 indicators present in default row

        expected = (
            WEIGHTS["vote_count"] * vc_score
            + WEIGHTS["popularity"] * pop_score
            + WEIGHTS["overview_length"] * ol_score
            + WEIGHTS["data_completeness"] * dc_score
        )
        assert score == pytest.approx(expected, abs=1e-9)

    def test_null_release_date_not_treated_as_unreleased(self) -> None:
        """None date → not unreleased → formula runs."""
        row = _default_scorer_row(release_date=None)
        score = compute_quality_score(row, TODAY)
        # Should produce a positive score, not 0.0.
        assert score > 0.0

    def test_score_in_zero_one_range(self) -> None:
        """Output always in [0.0, 1.0] for formula path."""
        for vc in [0, 10, 50, 100]:
            for pop in [0.0, 1.0, 5.0]:
                for ol in [0, 50, 150, 300]:
                    row = _default_scorer_row(
                        vote_count=vc, popularity=pop, overview_length=ol,
                    )
                    score = compute_quality_score(row, TODAY)
                    assert 0.0 <= score <= 1.0

    def test_weight_contributions_sum_correctly(self) -> None:
        """Manual weighted sum matches compute_quality_score output."""
        row = _default_scorer_row(
            vote_count=30,
            popularity=2.0,
            overview_length=80,   # → 0.5
            release_date="2015-01-01",
        )
        score = compute_quality_score(row, TODAY)

        vc_score = score_vote_count(30, "2015-01-01", TODAY, VoteCountSource.TMDB_NO_PROVIDER)
        pop_score = score_popularity(2.0)
        ol_score = 0.5
        dc_score = 1.0  # All indicators present

        expected = (
            WEIGHTS["vote_count"] * vc_score
            + WEIGHTS["popularity"] * pop_score
            + WEIGHTS["overview_length"] * ol_score
            + WEIGHTS["data_completeness"] * dc_score
        )
        assert score == pytest.approx(expected, abs=1e-9)

    def test_each_signal_independently_affects_score(self) -> None:
        """Toggling each of the 4 signals changes the score."""
        toggles = [
            ("vote_count",      0,   100),
            ("popularity",      0.0, 10.0),
            ("overview_length", 0,   300),
        ]
        for field, low, high in toggles:
            low_score = compute_quality_score(_default_scorer_row(**{field: low}), TODAY)
            high_score = compute_quality_score(_default_scorer_row(**{field: high}), TODAY)
            assert high_score > low_score, (
                f"Field '{field}' had no effect — weight may be zero or signal logic is broken"
            )

        # Data completeness: toggle all indicators off vs on.
        low_row = _default_scorer_row(
            genre_count=0, poster_url=None, has_revenue=0, has_budget=0,
            has_production_companies=0, has_production_countries=0,
            has_keywords=0, has_cast_and_crew=0,
        )
        high_row = _default_scorer_row()  # All indicators present
        assert compute_quality_score(high_row, TODAY) > compute_quality_score(low_row, TODAY)

    def test_vote_count_has_largest_weight_impact(self) -> None:
        """vote_count toggle produces the largest score delta (weight=0.50)."""
        deltas = {}

        deltas["vote_count"] = (
            compute_quality_score(_default_scorer_row(vote_count=100), TODAY)
            - compute_quality_score(_default_scorer_row(vote_count=0), TODAY)
        )
        deltas["popularity"] = (
            compute_quality_score(_default_scorer_row(popularity=10.0), TODAY)
            - compute_quality_score(_default_scorer_row(popularity=0.0), TODAY)
        )
        deltas["overview_length"] = (
            compute_quality_score(_default_scorer_row(overview_length=300), TODAY)
            - compute_quality_score(_default_scorer_row(overview_length=0), TODAY)
        )

        low_dc = _default_scorer_row(
            genre_count=0, poster_url=None, has_revenue=0, has_budget=0,
            has_production_companies=0, has_production_countries=0,
            has_keywords=0, has_cast_and_crew=0,
        )
        high_dc = _default_scorer_row()
        deltas["data_completeness"] = (
            compute_quality_score(high_dc, TODAY)
            - compute_quality_score(low_dc, TODAY)
        )

        assert deltas["vote_count"] == max(deltas.values())


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

    def test_weights_has_four_entries(self) -> None:
        """WEIGHTS contains exactly 4 signals."""
        assert len(WEIGHTS) == 4

    def test_weights_contains_expected_keys(self) -> None:
        """WEIGHTS keys are {vote_count, popularity, overview_length, data_completeness}."""
        expected = {"vote_count", "popularity", "overview_length", "data_completeness"}
        assert set(WEIGHTS.keys()) == expected

    def test_vote_count_is_dominant_weight(self) -> None:
        """vote_count weight ≥ all others."""
        vc_weight = WEIGHTS["vote_count"]
        for key, val in WEIGHTS.items():
            assert vc_weight >= val, f"vote_count weight ({vc_weight}) < {key} weight ({val})"


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

    def test_status_set_to_tmdb_quality_calculated(self, run_db) -> None:
        """Status changes from tmdb_fetched → tmdb_quality_calculated."""
        _seed_movie(run_db, 301)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        assert _read_status(run_db, 301) == "tmdb_quality_calculated"

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

    def test_idempotent(self, run_db) -> None:
        """Second run produces identical scores."""
        _seed_movie(run_db, 501)

        # First run.
        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()
        score_first = _read_score(run_db, 501)

        # Reset status to tmdb_fetched so second run picks it up.
        conn = sqlite3.connect(str(run_db))
        conn.execute("UPDATE movie_progress SET status = 'tmdb_fetched' WHERE tmdb_id = 501")
        conn.commit()
        conn.close()

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()
        score_second = _read_score(run_db, 501)

        assert score_first == pytest.approx(score_second, abs=1e-9)

    def test_batch_commit_correctness(self, run_db) -> None:
        """Scores are flushed for all movies including across COMMIT_EVERY boundary."""
        # Insert COMMIT_EVERY + 1 movies to ensure the partial-batch flush works.
        n = COMMIT_EVERY + 1
        for tmdb_id in range(1, n + 1):
            _seed_movie(run_db, tmdb_id)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        conn = sqlite3.connect(str(run_db))
        scored_count = conn.execute(
            "SELECT COUNT(*) FROM movie_progress WHERE stage_3_quality_score IS NOT NULL"
        ).fetchone()[0]
        conn.close()
        assert scored_count == n

    def test_unreleased_movie_scored_zero(self, run_db) -> None:
        """run() persists 0.0 for a future-dated movie."""
        future = (datetime.date.today() + datetime.timedelta(days=60)).isoformat()
        _seed_movie(run_db, 601, release_date=future)

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        assert _read_score(run_db, 601) == pytest.approx(0.0)

    def test_provider_movie_scored_one(self, run_db) -> None:
        """run() persists 1.0 for a movie with US watch providers."""
        _seed_movie(run_db, 701, watch_provider_keys=_pack_providers(8, 15))

        with patch("movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer.init_db",
                   side_effect=lambda: _make_run_conn(run_db)):
            run()

        assert _read_score(run_db, 701) == pytest.approx(1.0)
