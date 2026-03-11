"""
Unit tests for movie_ingestion.scoring_utils — shared scoring utilities.

Covers:
  - VoteCountSource enum
  - unpack_provider_keys  — BLOB decoder
  - score_vote_count      — log-scale + recency/classic age multipliers (TMDB and IMDB sources)
  - score_popularity      — log-scale
  - validate_weights      — weight sum guard
"""

import datetime
import math
import struct

import pytest

from movie_ingestion.scoring_utils import (
    POP_LOG_CAP,
    THEATER_WINDOW_DAYS,
    VC_CLASSIC_BOOST_CAP,
    VC_CLASSIC_RAMP_YEARS,
    VC_CLASSIC_START_YEARS,
    VC_RECENCY_BOOST_MAX,
    VoteCountSource,
    _VC_LOG_CAPS,
    score_popularity,
    score_vote_count,
    unpack_provider_keys,
    validate_weights,
)


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

# Fixed reference date used throughout to make all tests deterministic.
TODAY = datetime.date(2026, 3, 5)

# Convenience log caps for assertions.
TMDB_LOG_CAP = _VC_LOG_CAPS[VoteCountSource.TMDB]   # 2001
IMDB_LOG_CAP = _VC_LOG_CAPS[VoteCountSource.IMDB]   # 12001


# ---------------------------------------------------------------------------
# VoteCountSource enum
# ---------------------------------------------------------------------------


class TestVoteCountSource:
    """Tests for the VoteCountSource StrEnum."""

    def test_vote_count_source_tmdb_value(self) -> None:
        """VoteCountSource.TMDB has string value 'tmdb'."""
        assert VoteCountSource.TMDB == "tmdb"

    def test_vote_count_source_imdb_value(self) -> None:
        """VoteCountSource.IMDB has string value 'imdb'."""
        assert VoteCountSource.IMDB == "imdb"

    def test_tmdb_no_provider_value(self) -> None:
        """VoteCountSource.TMDB_NO_PROVIDER has string value 'tmdb_no_provider'."""
        assert VoteCountSource.TMDB_NO_PROVIDER == "tmdb_no_provider"

    def test_tmdb_no_provider_log_cap_is_101(self) -> None:
        """TMDB_NO_PROVIDER log cap is 101 (just above p99 ≈ 72 for no-provider population)."""
        assert _VC_LOG_CAPS[VoteCountSource.TMDB_NO_PROVIDER] == 101

    def test_tmdb_no_provider_saturates_at_100(self) -> None:
        """vc=100 with TMDB_NO_PROVIDER source produces a score ≈ 1.0."""
        score = score_vote_count(100, None, TODAY, VoteCountSource.TMDB_NO_PROVIDER)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_tmdb_no_provider_discriminates_low_range(self) -> None:
        """vc=10 with TMDB_NO_PROVIDER produces higher score than same vc with TMDB source.

        The lower cap (101 vs 2001) spreads discrimination across the low range,
        giving more resolution where the no-provider population lives.
        """
        no_prov_score = score_vote_count(10, None, TODAY, VoteCountSource.TMDB_NO_PROVIDER)
        tmdb_score = score_vote_count(10, None, TODAY, VoteCountSource.TMDB)
        assert no_prov_score > tmdb_score

    def test_all_three_sources_in_enum(self) -> None:
        """VoteCountSource enum has exactly 3 members."""
        assert len(VoteCountSource) == 3

    def test_vote_count_source_is_strenum(self) -> None:
        """All members are valid strings (StrEnum contract)."""
        for member in VoteCountSource:
            assert isinstance(member, str)
            assert member == member.value


# ---------------------------------------------------------------------------
# unpack_provider_keys
# ---------------------------------------------------------------------------


class TestUnpackProviderKeys:
    """Tests for the watch_provider_keys BLOB decoder."""

    def test_unpack_none_returns_empty_list(self) -> None:
        """None input returns an empty list."""
        assert unpack_provider_keys(None) == []

    def test_unpack_empty_bytes_returns_empty_list(self) -> None:
        """Zero-length bytes returns an empty list."""
        assert unpack_provider_keys(b"") == []

    def test_unpack_single_provider(self) -> None:
        """Single 4-byte packed integer is decoded to a one-element list."""
        blob = struct.pack("<1I", 42)
        assert unpack_provider_keys(blob) == [42]

    def test_unpack_multiple_providers(self) -> None:
        """Multiple packed integers are decoded in insertion order."""
        blob = struct.pack("<3I", 8, 15, 337)
        assert unpack_provider_keys(blob) == [8, 15, 337]

    def test_unpack_truncates_trailing_bytes(self) -> None:
        """A BLOB whose length is not a multiple of 4 has trailing bytes dropped."""
        # 5 bytes = 1 complete int + 1 leftover byte
        blob = struct.pack("<1I", 99) + b"\xff"
        assert unpack_provider_keys(blob) == [99]

    def test_unpack_preserves_order(self) -> None:
        """Output order matches byte order in BLOB."""
        ids = [100, 200, 300, 400, 500]
        blob = struct.pack(f"<{len(ids)}I", *ids)
        assert unpack_provider_keys(blob) == ids


# ---------------------------------------------------------------------------
# score_vote_count
# ---------------------------------------------------------------------------


class TestScoreVoteCount:
    """Tests for the log-scaled vote_count scorer with recency/classic multipliers."""

    # ---- Base score tests ----

    def test_zero_votes_returns_zero(self) -> None:
        """vc=0 → log10(1)=0 → score=0.0 regardless of source."""
        assert score_vote_count(0, None, TODAY, VoteCountSource.TMDB) == pytest.approx(0.0)
        assert score_vote_count(0, None, TODAY, VoteCountSource.IMDB) == pytest.approx(0.0)

    def test_one_vote_small_positive(self) -> None:
        """vc=1 → small positive value."""
        score = score_vote_count(1, None, TODAY, VoteCountSource.TMDB)
        assert 0.0 < score < 0.2

    def test_at_tmdb_cap_saturates(self) -> None:
        """vc=2000, source=TMDB, no date → approximately 1.0."""
        score = score_vote_count(2000, None, TODAY, VoteCountSource.TMDB)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_at_imdb_cap_saturates(self) -> None:
        """vc=12000, source=IMDB, no date → approximately 1.0."""
        score = score_vote_count(12000, None, TODAY, VoteCountSource.IMDB)
        assert score == pytest.approx(1.0, abs=0.01)

    def test_above_cap_clamps_to_one(self) -> None:
        """vc=50000 → exactly 1.0."""
        assert score_vote_count(50000, None, TODAY, VoteCountSource.TMDB) == pytest.approx(1.0)
        assert score_vote_count(50000, None, TODAY, VoteCountSource.IMDB) == pytest.approx(1.0)

    def test_tmdb_vs_imdb_same_vc_different_score(self) -> None:
        """Same vc produces higher score with TMDB cap (lower cap) than IMDB cap."""
        vc = 500
        tmdb_score = score_vote_count(vc, None, TODAY, VoteCountSource.TMDB)
        imdb_score = score_vote_count(vc, None, TODAY, VoteCountSource.IMDB)
        assert tmdb_score > imdb_score

    # ---- Release date / multiplier tests ----

    def test_no_release_date_no_multiplier(self) -> None:
        """release_date=None → base score without adjustment."""
        expected = math.log10(200 + 1) / math.log10(TMDB_LOG_CAP)
        assert score_vote_count(200, None, TODAY, VoteCountSource.TMDB) == pytest.approx(expected, abs=1e-9)

    def test_recency_boost_under_one_year(self) -> None:
        """Film < 1yr old gets ~2x multiplier (score higher than same vc without boost)."""
        six_months_ago = (TODAY - datetime.timedelta(days=180)).isoformat()
        base = score_vote_count(100, None, TODAY, VoteCountSource.TMDB)
        boosted = score_vote_count(100, six_months_ago, TODAY, VoteCountSource.TMDB)
        assert boosted > base
        # Should be close to 2x base (capped at 1.0)
        assert boosted == pytest.approx(min(base * VC_RECENCY_BOOST_MAX, 1.0), abs=0.05)

    def test_recency_boost_at_two_years_decays_to_one(self) -> None:
        """Film exactly 2yr old → multiplier ≈ 1.0."""
        date_2yr = TODAY.replace(year=TODAY.year - 2).isoformat()
        base = score_vote_count(100, None, TODAY, VoteCountSource.TMDB)
        result = score_vote_count(100, date_2yr, TODAY, VoteCountSource.TMDB)
        # max(1.0, min(2.0, 2.0/2.0)) = 1.0 → score unchanged
        assert result == pytest.approx(base, abs=0.01)

    def test_classic_boost_at_20_years(self) -> None:
        """Film 20yr old → multiplier = 1.0 (boost just starting, ramp term = 0)."""
        date_20yr = TODAY.replace(year=TODAY.year - VC_CLASSIC_START_YEARS).isoformat()
        base = score_vote_count(100, None, TODAY, VoteCountSource.TMDB)
        result = score_vote_count(100, date_20yr, TODAY, VoteCountSource.TMDB)
        assert result == pytest.approx(base, abs=0.01)

    def test_classic_boost_at_50_years_reaches_cap(self) -> None:
        """Film 50yr old → multiplier = 1.5 (capped)."""
        full_age = VC_CLASSIC_START_YEARS + VC_CLASSIC_RAMP_YEARS  # 50yr
        year = TODAY.year - full_age
        date = f"{year}-01-01"
        base = score_vote_count(100, None, TODAY, VoteCountSource.TMDB)
        boosted = score_vote_count(100, date, TODAY, VoteCountSource.TMDB)
        assert boosted == pytest.approx(min(base * VC_CLASSIC_BOOST_CAP, 1.0), abs=0.01)

    def test_classic_boost_beyond_cap(self) -> None:
        """Film 80yr old → multiplier still 1.5 (not growing beyond cap)."""
        base = score_vote_count(100, None, TODAY, VoteCountSource.TMDB)
        ancient = score_vote_count(100, "1946-01-01", TODAY, VoteCountSource.TMDB)
        assert ancient <= min(base * VC_CLASSIC_BOOST_CAP, 1.0) + 1e-9

    def test_middle_age_no_adjustment(self) -> None:
        """Film 10yr old → no recency or classic boost."""
        date_10yr = TODAY.replace(year=TODAY.year - 10).isoformat()
        base = score_vote_count(100, None, TODAY, VoteCountSource.TMDB)
        result = score_vote_count(100, date_10yr, TODAY, VoteCountSource.TMDB)
        assert result == pytest.approx(base, abs=1e-6)

    def test_invalid_date_string_uses_base(self) -> None:
        """release_date='not-a-date' → base score unchanged."""
        base = score_vote_count(200, None, TODAY, VoteCountSource.TMDB)
        result = score_vote_count(200, "not-a-date", TODAY, VoteCountSource.TMDB)
        assert result == pytest.approx(base, abs=1e-9)

    def test_result_never_exceeds_one(self) -> None:
        """High vc + max recency boost → clamped at 1.0."""
        recent = (TODAY - datetime.timedelta(days=200)).isoformat()
        assert score_vote_count(1500, recent, TODAY, VoteCountSource.TMDB) <= 1.0

    def test_age_floor_at_half_year(self) -> None:
        """Very new film (days=10) → age_years floored at 0.5, giving 2x recency boost."""
        ten_days_ago = (TODAY - datetime.timedelta(days=10)).isoformat()
        base = score_vote_count(100, None, TODAY, VoteCountSource.TMDB)
        boosted = score_vote_count(100, ten_days_ago, TODAY, VoteCountSource.TMDB)
        # Floor at 0.5yr → multiplier = min(2.0, 2.0/0.5) = min(2.0, 4.0) = 2.0
        assert boosted == pytest.approx(min(base * VC_RECENCY_BOOST_MAX, 1.0), abs=1e-6)


# ---------------------------------------------------------------------------
# score_popularity
# ---------------------------------------------------------------------------


class TestScorePopularity:
    """Tests for the log-scaled popularity scorer."""

    def test_zero_popularity_returns_zero(self) -> None:
        """popularity=0 → log10(1)=0 → score 0.0."""
        assert score_popularity(0.0) == pytest.approx(0.0)

    def test_negative_popularity_returns_zero(self) -> None:
        """popularity=-5 → max(val, 0) guard → score 0.0."""
        assert score_popularity(-5.0) == pytest.approx(0.0)

    def test_popularity_at_cap(self) -> None:
        """popularity=10 → approximately 1.0."""
        assert score_popularity(10.0) == pytest.approx(1.0, abs=0.01)

    def test_popularity_above_cap_clamps(self) -> None:
        """popularity=1000 → exactly 1.0."""
        assert score_popularity(1000.0) == pytest.approx(1.0)

    def test_popularity_one_intermediate(self) -> None:
        """popularity=1.0 → intermediate value between 0 and 1."""
        score = score_popularity(1.0)
        expected = math.log10(2) / math.log10(POP_LOG_CAP)
        assert score == pytest.approx(expected, abs=1e-9)
        assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# validate_weights
# ---------------------------------------------------------------------------


class TestValidateWeights:
    """Tests for the weight sum guard."""

    def test_valid_weights_passes(self) -> None:
        """Weights summing to exactly 1.0 → no exception."""
        validate_weights({"a": 0.6, "b": 0.4})

    def test_weights_summing_above_one_raises(self) -> None:
        """sum=1.1 → ValueError."""
        with pytest.raises(ValueError):
            validate_weights({"a": 0.6, "b": 0.5})

    def test_weights_summing_below_one_raises(self) -> None:
        """sum=0.9 → ValueError."""
        with pytest.raises(ValueError):
            validate_weights({"a": 0.5, "b": 0.4})

    def test_epsilon_tolerance(self) -> None:
        """sum=1.0 + 1e-10 → passes (within tolerance)."""
        # Construct weights that sum to 1.0 + 1e-10 (within 1e-9 tolerance)
        validate_weights({"a": 0.5, "b": 0.5 + 1e-10})

    def test_error_message_includes_label(self) -> None:
        """Custom label appears in the error message."""
        with pytest.raises(ValueError, match="MyLabel"):
            validate_weights({"a": 0.5}, label="MyLabel")

    def test_empty_weights_raises(self) -> None:
        """{} (sum=0) → ValueError."""
        with pytest.raises(ValueError):
            validate_weights({})
