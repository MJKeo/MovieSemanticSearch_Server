"""
Unit tests for movie_ingestion.imdb_quality_scoring.imdb_quality_scorer -- Stage 5.

Covers:
  - MovieContext dataclass
  - 8 signal scoring functions (all normalised to [0, 1])
  - compute_imdb_quality_score -- weighted composite
  - WEIGHTS module-level constant
"""

import datetime
import json
import math

import pytest

from movie_ingestion.imdb_quality_scoring.imdb_quality_scorer import (
    ALLOWED_TITLE_TYPES,
    BAYESIAN_C,
    BAYESIAN_M,
    BLEND_HIGH,
    BLEND_LOW,
    BLEND_MED,
    NOTABILITY_VC_LOG_CAP,
    PLOT_TEXT_LOG_CAP,
    STAGE5_POP_LOG_CAP,
    VC_TIER_LOW_CEILING,
    VC_TIER_MED_CEILING,
    WEIGHTS,
    MovieContext,
    _score_community_engagement,
    _score_critical_attention,
    _score_data_completeness,
    _score_featured_reviews_chars,
    _score_imdb_notability,
    _score_lexical_completeness,
    _score_plot_text_depth,
    _score_tmdb_popularity,
    compute_imdb_quality_score,
)
from movie_ingestion.scoring_utils import (
    VC_CLASSIC_BOOST_CAP,
    VC_CLASSIC_RAMP_YEARS,
    VC_CLASSIC_START_YEARS,
    VC_RECENCY_BOOST_MAX,
)


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

# Fixed reference date used throughout to make all tests deterministic.
TODAY = datetime.date(2026, 3, 5)


def _make_ctx(
    tmdb_overrides: dict | None = None,
    imdb_overrides: dict | None = None,
) -> MovieContext:
    """Build a MovieContext with sensible defaults for a high-scoring movie.

    All fields are populated so every signal returns a high score by default.
    Use tmdb_overrides/imdb_overrides to target specific fields for testing.
    """
    tmdb = {
        "tmdb_id": 12345,
        "release_date": "2020-06-15",
        "popularity": 5.0,
        "overview_length": 180,
        "maturity_rating": "PG-13",
        "budget": 50_000_000,
        "reviews": json.dumps(["A solid film with great performances."]),
        "has_production_companies": 1,
    }
    if tmdb_overrides:
        tmdb.update(tmdb_overrides)

    imdb = {
        "imdb_title_type": "movie",
        "imdb_rating": 7.2,
        "imdb_vote_count": 5000,
        "directors": ["Christopher Nolan"],
        "actors": [f"Actor{i}" for i in range(1, 11)],       # 10 for full sub-score
        "characters": [f"Char{i}" for i in range(1, 11)],     # 10 for full sub-score
        "overall_keywords": [f"kw{i}" for i in range(1, 7)],  # 6 for full sub-score
        "languages": ["English"],
        "countries_of_origin": ["United States"],
        "writers": ["Writer1"],
        "producers": ["Producer1"],
        "production_companies": ["Company1"],
        "featured_reviews": [
            {"text": "A" * 5000},
            {"text": "B" * 5000},
        ],
        "overview": "A detailed overview of the movie plot.",
        "plot_summaries": ["A longer plot summary with more details." * 3],
        "synopses": ["A full synopsis of the movie." * 5],
        "plot_keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"],
        "parental_guide_items": [
            {"category": "violence", "severity": "moderate"},
            {"category": "profanity", "severity": "mild"},
            {"category": "sex", "severity": "none"},
        ],  # 3 for full sub-score
        "maturity_rating": "R",
        "budget": 100_000_000,
        "metacritic_rating": 73,
        "reception_summary": "Widely acclaimed thriller.",
    }
    if imdb_overrides:
        imdb.update(imdb_overrides)

    return MovieContext(tmdb_id=tmdb["tmdb_id"], tmdb=tmdb, imdb=imdb)


# ---------------------------------------------------------------------------
# MovieContext dataclass
# ---------------------------------------------------------------------------


class TestMovieContext:
    """Tests for the MovieContext data container."""

    def test_movie_context_with_both_sources(self) -> None:
        """Constructs with tmdb dict and imdb dict."""
        ctx = _make_ctx()
        assert ctx.tmdb_id == 12345
        assert ctx.tmdb is not None
        assert ctx.imdb is not None


# ---------------------------------------------------------------------------
# Signal scoring functions — Relevance signals
# ---------------------------------------------------------------------------


class TestScoreImdbNotability:
    """Tests for _score_imdb_notability signal.

    Blends log-scaled vote count with Bayesian-adjusted IMDB rating using
    3 confidence tiers, then applies age multipliers (recency or classic).
    """

    # --- Zero/missing input ---

    def test_zero_votes_no_rating_returns_zero(self) -> None:
        """vc=0, no rating -> 0.0."""
        ctx = _make_ctx(imdb_overrides={"imdb_vote_count": 0, "imdb_rating": None})
        assert _score_imdb_notability(ctx, TODAY) == pytest.approx(0.0)

    def test_zero_votes_with_rating_returns_zero(self) -> None:
        """vc=0 with rating still returns 0.0 (vc > 0 check gates rating blend)."""
        ctx = _make_ctx(imdb_overrides={"imdb_vote_count": 0, "imdb_rating": 9.0})
        assert _score_imdb_notability(ctx, TODAY) == pytest.approx(0.0)

    # --- Pure vote-count fallback (no rating) ---

    def test_missing_rating_uses_pure_vote_base(self) -> None:
        """imdb_rating=None degrades to pure log-scaled vote count."""
        ctx = _make_ctx(imdb_overrides={"imdb_vote_count": 1000, "imdb_rating": None})
        score = _score_imdb_notability(ctx, TODAY)
        expected = math.log10(1001) / math.log10(NOTABILITY_VC_LOG_CAP)
        assert score == pytest.approx(expected, abs=1e-4)

    # --- Low confidence tier (vc < 100): almost pure vote count ---

    def test_low_tier_rating_barely_moves_score(self) -> None:
        """At vc=50, the 95/5 blend means rating has minimal influence."""
        vc = 50
        rating = 9.0
        ctx = _make_ctx(imdb_overrides={"imdb_vote_count": vc, "imdb_rating": rating})
        score = _score_imdb_notability(ctx, TODAY)

        # Compute expected: vote base with 95/5 blend.
        vote_base = math.log10(vc + 1) / math.log10(NOTABILITY_VC_LOG_CAP)
        bayesian = (vc / (vc + BAYESIAN_M)) * rating + (BAYESIAN_M / (vc + BAYESIAN_M)) * BAYESIAN_C
        rating_norm = bayesian / 10.0
        vote_w, rating_w = BLEND_LOW
        expected = vote_w * vote_base + rating_w * rating_norm
        assert score == pytest.approx(expected, abs=1e-4)

        # Rating component is tiny — score is very close to pure vote base.
        assert abs(score - vote_base) < 0.03

    def test_low_tier_boundary_at_99(self) -> None:
        """vc=99 uses low tier (< 100)."""
        ctx = _make_ctx(imdb_overrides={"imdb_vote_count": 99, "imdb_rating": 8.0})
        score = _score_imdb_notability(ctx, TODAY)

        vote_base = math.log10(100) / math.log10(NOTABILITY_VC_LOG_CAP)
        bayesian = (99 / (99 + BAYESIAN_M)) * 8.0 + (BAYESIAN_M / (99 + BAYESIAN_M)) * BAYESIAN_C
        rating_norm = bayesian / 10.0
        vote_w, rating_w = BLEND_LOW
        expected = vote_w * vote_base + rating_w * rating_norm
        assert score == pytest.approx(expected, abs=1e-4)

    # --- Medium confidence tier (100-999): rating has meaningful influence ---

    def test_med_tier_boundary_at_100(self) -> None:
        """vc=100 uses medium tier (>= 100)."""
        ctx = _make_ctx(imdb_overrides={"imdb_vote_count": 100, "imdb_rating": 8.0})
        score = _score_imdb_notability(ctx, TODAY)

        vote_base = math.log10(101) / math.log10(NOTABILITY_VC_LOG_CAP)
        bayesian = (100 / (100 + BAYESIAN_M)) * 8.0 + (BAYESIAN_M / (100 + BAYESIAN_M)) * BAYESIAN_C
        rating_norm = bayesian / 10.0
        vote_w, rating_w = BLEND_MED
        expected = vote_w * vote_base + rating_w * rating_norm
        assert score == pytest.approx(expected, abs=1e-4)

    def test_med_tier_rating_has_influence(self) -> None:
        """At medium tier, changing rating from 9.0 to 3.0 visibly changes score."""
        ctx_high = _make_ctx(imdb_overrides={"imdb_vote_count": 500, "imdb_rating": 9.0})
        ctx_low = _make_ctx(imdb_overrides={"imdb_vote_count": 500, "imdb_rating": 3.0})
        score_high = _score_imdb_notability(ctx_high, TODAY)
        score_low = _score_imdb_notability(ctx_low, TODAY)
        # 30% rating weight at medium tier means a 6-point rating difference
        # should produce a noticeable delta.
        assert score_high > score_low
        assert (score_high - score_low) > 0.05

    def test_tier_jump_at_100_changes_blend(self) -> None:
        """vc=99 (low tier) vs vc=100 (med tier) produce different scores
        despite nearly identical vote counts, because the blend weights change."""
        rating = 8.0
        ctx_99 = _make_ctx(imdb_overrides={"imdb_vote_count": 99, "imdb_rating": rating})
        ctx_100 = _make_ctx(imdb_overrides={"imdb_vote_count": 100, "imdb_rating": rating})
        score_99 = _score_imdb_notability(ctx_99, TODAY)
        score_100 = _score_imdb_notability(ctx_100, TODAY)
        # With a good rating (8.0), the medium tier's higher rating weight (30%
        # vs 5%) should push the score UP relative to the low tier, producing
        # a jump beyond what the 1-vote difference would normally yield.
        assert score_100 > score_99

    # --- High confidence tier (>= 1000): vote count dominates ---

    def test_high_tier_boundary_at_1000(self) -> None:
        """vc=1000 uses high tier (>= 1000)."""
        ctx = _make_ctx(imdb_overrides={"imdb_vote_count": 1000, "imdb_rating": 8.0})
        score = _score_imdb_notability(ctx, TODAY)

        vote_base = math.log10(1001) / math.log10(NOTABILITY_VC_LOG_CAP)
        bayesian = (1000 / (1000 + BAYESIAN_M)) * 8.0 + (BAYESIAN_M / (1000 + BAYESIAN_M)) * BAYESIAN_C
        rating_norm = bayesian / 10.0
        vote_w, rating_w = BLEND_HIGH
        expected = vote_w * vote_base + rating_w * rating_norm
        assert score == pytest.approx(expected, abs=1e-4)

    def test_high_tier_rating_less_influence_than_med(self) -> None:
        """High tier (85/15) gives less rating influence than medium (70/30)."""
        # Use the same vc for both — force into different tiers by choosing
        # a vc at the boundary. Instead, compare delta-from-rating at each tier.
        vc_med, vc_high = 500, 5000
        for vc in (vc_med, vc_high):
            ctx_r9 = _make_ctx(imdb_overrides={"imdb_vote_count": vc, "imdb_rating": 9.0})
            ctx_r3 = _make_ctx(imdb_overrides={"imdb_vote_count": vc, "imdb_rating": 3.0})
            score_r9 = _score_imdb_notability(ctx_r9, TODAY)
            score_r3 = _score_imdb_notability(ctx_r3, TODAY)
            if vc == vc_med:
                delta_med = score_r9 - score_r3
            else:
                delta_high = score_r9 - score_r3
        # Medium tier should show larger rating delta than high tier.
        assert delta_med > delta_high

    # --- Bayesian shrinkage ---

    def test_bayesian_shrinks_toward_mean(self) -> None:
        """Low vc shrinks bayesian toward C=6.0; high vc preserves actual rating."""
        rating = 9.0
        # At vc=1 (extreme low), bayesian is almost entirely the prior (C=6.0).
        bayesian_low = (1 / (1 + BAYESIAN_M)) * rating + (BAYESIAN_M / (1 + BAYESIAN_M)) * BAYESIAN_C
        assert bayesian_low == pytest.approx(BAYESIAN_C, abs=0.1)  # ~6.006

        # At vc=10000 (extreme high), bayesian is almost the actual rating.
        bayesian_high = (10000 / (10000 + BAYESIAN_M)) * rating + (BAYESIAN_M / (10000 + BAYESIAN_M)) * BAYESIAN_C
        assert bayesian_high == pytest.approx(rating, abs=0.2)  # ~8.86

    # --- Age multipliers: recency boost ---

    def test_recency_boost_recent_release(self) -> None:
        """Release 6 months ago -> 2x multiplier -> higher score."""
        recent_date = (TODAY - datetime.timedelta(days=180)).isoformat()
        old_date = "2020-06-15"  # ~5.7yr, no boost

        ctx_recent = _make_ctx(
            imdb_overrides={"imdb_vote_count": 50, "imdb_rating": None},
            tmdb_overrides={"release_date": recent_date},
        )
        ctx_old = _make_ctx(
            imdb_overrides={"imdb_vote_count": 50, "imdb_rating": None},
            tmdb_overrides={"release_date": old_date},
        )
        score_recent = _score_imdb_notability(ctx_recent, TODAY)
        score_old = _score_imdb_notability(ctx_old, TODAY)
        # Recent release gets ~2x boost.
        assert score_recent > score_old
        assert score_recent == pytest.approx(score_old * VC_RECENCY_BOOST_MAX, abs=0.01)

    def test_recency_boost_decays_at_two_years(self) -> None:
        """Release 2 years ago -> multiplier ~1.0 -> no boost."""
        two_years_ago = (TODAY - datetime.timedelta(days=730)).isoformat()
        five_years_ago = "2021-03-05"

        ctx_2yr = _make_ctx(
            imdb_overrides={"imdb_vote_count": 50, "imdb_rating": None},
            tmdb_overrides={"release_date": two_years_ago},
        )
        ctx_5yr = _make_ctx(
            imdb_overrides={"imdb_vote_count": 50, "imdb_rating": None},
            tmdb_overrides={"release_date": five_years_ago},
        )
        score_2yr = _score_imdb_notability(ctx_2yr, TODAY)
        score_5yr = _score_imdb_notability(ctx_5yr, TODAY)
        # At 2 years, boost is 2.0/2.0 = 1.0 — effectively no boost.
        assert score_2yr == pytest.approx(score_5yr, abs=0.01)

    # --- Age multipliers: classic boost ---

    def test_classic_boost_at_30_years(self) -> None:
        """30-year-old film gets classic boost of ~1.33x."""
        thirty_years_ago = "1996-03-05"
        ctx = _make_ctx(
            imdb_overrides={"imdb_vote_count": 100, "imdb_rating": None},
            tmdb_overrides={"release_date": thirty_years_ago},
        )
        score = _score_imdb_notability(ctx, TODAY)

        # Without boost, base score is pure vote_base.
        vote_base = math.log10(101) / math.log10(NOTABILITY_VC_LOG_CAP)
        # Classic boost at 30yr: 1.0 + (30 - 20) / 30 = 1.333
        age_years = (TODAY - datetime.date(1996, 3, 5)).days / 365.0
        classic = min(VC_CLASSIC_BOOST_CAP, 1.0 + max(0, age_years - VC_CLASSIC_START_YEARS) / VC_CLASSIC_RAMP_YEARS)
        expected = min(vote_base * classic, 1.0)
        assert score == pytest.approx(expected, abs=1e-3)

    def test_classic_boost_caps_at_50_years(self) -> None:
        """50-year-old film gets maximum classic boost of 1.5x."""
        fifty_years_ago = "1976-03-05"
        age_years = (TODAY - datetime.date(1976, 3, 5)).days / 365.0
        classic = min(VC_CLASSIC_BOOST_CAP, 1.0 + max(0, age_years - VC_CLASSIC_START_YEARS) / VC_CLASSIC_RAMP_YEARS)
        assert classic == pytest.approx(VC_CLASSIC_BOOST_CAP, abs=0.01)

    def test_no_age_boost_at_10_years(self) -> None:
        """10-year-old film gets no age boost (neither recency nor classic)."""
        ten_years_ago = "2016-03-05"
        ctx_10yr = _make_ctx(
            imdb_overrides={"imdb_vote_count": 100, "imdb_rating": None},
            tmdb_overrides={"release_date": ten_years_ago},
        )
        # Compare to a 5-year-old film — both should have multiplier 1.0.
        ctx_5yr = _make_ctx(
            imdb_overrides={"imdb_vote_count": 100, "imdb_rating": None},
            tmdb_overrides={"release_date": "2021-03-05"},
        )
        score_10yr = _score_imdb_notability(ctx_10yr, TODAY)
        score_5yr = _score_imdb_notability(ctx_5yr, TODAY)
        assert score_10yr == pytest.approx(score_5yr, abs=1e-4)

    # --- Edge cases ---

    def test_result_capped_at_one(self) -> None:
        """Very high vc + recent release -> capped at 1.0."""
        recent = (TODAY - datetime.timedelta(days=90)).isoformat()
        ctx = _make_ctx(
            imdb_overrides={"imdb_vote_count": 10000, "imdb_rating": 9.0},
            tmdb_overrides={"release_date": recent},
        )
        score = _score_imdb_notability(ctx, TODAY)
        assert score == pytest.approx(1.0)

    def test_invalid_release_date_uses_base(self) -> None:
        """Non-parseable release date -> uses base score unchanged."""
        ctx = _make_ctx(
            imdb_overrides={"imdb_vote_count": 1000, "imdb_rating": None},
            tmdb_overrides={"release_date": "not-a-date"},
        )
        score = _score_imdb_notability(ctx, TODAY)
        expected = math.log10(1001) / math.log10(NOTABILITY_VC_LOG_CAP)
        assert score == pytest.approx(expected, abs=1e-4)

    def test_no_release_date_uses_base(self) -> None:
        """release_date=None -> base score without age adjustment."""
        ctx = _make_ctx(
            imdb_overrides={"imdb_vote_count": 1000, "imdb_rating": None},
            tmdb_overrides={"release_date": None},
        )
        score = _score_imdb_notability(ctx, TODAY)
        expected = math.log10(1001) / math.log10(NOTABILITY_VC_LOG_CAP)
        assert score == pytest.approx(expected, abs=1e-4)

    def test_at_vote_count_log_cap(self) -> None:
        """vc at log cap -> vote_base approaches 1.0."""
        ctx = _make_ctx(imdb_overrides={"imdb_vote_count": 12000, "imdb_rating": None})
        score = _score_imdb_notability(ctx, TODAY)
        assert score > 0.99

    def test_deterministic(self) -> None:
        """Same inputs -> same output."""
        ctx = _make_ctx()
        s1 = _score_imdb_notability(ctx, TODAY)
        s2 = _score_imdb_notability(ctx, TODAY)
        assert s1 == s2


class TestScoreCriticalAttention:
    """Tests for _score_critical_attention signal.

    Counts presence of metacritic_rating and reception_summary out of 2.
    """

    def test_both_fields_present_returns_one(self) -> None:
        """metacritic_rating + reception_summary -> 2/2 = 1.0."""
        ctx = _make_ctx(imdb_overrides={
            "metacritic_rating": 73,
            "reception_summary": "Acclaimed film.",
        })
        assert _score_critical_attention(ctx) == pytest.approx(1.0)

    def test_only_metacritic_returns_half(self) -> None:
        """Only metacritic_rating -> 1/2 = 0.5."""
        ctx = _make_ctx(imdb_overrides={
            "metacritic_rating": 73,
            "reception_summary": None,
        })
        assert _score_critical_attention(ctx) == pytest.approx(0.5)

    def test_only_reception_summary_returns_half(self) -> None:
        """Only reception_summary -> 1/2 = 0.5."""
        ctx = _make_ctx(imdb_overrides={
            "metacritic_rating": None,
            "reception_summary": "A solid film.",
        })
        assert _score_critical_attention(ctx) == pytest.approx(0.5)

    def test_neither_present_returns_zero(self) -> None:
        """Neither field -> 0/2 = 0.0."""
        ctx = _make_ctx(imdb_overrides={
            "metacritic_rating": None,
            "reception_summary": None,
        })
        assert _score_critical_attention(ctx) == pytest.approx(0.0)

    def test_metacritic_zero_is_counted(self) -> None:
        """metacritic_rating=0 uses 'is not None' check -> counted as present."""
        ctx = _make_ctx(imdb_overrides={
            "metacritic_rating": 0,
            "reception_summary": None,
        })
        assert _score_critical_attention(ctx) == pytest.approx(0.5)

    def test_empty_string_reception_summary_not_counted(self) -> None:
        """reception_summary='' uses truthiness check -> not counted."""
        ctx = _make_ctx(imdb_overrides={
            "metacritic_rating": None,
            "reception_summary": "",
        })
        assert _score_critical_attention(ctx) == pytest.approx(0.0)


class TestScoreCommunityEngagement:
    """Tests for _score_community_engagement signal.

    Weighted sum of 4 fields: plot_keywords (wt=1), featured_reviews (wt=2),
    plot_summaries (wt=3), synopses (wt=4).  Total / 10.
    """

    def test_all_fields_full_returns_one(self) -> None:
        """5+ plot_keywords, 5+ featured_reviews, plot_summaries, synopses -> 1.0."""
        ctx = _make_ctx(imdb_overrides={
            "plot_keywords": [f"kw{i}" for i in range(5)],
            "featured_reviews": [{"text": "r"} for _ in range(5)],
            "plot_summaries": ["A summary."],
            "synopses": ["Full synopsis."],
        })
        assert _score_community_engagement(ctx) == pytest.approx(1.0)

    def test_all_fields_empty_returns_zero(self) -> None:
        """All empty -> 0/10 = 0.0."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [],
                "featured_reviews": [],
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"reviews": None},
        )
        assert _score_community_engagement(ctx) == pytest.approx(0.0)

    def test_plot_keywords_linear_to_cap(self) -> None:
        """3 plot_keywords -> min(3/5, 1.0) = 0.6 -> contributes 1 * 0.6."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": ["a", "b", "c"],
                "featured_reviews": [],
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"reviews": None},
        )
        # total = 1 * 0.6 = 0.6 -> 0.6 / 10 = 0.06
        assert _score_community_engagement(ctx) == pytest.approx(0.06, abs=1e-4)

    def test_plot_keywords_above_cap_clamps(self) -> None:
        """10 plot_keywords -> min(10/5, 1.0) = 1.0."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [f"kw{i}" for i in range(10)],
                "featured_reviews": [],
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"reviews": None},
        )
        # total = 1 * 1.0 = 1.0 -> 1.0 / 10 = 0.1
        assert _score_community_engagement(ctx) == pytest.approx(0.1, abs=1e-4)

    def test_featured_reviews_linear_to_cap(self) -> None:
        """3 featured_reviews -> min(3/5, 1.0) = 0.6 -> contributes 2 * 0.6."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [],
                "featured_reviews": [{"text": "r"} for _ in range(3)],
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"reviews": None},
        )
        # total = 2 * 0.6 = 1.2 -> 1.2 / 10 = 0.12
        assert _score_community_engagement(ctx) == pytest.approx(0.12, abs=1e-4)

    def test_featured_reviews_tmdb_fallback(self) -> None:
        """IMDB reviews=[], TMDB reviews JSON has 3 entries -> count=3."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [],
                "featured_reviews": [],
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"reviews": json.dumps(["r1", "r2", "r3"])},
        )
        # review_count=3 -> 2 * min(3/5, 1.0) = 2 * 0.6 = 1.2 -> 0.12
        assert _score_community_engagement(ctx) == pytest.approx(0.12, abs=1e-4)

    def test_featured_reviews_tmdb_not_used_when_imdb_present(self) -> None:
        """IMDB has 1 review, TMDB has 5 -> uses IMDB count of 1."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [],
                "featured_reviews": [{"text": "r"}],
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"reviews": json.dumps(["t1", "t2", "t3", "t4", "t5"])},
        )
        # review_count=1 (IMDB) -> 2 * min(1/5, 1.0) = 2 * 0.2 = 0.4 -> 0.04
        assert _score_community_engagement(ctx) == pytest.approx(0.04, abs=1e-4)

    def test_featured_reviews_malformed_tmdb_json(self) -> None:
        """IMDB=[], malformed TMDB JSON -> review_count stays 0."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [],
                "featured_reviews": [],
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"reviews": "not json"},
        )
        assert _score_community_engagement(ctx) == pytest.approx(0.0)

    def test_plot_summaries_binary(self) -> None:
        """plot_summaries present -> contributes weight 3."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [],
                "featured_reviews": [],
                "plot_summaries": ["A summary."],
                "synopses": [],
            },
            tmdb_overrides={"reviews": None},
        )
        # total = 3 -> 3 / 10 = 0.3
        assert _score_community_engagement(ctx) == pytest.approx(0.3, abs=1e-4)

    def test_synopses_binary(self) -> None:
        """synopses present -> contributes weight 4."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [],
                "featured_reviews": [],
                "plot_summaries": [],
                "synopses": ["Full synopsis."],
            },
            tmdb_overrides={"reviews": None},
        )
        # total = 4 -> 4 / 10 = 0.4
        assert _score_community_engagement(ctx) == pytest.approx(0.4, abs=1e-4)

    def test_partial_contributions(self) -> None:
        """Mixed partial inputs verified numerically."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": ["a", "b"],      # 2/5 = 0.4 -> 1 * 0.4 = 0.4
                "featured_reviews": [],            # 0 -> 0
                "plot_summaries": ["summary"],     # present -> 3
                "synopses": [],                    # absent -> 0
            },
            tmdb_overrides={"reviews": None},
        )
        # total = 0.4 + 0 + 3 + 0 = 3.4 -> 3.4 / 10 = 0.34
        assert _score_community_engagement(ctx) == pytest.approx(0.34, abs=1e-4)


class TestScoreTmdbPopularity:
    """Tests for _score_tmdb_popularity signal."""

    def test_zero_popularity(self) -> None:
        """popularity=0 -> 0.0."""
        ctx = _make_ctx(tmdb_overrides={"popularity": 0})
        assert _score_tmdb_popularity(ctx) == pytest.approx(0.0)

    def test_none_popularity_treated_as_zero(self) -> None:
        """popularity=None -> 0.0."""
        ctx = _make_ctx(tmdb_overrides={"popularity": None})
        assert _score_tmdb_popularity(ctx) == pytest.approx(0.0)

    def test_intermediate_popularity(self) -> None:
        """popularity=1.0 with STAGE5_POP_LOG_CAP=4.0 -> ~0.5."""
        ctx = _make_ctx(tmdb_overrides={"popularity": 1.0})
        score = _score_tmdb_popularity(ctx)
        expected = math.log10(2.0) / math.log10(STAGE5_POP_LOG_CAP)
        assert score == pytest.approx(expected, abs=1e-4)

    def test_at_cap_saturates(self) -> None:
        """popularity = cap - 1 -> score = 1.0."""
        # score_popularity(cap-1, cap) = log10(cap)/log10(cap) = 1.0
        ctx = _make_ctx(tmdb_overrides={"popularity": STAGE5_POP_LOG_CAP - 1})
        assert _score_tmdb_popularity(ctx) == pytest.approx(1.0)

    def test_above_cap_clamps(self) -> None:
        """popularity >> cap -> clamped to 1.0."""
        ctx = _make_ctx(tmdb_overrides={"popularity": 100.0})
        assert _score_tmdb_popularity(ctx) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Signal scoring functions — Data sufficiency signals
# ---------------------------------------------------------------------------


class TestScoreFeaturedReviewsChars:
    """Tests for _score_featured_reviews_chars signal.

    Linear blend: (min(total_chars/5000, 1.0) + min(review_count/5, 1.0)) / 2.
    IMDB primary, TMDB fallback only when IMDB contributes zero on both dims.
    """

    def test_no_reviews_no_tmdb_returns_zero(self) -> None:
        """No featured_reviews, no TMDB reviews -> 0.0."""
        ctx = _make_ctx(
            imdb_overrides={"featured_reviews": []},
            tmdb_overrides={"reviews": None},
        )
        assert _score_featured_reviews_chars(ctx) == pytest.approx(0.0)

    def test_single_review_partial_chars(self) -> None:
        """1 review with 2500 chars -> (0.5 + 0.2) / 2 = 0.35."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 2500}],
        })
        score = _score_featured_reviews_chars(ctx)
        # char_score = 2500/5000 = 0.5, count_score = 1/5 = 0.2
        assert score == pytest.approx(0.35, abs=1e-4)

    def test_both_dimensions_at_cap(self) -> None:
        """5 reviews totaling 5000+ chars -> 1.0."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 1000} for _ in range(5)],
        })
        score = _score_featured_reviews_chars(ctx)
        # char_score = 5000/5000 = 1.0, count_score = 5/5 = 1.0
        assert score == pytest.approx(1.0)

    def test_chars_above_cap_clamps(self) -> None:
        """1 review with 10000 chars -> char_score clamped to 1.0."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 10000}],
        })
        score = _score_featured_reviews_chars(ctx)
        # char_score = min(10000/5000, 1.0) = 1.0, count_score = 1/5 = 0.2
        assert score == pytest.approx(0.6, abs=1e-4)

    def test_count_above_cap_clamps(self) -> None:
        """10 short reviews -> count_score clamped to 1.0."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 100} for _ in range(10)],
        })
        score = _score_featured_reviews_chars(ctx)
        # char_score = min(1000/5000, 1.0) = 0.2, count_score = min(10/5, 1.0) = 1.0
        assert score == pytest.approx(0.6, abs=1e-4)

    def test_tmdb_fallback_when_imdb_empty(self) -> None:
        """IMDB reviews=[], TMDB reviews JSON has content -> uses TMDB."""
        tmdb_reviews = json.dumps(["A" * 2500, "B" * 2500])
        ctx = _make_ctx(
            imdb_overrides={"featured_reviews": []},
            tmdb_overrides={"reviews": tmdb_reviews},
        )
        score = _score_featured_reviews_chars(ctx)
        # total_chars = 5000 (from TMDB), review_count = 2 (from TMDB)
        # char_score = 1.0, count_score = 0.4
        assert score == pytest.approx(0.7, abs=1e-4)

    def test_tmdb_not_used_when_imdb_has_content(self) -> None:
        """IMDB has reviews -> TMDB ignored even if TMDB is richer."""
        ctx = _make_ctx(
            imdb_overrides={"featured_reviews": [{"text": "A" * 100}]},
            tmdb_overrides={"reviews": json.dumps(["B" * 5000])},
        )
        score = _score_featured_reviews_chars(ctx)
        # Uses IMDB: char_score = 100/5000 = 0.02, count_score = 1/5 = 0.2
        assert score == pytest.approx(0.11, abs=1e-4)

    def test_tmdb_not_used_when_imdb_has_count_but_no_chars(self) -> None:
        """IMDB reviews with empty text (count > 0) blocks TMDB fallback.

        The fallback gate requires BOTH total_chars == 0 AND review_count == 0.
        """
        ctx = _make_ctx(
            imdb_overrides={"featured_reviews": [{"text": ""}, {"text": ""}, {"text": ""}]},
            tmdb_overrides={"reviews": json.dumps(["T" * 5000])},
        )
        score = _score_featured_reviews_chars(ctx)
        # IMDB: chars=0, count=3 -> gate fails (count != 0) -> no TMDB fallback
        # char_score = 0, count_score = 3/5 = 0.6 -> (0 + 0.6) / 2 = 0.3
        assert score == pytest.approx(0.3, abs=1e-4)

    def test_malformed_tmdb_json_treated_as_no_reviews(self) -> None:
        """TMDB reviews='not json' -> 0.0 (no crash)."""
        ctx = _make_ctx(
            imdb_overrides={"featured_reviews": []},
            tmdb_overrides={"reviews": "not json"},
        )
        assert _score_featured_reviews_chars(ctx) == pytest.approx(0.0)

    def test_linear_scaling_at_midpoint(self) -> None:
        """Verify linear scaling at clear midpoints."""
        # 2 reviews with 1250 chars each = 2500 total
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 1250}, {"text": "B" * 1250}],
        })
        score = _score_featured_reviews_chars(ctx)
        # char_score = 2500/5000 = 0.5, count_score = 2/5 = 0.4
        assert score == pytest.approx(0.45, abs=1e-4)


class TestScorePlotTextDepth:
    """Tests for _score_plot_text_depth signal."""

    def test_all_empty_returns_zero(self) -> None:
        """No text in any field -> 0.0."""
        ctx = _make_ctx(
            imdb_overrides={
                "overview": "",
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"overview_length": 0},
        )
        assert _score_plot_text_depth(ctx) == pytest.approx(0.0)

    def test_short_overview_only(self) -> None:
        """150 chars overview -> intermediate score (~0.59)."""
        ctx = _make_ctx(
            imdb_overrides={
                "overview": "A" * 150,
                "plot_summaries": [],
                "synopses": [],
            },
        )
        score = _score_plot_text_depth(ctx)
        expected = math.log10(150 + 1) / math.log10(PLOT_TEXT_LOG_CAP)
        assert score == pytest.approx(expected, abs=0.01)

    def test_rich_text_saturates_at_cap(self) -> None:
        """5001+ total chars -> 1.0."""
        ctx = _make_ctx(
            imdb_overrides={
                "overview": "A" * 2000,
                "plot_summaries": ["B" * 2000],
                "synopses": ["C" * 2000],
            },
        )
        assert _score_plot_text_depth(ctx) == pytest.approx(1.0)

    def test_tmdb_overview_fallback_when_imdb_overview_empty(self) -> None:
        """IMDB overview='' -> uses tmdb.overview_length."""
        ctx = _make_ctx(
            imdb_overrides={
                "overview": "",
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"overview_length": 200},
        )
        score = _score_plot_text_depth(ctx)
        expected = math.log10(200 + 1) / math.log10(PLOT_TEXT_LOG_CAP)
        assert score == pytest.approx(expected, abs=0.01)

    def test_tmdb_fallback_not_used_when_imdb_has_overview(self) -> None:
        """IMDB overview present -> TMDB ignored."""
        ctx = _make_ctx(
            imdb_overrides={
                "overview": "X" * 100,
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"overview_length": 500},
        )
        score = _score_plot_text_depth(ctx)
        # Should use only IMDB's 100 chars, not TMDB's 500.
        expected = math.log10(100 + 1) / math.log10(PLOT_TEXT_LOG_CAP)
        assert score == pytest.approx(expected, abs=0.01)

    def test_combines_overview_summaries_synopses(self) -> None:
        """All three IMDB fields contribute to total."""
        ctx = _make_ctx(
            imdb_overrides={
                "overview": "A" * 100,
                "plot_summaries": ["B" * 200],
                "synopses": ["C" * 300],
            },
        )
        score = _score_plot_text_depth(ctx)
        total = 100 + 200 + 300
        expected = math.log10(total + 1) / math.log10(PLOT_TEXT_LOG_CAP)
        assert score == pytest.approx(expected, abs=0.01)


class TestScoreLexicalCompleteness:
    """Tests for _score_lexical_completeness signal.

    5 entities: actors (cap 10), characters (cap 10), writers (binary),
    producers (binary), production_companies (binary, IMDB -> TMDB fallback).
    Average of 5 sub-scores, then classic-film age boost.
    """

    def test_all_entities_full_returns_one(self) -> None:
        """10+ actors, 10+ characters, writers, producers, prodco -> 1.0."""
        ctx = _make_ctx(imdb_overrides={
            "actors": [f"A{i}" for i in range(10)],
            "characters": [f"C{i}" for i in range(10)],
            "writers": ["W1"],
            "producers": ["P1"],
            "production_companies": ["Co1"],
        })
        assert _score_lexical_completeness(ctx, TODAY) == pytest.approx(1.0)

    def test_all_entities_empty_returns_zero(self) -> None:
        """All empty -> 0.0."""
        ctx = _make_ctx(
            imdb_overrides={
                "actors": [],
                "characters": [],
                "writers": [],
                "producers": [],
                "production_companies": [],
            },
            tmdb_overrides={"has_production_companies": 0},
        )
        assert _score_lexical_completeness(ctx, TODAY) == pytest.approx(0.0)

    def test_actors_linear_to_cap(self) -> None:
        """5 actors -> actors_sub = 5/10 = 0.5."""
        ctx = _make_ctx(imdb_overrides={"actors": [f"A{i}" for i in range(5)]})
        score = _score_lexical_completeness(ctx, TODAY)
        # actors=0.5, chars=1.0, writers=1.0, producers=1.0, prodco=1.0
        # raw = (0.5 + 1 + 1 + 1 + 1) / 5 = 4.5 / 5 = 0.9
        assert score == pytest.approx(0.9, abs=1e-4)

    def test_actors_above_cap_clamps(self) -> None:
        """15 actors -> actors_sub clamped to 1.0."""
        ctx = _make_ctx(imdb_overrides={"actors": [f"A{i}" for i in range(15)]})
        assert _score_lexical_completeness(ctx, TODAY) == pytest.approx(1.0)

    def test_characters_linear_to_cap(self) -> None:
        """3 characters -> chars_sub = 3/10 = 0.3."""
        ctx = _make_ctx(imdb_overrides={"characters": [f"C{i}" for i in range(3)]})
        score = _score_lexical_completeness(ctx, TODAY)
        # actors=1.0, chars=0.3, writers=1.0, producers=1.0, prodco=1.0
        # raw = (1 + 0.3 + 1 + 1 + 1) / 5 = 4.3 / 5 = 0.86
        assert score == pytest.approx(0.86, abs=1e-4)

    def test_writers_binary(self) -> None:
        """writers=[] -> 0.0 sub-score; writers=['W1'] -> 1.0 sub-score."""
        ctx_empty = _make_ctx(imdb_overrides={"writers": []})
        ctx_present = _make_ctx(imdb_overrides={"writers": ["W1"]})
        score_empty = _score_lexical_completeness(ctx_empty, TODAY)
        score_present = _score_lexical_completeness(ctx_present, TODAY)
        # Difference should be exactly 1/5 = 0.2 (one binary sub-score).
        assert (score_present - score_empty) == pytest.approx(0.2, abs=1e-4)

    def test_producers_binary(self) -> None:
        """producers=[] -> 0.0 sub-score."""
        ctx = _make_ctx(imdb_overrides={"producers": []})
        score = _score_lexical_completeness(ctx, TODAY)
        # raw = (1 + 1 + 1 + 0 + 1) / 5 = 4/5 = 0.8
        assert score == pytest.approx(0.8, abs=1e-4)

    def test_production_companies_imdb_primary(self) -> None:
        """IMDB production_companies present -> 1.0 sub-score regardless of TMDB."""
        ctx = _make_ctx(
            imdb_overrides={"production_companies": ["Co1"]},
            tmdb_overrides={"has_production_companies": 0},
        )
        assert _score_lexical_completeness(ctx, TODAY) == pytest.approx(1.0)

    def test_production_companies_tmdb_fallback(self) -> None:
        """IMDB production_companies=[], TMDB has_production_companies=1 -> 1.0 sub-score."""
        ctx = _make_ctx(
            imdb_overrides={"production_companies": []},
            tmdb_overrides={"has_production_companies": 1},
        )
        assert _score_lexical_completeness(ctx, TODAY) == pytest.approx(1.0)

    def test_production_companies_both_empty(self) -> None:
        """Both IMDB and TMDB empty -> 0.0 sub-score."""
        ctx = _make_ctx(
            imdb_overrides={"production_companies": []},
            tmdb_overrides={"has_production_companies": 0},
        )
        score = _score_lexical_completeness(ctx, TODAY)
        # raw = (1 + 1 + 1 + 1 + 0) / 5 = 4/5 = 0.8
        assert score == pytest.approx(0.8, abs=1e-4)

    def test_classic_boost_at_30_years(self) -> None:
        """30-year-old film gets classic boost ~1.33x on raw score."""
        thirty_years_ago = "1996-03-05"
        # Use partial entities so raw score < 1.0 (boost won't be capped).
        ctx = _make_ctx(
            imdb_overrides={
                "actors": [f"A{i}" for i in range(6)],    # 6/10 = 0.6
                "characters": [f"C{i}" for i in range(6)],  # 6/10 = 0.6
                "writers": ["W1"],                           # 1.0
                "producers": [],                             # 0.0
                "production_companies": [],                  # 0.0
            },
            tmdb_overrides={
                "release_date": thirty_years_ago,
                "has_production_companies": 0,
            },
        )
        score = _score_lexical_completeness(ctx, TODAY)
        # raw = (0.6 + 0.6 + 1.0 + 0.0 + 0.0) / 5 = 2.2 / 5 = 0.44
        age_years = (TODAY - datetime.date(1996, 3, 5)).days / 365.0
        classic = min(VC_CLASSIC_BOOST_CAP, 1.0 + max(0, age_years - VC_CLASSIC_START_YEARS) / VC_CLASSIC_RAMP_YEARS)
        expected = min(0.44 * classic, 1.0)
        assert score == pytest.approx(expected, abs=1e-3)
        # Score should be higher than raw (boost applied).
        assert score > 0.44

    def test_classic_boost_does_not_exceed_one(self) -> None:
        """High raw score with classic boost -> capped at 1.0."""
        fifty_years_ago = "1976-03-05"
        ctx = _make_ctx(tmdb_overrides={"release_date": fifty_years_ago})
        score = _score_lexical_completeness(ctx, TODAY)
        # raw = 1.0, boost = 1.5 -> min(1.5, 1.0) = 1.0
        assert score == pytest.approx(1.0)

    def test_no_classic_boost_for_recent_film(self) -> None:
        """5-year-old film gets no classic boost."""
        ctx_5yr = _make_ctx(
            imdb_overrides={"actors": [f"A{i}" for i in range(5)]},
            tmdb_overrides={"release_date": "2021-03-05"},
        )
        ctx_10yr = _make_ctx(
            imdb_overrides={"actors": [f"A{i}" for i in range(5)]},
            tmdb_overrides={"release_date": "2016-03-05"},
        )
        score_5yr = _score_lexical_completeness(ctx_5yr, TODAY)
        score_10yr = _score_lexical_completeness(ctx_10yr, TODAY)
        # Both within the 2-20yr range -> no boost -> same score.
        assert score_5yr == pytest.approx(score_10yr, abs=1e-4)

    def test_invalid_release_date_no_boost(self) -> None:
        """Non-parseable release_date -> raw score unchanged."""
        ctx = _make_ctx(
            imdb_overrides={"actors": [f"A{i}" for i in range(5)]},
            tmdb_overrides={"release_date": "not-a-date"},
        )
        score = _score_lexical_completeness(ctx, TODAY)
        # raw = (0.5 + 1 + 1 + 1 + 1) / 5 = 0.9, no boost
        assert score == pytest.approx(0.9, abs=1e-4)

    def test_no_release_date_no_boost(self) -> None:
        """release_date=None -> raw score unchanged."""
        ctx = _make_ctx(
            imdb_overrides={"actors": [f"A{i}" for i in range(5)]},
            tmdb_overrides={"release_date": None},
        )
        score = _score_lexical_completeness(ctx, TODAY)
        assert score == pytest.approx(0.9, abs=1e-4)

    def test_composers_not_counted(self) -> None:
        """Composers field has no effect on score (was removed in v3)."""
        ctx_with = _make_ctx(imdb_overrides={"composers": ["M1", "M2"]})
        ctx_without = _make_ctx()
        score_with = _score_lexical_completeness(ctx_with, TODAY)
        score_without = _score_lexical_completeness(ctx_without, TODAY)
        assert score_with == pytest.approx(score_without)


class TestScoreDataCompleteness:
    """Tests for _score_data_completeness signal.

    5 fields: plot_keywords (cap 5), overall_keywords (cap 6),
    parental_guide_items (cap 3), maturity_rating (binary),
    budget (binary). Average of 5 sub-scores.
    """

    def test_all_fields_full_returns_one(self) -> None:
        """All fields at max -> 1.0."""
        ctx = _make_ctx(imdb_overrides={
            "plot_keywords": [f"k{i}" for i in range(5)],
            "overall_keywords": [f"o{i}" for i in range(6)],
            "parental_guide_items": [{"category": f"c{i}"} for i in range(3)],
            "maturity_rating": "R",
            "budget": 100_000_000,
        })
        assert _score_data_completeness(ctx) == pytest.approx(1.0)

    def test_all_fields_empty_returns_zero(self) -> None:
        """All empty -> 0.0."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [],
                "overall_keywords": [],
                "parental_guide_items": [],
                "maturity_rating": None,
                "budget": None,
            },
            tmdb_overrides={"maturity_rating": None, "budget": 0},
        )
        assert _score_data_completeness(ctx) == pytest.approx(0.0)

    def test_plot_keywords_linear_to_cap(self) -> None:
        """plot_keywords: 3 -> 3/5 = 0.6 sub-score."""
        ctx_3 = _make_ctx(imdb_overrides={"plot_keywords": ["a", "b", "c"]})
        ctx_5 = _make_ctx(imdb_overrides={"plot_keywords": ["a", "b", "c", "d", "e"]})
        ctx_7 = _make_ctx(imdb_overrides={"plot_keywords": [f"k{i}" for i in range(7)]})
        s3 = _score_data_completeness(ctx_3)
        s5 = _score_data_completeness(ctx_5)
        s7 = _score_data_completeness(ctx_7)
        # 3/5 < 5/5 = cap, 7 also caps at 1.0.
        assert s3 < s5
        assert s5 == pytest.approx(s7)  # both at cap

    def test_overall_keywords_linear_to_cap(self) -> None:
        """overall_keywords: 3 -> 3/6 = 0.5, 6 -> 1.0."""
        ctx_3 = _make_ctx(imdb_overrides={"overall_keywords": ["a", "b", "c"]})
        ctx_6 = _make_ctx(imdb_overrides={"overall_keywords": [f"o{i}" for i in range(6)]})
        s3 = _score_data_completeness(ctx_3)
        s6 = _score_data_completeness(ctx_6)
        assert s3 < s6

    def test_parental_guide_items_linear_to_cap(self) -> None:
        """parental_guide_items: 1 -> 1/3 = 0.333, 3 -> 1.0."""
        ctx_1 = _make_ctx(imdb_overrides={"parental_guide_items": [{"category": "v"}]})
        ctx_3 = _make_ctx(imdb_overrides={
            "parental_guide_items": [{"category": f"c{i}"} for i in range(3)],
        })
        s1 = _score_data_completeness(ctx_1)
        s3 = _score_data_completeness(ctx_3)
        assert s1 < s3

    def test_maturity_rating_imdb_primary(self) -> None:
        """IMDB maturity_rating='R' -> 1.0 sub-score regardless of TMDB."""
        ctx = _make_ctx(
            imdb_overrides={"maturity_rating": "R"},
            tmdb_overrides={"maturity_rating": None},
        )
        score = _score_data_completeness(ctx)
        # Should be same as default (maturity_rating present via IMDB).
        ctx_default = _make_ctx()
        assert score == pytest.approx(_score_data_completeness(ctx_default))

    def test_maturity_rating_tmdb_fallback(self) -> None:
        """IMDB maturity_rating=None, TMDB maturity_rating='PG-13' -> 1.0 sub-score."""
        ctx_imdb = _make_ctx(imdb_overrides={"maturity_rating": "R"})
        ctx_tmdb = _make_ctx(
            imdb_overrides={"maturity_rating": None},
            tmdb_overrides={"maturity_rating": "PG-13"},
        )
        assert _score_data_completeness(ctx_tmdb) == pytest.approx(
            _score_data_completeness(ctx_imdb),
        )

    def test_maturity_rating_both_empty(self) -> None:
        """Both IMDB and TMDB maturity_rating empty -> 0.0 sub-score."""
        ctx = _make_ctx(
            imdb_overrides={"maturity_rating": None},
            tmdb_overrides={"maturity_rating": None},
        )
        ctx_with = _make_ctx(imdb_overrides={"maturity_rating": "R"})
        # Difference should be exactly 1/5 = 0.2 (one binary sub-score).
        assert (_score_data_completeness(ctx_with) - _score_data_completeness(ctx)) == pytest.approx(0.2, abs=1e-4)

    def test_budget_imdb_primary(self) -> None:
        """IMDB budget present -> 1.0 sub-score."""
        ctx = _make_ctx(
            imdb_overrides={"budget": 100_000_000},
            tmdb_overrides={"budget": 0},
        )
        score = _score_data_completeness(ctx)
        ctx_default = _make_ctx()
        assert score == pytest.approx(_score_data_completeness(ctx_default))

    def test_budget_tmdb_fallback(self) -> None:
        """IMDB budget=None, TMDB budget present -> 1.0 sub-score."""
        ctx_imdb = _make_ctx(imdb_overrides={"budget": 100_000_000})
        ctx_tmdb = _make_ctx(
            imdb_overrides={"budget": None},
            tmdb_overrides={"budget": 50_000_000},
        )
        assert _score_data_completeness(ctx_tmdb) == pytest.approx(
            _score_data_completeness(ctx_imdb),
        )

    def test_budget_both_empty(self) -> None:
        """Both IMDB and TMDB budget empty -> 0.0 sub-score."""
        ctx = _make_ctx(
            imdb_overrides={"budget": None},
            tmdb_overrides={"budget": 0},
        )
        ctx_with = _make_ctx(imdb_overrides={"budget": 100_000_000})
        assert (_score_data_completeness(ctx_with) - _score_data_completeness(ctx)) == pytest.approx(0.2, abs=1e-4)

    def test_filming_locations_not_counted(self) -> None:
        """filming_locations has no effect on score (removed in v3)."""
        ctx_with = _make_ctx(imdb_overrides={"filming_locations": ["LA", "NYC", "London"]})
        ctx_without = _make_ctx(imdb_overrides={"filming_locations": []})
        assert _score_data_completeness(ctx_with) == pytest.approx(
            _score_data_completeness(ctx_without),
        )

    def test_single_field_contribution(self) -> None:
        """Only plot_keywords=5 present, all others empty -> 1/5 = 0.2."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [f"k{i}" for i in range(5)],
                "overall_keywords": [],
                "parental_guide_items": [],
                "maturity_rating": None,
                "budget": None,
            },
            tmdb_overrides={"maturity_rating": None, "budget": 0},
        )
        assert _score_data_completeness(ctx) == pytest.approx(0.2, abs=1e-4)


# ---------------------------------------------------------------------------
# compute_imdb_quality_score
# ---------------------------------------------------------------------------


class TestComputeImdbQualityScore:
    """Tests for the composite weighted quality score."""

    def test_perfect_movie_near_max(self) -> None:
        """All signals at max -> score near theoretical max."""
        ctx = _make_ctx()
        score = compute_imdb_quality_score(ctx, TODAY)
        # With all signals high, score should be well above 0.5.
        assert score > 0.5

    def test_minimal_movie_near_zero(self) -> None:
        """All signals at min -> score near 0.0."""
        ctx = _make_ctx(
            imdb_overrides={
                "imdb_title_type": "movie",
                "imdb_vote_count": 0,
                "imdb_rating": None,
                "metacritic_rating": None,
                "reception_summary": None,
                "featured_reviews": [],
                "overview": "",
                "plot_summaries": ["placeholder"],  # Need at least one text source
                "synopses": [],
                "actors": [],
                "characters": [],
                "writers": [],
                "producers": [],
                "production_companies": [],
                "plot_keywords": [],
                "overall_keywords": [],
                "parental_guide_items": [],
                "maturity_rating": None,
                "budget": None,
            },
            tmdb_overrides={
                "popularity": 0,
                "overview_length": 0,
                "maturity_rating": None,
                "budget": 0,
                "reviews": None,
                "has_production_companies": 0,
            },
        )
        score = compute_imdb_quality_score(ctx, TODAY)
        # All signals return near 0.0 (plot_text_depth > 0 due to placeholder).
        assert score < 0.1

    def test_weights_sum_to_one(self) -> None:
        """WEIGHTS dict sums to 1.0 (module-level guard)."""
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_individual_signal_contribution(self) -> None:
        """Verify that toggling a single signal changes score by weight * delta."""
        # Toggle both metacritic_rating and reception_summary: only affects
        # _score_critical_attention (0 -> 1.0), no other signal uses these fields.
        ctx_with = _make_ctx(imdb_overrides={
            "metacritic_rating": 73,
            "reception_summary": "Acclaimed.",
        })
        ctx_without = _make_ctx(imdb_overrides={
            "metacritic_rating": None,
            "reception_summary": None,
        })
        score_with = compute_imdb_quality_score(ctx_with, TODAY)
        score_without = compute_imdb_quality_score(ctx_without, TODAY)
        delta = score_with - score_without
        # critical_attention goes from 0.0 to 1.0 -> delta = weight * 1.0
        assert delta == pytest.approx(WEIGHTS["critical_attention"] * 1.0, abs=1e-9)

    def test_deterministic(self) -> None:
        """Same inputs -> same output."""
        ctx = _make_ctx()
        score1 = compute_imdb_quality_score(ctx, TODAY)
        score2 = compute_imdb_quality_score(ctx, TODAY)
        assert score1 == score2


# ---------------------------------------------------------------------------
# WEIGHTS module-level constant
# ---------------------------------------------------------------------------


class TestWeights:
    """Tests for the WEIGHTS constant."""

    def test_weights_sum_to_one(self) -> None:
        """abs(sum - 1.0) < 1e-9."""
        assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9

    def test_weights_has_eight_signals(self) -> None:
        """len(WEIGHTS) == 8."""
        assert len(WEIGHTS) == 8

    def test_all_weights_positive(self) -> None:
        """Every weight > 0."""
        for name, weight in WEIGHTS.items():
            assert weight > 0, f"Weight for '{name}' is not positive: {weight}"

    def test_weight_keys_match_expected(self) -> None:
        """WEIGHTS keys match the 8 signal function names."""
        expected_keys = {
            "imdb_notability",
            "critical_attention",
            "community_engagement",
            "tmdb_popularity",
            "featured_reviews_chars",
            "plot_text_depth",
            "lexical_completeness",
            "data_completeness",
        }
        assert set(WEIGHTS.keys()) == expected_keys


# ---------------------------------------------------------------------------
# ALLOWED_TITLE_TYPES constant
# ---------------------------------------------------------------------------


class TestAllowedTitleTypes:
    """Tests for the ALLOWED_TITLE_TYPES constant."""

    def test_allowed_title_types_contains_expected_values(self) -> None:
        """ALLOWED_TITLE_TYPES contains exactly the 4 expected types."""
        assert ALLOWED_TITLE_TYPES == {"movie", "tvMovie", "short", "video"}


# ---------------------------------------------------------------------------
# Title type early-return guard
# ---------------------------------------------------------------------------


class TestTitleTypeGuard:
    """Tests for the imdb_title_type early-return guard in compute_imdb_quality_score."""

    def test_returns_zero_for_tv_series(self) -> None:
        """imdb_title_type='tvSeries' returns 0.0."""
        ctx = _make_ctx(imdb_overrides={"imdb_title_type": "tvSeries"})
        assert compute_imdb_quality_score(ctx, TODAY) == 0.0

    def test_returns_zero_for_video_game(self) -> None:
        """imdb_title_type='videoGame' returns 0.0."""
        ctx = _make_ctx(imdb_overrides={"imdb_title_type": "videoGame"})
        assert compute_imdb_quality_score(ctx, TODAY) == 0.0

    def test_returns_zero_for_none_title_type(self) -> None:
        """imdb_title_type=None (not in ALLOWED_TITLE_TYPES) returns 0.0."""
        ctx = _make_ctx(imdb_overrides={"imdb_title_type": None})
        assert compute_imdb_quality_score(ctx, TODAY) == 0.0

    def test_returns_positive_for_movie(self) -> None:
        """imdb_title_type='movie' passes through to scoring."""
        ctx = _make_ctx(imdb_overrides={"imdb_title_type": "movie"})
        assert compute_imdb_quality_score(ctx, TODAY) > 0.0

    def test_returns_positive_for_tv_movie(self) -> None:
        """imdb_title_type='tvMovie' passes through to scoring."""
        ctx = _make_ctx(imdb_overrides={"imdb_title_type": "tvMovie"})
        assert compute_imdb_quality_score(ctx, TODAY) > 0.0

    def test_returns_positive_for_short(self) -> None:
        """imdb_title_type='short' passes through to scoring."""
        ctx = _make_ctx(imdb_overrides={"imdb_title_type": "short"})
        assert compute_imdb_quality_score(ctx, TODAY) > 0.0

    def test_returns_positive_for_video(self) -> None:
        """imdb_title_type='video' passes through to scoring."""
        ctx = _make_ctx(imdb_overrides={"imdb_title_type": "video"})
        assert compute_imdb_quality_score(ctx, TODAY) > 0.0


# ---------------------------------------------------------------------------
# Text source early-return guard
# ---------------------------------------------------------------------------


class TestTextSourceGuard:
    """Tests for the text source early-return guard in compute_imdb_quality_score."""

    def test_returns_zero_when_all_text_sources_empty(self) -> None:
        """No plot_summaries, synopses, or featured_reviews returns 0.0."""
        ctx = _make_ctx(imdb_overrides={
            "plot_summaries": [],
            "synopses": [],
            "featured_reviews": [],
        })
        assert compute_imdb_quality_score(ctx, TODAY) == 0.0

    def test_returns_positive_when_only_featured_reviews_present(self) -> None:
        """featured_reviews alone is sufficient to pass the text source guard."""
        ctx = _make_ctx(imdb_overrides={
            "plot_summaries": [],
            "synopses": [],
            "featured_reviews": [{"text": "A review."}],
        })
        assert compute_imdb_quality_score(ctx, TODAY) > 0.0

    def test_returns_positive_when_only_plot_summaries_present(self) -> None:
        """plot_summaries alone is sufficient to pass the text source guard."""
        ctx = _make_ctx(imdb_overrides={
            "plot_summaries": ["A summary."],
            "synopses": [],
            "featured_reviews": [],
        })
        assert compute_imdb_quality_score(ctx, TODAY) > 0.0

    def test_returns_positive_when_only_synopses_present(self) -> None:
        """synopses alone is sufficient to pass the text source guard."""
        ctx = _make_ctx(imdb_overrides={
            "plot_summaries": [],
            "synopses": ["A synopsis."],
            "featured_reviews": [],
        })
        assert compute_imdb_quality_score(ctx, TODAY) > 0.0
