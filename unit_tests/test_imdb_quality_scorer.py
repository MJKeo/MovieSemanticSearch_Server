"""
Unit tests for movie_ingestion.imdb_quality_scoring.imdb_quality_scorer — Stage 5.

Covers:
  - MovieContext dataclass
  - Hard filter predicates (_fails_* functions)
  - _evaluate_filters — priority-ordered filter evaluation
  - Signal scoring functions (8 signals)
  - compute_imdb_quality_score — weighted composite
  - WEIGHTS module-level constant
"""

import datetime
import json
import math
import struct

import pytest

from movie_ingestion.imdb_quality_scoring.imdb_quality_scorer import (
    PLOT_TEXT_LOG_CAP,
    WEIGHTS,
    MovieContext,
    _evaluate_filters,
    _fails_actors,
    _fails_characters,
    _fails_countries_of_origin,
    _fails_directors,
    _fails_imdb_json,
    _fails_imdb_rating,
    _fails_languages,
    _fails_overall_keywords,
    _fails_poster_url,
    _fails_release_date,
    _score_data_completeness,
    _score_featured_reviews_chars,
    _score_imdb_vote_count,
    _score_lexical_completeness,
    _score_metacritic_rating,
    _score_plot_text_depth,
    _score_tmdb_popularity,
    _score_watch_providers,
    compute_imdb_quality_score,
)
from movie_ingestion.scoring_utils import THEATER_WINDOW_DAYS


# ---------------------------------------------------------------------------
# Shared test infrastructure
# ---------------------------------------------------------------------------

# Fixed reference date used throughout to make all tests deterministic.
TODAY = datetime.date(2026, 3, 5)


def _make_ctx(
    tmdb_overrides: dict | None = None,
    imdb_overrides: dict | None = None,
    imdb_none: bool = False,
) -> MovieContext:
    """Build a MovieContext with sensible defaults for a passing movie.

    All fields are populated so every hard filter passes by default.
    Use tmdb_overrides/imdb_overrides to target specific fields, or
    imdb_none=True to simulate a missing IMDB JSON file.
    """
    tmdb = {
        "tmdb_id": 12345,
        "poster_url": "https://image.tmdb.org/poster.jpg",
        "release_date": "2020-06-15",
        "watch_provider_keys": struct.pack("<3I", 8, 15, 337),
        "popularity": 5.0,
        "overview_length": 180,
        "maturity_rating": "PG-13",
        "budget": 50_000_000,
        "reviews": json.dumps(["A solid film with great performances."]),
        "has_production_companies": 1,
    }
    if tmdb_overrides:
        tmdb.update(tmdb_overrides)

    imdb: dict | None = None
    if not imdb_none:
        imdb = {
            "imdb_rating": 7.2,
            "imdb_vote_count": 5000,
            "directors": ["Christopher Nolan"],
            "actors": ["Actor1", "Actor2", "Actor3", "Actor4", "Actor5"],
            "characters": ["Char1", "Char2", "Char3", "Char4", "Char5"],
            "overall_keywords": ["thriller", "drama", "mystery", "suspense"],
            "languages": ["English"],
            "countries_of_origin": ["United States"],
            "writers": ["Writer1"],
            "composers": ["Composer1"],
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
            "filming_locations": ["Los Angeles, CA"],
            "parental_guide_items": [{"category": "violence", "severity": "moderate"}],
            "maturity_rating": "R",
            "budget": 100_000_000,
            "metacritic_rating": 73,
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

    def test_movie_context_with_none_imdb(self) -> None:
        """imdb=None is valid (missing JSON case)."""
        ctx = _make_ctx(imdb_none=True)
        assert ctx.imdb is None
        assert ctx.tmdb is not None


# ---------------------------------------------------------------------------
# Hard filter predicates
# ---------------------------------------------------------------------------


class TestFailsImdbJson:
    """Tests for _fails_imdb_json predicate."""

    def test_fails_when_imdb_is_none(self) -> None:
        """imdb=None → True."""
        ctx = _make_ctx(imdb_none=True)
        assert _fails_imdb_json(ctx) is True

    def test_passes_when_imdb_present(self) -> None:
        """imdb={...} → False."""
        ctx = _make_ctx()
        assert _fails_imdb_json(ctx) is False


class TestFailsImdbRating:
    """Tests for _fails_imdb_rating predicate."""

    def test_fails_when_imdb_none(self) -> None:
        """imdb=None → True."""
        ctx = _make_ctx(imdb_none=True)
        assert _fails_imdb_rating(ctx) is True

    def test_fails_when_imdb_rating_none(self) -> None:
        """imdb_rating missing → True."""
        ctx = _make_ctx(imdb_overrides={"imdb_rating": None})
        assert _fails_imdb_rating(ctx) is True

    def test_passes_when_imdb_rating_present(self) -> None:
        """imdb_rating=7.0 → False."""
        ctx = _make_ctx(imdb_overrides={"imdb_rating": 7.0})
        assert _fails_imdb_rating(ctx) is False


class TestFailsDirectors:
    """Tests for _fails_directors predicate."""

    def test_fails_when_directors_empty(self) -> None:
        """directors=[] → True."""
        ctx = _make_ctx(imdb_overrides={"directors": []})
        assert _fails_directors(ctx) is True

    def test_fails_when_directors_missing(self) -> None:
        """Key absent → True."""
        ctx = _make_ctx()
        del ctx.imdb["directors"]
        assert _fails_directors(ctx) is True

    def test_passes_when_directors_present(self) -> None:
        """directors=['Name'] → False."""
        ctx = _make_ctx(imdb_overrides={"directors": ["Name"]})
        assert _fails_directors(ctx) is False


class TestFailsPosterUrl:
    """Tests for _fails_poster_url predicate (checks ctx.tmdb)."""

    def test_fails_when_poster_url_none(self) -> None:
        """poster_url=None → True."""
        ctx = _make_ctx(tmdb_overrides={"poster_url": None})
        assert _fails_poster_url(ctx) is True

    def test_fails_when_poster_url_empty(self) -> None:
        """poster_url='' → True."""
        ctx = _make_ctx(tmdb_overrides={"poster_url": ""})
        assert _fails_poster_url(ctx) is True

    def test_passes_when_poster_url_present(self) -> None:
        """poster_url='http://...' → False."""
        ctx = _make_ctx(tmdb_overrides={"poster_url": "http://example.com/poster.jpg"})
        assert _fails_poster_url(ctx) is False


class TestFailsImdbListPredicates:
    """Tests for the 5 structurally identical IMDB-list predicates:
    _fails_actors, _fails_characters, _fails_overall_keywords,
    _fails_languages, _fails_countries_of_origin.
    """

    @pytest.mark.parametrize("predicate,field", [
        (_fails_actors, "actors"),
        (_fails_characters, "characters"),
        (_fails_overall_keywords, "overall_keywords"),
        (_fails_languages, "languages"),
        (_fails_countries_of_origin, "countries_of_origin"),
    ])
    def test_fails_when_empty_list(self, predicate, field) -> None:
        """Empty list → True."""
        ctx = _make_ctx(imdb_overrides={field: []})
        assert predicate(ctx) is True

    @pytest.mark.parametrize("predicate,field", [
        (_fails_actors, "actors"),
        (_fails_characters, "characters"),
        (_fails_overall_keywords, "overall_keywords"),
        (_fails_languages, "languages"),
        (_fails_countries_of_origin, "countries_of_origin"),
    ])
    def test_fails_when_key_missing(self, predicate, field) -> None:
        """Key absent → True."""
        ctx = _make_ctx()
        del ctx.imdb[field]
        assert predicate(ctx) is True

    @pytest.mark.parametrize("predicate,field,value", [
        (_fails_actors, "actors", ["Actor1"]),
        (_fails_characters, "characters", ["Char1"]),
        (_fails_overall_keywords, "overall_keywords", ["keyword1"]),
        (_fails_languages, "languages", ["English"]),
        (_fails_countries_of_origin, "countries_of_origin", ["US"]),
    ])
    def test_passes_when_present(self, predicate, field, value) -> None:
        """Non-empty list → False."""
        ctx = _make_ctx(imdb_overrides={field: value})
        assert predicate(ctx) is False

    @pytest.mark.parametrize("predicate", [
        _fails_actors,
        _fails_characters,
        _fails_overall_keywords,
        _fails_languages,
        _fails_countries_of_origin,
    ])
    def test_fails_when_imdb_none(self, predicate) -> None:
        """imdb=None → True."""
        ctx = _make_ctx(imdb_none=True)
        assert predicate(ctx) is True


class TestFailsReleaseDate:
    """Tests for _fails_release_date predicate (checks ctx.tmdb)."""

    def test_fails_when_release_date_none(self) -> None:
        """release_date=None → True."""
        ctx = _make_ctx(tmdb_overrides={"release_date": None})
        assert _fails_release_date(ctx) is True

    def test_fails_when_release_date_empty_string(self) -> None:
        """release_date='' → True."""
        ctx = _make_ctx(tmdb_overrides={"release_date": ""})
        assert _fails_release_date(ctx) is True

    def test_passes_when_release_date_present(self) -> None:
        """release_date='2024-01-01' → False."""
        ctx = _make_ctx(tmdb_overrides={"release_date": "2024-01-01"})
        assert _fails_release_date(ctx) is False


# ---------------------------------------------------------------------------
# _evaluate_filters
# ---------------------------------------------------------------------------


class TestEvaluateFilters:
    """Tests for the priority-ordered filter evaluation."""

    def test_passing_movie_returns_none_and_empty(self) -> None:
        """All filters pass → (None, [])."""
        ctx = _make_ctx()
        primary, all_failing = _evaluate_filters(ctx)
        assert primary is None
        assert all_failing == []

    def test_single_failure_returns_reason(self) -> None:
        """Only one filter fails → (reason, [reason])."""
        ctx = _make_ctx(imdb_overrides={"directors": []})
        primary, all_failing = _evaluate_filters(ctx)
        assert primary == "no_directors"
        assert all_failing == ["no_directors"]

    def test_multiple_failures_returns_first_as_primary(self) -> None:
        """Two filters fail → primary is highest-priority."""
        # imdb_rating=None fails before directors=[]
        ctx = _make_ctx(imdb_overrides={"imdb_rating": None, "directors": []})
        primary, all_failing = _evaluate_filters(ctx)
        assert primary == "no_imdb_rating"
        assert "no_imdb_rating" in all_failing
        assert "no_directors" in all_failing

    def test_all_filters_fail_returns_first_as_primary(self) -> None:
        """imdb=None → primary is 'missing_imdb_json', all_failing contains all imdb-dependent reasons."""
        ctx = _make_ctx(imdb_none=True, tmdb_overrides={
            "poster_url": None,
            "release_date": None,
        })
        primary, all_failing = _evaluate_filters(ctx)
        assert primary == "missing_imdb_json"
        # All 10 filters should fail when imdb=None + poster_url=None + release_date=None
        assert len(all_failing) == 10

    def test_priority_order_matches_hard_filters_list(self) -> None:
        """Verify primary reason respects the declared priority order."""
        # Fail poster_url (priority 3) and release_date (priority 9) but pass IMDB filters.
        ctx = _make_ctx(
            tmdb_overrides={"poster_url": None, "release_date": None},
        )
        primary, all_failing = _evaluate_filters(ctx)
        # poster_url (index 3) should come before release_date (index 9)
        assert primary == "no_poster_url"
        assert all_failing.index("no_poster_url") < all_failing.index("no_release_date")


# ---------------------------------------------------------------------------
# Signal scoring functions
# ---------------------------------------------------------------------------


class TestScoreImdbVoteCount:
    """Tests for _score_imdb_vote_count signal."""

    def test_uses_imdb_vote_count_field(self) -> None:
        """imdb.imdb_vote_count=1000 → positive score."""
        ctx = _make_ctx(imdb_overrides={"imdb_vote_count": 1000})
        score = _score_imdb_vote_count(ctx, TODAY)
        assert score > 0.0

    def test_zero_votes_returns_zero(self) -> None:
        """imdb_vote_count=0 → 0.0."""
        ctx = _make_ctx(imdb_overrides={"imdb_vote_count": 0})
        score = _score_imdb_vote_count(ctx, TODAY)
        assert score == pytest.approx(0.0)

    def test_imdb_none_returns_zero(self) -> None:
        """ctx.imdb=None → 0.0 (defensive branch)."""
        ctx = _make_ctx(imdb_none=True)
        score = _score_imdb_vote_count(ctx, TODAY)
        assert score == pytest.approx(0.0)


class TestScoreWatchProviders:
    """Tests for _score_watch_providers signal (binary [-1, +1])."""

    def test_has_providers_returns_positive_one(self) -> None:
        """1+ providers → 1.0."""
        ctx = _make_ctx(tmdb_overrides={
            "watch_provider_keys": struct.pack("<1I", 8),
        })
        score = _score_watch_providers(ctx, TODAY)
        assert score == pytest.approx(1.0)

    def test_no_providers_past_theater_returns_negative_one(self) -> None:
        """0 providers, old release → -1.0."""
        ctx = _make_ctx(tmdb_overrides={
            "watch_provider_keys": b"",
            "release_date": "2020-01-01",
        })
        score = _score_watch_providers(ctx, TODAY)
        assert score == pytest.approx(-1.0)

    def test_no_providers_within_theater_window_returns_positive_one(self) -> None:
        """0 providers, recent release → 1.0."""
        recent = (TODAY - datetime.timedelta(days=THEATER_WINDOW_DAYS - 10)).isoformat()
        ctx = _make_ctx(tmdb_overrides={
            "watch_provider_keys": b"",
            "release_date": recent,
        })
        score = _score_watch_providers(ctx, TODAY)
        assert score == pytest.approx(1.0)

    def test_null_release_date_no_providers_returns_negative_one(self) -> None:
        """release_date=None, 0 providers → -1.0 (conservative)."""
        ctx = _make_ctx(tmdb_overrides={
            "watch_provider_keys": b"",
            "release_date": None,
        })
        score = _score_watch_providers(ctx, TODAY)
        assert score == pytest.approx(-1.0)

    def test_invalid_date_no_providers_returns_negative_one(self) -> None:
        """Malformed date, 0 providers → -1.0."""
        ctx = _make_ctx(tmdb_overrides={
            "watch_provider_keys": b"",
            "release_date": "not-a-date",
        })
        score = _score_watch_providers(ctx, TODAY)
        assert score == pytest.approx(-1.0)


class TestScoreFeaturedReviewsChars:
    """Tests for _score_featured_reviews_chars signal."""

    def test_no_reviews_returns_negative_one(self) -> None:
        """No featured_reviews, no TMDB reviews → -1.0."""
        ctx = _make_ctx(
            imdb_overrides={"featured_reviews": []},
            tmdb_overrides={"reviews": None},
        )
        score = _score_featured_reviews_chars(ctx)
        assert score == pytest.approx(-1.0)

    def test_short_reviews_returns_zero(self) -> None:
        """Total 2000 chars → 0.0 (≤3000 tier)."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 2000}],
        })
        score = _score_featured_reviews_chars(ctx)
        assert score == pytest.approx(0.0)

    def test_medium_reviews_returns_half(self) -> None:
        """Total 5000 chars → 0.5 (3001-8000 tier)."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 5000}],
        })
        score = _score_featured_reviews_chars(ctx)
        assert score == pytest.approx(0.5)

    def test_long_reviews_returns_one(self) -> None:
        """Total 10000 chars → 1.0 (8001+ tier)."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 10000}],
        })
        score = _score_featured_reviews_chars(ctx)
        assert score == pytest.approx(1.0)

    def test_tmdb_fallback_when_imdb_empty(self) -> None:
        """IMDB reviews=[], TMDB reviews JSON has content → uses TMDB."""
        tmdb_reviews = json.dumps(["A" * 5000])
        ctx = _make_ctx(
            imdb_overrides={"featured_reviews": []},
            tmdb_overrides={"reviews": tmdb_reviews},
        )
        score = _score_featured_reviews_chars(ctx)
        assert score == pytest.approx(0.5)

    def test_tmdb_not_used_when_imdb_has_content(self) -> None:
        """IMDB has reviews, TMDB also has → only IMDB counted."""
        # IMDB: 2000 chars → 0.0 tier.  TMDB: 5000 chars → would be 0.5 if used.
        ctx = _make_ctx(
            imdb_overrides={"featured_reviews": [{"text": "A" * 2000}]},
            tmdb_overrides={"reviews": json.dumps(["B" * 5000])},
        )
        score = _score_featured_reviews_chars(ctx)
        # Should use IMDB's 2000 chars → 0.0, not TMDB's 5000 → 0.5
        assert score == pytest.approx(0.0)

    def test_malformed_tmdb_json_treated_as_no_reviews(self) -> None:
        """TMDB reviews='not json' → -1.0 (no crash)."""
        ctx = _make_ctx(
            imdb_overrides={"featured_reviews": []},
            tmdb_overrides={"reviews": "not json"},
        )
        score = _score_featured_reviews_chars(ctx)
        assert score == pytest.approx(-1.0)

    def test_tier_boundary_3000_chars(self) -> None:
        """Exactly 3000 chars → 0.0 (≤3000 tier)."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 3000}],
        })
        score = _score_featured_reviews_chars(ctx)
        assert score == pytest.approx(0.0)

    def test_tier_boundary_3001_chars(self) -> None:
        """Exactly 3001 chars → 0.5."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 3001}],
        })
        score = _score_featured_reviews_chars(ctx)
        assert score == pytest.approx(0.5)

    def test_tier_boundary_8000_chars(self) -> None:
        """Exactly 8000 chars → 0.5 (≤8000 tier)."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 8000}],
        })
        score = _score_featured_reviews_chars(ctx)
        assert score == pytest.approx(0.5)

    def test_tier_boundary_8001_chars(self) -> None:
        """Exactly 8001 chars → 1.0."""
        ctx = _make_ctx(imdb_overrides={
            "featured_reviews": [{"text": "A" * 8001}],
        })
        score = _score_featured_reviews_chars(ctx)
        assert score == pytest.approx(1.0)


class TestScorePlotTextDepth:
    """Tests for _score_plot_text_depth signal."""

    def test_all_empty_returns_zero(self) -> None:
        """No text in any field → 0.0."""
        ctx = _make_ctx(
            imdb_overrides={
                "overview": "",
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"overview_length": 0},
        )
        score = _score_plot_text_depth(ctx)
        assert score == pytest.approx(0.0)

    def test_short_overview_only(self) -> None:
        """150 chars overview → intermediate score (~0.59)."""
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
        """5001+ total chars → 1.0."""
        ctx = _make_ctx(
            imdb_overrides={
                "overview": "A" * 2000,
                "plot_summaries": ["B" * 2000],
                "synopses": ["C" * 2000],
            },
        )
        score = _score_plot_text_depth(ctx)
        assert score == pytest.approx(1.0)

    def test_tmdb_overview_fallback_when_imdb_overview_empty(self) -> None:
        """IMDB overview='' → uses tmdb.overview_length."""
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
        """IMDB overview present → TMDB ignored."""
        ctx = _make_ctx(
            imdb_overrides={
                "overview": "X" * 100,
                "plot_summaries": [],
                "synopses": [],
            },
            tmdb_overrides={"overview_length": 500},
        )
        score = _score_plot_text_depth(ctx)
        # Should use only IMDB's 100 chars, not TMDB's 500
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
    """Tests for _score_lexical_completeness signal."""

    def test_all_entities_full_returns_one(self) -> None:
        """All 6 entities present with 5+ actors/chars → 1.0."""
        ctx = _make_ctx(imdb_overrides={
            "actors": ["A1", "A2", "A3", "A4", "A5"],
            "characters": ["C1", "C2", "C3", "C4", "C5"],
            "writers": ["W1"],
            "composers": ["M1"],
            "producers": ["P1"],
            "production_companies": ["Co1"],
        })
        score = _score_lexical_completeness(ctx)
        assert score == pytest.approx(1.0)

    def test_all_entities_empty_returns_negative_one(self) -> None:
        """All empty → -1.0."""
        ctx = _make_ctx(imdb_overrides={
            "actors": [],
            "characters": [],
            "writers": [],
            "composers": [],
            "producers": [],
            "production_companies": [],
        }, tmdb_overrides={"has_production_companies": 0})
        score = _score_lexical_completeness(ctx)
        assert score == pytest.approx(-1.0)

    def test_actors_below_threshold_half_score(self) -> None:
        """3 actors → 0.5 sub-score."""
        ctx = _make_ctx(imdb_overrides={
            "actors": ["A1", "A2", "A3"],
            "characters": ["C1", "C2", "C3", "C4", "C5"],
            "writers": ["W1"],
            "composers": ["M1"],
            "producers": ["P1"],
            "production_companies": ["Co1"],
        })
        score = _score_lexical_completeness(ctx)
        # total = 0.5 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 = 5.5 → (5.5 - 3) / 3 = 0.833...
        expected = (0.5 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0 - 3.0) / 3.0
        assert score == pytest.approx(expected, abs=1e-6)

    def test_actors_at_threshold_full_score(self) -> None:
        """5 actors → 1.0 sub-score."""
        ctx = _make_ctx(imdb_overrides={
            "actors": ["A1", "A2", "A3", "A4", "A5"],
        })
        # With default imdb having all other entities, sub-score for actors = 1.0
        score = _score_lexical_completeness(ctx)
        # All entities full → 1.0
        assert score == pytest.approx(1.0)

    def test_production_companies_tmdb_fallback(self) -> None:
        """IMDB empty, tmdb.has_production_companies=True → 1.0 sub-score."""
        ctx = _make_ctx(
            imdb_overrides={"production_companies": []},
            tmdb_overrides={"has_production_companies": 1},
        )
        score = _score_lexical_completeness(ctx)
        # prodco_sub is still 1.0 via TMDB fallback → all entities full → 1.0
        assert score == pytest.approx(1.0)

    def test_midpoint_score(self) -> None:
        """3 entities present at full → score = 0.0."""
        ctx = _make_ctx(imdb_overrides={
            "actors": ["A1", "A2", "A3", "A4", "A5"],
            "characters": ["C1", "C2", "C3", "C4", "C5"],
            "writers": ["W1"],
            "composers": [],
            "producers": [],
            "production_companies": [],
        }, tmdb_overrides={"has_production_companies": 0})
        score = _score_lexical_completeness(ctx)
        # total = 1.0 + 1.0 + 1.0 + 0.0 + 0.0 + 0.0 = 3.0 → (3.0 - 3) / 3 = 0.0
        assert score == pytest.approx(0.0)


class TestScoreDataCompleteness:
    """Tests for _score_data_completeness signal."""

    def test_all_fields_full_returns_one(self) -> None:
        """All 6 fields present at max tiers → 1.0."""
        ctx = _make_ctx(imdb_overrides={
            "plot_keywords": ["k1", "k2", "k3", "k4", "k5"],
            "overall_keywords": ["o1", "o2", "o3", "o4"],
            "filming_locations": ["LA"],
            "parental_guide_items": [{"category": "v"}],
            "maturity_rating": "R",
            "budget": 100_000_000,
        })
        score = _score_data_completeness(ctx)
        assert score == pytest.approx(1.0)

    def test_all_fields_empty_returns_negative_one(self) -> None:
        """All empty → -1.0."""
        ctx = _make_ctx(
            imdb_overrides={
                "plot_keywords": [],
                "overall_keywords": [],
                "filming_locations": [],
                "parental_guide_items": [],
                "maturity_rating": None,
                "budget": None,
            },
            tmdb_overrides={"maturity_rating": None, "budget": 0},
        )
        score = _score_data_completeness(ctx)
        assert score == pytest.approx(-1.0)

    def test_plot_keywords_tiered(self) -> None:
        """0→0.0, 2→0.5, 5→1.0."""
        # 0 keywords
        ctx0 = _make_ctx(imdb_overrides={"plot_keywords": []})
        s0 = _score_data_completeness(ctx0)

        # 2 keywords
        ctx2 = _make_ctx(imdb_overrides={"plot_keywords": ["a", "b"]})
        s2 = _score_data_completeness(ctx2)

        # 5 keywords
        ctx5 = _make_ctx(imdb_overrides={"plot_keywords": ["a", "b", "c", "d", "e"]})
        s5 = _score_data_completeness(ctx5)

        # Each tier increases the score
        assert s0 < s2 < s5

    def test_overall_keywords_tiered(self) -> None:
        """0→0.0, 1→0.25, 3→0.5, 4→1.0."""
        base_overrides = {
            "plot_keywords": [],
            "filming_locations": [],
            "parental_guide_items": [],
            "maturity_rating": None,
            "budget": None,
        }
        base_tmdb = {"maturity_rating": None, "budget": 0}

        # 0 keywords
        ctx0 = _make_ctx(imdb_overrides={**base_overrides, "overall_keywords": []}, tmdb_overrides=base_tmdb)
        s0 = _score_data_completeness(ctx0)

        # 1 keyword
        ctx1 = _make_ctx(imdb_overrides={**base_overrides, "overall_keywords": ["a"]}, tmdb_overrides=base_tmdb)
        s1 = _score_data_completeness(ctx1)

        # 3 keywords
        ctx3 = _make_ctx(imdb_overrides={**base_overrides, "overall_keywords": ["a", "b", "c"]}, tmdb_overrides=base_tmdb)
        s3 = _score_data_completeness(ctx3)

        # 4 keywords
        ctx4 = _make_ctx(imdb_overrides={**base_overrides, "overall_keywords": ["a", "b", "c", "d"]}, tmdb_overrides=base_tmdb)
        s4 = _score_data_completeness(ctx4)

        assert s0 < s1 < s3 < s4

    def test_maturity_rating_tmdb_fallback(self) -> None:
        """IMDB empty, tmdb.maturity_rating present → 1.0 sub-score."""
        ctx = _make_ctx(
            imdb_overrides={"maturity_rating": None},
            tmdb_overrides={"maturity_rating": "PG-13"},
        )
        score = _score_data_completeness(ctx)
        # Same as default with maturity_rating present via fallback
        ctx_with = _make_ctx(imdb_overrides={"maturity_rating": "R"})
        score_with = _score_data_completeness(ctx_with)
        assert score == pytest.approx(score_with)

    def test_budget_tmdb_fallback(self) -> None:
        """IMDB empty, tmdb.budget present → 1.0 sub-score."""
        ctx = _make_ctx(
            imdb_overrides={"budget": None},
            tmdb_overrides={"budget": 50_000_000},
        )
        score = _score_data_completeness(ctx)
        # Same as default with budget present via fallback
        ctx_with = _make_ctx(imdb_overrides={"budget": 100_000_000})
        score_with = _score_data_completeness(ctx_with)
        assert score == pytest.approx(score_with)


class TestScoreTmdbPopularity:
    """Tests for _score_tmdb_popularity signal."""

    def test_zero_popularity(self) -> None:
        """popularity=0 → 0.0."""
        ctx = _make_ctx(tmdb_overrides={"popularity": 0})
        score = _score_tmdb_popularity(ctx)
        assert score == pytest.approx(0.0)

    def test_none_popularity_treated_as_zero(self) -> None:
        """popularity=None → 0.0."""
        ctx = _make_ctx(tmdb_overrides={"popularity": None})
        score = _score_tmdb_popularity(ctx)
        assert score == pytest.approx(0.0)


class TestScoreMetacriticRating:
    """Tests for _score_metacritic_rating signal."""

    def test_present_returns_one(self) -> None:
        """metacritic_rating=73 → 1.0."""
        ctx = _make_ctx(imdb_overrides={"metacritic_rating": 73})
        score = _score_metacritic_rating(ctx)
        assert score == pytest.approx(1.0)

    def test_none_returns_zero(self) -> None:
        """metacritic_rating=None → 0.0."""
        ctx = _make_ctx(imdb_overrides={"metacritic_rating": None})
        score = _score_metacritic_rating(ctx)
        assert score == pytest.approx(0.0)

    def test_imdb_none_returns_zero(self) -> None:
        """ctx.imdb=None → 0.0."""
        ctx = _make_ctx(imdb_none=True)
        score = _score_metacritic_rating(ctx)
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_imdb_quality_score
# ---------------------------------------------------------------------------


class TestComputeImdbQualityScore:
    """Tests for the composite weighted quality score."""

    def test_perfect_movie_near_max(self) -> None:
        """All signals at max → score near theoretical max."""
        ctx = _make_ctx()
        score = compute_imdb_quality_score(ctx, TODAY)
        # With all signals maxed, score should be high (close to sum of positive weights)
        assert score > 0.5

    def test_minimal_movie_near_min(self) -> None:
        """All signals at min → score near theoretical min."""
        ctx = _make_ctx(
            imdb_overrides={
                "imdb_vote_count": 0,
                "featured_reviews": [],
                "overview": "",
                "plot_summaries": [],
                "synopses": [],
                "actors": [],
                "characters": [],
                "writers": [],
                "composers": [],
                "producers": [],
                "production_companies": [],
                "plot_keywords": [],
                "overall_keywords": ["x"],  # need at least 1 to pass hard filter
                "filming_locations": [],
                "parental_guide_items": [],
                "maturity_rating": None,
                "budget": None,
                "metacritic_rating": None,
            },
            tmdb_overrides={
                "watch_provider_keys": b"",
                "popularity": 0,
                "overview_length": 0,
                "maturity_rating": None,
                "budget": 0,
                "reviews": None,
                "has_production_companies": 0,
                "release_date": "2020-01-01",
            },
        )
        score = compute_imdb_quality_score(ctx, TODAY)
        assert score < 0.0

    def test_weights_sum_to_one(self) -> None:
        """WEIGHTS dict sums to 1.0 (module-level guard)."""
        total = sum(WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_individual_signal_contribution(self) -> None:
        """Verify that a single signal's contribution matches weight × signal value."""
        # Create two contexts identical except for metacritic_rating (binary 0/1)
        ctx_with = _make_ctx(imdb_overrides={"metacritic_rating": 73})
        ctx_without = _make_ctx(imdb_overrides={"metacritic_rating": None})
        score_with = compute_imdb_quality_score(ctx_with, TODAY)
        score_without = compute_imdb_quality_score(ctx_without, TODAY)
        delta = score_with - score_without
        assert delta == pytest.approx(WEIGHTS["metacritic_rating"] * 1.0, abs=1e-9)

    def test_deterministic(self) -> None:
        """Same inputs → same output."""
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
