"""
Unit tests for Movie class methods in schemas/movie.py.

These test the new Movie class (tracker-backed), NOT the old BaseMovie
in implementation/classes/movie.py. The Movie fixture builds objects
directly via constructor rather than from a SQLite database.
"""

import json
import importlib
import sqlite3
import sys
from types import ModuleType

import pytest

try:
    importlib.import_module("orjson")
except ModuleNotFoundError:
    orjson_module = ModuleType("orjson")
    orjson_module.dumps = lambda obj, *args, **kwargs: json.dumps(obj).encode("utf-8")
    orjson_module.loads = lambda data, *args, **kwargs: json.loads(
        data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else data
    )
    sys.modules["orjson"] = orjson_module

from implementation.classes.enums import BudgetSize
from movie_ingestion import tracker as tracker_module
from movie_ingestion.imdb_scraping.models import AwardNomination
from schemas.enums import AwardOutcome, BoxOfficeStatus, LineagePosition
from schemas.movie import Movie, TMDBData, IMDBData


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------

def _make_movie(**overrides) -> Movie:
    """Build a minimal valid Movie with targeted overrides.

    Accepts top-level shorthand keys that map into tmdb_data / imdb_data
    sub-models, plus direct metadata fields. Any key not recognized as
    a tmdb/imdb shorthand is passed through to Movie() directly.
    """
    tmdb_defaults = {
        "tmdb_id": 1,
        "imdb_id": "tt0000001",
        "title": "Spider-Man",
        "release_date": "2002-05-03",
        "duration": 121,
        "budget": 139_000_000,
        "maturity_rating": "PG-13",
    }
    imdb_defaults = {
        "tmdb_id": 1,
        "original_title": None,
        "overview": "A student gets spider-like abilities.",
        "imdb_rating": 7.4,
        "metacritic_rating": 73.0,
        "budget": None,
        "genres": ["Action", "Adventure"],
        "countries_of_origin": ["USA"],
        "production_companies": ["Columbia Pictures"],
        "filming_locations": ["New York", "Los Angeles", "Chicago", "London"],
        "languages": ["English", "Spanish"],
        "maturity_rating": None,
        "maturity_reasoning": [],
    }

    tmdb_overrides = overrides.pop("tmdb_data", {})
    imdb_overrides = overrides.pop("imdb_data", {})

    tmdb_data = TMDBData(**{**tmdb_defaults, **tmdb_overrides})
    imdb_data = IMDBData(**{**imdb_defaults, **imdb_overrides})
    return Movie(tmdb_data=tmdb_data, imdb_data=imdb_data, **overrides)


def _write_tracker_db_with_franchise(path, franchise_payload: dict) -> None:
    """Create a minimal tracker DB containing one movie with franchise metadata."""
    with sqlite3.connect(path) as db:
        db.executescript(tracker_module._SCHEMA_SQL)
        db.execute(
            """
            INSERT INTO tmdb_data (
                tmdb_id, imdb_id, title, release_date, duration, poster_url,
                watch_provider_keys, vote_count, popularity, vote_average,
                overview_length, genre_count, has_revenue, has_budget,
                has_production_companies, has_production_countries,
                has_keywords, has_cast_and_crew, budget, maturity_rating,
                reviews, collection_name, revenue
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "tt0000001",
                "Spider-Man",
                "2002-05-03",
                121,
                None,
                b"",
                100,
                1.0,
                7.0,
                120,
                2,
                1,
                1,
                1,
                1,
                1,
                1,
                139_000_000,
                "PG-13",
                json.dumps([]),
                "Spider-Man Collection",
                0,
            ),
        )
        db.execute(
            """
            INSERT INTO imdb_data (
                tmdb_id, imdb_title_type, original_title, maturity_rating, overview,
                imdb_rating, imdb_vote_count, metacritic_rating, reception_summary,
                budget, overall_keywords, genres, countries_of_origin,
                production_companies, filming_locations, languages, synopses,
                plot_summaries, plot_keywords, maturity_reasoning, directors,
                writers, actors, characters, producers, composers, review_themes,
                parental_guide_items, featured_reviews, awards, box_office_worldwide
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "movie",
                None,
                None,
                "A student gets spider-like abilities.",
                7.4,
                1000,
                73.0,
                "Well received.",
                None,
                json.dumps(["hero"]),
                json.dumps(["Action", "Adventure"]),
                json.dumps(["USA"]),
                json.dumps(["Columbia Pictures"]),
                json.dumps(["New York"]),
                json.dumps(["English"]),
                json.dumps([]),
                json.dumps([]),
                json.dumps([]),
                json.dumps([]),
                json.dumps(["Sam Raimi"]),
                json.dumps(["David Koepp"]),
                json.dumps(["Tobey Maguire"]),
                json.dumps(["Peter Parker"]),
                json.dumps([]),
                json.dumps([]),
                json.dumps([]),
                json.dumps([]),
                json.dumps([]),
                json.dumps([]),
                None,
            ),
        )
        db.execute(
            "INSERT INTO generated_metadata (tmdb_id, franchise) VALUES (?, ?)",
            (1, json.dumps(franchise_payload)),
        )
        db.commit()


# ---------------------------------------------------------------------------
# title_with_original
# ---------------------------------------------------------------------------

# TestTitleWithOriginal removed — title_with_original() was deleted from Movie.


# ---------------------------------------------------------------------------
# maturity_text_short
# ---------------------------------------------------------------------------

class TestMaturityTextShort:
    def test_prefers_imdb_reasoning(self):
        """Should join maturity_reasoning list with '. ' separator."""
        movie = _make_movie(
            imdb_data={"maturity_reasoning": ["Intense violence", "Brief language"]},
        )
        result = movie.maturity_text_short()
        assert result == "Intense violence. Brief language"

    def test_falls_back_to_mpa_description(self):
        """With no reasoning, should map known MPA ratings to descriptions."""
        movie = _make_movie(
            imdb_data={"maturity_reasoning": [], "maturity_rating": None},
            tmdb_data={"maturity_rating": "R"},
        )
        result = movie.maturity_text_short()
        assert "restricted" in result
        assert "mature audiences" in result

    def test_empty_for_nonstandard_rating(self):
        """Non-standard ratings like 'Not Rated' should return empty string."""
        movie = _make_movie(
            imdb_data={"maturity_reasoning": [], "maturity_rating": "Not Rated"},
        )
        assert movie.maturity_text_short() == ""

    def test_empty_when_no_maturity_data(self):
        """No reasoning and no rating should return empty string."""
        movie = _make_movie(
            imdb_data={"maturity_reasoning": [], "maturity_rating": None},
            tmdb_data={"maturity_rating": None},
        )
        assert movie.maturity_text_short() == ""

    def test_unrated_returns_empty(self):
        """'Unrated' is also non-standard — should return empty."""
        movie = _make_movie(
            imdb_data={"maturity_reasoning": [], "maturity_rating": "Unrated"},
        )
        assert movie.maturity_text_short() == ""


# ---------------------------------------------------------------------------
# resolved_budget
# ---------------------------------------------------------------------------

class TestResolvedBudget:
    def test_prefers_imdb(self):
        movie = _make_movie(
            imdb_data={"budget": 50_000_000},
            tmdb_data={"budget": 30_000_000},
        )
        assert movie.resolved_budget() == 50_000_000

    def test_falls_back_to_tmdb(self):
        movie = _make_movie(
            imdb_data={"budget": None},
            tmdb_data={"budget": 30_000_000},
        )
        assert movie.resolved_budget() == 30_000_000

    def test_zero_treated_as_missing(self):
        """Zero budget means 'unknown' — should return None."""
        movie = _make_movie(
            imdb_data={"budget": 0},
            tmdb_data={"budget": 0},
        )
        assert movie.resolved_budget() is None

    def test_none_when_both_missing(self):
        movie = _make_movie(
            imdb_data={"budget": None},
            tmdb_data={"budget": None},
        )
        assert movie.resolved_budget() is None

    def test_imdb_zero_falls_back_to_tmdb_positive(self):
        """IMDB budget=0 is falsy, should fall through to TMDB."""
        movie = _make_movie(
            imdb_data={"budget": 0},
            tmdb_data={"budget": 15_000_000},
        )
        assert movie.resolved_budget() == 15_000_000


# ---------------------------------------------------------------------------
# box_office_status
# ---------------------------------------------------------------------------

class TestBoxOfficeStatus:
    def test_none_when_release_date_missing(self):
        movie = _make_movie(tmdb_data={"release_date": None}, imdb_data={"budget": 10_000_000})
        assert movie.box_office_status() is None

    def test_none_when_release_date_invalid(self):
        movie = _make_movie(
            tmdb_data={"release_date": "not-a-date", "revenue": 50_000_000},
            imdb_data={"budget": 10_000_000},
        )
        assert movie.box_office_status() is None

    def test_none_when_pre_1980(self):
        movie = _make_movie(
            tmdb_data={"release_date": "1979-12-01", "revenue": 50_000_000},
            imdb_data={"budget": 10_000_000},
        )
        assert movie.box_office_status() is None

    def test_none_when_budget_missing(self):
        movie = _make_movie(
            tmdb_data={"revenue": 50_000_000, "budget": None},
            imdb_data={"budget": None},
        )
        assert movie.box_office_status() is None

    def test_none_when_budget_zero(self):
        movie = _make_movie(
            tmdb_data={"revenue": 50_000_000, "budget": 0},
            imdb_data={"budget": 0},
        )
        assert movie.box_office_status() is None

    def test_none_when_gross_missing(self):
        movie = _make_movie(
            tmdb_data={"revenue": None},
            imdb_data={"budget": 10_000_000, "box_office_worldwide": None},
        )
        assert movie.box_office_status() is None

    def test_none_when_gross_zero(self):
        movie = _make_movie(
            tmdb_data={"revenue": 0},
            imdb_data={"budget": 10_000_000, "box_office_worldwide": 0},
        )
        assert movie.box_office_status() is None

    def test_hit_when_ratio_at_least_three_and_budget_floor_met(self):
        movie = _make_movie(
            tmdb_data={"budget": 10_000_000},
            imdb_data={"budget": 10_000_000, "box_office_worldwide": 30_000_000},
        )
        assert movie.box_office_status() == BoxOfficeStatus.HIT

    def test_none_when_hit_ratio_met_but_budget_below_floor(self):
        movie = _make_movie(
            tmdb_data={"budget": 500_000},
            imdb_data={"budget": 500_000, "box_office_worldwide": 2_000_000},
        )
        assert movie.box_office_status() is None

    def test_flop_when_ratio_at_most_one(self):
        movie = _make_movie(
            tmdb_data={"budget": 20_000_000},
            imdb_data={"budget": 20_000_000, "box_office_worldwide": 20_000_000},
        )
        assert movie.box_office_status() == BoxOfficeStatus.FLOP

    def test_none_in_ambiguous_zone(self):
        movie = _make_movie(
            tmdb_data={"budget": 20_000_000},
            imdb_data={"budget": 20_000_000, "box_office_worldwide": 40_000_000},
        )
        assert movie.box_office_status() is None

    def test_prefers_imdb_gross_over_tmdb_revenue(self):
        movie = _make_movie(
            tmdb_data={"budget": 10_000_000, "revenue": 35_000_000},
            imdb_data={"budget": 10_000_000, "box_office_worldwide": 8_000_000},
        )
        assert movie.box_office_status() == BoxOfficeStatus.FLOP

    def test_falls_back_to_tmdb_revenue_when_imdb_gross_missing(self):
        movie = _make_movie(
            tmdb_data={"budget": 10_000_000, "revenue": 35_000_000},
            imdb_data={"budget": 10_000_000, "box_office_worldwide": None},
        )
        assert movie.box_office_status() == BoxOfficeStatus.HIT


# ---------------------------------------------------------------------------
# reception_score
# ---------------------------------------------------------------------------

class TestReceptionScore:
    def test_blended(self):
        """Both present: 40% IMDB×10 + 60% Metacritic."""
        movie = _make_movie(imdb_data={"imdb_rating": 7.0, "metacritic_rating": 80.0})
        expected = 0.4 * 70 + 0.6 * 80  # 28 + 48 = 76.0
        assert movie.reception_score() == pytest.approx(expected)

    def test_imdb_only(self):
        movie = _make_movie(imdb_data={"imdb_rating": 8.0, "metacritic_rating": None})
        assert movie.reception_score() == pytest.approx(80.0)

    def test_metacritic_only(self):
        movie = _make_movie(imdb_data={"imdb_rating": None, "metacritic_rating": 65.0})
        assert movie.reception_score() == pytest.approx(65.0)

    def test_none_when_both_missing(self):
        movie = _make_movie(imdb_data={"imdb_rating": None, "metacritic_rating": None})
        assert movie.reception_score() is None

    def test_zero_rating_treated_as_falsy(self):
        """0.0 ratings are falsy — should return None."""
        movie = _make_movie(imdb_data={"imdb_rating": 0.0, "metacritic_rating": 0.0})
        assert movie.reception_score() is None


# ---------------------------------------------------------------------------
# production_text
# ---------------------------------------------------------------------------

# TestProductionText removed — production_text() was deleted from Movie.


# ---------------------------------------------------------------------------
# languages_text
# ---------------------------------------------------------------------------

class TestLanguagesText:
    def test_primary_only(self):
        movie = _make_movie(imdb_data={"languages": ["English"]})
        result = movie.languages_text()
        assert result == "primary language: English"

    def test_primary_plus_additional(self):
        movie = _make_movie(imdb_data={"languages": ["English", "Spanish", "French"]})
        result = movie.languages_text()
        assert "primary language: English" in result
        assert "additional languages: Spanish, French" in result

    def test_empty_when_no_languages(self):
        movie = _make_movie(imdb_data={"languages": []})
        assert movie.languages_text() == ""


# ---------------------------------------------------------------------------
# release_decade_bucket
# ---------------------------------------------------------------------------

class TestReleaseDecadeBucket:
    def test_silent_era(self):
        movie = _make_movie(tmdb_data={"release_date": "1925-01-01"})
        assert movie.release_decade_bucket() == "Release date: 1920s, silent era & early cinema"

    def test_golden_age(self):
        movie = _make_movie(tmdb_data={"release_date": "1942-06-15"})
        assert movie.release_decade_bucket() == "Release date: 1940s, golden age of hollywood"

    def test_modern(self):
        movie = _make_movie(tmdb_data={"release_date": "2005-03-20"})
        assert movie.release_decade_bucket() == "Release date: 2000s, 00s"

    def test_empty_when_no_date(self):
        movie = _make_movie(tmdb_data={"release_date": None})
        assert movie.release_decade_bucket() == ""

    def test_invalid_date(self):
        movie = _make_movie(tmdb_data={"release_date": "not-a-date"})
        assert movie.release_decade_bucket() == ""

    def test_1930_boundary(self):
        """1930 should be golden age, not silent era."""
        movie = _make_movie(tmdb_data={"release_date": "1930-01-01"})
        assert "golden age" in movie.release_decade_bucket()

    def test_1950_boundary(self):
        """1950 should be modern format, not golden age."""
        movie = _make_movie(tmdb_data={"release_date": "1950-01-01"})
        result = movie.release_decade_bucket()
        assert "golden age" not in result
        assert "50s" in result


# ---------------------------------------------------------------------------
# budget_bucket_for_era
# ---------------------------------------------------------------------------

class TestBudgetBucketForEra:
    def test_small(self):
        """Budget below the era's small threshold."""
        movie = _make_movie(
            tmdb_data={"release_date": "2010-01-01", "budget": 5_000_000},
            imdb_data={"budget": 5_000_000},
        )
        assert movie.budget_bucket_for_era() == BudgetSize.SMALL

    def test_large(self):
        """Budget above the era's large threshold."""
        movie = _make_movie(
            tmdb_data={"release_date": "2010-01-01", "budget": 200_000_000},
            imdb_data={"budget": 200_000_000},
        )
        assert movie.budget_bucket_for_era() == BudgetSize.LARGE

    def test_mid_range_none(self):
        """Budget in the normal range returns None."""
        movie = _make_movie(
            tmdb_data={"release_date": "2010-01-01", "budget": 50_000_000},
            imdb_data={"budget": 50_000_000},
        )
        assert movie.budget_bucket_for_era() is None

    def test_none_when_no_budget(self):
        movie = _make_movie(
            tmdb_data={"budget": None},
            imdb_data={"budget": None},
        )
        assert movie.budget_bucket_for_era() is None

    def test_none_when_no_release_date(self):
        movie = _make_movie(
            tmdb_data={"release_date": None},
            imdb_data={"budget": 50_000_000},
        )
        assert movie.budget_bucket_for_era() is None

    def test_future_year_clamps_to_2020s(self):
        """Year 2035 should clamp to 2020s thresholds."""
        # 2020 large threshold is 185M, so 200M should be LARGE
        movie = _make_movie(
            tmdb_data={"release_date": "2035-01-01", "budget": 200_000_000},
            imdb_data={"budget": 200_000_000},
        )
        assert movie.budget_bucket_for_era() == BudgetSize.LARGE


# ---------------------------------------------------------------------------
# _interpolated_thresholds
# ---------------------------------------------------------------------------

class TestInterpolatedThresholds:
    def test_exact_decade(self):
        """Exact decade year should return that decade's thresholds."""
        movie = _make_movie()
        small, large = movie._interpolated_thresholds(2000)
        assert small == 12_000_000
        assert large == 110_000_000

    def test_mid_decade(self):
        """Mid-decade year should interpolate between adjacent decades."""
        movie = _make_movie()
        small, large = movie._interpolated_thresholds(1985)
        # 1980: (5M, 45M), 1990: (9M, 80M), t=0.5
        expected_small = 5_000_000 + 0.5 * (9_000_000 - 5_000_000)  # 7M
        expected_large = 45_000_000 + 0.5 * (80_000_000 - 45_000_000)  # 62.5M
        assert small == pytest.approx(expected_small)
        assert large == pytest.approx(expected_large)

    def test_clamp_below_1920(self):
        """Year below 1920 should use 1920 thresholds."""
        movie = _make_movie()
        small, large = movie._interpolated_thresholds(1900)
        expected = movie._DECADE_THRESHOLDS[1920]
        assert (small, large) == expected

    def test_clamp_above_2020(self):
        """Year at or above 2020 should use 2020 thresholds."""
        movie = _make_movie()
        small, large = movie._interpolated_thresholds(2025)
        expected = movie._DECADE_THRESHOLDS[2020]
        assert (small, large) == expected

    def test_boundary_no_interpolation(self):
        """Decade boundary year needs no interpolation (t=0)."""
        movie = _make_movie()
        small, large = movie._interpolated_thresholds(1920)
        expected = movie._DECADE_THRESHOLDS[1920]
        assert (small, large) == expected


# ---------------------------------------------------------------------------
# deduplicated_genres
# ---------------------------------------------------------------------------

class TestDeduplicatedGenres:
    def test_merges_and_deduplicates(self):
        """Overlapping genres from LLM and IMDB should be deduped."""
        from schemas.metadata import PlotAnalysisOutput, ElevatorPitchWithJustification
        plot_analysis = PlotAnalysisOutput(
            genre_signatures=["Action", "Thriller"],
            thematic_concepts=[],
            elevator_pitch_with_justification=ElevatorPitchWithJustification(
                explanation_and_justification="x", elevator_pitch="y",
            ),
            conflict_type=[],
            character_arcs=[],
            generalized_plot_overview="Overview.",
        )
        movie = _make_movie(
            imdb_data={"genres": ["Action", "Drama"]},
            plot_analysis_metadata=plot_analysis,
        )
        genres = movie.deduplicated_genres()
        assert "action" in genres
        assert genres.count("action") == 1  # no duplicates
        assert "thriller" in genres
        assert "drama" in genres

    def test_sorted(self):
        movie = _make_movie(imdb_data={"genres": ["Thriller", "Action", "Drama"]})
        genres = movie.deduplicated_genres()
        assert genres == sorted(genres)

    def test_without_plot_analysis(self):
        """When metadata is None, only IMDB genres used."""
        movie = _make_movie(imdb_data={"genres": ["Comedy"]})
        assert movie.deduplicated_genres() == ["comedy"]


# ---------------------------------------------------------------------------
# is_animation
# ---------------------------------------------------------------------------

class TestIsAnimation:
    def test_true(self):
        movie = _make_movie(imdb_data={"genres": ["Animation", "Comedy"]})
        assert movie.is_animation() is True

    def test_false(self):
        movie = _make_movie(imdb_data={"genres": ["Action", "Drama"]})
        assert movie.is_animation() is False

    def test_case_insensitive(self):
        """'animation' lowercase should still match."""
        movie = _make_movie(imdb_data={"genres": ["animation"]})
        assert movie.is_animation() is True


# ---------------------------------------------------------------------------
# reception_tier
# ---------------------------------------------------------------------------

class TestReceptionTier:
    @pytest.mark.parametrize("imdb_rating,metacritic,expected_tier", [
        (9.0, 90.0, "Universally acclaimed"),         # score = 90
        (7.0, 70.0, "Generally favorable reviews"),    # score = 70
        (5.0, 50.0, "Mixed or average reviews"),       # score = 50
        (3.0, 30.0, "Generally unfavorable reviews"),  # score = 30
        (1.0, 10.0, "Overwhelming dislike"),           # score = 10
    ])
    def test_all_boundaries(self, imdb_rating, metacritic, expected_tier):
        movie = _make_movie(
            imdb_data={"imdb_rating": imdb_rating, "metacritic_rating": metacritic},
        )
        assert movie.reception_tier() == expected_tier

    def test_none_when_no_score(self):
        movie = _make_movie(imdb_data={"imdb_rating": None, "metacritic_rating": None})
        assert movie.reception_tier() is None

    def test_exact_boundary_81(self):
        """Score of exactly 81 should be 'Universally acclaimed'."""
        # imdb_only: 8.1 * 10 = 81.0
        movie = _make_movie(imdb_data={"imdb_rating": 8.1, "metacritic_rating": None})
        assert movie.reception_tier() == "Universally acclaimed"

    def test_exact_boundary_61(self):
        """Score of exactly 61 should be 'Generally favorable reviews'."""
        movie = _make_movie(imdb_data={"imdb_rating": 6.1, "metacritic_rating": None})
        assert movie.reception_tier() == "Generally favorable reviews"


# ---------------------------------------------------------------------------
# release_ts
# ---------------------------------------------------------------------------

class TestReleaseTs:
    def test_valid_date_returns_unix_timestamp(self):
        """'2002-05-03' should produce the expected UTC Unix timestamp."""
        movie = _make_movie(tmdb_data={"release_date": "2002-05-03"})
        assert movie.release_ts() == 1020384000

    def test_none_when_release_date_missing(self):
        movie = _make_movie(tmdb_data={"release_date": None})
        assert movie.release_ts() is None

    def test_invalid_format_raises_value_error(self):
        """Malformed date string should raise ValueError from strptime."""
        movie = _make_movie(tmdb_data={"release_date": "not-a-date"})
        with pytest.raises(ValueError):
            movie.release_ts()

    def test_epoch_date(self):
        """'1970-01-01' should produce timestamp 0."""
        movie = _make_movie(tmdb_data={"release_date": "1970-01-01"})
        assert movie.release_ts() == 0


# ---------------------------------------------------------------------------
# maturity_rating_and_rank
# ---------------------------------------------------------------------------

class TestMaturityRatingAndRank:
    def test_known_rating_pg13(self):
        """'PG-13' should resolve to ('pg-13', 3)."""
        movie = _make_movie(
            imdb_data={"maturity_rating": "PG-13"},
            tmdb_data={"maturity_rating": None},
        )
        assert movie.maturity_rating_and_rank() == ("pg-13", 3)

    def test_unknown_rating_falls_back_to_unrated(self):
        """Unrecognized string should resolve to ('unrated', 999)."""
        movie = _make_movie(
            imdb_data={"maturity_rating": "XYZ-RATING"},
            tmdb_data={"maturity_rating": None},
        )
        assert movie.maturity_rating_and_rank() == ("unrated", 999)

    def test_prefers_imdb_over_tmdb(self):
        """IMDB maturity rating should take precedence over TMDB."""
        movie = _make_movie(
            imdb_data={"maturity_rating": "R"},
            tmdb_data={"maturity_rating": "PG"},
        )
        label, rank = movie.maturity_rating_and_rank()
        assert label == "r"
        assert rank == 4

    def test_falls_back_to_tmdb_when_imdb_missing(self):
        """When IMDB has no maturity rating, should use TMDB's."""
        movie = _make_movie(
            imdb_data={"maturity_rating": None},
            tmdb_data={"maturity_rating": "G"},
        )
        label, rank = movie.maturity_rating_and_rank()
        assert label == "g"
        assert rank == 1


# ---------------------------------------------------------------------------
# normalized_title_tokens
# ---------------------------------------------------------------------------

class TestMovieNormalizedTitleTokens:
    def test_title_only_no_original(self):
        """No original_title — returns only title tokens."""
        movie = _make_movie(
            tmdb_data={"title": "Spider-Man"},
            imdb_data={"original_title": None},
        )
        tokens = movie.normalized_title_tokens()
        assert tokens == ["spider-man", "spider", "man"]

    def test_different_original_title_appends_new_tokens(self):
        """Original title with new tokens appends them after title tokens."""
        movie = _make_movie(
            tmdb_data={"title": "Spider-Man"},
            imdb_data={"original_title": "El Hombre Araña"},
        )
        tokens = movie.normalized_title_tokens()
        # Title tokens come first, then unique original tokens
        assert tokens[:3] == ["spider-man", "spider", "man"]
        assert "el" in tokens
        assert "hombre" in tokens

    def test_same_original_title_no_duplicates(self):
        """original_title == title should not add duplicate tokens."""
        movie = _make_movie(
            tmdb_data={"title": "Spider-Man"},
            imdb_data={"original_title": "Spider-Man"},
        )
        tokens = movie.normalized_title_tokens()
        assert tokens == ["spider-man", "spider", "man"]

    def test_shared_tokens_deduplicated(self):
        """Tokens shared between title and original appear only once."""
        movie = _make_movie(
            tmdb_data={"title": "The Matrix"},
            imdb_data={"original_title": "Matrix Reloaded"},
        )
        tokens = movie.normalized_title_tokens()
        assert tokens.count("matrix") == 1
        assert "reloaded" in tokens
        assert "the" in tokens


# ---------------------------------------------------------------------------
# genre_ids
# ---------------------------------------------------------------------------

class TestGenreIds:
    def test_valid_genres_return_ids(self):
        """Known genre strings should map to their integer genre IDs."""
        movie = _make_movie(imdb_data={"genres": ["Action", "Drama"]})
        ids = movie.genre_ids()
        assert len(ids) == 2
        assert all(isinstance(gid, int) for gid in ids)

    def test_unknown_genre_is_skipped(self):
        """Unrecognized genre string should be silently skipped."""
        movie = _make_movie(imdb_data={"genres": ["Action", "NotARealGenre"]})
        ids = movie.genre_ids()
        assert len(ids) == 1  # Only Action resolves

    def test_empty_genres(self):
        movie = _make_movie(imdb_data={"genres": []})
        assert movie.genre_ids() == []


# ---------------------------------------------------------------------------
# watch_offer_keys
# ---------------------------------------------------------------------------

class TestWatchOfferKeys:
    def test_returns_tmdb_watch_provider_keys(self):
        """Should return the pre-decoded keys from tmdb_data."""
        movie = _make_movie(tmdb_data={"watch_provider_keys": [100, 200, 300]})
        assert movie.watch_offer_keys() == [100, 200, 300]

    def test_empty_when_no_providers(self):
        movie = _make_movie(tmdb_data={"watch_provider_keys": []})
        assert movie.watch_offer_keys() == []


# ---------------------------------------------------------------------------
# audio_language_ids
# ---------------------------------------------------------------------------

class TestAudioLanguageIds:
    def test_valid_language_returns_id(self):
        """Known language string should map to its integer ID."""
        movie = _make_movie(imdb_data={"languages": ["English"]})
        ids = movie.audio_language_ids()
        assert len(ids) == 1
        assert isinstance(ids[0], int)

    def test_unknown_language_is_skipped(self):
        """Unrecognized language string should be silently skipped."""
        movie = _make_movie(imdb_data={"languages": ["English", "Zzyzxian"]})
        ids = movie.audio_language_ids()
        assert len(ids) == 1  # Only English resolves

    def test_empty_languages(self):
        movie = _make_movie(imdb_data={"languages": []})
        assert movie.audio_language_ids() == []


# ---------------------------------------------------------------------------
# AwardNomination.ceremony_id
# ---------------------------------------------------------------------------

class TestAwardNominationCeremonyId:
    def test_ceremony_id_resolves_for_known_ceremony(self):
        """ceremony_id should return the correct AwardCeremony enum ID."""
        award = AwardNomination(
            ceremony="Academy Awards, USA",
            award_name="Oscar",
            category="Best Picture",
            outcome=AwardOutcome.WINNER,
            year=2024,
        )
        assert award.ceremony_id == 1

    def test_ceremony_id_resolves_for_festival(self):
        """Festival ceremonies (nullable category) should resolve correctly."""
        award = AwardNomination(
            ceremony="Cannes Film Festival",
            award_name="Palme d'Or",
            category=None,
            outcome=AwardOutcome.WINNER,
            year=2023,
        )
        assert award.ceremony_id == 4

    def test_ceremony_id_raises_for_unknown_ceremony(self):
        """Unknown ceremony string should raise KeyError."""
        award = AwardNomination(
            ceremony="Unknown Awards",
            award_name="Unknown",
            outcome=AwardOutcome.NOMINEE,
            year=2024,
        )
        with pytest.raises(KeyError):
            _ = award.ceremony_id


# ---------------------------------------------------------------------------
# from_tmdb_ids (batch loader)
# ---------------------------------------------------------------------------

class TestFromTmdbIds:
    def test_empty_list_returns_empty_dict(self):
        """Empty input should return empty dict without touching the DB."""
        result = Movie.from_tmdb_ids([])
        assert result == {}

    def test_missing_db_file_raises(self, tmp_path):
        """Non-existent DB path should raise FileNotFoundError."""
        fake_path = tmp_path / "nonexistent.db"
        with pytest.raises(FileNotFoundError):
            Movie.from_tmdb_ids([1, 2], tracker_db_path=fake_path)


class TestTrackerFranchiseLoading:
    def test_from_tmdb_id_parses_franchise_metadata(self, tmp_path):
        """Single-movie loader should parse generated_metadata.franchise into FranchiseOutput."""
        db_path = tmp_path / "tracker.db"
        _write_tracker_db_with_franchise(
            db_path,
            {
                "lineage_reasoning": "Identified the franchise lineage.",
                "lineage": "spider-man",
                "shared_universe": "marvel cinematic universe",
                "subgroups_reasoning": "This film belongs to a named phase.",
                "recognized_subgroups": ["phase three"],
                "launched_subgroup": False,
                "position_reasoning": "This is a sequel.",
                "lineage_position": "sequel",
                "crossover_reasoning": "No crossover here.",
                "is_crossover": False,
                "spinoff_reasoning": "Not a spinoff.",
                "is_spinoff": False,
                "launch_reasoning": "It did not launch the franchise.",
                "launched_franchise": False,
            },
        )

        movie = Movie.from_tmdb_id(1, tracker_db_path=db_path)

        assert movie.franchise_metadata is not None
        assert movie.franchise_metadata.lineage == "spider-man"
        assert movie.franchise_metadata.shared_universe == "marvel cinematic universe"
        assert movie.franchise_metadata.recognized_subgroups == ["phase three"]
        assert movie.franchise_metadata.lineage_position == LineagePosition.SEQUEL

    def test_from_tmdb_ids_parses_franchise_metadata(self, tmp_path):
        """Batch loader should expose franchise_metadata on returned Movie objects."""
        db_path = tmp_path / "tracker.db"
        _write_tracker_db_with_franchise(
            db_path,
            {
                "lineage_reasoning": "Identified a remake case.",
                "lineage": None,
                "shared_universe": None,
                "subgroups_reasoning": "No recognized subgroups.",
                "recognized_subgroups": [],
                "launched_subgroup": False,
                "position_reasoning": "This is a remake.",
                "lineage_position": "remake",
                "crossover_reasoning": "No crossover here.",
                "is_crossover": False,
                "spinoff_reasoning": "No spinoff relationship.",
                "is_spinoff": False,
                "launch_reasoning": "It did not launch a franchise.",
                "launched_franchise": False,
            },
        )

        result = Movie.from_tmdb_ids([1], tracker_db_path=db_path)

        assert 1 in result
        assert result[1].franchise_metadata is not None
        assert result[1].franchise_metadata.lineage is None
        assert result[1].franchise_metadata.lineage_position == LineagePosition.REMAKE


# ---------------------------------------------------------------------------
# Movie.source_material_type_ids()
# ---------------------------------------------------------------------------

from schemas.metadata import SourceMaterialV2Output
from schemas.enums import SourceMaterialType


class TestSourceMaterialTypeIds:
    def test_returns_ids_from_metadata(self):
        """Returns source material type IDs when metadata is present."""
        movie = _make_movie(
            source_material_v2_metadata=SourceMaterialV2Output(
                source_material_types=[
                    SourceMaterialType.NOVEL_ADAPTATION,
                    SourceMaterialType.TRUE_STORY,
                ],
            ),
        )
        assert movie.source_material_type_ids() == [1, 4]

    def test_returns_empty_when_metadata_none(self):
        """Returns empty list when source_material_v2_metadata is None."""
        movie = _make_movie()
        assert movie.source_material_type_ids() == []


# ---------------------------------------------------------------------------
# Movie.keyword_ids()
# ---------------------------------------------------------------------------

class TestKeywordIds:
    def test_returns_ids_for_known_keywords(self):
        """Known IMDB overall_keywords produce correct integer IDs."""
        movie = _make_movie(imdb_data={"overall_keywords": ["Action", "Drama"]})
        ids = movie.keyword_ids()
        # Should return non-empty list of ints for known keywords
        assert len(ids) > 0
        assert all(isinstance(i, int) for i in ids)

    def test_skips_unknown_keywords(self):
        """Unknown keyword strings are silently skipped."""
        movie = _make_movie(imdb_data={"overall_keywords": ["completely_nonexistent_keyword_xyz"]})
        assert movie.keyword_ids() == []

    def test_deduplicates(self):
        """Duplicate keywords produce unique IDs."""
        movie = _make_movie(imdb_data={"overall_keywords": ["Action", "Action"]})
        ids = movie.keyword_ids()
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# Movie.concept_tag_ids()
# ---------------------------------------------------------------------------

from schemas.metadata import (
    ConceptTagsOutput,
    NarrativeStructureAssessment,
    PlotArchetypeAssessment,
    SettingAssessment,
    CharacterAssessment,
    EndingAssessment,
    ExperientialAssessment,
    ContentFlagAssessment,
)
from schemas.enums import NarrativeStructureTag, EndingTag


class TestConceptTagIds:
    def test_returns_sorted_ids(self):
        """Returns sorted concept tag IDs from metadata."""
        output = ConceptTagsOutput(
            narrative_structure=NarrativeStructureAssessment(
                tags=[NarrativeStructureTag.PLOT_TWIST],
            ),
            plot_archetypes=PlotArchetypeAssessment(tags=[]),
            settings=SettingAssessment(tags=[]),
            characters=CharacterAssessment(tags=[]),
            endings=EndingAssessment(tag=EndingTag.HAPPY_ENDING),
            experiential=ExperientialAssessment(tags=[]),
            content_flags=ContentFlagAssessment(tags=[]),
        )
        movie = _make_movie(concept_tags_metadata=output)
        ids = movie.concept_tag_ids()
        assert ids == sorted(ids)
        assert 1 in ids   # PLOT_TWIST
        assert 41 in ids   # HAPPY_ENDING

    def test_returns_empty_when_metadata_none(self):
        """Returns empty list when concept_tags_metadata is None."""
        movie = _make_movie()
        assert movie.concept_tag_ids() == []


# ---------------------------------------------------------------------------
# AwardNomination.did_win()
# ---------------------------------------------------------------------------

class TestAwardNominationDidWin:
    def test_did_win_true_for_winner(self):
        """did_win() returns True for WINNER outcome."""
        award = AwardNomination(
            ceremony="Academy Awards, USA",
            award_name="Oscar",
            category="Best Picture",
            outcome=AwardOutcome.WINNER,
            year=2020,
        )
        assert award.did_win() is True

    def test_did_win_false_for_nominee(self):
        """did_win() returns False for NOMINEE outcome."""
        award = AwardNomination(
            ceremony="Academy Awards, USA",
            award_name="Oscar",
            category="Best Picture",
            outcome=AwardOutcome.NOMINEE,
            year=2020,
        )
        assert award.did_win() is False
