"""
Unit tests for movie_ingestion.tmdb_fetcher — Stage 2 TMDB detail fetching.

Covers watch-provider key encoding, field extraction, maturity rating and
review extraction, SQLite bulk persistence, single-movie async processing,
batch orchestration, error logging, and run().
"""

import json
import struct
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from movie_ingestion.tmdb_fetching.tmdb_fetcher import (
    _COMMIT_BATCH_SIZE,
    _MovieResult,
    _TMDB_CATEGORY_TO_ACCESS_TYPE,
    _extract_fields,
    _extract_review_contents,
    _extract_us_maturity_rating,
    _extract_watch_provider_keys,
    _fetch_all,
    _log_unexpected_error,
    _pack_provider_keys,
    _persist_movies,
    _process_movie,
    run,
)
from movie_ingestion.tracker import (
    _SCHEMA_SQL,
    MovieStatus,
    PipelineStage,
    batch_log_filter,
)


# ---------------------------------------------------------------------------
# Fixtures / Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def in_memory_db() -> sqlite3.Connection:
    """Return an in-memory SQLite connection with the full tracker schema."""
    db = sqlite3.connect(":memory:")
    db.executescript(_SCHEMA_SQL)
    db.commit()
    return db


def _sample_tmdb_response(**overrides) -> dict:
    """Build a realistic TMDB expanded-detail response with optional overrides."""
    base = {
        "id": 550,
        "imdb_id": "tt0137523",
        "title": "Fight Club",
        "release_date": "1999-10-15",
        "runtime": 139,
        "poster_path": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        "overview": "An insomniac office worker and a devil-may-care soap maker.",
        "genres": [{"id": 18, "name": "Drama"}, {"id": 53, "name": "Thriller"}],
        "revenue": 101209702,
        "budget": 63000000,
        "production_companies": [{"id": 508, "name": "Regency Enterprises"}],
        "production_countries": [{"iso_3166_1": "US", "name": "United States"}],
        "vote_count": 27000,
        "popularity": 62.5,
        "vote_average": 8.4,
        "keywords": {"keywords": [{"id": 825, "name": "support group"}]},
        "credits": {
            "cast": [{"id": 819, "name": "Edward Norton"}],
            "crew": [{"id": 7467, "name": "David Fincher", "job": "Director"}],
        },
        "watch/providers": {
            "results": {
                "US": {
                    "flatrate": [{"provider_id": 8, "provider_name": "Netflix"}],
                    "rent": [{"provider_id": 2, "provider_name": "Apple TV"}],
                    "buy": [{"provider_id": 2, "provider_name": "Apple TV"}],
                }
            }
        },
        "release_dates": {
            "results": [
                {
                    "iso_3166_1": "US",
                    "release_dates": [
                        {"certification": "R", "type": 3},
                    ],
                }
            ]
        },
        "reviews": {
            "results": [
                {"content": "A masterpiece of modern cinema."},
                {"content": "Mind-blowing twist ending."},
            ]
        },
    }
    base.update(overrides)
    return base


def _sample_extracted_fields(**overrides) -> dict:
    """Build a dict matching _extract_fields output for a sample movie."""
    from implementation.classes.enums import StreamingAccessType
    from implementation.misc.helpers import create_watch_provider_offering_key

    # Key encoding: (provider_id << 4) | type_id
    keys = sorted({
        create_watch_provider_offering_key(8, StreamingAccessType.SUBSCRIPTION.type_id),
        create_watch_provider_offering_key(2, StreamingAccessType.RENT.type_id),
        create_watch_provider_offering_key(2, StreamingAccessType.BUY.type_id),
    })

    base = {
        "tmdb_id": 550,
        "imdb_id": "tt0137523",
        "title": "Fight Club",
        "release_date": "1999-10-15",
        "duration": 139,
        "poster_url": "/pB8BM7pdSp6B6Ih7QZ4DrQ3PmJK.jpg",
        "watch_provider_keys": keys,
        "vote_count": 27000,
        "popularity": 62.5,
        "vote_average": 8.4,
        "overview_length": len(
            "An insomniac office worker and a devil-may-care soap maker."
        ),
        "genre_count": 2,
        "has_revenue": 1,
        "has_budget": 1,
        "has_production_companies": 1,
        "has_production_countries": 1,
        "has_keywords": 1,
        "has_cast_and_crew": 1,
        "budget": 63000000,
        "maturity_rating": "R",
        "reviews": json.dumps(
            ["A masterpiece of modern cinema.", "Mind-blowing twist ending."],
            ensure_ascii=False,
        ),
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# _pack_provider_keys
# ---------------------------------------------------------------------------


class TestPackProviderKeys:
    """Tests for the watch-provider key BLOB packing function."""

    def test_empty_list_returns_none(self) -> None:
        """Empty key list returns None (stored as NULL in SQLite)."""
        assert _pack_provider_keys([]) is None

    def test_single_key(self) -> None:
        """Single key packs to exactly 4 bytes (one unsigned 32-bit int)."""
        result = _pack_provider_keys([42])

        assert result == struct.pack("<1I", 42)
        assert len(result) == 4

    def test_multiple_keys_round_trip(self) -> None:
        """Pack then unpack produces the original sorted key list."""
        keys = [100, 200, 300]
        packed = _pack_provider_keys(keys)

        unpacked = list(struct.unpack(f"<{len(keys)}I", packed))

        assert unpacked == keys

    def test_max_uint32_value(self) -> None:
        """Packing handles the maximum 32-bit unsigned integer (2^32 - 1)."""
        max_val = 2**32 - 1
        packed = _pack_provider_keys([max_val])

        (unpacked,) = struct.unpack("<1I", packed)

        assert unpacked == max_val

    def test_zero_key(self) -> None:
        """Packing handles zero as a valid key."""
        result = _pack_provider_keys([0])

        assert result == struct.pack("<1I", 0)


# ---------------------------------------------------------------------------
# _extract_watch_provider_keys
# ---------------------------------------------------------------------------


class TestExtractWatchProviderKeys:
    """Tests for US-region watch provider key extraction."""

    def test_extract_all_three_categories(self) -> None:
        """Extracts keys from flatrate, rent, and buy categories."""
        raw = {
            "results": {
                "US": {
                    "flatrate": [{"provider_id": 8}],
                    "rent": [{"provider_id": 10}],
                    "buy": [{"provider_id": 15}],
                }
            }
        }
        keys = _extract_watch_provider_keys(raw)

        # All three categories produce keys
        assert len(keys) == 3

    def test_extract_empty_input(self) -> None:
        """Empty dict produces empty key list."""
        assert _extract_watch_provider_keys({}) == []

    def test_extract_no_us_region(self) -> None:
        """Non-US regions only produce empty key list."""
        raw = {"results": {"GB": {"flatrate": [{"provider_id": 8}]}}}

        assert _extract_watch_provider_keys(raw) == []

    def test_extract_deduplicates(self) -> None:
        """Same provider appearing twice in one category produces one key."""
        raw = {
            "results": {
                "US": {
                    "flatrate": [
                        {"provider_id": 8},
                        {"provider_id": 8},  # duplicate
                    ]
                }
            }
        }
        keys = _extract_watch_provider_keys(raw)

        assert len(keys) == 1

    def test_extract_returns_sorted(self) -> None:
        """Keys are returned in ascending sorted order."""
        raw = {
            "results": {
                "US": {
                    "flatrate": [{"provider_id": 100}],
                    "buy": [{"provider_id": 1}],
                }
            }
        }
        keys = _extract_watch_provider_keys(raw)

        assert keys == sorted(keys)

    def test_extract_missing_category(self) -> None:
        """Missing categories (rent/buy absent) are silently skipped."""
        raw = {"results": {"US": {"flatrate": [{"provider_id": 8}]}}}
        keys = _extract_watch_provider_keys(raw)

        assert len(keys) == 1

    def test_extract_key_formula(self) -> None:
        """Verify the (provider_id << 4) | type_id encoding for each access type."""
        from implementation.classes.enums import StreamingAccessType
        from implementation.misc.helpers import create_watch_provider_offering_key

        raw = {
            "results": {
                "US": {
                    "flatrate": [{"provider_id": 8}],
                    "rent": [{"provider_id": 8}],
                    "buy": [{"provider_id": 8}],
                }
            }
        }
        keys = _extract_watch_provider_keys(raw)

        expected = sorted([
            create_watch_provider_offering_key(8, StreamingAccessType.SUBSCRIPTION.type_id),
            create_watch_provider_offering_key(8, StreamingAccessType.RENT.type_id),
            create_watch_provider_offering_key(8, StreamingAccessType.BUY.type_id),
        ])
        assert keys == expected

        # Verify the bit-shift formula: (8 << 4) | type_id
        assert (8 << 4) | 1 in keys  # SUBSCRIPTION
        assert (8 << 4) | 2 in keys  # BUY
        assert (8 << 4) | 3 in keys  # RENT


# ---------------------------------------------------------------------------
# _extract_us_maturity_rating
# ---------------------------------------------------------------------------


class TestExtractUsMaturityRating:
    """Tests for US maturity rating extraction from release_dates data."""

    def test_extracts_us_certification(self) -> None:
        """Standard US entry with certification='PG-13' returns 'PG-13'."""
        raw = {
            "release_dates": {
                "results": [
                    {
                        "iso_3166_1": "US",
                        "release_dates": [
                            {"certification": "PG-13", "type": 3},
                        ],
                    }
                ]
            }
        }
        assert _extract_us_maturity_rating(raw) == "PG-13"

    def test_returns_none_when_no_us_region(self) -> None:
        """Only non-US countries present returns None."""
        raw = {
            "release_dates": {
                "results": [
                    {
                        "iso_3166_1": "GB",
                        "release_dates": [
                            {"certification": "15", "type": 3},
                        ],
                    }
                ]
            }
        }
        assert _extract_us_maturity_rating(raw) is None

    def test_returns_none_when_no_release_dates(self) -> None:
        """Empty or missing release_dates key returns None."""
        assert _extract_us_maturity_rating({}) is None
        assert _extract_us_maturity_rating({"release_dates": {"results": []}}) is None

    def test_returns_first_nonempty_certification(self) -> None:
        """US has multiple release dates, first empty, second 'R' — returns 'R'."""
        raw = {
            "release_dates": {
                "results": [
                    {
                        "iso_3166_1": "US",
                        "release_dates": [
                            {"certification": "", "type": 1},
                            {"certification": "R", "type": 3},
                        ],
                    }
                ]
            }
        }
        assert _extract_us_maturity_rating(raw) == "R"

    def test_strips_whitespace(self) -> None:
        """Certification ' PG-13 ' returns 'PG-13'."""
        raw = {
            "release_dates": {
                "results": [
                    {
                        "iso_3166_1": "US",
                        "release_dates": [
                            {"certification": " PG-13 ", "type": 3},
                        ],
                    }
                ]
            }
        }
        assert _extract_us_maturity_rating(raw) == "PG-13"

    def test_returns_none_for_empty_certifications(self) -> None:
        """US entries all have empty string certifications returns None."""
        raw = {
            "release_dates": {
                "results": [
                    {
                        "iso_3166_1": "US",
                        "release_dates": [
                            {"certification": "", "type": 1},
                            {"certification": "", "type": 3},
                        ],
                    }
                ]
            }
        }
        assert _extract_us_maturity_rating(raw) is None

    def test_handles_null_release_dates_key(self) -> None:
        """release_dates is None returns None."""
        raw = {"release_dates": None}
        assert _extract_us_maturity_rating(raw) is None


# ---------------------------------------------------------------------------
# _extract_review_contents
# ---------------------------------------------------------------------------


class TestExtractReviewContents:
    """Tests for review text extraction from TMDB reviews data."""

    def test_extracts_review_contents(self) -> None:
        """Two reviews with content returns JSON list with both strings."""
        raw = {
            "reviews": {
                "results": [
                    {"content": "Great movie!"},
                    {"content": "Loved it."},
                ]
            }
        }
        result = _extract_review_contents(raw)
        parsed = json.loads(result)

        assert parsed == ["Great movie!", "Loved it."]

    def test_returns_none_for_no_reviews(self) -> None:
        """Empty results list returns None."""
        raw = {"reviews": {"results": []}}
        assert _extract_review_contents(raw) is None

    def test_skips_reviews_without_content(self) -> None:
        """Mix of reviews with and without content only includes non-empty ones."""
        raw = {
            "reviews": {
                "results": [
                    {"content": "Good"},
                    {"content": ""},
                    {"author": "reviewer_no_content"},
                    {"content": "Bad"},
                ]
            }
        }
        result = _extract_review_contents(raw)
        parsed = json.loads(result)

        assert parsed == ["Good", "Bad"]

    def test_returns_none_for_missing_reviews_key(self) -> None:
        """No 'reviews' key returns None."""
        assert _extract_review_contents({}) is None

    def test_returns_none_for_null_reviews(self) -> None:
        """reviews=None returns None."""
        raw = {"reviews": None}
        assert _extract_review_contents(raw) is None

    def test_preserves_unicode_in_reviews(self) -> None:
        """Review with non-ASCII chars preserved in JSON output."""
        raw = {
            "reviews": {
                "results": [
                    {"content": "Un chef-d'œuvre cinématographique!"},
                ]
            }
        }
        result = _extract_review_contents(raw)
        parsed = json.loads(result)

        assert parsed == ["Un chef-d'œuvre cinématographique!"]


# ---------------------------------------------------------------------------
# _extract_fields
# ---------------------------------------------------------------------------


class TestExtractFields:
    """Tests for the TMDB JSON → flat dict field extraction."""

    def test_happy_path(self) -> None:
        """Complete TMDB response produces correct 21-key dict."""
        raw = _sample_tmdb_response()
        fields = _extract_fields(raw)

        expected = _sample_extracted_fields()
        assert fields == expected

    def test_raises_on_missing_id(self) -> None:
        """Missing 'id' key raises KeyError (structural API violation)."""
        raw = _sample_tmdb_response()
        del raw["id"]

        with pytest.raises(KeyError):
            _extract_fields(raw)

    def test_missing_optional_fields(self) -> None:
        """Minimal response with only 'id' produces safe defaults."""
        raw = {"id": 1}
        fields = _extract_fields(raw)

        assert fields["tmdb_id"] == 1
        assert fields["imdb_id"] is None
        assert fields["title"] is None
        assert fields["release_date"] is None
        assert fields["duration"] is None
        assert fields["poster_url"] is None
        assert fields["watch_provider_keys"] == []
        assert fields["vote_count"] == 0
        assert fields["popularity"] == 0.0
        assert fields["vote_average"] == 0.0
        assert fields["overview_length"] == 0
        assert fields["genre_count"] == 0
        assert fields["has_revenue"] == 0
        assert fields["has_budget"] == 0
        assert fields["has_production_companies"] == 0
        assert fields["has_production_countries"] == 0
        assert fields["has_keywords"] == 0
        assert fields["has_cast_and_crew"] == 0
        assert fields["budget"] == 0
        assert fields["maturity_rating"] is None
        assert fields["reviews"] is None

    @pytest.mark.parametrize("overview", [None, ""])
    def test_falsy_overview(self, overview) -> None:
        """None and empty string overviews both produce overview_length == 0."""
        raw = _sample_tmdb_response(overview=overview)
        fields = _extract_fields(raw)

        assert fields["overview_length"] == 0

    @pytest.mark.parametrize(
        "revenue, expected",
        [(100_000, 1), (1, 1), (0, 0), (None, 0)],
    )
    def test_revenue_boolean(self, revenue, expected) -> None:
        """has_revenue is 1 when revenue > 0, else 0."""
        raw = _sample_tmdb_response(revenue=revenue)
        fields = _extract_fields(raw)

        assert fields["has_revenue"] == expected

    @pytest.mark.parametrize(
        "budget, expected",
        [(1_000_000, 1), (1, 1), (0, 0), (None, 0)],
    )
    def test_budget_boolean(self, budget, expected) -> None:
        """has_budget is 1 when budget > 0, else 0."""
        raw = _sample_tmdb_response(budget=budget)
        fields = _extract_fields(raw)

        assert fields["has_budget"] == expected

    @pytest.mark.parametrize(
        "cast, crew, expected",
        [
            ([{"id": 1}], [{"id": 2}], 1),  # both present
            ([{"id": 1}], [], 0),             # cast only
            ([], [{"id": 2}], 0),             # crew only
            ([], [], 0),                       # neither
        ],
    )
    def test_cast_and_crew_requires_both(self, cast, crew, expected) -> None:
        """has_cast_and_crew requires both cast AND crew to be non-empty."""
        raw = _sample_tmdb_response(credits={"cast": cast, "crew": crew})
        fields = _extract_fields(raw)

        assert fields["has_cast_and_crew"] == expected

    def test_keywords_nested_access(self) -> None:
        """Keywords accessed via nested 'keywords.keywords' path."""
        raw = _sample_tmdb_response(
            keywords={"keywords": [{"id": 1, "name": "test"}]}
        )
        fields = _extract_fields(raw)

        assert fields["has_keywords"] == 1

    def test_keywords_none_inner(self) -> None:
        """Inner keywords list being None produces has_keywords == 0."""
        raw = _sample_tmdb_response(keywords={"keywords": None})
        fields = _extract_fields(raw)

        assert fields["has_keywords"] == 0

    def test_watch_providers_literal_slash_key(self) -> None:
        """Watch providers accessed via literal 'watch/providers' key."""
        raw = _sample_tmdb_response()
        # Verify the key with a slash is used
        assert "watch/providers" in raw

        fields = _extract_fields(raw)
        assert len(fields["watch_provider_keys"]) > 0

    def test_vote_count_default(self) -> None:
        """Missing vote_count defaults to 0."""
        raw = _sample_tmdb_response()
        del raw["vote_count"]
        fields = _extract_fields(raw)

        assert fields["vote_count"] == 0

    def test_overview_length_calculation(self) -> None:
        """overview_length equals len() of the overview string."""
        raw = _sample_tmdb_response(overview="Hello")
        fields = _extract_fields(raw)

        assert fields["overview_length"] == 5

    def test_production_companies_boolean(self) -> None:
        """has_production_companies is 1 when list is non-empty, else 0."""
        raw = _sample_tmdb_response(production_companies=[])
        assert _extract_fields(raw)["has_production_companies"] == 0

        raw = _sample_tmdb_response(
            production_companies=[{"id": 1, "name": "Test"}]
        )
        assert _extract_fields(raw)["has_production_companies"] == 1

    def test_production_countries_boolean(self) -> None:
        """has_production_countries is 1 when list is non-empty, else 0."""
        raw = _sample_tmdb_response(production_countries=[])
        assert _extract_fields(raw)["has_production_countries"] == 0

        raw = _sample_tmdb_response(
            production_countries=[{"iso_3166_1": "US", "name": "United States"}]
        )
        assert _extract_fields(raw)["has_production_countries"] == 1

    # --- New tests for the 3 added fields ---

    def test_budget_field_extracted(self) -> None:
        """Full response with budget=63000000 produces fields['budget']=63000000."""
        raw = _sample_tmdb_response(budget=63000000)
        fields = _extract_fields(raw)

        assert fields["budget"] == 63000000

    def test_budget_defaults_to_zero(self) -> None:
        """No budget key produces fields['budget']=0."""
        raw = _sample_tmdb_response()
        del raw["budget"]
        fields = _extract_fields(raw)

        assert fields["budget"] == 0

    def test_maturity_rating_extracted(self) -> None:
        """Response with US release_dates cert produces fields['maturity_rating']='PG-13'."""
        raw = _sample_tmdb_response(
            release_dates={
                "results": [
                    {
                        "iso_3166_1": "US",
                        "release_dates": [{"certification": "PG-13", "type": 3}],
                    }
                ]
            }
        )
        fields = _extract_fields(raw)

        assert fields["maturity_rating"] == "PG-13"

    def test_maturity_rating_none_when_missing(self) -> None:
        """No release_dates produces fields['maturity_rating']=None."""
        raw = _sample_tmdb_response()
        del raw["release_dates"]
        fields = _extract_fields(raw)

        assert fields["maturity_rating"] is None

    def test_reviews_extracted(self) -> None:
        """Response with reviews produces fields['reviews'] as JSON string."""
        raw = _sample_tmdb_response(
            reviews={
                "results": [
                    {"content": "Amazing film."},
                ]
            }
        )
        fields = _extract_fields(raw)

        parsed = json.loads(fields["reviews"])
        assert parsed == ["Amazing film."]

    def test_reviews_none_when_missing(self) -> None:
        """No reviews produces fields['reviews']=None."""
        raw = _sample_tmdb_response()
        del raw["reviews"]
        fields = _extract_fields(raw)

        assert fields["reviews"] is None


# ---------------------------------------------------------------------------
# _persist_movies
# ---------------------------------------------------------------------------


class TestPersistMovies:
    """Tests for bulk SQLite persistence of extracted movie fields."""

    def _make_result(self, **overrides) -> _MovieResult:
        """Build a _MovieResult with extracted fields for testing."""
        fields = _sample_extracted_fields(**overrides.pop("field_overrides", {}))
        defaults = {
            "tmdb_id": fields["tmdb_id"],
            "status": "fetched",
            "reason": None,
            "fields": fields,
        }
        defaults.update(overrides)
        return _MovieResult(**defaults)

    def test_bulk_inserts_tmdb_data(self, in_memory_db) -> None:
        """Pass 3 _MovieResults with fields — 3 rows in tmdb_data."""
        results = [
            self._make_result(tmdb_id=100, field_overrides={"tmdb_id": 100}),
            self._make_result(tmdb_id=200, field_overrides={"tmdb_id": 200}),
            self._make_result(tmdb_id=300, field_overrides={"tmdb_id": 300}),
        ]

        _persist_movies(in_memory_db, results)
        in_memory_db.commit()

        count = in_memory_db.execute("SELECT COUNT(*) FROM tmdb_data").fetchone()[0]
        assert count == 3

    def test_bulk_updates_movie_progress(self, in_memory_db) -> None:
        """3 results with status='fetched' — 3 movie_progress rows updated to tmdb_fetched."""
        for tid in (100, 200, 300):
            in_memory_db.execute(
                "INSERT INTO movie_progress (tmdb_id, status) VALUES (?, 'pending')",
                (tid,),
            )
        in_memory_db.commit()

        results = [
            self._make_result(tmdb_id=tid, field_overrides={"tmdb_id": tid})
            for tid in (100, 200, 300)
        ]

        _persist_movies(in_memory_db, results)
        in_memory_db.commit()

        statuses = in_memory_db.execute(
            "SELECT status FROM movie_progress ORDER BY tmdb_id"
        ).fetchall()
        assert all(s[0] == "tmdb_fetched" for s in statuses)

    def test_skips_missing_imdb_id_for_progress_update(self, in_memory_db) -> None:
        """Result with status='missing_imdb_id' — tmdb_data inserted but progress NOT updated."""
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (550, 'pending')"
        )
        in_memory_db.commit()

        result = _MovieResult(
            tmdb_id=550,
            status="missing_imdb_id",
            reason="missing_imdb_id",
            fields=_sample_extracted_fields(imdb_id=None),
        )

        _persist_movies(in_memory_db, [result])
        in_memory_db.commit()

        # tmdb_data row was inserted
        tmdb_row = in_memory_db.execute(
            "SELECT tmdb_id FROM tmdb_data WHERE tmdb_id = 550"
        ).fetchone()
        assert tmdb_row is not None

        # movie_progress was NOT updated to tmdb_fetched (still pending)
        status = in_memory_db.execute(
            "SELECT status FROM movie_progress WHERE tmdb_id = 550"
        ).fetchone()[0]
        assert status == "pending"

    def test_empty_results_is_noop(self, in_memory_db) -> None:
        """Empty list produces no DB writes."""
        _persist_movies(in_memory_db, [])

        count = in_memory_db.execute("SELECT COUNT(*) FROM tmdb_data").fetchone()[0]
        assert count == 0

    def test_does_not_commit(self, in_memory_db) -> None:
        """_persist_movies does not call db.commit()."""
        result = self._make_result()
        _persist_movies(in_memory_db, [result])

        # Row visible in same connection (uncommitted transaction)
        count = in_memory_db.execute(
            "SELECT COUNT(*) FROM tmdb_data WHERE tmdb_id = 550"
        ).fetchone()[0]
        assert count == 1

        # Roll back proves _persist_movies did not commit
        in_memory_db.rollback()
        count_after = in_memory_db.execute(
            "SELECT COUNT(*) FROM tmdb_data WHERE tmdb_id = 550"
        ).fetchone()[0]
        assert count_after == 0

    def test_packs_watch_provider_keys(self, in_memory_db) -> None:
        """Fields with watch_provider_keys=[100,200] stored as packed BLOB."""
        result = self._make_result(
            field_overrides={"watch_provider_keys": [100, 200]}
        )

        _persist_movies(in_memory_db, [result])
        in_memory_db.commit()

        blob = in_memory_db.execute(
            "SELECT watch_provider_keys FROM tmdb_data WHERE tmdb_id = ?",
            (550,),
        ).fetchone()[0]

        unpacked = list(struct.unpack(f"<{len(blob) // 4}I", blob))
        assert unpacked == [100, 200]

    def test_replace_on_duplicate(self, in_memory_db) -> None:
        """INSERT OR REPLACE: second insert with same tmdb_id overwrites."""
        result1 = self._make_result(field_overrides={"title": "Original"})
        _persist_movies(in_memory_db, [result1])
        in_memory_db.commit()

        result2 = self._make_result(field_overrides={"title": "Updated"})
        _persist_movies(in_memory_db, [result2])
        in_memory_db.commit()

        count = in_memory_db.execute(
            "SELECT COUNT(*) FROM tmdb_data WHERE tmdb_id = ?", (550,)
        ).fetchone()[0]
        title = in_memory_db.execute(
            "SELECT title FROM tmdb_data WHERE tmdb_id = ?", (550,)
        ).fetchone()[0]

        assert count == 1
        assert title == "Updated"


# ---------------------------------------------------------------------------
# _process_movie
# ---------------------------------------------------------------------------


class TestProcessMovie:
    """Tests for the single-movie async processing coroutine."""

    async def test_success_returns_fetched_result(self, mocker) -> None:
        """Successful fetch+extract with imdb_id returns _MovieResult(status='fetched')."""
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=_sample_tmdb_response(),
        )

        result = await _process_movie(MagicMock(), MagicMock(), 550)

        assert isinstance(result, _MovieResult)
        assert result.status == "fetched"
        assert result.reason is None

    async def test_success_includes_extracted_fields(self, mocker) -> None:
        """Result.fields contains all expected keys."""
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=_sample_tmdb_response(),
        )

        result = await _process_movie(MagicMock(), MagicMock(), 550)

        assert result.fields is not None
        assert "tmdb_id" in result.fields
        assert "imdb_id" in result.fields
        assert "budget" in result.fields
        assert "maturity_rating" in result.fields
        assert "reviews" in result.fields

    async def test_missing_imdb_id_returns_missing_result(self, mocker) -> None:
        """Extract succeeds but imdb_id=None returns _MovieResult(status='missing_imdb_id')."""
        raw = _sample_tmdb_response(imdb_id=None)
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=raw,
        )

        result = await _process_movie(MagicMock(), MagicMock(), 550)

        assert result.status == "missing_imdb_id"
        assert result.reason == "missing_imdb_id"

    async def test_missing_imdb_id_still_includes_fields(self, mocker) -> None:
        """Fields are present in result even when imdb_id missing."""
        raw = _sample_tmdb_response(imdb_id=None)
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=raw,
        )

        result = await _process_movie(MagicMock(), MagicMock(), 550)

        assert result.fields is not None
        assert result.fields["tmdb_id"] == 550

    async def test_tmdb_fetch_error_returns_error_result(self, mocker) -> None:
        """TMDBFetchError returns _MovieResult(status='error', reason='tmdb_fetch_error')."""
        from db.tmdb import TMDBFetchError

        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            side_effect=TMDBFetchError("test"),
        )

        result = await _process_movie(MagicMock(), MagicMock(), 550)

        assert result.status == "error"
        assert result.reason == "tmdb_fetch_error"
        assert result.fields is None

    async def test_value_error_returns_error_result(self, mocker) -> None:
        """ValueError returns _MovieResult(status='error', reason='tmdb_parse_error')."""
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            side_effect=ValueError("bad json"),
        )

        result = await _process_movie(MagicMock(), MagicMock(), 550)

        assert result.status == "error"
        assert result.reason == "tmdb_parse_error"
        assert result.fields is None

    async def test_404_returns_filtered_result(self, mocker) -> None:
        """None response returns _MovieResult(status='filtered', reason='tmdb_404')."""
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=None,
        )

        result = await _process_movie(MagicMock(), MagicMock(), 550)

        assert result.status == "filtered"
        assert result.reason == "tmdb_404"
        assert result.fields is None

    async def test_extract_error_returns_error_result(self, mocker) -> None:
        """Exception in _extract_fields returns _MovieResult(status='error', reason='tmdb_extract_error')."""
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value={"id": 1},
        )
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._extract_fields",
            side_effect=KeyError("unexpected"),
        )

        result = await _process_movie(MagicMock(), MagicMock(), 550)

        assert result.status == "error"
        assert result.reason == "tmdb_extract_error"
        assert result.fields is None

    async def test_empty_string_imdb_id_treated_as_missing(self, mocker) -> None:
        """Empty string imdb_id (falsy) treated as missing."""
        raw = _sample_tmdb_response(imdb_id="")
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=raw,
        )

        result = await _process_movie(MagicMock(), MagicMock(), 550)

        assert result.status == "missing_imdb_id"
        assert result.reason == "missing_imdb_id"


# ---------------------------------------------------------------------------
# _fetch_all
# ---------------------------------------------------------------------------


class TestFetchAll:
    """Tests for the batched async orchestration loop."""

    async def test_processes_in_batches(self, mocker) -> None:
        """1200 IDs produce 3 batches (500 + 500 + 200) with 3 commits."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
            return_value=_MovieResult(1, "fetched", None, _sample_extracted_fields()),
        )
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._persist_movies",
        )
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher.batch_log_filter",
        )
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.access_token", return_value="tok")

        ids = list(range(1200))
        counters = await _fetch_all(mock_db, ids)

        assert mock_db.commit.call_count == 3

    async def test_commits_after_each_batch(self, mocker) -> None:
        """501 IDs produce exactly 2 batches (500 + 1) with 2 commits."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
            return_value=_MovieResult(1, "fetched", None, _sample_extracted_fields()),
        )
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher._persist_movies")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.batch_log_filter")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.access_token", return_value="tok")

        counters = await _fetch_all(mock_db, list(range(501)))

        assert mock_db.commit.call_count == 2

    async def test_captures_gather_exceptions(self, mocker) -> None:
        """Unexpected exceptions from gather are logged and counted as errors."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        )
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher._persist_movies")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.batch_log_filter")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.access_token", return_value="tok")
        log_mock = mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._log_unexpected_error"
        )

        counters = await _fetch_all(mock_db, [1, 2, 3])

        assert counters["errors"] == 3
        assert log_mock.call_count == 3

    async def test_returns_counters_dict(self, mocker) -> None:
        """_fetch_all returns dict with fetched, filtered, and errors keys."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
            return_value=_MovieResult(1, "fetched", None, _sample_extracted_fields()),
        )
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher._persist_movies")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.batch_log_filter")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.access_token", return_value="tok")

        counters = await _fetch_all(mock_db, [1])

        assert "fetched" in counters
        assert "filtered" in counters
        assert "errors" in counters

    async def test_empty_pending_ids(self, mocker) -> None:
        """Empty ID list produces zero iterations and all-zero counters."""
        mock_db = MagicMock()
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.access_token", return_value="tok")

        counters = await _fetch_all(mock_db, [])

        assert counters == {"fetched": 0, "filtered": 0, "errors": 0}
        mock_db.commit.assert_not_called()

    async def test_persist_movies_called_per_batch(self, mocker) -> None:
        """_persist_movies called once per batch with collected _MovieResults."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
            return_value=_MovieResult(1, "fetched", None, _sample_extracted_fields()),
        )
        persist_mock = mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher._persist_movies")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.batch_log_filter")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.access_token", return_value="tok")

        await _fetch_all(mock_db, list(range(501)))

        # 2 batches → 2 calls
        assert persist_mock.call_count == 2

    async def test_batch_log_filter_called_for_filtered(self, mocker) -> None:
        """Filtered results cause batch_log_filter to be called with correct entries."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
            return_value=_MovieResult(1, "filtered", "tmdb_404", None),
        )
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher._persist_movies")
        blf_mock = mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.batch_log_filter")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.access_token", return_value="tok")

        await _fetch_all(mock_db, [1])

        blf_mock.assert_called_once()
        entries = blf_mock.call_args[0][1]
        assert len(entries) == 1
        assert entries[0][2] == "tmdb_404"  # reason

    async def test_error_results_added_to_filter_entries(self, mocker) -> None:
        """_MovieResult(status='error') included in batch_log_filter entries."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
            return_value=_MovieResult(1, "error", "tmdb_extract_error", None),
        )
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher._persist_movies")
        blf_mock = mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.batch_log_filter")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.access_token", return_value="tok")

        counters = await _fetch_all(mock_db, [1])

        blf_mock.assert_called_once()
        entries = blf_mock.call_args[0][1]
        assert len(entries) == 1
        assert entries[0][2] == "tmdb_extract_error"
        assert counters["errors"] == 1

    async def test_missing_imdb_id_counted_as_filtered(self, mocker) -> None:
        """_MovieResult(status='missing_imdb_id') increments counters['filtered']."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
            return_value=_MovieResult(1, "missing_imdb_id", "missing_imdb_id", _sample_extracted_fields()),
        )
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher._persist_movies")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.batch_log_filter")
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.access_token", return_value="tok")

        counters = await _fetch_all(mock_db, [1])

        assert counters["filtered"] == 1


# ---------------------------------------------------------------------------
# _log_unexpected_error
# ---------------------------------------------------------------------------


class TestLogUnexpectedError:
    """Tests for the append-only debug error log."""

    def test_writes_to_file(self, tmp_path, mocker) -> None:
        """Error entry written to file with tmdb_id and exception info."""
        log_file = tmp_path / "errors.log"
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._ERROR_LOG_PATH", log_file
        )

        try:
            raise ValueError("test error")
        except ValueError as exc:
            _log_unexpected_error(42, exc)

        content = log_file.read_text()
        assert "tmdb_id=42" in content
        assert "ValueError" in content
        assert "test error" in content

    def test_appends_not_overwrites(self, tmp_path, mocker) -> None:
        """Calling twice produces 2 separate entries in the file."""
        log_file = tmp_path / "errors.log"
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._ERROR_LOG_PATH", log_file
        )

        try:
            raise RuntimeError("first")
        except RuntimeError as exc:
            _log_unexpected_error(1, exc)

        try:
            raise RuntimeError("second")
        except RuntimeError as exc:
            _log_unexpected_error(2, exc)

        content = log_file.read_text()
        assert "tmdb_id=1" in content
        assert "tmdb_id=2" in content
        assert "first" in content
        assert "second" in content

    def test_includes_traceback(self, tmp_path, mocker) -> None:
        """Error entry includes a Python traceback."""
        log_file = tmp_path / "errors.log"
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._ERROR_LOG_PATH", log_file
        )

        try:
            raise TypeError("bad type")
        except TypeError as exc:
            _log_unexpected_error(99, exc)

        content = log_file.read_text()
        assert "Traceback" in content
        assert "TypeError" in content

    def test_timestamp_format(self, tmp_path, mocker) -> None:
        """Error entry includes a UTC timestamp in expected format."""
        log_file = tmp_path / "errors.log"
        mocker.patch(
            "movie_ingestion.tmdb_fetching.tmdb_fetcher._ERROR_LOG_PATH", log_file
        )

        try:
            raise Exception("ts test")
        except Exception as exc:
            _log_unexpected_error(7, exc)

        content = log_file.read_text()
        # Timestamp format: [YYYY-MM-DD HH:MM:SS UTC]
        assert "UTC]" in content
        # Verify it looks like a date
        assert "[20" in content  # Year starts with 20xx


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


class TestRun:
    """Tests for the Stage 2 entry point."""

    def test_prints_status_counts(self, mocker, capsys) -> None:
        """run() queries movie_progress and prints status information."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.init_db", return_value=mock_db)

        run()

        output = capsys.readouterr().out
        # When no pending movies exist, prints an informational message
        assert "No" in output or "no" in output or "0" in output

    def test_closes_db_connection(self, mocker) -> None:
        """db.close() called in finally block."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        mocker.patch("movie_ingestion.tmdb_fetching.tmdb_fetcher.init_db", return_value=mock_db)

        run()

        mock_db.close.assert_called_once()
