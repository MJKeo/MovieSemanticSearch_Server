"""
Unit tests for movie_ingestion.tmdb_fetcher — Stage 2 TMDB detail fetching.

Covers watch-provider key encoding, field extraction, SQLite persistence,
single-movie async processing, batch orchestration, error logging, and run().
"""

import struct
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from movie_ingestion.tmdb_fetcher import (
    _COMMIT_BATCH_SIZE,
    _TMDB_CATEGORY_TO_ACCESS_TYPE,
    _extract_fields,
    _extract_watch_provider_keys,
    _fetch_all,
    _log_unexpected_error,
    _pack_provider_keys,
    _persist_movie,
    _process_movie,
    run,
)
from movie_ingestion.tracker import _SCHEMA_SQL, PipelineStage


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
# _extract_fields
# ---------------------------------------------------------------------------


class TestExtractFields:
    """Tests for the TMDB JSON → flat dict field extraction."""

    def test_happy_path(self) -> None:
        """Complete TMDB response produces correct 18-key dict."""
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


# ---------------------------------------------------------------------------
# _persist_movie
# ---------------------------------------------------------------------------


class TestPersistMovie:
    """Tests for SQLite persistence of extracted movie fields."""

    def test_inserts_tmdb_data(self, in_memory_db) -> None:
        """_persist_movie inserts a complete row into tmdb_data."""
        fields = _sample_extracted_fields()
        _persist_movie(in_memory_db, fields)

        row = in_memory_db.execute(
            "SELECT tmdb_id, imdb_id, title, release_date, duration, "
            "vote_count, popularity, vote_average, overview_length, "
            "genre_count, has_revenue, has_budget, has_production_companies, "
            "has_production_countries, has_keywords, has_cast_and_crew "
            "FROM tmdb_data WHERE tmdb_id = ?",
            (550,),
        ).fetchone()

        assert row is not None
        assert row[0] == 550           # tmdb_id
        assert row[1] == "tt0137523"   # imdb_id
        assert row[2] == "Fight Club"  # title
        assert row[3] == "1999-10-15"  # release_date
        assert row[4] == 139           # duration
        assert row[5] == 27000         # vote_count
        assert row[10] == 1            # has_revenue
        assert row[11] == 1            # has_budget

    def test_updates_movie_progress(self, in_memory_db) -> None:
        """_persist_movie updates movie_progress status to 'tmdb_fetched'."""
        # Insert a pending row first
        in_memory_db.execute(
            "INSERT INTO movie_progress (tmdb_id, status) VALUES (?, 'pending')",
            (550,),
        )
        in_memory_db.commit()

        fields = _sample_extracted_fields()
        _persist_movie(in_memory_db, fields)

        row = in_memory_db.execute(
            "SELECT status, imdb_id FROM movie_progress WHERE tmdb_id = ?",
            (550,),
        ).fetchone()

        assert row[0] == "tmdb_fetched"
        assert row[1] == "tt0137523"

    def test_replace_on_duplicate(self, in_memory_db) -> None:
        """INSERT OR REPLACE overwrites existing tmdb_data row."""
        fields = _sample_extracted_fields()
        _persist_movie(in_memory_db, fields)

        # Insert again with different title
        fields["title"] = "Fight Club (Updated)"
        _persist_movie(in_memory_db, fields)

        count = in_memory_db.execute(
            "SELECT COUNT(*) FROM tmdb_data WHERE tmdb_id = ?", (550,)
        ).fetchone()[0]
        title = in_memory_db.execute(
            "SELECT title FROM tmdb_data WHERE tmdb_id = ?", (550,)
        ).fetchone()[0]

        assert count == 1
        assert title == "Fight Club (Updated)"

    def test_packs_watch_provider_keys_as_blob(self, in_memory_db) -> None:
        """Watch provider keys are stored as a packed binary BLOB."""
        fields = _sample_extracted_fields(watch_provider_keys=[100, 200, 300])
        _persist_movie(in_memory_db, fields)

        blob = in_memory_db.execute(
            "SELECT watch_provider_keys FROM tmdb_data WHERE tmdb_id = ?",
            (550,),
        ).fetchone()[0]

        unpacked = list(struct.unpack(f"<{len(blob) // 4}I", blob))
        assert unpacked == [100, 200, 300]

    def test_empty_watch_keys_stores_null(self, in_memory_db) -> None:
        """Empty watch_provider_keys list stores NULL in the BLOB column."""
        fields = _sample_extracted_fields(watch_provider_keys=[])
        _persist_movie(in_memory_db, fields)

        blob = in_memory_db.execute(
            "SELECT watch_provider_keys FROM tmdb_data WHERE tmdb_id = ?",
            (550,),
        ).fetchone()[0]

        assert blob is None

    def test_tolerates_missing_progress_row(self, in_memory_db) -> None:
        """UPDATE matches zero rows when no movie_progress entry exists — no error."""
        fields = _sample_extracted_fields()
        # No INSERT into movie_progress first
        _persist_movie(in_memory_db, fields)

        # tmdb_data row still inserted
        row = in_memory_db.execute(
            "SELECT tmdb_id FROM tmdb_data WHERE tmdb_id = ?", (550,)
        ).fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# _process_movie
# ---------------------------------------------------------------------------


class TestProcessMovie:
    """Tests for the single-movie async processing coroutine."""

    def _make_counters(self) -> dict:
        """Return a fresh counters dict."""
        return {"fetched": 0, "filtered": 0, "errors": 0}

    @pytest.mark.asyncio
    async def test_happy_path_increments_fetched(self, mocker, in_memory_db) -> None:
        """Successful fetch/extract/persist with imdb_id increments 'fetched'."""
        fields = _sample_extracted_fields()
        mocker.patch(
            "movie_ingestion.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=_sample_tmdb_response(),
        )
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._extract_fields", return_value=fields
        )
        mocker.patch("movie_ingestion.tmdb_fetcher._persist_movie")
        mocker.patch("movie_ingestion.tmdb_fetcher.log_filter")

        counters = self._make_counters()
        await _process_movie(MagicMock(), MagicMock(), 550, in_memory_db, counters)

        assert counters["fetched"] == 1
        assert counters["errors"] == 0
        assert counters["filtered"] == 0

    @pytest.mark.asyncio
    async def test_tmdb_fetch_error_logs_and_counts(self, mocker, in_memory_db) -> None:
        """TMDBFetchError logs 'tmdb_fetch_error' and increments errors."""
        from db.tmdb import TMDBFetchError

        mocker.patch(
            "movie_ingestion.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            side_effect=TMDBFetchError("test"),
        )
        log_mock = mocker.patch("movie_ingestion.tmdb_fetcher.log_filter")

        counters = self._make_counters()
        await _process_movie(MagicMock(), MagicMock(), 550, in_memory_db, counters)

        assert counters["errors"] == 1
        log_mock.assert_called_once()
        assert log_mock.call_args.kwargs["reason"] == "tmdb_fetch_error"

    @pytest.mark.asyncio
    async def test_value_error_logs_parse_error(self, mocker, in_memory_db) -> None:
        """ValueError logs 'tmdb_parse_error' and increments errors."""
        mocker.patch(
            "movie_ingestion.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            side_effect=ValueError("bad json"),
        )
        log_mock = mocker.patch("movie_ingestion.tmdb_fetcher.log_filter")

        counters = self._make_counters()
        await _process_movie(MagicMock(), MagicMock(), 550, in_memory_db, counters)

        assert counters["errors"] == 1
        # log_filter(db, tmdb_id, _STAGE, reason="tmdb_parse_error")
        assert log_mock.call_args.kwargs["reason"] == "tmdb_parse_error"

    @pytest.mark.asyncio
    async def test_404_none_response_logs_filtered(self, mocker, in_memory_db) -> None:
        """None response (404) logs 'tmdb_404' and increments filtered."""
        mocker.patch(
            "movie_ingestion.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=None,
        )
        log_mock = mocker.patch("movie_ingestion.tmdb_fetcher.log_filter")

        counters = self._make_counters()
        await _process_movie(MagicMock(), MagicMock(), 550, in_memory_db, counters)

        assert counters["filtered"] == 1
        assert log_mock.call_args.kwargs["reason"] == "tmdb_404"

    @pytest.mark.asyncio
    async def test_extract_error_logs_and_counts(self, mocker, in_memory_db) -> None:
        """Exception in _extract_fields logs 'tmdb_extract_error'."""
        mocker.patch(
            "movie_ingestion.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value={"id": 1},
        )
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._extract_fields",
            side_effect=KeyError("unexpected"),
        )
        log_mock = mocker.patch("movie_ingestion.tmdb_fetcher.log_filter")

        counters = self._make_counters()
        await _process_movie(MagicMock(), MagicMock(), 550, in_memory_db, counters)

        assert counters["errors"] == 1
        assert log_mock.call_args.kwargs["reason"] == "tmdb_extract_error"

    @pytest.mark.asyncio
    async def test_persist_exception_propagates(self, mocker, in_memory_db) -> None:
        """Exception in _persist_movie is NOT caught — propagates to gather."""
        mocker.patch(
            "movie_ingestion.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=_sample_tmdb_response(),
        )
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._extract_fields",
            return_value=_sample_extracted_fields(),
        )
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._persist_movie",
            side_effect=sqlite3.OperationalError("disk full"),
        )

        counters = self._make_counters()

        with pytest.raises(sqlite3.OperationalError, match="disk full"):
            await _process_movie(
                MagicMock(), MagicMock(), 550, in_memory_db, counters
            )

    @pytest.mark.asyncio
    async def test_missing_imdb_id_logs_filtered(self, mocker, in_memory_db) -> None:
        """None imdb_id logs 'missing_imdb_id' and increments filtered."""
        fields = _sample_extracted_fields(imdb_id=None)
        mocker.patch(
            "movie_ingestion.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=_sample_tmdb_response(),
        )
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._extract_fields", return_value=fields
        )
        persist_mock = mocker.patch("movie_ingestion.tmdb_fetcher._persist_movie")
        log_mock = mocker.patch("movie_ingestion.tmdb_fetcher.log_filter")

        counters = self._make_counters()
        await _process_movie(MagicMock(), MagicMock(), 550, in_memory_db, counters)

        assert counters["filtered"] == 1
        assert log_mock.call_args.kwargs["reason"] == "missing_imdb_id"
        # _persist_movie still called (before the IMDB check)
        persist_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_string_imdb_id_treated_as_missing(
        self, mocker, in_memory_db
    ) -> None:
        """Empty string imdb_id is falsy — treated as missing."""
        fields = _sample_extracted_fields(imdb_id="")
        mocker.patch(
            "movie_ingestion.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=_sample_tmdb_response(),
        )
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._extract_fields", return_value=fields
        )
        mocker.patch("movie_ingestion.tmdb_fetcher._persist_movie")
        log_mock = mocker.patch("movie_ingestion.tmdb_fetcher.log_filter")

        counters = self._make_counters()
        await _process_movie(MagicMock(), MagicMock(), 550, in_memory_db, counters)

        assert counters["filtered"] == 1
        assert log_mock.call_args.kwargs["reason"] == "missing_imdb_id"

    @pytest.mark.asyncio
    async def test_persist_called_before_imdb_check(self, mocker, in_memory_db) -> None:
        """_persist_movie is called even when imdb_id is missing."""
        fields = _sample_extracted_fields(imdb_id=None)
        mocker.patch(
            "movie_ingestion.tmdb_fetcher.fetch_movie_details",
            new_callable=AsyncMock,
            return_value=_sample_tmdb_response(),
        )
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._extract_fields", return_value=fields
        )
        persist_mock = mocker.patch("movie_ingestion.tmdb_fetcher._persist_movie")
        mocker.patch("movie_ingestion.tmdb_fetcher.log_filter")

        counters = self._make_counters()
        await _process_movie(MagicMock(), MagicMock(), 550, in_memory_db, counters)

        # _persist_movie was called BEFORE the imdb_id check
        persist_mock.assert_called_once_with(in_memory_db, fields)


# ---------------------------------------------------------------------------
# _fetch_all
# ---------------------------------------------------------------------------


class TestFetchAll:
    """Tests for the batched async orchestration loop."""

    @pytest.mark.asyncio
    async def test_processes_in_batches(self, mocker) -> None:
        """1200 IDs produce 3 batches (500 + 500 + 200) with 3 commits."""
        mock_db = MagicMock()
        process_mock = mocker.patch(
            "movie_ingestion.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
        )
        mocker.patch("movie_ingestion.tmdb_fetcher.access_token", return_value="tok")

        ids = list(range(1200))
        counters = await _fetch_all(mock_db, ids)

        assert process_mock.await_count == 1200
        assert mock_db.commit.call_count == 3

    @pytest.mark.asyncio
    async def test_commits_after_each_batch(self, mocker) -> None:
        """501 IDs produce exactly 2 batches (500 + 1) with 2 commits."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
        )
        mocker.patch("movie_ingestion.tmdb_fetcher.access_token", return_value="tok")

        counters = await _fetch_all(mock_db, list(range(501)))

        assert mock_db.commit.call_count == 2

    @pytest.mark.asyncio
    async def test_captures_gather_exceptions(self, mocker) -> None:
        """Unexpected exceptions from gather are logged and counted as errors."""
        mock_db = MagicMock()
        # Make every _process_movie call raise an exception
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        )
        mocker.patch("movie_ingestion.tmdb_fetcher.access_token", return_value="tok")
        log_mock = mocker.patch(
            "movie_ingestion.tmdb_fetcher._log_unexpected_error"
        )

        counters = await _fetch_all(mock_db, [1, 2, 3])

        assert counters["errors"] == 3
        assert log_mock.call_count == 3

    @pytest.mark.asyncio
    async def test_returns_counters_dict(self, mocker) -> None:
        """_fetch_all returns dict with fetched, filtered, and errors keys."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
        )
        mocker.patch("movie_ingestion.tmdb_fetcher.access_token", return_value="tok")

        counters = await _fetch_all(mock_db, [1])

        assert "fetched" in counters
        assert "filtered" in counters
        assert "errors" in counters

    @pytest.mark.asyncio
    async def test_empty_pending_ids(self, mocker) -> None:
        """Empty ID list produces zero iterations and all-zero counters."""
        mock_db = MagicMock()
        mocker.patch("movie_ingestion.tmdb_fetcher.access_token", return_value="tok")

        counters = await _fetch_all(mock_db, [])

        assert counters == {"fetched": 0, "filtered": 0, "errors": 0}
        mock_db.commit.assert_not_called()

    @pytest.mark.asyncio
    async def test_progress_reporting_at_boundary(self, mocker, capsys) -> None:
        """Progress is printed at _PROGRESS_INTERVAL boundaries."""
        mock_db = MagicMock()
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._process_movie",
            new_callable=AsyncMock,
        )
        mocker.patch("movie_ingestion.tmdb_fetcher.access_token", return_value="tok")

        # 10_000 is the progress interval — create exactly that many IDs
        # to trigger the boundary check.  Use a large batch size to keep
        # the test fast (one batch of 10K).
        mocker.patch("movie_ingestion.tmdb_fetcher._COMMIT_BATCH_SIZE", 10_000)

        counters = await _fetch_all(mock_db, list(range(10_000)))

        output = capsys.readouterr().out
        assert "Progress" in output


# ---------------------------------------------------------------------------
# _log_unexpected_error
# ---------------------------------------------------------------------------


class TestLogUnexpectedError:
    """Tests for the append-only debug error log."""

    def test_writes_to_file(self, tmp_path, mocker) -> None:
        """Error entry written to file with tmdb_id and exception info."""
        log_file = tmp_path / "errors.log"
        mocker.patch(
            "movie_ingestion.tmdb_fetcher._ERROR_LOG_PATH", log_file
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
            "movie_ingestion.tmdb_fetcher._ERROR_LOG_PATH", log_file
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
            "movie_ingestion.tmdb_fetcher._ERROR_LOG_PATH", log_file
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
            "movie_ingestion.tmdb_fetcher._ERROR_LOG_PATH", log_file
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

    def test_no_pending_movies_exits_early(self, mocker, capsys) -> None:
        """run() prints early-exit message when no pending movies exist."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = []
        mocker.patch("movie_ingestion.tmdb_fetcher.init_db", return_value=mock_db)

        run()

        output = capsys.readouterr().out
        assert "No pending movies" in output

    def test_processes_pending_movies(self, mocker) -> None:
        """run() calls _fetch_all with correct pending IDs."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = [
            (100,), (200,), (300,),
        ]
        mocker.patch("movie_ingestion.tmdb_fetcher.init_db", return_value=mock_db)
        # Use a plain MagicMock (not AsyncMock) so calling _fetch_all()
        # returns a non-coroutine sentinel that won't trigger "never awaited"
        mocker.patch("movie_ingestion.tmdb_fetcher._fetch_all", new=MagicMock())
        fetch_mock = mocker.patch(
            "movie_ingestion.tmdb_fetcher.asyncio.run",
            return_value={"fetched": 3, "filtered": 0, "errors": 0},
        )

        run()

        # asyncio.run is called with _fetch_all(db, [100, 200, 300])
        fetch_mock.assert_called_once()

    def test_prints_summary_statistics(self, mocker, capsys) -> None:
        """run() prints summary with counters after completion."""
        mock_db = MagicMock()
        mock_db.execute.return_value.fetchall.return_value = [(1,)]
        mocker.patch("movie_ingestion.tmdb_fetcher.init_db", return_value=mock_db)
        # Use a plain MagicMock (not AsyncMock) so calling _fetch_all()
        # returns a non-coroutine sentinel that won't trigger "never awaited"
        mocker.patch("movie_ingestion.tmdb_fetcher._fetch_all", new=MagicMock())
        mocker.patch(
            "movie_ingestion.tmdb_fetcher.asyncio.run",
            return_value={"fetched": 1, "filtered": 0, "errors": 0},
        )

        run()

        output = capsys.readouterr().out
        assert "Stage 2 Complete" in output
        assert "Fetched:" in output
