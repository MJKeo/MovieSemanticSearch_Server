"""Unit tests for the create_metadata_scores async entry point in db.metadata_scoring.

Tests are designed against the guide's intended behavior (guides/metadata_scoring_guide.md).
Bug-catching tests (Category 3) are expected to FAIL on the current implementation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import math
from types import ModuleType
from typing import Optional
from unittest.mock import AsyncMock, patch
import sys

import pytest

# ── Stub heavy dependencies before importing metadata_scoring ──────────

_temporary_stubs: list[str] = []


async def _stub_fetch_movie_cards(_: list[int]) -> list[dict]:
    return []


async def _stub_read_trending_scores() -> dict[int, float]:
    return {}


@dataclass(slots=True)
class _StubSearchCandidate:
    movie_id: int
    vector_score: float = 0.0
    lexical_score: float = 0.0
    metadata_score: float = 0.0


for _module_name, _attrs in (
    ("db.postgres", {"fetch_movie_cards": _stub_fetch_movie_cards}),
    ("db.redis", {"read_trending_scores": _stub_read_trending_scores}),
    ("db.search", {"SearchCandidate": _StubSearchCandidate}),
):
    try:
        importlib.import_module(_module_name)
    except ModuleNotFoundError:
        _stub_module = ModuleType(_module_name)
        for _attr_name, _attr_value in _attrs.items():
            setattr(_stub_module, _attr_name, _attr_value)
        sys.modules[_module_name] = _stub_module
        _temporary_stubs.append(_module_name)

from db.metadata_scoring import create_metadata_scores, ScoredPreference
from db.search import SearchCandidate
from implementation.classes.enums import (
    DateMatchOperation,
    Genre,
    MaturityRating,
    NumericalMatchOperation,
    RatingMatchOperation,
    ReceptionType,
    StreamingAccessType,
)
from implementation.classes.schemas import (
    DatePreference,
    DatePreferenceResult,
    GenreListPreference,
    GenreListPreferenceResult,
    LanguageListPreference,
    LanguageListPreferenceResult,
    MaturityPreference,
    MaturityPreferenceResult,
    MetadataPreferencesResponse,
    NumericalPreference,
    NumericalPreferenceResult,
    PopularTrendingPreference,
    ReceptionPreference,
    WatchProvidersPreference,
    WatchProvidersPreferenceResult,
)
from implementation.classes.languages import Language
from implementation.classes.watch_providers import StreamingService

for _module_name in _temporary_stubs:
    sys.modules.pop(_module_name, None)


# ── Helpers ──────────────────────────────────────────────────────────────

def _build_no_prefs(**overrides) -> MetadataPreferencesResponse:
    """Build a MetadataPreferencesResponse with all preferences inactive.

    Pass keyword overrides to replace specific preference fields.
    """
    defaults = dict(
        release_date_preference=DatePreference(result=None),
        duration_preference=NumericalPreference(result=None),
        genres_preference=GenreListPreference(result=None),
        audio_languages_preference=LanguageListPreference(result=None),
        watch_providers_preference=WatchProvidersPreference(result=None),
        maturity_rating_preference=MaturityPreference(result=None),
        popular_trending_preference=PopularTrendingPreference(
            prefers_trending_movies=False, prefers_popular_movies=False,
        ),
        reception_preference=ReceptionPreference(
            reception_type=ReceptionType.NO_PREFERENCE,
        ),
    )
    defaults.update(overrides)
    return MetadataPreferencesResponse(**defaults)


def _build_card(
    movie_id: int,
    *,
    genre_ids: list[int] | None = None,
    release_ts: int | None = None,
    runtime_minutes: int | None = None,
    maturity_rank: int | None = None,
    reception_score: float | None = None,
    popularity_score: float | None = None,
    audio_language_ids: list[int] | None = None,
    watch_offer_keys: list[int] | None = None,
) -> dict:
    return {
        "movie_id": movie_id,
        "genre_ids": genre_ids,
        "release_ts": release_ts,
        "runtime_minutes": runtime_minutes,
        "maturity_rank": maturity_rank,
        "reception_score": reception_score,
        "popularity_score": popularity_score,
        "audio_language_ids": audio_language_ids,
        "watch_offer_keys": watch_offer_keys,
    }


def _make_candidate(movie_id: int, *, vector_score: float = 0.5, lexical_score: float = 0.3) -> SearchCandidate:
    return SearchCandidate(movie_id=movie_id, vector_score=vector_score, lexical_score=lexical_score)


def _run(coro):
    """Convenience wrapper for asyncio.run."""
    return asyncio.run(coro)


# ── Patch targets ────────────────────────────────────────────────────────

_PATCH_FETCH = "db.metadata_scoring.fetch_movie_cards"
_PATCH_TRENDING = "db.metadata_scoring.read_trending_scores"


# ══════════════════════════════════════════════════════════════════════════
# Category 1: Early Returns / Edge Cases
# ══════════════════════════════════════════════════════════════════════════

class TestEarlyReturns:
    """Tests for early-return code paths and edge cases."""

    def test_empty_candidates_returns_empty_list(self):
        """Empty candidates list → returns [], no I/O calls."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value],
                    should_exclude=[],
                ),
            ),
        )
        mock_fetch = AsyncMock(return_value=[])
        mock_trending = AsyncMock(return_value={})

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, mock_trending):
            result = _run(create_metadata_scores(prefs, []))

        assert result == []
        mock_fetch.assert_not_called()
        mock_trending.assert_not_called()

    def test_no_active_preferences_returns_unchanged(self):
        """No active preferences → candidates returned unchanged, no I/O."""
        prefs = _build_no_prefs()
        candidates = [_make_candidate(1), _make_candidate(2)]
        mock_fetch = AsyncMock(return_value=[])
        mock_trending = AsyncMock(return_value={})

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, mock_trending):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result is candidates
        assert all(c.metadata_score == 0.0 for c in result)
        mock_fetch.assert_not_called()
        mock_trending.assert_not_called()

    def test_candidate_missing_from_movie_cards_stays_zero(self):
        """Candidate not in movie_cards → metadata_score stays 0.0."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value],
                    should_exclude=[],
                ),
            ),
        )
        candidates = [_make_candidate(999)]
        # Return cards that don't include movie_id=999
        mock_fetch = AsyncMock(return_value=[_build_card(1, genre_ids=[1])])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == 0.0


# ══════════════════════════════════════════════════════════════════════════
# Category 2: Single Preference Activation
# ══════════════════════════════════════════════════════════════════════════

class TestSinglePreference:
    """Each test activates exactly one preference.

    With a single active preference, metadata_score = scorer_output
    (weight cancels: w*score / w = score).
    """

    def test_genres_perfect_match(self):
        """All included genres match → score = 1.0."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value, Genre.COMEDY.value],
                    should_exclude=[],
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1, genre_ids=[Genre.ACTION.genre_id, Genre.COMEDY.genre_id, Genre.DRAMA.genre_id])
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(1.0)

    def test_genres_partial_match(self):
        """1 of 2 included genres match → score = 0.5."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value, Genre.COMEDY.value],
                    should_exclude=[],
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1, genre_ids=[Genre.ACTION.genre_id])
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(0.5)

    def test_genres_exclusion_penalty(self):
        """Excluded genre present → score = -2.0."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[],
                    should_exclude=[Genre.HORROR.value],
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1, genre_ids=[Genre.HORROR.genre_id])
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(-2.0)

    def test_release_date_in_range(self):
        """Release date within BETWEEN range → score = 1.0."""
        prefs = _build_no_prefs(
            release_date_preference=DatePreference(
                result=DatePreferenceResult(
                    first_date="2020-01-01",
                    match_operation=DateMatchOperation.BETWEEN,
                    second_date="2023-01-01",
                ),
            ),
        )
        # Mid-2021 timestamp
        mid_2021 = int(datetime(2021, 6, 15, tzinfo=timezone.utc).timestamp())
        candidates = [_make_candidate(1)]
        card = _build_card(1, release_ts=mid_2021)
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(1.0)

    def test_duration_outside_range(self):
        """Duration slightly outside range → partial score."""
        prefs = _build_no_prefs(
            duration_preference=NumericalPreference(
                result=NumericalPreferenceResult(
                    first_value=90,
                    match_operation=NumericalMatchOperation.BETWEEN,
                    second_value=120,
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        # 135 min = 15 min over; grace=30 → score = 1 - 15/30 = 0.5
        card = _build_card(1, runtime_minutes=135)
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(0.5)

    def test_maturity_exact_match(self):
        """Maturity rating exact match → score = 1.0."""
        prefs = _build_no_prefs(
            maturity_rating_preference=MaturityPreference(
                result=MaturityPreferenceResult(
                    rating=MaturityRating.PG_13.value,
                    match_operation=RatingMatchOperation.EXACT,
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1, maturity_rank=MaturityRating.PG_13.maturity_rank)
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(1.0)

    def test_watch_providers_include_match(self):
        """Watch provider in include list → score = 1.0."""
        from implementation.misc.helpers import create_watch_provider_offering_key
        from implementation.classes.watch_providers import STREAMING_PROVIDER_MAP

        # Pick a real service that has provider IDs
        service = StreamingService.NETFLIX
        provider_ids = STREAMING_PROVIDER_MAP.get(service, [])
        if not provider_ids:
            pytest.skip("No provider IDs for Netflix in STREAMING_PROVIDER_MAP")

        # Create an offer key using the first provider ID and subscription type
        offer_key = create_watch_provider_offering_key(provider_ids[0], StreamingAccessType.SUBSCRIPTION.type_id)

        prefs = _build_no_prefs(
            watch_providers_preference=WatchProvidersPreference(
                result=WatchProvidersPreferenceResult(
                    should_include=[service.value],
                    should_exclude=[],
                    preferred_access_type=None,
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1, watch_offer_keys=[offer_key])
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(1.0)

    def test_audio_language_include_match(self):
        """Audio language in include list → score = 1.0."""
        lang = Language.ENGLISH
        prefs = _build_no_prefs(
            audio_languages_preference=LanguageListPreference(
                result=LanguageListPreferenceResult(
                    should_include=[lang],
                    should_exclude=[],
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1, audio_language_ids=[lang.language_id])
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════
# Category 3: Bug-Catching Tests (expected to FAIL on current code)
# ══════════════════════════════════════════════════════════════════════════

class TestBugCatching:
    """Tests designed to expose known bugs in the current implementation.

    These tests should FAIL until the bugs are fixed:
    - Bug 1: reception_preference.result AttributeError (line 375)
    - Bug 2: popular_trending_preference.result AttributeError (lines 377-380)
    - Bug 3: Debug print statements in _score_maturity_rating (lines 202-204)
    """

    def test_trending_only_active(self):
        """BUG: Accessing .result on PopularTrendingPreference crashes.

        Also tests that TRENDING activates independently when
        prefers_trending_movies=True and prefers_popular_movies=False.
        """
        prefs = _build_no_prefs(
            popular_trending_preference=PopularTrendingPreference(
                prefers_trending_movies=True,
                prefers_popular_movies=False,
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1)
        mock_fetch = AsyncMock(return_value=[card])
        mock_trending = AsyncMock(return_value={1: 0.8})

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, mock_trending):
            result = _run(create_metadata_scores(prefs, candidates))

        # Only TRENDING should be active; score = trending_score = 0.8
        assert result[0].metadata_score == pytest.approx(0.8)

    def test_popular_only_active(self):
        """BUG: Accessing .result on PopularTrendingPreference crashes.

        Also tests that POPULAR activates independently when
        prefers_popular_movies=True and prefers_trending_movies=False.
        """
        prefs = _build_no_prefs(
            popular_trending_preference=PopularTrendingPreference(
                prefers_trending_movies=False,
                prefers_popular_movies=True,
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1, popularity_score=0.7)
        mock_fetch = AsyncMock(return_value=[card])
        mock_trending = AsyncMock(return_value={})

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, mock_trending):
            result = _run(create_metadata_scores(prefs, candidates))

        # Only POPULAR should be active; score = popularity_score = 0.7
        assert result[0].metadata_score == pytest.approx(0.7)

    def test_both_trending_and_popular_active(self):
        """BUG: Accessing .result on PopularTrendingPreference crashes.

        When both are active, weighted average of both scores.
        """
        prefs = _build_no_prefs(
            popular_trending_preference=PopularTrendingPreference(
                prefers_trending_movies=True,
                prefers_popular_movies=True,
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1, popularity_score=0.6)
        mock_fetch = AsyncMock(return_value=[card])
        mock_trending = AsyncMock(return_value={1: 0.8})

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, mock_trending):
            result = _run(create_metadata_scores(prefs, candidates))

        # TRENDING(w=2)*0.8 + POPULAR(w=2)*0.6 = (1.6 + 1.2) / 4 = 0.7
        assert result[0].metadata_score == pytest.approx(0.7)

    def test_neither_trending_nor_popular_not_in_active_set(self):
        """BUG: Accessing .result on PopularTrendingPreference crashes.

        When both are False, neither should be in the active set.
        """
        prefs = _build_no_prefs(
            popular_trending_preference=PopularTrendingPreference(
                prefers_trending_movies=False,
                prefers_popular_movies=False,
            ),
        )
        candidates = [_make_candidate(1)]
        mock_fetch = AsyncMock(return_value=[])
        mock_trending = AsyncMock(return_value={})

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, mock_trending):
            result = _run(create_metadata_scores(prefs, candidates))

        # No active prefs → candidates returned unchanged at 0.0
        assert result[0].metadata_score == 0.0
        mock_fetch.assert_not_called()

    def test_reception_critically_acclaimed(self):
        """BUG: Accessing .result on ReceptionPreference crashes (line 375).

        reception_type=CRITICALLY_ACCLAIMED should activate RECEPTION.
        """
        prefs = _build_no_prefs(
            reception_preference=ReceptionPreference(
                reception_type=ReceptionType.CRITICALLY_ACCLAIMED,
            ),
        )
        candidates = [_make_candidate(1)]
        # reception_score=95 → (95 - 55) / 40 = 1.0
        card = _build_card(1, reception_score=95.0)
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(1.0)

    def test_reception_poorly_received(self):
        """BUG: Accessing .result on ReceptionPreference crashes (line 375).

        reception_type=POORLY_RECEIVED should activate RECEPTION.
        """
        prefs = _build_no_prefs(
            reception_preference=ReceptionPreference(
                reception_type=ReceptionType.POORLY_RECEIVED,
            ),
        )
        candidates = [_make_candidate(1)]
        # reception_score=10 → (50 - 10) / 40 = 1.0
        card = _build_card(1, reception_score=10.0)
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(1.0)

    def test_reception_no_preference_stays_inactive(self):
        """reception_type=NO_PREFERENCE → RECEPTION not in active set."""
        prefs = _build_no_prefs(
            reception_preference=ReceptionPreference(
                reception_type=ReceptionType.NO_PREFERENCE,
            ),
        )
        candidates = [_make_candidate(1)]
        mock_fetch = AsyncMock(return_value=[])
        mock_trending = AsyncMock(return_value={})

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, mock_trending):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == 0.0
        mock_fetch.assert_not_called()

    def test_maturity_off_by_one_no_stdout(self, capsys):
        """BUG: Debug print() calls in _score_maturity_rating (lines 202-204).

        When maturity rank is off by 1, the scorer prints debug output.
        """
        prefs = _build_no_prefs(
            maturity_rating_preference=MaturityPreference(
                result=MaturityPreferenceResult(
                    rating=MaturityRating.PG_13.value,
                    match_operation=RatingMatchOperation.EXACT,
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        # PG_13 rank is 3; R rank is 4 → distance=1 → triggers print()
        card = _build_card(1, maturity_rank=MaturityRating.R.maturity_rank)
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        captured = capsys.readouterr()
        assert captured.out == "", f"Unexpected stdout output (debug prints): {captured.out!r}"
        # Score should be 0.5 for distance=1
        assert result[0].metadata_score == pytest.approx(0.5)


# ══════════════════════════════════════════════════════════════════════════
# Category 4: Multi-Preference Weighted Average
# ══════════════════════════════════════════════════════════════════════════

class TestWeightedAverage:
    """Test the weighted-average formula across multiple active preferences."""

    def test_two_prefs_genres_and_duration(self):
        """genres(w=5)*1.0 + duration(w=2)*0.5 → (5 + 1) / 7 ≈ 0.857."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value],
                    should_exclude=[],
                ),
            ),
            duration_preference=NumericalPreference(
                result=NumericalPreferenceResult(
                    first_value=90,
                    match_operation=NumericalMatchOperation.BETWEEN,
                    second_value=120,
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        # Genres: has ACTION → 1.0; Duration: 135 min → 0.5
        card = _build_card(1, genre_ids=[Genre.ACTION.genre_id], runtime_minutes=135)
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        expected = (5 * 1.0 + 2 * 0.5) / 7
        assert result[0].metadata_score == pytest.approx(expected)

    def test_three_prefs_genres_release_date_maturity(self):
        """genres(5)*1.0 + release_date(4)*1.0 + maturity(3)*1.0 → 1.0."""
        mid_2021 = int(datetime(2021, 6, 15, tzinfo=timezone.utc).timestamp())
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.DRAMA.value],
                    should_exclude=[],
                ),
            ),
            release_date_preference=DatePreference(
                result=DatePreferenceResult(
                    first_date="2020-01-01",
                    match_operation=DateMatchOperation.BETWEEN,
                    second_date="2023-01-01",
                ),
            ),
            maturity_rating_preference=MaturityPreference(
                result=MaturityPreferenceResult(
                    rating=MaturityRating.R.value,
                    match_operation=RatingMatchOperation.EXACT,
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(
            1,
            genre_ids=[Genre.DRAMA.genre_id],
            release_ts=mid_2021,
            maturity_rank=MaturityRating.R.maturity_rank,
        )
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        expected = (5 * 1.0 + 4 * 1.0 + 3 * 1.0) / 12
        assert result[0].metadata_score == pytest.approx(expected)

    def test_genre_exclusion_drags_average_negative(self):
        """genres(5)*-2.0 + duration(2)*1.0 → (-10 + 2) / 7 = -1.143."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[],
                    should_exclude=[Genre.HORROR.value],
                ),
            ),
            duration_preference=NumericalPreference(
                result=NumericalPreferenceResult(
                    first_value=90,
                    match_operation=NumericalMatchOperation.BETWEEN,
                    second_value=120,
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1, genre_ids=[Genre.HORROR.genre_id], runtime_minutes=100)
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        expected = (5 * -2.0 + 2 * 1.0) / 7
        assert result[0].metadata_score == pytest.approx(expected, abs=0.001)

    def test_all_wrapper_type_prefs_active(self):
        """All 6 wrapper-type prefs active → total_weight = genres(5)+release_date(4)+watch_providers(4)+audio_language(3)+maturity(3)+duration(2) = 21."""
        from implementation.misc.helpers import create_watch_provider_offering_key
        from implementation.classes.watch_providers import STREAMING_PROVIDER_MAP

        mid_2021 = int(datetime(2021, 6, 15, tzinfo=timezone.utc).timestamp())
        service = StreamingService.NETFLIX
        provider_ids = STREAMING_PROVIDER_MAP.get(service, [])
        if not provider_ids:
            pytest.skip("No provider IDs for Netflix")
        offer_key = create_watch_provider_offering_key(provider_ids[0], StreamingAccessType.SUBSCRIPTION.type_id)

        lang = Language.ENGLISH

        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value],
                    should_exclude=[],
                ),
            ),
            release_date_preference=DatePreference(
                result=DatePreferenceResult(
                    first_date="2020-01-01",
                    match_operation=DateMatchOperation.BETWEEN,
                    second_date="2023-01-01",
                ),
            ),
            duration_preference=NumericalPreference(
                result=NumericalPreferenceResult(
                    first_value=90,
                    match_operation=NumericalMatchOperation.BETWEEN,
                    second_value=120,
                ),
            ),
            maturity_rating_preference=MaturityPreference(
                result=MaturityPreferenceResult(
                    rating=MaturityRating.PG_13.value,
                    match_operation=RatingMatchOperation.EXACT,
                ),
            ),
            watch_providers_preference=WatchProvidersPreference(
                result=WatchProvidersPreferenceResult(
                    should_include=[service.value],
                    should_exclude=[],
                    preferred_access_type=None,
                ),
            ),
            audio_languages_preference=LanguageListPreference(
                result=LanguageListPreferenceResult(
                    should_include=[lang],
                    should_exclude=[],
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(
            1,
            genre_ids=[Genre.ACTION.genre_id],
            release_ts=mid_2021,
            runtime_minutes=100,
            maturity_rank=MaturityRating.PG_13.maturity_rank,
            watch_offer_keys=[offer_key],
            audio_language_ids=[lang.language_id],
        )
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        # All scores = 1.0, so weighted average = 1.0
        assert result[0].metadata_score == pytest.approx(1.0)


# ══════════════════════════════════════════════════════════════════════════
# Category 5: I/O / Data Fetching
# ══════════════════════════════════════════════════════════════════════════

class TestIOCalls:
    """Tests verifying correct I/O calls for data fetching."""

    def test_trending_active_calls_both_fetch_and_trending(self):
        """When TRENDING is active, both fetch_movie_cards and read_trending_scores are called."""
        prefs = _build_no_prefs(
            popular_trending_preference=PopularTrendingPreference(
                prefers_trending_movies=True,
                prefers_popular_movies=False,
            ),
        )
        candidates = [_make_candidate(1)]
        mock_fetch = AsyncMock(return_value=[_build_card(1)])
        mock_trending = AsyncMock(return_value={})

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, mock_trending):
            _run(create_metadata_scores(prefs, candidates))

        mock_fetch.assert_called_once()
        mock_trending.assert_called_once()

    def test_trending_inactive_does_not_call_read_trending(self):
        """When TRENDING is not active, read_trending_scores should not be called."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value],
                    should_exclude=[],
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        mock_fetch = AsyncMock(return_value=[_build_card(1, genre_ids=[Genre.ACTION.genre_id])])
        mock_trending = AsyncMock(return_value={})

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, mock_trending):
            _run(create_metadata_scores(prefs, candidates))

        mock_fetch.assert_called_once()
        mock_trending.assert_not_called()

    def test_correct_movie_ids_passed_to_fetch(self):
        """fetch_movie_cards receives the correct list of movie_ids."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value],
                    should_exclude=[],
                ),
            ),
        )
        candidates = [_make_candidate(10), _make_candidate(20), _make_candidate(30)]
        mock_fetch = AsyncMock(return_value=[])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            _run(create_metadata_scores(prefs, candidates))

        mock_fetch.assert_called_once_with([10, 20, 30])


# ══════════════════════════════════════════════════════════════════════════
# Category 6: Multiple Candidates
# ══════════════════════════════════════════════════════════════════════════

class TestMultipleCandidates:
    """Tests for scoring multiple candidates correctly."""

    def test_three_candidates_scored_independently(self):
        """3 candidates with different cards get different scores."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value, Genre.COMEDY.value],
                    should_exclude=[],
                ),
            ),
        )
        candidates = [_make_candidate(1), _make_candidate(2), _make_candidate(3)]
        cards = [
            _build_card(1, genre_ids=[Genre.ACTION.genre_id, Genre.COMEDY.genre_id]),  # 2/2 = 1.0
            _build_card(2, genre_ids=[Genre.ACTION.genre_id]),                          # 1/2 = 0.5
            _build_card(3, genre_ids=[Genre.DRAMA.genre_id]),                           # 0/2 = 0.0
        ]
        mock_fetch = AsyncMock(return_value=cards)

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(1.0)
        assert result[1].metadata_score == pytest.approx(0.5)
        assert result[2].metadata_score == pytest.approx(0.0)

    def test_returned_list_is_same_object(self):
        """create_metadata_scores mutates in place and returns the same list."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value],
                    should_exclude=[],
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        mock_fetch = AsyncMock(return_value=[_build_card(1, genre_ids=[Genre.ACTION.genre_id])])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result is candidates

    def test_vector_and_lexical_scores_untouched(self):
        """vector_score and lexical_score remain unchanged after scoring."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value],
                    should_exclude=[],
                ),
            ),
        )
        candidates = [_make_candidate(1, vector_score=0.77, lexical_score=0.33)]
        mock_fetch = AsyncMock(return_value=[_build_card(1, genre_ids=[Genre.ACTION.genre_id])])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].vector_score == pytest.approx(0.77)
        assert result[0].lexical_score == pytest.approx(0.33)


# ══════════════════════════════════════════════════════════════════════════
# Category 7: Additional Edge Cases
# ══════════════════════════════════════════════════════════════════════════

class TestAdditionalEdgeCases:
    """Additional edge case tests."""

    def test_movie_card_with_all_null_fields(self):
        """Movie card exists but all relevant fields are None → every scorer returns 0.0."""
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value],
                    should_exclude=[],
                ),
            ),
            duration_preference=NumericalPreference(
                result=NumericalPreferenceResult(
                    first_value=90,
                    match_operation=NumericalMatchOperation.EXACT,
                    second_value=None,
                ),
            ),
        )
        candidates = [_make_candidate(1)]
        # All fields None
        card = _build_card(1)
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(0.0)

    def test_single_candidate_single_preference_minimal(self):
        """Minimal path: 1 candidate, 1 preference, perfect match."""
        prefs = _build_no_prefs(
            duration_preference=NumericalPreference(
                result=NumericalPreferenceResult(
                    first_value=120,
                    match_operation=NumericalMatchOperation.EXACT,
                    second_value=None,
                ),
            ),
        )
        candidates = [_make_candidate(42)]
        card = _build_card(42, runtime_minutes=120)
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        assert result[0].metadata_score == pytest.approx(1.0)

    def test_reception_type_no_preference_does_not_crash_line_434(self):
        """BUG (minor): Line 434 runs ReceptionType() unconditionally.

        With reception_type=NO_PREFERENCE, this shouldn't crash
        (it's wasteful but not fatal). This test verifies the no-crash path.
        """
        prefs = _build_no_prefs(
            genres_preference=GenreListPreference(
                result=GenreListPreferenceResult(
                    should_include=[Genre.ACTION.value],
                    should_exclude=[],
                ),
            ),
            reception_preference=ReceptionPreference(
                reception_type=ReceptionType.NO_PREFERENCE,
            ),
        )
        candidates = [_make_candidate(1)]
        card = _build_card(1, genre_ids=[Genre.ACTION.genre_id])
        mock_fetch = AsyncMock(return_value=[card])

        with patch(_PATCH_FETCH, mock_fetch), patch(_PATCH_TRENDING, AsyncMock(return_value={})):
            result = _run(create_metadata_scores(prefs, candidates))

        # Only genres active → score = 1.0
        assert result[0].metadata_score == pytest.approx(1.0)
