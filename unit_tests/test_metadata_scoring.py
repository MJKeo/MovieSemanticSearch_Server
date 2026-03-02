"""Unit tests for metadata scoring helpers in db.metadata_scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
from itertools import product
import math
from types import ModuleType
import sys

import pytest

# metadata_scoring imports db.postgres/db.redis at module import time.
# Provide temporary stubs if those heavy modules are unavailable in test env.
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

from db.metadata_scoring import (
    _precompute_duration,
    _precompute_maturity,
    _precompute_release_date,
    _precompute_watch_providers,
    _score_audio_language,
    _score_duration,
    _score_genres,
    _score_maturity_rating,
    _score_popular,
    _score_reception,
    _score_release_date,
    _score_trending,
    _score_watch_providers,
)
from implementation.classes.enums import (
    DateMatchOperation,
    MaturityRating,
    NumericalMatchOperation,
    RatingMatchOperation,
    ReceptionType,
    StreamingAccessType,
)
from implementation.classes.watch_providers import STREAMING_PROVIDER_MAP, StreamingService
from implementation.misc.helpers import create_watch_provider_offering_key

for _module_name in _temporary_stubs:
    sys.modules.pop(_module_name, None)


_DAY_SECONDS = 86400
_ALL_METHOD_IDS = [m.type_id for m in StreamingAccessType]

_NETFLIX_ID = STREAMING_PROVIDER_MAP[StreamingService.NETFLIX][0]
_HULU_ID = STREAMING_PROVIDER_MAP[StreamingService.HULU][0]
_AMAZON_PRIME_ID = STREAMING_PROVIDER_MAP[StreamingService.AMAZON][0]


def _utc_ts(date_str: str) -> float:
    """Convert YYYY-MM-DD to a UTC timestamp."""
    return datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()


def _all_method_keys(provider_id: int) -> set[int]:
    """Return encoded offering keys for all access methods of one provider."""
    return {create_watch_provider_offering_key(provider_id, method_id) for method_id in _ALL_METHOD_IDS}


def _key(provider_id: int, access_type: StreamingAccessType) -> int:
    """Return one encoded watch provider offering key."""
    return create_watch_provider_offering_key(provider_id, access_type.type_id)


_RATED_MATURITY_TARGETS = [
    MaturityRating.G,
    MaturityRating.PG,
    MaturityRating.PG_13,
    MaturityRating.R,
    MaturityRating.NC_17,
]
_RATED_MATURITY_RANKS = [rating.maturity_rank for rating in _RATED_MATURITY_TARGETS]
_MATURITY_MATCH_OPERATIONS = [
    RatingMatchOperation.EXACT,
    RatingMatchOperation.GREATER_THAN,
    RatingMatchOperation.LESS_THAN,
    RatingMatchOperation.GREATER_THAN_OR_EQUAL,
    RatingMatchOperation.LESS_THAN_OR_EQUAL,
]


def _expected_valid_maturity_ranks(
    target: MaturityRating,
    match_operation: RatingMatchOperation,
) -> set[int]:
    """Return the rated maturity ranks allowed by the guide semantics."""
    if target == MaturityRating.UNRATED:
        return {MaturityRating.UNRATED.maturity_rank} if match_operation == RatingMatchOperation.EXACT else set()

    target_rank = target.maturity_rank

    if match_operation == RatingMatchOperation.EXACT:
        return {target_rank}
    if match_operation == RatingMatchOperation.GREATER_THAN:
        return {rank for rank in _RATED_MATURITY_RANKS if rank > target_rank}
    if match_operation == RatingMatchOperation.LESS_THAN:
        return {rank for rank in _RATED_MATURITY_RANKS if rank < target_rank}
    if match_operation == RatingMatchOperation.GREATER_THAN_OR_EQUAL:
        return {rank for rank in _RATED_MATURITY_RANKS if rank >= target_rank}
    if match_operation == RatingMatchOperation.LESS_THAN_OR_EQUAL:
        return {rank for rank in _RATED_MATURITY_RANKS if rank <= target_rank}

    raise AssertionError(f"Unhandled match operation: {match_operation}")


def _expected_maturity_score(
    movie_rank: int | None,
    target: MaturityRating,
    match_operation: RatingMatchOperation,
    lo: int,
    hi: int,
) -> float:
    """Return the maturity score implied by the guide, independent of production logic."""
    unrated_rank = MaturityRating.UNRATED.maturity_rank

    if movie_rank is None:
        return 0.0

    if movie_rank == unrated_rank:
        return 1.0 if target == MaturityRating.UNRATED and match_operation == RatingMatchOperation.EXACT else 0.0

    if target == MaturityRating.UNRATED:
        return 0.0

    valid_ranks = _expected_valid_maturity_ranks(target, match_operation)
    if movie_rank in valid_ranks:
        return 1.0
    if not valid_ranks:
        distance = min(abs(movie_rank - lo), abs(movie_rank - hi))
        return 0.5 if distance <= 1 else 0.0

    distance = min(abs(movie_rank - valid_rank) for valid_rank in valid_ranks)
    return 0.5 if distance == 1 else 0.0


def _expected_reception_score(
    reception_score: float | None,
    reception_type: ReceptionType,
) -> float:
    """Return the reception score implied by the guide, independent of production logic."""
    if reception_score is None:
        return 0.0

    if reception_type == ReceptionType.CRITICALLY_ACCLAIMED:
        raw_score = (reception_score - 55.0) / 40.0
    elif reception_type == ReceptionType.POORLY_RECEIVED:
        raw_score = (50.0 - reception_score) / 40.0
    else:
        raise AssertionError(f"Unhandled reception type: {reception_type}")

    return max(0.0, min(1.0, raw_score))


def test_score_release_date_returns_zero_when_release_is_missing() -> None:
    """Missing release timestamp should score zero."""
    assert _score_release_date(None, 0.0, 1.0, 365.0) == 0.0


@pytest.mark.parametrize("release_ts", [100.0, 150.0, 200.0])
def test_score_release_date_returns_full_credit_inside_range(release_ts: float) -> None:
    """Release dates in [lo, hi] (inclusive) should score full credit."""
    assert _score_release_date(release_ts, 100.0, 200.0, 365.0) == 1.0


def test_score_release_date_decays_linearly_outside_range_and_clamps() -> None:
    """Release-date score should decay linearly by day distance and clamp at zero."""
    lo = 10 * _DAY_SECONDS
    hi = 20 * _DAY_SECONDS
    grace_days = 10.0

    assert _score_release_date(8 * _DAY_SECONDS, lo, hi, grace_days) == pytest.approx(0.8)
    assert _score_release_date(22 * _DAY_SECONDS, lo, hi, grace_days) == pytest.approx(0.8)
    assert _score_release_date(0, lo, hi, grace_days) == 0.0
    assert _score_release_date(35 * _DAY_SECONDS, lo, hi, grace_days) == 0.0


def test_score_duration_returns_zero_when_runtime_is_missing() -> None:
    """Missing runtime should score zero."""
    assert _score_duration(None, 90.0, 120.0) == 0.0


@pytest.mark.parametrize("runtime", [90, 105, 120])
def test_score_duration_returns_full_credit_inside_range(runtime: int) -> None:
    """Runtimes in [lo, hi] (inclusive) should score full credit."""
    assert _score_duration(runtime, 90.0, 120.0) == 1.0


def test_score_duration_decays_linearly_outside_range_and_clamps() -> None:
    """Duration score should decay linearly by minute distance and clamp at zero."""
    assert _score_duration(80, 90.0, 120.0) == pytest.approx(2 / 3)
    assert _score_duration(75, 90.0, 120.0) == pytest.approx(0.5)
    assert _score_duration(130, 90.0, 120.0) == pytest.approx(2 / 3)
    assert _score_duration(150, 90.0, 120.0) == 0.0


# ── Additional _score_duration tests ──────────────────────────────────────


def test_score_duration_exactly_at_grace_boundary_scores_zero() -> None:
    """Runtime exactly 30 minutes outside the range should score 0.0."""
    assert _score_duration(60, 90.0, 120.0) == 0.0   # 30 below lo
    assert _score_duration(150, 90.0, 120.0) == 0.0   # 30 above hi


def test_score_duration_one_minute_beyond_grace_clamps_to_zero() -> None:
    """31 minutes outside the range must not go negative."""
    assert _score_duration(59, 90.0, 120.0) == 0.0
    assert _score_duration(151, 90.0, 120.0) == 0.0


def test_score_duration_one_minute_outside_range() -> None:
    """1 minute outside range scores 29/30."""
    assert _score_duration(89, 90.0, 120.0) == pytest.approx(29 / 30)
    assert _score_duration(121, 90.0, 120.0) == pytest.approx(29 / 30)


def test_score_duration_exact_match_range() -> None:
    """When lo == hi and runtime matches exactly, score should be 1.0."""
    assert _score_duration(100, 100.0, 100.0) == 1.0


def test_score_duration_exact_match_range_off_by_one() -> None:
    """When lo == hi, runtime 1 min off should score 29/30."""
    assert _score_duration(99, 100.0, 100.0) == pytest.approx(29 / 30)
    assert _score_duration(101, 100.0, 100.0) == pytest.approx(29 / 30)


def test_score_duration_zero_runtime() -> None:
    """Zero runtime should compute distance from lo normally."""
    assert _score_duration(0, 10.0, 20.0) == pytest.approx(2 / 3)   # 10 min away
    assert _score_duration(0, 30.0, 60.0) == 0.0                     # 30 min away
    assert _score_duration(0, 0.0, 100.0) == 1.0                     # inside range


def test_score_duration_runtime_at_lo_boundary() -> None:
    """Runtime exactly at lo should score 1.0."""
    assert _score_duration(90, 90.0, 120.0) == 1.0


def test_score_duration_runtime_at_hi_boundary() -> None:
    """Runtime exactly at hi should score 1.0."""
    assert _score_duration(120, 90.0, 120.0) == 1.0


def test_score_duration_open_upper_bound() -> None:
    """With hi=inf, any runtime >= lo scores 1.0; below lo decays."""
    assert _score_duration(100, 100.0, float("inf")) == 1.0
    assert _score_duration(500, 100.0, float("inf")) == 1.0
    assert _score_duration(90, 100.0, float("inf")) == pytest.approx(2 / 3)
    assert _score_duration(70, 100.0, float("inf")) == 0.0


def test_score_duration_open_lower_bound() -> None:
    """With lo=-inf, any runtime <= hi scores 1.0; above hi decays."""
    assert _score_duration(100, float("-inf"), 100.0) == 1.0
    assert _score_duration(0, float("-inf"), 100.0) == 1.0
    assert _score_duration(110, float("-inf"), 100.0) == pytest.approx(2 / 3)
    assert _score_duration(130, float("-inf"), 100.0) == 0.0


def test_score_duration_symmetric_decay() -> None:
    """Equal distance below lo and above hi should produce equal scores."""
    for dist in [1, 5, 10, 15, 20, 25, 29]:
        below = _score_duration(90 - dist, 90.0, 120.0)
        above = _score_duration(120 + dist, 90.0, 120.0)
        assert below == pytest.approx(above), f"Asymmetric at distance={dist}"


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        (0, 1.0),
        (5, 25 / 30),
        (10, 20 / 30),
        (15, 15 / 30),
        (20, 10 / 30),
        (25, 5 / 30),
        (30, 0.0),
    ],
)
def test_score_duration_parametrized_decay_curve(distance: int, expected: float) -> None:
    """Systematic distance-to-score mapping for the decay curve."""
    lo, hi = 100.0, 100.0
    assert _score_duration(100 - distance, lo, hi) == pytest.approx(expected)
    assert _score_duration(100 + distance, lo, hi) == pytest.approx(expected)


def test_score_duration_negative_runtime() -> None:
    """Negative runtime should compute distance from lo."""
    assert _score_duration(-10, 0.0, 100.0) == pytest.approx(2 / 3)   # 10 min below lo=0
    assert _score_duration(-30, 0.0, 100.0) == 0.0                     # 30 min below lo=0


def test_score_duration_very_large_runtime() -> None:
    """Very large runtime far beyond grace should clamp to 0.0."""
    assert _score_duration(10000, 90.0, 120.0) == 0.0


def test_score_duration_inverted_range_lo_greater_than_hi() -> None:
    """When lo > hi the range is empty; check behavior doesn't crash."""
    result = _score_duration(100, 120.0, 90.0)
    assert isinstance(result, float)


def test_score_duration_degenerate_range_at_zero() -> None:
    """lo == hi == 0 with runtime 0 should score 1.0."""
    assert _score_duration(0, 0.0, 0.0) == 1.0
    assert _score_duration(15, 0.0, 0.0) == pytest.approx(0.5)


def test_score_genres_returns_zero_when_genres_missing() -> None:
    """Missing genre_ids should score zero."""
    assert _score_genres(None, {1, 2}, {3}) == 0.0


def test_score_genres_exclusion_hit_takes_priority() -> None:
    """Any excluded genre in the movie should produce hard penalty -2.0."""
    assert _score_genres([1, 3], {1, 2}, {3, 4}) == -2.0


def test_score_genres_returns_fraction_of_inclusions_matched() -> None:
    """Genre score should be matched include fraction when includes are active."""
    assert _score_genres([1, 1, 3, 8], {1, 2, 3}, set()) == pytest.approx(2 / 3)


def test_score_genres_with_includes_but_no_matches_returns_zero() -> None:
    """If include list is active and there is no overlap, score should be zero."""
    assert _score_genres([7, 8], {1, 2, 3}, set()) == 0.0


@pytest.mark.parametrize(
    ("include_ids", "exclude_ids"),
    [
        (set(), {7}),
        (set(), set()),
    ],
)
def test_score_genres_without_includes_and_no_exclusion_hit_returns_one(
    include_ids: set[int],
    exclude_ids: set[int],
) -> None:
    """When only exclusions are in play and none match, score should be full credit."""
    assert _score_genres([1, 2], include_ids, exclude_ids) == 1.0


# ---------------------------------------------------------------------------
# Comprehensive _score_genres tests
# ---------------------------------------------------------------------------


class TestScoreGenresNoneAndEmpty:
    """None / empty input handling."""

    def test_none_genre_ids_with_empty_sets(self) -> None:
        assert _score_genres(None, set(), set()) == 0.0

    def test_empty_genre_list_no_includes_no_excludes(self) -> None:
        """Empty list (not None) with no prefs → fall-through 1.0."""
        assert _score_genres([], set(), set()) == 1.0

    def test_empty_genre_list_with_includes(self) -> None:
        """Empty list with includes → 0 matched / N includes = 0.0."""
        assert _score_genres([], {1, 2}, set()) == 0.0

    def test_empty_genre_list_with_excludes_no_hit(self) -> None:
        """Empty list can't hit any exclusion → 1.0."""
        assert _score_genres([], set(), {5, 6}) == 1.0


class TestScoreGenresExclusionPriority:
    """Exclusion checks must fire before inclusion logic."""

    def test_both_include_and_exclude_match(self) -> None:
        """Exclusion must win even when includes also match."""
        assert _score_genres([1, 2, 3], {1, 2}, {3}) == -2.0

    def test_all_genres_excluded(self) -> None:
        assert _score_genres([1, 2], set(), {1, 2}) == -2.0

    def test_single_exclusion_among_many_genres(self) -> None:
        assert _score_genres([1, 2, 3, 4, 5], set(), {5}) == -2.0

    def test_multiple_exclusion_hits_still_minus_two(self) -> None:
        """Multiple hits should not stack; still -2.0."""
        assert _score_genres([1, 2, 3], set(), {1, 2, 3}) == -2.0

    def test_genre_in_both_include_and_exclude(self) -> None:
        """A genre appearing in both sets → exclusion takes priority."""
        assert _score_genres([5], {5}, {5}) == -2.0


class TestScoreGenresInclusionFraction:
    """Inclusion fraction edge cases."""

    def test_all_includes_matched(self) -> None:
        assert _score_genres([1, 2, 3], {1, 2, 3}, set()) == pytest.approx(1.0)

    def test_single_include_single_match(self) -> None:
        assert _score_genres([1], {1}, set()) == pytest.approx(1.0)

    def test_single_include_no_match(self) -> None:
        assert _score_genres([2], {1}, set()) == pytest.approx(0.0)

    def test_large_include_set_one_match(self) -> None:
        includes = {10, 20, 30, 40, 50}
        assert _score_genres([10, 99], includes, set()) == pytest.approx(1 / 5)

    @pytest.mark.parametrize(
        ("genre_ids", "include_ids", "expected"),
        [
            ([1, 2], {1, 2, 3, 4}, 2 / 4),
            ([1], {1, 2}, 1 / 2),
            ([1, 2, 3], {1, 2, 3, 4, 5, 6}, 3 / 6),
            ([5, 6, 7], {5, 6, 7, 8, 9, 10, 11}, 3 / 7),
            ([1, 2, 3, 4, 5], {1, 2}, 2 / 2),
        ],
        ids=["2/4", "1/2", "3/6", "3/7", "superset-of-includes"],
    )
    def test_parametrized_fractions(
        self, genre_ids: list[int], include_ids: set[int], expected: float
    ) -> None:
        assert _score_genres(genre_ids, include_ids, set()) == pytest.approx(expected)


class TestScoreGenresDuplicates:
    """Duplicate genre_ids in movie list."""

    def test_duplicate_genre_ids_deduplicates(self) -> None:
        """[1,1,2] with include {1,2} → 2/2 = 1.0 after set conversion."""
        assert _score_genres([1, 1, 2], {1, 2}, set()) == pytest.approx(1.0)

    def test_many_duplicates_still_correct_fraction(self) -> None:
        assert _score_genres([3, 3, 3, 3], {3, 4}, set()) == pytest.approx(1 / 2)


class TestScoreGenresEmptySets:
    """Edge cases with empty include/exclude sets."""

    def test_empty_include_empty_exclude(self) -> None:
        assert _score_genres([1, 2], set(), set()) == 1.0

    def test_empty_exclude_with_active_includes(self) -> None:
        assert _score_genres([1, 3], {1, 2, 3}, set()) == pytest.approx(2 / 3)

    def test_empty_include_with_active_excludes_no_hit(self) -> None:
        assert _score_genres([1, 2], set(), {99}) == 1.0


class TestScoreGenresScoreRange:
    """Score should always be within [-2, 1]."""

    @pytest.mark.parametrize(
        ("genre_ids", "include_ids", "exclude_ids"),
        [
            ([1, 2, 3], {1, 2, 3}, set()),
            ([1, 2, 3], {1}, set()),
            ([1, 2, 3], set(), {4}),
            ([1, 2, 3], set(), {1}),
            ([1], {1, 2, 3, 4, 5}, set()),
            (None, {1}, {2}),
            ([], set(), set()),
        ],
    )
    def test_score_within_bounds(
        self,
        genre_ids: list[int] | None,
        include_ids: set[int],
        exclude_ids: set[int],
    ) -> None:
        score = _score_genres(genre_ids, include_ids, exclude_ids)
        assert -2.0 <= score <= 1.0


class TestScoreGenresDisjointOverlap:
    """Disjoint / overlapping set relationships."""

    def test_completely_disjoint(self) -> None:
        assert _score_genres([1, 2], {3, 4}, set()) == pytest.approx(0.0)

    def test_movie_superset_of_includes(self) -> None:
        assert _score_genres([1, 2, 3, 4, 5], {2, 3}, set()) == pytest.approx(1.0)

    def test_movie_subset_of_includes(self) -> None:
        assert _score_genres([1], {1, 2, 3}, set()) == pytest.approx(1 / 3)

    def test_partial_overlap(self) -> None:
        assert _score_genres([1, 2, 5], {1, 3, 5, 7}, set()) == pytest.approx(2 / 4)


def test_score_audio_language_returns_zero_when_languages_missing() -> None:
    """Missing audio language ids should score zero."""
    assert _score_audio_language(None, {1}, {2}) == 0.0


def test_score_audio_language_exclusion_hit_takes_priority() -> None:
    """Any excluded language present should produce hard penalty -2.0."""
    assert _score_audio_language([1, 4], {1, 2}, {4, 5}) == -2.0


def test_score_audio_language_include_overlap_returns_one() -> None:
    """With include list active, any overlap should yield 1.0."""
    assert _score_audio_language([4, 7], {1, 7}, set()) == 1.0


def test_score_audio_language_include_without_overlap_returns_zero() -> None:
    """With include list active and no overlap, score should be zero."""
    assert _score_audio_language([4, 7], {1, 2}, set()) == 0.0


@pytest.mark.parametrize(
    ("include_ids", "exclude_ids"),
    [
        (set(), {9}),
        (set(), set()),
    ],
)
def test_score_audio_language_without_includes_and_no_exclusion_hit_returns_one(
    include_ids: set[int],
    exclude_ids: set[int],
) -> None:
    """When only exclusions are active and none match, score should be full credit."""
    assert _score_audio_language([1, 2], include_ids, exclude_ids) == 1.0


# ---------------------------------------------------------------------------
# Comprehensive _score_audio_language tests
# ---------------------------------------------------------------------------


class TestScoreAudioLanguageNoneAndEmpty:
    """None / empty input handling."""

    def test_none_with_empty_sets(self) -> None:
        assert _score_audio_language(None, set(), set()) == 0.0

    def test_none_with_active_includes(self) -> None:
        assert _score_audio_language(None, {1, 2}, set()) == 0.0

    def test_none_with_active_excludes(self) -> None:
        assert _score_audio_language(None, set(), {3}) == 0.0

    def test_none_with_both_includes_and_excludes(self) -> None:
        assert _score_audio_language(None, {1}, {2}) == 0.0

    def test_empty_list_no_includes_no_excludes(self) -> None:
        """Empty list (not None) with no prefs → fall-through 1.0."""
        assert _score_audio_language([], set(), set()) == 1.0

    def test_empty_list_with_includes(self) -> None:
        """Empty list with includes → no overlap → 0.0."""
        assert _score_audio_language([], {1, 2}, set()) == 0.0

    def test_empty_list_with_excludes_no_hit(self) -> None:
        """Empty list can't hit any exclusion → 1.0."""
        assert _score_audio_language([], set(), {5, 6}) == 1.0


class TestScoreAudioLanguageExclusionPriority:
    """Exclusion checks must fire before inclusion logic."""

    def test_both_include_and_exclude_match(self) -> None:
        """Exclusion must win even when includes also match."""
        assert _score_audio_language([1, 2, 3], {1, 2}, {3}) == -2.0

    def test_all_languages_excluded(self) -> None:
        assert _score_audio_language([1, 2], set(), {1, 2}) == -2.0

    def test_single_exclusion_among_many_languages(self) -> None:
        """One excluded language among many non-excluded → still -2.0."""
        assert _score_audio_language([1, 2, 3, 4, 5], set(), {5}) == -2.0

    def test_multiple_exclusion_hits_still_minus_two(self) -> None:
        """Multiple hits should not stack; still -2.0."""
        assert _score_audio_language([1, 2, 3], set(), {1, 2, 3}) == -2.0

    def test_language_in_both_include_and_exclude(self) -> None:
        """A language appearing in both sets → exclusion takes priority."""
        assert _score_audio_language([5], {5}, {5}) == -2.0

    def test_exclusion_hit_with_no_includes_set(self) -> None:
        assert _score_audio_language([10, 20], set(), {20}) == -2.0

    def test_exclusion_hit_ignores_include_overlap(self) -> None:
        """Even if every include matches, one exclusion hit → -2.0."""
        assert _score_audio_language([1, 2, 3], {1, 2}, {3}) == -2.0


class TestScoreAudioLanguageInclusionBinary:
    """Inclusion is binary (any match → 1.0, no match → 0.0), not fractional."""

    def test_single_include_single_match(self) -> None:
        assert _score_audio_language([1], {1}, set()) == 1.0

    def test_single_include_no_match(self) -> None:
        assert _score_audio_language([2], {1}, set()) == 0.0

    def test_multiple_includes_one_match(self) -> None:
        """Any overlap at all → 1.0 (not a fraction like genres)."""
        assert _score_audio_language([3], {1, 2, 3}, set()) == 1.0

    def test_multiple_includes_all_match(self) -> None:
        assert _score_audio_language([1, 2, 3], {1, 2, 3}, set()) == 1.0

    def test_multiple_includes_none_match(self) -> None:
        assert _score_audio_language([4, 5], {1, 2, 3}, set()) == 0.0

    def test_partial_include_overlap_still_one(self) -> None:
        """Even partial overlap → 1.0, confirming binary behavior."""
        assert _score_audio_language([1, 99], {1, 2, 3, 4, 5}, set()) == 1.0

    def test_large_include_set_one_match(self) -> None:
        includes = {10, 20, 30, 40, 50}
        assert _score_audio_language([10, 99], includes, set()) == 1.0

    def test_large_include_set_no_match(self) -> None:
        includes = {10, 20, 30, 40, 50}
        assert _score_audio_language([1, 2, 3], includes, set()) == 0.0


class TestScoreAudioLanguageDuplicates:
    """Duplicate lang_ids in movie list."""

    def test_duplicate_lang_ids_still_matches(self) -> None:
        """[1,1,2] with include {1} → 1.0 after set conversion."""
        assert _score_audio_language([1, 1, 2], {1}, set()) == 1.0

    def test_duplicate_lang_ids_exclusion(self) -> None:
        """Duplicates in movie list don't affect exclusion result."""
        assert _score_audio_language([3, 3, 3], set(), {3}) == -2.0

    def test_many_duplicates_no_match(self) -> None:
        assert _score_audio_language([5, 5, 5], {1, 2}, set()) == 0.0


class TestScoreAudioLanguageEmptySets:
    """Edge cases with empty include/exclude sets."""

    def test_empty_include_empty_exclude(self) -> None:
        """No preferences at all → 1.0 (fall-through)."""
        assert _score_audio_language([1, 2], set(), set()) == 1.0

    def test_empty_exclude_with_active_includes_match(self) -> None:
        assert _score_audio_language([1, 3], {1, 2, 3}, set()) == 1.0

    def test_empty_exclude_with_active_includes_no_match(self) -> None:
        assert _score_audio_language([4], {1, 2, 3}, set()) == 0.0

    def test_empty_include_with_active_excludes_no_hit(self) -> None:
        assert _score_audio_language([1, 2], set(), {99}) == 1.0

    def test_empty_include_with_active_excludes_hit(self) -> None:
        assert _score_audio_language([1, 2], set(), {2}) == -2.0


class TestScoreAudioLanguageScoreRange:
    """Score should always be within [-2, 1]."""

    @pytest.mark.parametrize(
        ("lang_ids", "include_ids", "exclude_ids"),
        [
            ([1, 2, 3], {1, 2, 3}, set()),        # full include match
            ([1, 2, 3], {1}, set()),               # partial include match
            ([1, 2, 3], set(), {4}),               # exclude miss
            ([1, 2, 3], set(), {1}),               # exclude hit
            ([1], {1, 2, 3, 4, 5}, set()),         # small movie, big include
            (None, {1}, {2}),                       # None
            ([], set(), set()),                     # empty list
            ([1], set(), set()),                    # single lang, no prefs
            ([1, 2], {3, 4}, {5, 6}),              # no overlap anywhere
        ],
    )
    def test_score_within_bounds(
        self,
        lang_ids: list[int] | None,
        include_ids: set[int],
        exclude_ids: set[int],
    ) -> None:
        score = _score_audio_language(lang_ids, include_ids, exclude_ids)
        assert -2.0 <= score <= 1.0

    @pytest.mark.parametrize(
        ("lang_ids", "include_ids", "exclude_ids", "expected"),
        [
            ([1], {1}, set(), 1.0),
            ([1], {2}, set(), 0.0),
            ([1], set(), {1}, -2.0),
            (None, {1}, set(), 0.0),
            ([1], set(), set(), 1.0),
        ],
    )
    def test_exact_expected_values(
        self,
        lang_ids: list[int] | None,
        include_ids: set[int],
        exclude_ids: set[int],
        expected: float,
    ) -> None:
        """Verify specific expected outputs for canonical cases."""
        assert _score_audio_language(lang_ids, include_ids, exclude_ids) == expected


class TestScoreAudioLanguageDisjointOverlap:
    """Disjoint / overlapping set relationships."""

    def test_completely_disjoint_from_includes(self) -> None:
        assert _score_audio_language([1, 2], {3, 4}, set()) == 0.0

    def test_movie_superset_of_includes(self) -> None:
        """Movie has more languages than includes → still 1.0 (any match)."""
        assert _score_audio_language([1, 2, 3, 4, 5], {2, 3}, set()) == 1.0

    def test_movie_subset_of_includes(self) -> None:
        """Movie langs are subset of includes → still 1.0 (any match)."""
        assert _score_audio_language([1], {1, 2, 3}, set()) == 1.0

    def test_partial_overlap_with_includes(self) -> None:
        assert _score_audio_language([1, 2, 5], {1, 3, 5, 7}, set()) == 1.0

    def test_completely_disjoint_from_excludes(self) -> None:
        assert _score_audio_language([1, 2], set(), {3, 4}) == 1.0

    def test_movie_superset_of_excludes(self) -> None:
        """Movie has all excluded langs plus more → -2.0."""
        assert _score_audio_language([1, 2, 3], set(), {2}) == -2.0


class TestScoreAudioLanguageVsGenresBehavior:
    """Audio language inclusion is binary (any match → 1.0), unlike genres (fraction).

    These tests verify the key behavioral difference between the two scorers.
    """

    def test_single_match_out_of_many_includes_is_full_credit(self) -> None:
        """Genres would give 1/5 = 0.2. Audio language should give 1.0."""
        assert _score_audio_language([1], {1, 2, 3, 4, 5}, set()) == 1.0

    def test_no_match_with_includes_is_zero(self) -> None:
        """Both genres and audio language return 0.0 when no includes match."""
        assert _score_audio_language([99], {1, 2, 3, 4, 5}, set()) == 0.0

    def test_exclusion_penalty_same_as_genres(self) -> None:
        """Both genres and audio language return -2.0 on exclusion hit."""
        assert _score_audio_language([1], set(), {1}) == -2.0


def test_score_watch_providers_returns_zero_when_offers_missing() -> None:
    """Missing watch offers should score zero."""
    assert _score_watch_providers(None, set(), set(), set(), None) == 0.0


def test_score_watch_providers_exclusion_prefilters_before_include_check() -> None:
    """Excluded keys are removed before the include check; excluded-only offer scores zero."""
    netflix_all = _all_method_keys(_NETFLIX_ID)
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    assert (
        _score_watch_providers(
            [netflix_sub],
            netflix_all,           # exclude_key_set (flat set)
            netflix_all,           # include_any_keys
            {netflix_sub},         # include_desired_keys
            None,
        )
        == 0.0
    )


def test_score_watch_providers_include_desired_method_match_returns_one() -> None:
    """Desired method overlap should return full credit."""
    hulu_sub = _key(_HULU_ID, StreamingAccessType.SUBSCRIPTION)
    assert (
        _score_watch_providers(
            [hulu_sub],
            set(),
            _all_method_keys(_HULU_ID),
            {hulu_sub},
            None,
        )
        == 1.0
    )


def test_score_watch_providers_any_method_fallback_returns_half_credit() -> None:
    """Provider match on wrong method should return 0.5 when includes are active."""
    hulu_sub = _key(_HULU_ID, StreamingAccessType.SUBSCRIPTION)
    hulu_buy = _key(_HULU_ID, StreamingAccessType.BUY)
    assert (
        _score_watch_providers(
            [hulu_buy],
            set(),
            _all_method_keys(_HULU_ID),
            {hulu_sub},
            None,
        )
        == 0.5
    )


def test_score_watch_providers_include_active_without_overlap_returns_zero() -> None:
    """When include preference is active and there is no overlap, score should be zero."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    hulu_sub = _key(_HULU_ID, StreamingAccessType.SUBSCRIPTION)
    assert (
        _score_watch_providers(
            [netflix_sub],
            set(),
            _all_method_keys(_HULU_ID),
            {hulu_sub},
            None,
        )
        == 0.0
    )


def test_score_watch_providers_no_include_preference_and_no_exclusion_returns_one() -> None:
    """Without include or exclusion preference, any offer yields full credit."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    assert (
        _score_watch_providers(
            [netflix_sub],
            set(),
            set(),
            set(),
            None,
        )
        == 1.0
    )


def test_score_watch_providers_no_include_preference_still_respects_exclusions() -> None:
    """Excluded-only offer after pre-filter leaves empty set, scoring zero."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    assert (
        _score_watch_providers(
            [netflix_sub],
            _all_method_keys(_NETFLIX_ID),  # exclude_key_set (flat set)
            set(),
            set(),
            None,
        )
        == 0.0
    )


def test_score_watch_providers_access_type_only_matches_type_bits() -> None:
    """When only access_type_id is set, match on lower 4 bits of offer keys."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    assert (
        _score_watch_providers(
            [netflix_sub],
            set(),
            set(),
            set(),
            StreamingAccessType.SUBSCRIPTION.type_id,
        )
        == 1.0
    )
    assert (
        _score_watch_providers(
            [netflix_sub],
            set(),
            set(),
            set(),
            StreamingAccessType.BUY.type_id,
        )
        == 0.0
    )


def test_score_maturity_rating_returns_zero_when_movie_rank_missing() -> None:
    """Missing maturity rank should score zero."""
    assert _score_maturity_rating(None, MaturityRating.PG_13, 3, 3) == 0.0


def test_score_maturity_rating_handles_unrated_movie_and_target() -> None:
    """Unrated movies only match the exact unrated target."""
    unrated_rank = MaturityRating.UNRATED.maturity_rank
    assert _score_maturity_rating(unrated_rank, MaturityRating.UNRATED, unrated_rank, unrated_rank) == 1.0
    assert _score_maturity_rating(unrated_rank, MaturityRating.PG, 1, 3) == 0.0
    assert _score_maturity_rating(MaturityRating.PG.maturity_rank, MaturityRating.UNRATED, 999, 999) == 0.0


@pytest.mark.parametrize(
    ("movie_rank", "target", "lo", "hi"),
    [
        (3, MaturityRating.PG_13, 3, 3),
        (2, MaturityRating.PG_13, 2, 4),
        (4, MaturityRating.PG_13, 2, 4),
        (1, MaturityRating.PG_13, 1, 2),
        (5, MaturityRating.R, 4, 5),
    ],
)
def test_score_maturity_rating_returns_one_within_range(
    movie_rank: int,
    target: MaturityRating,
    lo: int,
    hi: int,
) -> None:
    """Ratings inside the valid range should get full credit."""
    assert _score_maturity_rating(movie_rank, target, lo, hi) == 1.0


@pytest.mark.parametrize(
    ("movie_rank", "target", "lo", "hi"),
    [
        (2, MaturityRating.PG_13, 3, 3),
        (4, MaturityRating.PG_13, 3, 3),
        (1, MaturityRating.PG_13, 2, 4),
        (5, MaturityRating.PG_13, 2, 4),
        (3, MaturityRating.PG, 1, 2),
        (3, MaturityRating.NC_17, 4, 5),
        (5, MaturityRating.NC_17, 6, 5),
    ],
)
def test_score_maturity_rating_returns_half_credit_for_off_by_one(
    movie_rank: int,
    target: MaturityRating,
    lo: int,
    hi: int,
) -> None:
    """Ratings one ordinal step from the nearest valid edge should get half credit."""
    assert _score_maturity_rating(movie_rank, target, lo, hi) == 0.5


@pytest.mark.parametrize(
    ("movie_rank", "target", "lo", "hi"),
    [
        (1, MaturityRating.PG_13, 3, 3),
        (5, MaturityRating.PG, 1, 2),
        (1, MaturityRating.NC_17, 4, 5),
        (999, MaturityRating.PG_13, 2, 4),
    ],
)
def test_score_maturity_rating_returns_zero_for_off_by_two_or_more(
    movie_rank: int,
    target: MaturityRating,
    lo: int,
    hi: int,
) -> None:
    """Ratings two or more steps from the valid range should score zero."""
    assert _score_maturity_rating(movie_rank, target, lo, hi) == 0.0


@pytest.mark.parametrize(
    "reception_type",
    [ReceptionType.CRITICALLY_ACCLAIMED, ReceptionType.POORLY_RECEIVED],
)
def test_score_reception_returns_zero_when_missing(reception_type: ReceptionType) -> None:
    """Missing reception score should return zero for both supported reception modes."""
    assert _score_reception(None, reception_type) == 0.0


@pytest.mark.parametrize(
    ("reception_score", "expected"),
    [
        (55.0, 0.0),
        (65.0, 0.25),
        (75.0, 0.5),
        (85.0, 0.75),
        (95.0, 1.0),
    ],
)
def test_score_reception_critically_acclaimed_matches_guide_anchor_points(
    reception_score: float,
    expected: float,
) -> None:
    """Critically acclaimed guide examples should map to their documented scores."""
    assert _score_reception(reception_score, ReceptionType.CRITICALLY_ACCLAIMED) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("reception_score", "expected"),
    [
        (50.0, 0.0),
        (40.0, 0.25),
        (30.0, 0.5),
        (20.0, 0.75),
        (10.0, 1.0),
    ],
)
def test_score_reception_poorly_received_matches_guide_anchor_points(
    reception_score: float,
    expected: float,
) -> None:
    """Poorly received guide examples should map to their documented scores."""
    assert _score_reception(reception_score, ReceptionType.POORLY_RECEIVED) == pytest.approx(expected)


@pytest.mark.parametrize(
    "reception_score",
    [54.999, 55.001, 94.999, 95.0, 95.001],
)
def test_score_reception_critically_acclaimed_clamps_at_ramp_boundaries(
    reception_score: float,
) -> None:
    """Critically acclaimed scoring should transition cleanly at both clamp boundaries."""
    expected = _expected_reception_score(reception_score, ReceptionType.CRITICALLY_ACCLAIMED)
    assert _score_reception(reception_score, ReceptionType.CRITICALLY_ACCLAIMED) == pytest.approx(expected)


@pytest.mark.parametrize(
    "reception_score",
    [50.001, 49.999, 10.001, 10.0, 9.999],
)
def test_score_reception_poorly_received_clamps_at_ramp_boundaries(
    reception_score: float,
) -> None:
    """Poorly received scoring should transition cleanly at both clamp boundaries."""
    expected = _expected_reception_score(reception_score, ReceptionType.POORLY_RECEIVED)
    assert _score_reception(reception_score, ReceptionType.POORLY_RECEIVED) == pytest.approx(expected)


@pytest.mark.parametrize(
    "reception_score",
    [-20.0, 0.0, 120.0, 1000.0],
)
def test_score_reception_critically_acclaimed_clamps_out_of_range_values(
    reception_score: float,
) -> None:
    """Critically acclaimed scoring should stay clamped for out-of-range inputs."""
    expected = _expected_reception_score(reception_score, ReceptionType.CRITICALLY_ACCLAIMED)
    assert _score_reception(reception_score, ReceptionType.CRITICALLY_ACCLAIMED) == pytest.approx(expected)


@pytest.mark.parametrize(
    "reception_score",
    [-20.0, 0.0, 120.0, 1000.0],
)
def test_score_reception_poorly_received_clamps_out_of_range_values(
    reception_score: float,
) -> None:
    """Poorly received scoring should stay clamped for out-of-range inputs."""
    expected = _expected_reception_score(reception_score, ReceptionType.POORLY_RECEIVED)
    assert _score_reception(reception_score, ReceptionType.POORLY_RECEIVED) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("reception_type", "reception_score", "expected"),
    [
        (ReceptionType.CRITICALLY_ACCLAIMED, 60.0, 0.125),
        (ReceptionType.CRITICALLY_ACCLAIMED, 72.5, 0.4375),
        (ReceptionType.CRITICALLY_ACCLAIMED, 82.5, 0.6875),
        (ReceptionType.POORLY_RECEIVED, 45.0, 0.125),
        (ReceptionType.POORLY_RECEIVED, 27.5, 0.5625),
        (ReceptionType.POORLY_RECEIVED, 12.5, 0.9375),
    ],
)
def test_score_reception_preserves_linearity_for_fractional_inputs(
    reception_type: ReceptionType,
    reception_score: float,
    expected: float,
) -> None:
    """Fractional inputs should follow the guide's linear ramp without bucketing or rounding."""
    assert _score_reception(reception_score, reception_type) == pytest.approx(expected)


def test_score_reception_critically_acclaimed_is_monotonic_non_decreasing() -> None:
    """Critically acclaimed scoring should never decrease as reception improves."""
    sample_scores = [0.0, 20.0, 55.0, 60.0, 75.0, 95.0, 120.0]
    results = [_score_reception(score, ReceptionType.CRITICALLY_ACCLAIMED) for score in sample_scores]

    for previous, current in zip(results, results[1:]):
        assert previous <= current


def test_score_reception_poorly_received_is_monotonic_non_increasing() -> None:
    """Poorly received scoring should never increase as reception improves."""
    sample_scores = [0.0, 10.0, 20.0, 30.0, 50.0, 80.0, 120.0]
    results = [_score_reception(score, ReceptionType.POORLY_RECEIVED) for score in sample_scores]

    for previous, current in zip(results, results[1:]):
        assert previous >= current


@pytest.mark.parametrize(
    ("reception_type", "reception_score"),
    list(product(
        [ReceptionType.CRITICALLY_ACCLAIMED, ReceptionType.POORLY_RECEIVED],
        [None, -50.0, 0.0, 10.0, 30.0, 55.0, 75.0, 95.0, 120.0],
    )),
)
def test_score_reception_always_returns_a_value_in_unit_interval(
    reception_type: ReceptionType,
    reception_score: float | None,
) -> None:
    """Reception scores should always be clamped to the unit interval."""
    result = _score_reception(reception_score, reception_type)
    assert 0.0 <= result <= 1.0


def test_score_trending_returns_zero_when_movie_not_present() -> None:
    """Missing movie IDs in trending map should score zero."""
    assert _score_trending(42, {1: 0.9}) == 0.0


def test_score_trending_returns_stored_score_for_present_movie() -> None:
    """Trending score should return the dict value for a present movie ID."""
    assert _score_trending(42, {42: 0.73}) == pytest.approx(0.73)


def test_score_popular_returns_zero_when_missing() -> None:
    """Missing popularity score should return zero."""
    assert _score_popular(None) == 0.0


@pytest.mark.parametrize(
    ("popularity_score", "expected"),
    [
        (0, 0.0),
        (0.63, 0.63),
        (1.25, 1.25),
        (-0.2, -0.2),
    ],
)
def test_score_popular_passes_through_numeric_values(
    popularity_score: float,
    expected: float,
) -> None:
    """Popularity scorer should pass through numeric value as float."""
    assert _score_popular(popularity_score) == pytest.approx(expected)


def test_precompute_release_date_between_sorts_bounds_and_uses_variable_grace() -> None:
    """Between mode should sort dates and use half-range grace within clamp limits."""
    lo, hi, grace_days = _precompute_release_date("2023-01-01", DateMatchOperation.BETWEEN, "2021-01-01")
    assert lo == _utc_ts("2021-01-01")
    assert hi == _utc_ts("2023-01-01")
    assert grace_days == 365.0


def test_precompute_release_date_between_applies_minimum_grace_clamp() -> None:
    """Small between ranges should clamp grace to minimum 365 days."""
    _, _, grace_days = _precompute_release_date("2021-01-01", DateMatchOperation.BETWEEN, "2021-06-01")
    assert grace_days == 365.0


def test_precompute_release_date_between_applies_maximum_grace_clamp() -> None:
    """Very wide between ranges should clamp grace to maximum 1825 days."""
    _, _, grace_days = _precompute_release_date("1900-01-01", DateMatchOperation.BETWEEN, "2100-01-01")
    assert grace_days == 1825.0


def test_precompute_release_date_after_builds_open_ended_upper_range() -> None:
    """After mode should set hi to +inf and fixed 3-year grace."""
    lo, hi, grace_days = _precompute_release_date("2010-05-20", DateMatchOperation.AFTER, None)
    assert lo == _utc_ts("2010-05-20")
    assert math.isinf(hi) and hi > 0
    assert grace_days == 1095.0


def test_precompute_release_date_before_builds_open_ended_lower_range() -> None:
    """Before mode should set lo to -inf and fixed 3-year grace."""
    lo, hi, grace_days = _precompute_release_date("2010-05-20", DateMatchOperation.BEFORE, None)
    assert math.isinf(lo) and lo < 0
    assert hi == _utc_ts("2010-05-20")
    assert grace_days == 1095.0


def test_precompute_release_date_between_with_missing_second_date_falls_back_to_exact() -> None:
    """Between with missing second date should behave as exact."""
    lo, hi, grace_days = _precompute_release_date("2010-05-20", DateMatchOperation.BETWEEN, None)
    expected = _utc_ts("2010-05-20")
    assert lo == expected
    assert hi == expected
    assert grace_days == 730.0


def test_precompute_release_date_unknown_operation_falls_back_to_exact() -> None:
    """Unknown operations should use exact-date behavior."""
    lo, hi, grace_days = _precompute_release_date("2010-05-20", "unexpected", "2015-01-01")
    expected = _utc_ts("2010-05-20")
    assert lo == expected
    assert hi == expected
    assert grace_days == 730.0


# ── _score_release_date — Boundary & Edge Cases ───────────────────────────


def test_score_release_date_exact_lo_boundary_returns_one() -> None:
    """release_ts exactly at lo should return 1.0."""
    lo = _utc_ts("2000-01-01")
    hi = _utc_ts("2005-01-01")
    assert _score_release_date(lo, lo, hi, 365.0) == 1.0


def test_score_release_date_exact_hi_boundary_returns_one() -> None:
    """release_ts exactly at hi should return 1.0."""
    lo = _utc_ts("2000-01-01")
    hi = _utc_ts("2005-01-01")
    assert _score_release_date(hi, lo, hi, 365.0) == 1.0


def test_score_release_date_one_day_below_lo() -> None:
    """One day below lo should decay by 1/grace_days."""
    lo = _utc_ts("2000-01-01")
    hi = _utc_ts("2005-01-01")
    grace_days = 365.0
    release_ts = lo - _DAY_SECONDS
    assert _score_release_date(release_ts, lo, hi, grace_days) == pytest.approx(1.0 - 1.0 / 365.0)


def test_score_release_date_one_day_above_hi() -> None:
    """One day above hi should decay by 1/grace_days."""
    lo = _utc_ts("2000-01-01")
    hi = _utc_ts("2005-01-01")
    grace_days = 365.0
    release_ts = hi + _DAY_SECONDS
    assert _score_release_date(release_ts, lo, hi, grace_days) == pytest.approx(1.0 - 1.0 / 365.0)


def test_score_release_date_exact_grace_distance_returns_zero() -> None:
    """Distance exactly equal to grace_days should score 0.0."""
    lo = _utc_ts("2000-01-01")
    hi = _utc_ts("2005-01-01")
    grace_days = 100.0
    release_ts = lo - (grace_days * _DAY_SECONDS)
    assert _score_release_date(release_ts, lo, hi, grace_days) == pytest.approx(0.0)


def test_score_release_date_just_inside_grace() -> None:
    """Distance = grace_days - 1 should give a small positive score."""
    lo = _utc_ts("2000-01-01")
    hi = _utc_ts("2005-01-01")
    grace_days = 100.0
    release_ts = lo - ((grace_days - 1) * _DAY_SECONDS)
    assert _score_release_date(release_ts, lo, hi, grace_days) == pytest.approx(1.0 / 100.0)


def test_score_release_date_half_grace_returns_half() -> None:
    """Distance = grace_days / 2 should score 0.5."""
    lo = _utc_ts("2000-01-01")
    hi = _utc_ts("2005-01-01")
    grace_days = 400.0
    release_ts = lo - (200.0 * _DAY_SECONDS)
    assert _score_release_date(release_ts, lo, hi, grace_days) == pytest.approx(0.5)


def test_score_release_date_beyond_grace_clamps_to_zero() -> None:
    """Distance > grace_days should clamp to 0.0, never go negative."""
    lo = _utc_ts("2000-01-01")
    hi = _utc_ts("2005-01-01")
    grace_days = 100.0
    release_ts = lo - (200.0 * _DAY_SECONDS)
    assert _score_release_date(release_ts, lo, hi, grace_days) == 0.0


def test_score_release_date_release_ts_zero_scores_normally() -> None:
    """release_ts = 0 (epoch) is a valid timestamp and should score normally."""
    lo = 0.0
    hi = 100.0 * _DAY_SECONDS
    grace_days = 50.0
    assert _score_release_date(0, lo, hi, grace_days) == 1.0


def test_score_release_date_negative_release_ts() -> None:
    """Pre-epoch (negative) timestamps should still decay correctly."""
    lo = 0.0
    hi = 100.0 * _DAY_SECONDS
    grace_days = 100.0
    release_ts = -50 * _DAY_SECONDS  # 50 days before epoch
    assert _score_release_date(release_ts, lo, hi, grace_days) == pytest.approx(0.5)


def test_score_release_date_point_range_at_point_returns_one() -> None:
    """Point range (lo == hi): movie exactly at lo should return 1.0."""
    point = _utc_ts("2015-06-15")
    assert _score_release_date(point, point, point, 730.0) == 1.0


def test_score_release_date_point_range_off_by_one_day() -> None:
    """Point range (lo == hi): 1 day off should decay by 1/grace_days."""
    point = _utc_ts("2015-06-15")
    grace_days = 730.0
    release_ts = point + _DAY_SECONDS
    assert _score_release_date(release_ts, point, point, grace_days) == pytest.approx(1.0 - 1.0 / 730.0)


def test_score_release_date_small_grace_steep_decay() -> None:
    """Very small grace_days (1 day) should have steep decay."""
    lo = _utc_ts("2020-01-01")
    hi = _utc_ts("2020-12-31")
    grace_days = 1.0
    release_ts = lo - _DAY_SECONDS  # 1 day outside
    assert _score_release_date(release_ts, lo, hi, grace_days) == pytest.approx(0.0)
    release_ts_half = lo - (_DAY_SECONDS * 0.5)  # half day outside
    assert _score_release_date(release_ts_half, lo, hi, grace_days) == pytest.approx(0.5)


def test_score_release_date_symmetry_below_lo_and_above_hi() -> None:
    """Same distance below lo and above hi should produce identical scores."""
    lo = _utc_ts("2000-01-01")
    hi = _utc_ts("2010-01-01")
    grace_days = 1000.0
    distance = 200 * _DAY_SECONDS
    score_below = _score_release_date(lo - distance, lo, hi, grace_days)
    score_above = _score_release_date(hi + distance, lo, hi, grace_days)
    assert score_below == pytest.approx(score_above)


# ── _score_release_date — Guide Example Verification ──────────────────────


def test_guide_example_80s_movies() -> None:
    """Verify guide table: '80s movies' BETWEEN 1980-01-01 and 1989-12-31."""
    lo, hi, grace_days = _precompute_release_date(
        "1980-01-01", DateMatchOperation.BETWEEN, "1989-12-31",
    )
    assert grace_days == 1825.0  # clamped to max

    one_year_before = lo - (365 * _DAY_SECONDS)
    assert _score_release_date(one_year_before, lo, hi, grace_days) == pytest.approx(0.80, abs=0.01)

    three_years_before = lo - (3 * 365 * _DAY_SECONDS)
    assert _score_release_date(three_years_before, lo, hi, grace_days) == pytest.approx(0.40, abs=0.01)


def test_guide_example_2015_films() -> None:
    """Verify guide table: '2015 films' EXACT 2015-01-01 → grace=730."""
    lo, hi, grace_days = _precompute_release_date(
        "2015-01-01", DateMatchOperation.EXACT, None,
    )
    assert lo == hi == _utc_ts("2015-01-01")
    assert grace_days == 730.0

    one_year_out = lo - (365 * _DAY_SECONDS)
    assert _score_release_date(one_year_out, lo, hi, grace_days) == pytest.approx(0.50, abs=0.01)

    three_years_out = lo - (3 * 365 * _DAY_SECONDS)
    assert _score_release_date(three_years_out, lo, hi, grace_days) == pytest.approx(0.0)


def test_guide_example_after_1980() -> None:
    """Verify guide table: 'after 1980' AFTER 1980-01-01 → grace=1095."""
    lo, hi, grace_days = _precompute_release_date(
        "1980-01-01", DateMatchOperation.AFTER, None,
    )
    assert lo == _utc_ts("1980-01-01")
    assert math.isinf(hi) and hi > 0
    assert grace_days == 1095.0

    one_year_before = lo - (365 * _DAY_SECONDS)
    assert _score_release_date(one_year_before, lo, hi, grace_days) == pytest.approx(0.67, abs=0.01)

    three_years_before = lo - (3 * 365 * _DAY_SECONDS)
    assert _score_release_date(three_years_before, lo, hi, grace_days) == pytest.approx(0.0)


# ── _precompute_release_date — Additional Coverage ────────────────────────


def test_precompute_release_date_exact_explicitly() -> None:
    """EXACT mode: lo == hi == first_ts, grace == 730."""
    lo, hi, grace_days = _precompute_release_date("2020-06-15", DateMatchOperation.EXACT, None)
    expected = _utc_ts("2020-06-15")
    assert lo == expected
    assert hi == expected
    assert grace_days == 730.0


def test_precompute_release_date_between_identical_dates() -> None:
    """BETWEEN with identical dates: range_width=0 → grace clamped to minimum 365."""
    lo, hi, grace_days = _precompute_release_date(
        "2020-06-15", DateMatchOperation.BETWEEN, "2020-06-15",
    )
    expected = _utc_ts("2020-06-15")
    assert lo == expected
    assert hi == expected
    assert grace_days == 365.0


def test_precompute_release_date_between_natural_order() -> None:
    """BETWEEN with first < second should assign lo/hi correctly."""
    lo, hi, grace_days = _precompute_release_date(
        "2000-01-01", DateMatchOperation.BETWEEN, "2010-01-01",
    )
    assert lo == _utc_ts("2000-01-01")
    assert hi == _utc_ts("2010-01-01")


def test_precompute_release_date_between_grace_at_minimum_boundary() -> None:
    """BETWEEN where range_width * 0.5 == 365 → grace == 365 (at lower clamp)."""
    # range_width = 730 days (no leap year) → 730 * 0.5 = 365 = minimum
    lo, hi, grace_days = _precompute_release_date(
        "2021-01-01", DateMatchOperation.BETWEEN, "2023-01-01",
    )
    range_days = (hi - lo) / _DAY_SECONDS
    assert range_days == 730.0
    assert grace_days == 365.0


def test_precompute_release_date_between_grace_at_maximum_boundary() -> None:
    """BETWEEN where range_width * 0.5 == 1825 → grace == 1825 (at upper clamp)."""
    # range_width = 3650 days → 3650 * 0.5 = 1825 = maximum
    lo, hi, grace_days = _precompute_release_date(
        "2010-01-01", DateMatchOperation.BETWEEN, "2020-01-01",
    )
    range_days = (hi - lo) / _DAY_SECONDS
    raw_grace = range_days * 0.5
    assert grace_days == max(365.0, min(raw_grace, 1825.0))
    assert grace_days == 1825.0


def test_precompute_release_date_between_grace_mid_range() -> None:
    """BETWEEN with 4-year range → grace = 730 (within clamp bounds)."""
    lo, hi, grace_days = _precompute_release_date(
        "2016-01-01", DateMatchOperation.BETWEEN, "2020-01-01",
    )
    range_days = (hi - lo) / _DAY_SECONDS
    expected_grace = range_days * 0.5
    assert 365.0 < expected_grace < 1825.0  # in mid-range
    assert grace_days == pytest.approx(expected_grace, abs=1.0)


def test_precompute_release_date_after_ignores_second_date() -> None:
    """AFTER mode should ignore second_date even if provided."""
    lo1, hi1, grace1 = _precompute_release_date("2010-05-20", DateMatchOperation.AFTER, None)
    lo2, hi2, grace2 = _precompute_release_date("2010-05-20", DateMatchOperation.AFTER, "2020-01-01")
    assert lo1 == lo2
    assert hi1 == hi2
    assert grace1 == grace2


def test_precompute_release_date_before_ignores_second_date() -> None:
    """BEFORE mode should ignore second_date even if provided."""
    lo1, hi1, grace1 = _precompute_release_date("2010-05-20", DateMatchOperation.BEFORE, None)
    lo2, hi2, grace2 = _precompute_release_date("2010-05-20", DateMatchOperation.BEFORE, "2020-01-01")
    assert lo1 == lo2
    assert hi1 == hi2
    assert grace1 == grace2


def test_precompute_duration_between_sorts_bounds() -> None:
    """Between mode should normalize numerical bounds to (lo, hi)."""
    lo, hi = _precompute_duration(120.0, NumericalMatchOperation.BETWEEN, 90.0)
    assert lo == 90.0
    assert hi == 120.0


def test_precompute_duration_greater_than_builds_open_ended_upper_range() -> None:
    """Greater-than mode should set hi to +inf."""
    lo, hi = _precompute_duration(100.0, NumericalMatchOperation.GREATER_THAN, None)
    assert lo == 100.0
    assert math.isinf(hi) and hi > 0


def test_precompute_duration_less_than_builds_open_ended_lower_range() -> None:
    """Less-than mode should set lo to -inf."""
    lo, hi = _precompute_duration(100.0, NumericalMatchOperation.LESS_THAN, None)
    assert math.isinf(lo) and lo < 0
    assert hi == 100.0


def test_precompute_duration_between_without_second_value_falls_back_to_exact() -> None:
    """Between with missing upper/lower value should fall back to exact."""
    lo, hi = _precompute_duration(100.0, NumericalMatchOperation.BETWEEN, None)
    assert lo == 100.0
    assert hi == 100.0


def test_precompute_duration_unknown_operation_falls_back_to_exact() -> None:
    """Unknown operations should use exact numeric bounds."""
    lo, hi = _precompute_duration(100.0, "unexpected", 200.0)
    assert lo == 100.0
    assert hi == 100.0


# ── Additional _precompute_duration tests ─────────────────────────────────


def test_precompute_duration_between_equal_values() -> None:
    """BETWEEN with equal first and second values should produce lo == hi."""
    lo, hi = _precompute_duration(100.0, NumericalMatchOperation.BETWEEN, 100.0)
    assert lo == 100.0
    assert hi == 100.0


def test_precompute_duration_exact_mode() -> None:
    """EXACT should set lo == hi == first_value."""
    lo, hi = _precompute_duration(42.0, NumericalMatchOperation.EXACT, None)
    assert lo == 42.0
    assert hi == 42.0


def test_precompute_duration_exact_ignores_second_value() -> None:
    """EXACT should produce lo == hi == first_value regardless of second_value."""
    lo, hi = _precompute_duration(42.0, NumericalMatchOperation.EXACT, 999.0)
    assert lo == 42.0
    assert hi == 42.0


def test_precompute_duration_greater_than_ignores_second_value() -> None:
    """GREATER_THAN should use first_value as lo and inf as hi, ignoring second_value."""
    lo, hi = _precompute_duration(50.0, NumericalMatchOperation.GREATER_THAN, 999.0)
    assert lo == 50.0
    assert math.isinf(hi) and hi > 0


def test_precompute_duration_less_than_ignores_second_value() -> None:
    """LESS_THAN should use first_value as hi and -inf as lo, ignoring second_value."""
    lo, hi = _precompute_duration(50.0, NumericalMatchOperation.LESS_THAN, 999.0)
    assert math.isinf(lo) and lo < 0
    assert hi == 50.0


@pytest.mark.parametrize("op", list(NumericalMatchOperation))
def test_precompute_duration_zero_first_value(op: NumericalMatchOperation) -> None:
    """Zero first_value should not cause errors for any operation."""
    lo, hi = _precompute_duration(0.0, op, 10.0)
    assert isinstance(lo, float)
    assert isinstance(hi, float)


@pytest.mark.parametrize("op", list(NumericalMatchOperation))
def test_precompute_duration_negative_first_value(op: NumericalMatchOperation) -> None:
    """Negative first_value should be handled without errors."""
    lo, hi = _precompute_duration(-50.0, op, 10.0)
    assert isinstance(lo, float)
    assert isinstance(hi, float)


def test_precompute_duration_very_large_first_value() -> None:
    """Very large first_value should not cause overflow or errors."""
    lo, hi = _precompute_duration(1e9, NumericalMatchOperation.EXACT, None)
    assert lo == 1e9
    assert hi == 1e9


def test_precompute_duration_between_then_score_integration() -> None:
    """End-to-end: precompute BETWEEN then score should produce expected result."""
    lo, hi = _precompute_duration(90.0, NumericalMatchOperation.BETWEEN, 120.0)
    assert _score_duration(105, lo, hi) == 1.0
    assert _score_duration(80, lo, hi) == pytest.approx(2 / 3)
    assert _score_duration(60, lo, hi) == 0.0


def test_precompute_duration_greater_than_then_score_integration() -> None:
    """End-to-end: precompute GREATER_THAN then score."""
    lo, hi = _precompute_duration(100.0, NumericalMatchOperation.GREATER_THAN, None)
    assert _score_duration(150, lo, hi) == 1.0
    assert _score_duration(100, lo, hi) == 1.0
    assert _score_duration(90, lo, hi) == pytest.approx(2 / 3)
    assert _score_duration(70, lo, hi) == 0.0


def test_precompute_duration_less_than_then_score_integration() -> None:
    """End-to-end: precompute LESS_THAN then score."""
    lo, hi = _precompute_duration(100.0, NumericalMatchOperation.LESS_THAN, None)
    assert _score_duration(50, lo, hi) == 1.0
    assert _score_duration(100, lo, hi) == 1.0
    assert _score_duration(110, lo, hi) == pytest.approx(2 / 3)
    assert _score_duration(130, lo, hi) == 0.0


def test_precompute_duration_exact_then_score_integration() -> None:
    """End-to-end: precompute EXACT then score."""
    lo, hi = _precompute_duration(100.0, NumericalMatchOperation.EXACT, None)
    assert _score_duration(100, lo, hi) == 1.0
    assert _score_duration(110, lo, hi) == pytest.approx(2 / 3)
    assert _score_duration(130, lo, hi) == 0.0
    assert _score_duration(85, lo, hi) == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("rating", "match_operation", "expected_target", "expected_lo", "expected_hi"),
    [
        ("pg-13", RatingMatchOperation.EXACT, MaturityRating.PG_13, 3, 3),
        ("pg", RatingMatchOperation.GREATER_THAN, MaturityRating.PG, 3, 5),
        ("pg-13", RatingMatchOperation.LESS_THAN, MaturityRating.PG_13, 1, 2),
        ("r", RatingMatchOperation.GREATER_THAN_OR_EQUAL, MaturityRating.R, 4, 5),
        ("r", RatingMatchOperation.LESS_THAN_OR_EQUAL, MaturityRating.R, 1, 4),
        ("g", RatingMatchOperation.LESS_THAN, MaturityRating.G, 1, 0),
        ("nc-17", RatingMatchOperation.GREATER_THAN, MaturityRating.NC_17, 6, 5),
        ("g", RatingMatchOperation.GREATER_THAN_OR_EQUAL, MaturityRating.G, 1, 5),
        ("nc-17", RatingMatchOperation.LESS_THAN_OR_EQUAL, MaturityRating.NC_17, 1, 5),
        ("unrated", RatingMatchOperation.EXACT, MaturityRating.UNRATED, 999, 999),
    ],
)
def test_precompute_maturity_resolves_rating_and_range_for_each_operation(
    rating: str,
    match_operation: RatingMatchOperation,
    expected_target: MaturityRating,
    expected_lo: int,
    expected_hi: int,
) -> None:
    """Maturity precompute should map operation strings to expected rank ranges."""
    target, lo, hi = _precompute_maturity(rating, match_operation)
    assert target is expected_target
    assert lo == expected_lo
    assert hi == expected_hi


def test_precompute_maturity_invalid_rating_raises_value_error() -> None:
    """Unknown maturity labels should raise ValueError from enum resolution."""
    with pytest.raises(ValueError):
        _precompute_maturity("x-rated", RatingMatchOperation.EXACT)


@pytest.mark.parametrize(
    ("target", "match_operation", "movie_rank"),
    list(product(_RATED_MATURITY_TARGETS, _MATURITY_MATCH_OPERATIONS, _RATED_MATURITY_RANKS + [MaturityRating.UNRATED.maturity_rank])),
)
def test_score_maturity_rating_matches_guide_matrix(
    target: MaturityRating,
    match_operation: RatingMatchOperation,
    movie_rank: int,
) -> None:
    """Guide-derived expected scores should match scorer behavior across the full rated matrix."""
    _, lo, hi = _precompute_maturity(target.value, match_operation)
    expected = _expected_maturity_score(movie_rank, target, match_operation, lo, hi)

    assert _score_maturity_rating(movie_rank, target, lo, hi) == expected


def test_precompute_watch_providers_resolves_names_and_builds_key_sets() -> None:
    """Provider names should resolve correctly and produce expected key sets."""
    exclude_key_set, include_any_keys, include_desired_keys, access_type_id = _precompute_watch_providers(
        should_include=["netflix", "hulu"],
        should_exclude=["amazon"],
        preferred_access_type=StreamingAccessType.SUBSCRIPTION,
    )

    amazon_all = {
        create_watch_provider_offering_key(pid, mid)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.AMAZON]
        for mid in _ALL_METHOD_IDS
    }
    assert exclude_key_set == amazon_all

    netflix_all = {
        create_watch_provider_offering_key(pid, mid)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
        for mid in _ALL_METHOD_IDS
    }
    hulu_all = {
        create_watch_provider_offering_key(pid, mid)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.HULU]
        for mid in _ALL_METHOD_IDS
    }
    assert include_any_keys == netflix_all | hulu_all
    assert include_desired_keys == {
        create_watch_provider_offering_key(pid, StreamingAccessType.SUBSCRIPTION.type_id)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX] + STREAMING_PROVIDER_MAP[StreamingService.HULU]
    }
    assert access_type_id is None  # access_type_id only set when no include list


def test_precompute_watch_providers_unknown_names_are_ignored() -> None:
    """Unrecognized provider names should be skipped cleanly."""
    exclude_key_set, include_any_keys, include_desired_keys, access_type_id = _precompute_watch_providers(
        should_include=["Not A Real Service"],
        should_exclude=["Also Fake"],
        preferred_access_type=None,
    )
    assert exclude_key_set == set()
    assert include_any_keys == set()
    assert include_desired_keys == set()
    assert access_type_id is None


def test_precompute_watch_providers_without_access_type_builds_include_any_only() -> None:
    """No preferred access type: include_any_keys is populated; include_desired_keys is empty."""
    _, include_any_keys, include_desired_keys, access_type_id = _precompute_watch_providers(
        should_include=["netflix"],
        should_exclude=[],
        preferred_access_type=None,
    )
    netflix_all = {
        create_watch_provider_offering_key(pid, mid)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
        for mid in _ALL_METHOD_IDS
    }
    assert include_any_keys == netflix_all
    assert include_desired_keys == set()
    assert access_type_id is None


def test_precompute_watch_providers_duplicate_includes_are_deduplicated_in_sets() -> None:
    """Duplicate include providers should not duplicate keys in include sets."""
    _, include_any_keys, include_desired_keys, _ = _precompute_watch_providers(
        should_include=["netflix", "netflix"],
        should_exclude=[],
        preferred_access_type=StreamingAccessType.RENT,
    )
    netflix_all = {
        create_watch_provider_offering_key(pid, mid)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
        for mid in _ALL_METHOD_IDS
    }
    netflix_rent = {
        create_watch_provider_offering_key(pid, StreamingAccessType.RENT.type_id)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
    }
    assert include_any_keys == netflix_all
    assert include_desired_keys == netflix_rent


def test_precompute_watch_providers_duplicate_excludes_are_merged_into_flat_set() -> None:
    """Duplicate excluded providers merge into a single flat key set."""
    exclude_key_set, _, _, _ = _precompute_watch_providers(
        should_include=[],
        should_exclude=["netflix", "netflix"],
        preferred_access_type=None,
    )
    netflix_all = {
        create_watch_provider_offering_key(pid, mid)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
        for mid in _ALL_METHOD_IDS
    }
    assert exclude_key_set == netflix_all


def test_precompute_watch_providers_access_type_without_include_sets_access_type_id() -> None:
    """When access_type is set but include list is empty, access_type_id is returned."""
    _, include_any_keys, include_desired_keys, access_type_id = _precompute_watch_providers(
        should_include=[],
        should_exclude=[],
        preferred_access_type=StreamingAccessType.SUBSCRIPTION,
    )
    assert include_any_keys == set()
    assert include_desired_keys == set()
    assert access_type_id == StreamingAccessType.SUBSCRIPTION.type_id


# ── _score_watch_providers — extended edge-case coverage ──────────────────


def test_score_wp_empty_offer_keys_list_returns_zero() -> None:
    """Empty list (not None) means the movie has no offerings → 0.0."""
    assert _score_watch_providers(
        offer_keys=[],
        exclude_key_set=set(),
        include_any_keys=set(),
        include_desired_keys=set(),
        access_type_id=None,
    ) == 0.0


def test_score_wp_include_only_no_access_type_match_returns_one() -> None:
    """Include-only branch: movie matches an included provider → 1.0."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    assert _score_watch_providers(
        offer_keys=[netflix_sub],
        exclude_key_set=set(),
        include_any_keys=_all_method_keys(_NETFLIX_ID),
        include_desired_keys=set(),
        access_type_id=None,
    ) == 1.0


def test_score_wp_include_only_no_access_type_no_match_returns_zero() -> None:
    """Include-only branch: movie has a different provider → 0.0."""
    hulu_sub = _key(_HULU_ID, StreamingAccessType.SUBSCRIPTION)
    assert _score_watch_providers(
        offer_keys=[hulu_sub],
        exclude_key_set=set(),
        include_any_keys=_all_method_keys(_NETFLIX_ID),
        include_desired_keys=set(),
        access_type_id=None,
    ) == 0.0


def test_score_wp_exclude_removes_keys_before_access_type_check() -> None:
    """Key matching access type is excluded first → 0.0."""
    netflix_rent = _key(_NETFLIX_ID, StreamingAccessType.RENT)
    assert _score_watch_providers(
        offer_keys=[netflix_rent],
        exclude_key_set={netflix_rent},
        include_any_keys=set(),
        include_desired_keys=set(),
        access_type_id=StreamingAccessType.RENT.type_id,
    ) == 0.0


def test_score_wp_exclude_partial_removal_leaves_matching_keys() -> None:
    """Movie has 2 providers; 1 is excluded, remaining matches include → 1.0."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    hulu_sub = _key(_HULU_ID, StreamingAccessType.SUBSCRIPTION)
    assert _score_watch_providers(
        offer_keys=[netflix_sub, hulu_sub],
        exclude_key_set=_all_method_keys(_HULU_ID),
        include_any_keys=_all_method_keys(_NETFLIX_ID),
        include_desired_keys=set(),
        access_type_id=None,
    ) == 1.0


def test_score_wp_both_set_movie_has_multiple_providers_best_match_wins() -> None:
    """Movie has keys from 2 included services; one at desired method → 1.0."""
    netflix_rent = _key(_NETFLIX_ID, StreamingAccessType.RENT)
    hulu_sub = _key(_HULU_ID, StreamingAccessType.SUBSCRIPTION)

    include_any = _all_method_keys(_NETFLIX_ID) | _all_method_keys(_HULU_ID)
    include_desired = {
        _key(_NETFLIX_ID, StreamingAccessType.RENT),
        _key(_HULU_ID, StreamingAccessType.RENT),
    }
    assert _score_watch_providers(
        offer_keys=[netflix_rent, hulu_sub],
        exclude_key_set=set(),
        include_any_keys=include_any,
        include_desired_keys=include_desired,
        access_type_id=None,
    ) == 1.0


def test_score_wp_both_set_no_desired_but_multiple_any_matches_returns_half() -> None:
    """Movie matches multiple included providers but none at desired method → 0.5."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    hulu_sub = _key(_HULU_ID, StreamingAccessType.SUBSCRIPTION)

    include_any = _all_method_keys(_NETFLIX_ID) | _all_method_keys(_HULU_ID)
    include_desired = {
        _key(_NETFLIX_ID, StreamingAccessType.RENT),
        _key(_HULU_ID, StreamingAccessType.RENT),
    }
    assert _score_watch_providers(
        offer_keys=[netflix_sub, hulu_sub],
        exclude_key_set=set(),
        include_any_keys=include_any,
        include_desired_keys=include_desired,
        access_type_id=None,
    ) == 0.5


def test_score_wp_exclude_only_with_remaining_keys_returns_one() -> None:
    """No include/access prefs; exclude removes some keys, others remain → 1.0."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    hulu_sub = _key(_HULU_ID, StreamingAccessType.SUBSCRIPTION)
    assert _score_watch_providers(
        offer_keys=[netflix_sub, hulu_sub],
        exclude_key_set=_all_method_keys(_HULU_ID),
        include_any_keys=set(),
        include_desired_keys=set(),
        access_type_id=None,
    ) == 1.0


def test_score_wp_exclude_only_removes_all_keys_returns_zero() -> None:
    """No include/access prefs; exclude removes ALL movie keys → 0.0."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    assert _score_watch_providers(
        offer_keys=[netflix_sub],
        exclude_key_set=_all_method_keys(_NETFLIX_ID),
        include_any_keys=set(),
        include_desired_keys=set(),
        access_type_id=None,
    ) == 0.0


def test_score_wp_access_type_only_with_multiple_keys_different_types() -> None:
    """Access-type-only branch: mixed types, one matches → 1.0."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    hulu_rent = _key(_HULU_ID, StreamingAccessType.RENT)
    assert _score_watch_providers(
        offer_keys=[netflix_sub, hulu_rent],
        exclude_key_set=set(),
        include_any_keys=set(),
        include_desired_keys=set(),
        access_type_id=StreamingAccessType.RENT.type_id,
    ) == 1.0


def test_score_wp_access_type_only_exclude_removes_only_matching_type() -> None:
    """Access-type set; matching-type key excluded, remaining has wrong type → 0.0."""
    netflix_rent = _key(_NETFLIX_ID, StreamingAccessType.RENT)
    hulu_sub = _key(_HULU_ID, StreamingAccessType.SUBSCRIPTION)
    assert _score_watch_providers(
        offer_keys=[netflix_rent, hulu_sub],
        exclude_key_set={netflix_rent},
        include_any_keys=set(),
        include_desired_keys=set(),
        access_type_id=StreamingAccessType.RENT.type_id,
    ) == 0.0


def test_score_wp_no_preferences_at_all_with_keys_returns_one() -> None:
    """All params empty/None, movie has keys → 1.0."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    assert _score_watch_providers(
        offer_keys=[netflix_sub],
        exclude_key_set=set(),
        include_any_keys=set(),
        include_desired_keys=set(),
        access_type_id=None,
    ) == 1.0


def test_score_wp_no_preferences_at_all_without_keys_returns_zero() -> None:
    """All params empty/None, movie has empty list → 0.0."""
    assert _score_watch_providers(
        offer_keys=[],
        exclude_key_set=set(),
        include_any_keys=set(),
        include_desired_keys=set(),
        access_type_id=None,
    ) == 0.0


def test_score_wp_duplicate_offer_keys_handled_correctly() -> None:
    """Duplicate keys in offer_keys should be deduplicated via set()."""
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    result_with_dupes = _score_watch_providers(
        offer_keys=[netflix_sub, netflix_sub, netflix_sub],
        exclude_key_set=set(),
        include_any_keys=_all_method_keys(_NETFLIX_ID),
        include_desired_keys=set(),
        access_type_id=None,
    )
    result_without_dupes = _score_watch_providers(
        offer_keys=[netflix_sub],
        exclude_key_set=set(),
        include_any_keys=_all_method_keys(_NETFLIX_ID),
        include_desired_keys=set(),
        access_type_id=None,
    )
    assert result_with_dupes == result_without_dupes == 1.0


def test_score_wp_include_desired_nonempty_but_include_any_empty() -> None:
    """Edge: include_desired has keys but include_any is empty.

    Should fall through to access_type or exclude branch, not the "both" branch,
    since the "both" branch requires include_any_keys to be truthy.
    """
    netflix_rent = _key(_NETFLIX_ID, StreamingAccessType.RENT)
    # include_desired non-empty, include_any empty → "both" branch is skipped.
    # With access_type_id set, should use the access-type-only branch.
    assert _score_watch_providers(
        offer_keys=[netflix_rent],
        exclude_key_set=set(),
        include_any_keys=set(),
        include_desired_keys={netflix_rent},
        access_type_id=StreamingAccessType.RENT.type_id,
    ) == 1.0

    # Same but movie has wrong access type → 0.0 via access-type-only branch.
    netflix_sub = _key(_NETFLIX_ID, StreamingAccessType.SUBSCRIPTION)
    assert _score_watch_providers(
        offer_keys=[netflix_sub],
        exclude_key_set=set(),
        include_any_keys=set(),
        include_desired_keys={netflix_rent},
        access_type_id=StreamingAccessType.RENT.type_id,
    ) == 0.0


# ── _precompute_watch_providers — extended edge-case coverage ─────────────


def test_precompute_wp_multi_provider_service_generates_keys_for_all_ids() -> None:
    """Netflix has 3 provider IDs; include should produce 3×3 = 9 any-keys."""
    _, include_any_keys, _, _ = _precompute_watch_providers(
        should_include=["netflix"],
        should_exclude=[],
        preferred_access_type=None,
    )
    netflix_pids = STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
    assert len(netflix_pids) == 3
    expected = {
        create_watch_provider_offering_key(pid, mid)
        for pid in netflix_pids
        for mid in _ALL_METHOD_IDS
    }
    assert include_any_keys == expected
    assert len(include_any_keys) == 9


def test_precompute_wp_multi_provider_service_exclude_generates_keys_for_all_ids() -> None:
    """Amazon has 4 provider IDs; exclude should produce 4×3 = 12 keys."""
    exclude_key_set, _, _, _ = _precompute_watch_providers(
        should_include=[],
        should_exclude=["amazon"],
        preferred_access_type=None,
    )
    amazon_pids = STREAMING_PROVIDER_MAP[StreamingService.AMAZON]
    assert len(amazon_pids) == 4
    expected = {
        create_watch_provider_offering_key(pid, mid)
        for pid in amazon_pids
        for mid in _ALL_METHOD_IDS
    }
    assert exclude_key_set == expected
    assert len(exclude_key_set) == 12


def test_precompute_wp_include_desired_keys_uses_only_preferred_method() -> None:
    """Include amazon with SUBSCRIPTION → include_desired has exactly 4 keys."""
    _, _, include_desired_keys, _ = _precompute_watch_providers(
        should_include=["amazon"],
        should_exclude=[],
        preferred_access_type=StreamingAccessType.SUBSCRIPTION,
    )
    amazon_pids = STREAMING_PROVIDER_MAP[StreamingService.AMAZON]
    expected = {
        create_watch_provider_offering_key(pid, StreamingAccessType.SUBSCRIPTION.type_id)
        for pid in amazon_pids
    }
    assert include_desired_keys == expected
    assert len(include_desired_keys) == 4


def test_precompute_wp_same_service_in_both_include_and_exclude() -> None:
    """Netflix in both include and exclude → both sets populated."""
    exclude_key_set, include_any_keys, _, _ = _precompute_watch_providers(
        should_include=["netflix"],
        should_exclude=["netflix"],
        preferred_access_type=None,
    )
    netflix_all = {
        create_watch_provider_offering_key(pid, mid)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
        for mid in _ALL_METHOD_IDS
    }
    assert exclude_key_set == netflix_all
    assert include_any_keys == netflix_all


def test_precompute_wp_all_empty_inputs() -> None:
    """Empty include, empty exclude, None access type → all empty."""
    exclude_key_set, include_any_keys, include_desired_keys, access_type_id = _precompute_watch_providers(
        should_include=[],
        should_exclude=[],
        preferred_access_type=None,
    )
    assert exclude_key_set == set()
    assert include_any_keys == set()
    assert include_desired_keys == set()
    assert access_type_id is None


def test_precompute_wp_exclude_only_no_include_no_access() -> None:
    """Exclude hulu, no include, no access → exclude set populated, rest empty."""
    exclude_key_set, include_any_keys, include_desired_keys, access_type_id = _precompute_watch_providers(
        should_include=[],
        should_exclude=["hulu"],
        preferred_access_type=None,
    )
    assert exclude_key_set == _all_method_keys(_HULU_ID)
    assert include_any_keys == set()
    assert include_desired_keys == set()
    assert access_type_id is None


def test_precompute_wp_multiple_services_in_include_keys_are_unioned() -> None:
    """Include [netflix, hulu] → keys are union of both services."""
    _, include_any_keys, _, _ = _precompute_watch_providers(
        should_include=["netflix", "hulu"],
        should_exclude=[],
        preferred_access_type=None,
    )
    netflix_keys = {
        create_watch_provider_offering_key(pid, mid)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
        for mid in _ALL_METHOD_IDS
    }
    hulu_keys = _all_method_keys(_HULU_ID)
    assert include_any_keys == netflix_keys | hulu_keys


def test_precompute_wp_multiple_services_in_exclude_keys_are_unioned() -> None:
    """Exclude [netflix, hulu] → keys are union of both services."""
    exclude_key_set, _, _, _ = _precompute_watch_providers(
        should_include=[],
        should_exclude=["netflix", "hulu"],
        preferred_access_type=None,
    )
    netflix_keys = {
        create_watch_provider_offering_key(pid, mid)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
        for mid in _ALL_METHOD_IDS
    }
    hulu_keys = _all_method_keys(_HULU_ID)
    assert exclude_key_set == netflix_keys | hulu_keys


def test_precompute_wp_access_type_with_empty_include_sets_access_type_id() -> None:
    """Empty include + RENT access type → access_type_id = RENT.type_id."""
    _, _, _, access_type_id = _precompute_watch_providers(
        should_include=[],
        should_exclude=[],
        preferred_access_type=StreamingAccessType.RENT,
    )
    assert access_type_id == StreamingAccessType.RENT.type_id


def test_precompute_wp_access_type_with_nonempty_include_clears_access_type_id() -> None:
    """Include netflix + RENT access type → access_type_id is None."""
    _, _, include_desired_keys, access_type_id = _precompute_watch_providers(
        should_include=["netflix"],
        should_exclude=[],
        preferred_access_type=StreamingAccessType.RENT,
    )
    assert access_type_id is None
    # But include_desired_keys should still be populated
    expected_desired = {
        create_watch_provider_offering_key(pid, StreamingAccessType.RENT.type_id)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
    }
    assert include_desired_keys == expected_desired


def test_precompute_wp_mixed_valid_and_invalid_names_in_include() -> None:
    """["netflix", "not_real"] → only netflix keys generated."""
    _, include_any_keys, _, _ = _precompute_watch_providers(
        should_include=["netflix", "not_real"],
        should_exclude=[],
        preferred_access_type=None,
    )
    netflix_keys = {
        create_watch_provider_offering_key(pid, mid)
        for pid in STREAMING_PROVIDER_MAP[StreamingService.NETFLIX]
        for mid in _ALL_METHOD_IDS
    }
    assert include_any_keys == netflix_keys


def test_precompute_wp_mixed_valid_and_invalid_names_in_exclude() -> None:
    """["not_real", "hulu"] → only hulu keys generated."""
    exclude_key_set, _, _, _ = _precompute_watch_providers(
        should_include=[],
        should_exclude=["not_real", "hulu"],
        preferred_access_type=None,
    )
    assert exclude_key_set == _all_method_keys(_HULU_ID)
