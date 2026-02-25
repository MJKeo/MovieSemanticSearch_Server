"""Unit tests for build_qdrant_filter in db.vector_search."""

import pytest

pytest.importorskip("qdrant_client")

import implementation.classes.enums as enums_module
from implementation.classes.enums import Genre
from implementation.classes.schemas import MetadataFilters

# db.vector_search imports MetadataFilters from enums; patch to match runtime expectation.
if not hasattr(enums_module, "MetadataFilters"):
    setattr(enums_module, "MetadataFilters", MetadataFilters)

from db.vector_search import build_qdrant_filter


def _condition_by_key(q_filter, key: str):
    """Return a must-condition by payload key."""
    assert q_filter is not None
    for condition in q_filter.must:
        if condition.key == key:
            return condition
    raise AssertionError(f"No condition found for key={key}")


def test_all_fields_none_returns_none() -> None:
    """No active metadata filters should produce no Qdrant filter object."""
    assert build_qdrant_filter(MetadataFilters()) is None


def test_release_range_both_bounds() -> None:
    """Release date min/max should map to a single release_ts range condition."""
    q_filter = build_qdrant_filter(MetadataFilters(min_release_ts=100, max_release_ts=200))
    assert q_filter is not None
    assert len(q_filter.must) == 1
    condition = _condition_by_key(q_filter, "release_ts")
    assert condition.range is not None
    assert condition.range.gte == 100
    assert condition.range.lte == 200


def test_runtime_range_both_bounds() -> None:
    """Runtime min/max should map to runtime_minutes range."""
    q_filter = build_qdrant_filter(MetadataFilters(min_runtime=90, max_runtime=150))
    assert q_filter is not None
    assert len(q_filter.must) == 1
    condition = _condition_by_key(q_filter, "runtime_minutes")
    assert condition.range is not None
    assert condition.range.gte == 90
    assert condition.range.lte == 150


def test_maturity_range_both_bounds() -> None:
    """Maturity rank min/max should map to maturity_rank range."""
    q_filter = build_qdrant_filter(MetadataFilters(min_maturity_rank=1, max_maturity_rank=4))
    assert q_filter is not None
    assert len(q_filter.must) == 1
    condition = _condition_by_key(q_filter, "maturity_rank")
    assert condition.range is not None
    assert condition.range.gte == 1
    assert condition.range.lte == 4


def test_genres_only_uses_genre_id_values() -> None:
    """Genres should map to MatchAny genre_ids using integer genre_id values."""
    q_filter = build_qdrant_filter(MetadataFilters(genres=[Genre.ACTION, Genre.DRAMA]))
    assert q_filter is not None
    assert len(q_filter.must) == 1
    condition = _condition_by_key(q_filter, "genre_ids")
    assert condition.match is not None
    assert condition.match.any == [Genre.ACTION.genre_id, Genre.DRAMA.genre_id]
    assert all(isinstance(v, int) for v in condition.match.any)


def test_watch_offer_keys_only() -> None:
    """Watch offer keys should map to a MatchAny watch_offer_keys condition."""
    q_filter = build_qdrant_filter(MetadataFilters(watch_offer_keys=[101, 202]))
    assert q_filter is not None
    assert len(q_filter.must) == 1
    condition = _condition_by_key(q_filter, "watch_offer_keys")
    assert condition.match is not None
    assert condition.match.any == [101, 202]


def test_all_filters_set() -> None:
    """All active filters should be ANDed together in must with five conditions."""
    q_filter = build_qdrant_filter(
        MetadataFilters(
            min_release_ts=100,
            max_release_ts=200,
            min_runtime=80,
            max_runtime=180,
            min_maturity_rank=1,
            max_maturity_rank=5,
            genres=[Genre.ACTION],
            watch_offer_keys=[1001],
        )
    )
    assert q_filter is not None
    assert len(q_filter.must) == 5
    assert [c.key for c in q_filter.must] == [
        "release_ts",
        "runtime_minutes",
        "maturity_rank",
        "genre_ids",
        "watch_offer_keys",
    ]


@pytest.mark.parametrize(
    ("kwargs", "key", "expected_gte", "expected_lte"),
    [
        ({"min_release_ts": 100}, "release_ts", 100, None),
        ({"max_release_ts": 200}, "release_ts", None, 200),
        ({"min_runtime": 80}, "runtime_minutes", 80, None),
        ({"max_runtime": 180}, "runtime_minutes", None, 180),
        ({"min_maturity_rank": 1}, "maturity_rank", 1, None),
        ({"max_maturity_rank": 5}, "maturity_rank", None, 5),
    ],
)
def test_single_bound_ranges(kwargs: dict, key: str, expected_gte: int | None, expected_lte: int | None) -> None:
    """A one-sided range should retain the set bound and leave the other as None."""
    q_filter = build_qdrant_filter(MetadataFilters(**kwargs))
    assert q_filter is not None
    assert len(q_filter.must) == 1
    condition = _condition_by_key(q_filter, key)
    assert condition.range is not None
    assert condition.range.gte == expected_gte
    assert condition.range.lte == expected_lte


def test_zero_values_are_respected() -> None:
    """Numeric zeros are valid active bounds and should produce conditions."""
    q_filter = build_qdrant_filter(
        MetadataFilters(min_release_ts=0, max_release_ts=0, min_runtime=0, max_runtime=0)
    )
    assert q_filter is not None
    assert len(q_filter.must) == 2
    release = _condition_by_key(q_filter, "release_ts")
    runtime = _condition_by_key(q_filter, "runtime_minutes")
    assert release.range.gte == 0 and release.range.lte == 0
    assert runtime.range.gte == 0 and runtime.range.lte == 0


def test_empty_genres_list_only_returns_none() -> None:
    """If the only provided metadata filter is genres=[], no Qdrant filter should be emitted."""
    assert build_qdrant_filter(MetadataFilters(genres=[])) is None


def test_empty_watch_offer_keys_only_returns_none() -> None:
    """If the only provided metadata filter is watch_offer_keys=[], no Qdrant filter should be emitted."""
    assert build_qdrant_filter(MetadataFilters(watch_offer_keys=[])) is None


def test_empty_list_filters_are_ignored_when_other_filters_exist() -> None:
    """Empty list filters should be skipped while other active filters are still materialized."""
    q_filter = build_qdrant_filter(MetadataFilters(min_runtime=95, genres=[], watch_offer_keys=[]))
    assert q_filter is not None
    assert len(q_filter.must) == 1
    assert q_filter.must[0].key == "runtime_minutes"


def test_min_greater_than_max_passes_through() -> None:
    """Range validation is not enforced here; impossible ranges are passed through unchanged."""
    q_filter = build_qdrant_filter(MetadataFilters(min_runtime=200, max_runtime=100))
    assert q_filter is not None
    condition = _condition_by_key(q_filter, "runtime_minutes")
    assert condition.range is not None
    assert condition.range.gte == 200
    assert condition.range.lte == 100


def test_negative_numeric_values_pass_through() -> None:
    """Negative bound values are passed through to Range without sanitization."""
    q_filter = build_qdrant_filter(MetadataFilters(min_release_ts=-10, max_release_ts=-1))
    assert q_filter is not None
    condition = _condition_by_key(q_filter, "release_ts")
    assert condition.range is not None
    assert condition.range.gte == -10
    assert condition.range.lte == -1


def test_condition_order_is_stable() -> None:
    """Conditions should be emitted in deterministic code order for stability."""
    q_filter = build_qdrant_filter(
        MetadataFilters(
            min_release_ts=1,
            min_runtime=2,
            min_maturity_rank=3,
            genres=[Genre.ACTION],
            watch_offer_keys=[9],
        )
    )
    assert q_filter is not None
    assert [c.key for c in q_filter.must] == [
        "release_ts",
        "runtime_minutes",
        "maturity_rank",
        "genre_ids",
        "watch_offer_keys",
    ]


def test_function_is_pure_input_not_modified() -> None:
    """build_qdrant_filter should not mutate incoming MetadataFilters or nested lists."""
    genres = [Genre.ACTION, Genre.DRAMA]
    watch_offer_keys = [101, 202]
    filters = MetadataFilters(genres=genres, watch_offer_keys=watch_offer_keys)

    original_genres = list(genres)
    original_watch_offer_keys = list(watch_offer_keys)

    _ = build_qdrant_filter(filters)

    assert genres == original_genres
    assert watch_offer_keys == original_watch_offer_keys
    assert filters.genres == original_genres
    assert filters.watch_offer_keys == original_watch_offer_keys


def test_distinct_calls_return_distinct_filter_objects() -> None:
    """Each invocation should return fresh model objects without shared state."""
    filters = MetadataFilters(min_release_ts=10)
    first = build_qdrant_filter(filters)
    second = build_qdrant_filter(filters)

    assert first is not None and second is not None
    assert first is not second
    assert first.must is not second.must
    assert first.must[0] is not second.must[0]


def test_genres_and_watch_offer_keys_are_deduped() -> None:
    """Genres and watch_offer_keys should be deduplicated before being sent to Qdrant."""
    q_filter = build_qdrant_filter(
        MetadataFilters(
            genres=[Genre.ACTION, Genre.ACTION, Genre.DRAMA],
            watch_offer_keys=[7, 7, 8],
        )
    )
    assert q_filter is not None

    genre_condition = _condition_by_key(q_filter, "genre_ids")
    watch_condition = _condition_by_key(q_filter, "watch_offer_keys")

    assert genre_condition.match is not None
    assert watch_condition.match is not None
    assert set(genre_condition.match.any) == {Genre.ACTION.genre_id, Genre.DRAMA.genre_id}
    assert len(genre_condition.match.any) == 2
    assert set(watch_condition.match.any) == {7, 8}
    assert len(watch_condition.match.any) == 2


def test_large_lists_supported() -> None:
    """Large list payloads should serialize correctly without truncation."""
    large_watch_keys = list(range(1000, 1300))
    large_genres = [Genre.ACTION, Genre.DRAMA, Genre.COMEDY] * 100

    q_filter = build_qdrant_filter(MetadataFilters(genres=large_genres, watch_offer_keys=large_watch_keys))
    assert q_filter is not None

    genre_condition = _condition_by_key(q_filter, "genre_ids")
    watch_condition = _condition_by_key(q_filter, "watch_offer_keys")

    assert genre_condition.match is not None
    assert watch_condition.match is not None
    assert set(genre_condition.match.any) == {Genre.ACTION.genre_id, Genre.DRAMA.genre_id, Genre.COMEDY.genre_id}
    assert len(watch_condition.match.any) == len(large_watch_keys)
    assert watch_condition.match.any[0] == 1000
    assert watch_condition.match.any[-1] == 1299


@pytest.mark.parametrize(
    ("kwargs", "key", "expected_gte", "expected_lte"),
    [
        ({"min_maturity_rank": 2}, "maturity_rank", 2, None),
        ({"max_maturity_rank": 4}, "maturity_rank", None, 4),
        ({"min_release_ts": 500}, "release_ts", 500, None),
        ({"max_release_ts": 1500}, "release_ts", None, 1500),
    ],
)
def test_single_sided_maturity_and_release_bounds_work(
    kwargs: dict, key: str, expected_gte: int | None, expected_lte: int | None
) -> None:
    """Single-sided release/maturity bounds should generate the expected one-sided ranges."""
    q_filter = build_qdrant_filter(MetadataFilters(**kwargs))
    assert q_filter is not None
    assert len(q_filter.must) == 1
    condition = _condition_by_key(q_filter, key)
    assert condition.range is not None
    assert condition.range.gte == expected_gte
    assert condition.range.lte == expected_lte
