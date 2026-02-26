"""Unit tests for enum conversion and string formatting behavior."""

import pytest

from implementation.classes.enums import MaturityRating, StreamingAccessType


@pytest.mark.parametrize(
    ("raw_method", "expected"),
    [
        ("subscription", StreamingAccessType.SUBSCRIPTION),
        ("buy", StreamingAccessType.BUY),
        ("rent", StreamingAccessType.RENT),
        ("invalid", None),
    ],
)
def test_streaming_access_type_from_string(raw_method: str, expected: StreamingAccessType | None) -> None:
    """StreamingAccessType.from_string should normalize case/spacing and reject unknown values."""
    assert StreamingAccessType.from_string(raw_method) == expected


@pytest.mark.parametrize(
    ("method_id", "expected"),
    [
        (1, StreamingAccessType.SUBSCRIPTION),
        (2, StreamingAccessType.BUY),
        (3, StreamingAccessType.RENT),
        (999, None),
    ],
)
def test_streaming_access_type_from_type_id(method_id: int, expected: StreamingAccessType | None) -> None:
    """StreamingAccessType.from_type_id should resolve known IDs and reject unknown ones."""
    assert StreamingAccessType.from_type_id(method_id) == expected


def test_maturity_rating_enum_labels_are_stable() -> None:
    """MaturityRating labels should remain stable for parsing and display."""
    assert MaturityRating.G.value == "g"
    assert MaturityRating.PG.value == "pg"
    assert MaturityRating.PG_13.value == "pg-13"
    assert MaturityRating.R.value == "r"
    assert MaturityRating.NC_17.value == "nc-17"
    assert MaturityRating.UNRATED.value == "unrated"


def test_maturity_rating_maturity_ranks_are_stable() -> None:
    """Maturity ranks should remain stable for persisted numeric comparisons."""
    assert MaturityRating.G.maturity_rank == 1
    assert MaturityRating.PG.maturity_rank == 2
    assert MaturityRating.PG_13.maturity_rank == 3
    assert MaturityRating.R.maturity_rank == 4
    assert MaturityRating.NC_17.maturity_rank == 5
    assert MaturityRating.UNRATED.maturity_rank == 999


def test_streaming_access_type_enum_values_are_stable() -> None:
    """StreamingAccessType values should remain stable for watch-offering keys."""
    assert StreamingAccessType.SUBSCRIPTION.type_id == 1
    assert StreamingAccessType.BUY.type_id == 2
    assert StreamingAccessType.RENT.type_id == 3
    assert StreamingAccessType.SUBSCRIPTION.value == "subscription"
    assert StreamingAccessType.BUY.value == "buy"
    assert StreamingAccessType.RENT.value == "rent"
