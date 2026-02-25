"""Unit tests for enum conversion and string formatting behavior."""

import pytest

from implementation.classes.enums import MaturityRating, WatchMethodType


@pytest.mark.parametrize(
    ("raw_rating", "expected"),
    [
        ("G", MaturityRating.G),
        ("pg", MaturityRating.PG),
        (" PG-13 ", MaturityRating.PG_13),
        ("r", MaturityRating.R),
        ("nc-17", MaturityRating.NC_17),
        ("Unrated", MaturityRating.UNRATED),
        ("completely unknown", MaturityRating.UNRATED),
    ],
)
def test_maturity_rating_from_string(raw_rating: str, expected: MaturityRating) -> None:
    """MaturityRating.from_string should parse known values and default unknown values."""
    assert MaturityRating.from_string(raw_rating) == expected


@pytest.mark.parametrize(
    ("rating", "expected"),
    [
        (MaturityRating.G, "G"),
        (MaturityRating.PG, "PG"),
        (MaturityRating.PG_13, "PG-13"),
        (MaturityRating.R, "R"),
        (MaturityRating.NC_17, "NC-17"),
        (MaturityRating.UNRATED, "Unrated"),
    ],
)
def test_maturity_rating_string_labels(rating: MaturityRating, expected: str) -> None:
    """MaturityRating.__str__ should return the expected human label for each member."""
    assert str(rating) == expected


@pytest.mark.parametrize(
    ("raw_method", "expected"),
    [
        ("subscription", WatchMethodType.SUBSCRIPTION),
        (" PURCHASE ", WatchMethodType.PURCHASE),
        ("rent", WatchMethodType.RENT),
        ("invalid", None),
    ],
)
def test_watch_method_type_from_string(raw_method: str, expected: WatchMethodType | None) -> None:
    """WatchMethodType.from_string should normalize case/spacing and reject unknown values."""
    assert WatchMethodType.from_string(raw_method) == expected


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (WatchMethodType.SUBSCRIPTION, "subscription"),
        (WatchMethodType.PURCHASE, "purchase"),
        (WatchMethodType.RENT, "rent"),
    ],
)
def test_watch_method_type_string_labels(method: WatchMethodType, expected: str) -> None:
    """WatchMethodType.__str__ should return the expected lowercase label."""
    assert str(method) == expected


def test_maturity_rating_enum_values_are_stable() -> None:
    """MaturityRating values should remain stable for persisted comparisons."""
    assert MaturityRating.G == 1
    assert MaturityRating.PG == 2
    assert MaturityRating.PG_13 == 3
    assert MaturityRating.R == 4
    assert MaturityRating.NC_17 == 5
    assert MaturityRating.UNRATED == 999


def test_watch_method_type_enum_values_are_stable() -> None:
    """WatchMethodType values should remain stable for watch-offering keys."""
    assert WatchMethodType.SUBSCRIPTION == 1
    assert WatchMethodType.PURCHASE == 2
    assert WatchMethodType.RENT == 3
