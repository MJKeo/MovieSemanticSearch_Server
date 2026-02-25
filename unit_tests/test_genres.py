"""Invariant tests for the Genre enum."""

from implementation.classes.enums import Genre
from implementation.misc.helpers import normalize_string


def test_genre_ids_are_unique() -> None:
    """Every Genre member must have a unique genre_id."""
    genre_ids = [genre.genre_id for genre in Genre]
    assert len(genre_ids) == len(set(genre_ids)), (
        "Duplicate genre_id values detected in Genre enum."
    )


def test_genre_values_are_unique() -> None:
    """Every Genre member must have a unique display value."""
    values = [genre.value for genre in Genre]
    assert len(values) == len(set(values)), (
        "Duplicate display values detected in Genre enum."
    )


def test_genre_normalized_names_match_normalize_string_of_value() -> None:
    """Every Genre.normalized_name must match normalize_string(Genre.value)."""
    for genre in Genre:
        assert genre.normalized_name == normalize_string(genre.value), (
            f"Genre.{genre.name}: normalized_name={genre.normalized_name!r} "
            f"does not equal normalize_string(value)={normalize_string(genre.value)!r}"
        )
