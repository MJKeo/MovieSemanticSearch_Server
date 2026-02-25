"""Invariant tests for the Language enum."""

from implementation.classes.languages import Language


def test_language_ids_are_unique() -> None:
    """Every Language member must have a unique language_id."""
    language_ids = [language.language_id for language in Language]
    assert len(language_ids) == len(set(language_ids)), (
        "Duplicate language_id values detected in Language enum."
    )


def test_language_values_are_unique() -> None:
    """Every Language member must have a unique display value."""
    values = [language.value for language in Language]
    assert len(values) == len(set(values)), (
        "Duplicate display values detected in Language enum."
    )
