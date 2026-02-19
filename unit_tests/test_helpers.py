"""Unit tests for helper utility functions."""

import pytest

from implementation.misc.helpers import create_watch_provider_offering_key, normalize_string


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("", ""),
        ("   ", ""),
        ("Spider-Man", "spider-man"),
        ("Ocean's Eleven", "oceans eleven"),
        ("L.A. Confidential", "la confidential"),
        ("The Lord of the Rings: The Two Towers", "the lord of the rings the two towers"),
        ("Se7en", "se7en"),
        ("AMELIE", "amelie"),
        ("Amelie", "amelie"),
        ("Jean-Luc Picard", "jean-luc picard"),
        ("  Mixed\tWhitespace\nText  ", "mixed whitespace text"),
    ],
)
def test_normalize_string_common_and_edge_cases(raw_text: str, expected: str) -> None:
    """normalize_string should normalize punctuation, case, and whitespace consistently."""
    assert normalize_string(raw_text) == expected


def test_create_watch_provider_offering_key_happy_path() -> None:
    """create_watch_provider_offering_key should combine IDs using bit packing."""
    # The lower 4 bits hold the watch method ID while upper bits hold provider ID.
    assert create_watch_provider_offering_key(11, 3) == (11 << 4) | 3


def test_create_watch_provider_offering_key_zero_boundary() -> None:
    """create_watch_provider_offering_key should support zero values."""
    assert create_watch_provider_offering_key(0, 0) == 0


def test_create_watch_provider_offering_key_large_provider_id() -> None:
    """create_watch_provider_offering_key should preserve large provider IDs."""
    provider_id = 2**20
    method_id = 2
    assert create_watch_provider_offering_key(provider_id, method_id) == (provider_id << 4) | method_id


def test_create_watch_provider_offering_key_max_method_id() -> None:
    """create_watch_provider_offering_key should handle max 4-bit method ID (15)."""
    result = create_watch_provider_offering_key(1, 15)
    assert result == (1 << 4) | 15
    assert result & 0xF == 15


def test_create_watch_provider_offering_key_method_id_overflow_bleeds_into_provider() -> None:
    """Method IDs exceeding 4 bits bleed into the provider portion (no guard in implementation)."""
    result = create_watch_provider_offering_key(1, 16)
    assert result & 0xF != 16


def test_create_watch_provider_offering_key_round_trip() -> None:
    """Provider and method IDs should be recoverable via bit masking."""
    provider_id, method_id = 42, 3
    key = create_watch_provider_offering_key(provider_id, method_id)
    assert key >> 4 == provider_id
    assert key & 0xF == method_id


# ================================
#  Unicode edge cases for normalize_string
# ================================


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Amélie", "amelie"),
        ("naïve café", "naive cafe"),
        ("Ñoño", "nono"),
        ("Strüdel", "strudel"),
        ("...!!!", ""),
        ("---", "---"),
        ("It's a 'test'", "its a test"),
        ("L.A. Confidential", "la confidential"),
    ],
)
def test_normalize_string_unicode_and_punctuation_edge_cases(raw_text: str, expected: str) -> None:
    """normalize_string should handle diacritics, multiple punctuation types, and apostrophes."""
    assert normalize_string(raw_text) == expected
