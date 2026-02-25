"""Unit tests for filterable watch provider lookup structures."""

from implementation.classes.watch_providers import (
    FILTERABLE_WATCH_PROVIDERS_MAP,
    FILTERABLE_WATCH_PROVIDER_IDS,
    FILTERABLE_WATCH_PROVIDER_NAMES,
    FILTERABLE_WATCH_PROVIDERS_NAME_TO_ID,
)
from implementation.misc.helpers import normalize_string


def test_watch_provider_ids_and_names_sets_match_source_map() -> None:
    """Derived ID/name sets should exactly match the source provider map."""
    assert FILTERABLE_WATCH_PROVIDER_IDS == set(FILTERABLE_WATCH_PROVIDERS_MAP.keys())
    assert FILTERABLE_WATCH_PROVIDER_NAMES == set(FILTERABLE_WATCH_PROVIDERS_MAP.values())


def test_watch_provider_name_to_id_map_has_full_coverage() -> None:
    """Normalized reverse lookup should include every provider once."""
    assert len(FILTERABLE_WATCH_PROVIDERS_NAME_TO_ID) == len(FILTERABLE_WATCH_PROVIDERS_MAP)
    assert set(FILTERABLE_WATCH_PROVIDERS_NAME_TO_ID.values()) == FILTERABLE_WATCH_PROVIDER_IDS


def test_watch_provider_name_to_id_map_round_trips_provider_names() -> None:
    """Each source provider name should resolve to its original provider ID."""
    for provider_id, provider_name in FILTERABLE_WATCH_PROVIDERS_MAP.items():
        normalized_provider_name = normalize_string(provider_name)
        assert FILTERABLE_WATCH_PROVIDERS_NAME_TO_ID[normalized_provider_name] == provider_id


def test_watch_provider_name_to_id_supports_normalized_queries() -> None:
    """Reverse lookup should work with normalized query variants."""
    assert FILTERABLE_WATCH_PROVIDERS_NAME_TO_ID[normalize_string("hbo max")] == 1899
    assert FILTERABLE_WATCH_PROVIDERS_NAME_TO_ID[normalize_string("  YOUTUBE TV ")] == 2528
    assert FILTERABLE_WATCH_PROVIDERS_NAME_TO_ID[normalize_string("AMC+")] == 526
