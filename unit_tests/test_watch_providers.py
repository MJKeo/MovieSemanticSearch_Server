"""Unit tests for StreamingService watch provider lookup structures."""

from implementation.classes.watch_providers import (
    StreamingService,
    STREAMING_PROVIDER_MAP,
    PROVIDER_ID_TO_SERVICE,
    STREAMING_SERVICE_DISPLAY_NAMES,
    STREAMING_SERVICE_ALIASES,
)


def test_streaming_provider_map_covers_all_services() -> None:
    """Every StreamingService enum member should have an entry in STREAMING_PROVIDER_MAP."""
    for service in StreamingService:
        assert service in STREAMING_PROVIDER_MAP, f"{service} missing from STREAMING_PROVIDER_MAP"
        assert len(STREAMING_PROVIDER_MAP[service]) > 0, f"{service} has empty provider list"


def test_provider_id_to_service_is_complete_reverse_lookup() -> None:
    """PROVIDER_ID_TO_SERVICE should contain exactly all IDs listed in STREAMING_PROVIDER_MAP."""
    expected_ids: set[int] = set()
    for ids in STREAMING_PROVIDER_MAP.values():
        expected_ids.update(ids)
    assert set(PROVIDER_ID_TO_SERVICE.keys()) == expected_ids


def test_provider_id_to_service_maps_to_correct_service() -> None:
    """Each provider ID should map back to the service that lists it."""
    for service, ids in STREAMING_PROVIDER_MAP.items():
        for pid in ids:
            assert PROVIDER_ID_TO_SERVICE[pid] == service


def test_display_names_covers_all_services() -> None:
    """Every StreamingService should have a display name."""
    for service in StreamingService:
        assert service in STREAMING_SERVICE_DISPLAY_NAMES, f"{service} missing display name"
        assert STREAMING_SERVICE_DISPLAY_NAMES[service]


def test_aliases_covers_all_services() -> None:
    """Every StreamingService should have at least one alias."""
    for service in StreamingService:
        assert service in STREAMING_SERVICE_ALIASES, f"{service} missing aliases"
        assert len(STREAMING_SERVICE_ALIASES[service]) > 0


def test_known_provider_ids_map_to_expected_services() -> None:
    """Spot-check well-known provider IDs resolve to the correct service."""
    assert PROVIDER_ID_TO_SERVICE[8] == StreamingService.NETFLIX    # Netflix core
    assert PROVIDER_ID_TO_SERVICE[15] == StreamingService.HULU      # Hulu core
    assert PROVIDER_ID_TO_SERVICE[9] == StreamingService.AMAZON     # Amazon Prime Video
    assert PROVIDER_ID_TO_SERVICE[1899] == StreamingService.MAX     # HBO Max
    assert PROVIDER_ID_TO_SERVICE[526] == StreamingService.AMC      # AMC+
    assert PROVIDER_ID_TO_SERVICE[2528] == StreamingService.YOUTUBE # YouTube TV
