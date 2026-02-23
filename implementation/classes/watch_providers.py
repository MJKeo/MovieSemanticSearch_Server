FILTERABLE_WATCH_PROVIDERS_MAP = {
    8: "Netflix",
    9: "Amazon Prime Video",
    15: "Hulu",
    337: "Disney Plus",
    1899: "HBO Max",
    386: "Peacock Premium",
    531: "Paramount Plus",
    350: "Apple TV",
    283: "Crunchyroll",
    257: "fuboTV",
    192: "YouTube",
    526: "AMC+",
    2528: "YouTube TV",
    43: "Starz",
    73: "Tubi TV",
    300: "Pluto TV",
    207: "The Roku Channel",
    538: "Plex",
    10: "Amazon Video",
}

FILTERABLE_WATCH_PROVIDER_IDS = set(FILTERABLE_WATCH_PROVIDERS_MAP.keys())
FILTERABLE_WATCH_PROVIDER_NAMES = set(FILTERABLE_WATCH_PROVIDERS_MAP.values())

# Reverse lookup: normalized provider name -> TMDB provider ID.
# Used at query time to convert provider name strings (from the query
# understanding WatchProvidersPreference) into TMDB IDs without hitting
# the database.  Import is intentionally deferred to module level to
# keep the import graph simple (helpers has no heavyweight deps).
from implementation.misc.helpers import normalize_string as _normalize_string

FILTERABLE_WATCH_PROVIDERS_NAME_TO_ID: dict[str, int] = {
    _normalize_string(name): provider_id
    for provider_id, name in FILTERABLE_WATCH_PROVIDERS_MAP.items()
}