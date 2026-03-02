from enum import Enum


class StreamingService(str, Enum):
    NETFLIX = "netflix"
    AMAZON = "amazon"
    HULU = "hulu"
    DISNEY = "disney"
    MAX = "max"
    PEACOCK = "peacock"
    PARAMOUNT = "paramount"
    APPLE = "apple"
    CRUNCHYROLL = "crunchyroll"
    FUBOTV = "fubotv"
    YOUTUBE = "youtube"
    AMC = "amc"
    STARZ = "starz"
    TUBI = "tubi"
    PLUTO = "pluto"
    ROKU = "roku"
    PLEX = "plex"
    SHUDDER = "shudder"
    MGM = "mgm"
    VUDU = "vudu"


# =============================================================================
# STREAMING_PROVIDER_MAP
#
# Maps each StreamingService enum to all TMDB watch provider IDs that a user
# would reasonably expect to see results for when they say
# "movies on [service name]".
#
# GUIDING PRINCIPLES:
#   1. "X [Platform] Channel" belongs to X (the content brand), NOT the
#      distribution platform. A user saying "on Starz" wants Starz content
#      whether accessed via the Starz app, Starz Amazon Channel, or Starz
#      Roku Premium Channel.
#
#   2. A user saying "on Amazon" means Amazon's own catalog (Prime Video,
#      rentals, etc.), NOT the hundreds of third-party add-on channels sold
#      through Amazon's marketplace. "Yipee Kids TV Amazon Channel" is not
#      Amazon content.
#
#   3. Same logic applies to Apple and Roku as distribution platforms — only
#      their own first-party content/storefront IDs map to APPLE/ROKU.
#
#   4. Tier variants of the same service (e.g., "Peacock Premium" vs
#      "Peacock Premium Plus", "Netflix" vs "Netflix Standard with Ads")
#      all map to the parent service.
#
#   5. Third-party niche channels distributed through Amazon/Apple/Roku
#      that are NOT one of our enum services are excluded entirely. They
#      don't belong to the platform and they don't belong to any of our
#      tracked services.
# =============================================================================

STREAMING_PROVIDER_MAP: dict[StreamingService, list[int]] = {
    # -------------------------------------------------------------------------
    # NETFLIX
    # -------------------------------------------------------------------------
    # 8    Netflix                        — core service
    # 175  Netflix Kids                   — Netflix's kids profile/section
    # 1796 Netflix Standard with Ads      — ad-supported Netflix tier
    StreamingService.NETFLIX: [8, 175, 1796],

    # -------------------------------------------------------------------------
    # AMAZON
    # -------------------------------------------------------------------------
    # Includes only Amazon's OWN content/storefront. Third-party channels
    # sold through Amazon (e.g., "Shudder Amazon Channel") map to their
    # respective content brand, not to Amazon.
    #
    # 9    Amazon Prime Video             — core subscription streaming
    # 10   Amazon Video                   — rental/purchase storefront
    # 613  Amazon Prime Video Free w/ Ads — free ad-supported tier
    # 2100 Amazon Prime Video with Ads    — ad-supported subscription tier
    StreamingService.AMAZON: [9, 10, 613, 2100],

    # -------------------------------------------------------------------------
    # HULU
    # -------------------------------------------------------------------------
    # 15   Hulu                           — core service
    StreamingService.HULU: [15],

    # -------------------------------------------------------------------------
    # DISNEY
    # -------------------------------------------------------------------------
    # 337  Disney Plus                    — core service
    # 508  DisneyNOW                      — Disney's free streaming app
    StreamingService.DISNEY: [337, 508],

    # -------------------------------------------------------------------------
    # MAX (formerly HBO Max)
    # -------------------------------------------------------------------------
    # 1899 HBO Max                        — core service (TMDB still uses old name)
    # 1825 HBO Max Amazon Channel         — same HBO/Max content via Amazon
    # 2365 HBO Max CNN Amazon Channel     — HBO Max content bundle via Amazon
    StreamingService.MAX: [1899, 1825, 2365],

    # -------------------------------------------------------------------------
    # PEACOCK
    # -------------------------------------------------------------------------
    # 386  Peacock Premium                — standard tier
    # 387  Peacock Premium Plus           — premium tier (no ads)
    StreamingService.PEACOCK: [386, 387],

    # -------------------------------------------------------------------------
    # PARAMOUNT
    # -------------------------------------------------------------------------
    # 531  Paramount Plus                 — core service (from original list)
    # 2616 Paramount Plus Essential       — ad-supported tier
    # 2303 Paramount Plus Premium         — premium tier
    # 582  Paramount+ Amazon Channel      — same Paramount+ content via Amazon
    # 633  Paramount+ Roku Premium Channel— same content via Roku
    # 2474 Paramount+ Originals Amazon Ch — Paramount+ originals via Amazon
    # 2475 Paramount+ MTV Amazon Channel  — Paramount+ MTV content via Amazon
    StreamingService.PARAMOUNT: [531, 2616, 2303, 582, 633, 2474, 2475],

    # -------------------------------------------------------------------------
    # APPLE
    # -------------------------------------------------------------------------
    # Includes Apple's own storefront and streaming service. Third-party
    # channels sold through Apple TV (e.g., "Starz Apple TV Channel") map
    # to their respective content brand.
    #
    # 350  Apple TV                       — core streaming service (Apple TV+)
    # 2    Apple TV Store                 — rental/purchase storefront
    # 2243 Apple TV Amazon Channel        — Apple TV+ content via Amazon
    StreamingService.APPLE: [350, 2, 2243],

    # -------------------------------------------------------------------------
    # CRUNCHYROLL
    # -------------------------------------------------------------------------
    # 283  Crunchyroll                    — core service
    # 1968 Crunchyroll Amazon Channel     — same Crunchyroll content via Amazon
    StreamingService.CRUNCHYROLL: [283, 1968],

    # -------------------------------------------------------------------------
    # FUBOTV
    # -------------------------------------------------------------------------
    # 257  fuboTV                         — core service
    StreamingService.FUBOTV: [257],

    # -------------------------------------------------------------------------
    # YOUTUBE
    # -------------------------------------------------------------------------
    # 192  YouTube                        — core platform (rental/purchase)
    # 188  YouTube Premium                — ad-free tier
    # 235  YouTube Free                   — free ad-supported content
    # 2528 YouTube TV                     — live TV service (may list movie availability)
    StreamingService.YOUTUBE: [192, 188, 235, 2528],

    # -------------------------------------------------------------------------
    # AMC
    # -------------------------------------------------------------------------
    # 80   AMC                            — base AMC platform
    # 526  AMC+                           — AMC's premium streaming service
    # 528  AMC+ Amazon Channel            — AMC+ content via Amazon
    # 1854 AMC Plus Apple TV Channel      — AMC+ content via Apple TV
    # 635  AMC+ Roku Premium Channel      — AMC+ content via Roku
    StreamingService.AMC: [80, 526, 528, 1854, 635],

    # -------------------------------------------------------------------------
    # STARZ
    # -------------------------------------------------------------------------
    # 43   Starz                          — core service
    # 1794 Starz Amazon Channel           — Starz content via Amazon
    # 1855 Starz Apple TV Channel         — Starz content via Apple TV
    # 634  Starz Roku Premium Channel     — Starz content via Roku
    StreamingService.STARZ: [43, 1794, 1855, 634],

    # -------------------------------------------------------------------------
    # TUBI
    # -------------------------------------------------------------------------
    # 73   Tubi TV                        — core service
    StreamingService.TUBI: [73],

    # -------------------------------------------------------------------------
    # PLUTO
    # -------------------------------------------------------------------------
    # 300  Pluto TV                       — core service
    # 1965 Pluto TV Live                  — live/linear channels on Pluto
    StreamingService.PLUTO: [300, 1965],

    # -------------------------------------------------------------------------
    # ROKU
    # -------------------------------------------------------------------------
    # Only Roku's own free streaming content. Third-party premium channels
    # sold through Roku (e.g., "Starz Roku Premium Channel") map to their
    # respective content brand.
    #
    # 207  The Roku Channel               — Roku's own free streaming
    StreamingService.ROKU: [207],

    # -------------------------------------------------------------------------
    # PLEX
    # -------------------------------------------------------------------------
    # 538  Plex                           — core service
    # 2077 Plex Channel                   — Plex's free streaming content
    StreamingService.PLEX: [538, 2077],

    # -------------------------------------------------------------------------
    # SHUDDER
    # -------------------------------------------------------------------------
    # 99   Shudder                        — core service
    # 204  Shudder Amazon Channel         — Shudder content via Amazon
    # 2049 Shudder Apple TV Channel       — Shudder content via Apple TV
    StreamingService.SHUDDER: [99, 204, 2049],

    # -------------------------------------------------------------------------
    # MGM
    # -------------------------------------------------------------------------
    # 34   MGM Plus                       — core service
    # 583  MGM+ Amazon Channel            — MGM+ content via Amazon
    # 636  MGM Plus Roku Premium Channel  — MGM+ content via Roku
    StreamingService.MGM: [34, 583, 636],

    # -------------------------------------------------------------------------
    # VUDU (Fandango at Home)
    # -------------------------------------------------------------------------
    # 7    Fandango At Home               — core service (formerly Vudu)
    # 332  Fandango at Home Free          — free ad-supported tier
    # 60   Fandango                       — Fandango's digital storefront
    StreamingService.VUDU: [7, 332, 60],
}


# =============================================================================
# REVERSE LOOKUP: provider_id → StreamingService
# Useful for checking if a movie's watch providers match a user's filter.
# =============================================================================
PROVIDER_ID_TO_SERVICE: dict[int, StreamingService] = {}
for service, provider_ids in STREAMING_PROVIDER_MAP.items():
    for pid in provider_ids:
        PROVIDER_ID_TO_SERVICE[pid] = service


# =============================================================================
# DISPLAY NAMES: What to show in UI filters and confirmations
# =============================================================================
STREAMING_SERVICE_DISPLAY_NAMES: dict[StreamingService, str] = {
    StreamingService.NETFLIX: "Netflix",
    StreamingService.AMAZON: "Amazon Prime Video",
    StreamingService.HULU: "Hulu",
    StreamingService.DISNEY: "Disney+",
    StreamingService.MAX: "Max",
    StreamingService.PEACOCK: "Peacock",
    StreamingService.PARAMOUNT: "Paramount+",
    StreamingService.APPLE: "Apple TV+",
    StreamingService.CRUNCHYROLL: "Crunchyroll",
    StreamingService.FUBOTV: "fuboTV",
    StreamingService.YOUTUBE: "YouTube",
    StreamingService.AMC: "AMC+",
    StreamingService.STARZ: "Starz",
    StreamingService.TUBI: "Tubi",
    StreamingService.PLUTO: "Pluto TV",
    StreamingService.ROKU: "The Roku Channel",
    StreamingService.PLEX: "Plex",
    StreamingService.SHUDDER: "Shudder",
    StreamingService.MGM: "MGM+",
    StreamingService.VUDU: "Fandango at Home",
}


# =============================================================================
# QUERY ALIASES: Terms the LLM should recognize and map to each enum value.
# This is reference documentation for your query understanding prompt, not
# runtime code. The LLM handles fuzzy matching naturally, but these ensure
# rebranded/colloquial names are covered.
# =============================================================================
STREAMING_SERVICE_ALIASES: dict[StreamingService, list[str]] = {
    StreamingService.NETFLIX: ["netflix"],
    StreamingService.AMAZON: ["amazon", "prime video", "prime", "amazon prime"],
    StreamingService.HULU: ["hulu"],
    StreamingService.DISNEY: ["disney", "disney plus", "disney+"],
    StreamingService.MAX: ["max", "hbo", "hbo max"],
    StreamingService.PEACOCK: ["peacock"],
    StreamingService.PARAMOUNT: ["paramount", "paramount plus", "paramount+"],
    StreamingService.APPLE: ["apple", "apple tv", "apple tv+"],
    StreamingService.CRUNCHYROLL: ["crunchyroll"],
    StreamingService.FUBOTV: ["fubo", "fubotv"],
    StreamingService.YOUTUBE: ["youtube", "youtube tv"],
    StreamingService.AMC: ["amc", "amc+", "amc plus"],
    StreamingService.STARZ: ["starz"],
    StreamingService.TUBI: ["tubi"],
    StreamingService.PLUTO: ["pluto", "pluto tv"],
    StreamingService.ROKU: ["roku", "roku channel"],
    StreamingService.PLEX: ["plex"],
    StreamingService.SHUDDER: ["shudder"],
    StreamingService.MGM: ["mgm", "mgm+", "mgm plus", "epix"],
    StreamingService.VUDU: ["vudu", "fandango", "fandango at home"],
}