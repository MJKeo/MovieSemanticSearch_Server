"""
Shared API response models.

Wire-format DTOs returned by the HTTP layer (api/main.py) to the
frontend. These are intentionally thin — only the fields needed to
render a result tile — so the wire payload stays small and the
contract is stable across endpoints.

Implemented as `msgspec.Struct` rather than a Pydantic model: the
search hot path serializes these tens of times per request and
msgspec.json.encode is ~10-50× faster than Pydantic's model_dump +
json.dumps round-trip. Pydantic still owns the LLM structured-output
boundary; msgspec earns its keep here on the wire-format hot path.
"""

from __future__ import annotations

import msgspec


class MovieCard(msgspec.Struct, omit_defaults=True, frozen=True):
    """Minimal movie summary returned by the search API.

    Backed by `public.movie_card`. `release_date` is the ISO
    (YYYY-MM-DD) form of the underlying `release_ts` Unix timestamp
    column — pre-formatted on the server so the client doesn't have to
    convert. `maturity_rating` is the canonical MPAA label
    ("G", "PG", "PG-13", "R", "NC-17") derived from the underlying
    `maturity_rank`; null when the movie is unrated. Every field except
    `tmdb_id` may be null when the underlying row is missing the data.
    """

    tmdb_id: int
    title: str | None = None
    release_date: str | None = None
    poster_url: str | None = None
    maturity_rating: str | None = None


# ---------------------------------------------------------------------------
# /movie_details endpoint payload
# ---------------------------------------------------------------------------
#
# The detail view is a curated projection of TMDB's /movie/{id} response
# (with appended credits/videos/images/external_ids/watch-providers/release-dates)
# merged with our locally-computed reception_score. The wire format is
# deliberately narrower than TMDB's raw payload — the frontend only needs
# a handful of fields, and a stable, typed contract keeps the API decoupled
# from upstream schema drift.


class CastMember(msgspec.Struct, omit_defaults=True, frozen=True):
    """One entry in the top-billed cast list (TMDB credits.cast)."""

    name: str
    character: str | None = None
    profile_url: str | None = None  # full https URL, not raw `profile_path`


class CrewMember(msgspec.Struct, omit_defaults=True, frozen=True):
    """One crew credit — a single (person, job) pair.

    `job` carries the canonical TMDB job label. A person credited for
    multiple jobs appears as multiple `CrewMember` entries (the backend
    does not merge them); consumers dedupe by person for display.
    """

    name: str
    job: str
    profile_url: str | None = None


class CrewGroup(msgspec.Struct, omit_defaults=True, frozen=True):
    """Crew for one TMDB department, e.g. "Directing", "Camera".

    `members` mirrors the TMDB credits list 1:1 — one entry per
    (person, job), no merging. `department` is the canonical TMDB
    department string. The server fixes both the group order and the
    member order; the frontend renders the arrays as-is.
    """

    department: str
    members: list[CrewMember] = []


class WatchProvider(msgspec.Struct, omit_defaults=True, frozen=True):
    """One US streaming/rental/purchase offering for the movie.

    `access_type` is the bucket the provider was found in on TMDB's
    watch/providers payload (`flatrate` for subscription, `buy`, `rent`).
    """

    provider_id: int
    name: str
    access_type: str
    logo_url: str | None = None


class MovieDetails(msgspec.Struct, omit_defaults=True, frozen=True):
    """Full movie detail payload for the `/movie_details/{tmdb_id}` endpoint.

    Combines TMDB's live data (overview, cast/crew, providers, trailer,
    images) with our locally-computed `reception_score`. Every optional
    field falls back to `None` / `[]` when TMDB omits it — the frontend
    must tolerate sparse payloads.
    """

    tmdb_id: int
    title: str | None = None
    original_title: str | None = None
    overview: str | None = None
    tagline: str | None = None
    release_date: str | None = None       # ISO YYYY-MM-DD
    runtime_minutes: int | None = None
    maturity_rating: str | None = None    # US certification ("PG-13", etc.)
    genres: list[str] = []
    # Keyword tags from our own OverallKeyword taxonomy (movie_card.keyword_ids),
    # e.g. "Splatter Horror", "Spaghetti Western", "Time Travel" — finer-grained
    # than `genres` and may overlap with it.
    keywords: list[str] = []
    spoken_languages: list[str] = []

    # Media
    poster_url: str | None = None
    backdrop_url: str | None = None
    trailer_url: str | None = None        # YouTube URL of primary trailer
    # Up to 5 extra artwork URLs for a gallery. Backdrops first (ranked by
    # TMDB vote_count), topped up with posters only if the movie has fewer
    # than 5 backdrops on file.
    additional_images: list[str] = []

    # Ratings
    reception_score: float | None = None  # 0–100, our custom score
    tmdb_vote_average: float | None = None
    tmdb_vote_count: int | None = None

    # People (curated from credits)
    # Single ranked crew list, capped at 12 distinct people. Priority order:
    # all directors, then top writers/producers, then the most important
    # remaining crew (cinematographer, composer, editor, …). Each selected
    # person contributes ALL their credits as separate entries — so the list
    # may hold more than 12 rows but only 12 distinct people. The frontend
    # groups by person for display.
    crew: list[CrewMember] = []
    cast: list[CastMember] = []           # top 12 by `order`
    # "See all" hints: True when the curated lists above dropped at least
    # one person TMDB lists — for `crew`, when more than 12 distinct crew
    # people exist. Each drives an independent "See all" link (pointing at
    # `/movie_credits` for the full, uncapped view) in its own section
    # header. Omitted when False per omit_defaults; the frontend treats
    # absent as False.
    cast_truncated: bool = False
    crew_truncated: bool = False

    # Streaming availability (US region only)
    watch_providers: list[WatchProvider] = []

    # External links
    tmdb_url: str = ""                    # always set
    imdb_url: str | None = None
    homepage: str | None = None


# ---------------------------------------------------------------------------
# /movie_credits endpoint payload
# ---------------------------------------------------------------------------
#
# The lazy "See all" companion to /movie_details: the complete, uncapped
# cast and crew for a single movie, fetched on demand. Credits-only — it
# carries no movie metadata (title, overview, posters, …) because the
# frontend already has those from the /movie_details call that rendered
# the page. Same upstream TMDB credits data as /movie_details, but with no
# caps and crew grouped by department instead of the three curated buckets.


class MovieCredits(msgspec.Struct, omit_defaults=True, frozen=True):
    """Full, uncapped cast & crew for the `/movie_credits/{tmdb_id}` endpoint.

    Reuses the `CastMember` / `CrewMember` shapes from `/movie_details`.
    Empty `cast` / `crew` drop out of the wire entirely via `omit_defaults`,
    so the frontend can branch on `?.length`.
    """

    tmdb_id: int
    cast: list[CastMember] = []   # full billed cast, in TMDB billing order
    crew: list[CrewGroup] = []    # grouped by department, server-fixed order
