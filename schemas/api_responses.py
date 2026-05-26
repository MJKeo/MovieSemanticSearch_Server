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
