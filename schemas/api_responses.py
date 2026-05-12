"""
Shared API response models.

Pydantic models returned by the HTTP layer (api/main.py) to the
frontend. These are intentionally thin — only the fields needed to
render a result tile — so the wire payload stays small and the
contract is stable across endpoints.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MovieCard(BaseModel):
    """Minimal movie summary returned by the search API.

    Backed by `public.movie_card`. `release_date` is the ISO
    (YYYY-MM-DD) form of the underlying `release_ts` Unix timestamp
    column — pre-formatted on the server so the client doesn't have to
    convert. Every field except `tmdb_id` may be null when the
    underlying row is missing the data.
    """

    model_config = ConfigDict(extra="forbid")

    tmdb_id: int
    title: str | None = None
    release_date: str | None = None
    poster_url: str | None = None
