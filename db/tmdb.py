"""
TMDB API client for fetching trending movie data.

Uses httpx.AsyncClient with Bearer token authentication. All pages are fetched
concurrently, bounded by a semaphore to respect TMDB's rate limits.
"""

import asyncio
import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TMDB_BASE_URL = "https://api.themoviedb.org/3"
_PAGE_SIZE = 20          # TMDB returns exactly 20 results per page
_SEMAPHORE_LIMIT = 10    # max concurrent requests (~40 req/10 s TMDB limit)
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles on each retry


def _access_token() -> str:
    token = os.getenv("TMDB_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("TMDB_ACCESS_TOKEN environment variable is not set")
    return token


async def _fetch_page(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    page: int,
) -> list[int]:
    """
    Fetch one page of weekly trending movies and return its movie IDs in page order.

    Retries on transient transport errors and 429 rate-limit responses with
    exponential back-off. Raises on non-retryable HTTP errors.
    """
    url = f"{_TMDB_BASE_URL}/trending/movie/week"

    for attempt in range(1, _MAX_RETRIES + 1):
        async with sem:
            try:
                response = await client.get(url, params={"page": page})
            except httpx.TransportError as exc:
                if attempt == _MAX_RETRIES:
                    raise
                wait = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "TMDB transport error on page %d (attempt %d/%d): %s — retrying in %.1fs",
                    page, attempt, _MAX_RETRIES, exc, wait,
                )
                await asyncio.sleep(wait)
                continue

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", _RETRY_BACKOFF_BASE * 2 ** attempt))
            if attempt == _MAX_RETRIES:
                response.raise_for_status()
            logger.warning("TMDB rate-limited on page %d — sleeping %.1fs", page, retry_after)
            await asyncio.sleep(retry_after)
            continue

        response.raise_for_status()
        results: list[dict[str, Any]] = response.json().get("results", [])
        return [entry["id"] for entry in results]

    raise RuntimeError(f"Failed to fetch TMDB page {page} after {_MAX_RETRIES} attempts")  # unreachable


async def fetch_trending_movie_ids(n: int = 500) -> list[int]:
    """
    Return the top-N weekly trending movie IDs from TMDB, in rank order.

    Pages are fetched concurrently. The result is trimmed to exactly `n`
    entries (or fewer if TMDB returns less data than requested).

    Args:
        n: Maximum number of movie IDs to return. Must be positive.

    Returns:
        List of TMDB movie IDs, rank-ordered (index 0 = rank 1).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    pages_needed = (n + _PAGE_SIZE - 1) // _PAGE_SIZE  # ceiling division
    print(f"pages_needed: {pages_needed}")
    sem = asyncio.Semaphore(_SEMAPHORE_LIMIT)
    headers = {"Authorization": f"Bearer {_access_token()}"}

    async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
        tasks = [_fetch_page(client, sem, page) for page in range(1, pages_needed + 1)]
        pages: list[list[int]] = await asyncio.gather(*tasks)

    # Flatten pages in order, then trim to at most n
    all_ids: list[int] = [movie_id for page_ids in pages for movie_id in page_ids]
    return all_ids[:n]
