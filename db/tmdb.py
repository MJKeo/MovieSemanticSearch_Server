"""
TMDB API client for fetching movie data.

Uses httpx.AsyncClient with Bearer token authentication. Provides two
entry points:
  - fetch_trending_movie_ids(): bulk-fetch weekly trending IDs (daily fetch job)
  - fetch_movie_details():      fetch expanded detail for a single movie (Stage 2)

The AdaptiveRateLimiter enforces TMDB's per-second request ceiling and
self-tunes based on 429 feedback from the server.
"""

import asyncio
import logging
import os

from dotenv import load_dotenv

load_dotenv()
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_TMDB_BASE_URL = "https://api.themoviedb.org/3"
_PAGE_SIZE = 20          # TMDB returns exactly 20 results per page
_SEMAPHORE_LIMIT = 10    # max concurrent requests (~40 req/10 s TMDB limit)
_MAX_ATTEMPTS = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles on each retry


def access_token() -> str:
    """Return the TMDB bearer token from the environment."""
    token = os.getenv("TMDB_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("TMDB_ACCESS_TOKEN environment variable is not set")
    return token


# ---------------------------------------------------------------------------
# Adaptive rate limiter — token-bucket with 429-driven feedback loop
# ---------------------------------------------------------------------------


class AdaptiveRateLimiter:
    """
    Token-bucket rate limiter with adaptive rate adjustment.

    Maintains a bucket of tokens that refill at ``current_rate`` per second,
    capped at ``burst``.  Each ``acquire()`` call consumes one token; callers
    sleep when the bucket is empty.

    On 429 signals: rate drops by 10%.
    After ``clean_window`` seconds with no 429s: rate increases by 5%, up to
    ``max_rate``.
    """

    def __init__(
        self,
        initial_rate: float = 36.0,
        max_rate: float = 40.0,
        burst: int = 5,
        clean_window: float = 120.0,
        increase_interval: float = 10.0,
    ) -> None:
        self.current_rate = initial_rate
        self.max_rate = max_rate
        self.burst = burst
        self.clean_window = clean_window
        # Minimum seconds between consecutive rate increases.  Separate from
        # clean_window because the concerns differ: clean_window gates the
        # *first* increase after a 429, while increase_interval throttles the
        # ramp-up cadence once recovery has begun.
        self.increase_interval = increase_interval

        # Token bucket state
        self.tokens: float = float(burst)
        self._last_refill = time.monotonic()
        self._last_429_time = 0.0
        self._lock = asyncio.Lock()

        # Global cooldown: when a 429 arrives, ALL coroutines must wait until
        # this monotonic timestamp before issuing any new request.  This
        # prevents the remaining queued coroutines from hammering TMDB with
        # requests that will also 429.
        self._cooldown_until = 0.0

        # Tracks when we last increased the rate so we only bump once per
        # increase_interval, not on every acquire() call after the clean
        # window elapses.
        self._last_increase_time = 0.0

        # Cumulative metrics
        self.total_requests = 0
        self.total_429s = 0

    async def acquire(self) -> None:
        """Block until a rate-limit token is available."""
        async with self._lock:
            now = time.monotonic()

            # Honour the global cooldown set by report_429().  Every coroutine
            # pauses here until the cooldown expires, ensuring we don't spray
            # requests into a known rate-limit window.
            if now < self._cooldown_until:
                wait = self._cooldown_until - now
                await asyncio.sleep(wait)
                now = time.monotonic()

            # Refill tokens proportional to elapsed time since last refill
            elapsed = now - self._last_refill
            self.tokens = min(self.burst, self.tokens + elapsed * self.current_rate)
            self._last_refill = now

            # Sleep if no token is available yet
            if self.tokens < 1.0:
                wait = (1.0 - self.tokens) / self.current_rate
                await asyncio.sleep(wait)
                self.tokens = 0.0
                self._last_refill = time.monotonic()
            else:
                self.tokens -= 1.0

            # Gently increase rate after a sustained clean window with no 429s.
            # Two separate timing checks:
            #   1. clean_window: must have zero 429s for this long before ANY
            #      increase (proves current rate is stable).
            #   2. increase_interval: minimum gap between consecutive bumps so
            #      we ramp gradually rather than jumping to max_rate instantly.
            if (
                now - self._last_429_time > self.clean_window
                and now - self._last_increase_time > self.increase_interval
                and self.current_rate < self.max_rate
            ):
                old_rate = self.current_rate
                self.current_rate = min(self.max_rate, self.current_rate * 1.05)
                self._last_increase_time = now
                if self.current_rate != old_rate:
                    print(f"  Rate limiter: increased to {self.current_rate:.1f} req/s")

            self.total_requests += 1

    def report_429(self, retry_after: float = 2.0) -> None:
        """
        Signal that TMDB returned 429.  Drops the rate by 10% and sets a
        global cooldown so that ALL coroutines pause before making new
        requests — not just the one that received the 429.

        Thread-safety note: this method mutates shared state (_last_429_time,
        current_rate, _cooldown_until) WITHOUT holding self._lock.  This is
        safe because (a) asyncio is single-threaded, and (b) this method is
        synchronous (no ``await``), so it runs atomically between event-loop
        yield points.  Do NOT add ``await`` calls to this method without
        acquiring the lock first.
        """
        now = time.monotonic()
        self.total_429s += 1
        # Ignore duplicate 429s within the cooldown window —
        # they're from requests already in flight, not a new signal
        if now < self._cooldown_until:
            return

        self._last_429_time = now
        old_rate = self.current_rate
        self.current_rate = max(1.0, self.current_rate * 0.90)

        # Set the global cooldown: no coroutine may issue a request until
        # this timestamp.  If a cooldown is already active (overlapping 429s),
        # extend it to whichever deadline is later.
        new_deadline = now + retry_after
        self._cooldown_until = max(self._cooldown_until, new_deadline)

        print(
            f"  Rate limiter: 429 received, reduced {old_rate:.1f} → "
            f"{self.current_rate:.1f} req/s, cooldown {retry_after:.0f}s "
            f"(total 429s: {self.total_429s}/{self.total_requests})"
        )

    def stats(self) -> str:
        """Human-readable one-liner of current state and cumulative metrics."""
        pct = (self.total_429s / max(1, self.total_requests)) * 100
        return (
            f"Rate: {self.current_rate:.1f} req/s | "
            f"Total: {self.total_requests:,} | "
            f"429s: {self.total_429s:,} ({pct:.2f}%)"
        )


# ---------------------------------------------------------------------------
# TMDBFetchError — raised for non-recoverable fetch failures
# ---------------------------------------------------------------------------


class TMDBFetchError(Exception):
    """Non-retryable failure when fetching from the TMDB API."""


# ---------------------------------------------------------------------------
# fetch_movie_details — single-movie expanded detail endpoint
# ---------------------------------------------------------------------------


async def fetch_movie_details(
    client: httpx.AsyncClient,
    rate_limiter: AdaptiveRateLimiter,
    tmdb_id: int,
    max_attempts: int = 3,
) -> dict | None:
    """
    Fetch the expanded detail payload for a single TMDB movie.

    Appends release_dates, keywords, watch/providers, and credits in one
    request.  Returns the parsed JSON dict on success, ``None`` for 404
    (movie deleted/missing on TMDB), or raises ``TMDBFetchError`` after
    exhausting retries on transient errors.

    429 responses are handled transparently: they signal the adaptive rate
    limiter and retry without consuming the transient-error retry budget.

    Args:
        client:       Shared httpx async client (must already carry auth headers).
        rate_limiter: Adaptive rate limiter governing request pacing.
        tmdb_id:      TMDB movie ID to fetch.
        max_attempts:  Maximum attempts before giving up on transient errors.

    Returns:
        Parsed JSON dict on 200, ``None`` on 404.

    Raises:
        TMDBFetchError: After all retries exhausted or on non-retryable status.
        ValueError:     If the 200 response body is not valid JSON.
    """
    url = f"{_TMDB_BASE_URL}/movie/{tmdb_id}"
    params = {"append_to_response": "release_dates,keywords,watch/providers,credits"}

    # Track transient-error retries separately from 429s.  429s are a normal
    # part of rate-limit convergence and should retry indefinitely (the rate
    # limiter's cooldown prevents runaway loops).  Only transport errors and
    # 5xx responses consume from the retry budget.
    transient_attempts = 0

    while transient_attempts < max_attempts:
        await rate_limiter.acquire()

        try:
            response = await client.get(url, params=params)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            transient_attempts += 1
            if transient_attempts >= max_attempts:
                raise TMDBFetchError(
                    f"Transport error for tmdb_id={tmdb_id} after "
                    f"{max_attempts} attempts: {exc}"
                ) from exc
            await asyncio.sleep(_RETRY_BACKOFF_BASE * 2 ** (transient_attempts - 1))
            continue

        # --- 200 OK: parse and return ---------------------------------
        if response.status_code == 200:
            # JSON decode failure is non-transient — raise immediately
            return response.json()

        # --- 404: movie does not exist on TMDB -------------------------
        if response.status_code == 404:
            return None

        # --- 429: rate limited — back off and retry --------------------
        # report_429 sets a global cooldown on the rate limiter so ALL
        # coroutines pause, not just this one.  The next acquire() call
        # (top of the loop) will honour that cooldown.  429s do NOT
        # consume the transient retry budget — they're a normal rate-limit
        # signal, not a transient error.
        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", 2))
            rate_limiter.report_429(retry_after)
            print(f"  429 for tmdb_id={tmdb_id}, backing off {retry_after:.0f}s")
            continue

        # --- 5xx: transient server error — back off and retry ----------
        if response.status_code >= 500:
            transient_attempts += 1
            if transient_attempts >= max_attempts:
                raise TMDBFetchError(
                    f"Server error {response.status_code} for tmdb_id={tmdb_id} "
                    f"after {max_attempts} attempts"
                )
            await asyncio.sleep(_RETRY_BACKOFF_BASE * 2 ** (transient_attempts - 1))
            continue

        # --- Anything else: non-retryable ------------------------------
        raise TMDBFetchError(
            f"Unexpected HTTP {response.status_code} for tmdb_id={tmdb_id}"
        )

    # Should only be reachable if max_attempts <= 0
    raise TMDBFetchError(
        f"Exhausted {max_attempts} retries for tmdb_id={tmdb_id}"
    )


# ---------------------------------------------------------------------------
# Trending endpoint (pre-existing daily functionality)
# ---------------------------------------------------------------------------


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

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        async with sem:
            try:
                response = await client.get(url, params={"page": page})
            except httpx.TransportError as exc:
                if attempt == _MAX_ATTEMPTS:
                    raise
                wait = _RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "TMDB transport error on page %d (attempt %d/%d): %s — retrying in %.1fs",
                    page, attempt, _MAX_ATTEMPTS, exc, wait,
                )
                await asyncio.sleep(wait)
                continue

        if response.status_code == 429:
            retry_after = float(response.headers.get("Retry-After", _RETRY_BACKOFF_BASE * 2 ** attempt))
            if attempt == _MAX_ATTEMPTS:
                response.raise_for_status()
            logger.warning("TMDB rate-limited on page %d — sleeping %.1fs", page, retry_after)
            await asyncio.sleep(retry_after)
            continue

        response.raise_for_status()
        results: list[dict[str, Any]] = response.json().get("results", [])
        return [entry["id"] for entry in results]

    raise RuntimeError(f"Failed to fetch TMDB page {page} after {_MAX_ATTEMPTS} attempts")  # unreachable


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
    headers = {"Authorization": f"Bearer {access_token()}"}

    async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
        tasks = [_fetch_page(client, sem, page) for page in range(1, pages_needed + 1)]
        pages: list[list[int]] = await asyncio.gather(*tasks)

    # Flatten pages in order, then trim to at most n
    all_ids: list[int] = [movie_id for page_ids in pages for movie_id in page_ids]
    return all_ids[:n]
