"""
Async HTTP client for IMDB GraphQL API fetching via DataImpulse proxies.

Provides a single fetch_movie() coroutine that handles:
  - Global semaphore-controlled concurrency
  - Per-request random User-Agent (desktop Chrome/Firefox only)
  - Random pre-request delay (0.05-0.25s) to break burst patterns
  - Retry with exponential backoff (up to 3 attempts)
  - Raw JSON response caching to disk for debugging / re-runs
"""

import asyncio
import json
import os
import random
import ssl
import time
from enum import StrEnum

import httpx
from fake_useragent import UserAgent

from movie_ingestion.tracker import INGESTION_DATA_DIR


# ---------------------------------------------------------------------------
# Fetch result types
# ---------------------------------------------------------------------------


class FetchResult(StrEnum):
    """Outcome of a single movie fetch attempt."""
    SUCCESS = "success"
    HTTP_404 = "http_404"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 3
_MIN_DELAY = 0.05
_MAX_DELAY = 0.25
# Successful fetches complete in <1s — fail fast on blocked/flagged IPs
# so retries rotate to a fresh proxy IP quickly.
_REQUEST_TIMEOUT = 5.0
_MAX_CONNECTIONS = 80
_JSON_CACHE_DIR = INGESTION_DATA_DIR / "imdb_graphql"

# IMDB's public GraphQL API endpoint. No authentication or WAF token required.
# The caching endpoint (caching.graphql.imdb.com) uses persisted queries;
# the direct endpoint accepts full query text, which is simpler.
_GRAPHQL_URL = "https://api.graphql.imdb.com/"

# Session-level headers applied to every request. Accept-Encoding enables
# gzip/brotli compression to minimize billable proxy bandwidth. Origin and
# Referer are required by the GraphQL endpoint (rejects requests without them).
_SESSION_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
    "Accept-Encoding": "gzip, deflate, br",
    "Origin": "https://www.imdb.com",
    "Referer": "https://www.imdb.com/",
}

# Single GraphQL query that replaces all 6 HTML page fetches. Returns every
# field the downstream parsers need: core metadata, credits, keywords with
# vote data, plot synopses/summaries, parental guide, and featured reviews.
_GRAPHQL_QUERY = """
query MovieData($id: ID!) {
  title(id: $id) {
    titleType { id }
    originalTitleText { text }
    certificate { rating ratingReason }
    plot { plotText { plainText } }
    ratingsSummary { aggregateRating voteCount }
    metacritic { metascore { score } }
    titleGenres { genres { genre { text } } }
    interests(first: 8) { edges { node { primaryText { text } } } }
    countriesOfOrigin { countries { text } }
    filmingLocations(first: 10) { edges { node { text } } }
    spokenLanguages { spokenLanguages { text } }
    productionBudget { budget { amount } }
    reviewSummary {
      overall { medium { value { plaidHtml } } }
      themes { label { value } sentiment }
    }
    companyCredits(first: 10, filter: { categories: ["production"] }) {
      edges { node { company { companyText { text } } } }
    }
    plots(first: 10) {
      edges { node { plotText { plainText } plotType } }
    }
    keywords(first: 50) {
      edges {
        node {
          keyword { text { text } }
          interestScore { usersInterested usersVoted }
        }
      }
    }
    parentsGuide { categories { category { text } severity { text } } }
    directors: credits(first: 10, filter: { categories: ["director"] }) {
      edges { node { name { nameText { text } } } }
    }
    writers: credits(first: 10, filter: { categories: ["writer"] }) {
      edges { node { name { nameText { text } } } }
    }
    cast: credits(first: 50, filter: { categories: ["actor", "actress"] }) {
      edges {
        node {
          name { nameText { text } }
          ... on Cast { characters { name } }
        }
      }
    }
    producers: credits(first: 10, filter: { categories: ["producer"] }) {
      edges { node { name { nameText { text } } } }
    }
    composers: credits(first: 5, filter: { categories: ["composer"] }) {
      edges { node { name { nameText { text } } } }
    }
    reviews(first: 10) {
      edges {
        node {
          summary { originalText }
          text { originalText { plainText } }
        }
      }
    }
  }
}
""".strip()


# ---------------------------------------------------------------------------
# Client initialization
# ---------------------------------------------------------------------------


def build_proxy_url() -> str:
    """
    Construct the DataImpulse proxy URL from environment variables.

    Format: http://{LOGIN}__cr.us:{PASSWORD}@{HOST}:{PORT}
    The __cr.us suffix activates US-only IP geo-targeting, ensuring
    IMDB serves MPAA maturity ratings, English content, and US-region
    metadata. Country targeting is free — no extra cost.
    """
    login = os.environ["DATA_IMPULSE_LOGIN"]
    password = os.environ["DATA_IMPULSE_PASSWORD"]
    host = os.environ.get("DATA_IMPULSE_HOST", "gw.dataimpulse.com")
    port = os.environ.get("DATA_IMPULSE_PORT", "823")
    return f"http://{login}__cr.us:{password}@{host}:{port}"


def create_client() -> httpx.AsyncClient:
    """
    Create the shared httpx.AsyncClient with proxy and connection limits.

    The caller is responsible for using this as an async context manager:
        async with create_client() as client:
            ...

    Connection pool is capped at 80 TCP connections to the proxy gateway,
    independently of the request-level semaphore.
    """
    proxy_url = build_proxy_url()
    limits = httpx.Limits(max_connections=_MAX_CONNECTIONS)
    timeout = httpx.Timeout(_REQUEST_TIMEOUT)

    return httpx.AsyncClient(
        proxy=proxy_url,
        headers=_SESSION_HEADERS,
        limits=limits,
        timeout=timeout,
        follow_redirects=True,
    )


def create_ua_generator() -> UserAgent:
    """
    Initialize the fake-useragent generator for desktop browsers only.

    Filtered to Chrome and Firefox — no Safari (low market share makes
    it unusual from residential IPs) and no mobile (different API behavior
    is possible with mobile User-Agents).
    """
    return UserAgent(browsers=["Chrome", "Firefox"], os=["Windows", "Mac OS X"])


# ---------------------------------------------------------------------------
# Core fetch function
# ---------------------------------------------------------------------------


async def fetch_movie(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    ua: UserAgent,
    imdb_id: str,
) -> tuple[FetchResult, dict | None]:
    """
    Fetch all IMDB data for a single movie via the GraphQL API.

    Sends a single POST request containing the full GraphQL query. The
    semaphore is acquired per-attempt and released before any backoff
    sleep, so retries do not hold a concurrency slot during the wait.

    On success: caches raw JSON to disk, returns (SUCCESS, title_data_dict).
    On HTTP 404 or null title: returns (HTTP_404, None) immediately.
    On failure after all retries: returns (FAILED, None).
    """
    tag = f"[{imdb_id}]"
    fetch_start = time.monotonic()
    last_failure_reason = "unknown"

    payload = {"query": _GRAPHQL_QUERY, "variables": {"id": imdb_id}}

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            async with semaphore:
                # Random delay after acquiring semaphore to break burst patterns
                # that could trigger detection even with IP rotation
                await asyncio.sleep(random.uniform(_MIN_DELAY, _MAX_DELAY))

                response = await client.post(
                    _GRAPHQL_URL,
                    json=payload,
                    headers={"User-Agent": ua.random},
                )
        except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError, ssl.SSLError) as exc:
            # Network-level failure — retry with exponential backoff
            last_failure_reason = type(exc).__name__
            if attempt < _MAX_RETRIES:
                backoff = random.uniform(0.2, 0.3)
                print(f"    {tag} Attempt {attempt}/{_MAX_RETRIES}: "
                      f"{last_failure_reason}, retrying in {backoff:.1f}s")
                await asyncio.sleep(backoff)
            continue

        # HTTP 404 — endpoint not found. Unlikely for GraphQL but handled.
        if response.status_code == 404:
            elapsed = time.monotonic() - fetch_start
            print(f"    {tag} -> HTTP_404 ({elapsed:.2f}s)")
            return (FetchResult.HTTP_404, None)

        # HTTP 403 or 429 — blocked or rate limited. Proxy rotation gives a
        # new IP automatically on the next attempt.
        if response.status_code in (403, 429):
            last_failure_reason = f"HTTP {response.status_code}"
            if attempt < _MAX_RETRIES:
                backoff = random.uniform(0.2, 0.3)
                print(f"    {tag} Attempt {attempt}/{_MAX_RETRIES}: "
                      f"{last_failure_reason}, retrying in {backoff:.1f}s")
                await asyncio.sleep(backoff)
            continue

        # HTTP 5xx — server error. Retry with backoff.
        if response.status_code >= 500:
            last_failure_reason = f"HTTP {response.status_code}"
            if attempt < _MAX_RETRIES:
                backoff = random.uniform(0.2, 0.3)
                print(f"    {tag} Attempt {attempt}/{_MAX_RETRIES}: "
                      f"{last_failure_reason}, retrying in {backoff:.1f}s")
                await asyncio.sleep(backoff)
            continue

        # HTTP 200 — parse the JSON response
        if response.status_code == 200:
            data = response.json()
            title_data = (data.get("data") or {}).get("title")

            # IMDB returns {"data": {"title": null}} for non-existent IDs
            # instead of an HTTP 404. Map this to our HTTP_404 result.
            if title_data is None:
                elapsed = time.monotonic() - fetch_start
                print(f"    {tag} -> HTTP_404 (null title, {elapsed:.2f}s)")
                return (FetchResult.HTTP_404, None)

            # Success — cache raw JSON response to disk before returning
            await _cache_json(imdb_id, title_data)
            return (FetchResult.SUCCESS, title_data)

        # Unexpected status code — treat as retriable
        last_failure_reason = f"unexpected HTTP {response.status_code}"
        if attempt < _MAX_RETRIES:
            backoff = 2 ** attempt + random.uniform(0, 1)
            print(f"    {tag} Attempt {attempt}/{_MAX_RETRIES}: "
                  f"{last_failure_reason}, retrying in {backoff:.1f}s")
            await asyncio.sleep(backoff)

    # All retries exhausted
    elapsed = time.monotonic() - fetch_start
    print(f"    {tag} -> FAILED ({last_failure_reason}, "
          f"exhausted {_MAX_RETRIES} retries, {elapsed:.2f}s)")
    return (FetchResult.FAILED, None)


# ---------------------------------------------------------------------------
# JSON response caching
# ---------------------------------------------------------------------------


def _ensure_cache_dir() -> None:
    """Create the JSON cache directory once at module level."""
    _JSON_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Create the cache directory at import time — avoids redundant
# mkdir syscalls during a full pipeline run.
_ensure_cache_dir()


async def _cache_json(imdb_id: str, title_data: dict) -> None:
    """
    Save the raw GraphQL title response to disk for debugging and re-runs.

    Path: ./ingestion_data/imdb_graphql/{imdb_id}.json

    Uses asyncio.to_thread to avoid blocking the event loop during
    the synchronous file write. This is a debugging cache, not a
    critical data artifact — if the process crashes mid-write, the
    file is simply truncated and will be overwritten on the next run.
    """
    cache_path = _JSON_CACHE_DIR / f"{imdb_id}.json"
    json_str = json.dumps(title_data, ensure_ascii=False)
    await asyncio.to_thread(cache_path.write_text, json_str, "utf-8")
