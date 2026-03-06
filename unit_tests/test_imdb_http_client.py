"""
Unit tests for movie_ingestion.imdb_scraping.http_client.

Tests the proxy URL construction, client creation, UA generator, and the
core fetch_movie coroutine with mocked httpx responses.
"""

import asyncio
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from movie_ingestion.imdb_scraping.http_client import (
    FetchResult,
    build_proxy_url,
    create_client,
    create_ua_generator,
    fetch_movie,
    _GRAPHQL_URL,
    _GRAPHQL_QUERY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Minimal valid GraphQL response with a non-null title
_VALID_TITLE_DATA = {
    "originalTitleText": {"text": "Test Movie"},
    "ratingsSummary": {"aggregateRating": 7.5, "voteCount": 1000},
}

_VALID_GRAPHQL_RESPONSE = {"data": {"title": _VALID_TITLE_DATA}}

# GraphQL response for a non-existent IMDB ID (null title)
_NULL_TITLE_RESPONSE = {"data": {"title": None}}


def _mock_json_response(status_code: int, json_data: dict | None = None) -> httpx.Response:
    """Build a real httpx.Response with a JSON body."""
    content = json.dumps(json_data or {}).encode("utf-8")
    return httpx.Response(
        status_code=status_code,
        content=content,
        headers={"content-type": "application/json"},
        request=httpx.Request("POST", _GRAPHQL_URL),
    )


def _mock_ua() -> MagicMock:
    """Return a mock UserAgent whose .random property returns a fixed string."""
    ua = MagicMock()
    ua.random = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) TestBrowser/1.0"
    return ua


# ---------------------------------------------------------------------------
# Tests: build_proxy_url
# ---------------------------------------------------------------------------


class TestBuildProxyUrl:
    """Tests for DataImpulse proxy URL construction."""

    def test_constructs_url_from_env_vars(self) -> None:
        """Returns http://{login}__cr.us:{password}@{host}:{port}."""
        env = {
            "DATA_IMPULSE_LOGIN": "mylogin",
            "DATA_IMPULSE_PASSWORD": "mypass",
            "DATA_IMPULSE_HOST": "proxy.example.com",
            "DATA_IMPULSE_PORT": "9999",
        }
        with patch.dict(os.environ, env, clear=False):
            url = build_proxy_url()
        assert url == "http://mylogin__cr.us:mypass@proxy.example.com:9999"

    def test_uses_default_host_and_port(self) -> None:
        """Missing host/port env vars use defaults gw.dataimpulse.com:823."""
        env = {
            "DATA_IMPULSE_LOGIN": "mylogin",
            "DATA_IMPULSE_PASSWORD": "mypass",
        }
        # Remove host/port if they exist to test defaults
        clean_env = {k: v for k, v in os.environ.items()
                     if k not in ("DATA_IMPULSE_HOST", "DATA_IMPULSE_PORT")}
        clean_env.update(env)
        with patch.dict(os.environ, clean_env, clear=True):
            url = build_proxy_url()
        assert url == "http://mylogin__cr.us:mypass@gw.dataimpulse.com:823"

    def test_raises_on_missing_login(self) -> None:
        """Missing DATA_IMPULSE_LOGIN raises KeyError."""
        env = {"DATA_IMPULSE_PASSWORD": "mypass"}
        clean_env = {k: v for k, v in os.environ.items()
                     if k != "DATA_IMPULSE_LOGIN"}
        clean_env.update(env)
        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(KeyError):
                build_proxy_url()

    def test_raises_on_missing_password(self) -> None:
        """Missing DATA_IMPULSE_PASSWORD raises KeyError."""
        env = {"DATA_IMPULSE_LOGIN": "mylogin"}
        clean_env = {k: v for k, v in os.environ.items()
                     if k != "DATA_IMPULSE_PASSWORD"}
        clean_env.update(env)
        with patch.dict(os.environ, clean_env, clear=True):
            with pytest.raises(KeyError):
                build_proxy_url()


# ---------------------------------------------------------------------------
# Tests: create_client
# ---------------------------------------------------------------------------


class TestCreateClient:
    """Tests for httpx.AsyncClient creation."""

    def test_returns_async_client(self) -> None:
        """Returns an httpx.AsyncClient instance with no cookies."""
        env = {
            "DATA_IMPULSE_LOGIN": "login",
            "DATA_IMPULSE_PASSWORD": "pass",
        }
        with patch.dict(os.environ, env, clear=False):
            client = create_client()
        assert isinstance(client, httpx.AsyncClient)
        # No cookies should be set (no WAF token needed)
        assert len(list(client.cookies.jar)) == 0


# ---------------------------------------------------------------------------
# Tests: create_ua_generator
# ---------------------------------------------------------------------------


class TestCreateUaGenerator:
    """Tests for the fake-useragent UserAgent generator."""

    def test_returns_user_agent_instance(self) -> None:
        """Returns a UserAgent instance."""
        from fake_useragent import UserAgent
        ua = create_ua_generator()
        assert isinstance(ua, UserAgent)


# ---------------------------------------------------------------------------
# Tests: fetch_movie
# ---------------------------------------------------------------------------


class TestFetchMovie:
    """Tests for the core async fetch_movie function with mocked HTTP."""

    @pytest.mark.asyncio
    async def test_success_returns_title_data(self) -> None:
        """HTTP 200 with valid GraphQL JSON returns (SUCCESS, title_data)."""
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_json_response(200, _VALID_GRAPHQL_RESPONSE))
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client._cache_json", new_callable=AsyncMock):
            with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
                result, data = await fetch_movie(
                    client, semaphore, _mock_ua(), "tt0000001"
                )
        assert result == FetchResult.SUCCESS
        assert data == _VALID_TITLE_DATA

    @pytest.mark.asyncio
    async def test_success_caches_json(self) -> None:
        """On success, _cache_json is called with the title data."""
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_json_response(200, _VALID_GRAPHQL_RESPONSE))
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client._cache_json", new_callable=AsyncMock) as mock_cache:
            with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
                await fetch_movie(
                    client, semaphore, _mock_ua(), "tt0000001"
                )
        mock_cache.assert_called_once_with("tt0000001", _VALID_TITLE_DATA)

    @pytest.mark.asyncio
    async def test_null_title_returns_http_404(self) -> None:
        """HTTP 200 with null title returns (HTTP_404, None) — movie doesn't exist."""
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_json_response(200, _NULL_TITLE_RESPONSE))
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
            result, data = await fetch_movie(
                client, semaphore, _mock_ua(), "tt9999999"
            )
        assert result == FetchResult.HTTP_404
        assert data is None
        # No retry — immediate return
        assert client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_http_404_returns_immediately(self) -> None:
        """HTTP 404 returns (HTTP_404, None) with no retry."""
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_json_response(404))
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
            result, data = await fetch_movie(
                client, semaphore, _mock_ua(), "tt0000001"
            )
        assert result == FetchResult.HTTP_404
        assert data is None
        assert client.post.await_count == 1

    @pytest.mark.asyncio
    async def test_http_403_retries(self) -> None:
        """HTTP 403 triggers retry (up to 3 attempts)."""
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_json_response(403))
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
            result, data = await fetch_movie(
                client, semaphore, _mock_ua(), "tt0000001"
            )
        assert result == FetchResult.FAILED
        assert client.post.await_count == 3

    @pytest.mark.asyncio
    async def test_http_5xx_retries(self) -> None:
        """HTTP 500 triggers retry with backoff."""
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_json_response(500))
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
            result, data = await fetch_movie(
                client, semaphore, _mock_ua(), "tt0000001"
            )
        assert result == FetchResult.FAILED
        assert client.post.await_count == 3

    @pytest.mark.asyncio
    async def test_network_error_retries(self) -> None:
        """httpx.NetworkError triggers retry with backoff."""
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.NetworkError("Connection reset"))
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
            result, data = await fetch_movie(
                client, semaphore, _mock_ua(), "tt0000001"
            )
        assert result == FetchResult.FAILED
        assert client.post.await_count == 3

    @pytest.mark.asyncio
    async def test_timeout_error_retries(self) -> None:
        """httpx.TimeoutException triggers retry with backoff."""
        client = AsyncMock()
        client.post = AsyncMock(side_effect=httpx.TimeoutException("Read timed out"))
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
            result, data = await fetch_movie(
                client, semaphore, _mock_ua(), "tt0000001"
            )
        assert result == FetchResult.FAILED
        assert client.post.await_count == 3

    @pytest.mark.asyncio
    async def test_all_retries_exhausted_returns_failed(self) -> None:
        """3 consecutive 429s returns (FAILED, None)."""
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_json_response(429))
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
            result, data = await fetch_movie(
                client, semaphore, _mock_ua(), "tt0000001"
            )
        assert result == FetchResult.FAILED
        assert data is None

    @pytest.mark.asyncio
    async def test_retry_then_success(self) -> None:
        """First attempt 500, second attempt 200 with data returns (SUCCESS, data)."""
        client = AsyncMock()
        client.post = AsyncMock(side_effect=[
            _mock_json_response(500),
            _mock_json_response(200, _VALID_GRAPHQL_RESPONSE),
        ])
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client._cache_json", new_callable=AsyncMock):
            with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
                result, data = await fetch_movie(
                    client, semaphore, _mock_ua(), "tt0000001"
                )
        assert result == FetchResult.SUCCESS
        assert data == _VALID_TITLE_DATA
        assert client.post.await_count == 2

    @pytest.mark.asyncio
    async def test_posts_correct_graphql_payload(self) -> None:
        """The POST body contains the GraphQL query and imdb_id variable."""
        client = AsyncMock()
        client.post = AsyncMock(return_value=_mock_json_response(200, _VALID_GRAPHQL_RESPONSE))
        semaphore = asyncio.Semaphore(1)

        with patch("movie_ingestion.imdb_scraping.http_client._cache_json", new_callable=AsyncMock):
            with patch("movie_ingestion.imdb_scraping.http_client.asyncio.sleep", new_callable=AsyncMock):
                await fetch_movie(
                    client, semaphore, _mock_ua(), "tt1234567"
                )

        # Verify the POST call
        call_kwargs = client.post.call_args[1]
        assert call_kwargs["json"]["query"] == _GRAPHQL_QUERY
        assert call_kwargs["json"]["variables"] == {"id": "tt1234567"}
        # URL should be the GraphQL endpoint
        call_args = client.post.call_args[0]
        assert call_args[0] == _GRAPHQL_URL
