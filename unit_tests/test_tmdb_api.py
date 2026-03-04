"""
Unit tests for db.tmdb — TMDB API client.

Covers access_token, AdaptiveRateLimiter (init, acquire, report_429, stats),
and fetch_movie_details (all HTTP status branches, retry logic, 429 handling).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from db.tmdb import (
    AdaptiveRateLimiter,
    TMDBFetchError,
    access_token,
    fetch_movie_details,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    status_code: int,
    json_data: dict | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """Build a real httpx.Response with controlled status, body, and headers."""
    response = httpx.Response(
        status_code=status_code,
        json=json_data,
        headers=headers or {},
        request=httpx.Request("GET", "https://api.themoviedb.org/3/movie/1"),
    )
    return response


# ---------------------------------------------------------------------------
# access_token
# ---------------------------------------------------------------------------


class TestAccessToken:
    """Tests for the access_token environment variable reader."""

    def test_returns_env_value(self, mocker) -> None:
        """access_token returns the TMDB_ACCESS_TOKEN environment variable."""
        mocker.patch("db.tmdb.os.getenv", return_value="my-secret-token")
        assert access_token() == "my-secret-token"

    def test_raises_when_env_var_missing(self, mocker) -> None:
        """access_token raises RuntimeError when TMDB_ACCESS_TOKEN is not set."""
        mocker.patch("db.tmdb.os.getenv", return_value=None)
        with pytest.raises(RuntimeError, match="TMDB_ACCESS_TOKEN"):
            access_token()

    def test_raises_when_env_var_empty_string(self, mocker) -> None:
        """access_token raises RuntimeError when TMDB_ACCESS_TOKEN is empty string."""
        mocker.patch("db.tmdb.os.getenv", return_value="")
        with pytest.raises(RuntimeError):
            access_token()


# ---------------------------------------------------------------------------
# AdaptiveRateLimiter — __init__
# ---------------------------------------------------------------------------


class TestAdaptiveRateLimiterInit:
    """Tests for AdaptiveRateLimiter constructor state initialization."""

    def test_init_default_values(self) -> None:
        """AdaptiveRateLimiter initializes with documented default values."""
        rl = AdaptiveRateLimiter()

        assert rl.current_rate == 36.0
        assert rl.max_rate == 40.0
        assert rl.burst == 5
        assert rl.clean_window == 120.0
        assert rl.tokens == 5.0
        assert rl.total_requests == 0
        assert rl.total_429s == 0
        assert rl._cooldown_until == 0.0
        assert rl._last_429_time == 0.0
        assert rl._last_increase_time == 0.0

    def test_init_custom_values(self) -> None:
        """AdaptiveRateLimiter accepts custom initial configuration."""
        rl = AdaptiveRateLimiter(
            initial_rate=10.0, max_rate=20.0, burst=2, clean_window=60.0,
        )

        assert rl.current_rate == 10.0
        assert rl.max_rate == 20.0
        assert rl.burst == 2
        assert rl.clean_window == 60.0
        assert rl.tokens == 2.0


# ---------------------------------------------------------------------------
# AdaptiveRateLimiter — acquire
# ---------------------------------------------------------------------------


class TestAdaptiveRateLimiterAcquire:
    """Tests for the acquire() async token-bucket method."""

    @pytest.mark.asyncio
    async def test_acquire_consumes_one_token(self, mocker) -> None:
        """acquire decrements tokens by 1.0 when tokens are available."""
        rl = AdaptiveRateLimiter()
        # Fix time so no refill happens
        mocker.patch("db.tmdb.time.monotonic", return_value=rl._last_refill)
        sleep_mock = mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()

        assert rl.tokens == pytest.approx(4.0)
        assert rl.total_requests == 1
        sleep_mock.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_acquire_sleeps_when_tokens_below_one(self, mocker) -> None:
        """acquire sleeps proportionally when token bucket is below 1.0."""
        rl = AdaptiveRateLimiter()
        rl.tokens = 0.5
        base_time = rl._last_refill

        # Return same time so no refill, then return updated time after sleep
        mocker.patch("db.tmdb.time.monotonic", return_value=base_time)
        sleep_mock = mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()

        expected_wait = (1.0 - 0.5) / 36.0
        sleep_mock.assert_awaited_once()
        assert sleep_mock.await_args[0][0] == pytest.approx(expected_wait)
        assert rl.tokens == 0.0

    @pytest.mark.asyncio
    async def test_acquire_refills_tokens_based_on_elapsed_time(self, mocker) -> None:
        """acquire refills tokens proportional to time elapsed since last refill."""
        rl = AdaptiveRateLimiter()
        rl.tokens = 0.0
        base_time = rl._last_refill
        # 0.1s elapsed at 36 req/s = 3.6 tokens refilled
        mocker.patch("db.tmdb.time.monotonic", return_value=base_time + 0.1)
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()

        # Refill: 0.0 + 0.1 * 36.0 = 3.6, then consume 1 = 2.6
        assert rl.tokens == pytest.approx(2.6)

    @pytest.mark.asyncio
    async def test_acquire_caps_tokens_at_burst(self, mocker) -> None:
        """acquire never refills tokens above the burst cap."""
        rl = AdaptiveRateLimiter()
        rl.tokens = 0.0
        base_time = rl._last_refill
        # 10s elapsed would give 360 tokens, but capped at burst=5
        mocker.patch("db.tmdb.time.monotonic", return_value=base_time + 10.0)
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()

        # Capped at burst=5, then consume 1 = 4.0
        assert rl.tokens == pytest.approx(4.0)

    @pytest.mark.asyncio
    async def test_acquire_honours_global_cooldown(self, mocker) -> None:
        """acquire sleeps until cooldown_until when in cooldown period."""
        rl = AdaptiveRateLimiter()
        rl._cooldown_until = 200.0
        # Align _last_refill with mock time so elapsed is non-negative and
        # tokens stay above 1.0 (avoids an extra time.monotonic call in the
        # token-wait branch).
        rl._last_refill = 100.0

        # now=100 (in cooldown) → sleep 100s → now=200 (cooldown expired).
        # Default of 200.0 prevents StopIteration if monotonic is called
        # during event-loop teardown.
        times = iter([100.0, 200.0])
        mocker.patch(
            "db.tmdb.time.monotonic",
            side_effect=lambda: next(times, 200.0),
        )
        sleep_mock = mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()

        # First sleep should be the cooldown wait (100s)
        first_sleep = sleep_mock.await_args_list[0][0][0]
        assert first_sleep == pytest.approx(100.0)

    @pytest.mark.asyncio
    async def test_acquire_increases_rate_after_clean_window(self, mocker, capsys) -> None:
        """acquire increases rate by 5% after clean_window with no 429s."""
        rl = AdaptiveRateLimiter(initial_rate=36.0, max_rate=40.0, clean_window=120.0)
        rl._last_429_time = 10.0  # Had a 429 at t=10
        rl._last_increase_time = 0.0  # Never increased before

        # now=200 => 200-10=190 > 120 (clean window) and 200-0=200 > 120
        mocker.patch("db.tmdb.time.monotonic", return_value=200.0)
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()

        assert rl.current_rate == pytest.approx(36.0 * 1.05)
        assert rl._last_increase_time == 200.0
        assert "increased" in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_acquire_rate_increase_capped_at_max_rate(self, mocker) -> None:
        """acquire rate increase never exceeds max_rate."""
        rl = AdaptiveRateLimiter(initial_rate=39.5, max_rate=40.0, clean_window=10.0)
        rl._last_429_time = 1.0
        rl._last_increase_time = 0.0

        mocker.patch("db.tmdb.time.monotonic", return_value=100.0)
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()

        # 39.5 * 1.05 = 41.475, capped at 40.0
        assert rl.current_rate == 40.0

    @pytest.mark.asyncio
    async def test_acquire_no_increase_when_no_429_history(self, mocker) -> None:
        """acquire does not increase rate if no 429 has ever been received."""
        rl = AdaptiveRateLimiter(initial_rate=36.0)
        # _last_429_time stays 0.0 (default)

        mocker.patch("db.tmdb.time.monotonic", return_value=rl._last_refill + 300.0)
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()

        assert rl.current_rate == 36.0  # Unchanged

    @pytest.mark.asyncio
    async def test_acquire_no_increase_within_same_increase_interval(self, mocker) -> None:
        """acquire only bumps rate once per increase_interval via _last_increase_time guard."""
        rl = AdaptiveRateLimiter(initial_rate=36.0, clean_window=120.0, increase_interval=10.0)
        rl._last_429_time = 10.0
        rl._last_increase_time = 195.0  # Increased 5s ago

        # now=200 => 200-10=190 > 120 (clean), but 200-195=5 < 10 (too soon)
        mocker.patch("db.tmdb.time.monotonic", return_value=200.0)
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()

        assert rl.current_rate == 36.0  # Unchanged

    @pytest.mark.asyncio
    async def test_acquire_no_increase_when_at_max_rate(self, mocker, capsys) -> None:
        """acquire skips rate increase when current_rate equals max_rate."""
        rl = AdaptiveRateLimiter(initial_rate=40.0, max_rate=40.0, clean_window=10.0)
        rl._last_429_time = 1.0
        rl._last_increase_time = 0.0

        mocker.patch("db.tmdb.time.monotonic", return_value=100.0)
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()

        assert rl.current_rate == 40.0
        assert "increased" not in capsys.readouterr().out

    @pytest.mark.asyncio
    async def test_acquire_increments_total_requests(self, mocker) -> None:
        """acquire increments total_requests on each call."""
        rl = AdaptiveRateLimiter()
        mocker.patch("db.tmdb.time.monotonic", return_value=rl._last_refill)
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        await rl.acquire()
        await rl.acquire()
        await rl.acquire()

        assert rl.total_requests == 3


# ---------------------------------------------------------------------------
# AdaptiveRateLimiter — report_429
# ---------------------------------------------------------------------------


class TestAdaptiveRateLimiterReport429:
    """Tests for the report_429 rate-drop and cooldown method."""

    def test_drops_rate_by_10_percent(self, mocker) -> None:
        """report_429 reduces current_rate by 10%."""
        mocker.patch("db.tmdb.time.monotonic", return_value=100.0)
        rl = AdaptiveRateLimiter(initial_rate=36.0)

        rl.report_429()

        assert rl.current_rate == pytest.approx(32.4)

    def test_rate_floor_at_one(self, mocker) -> None:
        """report_429 never reduces rate below 1.0."""
        mocker.patch("db.tmdb.time.monotonic", return_value=100.0)
        rl = AdaptiveRateLimiter(initial_rate=1.0)

        rl.report_429()

        assert rl.current_rate == 1.0

    def test_sets_cooldown_until(self, mocker) -> None:
        """report_429 sets cooldown_until to now + retry_after."""
        mocker.patch("db.tmdb.time.monotonic", return_value=100.0)
        rl = AdaptiveRateLimiter()

        rl.report_429(retry_after=3.0)

        assert rl._cooldown_until == 103.0

    def test_cooldown_merging_keeps_later_deadline(self, mocker) -> None:
        """report_429 keeps existing cooldown when it exceeds new deadline."""
        mocker.patch("db.tmdb.time.monotonic", return_value=105.0)
        rl = AdaptiveRateLimiter()
        rl._cooldown_until = 110.0  # Existing deadline is later

        rl.report_429(retry_after=2.0)  # New deadline = 107.0 < 110.0

        assert rl._cooldown_until == 110.0

    def test_cooldown_extends_when_new_is_later(self, mocker) -> None:
        """report_429 extends cooldown when new deadline exceeds current."""
        mocker.patch("db.tmdb.time.monotonic", return_value=105.0)
        rl = AdaptiveRateLimiter()
        rl._cooldown_until = 102.0  # Old deadline already passed

        rl.report_429(retry_after=5.0)  # New deadline = 110.0

        assert rl._cooldown_until == 110.0

    def test_increments_total_429s(self, mocker) -> None:
        """report_429 increments total_429s counter."""
        mocker.patch("db.tmdb.time.monotonic", return_value=100.0)
        rl = AdaptiveRateLimiter()

        rl.report_429()
        rl.report_429()
        rl.report_429()

        assert rl.total_429s == 3

    def test_updates_last_429_time(self, mocker) -> None:
        """report_429 records current monotonic time as _last_429_time."""
        mocker.patch("db.tmdb.time.monotonic", return_value=42.0)
        rl = AdaptiveRateLimiter()

        rl.report_429()

        assert rl._last_429_time == 42.0

    def test_prints_status_message(self, mocker, capsys) -> None:
        """report_429 prints rate reduction and cooldown information."""
        mocker.patch("db.tmdb.time.monotonic", return_value=100.0)
        rl = AdaptiveRateLimiter(initial_rate=36.0)

        rl.report_429(retry_after=3.0)

        output = capsys.readouterr().out
        assert "429 received" in output
        assert "36.0" in output  # old rate
        assert "32.4" in output  # new rate

    def test_default_retry_after(self, mocker) -> None:
        """report_429 uses default retry_after of 2.0 seconds."""
        mocker.patch("db.tmdb.time.monotonic", return_value=100.0)
        rl = AdaptiveRateLimiter()

        rl.report_429()  # No explicit retry_after

        assert rl._cooldown_until == 102.0


# ---------------------------------------------------------------------------
# AdaptiveRateLimiter — stats
# ---------------------------------------------------------------------------


class TestAdaptiveRateLimiterStats:
    """Tests for the stats summary method."""

    def test_format_with_requests(self) -> None:
        """stats returns formatted string with rate, total, and 429 percentage."""
        rl = AdaptiveRateLimiter()
        rl.total_requests = 100
        rl.total_429s = 5

        result = rl.stats()

        assert "Rate: 36.0" in result
        assert "Total: 100" in result
        assert "5.00%" in result

    def test_zero_requests_no_division_error(self) -> None:
        """stats avoids division by zero when total_requests is zero."""
        rl = AdaptiveRateLimiter()
        # Default: total_requests=0, total_429s=0

        result = rl.stats()

        assert "0.00%" in result


# ---------------------------------------------------------------------------
# fetch_movie_details
# ---------------------------------------------------------------------------


class TestFetchMovieDetails:
    """Tests for the async fetch_movie_details TMDB endpoint wrapper."""

    def _mock_rate_limiter(self) -> AdaptiveRateLimiter:
        """Create a rate limiter with acquire() as a no-op AsyncMock."""
        rl = MagicMock(spec=AdaptiveRateLimiter)
        rl.acquire = AsyncMock()
        rl.report_429 = MagicMock()
        return rl

    def _mock_client(self, responses: list[httpx.Response | Exception]) -> httpx.AsyncClient:
        """Create a mock httpx client returning responses in sequence."""
        client = MagicMock(spec=httpx.AsyncClient)
        client.get = AsyncMock(side_effect=responses)
        return client

    @pytest.mark.asyncio
    async def test_200_returns_parsed_json(self) -> None:
        """fetch_movie_details returns parsed JSON dict on 200 response."""
        data = {"id": 123, "title": "Test Movie"}
        client = self._mock_client([_make_response(200, json_data=data)])
        rl = self._mock_rate_limiter()

        result = await fetch_movie_details(client, rl, tmdb_id=123)

        assert result == data

    @pytest.mark.asyncio
    async def test_404_returns_none(self) -> None:
        """fetch_movie_details returns None on 404 response."""
        client = self._mock_client([_make_response(404)])
        rl = self._mock_rate_limiter()

        result = await fetch_movie_details(client, rl, tmdb_id=999)

        assert result is None

    @pytest.mark.asyncio
    async def test_429_calls_report_429_and_retries(self, mocker) -> None:
        """fetch_movie_details reports 429 to rate limiter and retries."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        data = {"id": 1, "title": "OK"}
        client = self._mock_client([
            _make_response(429, headers={"Retry-After": "2"}),
            _make_response(200, json_data=data),
        ])
        rl = self._mock_rate_limiter()

        result = await fetch_movie_details(client, rl, tmdb_id=1)

        assert result == data
        rl.report_429.assert_called_once_with(2.0)

    @pytest.mark.asyncio
    async def test_429_parses_retry_after_header(self, mocker) -> None:
        """fetch_movie_details parses Retry-After header for 429 responses."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([
            _make_response(429, headers={"Retry-After": "5"}),
            _make_response(200, json_data={"id": 1}),
        ])
        rl = self._mock_rate_limiter()

        await fetch_movie_details(client, rl, tmdb_id=1)

        rl.report_429.assert_called_once_with(5.0)

    @pytest.mark.asyncio
    async def test_429_uses_default_retry_after_when_header_missing(self, mocker) -> None:
        """fetch_movie_details defaults to 2s retry_after when Retry-After header absent."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([
            _make_response(429),  # No Retry-After header
            _make_response(200, json_data={"id": 1}),
        ])
        rl = self._mock_rate_limiter()

        await fetch_movie_details(client, rl, tmdb_id=1)

        rl.report_429.assert_called_once_with(2.0)

    @pytest.mark.asyncio
    async def test_429_does_not_consume_transient_retry_budget(self, mocker) -> None:
        """fetch_movie_details retries indefinitely on 429s without exhausting max_retries."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        # 5 consecutive 429s then a 200, with max_retries=3.
        # If 429s consumed the budget, this would raise after the 3rd.
        responses = [_make_response(429) for _ in range(5)]
        responses.append(_make_response(200, json_data={"id": 1}))

        client = self._mock_client(responses)
        rl = self._mock_rate_limiter()

        result = await fetch_movie_details(client, rl, tmdb_id=1, max_attempts=3)

        assert result == {"id": 1}
        assert rl.report_429.call_count == 5

    @pytest.mark.asyncio
    async def test_5xx_retries_with_backoff_and_raises_after_exhaustion(self, mocker) -> None:
        """fetch_movie_details raises TMDBFetchError after max_retries on 5xx."""
        sleep_mock = mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([
            _make_response(500),
            _make_response(502),
            _make_response(503),
        ])
        rl = self._mock_rate_limiter()

        with pytest.raises(TMDBFetchError, match="Server error"):
            await fetch_movie_details(client, rl, tmdb_id=1, max_attempts=3)

        # Backoff sleeps: 1.0 (after 1st 500), 2.0 (after 2nd 502), then raises on 3rd
        assert sleep_mock.await_count == 2
        assert sleep_mock.await_args_list[0][0][0] == pytest.approx(1.0)
        assert sleep_mock.await_args_list[1][0][0] == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_5xx_recovers_on_subsequent_200(self, mocker) -> None:
        """fetch_movie_details succeeds if 200 follows a 5xx before retries exhausted."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([
            _make_response(500),
            _make_response(200, json_data={"id": 1}),
        ])
        rl = self._mock_rate_limiter()

        result = await fetch_movie_details(client, rl, tmdb_id=1, max_attempts=3)

        assert result == {"id": 1}

    @pytest.mark.asyncio
    async def test_5xx_backoff_formula(self, mocker) -> None:
        """fetch_movie_details uses exponential backoff: 1.0 * 2^(attempt-1) seconds."""
        sleep_mock = mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([
            _make_response(500),
            _make_response(500),
            _make_response(200, json_data={"id": 1}),
        ])
        rl = self._mock_rate_limiter()

        await fetch_movie_details(client, rl, tmdb_id=1, max_attempts=3)

        sleep_durations = [call[0][0] for call in sleep_mock.await_args_list]
        assert sleep_durations == [pytest.approx(1.0), pytest.approx(2.0)]

    @pytest.mark.asyncio
    async def test_transport_error_retries_and_raises(self, mocker) -> None:
        """fetch_movie_details raises TMDBFetchError after exhausting retries on TransportError."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([
            httpx.TransportError("conn reset"),
            httpx.TransportError("conn reset"),
            httpx.TransportError("conn reset"),
        ])
        rl = self._mock_rate_limiter()

        with pytest.raises(TMDBFetchError, match="Transport error"):
            await fetch_movie_details(client, rl, tmdb_id=1, max_attempts=3)

    @pytest.mark.asyncio
    async def test_transport_error_recovers_on_success(self, mocker) -> None:
        """fetch_movie_details succeeds if request succeeds after a TransportError."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([
            httpx.TransportError("timeout"),
            _make_response(200, json_data={"id": 1}),
        ])
        rl = self._mock_rate_limiter()

        result = await fetch_movie_details(client, rl, tmdb_id=1, max_attempts=3)

        assert result == {"id": 1}

    @pytest.mark.asyncio
    async def test_timeout_error_retries_and_raises(self, mocker) -> None:
        """fetch_movie_details raises TMDBFetchError after exhausting retries on TimeoutException."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([
            httpx.TimeoutException("read timeout"),
            httpx.TimeoutException("read timeout"),
            httpx.TimeoutException("read timeout"),
        ])
        rl = self._mock_rate_limiter()

        with pytest.raises(TMDBFetchError, match="Transport error"):
            await fetch_movie_details(client, rl, tmdb_id=1, max_attempts=3)

    @pytest.mark.asyncio
    async def test_unexpected_status_raises_immediately(self, mocker) -> None:
        """fetch_movie_details raises TMDBFetchError immediately on unexpected status code."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([_make_response(403)])
        rl = self._mock_rate_limiter()

        with pytest.raises(TMDBFetchError, match="Unexpected HTTP 403"):
            await fetch_movie_details(client, rl, tmdb_id=1)

    @pytest.mark.asyncio
    async def test_url_includes_tmdb_id_and_append_params(self, mocker) -> None:
        """fetch_movie_details constructs correct URL with append_to_response params."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([_make_response(200, json_data={"id": 42})])
        rl = self._mock_rate_limiter()

        await fetch_movie_details(client, rl, tmdb_id=42)

        call_args = client.get.call_args
        assert "/movie/42" in call_args[0][0]
        assert "append_to_response" in call_args[1]["params"]
        assert "credits" in call_args[1]["params"]["append_to_response"]

    @pytest.mark.asyncio
    async def test_max_retries_zero_raises_immediately(self, mocker) -> None:
        """fetch_movie_details raises TMDBFetchError when max_retries is 0."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([])  # Never called
        rl = self._mock_rate_limiter()

        with pytest.raises(TMDBFetchError, match="Exhausted 0 retries"):
            await fetch_movie_details(client, rl, tmdb_id=1, max_attempts=0)

    @pytest.mark.asyncio
    async def test_mixed_429_and_5xx_sequence(self, mocker) -> None:
        """fetch_movie_details handles interleaved 429 and 5xx correctly."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        # 429s don't count toward budget, only 5xx does.
        # With max_retries=3: 2 transient (5xx) + any number of 429s should succeed.
        client = self._mock_client([
            _make_response(429),
            _make_response(500),
            _make_response(429),
            _make_response(500),
            _make_response(200, json_data={"id": 1}),
        ])
        rl = self._mock_rate_limiter()

        result = await fetch_movie_details(client, rl, tmdb_id=1, max_attempts=3)

        assert result == {"id": 1}
        assert rl.report_429.call_count == 2

    @pytest.mark.asyncio
    async def test_429_prints_message_with_movie_id(self, mocker, capsys) -> None:
        """fetch_movie_details prints message with tmdb_id on 429."""
        mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([
            _make_response(429),
            _make_response(200, json_data={"id": 42}),
        ])
        rl = self._mock_rate_limiter()

        await fetch_movie_details(client, rl, tmdb_id=42)

        output = capsys.readouterr().out
        assert "tmdb_id=42" in output

    @pytest.mark.asyncio
    async def test_transport_error_backoff_formula(self, mocker) -> None:
        """fetch_movie_details uses exponential backoff for transport errors."""
        sleep_mock = mocker.patch("db.tmdb.asyncio.sleep", new_callable=AsyncMock)

        client = self._mock_client([
            httpx.TransportError("err"),
            httpx.TransportError("err"),
            _make_response(200, json_data={"id": 1}),
        ])
        rl = self._mock_rate_limiter()

        await fetch_movie_details(client, rl, tmdb_id=1, max_attempts=3)

        sleep_durations = [call[0][0] for call in sleep_mock.await_args_list]
        assert sleep_durations == [pytest.approx(1.0), pytest.approx(2.0)]
