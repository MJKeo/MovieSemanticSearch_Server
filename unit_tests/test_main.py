"""Unit tests for api.main health check endpoint."""

from unittest.mock import AsyncMock, MagicMock

import pytest

# api.main imports redis and qdrant_client at module level; skip the entire
# module when those packages are not installed in the test environment.
pytest.importorskip("redis")
pytest.importorskip("qdrant_client")

from api.main import health_check  # noqa: E402


@pytest.mark.asyncio
async def test_health_check_all_services_ok(mocker) -> None:
    """health_check should return 'ok' for every service when all are reachable."""
    mocker.patch("api.main.check_postgres", new=AsyncMock(return_value="ok"))

    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mocker.patch("api.main.redis.Redis", return_value=mock_redis)

    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value = []
    mocker.patch("api.main.QdrantClient", return_value=mock_qdrant)

    result = await health_check()
    assert result == {"postgres": "ok", "redis": "ok", "qdrant": "ok"}


@pytest.mark.asyncio
async def test_health_check_postgres_failure_isolated(mocker) -> None:
    """health_check should report postgres error without affecting other services."""
    mocker.patch("api.main.check_postgres", new=AsyncMock(return_value="connection refused"))

    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mocker.patch("api.main.redis.Redis", return_value=mock_redis)

    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value = []
    mocker.patch("api.main.QdrantClient", return_value=mock_qdrant)

    result = await health_check()
    assert result["postgres"] == "connection refused"
    assert result["redis"] == "ok"
    assert result["qdrant"] == "ok"


@pytest.mark.asyncio
async def test_health_check_redis_failure_isolated(mocker) -> None:
    """health_check should report redis error without affecting other services."""
    mocker.patch("api.main.check_postgres", new=AsyncMock(return_value="ok"))

    mock_redis = MagicMock()
    mock_redis.ping.side_effect = ConnectionError("redis down")
    mocker.patch("api.main.redis.Redis", return_value=mock_redis)

    mock_qdrant = MagicMock()
    mock_qdrant.get_collections.return_value = []
    mocker.patch("api.main.QdrantClient", return_value=mock_qdrant)

    result = await health_check()
    assert result["postgres"] == "ok"
    assert result["redis"] == "redis down"
    assert result["qdrant"] == "ok"


@pytest.mark.asyncio
async def test_health_check_qdrant_failure_isolated(mocker) -> None:
    """health_check should report qdrant error without affecting other services."""
    mocker.patch("api.main.check_postgres", new=AsyncMock(return_value="ok"))

    mock_redis = MagicMock()
    mock_redis.ping.return_value = True
    mocker.patch("api.main.redis.Redis", return_value=mock_redis)

    mocker.patch("api.main.QdrantClient", side_effect=RuntimeError("qdrant unreachable"))

    result = await health_check()
    assert result["postgres"] == "ok"
    assert result["redis"] == "ok"
    assert result["qdrant"] == "qdrant unreachable"


@pytest.mark.asyncio
async def test_health_check_all_services_down(mocker) -> None:
    """health_check should report errors for all services when all are unreachable."""
    mocker.patch("api.main.check_postgres", new=AsyncMock(return_value="pg error"))

    mock_redis = MagicMock()
    mock_redis.ping.side_effect = ConnectionError("redis error")
    mocker.patch("api.main.redis.Redis", return_value=mock_redis)

    mocker.patch("api.main.QdrantClient", side_effect=RuntimeError("qdrant error"))

    result = await health_check()
    assert result["postgres"] == "pg error"
    assert result["redis"] == "redis error"
    assert result["qdrant"] == "qdrant error"
