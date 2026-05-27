"""
Redis async connection pool for the API service.

Uses redis.asyncio with an explicit ConnectionPool. decode_responses is False
because the embedding cache stores raw packed binary — not strings.
"""

import logging
import os
import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

logger = logging.getLogger(__name__)

_redis_pool: ConnectionPool | None = None
_redis_client: aioredis.Redis | None = None


def get_redis_client() -> aioredis.Redis:
    """Return the shared async Redis client backed by a connection pool."""
    if _redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() at startup.")
    return _redis_client


def redis_key(*parts: str) -> str:
    """Build an environment-prefixed Redis key from one or more parts.

    `REDIS_ENV` is resolved at call time, not import time, so the prefix
    picks up whatever the process environment holds when the key is
    built. This avoids a foot-gun where dotenv loading (or any other
    env setup) happens after `db.redis` is imported and a module-level
    constant would have already frozen the default.
    """
    env_prefix = os.getenv("REDIS_ENV", "unknown_env")
    return f"{env_prefix}:{':'.join(parts)}"


async def init_redis(
    host: str = os.getenv("REDIS_HOST", "redis"),
    port: int = int(os.getenv("REDIS_PORT", "6379")),
    max_connections: int = 10,
) -> None:
    """Call once at application startup (e.g. FastAPI lifespan)."""
    global _redis_pool, _redis_client
    _redis_pool = ConnectionPool(
        host=host,
        port=port,
        max_connections=max_connections,
        decode_responses=False,  # Embedding cache uses raw binary — never decode globally
    )
    _redis_client = aioredis.Redis(connection_pool=_redis_pool)
    await _redis_client.ping()  # Fail fast if Redis is unreachable at startup


async def close_redis() -> None:
    """Call at application shutdown."""
    global _redis_pool, _redis_client
    if _redis_client:
        await _redis_client.aclose()
    if _redis_pool:
        await _redis_pool.aclose()
    _redis_client = None
    _redis_pool = None


async def check_redis() -> str:
    """Ping Redis and return 'ok' or an error message string."""
    try:
        client = get_redis_client()
        await client.ping()
        return "ok"
    except Exception as e:
        return str(e)


# ---------------------------------------------------------------------------
# Trending scores
# ---------------------------------------------------------------------------

_TRENDING_LIVE_KEY = "trending:current"
_TRENDING_STAGING_KEY = "trending:next"


async def write_trending_scores(scores: dict[int, float]) -> None:
    """
    Atomically replace the live trending Hash with the given scores.

    Pattern: DEL staging → HSET staging → RENAME staging → live.
    RENAME is atomic in Redis, so the live key is never in a partial state.

    Args:
        scores: Mapping of TMDB movie_id → precomputed trending score [0, 1].
    """
    client = get_redis_client()
    live_key = redis_key(_TRENDING_LIVE_KEY)
    staging_key = redis_key(_TRENDING_STAGING_KEY)

    # Encode as str→str mapping for Redis (decode_responses=False means we
    # send plain Python strings and let redis-py encode them to bytes).
    mapping: dict[str, str] = {str(movie_id): str(score) for movie_id, score in scores.items()}

    # Prepare the staging key atomically (pipeline cuts round trips)
    pipe = client.pipeline(transaction=True)
    pipe.delete(staging_key)
    pipe.hset(staging_key, mapping=mapping)
    await pipe.execute()

    # Atomic swap: RENAME replaces live_key in a single operation
    await client.rename(staging_key, live_key)


async def read_trending_scores() -> dict[int, float]:
    """
    Load the full trending Hash from Redis into an in-memory dict.

    Returns an empty dict (with a WARNING log) if the key is absent —
    callers should treat missing trending data as graceful degradation,
    not an error.

    Returns:
        Mapping of TMDB movie_id (int) → trending score (float).
    """
    client = get_redis_client()
    live_key = redis_key(_TRENDING_LIVE_KEY)

    # hgetall returns dict[bytes, bytes] when decode_responses=False
    raw: dict[bytes, bytes] = await client.hgetall(live_key)

    if not raw:
        logger.warning("Redis key '%s' is absent or empty — trending scores unavailable", live_key)
        return {}

    return {int(k): float(v) for k, v in raw.items()}


# ---------------------------------------------------------------------------
# Movie details cache (backs the /movie_details API endpoint)
# ---------------------------------------------------------------------------
#
# Caches the *curated* msgspec-encoded MovieDetails payload (not the raw
# TMDB response), so warm hits skip both the TMDB round-trip and the
# build/encode step. 24h TTL is short enough that locked-in fields
# (reception_score, watch providers) stay reasonably fresh without
# hammering TMDB on every detail view.
#
# Namespace: `tmdb:movie:{id}` per docs/conventions.md. The cached value
# is the curated wire payload (not raw TMDB JSON), but the namespace
# stays under `tmdb:movie:` because that's the documented home for
# per-movie TMDB-derived cache entries.

_MOVIE_DETAILS_KEY_PARTS: tuple[str, ...] = ("tmdb", "movie")
_MOVIE_DETAILS_TTL_SECONDS = 24 * 60 * 60  # 24h


async def get_cached_movie_details(tmdb_id: int) -> bytes | None:
    """Return the cached MovieDetails JSON bytes for `tmdb_id`, or None.

    Returns the raw payload exactly as written by `cache_movie_details`
    so the API layer can pass it straight to a Response body — no
    decode/re-encode round trip on the warm path.
    """
    client = get_redis_client()
    key = redis_key(*_MOVIE_DETAILS_KEY_PARTS, str(tmdb_id))
    return await client.get(key)


async def cache_movie_details(tmdb_id: int, payload: bytes) -> None:
    """Write the encoded MovieDetails payload to Redis with a 24h TTL.

    `payload` must be the bytes the API endpoint will return verbatim
    (msgspec-encoded MovieDetails). Cache is serialization-agnostic —
    we never decode it here.
    """
    client = get_redis_client()
    key = redis_key(*_MOVIE_DETAILS_KEY_PARTS, str(tmdb_id))
    await client.set(key, payload, ex=_MOVIE_DETAILS_TTL_SECONDS)
