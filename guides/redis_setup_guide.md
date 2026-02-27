# Redis Connection Guide

## Context

Redis runs as a self-hosted Docker container on the same Docker Compose network as the API server. It is **never exposed to the internet** — all access is over Docker's internal network. It is a pure cache; no persistence (RDB/AOF) is needed or configured.

---

## Connection Model: Async Connection Pool

Use `redis.asyncio` (bundled with `redis-py`) with an explicit `ConnectionPool`. Do **not** use a bare single-connection client.

```python
import redis.asyncio as aioredis
from redis.asyncio.connection import ConnectionPool

_redis_pool: ConnectionPool | None = None
_redis_client: aioredis.Redis | None = None

def get_redis_client() -> aioredis.Redis:
    """Return the shared async Redis client backed by a connection pool."""
    if _redis_client is None:
        raise RuntimeError("Redis not initialized. Call init_redis() at startup.")
    return _redis_client

async def init_redis(host: str, port: int = 6379, max_connections: int = 10) -> None:
    """Call once at application startup (e.g. FastAPI lifespan)."""
    global _redis_pool, _redis_client
    _redis_pool = ConnectionPool(
        host=host,
        port=port,
        max_connections=max_connections,
        decode_responses=False,  # Keep as bytes — embedding cache uses raw binary
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
```

**Pool size:** 10 connections is appropriate for this deployment. Redis connections are cheap, but there's no need to go higher given the single-instance, low-QPS setup.

**`decode_responses=False`:** Required. The embedding cache stores raw packed binary (`struct.pack`) — not strings. A client with `decode_responses=True` will corrupt binary reads. Handle string decoding explicitly per call site where needed.

---

## Docker Compose Host

The `host` value passed to `init_redis()` should come from config/env, not be hardcoded. In Docker Compose, the service name is the hostname on the internal network:

```python
# config.py
REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")   # Docker service name
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
```

```yaml
# docker-compose.yml (reference)
services:
  redis:
    image: redis:7
    command: >
      redis-server
      --maxmemory 450mb
      --maxmemory-policy volatile-lru
      --save ""
      --appendonly no
    networks:
      - internal

  api:
    environment:
      - REDIS_HOST=redis
    networks:
      - internal
```

---

## FastAPI Lifespan Integration

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from config import REDIS_HOST, REDIS_PORT

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_redis(host=REDIS_HOST, port=REDIS_PORT)
    yield
    await close_redis()

app = FastAPI(lifespan=lifespan)
```

---

## Environment Prefix

Every key written to or read from Redis **must** be prefixed with the deployment environment. Apply this at the call site — do not bake it into the connection layer.

```python
ENV_PREFIX: str = os.getenv("ENV", "prod")  # "prod" | "staging"

def redis_key(*parts: str) -> str:
    return f"{ENV_PREFIX}:{':'.join(parts)}"

# Usage
key = redis_key("emb", model_name, text_hash)   # → "prod:emb:text-embedding-3-small:abc123"
key = redis_key("qu", f"v{QU_VERSION}", query_hash)  # → "prod:qu:v4:abc123"
key = redis_key("trending", "current")               # → "prod:trending:current"
key = redis_key("tmdb", "movie", str(movie_id))      # → "prod:tmdb:movie:42"
```

---

## Key Reference

| Namespace | Key Pattern | Type | TTL |
|---|---|---|---|
| Embedding cache | `{env}:emb:{model}:{hash}` | String (binary) | 7 days (604800s) |
| Query understanding | `{env}:qu:v{N}:{hash}` | String (JSON) | 1 day (86400s) |
| Trending set | `{env}:trending:current` | Set | **None** — atomic overwrite only |
| TMDB detail | `{env}:tmdb:movie:{id}` | String (JSON) | 1 day (86400s) |

`{N}` in the QU key is the QU schema version integer — store it as a named constant (`QU_VERSION`). Bump it on any deployment that changes the DAG output shape to avoid serving stale cached responses with a mismatched schema.

---

## Critical Constraints

- **`volatile-lru` memory policy** — keys with a TTL are eligible for LRU eviction; `trending:current` (no TTL) is never evicted. Do **not** use `allkeys-lru`.
- **No persistence** — `--save ""` and `--appendonly no`. Redis restarts cold and warms up via normal traffic. Nothing in Redis is a source of truth.
- **Binary embeddings only** — always use `struct.pack(">" + "f" * len(vector), *vector)` to store and `struct.unpack` to retrieve. Never serialize embedding vectors as JSON arrays.
- **`trending:current` is never expired** — refresh via atomic `RENAME` from a staging key (`trending:next`), never via `DEL` + `SADD` in sequence.