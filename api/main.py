from contextlib import asynccontextmanager
from fastapi import FastAPI
from db.postgres import pool, check_postgres
from db.qdrant import qdrant_client, check_qdrant
from db.redis import init_redis, close_redis, check_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for connection pool lifecycle management.

    Opens all connection pools on startup and closes them gracefully on shutdown.
    """
    # Open the Postgres pool and validate connectivity
    await pool.open()
    await pool.check()
    # Open the Redis pool and validate connectivity
    await init_redis()
    yield
    # Gracefully close all connections on shutdown
    await qdrant_client.close()
    await close_redis()
    await pool.close()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health_check():
    """
    Health check endpoint that validates connectivity to all external services.

    Returns a dictionary with status for each service:
    - postgres: 'ok' or error message (checked via connection pool)
    - redis: 'ok' or error message
    - qdrant: 'ok' or error message
    """
    results = {}
    results["postgres"] = await check_postgres()
    results["redis"] = await check_redis()
    results["qdrant"] = await check_qdrant()
    return results