import os
import redis
from contextlib import asynccontextmanager
from qdrant_client import QdrantClient
from fastapi import FastAPI
from db.postgres import pool, check_postgres
from db.qdrant import qdrant_client, check_qdrant


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler for connection pool lifecycle management.
    
    Opens the Postgres connection pool on startup and closes it gracefully on shutdown.
    This ensures connections are established after Docker Compose starts Postgres,
    and are properly cleaned up when the server shuts down.
    """
    # Open the pool and establish initial connections
    await pool.open()
    # Validate that connections actually work (fast-fail if Postgres is unreachable)
    await pool.check()
    yield
    # Gracefully close all connections on shutdown
    await qdrant_client.close()
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

    # Test Postgres via the connection pool
    results["postgres"] = await check_postgres()
    results["qdrant"] = await check_qdrant()

    # Test Redis
    try:
        r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379)
        r.ping()
        results["redis"] = "ok"
    except Exception as e:
        results["redis"] = str(e)

    return results