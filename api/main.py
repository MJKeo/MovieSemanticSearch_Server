import os
import redis
import psycopg2
from qdrant_client import QdrantClient
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health_check():
    results = {}

    # Test Postgres
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST"),
            dbname=os.getenv("POSTGRES_DB"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD")
        )
        conn.close()
        results["postgres"] = "ok"
    except Exception as e:
        results["postgres"] = str(e)

    # Test Redis
    try:
        r = redis.Redis(host=os.getenv("REDIS_HOST", "redis"), port=6379)
        r.ping()
        results["redis"] = "ok"
    except Exception as e:
        results["redis"] = str(e)

    # Test Qdrant
    try:
        client = QdrantClient(host=os.getenv("QDRANT_HOST", "qdrant"), port=6333)
        client.get_collections()
        results["qdrant"] = "ok"
    except Exception as e:
        results["qdrant"] = str(e)

    return results