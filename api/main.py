from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

import msgspec

from db.postgres import (
    check_postgres,
    fetch_movie_card_summaries,
    pool,
)
from db.qdrant import qdrant_client, check_qdrant
from db.redis import init_redis, close_redis, check_redis
from implementation.misc.event_loop import install_uvloop
from search_v2.similar_movies import run_similar_movies_for_ids
from search_v2.streaming_orchestrator import stream_full_pipeline

# Shared msgspec JSON encoder. msgspec.Struct types (e.g. MovieCard)
# encode natively without Pydantic's model_dump round-trip; ~10-50×
# faster than stdlib json + Pydantic on the wire-format hot path.
_json_encoder = msgspec.json.Encoder()

# Switch asyncio onto uvloop before uvicorn starts the event loop.
# ~2x faster than the default selector loop on socket-heavy workloads
# (concurrent Postgres + Qdrant + Redis + LLM calls). Idempotent and
# silently no-ops on platforms without uvloop. See
# implementation/misc/event_loop.py.
install_uvloop()


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


# ---------------------------------------------------------------------------
# Search endpoints
# ---------------------------------------------------------------------------


class QuerySearchBody(BaseModel):
    """Request body for POST /query_search."""

    query: str = Field(min_length=1)


class SimilaritySearchBody(BaseModel):
    """Request body for POST /similarity_search."""

    tmdb_ids: list[int] = Field(min_length=1)


@app.post("/query_search")
async def query_search(body: QuerySearchBody):
    """
    Stream the multi-channel search pipeline as Server-Sent Events.

    Events (in order, named on the SSE wire):
      - fetches_ready  — fires once after Steps 0+1. Lists every "fetch"
                          the pipeline will run (standard branches +
                          exact-title + similarity).
      - branch_traits  — fires per standard-flow branch when Step 2
                          completes (one per branch). Skipped for the
                          non-standard flows.
      - branch_results — fires per fetch when its execution finishes.
                          Per-fetch errors surface in the payload's
                          `branch_error` field, not the `error` event.
      - done           — terminal event with `total_elapsed` seconds.
      - error          — only for fatal failures (Step 0 unrecoverable).

    Returns:
      HTTP 200 with `text/event-stream` content. Empty `query` → 400.
    """
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be non-empty.")

    async def event_stream():
        # Translate (event_name, payload) pairs from the orchestrator
        # into SSE wire frames. msgspec.json.Encoder handles
        # MovieCard structs natively, so payloads can carry the
        # cards without a per-card model_dump materialization.
        # We let CancelledError propagate so Starlette can clean up
        # on client disconnect; the orchestrator's `finally` block
        # cancels any in-flight tasks before unwinding.
        async for event_name, payload in stream_full_pipeline(query):
            body = _json_encoder.encode(payload).decode("utf-8")
            yield f"event: {event_name}\ndata: {body}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Disable buffering at common reverse-proxy layers so events
            # flush to the client immediately rather than accumulating.
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/similarity_search")
async def similarity_search(body: SimilaritySearchBody) -> Response:
    """
    Run the similarity-search flow against a caller-supplied anchor set.

    Takes a list of TMDB IDs (the anchors) and returns ranked similar
    movies as an array of MovieCard objects (tmdb_id, title,
    release_date, poster_url), sorted by descending similarity score.
    Bypasses the natural-language pipeline entirely.

    Errors:
      - Empty `tmdb_ids` → 422 (pydantic).
      - Unknown TMDB IDs → 422 with the missing IDs in the detail.
    """
    try:
        result = await run_similar_movies_for_ids(body.tmdb_ids)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except LookupError as exc:
        # Raised when one or more tmdb_ids don't exist in movie_card.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Hydrate the ranked tmdb_ids into MovieCard summaries. Order is
    # preserved by fetch_movie_card_summaries, so the response stays
    # sorted by descending similarity score. MovieCard is a
    # msgspec.Struct — we encode directly with msgspec, skipping
    # Pydantic/jsonable_encoder for this hot path.
    ranked_ids = [r.movie_id for r in result.ranked]
    cards = await fetch_movie_card_summaries(ranked_ids)
    return Response(
        content=_json_encoder.encode(cards),
        media_type="application/json",
    )