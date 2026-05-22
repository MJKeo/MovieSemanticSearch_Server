from contextlib import asynccontextmanager
from typing import Optional
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
from implementation.classes.enums import Genre, StreamingAccessType
from implementation.classes.languages import Language
from implementation.classes.schemas import MetadataFilters
from implementation.classes.watch_providers import (
    STREAMING_PROVIDER_MAP,
    StreamingService,
)
from implementation.misc.event_loop import install_uvloop
from implementation.misc.helpers import create_watch_provider_offering_key
from search_v2.similar_movies import run_similar_movies_for_ids
from search_v2.streaming_orchestrator import stream_full_pipeline

# Shared msgspec JSON encoder. msgspec.Struct types (e.g. MovieCard)
# encode natively without Pydantic's model_dump round-trip; ~10-50×
# faster than stdlib json + Pydantic on the wire-format hot path.
_json_encoder = msgspec.json.Encoder()

# All StreamingAccessType method ids (SUBSCRIPTION/BUY/RENT). The UI filter
# carries no access-type preference, so we expand every selected provider
# across all methods — matches "available on this service, any way".
_ALL_STREAMING_METHOD_IDS: tuple[int, ...] = tuple(
    m.type_id for m in StreamingAccessType
)

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


class MetadataFiltersInput(BaseModel):
    """Wire-level mirror of MetadataFilters for the /query_search body.

    Every field is optional / nullable so unset filters are transmitted
    as None and converted into MetadataFilters with the corresponding
    attribute left as None (the "no filter on this axis" contract).

    The string-valued list fields take canonical enum values that match
    the labels the UI displays:
      - genres: ``Genre`` enum values ("Action", "Sci-Fi", …)
      - audio_languages: ``Language`` enum values ("English", "Spanish", …)
      - streaming_services: ``StreamingService`` enum values
        ("netflix", "max", …). Server-side these expand to the flat
        TMDB provider-id list via STREAMING_PROVIDER_MAP.

    Invalid enum values raise HTTPException 422 in _to_metadata_filters
    rather than silently dropping — wire input must be validated at
    the boundary per coding-standards.md.
    """

    min_release_ts:     Optional[int] = None
    max_release_ts:     Optional[int] = None
    min_runtime:        Optional[int] = None
    max_runtime:        Optional[int] = None
    min_maturity_rank:  Optional[int] = None  # 1=G..5=NC-17
    max_maturity_rank:  Optional[int] = None
    genres:             Optional[list[str]] = None
    audio_languages:    Optional[list[str]] = None
    streaming_services: Optional[list[str]] = None


def _to_metadata_filters(
    body_input: MetadataFiltersInput | None,
) -> MetadataFilters | None:
    """Translate the wire mirror into the internal MetadataFilters dataclass.

    Returns ``None`` when the input is None or every field is None
    (no-filter signal). Raises HTTPException 422 on any unknown enum
    value; this is the input-validation boundary for filters.
    """
    if body_input is None:
        return None

    # Resolve enum lists via direct value lookup. Genre / Language /
    # StreamingService all use string-valued members whose ``value``
    # matches the wire token.
    try:
        genres = (
            [Genre(name) for name in body_input.genres]
            if body_input.genres else None
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"unknown genre value: {exc}"
        ) from exc

    try:
        languages = (
            [Language(name) for name in body_input.audio_languages]
            if body_input.audio_languages else None
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"unknown audio language value: {exc}"
        ) from exc

    # Expand StreamingService values into the encoded watch_offer_keys that
    # movie_card.watch_offer_keys is indexed against. Each key packs
    # (provider_id, method_id) via create_watch_provider_offering_key
    # ((pid << 4) | mid), so raw provider IDs never appear in the column —
    # we must fan each service's providers out across every access-type
    # method (subscription/buy/rent) for the prefilter to match. Same
    # encoding pattern as _precompute_streaming_keys in
    # search_v2/endpoint_fetching/metadata_query_execution.py.
    watch_offer_keys: list[int] | None = None
    if body_input.streaming_services:
        try:
            services = [StreamingService(s) for s in body_input.streaming_services]
        except ValueError as exc:
            raise HTTPException(
                status_code=422,
                detail=f"unknown streaming_service value: {exc}",
            ) from exc
        flat: list[int] = []
        seen: set[int] = set()
        for svc in services:
            for pid in STREAMING_PROVIDER_MAP.get(svc, []):
                for mid in _ALL_STREAMING_METHOD_IDS:
                    key = create_watch_provider_offering_key(pid, mid)
                    if key not in seen:
                        seen.add(key)
                        flat.append(key)
        watch_offer_keys = flat or None

    filters = MetadataFilters(
        min_release_ts=body_input.min_release_ts,
        max_release_ts=body_input.max_release_ts,
        min_runtime=body_input.min_runtime,
        max_runtime=body_input.max_runtime,
        min_maturity_rank=body_input.min_maturity_rank,
        max_maturity_rank=body_input.max_maturity_rank,
        genres=genres,
        audio_languages=languages,
        watch_offer_keys=watch_offer_keys,
    )
    # If every field collapsed to None, return None so downstream
    # primitives short-circuit instead of building empty filter clauses.
    return filters if filters.is_active else None


class QuerySearchBody(BaseModel):
    """Request body for POST /query_search."""

    query: str = Field(min_length=1)
    filters: Optional[MetadataFiltersInput] = None


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

    # Translate the wire mirror into the internal MetadataFilters
    # dataclass once, at the boundary. Raises 422 on any unknown enum
    # value (genre / language / streaming_service); collapses to None
    # if every field is unset so the pipeline short-circuits filter
    # plumbing entirely on unfiltered queries.
    metadata_filters = _to_metadata_filters(body.filters)

    async def event_stream():
        # Translate (event_name, payload) pairs from the orchestrator
        # into SSE wire frames. msgspec.json.Encoder handles
        # MovieCard structs natively, so payloads can carry the
        # cards without a per-card model_dump materialization.
        # We let CancelledError propagate so Starlette can clean up
        # on client disconnect; the orchestrator's `finally` block
        # cancels any in-flight tasks before unwinding.
        async for event_name, payload in stream_full_pipeline(
            query, metadata_filters=metadata_filters,
        ):
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