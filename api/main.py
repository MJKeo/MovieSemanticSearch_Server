import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

import msgspec

logger = logging.getLogger(__name__)

from db.postgres import (
    check_postgres,
    fetch_movie_card_row,
    fetch_movie_card_summaries,
    pool,
)
from db.qdrant import qdrant_client, check_qdrant
from db.redis import (
    cache_movie_details,
    cache_similar_movies,
    check_redis,
    close_redis,
    get_cached_movie_details,
    get_cached_similar_movies,
    init_redis,
)
from db.tmdb import (
    AdaptiveRateLimiter,
    TMDBFetchError,
    build_api_tmdb_client,
    fetch_movie_details_for_endpoint,
)
from implementation.classes.enums import Genre, StreamingAccessType
from implementation.classes.languages import Language
from implementation.classes.schemas import MetadataFilters
from implementation.classes.watch_providers import (
    STREAMING_PROVIDER_MAP,
    StreamingService,
)
from implementation.misc.event_loop import install_uvloop
from implementation.misc.helpers import create_watch_provider_offering_key
from schemas.api_responses import (
    CastMember,
    CrewMember,
    MovieDetails,
    WatchProvider,
)
from schemas.entity_translation import PersonCategory
from search_v2.attribute_search import (
    DEFAULT_ATTRIBUTE_SEARCH_LIMIT,
    PersonSpec,
    run_attribute_search,
)
from search_v2.similar_movies import run_similar_movies_for_ids
from search_v2.streaming_orchestrator import stream_full_pipeline
from search_v2.title_search import (
    TITLE_SEARCH_DEFAULT_LIMIT,
    TITLE_SEARCH_MAX_LIMIT,
    run_title_search,
)

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
    # Build the shared TMDB httpx client + rate limiter used by /movie_details.
    # Stored on app.state (not module globals) so test clients can override
    # them via FastAPI's lifespan/dependency machinery if needed. TLS handshake
    # cost dominates per-request client construction, so we share one instance
    # across the process lifetime.
    app.state.tmdb_client = build_api_tmdb_client()
    app.state.tmdb_rate_limiter = AdaptiveRateLimiter()
    yield
    # Gracefully close all connections on shutdown
    await app.state.tmdb_client.aclose()
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
    # Optional follow-up correction the user supplied after receiving
    # a prior result set. When present and non-blank, Steps 0 and 1
    # of the search pipeline switch to clarification-mode prompts and
    # Step 1's main_rewrite replaces the verbatim raw query as the
    # primary search branch.
    clarification: Optional[str] = None
    filters: Optional[MetadataFiltersInput] = None


class SimilaritySearchBody(BaseModel):
    """Request body for POST /similarity_search."""

    tmdb_ids: list[int] = Field(min_length=1)


# Map the wire role string straight onto PersonCategory. Pydantic's
# Literal validator on PersonInput.role already rejects anything
# outside these five values, so the dict lookup is total over the
# accepted type.
_ROLE_LITERAL_TO_CATEGORY: dict[str, PersonCategory] = {
    "actor": PersonCategory.ACTOR,
    "director": PersonCategory.DIRECTOR,
    "writer": PersonCategory.WRITER,
    "producer": PersonCategory.PRODUCER,
    "composer": PersonCategory.COMPOSER,
}


class PersonInput(BaseModel):
    """One named-person filter on POST /attribute_search.

    `name` is the raw display name (we normalize server-side; the
    caller need not pre-normalize). `role`, when set, restricts the
    credit lookup to that one role's posting table; when omitted,
    any credit on the movie qualifies (union across actor /
    director / writer / producer / composer).
    """

    name: str = Field(min_length=1)
    role: Optional[
        Literal["actor", "director", "writer", "producer", "composer"]
    ] = None


# Upper bound on the number of person filters per request. Each
# unrestricted-role person fans out to 5 parallel posting-table
# lookups, so a tight cap keeps a single request from saturating
# the Postgres pool (max_size=10 per conventions.md). 20 is far
# beyond any legitimate browse-UI use case while still leaving
# headroom for "all-time favorite filmmakers" lists.
_MAX_ATTRIBUTE_SEARCH_PEOPLE = 20


class AttributeSearchBody(BaseModel):
    """Request body for POST /attribute_search.

    Both fields are optional. Sending an empty body (`{}`) returns
    the top movies overall ranked by the 80/20 popularity/reception
    prior — same shape as a "browse the catalog" request.

    `people=None` and `people=[]` are treated identically.
    """

    filters: Optional[MetadataFiltersInput] = None
    people: Optional[list[PersonInput]] = Field(
        default=None, max_length=_MAX_ATTRIBUTE_SEARCH_PEOPLE,
    )


def _to_person_specs(
    people_input: Optional[list[PersonInput]],
) -> list[PersonSpec]:
    """Translate wire-level PersonInput list into the internal PersonSpec list.

    - Strips whitespace from each name; drops persons whose name is
      blank after strip (defensive normalization at the boundary —
      mirrors the `clarification` handling in /query_search).
    - Resolves `role` (string Literal | None) to a `PersonCategory`,
      with the missing-role case landing on PersonCategory.UNKNOWN
      so the orchestrator's role dispatch is total.
    """
    if not people_input:
        return []

    specs: list[PersonSpec] = []
    for entry in people_input:
        stripped = entry.name.strip()
        if not stripped:
            continue
        role = (
            _ROLE_LITERAL_TO_CATEGORY[entry.role]
            if entry.role is not None
            else PersonCategory.UNKNOWN
        )
        specs.append(PersonSpec(name=stripped, role=role))
    return specs


@app.post("/query_search")
async def query_search(body: QuerySearchBody):
    """
    Stream the multi-channel search pipeline as Server-Sent Events.

    Events (in order, named on the SSE wire):
      - fetches_ready    — fires once after Steps 0+1. Lists every
                            "fetch" the pipeline will run (standard
                            branches + exact-title + similarity).
      - branch_traits    — fires per standard-flow branch when Step 2
                            completes (one per branch). Skipped for the
                            non-standard flows.
      - branch_categories — fires per standard-flow branch when Step 3
                            completes, after `branch_traits` and before
                            `branch_results`. Carries each trait's
                            decomposition into (category, expressions[])
                            pairs so the UI can expand traits before
                            results land. Same skip-set as
                            `branch_traits`.
      - branch_results   — fires per fetch when its execution finishes.
                            Per-fetch errors surface in the payload's
                            `branch_error` field, not the `error` event.
      - done             — terminal event with `total_elapsed` seconds.
      - error            — only for fatal failures (Step 0
                            unrecoverable).

    Returns:
      HTTP 200 with `text/event-stream` content. Empty `query` → 400.
    """
    query = body.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query must be non-empty.")

    # Normalize clarification at the boundary: treat empty strings and
    # whitespace-only input the same as omitted. This keeps the
    # no-clarification fast path stable — the pipeline only switches to
    # clarification-mode prompts when there is real correction text.
    # Downstream layers (stream_full_pipeline, run_full_pipeline,
    # run_step_0/1) re-normalize too; each is a separate public surface
    # also reachable from CLI runners, so each defends its own contract.
    clarification: Optional[str] = (
        body.clarification.strip() if body.clarification else None
    )
    if not clarification:
        clarification = None

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
            query,
            clarification=clarification,
            metadata_filters=metadata_filters,
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
    # Canonicalize anchors (dedup + sort) so the cache key is order- and
    # duplicate-insensitive. run_similar_movies_for_ids dedups again
    # internally and doesn't depend on input ordering.
    canonical_ids = sorted(dict.fromkeys(int(mid) for mid in body.tmdb_ids))

    # Redis warm path. Per the graceful-degradation convention
    # (docs/conventions.md), a Redis failure must NOT fail the request —
    # fall through to the cold path instead.
    try:
        cached = await get_cached_similar_movies(canonical_ids)
    except Exception:
        logger.warning(
            "similar_movies cache read failed for tmdb_ids=%s",
            canonical_ids,
            exc_info=True,
        )
        cached = None
    if cached is not None:
        return Response(content=cached, media_type="application/json")

    try:
        result = await run_similar_movies_for_ids(canonical_ids)
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
    encoded = _json_encoder.encode(cards)

    # Cache-write failures must NOT lose the response we already built —
    # log and continue.
    try:
        await cache_similar_movies(canonical_ids, encoded)
    except Exception:
        logger.warning(
            "similar_movies cache write failed for tmdb_ids=%s",
            canonical_ids,
            exc_info=True,
        )
    return Response(content=encoded, media_type="application/json")


@app.post("/attribute_search")
async def attribute_search(body: AttributeSearchBody) -> Response:
    """
    Hard-attribute browse search — no NLP, no LLM, no vector search.

    Intersects the supplied filters (genres, audio languages,
    streaming services, release/runtime/maturity ranges, and named
    people with optional role restriction) and returns the top
    matches ranked by the 80/20 popularity/reception neutral prior.

    Multiple `people` entries intersect (AND). A `role` on a person
    restricts the credit lookup to that one posting table; omitting
    `role` unions across all five role tables (any credit qualifies).
    Unresolvable person names produce zero matches silently — the
    response is an empty `[]` rather than an error.

    Returns:
      HTTP 200 with an array of MovieCard objects
      (tmdb_id, title, release_date, poster_url, maturity_rating),
      ranked by descending neutral-prior score, capped at 250.
      HTTP 422 on unknown genre / audio_language / streaming_service
      enum values (handled inside `_to_metadata_filters`).
    """
    # Translate the wire mirror into MetadataFilters once at the
    # boundary. Raises 422 on any unknown enum value; collapses to
    # None if every filter field is unset so the orchestrator's
    # SQL stays filter-clause-free on unfiltered queries.
    metadata_filters = _to_metadata_filters(body.filters)

    # Convert wire-level PersonInput list (strings + optional role
    # Literal) into the internal PersonSpec list (stripped name +
    # resolved PersonCategory). Blank-after-strip entries are
    # dropped here so the orchestrator never sees them.
    people_specs = _to_person_specs(body.people)

    ranked_ids = await run_attribute_search(
        people=people_specs,
        metadata_filters=metadata_filters,
        limit=DEFAULT_ATTRIBUTE_SEARCH_LIMIT,
    )

    # Hydrate ranked movie_ids into MovieCard summaries. Order is
    # preserved by `fetch_movie_card_summaries`, so the response
    # stays sorted by descending neutral-prior score. Same
    # msgspec-encode hot path as /similarity_search.
    cards = await fetch_movie_card_summaries(ranked_ids)
    return Response(
        content=_json_encoder.encode(cards),
        media_type="application/json",
    )


# Cache-Control header for /title_search responses. Five minutes is
# long enough to absorb a typeahead burst across sessions without
# making the cache visibly stale during ingest (which runs on a
# daily-ish cadence — see movie_ingestion/tmdb_fetching/daily_export.py).
# `public` so any intermediate cache (CDN, browser, reverse proxy) can
# serve the response without revalidation.
_TITLE_SEARCH_CACHE_CONTROL = "public, max-age=300"


@app.get("/title_search")
async def title_search(
    q: str = Query(..., description="Title query (trimmed server-side)"),
    limit: int = Query(
        TITLE_SEARCH_DEFAULT_LIMIT,
        description="Maximum number of results to return.",
    ),
) -> Response:
    """Lightweight title-only typeahead lookup.

    Backs the frontend's "pick similar-to" picker: called on every
    debounced keystroke, so the contract is minimal and the hot path
    is a single Postgres query against the trigram-indexed
    `title_normalized` column. No LLM, no vector search, no Qdrant,
    no Redis.

    Matching uses two priority tiers:
      1. Token-prefix — query is a prefix of any whitespace-delimited
         token in the title ("dark" → "The Dark Knight").
      2. Substring — query appears anywhere in the title but not at a
         token boundary ("ark kni" → "The Dark Knight").
    Within each tier results are ranked by the same 80/20
    popularity/reception blend `/attribute_search` uses, with
    `tmdb_id DESC` as a stable tie-break.

    Args:
      q: User input. Trimmed server-side; whitespace-only is rejected.
        Inputs longer than ~100 chars are truncated.
      limit: Max results to return. Default 10, hard cap 25. Out-of-
        range values 422.

    Errors:
      - 422 if `q` is empty / whitespace-only after trim.
      - 422 if `limit` is < 1 or > 25.

    Returns:
      HTTP 200 with an array of MovieCard objects (tmdb_id, title,
      release_date, poster_url, maturity_rating). Empty array on no
      matches (never 404). `Cache-Control: public, max-age=300` so
      identical queries are absorbed by intermediate caches.
    """
    # Validate the trimmed query at the boundary. Whitespace-only and
    # missing values both collapse to the same "empty" condition.
    trimmed = q.strip()
    if not trimmed:
        raise HTTPException(status_code=422, detail="q must be non-empty")

    # `limit` is validated as a closed range matching the spec. The
    # orchestrator trusts its caller for clamping, so we have to do it
    # here.
    if limit < 1 or limit > TITLE_SEARCH_MAX_LIMIT:
        raise HTTPException(status_code=422, detail="limit out of range")

    # Resolve to ranked movie_ids; then hydrate to the wire-format
    # MovieCard via the same single-query summary fetch all sibling
    # endpoints use. Order is preserved by `fetch_movie_card_summaries`
    # so the tier-then-popularity ordering survives intact.
    ranked_ids = await run_title_search(trimmed, limit=limit)
    cards = await fetch_movie_card_summaries(ranked_ids)
    return Response(
        content=_json_encoder.encode(cards),
        media_type="application/json",
        headers={"Cache-Control": _TITLE_SEARCH_CACHE_CONTROL},
    )


# ---------------------------------------------------------------------------
# /movie_details endpoint
# ---------------------------------------------------------------------------

# Base URLs for TMDB-hosted images and the public movie page. The TMDB
# payload carries relative paths (e.g. "/abc.jpg"); we join them onto the
# CDN base here so the frontend can render <img src=...> directly.
# Per-asset size suffixes (e.g. "w185", "w500", "w1280") slot into the
# {size} segment; the frontend almost always downscales, so we serve the
# nearest-larger TMDB-resized variant rather than the full-resolution
# "original" to cut payload weight on the cast/provider thumbnails.
_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p"
_TMDB_MOVIE_PAGE_BASE = "https://www.themoviedb.org/movie"
_IMDB_TITLE_PAGE_BASE = "https://www.imdb.com/title"
_YOUTUBE_WATCH_BASE = "https://www.youtube.com/watch"
_VIMEO_WATCH_BASE = "https://vimeo.com"

# TMDB image sizes, matched to expected render context. The frontend can
# always swap to a different size later by replacing the segment — but
# defaulting to the right size keeps the cold-path response small.
_POSTER_SIZE = "w500"      # poster art (movie tile / detail header)
_BACKDROP_SIZE = "w1280"   # hero banner — large but not full-res
_PROFILE_SIZE = "w185"     # cast/crew headshots (small thumbnail)
_LOGO_SIZE = "w92"         # provider logos (tiny icon)
# Gallery artwork served alongside the hero. Backdrops and posters share
# the w780 size — it's the largest size TMDB supports for both shapes,
# so the frontend can render a unified grid without per-image size logic.
_ADDITIONAL_IMAGE_SIZE = "w780"

# Cap on the additional_images gallery. Five is enough for a strip below
# the hero without bloating the cold-path response.
_ADDITIONAL_IMAGES_LIMIT = 5

# Cap the cast list at the top-billed N. TMDB returns the full crew which
# can be hundreds of entries on big productions; the detail page only
# needs the headliners.
_CAST_LIMIT = 12

# Per-bucket cap for crew lists. Most movies have 1–2 directors, but
# blockbusters can list 10+ producers / writers. Limit per bucket so
# the response stays small without dropping the primary credits.
_CREW_BUCKET_LIMIT = 5

# Job titles TMDB uses for each crew bucket. "Screenplay" and "Story"
# both contribute to the writers list since TMDB splits them out;
# "Executive Producer" rolls up into producers.
_WRITER_JOBS = {"Writer", "Screenplay", "Story"}
_PRODUCER_JOBS = {"Producer", "Executive Producer"}
_DIRECTOR_JOBS = {"Director"}


def _image_url(path: str | None, size: str) -> str | None:
    """Join a TMDB relative image path onto the CDN base at the given size.

    `size` must be a TMDB-served size segment ("w185", "w500", "w1280",
    "original", etc.). Returns ``None`` for empty/missing paths so the
    field is omitted via `omit_defaults` on the wire.
    """
    if not path:
        return None
    return f"{_TMDB_IMAGE_BASE}/{size}{path}"


def _extract_us_certification(release_dates: dict | None) -> str | None:
    """Return the US MPAA-style certification ("PG-13" etc.) or None.

    TMDB's release_dates payload is a list of country buckets, each with
    a list of dated releases. We pick the first US entry with a non-empty
    `certification` field — the others are typically blanks for
    non-theatrical release windows.
    """
    if not release_dates:
        return None
    for country in release_dates.get("results", []):
        if country.get("iso_3166_1") != "US":
            continue
        for entry in country.get("release_dates", []):
            cert = entry.get("certification")
            if cert:
                return cert
    return None


_SUPPORTED_VIDEO_SITES = {"YouTube", "Vimeo"}


def _video_url(site: str | None, key: str | None) -> str | None:
    """Build a watchable URL for a TMDB video entry, or None if unsupported.

    Only YouTube and Vimeo are wired up — those are the two hosts TMDB
    uses for ~99% of trailer entries. Other sites (e.g. self-hosted
    Lionsgate URLs occasionally appearing on indie releases) need
    bespoke URL templates we don't keep here.
    """
    if not key:
        return None
    if site == "YouTube":
        return f"{_YOUTUBE_WATCH_BASE}?v={key}"
    if site == "Vimeo":
        return f"{_VIMEO_WATCH_BASE}/{key}"
    return None


def _extract_trailer_url(videos: dict | None) -> str | None:
    """Pick the best trailer URL from TMDB's videos payload.

    Preference order: official Trailer → any Trailer → any Teaser.
    Within each tier YouTube wins over Vimeo because YouTube embeds are
    cheaper / more widely supported on the client side. Returns ``None``
    when nothing usable exists (e.g. older films with no video assets
    uploaded to TMDB).
    """
    if not videos:
        return None
    # Pre-filter to supported hosts and categorize by type in one pass.
    trailers_official: list[dict] = []
    trailers: list[dict] = []
    teasers: list[dict] = []
    for v in videos.get("results", []):
        if v.get("site") not in _SUPPORTED_VIDEO_SITES:
            continue
        kind = v.get("type")
        if kind == "Trailer":
            if v.get("official"):
                trailers_official.append(v)
            else:
                trailers.append(v)
        elif kind == "Teaser":
            teasers.append(v)

    # Within a tier, prefer YouTube over Vimeo via a stable sort. Python's
    # sort is stable, so TMDB's original ordering breaks further ties.
    def _youtube_first(v: dict) -> int:
        return 0 if v.get("site") == "YouTube" else 1

    for bucket in (trailers_official, trailers, teasers):
        if not bucket:
            continue
        bucket.sort(key=_youtube_first)
        url = _video_url(bucket[0].get("site"), bucket[0].get("key"))
        if url is not None:
            return url
    return None


def _extract_credits(
    credits: dict | None,
) -> tuple[list[CrewMember], list[CrewMember], list[CrewMember], list[CastMember]]:
    """Split TMDB's credits block into (directors, writers, producers, cast).

    Single pass over crew with set-membership lookups for bucketing.
    Cast is taken in TMDB's native order (already sorted by billing
    position), truncated to _CAST_LIMIT.
    """
    if not credits:
        return [], [], [], []

    directors: list[CrewMember] = []
    writers: list[CrewMember] = []
    producers: list[CrewMember] = []
    for entry in credits.get("crew", []):
        job = entry.get("job") or ""
        name = entry.get("name")
        if not name:
            continue
        member = CrewMember(
            name=name,
            job=job,
            profile_url=_image_url(entry.get("profile_path"), _PROFILE_SIZE),
        )
        if job in _DIRECTOR_JOBS and len(directors) < _CREW_BUCKET_LIMIT:
            directors.append(member)
        elif job in _WRITER_JOBS and len(writers) < _CREW_BUCKET_LIMIT:
            writers.append(member)
        elif job in _PRODUCER_JOBS and len(producers) < _CREW_BUCKET_LIMIT:
            producers.append(member)

    cast: list[CastMember] = []
    for entry in credits.get("cast", [])[:_CAST_LIMIT]:
        name = entry.get("name")
        if not name:
            continue
        cast.append(
            CastMember(
                name=name,
                character=entry.get("character") or None,
                profile_url=_image_url(entry.get("profile_path"), _PROFILE_SIZE),
            )
        )

    return directors, writers, producers, cast


def _extract_additional_images(
    images: dict | None,
    exclude_paths: set[str],
) -> list[str]:
    """Pick up to N gallery image URLs from the TMDB `images` sub-resource.

    Strategy:
      1. Sort backdrops by TMDB `vote_count` descending; take URLs in order,
         skipping any whose `file_path` is already surfaced as the primary
         poster/backdrop (passed in via `exclude_paths`) and any duplicates.
      2. If we still have headroom (fewer than the cap), top up from posters
         using the same sort + dedupe rule. Posters are only used as a
         fallback so the gallery stays predominantly landscape backdrops.

    Returns the deduped URL list, capped at `_ADDITIONAL_IMAGES_LIMIT`.
    Returns an empty list when `images` is missing or both sub-lists are
    empty.
    """
    if not images:
        return []

    # `seen` tracks file_paths already chosen (or excluded as primary media)
    # so we never emit the same image twice even if TMDB lists it in both
    # backdrops and posters (rare but possible for square-ish artwork).
    seen: set[str] = set(exclude_paths)
    picked: list[str] = []

    def _take_from(bucket: list[dict] | None) -> None:
        if not bucket:
            return
        # Sort by vote_count desc — TMDB's community-vetted popularity
        # signal. Missing/None vote_count sorts to the bottom (treated as 0).
        ranked = sorted(
            bucket,
            key=lambda entry: entry.get("vote_count") or 0,
            reverse=True,
        )
        for entry in ranked:
            if len(picked) >= _ADDITIONAL_IMAGES_LIMIT:
                return
            path = entry.get("file_path")
            if not path or path in seen:
                continue
            seen.add(path)
            picked.append(f"{_TMDB_IMAGE_BASE}/{_ADDITIONAL_IMAGE_SIZE}{path}")

    _take_from(images.get("backdrops"))
    # Only consult posters if backdrops didn't fill the quota — keeps the
    # gallery shape-consistent for movies with rich backdrop coverage.
    if len(picked) < _ADDITIONAL_IMAGES_LIMIT:
        _take_from(images.get("posters"))

    return picked


def _extract_us_watch_providers(watch_providers: dict | None) -> list[WatchProvider]:
    """Flatten the US region of TMDB's watch/providers payload.

    TMDB groups providers by access type (flatrate/buy/rent) inside the
    region bucket. We flatten them into a single list, tagging each with
    its `access_type` so the frontend can group / colour as it likes.
    Other regions are ignored — the UI is US-only.
    """
    if not watch_providers:
        return []
    us = watch_providers.get("results", {}).get("US")
    if not us:
        return []
    out: list[WatchProvider] = []
    # Order matches what most users care about: subscription first, then
    # rent (typically cheaper than buy), then buy.
    for access_type in ("flatrate", "rent", "buy"):
        for p in us.get(access_type, []) or []:
            provider_id = p.get("provider_id")
            name = p.get("provider_name")
            if provider_id is None or not name:
                continue
            out.append(
                WatchProvider(
                    provider_id=int(provider_id),
                    name=name,
                    logo_url=_image_url(p.get("logo_path"), _LOGO_SIZE),
                    access_type=access_type,
                )
            )
    return out


def _build_movie_details(
    tmdb_id: int,
    payload: dict,
    card_row: dict,
) -> MovieDetails:
    """Translate a TMDB detail payload + movie_card row into MovieDetails.

    Pure function — no I/O. All field mapping happens here so the endpoint
    handler stays focused on orchestration (cache → DB → TMDB → encode).

    `payload` is the TMDB JSON dict (with append_to_response sub-resources
    inlined). `card_row` is the matching row from public.movie_card and
    supplies the locally-computed reception_score.
    """
    directors, writers, producers, cast = _extract_credits(payload.get("credits"))

    # IMDb link: prefer external_ids.imdb_id (always present in the append
    # response when known), fall back to the top-level imdb_id field which
    # is sometimes populated when the appended block isn't.
    external_ids = payload.get("external_ids") or {}
    imdb_id = external_ids.get("imdb_id") or payload.get("imdb_id")

    # `homepage` is occasionally an empty string in TMDB; normalize to None
    # so the frontend can rely on omit_defaults behaviour.
    homepage = payload.get("homepage") or None

    # Build the dedupe set for additional_images using the *raw* TMDB paths
    # (not the joined CDN URLs) so the comparison matches regardless of the
    # size segment chosen for each surface.
    primary_image_paths = {
        p for p in (payload.get("poster_path"), payload.get("backdrop_path")) if p
    }

    return MovieDetails(
        tmdb_id=tmdb_id,
        title=payload.get("title"),
        original_title=payload.get("original_title"),
        overview=payload.get("overview") or None,
        tagline=payload.get("tagline") or None,
        release_date=payload.get("release_date") or None,
        # TMDB encodes "unknown runtime" as 0 (older films, recent indie
        # releases, foreign titles); collapse to None so the frontend can
        # omit the field instead of rendering "0 min".
        runtime_minutes=payload.get("runtime") or None,
        maturity_rating=_extract_us_certification(payload.get("release_dates")),
        genres=[g["name"] for g in payload.get("genres", []) if g.get("name")],
        spoken_languages=[
            lang.get("english_name") or lang.get("name")
            for lang in payload.get("spoken_languages", [])
            if lang.get("english_name") or lang.get("name")
        ],
        poster_url=_image_url(payload.get("poster_path"), _POSTER_SIZE),
        backdrop_url=_image_url(payload.get("backdrop_path"), _BACKDROP_SIZE),
        trailer_url=_extract_trailer_url(payload.get("videos")),
        additional_images=_extract_additional_images(
            payload.get("images"), primary_image_paths
        ),
        reception_score=card_row.get("reception_score"),
        tmdb_vote_average=payload.get("vote_average"),
        tmdb_vote_count=payload.get("vote_count"),
        directors=directors,
        writers=writers,
        producers=producers,
        cast=cast,
        watch_providers=_extract_us_watch_providers(payload.get("watch/providers")),
        tmdb_url=f"{_TMDB_MOVIE_PAGE_BASE}/{tmdb_id}",
        imdb_url=f"{_IMDB_TITLE_PAGE_BASE}/{imdb_id}" if imdb_id else None,
        homepage=homepage,
    )


@app.get("/movie_details/{tmdb_id}")
async def movie_details(tmdb_id: int) -> Response:
    """Return the full detail payload for a single movie.

    Pipeline:
      1. Try Redis (24h cache of the encoded MovieDetails bytes).
      2. Confirm the movie is in our index (`public.movie_card`); 404 if not.
      3. Fetch TMDB `/movie/{id}` with credits/videos/images/external_ids/
         watch-providers/release-dates appended.
      4. Build the curated MovieDetails struct (TMDB data + reception_score).
      5. Cache the encoded payload, return it.

    Errors:
      - 404 if `tmdb_id` is not in our `movie_card` index, or TMDB itself
        no longer serves the movie.
      - 502 if the TMDB fetch fails after retries (network / 5xx).
    """
    # 1. Redis warm path. Return cached bytes verbatim — they're already
    # the encoded MovieDetails the frontend expects. Per the
    # graceful-degradation convention (docs/conventions.md), a Redis
    # failure here must NOT fail the request — fall through to the
    # cold path instead.
    try:
        cached = await get_cached_movie_details(tmdb_id)
    except Exception:
        logger.warning(
            "movie_details cache read failed for tmdb_id=%s", tmdb_id, exc_info=True
        )
        cached = None
    if cached is not None:
        return Response(content=cached, media_type="application/json")

    # 2. Reject unknown movies before hitting TMDB. This is also the
    # source of our locally-computed reception_score.
    card_row = await fetch_movie_card_row(tmdb_id)
    if card_row is None:
        raise HTTPException(status_code=404, detail="movie not found")

    # 3. Fetch from TMDB. The shared client + rate limiter live on
    # app.state (initialized in the lifespan handler).
    try:
        tmdb_payload = await fetch_movie_details_for_endpoint(
            app.state.tmdb_client,
            app.state.tmdb_rate_limiter,
            tmdb_id,
        )
    except TMDBFetchError as exc:
        # Surface upstream failures as 502 (bad gateway) rather than 500
        # so the frontend can distinguish "TMDB is unhappy" from "our
        # server is broken" — different retry/UX semantics.
        raise HTTPException(
            status_code=502, detail=f"TMDB fetch failed: {exc}"
        ) from exc
    if tmdb_payload is None:
        # 404 from TMDB: the movie was removed upstream. Treat the same
        # as "not in our index" from the client's perspective.
        raise HTTPException(status_code=404, detail="movie not found on TMDB")

    # 4. Translate to the curated wire format.
    details = _build_movie_details(tmdb_id, tmdb_payload, card_row)

    # 5. Encode once, cache + return the same bytes so the warm-path
    # response is byte-identical to the cold-path response. Cache-write
    # failures must NOT lose the response we already built — log and
    # continue.
    encoded = _json_encoder.encode(details)
    try:
        await cache_movie_details(tmdb_id, encoded)
    except Exception:
        logger.warning(
            "movie_details cache write failed for tmdb_id=%s", tmdb_id, exc_info=True
        )
    return Response(content=encoded, media_type="application/json")