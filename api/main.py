import logging
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from enum import Enum
from typing import Annotated, Literal, NamedTuple, Optional, Union
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

import msgspec
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

# Configure application logging once at import time. Uvicorn only configures
# its own `uvicorn.*` loggers, leaving the root logger at WARNING with no
# handler — so without this, INFO-level logs from our app modules (e.g. the
# /health CF-Connecting-IP line below) would be silently dropped. basicConfig
# attaches a stderr StreamHandler to root at INFO; uvicorn's own loggers keep
# their handlers (propagate=False), so there is no duplication. Output lands on
# the container's stderr, visible via `docker compose logs -f api`.
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

from api.outcome import EndpointFailure, FailureReason, record_outcome
from db.postgres import (
    check_postgres,
    fetch_movie_card_row,
    fetch_movie_card_summaries,
    pool,
)
from db.qdrant import qdrant_client, check_qdrant
from db.redis import (
    cache_movie_credits,
    cache_movie_details,
    cache_similar_movies,
    check_redis,
    close_redis,
    get_cached_movie_credits,
    get_cached_movie_details,
    get_cached_similar_movies,
    init_redis,
    metadata_filters_fingerprint,
)
from db.tmdb import (
    AdaptiveRateLimiter,
    TMDBFetchError,
    build_api_tmdb_client,
    fetch_movie_credits_for_endpoint,
    fetch_movie_details_for_endpoint,
)
from implementation.classes.enums import Genre, StreamingAccessType
from implementation.classes.languages import Language
from implementation.classes.overall_keywords import (
    OverallKeyword,
    keyword_names_from_ids,
)
from implementation.classes.schemas import MetadataFilters
from implementation.classes.watch_providers import (
    STREAMING_PROVIDER_MAP,
    StreamingService,
)
from implementation.misc.event_loop import install_uvloop
from implementation.misc.helpers import create_watch_provider_offering_key
from observability.names import (
    CACHE_WRITE_OK,
    MOVIE_CREDITS_BUILD_AND_CACHE,
    MOVIE_CREDITS_CAST_COUNT,
    MOVIE_CREDITS_CREW_COUNT,
    MOVIE_CREDITS_PAYLOAD_CREATION,
    MOVIE_DETAILS_CACHE_WRITE,
    MOVIE_DETAILS_PAYLOAD_CREATION,
    MOVIE_PAYLOAD_SOURCE,
    MOVIE_TMDB_ID,
    TITLE_SEARCH_FUZZY_RESULT_COUNT,
    TITLE_SEARCH_LIMIT,
    TITLE_SEARCH_QUERY,
    TITLE_SEARCH_RESULT_COUNT,
)
from observability.tracing import setup_tracing
from schemas.api_responses import (
    CastMember,
    CrewGroup,
    CrewMember,
    MovieCredits,
    MovieDetails,
    WatchProvider,
)
from schemas.step_0_flow_routing import (
    CharacterFranchiseFlowData,
    ExactTitleFlowData,
    NonCharacterFranchiseFlowData,
    PersonFlowData,
    PersonReference,
    SimilarityFlowData,
    SimilarityReference,
    StudioFlowData,
    StudioReference,
)
from search_v2.attribute_search import (
    DEFAULT_ATTRIBUTE_SEARCH_LIMIT,
    PersonSpec,
    run_attribute_search,
)
from search_v2.similar_movies import run_similar_movies_for_ids
from search_v2.streaming_orchestrator import (
    RerunPlan,
    stream_full_pipeline,
    stream_rerun_pipeline,
)
from search_v2.query_input_validation import (
    MAX_CLARIFICATION_CHARS,
    MAX_QUERY_CHARS,
    QueryInputError,
    clean_clarification,
    clean_query,
)
from search_v2.title_search import (
    TITLE_SEARCH_DEFAULT_LIMIT,
    TITLE_SEARCH_MAX_LIMIT,
    run_title_search,
)

# Shared msgspec JSON encoder. msgspec.Struct types (e.g. MovieCard)
# encode natively without Pydantic's model_dump round-trip; ~10-50×
# faster than stdlib json + Pydantic on the wire-format hot path.
_json_encoder = msgspec.json.Encoder()

# Module-level tracer for the manual pipeline/cache spans this file creates
# (observability Phase 1c). Auto-instrumentation gives us the request/DB/
# cache/HTTP spans for free; this tracer is for the semantic child spans and
# request-scoped attributes that only the application layer can populate.
tracer = trace.get_tracer(__name__)


class MoviePayloadSource(str, Enum):
    """Where a served movie payload originated, recorded as the
    `movie.payload_source` span attribute on /movie_details and /movie_credits.

    The KEY names the dimension (origin of the payload); these members name the
    values. Role-based, not tech-based — `CACHE` (not "redis") survives a
    cache-technology swap; `TMDB` is the specific upstream, so a future upstream
    just adds a member rather than forcing a rename.

    Set ONLY at a success point (never optimistically), so a 404/502 response
    carries no source — absence means "no payload was served". Cache-hit rate is
    then `count(source=cache) / count(source in {cache, tmdb})` over successful
    requests. str-valued so the member serializes directly via `.value`.
    """

    CACHE = "cache"
    TMDB = "tmdb"


# The per-request outcome vocabulary (`FailureReason`) and its recording
# mechanism (`EndpointFailure`, `record_outcome`) live in `api/outcome.py`; the
# two 404 reasons that used to sit here as `NotFoundReason` are now
# `FailureReason.NOT_INDEXED` / `.TMDB_REMOVED` among the broader failure set.


# Manual span names and attribute keys are defined once in observability/names.py
# (imported above) and referenced as constants — never inline literals — so the
# namespace root is written once and a typo can't silently split a metric. See
# that module's docstring for the naming rules.

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

# Initialize OpenTelemetry tracing + auto-instrumentation for this app.
# Must run after the app exists (instrument_app wraps THIS instance) and
# before it serves traffic, so every request becomes a root span. Safe at
# import time: the httpx/psycopg/redis instrument() calls patch those
# libraries at the class level, so clients created later (e.g. the lifespan
# TMDB client) are still traced. Idempotent — guarded inside setup_tracing.
setup_tracing(app)


# ---------------------------------------------------------------------------
# CORS
# ---------------------------------------------------------------------------
# Origins permitted to call this API from a browser. A browser's `Origin`
# header is scheme + host + port ONLY — it never carries a path or trailing
# slash — so listing the bare origin here covers every sub-path the
# frontend serves (e.g. "/", "/similar", ...). Do NOT add a trailing slash
# or path: it would never match and silently block requests.
#
# Origins are enumerated explicitly (not "*") because allow_credentials is
# True — browsers reject wildcard origins on credentialed requests, and we
# send a credential (the device-ID / auth cookie). CORS is browser-enforced
# only and is not a defense against scripted (curl/Python) callers; it
# stops other websites' JS from calling this API in a user's browser.
ALLOWED_ORIGINS = [
    "https://www.cinemind.dev",   # production frontend (all sub-paths)
    "http://localhost:3000",      # local dev frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,            # required to send the device-ID / auth cookie
    allow_methods=["GET", "POST"],     # the verbs this API actually serves
    allow_headers=["Authorization", "Content-Type"],
)


@app.get("/health")
async def health_check(request: Request):
    """
    Health check endpoint that validates connectivity to all external services.

    Returns a dictionary with status for each service:
    - postgres: 'ok' or error message (checked via connection pool)
    - redis: 'ok' or error message
    - qdrant: 'ok' or error message
    """
    # Cloudflare puts the real client IP in CF-Connecting-IP; the socket peer
    # is Cloudflare's edge, so this header is the only way to see the true
    # caller. Header names are case-insensitive in Starlette's Headers map.
    # Falls back to None when the request did not transit Cloudflare (e.g. a
    # direct/local call), so this stays safe in non-proxied environments.
    logger.info("/health request from CF-Connecting-IP=%s", request.headers.get("CF-Connecting-IP"))

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
      - keywords: ``OverallKeyword`` enum values ("Splatter Horror",
        "Film Noir", "Spaghetti Western", …) — sub-genre / style tags from
        the keyword taxonomy. Matched against movie_card.keyword_ids.
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
    keywords:           Optional[list[str]] = None
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

    try:
        keywords = (
            [OverallKeyword(name) for name in body_input.keywords]
            if body_input.keywords else None
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"unknown keyword value: {exc}"
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
        keywords=keywords,
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
    # Optional UI hard filters mirroring /query_search. When present and
    # at least one field is set, the similarity pipeline restricts every
    # candidate-generation lane (Postgres + Qdrant) to filter-eligible
    # movies. None or all-None fields is the "no filter" signal and the
    # request takes the same fast path as before.
    filters: Optional[MetadataFiltersInput] = None


class PersonInput(BaseModel):
    """One named-person filter on POST /attribute_search.

    `name` is the raw display name (we normalize server-side; the
    caller need not pre-normalize). The flow is role-agnostic — any
    credit on the movie (actor / director / writer / producer /
    composer) qualifies — so there is no role field.
    """

    name: str = Field(min_length=1)


# Upper bound on the number of person filters per request. Each person
# fans out to 5 parallel posting-table lookups, so a tight cap keeps a
# single request from saturating the Postgres pool (max_size=10 per
# conventions.md). 20 is far beyond any legitimate browse-UI use case
# while still leaving headroom for "all-time favorite filmmakers" lists.
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

    Strips whitespace from each name; drops persons whose name is blank
    after strip (defensive normalization at the boundary — mirrors the
    `clarification` handling in /query_search).
    """
    if not people_input:
        return []

    specs: list[PersonSpec] = []
    for entry in people_input:
        stripped = entry.name.strip()
        if not stripped:
            continue
        specs.append(PersonSpec(name=stripped))
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
                            completes (one per branch). Carries the
                            branch-level `intent_exploration` reasoning
                            prose alongside the `traits[]`. Skipped for
                            the non-standard flows.
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
      HTTP 200 with `text/event-stream` content. Empty or over-length
      `query`/`clarification` → 400.
    """
    # Validate/normalize the two free-text fields at the boundary via the
    # shared validator: strip, enforce non-empty + length cap, and treat
    # blank clarification as omitted (keeps the no-clarification fast path
    # stable — the pipeline only switches to clarification-mode prompts
    # when there is real correction text). Downstream layers
    # (stream_full_pipeline, run_full_pipeline, run_step_0/1) re-validate
    # too; each is a separate public surface also reachable from CLI
    # runners, so each defends its own contract. A QueryInputError here
    # (empty or over-length input) maps to a user-facing 400.
    try:
        query = clean_query(body.query)
        clarification: Optional[str] = clean_clarification(body.clarification)
    except QueryInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

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


# ---------------------------------------------------------------------------
# /rerun_query_search — replay a prior search with new filters, bypassing
# Steps 0 (flow routing) and 1 (spin generation). The caller passes the same
# branch data it received in the original /query_search `fetches_ready` event;
# standard branches re-enter at Step 2, entity flows at their executor.
# ---------------------------------------------------------------------------


class StandardRerunBranch(BaseModel):
    """A standard-flow branch to replay from Step 2.

    `query` is the expanded branch sub-query (echoed from the original
    response's standard fetch `query`). `ui_label` is an optional display
    override (echoed from the fetch `label`); when omitted it defaults to the
    query. The branch's internal `kind` (identity/label-only, never affects
    scoring) is assigned positionally by `_to_rerun_plan`, so it is not part
    of the wire contract — replaying branches in their original order
    reproduces the same fetch ids.
    """

    type: Literal["standard"]
    query: str = Field(min_length=1)
    ui_label: Optional[str] = None


class ExactTitleRerunBranch(BaseModel):
    """Replay the exact-title entity flow. `release_year` is null when the
    user never stated a year."""

    type: Literal["exact_title"]
    title: str = Field(min_length=1)
    release_year: Optional[int] = None


class SimilarityRefInput(BaseModel):
    """One anchor reference for a similarity replay. `release_year` is null
    when the user never stated a year."""

    title: str = Field(min_length=1)
    release_year: Optional[int] = None


class SimilarityRerunBranch(BaseModel):
    """Replay the similarity ("movies like X") entity flow."""

    type: Literal["similarity"]
    references: list[SimilarityRefInput] = Field(min_length=1)


class NonCharacterFranchiseRerunBranch(BaseModel):
    """Replay the non-character franchise entity flow."""

    type: Literal["non_character_franchise"]
    canonical_name: str = Field(min_length=1)


class CharacterFranchiseRerunBranch(BaseModel):
    """Replay the character-franchise entity flow."""

    type: Literal["character_franchise"]
    canonical_name: str = Field(min_length=1)


class StudioRerunBranch(BaseModel):
    """Replay the studio entity flow (one or more studio names)."""

    type: Literal["studio"]
    canonical_names: list[str] = Field(min_length=1)


class PersonRerunBranch(BaseModel):
    """Replay the person entity flow (one or more person names)."""

    type: Literal["person"]
    canonical_names: list[str] = Field(min_length=1)


# Discriminated union over the branch `type` tag. Pydantic routes each entry
# to the matching model and rejects an unknown `type` with 422 at the
# request-parsing boundary.
RerunBranch = Annotated[
    Union[
        StandardRerunBranch,
        ExactTitleRerunBranch,
        SimilarityRerunBranch,
        NonCharacterFranchiseRerunBranch,
        CharacterFranchiseRerunBranch,
        StudioRerunBranch,
        PersonRerunBranch,
    ],
    Field(discriminator="type"),
]


class RerunSearchBody(BaseModel):
    """Request body for POST /rerun_query_search.

    `branches` is the set of pipeline branches to replay — the same branch
    data the client received in the original /query_search `fetches_ready`
    event. `filters` is the new hard-filter set to apply (identical shape to
    /query_search); unset / all-None means no filters.
    """

    branches: list[RerunBranch] = Field(min_length=1)
    filters: Optional[MetadataFiltersInput] = None


# Standard-flow kinds assigned positionally to replayed standard branches.
# `kind` is identity/label-only (not load-bearing in scoring); reusing the
# upstream literals keeps fetch ids in the shape the frontend expects, and
# replaying branches in their original order reproduces the original ids. The
# upstream pipeline never emits more than three standard branches, so the cap
# is exact for a faithful replay rather than an arbitrary limit.
_RERUN_STANDARD_KINDS: tuple[str, str, str] = ("original", "spin_1", "spin_2")

# Largest branch query the upstream pipeline can emit. A normal branch query
# is a raw query or a spin (≤150 chars), but the failed-clarification slot-1
# branch is the merge `f"{raw_query}. {clarification}"` — up to
# MAX_QUERY_CHARS + ". " + MAX_CLARIFICATION_CHARS. Bounding the rerun branch
# query at that ceiling (rather than MAX_QUERY_CHARS) lets every branch the
# pipeline can produce round-trip, while still bounding client-supplied Gemini
# input for cost / prompt-injection surface.
_MAX_RERUN_BRANCH_QUERY_CHARS = MAX_QUERY_CHARS + MAX_CLARIFICATION_CHARS + 2


def _clean_branch_query(raw: str) -> str:
    """Strip + non-empty + cap a standard-flow branch query.

    Unlike `clean_query` (200-char cap for fresh user input), the rerun cap is
    `_MAX_RERUN_BRANCH_QUERY_CHARS` so the largest branch query the upstream
    pipeline can emit round-trips instead of being falsely rejected. 400 on
    blank or over-cap, matching /query_search's QueryInputError → 400 contract.
    """
    query = raw.strip()
    if not query:
        raise HTTPException(
            status_code=400, detail="standard branch query must be non-empty."
        )
    if len(query) > _MAX_RERUN_BRANCH_QUERY_CHARS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"standard branch query must be at most "
                f"{_MAX_RERUN_BRANCH_QUERY_CHARS} characters (got {len(query)})."
            ),
        )
    return query


def _enforce_name_cap(name: str, flow_label: str) -> None:
    """422 if a (stripped, non-empty) entity name exceeds the free-text cap.

    Entity names are client-supplied and flow into LLM prompts (studio /
    character-franchise translators) and Postgres lookups, so they get the
    same length bound as the standard query for cost / injection-surface
    parity (see query_input_validation.MAX_QUERY_CHARS).
    """
    if len(name) > MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"{flow_label} branch name must be at most "
                f"{MAX_QUERY_CHARS} characters (got {len(name)})."
            ),
        )


def _clean_one(raw: str, flow_label: str) -> str:
    """Strip a single entity name; 422 if blank or over the length cap.

    The entity flow-data schemas strip-and-require their name fields, so an
    all-whitespace value would raise a pydantic ValidationError (→ 500) deep
    in construction. Validating here keeps it a clean 422 at the boundary.
    """
    name = raw.strip()
    if not name:
        raise HTTPException(
            status_code=422, detail=f"{flow_label} branch has a blank name"
        )
    _enforce_name_cap(name, flow_label)
    return name


def _clean_names(raw_names: list[str], flow_label: str) -> list[str]:
    """Strip names, drop blanks, cap length; 422 if none remain or any name is
    over the cap (mirrors `_to_person_specs`). Returns at least one name."""
    names = [n.strip() for n in raw_names if n.strip()]
    if not names:
        raise HTTPException(
            status_code=422, detail=f"{flow_label} branch has no usable names"
        )
    for name in names:
        _enforce_name_cap(name, flow_label)
    return names


def _to_rerun_plan(branches: list[RerunBranch]) -> RerunPlan:
    """Translate the wire branch list into a `RerunPlan` (branch plan + entity
    flow-data).

    Mirrors `_to_metadata_filters`: all boundary validation lives here.
    Standard branches become `(kind, branch_query, ui_label)` tuples (kind
    assigned positionally; label defaulting to the query); each entity branch
    becomes its executor's flow-data object.

    Raises HTTPException 400 on a blank/over-length standard query, and 422
    on a duplicate entity flow, a blank/over-length entity name, or more than
    three standard branches.
    """
    branch_plan: list[tuple[str, str, str]] = []
    exact_title_flow_data: ExactTitleFlowData | None = None
    similarity_flow_data: SimilarityFlowData | None = None
    non_character_franchise_flow_data: NonCharacterFranchiseFlowData | None = None
    character_franchise_flow_data: CharacterFranchiseFlowData | None = None
    studio_flow_data: StudioFlowData | None = None
    person_flow_data: PersonFlowData | None = None

    def reject_duplicate(existing: object, flow_label: str) -> None:
        # The orchestrator has exactly one slot per entity flow, so a second
        # branch of the same entity type is a malformed request.
        if existing is not None:
            raise HTTPException(
                status_code=422, detail=f"duplicate {flow_label} branch"
            )

    standard_count = 0

    for branch in branches:
        if isinstance(branch, StandardRerunBranch):
            if standard_count >= len(_RERUN_STANDARD_KINDS):
                raise HTTPException(
                    status_code=422,
                    detail="at most 3 standard branches may be replayed",
                )
            # Validate the branch query (strip, non-empty, widened cap) → 400.
            query = _clean_branch_query(branch.query)
            # kind is assigned positionally — identity/label-only, and unique
            # by construction (one per slot), so no collision check is needed.
            kind = _RERUN_STANDARD_KINDS[standard_count]
            ui_label = branch.ui_label if branch.ui_label else query
            branch_plan.append((kind, query, ui_label))
            standard_count += 1
        elif isinstance(branch, ExactTitleRerunBranch):
            reject_duplicate(exact_title_flow_data, "exact_title")
            exact_title_flow_data = ExactTitleFlowData(
                should_be_searched=True,
                exact_title_to_search=_clean_one(branch.title, "exact_title"),
                release_year=branch.release_year,
            )
        elif isinstance(branch, SimilarityRerunBranch):
            reject_duplicate(similarity_flow_data, "similarity")
            refs: list[SimilarityReference] = []
            for r in branch.references:
                title = r.title.strip()
                if not title:
                    continue  # drop blank refs; only 422 if none remain
                _enforce_name_cap(title, "similarity")
                refs.append(
                    SimilarityReference(
                        similar_search_title=title,
                        release_year=r.release_year,
                    )
                )
            if not refs:
                raise HTTPException(
                    status_code=422,
                    detail="similarity branch has no usable references",
                )
            similarity_flow_data = SimilarityFlowData(
                should_be_searched=True, references=refs
            )
        elif isinstance(branch, NonCharacterFranchiseRerunBranch):
            reject_duplicate(
                non_character_franchise_flow_data, "non_character_franchise"
            )
            non_character_franchise_flow_data = NonCharacterFranchiseFlowData(
                canonical_name=_clean_one(
                    branch.canonical_name, "non_character_franchise"
                ),
            )
        elif isinstance(branch, CharacterFranchiseRerunBranch):
            reject_duplicate(
                character_franchise_flow_data, "character_franchise"
            )
            character_franchise_flow_data = CharacterFranchiseFlowData(
                canonical_name=_clean_one(
                    branch.canonical_name, "character_franchise"
                ),
            )
        elif isinstance(branch, StudioRerunBranch):
            reject_duplicate(studio_flow_data, "studio")
            studio_flow_data = StudioFlowData(
                references=[
                    StudioReference(canonical_name=n)
                    for n in _clean_names(branch.canonical_names, "studio")
                ],
            )
        elif isinstance(branch, PersonRerunBranch):
            reject_duplicate(person_flow_data, "person")
            person_flow_data = PersonFlowData(
                references=[
                    PersonReference(canonical_name=n)
                    for n in _clean_names(branch.canonical_names, "person")
                ],
            )

    return RerunPlan(
        branch_plan=branch_plan,
        exact_title_flow_data=exact_title_flow_data,
        similarity_flow_data=similarity_flow_data,
        non_character_franchise_flow_data=non_character_franchise_flow_data,
        character_franchise_flow_data=character_franchise_flow_data,
        studio_flow_data=studio_flow_data,
        person_flow_data=person_flow_data,
    )


@app.post("/rerun_query_search")
async def rerun_query_search(body: RerunSearchBody):
    """
    Re-run a prior search with a new filter set, bypassing Steps 0 and 1.

    Accepts the branches to replay (the same branch data returned in the
    original /query_search `fetches_ready` event) plus an optional `filters`
    block (identical shape to /query_search). Standard branches re-enter the
    pipeline at Step 2 (`run_step_2` on the branch query → Step 3 → Stage 4);
    entity flows re-enter at their executor. Streams the same Server-Sent
    Event sequence as /query_search (fetches_ready → per-branch branch_traits
    / branch_categories / branch_results → done), so the frontend consumes it
    unchanged.

    Returns:
      HTTP 200 with `text/event-stream`. 400 on a blank/over-length standard
      branch query; 422 on a duplicate entity flow, a blank/over-length entity
      name, more than three standard branches, an unknown branch `type`, or an
      unknown filter enum value.
    """
    # Translate the wire branches into a RerunPlan at the boundary (raises
    # 400/422), then translate filters with the same helper /query_search uses.
    plan = _to_rerun_plan(body.branches)
    metadata_filters = _to_metadata_filters(body.filters)

    async def event_stream():
        async for event_name, payload in stream_rerun_pipeline(
            plan, metadata_filters=metadata_filters,
        ):
            encoded = _json_encoder.encode(payload).decode("utf-8")
            yield f"event: {event_name}\ndata: {encoded}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            # Disable buffering at common reverse-proxy layers so events flush
            # to the client immediately rather than accumulating.
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

    An optional ``filters`` field (same shape as /query_search) applies
    UI hard filters — release range, runtime, maturity rank, genres,
    audio languages, streaming services — to every candidate-generation
    lane. Filtered requests get their own cache slot via a short stable
    hash of the active filter config; unfiltered requests continue to
    use the legacy anchor-id-only cache key.

    Errors:
      - Empty `tmdb_ids` → 422 (pydantic).
      - Unknown TMDB IDs → 422 with the missing IDs in the detail.
      - Unknown enum value in `filters` → 422.
    """
    # Canonicalize anchors (dedup + sort) so the cache key is order- and
    # duplicate-insensitive. run_similar_movies_for_ids dedups again
    # internally and doesn't depend on input ordering.
    canonical_ids = sorted(dict.fromkeys(int(mid) for mid in body.tmdb_ids))

    # Translate wire filters at the boundary. Raises 422 on unknown enum
    # values inside _to_metadata_filters; an all-None MetadataFiltersInput
    # collapses back to None here.
    metadata_filters = _to_metadata_filters(body.filters)

    # Filtered queries get their own cache slot via a stable short hash
    # of the active filter config; unfiltered queries continue to use
    # the legacy anchor-id-only key shape so existing warm entries stay
    # valid. Per the graceful-degradation convention (docs/conventions.md),
    # a Redis failure must NOT fail the request — fall through to the
    # cold path instead.
    fingerprint = metadata_filters_fingerprint(metadata_filters)

    try:
        cached = await get_cached_similar_movies(
            canonical_ids, filter_fingerprint=fingerprint,
        )
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
        result = await run_similar_movies_for_ids(
            canonical_ids, metadata_filters=metadata_filters,
        )
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
    # log and continue. The fingerprint keeps filtered and unfiltered
    # writes in disjoint key slots.
    try:
        await cache_similar_movies(
            canonical_ids, encoded, filter_fingerprint=fingerprint,
        )
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

    Applies the supplied filters (genres, keywords, audio languages,
    streaming services, release/runtime/maturity ranges) and ranks named
    people by prominence, reusing the Step 0 person-search model.

    With no `people`, returns the catalog ranked by the 80/20
    popularity/reception neutral prior. With one or more `people`, each
    person is resolved (role-agnostically, across all credit types) to
    a per-movie prominence bucket — LEAD / MAJOR / RELEVANT / MINOR —
    via the same resolver the Step 0 person flow uses. Each bucket maps
    to an additive weight (LEAD=4 … MINOR=1) and weights are SUMMED
    across people (UNION — any credit qualifies, and crediting more or
    more-prominent people ranks higher). Within a prominence tier,
    movies are ordered by popularity_score (then movie_id).

    A single person with no metadata filters yields the SAME ordering
    as the Step 0 person flow by construction. Unresolvable person
    names contribute no credits silently — they neither error nor empty
    the result.

    Returns:
      HTTP 200 with an array of MovieCard objects
      (tmdb_id, title, release_date, poster_url, maturity_rating),
      ranked by descending summed prominence weight (popularity as the
      within-tier tie-break), capped at 250.
      HTTP 422 on unknown genre / keyword / audio_language /
      streaming_service enum values (handled inside `_to_metadata_filters`).
    """
    # Translate the wire mirror into MetadataFilters once at the
    # boundary. Raises 422 on any unknown enum value; collapses to
    # None if every filter field is unset so the orchestrator's
    # SQL stays filter-clause-free on unfiltered queries.
    metadata_filters = _to_metadata_filters(body.filters)

    # Convert wire-level PersonInput list into the internal PersonSpec
    # list (stripped name). Blank-after-strip entries are dropped here
    # so the orchestrator never sees them.
    people_specs = _to_person_specs(body.people)

    ranked_ids = await run_attribute_search(
        people=people_specs,
        metadata_filters=metadata_filters,
        limit=DEFAULT_ATTRIBUTE_SEARCH_LIMIT,
    )

    # Hydrate ranked movie_ids into MovieCard summaries. Order is
    # preserved by `fetch_movie_card_summaries`, so the response stays
    # in the orchestrator's ranked order (summed prominence band, or
    # the neutral prior on the no-people path). Same msgspec-encode hot
    # path as /similarity_search.
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
@record_outcome
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
        raise EndpointFailure(
            status_code=422,
            failure_reason=FailureReason.INVALID_PARAMETERS,
            detail="q must be non-empty",
        )

    # `limit` is validated as a closed range matching the spec. The
    # orchestrator trusts its caller for clamping, so we have to do it
    # here.
    if limit < 1 or limit > TITLE_SEARCH_MAX_LIMIT:
        raise EndpointFailure(
            status_code=422,
            failure_reason=FailureReason.INVALID_PARAMETERS,
            detail="limit out of range",
        )

    # Resolve to ranked movie_ids; then hydrate to the wire-format
    # MovieCard via the same single-query summary fetch all sibling
    # endpoints use. Order is preserved by `fetch_movie_card_summaries`
    # so the tier-then-popularity ordering survives intact.
    result = await run_title_search(trimmed, limit=limit)
    cards = await fetch_movie_card_summaries(result.movie_ids)

    # Record the request-scoped facts on the FastAPI request (server) span,
    # not the auto psycopg child spans: these describe THIS request and the
    # server span is the queryable root the two DB spans already hang off.
    # `get_current_span()` returns that active server span here (no manual
    # child span is warranted — title_search is a single unit of work).
    # `result_count` uses the hydrated cards (what the client actually got);
    # `fuzzy_result_count` > 0 means the fuzzy fallback fired, the signal for
    # likely typos or catalog gaps. NOTE: `query` is high-cardinality — fine
    # as a span attribute (per-trace) but must never become a metric label.
    span = trace.get_current_span()
    span.set_attribute(TITLE_SEARCH_QUERY, trimmed)
    span.set_attribute(TITLE_SEARCH_LIMIT, limit)
    span.set_attribute(TITLE_SEARCH_RESULT_COUNT, len(cards))
    span.set_attribute(TITLE_SEARCH_FUZZY_RESULT_COUNT, result.fuzzy_count)

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

# Global budget for the curated /movie_details crew list, counted in
# DISTINCT PEOPLE (not rows). Each selected person contributes all of their
# credits, so the emitted list can hold more rows than this — but never more
# than this many distinct people. 12 mirrors the cast cap for visual balance.
_CREW_PERSON_LIMIT = 12

# Guaranteed priority slots filled before the importance top-up. All
# directors are always included (subject only to the global cap); writers
# and producers get their top N each, in TMDB billing order.
_WRITER_PRIORITY_COUNT = 3
_PRODUCER_PRIORITY_COUNT = 3

# Job titles TMDB uses for each priority role. "Screenplay" and "Story"
# both count as writing since TMDB splits them out; "Executive Producer"
# rolls up into producers.
_WRITER_JOBS = {"Writer", "Screenplay", "Story"}
_PRODUCER_JOBS = {"Producer", "Executive Producer"}
_DIRECTOR_JOBS = {"Director"}

# Importance ranking for the fill phase (after directors + top writers/
# producers are placed). A person's rank is the lowest index any of their
# jobs maps to here; jobs absent from this tuple rank last and fall back to
# TMDB first-seen order. Leftover writers/producers (beyond the priority
# counts) re-enter here, ranked below the key below-the-line department heads
# (cinematographer, composer, editor, production designer) the details page
# most wants to surface.
_CREW_FILL_JOB_PRIORITY = (
    "Director of Photography",
    "Original Music Composer",
    "Music",
    "Editor",
    "Production Designer",
    "Writer",
    "Screenplay",
    "Story",
    "Producer",
    "Executive Producer",
    "Costume Design",
    "Casting",
    "Art Direction",
)
_CREW_FILL_JOB_RANK = {job: i for i, job in enumerate(_CREW_FILL_JOB_PRIORITY)}
# Sentinel rank for any job not in the priority tuple — sorts after all
# ranked jobs, leaving TMDB first-seen order as the effective tiebreak.
_CREW_FILL_RANK_FALLBACK = len(_CREW_FILL_JOB_PRIORITY)

# Department order for the uncapped /movie_credits crew list. We lead with
# the three departments the details page already surfaces so the full view
# reads as an expansion of what the user just saw, not a reshuffle. Every
# other department TMDB returns (Camera, Editing, Sound, Art, …) follows in
# TMDB's natural first-seen order, which is deterministic.
_PRIORITY_CREW_DEPARTMENTS = ("Directing", "Writing", "Production")


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


class _ExtractedCredits(NamedTuple):
    """Result of splitting a TMDB credits block into curated sections.

    `crew` is a single ranked list (see `_extract_credits`). The two
    truncation booleans tell the frontend whether to show a "See all" link
    per section. Each is derived by comparing the full pre-cap/pre-filter
    TMDB count against the count we actually return — never by referencing a
    cap constant — so the flags stay correct if the caps ever change. Named
    fields keep the call site self-documenting and immune to positional mixups.
    """

    crew: list[CrewMember]
    cast: list[CastMember]
    cast_truncated: bool
    crew_truncated: bool


class _CrewPerson(NamedTuple):
    """One person's grouped crew credits plus role flags for selection.

    `rows` holds every `CrewMember` credit for this person in TMDB order;
    `jobs` is the set of their job labels used for role/importance tests.
    `first_seen` is the person's position in TMDB's crew array — the stable
    tiebreak that makes selection deterministic.
    """

    rows: list[CrewMember]
    jobs: set[str]
    first_seen: int
    is_director: bool
    is_writer: bool
    is_producer: bool


def _person_fill_rank(jobs: set[str]) -> int:
    """Importance rank for the fill phase: lowest index across a person's jobs.

    Lower is more important. Jobs absent from `_CREW_FILL_JOB_PRIORITY` map to
    the fallback rank, leaving TMDB first-seen order as the effective tiebreak.
    """
    return min(
        (_CREW_FILL_JOB_RANK.get(job, _CREW_FILL_RANK_FALLBACK) for job in jobs),
        default=_CREW_FILL_RANK_FALLBACK,
    )


def _group_crew_by_person(crew_entries: list[dict]) -> list[_CrewPerson]:
    """Collapse TMDB crew rows into per-person buckets, in first-seen order.

    Rows are keyed by TMDB person `id`; a row missing `id` gets a per-row
    sentinel so distinct anonymous credits never merge. Nameless rows are
    skipped — they can't render. The backend deliberately keeps every credit
    (one `CrewMember` per job); consumers dedupe by person for display.
    """
    index_by_key: dict[object, int] = {}  # person key -> index into `buckets`
    buckets: list[dict] = []              # mutable accumulators, first-seen order
    for entry in crew_entries:
        name = entry.get("name")
        if not name:
            continue
        # `id` is TMDB's stable person identifier; fall back to the row's
        # object identity when it's absent so two anonymous credits stay split.
        key = entry.get("id")
        if key is None:
            key = ("anon", id(entry))
        job = entry.get("job") or ""
        member = CrewMember(
            name=name,
            job=job,
            profile_url=_image_url(entry.get("profile_path"), _PROFILE_SIZE),
        )
        idx = index_by_key.get(key)
        if idx is None:
            index_by_key[key] = len(buckets)
            buckets.append({"rows": [member], "jobs": {job}})
        else:
            buckets[idx]["rows"].append(member)
            buckets[idx]["jobs"].add(job)

    people: list[_CrewPerson] = []
    for first_seen, b in enumerate(buckets):
        jobs = b["jobs"]
        people.append(
            _CrewPerson(
                rows=b["rows"],
                jobs=jobs,
                first_seen=first_seen,
                is_director=bool(jobs & _DIRECTOR_JOBS),
                is_writer=bool(jobs & _WRITER_JOBS),
                is_producer=bool(jobs & _PRODUCER_JOBS),
            )
        )
    return people


def _select_crew_people(people: list[_CrewPerson]) -> list[_CrewPerson]:
    """Choose up to `_CREW_PERSON_LIMIT` people in priority order.

    Order: all directors, then the top writers, then the top producers, then
    the most important remaining crew by `_person_fill_rank`. A person already
    chosen is never taken again (a director-writer fills one slot), and every
    phase yields to the global cap.
    """
    selected: list[_CrewPerson] = []
    chosen: set[int] = set()  # first_seen indices already taken

    def take(person: _CrewPerson) -> None:
        chosen.add(person.first_seen)
        selected.append(person)

    def take_top(predicate, count: int) -> None:
        """Take up to `count` not-yet-chosen people matching `predicate`."""
        taken = 0
        for p in people:
            if len(selected) >= _CREW_PERSON_LIMIT or taken >= count:
                return
            if p.first_seen in chosen or not predicate(p):
                continue
            take(p)
            taken += 1

    # Phase 1: every director, first-seen order, subject only to the cap.
    for p in people:
        if len(selected) >= _CREW_PERSON_LIMIT:
            return selected
        if p.is_director:
            take(p)

    # Phases 2/3: top writers, then top producers.
    take_top(lambda p: p.is_writer, _WRITER_PRIORITY_COUNT)
    take_top(lambda p: p.is_producer, _PRODUCER_PRIORITY_COUNT)
    if len(selected) >= _CREW_PERSON_LIMIT:
        return selected

    # Phase 4: top up by importance, breaking ties on first-seen order.
    remaining = [p for p in people if p.first_seen not in chosen]
    remaining.sort(key=lambda p: (_person_fill_rank(p.jobs), p.first_seen))
    for p in remaining:
        if len(selected) >= _CREW_PERSON_LIMIT:
            break
        take(p)
    return selected


def _extract_credits(credits: dict | None) -> _ExtractedCredits:
    """Split TMDB's credits block into a curated crew list, cast, and flags.

    Crew is grouped by person and selected by priority (see
    `_select_crew_people`), then expanded so each chosen person contributes
    all of their credits. Cast is taken in TMDB's native order (already sorted
    by billing position), truncated to `_CAST_LIMIT`.
    """
    if not credits:
        return _ExtractedCredits([], [], False, False)

    people = _group_crew_by_person(credits.get("crew", []))
    selected = _select_crew_people(people)
    # Expand selected people back into a flat row list, preserving selection
    # order across people and TMDB order within each person.
    crew: list[CrewMember] = [member for person in selected for member in person.rows]

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

    # Truncation flags compare what we kept against the real totals — never a
    # cap constant. Crew counts DISTINCT PEOPLE: True iff we dropped anyone.
    cast_truncated = len(cast) < len(credits.get("cast", []))
    crew_truncated = len(selected) < len(people)

    return _ExtractedCredits(crew, cast, cast_truncated, crew_truncated)


def _build_movie_credits(tmdb_id: int, payload: dict) -> MovieCredits:
    """Build the full, uncapped MovieCredits payload from a TMDB detail dict.

    Pure function — no I/O. Unlike `_extract_credits` (which caps cast and
    keeps only three job buckets for the lean /movie_details view), this is
    the lazy "See all" companion: every billed cast member and every crew
    department TMDB returns, with no caps and no merging.

    Cast keeps TMDB's billing order. Crew is grouped by the canonical TMDB
    `department` field — one entry per (person, job), so a person credited
    for two jobs (or across two departments) appears once per credit. Group
    order leads with `_PRIORITY_CREW_DEPARTMENTS` (matching what the details
    page showed) and then follows TMDB's first-seen department order; member
    order within a group is TMDB's natural order. Both orders are fixed
    server-side per the backend-is-authoritative convention.
    """
    credits = payload.get("credits") or {}

    # Cast: full billed list in TMDB order. Skip nameless entries; normalize
    # blank character/profile to None so they drop via omit_defaults.
    cast: list[CastMember] = []
    for entry in credits.get("cast", []):
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

    # Crew: group by department in a single pass. A plain dict preserves
    # first-seen insertion order for both departments and the members within
    # each, which gives us the deterministic ordering the contract requires.
    grouped: dict[str, list[CrewMember]] = {}
    for entry in credits.get("crew", []):
        name = entry.get("name")
        if not name:
            continue
        # `department` is reliably present on TMDB crew rows; fall back to
        # "Crew" (itself a canonical TMDB department) so a real credit is
        # never silently dropped if it's ever missing.
        department = entry.get("department") or "Crew"
        grouped.setdefault(department, []).append(
            CrewMember(
                name=name,
                job=entry.get("job") or "",
                profile_url=_image_url(entry.get("profile_path"), _PROFILE_SIZE),
            )
        )

    # Emit the priority departments first (only if present), then the rest in
    # first-seen order (dict iteration preserves insertion order).
    ordered_departments = [d for d in _PRIORITY_CREW_DEPARTMENTS if d in grouped]
    ordered_departments += [
        d for d in grouped if d not in _PRIORITY_CREW_DEPARTMENTS
    ]
    crew = [
        CrewGroup(department=d, members=grouped[d]) for d in ordered_departments
    ]

    return MovieCredits(tmdb_id=tmdb_id, cast=cast, crew=crew)


async def _encode_and_cache_credits(tmdb_id: int, tmdb_payload: dict) -> bytes:
    """Build, encode, and best-effort cache the MovieCredits for a movie.

    The single write path for the `tmdb:credits:{id}` cache. Both callers
    funnel through here so the key is never written from two places:
      - /movie_credits caches the payload it just fetched, then returns it.
      - /movie_details cross-populates it for free from the detail payload
        it already fetched (the "See all" follow-up always comes after a
        details view), so that follow-up is a warm hit instead of a second
        TMDB round trip.

    Returns the encoded bytes so the caller can return them verbatim. A
    cache-write failure is logged but never fatal — per the
    graceful-degradation convention, Redis is an optimization, not a
    dependency.

    Observability: wrapped in a `movie_credits.build_and_cache` span so BOTH
    callers are covered by one instrumentation point — it appears as the main
    build path under /movie_credits and as the cross-populate step nested under
    /movie_details (in-process context nesting; no details-side code needed).
    The prefix names the payload (movie-credits), which is accurate under both
    parents; the root span identifies which endpoint actually ran.
    """
    with tracer.start_as_current_span(MOVIE_CREDITS_BUILD_AND_CACHE) as span:
        # Capture the built struct (rather than encoding inline) so we can read
        # its size for the count attributes below before serializing.
        credits = _build_movie_credits(tmdb_id, tmdb_payload)
        # `.crew` is grouped by department (list[CrewGroup]), so the true
        # crew-member total is the sum across groups, not len(crew) — that
        # would be the department count. Useful for the "See all" view: how
        # large do the uncapped cast/crew lists actually get?
        span.set_attribute(MOVIE_CREDITS_CAST_COUNT, len(credits.cast))
        span.set_attribute(
            MOVIE_CREDITS_CREW_COUNT,
            sum(len(group.members) for group in credits.crew),
        )
        encoded = _json_encoder.encode(credits)
        # Best-effort cache write. `cache.write_ok` makes the swallowed-failure
        # degradation queryable; the span is NOT marked ERROR — a failed cache
        # write is a degradation, not a request failure.
        try:
            await cache_movie_credits(tmdb_id, encoded)
            span.set_attribute(CACHE_WRITE_OK, True)
        except Exception:
            span.set_attribute(CACHE_WRITE_OK, False)
            span.add_event("credits cache write failed")
            logger.warning(
                "movie_credits cache write failed for tmdb_id=%s",
                tmdb_id,
                exc_info=True,
            )
        return encoded


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
    credits = _extract_credits(payload.get("credits"))

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

    # Genres come from our own (IMDB-derived) Postgres data, not TMDB.
    # movie_card stores them as stable numeric IDs; map each back to its
    # enum display name and drop any unrecognized IDs.
    genre_names = [
        genre.value
        for genre in (Genre.from_id(gid) for gid in card_row.get("genre_ids") or [])
        if genre is not None
    ]

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
        # Genres and spoken languages come from our own (IMDB-derived)
        # Postgres data, not TMDB. This deliberately replaces the TMDB values.
        genres=genre_names,
        # Keyword tags from our taxonomy (movie_card.keyword_ids), finer-grained
        # than `genres`. Excludes any keyword that duplicates a returned genre.
        keywords=keyword_names_from_ids(
            card_row.get("keyword_ids") or [], genre_names
        ),
        spoken_languages=[
            language.value
            for language in (
                Language.from_id(lid)
                for lid in card_row.get("audio_language_ids") or []
            )
            if language is not None
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
        crew=credits.crew,
        cast=credits.cast,
        cast_truncated=credits.cast_truncated,
        crew_truncated=credits.crew_truncated,
        watch_providers=_extract_us_watch_providers(payload.get("watch/providers")),
        tmdb_url=f"{_TMDB_MOVIE_PAGE_BASE}/{tmdb_id}",
        imdb_url=f"{_IMDB_TITLE_PAGE_BASE}/{imdb_id}" if imdb_id else None,
        homepage=homepage,
    )


async def _fetch_movie_payload(
    tmdb_id: int,
    *,
    span_name: str,
    tmdb_fetch: Callable[..., Awaitable[dict | None]],
) -> tuple[object, dict]:
    """Existence-gate a movie, then fetch its TMDB payload, under one span.

    Shared by /movie_details and /movie_credits so the cold-path fetch and its
    404/502 error contract live in exactly one place. Opens `span_name` (the
    endpoint's `*.payload_creation` span — the two network-bound calls the auto
    psycopg/httpx spans nest under) and:

      - returns `(card_row, tmdb_payload)` on success (both non-None);
      - raises `EndpointFailure` 404 (`NOT_INDEXED` — absent from our index — or
        `TMDB_REMOVED` — gone upstream), NOT marked as span errors (expected
        outcomes);
      - raises `EndpointFailure` 502 (`TMDB_FETCH_FAILED`) on `TMDBFetchError` —
        the span is marked ERROR + the exception recorded (a genuine upstream
        failure).

    In every case the failure carries a `FailureReason` that bubbles up to the
    `record_outcome` decorator, which sets `outcome.success`/`outcome.failure_reason`
    on the server span once — this helper never touches the server span itself.

    The span disables auto status/record so expected 404s stay UNSET, but an
    UNEXPECTED exception (DB error, malformed payload) is explicitly marked
    ERROR + recorded, so the span containing the failing op never reads green.
    """
    with tracer.start_as_current_span(
        span_name, record_exception=False, set_status_on_exception=False
    ) as span:
        try:
            # Reject unknown movies before hitting TMDB. One Postgres query, so
            # its auto psycopg span (nested here) already IS the index check —
            # no dedicated span needed.
            card_row = await fetch_movie_card_row(tmdb_id)
            if card_row is None:
                raise EndpointFailure(
                    status_code=404,
                    failure_reason=FailureReason.NOT_INDEXED,
                    detail="movie not found",
                )

            # Shared TMDB client + rate limiter live on app.state (lifespan).
            try:
                tmdb_payload = await tmdb_fetch(
                    app.state.tmdb_client, app.state.tmdb_rate_limiter, tmdb_id
                )
            except TMDBFetchError as exc:
                # Surface as 502 (bad gateway) so the frontend can tell "TMDB is
                # unhappy" from "our server is broken". A genuine failure of this
                # span's work → mark ERROR + record the root cause.
                span.set_status(Status(StatusCode.ERROR))
                span.record_exception(exc)
                raise EndpointFailure(
                    status_code=502,
                    failure_reason=FailureReason.TMDB_FETCH_FAILED,
                    detail=f"TMDB fetch failed: {exc}",
                ) from exc
            if tmdb_payload is None:
                raise EndpointFailure(
                    status_code=404,
                    failure_reason=FailureReason.TMDB_REMOVED,
                    detail="movie not found on TMDB",
                )

            return card_row, tmdb_payload
        except HTTPException:
            # Expected client/gateway outcomes (404s stay UNSET; the 502 already
            # marked ERROR above) — propagate without further span mutation.
            raise
        except Exception as exc:
            # Unexpected failure inside payload creation — mark + record so the
            # span reflects where it actually broke (the flags above suppress
            # this by default).
            span.set_status(Status(StatusCode.ERROR))
            span.record_exception(exc)
            raise


@app.get("/movie_details/{tmdb_id}")
@record_outcome
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
    # Capture the FastAPI request (server) span once. Inside the manual
    # child-span blocks below, trace.get_current_span() would return the child,
    # so request-scoped attributes (source, movie id, 404 reason) must be set on
    # this captured reference. movie.tmdb_id is set unconditionally so every
    # trace — success or error — is queryable by id (the path carries it, but
    # only buried inside the URL string, not as a clean attribute).
    request_span = trace.get_current_span()
    request_span.set_attribute(MOVIE_TMDB_ID, tmdb_id)

    # 1. Redis warm path. Return cached bytes verbatim — they're already
    # the encoded MovieDetails the frontend expects. Per the
    # graceful-degradation convention (docs/conventions.md), a Redis
    # failure here must NOT fail the request — fall through to the
    # cold path instead.
    try:
        cached = await get_cached_movie_details(tmdb_id)
    except Exception:
        # A read failure (Redis down) is distinct from a miss — both fall
        # through to the cold path, but the event surfaces the outage in the
        # trace, not just the log.
        request_span.add_event("cache read failed")
        logger.warning(
            "movie_details cache read failed for tmdb_id=%s", tmdb_id, exc_info=True
        )
        cached = None
    if cached is not None:
        # Success point: served from cache.
        request_span.set_attribute(
            MOVIE_PAYLOAD_SOURCE, MoviePayloadSource.CACHE.value
        )
        return Response(content=cached, media_type="application/json")

    # 2-3. Cold path: existence-gate + TMDB fetch under the shared
    # payload_creation span, which owns the 404/502 + unexpected-error contract.
    card_row, tmdb_payload = await _fetch_movie_payload(
        tmdb_id,
        span_name=MOVIE_DETAILS_PAYLOAD_CREATION,
        tmdb_fetch=fetch_movie_details_for_endpoint,
    )

    # 4. Build + encode the curated payload. Fast CPU, intentionally untimed
    # (no standalone span) — the meaningful latency is the network fetch above;
    # add a span only if the recombine itself ever gets slow. Encode once so the
    # cached and returned bytes are byte-identical.
    details = _build_movie_details(tmdb_id, tmdb_payload, card_row)
    encoded = _json_encoder.encode(details)

    # 5. Cache the encoded payload. Cache-write failures must NOT lose the
    # response we already built — record write_ok + an event and continue. This
    # is the ONLY write_ok movie_details sets; the credits cross-populate's
    # write_ok below is owned by the shared helper's span.
    with tracer.start_as_current_span(MOVIE_DETAILS_CACHE_WRITE) as cache_span:
        try:
            await cache_movie_details(tmdb_id, encoded)
            cache_span.set_attribute(CACHE_WRITE_OK, True)
        except Exception:
            cache_span.set_attribute(CACHE_WRITE_OK, False)
            cache_span.add_event("details cache write failed")
            logger.warning(
                "movie_details cache write failed for tmdb_id=%s",
                tmdb_id,
                exc_info=True,
            )

    # Cross-populate the /movie_credits cache from the same payload — the "See
    # all" follow-up always trails a details view, so warming it here turns that
    # into a cache hit for one extra build + Redis write, no extra TMDB trip.
    # Strictly best-effort: a build/encode failure here must NOT sink the details
    # response we already have (the helper only swallows cache-WRITE errors, not
    # build errors), so guard the whole call. Its own
    # `movie_credits.build_and_cache` span nests under this trace.
    try:
        await _encode_and_cache_credits(tmdb_id, tmdb_payload)
    except Exception:
        request_span.add_event("credits cross-populate failed")
        logger.warning(
            "movie_details credits cross-populate failed for tmdb_id=%s",
            tmdb_id,
            exc_info=True,
        )

    # Success point: served from a live TMDB fetch.
    request_span.set_attribute(MOVIE_PAYLOAD_SOURCE, MoviePayloadSource.TMDB.value)
    return Response(content=encoded, media_type="application/json")


@app.get("/movie_credits/{tmdb_id}")
@record_outcome
async def movie_credits(tmdb_id: int) -> Response:
    """Return the complete, uncapped cast & crew for a single movie.

    The lazy "See all" companion to /movie_details — credits only, no movie
    metadata. Same upstream credits data, but with no caps and crew grouped
    by department. Cached under `tmdb:credits:{id}` (24h TTL).

    Cache-first: the credits cache is normally warmed by the /movie_details
    view the user came from (which cross-populates it from its own fetch),
    so the common path is a hit with no TMDB call at all. The cold path is
    the rare fallback (details cache expired, served by a different instance,
    or a direct hit) and uses a lean credits-only TMDB fetch.

    Pipeline:
      1. Try Redis (24h cache of the encoded MovieCredits bytes).
      2. Confirm the movie is in our index (`public.movie_card`); 404 if not.
      3. Lean credits-only TMDB fetch.
      4. Build, cache, and return the uncapped MovieCredits payload.

    Errors:
      - 404 if `tmdb_id` is not in our `movie_card` index, or TMDB itself
        no longer serves the movie.
      - 502 if the TMDB fetch fails after retries (network / 5xx).
    """
    # Capture the request (server) span once — same rationale as
    # /movie_details: child-span blocks below shadow get_current_span(), and
    # movie.tmdb_id must be queryable on every path.
    request_span = trace.get_current_span()
    request_span.set_attribute(MOVIE_TMDB_ID, tmdb_id)

    # 1. Redis warm path. Return cached bytes verbatim. Per the
    # graceful-degradation convention (docs/conventions.md), a Redis
    # failure here must NOT fail the request — fall through to the cold path.
    try:
        cached = await get_cached_movie_credits(tmdb_id)
    except Exception:
        request_span.add_event("cache read failed")
        logger.warning(
            "movie_credits cache read failed for tmdb_id=%s", tmdb_id, exc_info=True
        )
        cached = None
    if cached is not None:
        # Success point: served from cache. This is the headline signal — the
        # credits cache is normally pre-warmed by /movie_details' cross-populate,
        # so a low cache rate here is how a silently-broken cross-populate shows.
        request_span.set_attribute(
            MOVIE_PAYLOAD_SOURCE, MoviePayloadSource.CACHE.value
        )
        return Response(content=cached, media_type="application/json")

    # 2-3. Cold path: existence-gate + lean credits-only TMDB fetch under the
    # shared payload_creation span (same 404/502 + unexpected-error contract as
    # /movie_details). card_row is unused beyond the existence gate here — no
    # reception fold-in — so it's discarded.
    _card_row, tmdb_payload = await _fetch_movie_payload(
        tmdb_id,
        span_name=MOVIE_CREDITS_PAYLOAD_CREATION,
        tmdb_fetch=fetch_movie_credits_for_endpoint,
    )

    # 4. Build, cache, and return — same bytes written and returned so the
    # warm-path response is byte-identical to this cold-path response. The
    # helper's `movie_credits.build_and_cache` span nests under this trace.
    encoded = await _encode_and_cache_credits(tmdb_id, tmdb_payload)

    # Success point: served from a live TMDB fetch.
    request_span.set_attribute(MOVIE_PAYLOAD_SOURCE, MoviePayloadSource.TMDB.value)
    return Response(content=encoded, media_type="application/json")