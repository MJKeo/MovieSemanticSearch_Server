# api/ — FastAPI Application

The HTTP interface for the search engine.

## What This Module Does

Provides the FastAPI application with lifecycle management
(opening/closing all database connection pools at startup/shutdown)
and API endpoints for search and movie detail retrieval.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app setup, lifespan context manager, health check endpoint, `/query_search`, `/similarity_search`, `/movie_details/{tmdb_id}`, `MetadataFiltersInput` request model, and `_build_movie_details` translator. |
| `cli_search.py` | CLI tool to run the full search pipeline from the terminal. Supports genre, maturity, runtime, and release date filters. Run via `python -m api.cli_search "query"`. |

## Boundaries

- **In scope**: HTTP request handling, connection pool lifecycle,
  endpoint definitions, request/response serialization.
- **Out of scope**: Search logic (delegated to `db/search.py`),
  query understanding (delegated to `implementation/llms/`),
  data models (in `implementation/classes/`).

## Endpoints

### `POST /query_search`
- **Request**: `QuerySearchBody` — `query` (string), optional `filters`
  (`MetadataFiltersInput`), optional `shown_movie_counts`.
- **Filters**: `MetadataFiltersInput` exposes six hard filters:
  - `release_date` — start/end Unix timestamps
  - `runtime` — min/max in minutes
  - `maturity` — min/max maturity rank
  - `genres` — list of genre display strings (e.g. `"Action"`)
  - `audio_languages` — list of language display strings (e.g. `"English"`)
  - `streaming_services` — list of `StreamingService` enum values
    (e.g. `"netflix"`, `"max"`); API expands these into encoded
    `watch_offer_keys` via `STREAMING_PROVIDER_MAP ×
    StreamingAccessType` fan-out (see Gotchas below).
- **Response**: Ranked results with score breakdowns.
- Filters are applied at retrieval time inside every V2 pipeline
  primitive (Postgres + Qdrant), not post-hoc — ensures filter-
  respecting candidates are retrieved rather than filtered out
  after the fact.

### `POST /similarity_search`
- **Request**: `SimilaritySearchBody` — `tmdb_ids` (non-empty list of int).
- **Response**: Array of `MovieCard` objects ranked by descending
  similarity score. Bypasses the natural-language pipeline.
- Errors: empty `tmdb_ids` → 422 (pydantic), unknown IDs → 422,
  invalid anchor → 400.

### `GET /movie_details/{tmdb_id}`
- **Response**: Curated `MovieDetails` msgspec struct (see
  `schemas/api_responses.py`) combining TMDB live data (overview,
  cast/crew, watch providers, trailer, images) with our locally-
  computed `reception_score`.
- **Flow**:
  1. Check Redis (`tmdb:movie:{id}`, 24h TTL) — return cached bytes
     verbatim on hit.
  2. On miss, confirm the movie is in `public.movie_card` (404 if not).
  3. Fetch `https://api.themoviedb.org/3/movie/{id}` with
     `append_to_response=credits,videos,images,external_ids,watch/providers,release_dates`.
  4. Translate to `MovieDetails` via `_build_movie_details`, cache, return.
- **Errors**: 404 if movie not in `movie_card` or TMDB returns 404;
  502 if the TMDB fetch fails after retries.
- **Graceful degradation**: Redis read/write failures are logged but
  do not fail the request — the cold path still serves the response.

### `GET /health`
- Validates connectivity to Postgres, Redis, and Qdrant.

## Lifecycle

The lifespan context manager handles:
1. **Startup**: Opens Postgres pool (`await pool.open()`,
   `await pool.check()`), initializes Redis (`await init_redis()`),
   builds the shared TMDB httpx client + `AdaptiveRateLimiter` and
   stores them on `app.state.tmdb_client` / `app.state.tmdb_rate_limiter`
   for the `/movie_details` endpoint.
2. **Shutdown**: Closes the TMDB httpx client and all connection pools
   gracefully.

Note: Qdrant connectivity is verified via the `/health` endpoint,
not at startup.

## Gotchas

- All database pools must be opened before the first request.
  The lifespan manager ensures this.
- TMDB detail responses are cached in Redis for 1 day under
  `tmdb:movie:{id}`. The cached payload is the *curated* MovieDetails
  wire format (not raw TMDB JSON), so warm hits skip both the upstream
  call and the build/encode step. The server acts as a caching layer —
  clients never call TMDB directly.
- The shared TMDB httpx client and rate limiter on `app.state` are
  reused across all `/movie_details` requests. Do not construct
  per-request clients (TLS handshake cost dominates).
- **Streaming-service filter key encoding**: `_to_metadata_filters`
  expands each `StreamingService` into encoded `(provider_id << 4) |
  method_id` values (not raw TMDB provider IDs). Raw provider IDs
  never match the encoded `watch_offer_keys` column; always use the
  fan-out over `StreamingAccessType` IDs, the same pattern as
  `_precompute_streaming_keys` in
  `search_v2/endpoint_fetching/metadata_query_execution.py`.
