# api/ — FastAPI Application

The HTTP interface for the search engine.

## What This Module Does

Provides the FastAPI application with lifecycle management
(opening/closing all database connection pools at startup/shutdown)
and API endpoints for search and movie detail retrieval.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app setup, lifespan context manager, health check endpoint, `/query_search` endpoint, `MetadataFiltersInput` request model. |
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

### `GET /movie/{movie_id}`
- Check Redis for cached TMDB JSON → return on hit
- On miss: call TMDB API → store in Redis (TTL 1 day) → return
- Proxies TMDB through the server to keep API secrets off the
  client, enable caching, and normalize responses.

### `GET /health`
- Validates connectivity to Postgres, Redis, and Qdrant.

## Lifecycle

The lifespan context manager handles:
1. **Startup**: Opens Postgres pool (`await pool.open()`,
   `await pool.check()`), initializes Redis (`await init_redis()`).
2. **Shutdown**: Closes all connection pools gracefully.

Note: Qdrant connectivity is verified via the `/health` endpoint,
not at startup.

## Gotchas

- All database pools must be opened before the first request.
  The lifespan manager ensures this.
- TMDB detail responses are cached in Redis for 1 day. The server
  acts as a caching proxy — clients never call TMDB directly.
- **Streaming-service filter key encoding**: `_to_metadata_filters`
  expands each `StreamingService` into encoded `(provider_id << 4) |
  method_id` values (not raw TMDB provider IDs). Raw provider IDs
  never match the encoded `watch_offer_keys` column; always use the
  fan-out over `StreamingAccessType` IDs, the same pattern as
  `_precompute_streaming_keys` in
  `search_v2/endpoint_fetching/metadata_query_execution.py`.
