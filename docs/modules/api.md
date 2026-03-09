# api/ — FastAPI Application

The HTTP interface for the search engine.

## What This Module Does

Provides the FastAPI application with lifecycle management
(opening/closing all database connection pools at startup/shutdown)
and API endpoints for search and movie detail retrieval.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app setup, lifespan context manager, health check endpoint. |

## Boundaries

- **In scope**: HTTP request handling, connection pool lifecycle,
  endpoint definitions, request/response serialization.
- **Out of scope**: Search logic (delegated to `db/search.py`),
  query understanding (delegated to `implementation/llms/`),
  data models (in `implementation/classes/`).

## Endpoints

### `POST /search` *(planned — not yet implemented)*
- **Request**: `query` (string), optional `filters` (release,
  runtime, maturity, genres, watch_providers, watch_methods),
  optional `shown_movie_counts` (client-provided session history)
- **Response**: Minimal card payload — `movie_id`, `title`, `year`,
  `poster_url`, `score`. Optional debug payload behind feature flag.
- Search response includes only what's needed to render title
  cards. Detail page data comes separately via `GET /movie/{id}`.

### `GET /movie/{movie_id}` *(planned — not yet implemented)*
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
