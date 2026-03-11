# Conventions

Coding patterns and conventions for this codebase. CLAUDE.md
references this file — detailed rationale and examples live here,
while CLAUDE.md stays lean.

## Cross-Codebase Invariants

These rules apply everywhere and are not negotiable:

- `movie_id` is always `tmdb_id` (BIGINT/uint64). Primary key in
  Postgres, point ID in Qdrant, identifier in all Redis keys.
  Never introduce a secondary ID system.
- String normalization runs identically at ingest and query time.
  A mismatch is a silent retrieval bug. The normalization function
  (`implementation/misc/helpers.py:normalize_string`) applies:
  Unicode NFC normalization, lowercase, diacritic removal,
  hyphen preservation, apostrophe/period removal.
- Qdrant scores are final. Vector similarity is not recomputed at
  reranking. The reranker uses Qdrant scores directly, normalized
  via exponential decay within each vector space.
- Never query Postgres per-candidate. All metadata enrichment uses
  a single `WHERE movie_id = ANY($1)` bulk fetch after merge.
- Never cache partial DAG outputs. The entire
  `QueryUnderstandingResponse` is one atomic Redis key.
- Embedding cache does not lowercase. Embedding models are
  case-sensitive. The QU cache normalizer lowercases; the embedding
  cache normalizer does not.
- Qdrant payload is for hard filters only. Full metadata lives in
  Postgres.
- Fetch the trending set from Redis once per request and check
  membership in-memory — never query Redis per-candidate.
- Keep reranking server-side for consistency and to protect scoring
  logic from client manipulation.

## Python Conventions

- Python 3.13, type hints on all functions
- UV for package management (not pip)
- Pydantic for data models and validation
- pytest with asyncio_mode = "auto" (no decorators needed for async)
- base_movie_factory fixture in conftest.py for test data
- asyncio for all I/O-bound operations (database queries, API calls,
  LLM calls). Use `asyncio.gather` for concurrent fan-out.
- httpx for async HTTP clients (TMDB, IMDB scraping)
- Separate async I/O from DB writes in pipeline stages. Async
  tasks (HTTP fetches, API calls) must not receive a DB connection
  — they return result objects (e.g., NamedTuples). The calling
  orchestrator collects results via `gather()`, then does all DB
  writes in a single synchronous batch. This prevents SQLite
  thread-safety issues and keeps error handling in one place.
- Prefer batch DB operations (`executemany`, bulk inserts) over
  row-at-a-time loops. Provide batch variants of helper functions
  (e.g., `batch_log_filter`) for use in async orchestrators.
- Push filtering to the data layer. Use SQL JOINs/WHERE clauses
  to scope data before loading into Python, rather than loading
  everything and filtering in application code. If a second data
  source needs the same scope, derive its filter set from the
  first query's results.
- When computing multiple independent metrics over the same large
  dataset (e.g., field coverage stats across 140K movies), do a
  single pass that updates all accumulators at once — not one full
  iteration per metric. Define a dataclass to hold running totals,
  populate it in the loop, then read from it in reporting functions.
- For bulk JSON I/O (hundreds+ files), use `orjson` with binary
  file handles (`rb`/`wb`) — it's 5-10x faster than stdlib `json`
  and avoids the encode/decode overhead of text mode. For bulk
  file reads that are I/O-bound (not CPU-bound), wrap with
  `ThreadPoolExecutor` to parallelize disk reads. Reserve stdlib
  `json` for small one-off reads where adding the dependency
  isn't justified.

## Error Handling

- **Ingestion pipeline**: Every error state has a defined behavior.
  No errors are silently swallowed. All filtering goes through the
  `log_filter()` helper in `tracker.py`, which atomically updates
  both `filter_log` and `movie_progress` tables.
- **TMDB API**: Adaptive rate limiter with automatic backoff on 429s.
  HTTP 404 = movie doesn't exist (filter out). HTTP 5xx = retry up
  to 3 times with exponential backoff, then filter out.
- **IMDB scraping**: Single GraphQL query per movie with per-request
  retry and exponential backoff. On failure after retries, the movie
  remains in `tmdb_quality_passed` status and is retried on restart.
  All IMDB fields default to `None` for scalars and `[]` for lists,
  so downstream pipeline handles missing data gracefully.
- **LLM calls**: Timeout set, bounded retries, structured logging
  (model, tokens, latency, error). If any required query understanding
  node fails after retries, the entire search request fails.
- **Redis cache misses**: Graceful degradation. Missing trending data
  = log warning, treat as empty set, do not fail the request.
  Missing QU cache = run full DAG. Missing embedding cache = call
  OpenAI embeddings API.
- **Database connections**: psycopg3 async pool with min_size=2,
  max_size=10, max_lifetime=1800s, timeout=5s.

## Naming Conventions

- **Test files**: `test_<module_name>.py` in `unit_tests/`
- **Pydantic models**: PascalCase. Query understanding outputs
  suffixed with `Response` (e.g., `ExtractedEntitiesResponse`,
  `ChannelWeightsResponse`, `MetadataPreferencesResponse`).
  Metadata models suffixed with `Metadata` (e.g.,
  `PlotEventsMetadata`, `ViewerExperienceMetadata`).
- **Enums**: PascalCase class names, UPPER_SNAKE values
  (e.g., `RelevanceSize.NOT_RELEVANT`, `MovieStatus.TMDB_FETCHED`)
- **Database functions**: Prefixed by operation type —
  `search_*` for retrieval, `fetch_*` for bulk lookups,
  `ingest_*`/`upsert_*` for writes
- **Vector spaces**: snake_case names matching `VectorName` enum
  (e.g., `dense_anchor_vectors`, `plot_events_vectors`)
- **Redis keys**: Colon-delimited namespaces —
  `emb:{model}:{hash}`, `qu:v{N}:{hash}`,
  `trending:current`, `tmdb:movie:{id}`
- **Watch provider keys**: Encoded as `provider_id << 2 | method_id`
  (packed uint32). The `create_watch_provider_offering_key()`
  helper in `implementation/misc/helpers.py` handles encoding.
- **Ingestion statuses**: Progress through pipeline stages —
  `pending` → `tmdb_fetched` → `tmdb_quality_passed` →
  `imdb_scraped` → `essential_data_passed` → `phase1_complete` →
  `phase2_complete` → `embedded` → `ingested`. Terminal:
  `filtered_out`.

## Scoring Conventions

All scores flowing through the reranking pipeline are normalized
to [0, 1] unless explicitly noted otherwise:

- **Vector scores**: Exponential decay from best candidate per
  space, then weighted sum across spaces. Always [0, 1].
- **Lexical scores**: F-score based (beta=2.0), normalized [0, 1].
- **Metadata scores**: Weighted average of per-preference scores.
  Most preferences score [0, 1], but genre/language exclusion
  violations produce -2.0 (intentional hard penalty).
- **Final score**: `w_L * lexical + w_V * vector + w_M * metadata`.
  Channel weights (`w_L`, `w_V`, `w_M`) are derived from
  `RelevanceSize` enums assigned by the LLM.
- **Quality reranking**: Bucket by rounded final_score
  (BUCKET_PRECISION=2), sort within buckets by normalized
  reception score. Reception normalization: [30.0, 90.0] → [0, 1],
  None → 0.5 (neutral).

## Caching Conventions

- *(Planned)* **QU cache**: Key will include prompt version prefix
  (`v{N}`). Bump version when any system prompt changes. Old keys
  expire within TTL (1 day) — no explicit invalidation needed.
- **Embedding cache**: Key includes model name. Binary serialization
  (not JSON) for float arrays.
- **TMDB detail cache**: TTL 1 day. Proxy TMDB through the server
  to keep API secrets off the client.
- **Trending set**: No TTL — key replaced atomically via RENAME.
  Stale trending data is better than missing trending data.

## Ingestion Conventions

- All pipeline stages are crash-safe and idempotent. Restarting
  picks up where it left off via the SQLite checkpoint tracker.
- Use `log_filter()` for all filtering — never write directly to
  `filter_log` or update `movie_progress.status` to `filtered_out`
  from stage modules.
- Commit every 500 movies for bounded data loss on crash.
- TMDB fetching is free (API key only, no proxy needed). IMDB
  scraping requires residential proxies. This cost asymmetry
  drives the TMDB-first funnel design.
- SQLite tracker must always set `PRAGMA journal_mode=WAL` and
  `PRAGMA synchronous=FULL` in `init_db()`. Never weaken the
  synchronous pragma — WAL mode without it risks corruption on
  crash.
- When multiple pipeline stages share scoring or analysis logic,
  extract to a shared module in `movie_ingestion/` with parametric
  inputs (config dataclasses, enums) for stage-specific
  differences. Each stage imports from the shared module but keeps
  its own configuration.
- Tracker DB identifiers (column names, status values, stage names)
  that could be ambiguous across pipeline stages must be prefixed
  with their data source scope (e.g., `tmdb_quality_passed`,
  `imdb_scraped`, `stage_3_quality_score`).
- Quality gates must use two distinct statuses: `*_calculated`
  (score written, no filtering) and `*_passed` (survived threshold).
  This keeps scoring and filtering independently re-runnable.

## Network & Retry Conventions

- Tune timeout and retry strategy to match the failure mode of
  the transport. With rotating proxies (fresh IP per request),
  use aggressive timeouts (2-5s) paired with higher retry counts
  (5+) and short backoffs (0.3-0.8s) — a slow response means a
  flagged IP, and waiting longer cannot help.
- For fixed-endpoint APIs (TMDB, OpenAI), use longer timeouts
  with exponential backoff, since the same server will eventually
  recover.
