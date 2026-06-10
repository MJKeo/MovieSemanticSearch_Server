# api/ — FastAPI Application

The HTTP interface for the search engine.

## What This Module Does

Provides the FastAPI application with lifecycle management
(opening/closing all database connection pools at startup/shutdown)
and API endpoints for search and movie detail retrieval.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app setup, lifespan context manager, health check endpoint, `/query_search`, `/similarity_search`, `/attribute_search`, `/title_search`, `/movie_details/{tmdb_id}`, `/movie_credits/{tmdb_id}`, `MetadataFiltersInput` request model, the `_build_movie_details` / `_build_movie_credits` translators, and the `_encode_and_cache_credits` shared cache writer. |
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
- **Request**: `SimilaritySearchBody` — `tmdb_ids` (non-empty list of
  int), optional `filters` (`MetadataFiltersInput`, same shape as
  `/query_search`).
- **Response**: Array of `MovieCard` objects ranked by descending
  similarity score. Bypasses the natural-language pipeline.
- **Filters**: when supplied and at least one field is set, every
  candidate-generation lane (Postgres director / franchise / studio /
  source / quality / themes / rare-medium and the Qdrant shape search)
  is constrained to filter-eligible movies — same pattern as the other
  Step-0 entity flows (`exact_title`, `franchise`, `studio`, `person`).
  An all-None filters payload (or omitting the field) is the explicit
  "no filter" signal and takes the same fast path as before.
- **Cache**: unfiltered requests use the legacy anchor-id-only Redis
  key. Filtered requests append a short stable BLAKE2b fingerprint of
  the active filter config (`...:f:<hex>`), so filtered and unfiltered
  warm hits live in disjoint key slots and pre-existing unfiltered
  entries stay valid. 24h TTL on every variant.
- Errors: empty `tmdb_ids` → 422 (pydantic), unknown IDs → 422,
  invalid anchor → 400, unknown filter enum value → 422.

### `POST /attribute_search`
- **Request**: `AttributeSearchBody` — both fields optional:
  - `filters`: same `MetadataFiltersInput` shape used by `/query_search`
    (genres, audio_languages, streaming_services, release_date /
    runtime / maturity ranges). Same 422 path on unknown enum values.
  - `people`: list of `PersonInput` entries, each `{name: str}` (no
    role — the flow is role-agnostic). Name is normalized server-side.
    Multiple `people` entries are **unioned and summed** (not
    intersected): a movie crediting any supplied person qualifies, and
    its score is the sum of each person's prominence weight on it.
    Empty list and `null` are treated identically.
- **Ranking** (people supplied): reuses the **Step 0 person-search
  model** (`search_v2/person_search.py`). Each person resolves to a
  per-movie prominence bucket via the shared `fetch_person_buckets`
  (role-agnostic, MIN bucket across all five credit tables): LEAD /
  MAJOR / RELEVANT / MINOR — the minor actor zone split at `zp=0.5`
  into "still relevant" vs "cameo", using the shared
  `search_v2/actor_zones.py` cutoffs; non-actor credits land in LEAD.
  Each bucket maps to an additive weight `(BUCKET_MINOR+1) - bucket`
  (LEAD=4 … MINOR=1), weights are **summed across people**, and the
  sum is the primary sort key (DESC). Within a tier, movies sort by
  `popularity_score` (NULLS-last) then `movie_id` DESC — identical to
  Step 0's `_sort_bucket`. Ranking happens in Python over the scored
  pool (one bulk popularity fetch), not in SQL.
  - **Single person, no metadata filters → identical ordering to the
    Step 0 person flow**, by construction (one summand → weight is
    constant within a bucket and strictly orders buckets; same
    within-bucket tie-break). The only divergence from Step 0 is the
    multi-person rule: Step 0 takes MIN bucket + `overlap_count`,
    whereas this endpoint sums weights so more/more-prominent credits
    rank higher.
- **Response**: Array of `MovieCard` objects, capped at 250 (Step 0's
  person flow caps at 100; the overlapping prefix is identically
  ordered). Unresolvable person names contribute no credits silently —
  they neither error nor empty the result.
- **No people supplied**: returns the catalog ranked by the same 80/20
  neutral prior used by the V2 reranker-only fallback path
  (`fetch_neutral_reranker_seed_ids`).
- **No NLP / no LLM / no vector search** — pure deterministic browse
  path. Hard `MetadataFilters` push down to the SQL layer inside every
  posting-table lookup, so the scored pool is already filter-respecting.

### `GET /title_search`
- **Request**: query params
  - `q` (string, required): title query. Trimmed server-side; values
    over ~100 chars are truncated before normalization.
  - `limit` (int, optional): max results. Default 10, hard cap 25.
- **Response**: Array of up to `limit` `MovieCard` summaries (same
  shape as `/similarity_search` and `/attribute_search`). Empty list
  on no matches (never 404). `Cache-Control: public, max-age=300`.
- **Matching**: NFC + diacritic-folded + lowercased (via the shared
  `normalize_string` used for ingest-time `title_normalized`). Two
  priority tiers — Tier 1 (token-prefix: query is a prefix of any
  whitespace-delimited token in the title) and Tier 2 (substring at
  any position). Tier 1 always ranks above Tier 2. Within each tier,
  the shorter normalized title wins ("John Wick" outranks "John Wick:
  Chapter 2" for query "john wi"); length ties fall through to the
  same 80/20 popularity/reception blend used by `/attribute_search`,
  with `movie_id DESC` as a final tie-break. Tier 3 fuzzy
  (edit-distance ≤ 2) is deliberately omitted for v1 — the spec marked
  it optional and it doesn't fit the p50<20ms target.
- **Errors**:
  - 422 `{ "detail": "q must be non-empty" }` when `q` is missing or
    whitespace-only after trim.
  - 422 `{ "detail": "limit out of range" }` when `limit < 1` or
    `limit > 25`.
- **No NLP / no LLM / no vector search / no Redis** — single Postgres
  query against `movie_card.title_normalized` (trigram GIN +
  text_pattern_ops indexes). Backs the frontend's typeahead picker
  in "pick similar-to" mode, called on every debounced keystroke.

### `GET /movie_details/{tmdb_id}`
- **Response**: Curated `MovieDetails` msgspec struct (see
  `schemas/api_responses.py`) combining TMDB live data (overview,
  cast/crew, watch providers, trailer, images) with our locally-
  computed `reception_score`. Crew is a single ranked `crew` list,
  not separate buckets: up to 12 distinct people (all directors, top 3
  writers, top 3 producers, then the most important remaining crew via
  `_select_crew_people`), each contributing all their credit rows
  (no backend dedupe). `crew_truncated` is True when >12 distinct crew
  people exist; the "See all" link points at `/movie_credits`.
- **Flow**:
  1. Check Redis (`tmdb:movie:v2:{id}`, 24h TTL) — return cached bytes
     verbatim on hit. (`v2` namespaces the single-`crew`-list payload
     shape; a bump avoids serving stale buckets-era bytes.)
  2. On miss, confirm the movie is in `public.movie_card` (404 if not).
  3. Fetch `https://api.themoviedb.org/3/movie/{id}` with
     `append_to_response=credits,videos,images,external_ids,watch/providers,release_dates`.
  4. Translate to `MovieDetails` via `_build_movie_details`, cache, return.
  5. Cross-populate the `/movie_credits` cache from the same payload via
     `_encode_and_cache_credits` — warms `tmdb:credits:{id}` for free so a
     follow-up "See all" is a cache hit. Best-effort; never blocks the
     response.
- **Errors**: 404 if movie not in `movie_card` or TMDB returns 404;
  502 if the TMDB fetch fails after retries.
- **Graceful degradation**: Redis read/write failures are logged but
  do not fail the request — the cold path still serves the response.

### `GET /movie_credits/{tmdb_id}`
- **Response**: `MovieCredits` msgspec struct (`schemas/api_responses.py`)
  — the complete, uncapped cast & crew, credits-only (no movie metadata).
  Backs the details page's "See all" affordance. Reuses the `CastMember` /
  `CrewMember` shapes; crew is grouped by TMDB `department` into `CrewGroup`s.
- **Flow** (cache-first): Redis (`tmdb:credits:{id}`, 24h TTL) hit returns
  verbatim → on miss, `movie_card` 404 check → **lean credits-only** TMDB
  fetch (`fetch_movie_credits_for_endpoint`, `append_to_response=credits`) →
  build + cache + return via `_encode_and_cache_credits`.
- **Ordering** (server-authoritative): cast in TMDB billing order; crew groups
  lead with Directing/Writing/Production then remaining departments in TMDB
  first-seen order; one entry per (person, job), no merging.
- **Errors / graceful degradation**: identical to `/movie_details` (404
  not-in-index / not-on-TMDB, 502 on TMDB failure; Redis failures non-fatal).
- **Caching**: the `tmdb:credits:{id}` cache is normally warmed by the
  `/movie_details` view the user came from (cross-population, see above), so
  the common path is a hit with no TMDB call. The cold-path fetch is the rare
  fallback and is credits-only to avoid pulling sub-resources it discards.
  Both writers funnel through `_encode_and_cache_credits` — one write codepath
  for the key.

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
