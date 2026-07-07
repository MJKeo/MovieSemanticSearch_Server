# api/ — FastAPI Application

The HTTP interface for the search engine.

## What This Module Does

Provides the FastAPI application with lifecycle management
(opening/closing all database connection pools at startup/shutdown)
and API endpoints for search and movie detail retrieval.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app setup, CORS middleware, lifespan context manager, health check endpoint, `/query_search` (with optional `clarification` field), `/rerun_query_search`, `/similarity_search`, `/attribute_search`, `/title_search`, `/movie_details/{tmdb_id}`, `/movie_credits/{tmdb_id}`, `MetadataFiltersInput` request model, the rerun request models + `_to_rerun_plan` translator, the `_build_movie_details` / `_build_movie_credits` translators, and the `_encode_and_cache_credits` shared cache writer. Calls `setup_tracing(app)` (from `observability/`) at import time, right after the app is constructed. |
| `outcome.py` | Per-request telemetry outcome mechanism: `FailureReason` enum, `EndpointFailure(HTTPException)`, and the `@record_outcome` decorator, which writes `outcome.success` / `outcome.failure_reason` exactly once on the request span. See the Observability section below. |
| `cli_search.py` | CLI tool to run the full search pipeline from the terminal. Supports genre, maturity, runtime, and release date filters. Run via `python -m api.cli_search "query"`. |

## Boundaries

- **In scope**: HTTP request handling, connection pool lifecycle,
  endpoint definitions, request/response serialization.
- **Out of scope**: Search logic (delegated to `db/search.py`),
  query understanding (delegated to `implementation/llms/`),
  data models (in `implementation/classes/`), OTel bootstrap and the
  telemetry name registry (delegated to `observability/` — see
  `docs/modules/observability.md`).

## Endpoints

### `POST /query_search`
- **Request**: `QuerySearchBody` — `query` (string), optional `clarification`
  (string — a follow-up correction, e.g. "less violent", "more 80s"),
  optional `filters` (`MetadataFiltersInput`), optional `shown_movie_counts`.
  When `clarification` is present, Step 0 and Step 1 see both the original
  query and the correction; Steps 2+ receive a single merged natural-language
  query per branch (slot 1 carries `main_rewrite`, spins operate on the
  corrected intent). Whitespace-only clarification is normalized to `None`.
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
- **Telemetry**: **Stage 0 (request boundary) instrumented; pipeline not yet.**
  Wrapped in `@record_outcome(success_on_return=False)` — failure-only mode,
  because the handler returns the `StreamingResponse` before the pipeline runs,
  so a clean return can't yet mean "succeeded" (the success verdict + stream-end
  rollups are deferred to a stream-aware mechanism). Request-span attributes,
  captured at handler entry from the raw wire body *before* validation
  (`_record_query_search_inputs`): `query_search.query` (truncated) +
  `query_search.query_chars`, `query_search.clarification` (+ `_chars`) when
  sent, one `filters.*` attr per sent field (`filters.genres`,
  `filters.min_runtime`, …; raw wire values, pre-translation), and
  `filters.active_count` (always) — the count of distinct filter *groups* the
  user engaged, where each min/max range (release date, runtime, maturity)
  counts once, not per bound. A `request rejected` span event (with `detail`)
  fires on the 400 (empty/over-length query), 422 unknown-filter-enum, and 422
  unknown-body-field paths; the verdicts are `invalid_parameters`,
  `invalid_filters`, and `invalid_parameters` respectively. Unknown body fields
  422 because `QuerySearchBody`/`MetadataFiltersInput` set `extra="forbid"`, and
  the framework 422 is recorded by the app-level `RequestValidationError`
  handler (`_on_request_validation_error`). No manual pipeline-stage spans yet —
  the rest of the pipeline (Steps 0–3, query generation, Stage 4, Qdrant,
  scoring) is Bites 1–9 in
  `observability_context/query_search_planning.md`.

### `POST /rerun_query_search`
- **Purpose**: Re-run a prior search with a new filter set while **bypassing
  Steps 0 (flow routing) and 1 (spin generation)** — filters never enter
  query understanding, so re-paying for those two Gemini calls is waste.
- **Request**: `RerunSearchBody` — `branches` (non-empty discriminated list,
  `type` tag) + optional `filters` (`MetadataFiltersInput`, same shape as
  `/query_search`). Each branch carries only what its flow strictly needs;
  this is the same branch data the client received in the original
  `/query_search` `fetches_ready` event:
  - `standard` — `query` (required); optional `ui_label` (display only).
    The internal `kind` is assigned positionally (not on the wire) — never
    affects scoring. Re-enters at **Step 2**.
  - `exact_title` — `title` (required), `release_year` (nullable).
  - `similarity` — `references[]` of `{ title, release_year? }`.
  - `non_character_franchise` / `character_franchise` — `canonical_name`.
  - `studio` / `person` — `canonical_names[]`.
  Entity branches re-enter at their **executor** (post-Step-0 point).
- **Translation**: `_to_rerun_plan` (boundary helper, mirrors
  `_to_metadata_filters`) demuxes the branch list into a `RerunPlan` — a
  `branch_plan` of `(kind, branch_query, ui_label)` tuples (kind assigned
  positionally `original` / `spin_1` / `spin_2`, so replaying branches in
  their original order reproduces the original fetch ids) plus the six entity
  `*FlowData` objects. `stream_rerun_pipeline` (in
  `search_v2/streaming_orchestrator.py`) accepts the `RerunPlan` and replays
  the shared downstream machine. Standard branch queries are capped at
  `MAX_QUERY_CHARS + MAX_CLARIFICATION_CHARS` (the failed-clarification merge
  ceiling) so any branch the pipeline can emit round-trips; entity names are
  capped at `MAX_QUERY_CHARS` for cost / injection parity with `/query_search`.
- **Response**: identical Server-Sent Event stream to `/query_search`
  (`fetches_ready` → per-branch `branch_traits` / `branch_categories`
  [standard only] / `branch_results` → `done`).
- **Errors**: 400 on a blank/over-length standard `query`; 422 on a duplicate
  entity flow, a blank/over-length entity name, more than 3 standard branches,
  empty `branches`, an unknown branch `type`, or an unknown filter enum value.
- **Telemetry**: auto-traced only today; not yet manually instrumented.

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
- **Telemetry**: auto-traced only today; introducing the manual Qdrant
  span (auto-instrumentation can't cover its gRPC client) is planned here
  — see Observability section below.

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
- **Telemetry**: auto-traced only today; not yet manually instrumented.

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
  exact-match priority tiers — Tier 1 (token-prefix: query is a prefix of
  any whitespace-delimited token in the title) and Tier 2 (substring at
  any position). Tier 1 always ranks above Tier 2. Within each tier,
  the shorter normalized title wins ("John Wick" outranks "John Wick:
  Chapter 2" for query "john wi"); length ties fall through to the
  same 80/20 popularity/reception blend used by `/attribute_search`,
  with `movie_id DESC` as a final tie-break.
- **Fuzzy fallback tier**: when the exact tiers return ≤ 3 results
  AND the normalized query is ≥ 3 chars AND remaining capacity > 0,
  a second `word_similarity` Postgres query runs (`pg_trgm` `<%` gate,
  threshold 0.45, ORDER BY `word_similarity DESC`). Results are appended
  below the exact tiers. The GUC `pg_trgm.word_similarity_threshold` is
  set to 0.45 inside an explicit transaction with `is_local=true` to
  prevent it from leaking onto the pooled connection. 0.45 captures
  one typo (~0.72 score) and dropped middle words (~0.87) while
  filtering most noise; threshold is the primary tuning knob.
- **Errors**:
  - 422 `{ "detail": "q must be non-empty" }` when `q` is missing or
    whitespace-only after trim.
  - 422 `{ "detail": "limit out of range" }` when `limit < 1` or
    `limit > 25`.
- **No NLP / no LLM / no vector search / no Redis** — single Postgres
  query against `movie_card.title_normalized` (trigram GIN +
  text_pattern_ops indexes). Backs the frontend's typeahead picker
  in "pick similar-to" mode, called on every debounced keystroke.
- **Telemetry**: wrapped in `@record_outcome` (`outcome.success` /
  `outcome.failure_reason=invalid_parameters` on the two 422s). No child
  span (single unit of work) — instead four attributes directly on the
  request span: `title_search.query` (raw text, high-cardinality —
  attribute only, never a metric label), `.limit`, `.result_count`
  (hydrated card count), `.fuzzy_result_count` (>0 = fuzzy fallback fired,
  a typo/catalog-gap signal). See Observability section below.

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
- **Telemetry**: wrapped in `@record_outcome`. Request-span attributes
  `movie.tmdb_id` (unconditional) and `movie.payload_source`
  (`cache`|`tmdb`, set only at a success point — absent on 404/502).
  Cold-path child spans: `movie_details.payload_creation` (card check +
  TMDB fetch; wraps the auto psycopg/httpx spans), `movie_details.cache_write`
  (the details cache SET, carries `cache.write_ok`), and
  `movie_credits.build_and_cache` (the cross-populate step, nests for free
  via the shared helper — see `/movie_credits` below). Warm path = request
  span + one auto redis GET, no child spans. Expected 404s leave spans
  UNSET; a 502 or unexpected error marks the relevant span ERROR +
  records the exception. See Observability section below.

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
- **Telemetry**: wrapped in `@record_outcome`. Same `movie.tmdb_id` /
  `movie.payload_source` request-span attributes as `/movie_details`. Cold-path
  child spans: `movie_credits.payload_creation` (index check + lean TMDB
  fetch) and `movie_credits.build_and_cache` (build + encode + cache SET,
  carries `cache.write_ok` plus `movie_credits.cast_count` / `.crew_count` —
  `crew_count` sums members across `CrewGroup` departments, not `len(crew)`).
  `build_and_cache` is the same span that runs as the `/movie_details`
  cross-populate step — one instrumented helper, two call sites. Warm path =
  request span + one auto redis GET. `source=tmdb` on this endpoint at any
  real rate is the signal that the details→credits cache warm-up is broken
  (see Observability section below).

### `GET /health`
- Validates connectivity to Postgres, Redis, and Qdrant.
- Logs the `CF-Connecting-IP` request header at INFO (the real client IP
  when the API is fronted by Cloudflare; `None` on direct/local calls).
  Visible on the api container's stderr via `docker compose logs -f api`.
  Note: `main.py` calls `logging.basicConfig(level=logging.INFO)` at import
  so app-module INFO logs emit at all — uvicorn alone only configures its
  own `uvicorn.*` loggers and leaves root at WARNING.
- **Telemetry**: no manual OTel spans/attributes and not wrapped in
  `@record_outcome` — deliberately excluded (three connectivity checks
  already fully covered by the auto-instrumented FastAPI/psycopg/redis/
  Qdrant-client spans, with no meaningful internal stage to name). The
  CF-Connecting-IP line above is plain logging, unrelated to tracing.

## Observability

OTel tracing bootstraps in `observability/tracing.py` (`setup_tracing(app)`,
called at import time right after `app = FastAPI(...)` is constructed) and
auto-instruments FastAPI, httpx, psycopg v3, and redis — every request on
every endpoint already gets a full network-hop trace for free. Manual spans
and attributes on top of that exist for `/title_search`, `/movie_details`,
`/movie_credits` (fully), and `/query_search` (Stage 0 request boundary only)
as noted inline above; see
`docs/modules/observability.md` for the bootstrap module itself and
`observability_context/observability_architecture.md` for the full as-built
catalog (source of truth — this section is a summary, not a replacement).

- **Per-request outcome verdict** — `outcome.success` (bool) /
  `outcome.failure_reason` (`FailureReason`: `invalid_parameters` |
  `invalid_filters` | `not_indexed` | `tmdb_removed` | `tmdb_fetch_failed` |
  `internal_error`) — is written on the request span by the `@record_outcome`
  decorator (`api/outcome.py`), applied to `/title_search`, `/movie_details`,
  `/movie_credits`, and `/query_search`. Endpoint code never sets this
  directly; it raises `EndpointFailure(failure_reason=...)` at each known
  failure site and the reason bubbles up to the one recording point.
  `/query_search` uses the decorator's **failure-only** form
  (`success_on_return=False`) — it records rejections/crashes but not
  `success=true`, because its handler returns a streaming response before the
  work runs (the success verdict awaits a stream-aware mechanism). The three
  remaining endpoints (`/rerun_query_search`, `/similarity_search`,
  `/attribute_search`) carry no `outcome.*` attribute yet.
- **`movie.tmdb_id` / `movie.payload_source`** — shared request-span
  attributes on `/movie_details` and `/movie_credits`. `payload_source`
  (`cache`|`tmdb`) is set only at a success point, never optimistically, so
  a 404/502 carries no source and cache-hit-rate stays honest.
- **Error contract on manual spans**: an expected 404 leaves the span
  UNSET (not an error); a 502 (TMDB fetch failure) or an unexpected
  exception marks the span ERROR + records the exception. Swallowed
  best-effort failures (cache read/write, credits cross-populate) surface
  as span *events*, never span errors, and don't affect `outcome.success`.
- **Partially / not yet instrumented**: `/query_search` has its Stage 0
  request boundary instrumented (outcome + input attributes above) but **no
  pipeline-stage spans** yet; `/rerun_query_search`, `/similarity_search`,
  `/attribute_search` get only the auto-traced network spans — no outcome
  attribute, no pipeline-stage spans, no `gen_ai.*` LLM attributes yet
  (planned: `observability_context/observability_todos.md` Phase 1c-1..1c-4;
  the `/query_search` pipeline is the highest-priority remaining target,
  carrying the LLM fan-out the latency goal cares most about — see the
  9-bite plan in `observability_context/query_search_planning.md`).
- **Naming**: every manual span name / attribute key referenced above is a
  `Name` constant from `observability/names.py` — never an inline string
  literal. See that module's docstring for the naming ruleset.

## Lifecycle

The lifespan context manager handles:
1. **Startup**: Opens Postgres pool (`await pool.open()`,
   `await pool.check()`), initializes Redis (`await init_redis()`),
   builds the shared TMDB httpx client + `AdaptiveRateLimiter` and
   stores them on `app.state.tmdb_client` / `app.state.tmdb_rate_limiter`
   for the `/movie_details` endpoint.
2. **Shutdown**: Closes the TMDB httpx client and all connection pools
   gracefully.

## CORS

`CORSMiddleware` is configured on the app with an explicit allow-list of
origins: `["https://www.cinemind.dev", "http://localhost:3000"]`,
`allow_credentials=True`, `methods=["GET", "POST"]`, and
`headers=["Authorization", "Content-Type"]`. Explicit enumeration (not
`"*"`) is required when `allow_credentials=True`. CORS is browser-enforced
only — not a defense against curl/scripted callers. Add the apex domain
explicitly if the non-www `cinemind.dev` is ever served.

Note: Qdrant connectivity is verified via the `/health` endpoint,
not at startup.

## Gotchas

- All database pools must be opened before the first request.
  The lifespan manager ensures this.
- TMDB detail responses are cached in Redis for 1 day under
  `tmdb:movie:v2:{id}` (namespace `v2` isolates the single-`crew`-list
  payload from older stale buckets-era bytes). The cached payload is the
  *curated* MovieDetails wire format (not raw TMDB JSON), so warm hits
  skip both the upstream call and the build/encode step. The server acts
  as a caching layer — clients never call TMDB directly.
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
- **The docker-compose `api` service is not tracing-ready yet**:
  `observability/` isn't volume-mounted and the OTel packages aren't in
  `api/requirements.txt`, so importing it in the container currently
  crashes (`ModuleNotFoundError: observability`). The instrumented path
  today is host-run (`uv run uvicorn api.main:app --reload`) only — see
  `docs/modules/observability.md`.
