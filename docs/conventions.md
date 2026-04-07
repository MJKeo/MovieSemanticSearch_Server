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
- `__str__()` methods on Pydantic schema classes that feed the embedding
  pipeline must normalize all concatenated text using `normalize_string()`
  (`implementation/misc/helpers.py`). This applies the same NFC normalization,
  lowercasing, and diacritic removal used at ingest time. Inconsistent
  normalization silently degrades embedding quality.
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
- When a function evaluates N independent cases with distinct logic
  (eligibility checks, validation rules, skip conditions), extract each
  case into its own function. The caller becomes a thin orchestrator —
  a loop or sequential calls — with no inlined case logic.
- All LLM generation functions — sync and async, across all providers —
  must return `Tuple[BaseModel, int, int]` (parsed_response, input_tokens,
  output_tokens). The unified router dispatches to provider functions by
  reference; any deviation in return shape is a silent runtime failure
  that only surfaces when a specific provider is exercised.
- Verify provider-specific API parameter support before setting kwargs.
  Cross-reference the latest online documentation for the exact model
  being configured — parameters that work for one provider or model
  family often don't exist or behave differently for another (e.g.,
  `reasoning_format` vs `include_reasoning`, temperature unsupported
  on reasoning models). Include a brief comment on non-obvious
  constraints.
- Jupyter notebooks must resolve the project root dynamically using a
  `find_project_root()` helper that walks up from `Path.cwd()` until
  `pyproject.toml` is found, then inserts that path into `sys.path`.
  Never use hardcoded `parent.parent` chains — they break silently when
  a notebook is opened from a different working directory.
- Default values in signatures, not body fallbacks. When a function
  parameter has a known default available at module scope, declare it
  in the signature (`param: str = DEFAULT`) rather than using `None`
  with a body-level fallback (`param = param or DEFAULT`). Reserve
  `None` defaults for cases where the default must be computed at call
  time or depends on other arguments.
- No backward-compatibility aliases. When renaming a variable,
  function, or parameter, update all references to use the new name
  directly. Don't create aliases (`new_name = old_name`) or
  re-exports to preserve the old name. Aliases add indirection
  without value and let stale references persist silently.
- Use current, non-deprecated API patterns. Always code against the
  current API of installed library versions. Do not rely on deprecated
  access patterns even if they still function. When unsure, verify
  the recommended usage for the installed version.
- When a set of related string constants appears in more than one
  module, define them as a `StrEnum` in a shared location. All callers
  reference the enum member — never hardcode the string value. `StrEnum`
  preserves string compatibility (SQLite, JSON, logs) while preventing
  typos and enabling IDE navigation.
- Function parameters exposed to callers should use the caller's
  domain vocabulary, not the implementation's internal units. Internal
  conversions (e.g., `max_movies = max_batches * batch_size`) belong
  inside the function body. When threading a value through multiple
  function layers, keep the parameter name consistent with its
  source — renaming mid-chain obscures what the value actually
  represents.
- When a function returns more than two related values, return a
  dataclass or Pydantic model — not a positional tuple. Named fields
  eliminate ordering bugs, make call sites self-documenting, and
  allow adding fields without breaking callers. If an existing
  Pydantic model already represents the data, return it directly
  rather than creating a parallel dataclass.

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
- **Exception class design**: Prefer shared exception classes parameterized
  with context fields over per-module subclasses. When N modules share the
  same failure mode (e.g., "generation failed for type X"), one class with
  a `generation_type` field scales to N modules; N subclasses proliferate
  with no added expressiveness.
- **Preserve retryable exception types**: When a function wraps
  exceptions into a generic type (e.g., `except Exception as e:
  raise ValueError(...)`), re-raise retryable exceptions (rate limits,
  transient network errors) before the catch-all. Callers need the
  original exception type to decide whether to retry or fail.
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
  `pending` → `tmdb_fetched` → `tmdb_quality_calculated` →
  `tmdb_quality_passed` → `imdb_scraped` → `imdb_quality_calculated` →
  `imdb_quality_passed` → `metadata_generated` → `ingested`.
  Terminal: `filtered_out`. Retryable: `ingestion_failed`
  (failure details in `ingestion_failures` table).

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
- Tracker DB is the authoritative data source for ingested movie
  data. When querying IMDB or TMDB fields for analysis, scoring,
  or pipeline logic, always read from the `imdb_data` / `tmdb_data`
  tables in `tracker.db`. Per-movie JSON files in
  `ingestion_data/imdb/` are raw scrape backups — they may be stale
  relative to the DB and must not be used as a data source.
- Tracker DB identifiers (column names, status values, stage names)
  that could be ambiguous across pipeline stages must be prefixed
  with their data source scope (e.g., `tmdb_quality_passed`,
  `imdb_scraped`, `stage_3_quality_score`).
- Quality gates must use two distinct statuses: `*_calculated`
  (score written, no filtering) and `*_passed` (survived threshold).
  This keeps scoring and filtering independently re-runnable.
- Deterministic derivation over LLM classification. Before assigning
  a classification task to an LLM generator, verify whether existing
  structured data (genres, keywords, title patterns) provides
  sufficient coverage to derive the answer deterministically. If
  coverage is near-100%, implement a rule-based function rather than
  paying for an LLM call that may introduce abstention bugs or
  hallucinations. LLM generation should be reserved for tasks where
  structured data is insufficient or ambiguous.

## Network & Retry Conventions

- Tune timeout and retry strategy to match the failure mode of
  the transport. With rotating proxies (fresh IP per request),
  use aggressive timeouts (2-5s) paired with higher retry counts
  (5+) and short backoffs (0.3-0.8s) — a slow response means a
  flagged IP, and waiting longer cannot help.
- For fixed-endpoint APIs (TMDB, OpenAI), use longer timeouts
  with exponential backoff, since the same server will eventually
  recover.

## Evaluation Conventions

- **Storage structure**: Evaluation results are stored as per-movie
  JSON files in `movie_ingestion/metadata_generation/evaluation_data/`
  (e.g., `reception_{tmdb_id}.json`). Analysis is done via
  `analyze_evaluations.py` in the same directory when present
  (removed after evaluation completes for a given type).
- **Candidate config**: LLM configurations under test are defined
  per metadata type in the evaluation notebook
  (`metadata_generation_playground.ipynb`). Each candidate specifies
  provider, model, and kwargs. Evaluation outputs are saved as
  per-movie JSON files in `evaluation_data/` (e.g.,
  `reception_{tmdb_id}.json`).
- **Reference-free pointwise evaluation**: For each (candidate, movie)
  pair, generate output, then score with a rubric-based LLM judge
  (Claude Opus 4.6, thinking disabled). The judge sees raw source data
  (the same movie fields the candidate saw), not generation instructions
  or reference outputs. Multi-run averaging (2 sequential runs for
  prompt caching). Idempotent — checks for existing rows before
  inserting. 429 rate limits trigger 30s sleep + retry.
- **Prompt reuse**: Evaluation pipelines must import prompt construction
  from the production generator, not duplicate it. Extract prompt builders
  as named public functions in the generator module (e.g.,
  `build_plot_events_prompts()`). Duplicating the prompt in the eval
  file creates silent drift — if the generator's prompt changes, the eval
  copy diverges without any error.
- **Judge prompt alignment**: Every judge prompt must include the raw
  source data that was available to the candidate, plus the candidate
  output. Rubric score anchors define quality criteria independently
  of the generation prompt. The judge evaluates against source data
  for factual verification and against the rubric's quality standards
  for style and completeness.

## Prompt Authoring Conventions

- **Cognitive-scaffolding field ordering.** Order structured output
  schema fields in cognitive progression: concrete/extractive fields
  before abstract/synthetic ones. Each field should scaffold the
  next — the model builds context as it generates. Never lead with
  synthesis (overview, core concept) that requires cold-start
  distillation. In schemas with reasoning/justification fields,
  place them immediately before the label or value they scaffold.
- **Within-section uniqueness only.** In generation prompts and
  evaluation rubrics, only enforce uniqueness within each section
  (no near-synonyms as section neighbors). Cross-section overlap is
  acceptable and beneficial for embedding quality — the same concept
  appearing in multiple sections reinforces the signal when it serves
  different analytical purposes.
- **Evidence inventory, not rationalization, for reasoning fields.**
  When adding reasoning/justification fields to structured output
  schemas, frame them as upstream evidence inventories ("cite specific
  input phrases") rather than downstream explanations ("why did you
  generate this"). Include an explicit empty-evidence path. Even
  though both are generated before the output in token order, only
  the inventory framing constrains over-inference. If the model may
  use parametric knowledge, explicitly state that an empty evidence
  inventory does not mandate empty output.
- **Explicit absence for primary inputs.** When building LLM prompts,
  absent primary inputs (those the system prompt teaches the model to
  rely on) should be included with the value `"not available"` rather
  than silently omitted. A model with minimal reasoning may not notice
  a missing field, but seeing an explicit absence signal calibrates
  confidence downward. Secondary inputs (keywords, genres) can still
  be omitted when empty.
- **Empty collections over skip-wrapper schemas.** When a structured
  output section is optional for some inputs, use the base type with
  `min_length=0` — not a wrapper with a `should_skip` boolean.
  Skip-flag wrappers force the model to manage a boolean alongside
  content fields and still populate dummy data when skipping. Empty
  lists achieve the same result (no contribution to `__str__()`)
  with less schema complexity.
- **Abstention-first for optional fields.** When a structured output
  field allows empty output and the model rarely produces it, lead
  the field's instructions with the abstention decision ("FIRST:
  determine whether...") before any extraction rules. List concrete
  content categories where emptiness is expected. Models default to
  "always produce something" — burying the empty-is-valid instruction
  at the end makes it an afterthought.
- **Merge ambiguous field boundaries.** When two structured output
  fields have ambiguous boundaries (the model frequently produces
  overlapping content) and the downstream consumer treats them
  identically (e.g., both feed the same embedding), merge into a
  single field. This eliminates a common failure mode (near-duplicate
  content across fields) with no downstream impact. Especially
  important for small models where cognitive budget spent
  distinguishing similar fields is wasted.
- **Principle-based constraints, not failure catalogs.** When updating
  LLM prompts based on evaluation findings or observed failures,
  express constraints as general principles the model can apply to
  novel cases — not enumerated lists of observed bad behaviors. A
  principle ("Only characters who actively drive plot decisions")
  scales better than a catalog of exclusions ("not plot devices, not
  kidnap targets, not one-mention characters..."). Reactive lists
  grow long, confuse the model, and only cover previously seen
  failures.
- **Brief pre-generation fields, no consistency coupling.** When
  adding a pre-generation reasoning field to a structured output
  schema for chain-of-thought quality improvement, keep it brief —
  a classification or label, not a summary or explanation. Never
  instruct the model to make downstream output "consistent with"
  the pre-generation field. Full sentences dominate autoregressive
  attention and act as templates; brief labels prime without
  constraining.
- **Programmatic prompt adaptation over self-classification.** When
  LLM behavior should differ based on input conditions (data quality,
  sparsity, field presence, content type), swap prompt sections or
  entire prompts programmatically rather than including all variants
  and asking the model to self-classify. Classify the condition in
  deterministic Python code; the model receives only the guidance
  relevant to its actual inputs. This saves reasoning tokens and
  eliminates misclassification risk.
- **Example-eval separation.** When a generation prompt uses concrete
  domain examples to disambiguate classification boundaries, draw
  examples from a different pool than the evaluation test data.
  Overlapping examples make it impossible to distinguish "learned
  the principle" from "memorized the example." Keep examples to 2-4
  and ensure the abstract rules stand alone without them.
