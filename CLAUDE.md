# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Documentation

- **Product context and priorities:** docs/PROJECT.md — read at session start
- **Decision records:** docs/decisions/ — check for precedent before tradeoff decisions
- **Module summaries:** docs/modules/ — read when entering a module for the first time in a session
- **Conventions:** docs/conventions.md — cross-codebase invariants and patterns
- **Transient context:** DIFF_CONTEXT.md — what changed recently and why
- **Action items:** docs/TODO.md — deferred TODOs discovered during sessions
- **Personal preferences:** docs/personal_preferences.md — communication and workflow preferences, read at session start

## Autonomous Documentation

After completing each implementation task, update DIFF_CONTEXT.md
per the context-tracking rule in .claude/rules/.

You MAY autonomously update docs/modules/ when you notice a doc
is stale while working in that module (include in your changeset).

You MAY autonomously add entries to docs/TODO.md when you discover
actionable items during implementation work. Use /save-todo or write
entries directly following the existing format.

You must NEVER autonomously modify docs/PROJECT.md, docs/conventions.md,
or docs/decisions/. See the docs-awareness rule for details.

## Session Learnings

When I run /safe-clear, Claude extracts learnings from the session:
- **Personal preferences** → docs/personal_preferences.md (auto-maintained)
- **Convention candidates** → docs/conventions_draft.md (staged for review)
- **Workflow suggestions** → docs/workflow_suggestions.md (staged for review)

Read docs/personal_preferences.md at session start to apply my
communication and workflow preferences.

## Commands

```bash
# Install dependencies (uses UV package manager, not pip)
uv sync

# Run all tests
pytest unit_tests/

# Run a single test file
pytest unit_tests/test_search.py -v

# Run a single test
pytest unit_tests/test_metadata_scoring.py::TestClassName::test_method_name -v

# Start all services (PostgreSQL, Redis, Qdrant, API)
docker-compose up

# Run the API locally (requires services running)
uvicorn api.main:app --reload
```

Python version is 3.13. Test runner uses `asyncio_mode = "auto"` (all async tests work without decorators).

**Environment:** A `.env` file is required with keys for `TMDB_API_KEY`, `OPENAI_API_KEY`, `MOONSHOT_API_KEY`, and Postgres/Redis/Qdrant connection strings. See `.env` for the full list.

## Architecture Overview

This is a **multi-channel movie search engine** that decomposes natural language queries via LLM and executes parallel retrieval across three channels:

### Search Pipeline (end-to-end flow)

```
User Query
    ↓
Query Understanding (parallel LLM calls)
    ├── Extract lexical entities (actors, directors, franchises, characters)
    ├── Assign channel weights (lexical vs. vector vs. metadata importance)
    ├── Extract metadata preferences (date, genre, runtime, streaming, ratings)
    ├── Generate vector subqueries (one per vector space)
    └── Assign vector space weights
    ↓
Parallel Retrieval
    ├── Lexical Search → PostgreSQL entity matching
    ├── Vector Search → Qdrant similarity across 8 embedding spaces
    └── Metadata Scoring → in-memory structured attribute scoring
    ↓
Score Merging: final = w_L*lex + w_V*vec + w_M*meta (all normalized [0,1])
    ↓
Quality Reranking: bucket by relevance, sort by reception score within buckets
    ↓
Fetch display metadata → return JSON
```

### Key Directories

| Path | Purpose |
|------|---------|
| `schemas/` | Shared Pydantic models and data classes used across multiple top-level modules (db, api, movie_ingestion) |
| `db/` | All database access: search orchestration, vector/lexical/metadata scoring, Postgres/Qdrant/Redis clients |
| `implementation/classes/` | Pydantic data models (`schemas.py`), `BaseMovie`, enums, watch providers |
| `implementation/llms/` | Shared LLM clients and routing for query understanding and metadata generation |
| `implementation/prompts/` | System prompts for each LLM task |
| `implementation/misc/` | Utilities: string normalization, SQL LIKE escaping |
| `implementation/notebooks/` | Jupyter notebooks for exploration, DB rebuilding, and evaluation |
| `movie_ingestion/` | Ingestion pipeline: `tracker.py` (shared state), `tmdb_fetching/` (TMDB export & detail fetch), `tmdb_quality_scoring/` (quality filtering), `imdb_scraping/` (IMDB data), `final_ingestion/` (Postgres + Qdrant upserts) |
| `unit_tests/` | pytest test suite (59 files); `conftest.py` provides `base_movie_factory` fixture |
| `docs/` | Project context, decision records, module summaries, conventions |

### Vector Search Design

8 named vectors per movie (OpenAI `text-embedding-3-small`, 1536 dims), stored in Qdrant with scalar quantization + memmap:
- `dense_anchor_vectors` — core thematic summary
- `plot_events_vectors` — plot + characters
- `plot_analysis_vectors` — themes, arcs, concepts
- `narrative_techniques_vectors` — storytelling style/tone
- `viewer_experience_vectors` — emotional tone, pacing
- `watch_context_vectors` — when/how to watch (date night, background, etc.)
- `production_vectors` — budget, locations, technical achievements
- `reception_vectors` — critical reception, awards, audience reaction

5-stage vector scoring pipeline (`db/vector_scoring.py`): execute → blend (80/20 subquery/original) → normalize (exponential decay) → weight → sum.

Each vector space has LLM-generated metadata covering 7 types: plot events, plot analysis, viewer experience, watch context, narrative techniques, reception, and production/source-of-inspiration style metadata. Generation now lives under `movie_ingestion/metadata_generation/` (generators, prompts, batch orchestration, and evaluations), and the resulting metadata is stored for embedding and ingestion.

### Data Stores

| Service | Technology | Purpose |
|---------|-----------|---------|
| PostgreSQL 15 | `db/postgres.py` | Movie metadata, lexical entities, posting lists |
| Qdrant | `db/qdrant.py`, `db/vector_search.py` | 8 vector spaces × 150K movies |
| Redis 7 | `db/redis.py` | Embeddings cache, query understanding cache, trending set, TMDB detail cache |

### Movie Ingestion Pipeline

The ingestion pipeline processes ~1M TMDB movies down to ~100K high-quality movies through a multi-stage funnel. All stages are crash-safe and idempotent — restarting picks up where it left off.

**Tracker system:** `movie_ingestion/tracker.py` is the shared backbone. It manages a SQLite database at `./ingestion_data/tracker.db` with two core tables:
- `movie_progress` — one row per movie, tracks status through the pipeline (status column progresses: `pending` → `tmdb_fetched` → `tmdb_quality_calculated` → `tmdb_quality_passed` → `imdb_scraped` → `imdb_quality_calculated` → `imdb_quality_passed` → `metadata_generated` → `ingested`; terminal: `filtered_out`; retryable: `ingestion_failed`)
- `filter_log` — append-only audit trail of every filtered movie with stage, reason, and optional details JSON
- `tmdb_data` — stores extracted TMDB fields needed by the quality scorer (vote counts, popularity, provider keys, boolean completeness flags)

The `log_filter()` helper handles both the filter_log INSERT and the movie_progress status update — never write to those tables directly from stage modules.

**Pipeline stages:**

```
Stage 1: TMDB Daily Export (movie_ingestion/tmdb_fetching/daily_export.py)
  └─ Downloads gzipped JSONL (~1M entries), stream-decompresses line by line
  └─ Filters: adult=False, video=False, popularity > 0
  └─ Result: ~800K movies inserted as 'pending' in movie_progress
  └─ Run: python -m movie_ingestion.tmdb_fetching.daily_export

Stage 2: TMDB Detail Fetching (movie_ingestion/tmdb_fetching/tmdb_fetcher.py)
  └─ Async HTTP fetches via httpx with adaptive rate limiting (db/tmdb.py)
  └─ Extracts fields into tmdb_data table, encodes watch provider keys as packed uint32 BLOBs
  └─ Filters out movies missing an IMDB ID (can't proceed to Stage 4)
  └─ Status: pending → tmdb_fetched
  └─ Run: python -m movie_ingestion.tmdb_fetching.tmdb_fetcher

Stage 3: TMDB Quality Funnel (two scripts, run in order)
  └─ Scorer: python -m movie_ingestion.tmdb_quality_scoring.tmdb_quality_scorer
     └─ Edge cases: unreleased → 0.0, has US watch providers → 1.0
     └─ No-provider formula (4 signals, weights sum to 1.0):
        vote_count (0.50), popularity (0.20), overview_length (0.15), data_completeness (0.15)
     └─ No hard filters — deliberately lenient, real quality gate is Stage 5
     └─ Status: tmdb_fetched → tmdb_quality_calculated
  └─ Filter: python -m movie_ingestion.tmdb_quality_scoring.tmdb_filter
     └─ Soft threshold: stage_3_quality_score < 0.2344 (inflection point from derivative analysis)
     └─ Status: tmdb_quality_calculated → tmdb_quality_passed (or filtered_out)

Stage 4: IMDB Scraping (movie_ingestion/imdb_scraping/)
  └─ Single GraphQL query per movie to api.graphql.imdb.com (replaces 6 HTML page fetches)
  └─ Routed through DataImpulse residential proxies with US geo-targeting
  └─ Async with semaphore-controlled concurrency (default 60), random UA rotation, exponential backoff
  └─ Extracts: credits, keywords (with community vote scoring), synopses, parental guide, reviews
  └─ Output: stored in `imdb_data` table in tracker.db (IMDBScrapedMovie Pydantic model)
  └─ Status: tmdb_quality_passed → imdb_scraped (or filtered_out)
  └─ Run: python -m movie_ingestion.imdb_scraping.run

Stage 5: IMDB Quality Filtering (movie_ingestion/imdb_quality_scoring/)
  └─ Two hard gates (title-type, missing-text) then combined TMDB+IMDB quality scorer (8 signals, weights sum to 1.0):
     imdb_notability (0.31), featured_reviews_chars (0.15),
     plot_text_depth (0.12), lexical_completeness (0.10),
     critical_attention (0.08), community_engagement (0.08),
     tmdb_popularity (0.08), data_completeness (0.08)
  └─ IMDB data primary, TMDB as fallback for overlapping fields
  └─ Soft threshold via derivative analysis of quality score distribution
  └─ Status: imdb_scraped → imdb_quality_passed (or filtered_out)
  └─ Run: python -m movie_ingestion.imdb_quality_scoring.imdb_quality_scorer

Stage 6+: LLM Generation → Embedding → Ingestion
  └─ movie_ingestion/metadata_generation/ — Stage 6 batch metadata generation pipeline
     └─ generators/ — per-type generation functions and prompt builders
     └─ batch_generation/ — run.py (CLI entry point), request_builder.py, openai_batch_manager.py, result_processor.py, generator_registry.py, pre_consolidation.py
     └─ helper_scripts/ — estimate_generation_cost.py, report_bucket_axis_performance.py
  └─ movie_ingestion/final_ingestion/vector_text.py — generates text for each of 8 vector spaces
  └─ movie_ingestion/final_ingestion/ingest_movie.py — embeds vector text via OpenAI and upserts final data into Postgres and Qdrant (embedding is integrated into Stage 8, not a separate step)
```

**`movie_ingestion/` subpackage structure:**

| Path | Purpose |
|------|---------|
| `tracker.py` | SQLite tracker DB init, `MovieStatus`/`PipelineStage` enums, `log_filter()`, atomic JSON I/O |
| `tmdb_fetching/daily_export.py` | Stage 1: stream-download and filter TMDB daily export |
| `tmdb_fetching/tmdb_fetcher.py` | Stage 2: async TMDB detail fetch, field extraction, watch provider key encoding |
| `tmdb_quality_scoring/tmdb_filter.py` | Stage 3: quality score threshold (no hard filters) |
| `tmdb_quality_scoring/tmdb_quality_scorer.py` | Quality scoring model (4 weighted signals) |
| `tmdb_quality_scoring/tmdb_data_analysis.py` | Diagnostic: per-attribute distributions from tmdb_data (informs funnel design) |
| `tmdb_quality_scoring/plot_tmdb_quality_scores.py` | Diagnostic: Gaussian-smoothed survival curve + derivatives (determines threshold) |
| `imdb_scraping/run.py` | Stage 4 entry point: batch orchestration with commit-per-batch |
| `imdb_scraping/scraper.py` | Per-movie orchestration: fetch → transform → persist |
| `imdb_scraping/http_client.py` | Async GraphQL client with proxy, retry, semaphore, and raw JSON caching |
| `imdb_scraping/parsers.py` | GraphQL response → `IMDBScrapedMovie` transformer (keyword scoring, synopsis priority) |
| `imdb_scraping/models.py` | Pydantic models for IMDB scraped data (`IMDBScrapedMovie` and sub-models) |
| `imdb_scraping/fix_stale_statuses.py` | One-off reconciliation script for stuck `tmdb_quality_passed` movies |
| `imdb_quality_scoring/imdb_quality_scorer.py` | Stage 5: two hard gates (title-type, missing-text) + combined TMDB+IMDB quality scorer (8 signals) |
| `scoring_utils.py` | Shared scoring utilities (vote_count, popularity, provider key unpacking) used by Stage 3 and Stage 5 |
| `imdb_quality_scoring/analyze_imdb_quality.py` | Diagnostic: per-field coverage and distribution report for scraped IMDB data |

**IMDB scraping environment:** Requires `DATA_IMPULSE_LOGIN` and `DATA_IMPULSE_PASSWORD` in `.env` for proxy access. Optional `DATA_IMPULSE_HOST`/`DATA_IMPULSE_PORT` (defaults to `gw.dataimpulse.com:823`).

### LLM Provider

LLM routing is centralized in `implementation/llms/generic_methods.py`
and is now **multi-provider**, not Kimi-only. Supported backends
include OpenAI, Kimi/Moonshot, Gemini, Groq, Alibaba/Qwen,
Anthropic, and WHAM. Query understanding remains a shared LLM layer,
while Stage 6 metadata generation lives in
`movie_ingestion/metadata_generation/` and calls the shared router.

Structured output handling varies by provider:
- OpenAI uses `chat.completions.parse()`
- Kimi uses OpenAI-compatible `chat.completions.create()` with an
  explicit JSON schema and manual validation
- Other providers are routed through provider-specific adapters in
  the same module

OpenAI is also used for embeddings (`text-embedding-3-small`, 1536
dims).

### Cross-Codebase Invariants

See docs/conventions.md for the full list. Key invariants:

- **`movie_id` is always `tmdb_id` (BIGINT/uint64).** Never introduce a secondary ID system.
- **String normalization runs identically at ingest and query time.** A mismatch is a silent retrieval bug.
- **Qdrant scores are final.** Not recomputed at reranking.
- **Never query Postgres per-candidate.** Bulk fetch with `WHERE movie_id = ANY($1)`.
- **Never cache partial DAG outputs.** Entire `QueryUnderstandingResponse` is one atomic Redis key.
- **Embedding cache does not lowercase.** QU cache normalizer lowercases; embedding cache does not.
- **Qdrant payload is for hard filters only.** Full metadata lives in Postgres.

### Coding Practices

See .claude/rules/coding-standards.md for the full coding standards.

Write code from the perspective of an expert senior software engineer.
Code must be generalizable, modular, and efficient. Always liberally
include comments explaining what you are doing and why.

When making tradeoff decisions, consult docs/PROJECT.md for the
priority ordering.
