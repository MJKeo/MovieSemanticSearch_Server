# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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
| `db/` | All database access: search orchestration, vector/lexical/metadata scoring, Postgres/Qdrant/Redis clients |
| `implementation/classes/` | Pydantic data models (`schemas.py`), `BaseMovie`, enums, watch providers |
| `implementation/llms/` | LLM calls: query understanding, structured output parsing, vector metadata generation |
| `implementation/prompts/` | System prompts for each LLM task |
| `implementation/misc/` | Utilities: string normalization, SQL LIKE escaping |
| `implementation/notebooks/` | Jupyter notebooks for exploration, DB rebuilding, and evaluation |
| `movie_ingestion/` | Ingestion pipeline: `tracker.py` (shared state), `tmdb_fetching/` (TMDB export & detail fetch), `tmdb_quality_scoring/` (quality filtering), `imdb_scraping/` (IMDB data) |
| `unit_tests/` | pytest test suite (27 files); `conftest.py` provides `base_movie_factory` fixture |
| `guides/` | Deep-dive architecture docs (23 markdown files) — **read the relevant guide before modifying any scoring logic** |

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

5-stage vector scoring pipeline (`db/vector_scoring.py`): execute → blend (80/20 original/subquery) → normalize (exponential decay) → weight → sum.

Each vector space has LLM-generated metadata (`implementation/llms/vector_metadata_generation_methods.py`) covering 7 types: plot events, plot analysis, viewer experience, watch context, narrative techniques, reception, and production. These are generated once during ingestion and stored in Qdrant payloads.

### Data Stores

| Service | Technology | Purpose |
|---------|-----------|---------|
| PostgreSQL 15 | `db/postgres.py` | Movie metadata, lexical entities, posting lists |
| Qdrant | `db/qdrant.py`, `db/vector_search.py` | 8 vector spaces × 150K movies |
| Redis 7 | `db/redis.py` | Embeddings cache, query understanding cache, trending set, TMDB detail cache |

### Movie Ingestion Pipeline

Movies are ingested in stages:
1. `movie_ingestion/tmdb_fetching/daily_export.py` — fetches bulk TMDB export (list of all movie IDs)
2. `movie_ingestion/tmdb_fetching/tmdb_fetcher.py` — fetches full TMDB detail per movie, rate-limited with adaptive backoff, stores raw data in SQLite tracker DB (`ingestion_data/`)
3. `implementation/llms/vector_metadata_generation_methods.py` — generates 7 LLM metadata types per movie
4. `implementation/vectorize.py` — embeds metadata into 8 vector spaces via OpenAI
5. `db/ingest_movie.py` — upserts final data into Postgres, Qdrant, and Redis

`movie_ingestion/tracker.py` manages per-movie processing state across all stages using a SQLite database.

### LLM Provider

The LLM calls use the **Moonshot/Kimi API** (`implementation/llms/generic_methods.py`) with structured output parsing via `client.beta.chat.completions.parse()`, not OpenAI Chat directly. OpenAI is used only for embeddings (`text-embedding-3-small`, 1536 dims).

### Coding Best Practices

Anytime you write code you must do so from the perspective of an expert senior software engineer with tons of industry experience. The code you write must adhere to industry best practices. Code should be generalizeable, modular, and efficient. Never implement hacky code, always assume your code will be highly scrutinized and "quick and dirty" code will be penalized.

Always liberally include comments in the code you write. It enables me to understand exactly what you are doing and why.

Additional Requirements:
- Do not run or fix unit tests unless EXPLICITLY asked to do so
- Do not add logging unless EXPLICITLY asked to do so