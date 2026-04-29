# PROJECT.md

## What This Is

A multi-channel movie search engine that decomposes natural language
queries via LLM and executes parallel retrieval across lexical,
vector, and metadata channels to return highly relevant movie
recommendations from a database of ~100K curated titles. Users
describe what they want in plain language — "a tense sci-fi thriller
like Arrival but more action-heavy" — and the system returns
ranked results by combining entity matching, semantic similarity
across 8 embedding spaces, and structured metadata preferences.

## Target Audience

People in the United States trying to find a movie to watch. The
most common use case is finding something to watch right now on
streaming platforms, but it also serves users looking for movies
currently in theaters that are worth seeing.

## Priorities (in order)

1. **Search quality** — results must feel relevant and "right"
   for the query. This is the core value proposition.
2. **Latency** — fast response times for a smooth user experience.
3. **Cost** — minimize LLM, embedding, infrastructure, and proxy
   spend across both search-time and ingestion-time operations.
4. **Code simplicity** — keep the codebase readable and
   maintainable for a solo developer.

Major wins in a lower priority can justify minor costs in a higher
priority (e.g., a large cost savings is worth a small latency
increase).

## Constraints

- Python 3.13
- PostgreSQL 15, Qdrant, Redis 7
- Single EC2 t3.large instance running all services via Docker
  Compose (2 vCPU, 8 GB RAM)
- LLM providers: Moonshot/Kimi API for V1 search-time query
  understanding (structured output); Gemini for V2 search-time query
  understanding (Steps 0/1/2); OpenAI gpt-5-mini and gpt-5.4-mini for
  ingestion-time metadata generation (model selected per generator —
  see ADR-039, ADR-043); OpenAI text-embedding-3-large (3072 dims)
  for all embeddings (per ADR-066)
- ~100K movies after quality filtering from ~1M TMDB daily exports
- US-focused: watch provider data, IMDB proxy geo-targeting, and
  content filtering all assume a US audience

## System Overview

### Search Pipeline

Multi-stage pipeline: natural language query → parallel LLM
decomposition (entities, channel weights, metadata prefs, vector
subqueries) → parallel retrieval across 3 channels (lexical via
Postgres, vector via Qdrant across 8 embedding spaces, metadata
scoring in-memory) → score merging → quality reranking → response.

### Ingestion Pipeline

Processes TMDB daily exports through a multi-stage funnel:
1. TMDB daily export download (~1M → ~800K)
2. TMDB detail fetching (~800K movies)
3. Quality scoring and filtering (~800K → ~100K)
4. IMDB scraping via GraphQL (~100K enriched)
5. IMDB quality filtering (combined TMDB+IMDB quality scorer; score is the sole filter)
6. LLM metadata generation (operational — gpt-5-mini selected after multi-candidate evaluation, multi-type batch pipeline running via generator registry; see ADR-039, ADR-043, ADR-044)
7. Vector text generation (`movie_ingestion/final_ingestion/vector_text.py`) — generates the text that gets embedded for each of 8 vector spaces
8. Database ingestion into Postgres and Qdrant (implemented) — embedding is integrated into Stage 8 inside `movie_ingestion/final_ingestion/ingest_movie.py` via `generate_vector_embedding()`. Vector text generation lives in `movie_ingestion/final_ingestion/vector_text.py`.

All stages are operational, crash-safe, and idempotent.

## Module Map

| Module | Purpose | Docs |
|--------|---------|------|
| db/ | Database access, search orchestration, all scoring | docs/modules/db.md |
| implementation/classes/ | Pydantic models, enums | docs/modules/classes.md |
| implementation/llms/ | LLM calls, structured output parsing | docs/modules/llms.md |
| implementation/prompts/ | System prompts for each LLM task | — |
| implementation/misc/ | String normalization, SQL escaping | — |
| movie_ingestion/ | Ingestion pipeline (TMDB → IMDB → LLM → embed → ingest) | docs/modules/ingestion.md |
| api/ | FastAPI application | docs/modules/api.md |
| schemas/ | Shared Pydantic models, enums, V2 translation schemas | docs/modules/schemas.md |
| search_v2/ | V2 search pipeline (Steps 0–2 + Stage 3/4) | docs/modules/search_v2.md |
| unit_tests/ | pytest suite (76 files) | — |

Module docs in docs/modules/ provide concise summaries with
boundaries, interactions, and gotchas. Decision records in
docs/decisions/ capture architectural choices with rationale
and alternatives considered.
