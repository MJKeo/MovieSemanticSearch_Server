# AGENTS.md

Instructions for Codex in this repository. This file is the
Codex-native bootstrap doc and should contain the practical guidance
needed to work here without separately re-reading Claude-specific
rule files every time.

The structured docs under `docs/` remain the source of truth for
product context, architecture, conventions, and decisions.

## Session Startup

Read these at the start of every session:

- `docs/PROJECT.md` for product context, priorities, constraints
- `docs/personal_preferences.md` for the user's collaboration style

Read these when relevant:

- `docs/modules/<module>.md` the first time you work in that module
  during a session
- `docs/decisions/` before making non-trivial tradeoff decisions
- `docs/conventions.md` when working across module boundaries or
  relying on shared invariants/patterns
- `docs/TODO.md` when starting work in an area, in case there is
  already queued follow-up work there
- `DIFF_CONTEXT.md` for current-session uncommitted context
- `search_improvement_planning/` when working on search redesigns or
  architecture changes in retrieval/query understanding

## Decision Framework

When priorities are in tension, optimize in this order:

1. Search quality
2. Latency
3. Cost
4. Code simplicity

For local code choices, use this implementation order:

1. Correctness and security
2. Readability and maintainability
3. Performance and efficiency

Major wins in a lower project priority can justify minor costs in a
higher one, but do that deliberately.

## Opinion And Assessment Rules

When asked for an opinion, assessment, or recommendation:

- Do not default to agreement
- Independently evaluate the user's position and any source material
- If you agree, explain why with concrete reasoning
- If you disagree or see weakness, say so directly and explain why
- Do not rubber-stamp with vague approval

Before giving an opinion, make sure you understand:

- The desired outcome
- What success looks like concretely
- What separates good from great in this context

If those are unclear, ask before advising.

Back claims with evidence:

- Do not rely only on memory for factual or current best-practice claims
- Use docs, code, decision records, and web research when warranted
- Distinguish facts, conventions, and your own reasoning
- Recommend approaches via explicit tradeoffs, not taste alone

For complex questions, reason in this order:

1. Break down the problem
2. Clarify the goal
3. Gather context from code/docs/decision records and external sources if needed
4. Evaluate realistic options with tradeoffs
5. Reach a conclusion tied back to the goal and evidence

## Documentation Rules

The repo docs are the persistent knowledge base for this project.
Do not rely on external memory systems for project knowledge.

Autonomous doc updates allowed:

- `docs/modules/` when you notice module docs are stale while
  working in that module; keep fixes proportional
- `docs/TODO.md` when you discover actionable follow-up work; use
  the existing format

Autonomous doc updates forbidden:

- `docs/PROJECT.md`
- `docs/conventions.md`
- `docs/decisions/`

Session learnings belong here:

- Personal preferences -> `docs/personal_preferences.md`
- Convention candidates -> `docs/conventions_draft.md`
- Workflow suggestions -> `docs/workflow_suggestions.md`

## Context Tracking

After completing each implementation task that writes or modifies
code, append a structured entry to `DIFF_CONTEXT.md` before
reporting results.

If `DIFF_CONTEXT.md` does not exist, create it with:

```md
# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.
```

Organize entries by intent, not by file.

Small change:

```md
## [brief description of intent]
Files: [paths] | [one sentence: what and why]
```

Medium change:

```md
## [intent]
Files: [paths]
Why: [motivation]
Approach: [what you did and why]
Design context: [relevant docs/decisions/PROJECT references]
Testing notes: [coverage or risks]
```

Large change:

```md
## [intent]
Files: [paths]

### Intent
[What this achieves and why]

### Key Decisions
[Justification, alternatives, references]

### Planning Context
[Planning choices that shaped implementation]

### Testing Notes
[Coverage needs, risks, edge cases]
```

Rules:

- Match entry length to the amount of reasoning a fresh reader needs
- Include decisions and justifications, not just what changed
- Reference permanent docs instead of restating them
- Do not rewrite old entries
- Do not include code snippets

## Coding Standards

Architecture and structure:

- Prefer small single-responsibility functions, roughly under 40 lines
- Extract shared logic instead of duplicating it
- Prefer dependency injection over hard-coded dependencies
- Keep public interfaces stable and implementation details private
- Follow the surrounding codebase's established patterns

Error handling:

- Validate inputs at system boundaries
- Prefer typed/custom exceptions over generic failures
- Fail with meaningful debugging context
- Never silently swallow exceptions; log or propagate them

Security:

- Sanitize user-provided input
- Never log secrets, tokens, or PII
- Use parameterized queries; never interpolate SQL or shell values
- If you notice a security issue, flag it and fix it

Performance:

- Prefer lazy evaluation and streaming for large data sets
- Avoid unnecessary allocations in hot paths
- Use appropriate data structures
- Do not add complexity before profiling justifies it

Code quality:

- Choose names that explain what and why, not how
- Add comments when they help explain intent or non-obvious behavior
- Avoid magic numbers/strings; use named constants
- Keep nesting shallow with early returns

## Test Boundaries

Unless the user explicitly asks:

- Do not read test files
- Do not edit test files
- Do not run tests

If another command incidentally triggers tests and they fail:

- Stop and report the failures
- Do not investigate or fix them in the same pass unless asked

Never modify tests just to make them pass. Tests define expected
behavior.

## Commands And Environment

```bash
# Install dependencies
uv sync

# Run all tests
pytest unit_tests/

# Run one test file
pytest unit_tests/test_search.py -v

# Run one test
pytest unit_tests/test_metadata_scoring.py::TestClassName::test_method_name -v

# Start services
docker-compose up

# Run API locally
uvicorn api.main:app --reload
```

Environment notes:

- Python 3.13
- `pytest` uses `asyncio_mode = "auto"`
- Production target is a single EC2 `t3.large` running all services
  via Docker Compose, so latency/cost wins must respect tight CPU/RAM
  limits
- `.env` must define `TMDB_API_KEY`, `OPENAI_API_KEY`,
  `MOONSHOT_API_KEY`, Postgres/Redis/Qdrant connection strings, and
  IMDB proxy credentials when scraping
- IMDB scraping uses DataImpulse proxies via
  `DATA_IMPULSE_LOGIN`/`DATA_IMPULSE_PASSWORD` and optionally
  `DATA_IMPULSE_HOST`/`DATA_IMPULSE_PORT`

## Project Overview

This is a multi-channel movie search engine for US users trying to
find something to watch, especially on streaming right now.

Search flow:

1. LLM query understanding decomposes a natural-language request
2. Retrieval runs in parallel across lexical, vector, and metadata channels
3. Scores are normalized and merged
4. Results are reranked and enriched with display metadata

Core directories:

- `db/` search orchestration, scoring, Postgres/Qdrant/Redis clients
- `api/` FastAPI application
- `implementation/classes/` Pydantic models and shared enums
- `implementation/llms/` shared LLM routing and structured output handling
- `implementation/prompts/` prompts for query understanding tasks
- `implementation/misc/` normalization and utility helpers
- `movie_ingestion/` TMDB -> IMDB -> metadata generation -> embed -> ingest
- `unit_tests/` pytest suite
- `docs/` project context, modules, decisions, conventions

## Search And Storage Architecture

Search combines:

- Lexical search in PostgreSQL
- Vector search in Qdrant
- Structured metadata scoring in memory

Final score is a weighted merge of lexical, vector, and metadata
channels after normalization to `[0, 1]`.

There are 8 vector spaces in Qdrant:

- `dense_anchor_vectors`
- `plot_events_vectors`
- `plot_analysis_vectors`
- `narrative_techniques_vectors`
- `viewer_experience_vectors`
- `watch_context_vectors`
- `production_vectors`
- `reception_vectors`

Embeddings use OpenAI `text-embedding-3-small` (1536 dims).

Data stores:

- PostgreSQL 15 for movie metadata and lexical entities
- Qdrant for vector retrieval
- Redis 7 for caches and trending data

## Ingestion Pipeline

High-level stages:

1. TMDB daily export download and initial filtering
2. TMDB detail fetching
3. TMDB quality scoring and filtering
4. IMDB scraping via GraphQL through US-targeted proxies
5. IMDB quality filtering
6. LLM metadata generation
7. Vector text generation in `movie_ingestion/final_ingestion/vector_text.py`
8. Ingestion into Postgres and Qdrant, with embedding generated inside
   `movie_ingestion/final_ingestion/ingest_movie.py`

Important tracker invariant:

- `movie_ingestion/tracker.py` owns pipeline progress tracking
- Use `log_filter()` rather than directly mutating tracker tables
- `implementation/vectorize.py` is legacy ChromaDB code and is not part
  of the active ingestion pipeline

## LLM Routing

LLM routing is centralized in
`implementation/llms/generic_methods.py` and is multi-provider.

Supported backends include OpenAI, Moonshot/Kimi, Gemini, Groq,
Alibaba/Qwen, Anthropic, and WHAM. Query understanding and metadata
generation share this routing layer.

Current operational split:

- Search-time query understanding uses Moonshot/Kimi structured output
- Ingestion-time metadata generation uses OpenAI `gpt-5-mini`
- Embeddings use OpenAI `text-embedding-3-small` (1536 dims)

Structured output handling is provider-specific. OpenAI uses parsed
structured outputs; other providers use adapter-specific handling and
validation in the same module.

## Cross-Codebase Invariants

- `movie_id` is always `tmdb_id`
- String normalization must match between ingest and query time
- Qdrant scores are final and are not recomputed during reranking
- Never query Postgres per candidate; bulk fetch with `WHERE movie_id = ANY($1)`
- Cache the full `QueryUnderstandingResponse` atomically, not partial DAG outputs
- Embedding-cache normalization and query-understanding-cache normalization differ; do not collapse them
- Qdrant payload is for hard filters only; full metadata lives in Postgres

## Working Style

- Prefer simple structural fixes over clever abstractions
- Teach while doing when explaining non-obvious design choices
- Prefer bulk operations over per-row loops
- Diagnose from hard data instead of locking onto early assumptions
- Save structured analysis output to files when it will be consumed downstream
- Treat analytical conclusions as hypotheses to validate, not overconfident verdicts
- When the user gives explicit criteria, evaluate strictly against those criteria
- Answer the scoped question that was asked; do not spiral outward
- Finish design discussion before writing docs or implementation when the user is still iterating
- Require explicit parameters instead of hiding important choices behind defaults when designing configurable APIs
