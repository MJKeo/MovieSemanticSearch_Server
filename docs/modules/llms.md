# implementation/llms/ — LLM Integration

All LLM API calls for both search-time query understanding and
ingestion-time vector metadata generation.

## What This Module Does

Provides two distinct LLM integration paths:
1. **Search-time**: Decomposes user queries into structured outputs
   (entities, weights, preferences, subqueries) via parallel LLM
   calls. Redis caching is planned but not yet implemented.
2. **Ingestion-time**: Generates 7 types of vector metadata per
   movie via LLM calls, producing the text that gets embedded into
   the 8 vector spaces.

## Key Files

| File | Purpose |
|------|---------|
| `generic_methods.py` | LLM client initialization and base call functions. Two clients: OpenAI (embeddings via `text-embedding-3-small`, 1536 dims; structured output via `chat.completions.parse()`) and Moonshot/Kimi (structured output via `chat.completions.create()` with explicit `response_format` JSON schema + manual `json.loads()` / `model_validate()`). Sync and async variants for both. |
| `query_understanding_methods.py` | Search-time DAG: 5 async functions that run in parallel with dependency management. Redis caching planned but not yet implemented (key format: `qu:v{N}:{hash}`, TTL 1 day). |
| `vector_metadata_generation_methods.py` | Ingestion-time: 7 generator functions (plot_events, plot_analysis, viewer_experience, watch_context, narrative_techniques, production, reception). Organized in two waves — Wave 1 runs plot_events/watch_context/reception in parallel, Wave 2 depends on plot_events output. |

## Boundaries

- **In scope**: LLM API calls, structured output parsing, embedding
  generation, QU caching logic.
- **Out of scope**: System prompts (live in `implementation/prompts/`),
  output schemas (live in `implementation/classes/schemas.py`),
  vector text construction (lives in `implementation/vectorize.py`).

## Search-Time Query Understanding DAG

The DAG produces a complete `QueryUnderstandingResponse` from a
user query. On Redis cache hit, the entire DAG is skipped.

| Function | Output | Dependencies |
|----------|--------|-------------|
| `extract_lexical_entities_async()` | Actors, directors, franchises, characters | None |
| `create_channel_weights_async()` | Lexical/vector/metadata relevance (RelevanceSize) | None |
| `extract_all_metadata_preferences_async()` | Genre, date, rating, streaming, duration, trending, reception, language, popularity | None |
| `create_single_vector_subquery_async()` | One subquery per non-anchor vector space (called per-space, not as a batch) | None |
| `create_single_vector_weight_async()` | Per-space relevance weight (called per-space, not as a batch) | None |

All functions run in parallel. When caching is implemented, the
cached blob will be the complete structured output — never cache
partial DAG results.

## Ingestion-Time LLM Generation

8 LLM calls per movie organized in two waves:

**Wave 1** (parallel): plot_events, watch_context, reception
**Wave 2** (parallel, depends on plot_events for plot_synopsis):
plot_analysis, viewer_experience, narrative_techniques,
production_keywords, source_of_inspiration

Current token usage: ~28K input + ~8K output per movie.
See decision record ADR-012 for optimization proposals.

## Gotchas

- *(Planned)* The QU cache key will include a prompt version prefix
  (`v{N}`). Bump the version when ANY system prompt changes.
- *(Planned)* QU cache will normalize query text (lowercase, trim,
  collapse whitespace). Embedding cache does NOT lowercase.
- `selected_filters`, `shown_movie_counts`, and trending state
  are applied downstream and never baked into cached QU results.
- Many LLM output fields (justification, explanation) exist for
  chain-of-thought quality but are not used in final embeddings.
