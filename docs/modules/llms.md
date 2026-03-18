# implementation/llms/ — LLM Integration

All LLM API calls for search-time query understanding, plus shared
provider clients and routing used by the ingestion-time metadata
generation pipeline.

## What This Module Does

Provides two distinct roles:

1. **Search-time LLM integration**: Decomposes user queries into
   structured outputs (entities, weights, preferences, subqueries)
   via parallel LLM calls using the Kimi/Moonshot API.

2. **Shared LLM infrastructure for ingestion-time generation**:
   The `LLMProvider` enum, `generate_llm_response_async` router,
   and all provider-specific async generation functions are defined
   here and imported by `movie_ingestion/metadata_generation/generators/`
   and `movie_ingestion/metadata_generation/evaluations/`.

The ingestion-time generation pipeline (generators, schemas, prompts)
lives in `movie_ingestion/metadata_generation/`. See
`docs/modules/ingestion.md` (Stage 6 section) and ADR-024.

## Key Files

| File | Purpose |
|------|---------|
| `generic_methods.py` | LLM client initialization, all provider-specific generation functions, `LLMProvider` enum, and `generate_llm_response_async` unified router. Seven providers: OpenAI, Kimi, Gemini, Groq, Alibaba, Anthropic, WHAM. Embeddings via OpenAI `text-embedding-3-small` (1536 dims). |
| `query_understanding_methods.py` | Search-time DAG: 5 async functions that run in parallel with dependency management. Uses Kimi for all QU calls. Redis caching planned but not yet implemented (key format: `qu:v{N}:{hash}`, TTL 1 day). |
| `vector_metadata_generation_methods.py` | Legacy ingestion-time generation functions. Superseded by `movie_ingestion/metadata_generation/generators/`. Not used in the active pipeline. Provides `TokenUsage` NamedTuple (imported by generators). |

## Boundaries

- **In scope**: Search-time LLM API calls, structured output parsing,
  embedding generation, QU caching logic, shared provider clients and
  routing for ingestion-time generators and evaluations.
- **Out of scope**: Ingestion-time metadata generator logic (now in
  `movie_ingestion/metadata_generation/generators/`), system prompts
  (live in `movie_ingestion/metadata_generation/prompts/` for ingestion
  and `implementation/prompts/` for search), output schemas (live in
  `implementation/classes/schemas.py` and
  `movie_ingestion/metadata_generation/schemas.py`).

## LLM Provider Architecture

Seven providers are supported, all accessed through `generate_llm_response_async`:

| Provider | Enum | Client | SDK | Structured Output Pattern |
|----------|------|--------|-----|--------------------------|
| OpenAI | `LLMProvider.OPENAI` | `async_openai_client` | `openai` | `chat.completions.parse()` with Pydantic model |
| Kimi (Moonshot) | `LLMProvider.KIMI` | `async_kimi_client` | `openai` (compatible) | `chat.completions.create()` with explicit JSON schema + manual `json.loads()` / `model_validate()` |
| Gemini | `LLMProvider.GEMINI` | `gemini_client` | `google-genai` | `response_mime_type` + `response_json_schema` in config dict |
| Groq | `LLMProvider.GROQ` | `async_groq_client` | `groq` | `json_schema` response_format with `strict: False` |
| Alibaba/Qwen | `LLMProvider.ALIBABA` | `async_alibaba_client` | `openai` (DashScope compatible) | `chat.completions.parse()` with Pydantic model |
| Anthropic | `LLMProvider.ANTHROPIC` | `async_anthropic_client` | `anthropic` | Tool-use pattern: schema registered as a tool, `tool_choice` forces the model to call it |
| WHAM | `LLMProvider.WHAM` | per-call `AsyncOpenAI` | `openai` | `responses.stream()` with `text_format` (Responses API, requires streaming) |

**Unified router (`generate_llm_response_async`)**: Dispatch table
`_PROVIDER_DISPATCH` maps `LLMProvider` → async function. Kimi is
special-cased via `_PROVIDERS_WITHOUT_MODEL_PARAM` (its model is
hardcoded internally). All other providers accept an explicit `model`
string. Provider-specific kwargs (e.g. `reasoning_effort` for OpenAI,
`enable_thinking` for Kimi, `temperature` for Gemini/Groq/Alibaba)
are passed through unchanged. Errors propagate without wrapping.

All provider methods return `Tuple[BaseModel, int, int]` (parsed
response, input tokens, output tokens).

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

## Gotchas

- *(Planned)* The QU cache key will include a prompt version prefix
  (`v{N}`). Bump the version when ANY system prompt changes.
- *(Planned)* QU cache will normalize query text (lowercase, trim,
  collapse whitespace). Embedding cache does NOT lowercase.
- `selected_filters`, `shown_movie_counts`, and trending state
  are applied downstream and never baked into cached QU results.
- Many LLM output fields (justification, explanation) exist for
  chain-of-thought quality but are not used in final embeddings.
- Kimi return type: both sync and async Kimi methods return
  `Tuple[BaseModel, int, int]`. Callers in `query_understanding_methods.py`
  unpack with `parsed, _, _`. A pre-existing bug (returning only the
  model, not the tuple) was fixed when the multi-provider work landed.
- Kimi schema name: uses `response_format.__name__` (actual class name),
  not `response_format.__class__.__name__` (which returns the metaclass).
- Gemini config: required keys (`response_mime_type`, `response_json_schema`,
  `system_instruction`) are set after kwargs spread, so they cannot be
  accidentally overridden by callers.
- Groq uses `strict: False` in its json_schema response_format for broader
  model compatibility.
- Alibaba/Qwen uses the DashScope US endpoint
  (`https://dashscope-us.aliyuncs.com/compatible-mode/v1`).
- Anthropic uses OAuth token authentication (`ANTHROPIC_API_KEY` env var)
  rather than a standard API key. `max_tokens` defaults to 4096 if not
  provided (required by Anthropic API). Extended thinking is supported
  via the `budget_tokens` kwarg — when present, `thinking` is enabled
  and `max_tokens` is expanded to `budget_tokens + 4096` to cover both
  thinking and output tokens. Temperature must not be set when thinking
  is enabled (Anthropic enforces this). See ADR-029.
- **`budget_tokens` is popped before forwarding** to the Anthropic API —
  it is not a native Anthropic parameter. Passing it through would cause
  an API error.
- **WHAM requires `api_key` (OAuth access_token) and `account_id`** at every
  call site. Call `get_valid_auth()` from `evaluations/openai_oauth.py` once
  before constructing concurrent tasks, then pass the result through.
  WHAM is evaluation-only; it uses `responses.stream()` (Responses API),
  not `chat.completions`. With any `reasoning_effort` other than `"none"`,
  GPT-5.4 rejects `temperature`, `top_p`, `max_output_tokens`, and
  `logprobs`. See ADR-030.
- Required env vars: `OPENAI_API_KEY`, `MOONSHOT_API_KEY`, `GOOGLE_API_KEY`,
  `GROQ_API_KEY`, `ALIBABA_API_KEY`, `ANTHROPIC_API_KEY`. WHAM uses OAuth
  tokens managed by `evaluations/openai_oauth.py` (no dedicated env var).
