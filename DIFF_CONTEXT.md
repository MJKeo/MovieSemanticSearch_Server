# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Implement plot events generator (new flow, async)

Files: implementation/llms/generic_methods.py, movie_ingestion/metadata_generation/inputs.py, movie_ingestion/metadata_generation/prompts/plot_events.py, movie_ingestion/metadata_generation/generators/plot_events.py, movie_ingestion/metadata_generation/errors.py

### Intent
First generator implementation for the redesigned LLM metadata pipeline. Creates an isolated async method for generating plot events metadata for a single movie, following the new flow spec (docs/llm_metadata_generation_new_flow.md Section 4.1).

### Key Decisions
- **Async OpenAI method:** Added `generate_openai_response_async` to generic_methods.py using the existing `async_openai_client`. Mirrors the sync version's signature and return type.
- **MultiLineList marker class:** Extended `build_user_prompt` in inputs.py to support long-text list formatting (newline-separated with `- ` prefix) via a `MultiLineList(list)` subclass. Regular lists remain comma-separated. Also added empty-list guard to skip empty lists entirely.
- **System prompt:** Copied from existing PLOT_EVENTS_SYSTEM_PROMPT with two changes: title described as "Title (Year)" format, and no-hallucination rule replaces "supplement with your own knowledge."
- **Single synopsis:** Generator takes only `plot_synopses[0]` (singular `plot_synopsis` in prompt) rather than the full list, since the first entry is the longest/most detailed.
- **Shared errors:** Created errors.py with `MetadataGenerationError` (API failure) and `MetadataGenerationEmptyResponseError` (None response) — reusable by all future generators. Stores generation_type and title for context.
- **TokenUsage reuse:** Imports `TokenUsage` NamedTuple from `implementation.llms.vector_metadata_generation_methods` rather than duplicating.
