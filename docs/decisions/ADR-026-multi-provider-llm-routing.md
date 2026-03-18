# [026] — Multi-Provider LLM Routing for Ingestion-Time Generation

## Status
Active

## Context

Stage 6 LLM metadata generation requires choosing a model (and provider)
for each of 7–8 generation types across ~112K movies. ADR-012 identified
model selection as the largest cost lever: switching from GPT-5-mini to
a cheaper model could reduce per-movie cost by 80%+. Validating which
model produces acceptable quality requires side-by-side comparison across
providers.

The existing codebase had sync and async methods for OpenAI and Kimi only.
Adding Gemini, Groq, and Alibaba/Qwen for comparison required either
duplicating call-site logic per provider or introducing a unified routing
layer.

## Decision

Add a `LLMProvider` enum and a `generate_llm_response_async` unified router
to `implementation/llms/generic_methods.py`. Originally five providers:
OPENAI, KIMI, GEMINI, GROQ, ALIBABA. Later expanded to seven with
ANTHROPIC (ADR-029) and WHAM (ADR-030).

**Dispatch table pattern**: `_PROVIDER_DISPATCH` maps each `LLMProvider`
to its async generation function. The router forwards provider-agnostic
params (prompts, response_format, model) and passes `**kwargs` unchanged
to the provider function, enabling provider-specific params
(e.g. `reasoning_effort`, `enable_thinking`, `temperature`) without
proliferating separate call signatures.

**Kimi special-case**: Kimi hardcodes its model internally, so it is
excluded from `model` parameter forwarding via `_PROVIDERS_WITHOUT_MODEL_PARAM`.

**No try/except in router**: Errors from provider functions propagate
unchanged. Each generator is responsible for wrapping in typed errors.

**Native SDKs for Gemini and Groq**: Gemini uses `google-genai` (not
the OpenAI-compatible endpoint) because its structured output config
differs. Groq uses the native `groq` SDK. Alibaba/Qwen uses the
OpenAI-compatible DashScope endpoint and reuses the existing `openai` package.

**Structured output patterns per provider**:
- OpenAI / Alibaba: `chat.completions.parse()` with Pydantic model
- Kimi / Groq: explicit JSON schema in `response_format`, manual `json.loads()` / `model_validate()`
- Gemini: `response_mime_type` + `response_json_schema` in config dict; required keys spread after kwargs so they cannot be overridden

## Alternatives Considered

1. **One-off per-provider call sites in each generator**: Would work but
   creates code duplication across 7 generators × 5 providers. Adding a
   new provider later requires editing every generator. Rejected.

2. **Wrap all providers behind the OpenAI-compatible endpoint**: Gemini
   and Groq both offer compatible endpoints, but they have limitations
   with structured output. Using native SDKs ensures full feature support.
   Rejected for Gemini and Groq; accepted for Alibaba (DashScope is stable).

3. **Abstract provider into a class hierarchy**: More extensible but adds
   complexity without clear benefit for the current 5-provider scope.
   PROJECT.md priority 4 (code simplicity) favors the dispatch table.

## Consequences

- All new ingestion-time generators call `generate_llm_response_async`
  with an explicit `provider` and `model` — no defaults on generator params.
  Caller (playground notebook, future orchestrator) is responsible for
  specifying these.
- New env vars required: `GOOGLE_API_KEY`, `GROQ_API_KEY`, `ALIBABA_API_KEY`.
- The `generate_llm_response_async` router is the canonical entry point
  for all ingestion-time LLM calls. The provider-specific functions in
  `generic_methods.py` are implementation details.
- Kimi return-type bug (was returning parsed model only, not the full tuple)
  fixed as part of this work. All 5 QU callers in
  `query_understanding_methods.py` updated to unpack with `parsed, _, _`.

## References

- ADR-012 (LLM generation cost optimization) — model selection analysis
- docs/modules/llms.md — provider architecture table and gotchas
- implementation/llms/generic_methods.py — full implementation
- movie_ingestion/metadata_generation/generators/plot_events.py — first consumer
