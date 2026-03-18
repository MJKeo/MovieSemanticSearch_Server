# [030] — WHAM Backend OAuth Integration for Evaluation Reference and Judge

## Status
Active

## Context

The evaluation pipeline (ADR-028) originally used Claude Opus as the reference
model and Claude Sonnet as the judge. Switching to GPT-5.4 for both roles
reduces cost. GPT-5.4 is exposed via the ChatGPT WHAM backend
(`chatgpt.com/backend-api/codex`), a ChatGPT-subscription-gated endpoint that
is not reachable through the standard OpenAI API (which requires an
`organization_id` not available via OAuth).

Three constraints distinguish WHAM from all other providers in `generic_methods.py`:
1. Auth is ChatGPT OAuth (PKCE + `ChatGPT-Account-Id` header), not a standard API key.
2. The API surface is the Responses API (`responses.stream()`), not `chat.completions`.
3. `stream=True` is mandatory; reasoning models reject `temperature`, `top_p`,
   `max_output_tokens`, and `logprobs` unless `reasoning_effort="none"`.

## Decision

Add `LLMProvider.WHAM` as a seventh provider in `generic_methods.py`, backed by
`generate_wham_response_async`. A new `openai_oauth.py` module in
`evaluations/` handles the full OAuth token lifecycle.

**Separate provider enum value** rather than overloading `LLMProvider.OPENAI`:
WHAM uses a different base URL, different auth headers, different API method
(`responses.stream` not `chat.completions.parse`), and has a distinct
set of supported parameters. A separate enum makes the difference explicit.

**`openai_oauth.py` PKCE flow**: Uses the Codex CLI's public `client_id`
(`app_EMoamEEZ73f0CkXaXp7hrann`), browser-based consent, and JWT decoding
(no signature verification, matching Codex CLI pattern) to extract
`account_id` and token expiry. Tokens are persisted to
`evaluation_data/openai_oauth_tokens.json` and auto-refreshed when they
expire within a 20-minute buffer.

**RFC 8693 token exchange was a dead end**: Standard OpenAI API requires an
`organization_id` in the JWT that OAuth tokens don't carry. WHAM bypasses
this by using the raw `access_token` directly.

**`responses.stream()` with `text_format`**: WHAM requires `stream=True` for
all requests; `responses.parse()` doesn't support streaming. Using `stream()`
with `text_format` satisfies both the streaming requirement and automatic
Pydantic parsing.

**`reasoning_effort` mapped to nested `reasoning.effort`**: Responses API
uses `{"effort": "low"}` rather than a flat kwarg.

## Alternatives Considered

1. **RFC 8693 token exchange**: Standard mechanism for OAuth-to-API-key exchange.
   Failed because the resulting token lacks `organization_id`. Removed after
   repeated 401 errors.

2. **Overload `LLMProvider.OPENAI` with WHAM-specific kwargs**: Would hide
   the auth, base URL, and API differences from callers. Rejected — the
   differences are too fundamental to paper over.

3. **Stay with Anthropic for reference and judge**: Claude Opus is higher
   quality than needed for the reference role; GPT-5.4 is sufficient and
   substantially cheaper via the ChatGPT subscription path.

## Consequences

- `LLMProvider.WHAM` requires `api_key` (OAuth access_token) and `account_id`
  kwargs at every call site. Callers must call `get_valid_auth()` before
  constructing tasks and pass the result through.
- WHAM judge calls must not include `temperature`, `max_output_tokens`, or
  `max_tokens` when `reasoning_effort` is anything other than `"none"`.
- Token persistence file (`evaluation_data/openai_oauth_tokens.json`) must
  be gitignored — it contains live OAuth credentials.
- The WHAM provider is evaluation-only. Production Stage 6 generation uses
  the standard OpenAI API (`LLMProvider.OPENAI`) or another provider chosen
  by the evaluation.
- End-to-end streaming + structured output path not yet fully verified against
  all candidate schemas.

## References

- ADR-026 (multi-provider routing) — `LLMProvider` enum and dispatch table
- ADR-028 (evaluation pipeline) — why a reference model is needed
- `implementation/llms/generic_methods.py` — `generate_wham_response_async`
- `movie_ingestion/metadata_generation/evaluations/openai_oauth.py` — OAuth lifecycle
