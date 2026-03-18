# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Switch evaluation pipeline from Anthropic to GPT-5.4 via ChatGPT WHAM backend
Files: `movie_ingestion/metadata_generation/evaluations/openai_oauth.py` (new),
`implementation/llms/generic_methods.py`, `movie_ingestion/metadata_generation/evaluations/plot_events.py`,
`movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py`

### Intent
Replace Claude Opus 4.6 (reference) and Claude Sonnet 4.6 (judge) with GPT-5.4 accessed
via ChatGPT OAuth through the WHAM backend, avoiding the standard OpenAI API which
requires an organization_id we can't provide via OAuth.

### Key Decisions
- **Separate `LLMProvider.WHAM`** rather than overloading the existing OpenAI provider,
  because WHAM uses the Responses API (`responses.stream()`) not `chat.completions`,
  requires different auth (OAuth + ChatGPT-Account-Id header), and has WHAM-specific
  constraints (stream=True mandatory, store=False mandatory, input_text content type).
- **`openai_oauth.py` created from scratch** — full OAuth2 PKCE flow with browser-based
  consent, JWT decode for account_id/expiry extraction (no signature verification,
  matching Codex CLI pattern), token persistence to `evaluation_data/openai_oauth_tokens.json`,
  and automatic refresh when expired.
- **RFC 8693 token exchange was a dead end** — standard API path requires `organization_id`
  in JWT which OAuth doesn't provide. WHAM bypasses this entirely by using the raw
  access_token directly.
- **Base URL is `chatgpt.com/backend-api/codex`** (not `/wham/v1`) — SDK appends
  `/responses` to base_url, so the final endpoint is `/backend-api/codex/responses`.
- **Must use `responses.stream()` not `responses.parse()`** — WHAM requires `stream=True`
  for all requests; `parse()` doesn't support streaming. `stream()` with `text_format`
  gives both mandatory streaming and automatic Pydantic parsing.
- **`reasoning_effort` mapped to nested `reasoning.effort`** for the Responses API.
  Both reference and judge calls use `reasoning_effort="low"` for speed.

### Planning Context
Iterated through several approaches: initial PKCE → added RFC 8693 exchange →
removed exchange after repeated 401 "missing organization_id" → switched to WHAM
backend based on external guide. Plan file at `.claude/plans/adaptive-snacking-hopcroft.md`.

### Testing Notes
- Full end-to-end WHAM flow not yet verified (OAuth works, structured output via
  streaming parse not yet confirmed working with PlotEventsOutput/PlotEventsJudgeOutput)
- `max_tokens` → `max_output_tokens` mapping needs confirmation with actual calls
- Token refresh flow tested implicitly (tokens saved with expiry from JWT)

## Multi-run judge averaging for plot_events evaluation
Files: `movie_ingestion/metadata_generation/evaluations/plot_events.py`

### Intent
Reduce LLM-as-judge scoring noise by running the judge 3 times per (candidate, movie) pair and storing averaged scores.

### Approach
- Added `judge_runs: int = 3` parameter to `run_evaluation()`
- Judge calls fire in parallel via `asyncio.gather` within each evaluation task's semaphore slot
- Scores averaged across runs (stored as REAL, not INTEGER)
- Reasoning concatenated with `--- Run N ---` delimiters for transparency
- Tokens summed across runs (actual API cost)
- Schema: score columns changed from INTEGER to REAL, added `judge_runs` column with idempotent ALTER TABLE migration
- Error handling: fail entire evaluation if any run fails (no partial averages)

## Parallelize reference generation and spread evaluation across candidates
Files: `movie_ingestion/metadata_generation/evaluations/plot_events.py`,
`movie_ingestion/metadata_generation/evaluations/run_evaluations_pipeline.py`

### Intent
Speed up Phase 0 (serial → parallel) and distribute Phase 1 across providers to reduce rate-limit pressure.

### Approach
- **Phase 0**: Converted serial for-loop to `asyncio.gather` + semaphore with `concurrency=15` (aggressive default). Per-request 429 retry preserved inside each task.
- **Phase 1**: Swapped task list iteration from `candidates × movies` to `movies × candidates` so concurrent semaphore slots fill with different candidates/providers. Bumped call-site concurrency from 3 to 10.

## Fix unsupported WHAM parameters in judge call
Files: `movie_ingestion/metadata_generation/evaluations/plot_events.py`

### Intent
Remove `temperature` from judge kwargs — not supported by WHAM/reasoning models (GPT-5.4 with reasoning_effort != "none"). Also confirmed `max_tokens`/`max_output_tokens` not supported by WHAM endpoint.

### Key Finding
Per OpenAI docs: GPT-5.4 only supports `temperature`, `top_p`, and `logprobs` when `reasoning_effort="none"`. With any other reasoning effort, these raise errors. WHAM also rejects `max_output_tokens`.

## Simplify analyze_results: drop median, reorder columns, add value/subset tables
Files: `movie_ingestion/metadata_generation/evaluations/shared.py`, `movie_ingestion/metadata_generation/evaluations/analyze_results.py`
Why: Median added noise to the output without providing actionable insight; dense/sparse breakdowns and a cost summary help pick the best value candidate.
Approach:
- `compute_score_summary` now computes only mean (removed median aggregation)
- Added `movie_ids` filter parameter to `compute_score_summary` for subset analysis
- Score table shows `overall_mean` as the first column after `candidate_id`
- New value-ranking table shows `candidate_id`, `overall_mean`, and `cost/1K movies`, sorted by overall descending
- Two additional score tables: "Dense movie performance" (ORIGINAL_SET_TMDB_IDS) and "Sparse movie performance" (MEDIUM + HIGH sparsity IDs)
- Extracted `_print_score_table` helper to avoid duplicating score table formatting
