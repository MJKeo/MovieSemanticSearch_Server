# [029] — Anthropic Extended Thinking Integration via `budget_tokens` Kwarg

## Status
Active

## Context

Claude Sonnet 4.6 was added as an evaluation candidate for `plot_events`
generation with extended thinking enabled (`budget_tokens=8000`). The
Anthropic SDK's extended thinking API has three constraints that differ
from all other providers in `generic_methods.py`:

1. Extended thinking is enabled via a `thinking` parameter dict
   (`{"type": "enabled", "budget_tokens": N}`) — not a simple flag.
2. `max_tokens` must be large enough to cover both thinking tokens and
   the structured output. Anthropic enforces this at the API level.
3. Temperature must not be set when thinking is enabled (Anthropic
   enforces 1.0 internally and rejects any explicit temperature).

The unified router (`generate_llm_response_async`) passes kwargs
through to provider functions, so a naive implementation would require
callers to manage all three constraints manually every time.

## Decision

Extended thinking is activated by passing `budget_tokens` as a kwarg
to `generate_anthropic_response_async`. The function:

1. Pops `budget_tokens` from kwargs (prevents it leaking to the API
   as an unknown param).
2. If present: constructs `thinking={"type": "enabled", "budget_tokens": N}`
   and expands `max_tokens` to `budget_tokens + 4096`.
3. If absent: behaves identically to before — `max_tokens` defaults to
   4096, no `thinking` key is set.

Callers express intent with a single kwarg (`budget_tokens=8000`) rather
than managing the three-constraint bundle themselves.

## Alternatives Considered

1. **Separate `generate_anthropic_thinking_response_async` function**:
   Would have duplicated the tool-use structured output logic. The
   `budget_tokens` kwarg keeps the implementation in one function and
   the dispatch table unchanged.

2. **Boolean `enable_thinking` flag (like Kimi)**: Kimi's flag controls
   a binary on/off without a configurable budget. Anthropic's thinking
   has a meaningful budget parameter that determines reasoning depth
   and cost; a boolean would discard that signal.

3. **Expose `thinking` dict directly via kwargs**: Would require callers
   to also remember to expand `max_tokens` and suppress `temperature`.
   The three-constraint bundle is fragile to get right at every call site.

## Consequences

- Callers that want extended thinking pass `budget_tokens=N`; callers
  that don't want it pass nothing (no API change).
- Temperature must not be set alongside `budget_tokens` — doing so will
  cause an Anthropic API error. This constraint is documented but not
  enforced in code (callers are trusted to follow the convention).
- `max_tokens` expansion formula: `budget_tokens + 4096`. The 4096
  ceiling covers the structured output; adjust if output schemas grow
  significantly larger.
- The judge in the evaluation pipeline now uses GPT-5.4 via WHAM
  (see ADR-030), not Anthropic. This consequence is no longer
  relevant to this ADR but is preserved for historical context.

## References

- ADR-026 (multi-provider routing) — `generate_llm_response_async` architecture
- ADR-028 (evaluation pipeline) — context for why thinking candidates are needed
- `implementation/llms/generic_methods.py` — `generate_anthropic_response_async`
- `movie_ingestion/metadata_generation/evaluations/plot_events.py` — `think-med` candidate
