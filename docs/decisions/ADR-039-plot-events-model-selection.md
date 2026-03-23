# [039] — plot_events Model Selection: gpt-5-mini

## Status
Active

## Context

After completing the reference-free evaluation pipeline (ADR-028, ADR-034),
a 21-movie 6-candidate evaluation was run to select the production model
for `generate_plot_events()`. Candidates: gpt-5-mini, gpt-5-nano,
gpt-5.4-nano, qwen3.5-flash, gpt-oss-120b, llama-4-scout. Prior
evaluation had selected `gemini-2.5-flash-lite__think-1k__short-prompt`
but that result was superseded by running the evaluation against the
finalized simplified schema (just `plot_summary`, no `setting` or
`major_characters`).

The 21-movie set used 3 movies from each of 6 sparsity groups (half of
each original 6-movie group). All non-Gemini candidates were evaluated;
Gemini was excluded because it requires `max_output_tokens` not `max_tokens`
and the generic router does not normalize this parameter — making it
incompatible with the batch pipeline's per-type kwargs design.

Key tradeoff: groundedness (factual accuracy, no hallucination) vs. cost.
The pipeline generates ~112K movies; every percentage point of hallucination
rate translates to thousands of degraded search results.

## Decision

Lock `generate_plot_events()` to OpenAI gpt-5-mini with
`{"reasoning_effort": "minimal", "verbosity": "low"}`.

Provider, model, and kwargs are module-level constants (`_PROVIDER`,
`_MODEL`, `_MODEL_KWARGS`) — not caller parameters. Production callers
pass no model arguments; the evaluation notebook passes explicit args
for candidate comparison.

## Alternatives Considered

1. **qwen3.5-flash ($10.98 at batch pricing)**: Scored 4.56 overall with
   consistent small inference leaps (slight hallucination). Rejected —
   at $17 total cost difference for ~112K movies, the groundedness gap
   (4.86 vs. lower) favors gpt-5-mini unambiguously.

2. **gpt-5-nano / gpt-5.4-nano**: Lower cost but measurably lower
   groundedness. Not worth the quality reduction for the most critical
   pipeline field — `plot_summary` is the primary input to 4 of 6
   downstream Wave 2 generators.

3. **llama-4-scout / gpt-oss-120b**: Did not match gpt-5-mini on
   groundedness at batch pricing.

4. **Keep provider/model as caller params (ADR-027 pattern)**: The
   ADR-027 rationale was that model selection wasn't finalized and
   callers needed to pass explicit params during evaluation. With selection
   finalized, keeping params open creates risk of accidental misconfiguration
   at scale. Module-level constants are greppable and auditable.

## Consequences

- `generate_plot_events()` signature no longer accepts `provider`,
  `model`, or `**kwargs`. Callers that previously passed these (the
  deleted `wave1_runner.py`, older evaluation code) will fail at import
  time if not updated.
- The evaluation notebook (cell 4) passes explicit args via the
  multi-candidate evaluation path — that path calls `generate_llm_response_async`
  directly, bypassing `generate_plot_events()`.
- ADR-027 (real-time first design) is partially superseded for plot_events:
  the generator no longer accepts provider/model overrides. ADR-027 remains
  active for all other generators.
- Batch pipeline cost for plot_events at 50% batch pricing:
  ~$17 total for ~112K movies at gpt-5-mini rates.

## References

- ADR-027 (generator real-time first design) — original provider/model param convention
- ADR-033 (plot_events two-branch design) — generator design context
- ADR-040 (plot_events schema simplification) — schema changes evaluated alongside model selection
- `movie_ingestion/metadata_generation/generators/plot_events.py`
