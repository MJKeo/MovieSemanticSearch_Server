# [043] — Reception Model Selection: gpt-5-mini with Minimal Reasoning

## Status
Active

## Context

After finalizing the reception schema (ADR-042) and system prompt, a 36-movie
evaluation across 6 input-richness buckets (ultra-thin 0-1K chars through
very-rich 10.5K+ chars) compared 3 candidates on 5 axes: faithfulness,
extraction_quality, synthesis_quality, proportionality, downstream_utility.

Candidates:
- `gpt-5-mini` with `reasoning_effort=low`
- `gpt-5-mini` with `reasoning_effort=minimal`
- `kimi-k2.5` with thinking disabled

A cost analysis script (`analyze_evaluations.py`) computed per-candidate
batch pricing across the full corpus using actual token data from the 36
eval movies scaled to ~109K eligible movies.

Key tradeoff: kimi-k2.5 scored highest overall but carries a ~$25 premium
over gpt-5-mini-minimal at batch pricing. The question was whether the
quality difference justified the cost.

## Decision

Lock `generate_reception()` to OpenAI gpt-5-mini with
`{"reasoning_effort": "minimal"}`.

Provider, model, and kwargs are module-level constants (`_PROVIDER`,
`_MODEL`, `_MODEL_KWARGS`) matching the plot_events pattern (ADR-039).
The function accepts only `movie`; downstream callers that previously passed
provider/model/kwargs will fail and must be updated.

**Rationale**: After removing the TMDB `overview` from inputs (it caused
parametric knowledge leaking on thin-input movies at minimal reasoning effort),
gpt-5-mini-minimal matched or exceeded low-reasoning quality on the revised
prompt. At batch pricing: gpt-5-mini-minimal ≈ $62, gpt-5-mini-low ≈ $86,
kimi-k2.5 ≈ $111. The ~$25 savings over gpt-5-mini-low and the elimination
of hallucination on thin-input movies via overview removal together make
minimal the correct choice.

## Alternatives Considered

1. **kimi-k2.5 no-thinking ($111)**: Strongest overall (4.29 avg vs 3.97-4.01
   for GPT variants). Won on 19/36 movies head-to-head; never lost on
   extraction or proportionality. Rejected — the ~$49 premium over
   gpt-5-mini-minimal is not justified for a generator whose extraction zone
   fields (the quality gap) are consumed by Wave 2 LLMs that can compensate
   for telegraphic signal.

2. **gpt-5-mini with reasoning_effort=low ($86)**: Slightly stronger than
   minimal on faithfulness (4.50 vs est. 4.47) but weaker on extraction
   depth. After overview removal fixed the thin-input failure mode, low
   reasoning added cost without measurable benefit. Rejected.

3. **Model tiering (cheaper models for thin buckets)**: Thin buckets
   (ultra-thin + very-thin) represent ~$9 combined at any candidate's pricing.
   Quality risk and implementation complexity of tiering outweigh savings.
   Rejected.

4. **Prompt caching as savings lever**: Already active in the Batch API for
   GPT-5 family models — not an untapped optimization.

5. **Include TMDB overview in inputs**: Initially included as cheap contextual
   grounding (~60 extra input tokens). At minimal reasoning effort, the model
   treated overview as source material for extraction despite the "context only"
   qualifier. Removed after evaluation confirmed hallucination improvement on
   thin-input movies; existing inputs (title, genres, reception_summary,
   attributes, reviews) are sufficient.

## Consequences

- `generate_reception()` signature no longer accepts `provider`, `model`, or
  `**kwargs`. Callers (notebooks, unit tests) that pass these will fail.
- Cost for reception generation at 50% batch pricing: ~$62 for ~109K movies.
- TMDB `overview` field is absent from reception generation inputs; any future
  prompt changes should not reintroduce it without evaluating thin-input
  behavior at the selected reasoning level.
- ADR-027 (real-time first design) is partially superseded for reception:
  the generator no longer accepts provider/model overrides.

## References

- ADR-027 (generator real-time first design) — original provider/model param convention
- ADR-039 (plot_events model selection) — established the locked module-constant pattern
- ADR-042 (reception schema redesign) — schema evaluated alongside model selection
- `movie_ingestion/metadata_generation/generators/reception.py`
- `movie_ingestion/metadata_generation/prompts/reception.py`
- `movie_ingestion/metadata_generation/evaluation_data/analyze_evaluations.py`
