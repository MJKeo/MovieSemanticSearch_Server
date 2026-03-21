# [034] — Reference-Free Evaluation with Opus 4.6 Judge

## Status
Active

## Context

ADR-028 established a two-phase evaluation pipeline: Phase 0 generates
reference outputs using GPT-5.4 via WHAM; Phase 1 scores candidates
against those references. Research (Yamauchi et al., arXiv:2506.13639)
showed rubric quality matters ~2.7× more than reference presence for
human judgment alignment on subjective metadata tasks. Additionally:

- References anchored judge scores, creating bias toward reference
  style rather than absolute quality against a rubric.
- WHAM backend required OAuth token acquisition and specific API
  call patterns (see ADR-030), adding maintenance overhead.
- The judge seeing the generation prompt instructions ("THE GENERATION
  PROMPT instructs:") rather than raw source data created ambiguity
  about what the judge was grading against.
- ADR-031 established 3-run judge averaging with staggered calls (run 1
  primes cache, runs 2-3 fire in parallel after). Statistical analysis
  showed the 3rd run contributes only 18% SE reduction versus 29% for
  run 2, making 2 runs sufficient on a 4-point discrete scale with a
  strong rubric.

## Decision

Remove reference-based evaluation entirely from the plot_events
evaluation pipeline. Switch the judge from GPT-5.4/WHAM to Claude
Opus 4.6/Anthropic with prompt caching.

**Reference removal**: `generate_reference_responses()`, the
`_CREATE_REFERENCES_TABLE` DDL, and all reference loading and passing
in `_evaluate_one()` are removed. The judge scores candidates directly
against a rubric anchored to the source data.

**Source data as judge context**: The judge sees raw movie fields
(labeled as SOURCE DATA) rather than the generation prompt's
instructions. The candidate's `build_plot_events_prompts()` output
already contains the labeled raw data fields — reused directly. Rubric
reframing: "THE GENERATION PROMPT instructs:" → "A HIGH-QUALITY OUTPUT
should:" makes quality criteria self-contained and independent of the
generation prompt contract.

**Judge model switch**: GPT-5.4/WHAM → Claude Opus 4.6/Anthropic.
Removes WHAM auth acquisition and all WHAM-specific kwargs.

**Prompt caching**: Added `cache_control` kwarg to
`generate_anthropic_response_async()`. When True, wraps system, user,
and tool content in cache_control blocks. Subsequent reads benefit from
Anthropic's 90% prompt cache discount.

**Reduced judge runs (3 → 2)**: Statistical analysis showed the 3rd run
provides only 18% SE reduction (versus 29% for run 2). 2 runs maintain
ranking stability on a 4-point discrete scale with a strong rubric.
Default `judge_runs` changed from 3 to 2.

**Sequential judge execution**: Replaced the stagger-then-parallel
pattern (run 1 alone, then runs 2-3 in parallel) with a simple
sequential loop. Both runs benefit from prompt caching (run 1 primes,
run 2 reads cached context). Simpler and correct.

**429 rate-limit retry**: `generate_anthropic_response_async` propagates
`anthropic.RateLimitError` (caught before the catch-all ValueError
wrapper). Per-call retry loop in the judge execution sleeps 30s and
retries indefinitely on 429.

**Thinking disabled explicitly**: `"thinking": {"type": "disabled"}` is
passed in judge kwargs. Anthropic API defaults to disabled when omitted,
but explicit is clearer and prevents accidental activation.

**Caveman-speak reasoning**: Judge reasoning fields are constrained to
one sentence, max 30 words, caveman-speak (no articles, no filler).
Added `Field(description=...)` constraints and a REASONING FORMAT
section in the system prompt. Compresses output tokens significantly
from ~2K unconstrained.

## Alternatives Considered

1. **Keep references with a different reference model**: References
   added anchoring bias for subjective metadata tasks regardless of
   quality. The Yamauchi et al. finding justified removal rather than
   swapping the reference generator.

2. **Keep GPT-5.4/WHAM as judge**: WHAM requires OAuth lifecycle
   management and has specific parameter constraints (see ADR-030).
   Opus 4.6 with native Anthropic SDK is simpler, and prompt caching
   makes multi-run evaluation cheaper.

3. **Keep 3 judge runs**: 18% SE reduction for run 3 does not justify
   the cost at Opus 4.6 pricing. 2 runs maintain ranking stability.

4. **Keep stagger-then-parallel pattern**: With sequential execution
   and 2 runs (not 3), the stagger pattern adds complexity without
   benefit. Sequential is simpler and achieves the same cache priming.

## Consequences

- `unit_tests/test_eval_plot_events.py` breaks on import (references
  removed function `generate_reference_responses`). Must be updated in
  a separate testing phase.
- Evaluation cost per pipeline run is reduced by ~33% (2 Opus runs
  vs 3 WHAM runs at lower per-call cost).
- ADR-028 Phase 0 description is now superseded for plot_events; other
  metadata types that haven't been evaluated yet should adopt this
  approach when their evaluation pipelines are built.
- The `cache_control` kwarg on `generate_anthropic_response_async` is
  available for any future Anthropic callers that want prompt caching.

## References

- ADR-028 (evaluation pipeline design) — original two-phase design
- ADR-030 (WHAM backend) — context for why WHAM was used and its complexity
- ADR-031 (multi-run averaging) — original 3-run decision
- Yamauchi et al., arXiv:2506.13639 — rubric vs reference importance
- `movie_ingestion/metadata_generation/evaluations/plot_events.py`
- `implementation/llms/generic_methods.py`
