# [045] — Wave 2 Generator Finalization Pattern: Tiered Eligibility, Justifications as Production Schema, Hardcoded Config

## Status
Active

## Context

Wave 1 generators (plot_events, reception) were finalized through systematic
multi-candidate evaluation and locked at module level. As Wave 2 generators
(plot_analysis, viewer_experience, narrative_techniques, source_of_inspiration,
production_keywords, watch_context) approached production, a repeatable
finalization pattern was needed covering three recurring decisions:

1. **Eligibility design**: the original boolean OR checks (e.g., "skip if
   no plot_synopsis AND no review_insights_brief AND no genre+keywords") were
   permissive but produced low-quality output on sparse inputs because the
   threshold was too low, not because the type was ineligible.
2. **Evaluation schema vs production schema**: Wave 2 evaluation used
   `WithJustificationsOutput` variants with chain-of-thought reasoning fields.
   Whether to deploy the justifications variant or the base variant was an
   open question.
3. **Model config locking**: when to freeze provider/model/kwargs into
   module-level constants vs leaving them as caller params.

## Decision

Establish the following finalization pattern for Wave 2 generators:

**Tiered eligibility replaces boolean OR checks.** Each tier requires a
higher-quality but less-available input source. Higher tiers require less
supporting evidence; lower tiers require more. If no tier passes, skip.
The tiers are evaluated in source-quality order (LLM-condensed > raw text >
review observations), so the model always receives the best available signal.
Thresholds are set from evaluation data: the lowest-quality bucket that
produced acceptable output (holistic >= 3.5) defines the minimum tier.

**The justifications (`WithJustificationsOutput`) schema becomes the production
schema.** Evaluation across viewer_experience (50 movies, 10 candidates) and
plot_analysis (70 movies, 4 candidates) showed justification schemas improve
section discipline (+0.72 to +0.98 for viewer_experience) by forcing the model
to locate input evidence before emitting output. The chain-of-thought is not
embedded — `__str__()` parity is maintained — so there is no downstream cost.

**Hardcode production config at module level when evaluation is complete.**
Once the evaluation winner is selected, remove `provider`/`model`/`**kwargs`
params from the generator function. This prevents accidental drift, makes
the production config auditable, and eliminates per-call parameter passing.

## Alternatives Considered

1. **Keep base schema in production, justifications for evaluation only**: Rejected
   because evaluation showed justification uplift is substantial and free —
   the reasoning is discarded at embedding time.

2. **Leave model params as caller params even after finalization**: Rejected
   because it creates implicit configuration that's easy to accidentally override
   and hard to audit. Module-level constants are the authoritative production config.

3. **Single permissive eligibility threshold**: Rejected because evaluation showed
   quality degrades sharply below certain input thresholds, producing output that
   would hurt retrieval rather than help it.

## Consequences

- Each newly finalized generator removes `provider`/`model`/`**kwargs` from its
  signature — callers (notebooks, tests) must be updated.
- Unit tests that relied on passing model params for test configuration need
  refactoring to mock at a lower level or use the fixed production config.
- The `WithJustificationsOutput` schema is both the evaluation and production
  schema — there is no need to maintain separate variants after finalization.
- `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` is renamed to `SYSTEM_PROMPT` in locked
  generators' prompt files. The non-justifications prompt is removed.

## References

- ADR-025 (schema design) — base vs justifications variant design
- ADR-039 (plot_events model selection) — first locked generator
- ADR-043 (reception model selection) — second locked generator
- `movie_ingestion/metadata_generation/generators/plot_analysis.py`
- `movie_ingestion/metadata_generation/generators/viewer_experience.py`
- `movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py`
