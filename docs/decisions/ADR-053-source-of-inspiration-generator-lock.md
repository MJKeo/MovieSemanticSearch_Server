# [053] — Source of Inspiration: Generator Lock to Production Defaults

## Status
Active

## Context

After two schema redesign iterations (ADR-051, ADR-052) and a 75-movie,
4-candidate evaluation using the final prompt and schema, source_of_inspiration
was ready to lock following the Wave 2 generator finalization pattern (ADR-045).

Evaluation results across 300 candidate evaluations:
- **gpt-5-mini non-reasoning (SYSTEM_PROMPT)**: selected as production winner
  with the highest overall accuracy. Designated as `low` reasoning_effort,
  `low` verbosity.
- Reasoning variants (SYSTEM_PROMPT_WITH_REASONING) showed mixed results;
  the evidence-inventory redesign (evidence fields as non-gating records rather
  than decision gates) resolved the anchoring-to-abstention bug from ADR-050,
  but the base prompt achieved comparable accuracy with lower token cost.
- gpt-4o-nano candidates were removed from consideration (lower accuracy,
  higher failure rate on source_material abstention).

The generator file still had `_DEFAULT_PROVIDER` and `_DEFAULT_MODEL` constants
(unlocked pattern) plus `provider`/`model`/`**kwargs` as callable parameters,
which is the pre-lock pattern from ADR-045.

## Decision

**Lock source_of_inspiration generator following ADR-045 pattern.** Remove
`provider`, `model`, and `**kwargs` parameters from `generate_source_of_inspiration()`.
Hardcode `LLMProvider.OPENAI`, `"gpt-5-mini"`, `reasoning_effort="low"`,
`verbosity="low"` directly in the function body. This makes the production config
auditable and prevents accidental drift.

**Use `SourceOfInspirationOutput` (base, no reasoning fields) as production schema.**
Unlike other Wave 2 generators that use the WithJustificationsOutput variant
in production, source_of_inspiration uses the base schema. The evidence-inventory
reasoning variant is available as `SourceOfInspirationWithReasoningOutput` (with
alias `SourceOfInspirationWithJustificationsOutput` for test compatibility) but
is not the production path.

**Retain `_DEFAULT_PROVIDER`/`_DEFAULT_MODEL` module-level constants in the
generator** as documentation of the config, consistent with the unlocked
generator convention. These constants are used internally only, not as callable
params.

## Alternatives Considered

1. **Use SYSTEM_PROMPT_WITH_REASONING and SourceOfInspirationWithReasoningOutput
   as production schema** (per ADR-045's default preference for justifications):
   Evaluation showed the base prompt achieves comparable accuracy. The reasoning
   fields add output tokens without measurable accuracy gain at gpt-5-mini scale.
   The cost savings at 112K corpus scale justify using the base prompt.

2. **Keep generator unlocked for future re-evaluation**: Rejected. The locked
   pattern (ADR-045) exists specifically to prevent configuration drift and make
   the production state auditable. Keeping it unlocked implies evaluation is
   ongoing when it is not.

## Consequences

- `generate_source_of_inspiration()` no longer accepts `provider`/`model`/
  `**kwargs` — any notebook or test code passing these params will break.
- Production schema is `SourceOfInspirationOutput` (base), not the reasoning
  variant. The `SourceOfInspirationWithReasoningOutput` schema remains available
  for any future evaluation runs.
- `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` alias was removed from the prompt file;
  `SYSTEM_PROMPT_WITH_REASONING` is the exported name. Schema alias
  `SourceOfInspirationWithJustificationsOutput` was preserved for test
  compatibility.
- source_of_inspiration joins the 7 other locked generators; all 8 generation
  types in the batch pipeline are now locked.

## References

- ADR-045 (Wave 2 finalization pattern — defines the locking convention)
- ADR-051 (schema redesign)
- ADR-052 (boundary refinement)
- ADR-050 (evidence-inventory pattern — predecessor reasoning approach)
- `movie_ingestion/metadata_generation/generators/source_of_inspiration.py`
- `movie_ingestion/metadata_generation/prompts/source_of_inspiration.py`
- `movie_ingestion/metadata_generation/schemas.py`
