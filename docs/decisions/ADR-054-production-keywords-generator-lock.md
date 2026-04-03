# [054] — Production Keywords: Generator Lock, 7-Category Prompt, and Evaluation Results

## Status
Active

## Context

Production keywords is the final Wave 2 generator to be finalized. After a prompt
rewrite (expanding from 4 to 6 then 7 categories) and a 48-movie, 4-candidate
evaluation, the generator was ready to lock following the Wave 2 finalization
pattern (ADR-045).

Evaluation results across 48 movies (192 candidate evaluations):
- **r2-5-mini-low** (OpenAI gpt-5-mini, reasoning_effort=low, no justifications):
  perfect precision (5.00), near-perfect recall (4.92), zero hard failures.
  Selected as production winner.
- Justifications variant added no measurable quality gain over the base prompt,
  unlike other Wave 2 types where justifications showed discipline improvement.
  Cost savings justify using the base schema.
- Gemini candidates failed generation for at least one movie (no `terms` payload),
  confirming OpenAI as the reliable provider.

The generator was the last of the 8 to remain unlocked (still accepting
`provider`/`model`/`**kwargs` as call params at the time of evaluation).

## Decision

**Lock production_keywords generator following ADR-045 pattern.** Remove
`provider`, `model`, and `**kwargs` parameters from `generate_production_keywords()`.
Hardcode `LLMProvider.OPENAI`, `"gpt-5-mini"`, `reasoning_effort="low"` as
module-level constants (`_PROVIDER`, `_MODEL`, `_KWARGS`).

**Use `ProductionKeywordsOutput` (base, no justifications) as production schema.**
Unlike most other Wave 2 generators where the justifications schema is the production
schema (per ADR-045), evaluation showed no benefit from justifications for this
classification task. The model selects verbatim from the input list, so chain-of-thought
does not improve output discipline here.

**Expand prompt from 6 to 7 categories by adding production era.** Evaluation
analysis revealed that decade/era keywords (e.g., "1960s", "silent film", "2020s anime")
are legitimate production context. Without an explicit category, the model handled
these inconsistently. Added category 7 with a NOTE explaining the boundary between
production era (when the movie was made) and in-universe time periods (when the story
is set). Also added "in-universe time periods" to the WHAT DOES NOT COUNT list.

**Retain two prompt variants** (`SYSTEM_PROMPT`, `SYSTEM_PROMPT_WITH_JUSTIFICATIONS`)
for backward compatibility with the evaluation notebook's `PlaygroundCandidate` path.
The production path uses only `SYSTEM_PROMPT`.

## Alternatives Considered

1. **Use `ProductionKeywordsWithJustificationsOutput` as production schema** (per
   ADR-045 default): Evaluation showed no uplift for this classification task.
   The model picks from a provided list — it does not compose text — so a reasoning
   field does not improve selection discipline. Cost savings favor the base schema.

2. **Keep generator unlocked for re-evaluation**: Rejected. All other 7 generators
   are locked; keeping production_keywords unlocked with no active evaluation ongoing
   creates configuration drift risk. ADR-045 requires locking after evaluation completes.

3. **Exclude production era from the prompt**: Rejected. Evaluation data showed
   inconsistent handling of decade keywords without explicit guidance. Adding the
   category and its boundary note materially reduced model disagreement.

## Consequences

- `generate_production_keywords()` no longer accepts `provider`/`model`/`**kwargs`.
  Any caller passing these params will break.
- All 8 Wave 2 generators are now locked. The batch pipeline has no unlocked generators.
- Production schema is `ProductionKeywordsOutput` (base), not the justifications variant.
  This is the only Wave 2 generator where the base schema is the production schema (per
  ADR-045's exception for types where justifications provide no measurable benefit).
- The evaluation notebook's playground cell uses its own `PlaygroundCandidate`-based
  call path and is unaffected by the generator lock.
- Production era keywords (e.g., "1960s", "silent film") are now explicitly in-scope
  and will be consistently selected across the corpus.

## References

- ADR-045 (Wave 2 finalization pattern — defines the locking convention)
- ADR-025 (schema design — base vs justifications variants)
- `movie_ingestion/metadata_generation/generators/production_keywords.py`
- `movie_ingestion/metadata_generation/prompts/production_keywords.py`
- `movie_ingestion/metadata_generation/schemas.py`
- `ingestion_data/production_keywords_eval_guide.md`
