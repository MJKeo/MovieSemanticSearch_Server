# [050] — Source of Inspiration: Evidence-Inventory Prompt Pattern and Abstention Rules

## Status
Active

## Context

Source-of-inspiration generation had two failure modes identified during
evaluation:

1. **Rationalization via single justification field.** The original prompt used a
   single post-hoc `justification` field for the entire output. Smaller models
   would decide on source labels first, then write justifications that rationalized
   the choice rather than constrained it — the same mechanism that caused problems
   in `watch_context`.

2. **Over-eager inference.** Models were too willing to infer loose inspirations
   or emit non-empty labels for original screenplays when direct source evidence
   was weak or absent. The prompt framing did not sharply distinguish explicit
   evidence (claims in the input) from parametric knowledge (model-internal film
   knowledge) from loose analogy (thematic resemblance without source relationship).

Additionally, the user prompt builder omitted absent primary inputs entirely
rather than passing them as explicit absence signals, which meant smaller models
saw no signal for "I don't have this data" and defaulted to inference.

## Decision

**Adopt the watch_context evidence-inventory pattern.** Replace the single
`justification` field with separate `source_reasoning` and
`production_medium_reasoning` fields placed *before* the lists they constrain.
Field descriptions frame them as evidence inventories: list what the inputs
say, not why the output is correct.

**Explicit absence signals in user prompt.** The user prompt builder was
updated to always include `merged_keywords` and `source_material_hint`, using
`"not available"` when absent. Models now see explicit lack-of-evidence
signals rather than omitted fields.

**Tightened abstention rules** (short operational instructions for small
models):
- Explicit input claims are authoritative — preserve them even if they seem
  imprecise.
- Parametric knowledge is only usable for very high-confidence cases (e.g., the
  film is a well-known stage musical adaptation).
- `"original screenplay"` must never appear in `sources_of_inspiration` output.
- Named categories that are loose analogies rather than source relationships
  (e.g., "inspired by real events" for fiction films with realistic themes)
  are forbidden.
- When no direct source evidence exists, emit an empty list.

**Backward-compatible aliases** were added for old prompt/schema names to
avoid a wider coordinated rename.

## Alternatives Considered

1. **Rewrite justification framing without structural change**: The field still
   appears after the output in CoT, so positional rationalization persists
   regardless of instruction text. Splitting into pre-constrained per-zone
   fields (source_reasoning before sources_of_inspiration, medium_reasoning
   before production_mediums) is a structural fix.

2. **Keep omitting absent fields from user prompt**: Omitting is cleaner but
   leaves smaller models without explicit absence signals. Testing on evaluation
   movies showed models inferred sources when `source_material_hint` was absent
   rather than treating absence as evidence of an original screenplay.

3. **Add explicit negative examples in the schema field description**: Schema
   descriptions must stay minimal per ADR-036. Negative examples belong in the
   prompt instructions, not field descriptions.

## Consequences

- `source_reasoning` and `production_medium_reasoning` fields are now part of
  the `SourceOfInspirationWithJustificationsOutput` schema, placed before the
  list fields they constrain.
- Any evaluation pipeline code importing the old justification field name
  needs updating.
- The user prompt always includes `merged_keywords` and `source_material_hint`
  — downstream prompt token counts will increase slightly for movies where
  these fields were previously omitted.

## References

- ADR-049 (watch_context evidence-inventory pattern origin)
- ADR-025 (schema design — justifications variant approach)
- ADR-036 (schema field description minimalism)
- `movie_ingestion/metadata_generation/prompts/source_of_inspiration.py`
- `movie_ingestion/metadata_generation/schemas.py`
- `movie_ingestion/metadata_generation/generators/source_of_inspiration.py`
