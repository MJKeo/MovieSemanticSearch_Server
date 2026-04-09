# [064] — Original Screenplay Represented as Empty Array, Not Enum Value

## Status
Active

## Context
The `SourceMaterialType` enum was designed to classify a movie's source
material for structured search (v2_data_needs.md item #7). An initial draft
included `ORIGINAL_SCREENPLAY` as a 11th enum value for movies with no
external source material. During implementation, the question arose: should
the LLM explicitly assign `ORIGINAL_SCREENPLAY`, or should an empty array
be used as the implicit signal?

## Decision
`ORIGINAL_SCREENPLAY` was removed from the enum. An empty `source_material_types`
list signals original screenplay — the LLM is never asked to assign it. V2
search queries for "original screenplays" filter for `source_material_type_ids
= EMPTY` (or equivalent).

## Alternatives Considered
**Keep ORIGINAL_SCREENPLAY as an explicit enum value.** Rejected because it
requires the LLM to reason negatively — "nothing applies, so assign this" —
which adds classification confusion with zero informational gain. The LLM
already emits an empty list when no types are present; `ORIGINAL_SCREENPLAY`
would just rename that state. A missing label is always better than a wrong one.

**Use a separate boolean field `is_original_screenplay`.** Rejected as
redundant — it would always equal `source_material_types.isEmpty()`. A
derivable signal should not be stored separately.

## Consequences
- Empty `source_material_types` arrays are unambiguous and correctly handled
  by the `SourceMaterialV2Output.embedding_text()` implementation (returns `""`).
- Search queries for original screenplays must use empty-array semantics,
  not keyword matching.
- Prompt design is simpler: the LLM is told to "identify what's present" with
  a high confidence threshold, never to assign a fallback category.
- Any future enum extension follows the same positive-only identification pattern.

## References
- `schemas/enums.py` — `SourceMaterialType` (10 values)
- `schemas/metadata.py` — `SourceMaterialV2Output.embedding_text()`
- `search_improvement_planning/source_material_type_enum.md` — full boundary notes
- `search_improvement_planning/v2_data_needs.md` — item #7
