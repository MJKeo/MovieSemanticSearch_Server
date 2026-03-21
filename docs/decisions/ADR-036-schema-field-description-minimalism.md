# [036] — Schema Field Description Minimalism: Behavioral Instructions Belong in Prompts

## Status
Active

## Context

Pydantic structured output schemas include `Field(description=...)` text
that is passed to the model alongside the schema definition. For
`PlotEventsOutput`, these descriptions contained behavioral instructions:
"Detailed chronological, spoiler-containing plot summary preserving
character names and locations." This was intended to guide output quality.

Evaluation of synthesis-branch results showed gpt-5-mini fabricating
~1000-token plots with invented character names from single-sentence
overviews. Diagnosis identified the schema's `description` fields as
a contributing cause: they created a strong competing signal that
overrode the system prompt's softer length and fabrication guidelines.

The model resolved the tension between competing instructions by
satisfying the schema's explicit demand for detail ("Detailed
chronological, spoiler-containing...") over the prompt's constraints
("proportional to input richness," "do not fabricate"). Schema
descriptions appear closer to the schema definition itself and may
receive higher implicit weight in some models' attention patterns.

The deeper problem: behavioral instructions in schema field descriptions
cannot be branch-specific. The same schema serves both the condensation
branch (where "preserve character names" is appropriate) and the
synthesis branch (where "never invent character names" is the opposite
instruction). Schema-level instructions must be generic; branch-specific
behavior cannot be expressed there.

## Decision

Strip `PlotEventsOutput` and `MajorCharacter` field descriptions to
minimal neutral labels only:
- `plot_summary`: "Chronological plot summary."
- `setting`: "Geographic and temporal setting."
- `major_characters`: "Key characters who actively drive plot decisions."
- `MajorCharacter.name`: "Character name."
- `MajorCharacter.role`: "Character's role and function in the plot."
- `MajorCharacter.goals`: "Character's goals or motivations."

All behavioral instructions moved into the FIELDS sections of each
branch-specific system prompt, where they can be tailored per task.

**Synopsis branch FIELDS**: Encourages preserving rich detail from
the comprehensive source, permitting character name retention.

**Synthesis branch FIELDS**: Explicit anti-fabrication rules — never
invent character names (use descriptive references), never invent plot
beats, keep output proportional to input richness, with concrete
guidance for sparse input.

Schema docstrings document the minimal-description decision so future
maintainers understand the intent.

## Alternatives Considered

1. **Keep behavioral instructions in schema, make them generic**: Generic
   instructions that work for both branches would be weaker than
   branch-specific ones and would still compete with prompt-level rules.

2. **Dual schemas (one per branch)**: Would allow full behavioral
   instructions per-branch at the schema level. Rejected — schema
   duplication for what is ultimately a prompt-level concern adds
   maintenance surface. The generator already branches on prompts;
   schemas should define structure only.

3. **Keep existing descriptions, add stronger prompt language**: Tested
   implicitly — the synthesis branch had strong prompt anti-fabrication
   language and still hallucinated because schema descriptions provided
   a competing "louder" signal.

## Consequences

- Behavioral instructions are now entirely prompt-owned. When a prompt
  is updated, the schema does not need to change.
- Future schema designers should default to neutral structural labels
  in field descriptions and put all generation guidance in system prompts.
- This pattern generalizes beyond plot_events — any schema where output
  behavior varies by context should follow this design.
- Unit tests for schema string representation are unaffected (schema
  structure unchanged, only field description strings changed).

## References

- ADR-025 (metadata generation schema design) — original schema design decisions
- ADR-033 (plot events cost optimization) — two-branch design that exposed this issue
- `movie_ingestion/metadata_generation/schemas.py` — `PlotEventsOutput`, `MajorCharacter`
- `movie_ingestion/metadata_generation/prompts/plot_events.py` — FIELDS sections
