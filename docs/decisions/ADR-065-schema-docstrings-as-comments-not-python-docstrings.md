# [065] — Pydantic Schema Class Documentation as # Comments, Not Docstrings

## Status
Active

## Context
`schemas/metadata.py` and `schemas/enums.py` contain Pydantic models and
enums used as `response_format` in OpenAI Batch API and structured output
calls. Python class docstrings propagate into JSON schema `description`
fields via Pydantic's `model_json_schema()`, which OpenAI's
`to_strict_json_schema()` then sends verbatim to the LLM on every API call.

Before this change, every generation call was leaking pipeline internals to
the LLM: model configuration choices ("gpt-5-mini, reasoning_effort=low"),
embedding decisions ("these terms are joined with commas and embedded"),
planned database schema details ("GIN index on movie_card"), and
file path references ("see ADR-054"). 17 class docstrings across the two
files were affected.

## Decision
All class-level docstrings in `schemas/metadata.py` and the `SourceMaterialType`
docstring in `schemas/enums.py` were converted to `#` comment blocks placed
above the class definition. Design context, ADR references, and behavioral
notes are fully preserved — they simply no longer appear in the JSON schema
payload. Field-level `Field(description=...)` annotations are intentional and
retained (they guide LLM output structure).

## Alternatives Considered
**Leave docstrings and accept leakage.** Rejected. Sending pipeline internals
to the LLM is unnecessary noise that could influence LLM behavior in
unpredictable ways, and leaking internal architecture is a bad practice
regardless of observed impact.

**Remove documentation entirely.** Rejected. The design context in these
comments is valuable for maintainability. The fix only changes the Python
construct used, not the content.

**Use `model_config = ConfigDict(json_schema_extra=...)` to suppress schema
descriptions.** Technically possible but complex to apply consistently across
all subclasses and sub-models. The `#` comment approach is zero-overhead and
immediately obvious to future contributors.

## Consequences
- No pipeline internals are sent to the LLM in any generation call.
- Developer-facing documentation is preserved and immediately visible in the
  source file, above the class definition.
- Future contributors must write class-level documentation as `#` comments
  in this module, not Python docstrings. This is a non-standard convention
  that should be noted when onboarding.
- Field-level `Field(description=...)` is unaffected and continues to guide
  LLM structured output.

## References
- `schemas/metadata.py` — all 9 `EmbeddableOutput` subclasses,
  `EmbeddableOutput` itself, `TagEvidence`, and `ConceptTagsOutput`
- `schemas/enums.py` — `SourceMaterialType`
- `docs/modules/schemas.md` — Gotchas section documents this invariant
