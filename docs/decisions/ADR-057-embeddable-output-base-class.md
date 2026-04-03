# [057] — EmbeddableOutput: Explicit embedding_text() Contract Replaces __str__()

## Status
Active

## Context

All 8 `*Output` metadata schemas used `__str__()` to produce the text
that gets embedded into Qdrant. This convention was implicit — there was
no type-level enforcement that a schema actually implemented it, callers
used `str(output)` without IDE guidance, and `__str__()` is conventionally
for human-readable display rather than machine-critical embedding input.
Critically, the `__str__()` methods did not consistently apply
`normalize_string()` — some applied it per-term, others called `.lower()`,
and a few had no normalization at all.

## Decision

Add `EmbeddableOutput(BaseModel)` as an abstract base class for all 8
`*Output` schemas. `EmbeddableOutput` declares:

```python
@abstractmethod
def embedding_text(self) -> str:
    ...
```

All 8 schemas now subclass `EmbeddableOutput` and implement
`embedding_text()`, which applies `normalize_string()` to produce
the text for vector embedding. Legacy `__str__()` methods are retained
for backward compatibility but are no longer the canonical path.

`WithJustificationsOutput` variants (used for evaluation) must produce
identical `embedding_text()` output to their base variants — this
invariant is tested.

## Alternatives Considered

1. **Keep `__str__()`-based convention**: No type enforcement, no IDE
   support, confusingly named for an operation that is machine-critical.
   Rejected — the lack of explicitness had already caused normalization
   inconsistencies across schemas.

2. **Use a standalone function `embed_text(output) -> str`**: Would avoid
   modifying the Pydantic models but loses co-location of embedding logic
   with the schema definition. When a schema's fields change, the embedding
   function is in a different file and easy to forget to update.

3. **Protocol instead of ABC**: A `typing.Protocol` with `embedding_text()`
   would provide the same static typing without inheritance. Rejected
   because it does not enforce implementation at class definition time —
   missing `embedding_text()` would only fail at use, not at class creation.

## Consequences

- Any new `*Output` schema that subclasses `EmbeddableOutput` but does
  not implement `embedding_text()` raises `TypeError` at instantiation.
- `normalize_string()` is now applied consistently inside each
  `embedding_text()` implementation, eliminating per-site normalization
  inconsistencies.
- `str(output)` still works (backward compatible) but `output.embedding_text()`
  is the canonical path for all embedding pipeline code.
- The `WithJustificationsOutput` parity test now validates
  `embedding_text()` output, not `__str__()`.

## References

- `schemas/metadata.py`
- `docs/modules/schemas.md`
- ADR-025 (original schema design — `__str__()` convention was established there)
- ADR-032 (WithJustificationsOutput str-parity invariant — updated to use embedding_text())
