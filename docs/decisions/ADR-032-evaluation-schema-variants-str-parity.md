# [032] — WithJustificationsOutput Variants with __str__() Embedding Parity

## Status
Active

## Context

During model evaluation for Stage 6 generators, candidates need to be
compared across both prompt variants (no justifications vs. with justifications)
and model/provider variants. The evaluation pipeline (ADR-028) uses
`EvaluationCandidate` with explicit `system_prompt` and `response_format`
overrides to test different schema variants.

The question arose: when evaluating `WithJustificationsOutput` schemas,
should the justification fields affect the embedded text? If yes, evaluation
outputs cannot be compared directly against production outputs; if no,
the embedding quality measurement remains valid across both variants.

Additionally, `WithJustificationsOutput` schemas are used in the playground
notebook to debug and compare generation quality. If their `__str__()` differs
from the base variant, any embedding-based comparison between variants would
measure the embedding difference, not the generation quality difference.

## Decision

All `WithJustificationsOutput` variants in `metadata_generation/schemas.py`
must produce **identical embedding text** to their base `Output` counterparts
via `__str__()`. Justification/explanation fields are excluded from `__str__()`
in the same way they are in the base variant.

This applies to: `PlotAnalysisWithJustificationsOutput`,
`ViewerExperienceWithJustificationsOutput`, `WatchContextWithJustificationsOutput`,
`NarrativeTechniquesWithJustificationsOutput`,
`ProductionKeywordsWithJustificationsOutput`,
`SourceOfInspirationWithJustificationsOutput`.

The `TermsWithJustificationSection` sub-model (adds a `justification` field
to `TermsSection`) places `justification` first in field order to encourage
chain-of-thought reasoning, but excludes it from `__str__()`.

This invariant is enforced by dedicated `__str__()` parity tests in the
test suite (one per generation type).

## Alternatives Considered

1. **Include justification text in `__str__()`**: Would mean the `WithJustifications`
   variant embeds different (longer) text. Rejected — evaluation would then
   measure embedding quality differences caused by the added justification text,
   not the underlying generation quality. Comparisons between variants would
   be confounded.

2. **Separate `WithJustifications` output from `Output` entirely (no `__str__()` on
   `WithJustifications`)**: Would prevent accidental embedding. Rejected —
   inconvenient for playground use where you want to inspect the full output
   without needing to manually strip justification fields first.

3. **Only use base `Output` schemas for evaluation, never `WithJustifications`**:
   Would work but loses the ability to test whether chain-of-thought justifications
   improve generation quality. The whole point of these variants is to compare
   output with and without justifications using the same embedding baseline.

## Consequences

- Evaluation outputs from `WithJustificationsOutput` candidates can be embedded
  and compared against production `Output` candidates on equal footing.
- Any new `WithJustificationsOutput` variant added in the future must follow
  this convention — justification fields excluded from `__str__()`.
- Each new variant requires a corresponding `__str__()` parity test to guard
  against accidental regressions.

## References

- ADR-025 (metadata generation schema design) — base justification-removal decision
- ADR-028 (evaluation pipeline) — `EvaluationCandidate` with `response_format` override
- docs/modules/ingestion.md (Output schemas subsection)
- movie_ingestion/metadata_generation/schemas.py
