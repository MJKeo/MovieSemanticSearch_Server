# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Decompose multi-case evaluations into per-case methods
**Observed:** User directed refactoring 8 inline skip conditions (if/else blocks inside _assess_wave1/_assess_wave2) into 8 individual _check_<type>() methods, each returning str | None. The orchestrator becomes a thin loop over individual checks.
**Proposed convention:** When a function evaluates N independent cases with distinct logic (e.g., eligibility checks, validation rules), extract each case into its own method rather than inlining all cases in a single function. Compose via a thin orchestrator.
**Sessions observed:** 1

## Consistent __str__() lowercasing on all embeddable schema models
**Observed:** Code review found 5 of 8 schema `__str__()` methods missing `.lower()` calls on terms, while the other 3 lowercased. Since these strings become embedding input text, inconsistent casing could affect embedding quality. All were fixed to lowercase uniformly.
**Proposed convention:** All Pydantic schema classes whose `__str__()` output feeds the embedding pipeline must lowercase their concatenated terms. When adding a new schema, match the lowercasing pattern of existing schemas.
**Sessions observed:** 1

## Shared exception classes over per-module custom errors
**Observed:** User directed replacing a `PlotEventsGenerationError` specific to plot_events.py with two shared exceptions (`MetadataGenerationError`, `MetadataGenerationEmptyResponseError`) in a central errors.py, parameterized by generation_type and title. Avoids proliferating N error classes as N generators are added.
**Proposed convention:** When multiple modules have identical failure modes differing only by context (e.g., which generation type failed), use a single shared exception class parameterized with context fields rather than creating per-module exception subclasses.
**Sessions observed:** 1
