# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Use StrEnum for domain constant sets, not bare strings
**Observed:** User corrected hardcoded `GENERATION_TYPE = "plot_events"` strings across 8 generator files, saying "We really shouldn't be hardcoding strings like 'plot_events' anywhere." Created `MetadataType(StrEnum)` as the canonical source, with all callers required to use enum members.
**Proposed convention:** When a set of related string constants is used across multiple modules (metadata types, pipeline stages, etc.), define them as a `StrEnum` in a shared location. Callers must reference the enum — never hardcode the string value. `StrEnum` keeps SQLite/JSON compatibility while preventing typos and enabling IDE support.
**Sessions observed:** 1
