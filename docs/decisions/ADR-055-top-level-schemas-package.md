# [055] — Extract Cross-Cutting Types into Top-Level schemas/ Package

## Status
Active

## Context

Shared Pydantic models and data classes were scattered between
`implementation/classes/` and `movie_ingestion/metadata_generation/schemas.py`.
This caused import friction: `db/` and `api/` modules couldn't import
generation-side types without pulling in `movie_ingestion/` as a dependency,
and `implementation/` was an increasingly awkward home for types that
needed to be shared across the entire codebase.

The project aims to phase out `implementation/` as a catch-all and replace
it with more purposeful top-level packages.

## Decision

Create a top-level `schemas/` package with the following modules:
- `metadata.py` — all 8 `*Output` Pydantic schemas (moved from
  `metadata_generation/schemas.py`)
- `enums.py` — `MetadataType` enum
- `data_types.py` — `MultiLineList`
- `movie_input.py` — `MovieInputData` + `load_movie_input_data()`
- `movie.py` — `Movie`, `TMDBData`, `IMDBData` (new, see ADR-056)

Types that are exclusively used by the generation pipeline's internal
orchestration remain in `movie_ingestion/metadata_generation/inputs.py`:
`build_custom_id`, `parse_custom_id`, `WAVE1_TYPES`, `WAVE2_TYPES`,
`ConsolidatedInputs`, `SkipAssessment`, `build_user_prompt`,
`Wave1Outputs`, `load_wave1_outputs`.

All 12 consumer files updated to import from new canonical locations.
No re-export shims — callers import from `schemas.*` directly.

## Alternatives Considered

1. **Keep types in `implementation/classes/`**: Would keep imports consistent
   with the existing codebase but perpetuates the `implementation/` catch-all.
   The path `implementation/classes/` implies these are implementation details
   rather than shared contracts.

2. **Move generation schemas into `db/`**: Wrong direction — `db/` is a
   runtime module; these types are data contracts used at ingest time.

3. **Create a separate `contracts/` or `models/` package**: Functionally
   equivalent to `schemas/`. Rejected in favor of `schemas/` which is the
   conventional name in FastAPI/Pydantic ecosystems and matches the
   existing file naming (`schemas.py` already existed in several modules).

## Consequences

- `db/`, `api/`, and `movie_ingestion/` can all import from `schemas.*`
  without circular dependencies.
- `metadata_generation/schemas.py` is deleted; existing tracker rows
  are unaffected (schema is purely for runtime validation).
- `implementation/` is one step closer to being fully retired.
- Any code still importing from `movie_ingestion.metadata_generation.schemas`
  will break at import time — this was the motivation for the test import
  update work that followed this change.

## References

- `docs/modules/schemas.md` (new module doc)
- `docs/modules/ingestion.md` (Stage 6 — Output schemas section)
- `docs/modules/classes.md`
- ADR-025 (original schema design decisions)
