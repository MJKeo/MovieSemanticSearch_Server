# [044] — Multi-Type Batch Pipeline via Generator Registry

## Status
Active

## Context

ADR-041 established the per-type batch pipeline architecture with
`eligibility`, `submit`, `status`, `process`, and `autopilot` commands.
The initial implementation was hardcoded to `plot_events`: request
building, result processing, and autopilot live generation all referenced
`plot_events` functions directly. With reception evaluation complete and
`generate_reception()` finalized (ADR-042, ADR-043), the pipeline needed
to support concurrent batch runs for multiple metadata types without
duplicating the orchestration logic.

The challenge was normalizing two generators with different prompt interfaces
into a single dispatch path:
- `plot_events`: `build_plot_events_prompts(movie)` returns `(user_prompt, system_prompt)` tuple
- `reception`: `build_reception_user_prompt(movie)` returns only `user_prompt`,
  with `SYSTEM_PROMPT` imported separately

## Decision

Introduce a `generator_registry.py` module with a `GeneratorConfig` dataclass
and thin adapter wrappers that normalize all generator interfaces to a common
`(user_prompt, system_prompt)` tuple contract.

**`generator_registry.py`** maps each `MetadataType` to a `GeneratorConfig`
containing: output schema, eligibility checker function, prompt builder
(normalized to tuple), live generator function, and model config (provider,
model, kwargs). `get_config(metadata_type)` is the single lookup entry point.

**CLI changes:**
- `eligibility`, `submit`, `autopilot` now require `--metadata <type>` arg,
  validated against registered types at parse time.
- `status` and `process` operate across all registered types automatically.
- `autopilot` runs live generation for the specified type, but polls and
  processes batches for all types.

**Dynamic SQL column names** in `request_builder.py` and `result_processor.py`
use `MetadataType` StrEnum values for column interpolation (e.g.,
`eligible_for_{type}`, `{type}_batch_id`). Safe because enum values are
controlled constants validated before any DB access.

**Dual schema registration**: `SCHEMA_BY_TYPE` in `result_processor.py` maps
`MetadataType` to output schemas independently of the registry, avoiding
import of generator modules (prompt builders, LLM callers) in the result
processing path. Maintenance cost accepted for the decoupling benefit.

**All print output prefixed** with `[{metadata_type}]` for clarity when
multiple types run concurrently.

## Alternatives Considered

1. **Separate run.py per metadata type**: Avoids the registry abstraction but
   duplicates all CLI boilerplate and batch orchestration. Not scalable to
   8 types.

2. **Normalize prompt interfaces in generator modules instead of adapters**:
   Would require changing `build_reception_user_prompt()` to return a tuple,
   breaking other callers (notebooks, evaluation runners). Adapter wrappers
   in the registry are a better boundary.

3. **Import generator modules in result_processor.py**: Would eliminate the
   `SCHEMA_BY_TYPE` duplication but pulls in prompt builders and LLM clients
   into the result processing path. Result processing should remain pure
   data — parse JSONL, validate schema, write DB.

4. **Wave-based reintroduction**: Rejected for the same reasons as ADR-041 —
   wave coupling adds CLI complexity without simplifying operations, and the
   operator manages type ordering anyway.

## Consequences

- `build_plot_events_requests()` and `process_plot_events_results()` removed
  entirely. Callers must use `build_requests(MetadataType.PLOT_EVENTS)` and
  the general `process_results()` path.
- Adding a new metadata type requires: registering a `GeneratorConfig` in
  `generator_registry.py` and adding a schema entry to `SCHEMA_BY_TYPE` in
  `result_processor.py`. No changes to `run.py`, `request_builder.py`, or
  `openai_batch_manager.py`.
- Existing unit tests for `run.py`, `request_builder.py`, and
  `result_processor.py` need updating for new function signatures.

## References

- ADR-041 (per-type batch pipeline architecture) — established the CLI design
- ADR-039 (plot_events model selection) — locked generator referenced by registry
- ADR-043 (reception model selection) — second locked generator in registry
- `movie_ingestion/metadata_generation/batch_generation/generator_registry.py`
- `movie_ingestion/metadata_generation/batch_generation/run.py`
- `movie_ingestion/metadata_generation/batch_generation/request_builder.py`
- `movie_ingestion/metadata_generation/batch_generation/result_processor.py`
