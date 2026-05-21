# [083] — SOLO TraitCombineMode: single-category passthrough with orchestrator trim

## Status
Active

## Context
`TraitCombineMode` controls how a trait's multiple category scores are combined into
a single trait score. FRAMINGS (MAX) and FACETS (geometric mean) handle multi-category
traits well. But some traits are inherently single-category — e.g., a trait that maps
exclusively to EMOTIONAL_EXPERIENTIAL has only one relevant category. Running these
through multi-category combine logic wastes a handler fan-out and can produce scores
inflated by the combine operation itself.

## Decision
Add `SOLO` as a third `TraitCombineMode` value with these semantics:
- The trait should be resolved by exactly one category call.
- The orchestrator, upon seeing `combine_mode=SOLO`, trims `category_calls` to
  `[:1]` before the handler fan-out, discarding any additional categories.
- Stage 4 Phase D passes the single category score through directly as the trait score
  (no combine operation needed).

`SOLO` is produced by Step 3 when the trait maps to a single category and the
additional-category space is empty or redundant.

## Alternatives Considered
- **Always use FRAMINGS with one category**: Works but the orchestrator still creates
  a fan-out coroutine for a known-single result, and the combine code must handle
  the degenerate one-element case explicitly.
- **Remove multi-category support for single-category traits at schema level**: Would
  require Step 3 to detect and refuse multi-category output for certain trait types —
  more brittle than a explicit SOLO mode.

## Consequences
- Slightly lower latency for SOLO traits (no unnecessary fan-out overhead).
- Orchestrator must implement the `[:1]` trim before dispatching; forgetting this
  would silently run extra handler calls.
- Step 3 must set `combine_mode=SOLO` accurately; incorrect classification allows
  category_calls[1:] to be silently dropped, which is safe but wastes generation.

## References
- docs/modules/search_v2.md — TraitCombineMode section, Orchestrator section
- search_v2/step_3.py, search_v2/full_pipeline_orchestrator.py
