# ADR-024 — Batch API Architecture for LLM Metadata Generation (Stage 6)

## Status
Active

## Context

Stage 6 generates 7 types of LLM vector metadata per movie (~112K movies).
The prior design in `implementation/llms/vector_metadata_generation_methods.py`
used per-movie real-time async calls with a two-wave dependency structure
(Wave 1: plot_events + reception; Wave 2: 5 generations that depend on
Wave 1 outputs). Running ~112K × 8 real-time calls would require sustained
concurrent API connections, is expensive at real-time pricing (~$10/1M tokens
vs. Batch API's ~$5/1M), and has no natural checkpoint — a crash mid-run
loses progress from the current in-flight batch.

## Decision

Restructure Stage 6 as a Batch API workflow with persistent state between
steps. Two batch submissions (Wave 1 and Wave 2) with SQLite state tracking.
The pipeline is broken into discrete CLI commands:
`submit --wave 1` → `status` → `process` (auto-submits Wave 2) → `status` → `process`.

**Key structural decisions**:

1. **Generators are transport-agnostic request builders**: each file in
   `generators/` returns a `body` dict suitable for both batch JSONL and
   real-time API calls. JSONL wrapping (`{custom_id, method, url, body}`)
   is handled exclusively by `request_builder.py`.

2. **`custom_id = "{tmdb_id}-{generation_type}"`**: encodes both movie and
   generation type for result routing. Avoids a separate lookup table.

3. **SQLite state in tracker.db**: two tables — `metadata_batches` (one row
   per batch submission) and `metadata_results` (one row per tmdb_id ×
   generation_type). `plot_synopsis` and `review_insights_brief` are scalar
   columns in `metadata_results` (not buried in `result_json`) so Wave 2
   request building can SELECT them directly.

4. **`result_json` stores full LLM response**: allows re-parsing with updated
   schemas without re-running the batch.

5. **`batch_manager.py` has no movie/generation knowledge**: it only uploads
   files and manages batches. This keeps the OpenAI API boundary clean and
   independently testable.

## Alternatives Considered

1. **Real-time async calls per movie** (prior approach): Rejected. Cost is
   2× higher at real-time pricing. No natural checkpoint — crashes lose
   progress. Sustained high concurrency increases timeout/rate-limit risk.

2. **Single batch for all 8 generations per movie**: Rejected. Wave 2
   generations (plot_analysis, viewer_experience, etc.) need `plot_synopsis`
   from plot_events and `review_insights_brief` from reception. These can
   only be computed after Wave 1 results are processed.

3. **Store intermediate outputs in movie_progress table**: Rejected.
   `movie_progress` is a narrow status-tracking table (see ADR-006).
   Generation state and intermediate outputs belong in dedicated tables.

## Consequences

- Stage 6 is not a single continuous process — it requires operator
  intervention between waves (check status, then run process). The CLI
  encodes the expected workflow.
- Total wall-clock time for Stage 6 is up to 48h (24h per wave), but
  compute cost is halved vs. real-time calls.
- Per-generation retry granularity: a failed generation for one movie
  does not block others. `metadata_results.status` tracks each independently.
- The `movie_ingestion/metadata_generation/` subpackage is scaffolded
  (docstrings only) at the time of this ADR; implementation is pending.

## References

- ADR-012 (LLM generation cost optimization) — original cost analysis
- docs/modules/ingestion.md (Stage 6 section)
- movie_ingestion/metadata_generation/ — full module scaffold
- docs/llm_metadata_generation_new_flow.md — detailed generation flow design
