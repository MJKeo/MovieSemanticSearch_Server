# [041] — Batch Pipeline: Per-Type Architecture with Autopilot

## Status
Active

## Context

ADR-024 designed Stage 6 as a two-wave Batch API workflow with SQLite state
in `metadata_batches` and `metadata_results` tables, and a CLI of
`submit --wave 1/2 → status → process`. Implementation of the first metadata
type (plot_events) revealed several problems with this design:

1. **Wave coupling introduced unnecessary complexity.** The wave concept
   requires the pipeline to know which types belong to Wave 1 vs Wave 2
   and to auto-submit Wave 2 from the `process` command. But the
   dependency is operator-managed anyway — the operator must wait for
   Wave 1 results before Wave 2 can use `plot_synopsis`. Modeling waves
   in the CLI added code complexity without simplifying operations.

2. **`metadata_batches` / `metadata_results` tables were over-engineered.**
   The design stored one row per tmdb_id × generation_type in
   `metadata_results`, with `plot_synopsis` and `review_insights_brief`
   as scalar columns. At ~112K movies × 8 types, this is ~900K rows.
   The actual access pattern is: "store result for one type, read it for
   the next type." A wide-table design (one row per movie, one column per
   type) is simpler and more cache-friendly.

3. **`wave1_runner.py` and `state.py` were evaluation artifacts.** The
   direct runner was built for notebook evaluation runs, not production.
   With evaluation complete and the batch pipeline replacing it, both
   files were obsolete.

4. **`batch_manager.py` naming was ambiguous.** Other metadata types may
   use non-OpenAI providers in the future; the file was renamed to
   `openai_batch_manager.py` to scope it explicitly.

5. **OpenAI's 5M enqueued token limit** was hit during initial test runs.
   The original fixed 10K batch size needed to become configurable
   (`--batch-size` flag), and the pipeline needed duplicate-submission
   guards and batch-level error capture.

6. **48h batch latency was unacceptable for initial validation.** Running
   a full batch just to confirm the pipeline works requires waiting up to
   24h. A hybrid approach — running live API calls between batch polls —
   would both validate the pipeline quickly and maximize throughput.

## Decision

Restructure the batch pipeline around three design changes:

**1. Per-type architecture instead of wave-based.** No wave concept in
the CLI. Each metadata type has its own `submit`, `status`, `process`
cycle. The operator runs types in dependency order. CLI commands:
`eligibility`, `submit`, `status`, `process`, `autopilot`.

**2. Wide-table schema in tracker.db.** Replace `metadata_batches` and
`metadata_results` with three tables:
- `metadata_batch_ids` — one row per movie, 8 batch_id columns. Populated
  on `submit`, cleared on `process`. No auto-seeded rows.
- `generated_metadata` — one row per movie, 8 JSON result columns +
  8 `eligible_for_<type>` integer columns.
- `generation_failures` — per-request failure log (tmdb_id,
  metadata_type, error_message). Separate from `generated_metadata`
  so "attempted and failed" is distinguishable from "not yet attempted."

**3. `autopilot` command with live generation interleaving.** Runs a
continuous loop: (1) 25 live `generate_plot_events()` calls at 5
concurrent, (2) check batch statuses, (3) process completed/failed
batches, (4) submit new batches to fill freed slots. Terminates when
no batches remain and no eligible movies exist. Live generation
(~30-60s for 25 requests) serves as the natural poll interval;
falls back to 60s sleep only when no live-eligible movies remain.

Additional hardening: duplicate-submission guard (LEFT JOIN against
`metadata_batch_ids` in `_get_pending_tmdb_ids()`), batch-level error
capture in `BatchStatus.errors`, failed-batch id clearing for
resubmission, configurable `--batch-size` and `--max-batches` flags.

Custom ID format: `{metadata_type}_{tmdb_id}` (e.g. `plot_events_12345`).
`rsplit('_', 1)` parsing is safe because tmdb_id is always a pure integer.
`build_custom_id(MetadataType, tmdb_id)` and `parse_custom_id(str)` in
`inputs.py` are the single authoritative implementations.

## Alternatives Considered

1. **Keep wave-based design with bug fixes**: Fixing the token limit and
   adding configurable batch size does not address the over-engineering
   of `metadata_batches`/`metadata_results` or the CLI coupling. Rejected
   in favor of a clean redesign.

2. **Keep `metadata_results` narrow table**: The wide-table design loses
   one capability: storing intermediate `plot_synopsis` and
   `review_insights_brief` as queryable scalar columns for Wave 2 request
   building (ADR-024, ADR-025). This capability is deferred — Wave 2
   request building will need to parse `generated_metadata.plot_events`
   JSON column to extract `plot_summary`. Acceptable because Wave 2
   has not been implemented yet.

3. **Pure batch (no live generation interleaving)**: Requires waiting
   up to 24h for the first batch to complete before any results are
   visible. Live interleaving validates the pipeline within minutes and
   provides continuous progress feedback.

4. **Separate process for live vs. batch generation**: Would require
   managing two running processes and coordinating their DB writes.
   Autopilot as a single loop is simpler to operate.

## Consequences

- `metadata_batches`, `metadata_results`, and `wave1_results` tables
  are dropped; migrations in `tracker.py` handle existing tracker DBs.
- `wave1_runner.py`, `state.py`, and `evaluations/` are deleted.
  `load_movie_input_data()` from `evaluations/shared.py` is moved to
  `inputs.py`.
- `batch_manager.py` → `openai_batch_manager.py`. Any import of the
  old name will fail.
- The `custom_id` format changes from `{tmdb_id}-{generation_type}`
  (ADR-024) to `{metadata_type}_{tmdb_id}`. Old-format custom IDs
  will fail `parse_custom_id()` with ValueError.
- `generate_plot_events()` is callable from the autopilot loop directly
  — its locked provider/model (ADR-039) means no per-call configuration
  is needed.

## References

- ADR-024 (original Batch API architecture) — superseded for implementation details
- ADR-025 (schema design) — `review_insights_brief` as scalar: deferred to Wave 2 impl
- ADR-027 (real-time generator contract) — live generation reused in autopilot
- ADR-039 (gpt-5-mini model selection) — enables no-arg live generation in autopilot
- `movie_ingestion/metadata_generation/batch_generation/run.py`
- `movie_ingestion/metadata_generation/batch_generation/openai_batch_manager.py`
- `movie_ingestion/tracker.py`
