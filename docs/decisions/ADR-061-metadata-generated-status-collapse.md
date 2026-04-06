# [061] — Collapse phase1_complete / phase2_complete into metadata_generated

## Status
Active

## Context

The metadata generation stage (Stage 6) previously used two intermediate
statuses to track progress through two generation waves:
- `phase1_complete`: plot_events + reception generated
- `phase2_complete`: all 8 types generated

The two-phase structure reflected an earlier pipeline design where Wave 1
(plot_events, reception) and Wave 2 (the remaining 6 types) were distinct
architectural steps with separate batch submission workflows. As the pipeline
matured into a unified multi-type batch runner (`batch_generation/run.py`),
the phase distinction lost its operational meaning. The only externally
meaningful status is "all 8 types are done."

109,277 movies were at `imdb_quality_passed` with at least one non-NULL
metadata column — an inconsistency caused by the migration.

## Decision

- Remove `PHASE1_COMPLETE` and `PHASE2_COMPLETE` from the `MovieStatus` enum.
- Add `METADATA_GENERATED = "metadata_generated"` as the single post-generation
  status.
- Add a migration in `init_db()` that collapses any lingering `phase1_complete`
  or `phase2_complete` rows to `metadata_generated` on startup.
- Update `ELIGIBLE_STATUSES` in `estimate_generation_cost.py` and any other
  scripts that referenced the old statuses.
- The SQL migration to advance 109,277 movies was run manually once.

## Alternatives Considered

1. **Keep phase1/phase2 for observability**: The batch runner already logs
   per-type completion; the status column is not the right granularity for
   per-type tracking. Per-type state is tracked in `generated_metadata` columns.

2. **Add per-type statuses (8 statuses, one per type)**: Over-engineered —
   the generated_metadata columns already track per-type completion with
   richer state (NULL / eligible=1 / eligible=0 / JSON result).

## Consequences

- `MovieStatus` enum is simpler: no PHASE1_COMPLETE or PHASE2_COMPLETE.
- Any code that branched on these statuses needs updating (estimate_generation_cost
  was the only non-test reference; tests updated separately).
- `init_db()` migration is idempotent — safe to re-run on already-migrated DBs.
- The `ingestion` pipeline status chain is now: ... → `imdb_quality_passed` →
  `metadata_generated` → `embedded` → `ingested`.

## References

- `movie_ingestion/tracker.py`
- `movie_ingestion/metadata_generation/helper_scripts/estimate_generation_cost.py`
- ADR-041 (batch pipeline per-type architecture — the change that made phases obsolete)
- ADR-044 (multi-type batch pipeline)
