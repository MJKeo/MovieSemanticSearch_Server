# ADR-006: SQLite Checkpoint Tracker for Ingestion

**Status:** Active

## Context

The ingestion pipeline runs for hours/days. It must be crash-safe
and resumable. Each movie progresses through multiple stages, and
the system needs to track which movies have completed which stages.

## Decision

Use a single SQLite database (`./ingestion_data/tracker.db`) as
both the checkpoint tracker and the data store for Stage 3 quality
scoring. Three core tables:

- `movie_progress` — one row per movie, `status` column tracks
  pipeline progression
- `filter_log` — append-only audit trail of every filtered movie
  with stage, reason, and optional details JSON
- `tmdb_data` — extracted TMDB fields for quality scoring

### Key Design Choices

1. **Single file, not flat JSON**: 950K JSON files = 3-5 GB with
   filesystem overhead. SQLite = ~310-360 MB, with atomic queries
   for Stage 3 ranking.
2. **Atomic transactions**: Data insert + status update in one
   transaction. Crash between the two = neither takes effect.
3. **`log_filter()` helper**: Central function that handles both
   the filter_log INSERT and the movie_progress status update.
   Stage modules never write to these tables directly.
4. **Commit every 500 movies**: Bounds data loss on crash to ~14
   seconds of re-fetch time. Prevents WAL bloat.

## Alternatives Considered

1. **Flat JSON files per movie**: 10x larger footprint, filesystem
   degradation at 950K files, no atomic queries.
2. **Postgres for tracking**: Overkill for a local laptop pipeline.
   Adds a dependency on a running Postgres instance.
3. **Redis for checkpointing**: Not durable by default. SQLite
   provides ACID guarantees on local disk.

## Consequences

- Resumability is automatic: query `WHERE status = 'pending'`
  returns only unprocessed movies.
- The `INSERT OR REPLACE` on `tmdb_data` is idempotent — partial
  writes are harmlessly overwritten on re-fetch.
- The database fits in RAM on any modern laptop, so Stage 3
  scoring queries are effectively instant.

## References

- guides/stage_2_tmdb_fetching.md (SQLite schema section)
- guides/full_movie_fetch_pipeline_guide.md (tracker system)
