# [062] — Batch Ingestion: Parallel Postgres + Qdrant, SAVEPOINTs, and Retryable Failure Status

## Status
Active

## Context

Stage 8 (database ingestion) previously had no CLI entry point. To ingest the
~109K eligible movies, either a notebook or a manual loop was required. Postgres
and Qdrant writes were serial. There was no structured way to record which movies
failed, why, or to retry them automatically on subsequent runs.

Key requirements:
- Ingest at scale without blocking one database on the other
- Isolate per-movie errors so one bad movie doesn't abort a batch
- Distinguish retryable failures (transient errors) from permanent ones (missing data)
- Make re-runs safe: already-ingested movies should be skipped or safely re-upserted

## Decision

**Parallel ingestion**: Postgres and Qdrant run concurrently per super-batch via
`asyncio.gather`. True parallelism: the Postgres path yields on async DB I/O;
the Qdrant path yields on OpenAI embedding calls and uses `asyncio.to_thread`
for sync Qdrant upserts.

**Postgres batching with nested SAVEPOINTs**: Each sub-batch runs as one
transaction. Per-movie, an outer SAVEPOINT wraps both `ingest_movie_card` and
`ingest_lexical_data` for atomicity. Inner SAVEPOINTs around each step let the
transaction continue after a step failure (clean state for the next step).
This collects separate error messages per step while keeping the batch
transaction open.

**`INGESTION_FAILED` status** in `MovieStatus`: Retryable (not terminal like
`FILTERED_OUT`). Set when any ingestion step fails. `_get_eligible_tmdb_ids`
fetches both `metadata_generated` and `ingestion_failed` movies, so failed
movies are retried automatically on the next run.

**`MissingRequiredAttributeError`** for unrecoverable failures: Movies missing
required attributes (ID, title, release date, duration, maturity rating) are
routed to `filtered_out` (terminal) via `batch_log_filter()`, not to
`ingestion_failed`. Retrying cannot fix missing data.

**`BatchIngestionResult` dataclass**: Structured return type for both batch
functions, carrying `succeeded_ids`, `failed_ids`, `filtered_ids`, and `errors`
(`IngestionError` dataclass per failure). Replaces the previous 3-tuple return.

**Intersection-based status tracking**: Only movies that succeed in BOTH Postgres
and Qdrant are marked `ingested`. Idempotent upserts make re-runs safe.

**Stale failure cleanup**: `_mark_ingested` clears old `ingestion_failures` rows
when movies succeed on retry, keeping the table current.

**`log_ingestion_failures()` in tracker.py** mirrors the `batch_log_filter`
pattern: bulk INSERT into `ingestion_failures` + bulk UPDATE of `movie_progress`
to `ingestion_failed`, does NOT commit (caller responsible).

## Alternatives Considered

1. **Serial Postgres → Qdrant**: Simpler but wastes time — Qdrant path (dominated
   by OpenAI embedding latency) and Postgres path (dominated by async DB I/O)
   have no data dependency and overlap naturally.

2. **Single savepoint per movie (no inner savepoints)**: Would abort the entire
   outer transaction on the first step failure, preventing lexical data from being
   attempted if movie_card fails. Inner savepoints allow collecting both step
   errors independently.

3. **No retryable status — log failures only**: Without `ingestion_failed` status,
   the next run would re-fetch all `metadata_generated` movies regardless of
   whether they were previously attempted. Retryable status makes eligible set
   explicit and avoids re-attempting permanent failures (which are `filtered_out`).

4. **Per-movie SQLite UPDATE in `_mark_ingested`**: `executemany` with N
   individual UPDATEs hits `SQLITE_MAX_VARIABLE_NUMBER`. `json_each()` with a
   single JSON array parameter is both safer and faster.

## Consequences

- Ingestion is now a CLI-invocable, resumable, observable operation.
- `BatchIngestionResult` is the canonical return type for batch ingestion; code
  that previously unpacked 3-tuples must be updated.
- `ingest_movies_to_qdrant_batched` now returns `BatchIngestionResult` instead
  of a count dict; notebook callers are affected.
- Materialized views (`title_token_doc_frequency`, `movie_popularity_scores`)
  are refreshed automatically after each ingestion run.
- The `ingestion_failures` table provides a structured audit trail for diagnosing
  per-movie failures by step label.

## References

- `movie_ingestion/final_ingestion/ingest_movie.py`
- `movie_ingestion/tracker.py`
- ADR-006 (SQLite tracker backbone — the durability model this failure table joins)
- ADR-014 (SQLite corruption prevention — durability context)
- `docs/modules/ingestion.md`
