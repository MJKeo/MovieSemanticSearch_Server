# ADR-014: SQLite Corruption Prevention — Durability Settings and HTTP/DB Separation

**Status:** Active

## Context

During a long Stage 4 IMDB scraping run, the tracker database
(`ingestion_data/tracker.db`) became corrupted when the process was
killed mid-batch. Root cause analysis identified two contributing
factors: (1) missing `PRAGMA synchronous=FULL` meant WAL commits
were not fsynced before the process exited, leaving the DB in an
unrecoverable state; (2) both Stage 2 and Stage 4 wrote to SQLite
inline during async HTTP fetching, making it hard to reason about
what state the DB was in at any given moment.

The database was recovered via `.dump` + `INSERT OR IGNORE` with no
data loss (~1M rows intact), but the corruption risk needed to be
eliminated structurally.

## Decision

Two changes applied together:

1. **Add `PRAGMA synchronous=FULL`** in `init_db()`, immediately
   after enabling WAL mode. This guarantees every `db.commit()` is
   fsynced to disk before returning. Combined with WAL, this provides
   full durability with minimal performance impact at the ~500-movie
   commit cadence.

2. **Separate async HTTP from synchronous DB writes** in Stages 2
   and 4. Async tasks now return result NamedTuples; all DB writes
   happen in bulk via `executemany` after `asyncio.gather()` completes,
   followed by a single `db.commit()`.

Additionally, `title` and `year` were removed from `filter_log`
(previously they required a per-movie SELECT inside `log_filter()`).
Those fields are available via JOIN on `tmdb_data` when needed.
A new `batch_log_filter()` helper enables bulk-insert for the new
gather-then-write pattern.

## Alternatives Considered

1. **aiosqlite**: Would allow truly async DB writes but adds a
   dependency and introduces thread-pool complexity. Unnecessary
   when the simpler gather-then-write pattern achieves the same
   isolation.
2. **Semaphore-guarded per-row writes**: Preserves the inline-write
   pattern but doesn't eliminate the fundamental mixing of async and
   sync. Still vulnerable to partial-batch state on kill.
3. **Write queue with background thread**: More complex plumbing with
   no benefit over gather-then-write at this scale (~500 movies/batch).
4. **Rely on WAL alone (no FULL sync)**: WAL without FULL sync is the
   root cause of the original corruption. Not viable.

## Consequences

- Corruption on process kill is eliminated: a killed process loses
  at most one in-progress batch (~500 movies), which is re-fetched
  on restart.
- HTTP and DB write logic are cleanly separated and independently
  testable.
- `filter_log` schema is simpler; existing databases are migrated via
  `ALTER TABLE filter_log DROP COLUMN title/year` in `init_db()`.
- Stage 3 (`tmdb_filter.py`) still uses the single-row `log_filter()`
  in its synchronous loop — this is fine and requires no change.
- `PRAGMA synchronous=FULL` adds a small per-commit fsync overhead,
  negligible at the batch commit frequency used here.

## References

- docs/decisions/ADR-006-sqlite-checkpoint-tracker.md (original tracker design)
- docs/modules/ingestion.md (HTTP/DB separation pattern section)
- movie_ingestion/tracker.py (`init_db`, `batch_log_filter`)
- movie_ingestion/imdb_scraping/run.py, scraper.py
- movie_ingestion/tmdb_fetching/tmdb_fetcher.py, daily_export.py
