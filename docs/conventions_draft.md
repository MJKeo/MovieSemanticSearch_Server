# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /promote-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Separate async I/O from DB writes in pipeline stages
**Observed:** Stages 2 and 4 were refactored to have async tasks
return results (NamedTuples) instead of writing to SQLite inline.
All DB writes happen in bulk after `asyncio.gather()` completes.
**Proposed convention:** Pipeline stages that use async HTTP must
not pass the DB connection into async tasks. Async tasks return
results; the orchestrator does bulk DB writes after gather.
**Sessions observed:** 1

## Use executemany for batch DB operations
**Observed:** Replaced per-row `db.execute()` calls with
`db.executemany()` for both INSERT and UPDATE operations across
pipeline stages. Added `batch_log_filter()` alongside existing
single-row `log_filter()`.
**Proposed convention:** When writing multiple rows in a pipeline
stage, always use `executemany` rather than looping `execute`.
Provide batch variants of helper functions (e.g., `batch_log_filter`)
for use in async batch orchestrators.
**Sessions observed:** 1

## SQLite safety pragmas
**Observed:** DB corruption traced partly to missing
`PRAGMA synchronous=FULL` with WAL mode. Added to `init_db()`.
**Proposed convention:** `init_db()` must always set both
`PRAGMA journal_mode=WAL` and `PRAGMA synchronous=FULL`.
Never remove or weaken the synchronous pragma.
**Sessions observed:** 1

## Fail-fast timeouts with rotating proxies
**Observed:** During IMDB scraping proxy tuning, reducing request
timeout from 30s to 2s was the single biggest speed improvement.
With per-request IP rotation, a slow response means a flagged IP —
waiting longer can't help.
**Proposed convention:** When using rotating proxy services, set
aggressive request timeouts (2-5s) rather than generous ones.
Pair with higher retry counts (5+) and short retry backoffs
(0.3-0.8s) since each retry gets a fresh IP.
**Sessions observed:** 1
