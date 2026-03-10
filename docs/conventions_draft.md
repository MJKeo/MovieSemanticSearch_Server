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
**Sessions observed:** 2

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

## Use orjson for bulk JSON I/O
**Observed:** Replaced stdlib `json.loads(f.read_text())` with
`orjson.loads(f.read())` (binary mode) for loading 140K IMDB JSON files.
Combined with ThreadPoolExecutor for parallel file I/O since the
workload is I/O-bound.
**Proposed convention:** When loading or writing JSON in batch
(hundreds+ files), use `orjson` with binary file handles (`rb`/`wb`).
For I/O-bound bulk loads, wrap with `ThreadPoolExecutor`. Keep stdlib
`json` for small one-off reads where the dependency isn't justified.
**Sessions observed:** 1

## Stage-prefixed quality score columns
**Observed:** When adding a second quality score (Stage 5), the
generic `quality_score` column was ambiguous. Renamed to
`stage_3_quality_score` and added `stage_5_quality_score`.
**Proposed convention:** Quality score columns in movie_progress
must be prefixed with their pipeline stage number
(`stage_N_quality_score`) to avoid ambiguity as more filtering
stages are added.
**Sessions observed:** 1

## Filter in SQL, not in Python
**Observed:** Analysis script was loading all IMDB JSONs then filtering
in Python. User corrected: use a JOIN in the SQL query to filter at the
DB level, then derive the file-load set from the query results. Avoids
round-tripping large ID sets through Python when SQL can do it directly.
**Proposed convention:** When loading data scoped to a subset of movies,
filter via SQL JOIN/WHERE rather than loading all data and filtering in
Python. If a second data source needs the same scope, derive its filter
set from the first query's results rather than issuing a separate
status query.
**Sessions observed:** 1

## Share scoring utilities across pipeline stages
**Observed:** Stage 3 and Stage 5 both needed vote_count scoring (with
age adjustments), popularity scoring, and provider key decoding. User
chose to extract shared functions to `scoring_utils.py` with parametric
caps (VoteCountSource enum) rather than duplicating or coupling stages.
**Proposed convention:** When multiple pipeline stages share scoring
logic, extract to `movie_ingestion/scoring_utils.py` with parametric
inputs for stage-specific differences. Each stage imports from the
shared module but keeps its own weight table and signal definitions.
**Sessions observed:** 1

## Single-pass accumulation for multi-metric analysis
**Observed:** Analysis script iterated the full 140K movie list ~92
times (once per field per section). Refactored to a single pass that
accumulates all field stats and composite values into dataclass
accumulators, then print functions read from the pre-computed results.
**Proposed convention:** When computing multiple independent metrics
over the same dataset, accumulate all results in a single iteration
rather than iterating per-metric. Use a stats accumulator dataclass
to hold intermediate results.
**Sessions observed:** 1
