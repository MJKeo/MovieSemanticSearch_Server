# [031] — Multi-Run Judge Averaging to Reduce Scoring Noise

## Status
Active

## Context

LLM-as-judge scoring has inherent stochastic variance — the same judge model
evaluating the same (candidate, movie) pair with identical parameters produces
slightly different scores across calls due to sampling. A single judge call
can swing a candidate's mean score by 0.1–0.2 points on a 1–4 scale, enough
to affect the ranking between close candidates.

The evaluation test set is 70 movies across ~7 active candidates, so the
per-run judge call count is ~490. At 3 runs, this becomes ~1,470 judge calls.
Cost must be weighed against noise reduction benefit.

## Decision

Run the judge `judge_runs=3` times per (candidate, movie) pair and average the
scores. The `judge_runs` parameter is configurable on `run_evaluation()`.

**Parallel judge calls**: The `judge_runs` coroutines fire concurrently via
`asyncio.gather` within a single semaphore slot, so the wall-clock cost of
3 runs is close to 1 run (limited by the slowest call, not summed latency).

**Averaged scores stored as REAL** (not INTEGER) in `plot_events_evaluations`.
The `judge_runs` column records how many runs contributed to each row, so
historical rows from before this change remain interpretable.

**Fail-all on any error**: If any of the `judge_runs` calls fails, the entire
evaluation for that (candidate, movie) pair is aborted (no partial averages
stored). Idempotent retry handles recovery on the next pipeline run.

**Reasoning concatenated** with `--- Run N ---` delimiters so the full
per-run chain-of-thought is preserved for debugging score variance.

**Schema migration**: `judge_runs INTEGER` column added via idempotent
`ALTER TABLE ... ADD COLUMN` (catches `OperationalError` if it already exists)
so existing `eval.db` instances are upgraded in place.

## Alternatives Considered

1. **Single judge run**: Simpler and cheaper, but score variance may cause
   ranking noise between close candidates. Rejected given the cost of
   making the wrong production model choice.

2. **5 or more runs**: Better noise reduction but linearly higher judge cost
   with diminishing returns beyond 3. 3 runs gives a 40% noise reduction
   (1/sqrt(3) vs 1/sqrt(1)) at manageable cost.

3. **Sequential runs with early exit on agreement**: More complex logic with
   no clear threshold for "sufficient agreement." Parallel gather is simpler
   and faster.

## Consequences

- Effective judge API spend is 3× per (candidate, movie) pair, partially
  offset by the fact that judge calls are cheap relative to generation calls.
- Score columns in `plot_events_evaluations` are REAL — SQL queries that
  previously assumed INTEGER must be updated if used outside `analyze_results.py`.
- The `judge_runs` column in each row makes the averaging factor transparent
  and enables future analysis of per-run variance.

## References

- ADR-028 (evaluation pipeline) — overall evaluation design
- `movie_ingestion/metadata_generation/evaluations/plot_events.py` — `run_evaluation()`
