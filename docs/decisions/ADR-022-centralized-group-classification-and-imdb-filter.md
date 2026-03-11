# ADR-022 — Centralize Group Classification in scoring_utils and Implement imdb_filter.py

## Status
Active

## Context

The PROVIDERS/NEW/OLD movie bucketing logic (classify by watch_provider_keys
and release recency) was independently reimplemented in three separate scripts:
`plot_quality_scores.py`, `sample_threshold_candidates.py`, and
`analyze_imdb_quality.py`. Each maintained its own SQL CASE expression and
Python grouping logic. Divergence risk was high — one script already had a
slightly different theater-window boundary.

Additionally, ADR-020 introduced `imdb_quality_calculated` as an intermediate
status to separate scoring from filtering at Stage 5, but the filter script
(`imdb_filter.py`) had not yet been implemented. Threshold values were known
from survival-curve analysis but only referenced informally in comments.

## Decision

1. **Canonical group classification in `scoring_utils.py`**: Added `MovieGroup`
   enum, `classify_movie_group()`, `passes_imdb_quality_threshold()`, and
   `IMDB_QUALITY_THRESHOLDS` as the single source of truth for group logic.
   Also added SQL fragment constants (`HAS_PROVIDERS_SQL`, `NO_PROVIDERS_SQL`,
   `THEATER_WINDOW_SQL_PARAM`) so diagnostic scripts use identical SQL
   conditions.

2. **`imdb_filter.py` implements Stage 5 threshold filtering**: Follows the
   `tmdb_filter.py` pattern — materialises all `imdb_quality_calculated` rows
   upfront, applies per-group thresholds in Python using the shared utilities,
   calls `log_filter()` for filtered movies, then advances survivors via a
   single bulk UPDATE.

3. **Diagnostic scripts refactored** to import from `scoring_utils.py` rather
   than duplicating classification logic.

## Alternatives Considered

1. **Keep classification logic in each script, add a comment to keep them in
   sync**: Rejected. The prior divergence proved this doesn't work in practice.
   A shared module eliminates the class of bugs.

2. **Move classification to `tracker.py`**: Rejected. `tracker.py` manages
   pipeline state (status transitions, filter log). Group classification is
   a scoring/filtering concern that belongs alongside the scoring utilities.

3. **Add classification as a column to the SQLite DB**: Rejected. The group
   is derived from `watch_provider_keys` + `release_date` + today's date.
   Storing it would require re-materialising whenever the theater window
   changes, and the derivation is fast enough to compute at query time.

## Consequences

- `scoring_utils.py` is now the authoritative source for both scoring
  primitives (vote count, popularity) and filtering logic (group
  classification, thresholds). Any threshold change requires editing
  only `IMDB_QUALITY_THRESHOLDS`.
- Stage 5 filtering is now complete: run `imdb_quality_scorer.py` first,
  then `imdb_filter.py`.
- `imdb_filter.py` is idempotent — re-running skips already-processed
  movies (only processes `imdb_quality_calculated` status).

## References

- ADR-020 (imdb_quality_calculated status) — established the two-step pattern
- ADR-021 (Stage 5 scorer v4) — determined final threshold values
- docs/modules/ingestion.md (Stage 5 section and scoring_utils.py entry)
- movie_ingestion/scoring_utils.py
- movie_ingestion/imdb_quality_scoring/imdb_filter.py
