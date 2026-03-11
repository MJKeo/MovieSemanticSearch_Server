# ADR-020 — Add imdb_quality_calculated Intermediate Status

## Status
Active

## Context

The Stage 5 scorer (ADR-019) originally advanced movies directly from
`imdb_scraped` to `imdb_quality_passed` after scoring. This was inconsistent
with Stage 3's two-step pattern: `tmdb_quality_calculated` (scored) →
`tmdb_quality_passed` (filtered). The single-step design meant the scorer
and threshold filter could not be run independently — re-running the scorer
after a threshold analysis round would also need to re-examine already-passed
movies, mixing concerns.

## Decision

Add `imdb_quality_calculated` as an intermediate status between `imdb_scraped`
and `imdb_quality_passed`, mirroring Stage 3's `tmdb_quality_calculated` →
`tmdb_quality_passed` two-step pattern.

- **Scorer** (`imdb_quality_scorer.py`): advances `imdb_scraped` →
  `imdb_quality_calculated`.
- **Filter** (future script): advances `imdb_quality_calculated` →
  `imdb_quality_passed` (or `filtered_out`) based on per-group thresholds.

Full status progression:
```
pending → tmdb_fetched → tmdb_quality_calculated → tmdb_quality_passed →
imdb_scraped → imdb_quality_calculated → imdb_quality_passed →
phase1_complete → phase2_complete → embedded → ingested
```

The prior intermediate status `essential_data_passed` (from the v1 hard-filter
design) was renamed to `imdb_quality_passed`; any existing `essential_data_passed`
rows were migrated to `imdb_quality_calculated` at startup.

## Alternatives Considered

1. **Single-step: scorer advances directly to imdb_quality_passed**: Rejected.
   It conflates scoring and threshold-filtering, preventing re-analysis of the
   score distribution without re-running the scorer. The Stage 3 precedent
   demonstrates the value of separation.

2. **Advance directly to imdb_quality_passed, add a separate re-score flag**:
   Rejected as unnecessarily complex. An intermediate status is the cleanest
   mechanism for expressing "scored but not yet filtered."

## Consequences

- analyze_imdb_quality.py queries all three Stage 5 statuses (imdb_scraped,
  imdb_quality_calculated, imdb_quality_passed) for complete diagnostic coverage.
- The scorer is idempotent within its scope: re-running processes remaining
  `imdb_scraped` movies; `imdb_quality_calculated` movies are not re-scored.
- docs/conventions.md line 120 references the old `essential_data_passed`
  name and needs updating via /solidify-draft-conventions.

## References

- ADR-019 (Stage 5 scorer v2) — introduced the scoring model requiring this separation
- ADR-017 (Stage 3 redesign) — precedent for two-step scorer/filter pattern
- docs/modules/ingestion.md (Tracker System section) — status progression
- movie_ingestion/tracker.py — MovieStatus enum definition
