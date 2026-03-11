# Conventions Draft

Observed patterns staged for review. Remove entries you disagree
with, then run /solidify-draft-conventions to merge the rest into
docs/conventions.md.

Entries are added automatically during /safe-clear based on
patterns observed in the session.

## Flat retry delay with rotating proxies
**Observed:** Exponential backoff is counterproductive when each retry
gets a fresh proxy IP. User asked for flat ~0.25s delay since there's
no reason to wait longer on later attempts — the new IP has no memory
of previous failures.
**Proposed convention:** When retrying through a rotating proxy pool
(where each attempt uses a different IP), use flat short delays
(~0.2-0.3s) instead of exponential backoff.
**Sessions observed:** 1

## Pipeline stage/status naming must use the data source prefix
**Observed:** User corrected a status name mistake where `tmdb_quality_passed`
was proposed for a Stage 5 (IMDB) operation. User said "all names relevant
to this stage should use 'imdb' instead of 'tmdb'." Resulting names:
`imdb_quality_passed` (status), `imdb_quality_funnel` (stage).
**Proposed convention:** Pipeline status and stage names must be prefixed
with the data source they primarily operate on (tmdb_ for TMDB-sourced
stages, imdb_ for IMDB-sourced stages). This prevents confusion about
which pipeline phase a status belongs to.
**Sessions observed:** 1

## Separate scoring from filtering with distinct statuses
**Observed:** User asked to add `imdb_quality_calculated` as an intermediate
status between `imdb_scraped` and `imdb_quality_passed`, matching the existing
Stage 3 pattern (`tmdb_quality_calculated` → `tmdb_quality_passed`). Scoring
and threshold filtering are separate operations with separate statuses.
**Proposed convention:** Quality scoring stages must use two distinct statuses:
`*_calculated` (score written, no filtering applied) and `*_passed` (survived
threshold filtering). This allows scoring and filtering to be run independently
and makes the pipeline state unambiguous.
**Sessions observed:** 1
