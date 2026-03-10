# TODO

Tracks actionable items discovered during development sessions.
Items here are things to address when the relevant work begins,
not urgent fixes.

## ~~Update tests for quality_score → stage_3_quality_score rename~~ DONE
Fixed in test implementation session. Both test_tracker.py and
test_tmdb_quality_scorer.py updated.

## ~~Implement combined TMDB+IMDB quality scorer~~ DONE
Implemented in this session. See DIFF_CONTEXT.md entry.

## Create Stage 5 survival curve analysis script
**Context:** After the combined scorer runs, need a script analogous
to `tmdb_quality_scoring/plot_quality_scores.py` that plots the
Stage 5 score distribution, computes Gaussian-smoothed survival
curve + derivatives, and identifies the threshold inflection point.
**When:** After the combined scorer is implemented and has run on
the full 120K movie set.
**See:** movie_ingestion/tmdb_quality_scoring/plot_quality_scores.py
(Stage 3 precedent)

## Run Stage 5 scorer on full dataset and verify score distribution
**Context:** The 8-signal scorer is implemented but hasn't been run on
the full ~120K movie set yet. Need to run `score_all()`, check the
summary stats, then run survival-curve analysis to pick a threshold.
**When:** Next session — immediate next step for Stage 5.
**See:** movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py

## ~~Update tests for scoring_utils refactor~~ DONE
Implemented test_scoring_utils.py (36 tests) and test_imdb_quality_scorer.py
(121 tests). Migrated shared function tests from test_tmdb_quality_scorer.py.

## Switch to residential proxies for database refresh pipeline
**Context:** Datacenter proxy IPs get flagged by IMDB, causing
mass timeouts and 502s. Residential IPs (real ISP addresses) are
much harder to block. DataImpulse offers residential on the same
platform — just change the proxy port/host in `build_proxy_url()`.
**When:** Building the daily update / database refresh pipeline.
**See:** memory/imdb-scraping.md for full tuning findings.
