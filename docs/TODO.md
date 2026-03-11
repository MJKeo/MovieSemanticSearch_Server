# TODO

Tracks actionable items discovered during development sessions.
Items here are things to address when the relevant work begins,
not urgent fixes.

## ~~Update tests for quality_score → stage_3_quality_score rename~~ DONE
Fixed in test implementation session. Both test_tracker.py and
test_tmdb_quality_scorer.py updated.

## ~~Implement combined TMDB+IMDB quality scorer~~ DONE
Implemented in this session. See DIFF_CONTEXT.md entry.

## ~~Create Stage 5 survival curve analysis script~~ DONE
Implemented as `movie_ingestion/imdb_quality_scoring/plot_quality_scores.py`
using shared `movie_ingestion/survival_curve_utils.py`. Not yet run
end-to-end (plt.show() blocks headless — see TODO below).

## ~~Guard plt.show() in survival curve utils for headless environments~~ DONE
Already implemented in `survival_curve_utils.py` lines 378-383 —
checks matplotlib backend for "agg" and skips `plt.show()` in
headless environments.

## Run Stage 5 scorer on full dataset and verify score distribution
**Context:** The 8-signal scorer is implemented but hasn't been run on
the full ~120K movie set yet. Need to run `score_all()`, check the
summary stats, then run survival-curve analysis to pick a threshold.
**When:** Next session — immediate next step for Stage 5.
**See:** movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py

## ~~Update tests for scoring_utils refactor~~ DONE
Implemented test_scoring_utils.py (36 tests) and test_imdb_quality_scorer.py
(121 tests). Migrated shared function tests from test_tmdb_quality_scorer.py.

## ~~Manually review threshold candidate samples~~ DONE
All 4 thresholds reviewed and analysis report written. See
`ingestion_data/threshold_analysis_report.md` for full findings.
Decision: fix formula first (watch_providers, runtime filter, min text volume
filter), then re-run survival curve and choose threshold from the new curve.

## Fix watch_providers signal in Stage 5 scorer
**Context:** Binary ±1.0 swing on 0.20 weight creates a ±0.40 total
swing — the single largest distortion in the formula. Cheap-to-license
junk (SyFy, Tubi) has 10-16 providers while quality films without US
streaming get -0.20. Miss Marple vs. Mothman Curse (0.43 threshold)
and One Night at McCool's vs. Bigfoot or Bust (0.54) are the starkest
examples. Fix: change from binary ±1.0 to tiered (e.g., 0 providers →
-0.5, 1-2 → 0.0, 3+ → +0.5) and/or reduce weight.
**When:** Before choosing the Stage 5 threshold.
**See:** movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py:272-296,
ingestion_data/threshold_analysis_report.md

## Add minimum runtime hard filter to Stage 5
**Context:** 8-minute cartoons, 25-minute shorts, 44-minute TV specials,
and YouTube content pass through Stage 5 scoring alongside feature films.
These fail both inclusion criteria (not a feature film, and thin data).
Runtime is in TMDB data. Suggested floor: 40-60 minutes.
**When:** Before choosing the Stage 5 threshold.
**See:** movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py,
movie_ingestion/tracker.py (runtime in tmdb_data schema)

## Add minimum text volume hard filter to Stage 5
**Context:** Movies with 0 plot keywords, 1-2 reviews, and no synopsis
pass scoring if they compensate on other signals (streaming, shallow
metadata). These will produce garbage embeddings at any threshold.
Suggested filter: total review_chars + synopsis_chars + plot_summary_chars
< 500 (or similar floor derived from what LLM generation actually needs).
**When:** Before choosing the Stage 5 threshold.
**See:** movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py,
ingestion_data/threshold_analysis_report.md

## Run survival counts at each candidate threshold
**Context:** Border analysis tells us about movies at the threshold
boundary, not what the full population looks like. Need COUNT queries at
0.27, 0.38, 0.43, 0.54 to understand how many movies survive at each
level and map against the ~100K catalog target.
**When:** After applying formula fixes above.
**See:** ingestion_data/threshold_analysis_report.md

## Switch to residential proxies for database refresh pipeline
**Context:** Datacenter proxy IPs get flagged by IMDB, causing
mass timeouts and 502s. Residential IPs (real ISP addresses) are
much harder to block. DataImpulse offers residential on the same
platform — just change the proxy port/host in `build_proxy_url()`.
**When:** Building the daily update / database refresh pipeline.
**See:** memory/imdb-scraping.md for full tuning findings.
