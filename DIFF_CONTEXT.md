# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Cleanup: removed quality_score_design.md and top_no_providers.py
Files: movie_ingestion/imdb_quality_scoring/quality_score_design.md (deleted),
movie_ingestion/imdb_quality_scoring/top_no_providers.py (deleted),
docs/modules/ingestion.md (updated references)
| Content from quality_score_design.md was already fully captured in
ingestion.md's Stage 5 section. Removed stale file references and
the top_no_providers.py Key Files entry from ingestion.md.

## Rewrite sample_threshold_candidates.py for per-group threshold sampling
Files: movie_ingestion/imdb_quality_scoring/sample_threshold_candidates.py
Why: Candidate thresholds were identified per provider group (has_providers,
no_providers_new, no_providers_old) from survival-curve analysis, but the
sampling script was group-unaware with a flat threshold list.
Approach: Single DB query with CASE expression classifies movies into 3 groups
(same logic as plot_quality_scores.py). Each group has its own thresholds:
has_providers [0.265, 0.547], no_providers_new [0.278, 0.441, 0.56],
no_providers_old [0.398, 0.47, 0.59, 0.645]. Samples 10 below/equal + 10
above per threshold, no cross-contamination between groups. Outputs 3 separate
JSON files to ingestion_data/. Bulk-loads TMDB + IMDB data once across all groups.

## Stage 5 quality scorer v2 → v3: formula refinements
Files: movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py
Why: Manual threshold evaluation on no_providers_old revealed garbage movies
(2.1/10, 2.8/10) scoring comparably to good movies (8.2/10). Three formula
flaws identified: featured_reviews ignored review count, community_engagement
used binary presence, and lexical_completeness didn't adjust for film age.

### Key changes
- **featured_reviews**: 4-tier char-only → linear chars+count blend (caps: 5000 chars, 5 reviews)
- **community_engagement**: plot_keywords and featured_reviews use linear-to-cap (5) instead of binary
- **lexical_completeness**: composers removed (not useful for search), actors/characters linear to 10,
  classic-film age boost added (1.0× at 20yr → 1.5× at 50yr, same ramp as vote_count)
- **data_completeness**: filming_locations removed (irrelevant for animated/short films),
  plot_keywords/overall_keywords/parental_guide_items use linear-to-cap instead of coarse tiers
- **Weights**: imdb_vote_count 0.27→0.25, community_engagement 0.08→0.10
- All non-binary attributes follow "good enough = 1.0" principle with linear growth to cap

### Design principle
Each attribute caps at 1.0 once it has "good enough" quantity for the app's needs,
with linear growth in the 0→cap range. Avoids over-rewarding movies just because
they have excess data (e.g. 80 actors vs 15).

## Stage 5 quality scorer v3 → v4: vote count × Bayesian rating blend
Files: movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py
Why: Threshold evaluation on has_providers group showed movies with terrible IMDB
ratings (2.4/10) but moderate vote counts scoring comparably to genuinely notable
films. The pure vote-count signal had no quality discrimination.

### Key changes
- **imdb_vote_count → imdb_notability**: replaced pure log-scaled vote count with
  a vote-count × Bayesian-adjusted-rating blend. Three confidence tiers based on
  rating-stability analysis of 113K has_providers movies:
  - Low (< 100 votes): 95/5 vote/rating — ratings are noise (std ~1.37)
  - Medium (100–999): 70/30 — ratings become meaningful
  - High (≥ 1000): 85/15 — movie is already notable, rating modulates modestly
- **Bayesian rating**: IMDB formula with m=500, C=6.0 (dataset mean). Shrinks
  noisy low-vote ratings toward the mean before blending.
- **Weights**: imdb_notability 0.25→0.31, critical_attention 0.12→0.08,
  community_engagement 0.10→0.08
- Age multipliers (recency + classic boost) still applied after blending

## Updated threshold candidates for per-group sampling
Files: movie_ingestion/imdb_quality_scoring/sample_threshold_candidates.py
| Updated candidate thresholds after v4 scorer analysis: has_providers [0.35, 0.486],
no_providers_new [0.37, 0.44, 0.5], no_providers_old [0.6, 0.621, 0.654].

## Stage 5 quality filter + centralized group classification
Files: movie_ingestion/scoring_utils.py, movie_ingestion/imdb_quality_scoring/imdb_filter.py (new),
movie_ingestion/imdb_quality_scoring/sample_threshold_candidates.py,
movie_ingestion/imdb_quality_scoring/plot_quality_scores.py,
movie_ingestion/imdb_quality_scoring/analyze_imdb_quality.py

### Intent
Apply per-group quality thresholds to imdb_quality_calculated movies (Stage 5 filtering).
Centralize the PROVIDERS/NEW/OLD bucketing logic that was duplicated across 3+ files.

### Key Decisions
- Added `MovieGroup` enum, `classify_movie_group()`, `passes_imdb_quality_threshold()`,
  and `IMDB_QUALITY_THRESHOLDS` to scoring_utils.py as the canonical group classification.
- Centralized SQL fragment constants (`HAS_PROVIDERS_SQL`, `NO_PROVIDERS_SQL`,
  `THEATER_WINDOW_SQL_PARAM`) in scoring_utils.py — previously private in each script.
- imdb_filter.py follows the tmdb_filter.py pattern: separate scoring from filtering,
  materialise rows upfront, periodic commits, bulk UPDATE survivors.
- Thresholds: has_providers=0.486, no_providers_new=0.55, no_providers_old=0.654.
- Status progression: imdb_quality_calculated → imdb_quality_passed (or filtered_out).
- Refactored sample_threshold_candidates.py and plot_quality_scores.py to import shared
  SQL constants. Refactored analyze_imdb_quality.py's `_classify_movie` to delegate to
  `classify_movie_group()` with a local name mapping.
