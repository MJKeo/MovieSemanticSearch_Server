# [037] — Title-Type and Missing-Text Quality Gates in IMDB Quality Scorer

## Status
Active

## Context

After quality filtering, 2,621 movies were found to have passed Stage 5
that should not have:

- **2,236 movies with invalid title types**: IMDB's `titleType.id` field
  (added in the `imdb_title_type` scraping change) revealed that some
  TMDB entries classified as movies are actually video games, TV series,
  TV episodes, or other non-movie content. TMDB's own classification is
  unreliable for this purpose.

- **385 movies missing all text sources**: No `plot_summaries`, no
  `synopses`, and no `featured_reviews`. LLM metadata generation cannot
  produce meaningful output for most vector spaces without any text.
  These movies would generate empty or hallucinated metadata that
  degrades search quality.

Both problems are early-exit conditions — movies that fail either check
have no path to useful search results regardless of their quality score.
The existing 8-signal scorer can assign positive scores to these movies
if other signals (vote count, watch providers, lexical completeness) are
strong, so a score-only filter does not catch them.

## Decision

Add two hard gates as early returns in `compute_imdb_quality_score()`,
returning 0.0 before any signal computation:

**Title-type gate**: `ALLOWED_TITLE_TYPES = {"movie", "tvMovie", "short", "video"}`.
Any movie whose `imdb_title_type` is not in this set — including None
(not yet scraped) — returns 0.0. Catches tvSeries, tvEpisode, videoGame,
etc.

Rationale for the allowed set: `movie` is the obvious inclusion.
`tvMovie` are made-for-TV films that are legitimately searchable.
`short` films are explicitly in scope (see ADR-001). `video` covers
direct-to-video releases. All other types (tvSeries, tvEpisode,
videoGame, tvMiniSeries, etc.) are out of scope for a movie search engine.

None is treated as failing the gate because title type cannot be
verified without IMDB data. Movies that genuinely lack a title type
in IMDB would pass after re-scraping returns a valid type; movies that
were never scraped properly would be caught.

**Missing-text gate**: Returns 0.0 when `plot_summaries` is empty AND
`synopses` is empty AND `featured_reviews` is empty. Without any text
source, Stage 6 metadata generation produces nothing usable for the
plot_events, plot_analysis, viewer_experience, watch_context,
narrative_techniques, or reception vector spaces.

**Data cleanup**: One-off scripts retroactively filtered 2,236 movies
with invalid title types and 385 movies missing all text from
`imdb_quality_passed` status, using `log_filter()` with stage
`imdb_quality_funnel` to maintain the audit trail.

## Alternatives Considered

1. **Handle title type in Stage 4 (at scrape time)**: Would prevent these
   movies from entering the pipeline earlier, saving scraping cost. Rejected
   for now — the `imdb_title_type` field was added after a large corpus was
   already scraped, and adding filters to Stage 4 while Stage 5 is the
   authoritative quality gate splits responsibility. Stage 5 is the correct
   place for "should this movie enter the index" decisions.

2. **Add title type as a quality signal (weighted, not binary)**: Title
   type is a categorical eligibility check, not a quality gradient — a
   video game is not a "low quality movie." Binary early exit is semantically
   correct and computationally cheaper than including it in the score formula.

3. **Treat None title type as passing**: Would allow movies without
   scraped title type data to proceed. Rejected — movies missing title type
   data are indistinguishable from non-movie content at this stage, and
   false positives in the index are worse than false negatives.

4. **Separate filter script for title-type gate**: The gate is a
   property of what passes quality, not a separate filtering stage. Keeping
   it in the scorer maintains the single-codepath write principle.

## Consequences

- ~2,600 non-movie/text-poor movies removed from `imdb_quality_passed`,
  preventing them from consuming Stage 6 LLM generation budget.
- Future movies with invalid title types or no text are automatically
  rejected at scoring time without a separate filtering step.
- `imdb_title_type = None` (not yet backfilled or re-scraped) fails the
  gate. If a large re-scrape is needed and some movies temporarily lack
  title type, they will be filtered out until the field is populated.
- The `ALLOWED_TITLE_TYPES` constant is the authoritative list of
  content types eligible for the search index.

## References

- ADR-016 (combined IMDB quality scorer) — original scorer design
- ADR-019 (Stage 5 scorer v2) — scorer evolution context
- `movie_ingestion/imdb_quality_scoring/imdb_quality_scorer.py`
- `movie_ingestion/imdb_scraping/models.py` — `IMDBScrapedMovie.imdb_title_type`
