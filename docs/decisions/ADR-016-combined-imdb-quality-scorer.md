# ADR-016 — Combined TMDB+IMDB Quality Scoring Model

## Status
Active

## Context

After Stage 4 (IMDB scraping), ~120K movies need to be filtered down
to ~80-100K before the expensive LLM metadata generation step (Stage 6,
estimated $250-2,500 for 100K movies). The existing Stage 5 only applies
hard filters on essential data presence (no directors, no actors, etc.)
but lacks a soft quality scorer analogous to Stage 3's TMDB quality
scorer.

At this point in the pipeline, we have the complete data picture from
both TMDB and IMDB sources. Rather than reusing the Stage 3 TMDB
quality score, we designed a fresh combined scorer that evaluates the
full dataset — IMDB data primary, TMDB as fallback — with weights
calibrated to the signals that matter most at this stage: data richness
for LLM generation and user relevance for the search app.

## Decision

Implement an 8-signal weighted quality scorer at Stage 5 that replaces
the TMDB-only scoring approach for the final quality gate. Weights sum
to 1.0; individual signal ranges are signed to penalize absence of
common fields.

### Scoring Model

| Signal                | Weight | Range      | Scale                          |
|-----------------------|--------|------------|--------------------------------|
| imdb_vote_count       | 0.22   | [0, 1]    | Log-scaled + recency/classic   |
| watch_providers       | 0.20   | [-1, +1]  | Binary with theater window     |
| featured_reviews_chars| 0.16   | [-1, +1]  | Tiered by total char count     |
| plot_text_depth       | 0.12   | [0, 1]    | Log-scaled composite           |
| lexical_completeness  | 0.10   | [-1, +1]  | Capped entity-type composite   |
| data_completeness     | 0.10   | [-1, +1]  | Tiered/binary field composite  |
| tmdb_popularity       | 0.06   | [0, 1]    | Log-scaled                     |
| metacritic_rating     | 0.04   | [0, +1]   | Binary bonus                   |

### Key Design Choices

1. **IMDB vote count only** — not both TMDB and IMDB. IMDB votes are
   more representative for a US audience and highly correlated with
   TMDB votes; including both adds complexity without discrimination.
   Recency/classic multipliers applied fresh (never applied to IMDB
   votes before).

2. **Binary watch providers** — simplified from Stage 3's 3-tier system
   to binary +1/-1. At this stage the question is "can users access
   this at all?" not "how many platforms carry it?" Harsh -1 penalty
   at weight 0.20 makes it nearly impossible for unwatchable movies
   to pass.

3. **Two composite signals** replace many small individual signals:
   - `data_completeness` (6 fields): vector-search-readiness — plot
     keywords, overall keywords (tiered), filming locations, parental
     guide, maturity rating, budget.
   - `lexical_completeness` (6 entity types): lexical-search-readiness —
     actors, characters (threshold at 5), writers, composers, producers,
     production companies (binary presence).

4. **plot_text_depth composite** — overview + plot_summaries + synopses
   as total chars rather than individual signals, because they are
   substitutes (total text budget matters, not which field contributes).

5. **TMDB popularity retained** at low weight (0.06) for short-term
   buzz signal that vote count and watch providers don't capture.

6. **Stage 3 TMDB quality score is NOT reused** as an input. Both
   scores are computed independently from raw fields.

### Full design specification

See `docs/modules/ingestion.md` (Stage 5 section) for complete signal
details, tier boundaries, formulas, and rationale.

## Alternatives Considered

1. **Reuse the Stage 3 TMDB quality score as an input signal**: Rejected
   because many TMDB-only fields now have richer IMDB equivalents. The
   weight each attribute should carry has changed non-uniformly, so
   recalculating from scratch is more accurate than adjusting a prior
   composite score.

2. **Use both TMDB and IMDB vote counts**: Rejected because they are
   highly correlated. A single authoritative source (IMDB) with
   recency/classic adjustment is cleaner.

3. **Score plot_keywords as a standalone weighted signal (0.10)**:
   Rejected because it overlaps with plot_text_depth — a movie with
   a rich synopsis needs fewer keywords. Folded into data_completeness
   composite instead.

4. **Tiered watch_providers (0/1-2/3+)**: Rejected in favor of binary.
   Stage 3 already graduated movies past the "how many providers"
   question. At Stage 5, the relevant question is accessibility at all.

5. **Many small individual signals (13 in early proposal)**: Rejected
   because sub-1% impact signals don't meaningfully shift movies across
   threshold boundaries. Consolidated into composites for cleaner
   discrimination.

## Consequences

- Stage 5 now performs both hard filtering (essential data checks) and
  soft scoring (combined quality score), replacing the previous
  hard-filter-only approach.
- The raw quality score is NOT normalised by this scorer. A separate
  pass will use survival-curve derivative analysis (same approach as
  Stage 3's `plot_quality_scores.py`) to find the threshold.
- Target survival: ~80-100K movies, but the actual cutoff depends on
  the shape of the score distribution.
- The scoring function loads both TMDB data and IMDB data from the
  tracker DB (`tmdb_data` and `imdb_data` tables) for each movie.
  *(Originally loaded IMDB data from per-movie JSON files; migrated
  to SQLite in ADR-023.)*
- Priority alignment: search quality (#1) drives the data-richness
  signals; cost (#3) drives the gating function itself (filtering
  before expensive LLM generation).

## References

- `docs/modules/ingestion.md` (Stage 5 section) — full scoring
  specification
- ADR-002 (TMDB-first quality funnel) — Stage 3 scorer, still active
  for the TMDB-only filtering stage
- ADR-012 (LLM generation cost optimization) — the cost pressure that
  motivates this quality gate
- `movie_ingestion/tmdb_quality_scoring/tmdb_quality_scorer.py` —
  Stage 3 scorer used as design precedent
- `ingestion_data/imdb_data_quality_report.json` — data coverage
  analysis that informed signal selection and weight calibration
