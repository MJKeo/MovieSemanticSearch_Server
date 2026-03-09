# ADR-002: TMDB-First Quality Funnel

**Status:** Active

## Context

The TMDB daily export contains ~1M movie entries. Downstream
stages (IMDB scraping, LLM generation, embedding) are expensive:
IMDB scraping costs ~$300-400 for 100K movies via proxy, LLM
generation costs ~$250-2,500 for 100K movies. Running these
on all 1M entries would cost 10x more.

## Decision

Fetch TMDB details for ALL ~950K candidates (free — API key only,
rate-limited at ~40 req/s), then apply a quality funnel to select
the top ~100K before running expensive stages.

### Quality Funnel Design

**5 hard filters** (disqualify immediately):
1. zero_vote_count
2. missing_or_zero_duration
3. missing_overview
4. no_genres
5. future_release

**10-signal weighted quality score** (weights sum to 1.0):
- vote_count (0.38) — dominant signal, log-scaled, age-adjusted
- watch_providers (0.25) — US streaming availability, tiered
- popularity (0.12) — TMDB's algorithmic activity score
- has_revenue (0.05), poster_url (0.05), overview_length (0.04),
  has_keywords (0.04), has_production_companies (0.03),
  has_budget (0.02), has_cast_and_crew (0.02)

**Soft threshold**: quality_score < -0.0441 (determined via
Gaussian-smoothed survival curve analysis).

## Alternatives Considered

1. **Run IMDB scraping on all 950K**: Cost prohibitive ($2,800-3,800
   in proxy bandwidth alone).
2. **Filter by vote_count only**: Loses nuance — new releases with
   high popularity but low vote counts would be excluded.
3. **Use TMDB popularity as pre-filter**: Popularity is volatile
   and would exclude classics with low recent activity but high
   lifetime engagement.

## Consequences

- TMDB fetching stage runs 7-8 hours at standard rate tier.
  This is wall time, not cost — TMDB is free.
- The quality scorer's vote_count weighting means the top 100K
  skews toward movies with audience validation (≥50-100 votes
  at the cutoff), which correlates with having IMDB reviews,
  ratings, and parental guide data for downstream enrichment.
- Vote count is the "universal meta-signal" — every other quality
  proxy (streaming coverage, poster presence, data completeness)
  correlates monotonically with it.

## References

- guides/full_movie_fetch_pipeline_guide.md
- guides/stage_2_tmdb_fetching.md
- guides/post_filter_data_analysis.md
