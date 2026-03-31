# [047] — IMDB GraphQL Query: Remove Artificial Field Limits

## Status
Active

## Context

The IMDB GraphQL query in `http_client.py` used `plots(first: 10)` and
similar limits on credits, company, and composer fields. An analysis of
86,622 scraped movies found that IMDB returns outlines and short summaries
before synopses in the plots list, so `first: 10` silently dropped synopses
for movies where the synopsis was entry 11 or later. Since synopses are the
primary input for plot_events Branch A (condensation path), silent truncation
produced lower-quality plot_summary output without any logging or detection.

## Decision

Remove artificial limits on fields used for lexical search and plot text:
- `plots`: raised to `100,000` so synopses are always reached; parser still
  extracts only the first synopsis and first 3 plot summaries.
- `credits`, `companies`, `composers`: set to `100,000` — these populate
  lexical search entities and there is no retrieval benefit to truncation.
- `interests`: raised to `100` (IMDB curates these to a bounded set).
- `keywords(50)` and `reviews(10)` unchanged — keywords are vote-ordered
  (top 50 is sufficient), reviews are engagement-ordered (top 10 sufficient).
- Parser `[:8]` cap on interests removed.

## Alternatives Considered

1. **Raise `plots` to a moderate number (e.g., 50)**: Safer but still
   potentially fragile for movies with many outlines. Setting to 100,000
   is effectively unlimited and removes the truncation risk entirely; IMDB
   response sizes are bounded by the actual data available.

2. **Keep credits at 10**: Original motivation was to limit response payload.
   With residential proxies completing requests in <1s, payload size is not
   a bottleneck, and truncated credits degrade lexical search quality.

## Consequences

- Previously scraped movies that lost synopses under `first: 10` are not
  retroactively fixed — only new scraping runs get the full data.
- Response payload size increases modestly for movies with many credits.
- Parser logic is unchanged; the raised limits only affect what raw data is
  available for the parser to select from.

## References

- `movie_ingestion/imdb_scraping/http_client.py`
- `movie_ingestion/imdb_scraping/parsers.py`
- ADR-009 (GraphQL migration) — original scraping approach
