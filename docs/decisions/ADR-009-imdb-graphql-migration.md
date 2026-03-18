# ADR-009: IMDB GraphQL Migration

**Status:** Active

## Context

The original IMDB scraping approach fetched 6 separate HTML pages
per movie (main page, plot summary, keywords, parental guide, full
credits, reviews), parsing `__NEXT_DATA__` JSON from each. This
required 600K HTTP requests for 100K movies, with significant
proxy bandwidth cost.

## Decision

Replace the 6 HTML page fetches with a single GraphQL query to
`api.graphql.imdb.com` per movie. The GraphQL endpoint returns
all the data previously spread across 6 pages in one response.

### What the GraphQL query extracts

- Credits (directors, writers, actors, characters, producers,
  composers)
- Keywords with community vote scoring
- Synopses and plot summaries (with priority selection)
- Parental guide (category + severity)
- Featured reviews (summary + text)
- Maturity rating, reception data (IMDb rating, Metacritic)
- Filming locations, budget, languages, countries, production
  companies, genres

### Implementation

- `imdb_scraping/http_client.py`: Async GraphQL client with
  proxy routing, semaphore-controlled concurrency (default 60),
  random UA rotation, exponential backoff
- `imdb_scraping/parsers.py`: GraphQL response transformer
- `imdb_scraping/models.py`: `IMDBScrapedMovie` Pydantic model

## Alternatives Considered

1. **Keep 6 HTML page approach**: 6x more requests, 6x more proxy
   bandwidth, more complex error handling for partial page failures.
2. **Use IMDB's official API**: Requires commercial license, not
   available for this project scale.

## Consequences

- Reduced proxy bandwidth by ~5-6x (one request vs six).
- Simplified error handling — one request succeeds or fails per
  movie, no partial-page logic needed.
- Keyword scoring formula preserved from original design:
  `score = usersInterested - 0.75 * dislikes`.
- Output format: originally per-movie JSON at
  `ingestion_data/imdb/{tmdb_id}.json`; later migrated to the
  `imdb_data` table in tracker.db (see ADR-023).

## References

- movie_ingestion/imdb_scraping/ (current implementation)
- docs/modules/ingestion.md (Stage 4 details)
