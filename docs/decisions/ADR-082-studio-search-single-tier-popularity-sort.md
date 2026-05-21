# [082] — Studio search: single-tier popularity sort with LLM translation

## Status
Active

## Context
Studio queries ("Pixar movies", "A24 films") need to map user-provided studio names to
canonical DB identifiers and return matching films. Unlike actor or franchise search,
there is no natural prominence hierarchy for a studio's relationship to its films —
a studio either produced a film or it didn't. Co-productions add minor complexity but
don't warrant separate buckets.

## Decision
Studio search is a 3-stage pipeline with a single result tier:
1. LLM translation via `studio_query_generation` — maps the user's studio name to
   canonical DB studio identifiers. Soft-degrades to deterministic fuzzy match on LLM
   failure (latency or API error).
2. `execute_studio_query` in ANY mode — returns all films matching any of the
   resolved studio IDs (co-productions included without demotion).
3. Popularity sort via `popularity_sort.py` — NULLS LAST DESC, movie_id DESC tiebreaker.

No sub-bucketing for co-productions; user expectation is "films associated with this
studio" not "films where this studio was the primary producer."

## Alternatives Considered
- **Primary-producer vs. co-producer tiers**: Adds complexity without clear user benefit;
  users querying "A24 films" want the full catalogue, not just sole-producer entries.
- **Deterministic name matching only**: Would miss variant studio names and
  abbreviations; LLM translation handles synonyms and common abbreviations robustly.

## Consequences
- Simple, fast: one LLM call (or none on fallback) + one DB query + sort.
- No per-film scoring; results ordered entirely by popularity.
- LLM failure path (deterministic fallback) must be tested to ensure it doesn't
  silently return empty results for valid studio names.

## References
- docs/modules/search_v2.md — Studio executor section
- search_v2/studio_search.py, search_v2/popularity_sort.py
