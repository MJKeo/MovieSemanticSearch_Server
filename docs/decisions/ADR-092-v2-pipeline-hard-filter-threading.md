# [092] — V2 pipeline hard-filter threading strategy

## Status
Active

## Context
The Gradio UI's six hard filters (release date, runtime, maturity, genres,
audio languages, streaming providers) were wired into the UI but never
sent to the API. The V2 pipeline issued all Postgres and Qdrant queries
without filtering. Post-filtering was unacceptable: top-K candidates could
all be excluded, leaving the UI empty even when filter-compatible movies
existed in the corpus.

## Decision
Thread `MetadataFilters` (the existing dataclass at
`implementation/classes/schemas.py:778`) through the entire V2 pipeline so
every query primitive applies filters at retrieval time. Two new helpers in
`db/postgres.py` centralize the SQL fragment generation:
- `_build_inline_movie_card_filter_clause` for posting-table queries
- `_build_direct_movie_card_filter_clause` for queries whose FROM is already `movie_card`

Both return `("", [])` when filters is None, making no-filter behavior
byte-identical to pre-change. Streaming-service values travel as
`StreamingService` enum names and are expanded at the API boundary into
encoded `(provider_id << 4) | method_id` values via fan-out over
`StreamingAccessType` IDs.

## Alternatives Considered
- **Pre-resolve to eligible-id set, thread everywhere**: rejected. A
  `HasIdCondition` over ~100K IDs is much slower than Qdrant payload
  `Range`/`MatchAny` index lookups; also bloats request memory and wastes
  work on loose filters.
- **Post-filter after retrieval**: rejected. Top-K candidates could all be
  excluded, producing empty results when matches exist below the cutoff.
- **Apply only to Postgres, skip Qdrant**: rejected. Semantic search would
  return filter-excluded candidates that get promoted to top positions.

## Consequences
- Semantic elbow-calibration probe (`_run_corpus_topn`) also filtered — the
  calibrated score threshold now reflects the filtered distribution, not the
  full corpus. Critical for correctness on tight filters.
- Similarity branch excluded by design (anchor-based search is not
  filter-relevant).
- Shorts subtraction (auxiliary blocklist) not filtered — filtering the
  blocklist would let out-of-range shorts survive as candidates.
- Trending via Postgres round-trip: `execute_trending_query` calls
  `fetch_movie_ids_matching_filters` to intersect the Redis hash.
- `UNRATED` movies (maturity_rank=999) are silently excluded by any
  maturity range filter — intentional and acceptable.

## References
- `api/main.py` (`MetadataFiltersInput`, `_to_metadata_filters`)
- `db/postgres.py` (hard-filter helpers)
- `search_v2/streaming_orchestrator.py`, `search_v2/stage_4_execution.py`,
  all `search_v2/endpoint_fetching/*.py` modules
- `run_gradio_ui.py` (UI controls)
