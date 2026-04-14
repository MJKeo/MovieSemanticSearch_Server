# db/ — Database Access & Search Orchestration

All database access, search orchestration, scoring, and reranking
logic lives here. This is the core runtime module — every search
request flows through it.

## What This Module Does

Orchestrates the full search pipeline: receives a parsed query,
fans out to lexical (Postgres), vector (Qdrant), and metadata
scoring channels in parallel, merges candidates, applies quality
reranking, and returns ranked results. Movie ingestion into Postgres/Qdrant now lives in
`movie_ingestion/final_ingestion/`.

## Key Files

| File | Purpose |
|------|---------|
| `search.py` | Main orchestrator. `search()` is the entry point — launches all channels in parallel, merges, reranks, returns. |
| `vector_search.py` | End-to-end vector search across 8 Qdrant collections. Two-stage: original query + LLM subqueries. Fires searches as embeddings arrive. |
| `vector_scoring.py` | 5-stage pipeline converting raw Qdrant scores → single [0,1] score. Stages: execution flags → blend (80/20) → normalize (exp decay, k=3.0) → weight → sum. |
| `lexical_search.py` | Entity-based search via Postgres inverted indexes. Resolves actors/directors/franchises/characters to term IDs, computes F-score (beta=2.0). |
| `metadata_scoring.py` | Scores candidates against LLM-extracted metadata preferences (genres, date, providers, language, maturity, reception, trending, popularity, budget_size). Weighted average, weights are static. |
| `reranking.py` | Quality-prior reranking: bucket by relevance score (precision=2), sort within buckets by reception score. |
| `postgres.py` | Async connection pool, all SQL operations, posting list queries (role-specific people tables: actor/director/writer/producer/composer, plus character/studio/franchise/title_token), and movie card bulk fetch/upsert. `movie_card` persists nullable text buckets for both `budget_bucket` and `box_office_bucket` alongside the canonical scalar/array metadata used by reranking and downstream filtering. Also owns the franchise upsert/delete helpers for `movie_franchise_metadata` and `lex.inv_franchise_postings`, and the awards upsert helper for `movie_awards` (ceremony_id, award_name, category, outcome_id, year). `PEOPLE_POSTING_TABLES` constant lists all 5 role-specific tables for search code that unions across roles. |
| `qdrant.py` | Minimal Qdrant async client singleton. |
| `redis.py` | Async Redis pool for all four cache namespaces. |
| `tmdb.py` | TMDB API client with adaptive token-bucket rate limiting. |
| `trending_movies.py` | Compute trending scores (concave decay: `1 - (rank/500)^0.5`), update Redis atomically via staging key + RENAME. |
| ~~`ingest_movie.py`~~ | Moved to `movie_ingestion/final_ingestion/ingest_movie.py`. |

## Boundaries

- **In scope**: All database reads/writes, search orchestration,
  all scoring math, cache management, TMDB API client.
- **Out of scope**: LLM calls (live in `implementation/llms/`),
  data models (live in `implementation/classes/`),
  ingestion pipeline orchestration (lives in `movie_ingestion/`).

## Key Patterns

- All retrieval is async. Lexical search starts as soon as entities
  are extracted (before embeddings complete). Vector channels start
  as their embeddings arrive.
- Qdrant responses are IDs + scores only — no payload fetched.
- Metadata enrichment is a single bulk Postgres query after merge.
- The trending set is fetched once from Redis per request,
  concurrent with the Postgres bulk fetch.
- Existing Postgres databases need manual rollout SQL when new
  `movie_card` columns are added; for `box_office_bucket` the
  required step is `ALTER TABLE public.movie_card ADD COLUMN IF NOT EXISTS box_office_bucket TEXT;`.

## Scoring Pipeline Constants

| Constant | Value | File | Purpose |
|----------|-------|------|---------|
| SUBQUERY_BLEND_WEIGHT | 0.8 | vector_scoring.py | Subquery vs original blend ratio |
| DECAY_K | 3.0 | vector_scoring.py | Exponential decay steepness |
| ANCHOR_MEAN_FRACTION | 0.8 | vector_scoring.py | Anchor weight as fraction of active-space mean |
| RELEVANCE_RAW_WEIGHTS | {SMALL: 1.0, MEDIUM: 2.0, LARGE: 3.0} | vector_scoring.py | RelevanceSize → raw weight dict |
| BUCKET_PRECISION | 2 | reranking.py | Relevance score rounding |
| RECEPTION_FLOOR/CEIL | 30.0/90.0 | reranking.py | Reception score normalization range |
| TITLE_SCORE_BETA | 2.0 | lexical_search.py | F-score beta for title matching |
| MAX_DF | 10000 | lexical_search.py | Max document frequency threshold |
| TITLE_SCORE_THRESHOLD | 0.15 | lexical_search.py | Minimum title match score |
| TITLE_MAX_CANDIDATES | 10,000 | lexical_search.py | Max title search candidates |

## Gotchas

- Vector scores from Qdrant are **final** — never re-fetched or
  recomputed after the initial search.
- A blended score of 0.0 for a participating space is a real
  score (candidate not found), not a missing value.
- The "promoted SMALL" edge case: when relevance is NOT_RELEVANT
  but a subquery exists, effective relevance becomes SMALL for
  weighting, but did_run_original stays False (search decision
  was already made using pre-promotion relevance).
- Missing reception_score defaults to 0.5 (neutral) in reranking,
  not 0 — prevents penalizing movies without ratings data.
- **`rescore=False` in all Qdrant searches**: Rescoring loads float32
  vectors from disk for a marginal precision gain; with int8 scalar
  quantization, in-RAM scoring is sufficient. `rescore=True` was
  causing 5-7s query times.
- **Qdrant candidate limit is 500 (not 2000)**: HNSW graph traversal
  depth (`ef`) silently follows `limit` when `limit > hnsw_ef`. A
  limit of 2000 was overriding the configured `hnsw_ef=128` and
  making graph traversal ~16x more expensive than intended. 500
  keeps ef above the configured hnsw_ef while cutting downstream
  Postgres fetch from ~13K to ~4K candidates.
- **Removed non-lexical dictionary tables**: `genre_dictionary`,
  `language_dictionary`, `maturity_dictionary`, and
  `watch_method_dictionary` are no longer in the schema. Their
  corresponding `batch_upsert_*_dictionary()` functions in
  `postgres.py` have been removed along with the CREATE TABLE
  statements. `batch_upsert_lexical_dictionary()` still exists
  and is unaffected (it targets `lex.lexical_dictionary`).
- **`idle_in_transaction_session_timeout = '2min'`** is set at the
  database level in the init script. Connections left idle in a
  transaction (e.g. from crashed ingestion runs) are forcibly
  terminated after 2 minutes, preventing zombie lock accumulation on
  ingestion tables.
