# ADR-011: Data Store Architecture (Postgres + Qdrant + Redis)

**Status:** Active

## Context

The system needs persistent storage for movie metadata (structured
queries + bulk enrichment), vector similarity search, and multiple
caching layers. Each workload has different access patterns.

## Decision

Three specialized data stores, each serving its strength:

### PostgreSQL 15

**Purpose**: Lexical search + movie card metadata for display and
reranking.

**Schema** (two schemas):
- `public.movie_card` — thin table for card rendering + reranking:
  movie_id (PK), tmdb_id, title, year, poster_url, release_ts,
  runtime_minutes, maturity_rank, genre_ids (INT[]),
  watch_offer_keys (INT[]), audio_language_ids (INT[]),
  reception_score, popularity_score, imdb_vote_count
- `lex.*` — inverted index posting tables for lexical search:
  inv_person_postings, inv_character_postings, inv_studio_postings,
  inv_title_token_postings. Each stores (term_id, movie_id) pairs.
  Dictionaries map normalized strings to term IDs.

**Access pattern**: Single bulk fetch via
`WHERE movie_id = ANY($1)` after candidate merge. Never per-candidate.

### Qdrant

**Purpose**: Vector similarity search across 8 named vector spaces.

**Collection**: `movies` with 8 named vectors per point (1536 dims
each, OpenAI text-embedding-3-small).

**Payload** (hard filters only): release_ts, runtime_minutes,
maturity_rank, genre_ids, watch_offer_keys. Create payload
indexes for all filter fields.

**Config**: Scalar quantization (int8) + memmap (on_disk: true).
See ADR-004.

**Access pattern**: IDs + scores only in responses. No payload
fetched. No stored vectors re-fetched after initial search.

### Redis 7

**Purpose**: Four cache namespaces.

| Key Pattern | Contents | TTL |
|-------------|----------|-----|
| `emb:{model}:{hash}` | Embedding vector (binary float array) | ~7 days |
| `qu:v{N}:{hash}` | Full QU output (JSON) | 1 day |
| `trending:current` | Hash of movie_id → trending score | None (atomic replace) |
| `tmdb:movie:{id}` | TMDB detail JSON | 1 day |

**Config**: decode_responses=False (raw binary), max_connections=10,
volatile-lru eviction policy (TTL-less keys immune to eviction).

## Key Invariants

- movie_id is always tmdb_id everywhere.
- Qdrant payload is for hard filters only — full metadata in Postgres.
- Never query Postgres per-candidate.
- Never cache partial DAG outputs in Redis.
- Embedding cache does NOT lowercase; QU cache does.
- Trending key has no TTL — replaced atomically via RENAME.

## Alternatives Considered

1. **Single Postgres for everything**: Postgres pgvector exists
   but doesn't support named vectors, scalar quantization, or
   the performance profile needed for 8-space ANN search.
2. **Elasticsearch for lexical**: Overkill for entity-based
   posting list lookups. Postgres inverted indexes are simpler
   and sufficient.
3. **Memcached instead of Redis**: No data structures (Hashes,
   Sets). Trending scores and embeddings benefit from Redis's
   richer types.

## References

- guides/postgres_database_structure.md
- guides/qdrant_database_structure.md
- guides/redis_database_structure.md
- guides/server_architecture_guide.md
