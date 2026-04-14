# [066] — Upgrade Embedding Model to text-embedding-3-large

## Status
Active

## Context
The V2 search system re-embeds the entire corpus with structured-label
vector text formats across all 8 vector spaces. Because a full re-embed
was already required, there was no migration cost to switching embedding
models. The existing model, `text-embedding-3-small` (1536 dims), had
been chosen for cost and latency before the corpus size and search
architecture were finalized. With the corpus now at ~100K movies and the
embedding format substantially changed, the model decision was revisited.

## Decision
Upgrade from `text-embedding-3-small` (1536 dims) to
`text-embedding-3-large` (3072 dims, native) for all 8 vector spaces.
Native 3072 dimensions were chosen over Matryoshka truncation because
the full re-embed eliminated any infrastructure compatibility constraint.

Single shared helper (`generate_vector_embedding` in
`implementation/llms/generic_methods.py`) is the sole call site; updating
`_EMBEDDING_MODEL` and the Qdrant init script (`db/init/02_qdrant_init.sh`)
was sufficient to cover all ingestion and search paths.

## Alternatives Considered
**Stay with text-embedding-3-small.** Cheapest option. Rejected because
the corpus re-embed was already required, making the cost delta negligible
for an MTEB lift upgrade.

**Voyage-3-large.** Top MTEB performer for retrieval at the time of
decision. Retained as the next revisit candidate if embeddings become
the bottleneck. Deferred because it requires a provider change and
separate SDK integration.

**Matryoshka truncation to 1536 dims.** Would have kept Qdrant vector
size unchanged. Rejected because the user planned a full Qdrant wipe and
rebuild, so there was no reason to accept reduced quality.

## Consequences
- All 8 Qdrant named vectors resized from 1536 → 3072 floats. Qdrant init
  script updated. Existing Qdrant collection must be wiped and rebuilt.
- Redis embedding cache coexists harmlessly: cache keys include the model
  name (`emb:{model}:{hash}`), so old 3-small entries and new 3-large
  entries do not collide. No manual flush needed.
- tiktoken encoding string updated from `text-embedding-3-small` to
  `text-embedding-3-large` (both share `cl100k_base` and the 8191 token
  limit — no functional change).
- End-to-end validation (Qdrant drop/recreate, re-embed ~100K movies,
  assert `len(embedding) == 3072`) is a separate operational step.

## References
- `implementation/llms/generic_methods.py` — `_EMBEDDING_MODEL` constant
- `movie_ingestion/final_ingestion/ingest_movie.py` — `EMBEDDING_MODEL`
- `db/init/02_qdrant_init.sh` — all 8 vector space size declarations
- `db/vector_search.py` — stale 1536-dim comments updated
- `docs/decisions/ADR-011-data-store-architecture.md` — embedding cache key format
- Project memory: `project_embedding_model_decision.md`
