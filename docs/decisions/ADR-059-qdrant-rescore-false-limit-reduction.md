# [059] — Qdrant Search: rescore=False and 500-candidate Limit

## Status
Active

## Context

Qdrant vector searches were taking 5-7 seconds per query. Investigation
identified two independent causes:

1. `rescore=True` (the Qdrant default) re-fetches full float32 vectors from
   disk after the initial HNSW pass to compute exact scores. With scalar
   quantization (int8), in-RAM scoring is already sufficient — rescoring
   provides marginal precision gain at significant I/O cost.

2. `limit=2000` silently overrides `hnsw_ef`. Qdrant sets `ef = max(limit, hnsw_ef)`,
   so a limit of 2000 was overriding the configured `hnsw_ef=128` and forcing
   graph traversal 16x deeper than intended. The Qdrant docs do not prominently
   warn about this interaction.

## Decision

- Set `rescore=False` on all Qdrant searches (all vector spaces). All scoring
  runs in RAM via int8 quantized vectors.
- Reduce the default candidate limit from 2000 to 500. At 500, `ef` is still
  well above the configured `hnsw_ef=128`, avoiding any regression in recall.

## Alternatives Considered

1. **Keep rescore=True, reduce limit only**: Would still incur disk I/O for
   rescoring on the smaller candidate set. The main latency driver is disk
   access, not graph traversal.

2. **Keep limit=2000, set explicit ef=128 override**: Qdrant's search API
   accepts an explicit `ef` override. This would decouple ef from limit, but
   it adds a non-obvious parameter and the 2000-candidate downstream Postgres
   fetch (~13K with 8 vector spaces) is also unnecessarily large.

3. **Tune hnsw_ef upward**: Increasing hnsw_ef to match the intended search
   depth is valid for quality tuning but doesn't solve the limit-overrides-ef
   interaction or the rescore disk I/O.

## Consequences

- Search latency drops from 5-7s to expected sub-second range per vector space.
- Downstream Postgres bulk fetch shrinks from ~13K to ~4K candidates (8 spaces × 500).
- Recall could theoretically decrease if the top result falls outside rank 500;
  this is acceptable given the downstream reranking and the quality-prior buckets
  that already tolerate approximate top-k.
- If future evaluation reveals recall regression for long-tail queries, limit can
  be tuned back up (e.g., to 750) without changing rescore behavior.

## References

- `db/vector_search.py`, `db/search.py`
- ADR-004 (Qdrant quantization and memmap — the storage config this interacts with)
- ADR-005 (vector scoring pipeline — what consumes these candidates)
