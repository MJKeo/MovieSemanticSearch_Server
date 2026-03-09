# ADR-004: Scalar Quantization + Memmap for Qdrant

**Status:** Active

## Context

150K movies × 8 vectors × 1536 dims × 4 bytes (float32) = ~7.4 GB
of vector data. This exceeds the available RAM on a t3.large
(8 GB total, ~3 GB available for Qdrant after other services).

## Decision

Enable two Qdrant features at collection creation time:

1. **Scalar quantization (int8)**: Compresses each float32 (4 bytes)
   to int8 (1 byte). Reduces vector storage to ~1.85 GB. Accuracy
   tradeoff is negligible for ANN retrieval.

2. **Memmap storage** (`on_disk: true`): Original unquantized vectors
   stored on EBS disk, not RAM. Qdrant loads hot pages on demand.

### Collection Config

```yaml
quantization_config:
  scalar:
    type: int8
    quantile: 0.99
    always_ram: false

vectors_config:
  on_disk: true
```

With both enabled, Qdrant's working RAM footprint is ~2.5–3 GB
(quantized vectors + HNSW graph).

## Alternatives Considered

1. **No compression**: Would require upgrading to a larger (more
   expensive) EC2 instance.
2. **Product quantization**: Higher compression ratio but
   significantly more accuracy loss.
3. **Binary quantization**: Too aggressive for 1536-dim vectors.

## Consequences

- These settings cannot be easily toggled after data is indexed
  without a full re-index. Must be set at collection creation time.
- Payload indexes must be created for all filter fields (range
  indexes for `release_ts`, `runtime_minutes`, `maturity_rank`;
  keyword/array indexes for `genre_ids`, `watch_offer_keys`).
- Qdrant payload stores only hard filter fields, not full metadata.
  Full metadata lives in Postgres (see ADR cross-codebase invariant).

## References

- guides/qdrant_database_structure.md
- guides/server_architecture_guide.md (section 1C)
