# [080] — Character-franchise search 7-tier prominence bucketing

## Status
Active

## Context
Character-franchise queries (e.g., "Batman movies", "MCU Spider-Man") need results
ranked by how central the character/franchise is to each film. The original design used
5 tiers derived from the franchise schema's two-axis structure (lineage × prominence).
Testing revealed the 5-tier model collapsed too many distinct prominence levels into the
same bucket, producing UI results where universe-adjacent films ranked alongside
mainline entries.

## Decision
Expand character-franchise search from 5 tiers to 7:
1. lineage-mainline — character is the primary subject of a canonical entry
2. top-billed-appearance — character appears prominently, top billing
3. lineage-ancillary — canonical but peripheral (side stories, shorts)
4. universe — same shared universe but character not present
5. prominent-appearance — meaningful role, not top billing
6. relevant-appearance — appears but secondary
7. minor-appearance — background or cameo

Results within each tier are sorted by popularity. Multi-character intersection takes
the MAX bucket index (weakest-link semantics: a film must satisfy both characters to
rank in a given tier).

## Alternatives Considered
- **Keep 5 tiers**: Simpler but loses the distinction between top-billed cross-franchise
  appearances and mainline entries — both mapped to tier 2.
- **Continuous prominence score**: More granular but harder to tune and explain; tier
  boundaries match how users mentally categorize franchise membership.

## Consequences
- Finer-grained tier separation produces more intuitive ordering for franchise queries.
- 7 tiers require the franchise schema to supply sufficient resolution; schema must
  not collapse these distinctions at ingest time.
- More buckets = more DB round-trips per tier unless pooled; current implementation
  pools all tiers in a single query and partitions in Python.

## References
- docs/modules/search_v2.md — Character-franchise executor section
- ADR-067: franchise schema two-axis rewrite (schema foundation)
- search_v2/character_franchise_search.py
