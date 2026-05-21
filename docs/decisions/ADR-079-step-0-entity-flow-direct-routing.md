# [079] — Step 0 entity-class direct routing with dedicated executors

## Status
Active

## Context
The V2 search pipeline originally funneled all queries through the same trait-based
Stage 3/4 machinery. Entity queries (actor, studio, franchise, character) have
fundamentally different retrieval semantics: they need posting-list lookups, bucketed
relevance by prominence, and popularity-sorted results — not trait scoring. Running
them through the semantic/trait pipeline produces poor quality and wastes latency on
irrelevant LLM calls.

## Decision
Step 0 classifies queries into 7 mutually exclusive flow types:
EXACT_TITLE, SIMILARITY, ACTOR, CHARACTER_FRANCHISE, NON_CHARACTER_FRANCHISE, STUDIO,
and GENERAL. Each non-GENERAL flow gets a dedicated executor module that bypasses the
Step 1/2/3 LLM pipeline entirely and returns results directly. The orchestrator reads
`flow_type` from the Step 0 output and dispatches accordingly before any further
steps run.

## Alternatives Considered
- **Route all queries through trait pipeline with entity boosting**: Tried but entity
  queries need bucketed prominence scoring that doesn't map to trait weights. Produces
  results sorted by tangential traits rather than actor billing prominence or franchise
  mainline status.
- **Single executor with flow-type switches**: Would create an unmaintainable monolith.
  Dedicated modules keep each flow's complexity isolated.

## Consequences
- Entity flows return results without Step 1/2/3 latency (large latency win for ~30–40%
  of queries estimated to be entity-type).
- Each executor can be tuned independently without risk of regressing other flows.
- Adding a new entity type requires a new executor + Step 0 flow class (well-defined
  extension point).
- Step 0 prompt must handle all 7 classes correctly; misclassification routes to wrong
  executor with no fallback.

## References
- docs/modules/search_v2.md — Entity-flow executors section
- ADR-076: five-step query understanding pipeline (original pipeline design)
