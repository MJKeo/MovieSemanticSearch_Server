# [071] — Hyphen variant expansion for deterministic entity resolution

## Status
Active

## Context
Entity names containing hyphens ("Spider-Man") exist in IMDB credits and
user queries in multiple surface forms: "spider-man", "spider man",
"spiderman". The LLM-based entity extraction cannot guarantee which form
it will emit, and users type all three variants. Without explicit handling,
a query for "spiderman" would retrieve zero credits despite 60+ Spider-Man
films in the corpus.

## Decision
Add `expand_hyphen_variants(normalized_string) -> list[str]` to
`implementation/misc/helpers.py`. Takes an already-normalized string and
returns the deduplicated set of {with-hyphen, hyphen→space, hyphen→empty}
forms. This helper is called symmetrically at both ingest and query time.

At ingest (`ingest_lexical_data`):
- Cast/character size denominators are frozen from distinct normalized
  names *before* expansion.
- Each distinct name contributes every hyphen variant as its own
  `term_id` in `lex.lexical_dictionary` / `lex.character_strings`.
- Posting rows carry the origin name's `billing_position`, so variants
  cannot create phantom cast members or shift prominence.

At query time (search_v2 Stage 3 entity executor):
- `expand_hyphen_variants` is applied to the normalized query string
  before posting-list lookup, producing the same set of term_ids.

## Alternatives Considered
- **Fuzzy / trigram match on entity names**: Would handle spelling
  variants too but introduces false positives (e.g. "Spider" matching
  "Sydney"). Entity matching has a higher precision requirement than
  title matching.
- **LLM normalization instruction**: The Stage 3 entity prompt could
  tell the model to emit canonical forms. Unreliable for small models
  and creates a query-only solution that does not help ingest-side
  retrieval consistency.
- **Single canonical form with aliases column**: Requires schema changes
  and a normalization-decision authority for which form is canonical.
  The posting-list approach makes all variants first-class without a
  canonical-form decision.

## Consequences
- "spider-man", "spider man", and "spiderman" resolve to the same credit
  set in both directions.
- `batch_insert_actor_postings` and `batch_insert_character_postings`
  signatures changed to take parallel `term_ids` + `billing_positions`
  lists instead of auto-generating positions from list index. Any caller
  or test using the old signature must be updated.
- Denominators (cast_size, character_cast_size) remain honest because
  they are frozen before expansion.
- Backfill required: `backfill/rebuild_lexical_postings.py` must re-run
  after this change lands.

## References
- docs/conventions.md — normalization symmetry invariant
- search_improvement_planning/entity_improvement.md
- ADR-069-person-posting-tables-by-role.md
- ADR-070-title-normalized-column-trigram-index.md
