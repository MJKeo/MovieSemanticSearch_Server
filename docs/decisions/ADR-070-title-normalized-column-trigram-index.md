# [070] — Normalized title column with trigram and btree indexes for symmetric title matching

## Status
Active

## Context
Stage 3 title matching in search_v2 used `ILIKE` against the raw
`movie_card.title` column. Diacritics (é, ü, ñ) and case differences
caused silent misses: a query for "Amelie" would not hit "Amélie" because
the two sides of the comparison were normalized differently (query via
`normalize_string`, stored value raw). V1 title-token infrastructure
was the previous workaround but was unread by search_v2 and could not
be efficiently indexed for prefix/contains matching.

## Decision
Add a `movie_card.title_normalized` column populated at ingest time by
running `normalize_string()` on the title. Back it with:
- A trigram GIN index (contains matching)
- A `text_pattern_ops` btree index (starts_with matching)

Stage 3's `_execute_title_pattern` now uses plain `LIKE` (not `ILIKE`)
because both sides are already normalized at their respective boundary.
The V1 title-token tables (`lex.title_token_strings`,
`lex.inv_title_token_postings`, `lex.title_token_doc_frequency`,
`movie_card.title_token_count`) and all helpers that wrote/read them are
removed from `db/init/01_create_postgres_tables.sql` and `db/postgres.py`.

## Alternatives Considered
- **Keep `ILIKE` on raw column with `pg_trgm`**: ILIKE + trigram works but
  does not normalize diacritics (ILIKE only folds case, not Unicode
  equivalences). "Amelie" vs "Amélie" would still miss.
- **Normalize query and use `ILIKE` on raw column**: Asymmetric — ingest
  writes the raw form, query normalizes; a future ingest rule change would
  silently break retrieval. Violates the normalization-symmetry convention.
- **Retain V1 title-token tables**: The V1 posting-list approach is not
  used by search_v2 and adds schema complexity with no active consumer.

## Consequences
- Symmetric title matching: diacritics, case, and Unicode equivalences
  are collapsed identically at ingest and query time.
- `LIKE` (not `ILIKE`) on the normalized column means the trigram and
  btree indexes are usable by Postgres (ILIKE is not index-compatible
  on these index types without `citext`).
- V1 title-token infrastructure is retired; any V1 search code that
  called the removed helpers will import-fail (stubs returning `{}`
  are left so `db/lexical_search.py` still loads — full V1 retirement
  is a separate step).
- **Backfill required**: `backfill/rebuild_lexical_postings.py` must run
  after schema migration and before search_v2 title_pattern is exercised
  in production.

## References
- docs/conventions.md — "String normalization runs identically at ingest
  and query time"
- search_improvement_planning/entity_improvement.md — companion plan
- docs/decisions/ADR-069-person-posting-tables-by-role.md
