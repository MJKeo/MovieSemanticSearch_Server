# [069] — Split Person Posting Tables by Role

## Status
Active

## Context
The V1 lexical search used a single monolithic `lex.inv_person_postings`
table for all people (actors, directors, writers, producers, composers).
The V2 search design per `search_improvement_planning/v2_data_architecture.md`
requires role-aware lexical search: actors, directors, writers, producers,
and composers need separate channels so the query understander can route
"movies directed by X" vs "movies starring X" to the correct posting table.
Actor billing position is a prominence signal (top-billed cast vs. minor
appearances) that the monolithic table could not capture.

## Decision
Replace `lex.inv_person_postings` with 5 role-specific tables:
- `lex.inv_actor_postings` — adds `billing_position INT` and `cast_size INT`
  columns for prominence scoring. `billing_position` is 1-based index
  derived from `imdb_data.actors` list order (IMDB preserves billing order).
  `cast_size` is `len(deduplicated actors)`.
- `lex.inv_director_postings`
- `lex.inv_writer_postings`
- `lex.inv_producer_postings`
- `lex.inv_composer_postings`

**Search compatibility preserved at launch**: `_build_people_union_cte()` in
`db/lexical_search.py` UNIONs all 5 tables for compound lexical search,
maintaining existing "people as one bucket" behavior. Role-specific routing
(e.g. only query `inv_director_postings` when the query specifies a
director) is a future V2 change.

**`PEOPLE_POSTING_TABLES`** constant in `db/postgres.py`: module-level list
used by both compound search and exclusion resolution. New role tables can
be added in one place.

**`_normalize_name_list()`** replaces `create_people_list()`: returns an
ordered list (not a set) to preserve actor billing order. Deduplicates via
a seen-set rather than a set constructor.

## Alternatives Considered
**Keep monolithic `inv_person_postings`.** Defers V2 role routing indefinitely
and loses actor billing data permanently (billing order is available at
ingest time but not recoverable later). Rejected.

**4 tables without composer.** Initial implementation shipped with 4 tables
(composers excluded — "no posting table needed for search"). Subsequently
added in a follow-up: composer data was already scraped/stored and excluding
it silently prevents "movies with music by X" queries.

**Separate billing_position table.** Would normalize prominence data.
Rejected as over-engineering — billing_position is a non-nullable attribute
of an actor's relationship to a specific movie, not a separate entity.

## Consequences
- The monolithic `inv_person_postings` table is dropped from the schema.
  Any existing Postgres database must be rebuilt from scratch (the project
  already wipes and rebuilds from `db/init/01_create_postgres_tables.sql`).
- Unit tests referencing the old `create_people_list()` function and
  `inv_person_postings` table needed updating.
- `PEOPLE_POSTING_TABLES` constant must be kept in sync with the DDL. Adding
  a new role requires updating both the SQL file and the constant.
- Role-specific query routing (the primary V2 motivation) is still deferred
  — this change provides the data foundation but does not change search
  behavior yet.

## References
- `db/init/01_create_postgres_tables.sql` — role-specific posting table DDL
- `db/postgres.py` — `PEOPLE_POSTING_TABLES`, `batch_insert_actor_postings()`, `_build_people_union_cte()`
- `movie_ingestion/final_ingestion/ingest_movie.py` — role-specific posting ingestion
- `db/lexical_search.py` — `_build_people_union_cte()`
- `search_improvement_planning/v2_data_architecture.md` — V2 role-aware search design
- `docs/decisions/ADR-011-data-store-architecture.md` — overall data-store architecture
