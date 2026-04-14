# [068] — Structured Award Data Storage in Postgres

## Status
Active

## Context
IMDB award nomination data was already scraped in Stage 4 (12 in-scope
ceremonies, stored as a JSON TEXT column in `imdb_data`). The V2 search
design needs award data in two forms: (1) deterministic filtering ("Oscar
winners", "Palme d'Or nominees") via a structured `movie_awards` table,
and (2) broad "award-winning movie" filtering via a GIN-indexed array
column on `movie_card`. Award ceremony names were previously hardcoded as
raw strings in `parsers.py` and `vector_text.py`, creating a fragile
multi-site dependency.

## Decision
Introduce `public.movie_awards` table and `award_ceremony_win_ids SMALLINT[]`
column on `movie_card`.

**`AwardCeremony` enum** (12 members in `schemas/enums.py`): consolidates
ceremony strings into a single source of truth. Each member carries the
IMDB `event.text` string as its value and a stable integer `ceremony_id`.
`CEREMONY_BY_EVENT_TEXT` lookup dict enables O(1) mapping from scraped
string to enum.

**`AwardOutcome` enum** (upgraded from plain StrEnum): `WINNER`/`NOMINEE`
each carry a stable `outcome_id` (1/2) for Postgres integer storage.
`AwardNomination.did_win()` provides intent-first access pattern.

**`movie_awards` table**: integer IDs for ceremony and outcome, TEXT for
`award_name` (e.g. "Oscar", "Palme d'Or") and `category` (nullable for
festival grand prizes with no named award). PK includes `award_name`
because some movies have multiple null-category awards at the same
ceremony/year (e.g. multiple Cannes Special Jury prizes). Ingested via
delete-then-insert per movie (not per-row upsert) since awards are always
a complete set. Deduplication before insertion groups by PK fields and
keeps the lowest `outcome_id` (best outcome) when IMDB returns multiple
nominee entries for the same award slot.

**`award_ceremony_win_ids SMALLINT[]` on `movie_card`**: GIN index for
broad "award-winning" filtering. Cannot use `gin__int_ops` (requires INT[])
but the column has at most 12 values per row so the impact is negligible.

**Category matching at search time**: `category` column stored as TEXT.
Fuzzy-matched against a small cached set of ~150 distinct categories via
`rapidfuzz`. A separate posting table was not built because award data has
structured enum IDs that map directly, unlike the open-ended text that
warrants lexical posting tables.

**`AwardNomination` and related sub-models** moved to `schemas/imdb_models.py`
so `db/` and `api/` containers can import them without mounting the
`movie_ingestion/` package.

## Alternatives Considered
**Raw TEXT columns for ceremony and outcome.** Simpler initial approach.
Rejected because string matching at search time is brittle and the enum
ID pattern was already established for `SourceMaterialType`, `AwardOutcome`,
and GIN array columns.

**Separate lexical posting table for awards.** Considered for category-level
search. Rejected because ceremony + outcome are structured (12 ceremonies,
2 outcomes), categories have low cardinality (~150 distinct values), and
fuzzy matching is adequate for that remaining dimension.

**Store only wins (not nominations) in movie_awards.** Simpler storage.
Rejected because nomination data enables richer search ("nominated for
Oscar", "multiple Grammy nominations") at trivial storage cost.

## Consequences
- Award ceremony string changes in IMDB scraper output require `AwardCeremony`
  enum updates. The `ceremony_id` is returned as `None` for unknown
  ceremonies rather than raising, so new ceremonies are silently dropped
  until the enum is updated.
- `award_ceremony_win_ids` uses `SMALLINT[]` (not `INT[]`), so it cannot
  use the `gin__int_ops` extension operator class. This is a known
  deviation from the `genre_ids`/`keyword_ids` pattern.
- IMDB can return duplicate award entries for the same (ceremony, award_name,
  category, year) when multiple people in the same category share a
  nomination. Deduplication keeps the best outcome_id.

## References
- `schemas/enums.py` — `AwardCeremony`, `AwardOutcome`
- `schemas/imdb_models.py` — `AwardNomination`
- `db/init/01_create_postgres_tables.sql` — `movie_awards` DDL
- `db/postgres.py` — `batch_upsert_movie_awards()`
- `movie_ingestion/final_ingestion/ingest_movie.py` — `ingest_movie_awards()`
- `docs/decisions/ADR-011-data-store-architecture.md` — overall data-store architecture
