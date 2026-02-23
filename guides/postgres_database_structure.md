# Postgres Database Structure Guide

This document is the authoritative reference for the Postgres database used by the movie search application. It is written for engineers ramping onto the project who have no prior context.

---

## Overview

Postgres serves two distinct purposes in this system:

1. **Canonical movie metadata** — a single "thin" table (`public.movie_card`) that stores the minimal set of fields required to render search result cards and compute reranking scores. This is queried in bulk once per search request after candidates have been retrieved from lexical and vector search.

2. **Lexical search** — a dedicated `lex` schema containing an inverted index system that supports keyword-style candidate retrieval. This runs in parallel with vector (Qdrant) search and contributes its own scored candidate set.

Postgres does **not** store full movie detail data (handled by TMDB + Redis) and does **not** store vector embeddings (handled by Qdrant). Its footprint is intentionally minimal — estimated RAM usage is 200–400 MB on the t3.large instance where it runs as a self-hosted Docker container.

A daily `pg_dump` is shipped to S3 for backup.

---

## Schemas

| Schema | Purpose |
|--------|---------|
| `public` | Canonical movie metadata for card rendering and reranking |
| `lex` | Lexical search — inverted indices, dictionary, and filter metadata |

---

## Required Extensions

These extensions must be installed before creating any `lex` schema objects. They enable fuzzy title-token matching and efficient integer array operations.

```sql
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS intarray;  -- Required for gin__int_ops on integer arrays
```

```sql
CREATE SCHEMA IF NOT EXISTS lex;
```

---

---

# Part 1 — `public` Schema: Movie Card Metadata

---

## `public.movie_card`

### What it is

The single source of truth for minimal movie metadata within the application. It is **not** a full movie detail store — that data lives in TMDB and is cached in Redis. This table contains exactly what is needed to:

- Render a search result card (title, year, poster)
- Compute the metadata preference component of the reranking score (runtime, release year, genre, maturity, watch offers, audio language, reception score)
- Construct the eligible candidate set for lexical hard filtering

### How it's used

During every `/search` request, after lexical and vector candidates have been gathered and merged, a single bulk Postgres query fetches metadata for all candidates at once:

```sql
SELECT movie_id, title, year, poster_url, release_ts, runtime_minutes,
       maturity_rank, genre_ids, watch_offer_keys, audio_language_ids, reception_score
FROM public.movie_card
WHERE movie_id = ANY($1)
```

This is done with a single query — never per-candidate. The results feed both the reranker (scoring) and the final API response payload (card rendering).

This table is also used during lexical search to build the **eligible set** — the subset of movies that pass hard metadata filters (release date range, runtime, maturity, genre, watch availability) before posting-list joins occur.

`title_token_count` is stored here to support title scoring in lexical search, where the scorer needs to know how many tokens the movie's title contains.

### Schema

```sql
CREATE TABLE public.movie_card (
  movie_id            BIGINT PRIMARY KEY,    -- Internal ID; equals Qdrant point_id and TMDB ID
  title               TEXT NOT NULL,
  year                INT,                   -- Release year (display use)
  poster_url          TEXT,                  -- TMDB poster path or full URL for card rendering

  -- Metadata used for hard filters and reranking preference scoring
  release_ts          BIGINT,                -- Unix timestamp (midnight UTC) of release date
  runtime_minutes     INT,
  maturity_rank       SMALLINT,              -- Ordinal encoding of content rating (G=1, PG=2, etc.)
  genre_ids           INT[]    NOT NULL DEFAULT '{}',
  watch_offer_keys    INT[]    NOT NULL DEFAULT '{}', -- Encoded (provider_id, method_id) pairs; see encoding note below
  audio_language_ids  INT[]    NOT NULL DEFAULT '{}',
  reception_score     FLOAT,                 -- Precomputed from IMDb/Metacritic; used in metadata preference scoring

  -- Lexical title scoring input
  title_token_count   INT      NOT NULL DEFAULT 0,   -- Number of tokens in the normalized movie title

  updated_at          TIMESTAMP NOT NULL DEFAULT now(),
  created_at          TIMESTAMP NOT NULL DEFAULT now()
);

-- Composite index for eligible-set construction under multi-filter queries.
-- When a user applies release year + runtime + maturity together (the common UI case),
-- this lets the planner satisfy all three range conditions in a single index scan
-- rather than choosing one single-column index and filtering the rest.
CREATE INDEX idx_movie_card_range
  ON public.movie_card (release_ts, runtime_minutes, maturity_rank);

-- Individual range indexes retained as fallback for single-filter queries
-- where the composite index would be less selective.
CREATE INDEX idx_movie_card_release_ts      ON public.movie_card (release_ts);
CREATE INDEX idx_movie_card_runtime_minutes ON public.movie_card (runtime_minutes);
CREATE INDEX idx_movie_card_maturity_rank   ON public.movie_card (maturity_rank);

-- GIN indexes for array overlap filters (genre and watch availability).
-- gin__int_ops is specified explicitly — it is the correct operator class for
-- integer arrays and ensures the && overlap operator is handled efficiently.
-- Requires the intarray extension (see prerequisites above).
CREATE INDEX idx_movie_card_genre_ids
  ON public.movie_card USING GIN (genre_ids gin__int_ops);

CREATE INDEX idx_movie_card_watch_offer_keys
  ON public.movie_card USING GIN (watch_offer_keys gin__int_ops);
```

### `watch_offer_keys` encoding

Watch availability is expressed as paired constraints — for example, "available to rent on Amazon" is a specific combination of provider and method. Storing provider IDs and method IDs in separate arrays would lose the pairing (you couldn't distinguish "Amazon stream OR Netflix rent" from "Amazon rent"). Instead, each pair is encoded into a single integer:

```
method IDs:  stream=1, rent=2, buy=3
watch_offer_key = (provider_id << 2) | method_id
```

The bit-shift by 2 works because there are only 3 method values, which fit in 2 bits. The lower 2 bits always encode the method; all higher bits encode the provider. No two valid combinations produce the same integer, and provider IDs have room for ~500 million values before any overflow risk.

Concrete example with Netflix (`provider_id = 8`):

```
Netflix stream: (8 << 2) | 1 = 32 | 1 = 33
Netflix rent:   (8 << 2) | 2 = 32 | 2 = 34
Netflix buy:    (8 << 2) | 3 = 32 | 3 = 35
```

At query time, the server expands the user's filter into a list of encoded keys and uses a GIN overlap check (`&&`). Examples:

- "Netflix rent only" → `[34]`
- "(Netflix OR Amazon) AND (stream OR rent)" → cross-product of 4 keys, expanded in server code

### Update cadence

| Field | Updated by |
|-------|-----------|
| `watch_offer_keys` | Weekly background job |
| All other fields | Daily new-movie ingestion job |

---

---

# Part 2 — `lex` Schema: Lexical Search

---

## Architecture overview

The lexical search system is a custom inverted index built on top of Postgres. It covers four "buckets" of matchable entities:

| Bucket | Match type | Table |
|--------|-----------|-------|
| Movie title tokens | Fuzzy (edit distance ≤ 1) | `lex.inv_title_token_postings` |
| People (actors, directors, writers, composers, producers) | Exact phrase | `lex.inv_person_postings` |
| Characters | Exact phrase | `lex.inv_character_postings` |
| Studios (production companies) | Exact phrase | `lex.inv_studio_postings` |

A movie becomes a **lexical candidate** if it matches at least one bucket (OR semantics). Hard metadata filters (release date, runtime, maturity, genre, watch offers) are applied first as an **AND** pre-filter to narrow the search space before posting-list joins.

All string terms — whether title tokens or entity phrases — are normalized and stored as integer IDs via a global dictionary (`lex.lexical_dictionary`). This dictionary is the backbone of the system.

---

## `lex.lexical_dictionary`

### What it is

A global, stable mapping from a normalized string to an integer `string_id`. Every term that appears in any posting table is registered here first. All posting tables store `term_id` (i.e., `string_id`) rather than raw strings.

### Why it exists

- **Compact storage**: posting tables store `BIGINT` IDs instead of variable-length `TEXT`, saving significant space across millions of rows.
- **Fast exact lookup**: people, character, and studio phrases are resolved to their `string_id` at query time and used for equality joins.

The normalizer must be **identical** at ingest time and query time. Any drift between the two will cause misses.

### Schema

```sql
CREATE TABLE lex.lexical_dictionary (
  string_id   BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  norm_str    TEXT NOT NULL UNIQUE,   -- UNIQUE implicitly creates a B-tree index; no separate index needed
  created_at  TIMESTAMP NOT NULL DEFAULT now()
);
```

> **Note:** No additional index is created on `norm_str` — the `UNIQUE` constraint already creates a B-tree index that serves exact lookups. The trigram index for fuzzy matching lives on `lex.title_token_strings` (see below), not here. Adding a redundant index would double write cost on every dictionary insert with no query benefit.

---

## `lex.title_token_strings`

### What it is

A dedicated lookup table containing only the normalized strings that are used as **title tokens** — a strict subset of `lex.lexical_dictionary`. This is the sole target for fuzzy title-token matching at query time.

### Why it exists

Fuzzy matching must be restricted to title token strings only. The naive approach — running trigram + Levenshtein against the full `lex.lexical_dictionary` — has two problems:

1. **False positives**: people names, studio names, and character phrases that happen to be similar to the query token would be returned as fuzzy matches.
2. **Unnecessary cost**: the trigram scan runs against a much larger string set than needed.

By maintaining a separate table for title token strings and placing the trigram GIN index here, fuzzy lookup is both fast and correctly scoped. `string_id` is a foreign key to `lex.lexical_dictionary`, so all ID references remain consistent across the system.

### Schema

```sql
CREATE TABLE lex.title_token_strings (
  string_id  BIGINT PRIMARY KEY REFERENCES lex.lexical_dictionary(string_id) ON DELETE CASCADE,
  norm_str   TEXT NOT NULL UNIQUE   -- Denormalized from lexical_dictionary for index locality
);

-- Trigram GIN index — used exclusively for fuzzy title-token shortlist generation.
-- At query time: find candidate string_ids via trigram similarity, then confirm
-- with Levenshtein ≤ 1. Only title token strings are indexed here.
CREATE INDEX idx_title_token_strings_trgm
  ON lex.title_token_strings USING GIN (norm_str gin_trgm_ops);
```

### Population

A row is inserted here whenever a new title token string is added to `lex.lexical_dictionary` during movie ingest. Deletions cascade from the dictionary via the foreign key.

---

## `lex.title_token_doc_frequency` (materialized view)

### What it is

A materialized view that counts how many movies each title token appears in (`doc_frequency`). Refreshed as part of the daily ingestion job.

### Why it exists

Tokens that appear in more than 10,000 movie titles are too common to be useful — they are effectively stop words for title matching (e.g., "the", "a", "of"). This view is the authoritative source for filtering them out at query time. Any `string_id` with `doc_frequency > 10,000` is excluded from title scoring regardless of what the fuzzy lookup returns.

> **Note:** With the introduction of `lex.title_token_strings`, this view no longer serves the secondary purpose of restricting fuzzy matching to title tokens — that is now handled by scoping the trigram index to `title_token_strings` directly. The view's sole remaining role is `max_df` enforcement.

### Schema

```sql
CREATE MATERIALIZED VIEW lex.title_token_doc_frequency AS
SELECT
  term_id,
  COUNT(*)::BIGINT AS doc_frequency,
  now()            AS updated_at
FROM lex.inv_title_token_postings
GROUP BY term_id;

CREATE UNIQUE INDEX idx_title_token_df_term_id
  ON lex.title_token_doc_frequency (term_id);
```

### Refresh

```sql
REFRESH MATERIALIZED VIEW CONCURRENTLY lex.title_token_doc_frequency;
```

Run this after each daily bulk ingest. The `CONCURRENTLY` option ensures reads are not blocked during the rebuild (requires the unique index above).

---

## Posting tables

Each posting table is an inverted index mapping `term_id → movie_id`. The design is **row-per-posting** (one row per term/movie pair) rather than storing arrays on the movie row. This makes it easy to:

- JOIN against an eligible-movie set
- Count distinct matched terms with `COUNT(DISTINCT term_id)`
- Delete or rebuild postings for a single movie (`DELETE WHERE movie_id = X`)

All four posting tables share the same structural pattern. `term_id` always references `lex.lexical_dictionary.string_id`.

### Clustering

After the initial bulk load, each posting table should be clustered by its primary key. This physically orders rows on disk by `(term_id, movie_id)`, so all postings for a given token are contiguous — reducing random I/O during posting list scans.

```sql
CLUSTER lex.inv_title_token_postings   USING inv_title_token_postings_pkey;
CLUSTER lex.inv_person_postings        USING inv_person_postings_pkey;
CLUSTER lex.inv_character_postings     USING inv_character_postings_pkey;
CLUSTER lex.inv_studio_postings        USING inv_studio_postings_pkey;
```

`CLUSTER` is a one-time physical reorder — it does not auto-maintain as new rows are inserted. Re-run periodically if write volume is high enough to significantly disorder the heap (unlikely at this scale with a daily batch ingest pattern).

---

### `lex.inv_title_token_postings`

#### What it is

The inverted index for movie title tokens. Each movie's normalized title is tokenized (words plus hyphen expansions), and one posting row is inserted per token per movie.

#### How it's used

At query time, the user's title-related query terms are tokenized and normalized. Each token is resolved to candidate `string_id` values via fuzzy lookup — trigram shortlist against `lex.title_token_strings.norm_str`, then Levenshtein ≤ 1 to confirm. Only `string_id` values with `doc_frequency ≤ 10,000` in `lex.title_token_doc_frequency` are used.

The posting join then counts how many query tokens matched each movie (`m`), along with the total query tokens (`k`) and the movie's total token count (`L` from `movie_card.title_token_count`). These feed the title scoring formula (β=2, threshold=0.15).

#### Schema

```sql
CREATE TABLE lex.inv_title_token_postings (
  term_id  BIGINT NOT NULL,
  movie_id BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX idx_title_postings_movie
  ON lex.inv_title_token_postings (movie_id);
```

---

### `lex.inv_person_postings`

#### What it is

The inverted index for people associated with a movie — covering actors, directors, writers, composers, and producers. Each person's normalized full name is treated as a single phrase (no tokenization). One posting row is inserted per person per movie.

#### How it's used

When a query contains named people, each name is normalized and looked up in `lex.lexical_dictionary` by exact match to get a `string_id`. The posting join counts `matched_people_count = COUNT(DISTINCT term_id)` for a movie, which feeds the people component of the lexical score.

#### Schema

```sql
CREATE TABLE lex.inv_person_postings (
  term_id  BIGINT NOT NULL,
  movie_id BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX idx_person_postings_movie
  ON lex.inv_person_postings (movie_id);
```

---

### `lex.inv_character_postings`

#### What it is

The inverted index for character names that appear in a movie. Each character name is a single phrase with exact matching only.

#### How it's used

Same pattern as people: exact dictionary lookup at query time, then `matched_character_count = COUNT(DISTINCT term_id)` in the posting join.

#### Schema

```sql
CREATE TABLE lex.inv_character_postings (
  term_id  BIGINT NOT NULL,
  movie_id BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX idx_character_postings_movie
  ON lex.inv_character_postings (movie_id);
```

---

### `lex.inv_studio_postings`

#### What it is

The inverted index for production company names (studios). Exact phrase matching only. Unlike people or titles, studios have no reception-based prior applied.

#### How it's used

Same pattern: exact dictionary lookup, then `matched_studio_count = COUNT(DISTINCT term_id)`.

#### Schema

```sql
CREATE TABLE lex.inv_studio_postings (
  term_id  BIGINT NOT NULL,
  movie_id BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX idx_studio_postings_movie
  ON lex.inv_studio_postings (movie_id);
```

---

## Reference / lookup tables (optional, non-critical)

These tables are not required for query execution — the server always passes pre-resolved integer IDs for genres, providers, methods, and maturity ratings. They exist purely for human introspection, admin tooling, and debugging.

```sql
CREATE TABLE lex.genre_dictionary (
  genre_id  INT  PRIMARY KEY,
  name      TEXT NOT NULL UNIQUE
);

CREATE TABLE lex.provider_dictionary (
  provider_id  INT  PRIMARY KEY,
  name         TEXT NOT NULL UNIQUE
);

CREATE TABLE lex.watch_method_dictionary (
  method_id  INT  PRIMARY KEY,
  name       TEXT NOT NULL UNIQUE   -- 'stream', 'rent', 'buy'
);

CREATE TABLE lex.maturity_dictionary (
  maturity_rank  SMALLINT PRIMARY KEY,
  label          TEXT NOT NULL UNIQUE  -- e.g., 'G', 'PG', 'PG-13', 'R'
);
```

---

---

# Part 3 — End-to-End Query Flow (Postgres's Role)

This section shows exactly where and how Postgres is touched during a single `/search` request.

### Step 1: Eligible set construction (lexical pre-filter)

If the query includes any hard metadata filters, the server builds a candidate-eligible set:

```sql
SELECT movie_id, title_token_count
FROM public.movie_card
WHERE
  release_ts BETWEEN $1 AND $2           -- AND if provided
  AND runtime_minutes BETWEEN $3 AND $4  -- AND if provided
  AND maturity_rank <= $5                -- AND if provided
  AND genre_ids && $6                    -- AND if provided (GIN overlap)
  AND watch_offer_keys && $7             -- AND if provided (GIN overlap)
```

When multiple range filters are active simultaneously — the common UI case — the planner uses `idx_movie_card_range` to satisfy all three scalar conditions in a single composite index scan, then applies GIN overlap filters on the result.

This result set is passed as a filter into the posting-table joins. Movies not in this set are excluded from lexical scoring regardless of what they match in the posting tables.

### Step 2: Posting-table joins (lexical scoring)

The server joins each active posting table against the eligible set, counting matched entities per movie. For title tokens, the join also computes the `m/k/L` inputs to the title scoring formula. Results from all four buckets are unioned (OR semantics) to produce the final set of lexical candidates.

Title token fuzzy resolution happens before this step: each query token is looked up against `lex.title_token_strings` (trigram shortlist → Levenshtein confirm → `max_df` filter), yielding a set of valid `string_id` values to use in the posting join.

### Step 3: Metadata enrichment (post-merge, all candidates)

After lexical and vector candidates are merged in-memory, a single bulk query retrieves metadata for all candidates:

```sql
SELECT movie_id, title, year, poster_url, release_ts, runtime_minutes,
       maturity_rank, genre_ids, watch_offer_keys, audio_language_ids, reception_score
FROM public.movie_card
WHERE movie_id = ANY($1)
```

This is the only time Postgres is queried for enrichment data. Results feed the reranker's metadata preference scoring and the final card payload returned to the client.

---

---

# Part 4 — Design Decisions Reference

| Decision | Rationale |
|----------|-----------|
| Row-per-posting in inverted indices | Simplifies joins, counting, and per-movie deletes vs. storing arrays on the movie row |
| Global dictionary with integer IDs | Compact posting tables; consistent IDs across ingest and query |
| Separate `lex.title_token_strings` table | Scopes trigram index to title tokens only — prevents fuzzy matching from touching people/studio/character strings and keeps the candidate shortlist small |
| No extra index on `lexical_dictionary.norm_str` | `UNIQUE` already creates a B-tree index; a second identical index wastes write overhead and cache with no query benefit |
| `max_df` materialized view | Enforces stop-word filtering for title tokens; sole responsibility now that fuzzy scoping is handled by `title_token_strings` |
| Composite range index on `movie_card` | Allows the planner to satisfy multi-filter eligible-set queries (release + runtime + maturity) in a single scan rather than falling back to per-column bitmap scans |
| Explicit `gin__int_ops` on array indexes | Ensures correct and efficient `&&` overlap behavior for integer arrays; avoids silent operator class mismatch across Postgres versions |
| `watch_offer_keys` encoded pairs | Preserves provider↔method co-occurrence; separate arrays cannot represent "rent on Amazon" correctly |
| `title_token_count` on `movie_card` | Collocated with other movie metadata; avoids a separate join during lexical scoring |
| Single table `movie_card` (no separate `lex.movies`) | One table serves both lexical pre-filtering and reranking enrichment; avoids data duplication and sync complexity |
| Single bulk enrichment query per request | Never query Postgres per-candidate; `WHERE movie_id = ANY($1)` with primary key index is fast at any realistic candidate set size |
| Cluster posting tables after bulk load | Places all postings for a given token physically adjacent on disk, reducing random I/O on long posting list scans |