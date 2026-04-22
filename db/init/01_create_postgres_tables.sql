-- Postgres bootstrap for movie-finder-rag.
-- This script is executed on first Docker initialization.

-- Required extensions for lexical search behavior and int[] GIN operator class.
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS intarray;

-- Auto-kill connections that sit idle in an open transaction for more than 2
-- minutes.  Prevents zombie transactions (e.g. from a crashed ingestion run)
-- from holding row-level locks indefinitely.
ALTER DATABASE moviedb SET idle_in_transaction_session_timeout = '2min';

-- Lexical search schema.
CREATE SCHEMA IF NOT EXISTS lex;

-- Canonical movie metadata used for card rendering, filtering, and reranking.
CREATE TABLE IF NOT EXISTS public.movie_card (
  movie_id            BIGINT PRIMARY KEY,
  title               TEXT NOT NULL,
  -- Normalized form of title (via implementation.misc.helpers.normalize_string).
  -- Stage 3 title_pattern lookups run ILIKE against this column so
  -- ingest-time and query-time normalization are symmetric (casing,
  -- diacritics, and most punctuation are collapsed; ASCII hyphens are
  -- preserved). Defaults to '' so the NOT NULL constraint is satisfied
  -- for any row inserted without an explicit value; ingestion always
  -- populates it.
  title_normalized    TEXT NOT NULL DEFAULT '',
  poster_url          TEXT,
  release_ts          BIGINT,
  runtime_minutes     INT,
  maturity_rank       SMALLINT,
  genre_ids           INT[] NOT NULL DEFAULT '{}',
  watch_offer_keys    INT[] NOT NULL DEFAULT '{}',
  audio_language_ids  INT[] NOT NULL DEFAULT '{}',
  country_of_origin_ids         INT[] NOT NULL DEFAULT '{}',
  source_material_type_ids INT[] NOT NULL DEFAULT '{}',
  keyword_ids         INT[] NOT NULL DEFAULT '{}',
  concept_tag_ids     INT[] NOT NULL DEFAULT '{}',
  award_ceremony_win_ids SMALLINT[] NOT NULL DEFAULT '{}',
  -- Normalized IMDB production_company IDs that this movie credits. Used by
  -- the freeform studio path to intersect (token → production_company_id)
  -- against (movie → production_company_ids). BIGINT[] because company IDs
  -- are BIGINT identities; no gin__int_ops (int[]-only opclass) — plain GIN
  -- works on bigint[].
  production_company_ids BIGINT[] NOT NULL DEFAULT '{}',
  -- Franchise lineage entry IDs — the direct sequel/prequel/reboot chain
  -- the movie belongs to (from movie_franchise_metadata.lineage). Stored
  -- separately from shared-universe entry IDs so stage-3 can score
  -- lineage matches higher than universe-only matches when the query
  -- spec sets prefer_lineage. Stage-3 retrieval resolves a query name
  -- through lex.franchise_token → lex.franchise_entry → this array (and
  -- shared_universe_entry_ids below) via the && GIN overlap operator.
  -- BIGINT[] to match franchise_entry_id width; plain GIN (no
  -- gin__int_ops, which is INT[]-only).
  lineage_entry_ids       BIGINT[] NOT NULL DEFAULT '{}',
  -- Shared-universe entry IDs — the broader multi-film universe the
  -- movie belongs to (from movie_franchise_metadata.shared_universe).
  -- A single franchise_entry_id can legitimately appear in this column
  -- for one movie (e.g., Puss in Boots carrying "Shrek" as its shared
  -- universe) and in lineage_entry_ids for another movie (e.g., Shrek 2
  -- carrying "Shrek" as its lineage); that's what makes
  -- lineage-vs-universe scoring work at query time.
  shared_universe_entry_ids BIGINT[] NOT NULL DEFAULT '{}',
  -- Subgroup entry IDs (one per element of movie_franchise_metadata.
  -- recognized_subgroups). Intersected with lineage_entry_ids /
  -- shared_universe_entry_ids at search time when the query carries
  -- both a lineage/universe name and a subgroup name (e.g., "MCU
  -- Phase One movies").
  subgroup_entry_ids     BIGINT[] NOT NULL DEFAULT '{}',
  imdb_vote_count     INT NOT NULL DEFAULT 0,
  popularity_score    FLOAT NOT NULL DEFAULT 0.0,
  reception_score     FLOAT,
  budget_bucket       TEXT,
  box_office_bucket   TEXT,
  updated_at          TIMESTAMP NOT NULL DEFAULT now(),
  created_at          TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_movie_card_range
  ON public.movie_card (release_ts, runtime_minutes, maturity_rank);

CREATE INDEX IF NOT EXISTS idx_movie_card_release_ts
  ON public.movie_card (release_ts);

CREATE INDEX IF NOT EXISTS idx_movie_card_runtime_minutes
  ON public.movie_card (runtime_minutes);

CREATE INDEX IF NOT EXISTS idx_movie_card_maturity_rank
  ON public.movie_card (maturity_rank);

-- Trigram GIN for ILIKE '%...%' containment matching (Stage 3
-- title_pattern endpoint's CONTAINS mode). Runs on the normalized
-- column so query-time normalize_string() on the pattern stays
-- symmetric with the stored form.
CREATE INDEX IF NOT EXISTS idx_movie_card_title_normalized_trgm
  ON public.movie_card USING GIN (title_normalized gin_trgm_ops);

-- Btree with text_pattern_ops accelerates LIKE '...%' starts-with
-- queries (Stage 3 title_pattern STARTS_WITH mode) without the trgm
-- index's per-match cost. Postgres picks the better index per query.
CREATE INDEX IF NOT EXISTS idx_movie_card_title_normalized_prefix
  ON public.movie_card (title_normalized text_pattern_ops);

-- Explicit gin__int_ops for efficient integer-array overlap filtering (&&).
CREATE INDEX IF NOT EXISTS idx_movie_card_genre_ids
  ON public.movie_card USING GIN (genre_ids gin__int_ops);

CREATE INDEX IF NOT EXISTS idx_movie_card_watch_offer_keys
  ON public.movie_card USING GIN (watch_offer_keys gin__int_ops);

CREATE INDEX IF NOT EXISTS idx_movie_card_audio_language_ids
  ON public.movie_card USING GIN (audio_language_ids gin__int_ops);

CREATE INDEX IF NOT EXISTS idx_movie_card_country_of_origin_ids
  ON public.movie_card USING GIN (country_of_origin_ids gin__int_ops);

CREATE INDEX IF NOT EXISTS idx_movie_card_source_material_type_ids
  ON public.movie_card USING GIN (source_material_type_ids gin__int_ops);

CREATE INDEX IF NOT EXISTS idx_movie_card_keyword_ids
  ON public.movie_card USING GIN (keyword_ids gin__int_ops);

CREATE INDEX IF NOT EXISTS idx_movie_card_concept_tag_ids
  ON public.movie_card USING GIN (concept_tag_ids gin__int_ops);

CREATE INDEX IF NOT EXISTS idx_movie_card_award_ceremony_win_ids
  ON public.movie_card USING GIN (award_ceremony_win_ids);

-- Production-company membership. Plain GIN (no gin__int_ops) because the
-- column is BIGINT[].
CREATE INDEX IF NOT EXISTS idx_movie_card_production_company_ids
  ON public.movie_card USING GIN (production_company_ids);

-- Franchise lineage membership. Plain GIN (BIGINT[]).
CREATE INDEX IF NOT EXISTS idx_movie_card_lineage_entry_ids
  ON public.movie_card USING GIN (lineage_entry_ids);

-- Shared-universe membership. Plain GIN (BIGINT[]).
CREATE INDEX IF NOT EXISTS idx_movie_card_shared_universe_entry_ids
  ON public.movie_card USING GIN (shared_universe_entry_ids);

-- Subgroup membership. Plain GIN (BIGINT[]).
CREATE INDEX IF NOT EXISTS idx_movie_card_subgroup_entry_ids
  ON public.movie_card USING GIN (subgroup_entry_ids);


-- Structured award nominations and wins for deterministic lookup.
-- Queried by ceremony_id + award_name + category + outcome_id, optionally filtered by year.
CREATE TABLE IF NOT EXISTS public.movie_awards (
  movie_id          BIGINT NOT NULL REFERENCES public.movie_card ON DELETE CASCADE,
  ceremony_id       SMALLINT NOT NULL,
  award_name        TEXT NOT NULL,
  category          TEXT NOT NULL DEFAULT '',
  -- Concept-tag ids derived from `category` via consolidate_award_categories.py
  -- and the 3-level taxonomy in schemas/award_category_tags.py. Stores the leaf
  -- tag plus every ancestor (mid, group), so a single GIN-indexed && lookup
  -- handles queries at any specificity ("any acting award" via the group id,
  -- "Best Actor" via the leaf id, etc.).
  category_tag_ids  INT[] NOT NULL DEFAULT '{}',
  outcome_id        SMALLINT NOT NULL,
  year              SMALLINT NOT NULL,
  -- Normalized-form resolution for the freeform award-name path. Written
  -- at ingest time by the same upsert that creates the row, using the
  -- entry id from lex.award_name_entry. Nullable to tolerate the rare
  -- case where normalize_award_string collapses to an empty string
  -- (punctuation-only names); query-side posting-list joins simply omit
  -- those rows. No FK declared here to avoid a forward-reference on the
  -- lex.award_name_entry table defined later in this file — matches the
  -- loose-reference convention already used for ceremony_id, outcome_id,
  -- and category_tag_ids.
  award_name_entry_id INT,
  PRIMARY KEY (movie_id, ceremony_id, award_name, category, year)
);

-- Covers: "Oscar winners", "Palme d'Or winners", "Best Picture nominees", "Cannes winners after 2000"
CREATE INDEX IF NOT EXISTS idx_awards_lookup
  ON public.movie_awards (ceremony_id, award_name, category, outcome_id, year);

-- Reverse lookup: given a movie, find all its awards (for display / card rendering)
CREATE INDEX IF NOT EXISTS idx_awards_movie
  ON public.movie_awards (movie_id);

-- Concept-tag overlap filter for the Stage-3 awards endpoint.
-- Lets queries like "any acting award" (tag id 10000) or "Best Actor"
-- (tag id 1) hit a single indexed && lookup regardless of specificity.
CREATE INDEX IF NOT EXISTS idx_awards_category_tag_ids
  ON public.movie_awards USING GIN (category_tag_ids gin__int_ops);

-- Freeform award-name entry resolution. Post-token-intersection, stage 3
-- filters movie_awards by WHERE award_name_entry_id = ANY(:ids), so a
-- btree here keeps that lookup cheap without carrying the GIN overhead
-- of the category-tag path.
CREATE INDEX IF NOT EXISTS idx_awards_entry
  ON public.movie_awards (award_name_entry_id);

-- Structured franchise metadata for search-time franchise retrieval/filtering.
CREATE TABLE IF NOT EXISTS public.movie_franchise_metadata (
  movie_id               BIGINT PRIMARY KEY REFERENCES public.movie_card ON DELETE CASCADE,
  lineage                TEXT,
  shared_universe        TEXT,
  recognized_subgroups   TEXT[] NOT NULL DEFAULT '{}',
  launched_subgroup      BOOLEAN NOT NULL DEFAULT FALSE,
  lineage_position       SMALLINT,
  is_spinoff             BOOLEAN NOT NULL DEFAULT FALSE,
  is_crossover           BOOLEAN NOT NULL DEFAULT FALSE,
  launched_franchise     BOOLEAN NOT NULL DEFAULT FALSE
);


-- Global dictionary of normalized lexical strings.
CREATE TABLE IF NOT EXISTS lex.lexical_dictionary (
  string_id   BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  norm_str    TEXT NOT NULL UNIQUE,
  created_at  TIMESTAMP NOT NULL DEFAULT now()
);

-- Inverted index postings for actor names (with billing metadata for prominence scoring).
CREATE TABLE IF NOT EXISTS lex.inv_actor_postings (
  term_id          BIGINT NOT NULL,
  movie_id         BIGINT NOT NULL,
  billing_position INT    NOT NULL,
  cast_size        INT    NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_actor_postings_movie
  ON lex.inv_actor_postings (movie_id);

-- Inverted index postings for director names.
CREATE TABLE IF NOT EXISTS lex.inv_director_postings (
  term_id   BIGINT NOT NULL,
  movie_id  BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_director_postings_movie
  ON lex.inv_director_postings (movie_id);

-- Inverted index postings for writer names.
CREATE TABLE IF NOT EXISTS lex.inv_writer_postings (
  term_id   BIGINT NOT NULL,
  movie_id  BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_writer_postings_movie
  ON lex.inv_writer_postings (movie_id);

-- Inverted index postings for producer names.
CREATE TABLE IF NOT EXISTS lex.inv_producer_postings (
  term_id   BIGINT NOT NULL,
  movie_id  BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_producer_postings_movie
  ON lex.inv_producer_postings (movie_id);

-- Inverted index postings for composer names.
CREATE TABLE IF NOT EXISTS lex.inv_composer_postings (
  term_id   BIGINT NOT NULL,
  movie_id  BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_composer_postings_movie
  ON lex.inv_composer_postings (movie_id);

-- Character names strings saving
CREATE TABLE IF NOT EXISTS lex.character_strings (
  string_id  BIGINT PRIMARY KEY REFERENCES lex.lexical_dictionary(string_id) ON DELETE CASCADE,
  norm_str   TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_character_strings_trgm
  ON lex.character_strings USING GIN (norm_str gin_trgm_ops);

-- Inverted index postings for character names (with billing metadata
-- for prominence scoring). Analogous to inv_actor_postings, but with
-- a distinct character_cast_size because characters are not 1:1 with
-- actors (aliases like "Peter Parker" + "Spider-Man" produce multiple
-- character rows for a single cast edge).
CREATE TABLE IF NOT EXISTS lex.inv_character_postings (
  term_id              BIGINT NOT NULL,
  movie_id             BIGINT NOT NULL,
  billing_position     INT    NOT NULL,
  character_cast_size  INT    NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_character_postings_movie
  ON lex.inv_character_postings (movie_id);

-- Normalized production-company registry. One row per distinct normalized
-- IMDB `production_companies` string. `canonical_string` preserves the first
-- raw form encountered (for display / debugging); `normalized_string` is the
-- lookup key produced by normalize_company_string() — normalize_string plus
-- the ordinal number-to-word rule (20th → twentieth) so numeric and worded
-- variants collide to the same row.
CREATE TABLE IF NOT EXISTS lex.production_company (
  production_company_id  BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  canonical_string       TEXT NOT NULL,
  normalized_string      TEXT NOT NULL UNIQUE
);

-- Token → production_company_id inverted index for the freeform studio path.
-- Every token produced by tokenize_company_string is recorded (no DF ceiling
-- at ingest). Query-side drops over-the-ceiling tokens using a percentage of
-- the dataset size. GIN on token for fast posting-list fetch.
CREATE TABLE IF NOT EXISTS lex.studio_token (
  token                  TEXT   NOT NULL,
  production_company_id  BIGINT NOT NULL REFERENCES lex.production_company ON DELETE CASCADE,
  PRIMARY KEY (token, production_company_id)
);

CREATE INDEX IF NOT EXISTS idx_studio_token_company
  ON lex.studio_token (production_company_id);

-- Brand → movie postings with prominence. `first_matching_index` is the
-- position of the brand's first matching company string in the movie's IMDB
-- production_companies list (0 = top credit). `total_brand_count` is the
-- number of distinct brands that tagged this movie, useful as a normalizer
-- when scoring prominence. Analogue of inv_actor_postings.billing_position /
-- cast_size.
CREATE TABLE IF NOT EXISTS lex.inv_production_brand_postings (
  brand_id              SMALLINT NOT NULL,
  movie_id              BIGINT   NOT NULL,
  first_matching_index  SMALLINT NOT NULL,
  total_brand_count     SMALLINT NOT NULL,
  PRIMARY KEY (brand_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_brand_postings_movie
  ON lex.inv_production_brand_postings (movie_id);

-- Materialized view used for DF-ceiling stop-word filtering in the freeform
-- studio path. DF is measured per canonical production_company string — since
-- lex.studio_token has PRIMARY KEY (token, production_company_id), a row
-- count per token is exactly the number of distinct production companies
-- containing that token. Query-side filters tokens whose doc_frequency
-- exceeds the empirically-chosen ceiling (see
-- search_improvement_planning/v2_search_data_improvements.md, "DF Ceiling
-- Determination").
CREATE MATERIALIZED VIEW IF NOT EXISTS lex.studio_token_doc_frequency AS
SELECT
  token,
  COUNT(*)::BIGINT AS doc_frequency,
  now()            AS updated_at
FROM lex.studio_token
GROUP BY token;

CREATE UNIQUE INDEX IF NOT EXISTS idx_studio_token_df_token
  ON lex.studio_token_doc_frequency (token);

-- Normalized franchise-entry registry. One row per distinct normalized
-- string seen in movie_franchise_metadata.lineage, .shared_universe, or any
-- element of .recognized_subgroups. `canonical_string` preserves the first
-- raw form encountered (for display / debugging); `normalized_string` is the
-- lookup key produced by normalize_franchise_string() — shared normalize_string
-- plus the ordinal and cardinal number-to-word rules so variants like
-- "phase 1"/"phase one" and "the lord of the rings"/"lord of the rings"
-- (after DF-ceiling filtering) collide to the same row. Lineage, universe,
-- and subgroup strings share this table: we never filter by where a string
-- came from at retrieval time, so separating them buys nothing.
CREATE TABLE IF NOT EXISTS lex.franchise_entry (
  franchise_entry_id  BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  canonical_string    TEXT NOT NULL,
  normalized_string   TEXT NOT NULL UNIQUE
);

-- Token → franchise_entry_id inverted index. Every token produced by
-- tokenize_franchise_string is recorded (no DF ceiling at ingest). Query-side
-- drops tokens whose doc_frequency exceeds the empirical ceiling, selected
-- after backfill (see search_improvement_planning/v2_search_data_improvements.md,
-- "Franchise Resolution" Stage C). PRIMARY KEY (token, franchise_entry_id) so a
-- row count per token is exactly the number of distinct franchise_entries that
-- contain the token — the DF denominator. Secondary index on franchise_entry_id
-- supports the materialized-view rebuild and ON DELETE CASCADE lookups.
CREATE TABLE IF NOT EXISTS lex.franchise_token (
  token               TEXT   NOT NULL,
  franchise_entry_id  BIGINT NOT NULL REFERENCES lex.franchise_entry ON DELETE CASCADE,
  PRIMARY KEY (token, franchise_entry_id)
);

CREATE INDEX IF NOT EXISTS idx_franchise_token_entry
  ON lex.franchise_token (franchise_entry_id);

-- Materialized view used for DF-ceiling stop-word filtering on the franchise
-- path. Same shape / rationale as lex.studio_token_doc_frequency — COUNT(*)
-- per token is exactly the number of distinct franchise_entry rows that
-- contain the token because of the (token, franchise_entry_id) PK upstream.
-- Refreshed CONCURRENTLY post-bulk-ingest; requires the unique index below.
CREATE MATERIALIZED VIEW IF NOT EXISTS lex.franchise_token_doc_frequency AS
SELECT
  token,
  COUNT(*)::BIGINT AS doc_frequency,
  now()            AS updated_at
FROM lex.franchise_token
GROUP BY token;

CREATE UNIQUE INDEX IF NOT EXISTS idx_franchise_token_df_token
  ON lex.franchise_token_doc_frequency (token);

-- Normalized award-name registry. One row per distinct normalized string
-- seen in public.movie_awards.award_name. `normalized` is the lookup key
-- produced by normalize_award_string() — shared normalize_string plus the
-- ordinal/cardinal number-to-word rules used by studio and franchise, so
-- surface variants (straight vs curly apostrophe, "Critics Week" vs
-- "Critics' Week", case/diacritic differences) collapse to a single
-- entry. Raw surface forms are preserved on movie_awards.award_name for
-- display/debug; they are not duplicated on the entry row. Entry id fits
-- in INT — ~600 distinct names observed today, no realistic path to 2B.
CREATE TABLE IF NOT EXISTS lex.award_name_entry (
  award_name_entry_id  INT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,
  normalized           TEXT NOT NULL UNIQUE
);

-- Token → award_name_entry_id inverted index for the freeform award-name
-- path. Every token produced by tokenize_award_string is recorded here
-- — no stoplist is applied at ingest. Query-side will drop above-ceiling
-- (and/or explicitly listed) tokens once the ceiling is selected
-- empirically against the materialized view below, matching how studio
-- and franchise stage the same decision. PRIMARY KEY
-- (token, award_name_entry_id) so COUNT(*) per token in the DF view is
-- exactly the number of distinct entries carrying the token.
CREATE TABLE IF NOT EXISTS lex.award_name_token (
  token                TEXT NOT NULL,
  award_name_entry_id  INT  NOT NULL REFERENCES lex.award_name_entry ON DELETE CASCADE,
  PRIMARY KEY (token, award_name_entry_id)
);

CREATE INDEX IF NOT EXISTS idx_award_name_token_entry
  ON lex.award_name_token (award_name_entry_id);

-- Materialized view for DF-ceiling stop-word filtering on the award-name
-- path. Same shape / rationale as lex.studio_token_doc_frequency and
-- lex.franchise_token_doc_frequency. Refreshed CONCURRENTLY post-ingest;
-- requires the unique index below.
CREATE MATERIALIZED VIEW IF NOT EXISTS lex.award_name_token_doc_frequency AS
SELECT
  token,
  COUNT(*)::BIGINT AS doc_frequency,
  now()            AS updated_at
FROM lex.award_name_token
GROUP BY token;

CREATE UNIQUE INDEX IF NOT EXISTS idx_award_name_token_df_token
  ON lex.award_name_token_doc_frequency (token);
