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
  imdb_vote_count     INT NOT NULL DEFAULT 0,
  popularity_score    FLOAT NOT NULL DEFAULT 0.0,
  reception_score     FLOAT,
  budget_bucket       TEXT,
  box_office_bucket   TEXT,
  title_token_count   INT NOT NULL DEFAULT 0,
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

-- Title-token-only string subset for fuzzy matching scope.
CREATE TABLE IF NOT EXISTS lex.title_token_strings (
  string_id  BIGINT PRIMARY KEY REFERENCES lex.lexical_dictionary(string_id) ON DELETE CASCADE,
  norm_str   TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_title_token_strings_trgm
  ON lex.title_token_strings USING GIN (norm_str gin_trgm_ops);

-- Inverted index postings for title tokens.
CREATE TABLE IF NOT EXISTS lex.inv_title_token_postings (
  term_id   BIGINT NOT NULL,
  movie_id  BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_title_postings_movie
  ON lex.inv_title_token_postings (movie_id);

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

-- Inverted index postings for character names so we can LIKE match
CREATE TABLE IF NOT EXISTS lex.inv_character_postings (
  term_id   BIGINT NOT NULL,
  movie_id  BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_character_postings_movie
  ON lex.inv_character_postings (movie_id);

-- Inverted index postings for studio names (legacy v1 path). New movie
-- ingestions no longer write here — the brand + freeform paths below have
-- replaced it. Kept in the schema so v1 compound lexical search and the
-- v2 stage-3 studio entity lookup can continue to serve from the frozen
-- snapshot until the query-side cutover retires them. Will be dropped as
-- part of that cutover, not this change.
CREATE TABLE IF NOT EXISTS lex.inv_studio_postings (
  term_id   BIGINT NOT NULL,
  movie_id  BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_studio_postings_movie
  ON lex.inv_studio_postings (movie_id);

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

-- Inverted index postings for franchise lineage/shared-universe names.
CREATE TABLE IF NOT EXISTS lex.inv_franchise_postings (
  term_id   BIGINT NOT NULL,
  movie_id  BIGINT NOT NULL,
  PRIMARY KEY (term_id, movie_id)
);

CREATE INDEX IF NOT EXISTS idx_franchise_postings_movie
  ON lex.inv_franchise_postings (movie_id);

-- Materialized view used for max_df stop-word filtering in title matching.
CREATE MATERIALIZED VIEW IF NOT EXISTS lex.title_token_doc_frequency AS
SELECT
  term_id,
  COUNT(*)::BIGINT AS doc_frequency,
  now()            AS updated_at
FROM lex.inv_title_token_postings
GROUP BY term_id;

CREATE UNIQUE INDEX IF NOT EXISTS idx_title_token_df_term_id
  ON lex.title_token_doc_frequency (term_id);
