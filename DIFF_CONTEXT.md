# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Documentation staleness fixes from full audit

Files: CLAUDE.md, docs/PROJECT.md, docs/conventions.md, docs/modules/classes.md, docs/modules/ingestion.md, docs/modules/api.md, docs/modules/db.md, docs/decisions/ADR-060-basemovie-to-movie-migration.md

Why: docs-auditor subagent identified 11 stale claims across permanent docs.

Fixes applied:
- Status chain updated from removed `phase1_complete`/`phase2_complete` to `metadata_generated` + `ingestion_failed` (conventions.md, CLAUDE.md)
- Stage 7 embedding corrected: it's integrated into Stage 8 inside `ingest_movie.py`, not a separate unimplemented step (CLAUDE.md, PROJECT.md, ingestion.md)
- `implementation/vectorize.py` correctly identified as legacy ChromaDB (CLAUDE.md, PROJECT.md, ingestion.md)
- Wrong filename `plot_quality_scores.py` → `plot_tmdb_quality_scores.py` (CLAUDE.md)
- BaseMovie no longer described as used in ingestion or db/ (classes.md, ADR-060)
- Dangling ADR-027 reference removed (ingestion.md)
- `cli_search.py` added to api.md key files table
- `batch_upsert_*_dictionary()` wording clarified to exclude `batch_upsert_lexical_dictionary` (db.md)

## Search system analysis and improvement planning

Files: search_improvement_planning/current_search_flaws.md, search_improvement_planning/types_of_searches.md, search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/open_questions.md

### Intent
Deep analysis of why the current search pipeline fails on multi-constraint queries
(e.g., "iconic twist ending" returns mid-tier thrillers instead of Fight Club, The
Sixth Sense, etc.) and planning for a redesigned search architecture.

### Key Findings
- Compared generated metadata for Fight Club, The Sixth Sense, Wild Things, and A
  Perfect Getaway — metadata quality is comparable across all four, ruling out
  metadata as the cause
- Root cause is architectural: additive scoring at every layer (vector scoring,
  channel merging) creates disjunctive results, rewarding movies that excel at one
  attribute over movies that satisfy multiple attributes simultaneously
- Secondary cause: embedding density effect — movies whose identity revolves around
  a single attribute (Wild Things = twists) have higher cosine similarity for that
  attribute than movies with richer, more distributed embeddings (Fight Club)

### Planning Decisions
- Proposed deal-breaker / preference / implicit hierarchy to replace flat additive
  scoring — deal-breakers gate the candidate set (conjunctive), preferences rank
  within it (additive)
- Proposed threshold + flatten approach for semantic deal-breakers: once a candidate
  passes retrieval threshold, its deal-breaker score is flattened to 1.0 to prevent
  embedding density bias
- Proposed 4-phase pipeline: Phase 0 (query understanding) → Phase 1 (deal-breaker
  retrieval) → Phase 2 (preference scoring) → Phase 3 (result assembly) → Phase 4
  (exploratory extension)
- Identified 14 query type categories with distinct retrieval needs (8 simple, 6
  complex), plus notes on subtypes and edge cases from exhaustive query analysis
- Multiple open questions documented around threshold selection, LLM classification
  reliability, graceful degradation strategy
- Added design principle: every query type needs a defined fallback path for when
  strict matching fails (user confidence doesn't guarantee correctness)

### Testing Notes
- Theories to validate: run "iconic twist ending" through notebook to inspect actual
  subqueries/weights/per-space scores; simulate threshold+flatten to test if
  reranking by reception would surface expected movies

## Integrated brainstorm notes into existing planning docs

Files: search_improvement_planning/current_search_flaws.md, search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/types_of_searches.md, search_improvement_planning/open_questions.md

Why: Stream-of-consciousness brainstorm notes organized and merged into the four
existing planning docs. Temporary brainstorm_organized.md removed after integration.

### What was added to each file

**current_search_flaws.md** — 6 new flaws (#7-12): production/lexical signal overlap,
franchise logic too broad, quality prior too weak/non-adaptive, metadata filters miss
temporal signals, lexical matching lacks actor prominence, observed query failure cases table.

**new_system_brainstorm.md** — Dynamic quality prior (strength varies by query),
ranking strategy output (sort/balance/superlative), query expansion from interpreted
intent, scoring function varies by query type (4 modes), cross-space rescoring,
trending candidate injection, similarity weaving into normal results, multi-interpretation
branching for ambiguous queries, 2 new design principles (#8-9).

**types_of_searches.md** — Franchise multi-level resolution, actor prominence scoring,
distinctive vs generic similarity decomposition, similarity weaving into other searches,
trending candidate injection, multi-interpretation branching, multi-audience queries,
temporal-establishment terms as cross-cutting concern.

**open_questions.md** — Scoring function selection, cross-space rescoring feasibility,
production/lexical overlap handling, actor prominence scoring, franchise entity
resolution, dynamic quality prior calibration, implicit temporal signals, multi-interpretation
branching UX. Two new untested theories.

## Scoring architecture refinement — union retrieval, full rescore, multi-vector thresholding

Files: all four search_improvement_planning/ files updated

### Intent
Refined the core scoring architecture based on discussion. Key shifts: candidate
retrieval is now union-based (at least one deal-breaker met), not intersection;
Phase 2 does a full rescore across all dimensions; deal-breaker conformance is the
primary sorting axis with bounded influence from preferences.

### Key Decisions

**Union-then-rescore:** Candidates enter the pool by meeting at least one deal-breaker.
Phase 2 rescores all candidates across all dimensions. Deal-breaker conformance (% met)
is the primary sort; preferences are bounded and cannot override conformance tier.
Pool sizes of a few thousand are fine.

**Semantic threshold mechanics updated:** Above threshold = 1.0 (capped), below
threshold = decay (not hard 0.0). Creates smooth fallback gradient — movies "close on
all 3 deal-breakers" rank above movies that "ace 2 but miss 1."

**Deal-breakers are typed, not symmetric:** "Funny horror" → horror is a deterministic
genre filter (binary), funny is semantic (threshold-capped). Genre narrows the set
first, semantic scores within it. Both are deal-breakers but different retrieval types.

**Multi-vector thresholding:** Single concept targeting multiple spaces (e.g., "twist
ending" in narrative_techniques + viewer_experience) → threshold each space separately,
take best score. Movie clears the deal-breaker if it passes in ANY target space.

**Production vector regeneration:** Committed to regenerating with tightened definition
(filming locations + production technique only). Worth the cost. Revisit whether the
lean vector justifies a dedicated space after regeneration.

**Graceful degradation is native:** Union retrieval + conformance scoring handles
partial matches and fallback in a single ranked list without explicit tier logic.

### Decisions Finalized
- Multi-vector scoring: max across spaces (decided)

### Open Questions Added
- Tiered sorting vs weighted formula for combining deal-breaker conformance with preference scores

## Open questions discussion — architecture decisions and constraint strictness model

Files: all four search_improvement_planning/ files updated

### Intent
Worked through all open questions from the planning docs. Made architecture decisions,
designed the constraint strictness model, and consolidated all outstanding empirical
tests.

### Key Decisions

**Phase 0 redesign:** Single structured LLM call producing full query decomposition
in one pass — intent articulation, query type classification (similarity vs all
others), lexical entities, metadata filters with strictness tiers, semantic
deal-breakers, semantic preferences, sorting criteria (null triggers default quality
composite), and quality prior weight (continuous 0.0-1.0).

**Channel weights eliminated:** Deal-breakers are equally important regardless of
channel. Phase 0's structured output implicitly routes to the right channels without
explicit weighting.

**Three-tier constraint strictness:** UI controls = hard filter (strict SQL WHERE);
NLP-extracted metadata = soft constraint (generous gate ±50% + preference decay);
vector similarity = semantic constraint (threshold-based). Strictness determined by
SOURCE of constraint, not attribute type. Handles imprecise user queries like "80s
action movies" including T2 (1991).

**Default quality composite:** When no explicit sorting criteria stated, pipeline
applies fixed `0.6 * reception + 0.4 * popularity`. Deterministic code, not
LLM-determined.

**Deal-breaker combination:** Binary deal-breakers (entity, metadata) filter via
intersection; spectrum deal-breakers (vector) threshold + flatten. Count passes,
rank by % met, fall back to highest % when no movie meets all.

**Production vector tightened:** Keywords restricted to production TECHNIQUE only
(visual, structural, process terms). Complements filming locations (WHERE) with
technique (HOW). Previous definition too broad, caused thematic bleed.

**Multi-interpretation branching:** Deferred to V2.

**Presentation:** Append (not weave) for exploratory results. Fixed depth trim
(top 25 or 40% score floor).

### Open Questions Added
- IMDB keyword vocabulary audit (prerequisite for keyword-based filtering design)
- Temporal bias representation in Phase 0 output

### Outstanding Tests Consolidated
7 empirical tests documented: subquery inspection, threshold+flatten simulation,
score distribution analysis, cross-channel intersection sizing, quality prior impact,
soft constraint decay validation, embedding density measurement.

## Data gap analysis and data layer design decisions

Files: all four search_improvement_planning/ files updated

### Intent
Mapped all 14 query types against the actual data stored in Postgres, Qdrant, and
LLM-generated metadata to identify gaps. Then made design decisions for each gap.

### Key Decisions

**New Postgres tables:**
- `movie_awards` — ceremony, category, outcome, year. Inverse lookup by award.
  Sourced from IMDB GraphQL. Also include award text in reception vector.
- `franchise_membership` — franchise_name, culturally_recognized_group (only internet-
  established terms, never hallucinated), franchise_role (STARTER/MAINLINE/SPINOFF/
  REBOOT/PREQUEL/REMAKE). Sourced from TMDB belongs_to_collection + LLM enrichment.
  Becomes its own lexical posting table replacing title+character hack.
- Role-specific person posting tables — inv_actor_postings (with billing_position +
  cast_size), inv_director_postings, inv_writer_postings, etc. Entity extraction gets
  role_hint field. Boost implied role when not stated; don't hard-filter.

**New movie_card fields:**
- country_of_origin_ids (INT[]) — Postgres only, not Qdrant payload
- box_office_bucket (TEXT) — era-adjusted like budget_bucket
- source_material_types (INT[] enum array) — constrained enum, not free text.
  Requires source_of_inspiration re-generation.

**Actor prominence:** 4 query modes (exclude non-major / boost by position / binary /
reverse). Prominence = 1.0 - (position / cast_size).

**Production medium:** Via keyword search against overall_keywords + production_keywords,
not a boolean. Maps to IMDB keyword vocabulary. Pilot case for keyword-based
deal-breaker filtering.

**Production vector scope:** After removals, contains only filming locations +
production keywords. Open question whether this justifies a dedicated space.

**Keyword-based deal-breaker filtering:** Use IMDB keywords as boost signal in
Phase 1 retrieval (not hard filter). Union of keyword matches + vector matches.
Keyword matches get automatic pass on deal-breaker threshold.

**Resolved:** production/lexical overlap (flaw #7) naturally resolved by removing
entity-adjacent content from production vector. Actor prominence and franchise
resolution now have concrete designs.

**Decisions NOT to change:**
- Parental guide: covered by watch_context + viewer_experience vectors, no structured
  change needed
- Filming locations: too diverse for structured search, stays in production vector
- Review themes: well captured by reception vector, no change

## Empirical test results → architecture revision

Files: all four search_improvement_planning/ files updated

### Intent
Integrated empirical test results from individual vector space testing notebook into
all planning docs. Results forced a fundamental revision of the Phase 1 architecture.

### Key Findings

**Embedding format is the deeper problem (new flaw #13):** The Sixth Sense doesn't
appear in top-1000 for "twist ending" in narrative_techniques even using exact
metadata wording. Scores 82% of max. Flat-list embedding format dilutes per-attribute
signal for multi-dimensional movies. This is a retrieval failure, not a scoring
failure — no architectural improvement helps if movies never enter the candidate pool.

**Semantic concepts cannot reliably generate candidates (new flaw #14):** "Funny
horror" had zero intersection between vector candidate sets. "Dark gritty Marvel"
missed Winter Soldier from vector results. Broad tonal/experiential concepts fail as
candidate generators via vector retrieval.

**Vector space routing is independently broken (new flaw #15):** watch_context had the
most twist content for The Sixth Sense but got zero weight. reception got 23.8% weight.
The routing system doesn't understand which spaces contain relevant signal.

**Architecture confirmed directionally correct:** Threshold + flatten + popularity
sorting surfaces Fight Club and Sixth Sense in top 10. Quality prior validated by
"silly comedies" comparison.

### Architecture Revisions

**Phase 1 fundamentally changed:** Candidates now generated exclusively from
deterministic channels (metadata filters, entity lookup, keywords). Semantic
deal-breakers do NOT generate candidates — they score in Phase 2 via cross-space
rescoring. Exception: pure-vibe queries with no deterministic anchors.

**Cross-space rescoring now REQUIRED:** Previously deferred as optional enhancement.
Now core to Phase 2 — fetch stored vectors from Qdrant for candidates, compute cosine
similarity against query embeddings per semantic concept.

**New prerequisite identified: structured-label embedding.** Proposed fix for flaw #13.
Embed vector text with structured labels preserving per-attribute context instead of
flat term lists. Generate search queries in the same structured shape. Testable on
small sample before full re-ingestion.

### Open Questions Updated
- 7 original tests closed with results; 3 new follow-up tests added (structured-label
  comparison, cross-space rescoring latency, metadata-anchored retrieval quality)
- Threshold question narrowed by empirical data (elbows at 100-200, 75-90% of max)
- Cross-space rescoring moved from "deferred" to "required"

### Design context
Empirical findings in current_search_flaws.md #13-15. Architecture revisions in
new_system_brainstorm.md Phase 1, "Deterministic Retrieval + Semantic Rescore,"
"Candidate Retrieval & Cross-Space Rescoring," and "Embedding Format: Structured
Labels." Test results and new tests in open_questions.md "Completed Tests" and
"Outstanding Tests (New)."

## V2 data architecture and data needs documentation

Files: search_improvement_planning/v2_data_architecture.md, search_improvement_planning/v2_data_needs.md

### Intent
Consolidated all data from the V1 system and all proposed V2 changes into a
single reference document (v2_data_architecture.md), and cataloged all data
that must be captured/generated/restructured before V2 search can be
implemented (v2_data_needs.md).

### Key Decisions
- Keywords: store `overall_keywords` only (not plot_keywords) as `keyword_ids INT[]`
  on movie_card with GIN index. Use lexical_dictionary for string→ID mapping. No
  inverse posting table — array overlap (`&&`) is sufficient since Phase 0 maps user
  language to keyword IDs before retrieval.
- Country of origin: new `Country` enum (same pattern as Language), `country_of_origin_ids`
  INT[] on movie_card with GIN, plus `inv_country_origin_postings` for lexical lookup.
- Source material: new `SourceMaterialType` enum (values TBD from audit of generated
  data), `source_material_type_ids` INT[] on movie_card with GIN, plus
  `inv_source_material_postings` for lexical lookup.
- Box office: `BoxOfficeBucket` enum (HIT/FLOP), movies < 75 days old always NULL.
- Franchise generation: LLM receives title, year, TMDB collection_name, production
  companies, keywords → generates franchise_name, franchise_role, culturally_recognized_group.
- TMDB `belongs_to_collection.name` not currently captured → needs new column on tmdb_data.

### What the docs contain
**v2_data_architecture.md** — Complete inventory: movie_card (existing + new columns),
movie_awards table, franchise_membership table, all lexical tables (V1 and V2),
hard filters, soft preferences, all 8 vector spaces (V1 and V2 content), enums
(existing + new), Redis, tracker DB.

**v2_data_needs.md** — 21 numbered work items with dependencies: 3 prerequisites
(keyword audit, source material enum derivation, country enum), 2 scraping targets
(awards, TMDB collection), 3 LLM generation tasks (franchise, source_of_inspiration
re-gen, production technique re-gen), 3 computed fields (box office, country mapping,
keyword mapping), 3 embedding regenerations, schema DDL for all new tables/columns/indexes.

## Anchor vector dropped, franchise search strategy decided

Files: search_improvement_planning/open_questions.md, search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/v2_data_architecture.md, search_improvement_planning/v2_data_needs.md

### Intent
Finalized two open design decisions from the V2 architecture discussion.

### Key Decisions

**Anchor vector dropped from V2.** Its generalist role is fully superseded by
deterministic candidate generation (Phase 1) and cross-space rescoring across
specialized vectors (Phase 2). Similarity queries use a weighted mix of targeted
vectors (plot_analysis, viewer_experience, etc.) for more precise control. V2 now
has 7 vector spaces, not 8. Embedding regeneration scope reduced from 800K to 700K.
Can reintroduce if clear failure examples emerge during testing.

**Franchise search strategy decided:**
- `franchise_name`: both ingestion LLM and search extraction LLM instructed to use
  the most common, fully expanded form (no abbreviations) — same convention as the
  lexical entity extractor for person names. Trigram matching via lexical dictionary.
  No enum or alias table needed.
- `franchise_role`: stored as integer enum ordinal, filtered with WHERE clause.
  Search extraction LLM receives the same enum definition.
- `culturally_recognized_group`: trigram matching on the post-franchise-lookup
  result set (3-30 movies). No separate index needed at that scale.

**Cascading updates:** Production vector open question updated (can't fold into
anchor since anchor is dropped — options are now keep/eliminate/repurpose).

## Added collection_name and revenue columns to tmdb_data

Files: movie_ingestion/tracker.py, movie_ingestion/tmdb_fetching/tmdb_fetcher.py
Why: V2 data architecture needs TMDB `belongs_to_collection.name` for franchise generation, and raw revenue for box office bucketing. Neither was persisted previously.
Approach: Added columns to CREATE TABLE schema + ALTER TABLE migrations in tracker.py. Extended _extract_fields() to pull collection_name from belongs_to_collection dict, and added both collection_name and revenue to the INSERT template and persist tuple in tmdb_fetcher.py. Revenue was already extracted but only used for the boolean has_revenue flag.

## IMDB GraphQL API research — awards and box office fields

Files: search_improvement_planning/v2_data_architecture.md, search_improvement_planning/v2_data_needs.md

### Intent
Researched the IMDB GraphQL API to determine exact field names, response structures,
and argument requirements for awards and box office data needed by V2 search.

### Key Decisions

**Awards — ceremony list expanded to 12:** Original 9 (Academy Awards, Golden Globes,
BAFTA, Cannes, Venice, Berlin, SAG, Critics Choice, Sundance) expanded to include
Razzie Awards, Film Independent Spirit Awards, and Gotham Awards. Razzies capture a
distinct search intent ("worst movies"); Spirit/Gotham capture indie prestige. DGA/WGA/PGA
excluded — rarely user search terms, typically 1 nomination per movie.

**Awards — `category` is nullable:** Festival grand prizes (Palme d'Or, Golden Lion,
Golden Bear) return null `category` from the API. Table schema updated to allow NULL
category with `COALESCE(category, '')` in the composite PK.

**Awards — SAG is "Actor Awards" in IMDB:** SAG-AFTRA Awards are listed as event
`"Actor Awards"` in the GraphQL API. Documented in the event.text mapping table.

**Awards — filtering yields ~10-50 rows per movie:** Movies have 300-570+ total
nominations (mostly regional critics circles), but filtering to 12 ceremonies produces
manageable counts.

**Box office — IMDB replaces TMDB as primary source:** `lifetimeGross(boxOfficeArea: DOMESTIC|WORLDWIDE)`
provides more complete data. Worldwide is inclusive of domestic (verified via Box Office
Mojo glossary). `openingWeekendGross(boxOfficeArea: DOMESTIC)` also available. TMDB
revenue column kept as supplementary signal.

**Box office — new imdb_data columns:** `box_office_domestic`, `box_office_worldwide`,
`opening_weekend_gross` added to tracker DB imdb_data table spec.

## Added awards and box office fetching to IMDB scraping pipeline

Files: movie_ingestion/imdb_scraping/models.py, movie_ingestion/imdb_scraping/http_client.py, movie_ingestion/imdb_scraping/parsers.py, movie_ingestion/tracker.py

### Intent
Extend the IMDB GraphQL scrape to fetch award nominations (filtered to 12 major
ceremonies) and box office revenue data, as required by the V2 data architecture
(v2_data_needs.md items #4 and #9).

### Approach
- Added `awardNominations(first: 250)` with `pageInfo` pagination support, plus
  `lifetimeGross(boxOfficeArea: WORLDWIDE)` to the GraphQL query.
- Award pagination: IMDB hard-caps at 250 edges per request. `_paginate_awards()`
  in http_client.py fetches remaining pages via cursor-based pagination and merges
  edges into title_data before the parser sees it. Most movies need 0 extra requests;
  heavily-awarded films (Parasite=570, Everything Everywhere=582) need 1-2 extra.
  Without pagination, major awards (Parasite's Best Picture Oscar) were missed.
- New `AwardNomination` Pydantic sub-model (ceremony, award_name, category, outcome,
  year). `award_name` is the specific prize name users search for ("Oscar", "Palme
  d'Or", "Golden Lion") — distinct from ceremony (organization) and category (specific
  category within the prize). Category is nullable for festival grand prizes.
- Parser filters to 12 in-scope ceremonies via `_IN_SCOPE_CEREMONIES` frozenset
  (exact `event.text` values verified via live API queries). Drops ~95% of
  nominations (regional critics circles).
- Box office: only worldwide lifetime gross, which is inclusive of domestic.
  Whole USD dollars. Non-USD currencies are flagged via print warning and stored
  as 0 to distinguish from missing data (None).
- V2 planning docs updated: `movie_awards` table schema now includes `award_name`
  column in all three planning files.
- Tracker DB: 2 new columns on `imdb_data` (awards TEXT, box_office_worldwide INTEGER)
  with ALTER TABLE migrations for existing databases.
- No changes to scraper.py or run.py — they handle data generically via
  `model_dump()` and the `IMDB_DATA_COLUMNS`-driven serialization.

## Backfill script for awards + box office on already-ingested movies

Files: movie_ingestion/imdb_scraping/backfill_awards_boxoffice.py
Why: ~100K already-ingested movies have NULL awards and box_office_worldwide.
Approach: Slimmed-down copy of run.py that queries `status='ingested'`, reuses
the same process_movie → serialize_imdb_movie → IMDB_INSERT_SQL pipeline, but
skips all status updates and filter logging. Failures are printed, not persisted.
Run: `python -m movie_ingestion.imdb_scraping.backfill_awards_boxoffice`

## Created Country enum for V2 data architecture

Files: implementation/classes/countries.py
Why: V2 data architecture (v2_data_needs.md item #3) requires a Country enum for `country_of_origin_ids` on movie_card. Follows the same pattern as the existing Language enum.
Approach: Extracted all 261 IMDB regions from the IMDB GraphQL API i18n data. 250 current countries/territories + 12 historical entities (Soviet Union, Yugoslavia, etc.) retained for older films. Each member is a (country_id, display_name) tuple with stable numeric IDs. Includes `COUNTRY_BY_NORMALIZED_NAME` dict for string→enum lookup, matching the Language enum pattern.

## Added box office revenue field to Movie object

Files: schemas/movie.py
Why: V2 needs box office data accessible on the Movie object for bucketing and display.
Approach: Added `revenue` (int | None) to TMDBData and `_TMDB_DATA_COLUMNS`, added `box_office_worldwide` (int | None) and `awards` (list[AwardNomination]) to IMDBData. All fields already existed in the tracker DB — they just weren't surfaced through the Pydantic models. Added explicit `AwardNomination.model_validate()` branch in `_build_imdb_data()` for consistency with other sub-model columns. Added `resolved_box_office_revenue()` method to Movie following the same IMDB-first-then-TMDB pattern as `resolved_budget()`. Zero and negative values treated as missing.

## IMDB keyword vocabulary audit — completed

Files: search_improvement_planning/keyword_vocabulary_audit.md (new), search_improvement_planning/v2_data_needs.md, search_improvement_planning/open_questions.md, search_improvement_planning/new_system_brainstorm.md, docs/TODO.md

### Intent
Completed the keyword vocabulary audit (v2_data_needs.md item #1) — prerequisite
for keyword-based deal-breaker filtering in V2 search. Unblocks task #11 (keyword
ID mapping).

### Key Findings
- `overall_keywords` is a curated 225-term genre/sub-genre taxonomy (not a
  free-form tagging system). 100% movie coverage, near-zero long tail, zero
  overlap with `plot_keywords`. Static mapping trivially feasible — no LLM
  translation needed.
- `plot_keywords` (114K terms) value is already absorbed by the metadata
  generation pipeline. Does not need a separate search path.
- Primary deal-breaker value: sub-genre precision across the entire genre
  space — 16 horror sub-types, 17 comedy sub-types, 3 western variants, etc.
  Exactly the deterministic signal vector search is weakest at.
- Holiday tags capture curated editorial judgment (Die Hard, Harry Potter
  correctly tagged). 83.8% christmas, remainder is other holidays and
  viewing traditions.
- 30 language/nationality tags complement structured country/language fields.

### Planning Context
Audit was originally blocked on by the keyword deal-breaker filtering design.
Finding that overall_keywords is a compact taxonomy (not 100K+ free-form tags)
simplifies the entire design: full vocabulary fits in the QU prompt, mapping
is static, and plot_keywords can be ignored for keyword_ids.

### Docs Updated
- v2_data_needs.md: task #1 marked completed with findings summary, task #11
  dependency note updated, dependency graph updated
- open_questions.md: keyword vocabulary mapping question marked DECIDED
- new_system_brainstorm.md: "Keyword-Based Deal-Breaker Filtering" section
  rewritten to reflect actual vocabulary structure and audit results
- docs/TODO.md: audit item marked done
