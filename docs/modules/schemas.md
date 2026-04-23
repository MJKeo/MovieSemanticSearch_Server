# schemas/ — Shared Pydantic Models & Data Types

Top-level package for shared Pydantic models, data classes, and enums
that are importable across `db/`, `api/`, and `movie_ingestion/`. Created
to give cross-cutting types a stable home as `implementation/` is phased
out.

## What This Module Does

Defines canonical types for:
- LLM output schemas used in both generation and embedding pipelines
- The `Movie` tracker-backed data loader for ingestion-time access
- Shared enums (`MetadataType`, `SourceMaterialType`) and utility types (`MultiLineList`)
- MovieInputData for tracker loading (generation-pipeline entry point)
- **Search-side schemas** for the V2 search pipeline: flow routing, query
  understanding, step-3 endpoint translation (entity, metadata, awards,
  franchise, keyword, semantic), and execution return types

## Key Files

| File | Purpose |
|------|---------|
| `metadata.py` | `EmbeddableOutput` base class + 10 embeddable `*Output` schema classes + `FranchiseOutput` and `ConceptTagsOutput` / `TagEvidence` (non-embeddable franchise/concept-tag classification). Each `EmbeddableOutput` subclass implements `embedding_text()` returning normalized text for vector embedding. `ConceptTagsOutput` produces integer concept_tag_ids via `all_concept_tag_ids()`, not embedding text. Legacy `__str__()` methods are retained for backward compatibility. Class docstrings are written as `#` comment blocks above each class — not as Python docstrings — to prevent them from leaking into the JSON schema payload sent to the LLM via `model_json_schema()`. |
| `movie.py` | `Movie`, `TMDBData`, `IMDBData` Pydantic models + `Movie.from_tmdb_id()` single-movie loader and `Movie.from_tmdb_ids()` batch loader. Joins `tmdb_data`, `imdb_data`, and `generated_metadata` from tracker.db in one query and returns fully typed objects with parsed metadata, including `franchise_metadata: FranchiseOutput | None`. |
| `enums.py` | `MetadataType` StrEnum (one value per generation type, 12 total including `PRODUCTION_TECHNIQUES`, `FRANCHISE`, `SOURCE_MATERIAL_V2`, and `CONCEPT_TAGS`), `BoxOfficeStatus` StrEnum (`HIT`, `FLOP`), `SourceMaterialType` enum (10 values with stable integer IDs for GIN-indexed storage), concept-tag enums grouped by category, `AwardCeremony` (12 members — IMDB event.text string as value, stable `ceremony_id` int for Postgres), `AwardOutcome` StrEnum (`WINNER`/`NOMINEE`, each with stable `outcome_id`), `LineagePosition` enum (sequel/prequel/remake/reboot), `BoxOfficeStatus` StrEnum. **Search-side additions**: `SearchFlow` (exact_title/similarity/standard/browse), `EndpointRoute` (7 values — entity/metadata/awards/franchise_structure/keyword/semantic/trending), `DealbreakDirection` (inclusion/exclusion), `SystemPrior` (enhanced/standard/inverted/suppressed), `AwardScoringMode` (FLOOR/THRESHOLD). |
| `data_types.py` | `MultiLineList` — a constrained list type used in generation schemas. |
| `imdb_models.py` | Shared IMDB sub-models: `AwardNomination`, `FeaturedReview`, `ParentalGuideItem`, `ReviewTheme`. Moved from `movie_ingestion/imdb_scraping/models.py` so `db/` and `api/` containers can import them without mounting `movie_ingestion/`. |
| `movie_input.py` | `MovieInputData` dataclass + `load_movie_input_data()` — loads raw tracker data into the form consumed by generator prompt builders. |
| `flow_routing.py` | **Search V2 Step 1** output schema. `FlowRoutingResponse` contains `primary_intent` plus `alternative_intents` and `creative_alternatives` fields. `AlternativeIntent` (competing readings) and `CreativeSpin` (productive sub-angles within a broad primary) are distinct classes. `CreativeSpin` has `spin_angle` replacing `difference_rationale`. `_validate_title_for_flow` helper enforces flow/title invariant on both types. |
| `query_understanding.py` | **Search V2 Step 2** output schema. `Step2AResponse` (concept extraction) + `QueryConcept` + `RetrievalExpression` + `Dealbreaker` / `Preference` with per-item `routing_rationale` and `route: EndpointRoute`. `validate_partition_completeness` validator. Schema represents the revised two-step concept-inventory design: Step 2A extracts concept list; Step 2B resolves each concept to expressions. |
| `endpoint_result.py` | **Search V2 Step 3 return shape**. `ScoredCandidate` (movie_id + score [0,1]) and `EndpointResult` (list of ScoredCandidate). Uniform shape across all 7 endpoints; orchestrator owns direction (inclusion/exclusion) and weighting. |
| `award_translation.py` | **Search V2 Step 3 award endpoint** output schema. `AwardQuerySpec` with `scoring_mode` (`AwardScoringMode`), `scoring_mark` (int ≥1), filter axes (`ceremonies`, `award_names`, `category_tags`, `outcome`, `years`), and `scoring_shape_label` reasoning field. `AwardYearFilter` sub-model (swaps transposed values). |
| `award_category_tags.py` | **Award category tag taxonomy**. `CategoryTag` — single `(str, Enum)` with 62 leaves (ids 1..99), 12 mid rollups (ids 100..199), 7 top groups (ids 10000..10006). Each member has `tag_id: int` and `level: int`. `LEVEL_*_TAGS` constants expose per-level views. `tags_for_category(raw_text) -> list[int]` is the ingest-time resolver (leaf + ancestor ids). `RAZZIE_TAG_IDS: frozenset[int]` enumerates every tag signaling Razzie intent (16 ids). `render_taxonomy_for_prompt()` generates the LLM-facing taxonomy section programmatically from the enum. Import-time assertion ensures `_TAG_DESCRIPTIONS` covers every member. |
| `franchise_translation.py` | **Search V2 Step 3 franchise endpoint** output schema. `FranchiseQuerySpec` with `franchise_or_universe_names` (list, max 3; renamed from `lineage_or_universe_names`), `recognized_subgroups` (list, max 3), `lineage_position` enum, four structural booleans. Validator: subgroup-only specs (no franchise name) are valid; at least one axis must be populated. |
| `metadata_translation.py` | **Search V2 Step 3 metadata endpoint** output schema. `MetadataTranslationOutput` with `constraint_phrases` (evidence inventory), `value_intent_label`, `target_attribute` (10-value enum), and sub-object fields per attribute. |
| `entity_translation.py` | **Search V2 Step 3 entity endpoint** output schema. |
| `keyword_translation.py` | **Search V2 Step 3 keyword endpoint** output schema. Uses `UnifiedClassification` as the selection type. |
| `unified_classification.py` | `UnifiedClassification` StrEnum — merges `OverallKeyword` (225), `SourceMaterialType` (10), and all `ConceptTag` values (25) into one vocabulary for the Step 3 keyword LLM. Built dynamically at import time. `CLASSIFICATION_ENTRIES` registry maps each name to `(display, definition, source, source_id, backing_column)`. `entry_for(member)` is the lookup entry point. `OverallKeyword` takes precedence on name collisions. `Genre` is excluded (fully subsumed by `OverallKeyword`). |
| `production_brands.py` | 31-brand registry with identity-test curation principle: a label belongs in a brand's roster only if a casual viewer typing `<brand> movies` would expect its films. Docstring updated to document the new principle; `_build_and_validate_registry()` import-time assertions still run. |

## Boundaries

- **In scope**: Shared data structures with no business logic.
  No database access (except `Movie.from_tmdb_id()` / `Movie.from_tmdb_ids()`,
  which are pure loaders). No LLM calls.
- **Out of scope**: Generation-pipeline-specific types that are not
  shared across modules. These remain in
  `movie_ingestion/metadata_generation/inputs.py`:
  `build_custom_id`, `parse_custom_id`, `WAVE1_TYPES`, `WAVE2_TYPES`,
  `ConsolidatedInputs`, `SkipAssessment`, `build_user_prompt`,
  `Wave1Outputs`, `load_wave1_outputs`, `extract_narrative_technique_terms`,
  `load_plot_analysis_output`.

## Key Types

**`EmbeddableOutput`** (`metadata.py`): Abstract base class for 10
embeddable `*Output` schemas. Declares `embedding_text() -> str` as the
canonical contract for producing normalized vector embedding text.
Replaces the previous `__str__()`-based convention, which was implicit
and inconsistently applied. `ConceptTagsOutput` intentionally does not
subclass `EmbeddableOutput` — it produces integer IDs, not embedding text.

**`*Output` schemas** (`metadata.py`): 10 embeddable Pydantic models for LLM
structured output — `PlotEventsOutput`, `ReceptionOutput`,
`PlotAnalysisOutput`, `ViewerExperienceOutput`, `WatchContextOutput`,
`NarrativeTechniquesOutput`, `ProductionKeywordsOutput`,
`ProductionTechniquesOutput`, `SourceOfInspirationOutput`,
`SourceMaterialV2Output`. Each
`embedding_text()` produces the text for its corresponding vector space.
`ReceptionOutput.embedding_text()` now emits labeled synthesis-zone lines
(`reception_summary:`, `praised:`, `criticized:`); deterministic award wins
are appended later in `vector_text.py`, not stored on the schema itself.
`ViewerExperienceOutput.embedding_text()` emits fixed-order labeled multiline
text with separate `*_negations:` lines for polarity-preserving retrieval.
`WatchContextOutput.embedding_text()` emits fixed-order labeled multiline
text with up to four section lines: `self_experience_motivations:`,
`external_motivations:`, `key_movie_feature_draws:`, and
`watch_scenarios:`; empty sections are omitted, and `identity_note` /
`evidence_basis` are excluded.
`NarrativeTechniquesOutput.embedding_text()` emits fixed-order labeled
multiline text with one line per populated section:
`narrative_archetype:`, `narrative_delivery:`, `pov_perspective:`,
`characterization_methods:`, `character_arcs:`,
`audience_character_perception:`, `information_control:`,
`conflict_stakes_design:`, and `additional_narrative_devices:`.
Empty sections are omitted, and justification/evidence fields are excluded.
`ProductionTechniquesOutput.embedding_text()` emits a normalized
comma-separated term list; the production vector wraps it as
`production_techniques: ...` and pairs it with scraped
`filming_locations: ...`. `ProductionKeywordsOutput` remains in the schema
set as a historical predecessor, but it is no longer the current
production-vector input.
`WithJustificationsOutput` variants exist for evaluation (identical
`embedding_text()` output to base variant).

**`ConceptTagsOutput`** (`metadata.py`): Multi-label binary classification
of concept tags by category. Contains 7 category fields: `narrative_structure`,
`plot_archetypes`, `settings`, `characters`, `experiential`, `content_flags`
(each a `list[TagEvidence]`), and `endings` (a single required `EndingAssessment`
with a `tag: EndingTag` field — not a list). The `endings` field uses a single
required enum rather than a list; `EndingTag` includes `NO_CLEAR_CHOICE` (id=-1)
as an explicit affirmative classification when evidence is ambiguous. A
`validate_tag_categories` model validator ensures each list category only
contains tags from its allowed set (defined in `CONCEPT_TAG_CATEGORIES`).
`all_concept_tag_ids()` extracts a sorted, deduplicated `list[int]` for
storage in `movie_card.concept_tag_ids`, filtering out id=-1 values.

**`ConceptTag`** (`enums.py`): IntEnum with a `concept_tag_id` attribute
per member. Grouped into 7 categories via `CONCEPT_TAG_CATEGORIES` dict
(maps category field name → frozenset of allowed `ConceptTag` values).

`SourceMaterialV2Output` outputs a `list[SourceMaterialType]`. An empty
list signals original screenplay — no enum value is assigned for that case.
`embedding_text()` returns `""` for empty lists.

**`SourceMaterialType`** (`enums.py`): `(str, Enum)` with both string
values (for Pydantic JSON schema enum constraints in LLM structured output)
and stable integer IDs (for future `movie_card.source_material_type_ids`
GIN-indexed storage). 10 values: `NOVEL_ADAPTATION`, `SHORT_STORY_ADAPTATION`,
`STAGE_ADAPTATION`, `TRUE_STORY`, `BIOGRAPHY`, `COMIC_ADAPTATION`,
`FOLKLORE_ADAPTATION`, `VIDEO_GAME_ADAPTATION`, `REMAKE`, `TV_ADAPTATION`.
See `docs/modules/ingestion.md` (SourceMaterialType section) for boundary notes.

**`Movie`** (`movie.py`): Central ingestion-time data object. Loads a
fully typed row from `tracker.db` including parsed IMDB JSON columns,
TMDB review JSON, provider-key blob unpacking, and all generated
metadata objects. Franchise classification is exposed directly as
`franchise_metadata: FranchiseOutput | None` for Stage 8 Postgres
projection into `movie_franchise_metadata` and `lex.inv_franchise_postings`.
`FranchiseOutput` uses a two-axis v8 schema: identity axis (`lineage`,
`shared_universe`, `recognized_subgroups`, `launched_subgroup`) and
narrative position axis (`lineage_position` enum, `is_spinoff` bool,
`is_crossover` bool), plus the `launched_franchise` flag. `is_crossover`
includes shared-universe team-up films. `validate_and_fix()` enforces
internal consistency after parsing (partial null-propagation, launches_subgroup
coupling, launched_franchise coherence). `FranchiseRole` enum is deleted.
Also includes `concept_tags_metadata` and `concept_tags_run_2_metadata` fields
for both generation runs; `concept_tag_ids()` merges both via set union.
Includes helper methods:
- `maturity_text_short()` — IMDB reasoning prose or MPA description fallback
- `deduplicated_genres()` — genre_signatures + IMDB genres, substring-deduped
- `reception_score()` / `reception_tier()` — blended IMDB + Metacritic score and tier label
- `is_animation()` — binary genre check
- `languages_text()` — labeled primary + additional languages
- `release_decade_bucket()` — semantic era label
- `budget_bucket_for_era()` — era-adjusted budget classification
- `resolved_box_office_revenue()` — IMDB worldwide gross when positive, falls back to TMDB revenue. Zero and negative values treated as missing.
- `box_office_status()` — clear financial outcome classifier (`HIT` / `FLOP` / null) from resolved budget and worldwide gross; only for release year 1980+ and only when both values are positive. `HIT` also requires budget >= $1M to avoid micro-budget ratio noise.

**Ingestion-compatible methods** added to `Movie` (mirrors `BaseMovie`
interface for the ingestion pipeline):
- `release_ts()` — unix timestamp for Postgres/Qdrant payload
- `maturity_rating_and_rank()` — tuple of `(MaturityRating, int)` for payload
- `normalized_title_tokens()` — deduplicated token set from both primary and
  original_title (original_title tokens merged after primary; first-seen order
  preserved). Foreign-language films are now searchable by original title.
- `genre_ids()` — list of integer genre IDs
- `watch_offer_keys()` — list of decoded uint32 provider keys (pre-decoded in TMDBData)
- `audio_language_ids()` — list of integer language IDs
- `source_material_type_ids()` — sorted list of `SourceMaterialType` IDs from `source_material_v2_metadata`
- `keyword_ids()` — sorted list of `OverallKeyword` IDs from `imdb_data.overall_keywords`
- `concept_tag_ids()` — sorted deduplicated list merging both `concept_tags_metadata` and `concept_tags_run_2_metadata` generation runs; id=-1 values filtered out
- `award_ceremony_win_ids()` — sorted list of `AwardCeremony.ceremony_id` values for ceremonies where the movie won at least one award

`Movie.from_tmdb_ids(tmdb_ids, tracker_db_path?)` is the batch loader: executes
one SQLite query with `json_each()` for N movies instead of N individual queries.
Reuses the same `_QUERY` column definitions and `_build_*` parsers as the
single-movie loader.

Default tracker DB path is resolved from the file's own location
(`schemas/movie.py` → repo root → `ingestion_data/tracker.db`),
making it stable across notebooks, shells, and other entry points.

**`TMDBData`** (`movie.py`): Typed sub-model for TMDB data. Now includes
`revenue: int | None` (raw TMDB revenue in USD; 0 stored as None) in addition
to the existing `has_revenue` boolean. Also includes `collection_name: str | None`
(the `belongs_to_collection.name` from TMDB, needed for franchise generation).

**`IMDBData`** (`movie.py`): Typed sub-model for IMDB data. Now includes
`awards: list[AwardNomination]` (nominations from 12 in-scope ceremonies,
parsed from JSON) and `box_office_worldwide: int | None` (IMDB worldwide
lifetime gross in USD; non-USD flagged as 0). `AwardNomination` is now in `schemas/imdb_models.py` (moved from
`movie_ingestion/imdb_scraping/models.py`). Fields: `ceremony` (str, IMDB
event text), `award_name` (str, e.g. "Oscar"), `category` (nullable for
festival grand prizes), `outcome` (`AwardOutcome` enum with
`outcome_id`), `year`. `AwardNomination.ceremony_id` property returns the
matching `AwardCeremony.ceremony_id` integer for Postgres storage, or
`None` for unknown ceremonies.

## Interaction with Other Modules

- `movie_ingestion/metadata_generation/` — all generators and all
  batch pipeline modules import `*Output` schemas from `schemas.metadata`
  and `MetadataType` from `schemas.enums`.
- `movie_ingestion/final_ingestion/vector_text.py` — all vector text
  functions accept `Movie` and call `embedding_text()` on metadata
  objects. `create_viewer_experience_vector_text()` is intentionally a
  thin wrapper over the schema's labeled multiline `embedding_text()`.
  `create_production_vector_text()` is the single production-vector
  formatter; `Movie` no longer duplicates that logic.
- `movie_ingestion/final_ingestion/ingest_movie.py` — all ingestion
  functions accept `Movie` exclusively. `BaseMovie` is no longer used
  in either Postgres or Qdrant ingestion paths.
- `movie_ingestion/metadata_generation/inputs.py` — imports
  `MovieInputData` from `schemas.movie_input`.
- `search_v2/` — all step-3 executor modules import their respective
  translation schemas from `schemas/`. `search_v2/stage_4/` consumes
  `EndpointResult` / `ScoredCandidate` for assembly.
- `implementation/classes/schemas.py` (legacy search-side schemas) —
  remains separate. V2 search uses `schemas/` directly.

## Gotchas

- `EmbeddableOutput.embedding_text()` is abstract — any new `*Output`
  schema that does not implement it will raise `TypeError` at
  instantiation (not at import time).
- **Class docstrings are deliberately avoided in `metadata.py` and
  `enums.py`.** Pydantic's `model_json_schema()` propagates Python
  docstrings into JSON schema `description` fields, which OpenAI's
  `to_strict_json_schema()` then sends to the LLM. Use `#` comment
  blocks above classes instead. Field-level `Field(description=...)`
  is intentional and kept.
- `Movie.from_tmdb_id()` applies a narrow compatibility normalization
  for known legacy key drift in stored metadata JSON:
  `justification` → `evidence_basis`, and obsolete
  source-of-inspiration evidence fields. This lets existing tracker
  rows validate against the current schema without requiring a
  database migration.
- Generation-side schemas (`schemas/metadata.py`) intentionally diverge
  from legacy search-side schemas (`implementation/classes/schemas.py`).
  V2 search uses `schemas/` directly, so this divergence is being
  resolved in place.
- `schemas/testing.ipynb` has been deleted — use the unit tests or a
  notebook in `implementation/notebooks/` for manual inspection.
- `normalized_title_tokens()` on `Movie` merges tokens from both
  `title` and `original_title`. Tests that only cover primary title
  tokens may need updating.
- `SourceMaterialType` uses `(str, Enum)` pattern (not `StrEnum`) because
  it needs a secondary `source_material_type_id: int` attribute per member.
  `__new__` carries both values.
- **`UnifiedClassification` loses IDE jump-to-definition** for its members
  because it is built dynamically at import time. `entry_for()` is the
  canonical lookup path for execution code.
- **`production_brands.py` roster semantics changed**: brand membership
  now uses an identity test (would a casual viewer expect this film under
  `<brand> movies`?) not corporate ownership. Tests referencing the old
  rosters (e.g. Miramax films under DISNEY) need to be regenerated.
- **`CategoryTag` ancestors stored at ingest**: `movie_awards.category_tag_ids`
  stores leaf + every ancestor id per row. A tag id can therefore appear at
  any level in the hierarchy — querying `&& ARRAY[10000]` (acting group) is
  intentional and correct.
