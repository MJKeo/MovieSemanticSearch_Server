# TODO

Tracks actionable items discovered during development sessions.
Items here are things to address when the relevant work begins,
not urgent fixes.

## Include imdb_vote_count in search reranking quality boost
**Context:** The search reranking process should use `imdb_vote_count` as
a signal in its automatic quality and relevance booster. This likely
requires adding `imdb_vote_count` as a column in the ingested movie
database (Postgres) so it's available at reranking time. Movies with
higher IMDB vote counts are generally better-known and more relevant
results, so this signal can help break ties and boost well-established
films in the final ranking.
**When:** When building or refining the reranking/quality-boost stage of
the search pipeline.
**See:** db/vector_scoring.py, db/ingest_movie.py, movie_ingestion/imdb_scraping/models.py


## Verify model IDs for playground notebook providers
**Context:** The metadata generation playground notebook (Cell 2) uses several model IDs that were specified
speculatively and may not be accurate. IDs to verify against each provider's current docs:
- Alibaba/DashScope: `qwen3.5-flash`
- Gemini: `gemini-2.5-flash-lite` (lite variant may have a different canonical ID)
- OpenAI-compatible: `gpt-oss-120b` (internal/routing alias, may not be stable)
- Groq: `meta-llama/llama-4-maverick-17b-128e-instruct` (verify exact string)
**When:** Before running the notebook for real model comparisons.
**See:** movie_ingestion/metadata_generation/metadata_generation_playground.ipynb (Cell 2)


## Add release year next to title in all LLM metadata generation
**Context:** The LLM metadata generation prompts should include the
release year alongside the movie title (e.g., "The Matrix (1999)")
across all 7 metadata types. This gives the LLM better temporal context
when generating plot analysis, viewer experience, reception, and other
metadata — helping it distinguish remakes, place films in their era,
and produce more accurate descriptions.
**Status:** COMPLETE. All 8 generators (plot_events, reception, plot_analysis,
viewer_experience, watch_context, narrative_techniques, production_keywords,
source_of_inspiration) now implemented with "Title (Year)" format.
**See:** movie_ingestion/metadata_generation/generators/plot_events.py,
movie_ingestion/metadata_generation/prompts/plot_events.py,
docs/llm_metadata_generation_new_flow.md


## Iterate on plot_events evaluation rubric after initial run
**Context:** The 2-dimension rubric (groundedness, plot_summary) was
simplified from 4 dimensions after evaluation showed setting and
character_quality were redundant. After running Phase 0 + Phase 1 on the
first movie(s), manually inspect judge reasoning in the
`plot_events_evaluations` table before running the full corpus —
calibration may still need adjustment based on observed judge behavior.
**When:** After first small-scale evaluation run completes.
**See:** movie_ingestion/metadata_generation/evaluations/plot_events.py (JUDGE_SYSTEM_PROMPT)


## ~~Implement request_builder.py for Batch API integration~~ DONE
Implemented as part of the batch generation pipeline build.


## Align search-side PlotAnalysis schema with generation-side redesign
**Context:** The generation-side PlotAnalysisOutput was redesigned (2026-03-24):
themes_primary + lessons_learned → thematic_concepts, conflict_scale → conflict_type,
field order changed for autoregressive scaffolding. The search-side schema in
`implementation/classes/schemas.py` still uses the old field names and structure.
Additionally, `implementation/vectorize.py` references old fields:
`plot_analysis_metadata.core_concept.core_concept_label`,
`plot_analysis_metadata.themes_primary`, `plot_analysis_metadata.lessons_learned`.
Both need updating to match the new generation output. The CoreConcept.__str__()
justification leak issue (from original TODO) still applies.
**When:** When deploying generation pipeline results to the production search index.
**See:** implementation/classes/schemas.py (PlotAnalysisMetadata, CoreConcept),
implementation/vectorize.py (create_plot_analysis_vector_text, dense anchor themes section),
movie_ingestion/metadata_generation/schemas.py (PlotAnalysisOutput)


## Align search-side WatchContextMetadata.__str__() to lowercase terms
**Context:** The generation-side `WatchContextOutput.__str__()` in
`movie_ingestion/metadata_generation/schemas.py` lowercases all terms
before joining (`", ".join(t.lower() for t in combined_terms)`). The
search-side `WatchContextMetadata.__str__()` in
`implementation/classes/schemas.py` does NOT lowercase
(`", ".join(combined_terms)`). This means embedding text will differ
between generation and search if terms contain uppercase characters.
The search-side schema should be updated to lowercase for consistency.
**When:** When deploying generation pipeline results to the production
search index.
**See:** implementation/classes/schemas.py (WatchContextMetadata),
movie_ingestion/metadata_generation/schemas.py (WatchContextOutput)


## ~~Remove _DEFAULT_KWARGS from remaining generators~~ DONE
Completed: removed `_DEFAULT_KWARGS` and `effective_kwargs` indirection from all 6 generators
(plot_analysis, viewer_experience, watch_context, narrative_techniques, production_keywords,
source_of_inspiration). They now pass `**kwargs` directly, matching reception's pattern.
plot_events unchanged (retains its defaults).


## ~~Retry failed Gemini plot_events generations with fallback provider~~ STALE
plot_events is now fixed to gpt-5-mini (no longer uses Gemini). The Gemini
content-filtering failures are no longer relevant. Any existing wave1_results
with NULL plot_events from the old Gemini runs will be regenerated with the
new model.


## ~~Remove debug print statements from metadata generators~~ DONE
Removed temporary user prompt print statements from all 6 generators.


## Update test_eval_plot_events.py for reference-free evaluation
**Context:** `unit_tests/test_eval_plot_events.py` imports `generate_reference_responses`
which was removed during the evaluation pipeline restructuring. The test file will fail
at collection time. Tests need updating to remove reference-related test cases and verify
the new reference-free flow (source data in judge prompt, staggered runs, Anthropic
provider defaults).
**When:** Next time evaluation tests are being worked on.
**See:** unit_tests/test_eval_plot_events.py,
movie_ingestion/metadata_generation/evaluations/plot_events.py



## ~~Backfill plot_summaries after IMDB re-scrape~~ STALE
The premise was wrong — the `imdb_data` table already has movies with both
synopses and plot_summaries. DB query (2026-03-21) shows: of 109,277
imdb_quality_passed movies, 22,655 have synopses across all plot_summary
counts (5,215 with 0 plots, 7,881 with 1, 4,665 with 2, 4,894 with 3).


## Handle long synopses (>8K chars) before embedding
**Context:** ~2,752 synopsis movies exceed 8K chars (~2K tokens), with some
reaching 60K chars. The embedding model (text-embedding-3-small) has a hard
8,191 token limit and quality degrades with longer inputs. LLM-based
distillation via gpt-5-nano was tested and abandoned — the model compressed
too aggressively (76% reduction vs target ~30%) and introduced hallucinations.
Alternative approaches: truncation to a char/token limit, or handling in the
plot_events generator (Option A prompt can instruct the LLM to work with a
truncated version).
**When:** Before generating production embeddings for synopsis movies.
**See:** docs/decisions/ADR-033-plot-events-cost-optimization.md (Section 1)


## Replace .lower() with normalize_string() in all generation-side __str__() methods
**Context:** docs/conventions.md states that `__str__()` methods on Pydantic
schema classes feeding the embedding pipeline must use `normalize_string()`
from `implementation/misc/helpers.py` (NFC normalization, lowercase, diacritic
removal). All generation-side schemas in `movie_ingestion/metadata_generation/schemas.py`
currently use `.lower()` instead. This includes PlotEventsOutput, ReceptionOutput,
PlotAnalysisOutput, ViewerExperienceOutput, WatchContextOutput, and all their
with-justifications variants. Should be a single cross-cutting change.
**When:** Before generating production embeddings from the new pipeline.
**See:** docs/conventions.md (lines 19-23),
movie_ingestion/metadata_generation/schemas.py (all __str__ methods),
implementation/misc/helpers.py (normalize_string)


## Update unit tests for batch_id() and custom_id format change
**Context:** `unit_tests/test_metadata_inputs.py` tests `MovieInputData.batch_id()` with
plain strings (e.g., `batch_id("plot_events")`) and asserts the old format `"12345-plot_events"`.
The signature now requires `MetadataType` enum values, and the format changed to
`"plot_events_12345"` (metadata_type first). Tests will fail at call time.
**When:** Next time inputs tests are being worked on.
**See:** unit_tests/test_metadata_inputs.py (lines 39, 43-44, 325, 330),
movie_ingestion/metadata_generation/inputs.py (build_custom_id, batch_id)


## Update remaining Wave 2 generators for individual reception observation fields
**Context:** The reception generator produces structured observation fields
(thematic_observations, emotional_observations, craft_observations, source_material_hint)
instead of a monolithic review_insights_brief. **plot_analysis is now updated** (2026-03-24)
to consume thematic_observations + emotional_observations directly. The remaining 5 Wave 2
generators still receive a backward-compatible concatenated review_insights_brief constructed
in pre_consolidation.py. Each should be migrated to consume targeted fields:
source_of_inspiration → thematic_observations + source_material_hint,
viewer_experience + watch_context → emotional_observations,
narrative_techniques → craft_observations.
**When:** When working on each Wave 2 generator's prompt/input redesign.
**See:** movie_ingestion/metadata_generation/pre_consolidation.py (concatenated review_insights_brief),
movie_ingestion/metadata_generation/generators/ (remaining Wave 2 generators)


## Update search-side ReceptionMetadata and embedding for new field names
**Context:** The generation-side ReceptionOutput renamed fields: new_reception_summary →
reception_summary, praise_attributes → praised_qualities, complaint_attributes →
criticized_qualities, and raised tag cap from 4→6. The search-side ReceptionMetadata in
implementation/classes/schemas.py and create_reception_vector_text() in vectorize.py still
use the old names. These need updating together, plus the empty-list guard fix in
vectorize.py (currently emits "Praises: " with nothing after it when list is empty).
**When:** When deploying new reception generation results to the search index.
**See:** implementation/classes/schemas.py (ReceptionMetadata),
implementation/vectorize.py (create_reception_vector_text),
movie_ingestion/metadata_generation/schemas.py (ReceptionOutput)


## Update test_reception_generator.py for revamped schema and signature
**Context:** test_reception_generator.py references old field names (new_reception_summary,
praise_attributes, complaint_attributes) and review_insights_brief as a model field.
The schema now uses reception_summary, praised_qualities, criticized_qualities, with
review_insights_brief as a @property. Additionally, `generate_reception()` no longer
accepts provider/model/kwargs — it only takes `movie`. Tests that pass those params
will fail at call time. Tests need updating for new field names, new observation fields,
the property behavior, and the simplified signature.
**When:** Next time reception tests are being worked on.
**See:** unit_tests/test_reception_generator.py,
movie_ingestion/metadata_generation/schemas.py (ReceptionOutput),
movie_ingestion/metadata_generation/generators/reception.py (generate_reception)


## Update unit tests for multi-type batch pipeline changes
**Context:** The batch pipeline was generalized from plot_events-only to multi-type.
`build_plot_events_requests()` and `process_plot_events_results()` were removed. `build_requests()`
and `process_results()` now take `MetadataType`. `cmd_eligibility`, `cmd_submit`, `cmd_autopilot`
now require `metadata_type` parameter. `_get_active_batch_ids()` returns `(batch_id, MetadataType)`
tuples. `_record_batch_ids()` and `_clear_batch_id()` take `metadata_type`. Any tests importing
old function names or using old signatures will fail.
**When:** Next time batch pipeline tests are being worked on.
**See:** movie_ingestion/metadata_generation/run.py, request_builder.py, result_processor.py,
generator_registry.py (new file)


## Keep SCHEMA_BY_TYPE in sync with GENERATOR_REGISTRY
**Context:** `result_processor.py:SCHEMA_BY_TYPE` duplicates schema info from
`generator_registry.py:GENERATOR_REGISTRY`. When adding new metadata types to the registry,
SCHEMA_BY_TYPE must also be updated or result processing will silently record failures
("No schema registered") instead of crashing loudly. Consider adding a startup assertion
or deriving SCHEMA_BY_TYPE from the registry.
**When:** When adding Wave 2 types to the batch pipeline.
**See:** movie_ingestion/metadata_generation/result_processor.py (SCHEMA_BY_TYPE),
movie_ingestion/metadata_generation/generator_registry.py (GENERATOR_REGISTRY)


## Update unit tests for ADR-033 signature changes
**Context:** The ADR-033 implementation changed signatures in plot_events and
source_of_inspiration generators. `build_plot_events_user_prompt` now returns
`Tuple[str, str]` instead of `str`. `build_source_of_inspiration_user_prompt`
and `generate_source_of_inspiration` no longer accept `plot_synopsis`. Unit
tests for both generators will fail at import/call time until updated.
**When:** Next time generator tests are being worked on.
**See:** unit_tests/test_source_of_inspiration_generator.py,
unit_tests/test_plot_events_generator.py (if it exists),
movie_ingestion/metadata_generation/generators/plot_events.py,
movie_ingestion/metadata_generation/generators/source_of_inspiration.py


## ~~Re-evaluate reception candidates with revised prompt~~ DONE
Completed: gpt-5-mini-minimal with revised prompt + no-overview was evaluated across all
36 movies. Matched or exceeded old low-reasoning quality. Reception generator finalized
with fixed config (gpt-5-mini, minimal reasoning, low verbosity).

## Update plot_events embedding to use synopsis when available, generated plot_summary as fallback
**Context:** The plot_events vector space embedding process should prefer
the IMDB synopsis (human-written, detailed) as the embedding input text
when one exists for a movie. For movies without a synopsis, the
LLM-generated `plot_summary` from the plot_events metadata should be used
instead. This aligns with ADR-033's two-branch strategy where synopsis
movies (Branch A) skip LLM generation for plot_events entirely — their
synopsis is the higher-quality signal for embedding. The embedding
pipeline needs to implement this conditional logic: check for synopsis
presence, use it if available, otherwise fall back to the generated
plot_summary output. **Important:** the same `MIN_SYNOPSIS_CHARS` threshold
(2,500 chars) from the plot_events generator must be applied here — synopses
below this length are too thin for faithful condensation, and the generated
plot_summary will be higher quality for embedding. See
`movie_ingestion/metadata_generation/generators/plot_events.py` for the
threshold constant and rationale.
**When:** When building the production embedding pipeline for plot_events
vectors (after ADR-033 implementation is complete).
**See:** docs/decisions/ADR-033-plot-events-cost-optimization.md,
implementation/vectorize.py, movie_ingestion/metadata_generation/schemas.py (PlotEventsOutput),
movie_ingestion/metadata_generation/generators/plot_events.py (MIN_SYNOPSIS_CHARS)


## Experiment: emotional_observations impact on plot_analysis quality
**Context:** emotional_observations is included as an input to plot_analysis experimentally.
The hypothesis is that thematic_observations alone is sufficient (emotional tone/mood is
more relevant to viewer_experience than thematic analysis), but emotional_observations may
help with genre_signature classification and character arc tone. Run A/B evaluation: same
movies with and without emotional_observations, compare output quality on a representative
sample including thin-input movies.
**When:** During plot_analysis evaluation pipeline runs.
**See:** movie_ingestion/metadata_generation/generators/plot_analysis.py,
movie_ingestion/metadata_generation/prompts/plot_analysis.py

## Experiment: plot_analysis quality thresholds for thin-input movies
**Context:** ~36K movies have only a short overview (~170 chars median) as their plot source
when plot_events doesn't run. Another ~28K have sub-600-char synopses/summaries. These thin
inputs may produce low-quality character_arcs and generalized_plot_overview. Need empirical
evaluation to determine if there's a minimum plot text length below which plot_analysis
should be skipped or outputs should be flagged as low-confidence. The skip condition
currently only requires one of plot_synopsis/thematic_observations/emotional_observations.
**When:** During plot_analysis evaluation pipeline runs.
**See:** movie_ingestion/metadata_generation/generators/plot_analysis.py (_best_plot_fallback),
movie_ingestion/metadata_generation/pre_consolidation.py (_check_plot_analysis)

## Update unit tests for plot_analysis schema and generator redesign
**Context:** unit_tests/test_plot_analysis_generator.py references old field names
(themes_primary, lessons_learned, conflict_scale, review_insights_brief) and old
function signatures. unit_tests/test_pre_consolidation.py references
review_insights_brief as a direct field on ReceptionOutput and old _check_plot_analysis
signature. Both will fail at collection/call time.
**When:** Next time plot_analysis or pre_consolidation tests are being worked on.
**See:** unit_tests/test_plot_analysis_generator.py,
unit_tests/test_pre_consolidation.py,
movie_ingestion/metadata_generation/schemas.py (PlotAnalysisOutput),
movie_ingestion/metadata_generation/generators/plot_analysis.py


