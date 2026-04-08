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
**See:** db/vector_scoring.py, movie_ingestion/final_ingestion/ingest_movie.py, movie_ingestion/imdb_scraping/models.py


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
**Context:** The generation-side PlotAnalysisOutput was redesigned (2026-03-24) and
further hardened (2026-03-24): themes_primary + lessons_learned → thematic_concepts,
conflict_scale → conflict_type, core_concept_label → elevator_pitch, CharacterArc
simplified to arc_transformation_label only, min_length → 0 on sparse-prone fields.
The search-side schema in `implementation/classes/schemas.py` still uses the old
field names and structure. Additionally, `movie_ingestion/final_ingestion/vector_text.py` references
old fields: `plot_analysis_metadata.core_concept.core_concept_label`,
`plot_analysis_metadata.themes_primary`, `plot_analysis_metadata.lessons_learned`.
Both need updating to match the new generation output. The CoreConcept.__str__()
justification leak issue (from original TODO) still applies.
**When:** When deploying generation pipeline results to the production search index.
**See:** implementation/classes/schemas.py (PlotAnalysisMetadata, CoreConcept),
movie_ingestion/final_ingestion/vector_text.py (create_plot_analysis_vector_text, dense anchor themes section),
schemas/metadata.py (PlotAnalysisOutput)


## Align search-side WatchContextMetadata.__str__() to lowercase terms
**Context:** The generation-side `WatchContextOutput.__str__()` in
`schemas/metadata.py` lowercases all terms
before joining (`", ".join(t.lower() for t in combined_terms)`). The
search-side `WatchContextMetadata.__str__()` in
`implementation/classes/schemas.py` does NOT lowercase
(`", ".join(combined_terms)`). This means embedding text will differ
between generation and search if terms contain uppercase characters.
The search-side schema should be updated to lowercase for consistency.
**When:** When deploying generation pipeline results to the production
search index.
**See:** implementation/classes/schemas.py (WatchContextMetadata),
schemas/metadata.py (WatchContextOutput)


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


## ~~Handle long synopses (>8K chars) before embedding~~ DONE
Completed (2026-04-03): Integrated token-limit fallback directly into
create_plot_events_vector_text(). Two-tier check: cheap character gate (15K chars)
skips tiktoken for 99.5% of movies, then exact token count for the ~561 that exceed
the gate. Falls back to shorter plot summaries/metadata when over limit.
**See:** movie_ingestion/final_ingestion/vector_text.py

## ~~Use create_plot_events_vector_text_fallback on token-limit embedding errors~~ DONE
Completed (2026-04-03): Addressed proactively in the text generation layer rather than
reactively in the embedding layer. Fallback is now private (_plot_events_fallback_text)
and called automatically by create_plot_events_vector_text() when token limit exceeded.
**See:** movie_ingestion/final_ingestion/vector_text.py


## ~~Replace .lower() with normalize_string() in all generation-side __str__() methods~~ DONE
Completed: all `__str__()` methods in `schemas/metadata.py` now use
`normalize_string()` from `implementation/misc/helpers.py` instead of `.lower()`.


## Update unit tests for batch_id() removal and schemas package split
**Context:** `MovieInputData.batch_id()` was removed — callers now use
`build_custom_id(movie_or_id, metadata_type)` directly. `test_metadata_inputs.py`
tests `batch_id()` extensively (lines 39-60, 331-343) and will fail. Additionally,
`MetadataType` moved to `schemas.enums`, `MovieInputData` to `schemas.movie_input`,
`MultiLineList` to `schemas.data_types`, and all Output schemas to `schemas.metadata`.
Tests importing from `movie_ingestion.metadata_generation.inputs` or
`movie_ingestion.metadata_generation.schemas` need import path updates.
**When:** Next time inputs or schema tests are being worked on.
**See:** unit_tests/test_metadata_inputs.py,
schemas/enums.py, schemas/movie_input.py, schemas/metadata.py,
movie_ingestion/metadata_generation/inputs.py (build_custom_id overloaded)


## ~~Update remaining Wave 2 generators for individual reception observation fields~~ DONE
All Wave 2 generators now consume individual observation fields directly:
plot_analysis (2026-03-24), viewer_experience (2026-03-24), narrative_techniques (2026-03-25),
source_of_inspiration (2026-03-25), watch_context (2026-03-31). The backward-compatible
concatenated review_insights_brief was removed from pre_consolidation.py — no consumers remain.
production_keywords does not use any reception fields.


## ~~Update search-side ReceptionMetadata and embedding for new field names~~ PARTIALLY DONE
create_reception_vector_text() in vector_text.py updated to use Movie with new field names
(reception_summary, praised_qualities, criticized_qualities). The search-side
ReceptionMetadata in implementation/classes/schemas.py still uses old names — needs
updating when deploying to the production search index.
**See:** implementation/classes/schemas.py (ReceptionMetadata),
movie_ingestion/final_ingestion/vector_text.py (create_reception_vector_text)


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
schemas/metadata.py (ReceptionOutput),
movie_ingestion/metadata_generation/generators/reception.py (generate_reception)


## Update unit tests for multi-type batch pipeline changes
**Context:** The batch pipeline was generalized from plot_events-only to multi-type.
`build_plot_events_requests()` and `process_plot_events_results()` were removed. `build_requests()`
and `process_results()` now take `MetadataType`. `cmd_eligibility`, `cmd_submit`, `cmd_autopilot`
now require `metadata_type` parameter. `_get_active_batch_ids()` returns `(batch_id, MetadataType)`
tuples. `_record_batch_ids()` and `_clear_batch_id()` take `metadata_type`. Any tests importing
old function names or using old signatures will fail.
**When:** Next time batch pipeline tests are being worked on.
**See:** movie_ingestion/metadata_generation/batch_generation/run.py, request_builder.py, result_processor.py,
generator_registry.py (new file)


## Keep SCHEMA_BY_TYPE in sync with GENERATOR_REGISTRY
**Context:** `result_processor.py:SCHEMA_BY_TYPE` duplicates schema info from
`generator_registry.py:GENERATOR_REGISTRY`. When adding new metadata types to the registry,
SCHEMA_BY_TYPE must also be updated or result processing will silently record failures
("No schema registered") instead of crashing loudly. Consider adding a startup assertion
or deriving SCHEMA_BY_TYPE from the registry.
**When:** When adding Wave 2 types to the batch pipeline.
**See:** movie_ingestion/metadata_generation/batch_generation/result_processor.py (SCHEMA_BY_TYPE),
movie_ingestion/metadata_generation/batch_generation/generator_registry.py (GENERATOR_REGISTRY)


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


## ~~Write empty source_material outputs as "original screenplay" during embedding~~ DONE
Completed (2026-04-03): Implemented in `SourceOfInspirationOutput.embedding_text()` with
`_is_likely_original()` helper. Defaults to "original screenplay" when source_material is
empty and franchise_lineage is empty or only contains "first"/"start" terms.
`create_production_vector_text()` also falls back to "source material: original screenplay"
when metadata is None entirely.
**See:** schemas/metadata.py, movie_ingestion/final_ingestion/vector_text.py

## ~~Update plot_events embedding to use synopsis when available, generated plot_summary as fallback~~ DONE
Completed: `create_plot_events_vector_text()` now implements a 4-level fallback hierarchy:
longest synopsis → generated plot_summary via metadata → longest plot_summary entry → overview.
Added `create_plot_events_vector_text_fallback()` for token-limit errors.
**See:** movie_ingestion/final_ingestion/vector_text.py


## ~~Experiment: emotional_observations impact on plot_analysis quality~~ EVALUATED
Evaluation complete (80 movies × 4 candidates). Results: emotional_observations helps with
low reasoning (+0.14 thematic, +0.19 arc) but slightly hurts with minimal reasoning (-0.14
thematic, -0.14 overview). Recommended candidate: gpt-5-mini-low with emotional. Prompt now
scopes emotional_observations to tone/mood evidence only (not arcs/conflict/plot).

## ~~Experiment: plot_analysis quality thresholds for thin-input movies~~ IMPLEMENTED
Tiered skip condition implemented based on 80-movie evaluation. Tier 1: plot_synopsis → always
eligible. Tier 2: plot fallback >= 400 chars → eligible. Tier 3: plot fallback 250-399 chars +
thematic_observations >= 300 chars → eligible. Otherwise skip. emotional_observations removed
from eligibility entirely.
**See:** movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py (_check_plot_analysis)

## ~~Re-evaluate plot_analysis with hardened prompt on 70-movie set~~ EVALUATED
Evaluation complete (70 movies × 8 candidates, 7 buckets). Key findings: arc_quality improved
dramatically (4.61 best vs 3.6–3.8 pre-hardening). Only 3 low scores (all score=2, all from
`with_emotional` on Queen concert film). Recommended production candidate changed from
`gpt-5-mini-low with emotional` to `minimal-justifications` (4.723 avg, $0.118/70 movies).
Emotional observations confirmed as net negative — exclude from production.
**See:** `/evaluate-metadata-results plot_analysis` report output (2026-03-24)

## ~~Finalize plot_analysis production config based on 70-movie evaluation~~ DONE
Implemented (2026-03-24): gpt-5-mini, minimal reasoning, low verbosity,
PlotAnalysisWithJustificationsOutput schema, emotional_observations removed.
Generator no longer accepts model params from callers.

## ~~Update unit tests for plot_analysis schema and generator redesign~~ DONE
Completed (2026-03-31): All 8 test files updated in a single session. 239 tests pass.


## ~~Test bucket: generalized_plot_overview-only movies for viewer_experience~~ EVALUATED
Bucket 5 tested in Round 1 (8 movies, 3 candidates). GPO performed surprisingly well
(3.66 avg, matching gold standard). The 2-layer abstraction concern was overblown —
GPO is a viable narrative source. Round 2 will further explore GPO-first fallback chain.
**See:** ingestion_data/viewer_experience_eval_guide.md (Round 1 Results)

## ~~Test bucket: observation-standalone movies for viewer_experience~~ EVALUATED
Buckets 6 and 7 tested in Round 1 (12 movies, 3 candidates). Obs-standalone is viable:
Bucket 6 averaged 3.53, Bucket 7 averaged 3.44 (all holistic scores ≥3). The issue is
section discipline (model fills ending_aftertaste/sensory_load without evidence), not
overall quality. gpt-5.4-nano handles sparse inputs significantly better in Bucket 7
(3.64 vs 3.33/3.36 for mini). No eligibility changes needed. Prompt improvements applied.
**See:** ingestion_data/viewer_experience_eval_guide.md (Round 1 Results)

## ~~Re-evaluate viewer_experience combined path necessity~~ DONE
Eligibility simplified in production config (2026-03-26). Old source-weighted combined
thresholds removed. New combined path is simply: GPO >= 200 + any usable observation.
The standalone-narrative and standalone-observation paths remain unchanged.
**See:** movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py (_check_viewer_experience)

## ~~Update unit tests for narrative_techniques input contract redesign~~ DONE
Completed (2026-03-31): All 8 test files updated in a single session. 239 tests pass.

## ~~Update unit tests for viewer_experience production config changes~~ DONE
Completed (2026-03-31): All 8 test files updated in a single session. 239 tests pass.

## Remove unused Optional wrapper schema classes
**Context:** OptionalTermsWithNegationsSection and OptionalTermsWithNegationsAndJustificationSection
in schemas.py are no longer referenced by any production schema after the viewer_experience
flattening. They are only imported by test files. The classes can be removed entirely once
the corresponding tests are updated.
**When:** When updating viewer_experience/schema tests (see above TODO).
**See:** schemas/metadata.py (lines 89-104, 494-509)

## ~~Re-scrape movies to pick up data truncated by old GraphQL limits~~ DONE
Re-scrape completed with updated GraphQL limits to recover truncated synopses and credits.

## Clean up scratch files left by explore agent in project root
**Context:** During narrative_techniques R2 evaluation analysis, an explore subagent
created 6 scratch files in the project root: `narrative_techniques_complete.json` (1.4MB),
`narrative_techniques_detailed.txt` (1.1MB), `EXTRACTION_SUMMARY.md`, `SAMPLE_DATA_EXAMPLES.md`,
`README_EXTRACTION.md`, `FINAL_VERIFICATION.txt`. These are data extraction artifacts with
no ongoing value.
**When:** Next time — just delete them.
**See:** project root directory

## ~~Evaluate merging production_keywords and source_of_inspiration into one generation~~ SUPERSEDED
Overlap reduced by source_of_inspiration redesign (2026-04-02): production_mediums moved
to deterministic derivation from genres+keywords, franchise lineage split into its own
field. source_of_inspiration is now narrowly scoped to source material identification
with parametric knowledge. production_keywords remains a pure keyword filter. The two
generators now have cleanly separated responsibilities with no output overlap.

## ~~Run viewer_experience ablation candidates to answer Q2~~ SUPERSEDED
Round 3 answered the input pruning question directly: tier1-pruned (remove keywords+arcs)
scored +0.031 over baseline, tier1-tier2-pruned (also remove thematic+genre) scored -0.069.
Production config removes keywords+arcs, keeps thematic+genre. Full ablation (remove all
observations) remains deprioritized — marginal cost too small to justify.
**See:** ingestion_data/viewer_experience_eval_guide.md (Round 3 Design)

## ~~Consider routing obs_standalone_minimal_context movies to nano model~~ SUPERSEDED
Round 2 showed gpt-5-mini-minimal-justifications scores 4.62 on obs_standalone_minimal
(Bucket 7) — higher than gold_standard. The justification schema eliminated the section
discipline gap that motivated nano routing. No model routing needed.
**See:** ingestion_data/viewer_experience_eval_guide.md (Round 2 Results)

## Update unit tests for watch_context input contract redesign
**Context:** The watch_context generator and pre_consolidation eligibility check both changed
signatures. `build_watch_context_user_prompt` and `generate_watch_context` now take
`genre_signatures`, `emotional_observations`, `craft_observations`, `thematic_observations`
instead of `review_insights_brief`. `_check_watch_context` now takes
`(genre_signatures, genres, emotional_observations, craft_observations, thematic_observations)`
instead of `(review_insights_brief, genres, merged_keywords, maturity_summary)`. The Phase 1
eligibility tightening (2026-04-01) added the 3 observation params and a second gate requiring
at least one observation field. Tests importing old signatures will fail at call time.
Additionally, `TermsWithJustificationSection.justification` was renamed to `evidence_basis` —
any test referencing the old field name will break.
Round 3 changes (2026-04-01): `WatchContextWithViewingAppealOutput` renamed to
`WatchContextWithIdentityNoteOutput` (`viewing_appeal_summary` → `identity_note`).
`SYSTEM_PROMPT_WITH_VIEWING_APPEAL` renamed to `SYSTEM_PROMPT_WITH_IDENTITY_NOTE`.
Any test imports of the old names will break.
Round 4 finalization (2026-04-01): `generate_watch_context()` no longer accepts
provider, model, system_prompt, response_format, or **kwargs. Tests passing those
params will fail. Return type changed from `WatchContextOutput` to
`WatchContextWithIdentityNoteOutput`.
**When:** Next time watch_context or pre_consolidation tests are being worked on.
**See:** unit_tests/test_watch_context_generator.py (if it exists),
unit_tests/test_pre_consolidation.py,
movie_ingestion/metadata_generation/generators/watch_context.py,
movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py

## ~~Evaluate identity_note variant on gold_standard + challenging_identity buckets~~ DONE
Evaluated (2026-04-01): R4 candidates (r4-identity-note-low, r4-identity-note-minimal) run across
all 50 movies. Qualitative review confirmed identity_note preserves challenging_identity priming
(correct ironic/visceral/camp classifications) without gold_standard regression (no template
over-constraining). Winner: r4-identity-note-minimal (gpt-5-mini, minimal reasoning, low verbosity,
SYSTEM_PROMPT_WITH_JUSTIFICATIONS + WatchContextWithIdentityNoteOutput). Generator finalized
to this config — no longer accepts configurable provider/model/kwargs params.

## ~~Populate challenging_identity eval bucket with 8-10 movies~~ DONE
Populated with 10 movies (2026-04-01): 9388, 10802, 619778, 23629, 798286, 2662, 792307,
755, 8424, 924. Spans tone-genre mismatch, quality-as-identity, mixed-valence, polarizing,
and non-obvious appeal categories. All verified watch_context eligible.

## ~~Derive production medium deterministically at embedding time (no LLM)~~ DONE
Completed (2026-04-03): Implemented as `Movie.production_medium_text()` in schemas/movie.py.
Simplified to binary classification: Animation genre → "animation", otherwise → "live action".
Finer-grained medium details (computer animation, stop motion, CGI, etc.) are already
captured by production_keywords — this field only provides the top-level signal.
**See:** schemas/movie.py, movie_ingestion/final_ingestion/vector_text.py

## ~~Redesign source_of_inspiration: narrow scope + add franchise_lineage~~ DONE
Completed (2026-04-02): Full prompt and schema rewrite. source_material now covers
adaptations + retellings/branches (remakes, reboots, reimaginings, spinoffs).
franchise_lineage is strictly linear story continuation with expanded vocabulary
(franchise starter, first in franchise, trilogy positions, series positions).
Open vocabulary (guidance not closed enum). Two new eval buckets added
(franchise_position, source_lineage_boundary). Prompt rewritten from scratch
optimized for gpt-5-mini with concrete movie examples at the classification boundary.
**See:** movie_ingestion/metadata_generation/prompts/source_of_inspiration.py,
ingestion_data/source_of_inspiration_eval_guide.md

## Update unit tests for production_keywords generator signature change
**Context:** `generate_production_keywords()` no longer accepts provider, model,
system_prompt, response_format, or **kwargs parameters — it only takes
`movie: MovieInputData`. Tests in `test_production_keywords_generator.py` that
pass those params will fail at call time.
**When:** Next time production_keywords tests are being worked on.
**See:** unit_tests/test_production_keywords_generator.py,
movie_ingestion/metadata_generation/generators/production_keywords.py

## Realign search prompts for remaining 6 vector spaces
**Context:** The plot_analysis search prompts (subquery + weight) were rewritten to match the
current metadata generation and embedding pipeline. The same misalignment pattern exists across
all other vectors — stale field names, content boundary errors (e.g., production prompts claim
cast/crew names are embedded but they aren't), experiential leakage into thematic prompts, etc.
The production vector has the most severe misalignment (cast/crew/character names listed as
embedded content but not actually present). The /realign-vector-search-prompts command was
created to repeat this process for each vector.
**When:** Before deploying updated search pipeline. Run one at a time:
plot_events, viewer_experience, watch_context, narrative_techniques, production, reception.
**See:** implementation/prompts/vector_subquery_prompts.py,
implementation/prompts/vector_weights_prompts.py,
.claude/commands/realign-vector-search-prompts.md

## ~~Run "iconic twist ending" through search notebook to validate scoring theory~~ DONE
Completed (2026-04-08): All 7 empirical tests run. Key findings: embedding format is the
deeper problem (Sixth Sense outside top-1000 for "twist ending"), semantic concepts cannot
reliably generate candidates (zero intersection for "funny horror"), vector space routing
is independently broken (watch_context had most twist content but got zero weight).
Architecture revised: Phase 1 now uses deterministic retrieval only, semantic concepts
rescore in Phase 2. Threshold+flatten confirmed as directionally correct. Quality prior
validated. Full results in search_improvement_planning/open_questions.md (Completed Tests).

## Test structured-label embedding format (highest priority)
**Context:** The single highest-leverage test for the search redesign. Current flat-list
embedding format dilutes per-attribute signal for multi-dimensional movies (Sixth Sense
scores only 82% of max for "twist ending" despite having explicit twist metadata).
Proposed fix: embed with structured labels ("information_control: plot twist / reversal")
instead of flat term lists, and generate search queries in the same structured shape.
Test on 10-20 movies without full re-ingestion: re-embed samples with structured format,
generate structured search queries, compare retrieval ranks for known-relevant movies.
**When:** Before implementing the search pipeline redesign — results determine whether
full re-ingestion is needed.
**See:** search_improvement_planning/open_questions.md (Test A),
search_improvement_planning/new_system_brainstorm.md (Embedding Format: Structured Labels)

## Test cross-space rescoring latency
**Context:** The revised architecture requires cross-space rescoring in Phase 2: fetch
stored vectors from Qdrant for all candidates, compute cosine similarity against query
embeddings per semantic concept. Need to verify latency stays under ~100ms for typical
loads (500 candidates × 2 spaces) and worst case (2000 candidates × 4 spaces).
**When:** During search pipeline implementation.
**See:** search_improvement_planning/open_questions.md (Test B)

## Test metadata-anchored retrieval quality
**Context:** Validates the revised Phase 1→Phase 2 flow end-to-end. For "funny horror
movies": retrieve all horror movies via genre filter, rescore on "funny" via cross-space
rescoring — do Shaun of the Dead, Tucker and Dale, Cabin in the Woods surface? For "dark
gritty Marvel": retrieve Marvel via lexical, rescore on "dark and gritty" — does Winter
Soldier surface?
**When:** During search pipeline implementation, after cross-space rescoring is built.
**See:** search_improvement_planning/open_questions.md (Test C)

## ~~Audit IMDB keyword vocabulary from scraped data~~ DONE
Completed. `overall_keywords` is a 225-term curated genre taxonomy; static mapping
confirmed as the right approach. See search_improvement_planning/keyword_vocabulary_audit.md.

## Fix report_bucket_axis_performance.py for flat-list bucket formats
**Context:** `report_bucket_axis_performance.py` expects bucket files to contain nested dicts
with `tmdb_ids`, `movies`, or `samples` keys. The watch_context bucket file uses a flat format
(`bucket_name -> [list of int IDs]`), which causes `_extract_tmdb_ids()` to return empty and
all movies to map to "unknown" bucket. The `build_id_to_bucket_map()` function filters on
`isinstance(bucket_payload, dict)` which skips list payloads entirely.
**When:** Next time the reporting script is used (low priority — manual Python works around it).
**See:** movie_ingestion/metadata_generation/helper_scripts/report_bucket_axis_performance.py (lines 47-86)
