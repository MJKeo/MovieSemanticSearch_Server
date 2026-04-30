# TODO

Tracks actionable items discovered during development sessions.
Items here are things to address when the relevant work begins,
not urgent fixes.

## Update plot_analysis unit tests for V2 structured-label embedding format
**Context:** `PlotAnalysisOutput.embedding_text()` and
`create_plot_analysis_vector_text()` were rewritten to the V2 structured-label
format (every field labeled with snake_case keys, `character_arcs` adjacent
to `themes`, TMDB-genre merge removed, `__str__()` delegates to
`embedding_text()`). Existing unit tests assert the old format and will fail.
Per .claude/rules/test-boundaries.md tests were not touched in the
implementation session — update them as part of the next dedicated testing
phase.
**When:** When running the next testing/validation pass on the ingestion
pipeline, or as part of the broader V2 rollout.
**See:** schemas/metadata.py (PlotAnalysisOutput.embedding_text), movie_ingestion/final_ingestion/vector_text.py (create_plot_analysis_vector_text), search_improvement_planning/v2_data_architecture.md §8.3

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
That old production-vector fallback has since been removed as part of the
production-vector narrowing; only the source-of-inspiration embedding retains
the historical "original screenplay" default.
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

## ~~Update v2_data_needs.md item #8 — superseded by concept tags~~ RESOLVED
**Resolution:** Docs were clarified to the opposite conclusion: concept tags do
not supersede production-technique metadata. Concept tags cover binary
story/content deal-breakers; production techniques cover real-world making-of
signals. Both remain part of the V2 plan, though the production vector may
still be evaluated later for whether it earns a dedicated slot.

## Measure FranchiseOutput null-pairing violation rates during evaluation
**Context:** The model_validator on FranchiseOutput was removed pre-evaluation. During
franchise evaluation, count how often gpt-5-mini produces: (1) franchise_role without
franchise_name, (2) culturally_recognized_groups without franchise_name, (3) franchise_name
without franchise_role. If rates are negligible (<1%), keep removed. If significant, implement
deterministic fixups in validate_and_fix() (e.g., clear orphaned role/groups when name is null).
**When:** During franchise evaluation phase.
**See:** schemas/metadata.py (FranchiseOutput), movie_ingestion/metadata_generation/generators/franchise.py

## Update franchise_metadata_planning.md for post-prompt-rewrite changes
**Context:** The planning doc is stale on: PREBOOT→REBOOT rename, singular→plural
culturally_recognized_groups, shared universe rule (use universe name when one exists),
public domain exclusion, parametric knowledge framing, collection_name caveat, "about"
counts as membership. Code and prompt are authoritative; planning doc should be updated
to match for reference consistency.
**When:** Before generating franchise at scale — the planning doc is the design reference.
**See:** search_improvement_planning/franchise_metadata_planning.md,
movie_ingestion/metadata_generation/prompts/franchise.py, schemas/metadata.py, schemas/enums.py

## Remove or pin conflicting `schemas` namespace package from venv
**Context:** A third-party `schemas` package installed in `.venv/lib/python3.13/site-packages/schemas`
shadows the local `schemas/` package when notebooks run with cwd != project root. This caused
`ModuleNotFoundError: No module named 'schemas.movie_input'` in test_concept_tags.ipynb even
with the project root on sys.path. The workaround is `importlib.invalidate_caches()` + purging
`sys.modules["schemas*"]`, but the root fix is removing the conflicting package (`pip uninstall schemas`
or adding it to a pip constraint/exclusion) so the local package wins without hacks.
**When:** Next time venv or dependency issues are being cleaned up.
**See:** movie_ingestion/metadata_generation/generators/test_concept_tags.ipynb (cell 2 import block)

## Update concept_tags.md planning doc for 25 tags and new definitions
**Context:** search_improvement_planning/concept_tags.md still documents 23 tags. Two new
tags were added (`bittersweet_ending`, `cliffhanger_ending`) and 6 definitions were refined
(`anti_hero`, `happy_ending`, `sad_ending`, `plot_twist`, `feel_good`, `open_ending`). The
planning doc should reflect the current 25-tag taxonomy with updated definitions and the
new ending spectrum (happy/bittersweet/sad) and open_ending/cliffhanger_ending split.
**When:** Before generating concept tags at scale — the planning doc is the design reference.
**See:** search_improvement_planning/concept_tags.md, schemas/enums.py,
movie_ingestion/metadata_generation/prompts/concept_tags.py

## Regenerate concept_tags_test_results.json with updated prompt/schema/definitions
**Context:** The test results JSON was generated before three major changes: (1) section-level
reasoning schema restructure (per-tag evidence+tag → per-section reasoning+tags), (2) endings
section rewrite replacing keyword shortcuts and factual ledger rules with comparative evaluation,
(3) 2 new tags and 6 definition refinements. Results need regenerating to evaluate whether
the combined structural + evaluation changes improve precision (especially endings) and
consistency. Also fix `female_protagonist` → `female_lead` in expected_tags (test data uses
old enum name).
**When:** Next concept tags evaluation session.
**See:** movie_ingestion/metadata_generation/generators/test_concept_tags.ipynb,
movie_ingestion/metadata_generation/generators/concept_tags_test_results.json

## Implement concept_tags production config: 2-run union + batch API
**Context:** Evaluation across three configurations (gpt-5-mini medium with reasoning fields,
gpt-5-mini medium without reasoning fields, gpt-5-mini low) concluded that the best production
setup is gpt-5-mini medium reasoning, reasoning fields removed from the output schema, run
twice per movie, union the tag sets. 2-run union of the "medium" file (no reasoning fields)
hit P=80.2% R=97.9% F1=88.2% with only 3 FNs across 40 movies — effectively eliminating false
negatives on deal-breaker tags. Cost is ~50% cheaper than the reasoning-fields-on variant
because output tokens drop sharply when the model no longer emits reasoning strings. Stacks
with OpenAI Batch API (another 50% off) for total production cost far below single-run
baseline. Production pipeline should run each movie twice and merge tag sets before storage.
**When:** When wiring concept_tags into the production batch pipeline.
**See:** movie_ingestion/metadata_generation/generators/concept_tags_test_results_medium.json,
schemas/metadata.py (ConceptTagsOutput), DIFF_CONTEXT.md (reasoning field removal entry)

## Rethink deterministic fixup hooks now that reasoning fields are gone
**Context:** Earlier concept_tags design called for deterministic post-generation fixups on
the chronic FP tags (`bittersweet_ending`, `anti_hero`, `kidnapping`) that would inspect the
reasoning field text for keywords like "cost"/"loss"/"moral ambiguity"/"captivity". With
reasoning fields removed from the output schema for cost reasons, those fixups can no longer
key off reasoning text. Options: (1) move fixups to look at `emotional_observations` input
text for the keywords, (2) add a separate lightweight validation pass, (3) keep reasoning
on just the endings category where it matters most. Evaluation showed ~8 anti_hero FPs,
~5 bittersweet FPs, ~3 kidnapping FPs per 2-run union across 40 movies — enough to be worth
addressing after shipping the baseline.
**When:** After shipping 2-run union baseline, if precision turns out to be the bottleneck.
**See:** schemas/metadata.py (ConceptTagsOutput.apply_deterministic_fixups),
movie_ingestion/metadata_generation/prompts/concept_tags.py

## Fix report_bucket_axis_performance.py for flat-list bucket formats
**Context:** `report_bucket_axis_performance.py` expects bucket files to contain nested dicts
with `tmdb_ids`, `movies`, or `samples` keys. The watch_context bucket file uses a flat format
(`bucket_name -> [list of int IDs]`), which causes `_extract_tmdb_ids()` to return empty and
all movies to map to "unknown" bucket. The `build_id_to_bucket_map()` function filters on
`isinstance(bucket_payload, dict)` which skips list payloads entirely.
**When:** Next time the reporting script is used (low priority — manual Python works around it).
**See:** movie_ingestion/metadata_generation/helper_scripts/report_bucket_axis_performance.py (lines 47-86)

## Complete the text-embedding-3-large migration (Qdrant rebuild + re-embed + doc sweep)
**Context:** Code-only flip to text-embedding-3-large landed in this session
(see DIFF_CONTEXT.md entry "upgrade embedding model to text-embedding-3-large").
The new model is wired into both ingestion and search via the shared
`generate_vector_embedding()` helper, and the Qdrant init script now declares
all 8 named vectors at size 3072. The following operational steps were
deliberately deferred and must run before the new model can be used in anger:
1. Drop the existing `movies_v1` Qdrant collection (currently sized 1536 — it
   will reject 3072-dim upserts). Let the init script recreate it, or POST
   the new schema manually.
2. Re-run Stage 8 ingestion (`movie_ingestion/final_ingestion/ingest_movie.py`)
   for every already-ingested movie to regenerate embeddings at 3072 dims.
3. Sweep and update remaining "1536 dims" / "text-embedding-3-small"
   references in docs: CLAUDE.md, AGENTS.md, docs/PROJECT.md, docs/modules/,
   docs/conventions.md, docs/decisions/ADR-011, search_improvement_planning/
   v2_data_architecture.md and v2_data_needs.md, .cursor/rules/.
4. Redis embedding cache keys are `emb:{model}:{hash}`, so old 3-small cache
   entries will coexist harmlessly — no manual flush required, but they can
   be pruned opportunistically.
**When:** When ready to actually deploy the upgraded embedding model to the
production search index. This is the structured-label re-embed milestone from
search_improvement_planning/v2_data_needs.md #12.
**See:** DIFF_CONTEXT.md (upgrade embedding model entry),
db/init/02_qdrant_init.sh,
movie_ingestion/final_ingestion/ingest_movie.py,
search_improvement_planning/v2_data_needs.md

## Hook query-time search into `movie_franchise_metadata`
**Context:** Franchise metadata is now projected into Postgres via
`public.movie_franchise_metadata` plus shared `lex.inv_franchise_postings`,
and `Movie` now loads `generated_metadata.franchise` as
`franchise_metadata`. What remains is the actual search-time usage:
entity extraction should route franchise terms into the new posting
table, metadata/scoring should use `lineage`, `shared_universe`,
`recognized_subgroups`, `lineage_position`, `is_spinoff`,
`is_crossover`, `launched_subgroup`, and `launched_franchise`, and
legacy franchise behavior still anchored in source-material phrases
should be removed. This is the remaining step for queries like
"Harry Potter spinoffs" and "Marvel movies from Phase 3."
**When:** Before the next search-quality iteration that depends on
franchise-aware retrieval.
**See:** schemas/metadata.py (FranchiseOutput),
movie_ingestion/final_ingestion/ingest_movie.py,
db/postgres.py,
db/lexical_search.py,
search_improvement_planning/franchise_metadata_planning.md.


## Update award-related unit tests for post-refinement signatures
**Context:** Four refinements were made to the movie_awards implementation that
change signatures and behavior tested in the previous session: (1) `AwardNomination.ceremony_id`
now returns `int | None` instead of raising `KeyError`; (2) `batch_upsert_movie_awards()`
now accepts `list[AwardNomination]` instead of `list[tuple[int, str | None, int, int]]`;
(3) `create_award_ceremony_win_ids()` now delegates to `Movie.award_ceremony_win_ids()`
instance method; (4) `_reception_award_wins_text()` was rewritten to use `AwardCeremony`
enum keys instead of the deleted `_RECEPTION_AWARD_CEREMONY_ORDER`. Tests in
test_enums.py, test_schemas_movie.py, test_ingest_movie.py, and test_postgres.py
will need updates to match the new signatures and None-handling behavior.
**When:** Next dedicated testing phase for the awards implementation.
**See:** unit_tests/test_enums.py, unit_tests/test_schemas_movie.py,
unit_tests/test_ingest_movie.py, unit_tests/test_postgres.py,
movie_ingestion/imdb_scraping/models.py (AwardNomination.ceremony_id),
db/postgres.py (batch_upsert_movie_awards),
schemas/movie.py (Movie.award_ceremony_win_ids)

## ~~Finalize Step 1 output schema for all three flows~~ DONE
Schema implemented in schemas/flow_routing.py.

## Add canonical-family grouping metadata to unified classification registry
**Context:** `schemas/unified_classification.py` merges OverallKeyword +
SourceMaterialType + ConceptTag into 259 entries, but does not tag each
entry with its canonical family from the 21 families defined in
`search_improvement_planning/finalized_search_proposal.md` §Endpoint 5
(Action/Combat/Heroics, Adventure/Journey/Survival, ..., Viewer Response /
Content Sensitivity). The step 3 keyword prompt will need family-grouped
presentation so the LLM can browse the taxonomy by section rather than as
a flat 259-item list. Add a `family: str` field to `ClassificationEntry`
and a family-mapping dict (or derive from the proposal's section headers).
**When:** When writing the step 3 keyword endpoint prompt.
**See:** schemas/unified_classification.py, search_improvement_planning/finalized_search_proposal.md §Endpoint 5 (lines ~1610-1742)

## ~~Develop keyword/concept tag trait description list for step 2 prompt~~ SUPERSEDED
Resolved differently: the step 2 prompt includes the full enumerated keyword
vocabulary (all ~192 genre/sub-genre keywords organized by family, 30 culture
keywords, 3 animation techniques, 10 source material types, and all 25 concept
tags across 7 categories). This gives the LLM the information it needs to make
accurate routing decisions about what the keyword endpoint covers vs what must
go to semantic. The trait-description approach was abandoned in favor of the
full list because the LLM needs to check whether specific concepts exist in
the vocabulary (e.g., "zombie" exists but "clown" doesn't).
**See:** search_v2/stage_2.py (keyword endpoint section in _ENDPOINTS)


## Factor ingestion-side `*Output.embedding_text()` through query-side `*Body` classes
**Context:** `schemas/semantic_bodies.py` duplicates the `embedding_text()` logic that lives on the ingestion-side `PlotEventsOutput`, `PlotAnalysisOutput`, `ViewerExperienceOutput`, `WatchContextOutput`, `NarrativeTechniquesOutput`, `ProductionTechniquesOutput`, and `ReceptionOutput` classes, plus `create_anchor_vector_text` / `create_production_vector_text`. Duplication was deliberate (makes drift visible in code review) but a cleaner long-term move is to have the ingestion `*Output` classes hold a `*Body` sub-model and delegate `embedding_text()` to it, so both sides share a single source of truth for the format.
**When:** When the next opportunity to touch ingestion-side schemas arises, or as a dedicated refactor once the step 3 semantic endpoint is in production and the format has proven stable.
**See:** schemas/semantic_bodies.py, schemas/metadata.py, movie_ingestion/final_ingestion/vector_text.py

## Author system prompt for step 3 semantic endpoint query generation
**Context:** `SemanticDealbreakerSpec` and `SemanticPreferenceSpec` in [schemas/semantic_translation.py](schemas/semantic_translation.py) are shape-only. The LLM-facing guidance — how to pick a space, how to map concepts to structured sub-fields within a space, how to decide primary vs. contributing weights, how to populate `signal_inventory` and `target_fields_label` — still needs to be written.
**When:** When implementing the step 3 semantic endpoint module.
**See:** search_improvement_planning/finalized_search_proposal.md (Endpoint 6: Semantic), schemas/semantic_translation.py, search_v2/stage_3/keyword_query_generation.py (prompt authoring pattern to follow)

## Implement step 3 semantic endpoint execution module
**Context:** `search_v2/stage_3/` needs `semantic_query_generation.py` + `semantic_query_execution.py` matching the pattern of the existing six endpoints. Execution must dispatch on the composition of step 2's output into D1 / D2 / P1 / P2 scenarios, run Qdrant `retrieve()` for score-only cases and top-N searches for candidate-generating cases, apply global elbow calibration for dealbreakers, and emit per-candidate scores to `dealbreaker_sum` / `preference_contribution`.
**When:** After the semantic system prompt is authored.
**See:** search_improvement_planning/finalized_search_proposal.md (Endpoint 6: Semantic → Execution Scenarios), schemas/semantic_translation.py, schemas/semantic_bodies.py

## ~~Cut stage-3 award endpoint over to token-intersection path~~ DONE
Shipped in the session ending 2026-04-20. `AWARD_QUERY_STOPLIST` +
`tokenize_award_string_for_query` added to `implementation/misc/award_name_text.py`;
new `fetch_award_name_entry_ids_for_tokens` posting-list helper in `db/postgres.py`;
`fetch_award_row_counts` signature changed to take `award_name_entry_ids: set[int] | None`
instead of `award_names: list[str] | None`, with the WHERE clause swapped to
`award_name_entry_id = ANY(...)`. `execute_award_query` now resolves `spec.award_names`
via token-intersection + across-name union, early-exits with empty `EndpointResult`
when the axis was populated but resolved to zero entry ids. Translator prompt's
`AWARD NAME SURFACE FORMS` section rewritten to describe base-form emission + shared
tokenizer (no more "one-character difference produces zero matches" language on the
prize axis); `CEREMONIES` section left intact because ceremony matching is still
exact enum. Related follow-ups tracked as separate TODOs below.

## Update award query-side tests for new tokenizer + DB helper signatures
**Context:** Per .claude/rules/test-boundaries.md, no test files were touched during the
query-side cutover. Test fixtures and assertions will need updating for: (1) new
`tokenize_award_string_for_query` and `AWARD_QUERY_STOPLIST`; (2) removal of
`_dedupe_nonempty` from `search_v2/stage_3/award_query_execution.py`; (3) new
`db.postgres.fetch_award_name_entry_ids_for_tokens` helper; (4) `fetch_award_row_counts`
parameter rename `award_names` → `award_name_entry_ids`; (5) the early-exit branch
in `execute_award_query` that returns empty when `spec.award_names` is populated but
resolves to no entry ids.
**When:** Dedicated test-updates phase after the query-side cutover stabilizes.
**See:** implementation/misc/award_name_text.py, db/postgres.py (fetch_award_row_counts, fetch_award_name_entry_ids_for_tokens), search_v2/stage_3/award_query_execution.py, unit_tests/

## DF-ceiling tuning + second-wave stopword candidates for award tokens
**Context:** Analogous to the franchise follow-up — query-side `AWARD_QUERY_STOPLIST`
is hand-curated from the post-backfill top-25 DF scan; planning doc Open Decisions
#1/#2 are resolved. After future ingest sweeps, inspect
`SELECT token, doc_frequency FROM lex.award_name_token_doc_frequency ORDER BY doc_frequency DESC LIMIT 25`
and extend the droplist if new domain-boilerplate or English-connective tokens surface.
Do NOT add a numeric DF ceiling — the tri-modal rationale (see Stopword Droplist
section of the planning doc) applies.
**When:** After each significant ingest sweep, and again periodically as the corpus grows.
**See:** search_improvement_planning/v2_search_data_improvements.md §Award Name Resolution ("Why Not a DF Ceiling"), implementation/misc/award_name_text.py (AWARD_QUERY_STOPLIST)

## ~~Implement Franchise Resolution plan~~ DONE
Ingest-side landed in prior session; query-side landed in the session ending 2026-04-20. Shipped: `lex.franchise_entry` / `lex.franchise_token` / `lex.franchise_token_doc_frequency`; `movie_card.franchise_name_entry_ids` / `subgroup_entry_ids` (BIGINT[], GIN); cardinal number-to-word rule; `FRANCHISE_STOPLIST` in `tokenize_franchise_string`; `FranchiseQuerySpec.franchise_or_universe_names` rename (user chose `franchise_or_universe_names` over the planning-doc's `franchise_names`); cross-field validator dropped so subgroup-only / structural-only specs are valid; `fetch_franchise_entry_ids_for_tokens` posting-list helper; `fetch_franchise_movie_ids` rewritten with spec-shape-aware from-clause branching; prompt rewritten with umbrella-vs-specific rule and scope-commitment in `concept_analysis`. See DIFF_CONTEXT.md (prior session + current session entries) for the full decision log. Remaining follow-ups tracked as separate TODOs below.

## Re-run franchise backfill under stopword-dropping tokenizer
**Context:** `FRANCHISE_STOPLIST` was added to `tokenize_franchise_string` in the query-side cutover. The ingest-side tokens in `lex.franchise_token` were stamped before the stoplist was wired, so they still contain `the` / `of` / `and` etc. Query-side tokenization now drops those, so the `lex.franchise_token` rows for stopword tokens are dead weight but not incorrect (they're just never queried). Re-running the franchise backfill script under the updated tokenizer cleans up the index so debuggability (DF view head) is not cluttered with rows that will never be hit. Precondition before production retrieval relies on the new executor.
**When:** Before shipping the query-side cutover to production. Sanity check after: `SELECT COUNT(*) FROM lex.franchise_token WHERE token IN ('the', 'of', 'and', 'a', 'in', 'to', 'on', 'my', 'i', 'for', 'at', 'by', 'with')` should return 0.
**See:** backfill_franchise_entries_and_tokens.py, implementation/misc/franchise_text.py (FRANCHISE_STOPLIST), DIFF_CONTEXT.md "Franchise resolution: query-side cutover"

## DF-ceiling tuning + second-wave stopword candidates for franchise tokens
**Context:** Planning doc Open Decision #1 intentionally deferred picking any numeric DF ceiling for franchise tokens — the current design uses a closed hand-curated `FRANCHISE_STOPLIST` instead, backed by the rationale that the DF distribution is tri-modal (stopwords / scaffolding / discriminative) with overlapping bands. After the backfill re-run in the prior TODO, inspect `SELECT token, doc_frequency FROM lex.franchise_token_doc_frequency ORDER BY doc_frequency DESC LIMIT 25` and add any obvious new English stopwords to `FRANCHISE_STOPLIST`. Do NOT add a numeric DF ceiling — the planning doc documents why.
**When:** After the backfill re-run, and again periodically as the corpus grows.
**See:** search_improvement_planning/v2_search_data_improvements.md §Franchise Resolution ("Why Not a DF Ceiling"), implementation/misc/franchise_text.py

## Narrow-inside-umbrella resolver rule for franchise (if eval shows prompt-only mitigation fails)
**Context:** Planning doc Open Decision #2. When a user asks "Doctor Strange in the MCU", the prompt currently instructs the LLM to emit only the narrow form (`["doctor strange"]`) because every Doctor Strange film is already MCU — so across-name union would OR-broaden the result. If evaluation shows the prompt-only mitigation misfires at scale, implement a resolver-side subset-elimination rule: when one name's entry-id set is a strict subset of another's in the same spec, drop the superset.
**When:** After eval data on stage-3 franchise retrieval quality is available.
**See:** search_improvement_planning/v2_search_data_improvements.md §Franchise Resolution Edge Cases / §Franchise-names OR semantics when user wanted AND, search_v2/stage_3/franchise_query_execution.py

## Update franchise query-side tests for renamed field + tokenizer + new DB helper signatures
**Context:** Per .claude/rules/test-boundaries.md, no test files were touched during the query-side cutover. Test fixtures and assertions will need updating for: (1) `FranchiseQuerySpec.franchise_or_universe_names` (renamed from `lineage_or_universe_names`); (2) removal of the `recognized_subgroups requires lineage_or_universe_names` validator (subgroup-only specs now valid); (3) `tokenize_franchise_string` now drops `FRANCHISE_STOPLIST` entries — any fixture depending on `the` / `of` / etc. surviving will break; (4) new `db.postgres.fetch_franchise_entry_ids_for_tokens` helper; (5) `db.postgres.fetch_franchise_movie_ids` signature changed from string-list variations to entry-id sets, and structural-flag args changed from `bool | None` → `bool`.
**When:** Dedicated test-updates phase after the query-side cutover stabilizes.
**See:** DIFF_CONTEXT.md "Franchise resolution: query-side cutover" testing notes, unit_tests/

## Update v2_search_data_improvements.md franchise_names reference to franchise_or_universe_names
**Context:** The planning doc's §Query-Time Resolution section says "Key rename: `lineage_or_universe_names` → `franchise_names`". The actual rename landed as `franchise_or_universe_names` (user preference — keeps "either lineage or universe" signal visible in the schema). The planning doc is now mildly stale on this detail.
**When:** Next time the planning doc is edited, or sooner if anyone references the rename.
**See:** search_improvement_planning/v2_search_data_improvements.md §Franchise Resolution / Query-Time Resolution (around line 1708), schemas/franchise_translation.py

## Run prefer_lineage migration against the active Postgres DB
**Context:** `db/migrate_split_franchise_columns.py` lands alongside the code changes but has not been executed. It drops `movie_card.franchise_name_entry_ids` + its GIN index and creates `lineage_entry_ids` / `shared_universe_entry_ids` + indexes, then backfills both columns from `movie_franchise_metadata`. The running DB still has the old schema. Run before exercising any stage-3 franchise search or the ingest pipeline, otherwise `upsert_movie_card` / `write_franchise_data` will fail on the missing new columns.
**When:** Before the next ingest run or stage-3 search test against the dev DB. Dry-run with `--schema-only` first if a staged rollout is preferred.
**See:** db/migrate_split_franchise_columns.py, DIFF_CONTEXT.md "Franchise prefer_lineage: review fixes + targeted migration script"

## Update Stage 1 flow-routing tests for schema reshape
**Context:** `FlowRoutingResponse` was restructured in this session:
(1) `alternative_intents` max_length dropped from 2 to 1;
(2) new required field `query_traits` (single-line string) inserted
at position 2 — final field order is ambiguity_analysis →
query_traits → primary_intent → alternative_intents →
creative_spin_analysis → creative_alternatives;
(3) `CreativeSpin` semantics loosened from "faithful narrowing" to
"trait preservation" (both narrowings and tangents valid). Any test
fixtures that construct `FlowRoutingResponse` or `CreativeSpin`
literals, assert on the old field order, or expect 2 alternatives
will fail. Per `.claude/rules/test-boundaries.md` no tests were
touched during implementation.
**When:** Next dedicated testing phase covering V2 search, or before
Stage 1 is wired into a production code path.
**See:** schemas/flow_routing.py, search_v2/stage_1.py,
DIFF_CONTEXT.md "Stage 1: cap alternatives at 1, add trait extraction, free spins from strict narrowing"

## Render `query_traits` + preserves/swaps annotations in debug_stage_1 compact summary
**Context:** [search_improvement_planning/debug_stage_1.py](search_improvement_planning/debug_stage_1.py)
`_render_compact_summary` doesn't print the new `query_traits`
field or the per-candidate `[preserves: X; swaps: Y]` annotation
from `creative_spin_analysis`. Both live in the JSON report but
aren't visible when eyeballing the stdout summary, which is the
primary inspection surface. Low priority — add two lines to the
renderer (one for query_traits after ambiguity_analysis, and the
spin annotation already embeds in creative_spin_analysis text).
**When:** Next time the Stage 1 debug script is used for an
evaluation pass.
**See:** search_improvement_planning/debug_stage_1.py
(`_render_compact_summary`, around lines 177-221)

## Optional code-level gate for exact_title / similarity no-spin enforcement
**Context:** The Stage 1 prompt now says exact_title and similarity
flows "default to `spin_potential: none` … err toward zero spins
unless an unusually strong angle exists." The soft framing was
deliberate, with the understanding that we could add a hard code
gate later. Debug run confirmed the soft rule can leak: `Scary
Movie` (exact_title) emitted one spin because the model judged a
horror-parody angle worth surfacing. If production traffic shows
this happening often enough to hurt result quality, add a
post-LLM guard in `route_query` (or the first consumer of
`FlowRoutingResponse`) that drops `creative_alternatives` when
`primary_intent.flow` is exact_title or similarity.
**When:** After some production traffic data shows whether the
soft rule holds up; revisit only if leak rate is non-trivial.
**See:** search_v2/stage_1.py (SYSTEM_PROMPT CREATIVE SPINS / creative_spin_analysis sections),
DIFF_CONTEXT.md "Stage 1: cap alternatives at 1, add trait extraction, free spins from strict narrowing"

## Rewrite Step 2B to consume new PlanningSlot shape from Step 2A
**Context:** Step 2A was rewritten this session (new `PlanningSlot` + `Step2AResponse`
schema, `interpret` verdict, decompose-first-then-group flow, three-condition fusion
rule). Step 2B was intentionally left broken: `search_v2/stage_2.py::run_stage_2`
raises `NotImplementedError`, `unit_tests/test_search_v2_stage_2.py` fails, and
notebook cell 4B (`search_v2/test_stage_1_to_4.ipynb`) still references
`step_2a_response.concepts` which no longer exists. The 2B rewrite should adopt the
same Stage-1 / Stage-2A prompt-authoring patterns now codified in
`search_improvement_planning/steps_1_2_improving.md` — per-slot verdict scaffold
(plan / skip), explicit "stay inside the slot's retrieval_shape" rule with a drift
example, single-family-per-expression discipline (each expression targets one
`EndpointRoute`), and brevity caps on reasoning fields. Step 2B is now the
highest-leverage next target for search-quality work.
**When:** Next major search-pipeline work session.
**See:** search_v2/stage_2.py (run_stage_2, _run_step_2b_for_concept, _STEP_2B_SYSTEM_PROMPT),
search_v2/stage_2a.py (new PlanningSlot shape that 2B must consume),
search_v2/test_stage_1_to_4.ipynb (cell 4B broken pending 2B rework),
search_improvement_planning/steps_1_2_improving.md ("What We Learned From The Step 2A Rewrite" + "Working Hypothesis Going Forward").

## Clean up dangling references to deleted search_v2/stage_2a.py and stage_2b.py
**Context:** `search_v2/stage_2a.py`, `search_v2/stage_2b.py`, and
`search_v2/prepass_explorations.py` were deleted in the session
ending 2026-04-23 when the step-2 pre-pass categorization was
finalized as `search_v2/step_2.py`. Four references survived the
cleanup because they sat outside the session's touch-scope:
1. `search_improvement_planning/debug_stage_2a.py` — imports
   `search_v2.stage_2a` (BranchKind, run_stage_2a); will
   `ImportError` on run.
2. `search_improvement_planning/debug_feedback_queries.py` —
   imports both `search_v2.stage_2a` and `search_v2.stage_2b`;
   will also `ImportError` on run.
3. `unit_tests/test_search_v2_stage_2.py` — uses `Step2AResponse`
   and `Step2BResponse` (which still exist in
   `schemas/query_understanding.py`, so this file may still
   import successfully, but it tests deleted behavior).
4. `Step2AResponse` / `Step2BResponse` / `PlanningSlot` /
   `CompletedSlot` / `Step2BResponse` / `RetrievalAction` in
   `schemas/query_understanding.py` — likely orphaned now that
   their only production consumers are gone. Verify no other
   stages import them before deleting.

Also note: the earlier TODO "Rewrite Step 2B to consume new
PlanningSlot shape from Step 2A" is STALE — the slot-partitioning
/ per-slot action-planning architecture was replaced wholesale by
the step-2 categorization pre-pass. Whether that functionality
needs to be re-introduced at a later stage is a separate design
question, not a rewrite of the old module.
**When:** Before the next search pipeline dev session that would
hit the broken debug scripts or the orphaned schemas.
**See:** search_improvement_planning/debug_stage_2a.py,
search_improvement_planning/debug_feedback_queries.py,
unit_tests/test_search_v2_stage_2.py,
schemas/query_understanding.py,
DIFF_CONTEXT.md "Step 2 (Query Pre-pass / Categorization)
migration and finalization"

## Update franchise tests for prefer_lineage + column split + dataclass return
**Context:** Expanded scope on top of the pre-existing "Update franchise query-side tests" TODO. New things that will break or need coverage: (1) `FranchiseQuerySpec` gained a `prefer_lineage: bool` field with two validator coercions (no-name-axis → False, SPINOFF → False); (2) `fetch_franchise_movie_ids` now returns `tuple[set[int], set[int]]` (lineage, universe-only) instead of `set[int]`; (3) `write_franchise_data` / `ingest_franchise_data` return a `FranchiseEntryIds` dataclass instead of a 2-tuple / 3-tuple; (4) `upsert_movie_card` + `update_movie_card_franchise_ids` take `lineage_entry_ids` + `shared_universe_entry_ids` instead of `franchise_name_entry_ids`; (5) the new scoring logic in `execute_franchise_query` produces 1.0 / 0.75 / 1.0-fallback scores that tests asserting binary {0.0, 1.0} will need updating for.
**When:** Dedicated test-updates phase — bundle with the other franchise test updates already tracked.
**See:** DIFF_CONTEXT.md "Franchise prefer_lineage" entries, search_v2/stage_3/franchise_query_execution.py, db/postgres.py `fetch_franchise_movie_ids`, schemas/franchise_translation.py

## Implement exact_title and similarity flow dispatch in orchestrator
**Context:** `search_v2/steps_0_2_orchestrator.py` currently handles all three flows at the routing level (step 0 decides which fire, the orchestrator surfaces `exact_title_flow_executed` / `similarity_flow_executed` flags), but the two non-standard flows are no-op placeholders — only the standard flow actually runs step-2 work. The budget rule for standard-flow branches (`3 - count(firing non-standard flows)`) already accounts for these flows running; once the actual search pipelines land, wire them into `run_steps_0_to_2` in parallel with the step-2 branches and remove the `# TODO` comments.
**When:** When the exact-title search and similarity search pipelines are built.
**See:** search_v2/steps_0_2_orchestrator.py (exact_title_flow_executed / similarity_flow_executed, TODO comments), schemas/step_0_flow_routing.py

## Update stage-3 executor tests for [0.5, 1.0] dealbreaker floor
**Context:** The dealbreaker-floor alignment session added `compress_to_dealbreaker_floor` and applied it in the dealbreaker branches of five executors: trending, award (THRESHOLD mode), metadata (release_date, runtime, popularity, reception), semantic (via in-place `_threshold_flatten` rewrite), plus a new `_score_country_position_dealbreaker` with discrete-position scoring (pos 1 → 1.0, pos 2 → 0.5, else drop). Any existing tests asserting raw [0, 1] values on dealbreaker-mode outputs from these executors will fail. Preference-mode behavior is byte-identical to before, so preference tests should not need changes. Per test-boundaries rule no tests were touched in implementation. New test coverage should validate: (a) dealbreaker outputs land in [0.5, 1.0] for every endorsed movie, (b) zero-score movies still get dropped from dealbreaker pools where applicable (award, popularity, reception, country positions ≥3), (c) preference paths still return raw [0, 1] with 0.0 for non-matches.
**When:** Next dedicated testing phase covering V2 search stage-3 executors.
**See:** search_v2/stage_3/result_helpers.py (compress_to_dealbreaker_floor), search_v2/stage_3/trending_query_execution.py, award_query_execution.py, metadata_query_execution.py, semantic_query_execution.py, DIFF_CONTEXT.md "Stage-3 dealbreaker scores aligned to [0.5, 1.0] floor"

## Rewrite semantic_query_generation.py for unified schema + category-handler prompt assembly
**Context:** `search_v2/stage_3/semantic_query_generation.py` still imports `SemanticDealbreakerSpec` and `SemanticPreferenceSpec` from `schemas/semantic_translation.py` — both were removed when the semantic schema was unified into `SemanticParameters` (+ `SemanticEndpointParameters` wrapper). The module is broken at import and no caller can reach it. Any rewrite should also align with the category-handler prompt-assembly design already scaffolded under `search_v2/stage_3/category_handlers/` rather than reinstating the old two-function `generate_semantic_dealbreaker_query` / `generate_semantic_preference_query` pair. The earlier TODO "Implement step 3 semantic endpoint execution module" is now done on the execution side; the generation side is what remains. Also fold in or supersede the pre-existing TODO "Author system prompt for step 3 semantic endpoint query generation" which was written against the obsolete two-spec shape.
**When:** As part of the category-handlers module buildout (handler.py / prompt_builder.py currently stubs), not as a standalone rewrite — the generation-side prompt is one of the category handler's inputs.
**See:** search_v2/stage_3/semantic_query_generation.py (broken imports), search_v2/stage_3/semantic_query_execution.py (new unified API), search_v2/stage_3/category_handlers/, schemas/semantic_translation.py (new `SemanticParameters` + `SemanticEndpointParameters`), search_improvement_planning/category_handler_planning.md §"Unified semantic schema"


## Update test_handler_output_schemas.py + category_handler_planning.md for MatchMode rename
**Context:** This session renamed `ActionRole` → `MatchMode` (values: `filter` / `trait`) on the category-handler wrappers, along with the field rename `action_role → match_mode`, shared-description constant rename `ACTION_ROLE_DESCRIPTION → MATCH_MODE_DESCRIPTION`, and a full rewrite of both `match_mode` and `polarity` field descriptions in LLM-grounded language (no more "candidate pool" / "orchestrator" vocabulary). Two surfaces were intentionally left untouched and need follow-up: (a) `unit_tests/test_handler_output_schemas.py` has a comment-only reference to "action_role + polarity" — skipped per test-boundaries rule; update when the next test pass runs; (b) `search_improvement_planning/category_handler_planning.md` still uses `ActionRole` / `candidate_identification` / `candidate_reranking` throughout the prose, tables, and code snippets — planning doc is a decision record, will adopt the new vocabulary in its next substantive edit rather than a mass find-and-replace.
**When:** Unit tests: next dedicated test phase covering the category-handler contract. Planning doc: next time it's edited for any reason (don't do a vocabulary-only pass).
**See:** DIFF_CONTEXT.md "Rename ActionRole → MatchMode" entry, schemas/endpoint_parameters.py, schemas/enums.py (MatchMode), unit_tests/test_handler_output_schemas.py, search_improvement_planning/category_handler_planning.md

## Wire `release_format` into the search side — deterministic router for media_type
**Context:** Substantially landed this session, with one piece pending. What's done: (a) `MEDIA_TYPE` value on `CategoryName` (in `schemas/trait_category.py` per the v3 reorg, not enums.py) routes to `EndpointRoute.MEDIA_TYPE`; (b) one `media_type` endpoint covers all three non-default values via the closed-enum wrapper `MediaTypeEndpointParameters` (formats: `Literal[TV_MOVIE, SHORT, VIDEO]` — MOVIE excluded as the default, UNKNOWN excluded as sentinel); (c) `execute_media_type_query` + `fetch_movie_ids_by_release_format` query `movie_card.release_format` with flat 1.0 scoring and the standard dual-mode (dealbreaker / preference) shape; (d) registry mapping in `endpoint_registry.py` and dispatch branch in `endpoint_executors.py`. What's pending: the deterministic phrase-matching router that constructs `MediaTypeEndpointParameters` from a trait's surface_text without an LLM call. Until that lands, MEDIA_TYPE is in `prompt_builder._ENDPOINT_PROMPTLESS` and `handler.run_handler` short-circuits the MEDIA_TYPE category to an empty `HandlerResult`. The architecture pivot away from the LLM handler is captured in DIFF_CONTEXT "media_type endpoint: revert to deterministic-routing direction"; the original TODO's three-endpoints-plus-`exclude_shorts` design was superseded by the single closed-enum endpoint. The `exclude_shorts` discovery default still wants resolving — likely as a deterministic rule wrapping the same router (set `polarity=negative, formats=[SHORT]` by default; flip when Step 2 emits a non-negated MEDIA_TYPE atom naming SHORT).
**When:** Next time search-side work picks up. The deterministic router is the load-bearing remaining work.
**See:** DIFF_CONTEXT.md "media_type endpoint: revert to deterministic-routing direction"; schemas/media_type_translation.py; search_v2/stage_3/media_type_query_execution.py; db/postgres.py `fetch_movie_ids_by_release_format`; search_v2/stage_3/category_handlers/handler.py (MEDIA_TYPE short-circuit at Step 0a.2); search_v2/stage_3/category_handlers/prompt_builder.py (`_ENDPOINT_PROMPTLESS`).

## Isolated unit tests for release_format helper + ingestion plumbing
**Context:** The integration test `unit_tests/test_release_format_backfill.py` covers the live-DB end (column shape, distribution, per-row spot checks, UNKNOWN bound) but doesn't exercise the smaller in-process units. Coverage candidates: `release_format_id_for_imdb_type` round-trip across all five enum members + None + an out-of-scope string (e.g., "tvSeries"); `upsert_movie_card` writes the value through to the SQL params tuple; backfill bucketing groups tmdb_ids correctly across mixed input; the `ALTER TABLE ADD COLUMN IF NOT EXISTS` is idempotent under repeated invocation.
**When:** Next dedicated test phase — kept out of the current changeset per the test-boundaries rule.
**See:** unit_tests/test_release_format_backfill.py (live-DB integration tests already in place), schemas/enums.py (`ReleaseFormat`, `release_format_id_for_imdb_type`), movie_ingestion/backfill/backfill_release_format.py.

## Tighten Step 2 carver-vs-qualifier on multi-title seed lists (q18 inception/interstellar/tenet)
**Context:** Experiment 7 (v10) shipped `intent_exploration` as primary source for atom generation and role_evidence — all four v9 regressions resolved and sole-trait carver-of-last-resort emerged organically. One residual issue documented in the writeup: q18 inception/interstellar/tenet has internal inconsistency. intent_exploration weighs "movies similar to these" as more likely; role_evidence cites "boundaries of the desired population (high-concept sci-fi)"; role commits carver; Step 3 then reads the qualifier-flavored prose and routes to attribute categories anyway. Three pieces (intent prose, role label, downstream routing) do not align. Possible tightening: when intent_exploration's weighed primary intent is shape (b) territory ("movies like X"), role MUST commit qualifier — block the carver path. User flagged this as "transient (an inevitable part of the LLM process)" so not urgent; revisit if the pattern recurs on other multi-title or named-entity-as-template queries.
**When:** Future Step 2 prompt iteration if multi-title-seed-list queries surface as a recurring failure mode in eval. Skip if it stays a one-off.
**See:** search_improvement_planning/steps_2_3_experimentation.md §"Experiment 7 — Lessons learned" item #5; /tmp/step3_runs_v10/q18.txt for the exact output; schemas/step_2.py `Trait.role_evidence`, search_v2/step_2.py `_CARVER_VS_QUALIFIER` PROCESS section.

## Tighten Step 3 aspect-layer DISTINCTNESS to reject near-synonym duplicates
**Context:** Experiment 8 (v11) batch shows 32 of 86 traits still produce `len(dimensions) < len(aspects)`. Roughly 20 of those 32 are Pattern A: the model emits two near-synonym phrasings of the same axis at the aspect layer ("high quality" + "positive reception"; "movie duration" + "120 minutes threshold"; "Christmas setting" + "Christmas as central subject"; "character identity" + "franchise membership"; "low popularity" + "obscurity"). The dimension layer correctly folds them into one — but the cleaner fix is upstream: the aspect-layer DISTINCTNESS test ("could a candidate film vary along one without varying along the other?") should reject duplicate axes before the model commits them. The current prompt has the principle but doesn't make the test mechanically checkable; the model often phrases the same axis twice and the principle doesn't fire. Likely fix: add a stricter pre-commit operational test (re-read the aspect list, ask "is this two ways of saying the same thing?" for each pair, collapse before output).
**When:** Next Step 3 prompt iteration. Check first whether Pattern A reproduces on a fresh batch — could be partly LLM variance.
**See:** search_improvement_planning/steps_2_3_experimentation.md §Experiment 8 (Pattern A breakdown of v11 silent drops); search_v2/step_3.py `_ASPECT_ENUMERATION` DISTINCTNESS section.

## Same-category dimensions still pre-merge in Step 3 despite v11 changes
**Context:** ~12 of 32 silent drops in v11 are Pattern B: genuinely independent axes get pre-merged at the dimension layer when both route to the same category. Three sub-patterns: (1) after-effect axes folded into in-film axes (q21 warm hug "positive emotional resonance" lost; q22 feel good "lighthearted accessible experience" lost; q03 rainy night "immersive emotional depth" lost); (2) adjacent technique/scope axes within one category (q25 violence vs gore merged; q32 prosthetics vs animatronics merged; q33 visibility vs canonical status merged; q41 villain victory vs subversion merged); (3) structural/genre/scale axes folded into tonal siblings (q13 hungover "easy-to-follow narrative" lost; q18 inception "psychological thriller atmosphere" folded; q18 interstellar "grand cosmic scale" lost). The "compression happens at category_calls" rule reduced silent drops 38% but didn't fully take — the model still anticipates the merge one layer too early. Likely fix: explicit "same-category at the dimension layer is allowed, separate dimensions still required" framing, or a preserve-aspect trace requirement at the dimension layer.
**When:** Next Step 3 prompt iteration. Lower priority than the aspect-distinctness fix above (Pattern A is more common); these are residual cases where the merge is structurally legitimate at the call layer but loses the per-aspect retrieval intent.
**See:** search_improvement_planning/steps_2_3_experimentation.md §Experiment 8 Pattern B traces (full aspect→dim→call breakdown for all 12 traits); search_v2/step_3.py `_DIMENSION_INVENTORY` "translate every aspect" section.
