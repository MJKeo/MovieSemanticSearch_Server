# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Viewer experience prompt hardening and schema simplification
Files: `movie_ingestion/metadata_generation/prompts/viewer_experience.py`, `movie_ingestion/metadata_generation/schemas.py`, `movie_ingestion/metadata_generation/generators/viewer_experience.py`, `ingestion_data/viewer_experience_eval_guide.md`

### Intent
Harden the viewer_experience prompt for gpt-5-mini reliability across the full input quality spectrum, following the patterns proven in plot_analysis and narrative_techniques hardening. Also simplify the output schema to remove unnecessary complexity.

### Key Decisions
- **Schema flattening:** Removed `OptionalTermsWithNegationsSection` wrapper from disturbance_profile, sensory_load, and emotional_volatility. All 8 sections now use the same flat `TermsWithNegationsSection` with min_length=0 on both terms and negations. `should_skip` boolean eliminated — empty lists serve the same purpose with less schema complexity. Applied to both `ViewerExperienceOutput` and `ViewerExperienceWithJustificationsOutput`. `__str__()` methods simplified to iterate all 8 sections uniformly.
- **Rules restructured into two subsections:** Split the 12 "CRITICAL phrasing rules" (which mixed output style with input interpretation) into "Input interpretation" (evidence hierarchy, confidence hints, support-only signals, not-available handling) and "Output style rules" (search-query phrasing, synonyms, negations format, no proper nouns). Evidence hierarchy is now prominently positioned in the input interpretation section rather than buried at rule 9.
- **Input field listing trimmed:** Removed the redundant 10-line input field catalog (model can see labeled fields in the user prompt). Replaced with interpretation-only guidance focused on what the model needs to *know about* each input (which is strongest, which is support-only, what narrative_input_source means).
- **Sparse-input guidance added:** New "WHEN INPUTS ARE THIN" subsection at the end of the prompt. Names which sections to leave empty without narrative evidence (ending_aftertaste, emotional_volatility, cognitive_complexity, sensory_load). Sets per-section cap of 0-3 terms/negations for thin inputs. Reinforces that genre + keywords alone produce generic output.
- **Explicit "not available" signals:** Generator now includes `emotional_observations: not available` (etc.) when observations, narrative_input, or character_arcs are absent, rather than silently omitting. Helps the model calibrate confidence vs treating missing fields as forgotten. Proven pattern from plot_analysis hardening.
- **Per-section "Empty when..." guidance:** Section descriptions for disturbance_profile, sensory_load, emotional_volatility, and ending_aftertaste now include explicit guidance on when to produce empty lists (replacing the old "Skip if..." / should_skip instructions).
- **Output expectations updated:** Changed "3-10 phrases" to "0-10 phrases" with "Empty when the section has no grounded evidence."
- **Eval guide updated:** Added two post-evaluation broader trend checks (negation under-production with min_length=0; separated terms/negations quality) and one key signal to watch (synonym runaway).

### Testing Notes
- All modified files compile cleanly via py_compile
- Unit tests for viewer_experience generator and metadata schemas will fail (schema changes, should_skip removal)
- `__str__()` methods still produce identical embedding text between variants
- `OptionalTermsWithNegationsSection` and `OptionalTermsWithNegationsAndJustificationSection` classes left in schemas.py but are now unused by ViewerExperience schemas

## Production keywords prompt improvements
Files: `movie_ingestion/metadata_generation/prompts/production_keywords.py`, `ingestion_data/production_keywords_eval_guide.md`
Why: Prompt review identified significant false-negative risk from blanket "not genres" exclusion that suppresses valid production keywords (animation, korean, tamil) and vague classification criteria.
Approach: (1) Replaced five abstract relevancy questions with four concrete keyword categories aligned with the search-side production vector space (medium, origin/language, source material, process). (2) Replaced genre exclusion with a positive principle: keywords describing how/where/in-what-form are production-relevant even if they also function as genre labels. (3) Softened parametric knowledge restriction to allow world-knowledge for classification decisions while preserving the no-invention guardrail. (4) Strengthened empty-output guidance — framed as expected rather than permitted. (5) Updated Bucket 6 eval guide description to correctly note that overall-keywords can carry real production signal.

## Narrative techniques prompt hardening and schema reordering
Files: `movie_ingestion/metadata_generation/prompts/narrative_techniques.py`, `movie_ingestion/metadata_generation/schemas.py`, `movie_ingestion/metadata_generation/generators/narrative_techniques.py`, `ingestion_data/narrative_techniques_eval_guide.md`

### Intent
Harden the narrative_techniques prompt for gpt-5-mini reliability across the full input quality spectrum (4 tiers), following the patterns that worked for plot_analysis prompt hardening.

### Key Decisions
- **Schema field reordering for cognitive scaffolding:** Reordered all 11 sections in both `NarrativeTechniquesOutput` and `NarrativeTechniquesWithJustificationsOutput` from taxonomic order to cognitive scaffolding order: easiest/most-concrete first (archetype, delivery, devices) → moderate (pov, characterization, arcs, perception) → hardest/most-abstract (information control, stakes, thematic delivery) → rare (meta). Follows the convention established in plot_analysis.
- **Empty lists as explicit default:** All section phrase counts changed from mandatory minimums (e.g., "1-2 phrases") to zero-based ranges (e.g., "0-2 phrases") with explicit "Empty when..." guidance per section. Added global rules: empty is correct default, thin inputs should produce 3-5 total tags, non-traditional content gets mostly empty sections.
- **Craft-only mode subsection:** Replaced the single vague "be more conservative" sentence with a dedicated "WHEN PLOT DATA IS ABSENT" subsection naming which sections should almost always be empty without plot data, and setting a 3-6 tag budget for craft-only movies.
- **Explicit "not available" signals:** Generator now includes `plot_synopsis: not available` and `craft_observations: not available` when those inputs are absent, rather than silently omitting the fields. Proven pattern from plot_analysis hardening.
- **`audience_character_perception` reframed:** Resolved the contradiction between "only include with clearly defined audience reception" and "do not use parametric knowledge." Reframed as a deliberate craft choice by writers/directors/performers — the model tags the intended audience response visible in input evidence, not real-world reception data.
- **Parametric knowledge eval note:** Added guidance to eval guide about monitoring for systematic under-filling on famous Tier 3 films, with a potential rule softening if observed.

### Testing Notes
- All modified files compile cleanly via py_compile
- Unit tests for narrative_techniques generator will fail (schema field order changed)
- `__str__()` methods still produce identical embedding text between variants

## Registered production_keywords in the batch generation pipeline
Files: `movie_ingestion/metadata_generation/generator_registry.py`, `movie_ingestion/metadata_generation/result_processor.py`, `ingestion_data/production_keywords_eval_guide.md`
Why: production_keywords generator existed but wasn't wired into the batch CLI. Input contract analysis confirmed current implementation (title + merged_keywords, eligibility >= 1) is already optimal — no changes to generator, prompt, schema, or pre_consolidation needed.
Approach: Added eligibility adapter (delegates to `_check_production_keywords` with computed `merged_keywords()`), prompt builder adapter, and registry entry with `reasoning_effort: low, verbosity: low`. Added `ProductionKeywordsOutput` to `SCHEMA_BY_TYPE` in result_processor. Wrote eval guide documenting 4 testable hypotheses (justification impact, hallucination rate, small-list behavior, structured data overlap).
Testing notes: Verified via import that registry and schema lookup both include production_keywords.

## Production keywords evaluation bucket design and movie selection
Files: `ingestion_data/production_keywords_eval_buckets.json`, `ingestion_data/production_keywords_eval_guide.md`
Why: Need evaluation buckets to test the 4 hypotheses before production generation: justification impact, hallucination rate, small-list behavior, and structured data overlap.
Approach: Designed 6 buckets × 8 movies = 48 total, crossing keyword count (rich/typical/small), keyword composition (both sources/overall-only), and expected production-keyword density (dense/none/mixed). Both candidates (with/without justification) run on all buckets for 96 total runs. Movies selected for genre, era, country, and data quality diversity within each bucket.
- **gold_standard:** 14-22 merged kw, both sources, well-known films — quality ceiling
- **typical:** 11-18 merged kw, moderate popularity — bread-and-butter workload
- **small_keyword_lists:** 1-5 merged kw — tests edge case behavior (Hypothesis 3)
- **production_dense:** adaptations, animations, documentaries with production terms — tests recall and overlap (Hypothesis 4)
- **no_production_expected:** pure fiction/thematic keywords — tests false positive rate
- **overall_keywords_only:** no plot_keywords, genre-like labels only — tests low-signal input handling

## Early truncation in build_requests to avoid wasted work
Files: `movie_ingestion/metadata_generation/request_builder.py`, `movie_ingestion/metadata_generation/run.py` | Added `max_batches` param to `build_requests` so it truncates the eligible movie list before loading data and building prompts, instead of building everything then discarding excess batches in the caller.

## Redesigned plot_analysis generator inputs and outputs
Files: `movie_ingestion/metadata_generation/schemas.py`, `movie_ingestion/metadata_generation/prompts/plot_analysis.py`, `movie_ingestion/metadata_generation/generators/plot_analysis.py`, `movie_ingestion/metadata_generation/pre_consolidation.py`

### Intent
Optimize the plot_analysis metadata generator for small LLMs (gpt-5-mini) and align with the redesigned reception output schema.

### Key Decisions
- **Input changes:** Replaced `review_insights_brief` (no longer exists on ReceptionOutput) with `thematic_observations` + `emotional_observations` (individual extraction-zone fields). emotional_observations included experimentally to test impact.
- **Plot fallback:** Added `_best_plot_fallback()` helper that selects the longest raw plot source (synopsis > plot_summary > overview) when Wave 1 plot_events didn't run. Uses distinct `plot_text` label (vs `plot_synopsis`) so the LLM knows the quality tier.
- **Merged themes + lessons:** `themes_primary` + `lessons_learned` → `thematic_concepts` (2-5 labels). Eliminates ambiguous theme/lesson distinction that small LLMs struggle with. No embedding impact (embedding model doesn't distinguish the source field).
- **Replaced conflict_scale with conflict_type:** `conflict_scale` (scale of consequences) → `conflict_type` (1-2 phrases for fundamental dramatic tension like "man vs nature", "individual vs system"). Fills a documented retrieval gap — the search subquery prompt already expects conflict types.
- **Field reordering for autoregressive generation:** genre_signatures → thematic_concepts → core_concept_label → conflict_type → character_arcs → generalized_plot_overview. Each field scaffolds the next (classify → analyze → distill → ground → synthesize).
- **Pre-consolidation fix:** Fixed latent AttributeError where `assess_skip_conditions` accessed non-existent `reception_output.review_insights_brief`. Now extracts individual observation fields and constructs a concatenated `review_insights_brief` for backward compatibility with other Wave 2 generators.

### Testing Notes
- Unit tests for plot_analysis generator and pre_consolidation will fail (schema changes, not updated intentionally)
- Downstream embedding logic (vectorize.py) references old field names — will need updating before deployment
- `PlotAnalysisOutput.__str__()` and `PlotAnalysisWithJustificationsOutput.__str__()` verified to produce identical embedding text

## Plot analysis prompt & schema hardening for small LLMs
Files: `movie_ingestion/metadata_generation/schemas.py`, `movie_ingestion/metadata_generation/prompts/plot_analysis.py`, `movie_ingestion/metadata_generation/generators/plot_analysis.py`

### Intent
Harden plot_analysis generation for gpt-5-mini reliability: reduce schema complexity, prevent forced hallucination on sparse data, and improve the LLM's ability to handle absent inputs.

### Key Decisions
- **CharacterArc simplified:** Removed `character_name` and `arc_transformation_description` from production schema — neither was embedded, and the conditional null logic on `character_name` was the highest-error instruction type for mini models. Production `CharacterArc` now has only `arc_transformation_label`.
- **CharacterArcWithReasoning added:** New schema for justification variant only. Has `reasoning` (chain-of-thought) before `arc_transformation_label`. Reasoning is never embedded.
- **min_length → 0:** `thematic_concepts`, `conflict_type`, and `character_arcs` now allow empty lists on both variants. Prevents schema from forcing hallucination when input data is too sparse. `genre_signatures` kept at min_length=2 (always have title + genres to work from).
- **core_concept_label → elevator_pitch:** Renamed across both schemas, both prompt variants, and all `__str__` methods. Field name itself communicates the desired register (conversational log-line, not abstract thematic tag). `CoreConceptWithJustification` renamed to `ElevatorPitchWithJustification`.
- **Explicit "not available" signals:** When plot_synopsis/plot_text and thematic_observations are absent, they are now explicitly included as "not available" in the user message rather than silently omitted. Helps the LLM calibrate confidence vs treating missing fields as forgotten.
- **Generalization reminder on field 6:** `generalized_plot_overview` now explicitly instructs: "replace all proper nouns with generic descriptions."
- **Sparse data guidance in preamble:** Added two new GENERAL RULES about "not available" handling and outputting empty lists when data is sparse.
- **Prompt field 4/5 split:** `_FIELDS_4_5` split into `_FIELD_4` (shared) + `_FIELD_5_NO_JUSTIFICATIONS` / `_FIELD_5_WITH_JUSTIFICATIONS` since character_arcs now differ between variants.

### Testing Notes
- Unit tests will fail due to schema changes (removed fields, renamed fields, new classes)
- `PlotAnalysisOutput.__str__()` and `PlotAnalysisWithJustificationsOutput.__str__()` still produce identical embedding text for matching inputs
- Justification ordering verified: all chain-of-thought fields (explanation_and_justification, reasoning) precede their corresponding label fields

## Plot analysis experiment 1: emotional_observations impact — bucket selection and notebook cell
Files: `ingestion_data/plot_analysis_eval_buckets.json`, `movie_ingestion/metadata_generation/metadata_generation_playground.ipynb`

### Intent
Design and implement the first evaluation experiment for the redesigned plot_analysis generator: does including emotional_observations as input change output quality?

### Key Decisions
- **8 buckets × 10 movies = 80 total:** Crosses two dimensions — observation richness (rich/moderate/thin) × plot data quality (rich plot_events / summaries fallback / overview only). Each bucket tests a different hypothesis about where emotional_observations might matter.
- **Thin observation buckets added:** 3 buckets for movies with thematic_observations <=200 chars (threshold broadened from 100 to get enough diversity). Most thin-observation movies have zero thematic but still have emotional observations (215-400 chars) — the reception generator extracted emotional tone but couldn't find thematic content.
- **Movie selection diversity:** Within each bucket, movies selected for genre, decade, popularity, and rating diversity. Rich-plot buckets have ~50/50 split of synopsis-condensed vs synthesis-sourced plot_events.
- **With-justifications schema for evaluation:** All runs use `PlotAnalysisWithJustificationsOutput` + `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` so we can inspect the LLM's reasoning alongside labels.
- **4 runs per movie:** 2 conditions (with/without emotional_observations) × 2 models (gpt-5-mini minimal reasoning, gpt-5-mini low reasoning).
- **Semaphore at 50:** All tasks across a bucket fire concurrently, throttled to 50 in-flight requests.
- **Composite label key:** `candidate__condition` (e.g., `gpt-5-mini-minimal__with_emotional`) for merge-safe re-runs.

## Manual plot_analysis evaluation files for 80-movie comparison set
Files: `movie_ingestion/metadata_generation/evaluation_data/plot_analysis_*_evaluation.json`
Why: The plot_analysis experiment needed per-candidate manual grades written to disk for later aggregate analysis without doing any trend analysis during this pass.
Approach: Read each source result file, judged the four candidates against the generation prompt on the five agreed axes, then wrote one `_evaluation.json` file per movie in source-candidate order. Used concise rubric-aligned reasoning strings and set `judge_tokens` to `null` because this pass was manual rather than model-judged.
Design context: Follows the evaluation setup documented in the bucket file and current-session plot_analysis experiment context in this diff log.

## Tiered plot_analysis skip conditions + prompt hardening from evaluation findings
Files: `movie_ingestion/metadata_generation/pre_consolidation.py`, `movie_ingestion/metadata_generation/prompts/plot_analysis.py`

### Intent
Data-driven refinements based on the 80-movie × 4-candidate evaluation results. Two categories: (1) skip condition to filter out movies that produce low-quality output, and (2) prompt changes to fix the dominant failure modes in movies that should still run.

### Key Decisions
- **Tiered skip condition replaces boolean check:** Old check was `plot_synopsis OR thematic_observations OR emotional_observations`. New tiered logic: always eligible if plot_synopsis exists; else eligible if plot fallback >= 400 chars; else eligible if plot fallback >= 250 chars AND thematic_observations >= 300 chars; otherwise skip. `emotional_observations` removed from eligibility — evaluation showed observation richness doesn't reliably predict quality, and emotional obs specifically introduced noise on non-narrative content.
- **Thresholds:** `_MIN_PLOT_FALLBACK_CHARS = 400`, `_MIN_PLOT_FALLBACK_WITH_OBSERVATIONS_CHARS = 250`, `_MIN_THEMATIC_OBSERVATIONS_CHARS = 300`. Chosen from evaluation data: Revenge Quest (prompt=373, avg=2.30) was the clearest failure; movies above 400 chars of plot fallback consistently scored 4.2+.
- **`_check_plot_analysis` signature changed:** Now takes `(plot_synopsis, thematic_observations, movie_input)` instead of `(plot_synopsis, thematic_observations, emotional_observations)`. Needs movie_input to compute plot fallback length.
- **Prompt: emotional_observations scoped to tone only:** Explicit instruction that emotional observations are evidence for tone/mood, not for arcs, conflict, or plot structure. Addresses Queen: A Night at the Odeon divergence (conflict_accuracy 5→2 with emotional).
- **Prompt: stronger abstention for character_arcs and conflict_type:** Both fields now lead with a "FIRST: determine whether..." gate that explicitly lists non-narrative content types (documentaries, concerts, observational films, shorts, experimental, anthology) as empty-list cases. Addresses 31/42 evaluation failures being on arc_quality.
- **Prompt: concrete transformation required for arcs:** New rule: "Only emit an arc when the input shows a CONCRETE before→after transformation — not just a role, trait, goal, situation, or hardship." Addresses the specific mechanism where gpt-5-mini turns character situations into fabricated arcs.
- **Prompt: relaxed overview for non-narrative content:** generalized_plot_overview now has separate instructions for narrative vs non-narrative films, allowing docs/concerts to "describe the central subject, situation, or progression honestly" instead of forcing setup-beats-resolution.

### Testing Notes
- Unit tests for pre_consolidation will fail (`_check_plot_analysis` signature changed)
- Prompt changes need re-evaluation on the 80-movie set to measure impact
Testing notes: Did not run tests per repo instructions; output shape was kept consistent with the expected evaluation file contract.

## Generic metadata evaluation report generator and plot_analysis report
Files: `movie_ingestion/metadata_generation/evaluation_data/analyze_evaluations.py`, `movie_ingestion/metadata_generation/evaluation_data/plot_analysis_evaluation_report.md`, `docs/modules/ingestion.md`
Why: The existing analyzer only handled the older reception-style aggregate table, but this session needed the fuller `/evaluate-metadata-results` workflow applied to the newly written plot_analysis judgments.
Approach: Replaced the narrow script with a metadata-type-aware Markdown report generator that loads the prompt, schema, bucket definitions, per-movie result JSONs, and per-movie evaluation JSONs; computes aggregate scores, per-bucket patterns, low-score failures, divergence cases, and recommendations; and can save the rendered report next to the evaluation artifacts. Also updated the ingestion module doc note so it describes the analyzer's broader behavior accurately.
Design context: Mirrors `.claude/commands/evaluate-metadata-results.md`; uses the plot_analysis prompt/schema/bucket artifacts already created this session. Preserves simple file-based analysis flow in `evaluation_data/` rather than adding a separate evaluation subsystem.
Testing notes: Did not run unit tests per repo instructions. Verified by executing `python movie_ingestion/metadata_generation/evaluation_data/analyze_evaluations.py plot_analysis --save`, which produced the saved report successfully after adding repo-root path bootstrapping.

## Extracted best_plot_fallback into MovieInputData method
Files: `movie_ingestion/metadata_generation/inputs.py`, `movie_ingestion/metadata_generation/generators/plot_analysis.py`, `movie_ingestion/metadata_generation/pre_consolidation.py`
Why: The plot fallback selection logic (longest of first synopsis, longest plot_summary, overview) was duplicated between `_best_plot_fallback()` in the generator and inline code in `_check_plot_analysis()`. Extracted to `MovieInputData.best_plot_fallback()` in inputs.py so both consumers use a single implementation. Also fixed stale docstring in generator that still described the old boolean skip condition.

## Updated plot_analysis evaluation buckets for new eligibility criteria
Files: `ingestion_data/plot_analysis_eval_buckets.json`
Why: The tiered eligibility criteria made many movies in the evaluation set ineligible. Updated the bucket file to reflect the new rules.
Approach: Removed `thin_obs_overview_only` bucket entirely (9/10 movies ineligible by definition — thematic <=200 blocks T3, overview rarely >=400 for T2). Replaced 7 ineligible movies in each of `rich_obs_overview_only`, `moderate_obs_thin_plot`, and `thin_obs_summaries_fallback` with eligible replacements selected for genre/decade/rating diversity. Added `eligibility_criteria` metadata to the bucket file documenting the tiered thresholds. Total: 70 movies across 7 buckets (was 80 across 8).

## Made the metadata evaluation analyzer compatible with current plot_analysis artifacts
Files: `movie_ingestion/metadata_generation/evaluation_data/analyze_evaluations.py`
Why: Running the repo's `/evaluate-metadata-results` equivalent for `plot_analysis` failed because the current dataset is not perfectly rectangular: four emotional-observation candidates only have 49/70 coverage, and the bucket JSON uses a leaner schema than the analyzer expected.
Approach: Hardened the analyzer to aggregate per candidate over the movies it actually appears in, show per-candidate coverage in the overview and aggregate table, skip absent candidates in bucket/divergence sections, and treat `experiment_hypothesis` as optional in bucket metadata.
Design context: Keeps the analysis flow aligned with `.claude/commands/evaluate-metadata-results.md` while matching the current `plot_analysis` evaluation artifacts under `movie_ingestion/metadata_generation/evaluation_data/` and `ingestion_data/plot_analysis_eval_buckets.json`.
Testing notes: Verified by running `python movie_ingestion/metadata_generation/evaluation_data/analyze_evaluations.py plot_analysis --save`, which completed successfully and rewrote the saved Markdown report.

## Tightened plot_analysis prompt abstention and label-specificity rules
Files: `movie_ingestion/metadata_generation/prompts/plot_analysis.py`
Why: Head-to-head review of `kimi-k2.5-no-thinking` vs `gpt-5-mini-minimal-justifications` showed GPT's remaining gap is concentrated in `arc_quality` and, secondarily, `genre_precision`. The recurring failure mode was weakly supported arcs/conflicts on non-traditional narrative material and end-state labels that described a role or status instead of a transformation.
Approach: Strengthened the global omission preference, added sharper guidance for avoiding broad genre labels, made `conflict_type` explicitly reject contrasts/retrospective framings that are not active recurring struggles, and tightened `character_arcs` so static role/status labels are rejected in favor of empty lists unless a concrete before→after change is visible.
Design context: Intentionally kept the new wording general rather than overfitting to a few eval titles; the goal is to shift GPT-5-mini toward Kimi's better abstention/label-grounding behavior without introducing brittle case-specific examples.
Testing notes: Prompt-only change; did not run tests per repo instructions.

## Finalized plot_analysis generation config
Files: `movie_ingestion/metadata_generation/generators/plot_analysis.py`, `movie_ingestion/metadata_generation/prompts/plot_analysis.py`, `movie_ingestion/metadata_generation/schemas.py`

### Intent
Lock in the evaluated winner for plot_analysis generation: gpt-5-mini with minimal reasoning, low verbosity, justification schema, no emotional_observations input.

### Key Decisions
- **Model params hardcoded:** `generate_plot_analysis` no longer accepts `provider`, `model`, `system_prompt`, `response_format`, or `**kwargs`. Production config is `_PROVIDER=OPENAI`, `_MODEL=gpt-5-mini`, `reasoning_effort="minimal"`, `verbosity="low"`, `_RESPONSE_FORMAT=PlotAnalysisWithJustificationsOutput`.
- **emotional_observations removed:** Dropped from `build_plot_analysis_user_prompt`, `generate_plot_analysis`, and the system prompt's INPUTS section. Evaluation showed it introduced noise (especially on non-narrative content) without quality benefit.
- **Single prompt variant:** Removed no-justification `SYSTEM_PROMPT`, `_FIELDS_2_3_NO_JUSTIFICATIONS`, `_FIELD_5_NO_JUSTIFICATIONS` from prompts. `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` renamed to `SYSTEM_PROMPT` as the sole export.
- **Schema docstrings updated:** `PlotAnalysisWithJustificationsOutput` marked as production schema; `PlotAnalysisOutput` retained for backward compatibility with evaluation pipelines.

### Testing Notes
- Unit tests for plot_analysis generator will need updates (removed params, changed return type annotation)
- Notebooks using `SYSTEM_PROMPT_WITH_JUSTIFICATIONS` or `emotional_observations` not updated per user request

## Registered plot_analysis in the batch generation pipeline
Files: `movie_ingestion/metadata_generation/generator_registry.py`, `movie_ingestion/metadata_generation/result_processor.py`, `movie_ingestion/metadata_generation/inputs.py`
Why: plot_analysis generator was finalized but not wired into the batch CLI (eligibility/submit/status/process/autopilot).
Approach: Added plot_analysis to `GENERATOR_REGISTRY` and `SCHEMA_BY_TYPE`. Because plot_analysis is a Wave 2 type that depends on Wave 1 outputs (plot_synopsis from plot_events, thematic_observations from reception), the adapters load those via `load_wave1_outputs_for_movie()` in inputs.py, which queries the `generated_metadata` table at call time. This keeps the generic pipeline interface (`MovieInputData → result`) intact. The loader lives in inputs.py (the data-loading module) rather than the registry, since it's a reusable data-loading function that future Wave 2 types will also need.
Testing notes: Verified via import that registry and schema lookup both include plot_analysis, and CLI shows it as a valid `--metadata` choice.

## Viewer experience eligibility and input contract redesign
Files: `movie_ingestion/metadata_generation/pre_consolidation.py`, `movie_ingestion/metadata_generation/generators/viewer_experience.py`, `movie_ingestion/metadata_generation/prompts/viewer_experience.py`

### Intent
Align viewer_experience with the finalized conservative eligibility rules and the new upstream input ladder built around finalized plot_events, reception, and plot_analysis outputs.

### Key Decisions
- **Narrative input ladder:** Added strict winner-takes-all narrative resolution in both eligibility and prompt building: `plot_summary` if present, else `best_plot_fallback()` if at least 500 chars, else `generalized_plot_overview` if at least 200 chars.
- **Eligibility thresholds:** Viewer experience now passes only when narrative or observation inputs are strong enough on their own, or when usable narrative and observation inputs combine to enough signal. Support-only inputs like genre context, keywords, and maturity no longer rescue eligibility.
- **Observation inputs split:** Replaced the old `review_insights_brief` concept with explicit `emotional_observations`, `craft_observations`, and `thematic_observations`, each with its own inclusion threshold. This preserves which review-derived signal is doing the work instead of flattening everything into one blob.
- **Plot-analysis support inputs:** Added `genre_signatures` as preferred genre context and `character_arcs` as a supportive input for ending-aftertaste and emotional-volatility judgments. Raw genres remain the fallback when genre signatures are absent.
- **Prompt contract update:** The viewer_experience prompt now teaches the model about `narrative_input_source`, direct observation priority (`emotional > craft > thematic`), and the fact that genre context, keywords, and character arcs are supportive rather than primary evidence.

### Testing Notes
- Ran `python -m py_compile movie_ingestion/metadata_generation/pre_consolidation.py movie_ingestion/metadata_generation/generators/viewer_experience.py movie_ingestion/metadata_generation/prompts/viewer_experience.py`
- Did not run unit tests per repo instructions.

## Viewer experience eligibility tightening and narrative resolution unification
Files: `movie_ingestion/metadata_generation/pre_consolidation.py`, `movie_ingestion/metadata_generation/generators/viewer_experience.py`

### Intent
Tighten viewer_experience eligibility based on analysis of threshold quality, and unify the duplicated narrative resolution logic between eligibility and prompt building.

### Key Decisions
- **Unified narrative resolution:** Replaced `_resolve_viewer_experience_narrative_input` (pre_consolidation) and `_resolve_narrative_input` (generator) with a single public `resolve_viewer_experience_narrative()` in pre_consolidation. Both eligibility checking and prompt building now call the same function, eliminating divergence risk.
- **plot_summary inclusion floor:** Added 400-char minimum for plot_summary to be accepted as narrative input. Previously any-length plot_summary was included. Short plot_summaries (from sparse movies that barely cleared the 600-char plot_events input gate) now fall through to raw plot fallback.
- **generalized_plot_overview standalone raised:** 300 → 350 chars. A 300-char thematic abstract (2 layers of LLM abstraction from source) was too thin to anchor 8 sections of felt-experience generation alone.
- **Source-weighted combined path:** Replaced flat 650-char combined threshold with source-specific thresholds: plot_summary 550, best_plot_fallback 700, generalized_plot_overview 750. Higher-quality narrative sources need less observation supplementation; abstract sources need more.
- **Observation standalone unchanged:** Kept current thresholds (emotional >= 160 standalone, combined >= 280 with emotional or craft). Will verify via test buckets whether tightening is needed.
- **Generalized overview standalone kept:** Not dropped pending test verification on a bucket of movies that fall back to this source.

### Testing Notes
- Both files compile cleanly via py_compile
- Did not run unit tests per repo instructions
- Test buckets needed: (1) generalized_plot_overview-only movies, (2) observation-standalone movies

## Unified observation filtering between eligibility and generator
Files: `movie_ingestion/metadata_generation/pre_consolidation.py`, `movie_ingestion/metadata_generation/generators/viewer_experience.py`
Why: The generator had its own `_filter_observations()` with duplicated threshold constants (`_MIN_EMOTIONAL_OBSERVATIONS_CHARS`, etc.) mirroring pre_consolidation's `_viewer_experience_observation_lengths()`. Same inclusion logic, different return types (strings vs lengths), independently maintained thresholds that could silently diverge.
Approach: Replaced both with a single public `filter_viewer_experience_observations()` in pre_consolidation that returns filtered strings. The eligibility check derives lengths from the non-None results. Generator imports and calls the shared function directly — no local threshold constants remain.

## Viewer experience evaluation bucket design and movie selection
Files: `ingestion_data/viewer_experience_eval_buckets.json`, `ingestion_data/viewer_experience_eval_guide.md`

### Intent
Design evaluation buckets to answer threshold leniency, observation contribution, narrative source quality, and observation-standalone viability questions before running production viewer_experience generation.

### Key Decisions
- **6 buckets, 42 movies total:** gold_standard (8), floor_plot_summary (8), raw_fallback_standalone (8), raw_fallback_with_observations (6), obs_standalone_with_context (6), obs_standalone_minimal_context (6). Bucket 5 (generalized_plot_overview) deferred — plot_analysis has 0 generated rows.
- **Combined-edge bucket dropped:** Only 1 movie in the 95K dataset qualifies solely through the combined path. Observation-standalone (280 combined chars) is permissive enough that the combined path is nearly always redundant.
- **Observation-standalone split:** The 45,552-movie obs-standalone population (48% of movies with reception) was split by supporting context level: 200-499 char fallback (Bucket 6) vs <200 char fallback (Bucket 7). This tests whether narrative context below inclusion thresholds still helps.
- **Ablation candidates on Buckets 1 and 3 only:** `no_observations` and `no_thematic_no_arcs` variants. These have strong standalone narrative, so quality delta is attributable to observations, not confounded by weak narrative.
- **Diversity selection:** Movies chosen for genre, era, rating, and vote-count diversity within each bucket pool.

### Planning Context
Original plan had 7 buckets including combined-edge. Data exploration revealed the combined path barely exists in practice, leading to redesign. The observation-standalone dominance was the biggest finding — nearly half the eligible population has no usable narrative input.

## Narrative techniques eligibility and input contract redesign
Files: `movie_ingestion/metadata_generation/pre_consolidation.py`, `movie_ingestion/metadata_generation/generators/narrative_techniques.py`, `movie_ingestion/metadata_generation/prompts/narrative_techniques.py`, `movie_ingestion/metadata_generation/inputs.py`, `docs/modules/ingestion.md`, `ingestion_data/narrative_techniques_eval_guide.md`

### Intent
Redesign narrative_techniques eligibility and input contract from scratch based on systematic analysis of upstream data, downstream usage, and per-input signal quality.

### Key Decisions
- **Tiered eligibility replaces boolean OR check:** Old check was `plot_synopsis OR review_insights_brief OR (genres AND keywords)`. New tiered logic: Tier 1 (plot_summary → always eligible), Tier 2 (best_plot_fallback >= 500 chars), Tier 3 (craft_observations >= 450 chars standalone), Tier 4 (fallback >= 300 + craft >= 300 combined). Drops genres+keywords-only path (would produce genre-speculative output).
- **Split review_insights_brief into craft_observations:** The concatenated blob mixed three observation types. craft_observations carries direct structural technique signal; emotional_observations is irrelevant (wrong domain); thematic_observations is uncertain (needs testing, ~50% noise). Generator now receives craft_observations only.
- **merged_keywords replaces overall_keywords:** Structural technique tags can appear in either keyword list. Prompt handles noise via "use only when consistent with primary evidence."
- **Plot fallback with quality-tiered labels:** When plot_summary is absent, best_plot_fallback() provides raw plot text. The prompt label distinguishes `plot_synopsis` (LLM-condensed) from `plot_text` (raw human-written) so the LLM can calibrate confidence.
- **Shared narrative resolution function:** `resolve_narrative_techniques_narrative()` in pre_consolidation serves both eligibility and prompt building. Follows the viewer_experience pattern (`resolve_viewer_experience_narrative`).
- **craft_observations filtering in generator:** `_filter_craft_observations()` uses the combined-path threshold (300 chars) as the inclusion floor, since eligible movies may have craft in the 300-449 range via Tier 4.
- **Unified Wave 1 loader:** Replaced both `load_wave1_outputs_for_movie()` (tuple) and the new `load_wave1_outputs_for_narrative_techniques()` (tuple) with a single `load_wave1_outputs()` returning a `Wave1Outputs` dataclass. All Wave 1 fields (plot_summary, thematic/emotional/craft observations, source_material_hint) are loaded in one query; callers access by attribute name instead of positional tuple.
- **No plot_analysis dependency:** Despite plot_analysis outputs mapping closely to several NT sections (character_arcs, conflict_type, genre_signatures), adding it would require Wave 2 ordering + create echo risk. NT derives its own technique labels from primary evidence.
- **Population: 90,168 eligible (82.5%)** vs ~105K+ under old rules. The ~19K excluded movies have thin plot data AND moderate-but-not-great craft observations.

### Testing Notes
- Unit tests for narrative_techniques generator and pre_consolidation will fail (changed signatures, new thresholds, removed inputs)
- Prompt INPUTS section updated to document quality-tiered narrative inputs and craft_observations
- Evaluation guide saved to ingestion_data/narrative_techniques_eval_guide.md with 8 open questions for bucket evaluation

## New command: /improve-metadata-prompt
Files: `.claude/commands/improve-metadata-prompt.md`
Why: Need a structured process to evaluate and improve system prompts for each metadata generation type before running first production generations. Existing commands cover input contract design (`/analyze-metadata-inputs`) and post-generation evaluation (`/evaluate-metadata-results`), but nothing for pre-generation prompt quality review.
Approach: 4-phase conversational command parameterized by `$ARGUMENTS` (metadata type). Phase 1 silently reads pipeline context, prompt, generator, schema, eval guide, and 3-5 real movies from tracker DB. Phase 2 presents understanding for confirmation. Phase 3 evaluates current prompt across 5 dimensions (clarity, ordering, input spectrum coverage, token efficiency, output alignment) from the perspective of gpt-5-mini with minimal reasoning in single-shot batch mode. Phase 4 proposes improvements with reasoning and open questions. Pauses after phases 2 and 3 for discussion.

## Refactored cmd_process for clarity and expired batch support
Files: `movie_ingestion/metadata_generation/run.py`, `movie_ingestion/metadata_generation/result_processor.py`
Why: Expired OpenAI batches have partial results available but cmd_process skipped them. Control flow was also hard to follow (nested if/elif/continue chains).
Approach: Restructured around a `match` statement on batch status. Extracted `_download_and_process_output`, `_download_and_process_errors`, and `_log_batch_errors` helpers to eliminate duplication. Added `expired` handling (downloads partial output + error file, clears batch IDs for resubmission) and explicit `cancelled` branch. Each status branch is self-contained with no fallthrough. Also fixed `process_error_file` to handle `"response": null` entries from expired batches — `.get("response", {})` doesn't use the default when the key exists with a None value; changed to `or {}` pattern.

## Source of inspiration input contract and eligibility redesign
Files: `movie_ingestion/metadata_generation/pre_consolidation.py`, `movie_ingestion/metadata_generation/generators/source_of_inspiration.py`, `movie_ingestion/metadata_generation/prompts/source_of_inspiration.py`, `docs/modules/ingestion.md`, `ingestion_data/source_of_inspiration_eval_guide.md`

### Intent
Replace the blunt `review_insights_brief` input (concatenated observation blobs) with the targeted `source_material_hint` field from reception's extraction zone, and add an abstention gate to the prompt for small-model reliability.

### Key Decisions
- **`review_insights_brief` → `source_material_hint`:** The source material signal was never in the thematic/emotional/craft observations — it was in the dedicated `source_material_hint` field. Swapping eliminates ~100-200 tokens of irrelevant observation text and replaces it with a ~10-20 token targeted classifying phrase.
- **Eligibility unchanged in practice:** `_check_source_of_inspiration` now checks `merged_keywords OR source_material_hint` instead of `merged_keywords OR review_insights_brief`. Near-zero skip rate (~21 movies lack all keywords).
- **Abstention gate added:** `sources_of_inspiration` section now leads with a "FIRST: determine whether..." gate instructing the model to check inputs for evidence before attempting classification. Addresses parametric knowledge hallucination risk on small models.
- **Evidence priority in prompt:** Explicit ordering: (1) input evidence (hint + keywords), (2) parametric knowledge for well-known films, (3) omit when uncertain. Replaces the flat "parametric knowledge allowed" guidance.
- **No plot data added:** Analysis confirmed plot_synopsis doesn't solve the parametric knowledge problem — it triggers the same inference through a different mechanism at ~200-500 token cost. `source_material_hint` + keywords + title cover the signal gap.

### Testing Notes
- All modified files compile cleanly via py_compile
- Unit tests for source_of_inspiration generator and pre_consolidation will fail (changed param names)
- Evaluation guide saved to ingestion_data/source_of_inspiration_eval_guide.md with 6 hypotheses and 6 bucket designs

## Source of inspiration prompt hardening for small-LLM reliability
Files: `movie_ingestion/metadata_generation/prompts/source_of_inspiration.py`
Why: Pre-generation prompt review identified gaps in evidence-gathering guidance for production_mediums, keyword description underselling available signals, and vague "movie-agnostic" instruction.
Approach: Four targeted changes (~50 tokens total): (1) Broadened `merged_keywords` description to note genre/format keywords are relevant for production_mediums and indirect source signals. (2) Added evidence-gathering guidance to production_mediums section — tells model where to find medium signals in keywords. (3) Replaced vague "Do not state specifically what the source is" with concrete "no titles, authors, or proper nouns" rule with example. (4) Added precision instruction: "be as specific as the evidence supports" so model prefers "based on a graphic novel" over "based on a book" when keywords warrant it. (5) Tightened abstention gate wording for consistency with evidence priority section. (6) Replaced mixed-media rule with clearer significance threshold guidance.
Testing notes: Prompt-only change; compiles cleanly via py_compile. Did not run tests per repo instructions.

## Populated Bucket 5 (generalized_plot_overview) in viewer_experience evaluation
Files: `ingestion_data/viewer_experience_eval_buckets.json`, `ingestion_data/viewer_experience_eval_guide.md`
Why: Bucket 5 was deferred pending plot_analysis generation (78,262 movies now generated). Needed movies where generalized_plot_overview is the sole narrative source to test the 2-layer abstraction path.
Approach: Queried tracker.db for movies with no plot_summary >= 400 and no raw_fallback >= 500 but generalized_plot_overview >= 200 chars (22,120 candidates). Selected 8 movies with diversity across genre (animation, fantasy/sci-fi, action/crime, biography/history, horror/comedy, documentary, film-noir/sport, romance), era (1947-2025), vote count (4.8K-138K), GPO length (320-537), observation quality (941-1550), and narrative path (5 standalone gpo >= 350, 3 combined gpo 200-349). Also updated deferred_candidates, data_availability_notes, eval guide movie counts, generation totals, and data availability section to reflect plot_analysis completion.

## Renamed plot_synopsis → plot_summary in Wave 2 generator parameters
Files: `movie_ingestion/metadata_generation/generators/viewer_experience.py`, `movie_ingestion/metadata_generation/generators/plot_analysis.py`, `movie_ingestion/metadata_generation/pre_consolidation.py`, `movie_ingestion/metadata_generation/generator_registry.py`, `movie_ingestion/metadata_generation/inputs.py`, `movie_ingestion/metadata_generation/schemas.py`
Why: The `plot_synopsis` parameter name in Wave 2 generators was misleading — it held the LLM-generated `plot_summary` from PlotEventsOutput, not the raw IMDB plot synopsis. Also added `= None` default to `build_viewer_experience_user_prompt()` to match `generate_viewer_experience()` signature (was causing TypeError when called via `**kwargs` unpacking in the notebook).
Approach: Renamed the Python parameter/variable to `plot_summary` everywhere it represents the Wave 1 output. Left `plot_synopsis` intact in LLM prompt labels (kwargs to `build_user_prompt()`) and system prompt strings, since those are the labels the LLM expects. Also left `plot_events.py` unchanged since it correctly uses raw IMDB `plot_synopses[0]`.

## Parallelized viewer_experience playground generation across all movies
Files: `movie_ingestion/metadata_generation/metadata_generation_playground.ipynb` (Cell 8)
Why: The generation loop ran movies sequentially (only parallelizing 3 candidates per movie). With high rate limits, all (movie × candidate) pairs can fire at once.
Approach: Replaced the nested `for bucket... for movie... await gather(candidates)` loop with a flat list of all (movie, candidate) coroutines launched via a single `asyncio.gather()`. Results are grouped back per-movie after completion for saving. The existing semaphore (40 concurrent) and token-bucket rate limiter (120/sec) still control concurrency. Added timing output and total cost/error summary.

## Viewer experience per-result scoring rubric
Files: `ingestion_data/viewer_experience_eval_guide.md`
Why: Needed a structured rubric for evaluating individual viewer_experience metadata results before running evaluations across the 50-movie eval set.
Approach: Designed 6 weighted axes (groundedness 0.25, specificity layering 0.20, retrieval alignment 0.20, section discipline 0.15, negation quality 0.10, term quality & diversity 0.10) plus an unweighted holistic score. Each axis has 1-5 descriptors. Key design choice: specificity axis rewards BOTH broad genre-level terms AND movie-specific differentiators (user explicitly corrected initial framing that penalized generic terms). Inserted before "After This Evaluation" section.

## /create-eval-rubric command
Files: `.claude/commands/create-eval-rubric.md`
Why: Reusable command for creating per-result scoring rubrics for any metadata type's evaluation, following the same process used for viewer_experience.
Approach: Scoped context reads to exactly 5 files (prompt, schema, eval guide, one sample result, viewer_experience rubric as template). 4-step workflow: read context → think through axes → present for user feedback → write after approval. Explicitly encodes the "generic terms are good" principle.

## /evaluate-metadata-results command
Files: `.claude/commands/evaluate-metadata-results.md`
Why: Reusable command for running per-movie evaluations against a rubric, producing structured JSON evaluation files.
Approach: Reads eval guide (for rubric), schema (for __str__ to know what's graded), prompt (for input awareness), and all result files. Key design principles: (1) grade only fields in __str__() — ignore justifications, (2) grade for retrieval quality not prompt compliance, (3) groundedness judged against LLM's inputs only unless schema explicitly allows parametric knowledge, (4) single-pass per movie with all candidates evaluated but scored independently, (5) outputs to {type}_{tmdb_id}_evaluation.json alongside result files.
