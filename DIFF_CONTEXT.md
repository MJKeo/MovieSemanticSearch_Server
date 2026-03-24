# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

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
