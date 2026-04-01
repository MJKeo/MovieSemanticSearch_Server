# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Watch context prompt rebuild and eval guide creation
Files: movie_ingestion/metadata_generation/prompts/watch_context.py, movie_ingestion/metadata_generation/schemas.py, ingestion_data/watch_context_eval_guide.md

### Intent
Rebuild the watch_context system prompt from scratch and create an evaluation
guide with A/B test tracking for Round 1 evaluation.

### Key Decisions
- **All term ranges 0-N** (was 4-8, 1-4, 1-4, 3-6): nonzero floors forced
  fabrication on sparse inputs. Schema field descriptions updated to match.
- **Explicit coverage principle**: "produce fewer terms when inputs are sparse"
  as a general principle rather than per-case rules.
- **Trimmed examples** (3-5 per section, was 6-15+): long example lists risk
  becoming vocabulary menus for low-reasoning models. A/B testing in Round 1
  will validate whether this is sufficient (eval guide H2).
- **Consistent section format**: replaced mixed rhetorical questions / direct
  instructions with uniform "captures / signal sources / examples" template.
- **Non-narrative content acknowledged**: documentaries, shorts, etc. get a
  principle-level note rather than enumerated special cases.
- **Suggestive signal routing**: input→section mapping changed from prescriptive
  ("Use to infer X") to suggestive ("strongest for X, but relevant across all").
  A/B testing in Round 1 will validate (eval guide H3).
- **Justification A/B test**: eval guide H1 tracks with-vs-without justifications.

### Testing Notes
- No generation run yet — prompt is pre-production (0/109K movies generated).
- Eval guide defines 5 input quality buckets and 4 hypotheses (3 A/B tests + 1 measurement).
- Schema changes are backward-compatible (added Field descriptions, no structural changes).

## Narrative techniques R2 evaluation decisions — prompt and eval guide updates
Files: movie_ingestion/metadata_generation/prompts/narrative_techniques.py, ingestion_data/narrative_techniques_eval_guide.md, ingestion_data/narrative_techniques_eval_buckets.json
Why: Apply decisions from R2 evaluation analysis (56 movies, 8 candidates, 7 buckets).
Approach:
- **Prompt**: Relaxed anti-duplication rule from global to within-section only. Cross-section overlap is acceptable for semantic retrieval.
- **Eval guide**: Updated Term Quality rubric axis to match (within-section uniqueness only). Updated input contract (craft threshold 450→400, thematic_observations excluded). Replaced speculative "After This Evaluation" section with a "Resolved Decisions" table covering all Q1-Q8 + R2 questions. Added Round 2 Learnings section with findings.
- **Eval buckets JSON**: Updated experiment_description to reflect resolved status. Added `candidates_r2_actual` alongside original design. Updated tier3 threshold to 400.
Design context: All questions now resolved. Production config is gpt-5-mini-low with justifications, merged keywords, no thematic observations, ACP kept, craft threshold at 400.

## Source-of-inspiration playground evaluation cell
Files: movie_ingestion/metadata_generation/metadata_generation_playground.ipynb
Why: Add the missing notebook cell for running `source_of_inspiration` playground evaluations with the requested candidate matrix and preserve prior per-movie evaluation results.
Approach: Replaced the placeholder notebook cell with a `source_of_inspiration` evaluation cell that loads eval-bucket movies, builds `MovieInputData`, extracts only `source_material_hint` from `generated_metadata.reception`, runs six OpenAI candidates (gpt-5-mini minimal/low and gpt-5.4-nano medium, each with and without the justifications prompt/schema) behind a 60-call semaphore, and merges new candidate results into existing `evaluation_data/source_of_inspiration_{tmdb_id}.json` files whether prior results are stored as a dict or legacy list.
Design context: Follows the established notebook evaluation pattern in `metadata_generation_playground.ipynb` and the `source_of_inspiration` generator contract in `movie_ingestion/metadata_generation/generators/source_of_inspiration.py`, where Wave 2 input dependency is only `source_material_hint`.
Testing notes: Did not run the notebook generation workload per repo guidance, but verified the notebook JSON parses and the inserted cell compiles with top-level `await`; main runtime risk is external API execution behavior when the cell is actually run.

## Watch context eval guide — new buckets and eval bucket selection
Files: ingestion_data/watch_context_eval_guide.md, ingestion_data/watch_context_eval_buckets.json
Why: Expand input quality buckets for evaluation coverage and create movie selections from actual pipeline data.
Approach:
- **New buckets added**: genre_signals_only (genre_signatures but no observations), sparse_full_coverage (all 3 observations but thin 15-60 words), single_observation (exactly 1 obs field).
- **single_observation removed**: DB query revealed only 11 movies in the entire pipeline have exactly 1 observation field (10 craft-only, 1 thematic-only, 0 emotional-only). Too rare to form a viable eval bucket. Deleted from both eval guide and buckets JSON.
- **Eval buckets JSON created**: 7 buckets × 8 movies each, all selected via SQL queries against tracker.db matching each bucket's criteria (observation lengths, genre_signatures presence, keyword counts, maturity ratings, documentary genre). Movies chosen for genre/style diversity within each bucket.
- **Final bucket set (7)**: gold_standard, mid_observations, genre_only_floor, genre_signals_only, sparse_full_coverage, documentary_nonfiction, maturity_edge.
Design context: Bucket criteria defined in ingestion_data/watch_context_eval_guide.md Section 7.

## Narrative techniques R3 evaluation pass
Files: movie_ingestion/metadata_generation/evaluation_data/narrative_techniques_*_evaluation.json
Why: Fulfill the `/evaluate-metadata-results narrative_techniques` workflow against the current R3 candidate-result files and persist per-movie scoring outputs for downstream review.
Approach: Read the live narrative-techniques eval guide, schema `__str__()` contract, prompt, and all 56 current result files; then wrote one `_evaluation.json` per movie using the rubric axes required by the command (groundedness, technique_coverage, technique_abstraction, section_discipline, retrieval_alignment, term_quality, holistic). Scoring was anchored to the embedded fields only and to what each file's `user_prompt` actually exposed, with a null-result candidate scored as unusable.
Design context: Follows `.claude/commands/evaluate-metadata-results.md` and the narrative-techniques rubric in `ingestion_data/narrative_techniques_eval_guide.md`. Evaluations intentionally judge retrieval usefulness of the embedded terms rather than prompt compliance, matching the guide's downstream-usage framing.
Testing notes: Spot-checked generated outputs at `narrative_techniques_278_evaluation.json` and `narrative_techniques_455839_evaluation.json`; 56 evaluation files were written on disk. These evaluation artifacts appear to be gitignored in this repo, so `git status` may not list them even though they exist.

## Lock narrative techniques generator to production config
Files: movie_ingestion/metadata_generation/generators/narrative_techniques.py
Why: Align the live generator with the finalized production contract instead of allowing caller overrides for prompt, schema, provider, or model.
Approach: Removed the configurable `provider`, `model`, `system_prompt`, `response_format`, and `**kwargs` parameters from `generate_narrative_techniques()`, switched the generator to the justifications prompt/schema variant, and hardcoded the OpenAI `gpt-5-mini` call with `reasoning_effort="minimal"` and `verbosity="low"`.
Design context: Matches the locked-generator pattern already used by other finalized Wave 2 generators and the project priority of stable ingestion-time metadata generation settings documented in `docs/PROJECT.md` and `docs/modules/ingestion.md`.
Testing notes: No tests run per repo instructions for this task; change is a focused generator signature/config update only.

## Wire narrative techniques into the generic batch pipeline
Files: movie_ingestion/metadata_generation/generator_registry.py, movie_ingestion/metadata_generation/result_processor.py
Why: Make `narrative_techniques` work end-to-end with the existing generic eligibility, batch request, and batch result processing flow.
Approach: Added registry adapters that load the required Wave 1 inputs (`plot_summary`, `craft_observations`) from `load_wave1_outputs()`, then wired `MetadataType.NARRATIVE_TECHNIQUES` to its production prompt builder, live generator, justifications schema, and finalized model kwargs. Added the same schema to `result_processor.SCHEMA_BY_TYPE` so completed batch outputs validate and persist correctly.
Design context: Follows the same Wave 2 registry pattern already used for `plot_analysis` and `viewer_experience`, where the batch pipeline stays generic and type-specific upstream dependencies are resolved inside thin adapters.
Testing notes: No tests run per repo instructions; request building and result processing are generic, so the main risk was missing one of the registry/schema integration points, which is now covered in both places.

## Streamline narrative techniques schema and prompt (11 → 9 sections)
Files: movie_ingestion/metadata_generation/schemas.py, movie_ingestion/metadata_generation/prompts/narrative_techniques.py, implementation/classes/schemas.py, implementation/prompts/vector_metadata_generation_prompts.py

### Intent
Reduce narrative_techniques from 11 to 9 sections based on R3 evaluation
analysis (56 movies, 7 candidates). Optimizes for gpt-5-mini with minimal
reasoning — the chosen production model.

### Key Decisions
- **Removed `thematic_delivery`**: top hallucination source across all
  candidates. Models consistently over-inferred delivery mechanisms from
  theme topics, failing the FIRST: gating check. Removing it improves
  groundedness without losing meaningful retrieval signal (theme delivery
  is already captured in plot_analysis vectors).
- **Merged `meta_techniques` into `additional_narrative_devices`**: meta
  techniques (fourth-wall breaks, genre deconstruction) are rare narrative
  devices that fit naturally as "additional devices." Merging eliminates
  a section that was empty for most movies.
- **Renamed `additional_plot_devices` → `additional_narrative_devices`**:
  broader name reflects the merged scope (now includes meta/self-aware
  elements, not just plot mechanics).
- **Moved catchall section to last position**: `additional_narrative_devices`
  is now section 9 (was section 3). Placed last so the model fills specific
  sections first, then deposits remaining devices into the catchall.
- **Trimmed reusability test examples** in _PREAMBLE from 8 to 4 (model
  already scores 4.0 on technique_abstraction).
- Updated both ingestion-side schemas (NarrativeTechniquesOutput,
  NarrativeTechniquesWithJustificationsOutput) and search-side schema
  (NarrativeTechniquesMetadata) plus the search-side prompt.

### Testing Notes
- Existing evaluation data uses the old 11-section schema and cannot be
  directly compared to new 9-section output.
- Test files reference old field names and will need updating.

## Re-run eligibility checks even when flags already exist
Files: movie_ingestion/metadata_generation/run.py
Why: The `eligibility` CLI command should refresh `eligible_for_<type>` for every `imdb_quality_passed` movie instead of skipping rows that were already evaluated, so reruns can pick up changed upstream inputs without manual flag clearing.
Approach: Removed the `eligible_for_<type> IS NULL` filter from `cmd_eligibility()`, reused the full `imdb_quality_passed` ID set as the evaluation target, and updated the command messaging to describe a full re-evaluation pass rather than only newly evaluated rows.
Design context: Matches the requested workflow for Stage 6 eligibility reruns in `movie_ingestion/metadata_generation/run.py`, where re-checking an already eligible movie is acceptable if upstream data changed.
Testing notes: No tests run per repo instructions; change is limited to CLI eligibility selection and status output.
