# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Add watch_context to batch generation flow
Files: movie_ingestion/metadata_generation/generator_registry.py, movie_ingestion/metadata_generation/result_processor.py
Why: `generate_watch_context()` was finalized, but the generic batch pipeline still could not build, submit, or validate watch-context batches because the metadata type was missing from the registry-backed integration path.
Approach: Added watch_context Wave 2 adapters in the generator registry that load plot_analysis genre_signatures plus Wave 1 observation fields, then registered the type with its locked production schema/prompt/model config. Added the same schema to `result_processor.SCHEMA_BY_TYPE` so completed batch outputs validate and store correctly.
Design context: Follows the existing Wave 2 integration pattern used by viewer_experience and narrative_techniques in `movie_ingestion/metadata_generation/generator_registry.py`. Uses the finalized watch-context production contract: identity_note schema + OpenAI gpt-5-mini with minimal reasoning.
Testing notes: Ran `python -m py_compile movie_ingestion/metadata_generation/generator_registry.py movie_ingestion/metadata_generation/result_processor.py` for a lightweight syntax/import sanity check. Did not run tests per repo instructions.

## Finalized watch_context generator to r4-identity-note-minimal config
Files: movie_ingestion/metadata_generation/generators/watch_context.py
Why: Round 4 evaluation confirmed identity_note variant works across both
gold_standard and challenging_identity buckets. Locking production config.
Approach: Removed all configurable params (provider, model, system_prompt,
response_format, **kwargs) from generate_watch_context(). Hardcoded the
winning config: OpenAI gpt-5-mini, reasoning_effort=minimal, verbosity=low,
SYSTEM_PROMPT_WITH_JUSTIFICATIONS, WatchContextWithIdentityNoteOutput.
Note: system prompt is SYSTEM_PROMPT_WITH_JUSTIFICATIONS (what was tested),
not SYSTEM_PROMPT_WITH_IDENTITY_NOTE (which has explicit identity_note
instructions but wasn't the tested config).
Testing notes: unit tests in test_watch_context_generator.py pass old kwargs
to generate_watch_context and will need updating.

## Watch context prompt and schema updates from Round 3 evaluation analysis
Files: movie_ingestion/metadata_generation/prompts/watch_context.py, movie_ingestion/metadata_generation/schemas.py, movie_ingestion/metadata_generation/generators/watch_context.py

### Intent
Apply Round 3 evaluation findings (50 movies, 9 candidates, 6 buckets) to
the watch_context generation pipeline. Three changes based on hypothesis
test results and failure pattern analysis.

### Key Decisions
- **H3 resolved: removed per-section "Signal sources" routing.** Round 3
  showed removing prescriptive signal-source lines produces equivalent or
  better quality (+0.03 composite overall, +0.26 on documentary_nonfiction,
  +0.20 on mid_observations). The general cross-section routing note in
  _INPUTS ("Each observation field is strongest for its namesake section but
  may inform any section. Use your judgment") remains as sole routing
  guidance. Per DIFF_CONTEXT entry from the H3 candidate setup.
- **H12 evolved: identity_note replaces viewing_appeal_summary.** Round 3
  showed the full-sentence viewing_appeal_summary improved identity accuracy
  on challenging movies (+0.32 composite on challenging_identity) but
  over-constrained output on rich-input movies (-0.49 on gold_standard).
  Root cause analysis: the summary was detailed enough (20-30 words) to act
  as a template the model faithfully expanded, suppressing secondary signals
  and creative phrasings. Fix: replace with a 2-8 word identity_note
  classification field. Brief enough to prime tone register (sincere vs
  ironic vs camp etc.) without providing a template to follow. No "anchor"
  or consistency constraint language — just classification then generation.
  New schema: WatchContextWithIdentityNoteOutput. New prompt:
  SYSTEM_PROMPT_WITH_IDENTITY_NOTE.
- **Coverage principle expanded for distinguishing context.** Round 3 failure
  analysis found the model ignoring distinguishing signals in
  overall_keywords and genre_signatures (e.g., Korean identity despite
  "Korean" in keywords). Added general guidance: "If overall_keywords or
  genre_signatures surface a distinctive identity — a national cinema, a
  cultural movement, a specific audience community — let that inform your
  terms."

### Planning Context
- H2 (expanded examples) resolved: no benefit overall, helps sparse but
  hurts rich. Keep 3-5 examples.
- H4 (thematic observations) resolved: clearly valuable, -0.17 composite
  and -0.52 coverage without them. Keep thematic observations.
- H7 (short observations) resolved: no degradation from thin observations.
  No per-field thresholds needed.
- Nano models confirmed non-viable: R3 high reasoning performed worse than
  R2 medium reasoning. All low scores in eval set belong to nano candidates.
- gpt-5-mini baselines are rock-solid across rounds: R2 and R3 identical at
  4.33 composite on shared movies.

### Testing Notes
- WatchContextWithViewingAppealOutput renamed to
  WatchContextWithIdentityNoteOutput. viewing_appeal_summary field replaced
  by identity_note. SYSTEM_PROMPT_WITH_VIEWING_APPEAL replaced by
  SYSTEM_PROMPT_WITH_IDENTITY_NOTE. Generator export updated. Any evaluation
  pipeline code referencing old names will need updating.
- The identity_note approach needs evaluation to confirm it preserves the
  challenging_identity gains while reducing gold_standard regression.
  Recommend running a focused eval on those two buckets before production.

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

## Watch context evaluation pass
Files: movie_ingestion/metadata_generation/evaluation_data/watch_context_*_evaluation.json, DIFF_CONTEXT.md
Why: Fulfill the `/evaluate-metadata-results watch_context` workflow against the current watch-context candidate result files and persist per-movie evaluation outputs for downstream review.
Approach: Read the watch-context eval guide, schema embedding contract, prompt, and all 56 current result files, then wrote one `_evaluation.json` per movie with rubric scores for groundedness, retrieval_alignment, section_discipline, coverage, and holistic quality. The scoring lens was anchored to each file's `user_prompt` evidence and to the embedded terms only, with sparse-input buckets penalized for overgenerated specifics and rich-input buckets rewarded for grounded downstream-useful occasion/motivation terms.
Design context: Follows `.claude/commands/evaluate-metadata-results.md` and `ingestion_data/watch_context_eval_guide.md`. The evaluations intentionally judge downstream retrieval usefulness based on what was present in `user_prompt`, not on system-prompt compliance.
Testing notes: Spot-checked generated evaluation artifacts for `watch_context_155556_evaluation.json`, `watch_context_294838_evaluation.json`, and `watch_context_10086_evaluation.json`; 56 evaluation files were written on disk.

## Watch context per-result scoring rubric
Files: ingestion_data/watch_context_eval_guide.md
Why: Add a formal scoring rubric for evaluating individual watch_context generation outputs, matching the viewer_experience rubric structure.
Approach: 4 weighted axes (Groundedness 0.30, Retrieval Alignment 0.25, Section Discipline 0.25, Coverage 0.20) + unweighted holistic score. Axes designed through iterative discussion: removed Specificity Layering (user doesn't want to penalize generics independently), merged synonym variation into Retrieval Alignment, created explicit Coverage axis as complement to Groundedness ("did we miss anything?" vs "did we add anything wrong?"), placed sanitization penalty in Coverage (sanitizing = removing information that should be present). No negation axis (watch_context has no negation fields, unlike viewer_experience).
Design context: Follows viewer_experience rubric format in the same eval guide. Axes informed by eval guide Section 5 evaluation axes and Round 1/2 failure mode analysis from viewer_experience.

## Watch context Phase 1 evaluation response — eligibility, rubric, and justification reframing
Files: movie_ingestion/metadata_generation/schemas.py, movie_ingestion/metadata_generation/prompts/watch_context.py, movie_ingestion/metadata_generation/pre_consolidation.py, movie_ingestion/metadata_generation/generators/watch_context.py, ingestion_data/watch_context_eval_guide.md, ingestion_data/watch_context_eval_buckets.json

### Intent
Apply Phase 1 evaluation findings (56 movies, 4 candidates, 7 buckets) to the
watch_context generation pipeline. Three changes based on evaluation data analysis.

### Key Decisions
- **Eligibility tightened: require ≥1 observation field.** Phase 1 showed all
  candidates score 1.6-2.5 on genre-only movies (genre_only_floor and
  genre_signals_only buckets). Terms are generic, undifferentiated, and may be
  inaccurate without observation grounding. Affects ~0.7% of pipeline (776 of
  109K movies). `_check_watch_context()` now requires `emotional_observations OR
  craft_observations OR thematic_observations` in addition to genre data. Call
  site in `assess_skip_conditions()` updated to pass observation fields through.
- **Justification → evidence_basis reframing.** Phase 1 showed justification
  candidates scored lower than no-justification on every axis. Root cause:
  justifications acted as post-hoc rationalization ("why I generated these")
  rather than upstream constraint ("what evidence do I have?"). Renamed field
  from `justification` to `evidence_basis`. Schema description now says "quote or
  closely paraphrase specific input phrases" with explicit "write 'No direct
  evidence' and leave terms empty" instruction. Prompt variant updated to frame
  as evidence inventory that constrains terms, not explains them.
- **Eval rubric updated: generic-but-accurate vs. inaccurate distinction.**
  Groundedness axis rewritten to distinguish three categories: grounded (traceable),
  generic-but-accurate (safe genre inference), and inaccurate/risky inference
  (experiential claims that could be wrong). Coverage axis updated to penalize
  missing specificity when inputs support it, but not penalize generic terms
  alongside specific ones.
- **Genre-only eval buckets removed.** genre_only_floor (8 movies) and
  genre_signals_only (8 movies) removed from eval_buckets.json since these
  movies are no longer eligible for generation. Eval guide bucket definitions
  removed. Hypotheses H6 and H10 marked as resolved.

### Planning Context
Analysis of Phase 1 data revealed: (1) no-justification + minimal reasoning
is the clear winner (3.73 overall, cheapest at $86/100K), (2) justifications
hurt quality via rationalization mechanism, (3) 99.3% of pipeline has
observation data so the eligibility gate has minimal impact, (4) the rubric
was over-penalizing generic-but-accurate inference while under-distinguishing
from genuinely inaccurate inference.

### Testing Notes
- `_check_watch_context()` signature changed: new optional params
  `emotional_observations`, `craft_observations`, `thematic_observations`.
  Existing unit tests in test_pre_consolidation.py use the old 2-param
  signature and will need updating.
- `TermsWithJustificationSection.justification` renamed to `evidence_basis`.
  Any code referencing the old field name will break.
- Eval bucket count reduced from 7 to 5 (56→40 movies in eval set).

## Evaluate watch_context generations against user_prompt-grounded retrieval quality
Files: movie_ingestion/metadata_generation/evaluation_data/watch_context_*_evaluation.json
Why: The watch_context eval set needed per-movie candidate scoring written out under the current rubric, with judgments anchored to what each model actually saw in `user_prompt` and to downstream embedding usefulness rather than prompt-compliance.
Approach: Read the command instructions, watch_context eval guide, bucket file, schema embedding contract, prompt file, and every watch_context result JSON. Wrote 56 `_evaluation.json` files covering 376 candidate evaluations with rubric scores for groundedness, retrieval_alignment, section_discipline, coverage, and holistic quality. The scoring lens stayed focused on embedded terms only, used the supplied `user_prompt` as the evidence base, and treated unsupported scenario/motivation invention as noise even when the system prompt might have encouraged more generation.
Design context: Follows `.claude/commands/evaluate-metadata-results.md` and `ingestion_data/watch_context_eval_guide.md`. This pass explicitly prioritizes the user’s instruction to judge from `user_prompt` evidence and downstream watch-context retrieval behavior, not from system-prompt desiderata.
Testing notes: No tests were run per repo instructions. Spot-checked generated evaluation JSONs after writing them, including rich-input, sparse-input, and trainwreck/bad-movie cases, to make sure the written reasoning matched the intended scoring lens.

## Watch context Phase 2 prompt/schema/eval improvements
Files: movie_ingestion/metadata_generation/prompts/watch_context.py, movie_ingestion/metadata_generation/schemas.py, movie_ingestion/metadata_generation/generators/watch_context.py, ingestion_data/watch_context_eval_guide.md, ingestion_data/watch_context_eval_buckets.json

### Intent
Apply Phase 2 evaluation findings (40 movies, 9 candidates across gpt-5-mini
and gpt-5.4-nano) to improve prompt, schema, and evaluation infrastructure.

### Key Decisions
- **Evidence_basis interpretation guidance added.** Phase 2 showed the evidence_basis
  mechanism works overall (+0.88 section discipline for mini-low) but can backfire
  when the model misinterprets cited evidence (Bigfoot: nano cited "terrible animation"
  → generated "nightmare fuel" instead of "so bad it’s good"). Added prompt guidance
  to verify terms reflect how viewers would RESPOND TO cited evidence, not just the
  topic. Schema description updated to match.
- **Coverage counterbalance added.** The coverage principle was one-sided — all about
  restraint. Data showed gpt-5-mini-minimal scores 3.98 groundedness but only 3.02
  coverage (playing it too safe). Added "when inputs DO provide specific signals,
  capture them" counterbalance.
- **Dynamic input-richness guidance.** Instead of asking the LLM to self-classify
  input richness (wastes reasoning tokens, risk of wrong classification), Python-side
  `_classify_input_richness()` measures observation presence/length and prepends a
  calibration line to the user prompt. Rich inputs get no extra guidance; sparse inputs
  get "favor directly-informed sections, keep others empty."
- **Viewing appeal pre-anchor schema (H12 A/B test).** New
  `WatchContextWithViewingAppealOutput` adds a `viewing_appeal_summary` field before
  sections. Forces identity classification before term generation. Corresponding
  `SYSTEM_PROMPT_WITH_VIEWING_APPEAL` prompt variant added. Not adopted for production
  yet — tracked as H12 hypothesis for next evaluation round.
- **Nano models excluded from production consideration.** All nano candidates failed
  catastrophically on identity-ambiguous movies (all scored 1 on Bigfoot while all
  mini candidates scored 4). Failure mode is identity misread, not output calibration.
- **H1 marked RESOLVED.** Evidence_basis works. `WatchContextWithJustificationsOutput`
  is the recommended production schema.
- **New `challenging_identity` bucket added to eval guide.** Tests tone-genre mismatch,
  quality-as-identity, mixed-valence reception, non-obvious experiential appeal, and
  polarizing movies. Empty in eval_buckets.json — needs movie selection.
- **Test set expansion guidance added.** Section 7.1 in eval guide provides criteria
  for when/how to add movies and test set hygiene rules.

### Testing Notes
- `_classify_input_richness()` is a new function — no existing tests cover it.
- `WatchContextWithViewingAppealOutput` and `SYSTEM_PROMPT_WITH_VIEWING_APPEAL` are
  new — evaluation pipeline callers need to import them for H12 testing.
- `challenging_identity` bucket is empty — needs movie selection before next eval round.
- Evidence_basis schema description changed — evaluation prompts that reference the
  old description text may need updating.

## Populate challenging_identity eval bucket with 10 movies
Files: ingestion_data/watch_context_eval_buckets.json
Why: The bucket was empty after Phase 2 added it. Needed movie selection before H12 A/B test.
Approach: Queried tracker.db for watch_context-eligible movies (genres + ≥1 observation) with
>10K IMDB votes, filtered by text signals matching the 5 challenge categories (tone-genre
mismatch, quality-as-identity, mixed-valence, polarizing, non-obvious appeal). Selected 10
movies spanning diverse genres and multiple categories: Thank You for Smoking (9388),
Showgirls (10802), Malignant (619778), Sucker Punch (23629), Beau Is Afraid (798286),
House of 1000 Corpses (2662), Poor Things (792307), From Dusk Till Dawn (755),
Love Me If You Dare (8424), Dawn of the Dead 2004 (924). All verified eligible via
check_watch_context() criteria.

## Watch context H4 hypothesis test: no thematic observations candidate
Files: movie_ingestion/metadata_generation/prompts/watch_context.py, movie_ingestion/metadata_generation/generators/watch_context.py, metadata_generation_playground.ipynb
Why: Set up r3-no-thematic-obs candidate to test whether thematic_observations add value for experiential reframing (H4 in eval guide).
Approach: Created no-thematic prompt variants (`_INPUTS_NO_THEMATIC`, `_SECTIONS_NO_THEMATIC`, `SYSTEM_PROMPT_WITH_JUSTIFICATIONS_NO_THEMATIC`) that remove all thematic_observations references. Added `exclude_thematic` parameter to `build_watch_context_user_prompt` and `generate_watch_context` to omit thematic_observations from user prompt entirely. Updated notebook cell 9 to single `r3-no-thematic-obs` candidate (gpt-5-mini, low reasoning, justifications). These prompt/generator changes are temporary — will be reverted after evaluation.

## Watch context H3 hypothesis test: no signal sources candidate
Files: movie_ingestion/metadata_generation/prompts/watch_context.py, movie_ingestion/metadata_generation/generators/watch_context.py, movie_ingestion/metadata_generation/metadata_generation_playground.ipynb
Why: Set up r3-no-signal-sources candidate to test whether removing per-section "Signal sources" routing guidance produces equivalent or better section assignment (H3 in eval guide).
Approach: Created `_SECTIONS_NO_SIGNAL_SOURCES` variant identical to `_SECTIONS` but with all "Signal sources:" lines removed. Assembled as `SYSTEM_PROMPT_NO_SIGNAL_SOURCES` using `_OUTPUT_WITH_JUSTIFICATIONS` (evidence_basis). The general cross-section routing in `_INPUTS` ("Each observation field is strongest for its namesake section but may inform any section. Use your judgment") remains as the sole routing guidance. Exported from generator module. Updated notebook cell 9 to single `r3-no-signal-sources` candidate (gpt-5-mini, low reasoning, justifications).

## Watch context Round 3 hypothesis test candidate planning
Files: ingestion_data/watch_context_eval_guide.md
Why: Define which hypotheses get dedicated Round 3 candidates and document them in the eval guide.
Approach: Added "Round 3 Hypothesis Test Candidates" section with 4 candidates, each isolating one variable from the r3-gpt5mini-low-just baseline. Marked H5 (genre_signatures vs raw genres), H8 (sparse input behavior), and H9 (explicit language) as REMOVED with rationale. H7 (short observation impact) kept as hypothesis but no dedicated candidate — will be assessed observationally via sparse_full_coverage bucket results. Run order: H12 → H4 → H2 → H3.

Final candidate set:
- `r3-viewing-appeal` (H12): viewing_appeal_summary pre-anchor for identity accuracy
- `r3-no-thematic-obs` (H4): thematic observations ablation
- `r3-expanded-examples` (H2): 8-12 prompt examples per section
- `r3-no-signal-sources` (H3): remove prescriptive signal source routing

## Evaluate watch_context generations against user_prompt-grounded retrieval quality (current 50-movie set)
Files: movie_ingestion/metadata_generation/evaluation_data/watch_context_*_evaluation.json
Why: Fulfill the watch_context evaluation pass for the current result files using the repo rubric, while strictly judging from the evidence actually present in each file's `user_prompt` and from downstream retrieval usefulness.
Approach: Read `.claude/commands/evaluate-metadata-results.md`, `ingestion_data/watch_context_eval_guide.md`, the watch_context schema embedding contract, the watch_context prompt, the bucket file, and all current watch_context result JSONs. Wrote 50 `_evaluation.json` files covering 440 candidate evaluations with rubric scores for groundedness, retrieval_alignment, section_discipline, coverage, and holistic quality. The scoring lens stayed anchored to embedded terms only and treated unsupported scenario/motivation invention as a downgrade even when the system prompt might encourage broader generation.
Design context: Matches the command workflow and the watch_context rubric's downstream-usage framing. This pass explicitly prioritizes the user's instruction to grade only against `user_prompt` evidence and retrieval value, not against system-prompt compliance.
Testing notes: No tests run per repo instructions. Spot-checked generated evaluation artifacts including `watch_context_644239_evaluation.json` (Bigfoot identity-misread failure case) and `watch_context_10802_evaluation.json` (Showgirls camp/identity success case) after writing the full set.
