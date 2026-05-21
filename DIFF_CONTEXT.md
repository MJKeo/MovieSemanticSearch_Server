# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Step 1: dropped structured decomposition, freeform prompt + thinking budget 1024
Files: schemas/step_1.py, search_v2/step_1.py, search_v2/full_pipeline_orchestrator.py, run_search.py, docs/modules/search_v2.md

Why: prior spins under the rigid hard_commitments / soft_interpretations / open_dimensions decomposition kept producing single-token tweaks on the original query (e.g. "action movies" → "animated action movies"). Preserving every hard commitment forced every spin to share the dominant constraint, so all three branches collapsed onto largely the same result set and wasted two slots of compute. User flagged this as a fundamental misunderstanding of what spins should do.

Approach: stripped `Step1Response` down to just `spins: List[Spin]` and `Spin` down to `query` + `ui_label`. Removed `hard_commitments`, `soft_interpretations`, `open_dimensions`, `original_query_label`, `branching_opportunity`, and `distinctness`. Rewrote the system prompt as freeform principles (vague queries → commit to a specific interpretation; specific queries → branch outward by dropping or generalizing the dominant anchor; the two spins explore different directions from each other) with the "dark Marvel → mature superhero" worked example. Switched the Gemini config from `thinking_level="minimal"` to `thinking_budget=1024` so the model has room to reason holistically about what to keep vs drop. Field descriptions reframed around the result-set-difference goal rather than slot-filling.

Design context: per personal_preferences.md "Principle-based prompt instructions, not reactive failure lists" and "Examples in LLM field descriptions become templates the model imitates" — the prompt body keeps a single worked example, and field descriptions stay principle-level. Per "Lock LLM params into module constants once a stage is finalized" the new thinking_budget value lives as a module constant; callers cannot override.

Downstream: full_pipeline_orchestrator's original-branch label now hard-codes "Original Query" (was sourced from `step1.original_query_label`). run_search.py's step-1 CLI prints `ui_label` instead of `branching_opportunity`.

Testing notes: user will run tests themselves. Worth eyeballing: (1) spin distinctness on queries with one dominant anchor (Marvel, Disney, specific actor) — verify the model drops the anchor when it should; (2) vague-query handling ("I want to feel something") — verify both spins commit to distinct readings rather than paraphrasing; (3) latency change from thinking_budget=1024 vs prior `minimal`.

Review fixes (post-write): (1) trimmed schemas/step_1.py field descriptions to one definitional sentence each so the system prompt is the sole source of how-to-think guidance (per the "pick one per schema" convention) — saves per-call tokens and removes the maintenance trap of two surfaces saying the same thing; (2) removed `Spin` / `Step1Response` class docstrings (Pydantic propagates them into the JSON schema description sent on every API call — convention forbids); developer documentation now lives in `#` comment blocks above each class; (3) swapped the system-prompt worked example from "Dark Marvel movies" to "Cozy Studio Ghibli films" — Marvel appears as actual test queries in search_v2/test_queries.md (lines 229, 676), Ghibli only as analyst commentary, so the example no longer brushes against eval-overlap territory.

Follow-up tweak: added an `exploration: str` field to `Step1Response` ahead of `spins` — a freeform brainstorm of alternative search directions worth considering for the query — and flipped `thinking_budget` from 1024 to 0. The model now surfaces its reasoning visibly in `exploration` rather than internally; spending tokens on hidden thinking on top of a visible scratchpad is redundant. Field order respects the cognitive-scaffolding convention (concrete/exploratory before committed). Model stays at `gemini-3-flash-preview` (user's chosen value). System-prompt OUTPUT block updated to introduce exploration first; module-level comments in both schema and executor rewritten so they no longer describe Step 1's reasoning as hidden-thinking-budget-only. Added a smoke-test runner at `search_v2/run_step_1.py` (default query "dark gritty marvel movies" to exercise the dominant-anchor path).

Iteration round (multi-pass, user-directed): tested the spin generator against ~12 fixed queries between each prompt change, with the user analyzing failure patterns and prescribing the next reframe. End state after the session:

(1) Output-length constraint on exploration. User noted first runs produced multi-sentence exploration paragraphs that ballooned latency. Added "2-3 compact, telegraphic sentences" to both the schema description and the OUTPUT block of the system prompt (this is a format spec at point of generation, not how-to-think duplication — the lean-fields convention's carve-out).

(2) Three-step reasoning pattern in HOW_TO_THINK. Replaced the loose "read the query / consider directions" framing with a numbered three-step scaffold the model executes inside the exploration field: Step 1 read the user (interpret intent from query cues), Step 2 find adjacent territory (vague→commit to specific reading; specific→build from scratch as semantic neighbor), Step 3 pressure-test redundancy (compare candidate result sets against original and each other). Each step explicitly informs the next.

(3) Removed all worked examples from the system prompt. Ghibli, "feel something," and Marvel illustrative passages all stripped. Per user direction: no few-shot examples in the prompt at all; generalized guidance only. The Avoid list was reframed from vocabulary callouts ("iconic / highly-rated / prestigious") to category definitions ("words that ask the search for prestige, popularity, or judgment rather than describe what kind of movie the user wants").

(4) Spins → must trace back to exploration candidates, not be generated independently. Added "refine, do not copy" framing — the spin's content must derive from an angle surfaced in exploration; the second spin must explicitly compare itself against the first and pick a different candidate if the result sets would overlap.

(5) Naming rule for entities. Movie/show titles are never allowed inside a spin query. Brand, studio, director, and actor names are allowed only when (a) introducing a NEW such entity the original didn't mention AND (b) the spin's whole shape pivots on it — never as enumerated examples ("from X, Y, or Z"). Entities named in the original should be dropped from the spin (retaining them just preserves the same retrieval anchor).

(6) Specific-query reframe (final correction from the user). Earlier iterations framed specific-query spin construction as "transform the original by dropping some traits" or "drop majority of anchors" — including a trait-counting gut check. User corrected the framing entirely: spins for specific queries are built FROM SCRATCH using the viewer's underlying taste as the starting point, NOT by transforming the original's text. The spin's relationship to the original is semantic neighborhood (same viewer, same general territory of films), not textual descent (same anchors with edits). Removed the trait-counting paragraph; reframed both the SPECIFIC-handling section and the lead Avoid item ("constructed as variations of the original (paraphrases, narrower slices, 'the original but [adjective]')") around construction pattern, not trait preservation. Vague-query handling unchanged: still framed as refinement (adding specificity via interesting traits to make implicit readings concrete).

(7) `ui_label` length spec softened. Dropped "2-5 word Title Case" hard count from the schema description (which had pushed the model into mangled compressions like "Nostalgic European Rom-Set Rom-Coms"). Replaced with "Short Title Case label for the browsing UI. Pithy enough to read at a glance but not so compressed it loses meaning."

End-of-session test results: exploration text now consistently opens with a viewer-intent read instead of "Direction 1: X. Direction 2: Y." preview. Single-anchor specific queries (Tom Hanks, Scorsese, Pixar) drop the named anchor and build genuine sibling searches; the brand/studio permission is used correctly to name NEW pivots (DreamWorks/Sony for the Pixar spin) rather than retaining the original's anchor. Sibling distinctness between the two spins improved across most queries. Vague-query handling stayed strong throughout.

Two ceilings persisted at session end (user accepted): (a) multi-anchor specific queries where the combination is the recognizable concept — "sci-fi adventures with mentorship arc," "John Wick but woman," "90s romantic comedies set in NYC" — still produce spins that anchor-lock on the combination because the model reads the combination as the viewer's taste itself rather than as the surface query, and lifting to a trans-genre underlying experience didn't happen spontaneously. (b) Evaluative language ("heartfelt," "stunning," "magical," "classics") still leaks into queries and labels despite the principle-based avoid item. Both deferred to docs/TODO.md.

Follow-up: swapped Step 1 model from `gemini-3-flash-preview` / `thinking_config={"thinking_budget": 0}` to `gemini-3.5-flash` / `thinking_config={"thinking_level": "minimal"}`. A/B run across 6 representative queries (vague mood / single-anchor specific / multi-anchor composite / comparison-shaped / evaluative modifier / topical) showed: ~6% lower average latency (2.11s vs 2.24s) and substantially tighter variance (max 2.42s vs 4.00s — no p99 outlier on the gemini-3.5-flash side). Quality roughly comparable on most queries, with the new model handling multi-anchor anchor-lock slightly better ("90s NYC romcom" sheds both anchors instead of preserving them) and giving a stronger lane split on topical queries; old model edged out the new on evaluative-language hygiene for the "stunning visual movies" probe. Aligns Step 1 with Step 3's model + thinking framing for the search front-end. Module comment in `search_v2/step_1.py` updated to describe the new config; no schema or prompt changes.

## Step 3: `loose5` shipped — min_length=5 + filler-pruning framing + gemini-3.5-flash with minimal thinking
Files: schemas/step_3.py, search_v2/step_3.py

### Intent
Ship the `loose5` variant from the `category_candidates` experiment.
Combines three changes to Step 3 that the experiment showed work
better together than any of them individually:
1. Schema floor of `min_length=5` on `Dimension.category_candidates`
   (forces broad exploration of routing options).
2. Prompt + schema framing that explicitly tells the model the
   floor will produce filler and the routing-exploration step is
   responsible for ruthlessly pruning it.
3. Model swap to `gemini-3.5-flash` with `thinking_config={"thinking_level": "low"}`
   (minimal thinking enabled — distinct from `thinking_budget=0`
   which disables thinking entirely).

### Key Decisions
- **Why ship all three together, not the schema floor alone.**
  The prior `min5` experiment (schema floor only) showed real
  regressions: the model treated `min_length` as both floor AND
  ceiling, and committed weak adjacents as hedges. The added
  "expect filler / be ruthless" framing in routing_exploration is
  what unblocks the model to drop the weak candidates instead of
  committing them. See `search_v2/category_candidates_experiment/ANALYSIS_HANDOFF.md`
  for full per-variant comparison.
- **Why gemini-3.5-flash + thinking_level="low".** Faster than
  `gemini-3-flash-preview + thinking_budget=0` for this workload
  (loose5 wall-clock 194.8s vs min5's 247.8s at the same min_length=5).
  Minimal thinking gives the model some reasoning headroom without
  the cost of higher thinking levels.
- **Schema and prompt changes are coordinated.** The schema
  descriptions on `category_candidates`, `routing_exploration`, and
  `category_calls` all reinforce the same framing the prompt sections
  `_PER_DIMENSION_CANDIDATES` and `_ROUTING_EXPLORATION` carry.
  Schema = micro-prompts; both layers had to move together.
- **n_dims dropped ~22% (358 → 272) in the experiment.** Unknown
  whether this is the new model or the framing change. Worth
  watching downstream — fewer dimensions = fewer routing chances.
  Did not block ship per user direction.

### Planning Context
Experiment ran 4 variants × 25 queries × 3 runs against a fixed
Step 2 output (see `search_v2/category_candidates_experiment/`).
Per-variant deltas vs `base`:
- min3 (schema floor only): some wins (Studio/brand on Ghibli,
  Story archetype on war "epics") but real regression on MMA
  (lost Element/motif presence).
- min5 (deeper schema floor): same shape, regression persists.
- loose5 (schema + framing + new model): best within-variant 3-run
  stability (75% vs 63-65%), MMA regression fixed, granularity
  gate applied more consistently, output tokens 15% lower than min5
  despite same floor, wall-clock comparable to min3.

User confirmed acceptance of the trade-offs the analysis flagged
(Studio Ghibli's Studio/brand dropped per granularity gate; "war"
trait collapsed to Genre only; "dark" trait dropped Story
archetype). All judged as net-positive or acceptable.

### Testing Notes
- The experiment harness (`run_step_3_batch.py`) is what was used
  to validate. 75 result files per variant exist in
  `search_v2/category_candidates_experiment/results/`.
- No production tests run (per test-boundaries rule). The loose5
  behavior is verified by 75 fresh step-3 runs across the 25-query
  suite with zero per-trait LLM failures.
- Watch downstream: the n_dims drop and per-trait commit shrinkage
  could affect Stage 4 trait_score composition. If a query that
  worked under base now scores zero, that's the place to look.
- 25 experiment queries from `search_improvement_planning/rescore_overhal_queries.md`
  must continue to be kept out of any prompt/schema example set.

## Steps 0/1/2: swap to gemini-3.5-flash with `thinking_level="minimal"`
Files: search_v2/step_0.py, search_v2/step_1.py, search_v2/step_2.py

### Intent
Bring the front-end query-understanding stages onto the same model
family as the recently-shipped Step 3, but at the cheaper
`thinking_level="minimal"` setting (one level below `"low"`).
Verified valid via API probe — `minimal` is accepted by
`gemini-3.5-flash` and is strictly lower than `"low"`.

### Key Decisions
- **Step 3 stays on `"low"`, not `"minimal"`.** Step 3 is the
  routing-decision stage with the most reasoning load (per-trait
  category routing, granularity gate, combine-mode selection); the
  user explicitly asked for the others to be "minimal thinking
  (NOT low thinking)" so the lighter front-end stages get the
  cheaper config while routing keeps the headroom.
- **`implicit_expectations.py` left untouched.** The user gated the
  change with "ONLY if [implicit priors is] not already using a
  gemini model" — it is already on `gemini-3-flash-preview`, so
  per the condition we leave it alone.
- **Temperatures unchanged** (step_0: 0.1, step_1: 0.35, step_2:
  0.35). Only the model + thinking_config changed.
- **Module-doc comments updated** in all three files to reflect
  the new model + thinking level.

### Testing Notes
- Smoke-tested all three stages end-to-end against
  `"heartwarming holiday films"`: step_0 ran in <1s, step_1 in
  2.52s, step_2 in 5.04s with 2 traits returned. Zero failures.
- No suite-level regression test for steps 0/1/2 — the
  category_candidates_experiment harness exercises step_2 + step_3
  but not 0 or 1. Watch for behavior drift in production.

## New `CategoryCombineType.CONSENSUS` + SENSITIVE_CONTENT switched to it
Files: schemas/enums.py, schemas/trait_category.py, search_v2/stage_4_execution.py

Why: Under `ALTERNATIVES` (max), one endpoint scoring high could
over-promote SENSITIVE_CONTENT on its own — e.g. SEMANTIC matching
"gory" descriptive plot prose while KW disagrees and META is silent.
The user wants consensus credit: movies that match across the
committed endpoints should rank above movies that spike on a single
endpoint.

Approach: Added a new `CONSENSUS` combine type that takes the
geometric mean over committed-call scores with `_CONSENSUS_FOLD_FLOOR
= 0.1`, mirroring the existing FACETS across-category fold but at
the within-category level. SENSITIVE_CONTENT switched from
ALTERNATIVES → CONSENSUS. Walk-then-commit handler gating means the
fold only sees committed endpoints, so single-endpoint commits (e.g.
"famous for gratuitous gore" → SEMANTIC-only) degenerate to
passthrough; multi-endpoint commits get pulled toward agreement.

Design context: Rationale tradeoff documented in
search_improvement_planning/rescore_overhaul.md (ALTERNATIVES — max);
this change explicitly diverges for SENSITIVE_CONTENT to combat
single-endpoint over-promotion. Same EPS as FACETS for shared
calibration. Score impact is intentionally aggressive — a 3-spec
[0.95, 0.05, 0.05] commit drops from 0.95 (max) to ~0.21
(consensus). Reversible by flipping CategoryCombineType back to
ALTERNATIVES on the SENSITIVE_CONTENT enum row.

Testing notes:
- `unit_tests/test_schema_factories.py` exercises SENSITIVE_CONTENT
  routing but does not assert combine_type — should still pass.
- No existing combine_calls(CONSENSUS) test; eyeball validation via
  the next end-to-end run on queries like "no gore", "not too
  bloody", "famous for over-the-top gore".
- TARGET_AUDIENCE still on ALTERNATIVES (also bucket 8); revisit
  after a CONSENSUS query window if the same pattern shows up there.

## concept_tags: feed 4 baseline-derived inputs to the classifier
Files: movie_ingestion/metadata_generation/inputs.py, movie_ingestion/metadata_generation/generators/concept_tags.py, movie_ingestion/metadata_generation/prompts/concept_tags.py, movie_ingestion/metadata_generation/batch_generation/generator_registry.py, run_concept_tags_generation.py, concept_tags_table.md

Why: the post-baseline definition table identified four already-generated upstream fields that target specific failure clusters from the 23-movie eval — ending_aftertaste (5 ending failures), NT character_arcs.terms (4 ANTI_HERO false positives via "redemption arc" disqualifier), craft_observations (Kill Bill NONLINEAR_TIMELINE miss + craft-tag nuance), parental_guide_items (ANIMAL_DEATH severity + KIDNAPPING corroboration). All four were produced upstream but not routed into the concept_tags prompt.

Approach:
- `inputs.py` — added `load_viewer_experience_output()` mirroring the NT loader (normalize_legacy_metadata_payload route, returns None on failure). Updated `extract_narrative_technique_terms` to include `character_arcs` (now 7 sections, not 6) since film-language arc labels were the only NT signal currently dropped that disambiguates ANTI_HERO. Wave1Outputs already exposed `craft_observations`; added a docstring note about why concept_tags consumes it.
- `generators/concept_tags.py` — added two new formatters (`_format_ending_aftertaste`, `_format_parental_guide_items`) and threaded two new optional kwargs (`craft_observations`, `ve_output`) through both `build_concept_tags_user_prompt` and `generate_concept_tags`. parental_guide_items capped at 20 items to bound prompt length; ending_aftertaste flattens terms+negations into a single line. New inputs are keyword-only with None defaults to keep existing callers compiling.
- `prompts/concept_tags.py` — added input-block descriptions for craft_observations, ending_aftertaste, parental_guide_items, and the expanded narrative_technique_terms (now 7 sections). Wove the new signals into the tag definitions where they apply: PLOT_TWIST / NONLINEAR_TIMELINE / UNRELIABLE_NARRATOR / BREAKING_FOURTH_WALL gained craft_observations references; OPEN_ENDING and CLIFFHANGER_ENDING gained ending_aftertaste references; ANTI_HERO got an explicit "redemption arc DISQUALIFIES" rule keyed off the NT character_arcs section; ANIMAL_DEATH and KIDNAPPING gained parental_guide_items references with severity-driven decision rules. The endings HOW-TO block was rewritten to make ending_aftertaste the primary step (step 1) with concrete term-to-tag mappings, and added explicit "trust ending_aftertaste over inference from final state of affairs" guidance to close the Inception/Conjuring/Get Out HAPPY misses.
- `run_concept_tags_generation.py` — added `viewer_experience` to the join SQL, parses both `emotional_observations` and `craft_observations` from ReceptionOutput in one pass, parses ViewerExperienceOutput, expanded the per-movie tuple to 7 fields, threaded both new inputs through `run_three_times` and the per-movie dispatcher.
- `batch_generation/generator_registry.py` — concept_tags adapters (`_concept_tags_prompt_builder` and `_concept_tags_live_generator`) now also load VE via the new loader and pass craft_observations from Wave1Outputs. Other generators untouched.
- `concept_tags_table.md` — moved the four new fields from "Missing data (already generated)" to "Evidence (how to use it)" for every row that cited them, retained the residual missing-data entries that remain unused, and rewrote the cross-cutting observations + "Already-generated but NOT currently fed" inventory to reflect what's now in vs. still held back. Held-back fields (elevator_pitch, identity_note, thematic_concepts, genres, raw featured_reviews) are documented with rationale.

Design context: the baseline-derived prioritization (Tier 1 = compound leverage across 3+ failure clusters) was explicitly endorsed by the user. Per personal_preferences "principle-based prompt instructions" — the prompt body uses the new fields as primary signals where they win and as supporting signals where they corroborate, rather than as reactive failure-mode patches.

Testing notes:
- No code-correctness tests added (per rule). Validation is via re-running the 23-movie baseline eval (`python run_concept_tags_generation.py --test-name baseline_v2`) and diffing against `baseline/`.
- Expected closures (predict before re-running): 3 HAPPY misses → HAPPY, 2 BITTERSWEET over-corrections (Graduate, John Wick) → NO_CLEAR/HAPPY, Kill Bill NONLINEAR_TIMELINE present, 3 ANTI_HERO false positives (Phil, Frank, Cobb) gone — Taken's ANTI_HERO false positive depends on whether the NT char_arcs section actually labels him "rescue arc" or similar.
- Watch for regressions: prompt is now ~25% longer; cost/latency should bump proportionally. ENSEMBLE_CAST and FEMALE_LEAD were correct in baseline — should not drift now.

## concept_tags: collapse per-category enums into a single master ConceptTag with attribute-driven prompt assembly
Files: schemas/enums.py, movie_ingestion/metadata_generation/prompts/concept_tags_assembly.py (new), movie_ingestion/metadata_generation/prompts/concept_tags.py, run_concept_tags_generation.py

### Intent
Tag metadata previously lived in two places: numeric IDs + slug values + short `description` on seven per-category `str, Enum` classes, AND selection criteria + boundary cases + evidence rules as a 450-line hard-coded prose blob (`_TAG_DEFINITIONS`) inside the system prompt. The split made the taxonomy hard to evolve — every boundary tweak required editing two surfaces — and the boundary content was studded with movie-name examples (Mad Max, Catch Me If You Can, …) that overlap with the eval set. This refactor consolidates everything onto one `ConceptTag` master enum + a parallel `ConceptTagCategory` enum, both carrying their own metadata as attributes, with the prompt's tag-definitions section programmatically assembled from those attributes and example-driven bullets paraphrased into generalized principles.

### Key Decisions
- **Master enum carries rich attributes; per-category enums stay as dynamic subset classes.** ConceptTag uses the same `str, Enum` + custom `__new__` pattern already established in this file (`LineagePosition`, `SourceMaterialType`, `ReleaseFormat`) and adds `concept_tag_id`, `category`, `description`, `selection_criteria`, `boundary_cases`, `long_form_instructions` per member. The seven per-category enums (`NarrativeStructureTag` etc.) are dynamically built via a `_build_category_enum` helper that filters ConceptTag by category and creates a real `str, Enum` subclass via the functional `Enum(...)` API. Each derived member gets `concept_tag_id` and `description` attached post-construction so existing call sites (`apply_deterministic_fixups`, `all_concept_tag_ids`, `unified_classification`) keep working without modification. Pydantic v2 generates the same per-category JSON-schema enum constraint for structured-output enforcement (verified by inspecting `NarrativeStructureAssessment.model_json_schema()` — enum value list is byte-identical to pre-refactor).
- **ConceptTagCategory carries section-level metadata** (display_label, intro_text, cardinality, field_name, enum_class_name, section_instructions, cross_tag_note). User-confirmed design (over a sidecar dict). The ENDINGS HOW-TO 4-step block lives as `section_instructions`; cross-tag relationship notes (OPEN/CLIFFHANGER independence; structural-vs-emotional independence) live as `cross_tag_note`.
- **FEMALE_LEAD's 3-step reasoning** lives as `long_form_instructions` on that single tag — kept variable-length to avoid forcing a one-size-fits-all per-tag content schema.
- **Slug values and numeric IDs are byte-identical** to pre-refactor. Postgres rows referencing `movie_card.concept_tag_ids` keep working; tracker.db JSON deserializes through `ConceptTagsOutput.validate_and_fix()` unchanged. Round-tripped all 23 baseline JSON files as verification.
- **Movie-name examples paraphrased**, not removed wholesale. Boundary cases were rewritten as generalized concepts ("captives serving as the premise for a different central plot" instead of "Mad Max over-tag") to avoid polluting evaluation when the eval set names the same titles. Smoke-tested the assembled prompt for occurrences of any of the 23 baseline movie titles — no leaks.
- **Prompt assembly lives in a sibling module** (`concept_tags_assembly.py`) rather than inside `concept_tags.py` or `schemas/enums.py`. Keeps schema pure data, prompt-format concerns separate from content. The `concept_tags.py` module still owns the four hand-written framing blocks (_TASK, _EVIDENCE, _INPUTS, _OUTPUT) and re-exports `SYSTEM_PROMPT = build_system_prompt()` so external imports remain unchanged.
- **LIST_CATEGORIES in run_concept_tags_generation.py is now derived** from `ConceptTagCategory.cardinality == "multi"` rather than hand-coded. Verified the derived list equals the previous hand-coded list exactly.

### Planning Context
User-approved plan at /Users/michaelkeohane/.claude/plans/purrfect-fluttering-puddle.md. Pre-implementation Explore agent mapped consumers; verified search_v2/* and db/postgres.py work with raw int[] only and don't import the enum classes — so dynamic per-category enums don't ripple beyond the Pydantic-typed Assessment models. Pre-implementation smoke test confirmed post-construction attribute attachment on functional-Enum members works under Python 3.13 + Pydantic v2.

### Testing Notes
- All verification steps in the plan ran clean: 26 (slug, id) pairs unchanged; LIST_CATEGORIES derivation byte-identical; per-category enum JSON schemas byte-identical to pre-refactor; all 23 baseline result JSON files round-trip through `ConceptTagsOutput.validate_and_fix()` with TWIST_VILLAIN → PLOT_TWIST fixup still firing.
- Final SYSTEM_PROMPT is ~32,185 chars and contains all expected markers (TAG DEFINITIONS, all 7 section headers, FEMALE_LEAD 3-step block, ENDINGS HOW-TO, cross-tag notes for NARRATIVE STRUCTURE and ENDINGS, Check/NOT pattern per tag).
- No test files touched per the test-boundaries rule. Next end-to-end check is to re-run the 23-movie baseline (`python run_concept_tags_generation.py --test-name baseline_v3`) and diff against `concept_tags_results/baseline/` and `baseline_v2`. The refactor is content-neutral structurally; observable differences should be small and explainable by the paraphrased boundary cases.
- Risk to monitor: paraphrased boundaries are necessarily slightly different wording — the LLM may interpret subtly differently. Worth eyeballing any movie that flips a tag between baseline_v2 and baseline_v3 to make sure the new boundary phrasing didn't drift semantics.

## Tighten five regressed tag definitions identified by improved_context eval
Files: schemas/enums.py

### Intent
The improved_context eval (results in concept_tags_results/improved_context vs concept_tags_results/baseline) showed a mixed outcome: real wins on ANTI_HERO discipline and narrative-structure over-tagging, but a catastrophic ending-classification regression. 8 of 23 movies got pulled toward BITTERSWEET_ENDING when the table-expected tag was HAPPY or SAD (Frozen, Deadpool, Catch Me If You Can, Mad Max, Fight Club, Pulp Fiction, The Conjuring, Marley & Me). Other regressions: Pulp Fiction lost a legitimate ANTI_HERO (the redemption-arc disqualifier over-reached), The Mist gained an ANTI_HERO false positive (single moral choice under duress misread), Star Wars gained an ENSEMBLE_CAST false positive (hero-with-allies misread as ensemble), The Conjuring gained an OPEN_ENDING false positive (franchise-hook coda misread as open central question). This change rewrites the five tags that had at least one IMPR failure: BITTERSWEET_ENDING, HAPPY_ENDING, OPEN_ENDING, ANTI_HERO, ENSEMBLE_CAST — plus the ENDINGS HOW-TO section_instructions block.

### Key Decisions
- **Lead each definition with the discriminating question, not the surface description.** ENSEMBLE_CAST, BITTERSWEET_ENDING, and OPEN_ENDING all suffered from having their disqualifying tests buried in selection_criteria. The model evaluated the surface description first, satisfied it, and tagged without ever applying the test. Rewrote so the discriminating question appears in the description field itself — ensemble's removal test, bittersweet's "would HAPPY OR SAD be defensible?" test, open_ending's "did the film resolve the main conflict it posed?" test.
- **Name common false-positive patterns explicitly by archetype.** ENSEMBLE_CAST gained an explicit "HERO-WITH-ALLIES PATTERN" callout (single protagonist with mentor / love interest / sidekick / antagonist) as a named anti-pattern. OPEN_ENDING gained a "franchise-hook coda, post-credits tease, universe-continues epilogue" exclusion. BITTERSWEET gained explicit "genre films where protagonists pay a price and then win — that is the DEFAULT STRUCTURE of a happy ending, not bittersweet" boundary. ANTI_HERO gained a "single act of extreme moral consequence under impossible duress" exclusion.
- **Distribute frequency claims to push correct base rates.** Both HAPPY_ENDING and the ENDINGS HOW-TO block now explicitly state HAPPY is the empirically dominant ending tag and BITTERSWEET is uncommon. Added a "BASE RATES" preamble to the HOW-TO listing the four tags' relative frequency. Added explicit "DEFAULT RULES" — when ambiguous between HAPPY and BITTERSWEET, pick HAPPY; same for SAD vs BITTERSWEET. The intent is to make BITTERSWEET feel like the high-bar exception it should be rather than the sophisticated middle ground the model was treating it as.
- **Fixed the HOW-TO term-to-tag mapping.** Previously `"achievement at a cost"` mapped to BITTERSWEET — but that phrase describes the structure of every satisfying happy ending in genre cinema. Moved it to the HAPPY_ENDING row alongside "earned" and "hard-won". BITTERSWEET now requires explicit audience-leaving-mixed language ("unresolvably mixed", "unable to celebrate fully", "a knot despite the win"). This single change is expected to recover most of the 8 ending regressions.
- **Added a closing-scene heuristic** to both HAPPY_ENDING and the HOW-TO. Naming the specific final-beat genres (triumphant kiss, family reunion hug, threat-defeated cheer, platform-raise / mountaintop moment) gives the model a concrete shape to match rather than abstract emotion-reasoning.
- **Scoped the ANTI_HERO redemption-arc disqualifier** to single-protagonist films. For ensembles with multiple decision-driving protagonists, evaluate each independently — if any one is consistently anti-heroic, the film qualifies. This addresses the Pulp Fiction false negative without losing the Catch Me If You Can / Groundhog Day / Inception wins. Also broadened the "fundamentally moral" boundary to make the principle explicit: moral mode is measured across the runtime, not at the most extreme moment.

### Planning Context
Analysis was a post-eval diagnostic exercise comparing concept_tags_results/baseline (the original) to concept_tags_results/improved_context (the previous prompt-update run). The user explicitly asked for definition-level changes only — no input-side changes, no eval reruns. The five tags addressed are exactly those with at least one IMPR-side regression; tags that only saw IMPR wins (PLOT_TWIST, NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR, KIDNAPPING, UNDERDOG) were left alone. No movie names introduced into definitions; verified via the same banned-substring smoke check used in the master-enum refactor.

### Testing Notes
- 26 tag IDs unchanged; ALL_CONCEPT_TAGS still 25; legacy result JSONs still round-trip through ConceptTagsOutput.validate_and_fix.
- Prompt length grew from ~32,185 chars to ~39,117 chars — about 22% larger. The five rewritten tags are intentionally more verbose since the prior versions failed by being too brief at the discriminating-question level. If the eval shows the model loses attention to other tags due to length, the fix is to compress non-regressed tags (FEEL_GOOD, TEARJERKER, TIME_LOOP, etc.) — not to re-shrink the five tags we just expanded.
- The most concrete predictions to check on the next baseline eval run:
  (1) BITTERSWEET_ENDING should drop from 11 movies to ~3-4 (only La La Land, Rocky, John Wick should remain BITTERSWEET; The Graduate and Kill Bill should land NO_CLEAR per the table; Frozen/Deadpool/Mad Max/Catch Me/Fight Club/Pulp Fiction/Conjuring/Marley & Me should leave BITTERSWEET).
  (2) Star Wars should drop ENSEMBLE_CAST.
  (3) The Conjuring should drop OPEN_ENDING.
  (4) Pulp Fiction should re-acquire ANTI_HERO.
  (5) The Mist should drop ANTI_HERO.
- If (1) doesn't largely materialize, the diagnosis was wrong about the BITTERSWEET cause and the failure is upstream in the ending_aftertaste generator rather than in the prompt's discrimination criteria.

## Per-call LLM timeout override; concept_tags eval uses 60s
Files: implementation/llms/generic_methods.py, movie_ingestion/metadata_generation/generators/concept_tags.py, run_concept_tags_generation.py
Why: the eval script was hitting frequent `asyncio.wait_for` timeouts. The router's `LLM_PER_ATTEMPT_TIMEOUT_SECONDS=25.0` was tuned for fast non-reasoning calls, but concept_tags runs gpt-5-mini at medium reasoning effort over a heavy multi-section prompt and routinely lands in the 25-40s range. The eval also fans out ~24 movies × 3 runs = ~72 concurrent calls, which inflates tail latency further. Both attempts share the same too-tight budget, so the built-in retry doesn't rescue these.
Approach: added an optional `timeout: float | None = None` kwarg to `generate_llm_response_async` that overrides the module default when provided; forwarded the same kwarg through `generate_concept_tags`; set `LLM_TIMEOUT_SECONDS = 60.0` in the run script and pass it on each `generate_concept_tags` call. Kept it as a per-call override rather than mutating the router constant so non-reasoning callers retain the tighter SLA and a single eval can't quietly relax the global default.
Testing notes: rerun `python run_concept_tags_generation.py --test-name <name>` and confirm the timeout warnings disappear (or at least drop sharply). Other generators are unaffected since they don't pass `timeout`.

## Add evidence-first `reasoning` field to every concept_tags Assessment
Files: schemas/metadata.py, run_concept_tags_generation.py, movie_ingestion/metadata_generation/prompts/concept_tags.py

### Intent
Force the concept_tags model to externalize its evidence walk-through BEFORE committing to tags, rather than emitting tags first and reasoning post-hoc. Each per-category Assessment (`NarrativeStructureAssessment`, `PlotArchetypeAssessment`, `SettingAssessment`, `CharacterAssessment`, `EndingAssessment`, `ExperientialAssessment`, `ContentFlagAssessment`) now declares a required `reasoning: str` field positioned BEFORE its `tags`/`tag` field. Since OpenAI structured outputs emit JSON properties in schema declaration order, the model must produce the reasoning string first. Main system prompt was deliberately left untouched per user direction — this is purely a schema-level intervention.

### Key Decisions
- **Field placement BEFORE the tag list.** Pydantic v2 preserves class-declaration order in `model_json_schema()`'s `properties`, and OpenAI structured outputs respect that order. Putting `reasoning` first is the lever that forces evidence-before-conclusion at the token-generation level. Verified via smoke test that schema property order is `[reasoning, tags]` for all 6 multi-label categories and `[reasoning, tag]` for endings.
- **Required, not optional.** The reasoning field is declared with `Field(...)` (no default), making it required. An optional field with a default lets the model skip it and defeats the entire purpose. The cost is that legacy ConceptTagsOutput JSON without a `reasoning` field will fail `model_validate`. Acceptable — the user is iterating and will regenerate. If legacy tracker.db rows turn out to need re-loading, the cheap fix is a `model_validator(mode='before')` that injects `reasoning=""` for missing keys.
- **Per-category description names the primary evidence sources for that category.** Each reasoning field's `description=` reiterates which inputs the model should examine for that section (e.g., NarrativeStructure → craft_observations + narrative_technique_terms sections; Endings → ending_aftertaste + emotional_observations + closing-scene plot beat; Experiential → emotional_observations as PRIMARY/AUTHORITATIVE). The Endings description is the longest because that category had the worst regression history — it inlines the celebration-vs-unresolved-language distinction, the closing-scene test, and the base-rate rule (HAPPY dominant, BITTERSWEET uncommon).
- **Closing every description with "EVIDENCE FIRST, conclusion second — never state the conclusion and then write the evidence that justifies it."** The negative-form constraint is intentional. The failure mode we're attacking is post-hoc justification, where the model decides the tag and then writes plausible-sounding evidence to support it. Naming the anti-pattern explicitly is more effective than only describing the positive instruction.
- **`majority_merge()` populates reasoning by concatenating per-run reasonings.** The merged Assessment is a programmatic synthesis (majority vote across 3 runs), so its reasoning is the joined `[Run 1] ... [Run 2] ... [Run 3] ...` audit trail of the underlying runs. This is more useful for debugging disagreements than a synthesized summary would be, and it preserves the model's actual evidence walk for every run in the saved JSON.

### Planning Context
This is the follow-up intervention after the three-way REV vs IMPR vs BASE comparison revealed that 7 of 8 BITTERSWEET regressions persisted in revised_definitions despite the targeted prompt rewrite of BITTERSWEET / HAPPY / OPEN / ANTI_HERO / ENSEMBLE definitions. The diagnostic conclusion was that the bottleneck is partly upstream in the `ending_aftertaste` generator AND partly in the consumer not being forced to reconcile contradictory evidence before deciding. Externalized reasoning attacks the second half: the model now has to write its evidence walk down before it can emit a tag, which makes "ending_aftertaste says X, but closing scene shows Y, base rate is Z" reconciliation visible and pressureable. The user explicitly said to leave the main prompt untouched and test the schema-level lever in isolation.

### Testing Notes
- User will rerun `python run_concept_tags_generation.py --test-name <name>`. The saved JSON in `concept_tags_results/<name>/<movie>.json` will now contain per-category `reasoning` strings under `merged.*.reasoning` and `individual_runs[].*.reasoning`. Existing baseline/improved_context/revised_definitions JSON files cannot be re-loaded as ConceptTagsOutput (missing required field) but they don't need to be — they're for visual comparison only.
- Watch for: (1) whether the explicit reasoning narrows the BITTERSWEET over-tag specifically (does the model now write "ending_aftertaste says 'sacrifice'… but closing scene is triumph… HAPPY" rather than just "ending_aftertaste says 'sacrifice' → BITTERSWEET"); (2) whether reasoning length blows up token budgets (gpt-5-mini at medium reasoning effort + ~39K-char prompt + now extra per-section reasoning could push total tokens significantly); (3) whether the new PLOT_TWIST false positives in REV (12 Angry Men, Erin Brockovich) disappear once the model has to write the evidence for the twist before claiming it.
- Smoke-tested: schema field order verified `reasoning` first in all 7 Assessments; reasoning marked required in JSON schema; full ConceptTagsOutput construct + JSON round-trip clean; `majority_merge()` correctly joins per-run reasonings into `[Run 1] ... [Run 2] ...` format.

## Remove contaminated upstream inputs from concept_tags consumer (ending_aftertaste + 2 NT subsections)
Files: movie_ingestion/metadata_generation/generators/concept_tags.py, movie_ingestion/metadata_generation/inputs.py, movie_ingestion/metadata_generation/prompts/concept_tags.py, run_concept_tags_generation.py, schemas/enums.py, schemas/metadata.py, concept_tags_table.md

### Intent
The reasoning-field eval (reasoning_fields test) made the contamination chain unambiguously visible: every BITTERSWEET over-tag (7+ movies) was driven by the upstream `ending_aftertaste` literally containing the word "bittersweet", and every Catch Me / Deadpool / 12 Angry Men ANTI_HERO + NONLINEAR + PLOT_TWIST false positive was driven by upstream `narrative_technique_terms` subsections that emit tag-shaped labels ("antihero maturation arc", "sympathetic antihero", "intercut flashback structure", "evidence-driven reversal"). The consumer's reasoning text showed it was operating as a faithful label-transcriber — the schema description's anti-pattern guidance couldn't override an upstream signal that literally named the tag. Per user direction, remove the three contaminated inputs from the consumer (rather than regenerate the upstream ViewerExperience / NarrativeTechniques generators) and re-derive their evidence from rawer fields already in the input set.

### Key Decisions
- **Removed `ending_aftertaste` entirely** (Tier 1). Replaced as the PRIMARY ending signal by a two-stage process inlined into the EndingAssessment.reasoning description and the ENDINGS section_instructions HOW-TO: (1) the literal closing scene from `plot_summary` is the new PRIMARY (celebration beat → HAPPY; loss beat with no upswing → SAD; quiet contemplative beat → BITTERSWEET candidate; existential beat → NO_CLEAR_CHOICE candidate); (2) `emotional_observations` filtered for end-state language is the supporting signal. The base-rate rule and HAPPY/BITTERSWEET default rule are preserved.
- **Removed NT `character_arcs.terms` and `audience_character_perception.terms`** (Tier 2). These were emitting "antihero maturation arc" / "sympathetic antihero" literally and driving Catch Me's ANTI_HERO regression. The CharacterAssessment.reasoning description now points the model at `plot_summary` (raw protagonist behavior) + `character_arc_labels` from PlotAnalysis (thematic arc transformations like "impostor to reconciled contributor", which still encode the redemption-arc disqualifier without using the literal word "antihero") + `conflict_type`. The ANTI_HERO selection_criteria in the enum was rewritten to explicitly forbid deriving from upstream-labeled terms.
- **Kept `narrative_technique_terms` (now 5 sections)**: narrative_archetype, narrative_delivery, pov_perspective, information_control, additional_narrative_devices. These remain tag-adjacent but the eval showed they do real positive work (Erin Brockovich plot_twist correctly dropped because the model contrasted "information_control: slow-burn reveal" with "recontextualizing twist"). Added a one-line warning in the prompt _INPUTS block: "these labels may use vocabulary that overlaps with concept-tag names ... Treat them as descriptive shorthand to investigate, not as direct classifications."
- **Schema-level CharacterAssessment.reasoning explicitly instructs "DO NOT derive ANTI_HERO from upstream-labeled terms — derive it from what the protagonist actually does in plot_summary."** Mirrored in the enum's ANTI_HERO selection_criteria. Belt-and-suspenders: the input is removed AND the schema description forbids the failure mode, so even if a future upstream change re-introduces tag-shaped labels somewhere, the consumer prompt is pointed away from them.
- **The eval runner (run_concept_tags_generation.py)** drops the `ve_output` join from its SQLite query, the `ViewerExperienceOutput` import, the `ve` tuple element, the `run_three_times(ve)` parameter, and the `generate_concept_tags(ve_output=)` kwarg. End-to-end signature cleanup so the new run won't carry stale wiring.
- **`load_viewer_experience_output`** in inputs.py is kept (its docstring updated) — other future consumers may want to read ViewerExperience output even though concept_tags no longer does.

### Planning Context
This is the upstream-label-contamination fix that the reasoning-field eval was designed to enable diagnosis of. The three-way comparison (BASE vs revised_definitions vs reasoning_fields) showed that the BITTERSWEET regression was unmovable via prompt-side tightening alone (the prompt explicitly told the model "earned victory is HAPPY, not bittersweet" and the model still picked bittersweet because upstream literally said "bittersweet"). The user explicitly chose removal-and-rederive over upstream-regeneration ("we won't regenerate viewer experience"). Two NT subsections were removed (Tier 2) because they had the same contamination pattern as ending_aftertaste — literal tag-name words in upstream output. The two-NT-subsection removal is moderately conservative: narrative_delivery and information_control were also tag-adjacent but had genuine positive use cases in the eval, so they're kept with a vocabulary-overlap warning in the prompt.

### Testing Notes
- User will rerun `python run_concept_tags_generation.py --test-name <name>`. Predictions to watch:
  1. **BITTERSWEET over-tag should collapse**: Catch Me, Deadpool, Mad Max, Fight Club, Pulp Fiction, The Conjuring, John Wick, Marley & Me should now lean HAPPY or SAD per their closing scene + emotional_observations. The bittersweet bar is now "audience explicitly says they cannot decide between happy and sad", which most of these films don't meet.
  2. **Catch Me ANTI_HERO should drop**: the upstream "antihero maturation arc" label is no longer in input. character_arc_labels "impostor to reconciled contributor" is now the primary arc signal — and it lands on a moral end-state, which is a hard disqualifier.
  3. **Deadpool NONLINEAR_TIMELINE should drop**: same input file as before BUT `audience_character_perception` was the contaminant, not the source of nonlinear. Actually `narrative_delivery: 'intercut flashback structure'` is still in input, so this might NOT improve — watch closely. If it persists, narrative_delivery may need a follow-up audit.
  4. **12 Angry Men PLOT_TWIST should drop**: same story — `information_control: 'evidence-driven reversal'` is still in input. The Tier-3 holdout. Watch for whether the prompt's new "vocabulary overlap" warning is enough.
  5. **Kill Bill CLIFFHANGER**: should be re-acquired now that the schema description for narrative_structure no longer assumes `ending_aftertaste` exists. NarrativeStructureAssessment.reasoning now explicitly points cliffhanger detection at plot_summary's closing scene + emotional_observations.
- Risk to watch: removing the primary ending signal could push more endings to NO_CLEAR_CHOICE if the model can't confidently call a closing scene from plot_summary alone. The schema description preserves the HAPPY default rule to mitigate.
- Failure-mode fallback: if (1) doesn't materialize and bittersweet over-tag persists, the contamination is in `emotional_observations` too (less likely — that field describes whole-movie audience reaction, not the ending tag specifically). If (3) and (4) don't improve, Tier-3 NT subsections (information_control, narrative_delivery) need a follow-up removal pass.
- Smoke-tested: tag IDs unchanged (PLOT_TWIST=1, ANTI_HERO=33, BITTERSWEET=43, NO_CLEAR=-1); ALL_CONCEPT_TAGS=25; NT extraction yields 5 sections; system prompt no longer contains "ending_aftertaste" or "audience_character_perception" (length stayed at ~39.5K chars — roughly equal); generate_concept_tags signature no longer carries ve_output; EndingAssessment.reasoning leads with closing-scene PRIMARY, no ending_aftertaste references; full eval-runner import chain clean.

## Bind tags-field outputs to reasoning + rewrite 8 tag definitions
Files: schemas/metadata.py, schemas/enums.py

### Intent
Two interventions in one pass to address failure patterns from the gpt-5.4-mini @ reasoning=none eval (concept_tags_results/gpt_54):

1. **Schema-level: bind every Assessment's `tags`/`tag` field to its `reasoning` field as source of truth.** The prior tags-field descriptions said only "Supported tags. Empty list is correct when no tags apply." — they did not reference the reasoning field at all. The reasoning was being written diligently and then ignored at the tags step ("Pattern 6" in the eval analysis — e.g. 12 Angry Men whose reasoning argued AGAINST plot_twist but whose tags field included it). The reasoning field is now explicitly named as the SOURCE OF TRUTH; the tags must follow from the reasoning's conclusions.

2. **Content-level: rewrite eight tag definitions** per user-vetted feedback after reviewing the gpt_54 regression analysis. Each rewrite was hand-approved.

### Key Decisions

- **TWIST_VILLAIN — moral-category flip, not identity hidden.** User reframed: the surprise is that a character presented as good/ally/trusted IS the antagonist. Surprising-evil-motivation alone (a known villain whose plan or scope is more elaborate) does not qualify. Rewrote description + selection_criteria + boundary_cases around the category-flip principle.

- **UNDERDOG — drop the identity-vs-setting distinction; the central dramatic question is what matters.** User: Star Wars (small rebellion vs large empire) IS underdog (setting-level asymmetry where the protagonist's side is the weaker one AND the film's central question is whether they prevail); Get Out is NOT (asymmetry exists but the central question is escape/revelation, not improbable victory). New definition: the protagonist or the side they belong to is structurally weaker AND the central dramatic question the film is built around is whether they can prevail. Setting-level asymmetry qualifies when the protagonist is on the disadvantaged side and the film is ABOUT that improbable rise. Input signal language generalized (no specific verbatim phrases like "underdog rising" — describes the TYPES of words to look for).

- **FEMALE_LEAD — expanded to all-female lead groups.** User: Frozen is clearly female_lead (two female co-leads); Birds of Prey is female_lead (all-female ensemble). The disqualifier is the presence of ANY male character in a lead role — single male lead, mixed-gender duo/trio, or ensemble that includes male leads. Rewrote description, selection_criteria, boundary_cases, AND the 3-step long_form_instructions to identify "lead role(s)" (one or many) instead of "single core character", then check the gender of each.

- **ANTI_HERO — redemption arc no longer disqualifies.** User: "If they are presented as an anti-hero that finds their way they were still originally an anti-hero and that counts in my book." Removed the HARD DISQUALIFIER clause that excluded any arc landing on a moral end-state. New criterion: substantive anti-heroic operation must occupy a meaningful portion of the runtime — whether or not the arc ends in redemption. The character_arc_labels arc-from-compromised-to-moral now counts as POSITIVE evidence the character WAS an anti-hero. CharacterAssessment.reasoning field also updated to drop the "HARD ANTI_HERO disqualifier" wording (it directly contradicted the new tag definition).

- **CLIFFHANGER_ENDING — intentionality is the bar.** User: cliffhanger is about deliberate sequel setup that leaves the audience on the edge of their seat between films. Plot holes or thematic ambiguity (Inception) are NOT cliffhangers. Being one of a series alone does NOT qualify — the nature of the ending must show deliberate continuation. Added explicit reviewer-commentary signal (how reviewers DESCRIBED the ending) as a primary evidence channel, alongside plot_summary closing-beat detection.

- **FEEL_GOOD — holistic emotional read, no word lists.** User: "Rather than listing words that are good or bad let's explain how to analyze the inputs." Removed the explicit term-list ("uplifting / heartwarming / joyful / ..."). New selection_criteria asks the model to read the FULL emotional landscape described and apply a holistic test: is the dominant tone OVERWHELMINGLY warm/hopeful/lifting? A meaningful counterweight of heavy emotions (grief, dread, devastation, sustained dark tension) means MIXED experience, not feel_good.

- **TEARJERKER — strict literal test on direct crying language.** User: "Is there a direct statement of the audience crying in the emotional craft or wherever else we're reading from? If yes, include, if no, exclude. Do not derive from anything else." Removed the broader term list. New rule: tag only when the available text DIRECTLY, EXPLICITLY states audiences cried / sobbed / wept / shed tears. Synonyms for emotional impact (moving / touching / poignant / devastating / heartbreaking / emotionally wrecking) do NOT qualify. Strict literal test by design — false-positive rate over recall.

- **ANIMAL_DEATH — explicit two-tier fallback.** User: tiered fallback (parental_guide_items first, plot_summary/plot_keywords as fallback when parental_guide_items has no entry or only mild). Rewrote selection_criteria as TIER 1 (parental_guide_items moderate-or-severe → tag) and TIER 2 (when parental_guide_items has no entry at all or only mild → fall back to plot-level evidence for the final decision).

- **All 7 Assessment `tags` / `tag` field descriptions rewritten.** Replaced "Supported tags. Empty list is correct when no tags apply." with a description that names the reasoning field as SOURCE OF TRUTH and explicitly requires the tags to follow from the reasoning's conclusions. Derivation from reasoning is allowed when the reasoning's conclusions logically imply a tag (the reasoning need not literally name it), but the derivation must follow from the reasoning, not from re-examined raw evidence. EndingAssessment's `tag` field got the same treatment scoped to single-tag selection.

### Planning Context
Vetting checkpoint where the user reviewed nine proposed rule changes, accepted some (PLOT_TWIST OK, OPEN_ENDING OK), rejected/reframed others (FEMALE_LEAD needed full reframe, ANTI_HERO redemption-disqualifier reversed, FEEL_GOOD/TEARJERKER required removing word lists), and added a new schema-level intervention (#5 — bind tags to reasoning). The user explicitly asked for the schema change to land FIRST as the highest-leverage edit. No movie names introduced into any definition; generalized principles only — preserves disjoint sets between prompt examples and the 23-movie eval set.

### Testing Notes
- Smoke-tested: 25 tags, IDs preserved (PLOT_TWIST=1, NO_CLEAR_CHOICE=-1); all 7 Assessment schemas still emit `[reasoning, tags]` or `[reasoning, tag]` in that order.
- No code-correctness tests run (per rule).
- Next concrete validation: rerun `python run_concept_tags_generation.py --test-name <name>` on the 23-movie set and check whether:
  1. Pattern-6 reasoning/output desyncs disappear (12 Angry Men no longer tags PLOT_TWIST when reasoning argued against it).
  2. UNDERDOG correctly fires for Star Wars and does NOT fire for Get Out.
  3. FEMALE_LEAD correctly fires for Frozen (two female co-leads) and Birds of Prey (all-female ensemble) while remaining disqualified for any mixed-gender lead structure.
  4. ANTI_HERO no longer drops legitimate cases where the arc lands in redemption (Pulp Fiction-style ensemble anti-heroes recovered; Catch Me If You Can stays included if upstream evidence supports it).
  5. CLIFFHANGER_ENDING fires only when reviewers describe deliberate sequel setup; Inception-style ambiguity stays in OPEN_ENDING.
  6. FEEL_GOOD over-tagging on tonally mixed films (action-comedies with grim mid-acts) drops; warm, overwhelmingly hopeful films are unaffected.
  7. TEARJERKER drops to near-zero on films without explicit crying language; remains on films where reviewers report tears.
  8. ANIMAL_DEATH gains plot-only positives that parental_guide_items missed, without false positives from unrelated "animal" mentions.
- Risk to monitor: the reasoning-as-source-of-truth binding pushes the failure mode upstream into reasoning quality. If reasoning still over-commits, the next lever is the per-category reasoning field descriptions (currently still call out specific evidence sources per tag, which is fine).

## Flip ANIMAL_DEATH tiered evidence: plot first, parental_guide as fallback
Files: schemas/enums.py

Why: definition_revision's tiered fallback (parental_guide TIER 1, plot TIER 2) misfired on Marley & Me — all 3 runs cited the dog's euthanasia in plot_summary, then refused to tag because parental_guide has unrelated entries (Sex & Nudity, Violence & Gore, Profanity, Frightening) but no animal-specific advisory. The model was treating "parental_guide is present and silent on animals" as evidence-of-absence rather than as fall-through to plot.

Approach: reversed the tier order. TIER 1 is now plot_summary + plot_keywords — any evidence of an animal death qualifies (the death does not need to be violent, severe, central, or extensively depicted; on-screen, off-screen, euthanasia, killed-by-antagonist, killed-in-passing all count). TIER 2 (used only when TIER 1 has no plot evidence at all) consults parental_guide_items for an animal-specific category at any severity. The selection_criteria now explicitly states that unrelated parental_guide entries are not a signal either way — only an animal-specific advisory counts. Boundary cases tightened around "animal merely mentioned without dying" to keep the broader plot-first threshold from leaking false positives.

Testing notes: re-ran the 23-movie eval as `concept_tags_results/animal_death_improvement` and compared ANIMAL_DEATH outcomes against baseline / pollution_removed / gpt_54 / definition_revision:
- **Marley & Me recovered** (DR: [], new: [animal_death]) — all 3 runs correctly cite plot's "Marley's euthanasia" / "loss of pet" / "mourn the loss of the dog" as direct evidence.
- **John Wick preserved** (3/3 runs cite "the puppy is killed" from plot_summary).
- **The Conjuring preserved** (3/3 runs cite "their dog dies" from plot_summary).
- **Get Out remained untagged** (plot only says car "hits a deer", does not state the deer dies — defensible call; all 3 runs explicitly reasoned through the distinction).
- **Zero new false positives** across the other 19 movies — none had any animal-death plot evidence to trigger TIER 1.

Run-to-run variance on the rest: 12 unrelated tag cells flipped between definition_revision and animal_death_improvement (≈7% of 161 non-animal-death cells), in both directions. These are sampling noise from gpt-5.4-mini @ reasoning_effort=none, not directional effects of the prompt change (which touched only the ANIMAL_DEATH literal). Side-effect-positive flips observed: Star Wars ANTI_HERO false-positive disappeared; The Mist regained SINGLE_LOCATION. Side-effect-negative flips observed: Paddington 2 lost PLOT_TWIST + TWIST_VILLAIN; Inception/Taken lost ANTI_HERO. None of these correlate with the edit.
