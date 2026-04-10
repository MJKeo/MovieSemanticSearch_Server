# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Add concept_tags_run_2 column to generated_metadata
Files: movie_ingestion/tracker.py | Added migration `ALTER TABLE generated_metadata ADD COLUMN concept_tags_run_2 TEXT` to store results of a second concept tags generation run; unioning both runs improves recall. Not written to yet. Ran `init_db()` to apply.

## Remove reasoning fields from ConceptTagsOutput
Files: schemas/metadata.py, movie_ingestion/metadata_generation/prompts/concept_tags.py | Dropped the `reasoning` field from all 7 per-category assessment classes and reworded the system prompt (module docstring, endings "HOW TO THINK THROUGH" block, OUTPUT FORMAT) so it still instructs the model how to evaluate each tag internally without claiming there is a reasoning field to emit. Empirically produced better tag quality on gpt-5-mini in test_concept_tags.ipynb runs; plan is to generate twice and union the results per run to improve recall. Stale field descriptions on `ConceptTagsOutput.narrative_structure` updated to drop the "evaluate each tag" phrasing.

## Capture per-run token usage in concept tags test notebook
Files: movie_ingestion/metadata_generation/generators/test_concept_tags.ipynb | Generation loop now stores the `TokenUsage` returned by `generate_concept_tags` in a new `all_usage` dict, and the save-to-JSON cell persists it as a `"usage"` list aligned positionally with `"runs"` for cost analysis and candidate comparison.

## ConceptTagsOutput schema refinements
Files: schemas/metadata.py
Why: Endings field allowed multiple tags but the prompt and spec define it as zero-or-one. Field descriptions were stale (listing enum values redundantly).
Approach: Changed `endings` to `conlist(EndingEvidence, max_length=1)` enforcing the single-tag constraint at the schema level. Rewrote all 7 category descriptions to describe the category's purpose rather than listing enum members, since the enum already documents the options.

## Concept tag generation pipeline (new metadata type)

Files: schemas/enums.py, schemas/metadata.py, movie_ingestion/metadata_generation/inputs.py, movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py, movie_ingestion/metadata_generation/prompts/concept_tags.py (new), movie_ingestion/metadata_generation/generators/concept_tags.py (new), movie_ingestion/metadata_generation/batch_generation/generator_registry.py, movie_ingestion/metadata_generation/batch_generation/result_processor.py, movie_ingestion/tracker.py

### Intent
Adds CONCEPT_TAGS as a new Wave 2 metadata generation type. Classifies 23 binary concept tags across 7 categories (narrative structure, plot archetypes, settings, characters, endings, experiential, content flags) via LLM multi-label classification. Tags enable deterministic Phase 1 search retrieval via Postgres INT[] array containment queries.

### Key Decisions
- **7 per-category enums replace single ConceptTag enum** (`NarrativeStructureTag`, `PlotArchetypeTag`, `SettingTag`, `CharacterTag`, `EndingTag`, `ExperientialTag`, `ContentFlagTag`). JSON schema self-enforces category membership — eliminates runtime model_validator. `ALL_CONCEPT_TAGS` flat tuple for codebase consumers.
- **7 per-category evidence classes** (`NarrativeStructureEvidence`, etc.) each with typed tag field. Replaces single `TagEvidence` class.
- **`validate_and_fix(content)` classmethod on `EmbeddableOutput`** — single entry point for batch result processor to validate raw LLM JSON and apply deterministic fixups. Default: pure validation. `ConceptTagsOutput` (inherits `BaseModel`, not `EmbeddableOutput` — first non-embeddable output) overrides with its own `validate_and_fix` that calls `apply_deterministic_fixups()` after validation. Result processor calls `schema_class.validate_and_fix()` uniformly for all types.
- **Category-level arrays with evidence-before-tag ordering** chosen over boolean grid (false-negative bias) and flat array (category-skipping). Evidence field forces chain-of-thought before tag commitment.
- **Post-generation fixup**: TWIST_VILLAIN implies PLOT_TWIST — handled deterministically via `ConceptTagsOutput.apply_deterministic_fixups()` instance method, called by `validate_and_fix()` override and the live generator. Fixup logic lives on the schema class, not in generators or processors.
- **Prompt section ordering**: task → inputs → tag definitions → evidence discipline → output. Evidence discipline placed after tag definitions for recency advantage during generation.
- **Parametric knowledge reframed** as "high-confidence fallback" (95%+ confidence, culturally unambiguous) instead of "tiebreaker". Input evidence trusted over parametric knowledge on conflict.
- **No tag count anchoring** — removed "2-6 tags total" from prompt. Classification is purely evidence-based.
- **Single prompt path** — no separate keyword-only prompt. Investigated the 20.7K keyword-only-eligible movies and found 99.7% have emotional_observations, making a distinct prompt unnecessary.
- **Eligibility gate**: plot_summary exists OR best_plot_fallback >= 250 chars OR plot_keywords >= 3.
- **Six LLM inputs** (~310-1140 tokens): title_with_year, plot_keywords, plot_summary/plot_text (quality-tiered), emotional_observations, narrative_technique_terms (6 of 9 sections, terms only), plot_analysis fields (arc labels + conflict_type).

### Planning Context
See search_improvement_planning/concept_tags.md for full tag definitions and design rationale. See search_improvement_planning/concept_tags_notes.md for prompt evaluation session notes.

### Testing Notes
- Schema validation: verify tags in wrong categories are rejected by typed enum constraints (no model_validator needed)
- Fixup: verify TWIST_VILLAIN without PLOT_TWIST inserts PLOT_TWIST; no duplicate when both present; no-op when neither present
- Eligibility: test all 3 eligible paths + skip case
- Prompt builder: test with all-present, all-absent, and partial inputs
- End-to-end: run eligibility evaluation, then live generation on small batch to inspect output quality

## Documentation staleness audit fixes
Files: schemas/metadata.py, docs/modules/ingestion.md, docs/modules/schemas.md, docs/modules/llms.md, docs/conventions.md, docs/decisions/ADR-065-schema-docstrings-as-comments-not-python-docstrings.md, docs/decisions/ADR-058-vector-text-formatting-conventions.md, movie_ingestion/metadata_generation/batch_generation/pre_consolidation.py

### Intent
Full docs-auditor pass found 15 stale references — all traced to the concept_tags type being added without a doc sweep. Also fixed two pre-existing convention/ADR issues.

### Key Changes
- **schemas/metadata.py**: Converted `TagEvidence` and `ConceptTagsOutput` class docstrings to `#` comment blocks per ADR-065 (these were leaking into LLM JSON schema payload)
- **ingestion.md**: Updated all "8 generators"→10, "9 batch_id columns"→10, "9 JSON result columns"→10, Wave 2 eligibility count 6→7
- **schemas.md**: MetadataType count 9→10, added `ConceptTagsOutput`/`ConceptTag`/`TagEvidence`/`CONCEPT_TAG_CATEGORIES` to Key Types, updated boundary list with `extract_narrative_technique_terms` and `load_plot_analysis_output`
- **conventions.md**: Added missing `embedded` status to pipeline chain; updated `__str__()` normalization rule to reference `embedding_text()` per ADR-057
- **ADR-065**: Updated reference count from "8 `*Output` classes" to include 9 EmbeddableOutput subclasses + ConceptTagsOutput/TagEvidence
- **ADR-058**: Struck stale open consequence about unwired fallback (now integrated)
- **llms.md**: Documented `verbosity` kwarg; updated TokenUsage import count 8→10
- **pre_consolidation.py**: Fixed docstring count from "Eight" to "Nine" eligibility methods

## Concept tags deterministic test suite

Files: movie_ingestion/metadata_generation/generators/concept_tags.py (modified), movie_ingestion/metadata_generation/generators/test_concept_tags.ipynb (new)

### Intent
Built a notebook-based evaluation harness for concept tag generation with 38 test movies across 4 buckets (core coverage, dense/messy, sparse, targeted challenges). Tests multiple LLM candidates (OpenAI gpt-5-mini, Gemini flash-lite) with 3 runs each for consistency measurement.

### Key Changes
- **Generator signature extended**: `generate_concept_tags()` now accepts keyword-only `provider`, `model`, and `llm_kwargs` overrides. Defaults unchanged — production callers are unaffected.
- **Test notebook**: 38 movies with expected tag sets, 4 LLM candidates, sequential generation loop with error capture, JSON serialization of full ConceptTagsOutput per run.
- **Notebook import fix**: Kernel cwd is the notebook's directory (not project root), and a third-party `schemas` namespace package in site-packages shadows the local `schemas/` package. Fix: `sys.path.insert(0, project_root)` + `importlib.invalidate_caches()` + purge stale `sys.modules["schemas*"]` entries.
- **Gemini model version**: Changed from `gemini-2.5-flash-lite-preview-06-17` to `gemini-3.1-flash-lite-preview`.

## Concept tags expected_tags overhaul from evaluation analysis

Files: movie_ingestion/metadata_generation/generators/test_concept_tags.ipynb (cell 2)

### Intent
Comprehensive review of all 38 test movies against gpt-5-mini candidate outputs and user prompts (input evidence). Updated expected_tags to reflect what the model should actually produce given its inputs.

### Key Changes
- **2 removals**: The Skin I Live In and Light of My Life both lost `sad_ending` (protagonist escapes/input lacks ending evidence respectively)
- **51 additions across 27 movies**: Most common additions were `anti_hero` (11 movies), `happy_ending` (11), `plot_twist` (7), `feel_good` (6), `revenge` (5)
- **Oldboy**: did NOT add `kidnapping` despite 6/9 candidate consensus — user clarified imprisonment ≠ kidnapping
- **Machete**: did NOT add `plot_twist` despite 5/9 — user clarified early betrayal is plot setup, not a recontextualizing twist
- **Tags NOT added despite high consensus**: Alien `happy_ending` (survival ≠ happy), Ferris Bueller `anti_hero` (trickster ≠ morally ambiguous), Taken `revenge` (rescue mission not vengeance), Mad Max `anti_hero` (reluctant ≠ morally ambiguous), Little Miss Sunshine `female_protagonist` (contradicts ensemble_cast)
- Total expected tag instances: 85 → 138. All 23/23 tags still represented.

## New tags + definition refinements from eval analysis

Files: schemas/enums.py, schemas/metadata.py, movie_ingestion/metadata_generation/prompts/concept_tags.py, movie_ingestion/metadata_generation/generators/concept_tags.py, movie_ingestion/metadata_generation/generators/test_concept_tags.ipynb

### Intent
Added 2 new concept tags and refined 5 existing definitions based on systematic model misclassification patterns from the gpt-5-mini evaluation.

### New Tags (23 → 25)
- **`cliffhanger_ending`** (NarrativeStructureTag, ID 9): Story ends with central conflict clearly unresolved and setup for continuation. Distinct from `open_ending` (artistic ambiguity vs unfinished story).
- **`bittersweet_ending`** (EndingTag, ID 43): Resolution contains significant positive AND negative elements. Fills the gap between `happy_ending` and `sad_ending` that was causing forced misclassification.

### Definition Refinements
- **`anti_hero`**: Added "tension or discomfort" threshold — reluctant heroes, flawed-but-good characters, and fully-redeeming protagonists don't qualify. Addresses over-application to any protagonist who does illegal things.
- **`happy_ending`**: Added bittersweet boundary — survival alone isn't sufficient, success at significant cost is bittersweet. Addresses tagging horror survival movies as happy.
- **`sad_ending`**: Added bittersweet boundary — loss tempered by hope/meaning is bittersweet.
- **`plot_twist`**: Added "must change understanding of events already shown" — early betrayals/deceptions before audience has formed expectations are plot setups, not twists.
- **`feel_good`**: Added subject matter compatibility requirement — cathartic satisfaction from violent/dark material is not feel_good.
- **`open_ending`**: Narrowed to artistic ambiguity with complete narrative arc. Sequel cliffhangers directed to `cliffhanger_ending`.

### Test Set Changes
- Terrifier 3: `open_ending` → `cliffhanger_ending`
- Schindler's List: `sad_ending` → `bittersweet_ending`
- Kill Bill Vol. 1: added `cliffhanger_ending` (daughter alive reveal), kept `anti_hero`
- Added Star Wars (1977): `happy_ending`, `feel_good`, `underdog` — resolved central conflict, NOT cliffhanger
- Added Empire Strikes Back (1980): `plot_twist`, `cliffhanger_ending`, `sad_ending` — opposite emotional valence
- Total: 38 → 40 movies, 25/25 tags represented, 145 expected tag instances

## Franchise metadata generation — full planning doc
Files: search_improvement_planning/franchise_metadata_planning.md (new), search_improvement_planning/new_system_brainstorm.md, search_improvement_planning/v2_data_needs.md, search_improvement_planning/v2_data_architecture.md, search_improvement_planning/open_questions.md

### Intent
Created comprehensive planning document for franchise metadata generation and updated all related V2 planning docs with finalized design decisions.

### Key Decisions
- **Franchise = real-world IP, not just film series:** Any recognizable IP or brand from any medium (video games, toys, books, comics, TV, etc.). Franchise name is the IP name (e.g., "mario"), not the film series name.
- **Drop `franchise_name` column, keep only `franchise_name_normalized`:** No display-name column. Same for `culturally_recognized_group` — stored normalized.
- **Finalized LLM input set (7 fields):** title, release_year, overview (identification aid only), TMDB collection_name, production_companies, overall_keywords, characters. No generated-metadata dependencies.
- **`culturally_recognized_group` is globally scoped:** Any market, not just US. American-market term takes precedence in rare conflicts.
- **No eligibility gate — all movies eligible:** Franchise generation is identification/classification (not content analysis), so sparse input causes conservative nulls rather than hallucination. False negatives are worse than false positives.
- **Wave-independent:** No dependency on Wave 1 or Wave 2 outputs. Can run in parallel with or independently from existing generation waves.

### Planning Context
See search_improvement_planning/franchise_metadata_planning.md for full specification covering output fields, LLM inputs with justifications, eligibility rationale, prompt strategy, pipeline placement, storage schema, and downstream search usage.

## Concept tags: 4-tag prompt refinement from gpt-5-mini eval analysis
Files: movie_ingestion/metadata_generation/prompts/concept_tags.py, schemas/metadata.py

### Intent
Targeted fixes for 4 tags identified via 40-movie x 3-run evaluation (120 classifications, 35% exact-match rate). Addresses systematic false positives in underdog (53.6% precision), bittersweet_ending (14.3%), ensemble_cast (55.6%), and systematic false negatives in sad_ending when cliffhanger is present (Terrifier 3 0/3, Empire Strikes Back 1/3).

### Key Changes
- **underdog**: Replaced definition with official ("expected to lose or fail due to lack of resources, talent, or status"). Added emotional engine test ("audience roots for them BECAUSE they are disadvantaged"). Explicitly demoted conflict_type as evidence source via NOTE. Added generic NOT examples (skilled grifters, structurally weaker faction, lone dissenter in debate). Removed conflict_type from Check line.
- **ensemble_cast**: Adopted official definition (multiple main characters share roughly equal importance, screen time, and storyline focus). Added group/event-as-protagonist framing. Added removal test ("if removing one character's arc collapses the film, that character is the lead"). Added SIGNAL CHECK warning about long plot_summaries with many named characters.
- **bittersweet_ending**: Added official definition anchor ("protagonist achieves main goal but suffers significant, concrete loss or cost"). Added NOT boundary: "structural ambiguity (open/ambiguous ending) is NOT emotional ambiguity — narrative uncertainty is not a loss."
- **endings section**: Reframed category header from "how the resolution feels" to "how the audience FEELS when the credits roll." Replaced "resolution" with "film's final moments" / "ending" throughout all 3 tag definitions. Added "Not all movies have resolutions, but all movies have endings." Added cliffhanger+sad example to SAD_ENDING. Changed 3-step to 4-step decision process: Steps 2-3 evaluate happy/sad with critical evidence assessment (not case-building), Step 4 evaluates bittersweet only as last resort with explicit "select NONE" for ambiguous/insufficient evidence. Step 2 explicitly instructs model to dismiss weak options upfront and to treat emotional_observations as primary over plot event inference.
- **EndingAssessment schema**: Updated reasoning description to reference 4-step process and "how the audience feels when credits roll." Updated tags description to frame empty list as correct when evidence is ambiguous.
- **ConceptTagsOutput.endings field**: Updated description to frame as audience emotion independent of structural tags.

## Endings decision process: sequential → parallel → evidence distillation
Files: movie_ingestion/metadata_generation/prompts/concept_tags.py, schemas/metadata.py, schemas/enums.py

### Intent
Post-evaluation analysis of 2-phase parallel approach showed bittersweet_ending at 16.7% precision (2 TP, 10 FP). Two root causes: (1) multi-way competition frame made NONE the hardest option to select — it needed to "win" against affirmative options rather than being the default; (2) schema used `conlist(EndingTag, max_length=1)` where empty list `[]` created completion pressure to fill the field after writing reasoning.

### Key Changes
- **Decision process → evidence distillation**: Replaced 2-phase argue-for-each-option approach with 3-step factual extraction: (1) extract ending-specific emotional_observations, filtering out journey-level emotions; (2) summarize final state of affairs (factual, no interpretation); (3) note ending-related plot_keywords. Tag selection follows from extracted evidence rather than per-option argumentation.
- **NO_CLEAR_CHOICE enum value**: Added to `EndingTag` with `concept_tag_id=-1`. Explicit "none of the above" option that the model selects as an affirmative classification rather than leaving a list empty.
- **EndingAssessment schema**: Changed `tags: conlist(EndingTag, max_length=1)` → `tag: EndingTag` (single required field). Eliminates empty-list completion pressure — choosing NO_CLEAR_CHOICE feels identical to choosing any other tag.
- **ALL_CONCEPT_TAGS**: Filters out tags with `concept_tag_id < 0` so NO_CLEAR_CHOICE never reaches storage or search.
- **all_concept_tag_ids()**: Updated to handle single `endings.tag` field and skip classification-only values.
- **_deduplicate_tags()**: Skips endings (no longer a list).
- **Bittersweet NOT list**: Added "losses that occur mid-film but not at the ending do not make an ending bittersweet" — addresses failure mode 1 (mid-film losses treated as ending evidence).
- **Removed per-tag Evidence lines**: Tag definitions now rely on the shared reasoning steps for evidence sourcing rather than repeating evidence instructions per tag.
