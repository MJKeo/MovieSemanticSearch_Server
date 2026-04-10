# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Franchise generator v3 — prompt + schema rewrite from test analysis
Files: movie_ingestion/metadata_generation/prompts/franchise.py, schemas/metadata.py

### Intent
Full rewrite of the franchise generator prompt and FranchiseOutput schema
descriptions, driven by per-candidate analysis of the v2 test run (79
movies × 4 models × 3 samples). The v2 evaluation surfaced three failure
classes the v2 prompt could not fix without a structural redesign:
(1) spinoff was being over-applied to shared-universe pillar films
(Wonder Woman, Black Panther) and legacy sequels (Creed) — every wrong
trace followed the same shortcut "character appeared before + now has
own film = spinoff"; (2) "prefer missing over wrong" framing caused
medium/low tiers to correctly punt to role=None on Transformers,
Guardians of the Galaxy, Dragon Ball Z, Rurouni Kenshin, fighting the
SOT — the prompt was biasing toward an outcome rather than critical
thinking; (3) culturally_recognized_groups failures dominated the raw
row count because the prompt used brainstorm-then-filter framing and
the Raiders-of-the-Lost-Ark → ["original trilogy"] example trained
models to pad every trilogy with "original trilogy".

### Key Decisions
- Prompt architecture rewritten around concept_tags.py patterns:
  evidence hierarchy (direct / concrete inference / parametric) stated
  up front; per-field numbered procedural walkthroughs replace
  free-form "Step 1 Evidence / Step 2 Analysis" bullets; comparative
  evaluation for franchise_role ("build case for each candidate role,
  commit to strongest") replaces first-match top-to-bottom scan;
  EMPTY-THEN-ADD framing for culturally_recognized_groups replaces
  brainstorm-then-filter; IS NOT clauses added to every field.
- Spinoff operationalized with three constraints that must ALL hold:
  (a) MINOR IN SOURCE — the character/element/subplot was minor, not
  a lead, co-lead, or planned shared-universe pillar; (b) GOES
  SOMEWHERE NEW — new protagonist/story spine/setting focus;
  (c) LEAVES THE SOURCE BEHIND — source film's main characters and
  plot are not the focus here (this is what distinguishes spinoffs
  from legacy sequels like Creed, where Rocky is a returning major
  character). Explicit "Examples where constraints fail" block lists
  Wonder Woman, Black Panther, Creed, Ghostbusters: Afterlife as
  mainline not spinoff, with the specific constraint each fails.
- NULL role is now a first-class answer with two documented valid
  cases, not a hedge: (i) first theatrical entry in a franchise
  whose prior existence is limited to non-cinematic media (toys,
  games, TV, anime, manga, comics-only) — Transformers 2007, Sonic
  2020, Detective Pikachu, Iron Man 2008, Rurouni Kenshin Origins,
  X-Men 2000 (Fox continuity); (ii) documentaries. "Null is NOT
  correct when choosing between two plausible roles — commit to the
  stronger" is stated explicitly. SOT needs realignment on the null
  cases in a separate pass before re-running the test set.
- "Prefer missing over wrong" language removed throughout. Replaced
  with positive commitment framing at the top of the prompt and in
  every field's procedure: "when evidence points to a franchise,
  commit; null is reserved for genuine non-fit, not uncertainty".
- culturally_recognized_groups is rewritten with three named evidence
  tiers (studio/official usage, mainstream critical convention,
  widely-used fan terminology), and the IS NOT filter now explicitly
  blocks bare sub-series restatements ("minions", "creed", "guardians
  of the galaxy", "the avengers", "puss in boots", "hobbs and shaw",
  "fantastic beasts", "iron man trilogy") which were the single
  biggest noise source in the v2 test. The Raiders → ["original
  trilogy"] example booby trap is removed; "original trilogy" is now
  explicitly scoped to Star Wars only. The normalization rule to
  drop first names on director-era labels is kept, with the Sherlock
  Holmes case added as an explicit example.
- launches_subgroup is coupled to culturally_recognized_groups via
  an internal consistency check ("if true, the launched subgroup's
  label MUST appear in the groups list"). Addresses action item #8
  from the v2 iterations report. Mad Max 1979 is given as an explicit
  FALSE example because "the original Mad Max trilogy" is not a
  culturally-used label despite being first of a trilogy.
- is_prequel is tightened with IS NOT clauses covering reboots set
  early (Batman Begins), flashback-heavy mainline films, and films
  set in the same time period as an earlier entry.
- Top-level entity rule is elevated into the franchise_name procedure
  as a forced step (not just stated in a separate section) — this is
  the minimal-tier failure mode where models say name='batman' /
  'x-men' / 'spider-man' because they don't re-apply the rule at
  decision time.
- INPUTS section now annotates every input with its reasoning role
  (primary / confirming / cross-reference signal) and which fields
  it serves. top_billed_cast is explicitly called out as the primary
  signal for the spinoff "prior-role prominence test".
- FranchiseOutput schema descriptions encode the same procedural
  walkthroughs in compact numbered form so the structured-output
  JSON schema itself carries the procedure, not just the system
  prompt. FranchiseRole enum unchanged (no roles added or removed;
  null covers the new non-cinematic-IP-first-theatrical case).

### Planning Context
Evaluation report lives at
search_improvement_planning/franchise_test_iterations.md (analyses
v1 and v2 runs). Per-candidate analysis in this session's chat
history identified which failures were genuinely harmful vs.
debatable vs. inconsequential and mapped each recurring failure
pattern to a specific prompt-language root cause. The rewrite
targets those root causes rather than patching individual failures.

### Testing Notes
- Next step is to re-run the v2 test set (franchise_test_results.json)
  against the new prompt and compare per-candidate H/D/I counts. Expect
  large drops in culturally_recognized_groups noise and in spinoff
  over-application; expect Transformers / Guardians / Dragon Ball Z /
  Rurouni Kenshin role=None to now be correct (not D) provided SOT is
  also updated.
- SOT needs pre-run realignment for the explicit null-role cases
  documented in the new prompt. Without SOT updates, re-run numbers
  will still show Transformers-class failures as wrong even though
  they are now the prompt's intended answer.
- Verify launches_subgroup / culturally_recognized_groups co-occurrence
  after re-run (action item #8 from iterations report).

## Franchise generator v2 — definitions, schema, prompt, SOT
Files: schemas/enums.py, schemas/metadata.py, movie_ingestion/metadata_generation/prompts/franchise.py, movie_ingestion/metadata_generation/generators/test_franchise.ipynb

### Intent
Re-ground the franchise generator on a tightened, internally consistent set
of definitions. Analysis of the last test-set run (76 movies × 3 tiers × 3
samples) showed the majority of harmful failures traced back to four
definitional holes: loose role definitions, umbrella-vs-lineage tension in
`franchise_name`, no way to express prequels or subgroup-launches, and no
`crossover` role. This change locks in resolutions for all four, updates the
Pydantic schema + enum to match, rewrites the system prompt, and updates the
test notebook SOT and evaluation cells. No DB/ingestion code touches franchise
fields yet so the change is contained to the generator + tests.

### Key Decisions
- `franchise_name` now always carries the TOP-LEVEL brand entity (Marvel,
  DC Comics, Harry Potter, Godzilla). Sub-groupings (MCU, DCEU, Dark Knight
  Trilogy, Raimi Trilogy, MonsterVerse, Wizarding World, Michael Bay era,
  actor runs, phase N, saga N) all move to `culturally_recognized_groups`.
  This kills the umbrella-vs-lineage flip-flop and makes "marvel movies"
  vs "MCU movies" a group-filter question rather than a naming question.
- Role definitions tightened: `starter` now requires that the franchise
  ENTITY did not exist before the film in any form. Books / comics / games
  do NOT block starter (so Jurassic Park 1993, LOTR 2001 are still
  starters); toys / products / TV / prior theatrical entries DO block it
  (so Barbie 2023, Transformers 2007, Super Mario 2023 are not).
  `mainline` is explicitly demoted to the fallback-of-last-resort — only
  assign it after ruling out all other roles, AND null-role with a
  populated `franchise_name` is a valid preferred output when no role
  cleanly fits (Iron Man 2008, Barbie 2023, Sonic 2020).
- `spinoff` redefined to cover minor character OR story element OR subplot
  focus. Creed is no longer a spinoff (new lead character not in prior
  Rocky films + Rocky is a major co-lead) — it's now mainline. Rogue One
  is a spinoff via the "Death Star plans" subplot hook, plus `is_prequel`.
- `remake` vs `reboot` split via story-spine-preservation: Total Recall
  2012 is a remake (same "artificial memories → spy awakening" spine);
  Jungle Book 2016 is a remake; Super Mario 2023 is a reboot.
- Two new orthogonal boolean fields: `is_prequel` (set chronologically
  before a prior-released entry in the same franchise; orthogonal to
  role so Rogue One can be spinoff+prequel and Episode I can be
  mainline+prequel) and `launches_subgroup` (first entry of a notable
  culturally-recognized subgroup, e.g. Iron Man 2008 → MCU, Spider-Man
  2002 → Raimi trilogy, Godzilla 2014 → MonsterVerse).
- New `crossover` role for Freddy vs Jason / AvP / Godzilla vs Kong.
  Dominant parent franchise goes in `franchise_name`; the secondary is
  recoverable via character search.
- Group label normalization is handled PROMPT-FIRST rather than via an
  ingest-time normalizer. Added explicit rules to the prompt: lowercase,
  digits spelled as words ("phase three" not "phase 3"), "&" expanded to
  "and", most-common canonical phrasing ("raimi trilogy" not "sam raimi
  trilogy"), never restate the franchise name ("marvel film series" /
  "godzilla films" forbidden). If the next test run shows the model
  can't hold these rules, escalate to a programmatic normalizer.
- FranchiseRole enum gains `CROSSOVER = 6`. FranchiseOutput gains
  `is_prequel_reasoning` + `is_prequel` and `launches_subgroup_reasoning`
  + `launches_subgroup` (reasoning-before-decision pattern matches the
  existing fields). All existing field descriptions updated to reference
  the top-level-entity rule and the null-role-valid carve-out.
- Test notebook SOT updated for ~27 entries: Creed → mainline; all
  MCU/DCEU/Batman/Spider-Man/X-Men/Logan films re-pointed at
  Marvel/DC Comics as franchise_name with cinematic-universe labels
  moved to groups; Iron Man → role=null + launches_subgroup=True;
  Super Mario 2023 / Transformers 2007 → reboot; Barbie / Sonic /
  Resident Evil / Scooby-Doo → null role; Sherlock Holmes 2009 →
  reboot; Detective Pikachu → reboot (name="Pokemon"); Rogue One,
  Solo, Fantastic Beasts, Phantom Menace → is_prequel=True;
  Minions → is_prequel=True; Batman Begins, Casino Royale,
  Spider-Man 2002, X-Men 2000, Rise of Apes, Godzilla 2014,
  Transformers 2007, Sherlock Holmes 2009, Phantom Menace →
  launches_subgroup=True; all group labels normalized to words.
  New counts: 57 franchise / 19 standalone / 5 null-role-with-name
  / 5 prequels / 10 subgroup-launchers.
- Test notebook evaluation helpers (cell 8) extended with two new
  RunResult values (`PREQUEL_WRONG`, `LAUNCHES_WRONG`), new
  check_null_pairing semantics (name-without-role no longer a
  violation; is_prequel=True without name IS a violation), and new
  classify_run signature with optional `expected_is_prequel` /
  `expected_launches_subgroup` defaulted to False. Cells 9-13 updated
  to carry the new expected fields through per-candidate summary,
  per-bucket breakdown, inspector, majority-vote, and null-pairing
  report.

### Planning Context
Decision trace lives in /Users/michaelkeohane/.claude/plans/elegant-meandering-sundae.md
and in search_improvement_planning/franchise_test_iterations.md. The locked
decisions listed above were all validated against specific failure cases
from the prior run (Creed, Barbie, Iron Man, Super Mario 2023, Total Recall
2012, Jungle Book 2016, Rogue One, Detective Pikachu, Fantastic Beasts,
Godzilla 2014) before the prompt was rewritten.

### Testing Notes
End-to-end: re-run test_franchise.ipynb cells 3→6 to regenerate
franchise_test_results.json under the new prompt/schema, then 9/10/11/12/13
for the updated metrics. Spot-check: Creed majority → mainline; Iron Man
majority → franchise=Marvel, role=null, launches_subgroup=True; Super Mario
2023 → reboot; Barbie → null role; Fantastic Beasts → spinoff+prequel with
franchise=Harry Potter; Rogue One is_prequel=True; any MCU film groups use
"phase one/two/three" in words; Spider-Man 2002 groups use "raimi trilogy".
Read 2–3 franchise_name_reasoning traces to confirm the top-level rule is
actually being applied. If normalization inconsistencies persist (digit
phase labels etc.) escalate to a shared ingest-time normalizer applied at
both SOT-load and generation-write time.

Known gaps left for follow-up: no crossover movies in the test set (new
`crossover` role is untested); no documentary entries; Jurassic Park is
present but LOTR / Harry Potter 1 are not, so the "books don't block
starter" rule is only lightly tested.

## Add per-field evidence→analysis reasoning to FranchiseOutput
Files: schemas/metadata.py, movie_ingestion/metadata_generation/prompts/franchise.py
Why: Franchise classification is a multi-step judgment (identify the IP, classify the role, enumerate established sub-groupings) and the model was committing to answers without visibly weighing the signals. Per-field CoT scaffolding has been effective elsewhere in the metadata schema (CharacterArcWithReasoning, ElevatorPitchWithJustification) so adopt the same pattern here.
Approach: Added three required reasoning fields to `FranchiseOutput` — `franchise_name_reasoning`, `franchise_role_reasoning`, `culturally_recognized_groups_reasoning` — each placed immediately BEFORE its decision field so OpenAI structured-output field order forces the model to produce the reasoning first. Each reasoning field's description mandates a two-step process: Step 1 gather evidence (list relevant signals), Step 2 analyze that evidence and commit to the decision. For the role and groups fields, when `franchise_name` is null the reasoning should be "N/A — standalone" (kept required rather than optional to avoid an extra conditional branch in the schema). Rewrote the prompt's FIELDS section to mirror the schema: each numbered field now opens with an explicit Step 1 / Step 2 breakdown of what evidence to gather and how to analyze it, with a leading paragraph establishing the "evidence → analysis → decision, never skip straight to the answer" rule.
Design context: Pattern matches existing `*WithReasoning` / `*WithJustification` sub-models in schemas/metadata.py — reasoning fields are generated first, are never embedded (franchise output is non-embeddable anyway), and exist purely to scaffold higher-quality labels. Chose required strings over Optional so the model cannot silently skip the reasoning step.
Testing notes: Re-run franchise generation on the existing test set (movie_ingestion/metadata_generation/generators/test_franchise.ipynb, franchise_test_results.json) and spot-check that (a) the reasoning fields show genuine evidence enumeration rather than restating the decision, (b) franchise_name / role / groups accuracy is equal or better than the prior run, (c) the "N/A — standalone" convention is followed when franchise_name is null. Also verify the batch pipeline still parses FranchiseOutput cleanly — extra_forbid is on so any drift in the response schema will fail loudly.

## Fix female_lead over-inclusion via top_billed_cast + stricter reasoning flow
Files: schemas/movie_input.py, movie_ingestion/metadata_generation/generators/concept_tags.py, movie_ingestion/metadata_generation/prompts/concept_tags.py
Why: `female_lead` was firing on nearly every movie. Root cause was twofold: (1) the concept_tags prompt has a global "when debatable, include it — a missing tag is worse than an extra tag" bias that conflicts with a binary-gender question that should default to no, and (2) the LLM had no reliable prominence signal — it had to infer protagonist gender from plot_summary alone, which skews toward any named female character regardless of actual lead status.
Approach:
  - Added `actors: list[str]` to MovieInputData and fetched `i.actors` in `load_movie_input_data`. The column already exists in `imdb_data`.
  - Added `MovieInputData.top_billed_cast(n=5)` which pairs actors[:n] with characters[:n] and renders "Character (Actor), ..." in billing order. Pairing is positional — mostly correct, may drift slightly past the top billed few for rare multi-role actors, which is acceptable for a prominence signal.
  - `build_concept_tags_user_prompt` now passes `top_billed_cast` as a new labeled input (fallback "not available" when no actors).
  - Rewrote the FEMALE_LEAD tag definition as an explicit 3-step reasoning flow: (1) does the story have a single core protagonist at all? If not (two-hander, ensemble, trio), do not tag; (2) if yes, name them — use plot_summary as primary, cross-reference top_billed_cast for prominence, treat top-billed-man + no clearly centered woman as strong negative signal; (3) is that single core character female? Tag only if yes with high confidence.
  - Added an explicit override: "The 'when debatable, include it' rule does NOT apply to FEMALE_LEAD — a false positive here is worse than a miss." Also made two-handers ineligible (the previous version tagged two-handers if one co-lead was female, which was a major over-inclusion vector).
  - Added a new INPUTS entry for top_billed_cast explaining it's a cross-reference prominence signal, not a determinant.
  - Updated the module docstring failure-mode note to describe the new FEMALE_LEAD reasoning flow.
Design context: Chose n=5 (not n=3) for top_billed_cast so the model can see the drop-off past position 3 — that drop-off is what distinguishes a genuine ensemble from "single lead + two notable supports." 5 also amortizes across future ENSEMBLE_CAST tightening.
Testing notes: Need to re-run concept_tags over the validation set to confirm female_lead recall drops to a sane rate without collapsing legitimate female-lead classifications (e.g. Erin Brockovich, Lady Bird, Promising Young Woman). Watch for two-handers (e.g. Thelma & Louise) which are now deliberately excluded — if the user wants those tagged, the rule needs to loosen back up.

## Collapse franchise prompt title/year into single title_with_year line
Files: movie_ingestion/metadata_generation/generators/franchise.py, movie_ingestion/metadata_generation/prompts/franchise.py | Pass `movie.title_with_year()` as a single `title_with_year` field in the user prompt instead of separate `title` and `release_year` lines; updated the INPUTS section of the system prompt to match.

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

## Franchise generator v2 follow-ups — SOT consistency + richer inputs
Files: schemas/movie_input.py, movie_ingestion/metadata_generation/generators/franchise.py, movie_ingestion/metadata_generation/prompts/franchise.py, movie_ingestion/metadata_generation/generators/test_franchise.ipynb, search_improvement_planning/franchise_test_iterations.md

### Intent
Post-v2 verification surfaced (a) three SOT rows that violated the locked Marvel-predates-starter rule or had `launches_subgroup=True` paired with an empty `culturally_recognized_groups` list (logically inconsistent under the v2 definitions), and (b) three cheap, high-signal inputs the franchise user prompt wasn't passing. This entry captures both fixes so the next full-run numbers are trustworthy.

### Key Changes
- **SOT — Spider-Man (2002) & X-Men (2000)**: `expected_role: "starter" → None`. Marvel is the top-level brand and predates both films, which blocks starter per the v2 definition. Both now match the Iron Man (1726) pattern: `franchise_name="Marvel"`, `role=None`, `launches_subgroup=True`, `groups=["raimi trilogy"] / ["x-men"]`. Null-role-with-name count rises from 5 to 7.
- **SOT — launches_subgroup ↔ groups invariant**: Every SOT row with `launches_subgroup=True` now carries a label in `expected_groups`. Filled in: Rise of the Planet of the Apes 2011 → `["caesar trilogy"]`, Transformers 2007 → `["bay era"]`, Sherlock Holmes 2009 → `["ritchie sherlock holmes films"]`. Verified no remaining `launches=True & groups=[]` rows.
- **MovieInputData.directors**: Added `directors: list[str]` field. `load_movie_input_data` now SELECTs and parses `i.directors` from the tracker's `imdb_data` table (already populated by the IMDB scraper). No migration needed.
- **Franchise user prompt — new inputs**: `build_franchise_user_prompt` now passes `release_year` as a separate labeled line, `directors` (explicitly labeled), and `top_billed_cast(5)` (actor+character pairings) alongside the existing `characters[:5]`. Release year supports chronology reasoning for `is_prequel`; directors + cast pairings are high-signal for director-era / actor-run subgroup detection ("raimi trilogy", "bay era", "daniel craig era").
- **Prompt INPUTS section**: Extended to document the three new fields and what signal they carry, so the model explicitly knows to lean on them.
- **franchise_test_iterations.md**: Added action-item #8 to check in the next run how often `launches_subgroup=True` co-occurs with an empty `culturally_recognized_groups`. If frequent, we'll tighten the prompt to require the launching label be present in the groups list.

### Verification
- Schema import + prompt build smoke test run with a populated Spider-Man (2002) fixture and a sparse fixture — both render cleanly, missing fields show `"not available"`.
- Cell 3 reparse: 76 entries, null-role-with-name count 7, zero `launches=True & groups=[]` rows.
- No consumer outside the generator + test notebook reads `movie_input.directors` yet, so the schema change is isolated.

## remove stale implementation/ files (chromadb-era dead code)
Files: implementation/vectorize.py, implementation/searching/, implementation/visualize.py, implementation/chroma_db/, AGENTS.md, docs/modules/ingestion.md, docs/modules/classes.md, docs/PROJECT.md, .claude/commands/analyze-metadata-inputs.md, search_improvement_planning/v2_data_needs.md, memory/project_embedding_model_decision.md
Why: User pruned implementation/scraping, implementation/generated_data, implementation/notebooks. Verified vectorize.py/searching/visualize.py/chroma_db/ were also stale (chromadb imports, BaseMovie deps, zero external importers; live replacements live at movie_ingestion/final_ingestion/vector_text.py and movie_ingestion/final_ingestion/ingest_movie.py) and removed them.
Approach: (1) confirmed no external importers via grep; (2) confirmed unit_tests/test_ingest_movie.py already has defensive ModuleNotFoundError fallback stubs for implementation.vectorize so deletion is safe; (3) test_vector_text.py imports from movie_ingestion.final_ingestion.vector_text, not the deleted module; (4) deleted files; (5) rewrote stale doc references to point at the live movie_ingestion paths. Also saved embedding-model decision (upgrade to text-embedding-3-large, Voyage-3-large as fallback) to search_improvement_planning/v2_data_needs.md #12 and updated the stale memory record that previously said "stay with 3-small".
Testing notes: test_ingest_movie.py fallback path is now the only path (module no longer importable). Worth a pytest run to confirm. ADR-060 still references implementation/vectorize.py as a historical note — left intact since ADRs are point-in-time decision records.
