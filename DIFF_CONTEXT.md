# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Step 0 (Flow Routing) implementation
Files: schemas/enums.py, schemas/step_0_flow_routing.py, search_v2/step_0.py

### Intent
Implements Step 0 of the V2 search pipeline redesign per
`search_improvement_planning/steps_1_2_improving.md`. Step 0 is a narrow
classifier that decides which of the three major search flows
(exact_title, similarity, standard) should execute for a given raw
query, carries title payloads where needed, and picks the primary_flow
for result-list ordering. Runs in parallel with the existing Step 1
(stage_1.py); the merge happens in code afterward. Step 1 will be
re-shaped in a separate task.

### Key Decisions
- **Observations-first schema**: four extractive observation fields
  (titles_observed, qualifiers, ambiguous_title_phrases; similarity
  reference is carried by similarity_flow to avoid duplication) come
  before three per-flow decision fields and the primary_flow enum.
  FACTS → DECISION pattern, mirroring the award-endpoint scoped
  reasoning convention.
- **Three dedicated flow fields** instead of a `list[Flow]`: rules out
  duplicates/unused enum values by construction and matches the user's
  explicit request. `standard_flow` is a bool since it has no payload;
  the other two carry their title string directly.
- **Parameterizable provider/model/kwargs**: mirrors Stage 2A's pattern
  rather than Stage 1's pinned-config pattern because the user wants
  callers to pick the backend (no defaults).
- **AmbiguousLean enum added to schemas/enums.py** rather than the
  schema file, adjacent to SearchFlow and QueryAmbiguityLevel, per the
  existing enums-are-centralized convention.
- **Three Pydantic validators on Step0Response**: (1) at least one flow
  must fire, (2) primary_flow must correspond to a firing flow, (3)
  non-null flow titles must be non-empty. These are correctness gates
  the schema can enforce on its own rather than depending on prompt
  compliance.

### Planning Context
The schema shape was iterated in conversation — started from
FlowRoutingOutput with plausibility labels, collapsed through several
rounds of simplification ("evidence over yes/no", "three flow fields
not an array", "primary_flow enum at the end") before landing on the
final structure. The prompt's boundary examples section codifies the
canonical multi-flow cases (bare ambiguous phrase, similarity with
modifier, title-in-sentence) that drove the schema shape.

### Testing Notes
Import, schema emission, and validator rejection cases were smoke-tested
manually. No unit tests written per test-boundaries rule. Smoke queries
to try in a notebook once a provider/model is chosen:
"Interstellar", "scary movie", "movies like Inception", "I want to
watch Inception tonight", "Godfather and Goodfellas",
"movies where things blow up", "surprise me", "Intersteller".

## Step 2 (Query Pre-pass / Categorization) migration and finalization
Files: schemas/enums.py, schemas/step_2.py, search_v2/step_2.py
Deleted: search_v2/stage_2a.py, search_v2/stage_2b.py, search_v2/prepass_explorations.py

### Intent
Consolidates the step-2 categorization pre-pass into a permanent
module. Previously: experimental prepass_explorations.py scratchpad,
plus the older stage_2a.py / stage_2b.py pair that were superseded
by the category-taxonomy approach. This session finalizes the
pre-pass design (31-category taxonomy, four language types,
coverage-evidence decomposition, clarifying-evidence narrowing
rule), migrates it to a production module, and deletes the
superseded files.

### Key Decisions
- **CategoryName enum + concept descriptions in schemas/enums.py.**
  31 members (30 structured cats + Interpretation-required fallback).
  Descriptions cover attributes contained / concepts covered /
  questions answered but deliberately exclude routing, mechanism,
  and endpoint details — this enum is the LLM's view of the
  taxonomy, not the dispatcher's. Uses the tuple-constructor
  pattern already established by `NarrativeStructureTag` /
  `LineagePosition` so each member carries a `.description`
  attribute alongside its string value.
- **Output schemas in schemas/step_2.py.** `LanguageType`,
  `FitQuality`, `CoverageEvidence`, `RequirementFragment`,
  `Step2Response`. `CoverageEvidence` field order is
  evidence-first: captured_meaning → category_name → fit_quality →
  atomic_rewrite. User explicitly pushed this ordering after
  observing the LLM rationalizing categories in the old ordering
  ("decide evidence before judgment").
- **`fit_quality` gains "no_fit".** Third value for the case where
  captured_meaning turned out to be speculative / empty and should
  be discarded downstream — the auto-regressive "now-I-must-pick-a-
  category" trap was a real failure mode. Interpretation-required
  still handles "real but no structured fit"; no_fit is the
  explicit abandon path.
- **CLARIFYING EVIDENCE prompt section.** Teaches the model a
  definitional-vs-stylistic exclusion test for multi-dimension
  entities (Spider-Man is character AND franchise) against
  meta-relation qualifiers (spinoff, parody, inspired-by). Written
  principle-first (the HP example that prompted it is deliberately
  kept out of the prompt). Narrow rule: reading-narrowing, not
  fragment-erasing.
- **Taxonomy prompt section built programmatically from the enum.**
  `_build_category_taxonomy_section()` iterates `CategoryName` and
  emits "Cat: <name>\n  <description>" — prompt text and
  structured-output vocabulary cannot drift because they're
  sourced from the same enum.
- **Finalized Gemini 3 Flash, no model parameters on the executor.**
  `run_step_2(query)` hard-codes provider=Gemini,
  model=gemini-3-flash-preview, thinking_budget=0, temperature=0.35.
  User asked to lock the config once the pre-pass was dialed in,
  removing provider/model overrides from the executor entirely.
  Differs from Step 0 / Step 1 / Stage 2A intentionally — this
  step is finalized, those are still in-tuning.

### Planning Context
Pre-pass was iterated through ~5 major revisions over two
sessions. Key design moves:
- Single-pass schema (no separate "problematic requirements" list)
  — one decomposition per attribute fragment, atoms emerge via
  coverage_evidence entries.
- Four language types (attribute / selection_rule / role_marker /
  polarity_modifier) — small words like "starring", "first", "not
  too" carry real signal and get their own fragment type.
- Specificity preservation ("brother" stays "brother", not
  "sibling"; "1990s" not "older") — explicit prompt rule after
  observing generalization failures in earlier probes.
- `category_name` typed as the `CategoryName` enum (not a free-form
  string) — prevents hallucinated categories and makes vocabulary
  changes a structured edit rather than a prompt edit.

### Testing Notes
Per test-boundaries rule, no tests written or modified.
Smoke-verified: 31 categories load, `.description` property works,
SYSTEM_PROMPT builds (~30K chars) and includes CLARIFYING EVIDENCE
+ full taxonomy, `run_step_2` signature is just `query`.

Dangling references flagged to the user (left untouched — see
follow-up TODO): `search_improvement_planning/debug_stage_2a.py`
and `debug_feedback_queries.py` still import `search_v2.stage_2a`
/ `stage_2b`; `unit_tests/test_search_v2_stage_2.py` uses
`Step2AResponse` / `Step2BResponse`; those response classes
still live in `schemas/query_understanding.py` and may be
orphaned.

## Step 1 (Spin Generation) implementation
Files: schemas/step_1.py (new), search_v2/step_1.py (new), search_v2/stage_1.py (deleted)

### Intent
Reshapes Step 1 per the finalized design in
`search_improvement_planning/steps_1_2_improving.md` and the follow-on
discussion that simplified the task further. Step 1 now runs in
parallel with Step 0 on the raw query and generates exactly two
creative "spins" plus UI labels for all three branches (original +
two spins). The original query passes through downstream verbatim —
Step 1 no longer owns a "faithful primary rewrite" because Step 2
produces `rewritten_query` with specificity-preservation enforced by
its schema, removing the redundant burden from Step 1's prompt.

### Key Decisions
- **Observations-first schema**: three decomposition fields
  (hard_commitments, soft_interpretations, open_dimensions) come
  before any spin is committed to. Each spin's branching_opportunity
  must ground in a specific item from soft_interpretations or
  open_dimensions — keeps small models from free-associating spins.
- **Two spins, always**: previous design varied spin count by
  "ambiguity tier". Simpler contract is always exactly two
  (min_length=max_length=2). Step 0's merge trims from the tail if
  non-standard flows consume slots, so the worst case is a wasted
  small-model call on title/similarity-only queries — accepted cost
  per the planning doc.
- **Distinctness as a first-class field**: each Spin carries a
  `distinctness` field stating how its result set differs from BOTH
  the original and the sibling spin, in retrieval terms. Prevents
  the common failure mode where two spins pull the same lever in
  slightly different directions.
- **Spin construction shapes**: three explicit patterns in the
  prompt — reinterpretation (commit a soft term to one reading),
  narrowing (add a constraint on an open dimension), and adjacent
  swap (fallback for fully specified queries with no soft terms).
- **UI labels for all three branches**: `original_query_label` plus
  one label per spin. 2-5 words, Title Case, lean into the
  distinguishing lever for spin labels.
- **Model choice**: Gemini 3.1 Flash Lite Preview (faster than
  step_2's Flash) with thinking disabled and temperature 0.35 —
  matches step_2's kwargs except the lighter model. Step 1's task
  is more open-ended than step 2's but operates on less structure,
  and its output drives UI copy rather than downstream
  decomposition, so the latency win is worth the capability trade.

### Planning Context
Earlier conversation pinned down that Step 1's role is purely the
*branching layer* for the standard flow — it turns one raw query
into the set of intents Step 2 decomposes independently. Step 0
handles flow routing and title extraction; Step 2 handles per-intent
decomposition with specificity preservation. Step 1 sits in the gap
by generating alternate exploration directions that nothing else in
the pipeline produces.

### Testing Notes
- Distinctness is the hardest quality bar: eval should check whether
  the two spins' retrieved candidate sets have low Jaccard overlap,
  not just whether the `distinctness` field reads plausibly.
- Specificity-preservation is Step 2's job now, but Step 1's spin
  `query` strings should still preserve hard_commitments verbatim —
  worth a unit check that named entities in the original appear in
  both spin queries unless the spin's lever explicitly swaps that
  axis.
- UI label variety: check that labels across the three branches
  don't all end with the same noun ("Movies", "Films").

## Steps 0-2 orchestrator
Files: search_v2/steps_0_2_orchestrator.py, search_v2/test_search_v2.ipynb
Why: Need a single entry point that runs the front half of the V2 pipeline (steps 0+1 in parallel, then per-flow dispatch into step 2) so callers and notebooks stop hand-wiring it.
Approach: asyncio.gather steps 0/1 with return_exceptions=True. Step 0 failure raises (no routing → no dispatch). Step 1 failure is captured per-flow: exact_title and similarity flows are independent of step 1, and the standard flow degrades to a single original-query step-2 branch when spins are unavailable. Standard-flow branch budget = 3 minus the count of non-standard flows that fired (drawing original → spin_1 → spin_2 in priority order); branches run in parallel with per-branch error isolation. Exact_title and similarity flows are TODO no-ops surfaced as boolean flags on the result. Returns dataclasses (OrchestratorResult, Step2Branch) since the orchestrator is library-only and never serialized — the notebook consumes the structure directly.
Notebook: dropped the standalone Step 0 cell. Added a Step 1 isolation cell and an orchestrator cell (uses run_steps_0_to_2). Setup cell now also imports run_step_1, run_steps_0_to_2, and the Step1/Step2 schemas.

## Orchestrator review follow-ups
Files: search_v2/steps_0_2_orchestrator.py, search_v2/step_1.py
- Step2Branch.input_tokens/output_tokens/elapsed are now Optional — failed branches carry None instead of 0 so token aggregators don't silently under-count a failure as a free success.
- Step 1 model reverted to `gemini-3-flash-preview` to match steps 0 and 2. The three front-half prompts now share one backend profile (Gemini 3 Flash, thinking disabled, temperature 0.35).
