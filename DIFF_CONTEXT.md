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

## Step 2 prompt: tighten multi-dimension entity rule
Files: search_v2/step_2.py | Reframed multi-dimension entity guidance so Named character + Franchise/universe lineage only co-emit when each reading INDEPENDENTLY holds. Added three canonical shapes (persona+franchise like Spider-Man; persona-only like Neo; franchise-only like Star Wars) across `_TASK_AND_OUTCOME`, `_COVERAGE_EVIDENCE_RULES`, `_ATOMIZATION_PATTERNS` #11, and `_GROUND_RULES` #4. Fixes a regression where "Star Wars" was emitting a spurious Named character entry because every prior example paired franchise with a same-named character.

## Step 2 schema migration: collapse fragment types, nest modifiers, drop per-fragment rewrite
Files: schemas/enums.py, schemas/step_2.py, search_v2/step_2.py, search_v2/test_search_v2.ipynb

### Intent
Restructures the step-2 output so each requirement is a self-contained package the downstream categorizer can dispatch per coverage entry without reassembling cross-fragment context. Previous shape had four language types at the fragment level (attribute / selection_rule / role_marker / polarity_modifier), a per-fragment `full_rewrite`, and a top-level `rewritten_query`. Step 3 was re-parsing the smoothed prose to recover signal that step 2 had already computed — a double translation that cost tokens and lost information.

### Key Decisions
- **Every fragment is an attribute.** Role markers and polarity phrases are no longer standalone fragments; they nest inside the adjacent attribute's `modifiers` list as `Modifier` objects carrying `original_text`, `effect`, and `type` (POLARITY_MODIFIER | ROLE_MARKER). Supports multiple modifiers per attribute (resolves stacking like "not preferably too violent"). `LanguageType` enum reduced from four values to two.
- **Selection rules collapse into attributes via a new Chronological category.** "First", "last", "earliest", "latest", "most recent" are attribute fragments mapped to `CategoryName.CHRONOLOGICAL`. Reception-framed superlatives ("best") and popularity framings ("rarely-seen") route to RECEPTION_QUALITY / STRUCTURED_METADATA as attributes — not as selection rules. The prompt was previously inconsistent with the taxonomy on this point; that's now fixed. Result-count limits ("top 10") deliberately NOT modeled — user's call to avoid any schema lever that could truncate results if the LLM gets it wrong. Bare era framing ("recent", "newer") stays with STRUCTURED_METADATA; only ordinal-position phrasing routes to Chronological — the CHRONOLOGICAL description spells this out.
- **Dropped `full_rewrite` and `rewritten_query`.** Captured_meaning is already dimension-aware and atomic_rewrite already reflects nested modifiers, so the smoothed prose layers were redundant at a meaningful token cost. `description` kept on the fragment per user preference — aligns the LLM before it starts atomizing.
- **Per-coverage-entry dispatch, application-side flattening.** LLM still emits the nested fragment→coverage shape for natural grouping; downstream flattens to `(query_text, modifiers[], captured_meaning, category, fit_quality, atomic_rewrite)` packages before dispatch, duplicating fragment-level fields into each package. `no_fit` entries pruned at flattening, never dispatched.
- **`FitQuality` and `LanguageType` promoted to StrEnums in schemas/enums.py.** Previously Literal types inside schemas/step_2.py; now proper enums for downstream ergonomics. `PolarityModifier` / `RoleMarker` deliberately NOT enums — freeform strings are fine since they only feed the next LLM as context.
- **Prompt rewrite.** `_LANGUAGE_TYPES` section renamed conceptually to FRAGMENT STRUCTURE (one attribute fragment per chunk; two kinds of nested modifiers). `_CHUNKING_RULES` now teaches modifier nesting and treats chronology as its own attribute fragment. `_COVERAGE_EVIDENCE_RULES` adds a Chronological example and requires atomic_rewrite to surface modifier signal. `_ATOMIZATION_PATTERNS` gains a new item 8b for chronological ordinals. `_REWRITE_RULES` deleted entirely. `_GROUND_RULES` rewritten — rule 3 names "best"/"rarely-seen" as attributes (not selection rules), rule 6 replaced with the modifier→atomic_rewrite surfacing requirement. Pattern 7 atomic_rewrite rephrased to avoid polarity-like prose ("not too heavy" → "light rather than heavy") since user-typed polarity now has a structured home.

### Planning Context
Shape was iterated in conversation through several rounds: first confirmed that role_markers and polarity_modifiers are purely context-adders (never standalone), then collapsed selection_rules into attributes after observing the taxonomy already absorbs "best" and "rarely-seen", then added CHRONOLOGICAL as the only residual category needed for ordinal picks. User explicitly declined a result_limit field to avoid any lever that could silently truncate results on LLM error. Modifiers-as-list with (original_text, effect, type) was the final shape — supports chaining and preserves verbatim spans for debugging alongside the brief semantic note.

### Testing Notes
- schema import + field shape verified via python -c smoke test.
- SYSTEM_PROMPT composition verified: CHRONOLOGICAL present, full_rewrite / rewritten_query / selection_rule references purged.
- Notebook cells (Cell 4 Step 2 isolated and Steps 0-2 orchestrator display) updated to the new shape; previous cells referenced `frag.type`, `frag.full_rewrite`, `branch.response.rewritten_query` and would have crashed on next run.
- Eval surface worth verifying in notebook: that modifiers actually populate for queries like "starring Tom Hanks" and "not too violent"; that "first Indiana Jones" emits a Chronological fragment separate from the Indiana Jones fragment; that "recent Nolan films" stays on STRUCTURED_METADATA and does NOT get a spurious Chronological entry.

## Stage-3 dealbreaker scores aligned to [0.5, 1.0] floor
Files: search_v2/stage_3/result_helpers.py, search_v2/stage_3/trending_query_execution.py, search_v2/stage_3/award_query_execution.py, search_v2/stage_3/metadata_query_execution.py, search_v2/stage_3/semantic_query_execution.py

Why: keyword/studio/franchise/entity already emitted dealbreaker scores in [0.5, 1.0] ("dealbreaker-eligible band"), but trending, award (THRESHOLD mode), semantic, and five metadata attributes (release_date, runtime, country_of_origin, popularity, reception) did not. Downstream candidate-pool aggregation assumes a uniform floor; the outliers let weak matches land arbitrarily close to 0.

Approach: added `compress_to_dealbreaker_floor(raw) = 0.5 + 0.5 * raw` to result_helpers.py (mirrors the pattern in entity's private `_compress_to_floor`). Applied at each offender's dealbreaker branch only (gated on `restrict_to_movie_ids is None`). Preference paths keep raw [0, 1] scoring so non-matches stay at 0.0 and ranking gradient is preserved. Trending compresses at executor level — Redis values and `compute_trending_score` untouched. Award's fast path already emits 1.0, standard path compresses post-zero-drop. Metadata release_date/runtime/popularity/reception compress post-existing-zero-drop. Semantic modifies `_threshold_flatten` directly since it's only reachable from dealbreaker paths. Country-of-origin's exp-decay was inappropriate to compress (pos 3+ would get promoted to 0.5); added a sibling `_score_country_position_dealbreaker` that returns 1.0/0.5/0.0 for positions 1/2/≥3 and drops zero scores.

Design context: dealbreaker-only scope confirmed with user — compressing both paths would warp preference ranking by inflating weak matches to 0.5.

Testing notes: behavior-boundary spot checks to run in a notebook — trending rank 1→~0.98 (compression of raw ~0.955), rank n→0.5; award THRESHOLD count=1,mark=10→0.55, count≥mark→1.0; release_date in-window→1.0, grace-edge→0.5; runtime ±30min edge→0.5; country pos 1→1.0, pos 2→0.5, pos 3→dropped; popularity/reception raw 0→dropped, raw 0.01→~0.505; semantic sim>=elbow→1.0, sim just above floor→~0.5, sim<=floor→dropped. Preference-path byte-identical behavior preserved (else branches unchanged).
