# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## category_handlers module scaffold + schema-file relocation
Files: search_v2/stage_3/category_handlers/{__init__,handler,prompt_builder,handler_result,endpoint_registry,schema_factories}.py, search_v2/stage_3/category_handlers/prompts/{shared,buckets,endpoints,categories}/, search_improvement_planning/category_handler_planning.md

### Intent
Stand up the step-3 category-handler module and finalize its shape
in the planning doc. All step-3 handler code (schema factories,
endpoint registry, handler runtime, prompt builder, prompt chunks)
now lives under `search_v2/stage_3/category_handlers/` rather than
split between `schemas/` and a future handler package.

### Key Decisions
- **Module lives under stage_3, not schemas/.** Handlers are
  execution code, not data contracts. `schemas/` stays reserved
  for contracts that cross the handler boundary (e.g.
  `EndpointParameters` + per-endpoint wrappers).
- **Moved both schema files into the new module.**
  `schemas/endpoint_registry.py` → `category_handlers/endpoint_registry.py`
  and `schemas/handler_outputs.py` → `category_handlers/schema_factories.py`
  (renamed to match the "factories" role). Import in schema_factories
  updated to reference the sibling. Smoke-tested: all 31 OUTPUT_SCHEMAS
  still build at import.
- **No separate category registry module.** `CategoryName.bucket` and
  `CategoryName.endpoints` already serve that role on the enum —
  overturning the "Category registry location and format" item in
  the planning doc's deferred-items list.
- **Handler scoped to a single category.** `handler.py` never fans
  out across `coverage_evidence` entries; that lives one layer up
  in the orchestrator. Simpler contract, easier to test.
- **Dispatcher vs runtime split rejected.** Earlier sketch had
  both; in practice it's one "run one handler" function. Fan-out
  is an orchestrator concern, not a stage-3 one.
- **Prompt chunk organization by keying axis, not by endpoint.**
  Four axes from the planning doc (shared / bucket / endpoint /
  category) become four subfolders under `prompts/`. Few-shot
  examples are per-category only (user correction — not per-bucket).
- **HandlerResult as a dataclass, not Pydantic.** Internal container
  with mutable defaults. No serialization at this boundary.

### Planning doc additions
- New "Finalized section table" subsection under "Prompt assembly"
  enumerating the 8 prompt sections and their keying axes.
- New "Category-handlers module layout (finalized)" subsection
  under "Pre-implementation setup"; replaces the stale "Dispatcher
  location" and "Deferred to implementation time" bullets that
  were resolved in this session.

### Testing Notes
- Smoke-tested: `OUTPUT_SCHEMAS` builds all 31 schemas after the
  move; `HandlerResult()` instantiates with empty defaults.
- `handler.py` and `prompt_builder.py` are intentional stubs — no
  behavior yet.
- `prompts/` subdirectories are empty; authoring content is the
  next task. Git won't track empty dirs, so they re-emerge when
  their first files are added.

## Category-handler output schemas: dynamic per-category assembly
Files: schemas/enums.py, schemas/endpoint_registry.py, schemas/handler_outputs.py

### Intent
Wire up the dynamic output-schema machinery the step-3 category
handlers will use. Step 3 emits structured output whose shape is
bucket-level (single / mutex / tiered / combo) and whose endpoint-
specific atoms come from the seven existing `EndpointParameters`
wrappers. This commit delivers the building blocks: a `HandlerBucket`
enum, a `bucket` attribute on every CategoryName, an EndpointRoute→
wrapper registry, and a handler_outputs module that eagerly builds
one Pydantic output class per category at import.

### Key Decisions
- **`HandlerBucket` as a separate enum** (not methods on an existing
  enum). Keeps the bucket vocabulary addressable independently of
  CategoryName; factories dispatch on bucket identity, not category
  count.
- **`bucket` baked onto CategoryName** alongside `endpoints`.
  Assignments pulled from
  `search_improvement_planning/category_handler_planning.md` §"The
  four handler types" (Cats 1–31); INTERPRETATION_REQUIRED (not in
  planning doc) defaulted to SINGLE as the fallback category has a
  single SEMANTIC endpoint.
- **Registry in a sibling module, not on EndpointRoute**. Wrapper
  modules already import from `schemas.enums`; attaching wrappers
  to the enum would create a cycle. `schemas/endpoint_registry.py`
  owns the import fanout.
- **TRENDING → None** in the registry. Trending is handled by a
  deterministic code path, not a handler LLM. Categories whose
  endpoint list resolves to no wrappers get no entry in
  OUTPUT_SCHEMAS; `get_output_schema(CategoryName.TRENDING)` raises
  KeyError by design — callers special-case upstream.
- **Eager build at module import** instead of `@functools.cache`.
  32 known keys × cheap `create_model` calls = ~1s once, errors
  surface at startup, no cold-request penalty. Factory functions
  are pure; the cache dict lives in the module.
- **No method on the enum for schema lookup**. The earlier
  consideration of `CategoryName.output_schema` as a property
  required a lazy import to dodge circular deps. Went with plain
  `handler_outputs.get_output_schema(category)` — same ergonomics,
  zero import gymnastics.
- **Per-bucket output shape follows planning doc verbatim**:
  - Single: `requirement_aspects` (with relation_to_endpoint +
    coverage_gaps), `should_run_endpoint`, `endpoint_parameters?`
  - Mutex: `requirement_aspects` (with per-endpoint coverage +
    `best_endpoint` Literal), `endpoint_to_run` Literal over
    endpoints + "None", `endpoint_parameters?` as Union of wrappers
  - Tiered: same as Mutex + `performance_vs_bias_analysis` placed
    between aspects and the final pick, so the model reasons about
    the bias before committing
  - Combo: `requirement_aspects`, `overall_endpoint_fits`,
    `per_endpoint_breakdown` with *named fields per endpoint*
    (enumerated-not-freeform to prevent silent omission). Each
    breakdown entry has `should_run_endpoint` + optional wrapper.
- **Shared Field descriptions as module constants** in
  handler_outputs.py. Prevents wording drift across four bucket
  factories. Calibrated for small / instruction-tuned models:
  phrasal cues, anti-failure-mode framing ("false is a valid and
  preferred answer when ..."), concrete direction over abstract
  framing. Follows the same pattern established in
  schemas/semantic_translation.py's per-space descriptor constants.
- **`_HandlerOutputBase` carries `ConfigDict(extra="forbid")`** for
  every dynamically-generated class. OpenAI structured output
  requires `additionalProperties: false` on every sub-object.
- **`__module__` set to "schemas.handler_outputs"** on every
  generated class. Keeps tracebacks and repr pointing somewhere
  useful rather than pydantic internals.

### Testing Notes
- All 31 non-TRENDING categories build clean schemas at import.
- `model_json_schema()` generates for all four bucket shapes
  (spot-checked: CREDIT_TITLE 6KB, TOP_LEVEL_GENRE 32KB,
  SPECIFIC_SUBJECT 33KB, TARGET_AUDIENCE 50KB — within OpenAI
  structured-output limits but worth smoke-testing the largest
  combos against the live API before production use).
- Combo's per_endpoint_breakdown field keys are endpoint values
  (keyword / metadata / semantic / ...), all valid Python
  identifiers. No sanitization needed today; revisit if any future
  EndpointRoute value contains hyphens or reserved words.
- `best_endpoint` Literal includes only the category's candidate
  endpoints (not "None"); `endpoint_to_run` Literal includes
  "None" as a valid no-fire outcome.
- No validator enforcing `should_run_endpoint == (endpoint_parameters
  is not None)` — relying on handler-side conversion. Add if drift
  is observed in practice.


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

## Category-handler schemas: EndpointParameters wrapper, HandlerResult, unified semantic schema
Files: schemas/enums.py, schemas/endpoint_parameters.py (new), schemas/semantic_translation.py (rewrite), schemas/keyword_translation.py, schemas/metadata_translation.py, schemas/award_translation.py, schemas/franchise_translation.py, schemas/studio_translation.py, schemas/entity_translation.py

### Intent
First implementation pass on the category-handler design finalized in `search_improvement_planning/category_handler_planning.md`. Introduces the shared schemas every downstream piece (handler LLM output, dispatcher routing, orchestrator consolidation) depends on. No handler code, factory, dispatcher, or `CategoryName` registry yet — those come in later passes.

### Key Decisions
- **`ActionRole` + `Polarity` enums added to `schemas/enums.py`.** Their 2×2 product mechanically selects the `HandlerResult` bucket (inclusion_candidates / exclusion_ids / preference_specs / downrank_candidates). Keeping them as two orthogonal axes — rather than a flat 4-value enum — matches how the handler LLM reasons about each finding.
- **`EndpointParameters` abstract base + concrete per-endpoint subclasses.** Base lives in `schemas/endpoint_parameters.py`; each per-endpoint subclass lives in its own `*_translation.py` and declares a typed `parameters` field. Concrete subclassing (not pydantic generics) to keep OpenAI structured-output JSON schemas flat and clean. Seven wrappers total: keyword / metadata / award / franchise / studio / entity / semantic. Trending deliberately skipped (no translation spec, dispatched deterministically).
- **Existing `*QuerySpec` classes left untouched.** Reasoning fields stay bundled for now per "keep, revisit once v2 handlers run" decision. Only stale comments about direction/exclusion being a "step 4 concern" were rewritten to reference the wrapper's `polarity` field.
- **`HandlerResult` defaults to all-empty buckets.** Lets the soft-fail retry path return `HandlerResult()` cleanly when a handler double-fails.
- **Semantic endpoint unified into one shape.** Replaced `SemanticDealbreakerSpec` (7 spaces, single pick, no weight) and `SemanticPreferenceSpec` (8 spaces, multi, weighted) with a single `SemanticParameters` + `SemanticEndpointParameters` wrapper. Anchor dropped from both paths; 7-space enum (`SemanticSpace`) replaces the old `DealbreakerSpace`/`VectorSpace` pair. `PreferenceSpaceWeight` renamed to `SpaceWeight` for clarity.
- **`primary_vector` placed LAST in `SemanticParameters`.** Deliberate field-order design: emitting it before `space_queries` pressures the LLM to collapse the list prematurely. Placing it after reframes the pick as a retrospective summary over an already-populated inventory. Guarded by a new `_primary_vector_in_space_queries` validator so the chosen space must match one populated above. Dealbreaker path uses `primary_vector` and ignores weight; preference path uses weights and ignores `primary_vector`. Symmetric shape, no runtime conditional.

### Intentional breakage (not fixed in this pass)
Deleting the old `SemanticDealbreakerSpec` / `SemanticPreferenceSpec` breaks imports in:
- `search_v2/stage_3/semantic_query_generation.py`
- `search_v2/stage_3/semantic_query_execution.py`
- `search_v2/test_stage_3.ipynb`

These will be rebuilt against `SemanticEndpointParameters` when the new category-handler flow replaces stage_3. Verified via `uv run python -c "import search_v2.stage_3.semantic_query_generation"` — expected ImportError fires.

### Testing Notes
Smoke-verified via `uv run python`:
- All new classes import cleanly from their modules.
- `HandlerResult()` default-constructs with four empty buckets.
- `SemanticEndpointParameters.model_json_schema()` generates without error.
- Valid `SemanticParameters` with two entries and matching `primary_vector` constructs.
- Invalid cases raise `ValidationError`: `primary_vector` not in populated spaces; duplicate space in `space_queries`.
No unit tests written per test-boundaries rule.

## CategoryName.endpoints attribute + Cat 32 Chronological doc entry
Files: schemas/enums.py, search_improvement_planning/query_categories.md

### Intent
`CategoryName` carried a human-readable `description` but no machine-readable dispatch metadata. Downstream category handlers and the step-3 dispatcher had no authoritative source for which `EndpointRoute`(s) a given category may fire or in what priority order. This pass attaches a `.endpoints` tuple to every `CategoryName` value so the taxonomy itself is the source of truth for routing priority.

### Key Decisions
- **`EndpointRoute` moved above `CategoryName`** in `schemas/enums.py`. Enum members reference `EndpointRoute` at class-body evaluation time, so the dependency had to be declared first. Old location (further down the file with the other step-3 enums) was deleted.
- **Tuple-constructor extended, not replaced.** `CategoryName.__new__` now takes three positional args `(value, description, endpoints)`; the `description: str` class annotation gains a sibling `endpoints: tuple["EndpointRoute", ...]`. Same tuple-constructor pattern already used for `NarrativeStructureTag` / `LineagePosition` and validated by the "Enum vocabularies for LLM structured output carry prompt-ready descriptions as enum attributes" convention in conventions_draft.md.
- **Priority ordering, not set.** User specified the list must be priority-ordered even though ordering only strictly matters for tiered categories — "good principle overall" per the request. Tiered cats lead with the canonical tier (KEYWORD before SEMANTIC for the keyword-first tiered family); combo cats lead with the primary channel (SEMANTIC before METADATA for RECEPTION_QUALITY / CURATED_CANON; KEYWORD before METADATA for CULTURAL_TRADITION; KEYWORD → METADATA → SEMANTIC for the gate+inclusion shapes TARGET_AUDIENCE / SENSITIVE_CONTENT).
- **Semantic sub-spaces consolidated.** The planning doc references ANC / P-EVT / P-ANA / NRT / VWX / CTX / RCP / PRD / INTERP as distinct semantic surfaces; all roll up to `EndpointRoute.SEMANTIC` in the enum per the current endpoint-family design (one semantic endpoint internally dispatches across vector spaces).
- **CHRONOLOGICAL → METADATA, INTERPRETATION_REQUIRED → SEMANTIC.** Chronological picks an ordinal on `movie_card.release_date` (pure metadata sort). Interpretation-required routes to the LLM semantic fallback per Cat 31 in the planning doc.

### Planning Context
The planning doc (`search_improvement_planning/query_categories.md`) listed Cats 1–31 but did not contain the CHRONOLOGICAL category that exists in the enum (added when step_2's selection-rule language type collapsed into attributes — see the earlier "Step 2 schema migration: collapse fragment types" entry above). User directed adding Cat 32 to the doc to keep the planning surface in sync with the enum. New "Ordinal selection" section numbered Cat 32 (not slotted between Cats 10 and 11) to preserve stable numbering for the existing 30 categories; justification note in the doc explains the numbering choice. Updated the "single endpoint" orchestration list and the data-gap table to reference Cat 32.

### Testing Notes
Verified via `uv run python -c "from schemas.enums import CategoryName, EndpointRoute; print(len(CategoryName)); [print(c.name, '->', [e.value for e in c.endpoints]) for c in CategoryName]"` — 32 members load, every one carries a non-empty endpoints list matching the planning-doc mapping. No unit tests written per test-boundaries rule. Downstream category handlers can now iterate `category.endpoints` for dispatch priority instead of maintaining a parallel routing map.

## Semantic executor: align with unified SemanticParameters + action_role dispatch
Files: search_v2/stage_3/semantic_query_execution.py, search_v2/stage_4/dispatch.py

### Intent
`schemas/semantic_translation.py` was rewritten to collapse `SemanticDealbreakerSpec` + `SemanticPreferenceSpec` into a single `SemanticParameters` model (with `space_queries` + retrospective `primary_vector`), rename `PreferenceSpaceWeight` → `SpaceWeight`, and drop the anchor space entirely. The executor still spoke the old dual-schema / two-entry-point shape, so every import and every `spec.body.space.value` / `spec.body.content` access broke. This pass re-plumbs the executor around the unified schema and the new category-handler dispatch signal.

### Key Decisions
- **Single public entry point, not two.** `execute_semantic_query(params, *, action_role, restrict_to_movie_ids, qdrant_client)` replaces the former dealbreaker/preference pair. The caller must pass `action_role` (from the enclosing `SemanticEndpointParameters` wrapper); that selects the dealbreaker vs preference branch inside the executor, while `restrict_to_movie_ids` still picks D1 vs D2 and P1 vs P2 within each branch. Matches the "symmetric shape, no runtime schema conditional" directive in category_handler_planning.md §"Unified semantic schema".
- **`_entry_for_primary_vector` helper.** Pulled out so the dealbreaker helpers don't re-implement the linear scan. Raises loudly on a primary_vector-without-matching-entry rather than silently picking entry[0] — the schema validator already guarantees correctness, so a raise only fires on validator bypass, which is worth surfacing.
- **Polarity is NOT consulted in the executor.** Per the planning doc's "From LLM output to return buckets" table, polarity routes at the orchestrator level. The executor returns the same `EndpointResult` shape whether the finding is positive or negative; existing D1/D2/P1/P2 scoring semantics are unchanged.
- **AnchorBody removed from the `SemanticBody` union** since `SemanticSpace` has 7 non-anchor members — no entry can carry an `AnchorBody` under the new schema.
- **Stage-4 boundary maps role strings → ActionRole.** `TaggedItem.role` is still the pre-category-handler Literal["inclusion_dealbreaker", "exclusion_dealbreaker", "preference"]. Mapped locally in `dispatch.py` (`"preference" → CANDIDATE_RERANKING`, else `CANDIDATE_IDENTIFICATION`) rather than touching `TaggedItem` — stage 4 has its own in-progress refactor per the planning doc and I don't want to merge concerns.
- **action_role validation hoisted out of the retry loop.** The initial cut put the `else: raise ValueError(...)` inside the `try/except Exception` block, which laundered a contract violation into a silent empty-result soft-fail with misleading "retrying" logs. Fixed during code review: validate action_role before the retry loop so non-retryable exceptions surface, in line with conventions.md §"Preserve retryable exception types".
- **Per-branch log context.** The retry/error log lines used to include `primary_vector=` *and* `spaces=` regardless of which branch ran — misleading for the reranking path, which explicitly ignores primary_vector. Now a `log_context` string is built once per branch (identification → just primary_vector; reranking → just spaces) and reused by both the warning and error logs.

### Planning Context
Follows the "Unified semantic schema" section of `search_improvement_planning/category_handler_planning.md` (lines 910-968): one Pydantic model, anchor excluded in both paths, action_role drives which fields the executor consults (dealbreaker → primary_vector entry only; reranking → all entries with weights). The executor is intentionally still direction-agnostic — polarity becomes the orchestrator's job per the "From LLM output to return buckets" 2×2.

### Testing Notes
Module imports clean (`python -c "from search_v2.stage_3 import semantic_query_execution"`). `search_v2/stage_3/semantic_query_generation.py` is still broken against the new schema (imports `SemanticDealbreakerSpec` / `SemanticPreferenceSpec`) — that file needs its own rewrite aligned with the category-handler prompt-assembly design and is outside this change's scope. D1/D2/P1/P2 scoring math is byte-identical to before — the change is purely how the executor reads the input, not how it produces the output.

## Reorder EndpointParameters fields to action_role → parameters → polarity
Files: schemas/endpoint_parameters.py, schemas/entity_translation.py, schemas/keyword_translation.py, schemas/metadata_translation.py, schemas/semantic_translation.py, schemas/franchise_translation.py, schemas/award_translation.py, schemas/studio_translation.py

### Intent
Change the LLM-facing field order on every `EndpointParameters` subclass from `action_role → polarity → parameters` to `action_role → parameters → polarity`. Generating `polarity=negative` immediately before the `parameters` block was creating a plausible double-negative failure: small instruction-tuned models can pattern-match on the nearby "negative" token and invert the parameter content (e.g. "not Tom Hanks" instead of "Tom Hanks" with a negative polarity routing tag). Moving `polarity` to last reframes it as a retrospective routing judgment over already-populated parameters — same pattern the planning doc already uses for `primary_vector` inside `SemanticParameters`.

### Key Decisions
- **Pattern 1 (static redeclaration) over Pattern 2 (factory-built wrappers).** The user explicitly rejected a factory-based approach after weighing tradeoffs. Pattern 2 would have coupled class identity to a singleton-cache invariant that the orchestrator's `isinstance`-based preference routing depends on, and degraded IDE/static-analysis support. Pattern 1 costs a handful of redeclaration lines per subclass but keeps every wrapper statically inspectable.
- **Pydantic v2 preserves base-class field positions under inheritance.** Initial implementation redeclared `action_role` and `polarity` on every subclass expecting this to move them to subclass declaration order — it does not; Pydantic appends new subclass fields after inherited ones regardless of redeclaration. Fix: strip `action_role` and `polarity` off the `EndpointParameters` base entirely. The base is now a pure marker (still load-bearing for the orchestrator's `isinstance` check on `HandlerResult.preference_specs`), and each subclass declares all three fields fresh in canonical order.
- **Guardrail via `__pydantic_init_subclass__`.** Because the base no longer declares the three fields, a future endpoint could silently ship without them or in the wrong order and the JSON schema would not reveal the mistake until runtime. Added a subclass-construction hook that enforces both field presence and relative order (`action_role` before `parameters` before `polarity`). Not `__init_subclass__` — Pydantic populates `model_fields` after that hook runs, so the check has to use the Pydantic-specific hook.
- **Shared description constants in `endpoint_parameters.py`.** `ACTION_ROLE_DESCRIPTION` and `POLARITY_DESCRIPTION` live at module scope so all seven wrappers import identical wording. Also rewrote the polarity description to include explicit routing-tag framing: "This field is a routing signal for the orchestrator, not a content modifier: `parameters` always describes the target concept directly…" plus a worked example ("no 'not Tom Hanks', no anti-keyword lists, no inverted searches"). Belt-and-suspenders for the ordering fix: if the ordering argument doesn't fully eliminate the risk, the description itself now reinforces it.
- **Appended a "describe target directly" sentence to every subclass's `parameters` description.** Same risk, opposite direction: the `parameters` field's own description now tells the LLM not to encode negation inside it. Applied consistently across all seven endpoint wrappers.

### Planning Context
Conversation-driven refinement of the "EndpointParameters base class" section of `search_improvement_planning/category_handler_planning.md`. The planning doc enumerates the three fields but does not prescribe their order; this change establishes `action_role → parameters → polarity` as the canonical order with the double-negative rationale captured in `endpoint_parameters.py` comments.

### Testing Notes
Smoke-tested via `model.model_json_schema()["properties"]` for all seven wrappers (EntityEndpointParameters, KeywordEndpointParameters, MetadataEndpointParameters, SemanticEndpointParameters, FranchiseEndpointParameters, AwardEndpointParameters, StudioEndpointParameters) — all emit properties in the order `action_role, parameters, polarity`. Guardrail rejects both missing-field and wrong-order subclasses (verified with two synthetic broken subclasses). `get_output_schema(category)` in `search_v2/stage_3/category_handlers/schema_factories.py` still composes cleanly across all four buckets (single / mutex / tiered / combo). Existing `unit_tests/test_handler_output_schemas.py` was added this session for this contract but left untouched per the test-boundaries rule.

## Rename ActionRole → MatchMode (values: filter | trait); rewrite descriptions in LLM-grounded language
Files: schemas/enums.py, schemas/endpoint_parameters.py, schemas/entity_translation.py, schemas/keyword_translation.py, schemas/metadata_translation.py, schemas/semantic_translation.py, schemas/franchise_translation.py, schemas/award_translation.py, schemas/studio_translation.py, search_v2/stage_3/semantic_query_execution.py, search_v2/stage_3/category_handlers/schema_factories.py, search_v2/stage_3/category_handlers/handler_result.py, search_v2/stage_3/category_handlers/prompts/shared/shared_vocabulary.md, search_v2/stage_3/category_handlers/prompts/endpoints/semantic.md, search_v2/stage_4/dispatch.py

### Intent
Two separate but related concerns, done in one pass:

1. **Rename the hard/soft classification enum and field** to terms the LLM can ground from the handler prompt alone. Prior names (`ActionRole`, `candidate_identification`, `candidate_reranking`) referenced system concepts — "candidate pool", "reranking" — that appear nowhere in the handler's visible context. The LLM either inferred meanings inconsistently or treated the tokens as inert, wasting attention.
2. **Rewrite the `match_mode` and `polarity` field descriptions** to drop system-architecture vocabulary entirely. The handler prompt surfaces no knowledge of the orchestrator, candidate pools, inclusion_candidates, or subtract/downrank mechanics — so those terms were actively misleading for an LLM deciding between two enum values.

### Key Decisions
- **`match_mode: filter | trait` over alternatives.** Conversation iterated through `action_role: candidate_identification / candidate_reranking` (rejected — system-speak), `strictness: requirement / preference` (rejected — "requirement" collides with `requirement_aspects` reasoning field), `constraint / preference` (rejected — doesn't convey the binary-vs-descriptive asymmetry), before landing on `filter / trait`. `filter` names the binary yes/no pool-reduction role; `trait` names the descriptive attribute role. Both combine freely with either polarity (filter+positive includes; filter+negative excludes; trait+positive boosts; trait+negative demotes) — the rename explicitly removes the "filter means remove" misreading risk.
- **Description rewrite principle: only use concepts the LLM can ground from the prompt it sees.** No "orchestrator", "candidate pool", "inclusion_candidates", "downrank", "already-decided pool", etc. Descriptions now speak in user-intent terms ("drops out entirely" / "still qualifies, just with a lower score") with concrete example quadrants ("must star Tom Hanks is filter+positive; no horror is filter+negative; preferably funny is trait+positive; not too violent is trait+negative") so the LLM has worked examples to pattern-match against.
- **Polarity description keeps the no-double-negation paragraph but in LLM-grounded language.** Stripped "routing signal for the orchestrator" → replaced with direct instruction: "parameters always describes the target concept directly, regardless of polarity. When polarity is 'negative', do NOT negate the parameters." Worked example remains ("for 'no Tom Hanks', parameters describes Tom Hanks and polarity is negative").
- **The separate `ActionRole` in `schemas/query_understanding.py` is untouched.** That enum (`INCLUSION / EXCLUSION / PREFERENCE`) is a different axis used by Stage 2B and survived under the same name. Distinct module, distinct semantics — not swept by this rename. Stage 4's `flow_detection.py` continues to import it.
- **Shared vocabulary prompt updated; `input_spec.md` untouched.** `input_spec.md` never referenced the old enum, so no change. `shared_vocabulary.md` was rewritten to define `match_mode` with the filter/trait framing and an explicit orthogonality note ("filter does not mean remove"). `semantic.md` updated to replace "candidate-identification" / "candidate-reranking" phrasings with "filter-mode" / "trait-mode".
- **Stage-4 `dispatch.py` rename.** The semantic executor's kwarg renamed from `action_role` → `match_mode` (no external callers yet beyond dispatch). Mapping at the stage-4 boundary now maps `item.role == "preference"` → `MatchMode.TRAIT`, else `MatchMode.FILTER`.

### Planning Context
Extends the "Unified semantic schema" / "EndpointParameters base class" sections of `search_improvement_planning/category_handler_planning.md`. The planning doc's discussion of `action_role` / `candidate_identification` / `candidate_reranking` is now stale vocabulary — the code and prompts have moved on. Not rewriting the planning doc in this pass (it's a decision record, not a spec); future planning-doc edits will adopt the new vocabulary as they land.

### Testing Notes
End-to-end verification: all 7 EndpointParameters subclasses still emit JSON-schema properties in the order `match_mode, parameters, polarity`, with enum values correctly `['filter', 'trait']`. Guardrail still fires for missing-field and wrong-order subclasses. `get_output_schema(category)` still composes cleanly across sampled categories in all four buckets. No remaining `ActionRole` / `action_role` / `candidate_identification` / `candidate_reranking` references in any touched code file. Pre-existing breakage in `search_v2/stage_3/semantic_query_generation.py` (still references removed `SemanticDealbreakerSpec` / `SemanticPreferenceSpec` from the earlier unified-schema work) is unchanged and out of scope. `unit_tests/test_handler_output_schemas.py` still has a comment-only reference to "action_role" — left untouched per the test-boundaries rule; will update when test rewrite is scheduled.

## Authored shared / bucket / endpoint prompt chunks for category handlers
Files: search_v2/stage_3/category_handlers/prompts/shared/{role,shared_vocabulary,input_spec}.md, search_v2/stage_3/category_handlers/prompts/buckets/{single,mutex,tiered,combo}_{objective,guardrails}.md, search_v2/stage_3/category_handlers/prompts/endpoints/{keyword,metadata,semantic,award,franchise,studio,entity}.md, search_improvement_planning/category_handler_planning.md

### Intent
First substantive authoring pass for the category-handler prompt chunks. Fills 3 of the 8 plug-in pieces (shared role/vocab/input-spec) plus all 8 bucket-keyed pieces (core objective + failure-mode guardrails for single/mutex/tiered/combo) plus all 7 per-endpoint context chunks. Category-specific `notes.md` + `examples.md` remain to be authored per category.

### Key Decisions
- **Endpoint chunk content split.** Everything that was task framing, input description, positive-presence invariant, output-field schema-order scaffolding, or worked I/O examples was pruned from the old `stage_3/*_query_generation.py` system prompts — those responsibilities now live in role.md / input_spec.md / `action_role`+`polarity` Field descriptions / bucket objectives / per-category `examples.md`. What's preserved in each endpoint chunk is endpoint-specific semantic knowledge: vocabulary (closed enums, taxonomies), capability boundaries (what this endpoint can/can't express), parameter-semantic rules (how user phrasings resolve to literal parameter values), canonicalization / tokenizer rules, disambiguation principles for near-collision options.
- **Dynamic registry content via `{{PLACEHOLDER}}` markers.** Classification registry (keyword), tracked/free streaming services (metadata), ceremony mappings / award name surface forms / category tag taxonomy (award), brand registry (studio) stay single-sourced in the existing `render_*_for_prompt()` helpers. Endpoint markdown files use `{{CLASSIFICATION_REGISTRY}}` / `{{CEREMONY_MAPPINGS}}` / etc. — `prompt_builder.py` will substitute at build time. Avoids duplicating vocabulary between the old SYSTEM_PROMPT modules and the new chunks.
- **Semantic chunk unified (no anchor).** The new `SemanticParameters` enum has 7 non-anchor spaces; `semantic.md` describes those 7 only. Both `match_mode=filter` (dealbreaker-style) and `match_mode=trait` (preference-style) share the same schema, so the chunk documents space selection, body-authoring principles, decomposition discipline, and `central`/`supporting` weight semantics once — not twice.
- **Bucket-specific guardrails over generic.** Each of the 4 bucket failure-mode chunks covers 3 shared failure modes (ambiguous / out-of-scope / self-contradictory) plus bucket-specific pitfalls (tiered: bias tiebreaker discipline; combo: don't-skip-candidates + don't-fire-everything-just-because). The "correct no-fire emission" is named with the bucket's exact schema shape (`should_run_endpoint: false` vs `endpoint_to_run: "None"` vs per-endpoint breakdown with all false).
- **No "don't reach for anchor" framing in semantic.md.** Early draft had a line warning the LLM off the anchor space; removed after user pointed out anchor isn't in the enum or input context — mentioning it only surfaces a concept the LLM wouldn't otherwise consider. Corollary to the existing "grounded vocabulary" rule: even concepts the LLM CAN ground shouldn't appear if they're outside its actual decision surface.
- **Planning doc updates.** Added an "Input spec" chunk to the finalized section table (slot 4, between endpoint context and core objective), new "Input spec stays descriptive, not prescriptive" subsection explaining why decision rules live on Field descriptions and not in the spec. Chunk-ordering rationale updated.

### Planning Context
Follows the eight-chunk composition laid out in `search_improvement_planning/category_handler_planning.md` §"Full system-prompt composition". Shared + bucket + endpoint authored first because they depend only on the category registry on `CategoryName`; per-category `notes.md` and `examples.md` will be authored one category at a time as handlers start running.

### Cross-verification against EndpointParameters subclasses
Spot-checked each endpoint chunk against its `*EndpointParameters.parameters` subclass (keyword / metadata / semantic / award / franchise / studio / entity): every enum value, attribute, sub-object, and axis mentioned in the chunk exists on the schema; nothing extra is described that the schema doesn't accept. Verified `SemanticSpace` has exactly 7 non-anchor members after the earlier unification. MatchMode / Polarity terminology (filter / trait / positive / negative) used consistently.

### Testing Notes
No runtime wiring — `prompt_builder.py` is still a stub. Chunks are authored content only; the build-time substitution of `{{PLACEHOLDER}}` markers for registry content is next in that module.

## Cat 20 (Plot events + narrative setting) handler prompt chunks
Files: search_v2/stage_3/category_handlers/prompts/categories/additional_objective_notes/cat_20_plot_events.md, search_v2/stage_3/category_handlers/prompts/categories/few_shot_examples/cat_20_plot_events.md | Authored the per-category notes chunk and 5 few-shot examples for the Single/Semantic Cat 20 handler. Notes anchor the category on `plot_events` as the only space that carries raw synopsis prose at ingest, with `primary_vector: "plot_events"` fixed; draws boundaries against Cat 13 (filming vs. narrative location), Cat 10 (release era vs. story era), Cat 21 (themes vs. events), Cat 15 (pattern labels vs. concrete events), and Cat 22 (tone vs. events). Examples cover: concrete plot-event fire (heist/betrayal), narrative time+place fire (1940s Berlin), narrative-place-only fire (Tokyo), plus two no-fires (filmed-in-Tokyo → Cat 13; thematic-grief → Cat 21).

## Cat 24 (Craft acclaim) handler prompt chunks
Files: search_v2/stage_3/category_handlers/prompts/categories/additional_objective_notes/cat_24_craft_acclaim.md, search_v2/stage_3/category_handlers/prompts/categories/few_shot_examples/cat_24_craft_acclaim.md | Authored the per-category notes chunk and 5 few-shot examples for the Single/Semantic Cat 24 handler. Notes anchor reception.praised_qualities as always in scope with `primary_vector: "reception"` fixed, and describe when production (visual/technical craft) or narrative_techniques (dialogue/writing craft) spaces co-fire per axis. Boundaries drawn against Cat 1 (named director), Cat 29 (named below-the-line creator), Cat 25 (axis-less acclaim → metadata), and Cat 22 (viewer experience). Examples cover: visual acclaim (reception + production), musical acclaim (reception only), dialogue acclaim (reception + narrative_techniques), no-fire on named cinematographer (Deakins → Cat 29), no-fire on named director (Nolan → Cat 1).

## Cat 21 (Kind of story / thematic archetype) handler prompt chunks
Files: search_v2/stage_3/category_handlers/prompts/categories/additional_objective_notes/cat_21_kind_of_story.md, search_v2/stage_3/category_handlers/prompts/categories/few_shot_examples/cat_21_kind_of_story.md | Authored the per-category notes chunk and 6 few-shot examples for the Tiered Keyword/Semantic Cat 21 handler. Notes lead with the spectrum-escape mechanic unique to this category — gradient framings ("kind of", "leans", "touches of") bypass Keyword even when a registry member would match, because binary posting-list membership cannot rank graded intent. Boundaries drawn against Cat 7 (static type vs arc trajectory), Cat 15 (recognized story-pattern label vs abstract theme), and Cat 17 (audience framing vs story-pattern framing). Examples: clean keyword win on COMING_OF_AGE (binary coming-of-age), clean keyword win on SURVIVAL (binary man-vs-nature), spectrum escape on "kind of about grief" (semantic direct), spectrum escape on "leans redemptive" (semantic direct despite hypothetical tag), lower-tier win on forgiveness (no tag exists, binary → semantic), no-fire on "deep themes". Chose COMING_OF_AGE and SURVIVAL for the keyword-win examples because the registry has no REDEMPTION or FOUND_FAMILY tags — adjusted per the spec's "if the tag exists" caveat.

## Prompt builder + chunk-file rename
Files: search_v2/stage_3/category_handlers/prompt_builder.py, search_v2/stage_3/category_handlers/prompts/endpoints/{award.md→awards.md,franchise.md→franchise_structure.md}, search_v2/stage_3/category_handlers/prompts/categories/additional_objective_notes/cat_*_*.md → <category_lower>.md (31 files), search_v2/stage_3/category_handlers/prompts/categories/few_shot_examples/cat_*_*.md → <category_lower>.md (31 files), search_v2/stage_3/category_handlers/prompts/shared/input_spec.md, search_improvement_planning/category_handler_planning.md

### Intent
Implement the handler prompt builder and lock in file-naming
conventions so the eight chunks can be looked up mechanically from
`CategoryName.name.lower()` / `EndpointRoute.value` with no
intermediate mapping table.

### Key Decisions
- **Two entry points: `build_system_prompt(category)` and
  `build_user_message(raw_query, overall_query_intention_exploration,
  target_entry, parent_fragment, sibling_fragments)`.** System prompt
  is a pure function of the category (enabling per-category caching
  by the caller across multiple coverage_evidence atoms in one
  query); user message is a pure function of the per-call payload.
- **Endpoint prompt files renamed to match `EndpointRoute.value`
  exactly.** `award.md → awards.md` and `franchise.md →
  franchise_structure.md`. `f"endpoints/{route.value}.md"` now works
  uniformly.
- **Per-category files renamed to `<CategoryName.name.lower()>.md`.**
  Dropped the `cat_NN_` prefix so the lookup is `category.name.lower()`
  with no NN→enum mapping. Used enum `.name` (UPPER_SNAKE) rather
  than `.value` because values contain spaces/slashes that aren't
  filesystem-safe. 31 files renamed in each of
  `additional_objective_notes/` and `few_shot_examples/`.
- **Import-time caching of shared / bucket / endpoint chunks;
  call-time fallback for per-category chunks.** Missing shared /
  bucket / endpoint files fail loudly at import (these are
  invariants of the stack). Missing per-category files surface at
  `build_system_prompt()` call time with a clear `FileNotFoundError`
  naming the expected path — keeps import from breaking while
  categories are being authored. All 31 non-TRENDING categories
  built cleanly in the smoke test.
- **`build_system_prompt(CategoryName.TRENDING)` raises `ValueError`.**
  TRENDING has no LLM codepath; the explicit raise catches dispatch
  mistakes that would otherwise funnel through the wrong branch.
  Mirrors `schema_factories.get_output_schema`'s KeyError-for-TRENDING
  convention.
- **XML serialization uses fully-tagged nested modifier elements
  (Option A).** Chosen over attributes or flat prose because small
  models parse element text more reliably, escaping is uniform
  across all leaves, and the shape extends cleanly if `Modifier`
  gains fields.
- **All user-derived leaf text escaped via `xml.sax.saxutils.escape`.**
  CDATA explicitly avoided — structured-output models parse escaped
  text more reliably. Empty `<modifiers>` / `<sibling_fragments>`
  render as explicit empty tags so the slot is visible to the LLM.
- **Enum leaves emit `.value`.** `fit_quality` → `clean` / `partial`
  (matches shared vocabulary literals); `type` on modifiers →
  `role_marker` / `polarity_modifier`. Updated
  `prompts/shared/input_spec.md` to use lowercase modifier-type
  literals (`polarity_modifier` / `role_marker`) so the spec text
  matches what the builder actually emits — the alternative (switch
  to `.name` for LanguageType only) would have introduced a one-off
  rule inconsistent with the `fit_quality` decision.

### Planning doc updates
- Rewrote the category-handlers module-layout tree to reflect the
  flat two-subdir-per-chunk-type structure on disk (the earlier
  `cat_NN_<slug>/notes.md + examples.md` nested layout was never
  implemented).
- Documented the filename-derivation convention
  (`CategoryName.name.lower()` for categories; `EndpointRoute.value`
  for endpoints) with explicit rationale for using enum `.name` over
  `.value`.
- Added the modifier XML example and the text-escape rule to
  §"Input serialization — XML tags".
- New §"Prompt builder behavior" capturing the two-function
  interface, import-time vs call-time loading, and TRENDING-raises.

### Testing Notes
- Import-time smoke: `import prompt_builder` succeeds; all shared /
  bucket / endpoint chunks load; 31 non-TRENDING per-category chunks
  load.
- `build_system_prompt` tested for every category: 31/31 succeed,
  TRENDING raises `ValueError` as designed. Largest assembled prompt
  (SPECIFIC_SUBJECT, tiered) is ~46K chars — within provider limits
  and comparable to the old per-endpoint SYSTEM_PROMPT sizes.
- `build_user_message` smoke with `&`/`<`/`>` in user text confirms
  escape pipeline; nested modifier XML renders in the expected
  shape; empty siblings modifiers render as `<modifiers></modifiers>`.
- No unit tests written per test-boundaries rule.

## prompt_builder code-review follow-ups
Files: search_v2/stage_3/category_handlers/prompt_builder.py
- Memoized `build_system_prompt` with `@functools.cache` — output is deterministic per CategoryName, removes the per-caller caching burden the planning doc punted.
- Sibling `<fragment>` children now indent 2 spaces inside `<sibling_fragments>` to match the rest of the XML (parent_fragment, target_entry, modifiers). `_serialize_fragment` takes an `indent` kwarg; parent uses `""`, siblings use `"  "`.
- `build_system_prompt` now raises `ValueError` if the category's endpoint set resolves to zero LLM-wrapper endpoints. Impossible today (TRENDING raises first) but future-proofs against a later category that might slip past the dispatch guard.
- `build_user_message` asserts `parent_fragment.coverage_evidence == []` and the same on every sibling. Makes dispatch bugs that forget to strip coverage_evidence a loud failure instead of a silently-bloated payload.

### Testing Notes
- Verified via smoke test: cache returns identical objects across repeat calls (`CacheInfo(hits=1, misses=1)`); empty-endpoint raise fires with a clear message naming the category + declared routes; coverage_evidence asserts fire for both parent and sibling misuse.
- unit_tests/test_prompt_builder_system_prompt.py was authored in the same session and still passes the build surface (it only touches `build_system_prompt`, which is unchanged apart from the cache and the new empty-endpoint raise — both transparent to the 31 live categories).

## prompt_builder: drop coverage_evidence-stripping precondition on build_user_message
Files: search_v2/stage_3/category_handlers/prompt_builder.py | Removed the assertion that parent_fragment / sibling_fragments arrive with coverage_evidence=[]. The serializer only reads query_text, description, and modifiers, so forcing callers to strip the field served no purpose beyond making dispatch code awkward. Callers can now pass RequirementFragment objects straight through from Step 2; coverage_evidence is simply ignored. Planning doc §"Handler input data" description of "RequirementFragment without coverage_evidence" still holds — it describes what the handler *sees*, not a precondition on the caller.

## Implicit expectations step scaffold
Files: schemas/implicit_expectations.py, search_v2/implicit_expectations.py

### Intent
Add a new post-Step-2 LLM step that classifies every Step-2
requirement fragment as explicit quality / explicit notability /
other, then derives whether any implicit quality/notability gap
remains for reranking to backfill.

### Key Decisions
- **Kept the Step 0-2 split.** Schema definitions live in
  `schemas/implicit_expectations.py`; execution, prompt text, XML
  user payload building, and CLI live in
  `search_v2/implicit_expectations.py`.
- **Derived booleans in code, not in the LLM output.**
  `implicitly_expects_quality` / `implicitly_expects_notability` are
  computed deterministically from the emitted signal list.
- **One row per Step-2 requirement is enforced at runtime.** The
  executor validates both row count and exact
  `query_span == requirement.query_text` ordering so the step cannot
  silently merge or drop fragments.
- **Nullable strength stays narrow but load-bearing.**
  `strength_of_implicit_expectations` is `strong` / `suppressed` /
  `null`; `null` is forced when both explicit quality and explicit
  notability are present. Result-model validation also rejects
  `null` when an implicit gap still remains.
- **Schema descriptions carry the output guidance.** Following the
  newer Step 3 pattern, the LLM-facing explanation for output fields
  lives in `Field(description=...)` strings rather than the system
  prompt.
- **XML payload mirrors the Stage 3 prompt-builder style.** The
  module emits escaped nested tags for the raw query, overall Step-2
  intention, modifiers, and full coverage-evidence entries so the
  Gemini model receives the full fragment structure in a stable
  shape.

### Testing Notes
- Ran `python3 -m py_compile schemas/implicit_expectations.py
  search_v2/implicit_expectations.py` successfully.
- No pytest or live query runs performed per test-boundaries rule.

## Implicit expectations review follow-up: remove strength field
Files: schemas/implicit_expectations.py, search_v2/implicit_expectations.py | Simplified the step after review: dropped `strength_of_implicit_expectations` from both the LLM output schema and the derived result model, removed the prompt section that asked the small model for a global suppression judgment, and kept the step focused on per-fragment explicit quality/notability classification only. Also tightened the quality examples so ambiguous "prestige" language no longer counts as explicit quality by default. Re-ran `python3 -m py_compile` on both files successfully.

## Notebook harness for implicit expectations step
Files: search_v2/test_search_v2.ipynb | Added a new bottom markdown/code cell pair that runs `run_step_2(query)`, feeds the result into `run_implicit_expectations(query, step_2_response)`, and prints the derived booleans, per-fragment explicit signals, token counts, and full JSON. Validated the notebook remains parseable by loading the `.ipynb` as JSON with Python.

## Category handler runtime driver (handler.py)
Files: search_v2/stage_3/category_handlers/handler.py

### Intent
Fill in the runtime driver that runs one category handler on one
coverage_evidence entry. Closes out the step-3 category-handler stack
— prompt_builder, schema_factories, endpoint_registry, and per-endpoint
EndpointParameters wrappers were already in place.

### Key Decisions
- **TRENDING short-circuits before the no_fit check.** Per user
  decision — trending is deterministic and the early branch keeps the
  LLM codepath clear of any trending-specific affordances. no_fit
  short-circuit sits second as defense-in-depth (dispatch filters
  these in practice).
- **Route extraction keyed on bucket shape, not a wrapper→route map.**
  SINGLE reads `category.endpoints[0]`; MUTEX/TIERED reads
  `output.endpoint_to_run`; COMBO iterates `per_endpoint_breakdown`'s
  field names (which are route.value strings by construction in
  schema_factories). No new registry module needed.
- **Semantic FILTER+NEGATIVE routes to downrank_candidates, not
  exclusion_ids.** Override of the base 2×2 in the planning doc —
  semantic similarity is soft, so a "not scary" match is a gradient
  downrank rather than a hard set removal. Implemented as an
  isinstance check on SemanticEndpointParameters in _classify_wrapper.
- **Additive score consolidation within inclusion_candidates /
  downrank_candidates.** Multiple fired endpoints (COMBO) can surface
  the same tmdb_id; the handler sums their scores so the orchestrator
  sees a single consolidated contribution per ID. exclusion_ids uses
  set union (natively idempotent).
- **Parallel endpoint execution with per-endpoint soft-fail.**
  asyncio.gather(..., return_exceptions=True) wrapped in 20s
  wait_fors; any exception is logged and the outcome dropped while
  siblings continue landing.
- **LLM call retries once.** Matches planning doc §"Error handling".
  Prompt build sits outside the retry loop — only the network call is
  re-attempted. Second failure returns an empty HandlerResult rather
  than raising, consistent with the soft-fail contract.
- **TIMEOUT_SECONDS is a local constant.** Couldn't import from
  stage_4/dispatch because that module's import chain still references
  the pre-unification SemanticDealbreakerSpec. Duplicated the 20.0
  value with a comment pointing back to dispatch as the intended
  source of truth once the semantic generator imports are cleaned up.

### Testing Notes
- Import smoke test: `uv run python -c "from
  search_v2.stage_3.category_handlers.handler import run_handler"`
  passes.
- Unit tests for handler.py are a follow-up step (not in scope for
  this change per test-boundaries rule).
- End-to-end validation will cover: SINGLE/MUTEX/TIERED/COMBO each
  firing, TRENDING short-circuit, NO_FIT short-circuit, LLM timeout
  → empty result, per-endpoint exception → sibling results still land.

## Implicit expectations: add `both` signal type
Files: schemas/implicit_expectations.py, search_v2/implicit_expectations.py | Expanded `signal_type` from `quality | notability | other` to `quality | notability | both | other` so a single fragment can explicitly cover both axes at once (e.g. "classics"). Updated the system prompt and schema descriptions to teach the new label, kept "prestige" out of the quality examples, and changed boolean derivation so `both` disables both implicit expectations. Re-ran `python3 -m py_compile` on both files successfully.

## Implicit expectations prompt/schema clarification for `both`
Files: schemas/implicit_expectations.py, search_v2/implicit_expectations.py | Strengthened the small-model guidance for `both`: the schema field description now includes contrast cases (`best` = quality only, `hidden gems` = notability only, `prestige` / `arthouse` = other by default), and the system prompt now has an explicit EXAMPLES subsection with positive `both` cases plus tricky near-misses that should not be classified as `both`. Re-ran `python3 -m py_compile` on both files successfully.

## Implicit expectations semantic correction: `best` and `hidden gems` are `both`
Files: schemas/implicit_expectations.py, search_v2/implicit_expectations.py | Corrected the `both` teaching examples after review: `best` and `hidden gems` now count as `both` in both the schema description and the system prompt. Removed `best` from the quality-only examples and `hidden gems` from the notability-only examples, added them to the positive `both` list, and replaced the contrast examples with cleaner single-axis cases (`critically acclaimed`, `blockbuster`, `obscure`). Re-ran `python3 -m py_compile` on both files successfully.

## Implicit expectations redesign: observations-first + ordering analysis
Files: schemas/implicit_expectations.py, search_v2/implicit_expectations.py, search_v2/test_search_v2.ipynb | Reworked the step around a new schema shape: `query_intent_summary`, per-fragment `explicit_signals` with `explicit_axis=quality|notability|both|neither`, a prose `explicit_ordering_axis_analysis`, then four booleans (`explicitly_addresses_*`, `should_apply_*_prior`). Removed the separate `ImplicitExpectationsLLMOutput` model and now parse directly into `ImplicitExpectationsResult`. The prompt was rewritten to follow that field order explicitly, to treat trending/chronology/semantic-extremeness language as `neither` at the fragment level but discuss it in the ordering-analysis field, and to include worked examples for `comedies`, `best comedies`, `hidden gem comedies`, `trending comedies`, `most recent horror movies`, `scariest horror movies`, and `prestige thrillers`. Updated the notebook test cell to print the new fields and cleared its stale output. Validated with `python3 -m py_compile` on both Python files and a JSON parse of `search_v2/test_search_v2.ipynb`.

## Stage 3 handler LLM pinned to gpt-5.4-mini
Files: search_v2/stage_3/category_handlers/handler.py | Hardcoded the step 3 handler LLM call to always use LLMProvider.OPENAI, model `gpt-5.4-mini`, `reasoning_effort="none"`, `verbosity="low"`. Dropped the `provider` and `model` parameters from `run_handler` and `_run_handler_llm` since the choice is now fixed; no production callers were passing them yet (only notebooks mirrored the flow with their own inline LLM calls).
