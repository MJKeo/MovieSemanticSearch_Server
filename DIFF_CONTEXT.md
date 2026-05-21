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
