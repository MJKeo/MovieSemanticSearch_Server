# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

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
