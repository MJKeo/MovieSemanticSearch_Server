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
