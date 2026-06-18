# [090] — Step 3 loose5: coordinated schema floor + framing + model changes

## Status
Active — model selection (Decision item 3) superseded by ADR-100:
Step 3 now uses OpenAI gpt-5.4-mini, not gemini-3.5-flash. The schema
floor (item 1) and prune-ruthlessly framing (item 2) remain active.

## Context
Step 3 was producing shallow category_candidates lists — the model
would commit 2-3 categories and stop, leaving relevant routing options
unexplored. An experiment across 4 variants × 25 queries × 3 runs
tested options: `base` (unchanged), `min3` (schema floor only, min_length=3),
`min5` (deeper schema floor, min_length=5), `loose5` (schema floor +
"expect filler / prune ruthlessly" framing + model swap).

The prior min5 experiment (schema floor only) showed regressions: the
model treated `min_length` as both floor AND ceiling, committed weak
adjacents as hedges rather than surfacing them for later pruning.

## Decision
Ship all three changes together as `loose5`:
1. `category_candidates` schema floor `min_length=5` to force broad exploration.
2. `routing_exploration` prompt + schema framing that explicitly tells the model
   the floor will produce fillers and the routing step must prune ruthlessly.
3. Model swap to `gemini-3.5-flash` with `thinking_level="low"` (distinct from
   `thinking_budget=0` which disables thinking entirely).
   **(Superseded by ADR-100 — Step 3 now uses OpenAI gpt-5.4-mini at
   reasoning low / verbosity low.)**

The framing change is the key enabler: without it, the floor produces committed
hedges. With it, the model surfaces 5+ options and then drops the weak ones in
`routing_exploration` before committing.

## Alternatives Considered
- **Schema floor only (min5)**: regression on MMA query (lost
  Element/motif presence); model anchored min_length as ceiling too.
- **Model swap only**: did not address the shallow-exploration failure mode.
- **Prompt framing only without floor**: less effective; model reverted to
  2-3 candidates without the floor's structural pressure.

## Consequences
- `n_dims` dropped ~22% (358 → 272) in the experiment — unknown whether
  model or framing. Fewer dimensions = fewer routing chances; watch downstream.
- Output tokens 15% lower than min5 despite same floor; wall-clock comparable.
- 3-run stability improved: 75% vs 63-65% for other variants.
- Step 3 stays on `"low"` thinking while Steps 0/1/2 use `"minimal"` — step 3
  carries the heaviest routing reasoning load (per-trait category routing,
  granularity gate, combine-mode selection).

## References
- `search_v2/step_3.py`, `schemas/step_3.py`
- `search_v2/category_candidates_experiment/ANALYSIS_HANDOFF.md`
- ADR-088 (FACETS soft fold)
