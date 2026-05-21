# `category_candidates` Minimum-Count Experiment — Analysis Handoff

This document is the entry point for the analysis-phase LLM. It
explains what was run, where every file lives, and the hypothesis
the data was gathered to test. The analysis itself has NOT been
performed — that is the next agent's job.

## Hypothesis being tested

Step 3 currently emits as few category candidates per dimension as
the prompt invites. The thesis is that the LLM is short-circuiting
the "lean broad" instruction and committing to a routing decision
without honestly considering adjacent categories. If we force a
minimum count of distinct candidates (3, then 5) via the response
schema and reaffirm "each must be unique and genuinely
applicable" in the field description, the downstream
`routing_exploration` / `combine_mode` / `category_calls` commits
should be better grounded because they have more competing options
to choose from.

The experiment is purely on Step 3. Step 2 output is held fixed
across all variants by serializing it to disk and reusing it.

## Variant inventory

| Prefix | `category_candidates` schema | Stage |
|--------|------------------------------|-------|
| `base` | original (no `min_length`)   | baseline before any change |
| `min3` | `min_length=3` + updated prose ("MINIMUM 3 candidates per dimension, each distinct, most-applicable adjacent categories preferred") | first test |
| `min5` | `min_length=5` + updated prose ("MINIMUM 5 candidates per dimension, each distinct, most-applicable adjacent categories preferred") | second test |

The schema was **restored to the baseline** after the `min5` run
completed. Re-applying either patch requires a one-line edit to
`schemas/step_3.py` — the field is `Dimension.category_candidates`
around line 156.

## File layout

```
search_v2/category_candidates_experiment/
├── __init__.py                  (empty marker)
├── queries.py                   (the 25-query suite + slugify helper)
├── run_step_2_batch.py          (Step 2 batch runner)
├── run_step_3_batch.py          (Step 3 batch runner; takes <prefix>)
├── step_2_results.json          (Step 2 outputs — fixed input for every Step 3 run)
├── ANALYSIS_HANDOFF.md          (this file)
└── results/
    ├── base_<slug>.json   × 25
    ├── min3_<slug>.json   × 25
    └── min5_<slug>.json   × 25
```

Slug rule: lowercase first four words of the query, non-alphanumerics
stripped, joined with `_`. The mapping is implemented in
`queries.slugify_first_four`.

### Per-query JSON shape (`results/<prefix>_<slug>.json`)

```jsonc
{
  "query": "<query string>",
  "prefix": "<prefix>",
  "step_2_output": { ... full QueryAnalysis.model_dump() ... },
  "runs": [
    {
      "run_index": 0,                  // 0, 1, or 2 — three repeats per query
      "elapsed_seconds": 5.5,
      "trait_results": [
        {
          "trait_surface_text": "...",
          "trait": { ... full Trait.model_dump() ... },
          "decomposition": { ... full TraitDecomposition.model_dump() ... },
          "input_tokens": 12345,
          "output_tokens": 678,
          "elapsed_seconds": 4.9
          // "error": "<only present if the LLM call failed>"
        },
        ...
      ]
    },
    { "run_index": 1, ... },
    { "run_index": 2, ... }
  ]
}
```

All 75 files exist with zero per-trait LLM failures across all three
runs (verified at completion time). One file per (variant, query)
pair, three full `runs` inside each file.

## How to compare

Suggested comparison axes — each diff is `base → min3 → min5` per
query, per trait, per run:

1. **Candidate-count distribution.** Per dimension, `len(category_candidates)`
   across runs. Confirms the schema constraint actually bound and
   surfaces how often the LLM voluntarily exceeded the minimum.
2. **Candidate quality.** Are the "extra" candidates substantive
   adjacencies or filler? Read `what_this_covers` /
   `what_this_misses`. Padding shows up as generic prose or
   `what_this_misses` = "nothing" on a category whose presence is
   indefensible.
3. **Routing-exploration depth.** Does `routing_exploration` change?
   Does it engage with the extra candidates (dedup / drop) or
   ignore them?
4. **Combine-mode drift.** Per trait, did `combine_mode` flip
   (SOLO / FRAMINGS / FACETS)? Flips that survive across all 3
   runs are stronger evidence than single-run flips.
5. **Final commit shape.** Per trait, did `category_calls` change
   shape — number of calls, categories chosen, expressions per
   call? Per the `rescore_overhal_queries.md` rubric, the
   per-(trait, category) commit shape is the load-bearing
   measurement; individual keyword commits drift across runs by
   design.
6. **Run-to-run variance.** Within each variant, how much do the
   three runs disagree? This is the noise floor against which
   between-variant deltas must be judged.

### Inputs the analyst will need from elsewhere in the repo

- **The verification rubric and per-query expectations** —
  `search_improvement_planning/rescore_overhal_queries.md`. Read
  the per-query "what to look for" notes; the 25-query suite was
  authored against this rubric and the "diagnostic shape" notes
  are the right yardstick. **Do not feed any of the 25 queries
  into a prompt or schema example** under any circumstance.
- **Step 3 schema source of truth** —
  `schemas/step_3.py`. `Dimension.category_candidates` is the
  field that was patched.
- **Step 3 prompt source of truth** — `search_v2/step_3.py`. Two
  sections directly govern candidate cardinality:
  `_PER_DIMENSION_CANDIDATES` (the "lean broad" instructions)
  and `_ROUTING_EXPLORATION` (where pruning happens).
- **Category taxonomy** — `schemas/trait_category.py` (loaded into
  the prompt by `_build_full_category_taxonomy_section` in
  step_3.py). Necessary context for judging whether a candidate
  is a real adjacency or filler.
- **Project context** — `docs/PROJECT.md`, `docs/conventions.md`.
  Tradeoff priority order matters when judging whether the cost
  (more tokens, more LLM time) is worth the gain.

## Cost / latency reference (for context only)

Wall-clock totals for the Step 3 batch over the whole 25-query
suite, three runs each (sequential across queries, parallel runs &
parallel traits within each run):

- `base`  — 163.7 s
- `min3`  — 197.4 s  (≈ +21 %)
- `min5`  — 247.8 s  (≈ +51 %)

These are wall-clock not LLM tokens; each file's `trait_results[*]`
carries per-call `input_tokens` / `output_tokens` if a token-cost
comparison is wanted.

## What the analyst should produce

A written report comparing the three variants against the
`rescore_overhal_queries.md` rubric, with a clear recommendation:
keep `base`, ship `min3`, ship `min5`, or move on with a different
intervention. The recommendation must be grounded in concrete
per-query examples drawn from the JSON files in `results/`, not in
plausibility arguments about the schema change.
