# search_v2/ — V2 Search Pipeline

Multi-step query understanding and search execution pipeline. The
front half (Steps 0–2) decomposes a natural-language query into
structured trait atoms; the back half (Stage 3 + Stage 4) routes
those atoms to typed endpoint LLMs and assembles a ranked candidate
list.

## What This Module Does

Receives a user query (text), decomposes it through three front-end
steps (flow routing, spin generation, holistic read) and two
back-end stages (endpoint execution, assembly/reranking), and
returns a ranked candidate list.

## Pipeline Steps

```
Step 0 (step_0.py): Flow routing
  → Step0Response: which of three flows fires
    (exact_title / similarity / standard) + title payloads
  → Runs in parallel with Step 1 on the raw query

Step 1 (step_1.py): Spin generation (standard flow only)
  → Step1Response: two distinct creative spins + UI labels
  → Runs in parallel with Step 0 on the raw query
  → Spin scope: narrowing / reinterpretation / adjacent-swap levers
  → Hard commitments preserved verbatim in spin queries

Step 2 (step_2.py): Query analysis (combined holistic read + atomization)
  → QueryAnalysis.holistic_read — faithful prose read of the query
  → QueryAnalysis.atoms — per-criterion atoms, each with surface_text,
    modifying_signals (unified raw-signal list), and evaluative_intent
    (consolidated 1-2 sentence semantic statement)
  → Stays descriptive on surface_text + modifying_signals;
    evaluative_intent is the one place where light inference is permitted
  → Does NOT commit polarity/salience numbers, category labels, or
    downstream channel routing
  → Feeds Stages 3-5 (reconstruction test, literal test, trait
    commitment) — those stages not yet landed

Steps 0+1 Orchestrator (steps_0_2_orchestrator.py):
  → asyncio.gather for Step 0 + Step 1 in parallel
  → Standard-flow branch budget: 3 minus non-standard flows fired
  → Step 1 failure degrades gracefully (branches drop to original only)
  → Returns OrchestratorResult + list[Step2Branch]
```

## Back-End Stages (Stage 3 / Stage 4)

Stage 3 and Stage 4 are carried over from the earlier V2 work and
are still operational for the old stage_2a/2b concept-routing flow.
The new step-2 holistic-read path feeds into category-handler
routing (stage_3/category_handlers/) which is still under
construction.

```
Stage 3 (stage_3/): Endpoint translation + execution (9 endpoints)
  ├── entity_query_generation.py + entity_query_execution.py
  ├── metadata_query_generation.py + metadata_query_execution.py
  ├── award_query_generation.py + award_query_execution.py
  ├── franchise_query_generation.py + franchise_query_execution.py
  ├── keyword_query_generation.py + keyword_query_execution.py
  ├── studio_query_generation.py + studio_query_execution.py
  ├── semantic_query_execution.py
  ├── trending_query_execution.py     (deterministic, no LLM call)
  └── media_type_query_execution.py   (closed-enum wrapper + executor in place;
                                       no _query_generation module — MEDIA_TYPE
                                       routes deterministically by matching
                                       Step-3 expressions against the
                                       non-default ReleaseFormat values, not
                                       via an LLM handler)
  All executors return EndpointResult (list[ScoredCandidate]).

  stage_3/category_handlers/ — new handler scaffolding for the
    step-2-routed path. In progress:
    handler.py, prompt_builder.py, handler_result.py,
    endpoint_registry.py, schema_factories.py, prompts/

Stage 4 (stage_4/): Assembly & reranking
  → Flattens expressions → concept-level inclusion/exclusion aggregation
  → run_stage_4() is the public entry point
```

## Key Files

| File | Purpose |
|------|---------|
| `step_0.py` | Flow routing. `run_step_0()` returns `Step0Response`. Narrow classifier: fires exact_title / similarity / standard flows, carries title payload. Observations-first schema. Runs parallel with Step 1. |
| `step_1.py` | Spin generation. `run_step_1()` returns `Step1Response`. Produces two distinct spins plus UI labels. Always exactly two spins. `distinctness` field requires result-set divergence from both original and sibling. |
| `step_2.py` | Query analysis — combined Stage 1+2 of the 5-stage trait decomposition. `run_step_2()` returns `QueryAnalysis` (holistic_read + atoms with modifying_signals + evaluative_intent). Model hard-coded to Gemini 3 Flash (no thinking, temperature 0.35). System prompt loads sections this stage applies (atomicity, modifier vs trait, evaluative intent) plus background sections later stages use (carver vs qualifier, polarity, salience, category taxonomy). |
| `run_step_2.py` | CLI runner for step_2. Prints full JSON response + timing + tokens. Default query exercises role markers, polarity, chronological, and multi-dimension entities. |
| `steps_0_2_orchestrator.py` | Orchestrator for the front half. Parallel Steps 0+1, then Step 2 per branch. Standard-flow branch budget = 3 minus non-standard flows. |
| `implicit_expectations.py` | Implicit-prior state management (quality/notability priors). |
| `stage_3/category_handlers/` | Category handler module (scaffolded, prompts in progress). `handler.py` is scoped to a single category — fan-out lives one level up. |

## Step 2 / Query Analysis Design

Step 2 (`search_v2/step_2.py` + `schemas/step_2.py`) combines Stages
1+2 of a planned 5-stage trait decomposition pipeline into a single
LLM call:

| Stage | Purpose |
|-------|---------|
| 1+2 | Query analysis (holistic read + atomization with evaluative intent — single call, landed) |
| 3 | Reconstruction test |
| 4 | Literal test (parametric expansion) |
| 5 | Trait commitment + category grounding |

Stages 3–5 are not yet landed. The full design rationale lives in
`search_improvement_planning/v3_step_2_rethinking.md`.

Key design decisions:
- **Per-criterion evaluative intent**, not a graph of edges. Each
  atom carries a 1-2 sentence prose statement of what scoring on it
  actually means once context is integrated. The intent is the
  load-bearing semantic field downstream consumes.
- **Unified `modifying_signals` list per atom.** Both adjacent
  qualifiers and cross-criterion modifiers land on the same list —
  conceptually they're the same thing (something-shaping-this-
  criterion's-meaning). Each entry is `surface_phrase` (verbatim
  user text) + `effect` (freeform concise description).
- **No closed enum on modifier kind, no SHALLOW/DEEP depth, no
  positional pointers.** All three were experimentally rejected via
  the 34-query test set: bucket-forcing on `kind`, false-precision
  on `depth`, count-fragility on `modifier_atom_index`.
- **Modal vocabulary recommended, not forced.** SOFTENS / HARDENS /
  FLIPS POLARITY / CONTRASTS still keys downstream parsing for
  modal cases; non-modal effects use plain-words descriptions.
- **Surface-text discipline preserved.** `surface_text` is exact
  substring of the query with modifying language stripped; no
  paraphrase, no expansion of named things.
- **Inference license confined to `evaluative_intent`.** The other
  fields stay strictly descriptive.

## Category Taxonomy

`schemas/trait_category.py` houses `CategoryName` — the canonical
43-category vocabulary for grounding trait atoms. Each member carries
`description`, `boundary`, `edge_cases`, `good_examples`,
`bad_examples`, `endpoints` tuple, and `HandlerBucket` enum. The full
taxonomy and split rationale live in
`search_improvement_planning/query_categories.md`.

Notable taxonomy design choices:
- Two parametric-expansion categories (Cat 43 "Like media reference"
  and Cat 45 "Generic catch-all") were removed. Their work now
  belongs to Step 4 (literal test + parametric resolution). Numbering
  gaps at 43/45 preserved for cross-reference stability.
- Cat 6 CHARACTER_FRANCHISE absorbs dual-nature referents (name that
  is both a character and a franchise). 1:1 trait→category is the
  global rule; Cat 6 is the only exception via combo orchestration.

## Endpoint Contract

All endpoint executors follow the same dual-mode signature:
- `restrict_to_movie_ids=None` → dealbreaker mode: generates own candidate pool
- `restrict_to_movie_ids=set[int]` → preference mode: scores exactly the supplied ids
- `restrict_to_movie_ids=set()` → short-circuit: returns empty `EndpointResult`

Failures retry once; second failure returns empty `EndpointResult` (soft failure).

## Dealbreaker Score Floor

All endpoint executors emit dealbreaker scores in `[0.5, 1.0]`
("dealbreaker-eligible band") via `compress_to_dealbreaker_floor(raw) = 0.5 + 0.5 * raw`.
Preference paths keep raw `[0, 1]` scoring. Country-of-origin uses a
3-bucket position score (1.0 / 0.5 / dropped) rather than exponential
decay for the dealbreaker path.

## Key Patterns

- **Positive-presence invariant**: all step-3 endpoint specs express
  presence of an attribute, not absence. Exclusion direction is
  carried by `ActionRole` / `Polarity` in the category handler layer.
- **Concept-level aggregation in Stage 4**: expressions from the same
  concept contribute one inclusion max and one exclusion max. Prevents
  a multi-expression concept from dominating.
- **Token-index resolution**: franchise and award endpoints resolve
  LLM surface forms via posting-list helpers. Empty resolution →
  empty result, not a broadened query.
- **Implicit-prior split**: 80/20 popularity/reception when both
  active (not 50/50). Both axes claim up to `IMPLICIT_PRIOR_CAP=0.25`;
  single-axis cases claim the full cap.

## Interactions

- `schemas/` — step-0/1/2 schemas, category taxonomy, endpoint
  translation schemas, `EndpointResult` / `ScoredCandidate`.
- `db/postgres.py` — posting-list reads for franchise, award, entity, studio.
- `db/qdrant.py` — semantic endpoint vector searches.
- `db/redis.py` — trending endpoint hash reads.
- `implementation/llms/generic_methods.py` — shared LLM router for all generator modules.

## Gotchas

- **`step_2.py` is NOT the old Stage 2A/2B.** `stage_2a.py` and
  `stage_2b.py` are deleted. `step_2.py` is the combined Stage 1+2
  of the new 5-stage pipeline (query analysis: holistic read +
  atomization + evaluative intent). Any code importing old
  `Step2AResponse` / `Step2BResponse` will fail.
- **`schemas/step_2.py` holds `QueryAnalysis`.** Top-level fields
  are `holistic_read: str` + `atoms: list[Atom]`. Each atom has
  `surface_text`, `modifying_signals`, `evaluative_intent`, and
  optional `candidate_internal_split`. The intermediate-design
  types (`AbsorbedModifier`, `IncomingModification`,
  `AbsorbedModifierKind`, `ModificationDepth`) and the older
  single-field `Step2Response` are both deleted; any code
  importing them will fail.
- **Category taxonomy lives in `schemas/trait_category.py`, not
  `schemas/enums.py`.** `CategoryName` was moved and rebuilt; old
  import paths will fail. 43 active members (not 32 or 44 or 45).
- **Semantic dealbreaker scoring uses a top-2000 probe.** `kneed>=0.8`
  is a required dependency for Kneedle elbow detection.
- **`stage_4/priors.py` is deleted.** No replacement — priors are
  out of scope for the V2 runtime orchestrator.
- **Stoplist asymmetry for awards**: ingest writes every token; query
  drops `AWARD_QUERY_STOPLIST`. Intentional — lets the droplist be
  revised without re-ingesting.
