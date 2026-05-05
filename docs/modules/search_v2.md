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

Step 3 (step_3.py): Trait decomposition (per-trait LLM call)
  → TraitDecomposition: target_population + trait_role_analysis +
    aspects + dimensions + category_calls
  → One call per committed trait; orchestrator fans out in parallel
  → Reads qualifier_relation as the carver-vs-qualifier signal
    (Step 2 no longer commits a binary `role` field)

Full pipeline orchestrator (full_pipeline_orchestrator.py):
  → run_full_pipeline(query, *, skip_bypass_steps_0_1=False)
  → Steps 0 + 1 in parallel (or skipped); Step 2 per branch;
    implicit-prior policy per branch runs from Step-2 committed traits
    in parallel with Step 3 per trait; per-CategoryCall handler-LLM
    fired immediately as each Step 3 returns (does not wait on sibling
    traits)
  → 25s timeout + 1 retry on every individual LLM call
  → After Step 3 + handler-LLM, applies the reranker-only candidate
    fallback (promotes tier-1+ rerankers or emits a NEUTRAL_SEED spec)
    and the default shorts-exclusion auxiliary, then calls
    stage_4_execution.execute_branches to actually fire endpoints and
    rank candidates per branch. The full orchestrator then applies the
    implicit-prior multiplicative post-rerank when active.
  → Returns FullPipelineResult: per-branch trait/category/endpoint
    specs (`branches`) + auxiliary specs + per-branch ranked candidate
    lists (`branch_results: list[BranchRankedResults]`).
  → Soft-fails per branch / per trait / per CategoryCall / per
    endpoint call; only Step 0 failure is fatal.
```

## Stage 4 — Execution & Ranking (`stage_4_execution.py`)

Stage 4 is the execution + scoring layer that turns generated endpoint
specs into a ranked candidate list. Public entry:
`execute_branches(branches, auxiliary_specs) -> list[BranchRankedResults]`.

It implements the recursive granularity rule from
`search_improvement_planning/search_method_deterministic_logic.md`
(§3 / §5 / §6 / §7 / §8):

```
query
  └── traits             (across-trait at branch level)
       └── categories    (intra-trait at trait level)
            └── calls    (intra-category at category level)
```

At every level a node is either **candidate-generating** (≥1
positive-polarity CANDIDATE_GENERATOR call somewhere in its subtree)
or **pure-reranker**. Cand-gen nodes execute generators in isolation
then run rerankers within their own scope; pure-reranker nodes defer
one level up. Composites use **nested equal-weight averaging** —
calls average within a category, categories average within a trait,
traits combine across the branch via §9's weighted sum.

### Branch-level scoring (§9)

```
final_score(movie) = Σ over traits of (trait_score × trait_weight × polarity_sign)

trait_weight = commitment_multiplier × rarity_factor      (cand-gen traits)
             = commitment_multiplier × 1.0                 (pure-rer / negative)
polarity_sign = +1 (positive)  or  -1 (negative)
```

Commitment multipliers: `required=3.0 / elevated=1.75 / neutral=1.0 /
supporting=0.6 / diminished=0.35`. Rarity tiers (corpus N≈150K):
`<0.1%→1.5 / <1%→1.2 / <10%→1.0 / <30%→0.75 / else 0.5`.

A cand-gen trait contributes 0 to candidates it didn't generate
(opportunity cost, §8) — missing positive ≠ active subtraction.

### Implicit-prior post-rerank

`search_v2.implicit_expectations` runs after Step 2 and consumes
`QueryAnalysis.intent_exploration` plus the committed Step-2 traits.
It does not rediscover criteria from the raw query. Its output records
per-trait prior-axis evidence, an explicit ordering-axis analysis,
query specificity / prior room, and final quality/popularity
`PriorDecision`s.

The full orchestrator applies active priors after Stage 4 base scoring:

```
boost = quality_cap * quality_signal + popularity_cap * popularity_signal
prior_base = positive_total if positive_total > 0 else 1.0
boosted_score = base_score + prior_base * boost
```

Boost caps are `quality: 0 / 0.025 / 0.06 / 0.10` and
`popularity: 0 / 0.05 / 0.12 / 0.20` for
`none / light / normal / strong`. Inverse directions use
`1 - normalized_signal`. Missing reception or popularity data has no
effect on the boost. The neutral `prior_base = 1.0` fallback lets
pure-negative queries (for example, "not scary") still use implicit
priors for ordering without magnifying negative penalties.

### Negative-trait scoring (gate × noisy-OR, three-bin)

`handler.determine_operation_type` short-circuits every
negative-polarity call to `POOL_RERANKER` regardless of route, so the
"would-be-generator vs would-be-reranker" partition is **not readable
off `spec.operation_type`** at execution time. Stage 4 re-derives it
by calling `determine_operation_type(category, route, Polarity.POSITIVE)`
per call, then further partitions the would-be-generator calls by
whether the category is *authoritative* about the negative concept
(membership definitively answers "is this a member?", e.g.
PERSON_CREDIT, NAMED_CHARACTER, RELEASE_DATE, MEDIA_TYPE) versus
*evidential* (a high-precision but low-recall proxy, e.g.
KEYWORD-style tags, GENRE, archetypes, TRENDING, CENTRAL_TOPIC).

The three bins drive different aggregation shapes:

```
G_a = would-be CANDIDATE_GENERATOR with authoritative category
      (specific-entity / structured-metadata)
G_e = would-be CANDIDATE_GENERATOR with evidential category
      (keyword tags, archetypes, fuzzy descriptors)
R   = would-be POOL_RERANKER (continuous similarity / prior)

gate  = ∏ G_a                         when present
fuzzy = 1 − ∏(1 − s)  over G_e ∪ R    when present

trait_score = gate × fuzzy   when both present
            = gate           G_a only
            = fuzzy          G_e ∪ R only
            = 0.0            all empty
```

Why the asymmetry: a single fuzzy proxy ("not scary" decomposed to
KEYWORD:horror) under-recalls and shouldn't gate a more comprehensive
semantic R. But authoritative G calls describe required parts of a
conjunctive concept ("not Joaquin Phoenix Joker" → PERSON_CREDIT:JP +
NAMED_CHARACTER:Joker) and SHOULD gate — otherwise R's similarity to
the JP-Joker vibe over-penalizes Heath Ledger Jokers. Authoritative
vs evidential is the discriminator; G_e merges into the noisy-OR with
R because both are alternative evidence rather than required parts.

The authoritative set is hardcoded in
`stage_4_execution._AUTHORITATIVE_NEGATION_CATEGORIES` (co-located
with the consumer rather than alongside `_SEMANTIC_PROMOTION_TIERS`
in `full_pipeline_orchestrator.py` because the orchestrator imports
from `stage_4_execution` at load time).

Failed calls are *dropped from their bin* rather than counted as
confirmed-zero — option A from review. A single transient endpoint
failure can no longer zero out the gate or saturate the noisy-OR.
Sign is applied at the §9 aggregation layer, not inside the trait.

### Auxiliary specs (NEUTRAL_SEED + shorts exclusion)

`auxiliary_endpoint_specs` carries up to two entries from upstream:
the reranker-only fallback's `NEUTRAL_SEED` (additive) and the default
shorts-exclusion `MEDIA_TYPE` (subtractive). Stage 4 applies them at
branch level, in order:

1. If `branch_pool` is empty AND a NEUTRAL_SEED spec is present, fetch
   `db.postgres.fetch_neutral_reranker_seed_ids()` and seed the pool.
2. If a MEDIA_TYPE spec is present, fetch the SHORT-format movie IDs
   and **subtract** them from the pool. The MEDIA_TYPE spec is tagged
   `CANDIDATE_GENERATOR` upstream so its executor returns the SHORT
   set; Stage 4 uses those IDs as a blocklist, not as a positive
   contribution.

Neither contributes a `trait_score`. Pure rerankers run *after*
auxiliary application so they only score the surviving pool.

### Rarity bookkeeping

Rarity uses a per-trait union over the trait's would-be-generator
calls (per §7 + the user's clarification on semantic-promoted traits):

- **Promoted call** (`spec.was_promoted=True`, set by
  `_apply_reranker_only_candidate_fallback`): only post-elbow 1.0
  scoring movies count.
- **Regular finder call**: every matched candidate counts (every key
  in the call's score map).
- A trait mixing both unions both sets.

### Soft-failure semantics

`_dispatch_call` returns `dict[int, float] | None`. None ≡ failed
call. Positive-trait paths fold None into `{}` (the call simply
doesn't contribute candidates / scores); negative-trait scoring drops
None calls from both G and R partitions.

## Back-End Stages (Stage 3 / Stage 4)

Stage 3 owns endpoint translation + execution. Stage 4 (the new
`stage_4_execution.py`, not the legacy `stage_4/` directory) owns
recursive scoring + ranking — see the **Stage 4 — Execution &
Ranking** section above for the full design.

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
    handler.py, generated_endpoint_spec.py, prompt_builder.py,
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
| `full_pipeline_orchestrator.py` | End-to-end orchestrator. `run_full_pipeline(query, *, skip_bypass_steps_0_1=False)` runs Steps 0+1 in parallel (or skipped) → Step 2 per branch → Step 3 per trait → per-CategoryCall handler-LLM endpoint-spec generation → reranker-only fallback + auxiliary-spec planning → `stage_4_execution.execute_branches`. Returns `FullPipelineResult` with per-branch specs (`branches`) and ranked candidates (`branch_results`). 25s timeout + 1 retry per LLM call. |
| `stage_4_execution.py` | Stage 4 execution + ranking. `execute_branches(branches, auxiliary_specs)` returns `list[BranchRankedResults]`. Implements recursive granularity (category → trait → branch) with nested equal-weight averaging, three-bin negative-trait scoring (`gate × fuzzy` where gate = ∏ authoritative-G and fuzzy = noisy-OR over evidential-G ∪ R; would-be partition derived via `determine_operation_type(category, route, POSITIVE)` then split by `_AUTHORITATIVE_NEGATION_CATEGORIES`), commitment × rarity weighting, NEUTRAL_SEED additive seeding, and MEDIA_TYPE shorts subtraction. Per-call soft-fail returns None so failed calls are dropped from their bin instead of zeroing the gate or saturating the noisy-OR. |
| `step_3.py` | Trait decomposition. `run_step_3(trait)` returns `TraitDecomposition`. One LLM call per committed Trait. Reads `qualifier_relation` (with `"n/a"` sentinel) as the carver-vs-qualifier signal — `role` was removed from Step 2. |
| `endpoint_fetching/category_handlers/output_extractor.py` | `extract_fired_endpoints(category, output)`: per-bucket extraction of `(EndpointRoute, EndpointParameters)` pairs from a handler-LLM structured output. |
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

**Exception: `semantic_query_execution.py` emits raw `[0, 1]` on every
path** (carver candidate-generator, carver reranker, qualifier
reranker, qualifier promoted). The dealbreaker compression was dropped
when the executor was reworked so that `role` drives within-space
normalization (corpus-calibrated elbow for carver/qualifier-promoted,
pool-relative rescale for qualifier+restrict) and cross-space
combination (`max()` for carver, `Σ(w·score)/Σw` for qualifier).

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
- **Stage 4 lives in `stage_4_execution.py`, not `stage_4/`.** The
  legacy `stage_4/` directory's assembly/reranking code is superseded
  by the recursive granularity executor described above. Code
  importing the old `run_stage_4` will fail.
- **Negative-trait `operation_type` is uniformly `POOL_RERANKER`.**
  `determine_operation_type` short-circuits negative polarity to
  `POOL_RERANKER` regardless of route, so the spec's
  `operation_type` does NOT identify which negative-trait calls
  would have been candidate-generators in positive polarity. Stage 4
  re-derives the would-be type using `determine_operation_type(category,
  route, Polarity.POSITIVE)`, then further partitions the would-be
  generators by `_AUTHORITATIVE_NEGATION_CATEGORIES` to drive the
  three-bin `gate × fuzzy` formula.
- **Stoplist asymmetry for awards**: ingest writes every token; query
  drops `AWARD_QUERY_STOPLIST`. Intentional — lets the droplist be
  revised without re-ingesting.
