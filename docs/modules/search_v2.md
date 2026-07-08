# search_v2/ — V2 Search Pipeline

Multi-step query understanding and search execution pipeline. The
front half (Steps 0–2) decomposes a natural-language query into
structured trait atoms; the back half (Stage 3 + Stage 4) routes
those atoms to typed endpoint LLMs and assembles a ranked candidate
list. Standalone entity-flow executors handle exact-title, similarity,
person, studio, and character/franchise queries outside the standard
trait pipeline.

## What This Module Does

Receives a user query (text), decomposes it through three front-end
steps (flow routing, spin generation, holistic read) and two
back-end stages (endpoint execution, assembly/reranking), and
returns a ranked candidate list. When Step 0 identifies an entity
flow (exact title, similarity, person, studio, character/franchise,
non-character franchise), the dedicated executor runs in place of
the standard pipeline.

## Pipeline Steps

```
Step 0 (step_0.py): Flow routing
  → Step0Response: which of seven entity flows fires
    (exact_title / similarity_to_titles / character_franchise /
     non_character_franchise / studio / person / none_of_the_above)
    plus an optional standard co-fire
  → SimilarityFlowData now carries a list[SimilarityReference]
    (length 1 for single-anchor, length N for multi-anchor frames
    like "Inception meets Arrival")
  → ExactTitleFlowData and SimilarityFlowData carry optional
    release_year (only when user explicitly states it; never inferred)
  → Similarity, studio, and person flows are list-shaped (one-or-more
    entities); exact_title and character/franchise take exactly one entity
  → The person flow is role-agnostic — any credited role (actor,
    director, writer, producer, composer) qualifies the span; no
    preferred role
  → Observations-first schema. Runs parallel with Step 1.

Step 1 (step_1.py): Spin generation (standard flow only)
  → Standard path: `Step1Response` — `exploration` scratchpad (2-3
    telegraphic sentences) + exactly two `Spin(query, ui_label)` objects
  → Clarification path: `Step1ClarificationResponse` — `exploration` +
    `main_rewrite: Spin` (faithful translation of the merged intent) +
    `spins: list[Spin]` (creative angles on the rewritten intent). Schema
    reuses the same `Spin` class so downstream branch-plan unpacking is
    identical for both paths.
  → Runs in parallel with Step 0 on the raw query
  → Freeform: no structured decomposition fields — the model
    reasons visibly in the `exploration` field and commits to spins;
    hidden thinking is disabled (Gemini 3.5 Flash with
    `thinking_level="minimal"`)
  → Goal: each spin must produce a visibly different result set
    from both the original query and the sibling spin; the model is
    free to drop or replace original-query anchors when keeping them
    would make every spin collapse onto the same result set
  → When a clarification is present, slot 1 in the branch plan carries
    `main_rewrite.query` + `main_rewrite.ui_label` (faithful rewrite,
    not the raw original query). Slot 1 always uses the user's own
    words verbatim — no static "Original Query" placeholder.

Step 2 (step_2.py): Query analysis (combined holistic read + atomization)
  → QueryAnalysis.intent_exploration — faithful intent read (replaces
    holistic_read; promoted to primary evidence source for role decisions)
  → QueryAnalysis.atoms — per-criterion atoms, each with surface_text,
    split_exploration (was split_note), standalone_check (was redundancy_note),
    modifying_signals, and evaluative_intent (faithful restatement,
    bounded by surface_text + modifying_signals; near-paraphrase is
    correct when no signals exist)
  → QueryAnalysis.traits — committed layer on top of atoms; each Trait
    carries surface_text, evaluative_intent, qualifier_relation,
    anchor_reference, polarity, commitment_evidence, commitment
    (required/elevated/neutral/supporting/diminished), contextualized_phrase,
    relationship_role (INDEPENDENT/POSITIONING_REFERENCE/POSITIONING_QUALIFIER),
    replaces_axis, axes_replaced_by_siblings
  → Model: `gemini-3.5-flash` with `thinking_level="minimal"`, temperature 0.35
  → QueryAnalysis.traits is the closed set of explicit user intent Step 3 consumes

Step 3 (step_3.py): Trait → category-call decomposition (per-trait LLM call)
  → TraitDecomposition: trait_restatement (verbatim quotes of
    contextualized_phrase + evaluative_intent — first field, auto-regressive
    anchor) + target_population + trait_role_analysis + aspects + dimensions
    (expression = verbatim copy of one aspect string; routing only, no
    translation) + category_calls + combine_mode + consolidation_analysis
  → aspects now enumerate distinct, non-overlapping, comprehensive PARTS
    of the trait (not the whole restated alongside its parts). Described
    plot/story shapes are kept whole as one aspect — splitting a story's
    character archetypes, events, and relations into peer aspects is the
    primary over-fragmentation failure mode Step 3 guards against.
  → Each CategoryCandidate now carries a `fit: CandidateFit` enum:
    CLEAN_OWNERSHIP (category squarely claims this aspect) /
    COULD_CONSOLIDATE (usable but better folded into a broader call) /
    LIKELY_DISREGARD (category can only describe this via an absence —
    inclusion-ineligible). Fit is decided at candidate enumeration time,
    not at the call layer.
  → `consolidation_analysis` (was `routing_exploration`): the model
    explores options, places the trait on the breadth↔single-shape spectrum
    (continuous), then selects the minimum-viable call set — folding finer
    aspects into broader calls rather than spawning fragile separate facets.
  → combine_mode: SOLO (one category cleanly covers all dims; orchestrator
    keeps first call, drops extras) / FRAMINGS (alternative homes for one
    signal; Stage-4 MAX) / FACETS (compound concept; Stage-4 geometric-mean
    with floor=0.1). combine_mode reads off the spectrum placement.
  → relationship_role (INDEPENDENT / POSITIONING_REFERENCE /
    POSITIONING_QUALIFIER) is the PRIMARY structural signal; sibling
    structural fields (no interpretive prose) surface so positioning
    references can drop replaced axes without leaking sibling decompositions
  → Step 3 is polarity-agnostic; every call describes presence of the attribute
  → Siblings list passed to _build_user_prompt so POSITIONING_REFERENCE traits
    can honor cross-trait scope replacement
  → Model: OpenAI `gpt-5.4-mini` with `reasoning_effort="low"`, `verbosity="low"`
    (caller cannot override — see step_3.py executor comment; chosen over
    Gemini 3.5 Flash after the consolidation experiments).
    `category_candidates` has a schema floor of `min_length=5`.

Full pipeline orchestrator (full_pipeline_orchestrator.py):
  → run_full_pipeline(query, *, skip_bypass_steps_0_1=False)
  → Steps 0 + 1 in parallel (or skipped); Step 2 per branch;
    implicit-prior policy per branch runs in parallel with Step 3 per trait;
    per-CategoryCall handler-LLM fired immediately as each Step 3 returns
    (does not wait on sibling traits)
  → SOLO combine_mode: orchestrator trims category_calls to [:1] before
    handler fan-out so dropped categories never reach endpoint generation
  → Sibling-task context injected into handler user message:
    <sibling_categories combine_mode="..."> block listing parallel
    category retrieval_intents (instruction-time, parallel-safe)
  → 25s timeout + 1 retry on every individual LLM call
  → After Step 3 + handler-LLM, applies reranker-only candidate fallback
    (promotes tier-1+ rerankers or emits a NEUTRAL_SEED spec) and the
    default shorts-exclusion auxiliary, then calls
    stage_4_execution.execute_branches to actually fire endpoints and
    rank candidates per branch. The full orchestrator then applies
    implicit-prior multiplicative post-rerank when active.
  → Returns FullPipelineResult: per-branch trait/category/endpoint
    specs (branches) + auxiliary specs + per-branch ranked candidate
    lists (branch_results: list[BranchRankedResults]).
  → Soft-fails per branch / per trait / per CategoryCall / per
    endpoint call; only Step 0 failure is fatal.
```

## Entity-Flow Executors (Step 0 fast paths)

### Exact-title search (`exact_title_search.py`)
Seven-tier scoring scheme: 1.0 seed (title+year match), 0.75 lineage
sibling, 0.625 seed-lineage→candidate-universe, tier 6.5 close-title-match
(query tokens ⊆ doc tokens, score 0.80–0.95 based on length ratio;
year-mismatch variant 0.40–0.475), 0.5 title-only year-mismatch,
0.25 seed-universe→candidate-universe, 0.125
seed-universe→candidate-lineage. No LLM call. Year from
`flow_data.release_year` only — never inferred.
Close-match candidates do NOT seed franchise fan-out — only 1.0 seeds
anchor fan-out. The close-match tier uses `fetch_close_title_candidates`
(pg_trgm `<%` operator) with a strict Python token-subset gate.

### Similarity search (`similar_movies.py`)
Standalone "movies like X" flow. Supports single-anchor and
multi-anchor (Step 0 now routes "Inception meets Arrival" here via
`list[SimilarityReference]`). Lane-based: shape (Qdrant vectors),
director (curated 63-entry auteur list, multiplicative-on-shape),
franchise (5-tier lineage/subgroup/universe), studio (multiplicative),
source (IDF-weighted), themes (IDF-weighted + compounding bonus),
cast (N-anchor bucket-with-floor), quality (always-on), format,
country/language, specific-award. V3.4 Bucket-Weaver assembles top
10 from 5 named buckets with MMR; franchise-fatigue (≤0.34 ratio)
and director gap-boost prevent over-clustering. `active_anchor_types`
appends `director_signature` in the multi-anchor path whenever
`has_auteur_anchor` (union of anchor directors ∩ the curated auteur
set) — this mirrors the single-anchor gate and was a correctness fix;
it means "≥1 anchor was directed by a curated auteur", NOT "anchors
share a director" (that's a separate, currently-untraced director
*cohesion* signal). Title-based resolution (`_resolve_similarity_reference`)
returns the winning row itself, and the engine
(`run_similar_movies_for_ids`) reuses those already-fetched anchor rows
instead of re-querying Postgres by id — the title path now issues one
fewer `movie_card` read than the raw-ID path (`/similarity_search`,
`run_similar_movies_batch.py`), which still fetches all anchors by id.

### Character-franchise search (`character_franchise_search.py`)
Fires when Step 0 selects `EntityFlow.CHARACTER_FRANCHISE`. Uses
one `character_franchise_fanout_call` LLM round-trip to expand
character forms + franchise forms. Seven disjoint tiers: lineage-
mainline (NOT is_spinoff AND release_format=MOVIE) → top-billed-
appearance (billing_position ≤ 3, outside lineage) → lineage-
ancillary → universe → prominent-appearance (DEFAULT ≥ 0.70 AND
billing > 3) → relevant-appearance → minor-appearance. Within-tier
sort is by `popularity_score DESC`. Tier 2 sits between the two
lineage halves so a top-billed appearance never outranks a major
theatrical Batman but always outranks obscure franchise entries.
`CHARACTER_FRANCHISE_FANOUT` handler output hard-forces
`prominence_mode=CENTRAL` and `prefer_lineage=True`; the fanout
prompt explicitly forbids copying umbrella-universe names into
`franchise_forms`.

### Non-character franchise search (`non_character_franchise_search.py`)
Fires when Step 0 selects `EntityFlow.NON_CHARACTER_FRANCHISE`.
Returns `NonCharacterFranchiseSearchResult` with `primary_franchise`
(lineage members) and `secondary_franchise` (universe-only), each
sorted by `popularity_score DESC`. Single Postgres round-trip.

### Studio search (`studio_search.py`)
Fires when Step 0 selects `EntityFlow.STUDIO`. Translates studio
names via the Step 3 studio translator (one LLM call, soft-degrades
to deterministic on failure), runs `execute_studio_query` in ANY
mode, popularity-sorts matched IDs. Single tier — co-productions not
separately bucketed.

### Person search (`person_search.py`)
Fires when Step 0 selects `EntityFlow.PERSON`. No LLM call. Each
named person is resolved to a term_id and looked up against ALL
five role posting tables (actor, director, writer, producer,
composer) in parallel — the executor is role-agnostic, with no
preferred role. Per-(person, movie) bucket is the MIN across roles:
actor credits use the sqrt-adaptive cast-zone model
(`actor_zones.py`) to assign one of four prominence buckets (lead /
supporting / minor top-half / minor bottom-half); director / writer
/ producer / composer credits carry no billing data so they
uniformly land in bucket 1 (lead).

Multi-person queries (e.g. "Spielberg and Williams") use UNION
semantics — any movie where any named person has any credit is a
match. For each matched movie the executor tracks (a) the MIN
bucket across the named people who appear ("best credit by any
named person") and (b) an overlap_count of how many of the named
people appear. Within-bucket sort is (overlap_count DESC,
popularity DESC, movie_id DESC) so the intersection of all named
people surfaces above single-person matches in the same prominence
tier. Single-person queries collapse to pure popularity DESC.

### Supporting modules
- `actor_zones.py` — sqrt-adaptive zone cutoffs and in-zone relative
  position. Shared between `person_search.py` and
  `endpoint_fetching/entity_query_execution.py` so tuning parameters
  stay synchronized.
- `popularity_sort.py` — shared popularity-sort helper used by
  character_franchise_search and studio_search. The person flow
  inlines an equivalent sort key so it can carry overlap_count as
  the primary within-bucket sort component.
- `auteur_directors.py` — 63-entry curated auteur frozenset, lazily
  resolved to `lex.lexical_dictionary.string_id` on first call.
- `production_medium_registry.py` — 6×6 technique-only medium
  similarity matrix (MEDIUM_SIMILARITY) + `medium_score()` helper.
- `format_registry.py` — format bucket priority and
  `FORMAT_CROSS_CATEGORY_MULTIPLIER = 0.35`.
- `country_language_registry.py` — nation/language OverallKeyword IDs.
- `award_taxonomy.py` — three-tier specific-award scoring.
- `similar_studio_registry.py` — production-company similarity data.
- `run_step_0_batch.py` — batch runner for Step 0 across N queries;
  writes per-query files to `step_0_results/`. Includes person and
  studio search result display.

## Deterministic Endpoint Backers (no NLP / LLM / vector search)

Three standalone modules back API endpoints outside the trait pipeline
and the Step-0 entity flows. None makes an LLM or vector call — every
input is a closed enum / numeric range (pushed to the SQL layer via
`MetadataFilters`), a person name resolved against the lexical posting
tables, or a normalized title match.

### Attribute search (`attribute_search.py`)
Backs `POST /attribute_search`. Hard-attribute browse: with no people
supplied, returns the whole `movie_card` table ranked by the neutral
80/20 popularity/reception prior (`fetch_neutral_reranker_seed_ids`),
filter-respecting. With one or more people supplied, reuses the Step 0
person-prominence model verbatim via the shared `fetch_person_buckets`
(role-agnostic, 4 buckets LEAD/MAJOR/RELEVANT/MINOR) — the sole
divergence from Step 0 is that per-person bucket weights are **summed**
across people (Step 0 takes MIN bucket + `overlap_count`), so a movie
crediting more/more-prominent people ranks higher.

### Title-only typeahead (`title_search.py`)
Backs `GET /title_search`. Title-search for the frontend's "pick
similar-to" mode, called on every debounced keystroke. Single Postgres
query over `movie_card.title_normalized` with two exact tiers
(token-prefix > substring) plus a `pg_trgm` `word_similarity` fuzzy
fallback when the exact scan underfills; within-tier sort is
shorter-normalized-title-first, then the same 80/20 popularity/reception
prior `/attribute_search` uses, then `movie_id DESC`.

### Query input validation (`query_input_validation.py`)
Single source of truth for sanitizing the two user free-text fields
(`query` and optional `clarification`): strip, non-empty, length cap.
Every public surface that accepts them (the API endpoint, both
orchestrators, and the individually-callable Steps 0/1) routes through
here so the rules are enforced identically. Over-length input is
**rejected** (API → 400), never silently truncated, so a query is never
sliced mid-clause into corrupted intent.

## Stage 4 — Execution & Ranking (`stage_4_execution.py`)

Stage 4 is the execution + scoring layer that turns generated endpoint
specs into a ranked candidate list. Public entry:
`execute_branches(branches, auxiliary_specs) -> list[BranchRankedResults]`.

It implements the 5-phase pipeline from
`search_improvement_planning/rescore_overhaul.md`. The load-bearing
property is **separation of pool definition from per-trait scoring**:
all positive generators across all traits build one union, all
positive rerankers score that finalized union, then per-trait
scoring composes per-call scores via the category's combine type
and takes max across categories. This fixes the prior trait-local
reranker bug — in queries like "dark gritty marvel movies", the
dark/gritty rerankers now score the marvel candidates produced by
sibling traits.

```
Phase A — LLM generation (Steps 0/1/2/3 + handler-LLM, upstream)
Phase B — Pool definition: positive generators in parallel → union
          → shorts subtraction
          → Filter-active tiered promotion loop: when a hard filter
            is active and len(union) < CANDIDATE_FLOOR (25), promote
            the lowest-tier reranker(s) into generator role, dispatch
            them, merge into union, repeat until floor met or tiers
            exhausted (parallel within tier, serial across tiers).
            Unfiltered base case is unchanged ("doesn't exist means
            doesn't exist")
          → Neutral-seed when no positive generator was attempted
            pipeline-wide (aux-spec path) OR when the filter-active
            tiered loop exhausted with an empty union (direct path,
            bypasses aux-spec gate)
          → Post-hoc generator-spec dedup: group by (route, params) →
            one _dispatch_call per unique group, result broadcast to
            all shared CallKeys. The promotion loop carries a
            cross-iteration cache so a spec dispatched in an earlier
            tier is not re-dispatched if a later tier holds an
            identical (route, params) spec
Phase C — Reranker pass: positive rerankers in parallel against the
          finalized union (no trait-local scoping). Specs that were
          promoted to generators during the tiered loop are removed
          from this pass so their generator score isn't overwritten
Phase D — Per-trait scoring (positive + negative), described below
Phase E — Branch aggregation: Σ trait_score × weight × sign
```

### Within-category combine

Each `CategoryName` member declares a `combine_type` (in
`schemas/enums.CategoryCombineType`):

- `SINGLE` — passthrough (one orchestrator-visible call)
- `ADDITIVE` — product across calls (multiple calls together complete
  the picture; strict — any 0 zeros the category)
- `ALTERNATIVES` — max across calls (each call is an alternative way
  of finding the trait; matching any one is sufficient)
- `CONSENSUS` — geometric mean across committed calls (same `_CONSENSUS_FOLD_FLOOR
  = 0.1` as FACETS but within one category rather than across categories).
  Used for `SENSITIVE_CONTENT` to require multi-endpoint agreement and prevent
  a single endpoint spike from over-promoting. A [0.95, 0.05, 0.05] commit
  drops from 0.95 (ALTERNATIVES max) to ~0.21 (CONSENSUS geomean).
- `NO_OP` — category never fires (e.g. `BELOW_THE_LINE_CREATOR`'s
  `EXPLICIT_NO_OP` bucket); skipped by across-category max

The 43-member assignment is locked in `schemas/trait_category.py`:
27 SINGLE, 11 ADDITIVE, 3 ALTERNATIVES, 1 CONSENSUS, 1 NO_OP. Rationale per
category lives in `search_improvement_planning/rescore_overhaul.md`.

Phase D additionally skips categories whose handler emitted zero
generated_specs before they reach `combine_calls` (prevents an
abstaining category from zeroing a FACETS-PRODUCT trait).

### Across-category combine

Branches on the trait's `combine_mode` (committed by Step 3 in
`schemas.enums.TraitCombineMode`):

- `SOLO` — one category cleanly covers all dimensions; Stage 4
  passthrough (the category score IS the trait score). Orchestrator
  already trimmed category_calls to one, so Stage 4 raises a warning
  if >1 scores arrive.
- `FRAMINGS` — categories are alternative homes for the same
  underlying thing; matching any one is sufficient evidence of the
  criterion. `trait_score = max(category_score_j)` (NO_OP categories
  skipped). Identity-style traits (Marvel = STUDIO_BRAND ∨
  FRANCHISE_LINEAGE; Christmas = SEASONAL_HOLIDAY ∨ NARRATIVE_SETTING)
  commit FRAMINGS.
- `FACETS` — categories cover different axes of a compound concept;
  ALL facets must fire to a degree for the criterion to be met.
  `trait_score = geomean([max(s, 0.1) for s in scores])` — geometric
  mean with floor 0.1 per score so a single-zero on a 2-cat trait
  scores ~0.316 instead of collapsing to zero. Compound aesthetic /
  cultural concepts (bro movie, cottagecore, dark gritty) commit FACETS.

The fold lives in `combine_categories(combine_mode, category_scores)`
in `stage_4_execution.py`. Empty `category_scores` returns 0.0 in
all modes (a trait whose every category went silent contributes
nothing). `TraitWithEndpoints.combine_mode` defaults to FRAMINGS so
failure paths (Step 3 errors) and test mocks land on V3-equivalent
MAX behavior.

### Branch-level scoring

```
final_score(movie) = Σ over traits of (trait_score × trait_weight × polarity_sign)

trait_weight = commitment_multiplier × rarity_factor      (pure-generator traits)
             = commitment_multiplier × 1.0                 (mixed / pure-reranker / negative)
polarity_sign = +1 (positive)  or  -1 (negative)
```

Commitment multipliers: `required=3.0 / elevated=1.75 / neutral=1.0 /
supporting=0.6 / diminished=0.35`. Rarity tiers (corpus N≈150K):
`<0.1%→1.5 / <1%→1.2 / <10%→1.0 / <30%→0.75 / else 0.5`.

### Trait classification (rarity bookkeeping only)

Traits are classified by the operation_type of their calls:
**pure-generator** (every call is CANDIDATE_GENERATOR), **mixed**
(at least one of each), or **pure-reranker** (every call is
POOL_RERANKER). Rarity weighting only applies to pure-generator
traits — mixed and pure-reranker traits get `rarity_factor = 1.0`.

### Implicit-prior post-rerank

`search_v2.implicit_expectations` runs after Step 2 and consumes
`QueryAnalysis.intent_exploration` plus the committed Step-2 traits.
It does not rediscover criteria from the raw query. Its output records
per-trait prior-axis evidence, an explicit ordering-axis analysis,
query specificity / prior room, and final quality/popularity
`PriorDecision`s.

The full orchestrator applies active priors after Stage 4 base scoring.
**Axis selection**: if `popularity_prior.direction != "none"` and
`popularity_cap > 0`, use popularity only; otherwise (popularity
inactive) fall back to quality alone.

```
boost = popularity_cap * popularity_signal   (or quality_cap * quality_signal)
prior_base = positive_total if positive_total > 0 else 1.0
boosted_score = base_score + prior_base * boost
```

Boost caps are `quality: 0 / 0.025 / 0.06 / 0.10` and
`popularity: 0 / 0.05 / 0.12 / 0.20` for
`none / light / normal / strong`. Both signals use the same sigmoid
curves as the metadata endpoint (`score_reception_prior()` /
`score_popularity_prior()` wrappers in `metadata_query_execution.py`).
Missing reception or popularity data has no effect (returns 0.0).
The neutral `prior_base = 1.0` fallback lets pure-negative queries
still use implicit priors for ordering without magnifying negative
penalties.

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

`CHRONOLOGICAL` is intentionally **excluded** from the authoritative
set — CHRONOLOGICAL+METADATA routes as POOL_RERANKER for positive
polarity too (never reaches the G partition), and approximate
phrasings like "old" / "recent" are better served by the fuzzy OR.

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
confirmed-zero. A single transient endpoint failure can no longer
zero out the gate or saturate the noisy-OR.

### Auxiliary specs (NEUTRAL_SEED + shorts exclusion)

`auxiliary_endpoint_specs` carries up to two entries from upstream:
the reranker-only fallback's `NEUTRAL_SEED` and the default
shorts-exclusion `MEDIA_TYPE`. Both are applied during Phase B:

1. **Shorts subtraction** — fetched once into a set by
   `_fetch_shorts_ids`, then re-applied via `_apply_shorts_subtraction`
   after the initial generator dispatch AND after each iteration of
   the filter-active tiered promotion loop (set difference is
   idempotent; the fetch is not repeated). Blocklist semantics —
   never a positive contribution.
2. **Neutral seed** — two trigger paths share the same fetch
   (`fetch_neutral_reranker_seed_ids` via the unified `_seed_neutral_pool`
   helper):
   - *Aux-spec gated path* (`_seed_from_neutral`): fires when **zero
     positive generators were attempted pipeline-wide** and the
     orchestrator's pre-execution fallback added a `NEUTRAL_SEED`
     aux spec for the branch.
   - *Filter-active direct path*: fires when a hard filter is active,
     generators were attempted, and the tiered promotion loop
     exhausted with an empty union. The orchestrator can't predict
     this case so no aux spec is present; `_run_branch` invokes
     `_seed_neutral_pool` directly.

   Both paths use the same DB-module constants
   `NEUTRAL_RERANKER_SEED_LIMIT` (2000),
   `NEUTRAL_RERANKER_SEED_POPULARITY_WEIGHT = 0.8`,
   `NEUTRAL_RERANKER_SEED_RECEPTION_WEIGHT = 0.2`.

`CANDIDATE_FLOOR = 25` is the soft target for the tiered promotion
loop. Below it the loop promotes the next tier; meeting or exceeding
it terminates the loop. Tiers and their category memberships live in
`search_v2/promotion_tiers.py` (extracted from
`full_pipeline_orchestrator.py` to avoid an import cycle with Stage 4).

### Score breakdown and diagnostics

`BranchRankedResults` carries:
- `ranked: list[tuple[movie_id, score]]`
- `score_breakdowns: dict[int, ScoreBreakdown]`

`ScoreBreakdown` has `positive_total`, `negative_total`,
`implicit_prior_boost` (boost fraction applied post-rerank), and
`trait_contributions: list[TraitContribution]`.

`TraitContribution` has `surface_text`, `commitment`, `contribution`,
`trait_score`, `weight`, and `category_scores: list[CategoryScore]`.

`CategoryScore` has `category_name`, `combine_type`, `score`,
`expressions`, `retrieval_intent`, and `endpoint_scores: list[EndpointScore]`.

`run_orchestrator.py` (at the **repo root**, not under `search_v2/`)
renders a four-level tree: result header →
trait → category (expressions + retrieval_intent + cat_score) →
endpoint (route + score). It runs `run_full_pipeline` with
`skip_bypass_steps_0_1=True`.

## Back-End Stages (Stage 3 / Stage 4)

Stage 3 owns endpoint translation + execution (10 endpoints).
Stage 4 (`stage_4_execution.py`) owns recursive scoring + ranking.

```
Stage 3 (endpoint_fetching/): Endpoint translation + execution
  ├── entity_query_generation.py + entity_query_execution.py
  ├── metadata_query_generation.py + metadata_query_execution.py
  ├── award_query_generation.py + award_query_execution.py
  ├── franchise_query_generation.py + franchise_query_execution.py
  ├── keyword_query_generation.py + keyword_query_execution.py
  ├── studio_query_generation.py + studio_query_execution.py
  ├── semantic_query_execution.py
  ├── trending_query_execution.py    (deterministic, no LLM call)
  ├── media_type_query_execution.py  (deterministic; MEDIA_TYPE category
  │                                   routes via deterministic phrase-matcher
  │                                   in category_handlers/media_type_router.py;
  │                                   no LLM handler)
  └── chronological_query_execution.py  (wired via endpoint_executors.py;
                                         spec flows through HandlerResult.preference_specs;
                                         ChronologicalQuerySpec is a bespoke
                                         EndpointParameters subclass)
  All executors return EndpointResult (list[ScoredCandidate]).

  category_handlers/ — handler scaffolding for the step-3-routed path:
    handler.py, generated_endpoint_spec.py, prompt_builder.py,
    endpoint_registry.py, schema_factories.py, output_extractor.py,
    media_type_router.py, prompts/
```

`AwardQueryPlan` (award endpoint) supports one or more `AwardSearch`
entries with `AwardCombineMode` (`any` / `average`); searches run
concurrently via `asyncio.gather`.

## Key Files

| File | Purpose |
|------|---------|
| `step_0.py` | Flow routing. `run_step_0()` returns `Step0Response`. Routes to one of seven entity flows plus optional standard co-fire. SimilarityFlowData now carries `list[SimilarityReference]`. |
| `step_1.py` | Spin generation. `run_step_1()` returns `Step1Response` (an `exploration` scratchpad + `spins`: exactly two `Spin(query, ui_label)`). Gemini 3.5 Flash with `thinking_level="minimal"`. |
| `step_2.py` | Query analysis. `run_step_2()` returns `QueryAnalysis` (intent_exploration + atoms + traits). Model: `gemini-3.5-flash` with `thinking_level="minimal"`, temperature 0.35. |
| `step_3.py` | Trait decomposition. `run_step_3(trait, siblings=None)` returns `TraitDecomposition` (includes trait_restatement + combine_mode). OpenAI `gpt-5.4-mini` with `reasoning_effort="low"`, `verbosity="low"` (locked, no caller override). |
| `run_step_0.py` | CLI runner for step_0. Conditionally invokes entity-flow executors and prints results. |
| `run_step_1.py` | Smoke-test runner for Step 1. Default query exercises the dominant-anchor path. |
| `run_step_0_batch.py` | Batch runner; writes per-query files to `step_0_results/`. |
| `run_test_queries.py` | Batch runner for Step 2+3 over test_queries.md queries with asyncio.Semaphore concurrency. |
| `run_specs.py` | Diagnostic runner: Step 2→3→handler-LLM end-to-end, outputs combine_mode / combine_type / fired endpoints / keyword commits. |
| `run_query_generation.py` | CLI wrapper: raw query → Step 2 → Step 3 fan-out → per-handler query-generation stage, printing each stage's payload. |
| `run_handler_call.py` | Focused per-handler tester. Skips Steps 2/3 and injects a `CategoryCall` directly — for iterating on the body-authoring side of the semantic/keyword/metadata handlers without Step 2/3 variability. |
| `full_pipeline_orchestrator.py` | End-to-end orchestrator. `run_full_pipeline()` runs Steps 0+1→Step 2→Step 3→handler-LLM→Stage 4. |
| `streaming_orchestrator.py` | **Production HTTP API entry point.** `stream_full_pipeline()` runs Steps 0/1, derives the branch plan + entity flow-data, then delegates to the shared `_stream_from_branch_plan()` (build fetches → launch tasks → merge loop), emitting Server-Sent Events to `/query_search`. `stream_rerun_pipeline()` is the second public entry: it accepts a pre-built `RerunPlan` (branch plan + entity flow-data) + filters and replays `_stream_from_branch_plan()` directly, **bypassing Steps 0/1** — backs `/rerun_query_search`. `BranchKind` is identity/label-only (not load-bearing in scoring). |
| `stage_4_execution.py` | Stage 4 execution + ranking. 5-phase pipeline. |
| `exact_title_search.py` | Exact-title entity-flow executor. 7-tier scoring (includes close-title-match tier 6.5). |
| `similar_movies.py` | Similarity entity-flow executor. V3.4+ lane architecture. |
| `character_franchise_search.py` | CHARACTER_FRANCHISE entity-flow executor. 7-tier reranking. |
| `non_character_franchise_search.py` | NON_CHARACTER_FRANCHISE executor. Primary+secondary franchise buckets. |
| `studio_search.py` | STUDIO entity-flow executor. LLM translation + popularity sort. |
| `person_search.py` | PERSON entity-flow executor. Role-agnostic union across actor/director/writer/producer/composer postings. 4-bucket prominence (actor billing only) + (overlap_count DESC, popularity DESC) within-bucket sort. |
| `actor_zones.py` | Sqrt-adaptive cast-zone primitives shared between person_search (actor table) and entity_query_execution. |
| `popularity_sort.py` | Shared popularity-sort helper for entity-flow executors. |
| `implicit_expectations.py` | Implicit-prior state management (quality/popularity priors). |
| `vague_temporal_vocabulary.py` | Shared vague-time/duration mappings injected into Steps 2/3 and metadata handler. |
| `attribute_search.py` | Backs `POST /attribute_search`. Deterministic hard-attribute browse; sums Step-0 person-prominence bucket weights across people. No LLM/vector. |
| `title_search.py` | Backs `GET /title_search`. Deterministic title typeahead over `movie_card.title_normalized` (two exact tiers + `pg_trgm` fuzzy fallback). No LLM/vector/Redis. |
| `query_input_validation.py` | Single source of truth for `query`/`clarification` sanitization (strip, non-empty, length cap); used by the API, both orchestrators, and Steps 0/1. |

## Step 2 / Query Analysis Design

Step 2 (`search_v2/step_2.py` + `schemas/step_2.py`) combines Stages
1+2 of the trait decomposition pipeline into a single LLM call:

| Layer | Purpose |
|-------|---------|
| Atoms | Query analysis (holistic read + atomization with evaluative intent) |
| Traits | Committed layer: role/polarity/commitment/relationship per atom |

Key design decisions:
- **`intent_exploration` as primary role-evidence source.** Replaces
  `holistic_read`. Promoted to the first field the LLM generates;
  feeds all downstream role/relationship decisions as the primary
  frame rather than one-of-three evidence sources.
- **Five-level `commitment` axis** (required/elevated/neutral/
  supporting/diminished). Replaces `role` (carver/qualifier) + `salience`.
  Explicit signals occupy extremes; structural signals fill inner buckets.
- **`relationship_role`** (INDEPENDENT / POSITIONING_REFERENCE /
  POSITIONING_QUALIFIER) plus paired `replaces_axis` /
  `axes_replaced_by_siblings`. Drives cross-trait scope-replacement
  at Step 3 without leaking sibling interpretive prose.
- **Evaluative intent is a faithful restatement** bounded by
  surface_text + modifying_signals. Near-paraphrase is correct when
  no signals exist; inference license is restricted to integrating
  signal effects.
- **Unified `modifying_signals` list per atom.** Both adjacent
  qualifiers and cross-criterion modifiers land on the same list.
  Each entry is `surface_phrase` (verbatim user text) + `effect`
  (freeform concise description).
- **`split_exploration` / `standalone_check`** are pure evidence-
  gathering fields (no embedded verdict). Commit phase acts on them.
- **Surface-text discipline preserved.** `surface_text` is exact
  substring of the query with modifying language stripped.

## Step 3 Trait Decomposition Design

Step 3's key discipline rules:
- **`trait_restatement` is the first output field.** Verbatim quotes
  of contextualized_phrase + evaluative_intent + role fields. Every
  downstream Step-3 field must trace clauses to content in those quotes.
- **`Dimension.expression` = verbatim copy of one aspect string.**
  The dimension layer routes; it does not re-author. Character-for-
  character diff equality against the aspects list is the test.
- **Aspects enumerate non-overlapping, comprehensive PARTS.** The
  whole trait restated alongside its parts is not a valid aspect list
  — the parts alone are. Described plot/story shapes are kept whole:
  a story's character archetypes, events, and relations are co-varying
  parts of one concept, not independently-varying aspects.
- **Minimum-viable call set.** `consolidation_analysis` places the
  trait on the breadth↔single-shape spectrum and selects the fewest
  calls: fold finer parts into a broader call rather than spawning
  brittle separate facets. `CandidateFit.LIKELY_DISREGARD` marks
  candidates that can only describe the trait by describing an
  absence — inclusion-ineligible regardless of how many calls there are.
- **Aspect width preservation.** Aspects must preserve the user's
  stated constraint at the same granularity — no implicit broadening
  (widening "winning" to "won or nominated"), narrowing, or invented
  conditions.
- **Example-in-expression rule.** Expressions and retrieval_intent
  contain named entities only when the trait named them; never
  model-supplied exemplars (no "such as X, Y, or Z" clauses).
- **Identity-vs-attribute rule.** For POSITIONING_REFERENCE traits,
  only attribute categories are valid (what the entity is LIKE).
  Identity categories (PERSON_CREDIT, TITLE_TEXT, STUDIO_BRAND,
  FRANCHISE_LINEAGE, etc.) are off-limits for qualifiers.
- **Category-aware decomposition.** Decompose only as deep as distinct
  categories require; one category covering the trait → one dimension.

## Category Taxonomy

`schemas/trait_category.py` houses `CategoryName` — the canonical
43-category vocabulary. Each member carries `description`, `boundary`,
`edge_cases`, `good_examples`, `bad_examples`, `endpoints` tuple,
`HandlerBucket` enum, and `combine_type`. The full taxonomy and split
rationale live in `search_improvement_planning/query_categories.md`.

Notable taxonomy changes since ADR-078:
- `TARGET_AUDIENCE` flipped to `CategoryCombineType.ALTERNATIVES` (was ADDITIVE)
  so a single endpoint score suffices rather than requiring KW × META × SEM product.
- `SENSITIVE_CONTENT` switched to `CategoryCombineType.CONSENSUS` (geometric
  mean over committed endpoints) to require multi-endpoint agreement and prevent
  a single SEMANTIC spike from over-promoting content-flagged movies. Description
  reframed to presence/intensity of mature content only — the category never
  indexes absence. "Safe for kids" / avoidance asks route to `TARGET_AUDIENCE`.
- `MATURITY_RATING` description loosened to allow sparing proxy use (e.g.
  "safe for kids" → G/PG ceiling) when it is the best available inclusion
  signal; fires on an explicit rating OR a clear suitability proxy, never
  as a default routing target.
- `EMOTIONAL_EXPERIENTIAL`, `SEASONAL_HOLIDAY`, `SPECIFIC_PRAISE_CRITICISM`
  re-bucketed to `SINGLE_NON_METADATA_ENDPOINT` with semantic-only endpoint
  (was SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT + KEYWORD).
- `CENTRAL_TOPIC` and `ELEMENT_PRESENCE` re-bucketed to
  `SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT` (was
  PREFERRED_REPRESENTATION_FALLBACK); keyword fires only when a single
  member cleanly covers the subject — no ANY-union stitching.
- `CHRONOLOGICAL` moved to `SINGLE_NON_METADATA_ENDPOINT` with a
  bespoke `ChronologicalQuerySpec` (direction + is_top_n_active + top_n).
- Category definition audit (A1–A4, B1/B3/B6): `STORY_THEMATIC_ARCHETYPE`
  owns the whole-film elevator-pitch shape; `PLOT_EVENTS` owns a beat
  within a larger story; `CHARACTER_ARCHETYPE` is a static type (no arc);
  `ELEMENT_PRESENCE` is explicit fallback only. `COUNTRY_OF_ORIGIN` /
  `CULTURAL_TRADITION` / `FILMING_LOCATION` / `NARRATIVE_SETTING` each have
  a "how to tell" discriminator test. `TARGET_AUDIENCE` = explicitly named
  audience (people); `VIEWING_OCCASION` = occasion not defined by named people.
  Reception categories (NUMERIC / GENERAL_APPEAL / CULTURAL_STATUS /
  SPECIFIC_PRAISE_CRITICISM / AWARDS) have crisp generalized splits.
- Step-3 fit criterion added: candidates are ranked against their own
  taxonomy entry's definition; `CLEAN_OWNERSHIP` requires the category's
  definition to squarely claim the aspect.

## Endpoint Contract

All endpoint executors follow the same dual-mode signature:
- `restrict_to_movie_ids=None` → dealbreaker mode: generates own candidate pool
- `restrict_to_movie_ids=set[int]` → preference mode: scores exactly the supplied ids
- `restrict_to_movie_ids=set()` → short-circuit: returns empty `EndpointResult`

Failures retry once; second failure returns empty `EndpointResult` (soft failure).

`build_endpoint_coroutine` takes a `GeneratedEndpointSpec` (not
separate route/wrapper) and applies three pre-dispatch gates:
1. POOL_RERANKER + falsy restrict → empty result
2. params is None → log warning + empty result
3. CANDIDATE_GENERATOR → locally rebind restrict=None

**TRENDING special-case in `_dispatch_call`**: when `spec.route is
EndpointRoute.TRENDING`, `_dispatch_call` calls `execute_trending_query`
directly instead of routing through `build_endpoint_coroutine` (which
rejects params=None). Mirrors the same special-case in
`category_handlers.handler._dispatch_one_spec`.

## Dealbreaker Score Floor

All endpoint executors emit dealbreaker scores in `[0, 1]` (raw, not
compressed). The old `compress_to_dealbreaker_floor(raw) = 0.5 + 0.5 * raw`
helper is retained in `result_helpers.py` but is no longer used by
the semantic or metadata executors.

**Exception: `semantic_query_execution.py` emits raw `[0, 1]` on every
path** — role is now LLM-committed inside `SemanticParameters.role`
(`SemanticRetrievalShape.CARVER` / `QUALIFIER`); executor dispatches
internally. Carver uses equal-vote multi-space scoring with corpus-
calibrated elbow; qualifier uses weighted-sum with CENTRAL=2.0 /
SUPPORTING=1.0.

**Metadata executor** (`metadata_query_execution.py`) uses
threshold-anchored sigmoid curves for popularity and reception scoring.
Public `score_popularity_prior()` and `score_reception_prior()` wrappers
allow the implicit-prior post-rerank to use the same curves.

## Key Patterns

- **Positive-presence invariant**: all step-3 endpoint specs express
  presence of an attribute, not absence. Exclusion direction is
  carried by `ActionRole` / `Polarity` in the category handler layer.
- **SOLO trim at orchestrator**: if combine_mode=SOLO and LLM emitted
  >1 category_calls, orchestrator keeps only [:1] before handler fan-out.
- **Sibling-task context injection**: handler user message includes
  a `<sibling_categories>` block listing parallel category retrieval_intents.
- **Concept-level aggregation in Stage 4**: expressions from the same
  concept contribute one inclusion max and one exclusion max.
- **Token-index resolution**: franchise and award endpoints resolve
  LLM surface forms via posting-list helpers. Empty resolution →
  empty result, not a broadened query.
- **Implicit-prior split**: popularity-only when active, quality
  fallback when popularity is inactive (not 50/50 sum).
- **Shorts promotion**: `ingest_movie.py` forces `release_format=SHORT`
  when `runtime_minutes <= 40`, regardless of IMDB title-type.

## Hard Filters

`MetadataFilters` (from `implementation/classes/schemas.py`) is threaded
through every V2 pipeline primitive at retrieval time:

- **Applied at primitive, not pre-resolved IDs.** Each Postgres and Qdrant
  primitive folds conditions into its own WHERE / payload-Filter.
- **Similarity branch threaded with filters.** Both `run_similarity_search`
  (Step-0 `similarity_to_titles`) and the standalone
  `run_similar_movies_for_ids` (backing `/similarity_search`) accept a
  `metadata_filters` arg that flows into every candidate-generation lane
  (Postgres director / franchise / studio / source / quality / themes /
  rare-medium) and into the Qdrant shape search via `build_qdrant_filter`.
  When the filter is active the shape lane's `qdrant_limit` and the
  quality lane's `LIMIT` are both 2×'d so the post-filter pool stays
  usefully sized (filtered HNSW returns fewer points per traversal step,
  and the quality LIMIT is applied before the filter on the SQL side).
- **Shorts subtraction not filtered.** Shorts are a blocklist; applying the
  filter to them would let out-of-range shorts survive as candidates.
- **Semantic elbow calibration unfiltered, candidate pool filtered.**
  `_run_corpus_topn` is unconditionally unfiltered — the elbow is an
  absolute "is-about-X" bar that's a property of the global corpus,
  not the filtered slice; calibrating against a filtered slice would
  inflate the elbow toward the filtered noise floor and manufacture
  false positives. `_run_corpus_topn_filtered` is the separate filtered
  probe used to populate the candidate pool on carver-unrestricted /
  qualifier-promoted generator paths. In the no-filter case those
  paths fire a single probe per space (byte-identical to pre-fix
  behavior); in the filter-active case they fire calibration and pool
  probes in parallel per space.
- **Trending via Postgres round-trip.** `execute_trending_query` calls
  `fetch_movie_ids_matching_filters` to intersect the Redis hash against
  an eligible-movie-card scan in one Postgres round-trip.
- **No-filter regression.** When filters is None, helpers return `("", [])`
  and the query plan is byte-identical to pre-filter behavior.

## Observability

The `/query_search` streaming path (`streaming_orchestrator.py`) carries manual
OTel spans through most of the pipeline: `query_search.step_0`/`.step_1`
(flow routing + spin generation), one `query_search.branch` span per fetch
(also emitted on rerun, which shares `_stream_from_branch_plan`), the
standard-branch trait pipeline (`full_pipeline_orchestrator.py` /
`category_handlers/handler.py`: `.step_2`/`.trait`/`.step_3`/`.query_generation`),
the six entity-flow executors' `branch_*` attributes + per-flow child spans,
the Stage 4 execution spans (`stage_4_execution.py`: `query_search.generators`
/ `.promotion` / `.neutral_seed` / `.rerankers` / `.negatives` and a per-call
`.dispatch` span, plus fallback/dedup/shorts/timeout events — the pool-definition
+ reranker/negative-dispatch half; scoring/aggregation is a later bite), and
manual Qdrant probe spans (`semantic_query_execution.py`'s
`query_search.semantic_qdrant`, `similar_movies.py`'s `similarity_qdrant`) that
close the gRPC auto-instrumentation gap. Every routed LLM call
(`implementation/llms/generic_methods.py`) nests a shared `llm.generate` span
under whichever step/branch/handler span is current, so cost/tokens/retries
attribute automatically to the right stage. This module never imports the OTel
SDK directly for its own logic — spans wrap existing call sites without
changing return values or control flow. Source of truth for the full span/
attribute catalog: `observability_context/observability_architecture.md`;
summary: `docs/modules/api.md`'s Observability section.

## Interactions

- `schemas/` — step-0/1/2/3 schemas, category taxonomy, endpoint
  translation schemas, `EndpointResult` / `ScoredCandidate`.
- `db/postgres.py` — posting-list reads for franchise, award, entity,
  studio; materialized views for director strength, franchise
  confidence, trait IDF; similarity signal rows.
- `db/qdrant.py` — semantic endpoint vector searches; `query_batch_points`
  for 8-space shape searches.
- `db/redis.py` — trending endpoint hash reads.
- `implementation/llms/generic_methods.py` — shared LLM router for all
  generator modules. Clients are lazy-constructed via `_LazyClient`
  proxy (first attribute access triggers build).

## Gotchas

- **`step_2.py` is NOT the old Stage 2A/2B.** `stage_2a.py` and
  `stage_2b.py` are deleted.
- **`schemas/step_2.py` holds `QueryAnalysis`.** `holistic_read` has
  been replaced by `intent_exploration`. The old types
  (`AbsorbedModifier`, `IncomingModification`, `AbsorbedModifierKind`,
  `ModificationDepth`, `Step2Response`) are deleted.
- **`Trait.role` (carver/qualifier) is removed.** It is replaced by
  `commitment` (five-level) + `relationship_role` (structural triplet).
  `schemas/enums.py:Role` is retained only for legacy consumers of
  `semantic_query_execution.py`; the semantic schema now uses
  `SemanticRetrievalShape` (LLM-committed per call).
- **Category taxonomy lives in `schemas/trait_category.py`, not
  `schemas/enums.py`.** 43 active members.
- **`stage_4/priors.py` is deleted.** Priors live in the orchestrator.
- **Stage 4 lives in `stage_4_execution.py`, not `stage_4/`.** The
  legacy `stage_4/` directory is superseded.
- **Negative-trait `operation_type` is uniformly `POOL_RERANKER`.**
  Stage 4 re-derives would-be type via
  `determine_operation_type(category, route, Polarity.POSITIVE)`.
- **`stage_3/` is deleted.** Active code lives in
  `search_v2/endpoint_fetching/`. Any import from `search_v2.stage_3`
  will fail. Only `unit_tests/` still imports from there (untouched
  per test-boundary rule).
- **`CarverSemanticEndpointParameters` / `QualifierSemanticEndpointParameters`
  are deleted.** Replaced by unified `SemanticEndpointParameters` with
  a `role_exploration` + `role` (`SemanticRetrievalShape`) commit.
- **`EntityEndpointParameters` union wrapper is deleted.** Each entity
  category (`PERSON_CREDIT`, `NAMED_CHARACTER`, `TITLE_TEXT`) now
  emits its narrowed spec directly.
- **`ChronologicalQuerySpec` uses bespoke `EndpointRoute.CHRONOLOGICAL`.**
  Executor is fully wired via `endpoint_executors.py` (`execute_chronological_query`);
  spec flows through `preference_specs` and is dispatched by route.
- **Stoplist asymmetry for awards**: ingest writes every token; query
  drops `AWARD_QUERY_STOPLIST`. Intentional — lets the droplist be
  revised without re-ingesting.
- **LLM client construction is lazy.** `_LazyClient` proxy in
  `generic_methods.py` builds underlying clients on first attribute
  access. Proxy `__getattr__` is not thread-safe (documented); safe
  under single-event-loop deployment.
- **api/requirements.txt must be manually updated** when new deps are
  added via `uv add`. The API Dockerfile uses requirements.txt rather
  than `uv export` to avoid transitive bloat. Docker base image is
  `python:3.13-slim`.
