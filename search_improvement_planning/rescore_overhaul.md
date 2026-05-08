# Rescore Overhaul

Conceptual plan for reworking how scoring composes across calls,
categories, and traits in the Stage 4 execution layer. Written from
the perspective of "what should happen and why." Implementation
shape (which functions move where, which existing code paths
collapse) is downstream of this doc.

## Problem this fixes

Today's classification rule in
[stage_4_execution.py](../search_v2/stage_4_execution.py)
treats a trait as candidate-generating if any of its sub-calls is a
positive-polarity pool finder. The mix-resolution rule then scopes
that trait's reranker calls to the trait's own narrow generator pool.

This breaks on the common case where a trait that is intuitively a
qualifier (modifies the population other traits define) happens to
contain at least one membership-style sub-call. The reranker calls
inside it never see the candidates produced by the actual
population-defining traits — they only ever score within their own
sub-pool, missing most of the relevant union.

Concrete instance: in **"dark, gritty marvel movies"**, "dark" might
decompose to a KEYWORD generator plus semantic rerankers. The
KEYWORD makes the trait candidate-generating by today's rule. The
semantic rerankers then score only the keyword-matched movies, never
the marvel candidates produced by the marvel trait. The user
explicitly wants those rerankers to evaluate the marvel pool.

The fundamental fix: separate **candidate-set definition** from
**per-trait scoring**, and let every reranker call score the full
candidate set regardless of which trait it lives in.

## Fix in one paragraph

Run all positive-polarity candidate generators across all traits to
produce a single union of candidate IDs (with auxiliary shorts
subtraction and the neutral-seed fallback applied as today). Run
all positive-polarity rerankers against that finalized union, not
against trait-local pools. Then score every candidate against every
trait via a hierarchical combine — per-category combine first
(declared on the CategoryName enum: additive, alternatives, or
single), then across-category combine via max. Weight each trait
by `commitment × (rarity_factor if pure-generator else 1.0)`.
Negative polarity stays on the existing gate × fuzzy formula.

## Core principles

1. **Concrete logic over LLM judgment** wherever the data already
   determines the answer. The pure-generator vs mixed vs pure-reranker
   classification is fully derivable from the sub-call types Step 3
   and the handler already produce — no new LLM-emitted role field.
2. **One trait = one criterion** (Step 2's atomization rule).
   Multiple categories within a trait are different framings of the
   same criterion, not facets of a compound concept. This drives the
   across-category combine to max.
3. **Bounded [0,1] at every aggregation level.** Per-call, per-
   category, per-trait. No level can produce a number above 1.0.
4. **Soft scoring everywhere.** No positive trait categorically
   excludes a candidate. The single hard exclusion is the auxiliary
   shorts subtraction, which is a default user-preference filter, not
   a trait.
5. **Polarity asymmetry preserved.** Positive polarity uses the
   combine described here; negative polarity continues to use the
   three-bin gate × fuzzy formula already in
   [stage_4_execution.py](../search_v2/stage_4_execution.py).
6. **Endpoint internals belong to the executor.** "One call" at the
   orchestrator's level is one endpoint invocation — regardless of
   how many internal targets (vector spaces, entity list members,
   keyword groups) the call covers. SEMANTIC across multiple vector
   spaces is one call, not many. The orchestrator consumes a single
   [0,1] score per call and never looks inside. This boundary is what
   makes the within-category combine modes (additive / alternatives /
   single) tractable — the modes describe how *orchestrator-visible*
   calls compose, not how an executor stitches its own internals.

## Pipeline phases

The full pipeline gains a sharper separation between pool definition
and scoring. Steps 0–3 plus the handler-LLM remain unchanged.

### Phase A — LLM generation (unchanged)

Steps 0/1/2/3 plus the per-CategoryCall handler-LLM produce the
trait set with per-call generated specs. This phase is governed by
[full_pipeline_orchestrator.py](../search_v2/full_pipeline_orchestrator.py)
and is out of scope for this overhaul.

### Phase B — Pool definition

1. Run every **positive-polarity candidate-generator call** across
   every trait in parallel. Each call returns a `dict[movie_id,
   score]` for the movies it matched.
2. Cache the per-call score maps. Do not aggregate them into
   per-trait scores yet — they are reused in Phase D.
3. Take the union of all generator-call IDs into a single candidate
   set.
4. Apply the auxiliary **shorts subtraction**: remove any movie
   whose release format is SHORT, unless the user explicitly
   expressed a media-type preference (existing
   `_build_shorts_exclusion_spec` logic).
5. **Neutral-seed fallback**: only when the entire query produced
   zero positive-polarity candidate-generator calls (i.e., every
   trait is pure-reranker and tier-fallback promotion did not turn
   up a promotable category). When this fires, seed with the top-N
   blended popularity/reception movies as today; the seed's scores
   are not used in scoring (implicit priors handle quality and
   popularity contribution).
6. **Empty result is empty result.** If at least one generator call
   was attempted but the union after shorts subtraction is empty,
   return no results. Do not seed in this case — empty means the
   user's stated constraints excluded everything in the corpus, and
   the right behavior is to say so.

### Phase C — Reranker pass

Run every **positive-polarity reranker call** across every trait
against the finalized candidate ID set. Each call returns a
`dict[movie_id, score]` covering every movie in the union (or a
subset; missing movies are treated as 0 for that call).

This is the load-bearing change from current behavior: rerankers no
longer scope to trait-local pools. A semantic reranker inside trait
A scores every candidate, including those brought in only by trait
B's generators.

Cache the per-call score maps for Phase D.

### Phase D — Per-trait scoring

For each candidate in the union, for each trait in the query:

1. **Per-call scores.** Look up the candidate's score for each of
   the trait's calls from the cached score maps. Calls that didn't
   fire on this candidate contribute 0. Failed calls drop entirely
   (do not count as 0; do not count as anything).
2. **Per-category combine.** Apply the category's declared combine
   rule (additive/alternatives/single, see below) to the call
   scores within that category to produce one per-category score
   in [0,1].
3. **Across-category combine.** Take max over the per-category
   scores to produce trait_score in [0,1].
4. **Trait weighting.** Compute trait_weight (see below) and apply
   polarity sign:
   - Positive polarity: contribution = `+ trait_score × trait_weight`
   - Negative polarity: handled separately via gate × fuzzy
     (unchanged from current implementation)

### Phase E — Branch aggregation and implicit priors

`final_score(candidate) = Σ trait_contribution` across all traits
for that candidate, with negative polarity contributions computed
per the existing gate × fuzzy formula and applied as a downward
push.

Then apply the multiplicative implicit-prior boost (existing
[implicit_expectations.py](../search_v2/implicit_expectations.py)
logic, unchanged).

## Per-call score conventions

Every call — generator or reranker — returns scores in [0,1]:

- **KEYWORD, FRANCHISE, STUDIO, AWARD, TRENDING, MEDIA_TYPE**:
  binary or graded membership. Today these are 1.0 on hit, 0 on
  miss, with some endpoint-specific gradient (multi-tag overlap,
  award count thresholds).
- **ENTITY (PERSON_CREDIT)**: graded by role tier (lead 1.0,
  supporting 0.7, minor 0.3, etc.).
- **METADATA range columns**: 1.0 inside requested range with soft
  exponential falloff outside.
- **METADATA prior columns**: continuous score from existing prior
  scoring logic.
- **SEMANTIC**: continuous cosine via elbow normalization.

Per-call scoring and within-call aggregation (across vector spaces
for SEMANTIC, across multiple entities for ENTITY, etc.) is the
endpoint's responsibility. The orchestrator consumes the final
[0,1] number and does not look inside the call.

## Within-category combine

Each `CategoryName` member declares a combine type. Three modes:

### ADDITIVE — multiply

`category_score = ∏ call_score_i`

For categories where multiple endpoints together complete the
picture: KEYWORD + SEMANTIC in EMOTIONAL_EXPERIENTIAL or
CENTRAL_TOPIC, both probing different aspects of the same intent.

Properties:
- Bounded [0,1] (product of [0,1] values stays in [0,1])
- Strict — any 0 zeros the category
- Gradient preserved when both signals are non-zero
  (1 × 0.95 = 0.95 differentiates from 1 × 0.3 = 0.3)
- Symmetric — neither call type is privileged

The strictness is intentional: the additive interpretation says all
sub-signals must be present for the category to fire. A movie that
fails the canonical tag (KW = 0) gets 0 on this category even with
strong gradient evidence elsewhere — the across-category max
provides the recovery path when other categories of the same trait
fire.

### ALTERNATIVES — max

`category_score = max(call_score_i)`

For categories where each endpoint is a distinct way of finding the
trait, and matching any one is sufficient evidence. The categories
this applies to are a small set; default presumption for mixed-
endpoint categories is ADDITIVE unless explicitly tagged otherwise.

### SINGLE — passthrough

`category_score = call_score`

For categories that fire exactly one endpoint by definition. The
combine rule is trivial; surfacing it as a third mode keeps the
classification exhaustive and self-documenting on the CategoryName
enum.

### What counts as one call

The combine modes apply to **orchestrator-level calls**, not to
internal endpoint operations. Each endpoint invocation is one call
from the orchestrator's perspective, regardless of how many internal
targets it scores.

The clearest case is SEMANTIC. The semantic executor takes a params
object that may target several vector spaces in a single call —
e.g., `reception_vectors` and `production_vectors` for
VISUAL_CRAFT_ACCLAIM, or `viewer_experience_vectors`,
`watch_context_vectors`, and `reception_vectors` for
EMOTIONAL_EXPERIENTIAL. All cross-space aggregation, blending, and
elbow normalization happens inside the executor and surfaces a
single [0,1] score per movie. From the orchestrator's perspective
that is one call, full stop.

The same boundary applies to ENTITY (one call against a list of
person entities), KEYWORD (one call against a list of tag groups),
and any other endpoint that accepts list-shaped specs. The
orchestrator consumes the final [0,1] number and never looks inside.

Two consequences fall out of this:

- A category whose only endpoints are semantic vector spaces — e.g.,
  VISUAL_CRAFT_ACCLAIM, DIALOGUE_CRAFT_ACCLAIM, NAMED_SOURCE_CREATOR —
  is **SINGLE-combine**, not additive across multiple semantic calls.
  Multi-vector-space concerns are not visible at the rescore layer.
- A category with KEYWORD plus semantic spaces — e.g., CENTRAL_TOPIC,
  ELEMENT_PRESENCE, EMOTIONAL_EXPERIENTIAL — has exactly two
  orchestrator-visible calls (one KW, one SEM), regardless of how
  many vector spaces SEM internally targets. These are the
  ADDITIVE-combine cases.

This is also what closes the older question about "should we merge
related semantic calls?" There is nothing to merge at the
orchestrator level — the executor already returns one score per
SEMANTIC invocation.

## Across-category combine

`trait_score = max(category_score_j)` across all the trait's
categories.

Why max universally:

- **Step 2's atomization rule** guarantees one trait = one
  criterion. Multiple categories within a trait are different
  framings of that single criterion, not facets of a compound.
- **Strongest framing wins**, others can't help or hurt. This
  matches the user-intent reading of "is this a Marvel movie?" or
  "is this a WWII movie?" or "is this dark?": match any framing
  strongly = strong evidence of the trait.
- **Noisy-OR would over-reward correlated framings.** Multiple
  framings of the same identity are correlated probes, not
  independent evidence sources; compounding them inflates the
  score above what any single framing actually justifies.
- **Average would punish missing framings.** A Marvel Studios film
  not featuring a specific Marvel character would get its score
  dragged down by a 0 on the CHARACTER framing, violating the
  "matching one is sufficient" semantics.

This is uniform across pure-generator, mixed, and pure-reranker
traits. Rarity weighting (next section) applies to the trait_score
for pure-generators; the across-category combine itself doesn't
change shape based on classification.

## Trait classification (for rarity only)

Three structural classes derived from the trait's sub-call types:

- **Pure-generator**: every call across every category is a
  candidate-generator (generator routes per the existing endpoint
  classification).
- **Pure-reranker**: every call is a reranker.
- **Mixed**: at least one of each.

This classification is used **only** for rarity bookkeeping. The
combine rules above are uniform across all three classes. The
classification does not gate which calls fire, where they fire, or
how they score individual candidates.

## Trait weighting

```
trait_weight = commitment_multiplier × rarity_factor
```

**Commitment multipliers** (unchanged from current implementation):
required 3.0 / elevated 1.75 / neutral 1.0 / supporting 0.6 /
diminished 0.35.

**Rarity factor**:
- Pure-generator traits: corpus-rarity tier as today (ULTRA_RARE
  1.5 / RARE 1.2 / MODERATE 1.0 / COMMON 0.75 / VERY_COMMON 0.5),
  computed from the union of the trait's generator-call match sets
  per the existing rule.
- Mixed and pure-reranker traits: `rarity_factor = 1.0`.

The restriction to pure-generator traits is principled. Rarity asks
"how identifying is the matched set?" That question is well-posed
when the trait's full evidence is membership-based. In a mixed
trait, the generator calls are typically narrow proxies for a
broader concept the rerankers fill out — rewarding the trait by the
narrow proxy's rarity overweights it. A 500-movie keyword set
inside a mixed trait would tag the trait ULTRA_RARE (1.5×) when the
underlying concept is broad, distorting the trait's weight relative
to its actual contribution.

## Negative polarity (unchanged)

Negative polarity continues to use the three-bin gate × fuzzy
formula already implemented in
[stage_4_execution.py](../search_v2/stage_4_execution.py):

- **G_a**: would-be candidate-generator calls in authoritative
  categories (specific-entity / structured-metadata) — combine via
  product (gate).
- **G_e**: would-be candidate-generator calls in evidential
  categories (keyword tags, archetypes, fuzzy descriptors) — combine
  with R via noisy-OR (fuzzy).
- **R**: would-be pool-reranker calls (continuous similarity /
  prior) — combine with G_e via noisy-OR (fuzzy).

`trait_score = gate × fuzzy` when both partitions present; gate
alone or fuzzy alone otherwise. Sign applied at branch aggregation.

The asymmetry between positive and negative polarity is intentional
and preserved. Positive-polarity rerank scope changes; negative
polarity does not.

## Auxiliary specs

### Shorts subtraction (unchanged)

When no trait emitted a MEDIA_TYPE call, fetch all SHORT-format
movies and subtract them from the union before reranker scoring
runs. This is the only categorical exclusion in the system.

### Neutral seed (semantics tightened)

Fires only when the entire query produced zero positive-polarity
candidate-generator calls — i.e., every trait is structurally
pure-reranker and tier-fallback promotion did not promote any
category to candidate-generator duty. In that case, seed the pool
with the top-N blended popularity/reception movies via existing
`fetch_neutral_reranker_seed_ids()` logic. Seed scores do not enter
trait scoring; quality/popularity contribution is handled by
implicit priors at branch aggregation.

If at least one positive-polarity generator ran but the union ended
up empty (either because every generator returned 0 candidates or
because shorts subtraction emptied an all-shorts pool), **return no
results**. The user's stated constraints excluded everything in the
corpus; the right behavior is to surface that, not to substitute a
neutral seed and pretend something matched. Per
[search_method_deterministic_logic.md §11](search_method_deterministic_logic.md):
"if something truly doesn't exist, then it doesn't exist."

## Branch aggregation

```
final_score(candidate) =
    Σ over positive-polarity traits ( trait_score × trait_weight )
  - Σ over negative-polarity traits ( negative_trait_score × trait_weight )
```

Negative trait scores come from the gate × fuzzy formula above.
After base scoring, apply the multiplicative implicit-prior boost
per existing
[implicit_expectations.py](../search_v2/implicit_expectations.py)
logic.

## Worked examples

### "dark, gritty marvel movies"

Three traits. Marvel is pure-generator (STUDIO + FRANCHISE alternatives;
both candidate-generator calls). Dark and gritty are pure-reranker
(EMOTIONAL_EXPERIENTIAL with SEMANTIC reranker only — at least under
the empirical Step 3 routing observed for this query; a KEYWORD call
on this category would make them mixed).

- **Phase B (pool)**: marvel STUDIO and FRANCHISE generators run;
  union ≈ MCU films (~30 movies). No shorts to subtract. Neutral
  seed does not fire (a generator ran).
- **Phase C (rerank)**: dark and gritty SEMANTIC rerankers score
  every movie in the union. **This is the bug fix** — they now see
  every marvel movie, not just keyword-matched ones.
- **Phase D (scoring)** for an actual Marvel Studios non-character-
  specific film:
  - marvel: STUDIO=1, FRANCHISE=1, alternatives → max=1.0;
    across-category trivial; trait_score = 1.0
  - dark: EMOTIONAL_EXPERIENTIAL single-call SEM=0.7;
    trait_score = 0.7
  - gritty: same shape, SEM=0.6; trait_score = 0.6
- **Phase D (weighting)**: marvel commitment elevated × rarity
  ULTRA_RARE = 1.75 × 1.5 = 2.625; dark and gritty commitment
  neutral × rarity 1.0 = 1.0 each
- **Final**: 1.0×2.625 + 0.7×1.0 + 0.6×1.0 = 3.925

A non-marvel dark gritty thriller pulled in only by another query's
generators (not applicable here — but in a query like "dark gritty
movies" with no marvel trait, the trait-set itself shifts and dark/
gritty become candidate sources via tier-fallback or their own
generator routes). Within this query, no non-marvel movie is in the
union.

### "movies about WWII"

One trait. WWII is mixed: CENTRAL_TOPIC fires KEYWORD generator
[WAR, WAR_EPIC, HISTORY] (ALL) plus SEMANTIC reranker; NARRATIVE_
SETTING fires SEMANTIC reranker.

- **Phase B**: KW generator runs, union = movies tagged all three
  keywords.
- **Phase C**: SEM(CENTRAL_TOPIC) and SEM(NARRATIVE_SETTING) score
  every movie in the union.
- **Phase D** for Saving Private Ryan:
  - CENTRAL_TOPIC additive: KW=1 × SEM=0.95 = 0.95
  - NARRATIVE_SETTING single: SEM=0.9
  - Across max: 0.95
  - trait_score = 0.95
- **Phase D** for a tangentially-WWII film tagged all three:
  - CENTRAL_TOPIC: 1 × 0.3 = 0.3
  - NARRATIVE_SETTING: 0.5
  - max: 0.5
  - trait_score = 0.5

Granularity preserved. The multiply rule for additive
within-category does not flatten KW=1 into 1.0 trait_score.

### "WWII movies starring Tom Hanks"

Two traits. WWII is mixed (as above; in this query Step 3 may drop
CENTRAL_TOPIC entirely when a stronger generator-trait sits
alongside, observed empirically). Tom Hanks is pure-generator
(PERSON_CREDIT single call).

- **Phase B**: WWII KW generator (if CENTRAL_TOPIC retained) +
  Tom Hanks ENT generator both run; union = WAR-tagged movies ∪
  Tom Hanks's filmography.
- **Phase C**: WWII semantic rerankers score every movie in the
  union, including Tom Hanks's non-WWII films.
- **Phase D** for Saving Private Ryan: high on both traits.
- **Phase D** for You've Got Mail (Tom Hanks comedy): 0 or low on
  WWII (KW miss + low SEM), 1.0 on Tom Hanks. Drops to mid-pack
  at branch aggregation.
- **Phase D** for Schindler's List (WWII without Hanks): high on
  WWII, 0 on Tom Hanks. Sits below Hanks-WWII films but above
  Hanks-comedies.

### "feel-good Christmas movies"

Two mixed traits. Christmas: SEASONAL_HOLIDAY with KW:HOLIDAY +
SEM. Feel-good: EMOTIONAL_EXPERIENTIAL with KW:FEEL_GOOD + SEM.
Both extend the union.

The "no harm no foul" property carries the result: commitment
asymmetry (Christmas elevated 1.75, feel-good neutral 1.0) ensures
Christmas movies dominate the ranking. A non-Christmas feel-good
film leaks into the union via feel-good's KW generator but scores
0 on the Christmas trait (KW miss + low SEM) and only ~1.0 from
feel-good — well below a Christmas+feel-good film at ~2.75.

## What changes vs current implementation

- **Reranker scope**: from trait-local pool to full union.
  Mechanically: rerankers run after pool is finalized in Phase B,
  not within the candidate-generating-trait subtree.
- **Within-category combine**: new per-category metadata on
  `CategoryName` (additive / alternatives / single). Per-category
  combine logic uses that metadata.
- **Across-category combine**: changes from nested equal-weight
  averaging to max.
- **Within-trait combine**: collapses; no separate "trait composite
  score" formula beyond the across-category max.
- **Rarity application**: restricted to pure-generator traits.
  Mixed and pure-reranker traits use rarity_factor = 1.0.
- **Trait classification rule**: simplified to pure-generator /
  mixed / pure-reranker derived from sub-call types. Used only for
  rarity bookkeeping.
- **Neutral seed semantics**: only fires when zero positive
  generators ran across the entire query. Empty union after
  generator execution returns empty results.

## What stays the same

- All of Phase A (Steps 0/1/2/3, handler-LLM, spec generation).
- Tier-fallback promotion (§10 of
  [search_method_deterministic_logic.md](search_method_deterministic_logic.md)).
- Auxiliary shorts subtraction.
- Negative-polarity gate × fuzzy three-bin formula.
- Implicit-prior multiplicative boost.
- Per-call scoring (endpoints own their internal aggregation).
- Commitment multipliers and rarity tier values.

## Out of scope

- Adding a Step 2 LLM-emitted role label. The classification is
  derivable from sub-call types; no new LLM judgment needed.
- Restructuring the trait/category taxonomy.
- Tuning category-level additive vs alternatives flags. Initial
  values to be set when the enum field is added; empirical tuning
  is a follow-up concern.
- Changing handler-LLM behavior. Handler-level context-awareness
  (the WWII + Tom Hanks case where CENTRAL_TOPIC was dropped
  entirely) becomes a quality-of-result bonus rather than something
  the scoring layer depends on.
- Improving keyword query breadth for additive categories. The
  upstream concern is real (see Sharp Edges below) but the response
  lives in handler-LLM spec generation, not in the rescore math.

## Sharp edges to watch

The multiply-for-additive rule is strict. A movie missing the
canonical tag in an additive category zeros that category, even
with strong gradient evidence. The across-category max provides
recovery when other categories of the same trait fire, but if every
category of a trait happens to be additive multi-call with KW + SEM,
the strictness compounds.

**This strictness is accepted as-is.** The implication is explicit
and bounded: for additive categories where KW carries the
membership signal and SEM carries the gradient — CENTRAL_TOPIC,
ELEMENT_PRESENCE, CHARACTER_ARCHETYPE, NARRATIVE_DEVICES,
STORY_THEMATIC_ARCHETYPE, plus EMOTIONAL_EXPERIENTIAL,
SEASONAL_HOLIDAY, TARGET_AUDIENCE, SENSITIVE_CONTENT, CULTURAL_STATUS,
SPECIFIC_PRAISE_CRITICISM, NAMED_SOURCE_CREATOR — a movie that the
keyword vocabulary fails to tag (true vocab gap, not genuine
absence) will score 0 on that category regardless of SEM strength.
The across-category max recovers the trait when another category
fires; for traits whose only category is KW + SEM additive, the
multiply does the gating intentionally.

The remediation lives upstream — the handler-LLM is expected to
keep KW tag sets broad enough to absorb realistic vocab variance,
and the routing rules in
[query_categories.md](query_categories.md) keep KW as a *broad
fallback* signal rather than a precise filter. If evaluation
reveals systematic miss patterns, the response is to broaden KW
spec generation, not to soften the combine rule. Multiply's
strictness is a feature: it forces categorical evidence to be
present before the category can fire.

## Implementation hooks (for follow-up planning, not part of this doc)

- Adding the combine-type field to `CategoryName` in
  [schemas/trait_category.py](../schemas/trait_category.py).
- Restructuring `execute_branches` in
  [stage_4_execution.py](../search_v2/stage_4_execution.py) into the
  five phases above.
- Migrating away from the recursive granularity rule (the existing
  category → trait → branch nested combine) toward the flat
  per-trait scoring against a finalized union.

These are sequenced after the conceptual frame is locked. Treat
this doc as the source of truth for the conceptual model; the
implementation plan derives from it.

---

# V5 — Query-generation findings: keyword over-eagerness collides
# with ADDITIVE × FACETS strictness

V4 introduced the trait-level FACETS combine_mode (PRODUCT across
categories) and inherited the V3 within-category ADDITIVE rule
(PRODUCT across endpoints). The two strictnesses stack: an ADDITIVE
category with KW + SEM zeroes on a KW miss; a FACETS trait zeroes
when *any* category zeroes. This section documents what the keyword
handler actually commits today under that math, where it goes
wrong, and how to fix the LLM commitments and the scoring math
together. Findings come from running
[run_specs.py](../search_v2/run_specs.py) over a 26-query suite
(default suite + a refinement batch tuned to test specific
hypotheses).

## Headline number

**46% of generated categories carry the "ADDITIVE_KW_RISK" trip-wire**
(18 / 39 in the refinement batch, 12 / 26 in the default batch).
Trip-wire fires when a category's combine_type is ADDITIVE *and* the
keyword endpoint is one of the firing endpoints. Under the V3/V4
math any movie missing the keyword tag scores 0 on that category
regardless of how strong the semantic gradient is.

## Diagnostic shapes (from
[run_specs.py](../search_v2/run_specs.py))

The script runs Step 2 → Step 3 (with the V4 sibling contract) →
per-CategoryCall handler-LLM, then surfaces per (trait, category):

- trait `combine_mode` (FRAMINGS vs FACETS — across-category fold)
- category `combine_type` (SINGLE / ADDITIVE / ALTERNATIVES — within-
  category fold)
- fired endpoint routes
- for KEYWORD endpoints: `finalized_keywords` + `scoring_method`
  (ANY vs ALL)
- for SEMANTIC endpoints: `role` (carver vs qualifier) + targeted
  vector spaces

Trip-wire flag `ADDITIVE_KW_RISK = (combine_type == ADDITIVE) AND
(KEYWORD ∈ fired_routes)` is computed mechanically per category.
Confirmed numerically:

```
combine_calls(ADDITIVE, [0.0_KW, 0.92_SEM]) → 0.0
combine_calls(ADDITIVE, [0.5_KW, 0.92_SEM]) → 0.46
combine_calls(SINGLE, []) → 0.0  (handler abstained)
combine_categories(FACETS, [0.0, 0.95]) → 0.0
combine_categories(FRAMINGS, [0.0, 0.95]) → 0.95
```

## Failure mode catalogue

### F1 — Vibe-only categories with a thin keyword commitment

The keyword handler fires a narrow registry member on a query whose
real signal is semantic. KW miss zeros the category; if it is the
only category (single-trait FRAMINGS), the trait dies entirely.

| Query | Trait → Category | Finalized keywords | Risk |
|---|---|---|---|
| `feel-good Christmas movies` | feel-good → EMOTIONAL_EXPERIENTIAL | `[FEEL_GOOD]` ANY | KW miss → trait_score=0 (FRAMINGS, single category) |
| `cozy fall movies` | cozy → EMOTIONAL_EXPERIENTIAL | `[FEEL_GOOD]` ANY | same |
| `movies with a twist ending` | twist ending → EMOTIONAL_EXPERIENTIAL | `[PLOT_TWIST]` ANY | same |
| `films with a haunting bittersweet tone` | haunting bittersweet tone → EMOTIONAL_EXPERIENTIAL | `[BITTERSWEET_ENDING]` ANY | tag is endings-specific, not tone |
| `slow burn character studies` | slow burn → EMOTIONAL_EXPERIENTIAL | `[THRILLER, PSYCHOLOGICAL_THRILLER, DRAMA]` ANY | none of these mean "slow" — miss is brutal |
| `cerebral psychological thrillers` | (whole) → STORY_THEMATIC_ARCHETYPE | `[PSYCHOLOGICAL_DRAMA]` ANY | wrong genre tag for a thriller query |

The corollary case where the handler *does* abstain — `movies
about loneliness` — produces the right shape: both categories pure
SEM, no ADDITIVE_KW_RISK flag, and the trait scores cleanly off the
semantic axis. Existence of clean abstention proves the prompt
already permits it; the handler just doesn't reach it often enough.

### F2 — ALL scoring chosen for paraphrase-cluster keywords

The keyword endpoint prompt notes ANY is the default when "the brief
is silent on combination — typical for single-attribute calls." But
the handler routinely commits ALL when the finalized list contains
two or three keywords that are *paraphrases of one concept*, not
distinct facets. ALL on a paraphrase cluster forces a movie to be
tagged with every member, and tagging is incomplete enough that this
penalises clean matches by 33–66%.

| Query | Category | finalized_keywords / scoring | Issue |
|---|---|---|---|
| `movies about WWII` | CENTRAL_TOPIC | `[WAR, HISTORY]` **ALL** | WAR + HISTORY are stacked framings of "WWII"; ALL forces both |
| `scary monster movies` | GENRE | `[HORROR, MONSTER_HORROR]` **ALL** | HORROR is broad parent of MONSTER_HORROR; many monster films lack the narrow tag |
| `dystopian sci-fi` | STORY_THEMATIC_ARCHETYPE | `[DYSTOPIAN_SCI_FI, POST_APOCALYPTIC]` **ALL** | adjacent neighbours, not a compound |
| `mind-bending sci-fi` | NARRATIVE_DEVICES | `[NONLINEAR_TIMELINE, UNRELIABLE_NARRATOR, PLOT_TWIST]` **ALL** | three alternative devices — Inception has 2, Memento 3, Eternal Sunshine 1 |
| `running movies` | CENTRAL_TOPIC | `[SPORT, BIOGRAPHY, TRUE_STORY]` **ALL** | each is a separate axis; demanding all three is a 3-way conjunction |
| `biographical dramas about musicians` | ADAPTATION_SOURCE | `[BIOGRAPHY, TRUE_STORY]` **ALL** | paraphrases — "Bohemian Rhapsody" tagged BIOGRAPHY but not TRUE_STORY → 0.5 |
| `cheap shark movies` | shark movies → CENTRAL_TOPIC | `[MONSTER_HORROR, SURVIVAL, SEA_ADVENTURE]` **ALL** | three alternative homes; Jaws tagged 2/3 |
| `slow burn character studies` | character studies → STORY_THEMATIC_ARCHETYPE | `[PSYCHOLOGICAL_DRAMA, BIOGRAPHY]` **ALL** | character study ≠ biography |

### F3 — Over-coverage keywords get committed despite the prompt's own warning

The keyword endpoint prompt explicitly cites the SPORT-for-running
example as the canonical over-coverage case ("SPORT covers running
but ALSO pulls football, basketball, hockey, golf"). The handler
reads this in-prompt and *still commits SPORT* for `running movies`.
The strengths/weaknesses analysis is purely informational — listing
"weaknesses: over-coverage: pulls football/basketball/hockey" does
not block the commitment.

| Query | finalized_keywords | Over-coverage |
|---|---|---|
| `running movies` | ELEMENT_PRESENCE: `[SPORT]` ANY | the canonical example, used anyway |
| `movies with horses` | ELEMENT_PRESENCE: `[ANIMAL_ADVENTURE, WESTERN, SPORT]` ANY | WESTERN over-pulls all westerns; SPORT over-pulls non-horse sports |
| `dark gritty marvel movies` | gritty → EMOTIONAL_EXPERIENTIAL: `[DRAMA, FILM_NOIR, THRILLER]` ANY | DRAMA matches every drama; near-no-op |
| `violent action movies` | violent → SENSITIVE_CONTENT: `[SPLATTER_HORROR, ACTION, MARTIAL_ARTS]` ANY | ACTION duplicates the sibling action trait; SPLATTER_HORROR over-pulls horror |
| `shitty shark movies` | sharks → ELEMENT_PRESENCE: `[SURVIVAL, HORROR]` ANY | neither names sharks; SURVIVAL pulls plane crash, desert, etc. |

ANY mode hides the over-coverage from the per-trait score (matching
any of the over-broad set still scores 1.0), but the over-broad set
inflates the candidate pool produced by the generator pass. The
candidate pool then carries SEM scores that pretend the movie is in
play when its only attachment was the over-broad keyword.

### F4 — Cross-trait keyword duplication double-counts the same evidence

When two traits in one query both route to keyword and converge on
the same registry member, that member contributes twice to the
candidate pool and twice to scoring (once per trait's fold). No
de-duplication across traits.

| Query | Duplication |
|---|---|
| `boxing movies` | trait → CENTRAL_TOPIC `[BOXING]` AND trait → GENRE `[BOXING]` |
| `violent action movies` | violent → SENSITIVE_CONTENT `[ACTION, ...]` AND action → GENRE `[ACTION, ...]` |
| `biographical dramas about musicians` | biographical dramas → ADAPTATION_SOURCE `[BIOGRAPHY, ...]` AND musicians → CENTRAL_TOPIC `[BIOGRAPHY, ...]` |

### F5 — Empty-spec categories under FACETS zero the trait

This is a code bug independent of the LLM's choices. When the
handler abstains entirely from a category (no specs generated), the
scoring path produces:

```
combine_calls(SINGLE, []) → 0.0   (or any non-NO_OP type)
```

That 0.0 then enters `combine_categories(FACETS, [0.0, …])` and
zeros the trait via PRODUCT.

Observed in:

- `underrated indie films` → indie films trait STUDIO_BRAND
  (handler abstained → no specs) + FINANCIAL_SCALE (metadata fired)
  under FACETS → trait_score = 0 × metadata = 0.
- `movies about WWII` → NARRATIVE_SETTING fired SEM but if it had
  abstained, FACETS WWII trait would have died on the empty
  category alone.

The fix is mechanical: filter empty-spec categories out of
`category_scores` before the across-category fold, treating them
identically to NO_OP. NO_OP is filtered at
[stage_4_execution.py:746](../search_v2/stage_4_execution.py#L746)
already — empty-spec cases just need the same treatment.

## Root causes

The handler-LLM and the scoring math both contribute. Fixing one
without the other still leaves cases on the table.

**RC1 — KW commitment has no awareness of downstream multiply.** The
keyword handler prompt is a registry-fit exercise. It does not know
that its commitment will be multiplied by the SEM commitment in an
ADDITIVE category. If the LLM understood "a brittle KW choice will
zero this category even on a strong SEM hit," it would abstain more
aggressively. Today the structural pressure is to commit at least
one keyword on every fire — `finalized_keywords: min_length=1` on
the schema enforces this for single-endpoint buckets, and the
multi-endpoint walk-then-commit shape pressures the LLM to fill the
slot when keyword is one of the routed endpoints.

**RC2 — Strengths/weaknesses analysis is purely descriptive.** The
PotentialKeyword schema asks for under-coverage and over-coverage
weaknesses, but nothing in the commitment phase reads them as a
go/no-go gate. A keyword whose weakness is "over-coverage: pulls
football, basketball, hockey, golf" still ends up in finalized_
keywords. The analysis-then-commit shape is a cognitive scaffold,
not a constraint.

**RC3 — ANY-vs-ALL discriminator is mis-tuned.** The prompt's read-
back ("does retrieval_intent treat the finalized members as
substitutable, or as each-matters?") biases the LLM toward ALL when
it can name a difference between members, even when the difference
is paraphrastic. WAR vs HISTORY are different *concepts* but both
*paraphrase the same WWII intent*. The LLM correctly distinguishes
them and incorrectly concludes ALL.

**RC4 — ADDITIVE strictness is uniform across category shapes.** The
V3 ADDITIVE rule treats every (KW + SEM) pair the same: KW is a
gate, SEM is a refiner, KW miss zeros. But the registry's coverage
is uneven: GENRE has nearly-clean coverage (HORROR, COMEDY, DRAMA),
while EMOTIONAL_EXPERIENTIAL covers maybe 20% of the experiential
space the user can describe. ADDITIVE-strict on a category whose
KW coverage is structurally incomplete is a category-level mismatch.

**RC5 — Empty-spec category folds as 0.0, not NO_OP.** This is the
F5 code bug. The semantic intent of an empty-spec category is "the
handler had nothing to fire here" — same as NO_OP at runtime — but
the scoring layer treats it as "the category fired with zero
score."

## Decisions

The following decisions emerged from a multi-round critique of the
initial proposed fixes. Earlier proposals (telling the keyword
handler about downstream multiply, adding a `commitment_verdict`
schema field) are superseded by simpler structural changes — most
notably removing the keyword endpoint from categories where
registry coverage is structurally too thin to carry the attribute,
and rewriting the commitment guidance around a generalized
superset test rather than category-by-category enumerations.

### D1. Per-category keyword configuration

The 11 categories that combine KW × SEM under V3 ADDITIVE-multiply
do not all benefit from the keyword endpoint. After examining each
against [query_categories.md](query_categories.md) and the
UnifiedClassification registry, three responses apply.

**Remove KW from category endpoint set.** Registry coverage is too
thin to carry the attribute space the category serves. Switching
combine_type to ALTERNATIVES does not help: KW=1 on a near-neighbor
match dominates the SEM gradient via MAX, so a noisy KW commit
silently overrides the more meaningful semantic signal. The user
phrases the KW would catch (TEARJERKER, FEEL_GOOD, etc.) are also
caught by SEM via reception / viewer-experience prose, so removal
is not a coverage loss.

- **SEASONAL_HOLIDAY** — `query_categories.md` itself states
  "no channel is authoritative — proxy tags are inherently
  approximate." With KW gone, CTX `watch_scenarios` + P-EVT
  setting prose carry the seasonal signal.
- **EMOTIONAL_EXPERIENTIAL** — ~6 KW tags for hundreds of
  experiential vibes. VWX + RCP + CTX cover the surface without a
  KW gate.
- **CULTURAL_STATUS** — "Classic", "underrated", "cult" are not
  tagged canonically. RCP carries the cultural-position prose.
- **SPECIFIC_PRAISE_CRITICISM** — Aspect-level praise/criticism
  lives in RCP `praised_qualities` / `criticized_qualities`.

**Switch combine_type from ADDITIVE to ALTERNATIVES.** KW carries
genuine canonical signal that the user cares about, but it is one
alternative path alongside the other firing endpoints, not a
multiplier gate. Either path alone is sufficient evidence.

- **TARGET_AUDIENCE** — KW (FAMILY_MOVIE / KIDS / TEENAGERS) and
  CTX (`watch_scenarios`) are alternative inclusion paths beneath
  the META.maturity_rank gate.
- **SENSITIVE_CONTENT** — KW for specific content flags
  (ANIMAL_DEATH-style) and VWX `disturbance_profile` for intensity
  gradient are alternative ways to detect content concerns.

**Keep ADDITIVE; rely on the superset commitment principle (D2)
to gate firing.** Registry has clean coverage when the attribute is
canonical; the multiply gating is the right shape; the failure case
is the handler firing keyword on non-canonical attributes where the
registry has no member that names what the user asked for.

- **CENTRAL_TOPIC** — canonical historical events, persons, named
  subjects (BIOGRAPHY, TRUE_STORY, named-event tags) gate cleanly
  when present; abstain when absent.
- **ELEMENT_PRESENCE** — canonical elements (ZOMBIE_HORROR, ROBOT,
  etc.) gate cleanly when present; abstain when registry has no
  matching element tag.
- **STORY_THEMATIC_ARCHETYPE** — canonical archetypes (REDEMPTION,
  REVENGE, FOUND_FAMILY) gate cleanly when present; abstain on
  vague archetypes.

**Keep ADDITIVE unchanged; no abstention rule needed.** Registry
coverage is consistently strong; multiply gating works as designed;
non-canonical commits are rare in observation.

- **CHARACTER_ARCHETYPE** — ANTI_HERO, FEMALE_LEAD, FEMME_FATALE
  registry coverage is dense.
- **NARRATIVE_DEVICES** — PLOT_TWIST, NONLINEAR_TIMELINE,
  UNRELIABLE_NARRATOR registry coverage is dense.

### D2. Keyword commitment principle: superset test

Replaces the analysis-then-commit cognitive scaffold with an
operational test the LLM applies before finalizing keywords. The
test is generalized — it applies to every keyword commitment,
every category, every registry member; it does not enumerate
specific keyword pairs because enumeration trains the LLM to
pattern-match on examples instead of evaluating the underlying
question.

> Fire keyword only when the keyword — or the ANY-mode union of
> keywords — is a true superset of the movies the user is asking
> for. A superset means: every movie that genuinely satisfies the
> user's attribute would carry at least one of the chosen
> keywords.
>
> **Over-pull is acceptable.** The keywords also covering unrelated
> movies is not a failure of this test. Semantic refinement on the
> same call narrows the noise, and broadness in this trait is
> recovered by another trait's specificity.
>
> **Gaps fail the test.** If a movie that genuinely satisfies the
> user's attribute could carry none of your chosen keywords, the
> set is not a superset. Firing will zero genuine matches under
> ADDITIVE-multiply. Abstain.
>
> **Stretching intent fails the test.** If the keywords name
> something semantically adjacent to the user's attribute rather
> than the attribute itself, you are stretching. Firing will
> tag-match adjacent-but-irrelevant movies at 1.0 while genuinely
> relevant movies that lack those tags score 0. Abstain.

This is operationalized at two layers:

1. **Keyword endpoint prompt** — the commitment phase reads the
   superset test as the gate between candidates and
   `finalized_keywords`. A candidate that fails any of the three
   conditions does not enter the commit set.
2. **Bucket prompt for multi-endpoint categories** — partial
   abstention is sanctioned. A keyword endpoint that fails the
   superset test abstains while semantic and other endpoints
   continue to fire. The current bucket prompt only sanctions
   "all endpoints abstain together," which is the structural
   cause of the handler reaching for nearest-neighbor on
   sharks-style queries.

### D3. Keyword scoring_method default: singular vs plural intent

Replaces the "substitutable vs each-matters" discriminator with a
generalized framing at the user-intent level. As with D2, this is
intentionally generalized — no enumerated examples — so the LLM
applies the underlying test rather than pattern-matches on listed
keyword pairs.

> The scoring_method defaults to ANY. ALL is reserved for the case
> where the user named multiple distinct attributes that should
> compound, each independently demanded.
>
> **Singular intent → ANY.** The user's expression names one
> attribute. The keyword commit may include multiple registry
> members because the LLM has converted that one attribute into
> several registry surface forms — paraphrases, alternative routes,
> sub-form alternatives. Matching any one is sufficient evidence
> the user's one thing is present.
>
> **Plural intent → ALL is on the table.** The user's expression
> names multiple distinct attributes the user wants present
> together — separate things, each independently demanded,
> compoundable. Each must be matched for the call's intent to be
> satisfied.
>
> **Operational test:** read the call's expressions. One
> expression with multiple keywords commits to ANY. Multiple
> expressions naming genuinely distinct attributes that the user
> conjoined may commit to ALL.

### D4. Code change: filter empty-spec categories from across-category fold

A category whose handler emitted no specs should be treated
identically to NO_OP — skipped from the across-category combine.
Currently empty-spec categories return 0.0 from `combine_calls`,
which under FACETS-PRODUCT zeros the entire trait and under
FRAMINGS-MAX is filtered by the max anyway (so no FRAMINGS
regression). The fix is in
[stage_4_execution.py](../search_v2/stage_4_execution.py) where
`live_cats` is assembled — extend the existing NO_OP filter to
also skip categories whose `cc.generated_specs` is empty.

Monotonic-safe: under FRAMINGS-MAX, removing a 0.0 from the
candidate set can only raise or hold the max. Under FACETS-PRODUCT,
removing a 0.0 makes the product non-zero, which is the desired
outcome (a category the handler abstained on should not contribute
zero evidence to the trait's compound scoring).

### D5. Code change: post-hoc dedup of identical generator specs

When two traits commit identical (route, params) generator specs
(same registry members, same scoring_method), execute the
underlying DB query once and feed the result map to both traits'
scoring paths. Pure perf optimization with no semantic change —
each trait still scores the keyword's contribution per its own
ANY/ALL commitment. Lives in Phase B (pool definition).

This addresses the F4 cross-trait keyword duplication observation
without breaking the per-call isolation principle that keeps the
handler-LLM cleanly scoped to one CategoryCall at a time.

### Out of decisions

The following V5 fix candidates were considered and rejected:

- **Telling the keyword handler about downstream multiply
  consequences in prose.** The handler-LLM is mediocre at
  consequentialist reasoning; structural changes (D1's category
  reclassification + D2's superset test) achieve the same outcome
  without relying on the LLM to remember a multiply warning over
  hundreds of tokens of prose.
- **Adding a `commitment_verdict: Literal["commit", "abstain"]`
  schema field on PotentialKeyword.** Subsumed by D2's superset
  test in the prompt — operationalizing the gate via prompt
  guidance avoids a multi-file schema migration and keeps the
  PotentialKeyword shape stable.
- **Per-category ADDITIVE-strictness softening as a global
  parameter.** Rejected at V3 design time and remains rejected.
  D1's per-category response is finer-grained and addresses the
  specific cases where strictness is wrong without weakening the
  rule globally.

## Verification artifact

The runner used to produce these findings:
[search_v2/run_specs.py](../search_v2/run_specs.py). Reproduce with:

```bash
python -m search_v2.run_specs                    # default 10-query suite
python -m search_v2.run_specs --suite path.txt   # custom suite
python -m search_v2.run_specs --json out.json    # machine-readable
```

Default and refinement suite outputs were captured to
`/tmp/run_specs_default.json` and `/tmp/run_specs_refinement.json`
during this investigation; regenerate as needed since the LLM is
non-deterministic and exact keyword commitments will drift.

## Implementation plan

Concrete steps to execute D1 through D5. Each phase is independently
shippable; recommended sequence below assumes each phase ships +
stabilizes before the next so its marginal effect on the V5 suite
metrics is observable in isolation.

### Pre-implementation correction: CULTURAL_STATUS already has no KW

Verification of the live enum at
[schemas/trait_category.py:1065-1098](../schemas/trait_category.py#L1065-L1098)
shows CULTURAL_STATUS endpoints are
`(EndpointRoute.SEMANTIC, EndpointRoute.METADATA)` — KEYWORD is
already absent. The earlier V5 narrative claimed CULTURAL_STATUS
needed KW removal; this was incorrect. CULTURAL_STATUS keeps its
current `ADDITIVE(SEMANTIC × METADATA)` shape, which composes RCP
status prose with quality/popularity priors — both genuinely
different signals that benefit from compounding.

The actual REMOVE-KW set is 3 categories, not 4:
**SEASONAL_HOLIDAY, EMOTIONAL_EXPERIENTIAL, SPECIFIC_PRAISE_CRITICISM.**

### Current state reference table

For each affected category as of this writing
([schemas/trait_category.py](../schemas/trait_category.py)):

| Category | Line | Endpoints | Bucket | Combine |
|---|---|---|---|---|
| CENTRAL_TOPIC | 262 | (KEYWORD, SEMANTIC) | PREFERRED_REPRESENTATION_FALLBACK | ADDITIVE |
| ELEMENT_PRESENCE | 288 | (KEYWORD, SEMANTIC) | PREFERRED_REPRESENTATION_FALLBACK | ADDITIVE |
| CHARACTER_ARCHETYPE | 313 | (KEYWORD, SEMANTIC) | PREFERRED_REPRESENTATION_FALLBACK | ADDITIVE |
| NARRATIVE_DEVICES | 701 | (KEYWORD, SEMANTIC) | PREFERRED_REPRESENTATION_FALLBACK | ADDITIVE |
| TARGET_AUDIENCE | 731 | (KEYWORD, METADATA, SEMANTIC) | AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST | ADDITIVE |
| SENSITIVE_CONTENT | 761 | (KEYWORD, METADATA, SEMANTIC) | AUDIENCE_SUITABILITY_DETERMINISTIC_FIRST | ADDITIVE |
| SEASONAL_HOLIDAY | 786 | (SEMANTIC, KEYWORD) | SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT | ADDITIVE |
| STORY_THEMATIC_ARCHETYPE | 873 | (KEYWORD, SEMANTIC) | PREFERRED_REPRESENTATION_FALLBACK | ADDITIVE |
| EMOTIONAL_EXPERIENTIAL | 901 | (SEMANTIC, KEYWORD) | SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT | ADDITIVE |
| CULTURAL_STATUS | 1065 | (SEMANTIC, METADATA) | SEMANTIC_PREFERRED_DETERMINISTIC_SUPPORT | ADDITIVE |
| SPECIFIC_PRAISE_CRITICISM | 1099 | (SEMANTIC, KEYWORD) | PREFERRED_REPRESENTATION_FALLBACK | ADDITIVE |

Verify line numbers against the live file before editing — they
will drift as enum members are added/reordered.

### Phase 1 — Code changes

**Change 1.1 — Filter empty-spec categories from the across-category
fold (D4).**

File: [search_v2/stage_4_execution.py](../search_v2/stage_4_execution.py),
inside `_score_positive_trait` around line 742-753 where `live_cats`
is assembled.

Current logic skips NO_OP categories at the top of the loop:

```python
for cat_idx, cc in enumerate(trait.category_calls):
    if cc.handler_error is not None:
        continue
    combine_type = cc.category.combine_type
    if combine_type is CategoryCombineType.NO_OP:
        continue
    live_pairs: list[...] = []
    for spec_idx, spec in enumerate(cc.generated_specs):
        ...
    live_cats.append((cc, combine_type, live_pairs))
```

Extend the NO_OP filter to also skip empty-spec categories:

```python
for cat_idx, cc in enumerate(trait.category_calls):
    if cc.handler_error is not None:
        continue
    combine_type = cc.category.combine_type
    if combine_type is CategoryCombineType.NO_OP:
        continue
    if not cc.generated_specs:
        # Handler abstained — semantically identical to NO_OP
        # at scoring time. Skip so the empty fold doesn't
        # produce a 0.0 that would zero a FACETS trait.
        continue
    live_pairs: list[...] = []
    ...
```

`combine_calls` itself stays unchanged — the fix is at the upstream
filter, not inside the math. This way `combine_calls(SINGLE, [])`
remains 0.0 (correct semantics if the call ever did need to be made
with empty scores) but the empty-spec categories never reach
`combine_calls` in the first place.

The negative-trait scoring path
([stage_4_execution.py:921](../search_v2/stage_4_execution.py#L921))
does not need this change — the gate × fuzzy formula partitions
calls structurally and doesn't fold through per-category empty
checks.

**Verification:** confirmed by inspection that under FRAMINGS-MAX
removing a 0.0 entry can only raise or hold the max, and under
FACETS-PRODUCT removing a 0.0 makes the product non-zero. Monotonic-
safe for both combine_modes.

**Change 1.2 — Post-hoc dedup of identical generator specs (D5).**

File: [search_v2/stage_4_execution.py](../search_v2/stage_4_execution.py),
Phase B (pool definition).

Current behavior: each trait's CategoryCalls each contribute
`generated_specs` which all run independently in parallel. When two
traits commit identical (route, params), the underlying DB query
executes twice.

Implementation outline:

1. Walk all traits' positive-polarity generator specs into a flat
   list of `(trait_idx, cat_idx, spec_idx, spec)` tuples.
2. Build a dedup key from each spec — `(spec.route, dedup_hash(spec.params))`
   where `dedup_hash` serializes the pydantic params model
   deterministically (e.g., `json.dumps(spec.params.model_dump(mode="json"), sort_keys=True)`).
3. Group tuples by dedup key. Execute one representative spec per
   group, get back its `dict[movie_id, score]`.
4. Distribute the result map: for every (trait_idx, cat_idx,
   spec_idx) tuple in the group, register the result map at the
   same coordinates the per-trait scoring loop would look up
   (`call_score_maps[(trait_idx, cat_idx, spec_idx)]`).

Score-side semantics unchanged — each trait still reads the result
map per its own ANY/ALL commit, scoring_method, etc. Per-trait
rarity computation is unaffected because rarity is computed from the
matched-set of the generator's score map, which is identical across
all traits sharing the spec.

**Edge case:** if two traits commit specs that look identical but
differ in opaque fields the executor consumes, the dedup will fold
them incorrectly. Mitigate by using the full `model_dump(mode="json")`
serialization as the hash key — anything that affects execution
output should be in `model_dump`. Excluded fields (private state,
non-serializable handles) are the executor's responsibility to
exclude from `model_dump` already.

**Performance bound:** dedup is O(N_specs) where N_specs is total
generator count across all traits. Typical query has 2-5 traits ×
1-3 generators per trait = 5-15 specs. Dedup overhead is negligible
relative to DB call latency.

### Phase 2a — combine_type changes (single-line edits)

File: [schemas/trait_category.py](../schemas/trait_category.py).

**TARGET_AUDIENCE (line ~759):**

Change `CategoryCombineType.ADDITIVE` to
`CategoryCombineType.ALTERNATIVES` on the last positional arg of
the enum tuple. No other changes — endpoints, bucket, and per-
category prompts stay as-is.

**SENSITIVE_CONTENT (line ~784):**

Same single-line change.

**Why no prompt updates needed for Phase 2a:** The prompts for
TARGET_AUDIENCE / SENSITIVE_CONTENT do not name the combine_type or
multiply behavior anywhere. The combine_type is read only by
`stage_4_execution.combine_calls`, which honors the new value
immediately.

**Risk surface:** under V3 ADDITIVE, KW × META × SEM compounded.
Under V5 ALTERNATIVES, MAX(KW, META, SEM) takes whichever is
strongest. For TARGET_AUDIENCE: a movie inside the maturity range
gets META=1.0 → category=1.0 regardless of KW/SEM, which encodes
"maturity-eligible" as the passing signal. For movies outside the
range, META falls off and KW/SEM still scores. This is closer to
the intended "gate + inclusion" semantics than ADDITIVE was. Same
reasoning for SENSITIVE_CONTENT.

### Phase 2b — KW removal (per-category atomic changesets)

Three categories: SEASONAL_HOLIDAY, EMOTIONAL_EXPERIENTIAL,
SPECIFIC_PRAISE_CRITICISM. Each is a self-contained changeset
touching 4-5 files. Ship one category at a time, evaluate, then
proceed to the next — do not batch.

**Common transition logic:** all three categories are currently
two-endpoint `(SEMANTIC, KEYWORD)` shapes. Removing KEYWORD
collapses to a single-endpoint SEMANTIC shape. Per the
"What counts as one call" section of this doc, a category whose
only endpoint is SEMANTIC is SINGLE-combine at the orchestrator
level (semantic internally aggregates across vector spaces in one
call). The bucket therefore moves to `SINGLE_NON_METADATA_ENDPOINT`.

**Per-category changes for each of the three:**

1. **`schemas/trait_category.py`**:
   - Drop `EndpointRoute.KEYWORD` from the endpoints tuple
   - Change `HandlerBucket` to `SINGLE_NON_METADATA_ENDPOINT`
   - Change `CategoryCombineType.ADDITIVE` to
     `CategoryCombineType.SINGLE`

2. **`search_v2/endpoint_fetching/category_handlers/prompts/categories/additional_objective_notes/<category>.md`**:
   - Read the file in full
   - Remove sections naming the keyword endpoint or the
     `keyword_walk` / `finalized_keywords` / `scoring_method`
     fields
   - Remove guidance about KW-vs-SEM coverage tradeoffs (the file
     was authored under the multi-endpoint shape)
   - Preserve guidance about the category's domain (what the
     category is about, what's in vs out of scope) — that is
     unchanged

3. **`search_v2/endpoint_fetching/category_handlers/prompts/categories/few_shot_examples/<category>.md`**:
   - Read the file in full
   - Remove example outputs that include `keyword_walk`,
     `coverage_assignments` with a keyword entry, or
     `keyword_parameters` blocks
   - If all examples were multi-endpoint, replace with new examples
     that match the SINGLE_NON_METADATA_ENDPOINT bucket's expected
     output shape — a single SemanticEndpointParameters payload

4. **Verify `_BUCKET_OBJECTIVES` and `_BUCKET_GUARDRAILS` cover
   `SINGLE_NON_METADATA_ENDPOINT`** in
   [search_v2/endpoint_fetching/category_handlers/prompts/buckets/](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/).
   `single_non_metadata_endpoint_objective.md` and
   `single_non_metadata_endpoint_guardrails.md` already exist (they
   power other SINGLE-bucket categories), so no new authoring is
   needed — just verify they're present and that the bucket
   produces appropriate output schema for a SEMANTIC-only
   category.

5. **Verify schema generation:** [schema_factories.py](../search_v2/endpoint_fetching/category_handlers/schema_factories.py)
   keys per-category schemas off the bucket. With the bucket changed
   to SINGLE_NON_METADATA_ENDPOINT, the LLM will be asked to emit a
   schema with only one endpoint slot (SEMANTIC). Run a smoke test
   on each category after the change to confirm the schema is
   generated cleanly and the LLM produces a valid output.

**SEASONAL_HOLIDAY specific notes:**
- The query_categories.md doc Cat 29 description references
  KW + CTX + P-EVT additive combo. After this change, the doc is
  out of date with respect to the implementation. **Update
  query_categories.md Cat 29** to reflect SEMANTIC-only.

**EMOTIONAL_EXPERIENTIAL specific notes:**
- query_categories.md Cat 33 description references VWX + CTX +
  RCP + KW. **Update query_categories.md Cat 33** to remove KW
  from the endpoints list.

**SPECIFIC_PRAISE_CRITICISM specific notes:**
- query_categories.md Cat 40 description references RCP + KW
  additive combo. **Update query_categories.md Cat 40** to remove
  KW.

**Sequence within Phase 2b:** start with EMOTIONAL_EXPERIENTIAL —
it has the highest ADDITIVE_KW_RISK rate in the V5 suite (most
queries with vibe-trait failures route here). Then SEASONAL_HOLIDAY
(seasonal proxy chain is the second-clearest doc-vs-implementation
mismatch). Then SPECIFIC_PRAISE_CRITICISM (lowest-frequency, easiest
to verify last).

After each category lands, re-run
`python -m search_v2.run_specs` and confirm:
- The category no longer flags ADDITIVE_KW_RISK on any V5 suite
  query
- The semantic-only output is structurally valid for the queries
  that previously routed to the multi-endpoint shape
- No regressions on queries that didn't previously trigger the
  category

### Phase 3 — Prompt rewrites

**Change 3.1 — Keyword endpoint prompt: superset test (D2).**

File: [search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md).

Current structure (relevant sections):
- "What does NOT belong here" — unchanged
- "Where the keyword analysis lives" — unchanged
- "Classification registry" — unchanged
- "Reading inputs as keyword facets" — unchanged
- "Surface forms and aliases" — unchanged
- "Authoring `strengths` and `weaknesses` per candidate" — REWRITE
- "Near-collision disambiguation" — REWRITE
- "Reading the brief for scoring_method" — see Change 3.2

**Replace "Authoring strengths and weaknesses" + "Near-collision
disambiguation" with a single "Commitment: superset test"
section.** The strengths/weaknesses fields stay in the schema (the
LLM walks them as cognitive scaffold) but the gate from candidates
to `finalized_keywords` becomes the superset test.

Suggested section content (verbatim from D2):

> ## Commitment: superset test
>
> Fire keyword only when the keyword — or the ANY-mode union of
> keywords — is a true superset of the movies the user is asking
> for. A superset means: every movie that genuinely satisfies the
> user's attribute would carry at least one of the chosen keywords.
>
> **Over-pull is acceptable.** The keywords also covering unrelated
> movies is not a failure of this test. Semantic refinement on the
> same call narrows the noise, and broadness in this trait is
> recovered by another trait's specificity.
>
> **Gaps fail the test.** If a movie that genuinely satisfies the
> user's attribute could carry none of your chosen keywords, the
> set is not a superset. Firing will zero genuine matches under
> ADDITIVE-multiply. Abstain.
>
> **Stretching intent fails the test.** If the keywords name
> something semantically adjacent to the user's attribute rather
> than the attribute itself, you are stretching. Firing will
> tag-match adjacent-but-irrelevant movies at 1.0 while genuinely
> relevant movies that lack those tags score 0. Abstain.
>
> Apply the test once over the union of your finalized members in
> ANY mode. If the union passes the test, commit; if it fails on
> any of the three conditions, drop members until the remainder
> passes — or abstain entirely if no remaining subset passes.

Do **not** include category-specific examples in this section. The
test is generalized so the LLM evaluates the underlying property
rather than pattern-matches on enumerated keyword pairs.

**Schema implication:** the multi-endpoint shape (subintent) already
allows abstention via not-including-keyword in `coverage_assignments`.
The single-endpoint shape (`KeywordQuerySpec` with `min_length=1`
on `finalized_keywords`) does not currently allow full abstention.
Single-endpoint categories that route exclusively to keyword
(GENRE when canonical, AWARDS when KW path, etc.) will still need
to commit at least one member; the superset test still applies as
guidance to pick the best fit. Do **not** relax `min_length=1` on
the single-endpoint schema — those categories route to keyword
because the user's attribute is registry-clean, so abstention is
not the failure mode there.

**Change 3.2 — Keyword endpoint prompt: scoring_method singular vs
plural (D3).**

Same file, "Reading the brief for scoring_method" section. Replace
the existing ANY/ALL discriminator with the singular/plural framing
(verbatim from D3):

> ## Reading the brief for scoring_method
>
> The scoring_method defaults to ANY. ALL is reserved for the case
> where the user named multiple distinct attributes that should
> compound, each independently demanded.
>
> **Singular intent → ANY.** The user's expression names one
> attribute. The keyword commit may include multiple registry
> members because you have converted that one attribute into
> several registry surface forms — paraphrases, alternative routes,
> sub-form alternatives. Matching any one is sufficient evidence
> the user's one thing is present.
>
> **Plural intent → ALL is on the table.** The user's expression
> names multiple distinct attributes the user wants present
> together — separate things, each independently demanded,
> compoundable. Each must be matched for the call's intent to be
> satisfied.
>
> **Operational test:** read the call's expressions. One
> expression with multiple keywords commits to ANY. Multiple
> expressions naming genuinely distinct attributes that the user
> conjoined may commit to ALL.
>
> When N=1 (one finalized keyword) the two modes are mathematically
> identical — default to ANY and move on.

Do **not** include category-specific examples in this section
either.

**Update the matching schema field description.** [schemas/keyword_translation.py:215-238](../schemas/keyword_translation.py#L215-L238)
contains the `scoring_method` field's description. Edit the prose
to match the singular-vs-plural framing — the schema description is
treated as a micro-prompt and must not contradict the endpoint
prompt.

**Change 3.3 — Bucket prompt: sanction partial abstention.**

File: [search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md).

Current "coverage exploration" phase has three local tests: Fire
test, Drop test, Over-coverage refinement. Add a fourth:

> - **Superset test (per endpoint):** apply the endpoint's own
>   commitment principle to its candidates. If the candidates fail
>   the endpoint's commitment criteria (e.g., the keyword
>   candidates fail the keyword endpoint's superset test —
>   gaps in coverage of the user's attribute, or stretching intent
>   beyond what the registry names), drop that endpoint from
>   coverage_assignments even if other endpoints fire. This is
>   partial abstention — sanctioned alongside the existing whole-
>   call abstention pathway.

Update the section that currently says **"Empty `coverage_assignments`
is valid only when ALL declared endpoint walks surfaced no useful
candidate"** to clarify:

> Empty `coverage_assignments` is valid when no endpoint walk
> surfaced a candidate that passes both the local fire/drop tests
> and the endpoint's own commitment criteria. Partial commitment
> (some endpoints fire, others abstain via the superset test or
> equivalent) is also valid — the per-endpoint criteria are
> independent.

**Audit other multi-endpoint bucket prompts** for the same wording
pattern:
- `audience_suitability_deterministic_first_objective.md`
- `character_franchise_fanout_objective.md`
- `semantic_preferred_deterministic_support_objective.md`

Apply the same partial-abstention sanction wherever the prompt
currently treats abstention as all-or-nothing.

### Recommended sequence

1. **Phase 1** (changes 1.1, 1.2). Pure code, monotonic-safe. Lands
   without LLM behavior changes. Re-run V5 suite to confirm
   ADDITIVE_KW_RISK count holds steady (Phase 1 doesn't reduce
   risk; it just stops empty-spec from zeroing FACETS traits).

2. **Phase 2a** (TARGET_AUDIENCE + SENSITIVE_CONTENT combine_type
   flips). Single-line edits, no prompt churn. Re-run V5 suite —
   the trip-wire flag stays on these categories (they still have
   KW + ADDITIVE-multiply problem? No — combine_type is now
   ALTERNATIVES, so the trip-wire formula
   `combine_type==ADDITIVE AND KEYWORD ∈ fired` no longer fires.
   Confirm flag clears).

3. **Phase 2b**, one category at a time:
   - EMOTIONAL_EXPERIENTIAL first (highest impact).
   - SEASONAL_HOLIDAY second.
   - SPECIFIC_PRAISE_CRITICISM third.
   - Between each: re-run V5 suite, verify the category is no
     longer in any ADDITIVE_KW_RISK report, verify no schema-
     generation errors, smoke-test 3-5 queries that previously
     hit the category.

4. **Phase 3** prompt rewrites:
   - 3.1 + 3.2 together (both edit keyword.md and the schema
     description; do as one changeset).
   - 3.3 separately, with the audit of sibling bucket prompts.
   - After each, re-run V5 suite and count: ADDITIVE_KW_RISK
     categories on the 5 still-keyword-firing categories
     (CENTRAL_TOPIC, ELEMENT_PRESENCE, CHARACTER_ARCHETYPE,
     STORY_THEMATIC_ARCHETYPE, NARRATIVE_DEVICES) plus any
     remaining keyword commits in TARGET_AUDIENCE /
     SENSITIVE_CONTENT under ALTERNATIVES.

### Out of implementation scope

Per the V5 "Out of decisions" section, the following candidates
were considered and rejected; do **not** add them to the
implementation:

- Telling the keyword handler about downstream multiply consequences
  in prose.
- Adding `commitment_verdict: Literal["commit", "abstain"]` schema
  field on `PotentialKeyword`.
- Per-category ADDITIVE-strictness softening as a global parameter.
- Threading sibling-trait info into the handler's user message
  (architectural reversal of per-call isolation).
- Step 3 keyword-namespace reservations (pushes registry knowledge
  upstream into Step 3).

### Verification plan

After each phase, the V5 runner produces the headline metric:

```bash
python -m search_v2.run_specs --json /tmp/v5_after_phase_N.json
```

Compare ADDITIVE_KW_RISK count against the pre-Phase-1 baseline
(46% / 12-of-26 default suite, 18-of-39 refinement suite). Expected
trajectory:
- After Phase 1: same 46% (Phase 1 fixes a different failure mode —
  empty-spec trait death — not the trip-wire population).
- After Phase 2a: trip-wire on TARGET_AUDIENCE / SENSITIVE_CONTENT
  clears (combine_type no longer ADDITIVE).
- After Phase 2b (each): trip-wire on the just-removed category
  clears (KEYWORD no longer fires there).
- After Phase 3: trip-wire on the 5 still-keyword-firing categories
  drops as the handler abstains more aggressively under the
  superset test. Target: <20% on the V5 suite, with abstentions
  concentrated on non-canonical attributes (sharks, vibes).

**Sticky cases to watch through all phases:**
- "movies about WWII" CENTRAL_TOPIC — should keep firing keyword
  (canonical historical event); ANY/ALL should be ANY after Phase 3.
- "shitty shark movies" ELEMENT_PRESENCE — should abstain from
  keyword after Phase 3 (sharks not in registry; superset test
  rejects MONSTER_HORROR/SURVIVAL/HORROR).
- "feel-good Christmas movies" — both trait categories
  (EMOTIONAL_EXPERIENTIAL, SEASONAL_HOLIDAY) become semantic-only
  after Phase 2b. Verify the semantic call still produces
  reasonable rankings.
- "movies about loneliness" — should remain pure-SEM (already does
  today via correct abstention). Should not regress.

### Files touched (full inventory)

Code:
- [search_v2/stage_4_execution.py](../search_v2/stage_4_execution.py) — Phase 1.1 + 1.2

Schemas:
- [schemas/trait_category.py](../schemas/trait_category.py) — Phase 2a (2 categories) + Phase 2b (3 categories)
- [schemas/keyword_translation.py](../schemas/keyword_translation.py) — Phase 3.2 (scoring_method field description)

Prompts (endpoint-level):
- [search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md](../search_v2/endpoint_fetching/category_handlers/prompts/endpoints/keyword.md) — Phase 3.1 + 3.2

Prompts (bucket-level):
- [search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md](../search_v2/endpoint_fetching/category_handlers/prompts/buckets/preferred_representation_fallback_objective.md) — Phase 3.3
- Plus `audience_suitability_deterministic_first_objective.md`,
  `character_franchise_fanout_objective.md`,
  `semantic_preferred_deterministic_support_objective.md` — Phase 3.3 audit

Prompts (category-level), Phase 2b only:
- `additional_objective_notes/seasonal_holiday.md`
- `additional_objective_notes/emotional_experiential.md`
- `additional_objective_notes/specific_praise_criticism.md`
- `few_shot_examples/seasonal_holiday.md`
- `few_shot_examples/emotional_experiential.md`
- `few_shot_examples/specific_praise_criticism.md`

Documentation:
- [search_improvement_planning/query_categories.md](query_categories.md) — Phase 2b (Cat 29, Cat 33, Cat 40 endpoint lists)
- This doc — already updated through V5.
