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

### Multi-call same-category

When a single category fires multiple calls of the same endpoint
type (e.g., PERSON_CREDIT with two ENT calls because the spec is
shaped to handle multiple entities), the endpoint owns the scoring
logic for that case. The ENT executor takes a list of entities and
returns one [0,1] score per movie covering all of them. The
orchestrator does not see the internal aggregation.

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
- Improving keyword query breadth for additive categories. If KW
  tag sets have meaningful coverage gaps, the multiply rule will
  occasionally zero out additive categories for strong-SEM-untagged
  candidates. Treated as a separate concern: ensure handler-LLM
  keeps keyword sets broad when the category is additive, and accept
  that bad keyword data produces bad multiplied scores.

## Sharp edges to watch

The multiply-for-additive rule is strict. A movie missing the
canonical tag in an additive category zeros that category, even
with strong gradient evidence. The across-category max provides
recovery when other categories of the same trait fire, but if every
category of a trait happens to be additive multi-call with KW + SEM,
the strictness compounds. Expected to be rare in practice given how
Step 3 currently routes; flagged as a thing to monitor in evaluation.

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
