# Step 4 Planning

Detailed planning for Step 4 of the V2 search pipeline: candidate assembly,
exclusion handling, final scoring, reranking, debug output, and final result
payload shaping.

This file is the fine-detail companion to
`search_improvement_planning/finalized_search_proposal.md` for Step 4 only.
Use the finalized proposal for the system-level architecture and this file for
Step-4-specific implementation details, clarified edge cases, and unresolved
questions.

## Scope

Step 4 is responsible for:

1. Receiving one fully isolated Step-1 interpretation branch at a time
2. Coordinating all Step 3 translation/execution work for that branch
3. Assembling the candidate pool
4. Applying exclusions
5. Computing final scores and reranking
6. Fetching the minimal display fields for the top 100 movies
7. Returning the top 100 results plus debug data

## Branch Isolation

Each Step-1 interpretation branch runs through its own independent Step 2 ->
Step 3 -> Step 4 flow.

Practical meaning:

- Step 2 only sees one interpretation at a time
- Step 3 only sees one interpretation at a time
- Step 4 only assembles/ranks candidates for one interpretation at a time
- Different Step-1 branches do not share candidates, scores, or endpoint state

## Finalized Decisions

### Step 3 / Step 4 coordination

- All Step 3 LLM translations for the branch should be kicked off immediately
- Any Step 3 execution that does not depend on the final candidate pool should
  run as soon as its translation returns
- Any Step 3 execution that needs candidate IDs should wait until:
  - its own LLM translation has returned, and
  - all candidate-generating work for the branch has finished and been
    consolidated into a single deduplicated ID set
- Initial timeout budget for this orchestration is 20 seconds

### Candidate generation scenarios

#### Standard inclusion flow

When one or more inclusion dealbreakers exist and at least one of them can
generate candidates deterministically:

- candidate-generating inclusion endpoints contribute their matched IDs
- those ID sets are unioned and deduplicated
- semantic inclusion dealbreakers do not generate candidates in this flow;
  they only score the assembled pool

#### D2: semantic-only inclusion flow

When inclusion dealbreakers exist but all of them are semantic:

- semantic dealbreakers generate candidates
- cap candidate generation at 2000 per vector search
- union and deduplicate those IDs into the branch candidate pool

#### Zero inclusion dealbreakers, preferences exist

When there are no inclusion dealbreakers, but there are preferences, those
preferences become the source of candidate generation for the branch.

This applies even if exclusion dealbreakers exist.

- all preferences are eligible to participate in candidate generation
- `dealbreaker_sum` starts at 0 for every seeded candidate
- preferences and priors determine positive ranking movement
- exclusions then filter or penalize that pool

This is intentionally different from the pure exclusion-only browse fallback.
If the user gave positive preferences, we should use them as the positive
retrieval signal rather than falling back to generic browse ordering.

#### Zero inclusion dealbreakers, zero preferences

When the branch contains only exclusion dealbreakers:

- seed candidates from `movie_card`
- order that seed pool by an effective prior-based browse score derived from
  `quality_prior` and `notability_prior`
- initial working seed size: top 2000
- `dealbreaker_sum` starts at 0 for every seeded candidate
- apply exclusions after seeding
- final ranking comes from prior contribution minus any semantic exclusion
  penalties

Important: the browse seed ordering is only a candidate-generation heuristic.
It does not give seeded movies any positive dealbreaker credit.

### Exclusion handling

#### Deterministic exclusions

- hard-remove candidates from the pool
- applies to entity, metadata, awards, franchise, keyword, and any other
  endpoint with binary/categorical exclusion semantics

#### Semantic exclusions

- never produce a standalone exclusion ID list
- cannot hard-filter based on vector similarity alone
- only downrank candidates that are already in the seeded pool
- use global-corpus elbow/floor calibration rather than candidate-relative
  calibration

This matters operationally: if a branch has only semantic exclusions and no
positive inclusion signal, Step 4 must seed candidates first before semantic
exclusion scoring can do anything.

### Final scoring model

Working formula remains:

```text
final_score = dealbreaker_sum + preference_contribution - exclusion_penalties
```

With these Step-4 clarifications:

- `dealbreaker_sum` is only for inclusion dealbreakers
- exclusion-only seed candidates begin with `dealbreaker_sum = 0`
- preference-generated seed candidates in the zero-inclusion flow also begin
  with `dealbreaker_sum = 0`
- final scores are intentionally unbounded below zero

### Priors in Step 4

`quality_prior` and `notability_prior` are part of preference-side ranking,
not dealbreaker scoring.

They should influence:

- normal reranking when inclusion dealbreakers already produced candidates
- preference-driven candidate flows with zero inclusion dealbreakers
- browse-style exclusion-only fallback ordering

Current meaning of the prior inputs:

- `quality_prior` operates on the quality-side signal used by Step 4
- `notability_prior` operates on the movie's notability / mainstream-recognition
  signal

### Popularity score definition

`popularity_score` is not a TMDB popularity field.

In this codebase it is:

- sourced from IMDb vote count
- converted to a global percentile
- transformed by a sigmoid
- stored on `public.movie_card.popularity_score`

It represents stable "widely known / lifetime engagement" rather than "hot
right now." Trending remains the TMDB-driven temporal signal.

Because Step 4 already uses `popularity_score`, the old TODO about "using
`imdb_vote_count` in reranking" was stale and has been removed.

## Per-Endpoint Orchestration

Per-endpoint behavior across the three roles a Step 2 item can take:
inclusion dealbreaker, preference, and exclusion dealbreaker. Scoring
formulas are intentionally out of scope here — see Phase 4c of the
finalized proposal for that. This section covers translation,
execution, and candidate-generation behavior only.

The "candidate generator" terminology is shorthand for: this endpoint
produces an ID set that is unioned into the branch candidate pool
during Phase 4a. Endpoints that "do not generate candidates" still run
their queries — they just do so against the already-assembled pool to
produce per-candidate signal for Phase 4c.

### Endpoint 1: Entity Lookup

- **Dealbreaker:** Step 3 LLM emits an `EntityQuerySpec` in
  positive-presence form (even when the Step 2 direction is
  exclusion). Execution runs exact-match lookups against the relevant
  posting table(s) — single `person_category` table, or all 5 in
  `broad_person` mode with cross-table dedup. **Generates
  candidates** (no pool size cap; worst case ~7K). Empty result is a
  valid outcome.
- **Preference:** Same translation and same lookup mechanics.
  **Does not generate candidates** in the standard inclusion flow;
  used to score pool members only. Becomes a candidate generator in
  the zero-inclusion preference flow.
- **Exclusion:** Same lookup; matched IDs are hard-removed from the
  pool in Phase 4b.

### Endpoint 2: Movie Attributes

- **Dealbreaker:** Step 3 LLM emits a `MetadataTranslationOutput`
  with a single `target_attribute`. Execution runs one column query
  (SQL `WHERE` or GIN `&&`) using a deterministically widened
  "generous gate" around the literal LLM translation. **Generates
  candidates.** Movies with null data on the target column are NOT
  excluded from the pool.
- **Preference:** Same translation. **Does not generate candidates**
  in the standard flow; runs a value lookup against the same column
  for the assembled pool. Becomes a candidate generator (using the
  same generous gate) in the zero-inclusion preference flow.
- **Exclusion:** Translation is inclusion-only. Step 4 inverts /
  removes the matched ID set in Phase 4b.

### Endpoint 3: Awards

- **Dealbreaker:** Step 3 LLM emits an `AwardQuerySpec` (filter axes
  plus scoring mode/mark). Execution dispatches to either the
  `award_ceremony_win_ids` fast path or `COUNT(*) FROM movie_awards
  WHERE ...` with active filters. Razzie is auto-excluded unless
  explicitly named in `ceremonies`. **Generates candidates** (one row
  per matched movie).
- **Preference:** Same translation and same query. **Does not
  generate candidates** in the standard flow; pulls counts/matches
  for assembled pool members. Becomes a candidate generator in the
  zero-inclusion preference flow.
- **Exclusion:** Same query; matched IDs hard-removed in Phase 4b.

### Endpoint 4: Franchise Structure

- **Dealbreaker:** Step 3 LLM emits a `FranchiseQuerySpec` with up
  to 5 axes (`lineage_or_universe_names`, `recognized_subgroups`,
  `lineage_position`, `structural_flags`, `launch_scope`). Execution
  does exact-match against normalized stored strings and ANDs across
  populated axes against `movie_franchise_metadata`. **Generates
  candidates** (naturally small sets, no cap). Zero-result is
  accepted.
- **Preference:** Same translation and same query. **Does not
  generate candidates** in the standard flow; tests pool membership
  for axis match. Becomes a candidate generator in the zero-inclusion
  preference flow.
- **Exclusion:** Same query; matched IDs hard-removed in Phase 4b.

### Endpoint 5: Keywords & Concept Tags

- **Dealbreaker:** Step 3 LLM picks exactly one
  `UnifiedClassification` registry entry. Execution issues one GIN
  `&&` overlap query against the resolved `(backing_column,
  source_id)` pair on `movie_card`. **Generates candidates.**
- **Preference:** Same translation and same overlap query. **Does
  not generate candidates** in the standard flow; checks pool
  membership of the resolved ID. Becomes a candidate generator in
  the zero-inclusion preference flow. Thematic-centrality nuance
  ("Christmas is central, not backdrop") is delegated to a paired
  semantic preference, not handled here.
- **Exclusion:** Same overlap query; matched IDs hard-removed in
  Phase 4b.

### Endpoint 6: Semantic

Two execution scenarios per role; the active scenario depends on what
else exists in the branch.

- **Dealbreaker — D1 (score-only):** Triggered when ≥1 non-semantic
  inclusion dealbreaker exists. Step 3 LLM emits a
  `SemanticDealbreakerSpec` (1 of 7 non-anchor spaces, structured-
  label body). Execution runs a global calibration probe plus a
  single filtered `query_points` call with `HasIdCondition` over the
  assembled pool. **Does not generate candidates.**
- **Dealbreaker — D2 (candidate-generating):** Triggered when zero
  non-semantic inclusion dealbreakers exist and ≥1 semantic
  inclusion dealbreaker exists. Each semantic dealbreaker
  independently runs top-N (cap 2000) against the full corpus on its
  selected space. Union across dealbreakers = candidate pool.
  **Generates candidates.** No cross-dealbreaker scoring — each
  semantic dealbreaker scores only the movies its own probe
  retrieved.
- **Preference — P1 (score-only):** Triggered when ≥1 inclusion
  dealbreaker exists (semantic or non-semantic). Step 3 LLM emits a
  `SemanticPreferenceSpec` (1+ of 8 spaces, per-space structured
  bodies, `central` / `supporting` weights). Execution runs one
  filtered `query_points` per selected space against the pool, in
  parallel. **Does not generate candidates.**
- **Preference — P2 (candidate-generating):** Triggered when zero
  inclusion dealbreakers exist. Each selected space runs top-N (cap
  2000) against the full corpus; union across spaces forms the pool.
  Pool members missing a per-space cosine are backfilled via one
  filtered `query_points` per space (only non-empty backfills fire).
  **Generates candidates.**
- **Exclusion:** Uses the dealbreaker execution path identically
  (single non-anchor space, global calibration). **Never generates a
  standalone exclusion ID list and never hard-filters** — only
  downranks already-seeded candidates in Phase 4b.

### Endpoint 7: Trending

- **Dealbreaker:** No LLM call. Execution reads the
  `trending:current` Redis hash and emits every present ID (typically
  the TMDB top-500). **Generates candidates** (full hash, no
  non-zero cutoff applied at this stage — tail-of-hash zeros still
  enter the pool but contribute nothing to scoring).
- **Preference:** Same Redis read; pulls the precomputed trending
  score for each assembled pool member. **Does not generate
  candidates** in the standard flow. Becomes a candidate generator
  in the zero-inclusion preference flow.
- **Exclusion:** Not a meaningful exclusion concept. Treated as a
  no-op if it appears.

## Execution & Aggregation Pipeline

How the per-endpoint pieces above compose into a single ordered run
for one Step-1 branch. Three concerns: execution ordering, candidate
pool assembly, and final score computation.

### 1. Execution ordering

**Timeout model.** Each Step-3 LLM translation call runs under its
own independent 20-second timeout. If a single translation stalls,
that endpoint receives an empty return object and the rest of the
branch continues without it. There is no overarching branch-level
budget — the branch finishes when every endpoint has either returned
or hit its own timeout. Soft-failure semantics: debug records the
timeout, but no other endpoint is blocked.

**Step A — flow scenario detection (synchronous, immediately after
Step 2 returns).** Inspect Step-2 output and pick exactly one flow:

| Flow | Trigger |
|---|---|
| **Standard inclusion** | ≥1 non-semantic inclusion dealbreaker exists |
| **D2 semantic-only inclusion** | ≥1 inclusion dealbreaker, all of them semantic. Deterministic exclusions may coexist; they do not change the flow choice. |
| **P2 preference-driven** | Zero inclusion dealbreakers, ≥1 preference exists. Exclusions may coexist; they do not change the flow choice. |
| **Exclusion-only browse** | Zero inclusion dealbreakers AND zero preferences. Only exclusions present. |

The flow choice determines which role each endpoint plays
(candidate-generator vs scorer-only) and is recorded in debug.

**Step B — fan out all Step-3 LLM translations in parallel.** Every
endpoint instance with an LLM step (entity, metadata, awards,
franchise, keyword, semantic dealbreakers, semantic preferences,
semantic exclusions) starts its translation immediately under its
own 20-second timeout. No inter-endpoint dependency at the
translation layer. Trending has no LLM step.

**Step C — execute each endpoint as soon as its translation returns,
gated by whether it needs the assembled pool.**

- **Pool-independent execution** (runs the moment translation
  returns): all candidate-generating inclusion endpoints in the
  active flow, all candidate-generating preferences (P2 only), all
  deterministic exclusion lookups, the global calibration probe for
  every semantic dealbreaker / exclusion (D1 and D2 both need it),
  and the trending Redis read.
- **Pool-dependent execution** (waits until the candidate pool is
  fully assembled and deduplicated): semantic dealbreaker scoring
  in D1, semantic preference scoring in P1, the score-only scoring
  pass for any deterministic preference in the standard flow, and
  the semantic-exclusion `HasIdCondition` scoring call.

**Step D — assembly barrier.** When all **candidate-generating
inclusion endpoints in the active flow** have returned (or
soft-failed or hit their own timeout), union and dedupe their ID
sets into the branch candidate pool. Only inclusion candidate
generators participate in this union — preferences (in the standard
flow), semantic dealbreaker scoring queries, and exclusion endpoints
all sit outside it. This barrier releases the pool-dependent
endpoints from Step C.

**Step E — apply deterministic exclusions, then score the pool.**
Deterministic exclusion ID sets are subtracted from the pool. All
remaining scoring queries run in parallel against the pruned pool.
When every scoring call returns, score composition (Section 3
below) executes.

**Edge cases for ordering:**

- **D2 dual-purpose probe.** In the D2 flow, each semantic
  dealbreaker's top-N probe IS both the candidate-generating call
  AND the calibration sample — one Qdrant call serves both, not
  two.
- **Exclusion-only browse, sub-cases.** True exclusion-only browse
  is the *no-preferences* form of the zero-inclusion flow. There
  are two structurally different zero-inclusion variants and they
  must not be conflated:
  - **Exclusions + preferences (this is P2, not browse).** The
    branch is preference-driven — the preferences themselves
    become the candidate generators per the standard P2 flow.
    Exclusions are then applied to that pool in the normal
    Phase 4b way (deterministic exclusions hard-remove,
    semantic exclusions penalize during scoring). The presence
    of exclusions does not change the flow choice from P2.
  - **Exclusions only, no preferences (true browse fallback).**
    No semantic search runs as a candidate generator. The seed
    is `movie_card` top 2000 ordered by the prior-based browse
    score derived from `quality_prior` and `notability_prior`.
    Deterministic exclusion translations and semantic exclusion
    translations still fan out in parallel under their own
    20-second timeouts. After the seed is assembled, deterministic
    exclusions hard-remove from it; semantic exclusions then run
    against the pruned seed and contribute penalties at scoring
    time. Final ranking is `(prior-based preference contribution)
    − exclusion penalties`. `dealbreaker_sum = 0` for every
    candidate.
- **Endpoint soft-failure.** Any translation or execution call that
  errors or hits its own 20s timeout returns an empty contribution.
  The branch continues. Debug distinguishes soft-fail from
  no-match.

### 2. Candidate pool assembly

The single deduplicated branch pool is built once per branch in
Step D. Composition by flow:

| Flow | Sources unioned into pool |
|---|---|
| **Standard inclusion** | Every non-semantic inclusion dealbreaker's candidate set (entity / metadata / awards / franchise / keyword / trending). Semantic inclusion dealbreakers contribute nothing **in this flow only** — their D1 scoring runs against the pool the deterministic endpoints assembled. |
| **D2 semantic-only** | Each semantic inclusion dealbreaker's top-N probe (cap 2000 per dealbreaker). Used whenever every inclusion dealbreaker is semantic, even when deterministic exclusions also exist in the same branch — exclusions are not inclusions and do not change the flow. |
| **P2 preference-driven** | Every candidate-generating preference's result set. Non-semantic preferences contribute their natural set; semantic preferences contribute their per-space top-N (cap 2000 per space). |
| **Exclusion-only browse** | Top 2000 movie IDs from `movie_card` ordered by the prior-based browse score. |

After the union/dedupe, **all deterministic exclusion ID sets are
subtracted from the pool** (Phase 4b hard filter). Semantic
exclusions never modify pool membership — they only contribute a
penalty term during scoring.

**Edge cases for pool assembly:**

- **"Semantic dealbreakers contribute nothing" is flow-scoped.**
  This statement applies only when at least one non-semantic
  inclusion dealbreaker exists (the standard flow). When the only
  inclusion dealbreakers in a branch are semantic, we are in D2
  and those semantic dealbreakers ARE the candidate generators —
  even if deterministic exclusions also exist in the same branch.
  This does not conflict with the rest of the plan; D2's trigger
  makes no statement about exclusions, so a "semantic-inclusions +
  deterministic-exclusions" branch (e.g., "good date night movies
  not with adam sandler") cleanly stays in D2.
- **Empty pool.** If every candidate-generating endpoint returns
  empty (or all results get pruned by deterministic exclusions),
  the branch returns zero results. Scoring is skipped.
- **Asymmetric pool membership.** A movie matched by one inclusion
  dealbreaker but not by another still enters the pool — the union
  is OR, not AND. The continuous scoring model handles the
  intersection by giving 0 credit for missed dealbreakers (see the
  default-zero rule in Section 3).
- **Trending tail-of-hash zeros.** The trending dealbreaker
  contributes every hash entry to the pool, including 0.0-score
  entries. They survive into scoring and add 0 to `dealbreaker_sum`.
- **No further trimming during assembly.** Even a 7K-pool from a
  major-studio entity union flows directly into scoring. Trimming
  only happens at the final top-100 cut.
- **Per-endpoint caps don't bound the union.** The 2000 caps apply
  per vector search and per browse seed; the unioned pool can
  exceed any single endpoint's cap.
- **Outstanding (backfill).** If the post-exclusion pool is smaller
  than 100, we currently return whatever remains. Whether to
  backfill from a fallback browse query is open.

### 3. Final score calculation

Every candidate in the assembled, exclusion-pruned pool gets one
scalar:

```text
final_score = dealbreaker_sum + preference_contribution - exclusion_penalties
```

All three terms are computed against the assembled pool, not against
any endpoint's internal result set. Per-endpoint scoring formulas
are out of scope here — see Phase 4c of the finalized proposal for
those.

**Default-zero rule for missing IDs.** When an endpoint's returned
result set does not contain a given candidate ID:

- For inclusion dealbreakers and preferences: the candidate scores
  **0** for that endpoint. No contribution.
- For deterministic exclusions: a missing ID means the candidate
  did not match the exclusion criteria — so it is **not removed and
  not penalized**. Same effect as scoring 0.
- For semantic exclusions: a missing ID (or one below the elbow
  floor) contributes **0** to `exclusion_penalties`. Not excluded,
  not penalized.

This single rule applies uniformly across all roles — every endpoint
contributes 0 when the candidate isn't in its result set. Missing
from an inclusion endpoint hurts a candidate (0 instead of >0);
missing from an exclusion endpoint helps it (0 instead of penalty).

**`dealbreaker_sum`** — sum across all inclusion dealbreakers in the
branch. Each dealbreaker yields a per-candidate score in [0, 1] (or
0 by the default-zero rule). Gradient dealbreakers (metadata,
semantic, actor prominence, trending) compute a real score for every
candidate from the candidate's own data. Bounded above by the
dealbreaker count, below by 0.

**`preference_contribution`** — single weighted average across every
preference and every active prior:

```text
preference_contribution = P_CAP × ( Σ(w_i × score_i) / Σ(w_i) )
```

P_CAP starts at 0.9. Weights:

| Signal | Weight |
|---|---|
| Regular preference | 1.0 |
| Primary preference (`is_primary_preference=true`) | 3.0 |
| Quality / notability prior — enhanced or inverted | 1.5 |
| Quality / notability prior — standard | 0.75 |
| Quality / notability prior — suppressed | 0.0 |

When the denominator is 0 (no active preferences and both priors
suppressed), the term is 0.

**`exclusion_penalties`** — sum across semantic exclusions only:

```text
exclusion_penalties = Σ_excl ( E_MULT × match_score_excl )
```

E_MULT starts at 2.0. Deterministic exclusions are not represented
here — they were already enforced by pool subtraction in Phase 4b.

**Final ranking.** Sort the pool by `final_score` descending, take
the top 100, fetch display fields via `fetch_movie_cards()`, and
restore rank order (the fetch does not preserve input order).

**Edge cases for scoring:**

The D2, P2, and exclusion-only browse flows share one structural
property: they have no candidate-generating *non-semantic* inclusion
dealbreakers. The score formula is **identical** in all three flows —
what differs is which terms are non-zero and how they were
populated. Per-flow specifics:

- **D2 (semantic-only inclusion).** There ARE inclusion dealbreakers
  — they're just all semantic. Each contributes its elbow-calibrated
  [0, 1] score to `dealbreaker_sum`, but only for the candidates its
  own probe retrieved (no cross-dealbreaker scoring). A candidate
  retrieved by semantic dealbreaker A but not B gets A's real score
  plus 0 for B by the default-zero rule. So `dealbreaker_sum` is
  non-zero in D2; it's just produced asymmetrically. Preferences
  (P1 against the D2-assembled pool) and exclusions then layer on
  normally.
- **P2 (preference-driven).** Zero inclusion dealbreakers exist by
  definition, so `dealbreaker_sum = 0` for every candidate. The
  preferences that generated the candidates ALSO score them — they
  are not double-counted as dealbreakers, they remain preferences.
  Ranking is driven entirely by `preference_contribution −
  exclusion_penalties`. Pool composition guarantees every candidate
  has at least some preference signal, so the preference weighted
  average is meaningful.
- **Exclusion-only browse.** Zero inclusion dealbreakers AND zero
  preferences. `dealbreaker_sum = 0` for every candidate.
  `preference_contribution` reduces to whatever the priors
  contribute under their Step-2-assigned weights (defaults are
  standard / standard). Final ranking is `(P_CAP × prior weighted
  average) − exclusion_penalties`. The browse seed's prior-based
  ordering and the prior-driven preference contribution use the
  same underlying signals — that's intentional, not a double count.

Other scoring edge cases:

- **Unbounded below zero.** Final scores can go arbitrarily negative
  when semantic exclusions match strongly. Intentional — that's
  what makes exclusions meaningful.
- **Score band guarantee.** Because each dealbreaker contributes up
  to 1.0 and `preference_contribution ≤ P_CAP < 1.0`, preferences
  alone cannot overcome a full dealbreaker miss. They CAN overcome
  partial misses (gradient dealbreakers, actor billing decay,
  semantic decay).
- **Asymmetric per-endpoint scoring cost.** Binary dealbreaker
  scoring is a set-membership test against the endpoint's result
  set — fast, no per-candidate query. Gradient dealbreaker scoring
  needs the underlying column or vector value for every pool member
  (one bulk SQL fetch for metadata, one filtered `query_points`
  call per dealbreaker for semantic).
- **Ties.** Identical `final_score` candidates stay in whatever
  order they happen to land in after the sort. They're rare enough
  that imposing a deterministic tiebreaker isn't worth the
  complexity.

## Initial implementation parameters

- Step-1 branches are fully isolated
- Candidate cap for vector candidate generation: 2000 per vector
- Seed size for browse-style `movie_card` fallback: 2000
- Top results returned: 100
- Initial returned fields per result:
  - `tmdb_id`
  - `movie_title`
  - `release_date`
  - `poster_url`
- Existing `movie_card` fields are sufficient for this initial payload
- Ignore caching for the initial implementation

## Debug requirements

Debug output should include enough detail to understand both latency and score
composition.

Minimum Step-4 debug expectations:

- final branch-level scenario chosen
  - standard inclusion flow
  - D2 semantic-only inclusion flow
  - zero-inclusion preference-generated flow
  - zero-inclusion browse fallback
- per-endpoint timing split into:
  - `llm_generation_ms`
  - `query_execution_ms`
- candidate counts after major phases:
  - seeded count
  - count after deterministic exclusions
  - count after scoring/reranking trimming
- per-result score breakdown suitable for debugging
  - dealbreaker sum
  - preference contribution
  - exclusion penalties
  - final score

## Important implementation considerations

- `fetch_movie_cards()` does not guarantee the same order as the ranked input
  IDs; Step 4 must restore ranking order after fetching display rows
- endpoint soft-failures should remain distinguishable in debug output even
  when the branch continues successfully
- preference-mode endpoint outputs can legitimately contain explicit `0.0`
  rows; Step 4 must not treat those as missing results
- semantic exclusions should run against the already-seeded pool; they are not
  a candidate-generation mechanism
- browse seeding should not accidentally leak into scoring as implicit
  dealbreaker credit

## Outstanding questions

### Prior-based browse score formula

The branch-level rule is finalized: use the two priors to order the exclusion-
only browse seed pool.

Still open:

- should browse seeding use the exact same weighted-average prior formula as
  normal `preference_contribution`, or a simpler fetch-time ordering formula?
- how should `suppressed` behave in fetch ordering when both priors are
  suppressed?

### Zero-inclusion preference generation mechanics

The branch-level rule is finalized: when preferences exist and there are no
inclusion dealbreakers, preferences generate candidates.

Still open:

- exact per-endpoint caps and merge behavior when multiple non-semantic
  preferences all generate natural candidate sets
- whether all preference endpoints should use a common cap or endpoint-specific
  caps

### Backfill

Resolved:

- tie-breaking: identical `final_score` candidates stay in whatever order
  they land in after the sort. No deterministic tiebreaker — ties are rare
  enough that the complexity isn't justified.

Still open:

- what to do if exclusions or missing `movie_card` rows leave fewer than 100
  valid displayable results after initial trimming

### Debug schema shape

Still open:

- exact response structure for Step-4 debug data
- whether per-endpoint score breakdowns should be returned inline with each
  result or only in a parallel debug object
