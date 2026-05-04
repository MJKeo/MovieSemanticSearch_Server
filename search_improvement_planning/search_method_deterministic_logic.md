# Search Method Deterministic Logic

This document defines the deterministic logic that runs *between*
Step 2 (trait extraction) and final result assembly. It covers:

- The trait payload Step 2 emits.
- Per-endpoint mode (candidate finder vs pool reranker).
- The granularity hierarchy that orchestrates execution.
- Trait classification (candidate-generating vs pure-reranker).
- Execution flow.
- Within-trait scoring, including elbow normalization for semantic.
- Across-trait weighting via commitment + rarity.
- Polarity rules (positive opportunity cost vs negative active
  subtraction).
- Final score formula.
- Tier-fallback promotion (when no candidate-generating traits
  exist in the query).
- Edge cases.
- Open items to tune empirically.

The decisions in this doc are **deterministic from the trait
surface** — no LLM judgment is invoked between Step 2 commitment
and final scoring beyond what's already encoded in the trait
payload.

The underlying design principle: **everything is soft scoring.**
No trait ever filters categorically. UI filters handle the
"literally only Netflix" case; the semantic search system itself
never categorically excludes a movie based on a single trait
interpretation. This protects against catastrophic failure when
the LLM mis-interprets user intent.

---

## 1. Step 2 trait payload

Each atomic trait the LLM emits carries:

```
{
  category:    one of the 43 enum values (per query_categories.md)
  polarity:    positive | negative
  commitment:  required | elevated | neutral | supporting | diminished
}
```

**Removed from earlier designs:**

- `role` (carver/qualifier) — orchestration mode is now
  deterministic from the endpoint, not from a per-trait label.
  The LLM no longer commits to this and we no longer ignore a
  field it produced.
- `salience` as a qualifier-only field — generalized into
  `commitment` for all traits.

**Commitment axis.**

Five levels on a single importance axis. Step 2 surveys two signal
channels — **explicit** (language whose function fixes the trait's
strength) and **structural** (surface position, content load,
removability against the query's primary intent) — with the
**explicit channel taking precedence**: when it fires, the trait
commits to one of the two extremes regardless of structural
prominence; the structural channel sets the level only when the
explicit channel is silent.

| Level | Meaning | Recognition |
|---|---|---|
| `required` | User has expressed an inviolable commitment. Largest weight in final scoring (per §7); reward or penalty direction is set by polarity. **Does not** hard-gate. | Explicit channel fires with strong-assertion language: phrasing that names an inviolable constraint; phrasing that frames the trait as a precondition for the candidate being a viable watch at all (access, language, format, or viewer-fit gate) rather than a preference within the watchable space; phrasing that asserts an exclusion (the trait names something out of scope) rather than expressing a preference against (something to be downranked). |
| `elevated` | Query's structure presents the trait as the load-bearing axis the search is fundamentally about. Removing it would change what kind of movie is being asked for. | Explicit channel silent. Structural prominence: headline / leading position, bulk of the query's content load, or names the population other traits qualify. |
| `neutral` | Co-equal criterion among peers. Default level when both channels read silent or balanced. | Explicit channel silent; structural prominence balanced — no peer overshadows the trait and the trait does not overshadow peers. |
| `supporting` | Trait sits as a structural refinement on a population other traits define. Lower weight reflects structural role, not expressed softening. | Explicit channel silent. Structural prominence: trailing position, modest content load, refinement on a peer's population. |
| `diminished` | User has actively softened the trait. Lowest weight; the strongest low-weight signal because it derives from the user's own language. | Explicit channel fires with soft-framing language: phrasing whose function is to soften the trait's claim and invite the system to set it aside in exchange for matches on other axes. Includes the "not too X" special case (softened preference against, not assertion against). |

The recognition is by **function**, not by surface token. Strong
assertion does not have to take a canonical "must" form to count;
soft framing does not have to take a canonical "ideally" form. The
LLM commits the extreme when the language *functions* as assertion
or softening, even when the canonical token is absent.

Conversely: REQUIRED and DIMINISHED commit only when the explicit
channel fires. Strong implicit prominence alone commits ELEVATED,
not REQUIRED. Structural triviality alone commits SUPPORTING, not
DIMINISHED.

---

## 2. Per-endpoint mode

The endpoint determines whether a call runs as a **candidate
finder** (produces its own pool by membership / top-K) or a
**pool reranker** (scores an existing candidate pool with
additive signal).

The principle: endpoints whose signal is discrete membership and
composes cleanly under additive scoring are candidate finders.
Endpoints whose signal is continuous similarity, blurs across
many candidates, or shifts meaning depending on surrounding
traits are rerankers — their job is to sharpen ordering within
an already-narrowed pool.

### Always candidate finder

These endpoints always run as candidate finders. Posting hits,
array overlaps, tag presence, and concrete records produce clean
membership signals. Multiple traits hitting the same endpoint
combine cleanly (intersection / union / additive score sum) and
produce a coherent "matches all of these" pool.

| Endpoint | Mechanism | Why finder-only |
|---|---|---|
| **ENT** | Posting tables for actor / director / writer / composer / character / title | Posting hit is near-binary; multiple credits intersect cleanly. The signal never shifts meaning with context. |
| **FRA** | Two-phase token resolution → array overlap on lineage / shared_universe / subgroup | Lineage membership is discrete; "Star Wars OR MCU" composes by union without ambiguity. |
| **STU** | ProductionBrand enum path or DF-filtered token intersect | Brand membership is discrete; multiple studios compose by union. |
| **KW** | Keyword tag presence (single or multi-overlap) | Tags are binary presence; ZOMBIE + COMEDY + REDEMPTION sums into a clean "has all three" signal. Even when KW augments a semantic call (Bucket 6), the endpoint's job is still membership lookup — only the orchestration merges. |
| **AWD** | `movie_awards` filter with COUNT thresholds; fast path on `award_ceremony_win_ids` | Concrete ceremony records; thresholds are filters by construction. |
| **TRENDING** | Live-refreshed trending set | Returns a fixed set by definition; "kind of trending" has no meaning. |

### Always pool reranker

These endpoints always run as rerankers when at least one
candidate-generating trait exists in the query. Their signal is
continuous similarity (vector cosine) or continuous prior, and
their meaning shifts depending on the surrounding traits.

The justification is mechanical: multiple semantic / prior signals
compose by additive cosine across the same candidates, which
expresses conjunction ("matches all of these traits") better than
any single space's top-K can. Pool-finder mode for these
endpoints — top-K of one cosine signal — cannot represent
conjunction across multiple signals.

| Endpoint | Why reranker-only |
|---|---|
| **P-EVT** | Continuous cosine over plot-event / setting prose. Multiple subqueries on the same space don't combine as "AND." |
| **P-ANA** | Theme intensity is spectrum scoring; "grief + redemption" sum is fuzzy "near both" rather than "is both." |
| **VWX** | Tone / palette / tension are inherently context-dependent feel axes. "Slow burn" means different things across genres. |
| **CTX** | Watch-context fit is a judgment over candidates, not a category of movies. |
| **NRT** | Stylistic / craft scoring axes; "Sorkin-style" is a ranking signal, not membership. |
| **PRD** | Production / filming-location prose is similarity-shaped even where it's the only channel for the data (Cat 24). |
| **RCP** | Reception is ranking signal by construction — praised/criticized prose is aspect-level scoring. |

When **no** candidate-generating trait exists anywhere in the
query, the tier-fallback promotion rule (§9) elevates one or more
of these endpoints to candidate-finder duty for that query only.

### META — split by attribute

META is heterogeneous and splits cleanly by column. The split is
deterministic from the category that fired the trait, so step 2
already labels the right mode.

#### Pool-finder columns

These columns are filters or bucket lookups. Multiple traits
compose by AND without ambiguity ("90s + under 2 hrs + on Netflix
+ rated R" is unambiguously "matches all four"). Despite the word
"filter" or "gate" in the mechanism descriptions below, none of
these hard-exclude — they produce per-candidate scores in [0, 1]
with soft falloff outside the matched range.

| Column | Category | Mode |
|---|---|---|
| `release_date` | Cat 13 (era / range) | Range filter or soft-falloff range |
| `runtime` | Cat 14 | Range filter with soft-falloff; system 60-min floor enforced at dispatcher |
| `maturity_rank` | Cat 15, gate side of Cat 27 / Cat 28 | Soft scoring |
| `audio_language_ids` | Cat 16 | Filter |
| `providers` | Cat 17 | Packed-uint32 decode filter |
| `budget_bucket` | Cat 18 | Bucket lookup |
| `box_office_bucket` | Cat 18 | Bucket lookup |
| `reception_score` (threshold) | Cat 19 | Numeric threshold filter |
| `country_of_origin` | Cat 20 | Filter |
| `media_type` | Cat 21 | Filter |

#### Reranker columns

These columns are continuous priors. The doc defines them as
"additive lift, not hard threshold."

| Column | Category | Mode |
|---|---|---|
| `reception_score` (prior) | Cat 38 | Additive numeric prior |
| `popularity_score` | Cat 38, Cat 39 | Additive numeric prior |

#### One column whose mode varies — `release_date`

`release_date` flips deterministically by category:

- **Pool finder** — when fired by Cat 13 (era / range / decay
  framings: "90s movies," "before 2000," "modern," "old-school").
- **Reranker** — when fired by Cat 44 (chronological ordinal:
  "first," "last," "earliest," "latest," "most recent"). Cat 44
  is sort-and-pick over an already-scoped pool, which is
  reranker-shaped by definition.

No other META column varies. The category routing already encodes
the right mode.

---

## 3. Granularity hierarchy

Execution is structured as four nested levels:

```
query
  └── traits           (1 query → N traits)
       └── categories  (1 trait → 1 category, but each trait may
                        produce multiple endpoint calls within
                        that category)
            └── endpoint calls
```

**Mix resolution rule:** when a candidate-generating call coexists
with reranker calls, the mix is resolved at the **lowest
granularity level where it occurs.** A reranker call within a
category that has a pool finder reranks within that category's
pool-finder output. A reranker category within a trait that has a
pool-finder category reranks within the trait's pool. A pure-
reranker trait reranks within the cross-trait union.

This keeps semantic refinement scoped to the structural carve it's
meant to refine, rather than letting it leak across unrelated
parts of the query.

---

## 4. Trait classification

A trait is classified deterministically as one of two kinds:

| Kind | Definition |
|---|---|
| **Candidate-generating** | Has at least one **positive-polarity pool-finder endpoint call** anywhere in its call tree. |
| **Pure-reranker** | All endpoint calls are rerankers, OR all pool-finder calls are negative-polarity. |

**Negative-polarity pool finders are always treated as pure-
reranker for orchestration purposes.** They never add candidates
to the pool — we are not searching *for* movies that match them.
Instead they score against the existing union as downrankers.

---

## 5. Execution flow

### Step A — LLM generation phase (parallel)

All trait payloads, including all category and endpoint call
parameters across all traits, are generated by Step 2 in parallel.
Nothing is gated on execution outcomes during generation. Only the
execution phase is sequenced.

### Step B — Pool-availability check

Inspect the trait set:

- If at least one trait is **candidate-generating** → proceed to
  Step C.
- If every trait is **pure-reranker** (no positive-polarity pool
  finder anywhere in the query) → enter **tier-fallback
  promotion** (§9), then proceed to Step C with the updated
  trait set.

This check distinguishes "no candidates exist" (the empty-result
case in §10) from "we have no way to find candidates" (the
tier-fallback case). They are fundamentally different and produce
different behaviors.

### Step C — Candidate-generating traits execute in isolation, in parallel

Each candidate-generating trait produces its own scored candidate
set. Within a single candidate-generating trait:

- Pool-finder calls produce the candidate scope at their
  granularity level.
- Reranker calls scoped to that level score within it (not against
  the broader corpus).
- Per-call scores are normalized to [0, 1] (elbow normalization
  for semantic, native [0, 1] for finders — see §6).
- Trait-level score per candidate = equal-weight sum of all call
  scores ÷ number of unique calls.

### Step D — Union the candidate sets

Union all candidate-generating traits' outputs into a single pool.
A movie scored by Trait A but not Trait B appears in the union
with Trait B's contribution = 0 (opportunity cost — see §8).

### Step E — Pure-reranker traits and negative-polarity finder traits execute against the union

These traits do not add candidates; they only score what's already
in the union. Their per-trait scoring follows the same equal-
weight composite rule as in Step C.

### Step F — Final scoring

Aggregate per-candidate scores per §7 and §8 to produce the final
ranked list.

---

## 6. Within-trait scoring

All endpoint calls produce a score in [0, 1].

### Finders

Score by mechanism:

- **Postings (ENT):** role tier (e.g. lead 1.0, supporting 0.7,
  minor 0.3, default 0.5).
- **Tag presence (KW, FRA, STU, AWD):** 1.0 if matched, 0 if not;
  multi-overlap can produce intermediate values.
- **META range columns:** 1.0 inside the requested range, soft
  exponential falloff outside.
- **TRENDING:** included with weighting per trending score, or 0.

**No compressed [0.5, 1] range** — the legacy carver compression
is dropped. Match quality is reflected truthfully in [0, 1] across
all endpoints.

### Semantic rerankers — elbow normalization

For each semantic reranker call, score every candidate in the
scoped pool by cosine similarity, then apply elbow normalization:

1. Find the top cosine score across the pool.
2. Compute the clamp threshold: `top × 0.85`.
3. Any candidate with cosine ≥ clamp threshold → normalized
   score = 1.0.
4. Linear gradient from `clamp_threshold` down to 0:
   `score = max(0, cosine / clamp_threshold)`.

This applies whether the semantic call is acting as a within-trait
reranker, a cross-trait reranker, or a tier-fallback promoted
finder. The pool being scored differs across those cases; the
normalization shape is the same.

**Fallback for noisy / small / no-clear-elbow cases:** TBD (see
§11 open items).

### Composite trait score

For a given candidate within a given trait:

```
trait_score = Σ (call_score_i) / number_of_unique_calls
```

Equal weight across all calls within the trait. Multi-endpoint
trait scoring (e.g. Cat 33 firing VWX + CTX + RCP + KW) treats
all endpoints equally; per-endpoint base weights are not used.

---

## 7. Across-trait weighting

For each trait, compute a per-trait weight:

```
trait_weight = commitment_multiplier × rarity_factor
```

### Commitment multiplier

| Commitment   | Multiplier |
|--------------|-----------:|
| `required`   | 3.0        |
| `elevated`   | 1.75       |
| `neutral`    | 1.0        |
| `supporting` | 0.6        |
| `diminished` | 0.35       |

The scale is asymmetric by construction: explicit signals (REQUIRED,
DIMINISHED) reach the extremes because the user named the strength
themselves; structural signals (ELEVATED, SUPPORTING) reach the
inner half-step because they reflect inferred prominence rather
than expressed strength; NEUTRAL sits at 1.0 as the true middle.

The values form an approximately geometric scale with ratio √e ≈
1.65 per step (log-space gaps of ≈ 0.5 between adjacent levels).
This means each level is "noticeably more important than the one
below" by a constant factor rather than a constant absolute
amount, which matches how commitment intuitively scales.

The required multiplier is the lever that gives required traits
real teeth without making them de facto hard gates. Set too low
and the system ignores strong user constraints; set too high and
missing a required trait categorically excludes the candidate.
3.0 is calibrated so that in a benchmark query with 1 REQUIRED
trait and 3 NEUTRAL traits, a candidate that matches REQUIRED but
nothing else ties a candidate that misses REQUIRED but matches
all three neutrals — the "soft gate" pivot point.

### Rarity factor

Rarity weighting applies **only to candidate-generating traits**.
Pure-reranker traits and negative-polarity finder traits use
`rarity_factor = 1.0`.

| Trait kind | Rarity formula |
|---|---|
| Candidate-generating (positive-polarity finder) | log-based transform of `(candidate_count / corpus_size)`. Rarer = higher. |
| Semantic-promoted via tier fallback | log-based transform of `(count of movies that scored 1.0 on at least one of the trait's semantic fetches) / corpus_size`. |
| Pure-reranker | `rarity_factor = 1.0` |
| Negative-polarity finder | `rarity_factor` computed by finder formula (DF / corpus). Even though they don't add candidates, their match prevalence in the corpus determines per-match penalty magnitude. Rare negative trait → big per-match penalty. Common negative trait → small per-match penalty. |

**Marking semantic-promoted traits.** When tier-fallback
promotion elevates a reranker to candidate-generating duty, the
trait must be explicitly tagged so the rarity calculation uses
the semantic rule (1.0-scoring movies) rather than the finder
rule (matched candidates).

**Rarity tiers.** The log-based transform is implemented as
discrete tiers over corpus fraction. Tiered (rather than
continuous) because the natural granularity of trait specificity
is coarse — a specific actor, a niche keyword, a broad genre, a
near-universal descriptor — and tiers are easier to tune
empirically by shifting boundaries than by retuning a slope.
Corpus size N ≈ 150K movies.

| Tier        | Corpus fraction | df range (N ≈ 150K) | Rarity factor |
|-------------|----------------:|--------------------:|--------------:|
| Ultra-rare  | < 0.1%          | < 150               | 1.5           |
| Rare        | 0.1% – 1%       | 150 – 1,500         | 1.2           |
| Moderate    | 1% – 10%        | 1,500 – 15,000      | 1.0           |
| Common      | 10% – 30%       | 15,000 – 45,000     | 0.75          |
| Very common | > 30%           | > 45,000            | 0.5           |

Bounded [0.5, 1.5] — a 3× span between the rarest and most-common
traits. Wide enough that a specific actor outweighs a broad genre
on rarity alone, narrow enough that rarity refines but does not
dominate commitment. Combined with commitment, the full
`commitment × rarity` weight ranges from 0.175 (diminished × very
common) to 4.5 (required × ultra-rare).

---

## 8. Polarity rules

| Polarity | Match | Miss |
|---|---|---|
| **Positive** | Reward: `+ trait_score × trait_weight` | No reward (opportunity cost). Nothing subtracted. |
| **Negative** | Penalty: `− trait_score × trait_weight` | No penalty (the candidate "got lucky"). Nothing added. |

The asymmetric framing matters:

- **Missing a positive-polarity trait** is opportunity cost — the
  candidate forgoes a potential reward but has nothing taken
  away. This is the case for the vast majority of "trait didn't
  fire on this candidate" outcomes.
- **Matching a negative-polarity trait** is active subtraction —
  a real negative contribution to the score. This is structurally
  different: the candidate had something taken away.

This distinction is important when reasoning about why a
candidate scored low: a missed positive trait is "didn't earn
points," a matched negative trait is "lost points."

**Negative-polarity finders never add candidates to the pool.**
They orchestrate as pure rerankers (§4) — they wait for the union
and then apply their negative contribution to candidates that
match. A "no horror" trait does not cause horror movies to be
fetched into the pool only to be downranked; horror movies enter
the pool only if other traits brought them in, and then the
negative trait penalizes their score.

---

## 9. Final score formula

```
final_score(candidate) =
    Σ over all traits ( trait_score × trait_weight × polarity_sign )

where polarity_sign = +1 for positive polarity
                     -1 for negative polarity
```

**No query-length normalization.** Scores are relative within a
single query and not user-facing, so absolute magnitude across
queries doesn't matter.

---

## 10. Tier-fallback promotion

Triggered **only** when Step B determines that no candidate-
generating traits exist in the query (every trait is structurally
pure-reranker — no positive-polarity pool finder anywhere). This
is distinct from the empty-result case where candidate-generating
traits exist but all returned 0 results — see §11.

### Tier criteria

The criterion for tier assignment:

1. **Sharpness** — does the category's prose unambiguously
   identify a small subset of the corpus?
2. **Context-independence** — does the trait mean the same thing
   without other traits present?
3. **Top-K density** — are the top-ranked candidates by cosine
   likely to all be genuine matches?

Categories that score high on all three (concrete subjects,
settings, plot events) belong in the top tier. Categories that
score low (vibes, watch-context fit) belong near the bottom. Some
rerankers are never promotable: negative-polarity calls describe
what to penalize, not what to fetch, and some signals only make
sense once an independent pool already exists.

Promotion tiers apply only to **positive-polarity reranker calls**.
Negative-polarity calls are always Tier NP (never promote), even
when the underlying endpoint would normally be a pool finder.

### Tier 1 — Concrete fact / specific identifier

Top-K is tight and accurate. Promote first.

| Category | Endpoint(s) |
|---|---|
| Cat 8 — Central topic / about-ness | P-EVT |
| Cat 30 — Plot events | P-EVT |
| Cat 31 — Narrative setting | P-EVT |
| Cat 24 — Filming location | PRD |
| Cat 42 — Named source creator | P-EVT + RCP |

### Tier 2 — Concrete element / structural feature

Sharper than abstract types, broader than facts. Top-K is mostly
relevant with some adjacent matches.

| Category | Endpoint(s) |
|---|---|
| Cat 9 — Element / motif presence | P-EVT |
| Cat 22 (P-ANA fallback) — Qualifier-laden subgenre | P-ANA |
| Cat 25 — Format + visual-format specifics | PRD |
| Cat 26 — Narrative devices + structural form | NRT |

### Tier 3 — Abstract type / archetype

Type / theme intensity is fuzzy. Top-K identifies a meaningful
pool but with higher noise.

| Category | Endpoint(s) |
|---|---|
| Cat 10 — Character archetype | NRT |
| Cat 32 — Story / thematic archetype | P-ANA |

### Tier 4 — Reception / praise prose

Pool-finder mode produces "movies known for X" — a real defined
pool, but defined by reception rather than content.

| Category | Endpoint(s) |
|---|---|
| Cat 35 — Visual craft acclaim | RCP + PRD |
| Cat 36 — Music / score acclaim | RCP |
| Cat 37 — Dialogue craft acclaim | RCP + NRT |
| Cat 39 — Cultural status | RCP |
| Cat 40 — Specific praise / criticism | RCP |

### Tier 5 — Audience / sensitivity / seasonal

Mostly fuzzy semantic surface. These categories normally have a
structured gate (META.maturity_rank, KW proxy chains) and rarely
reach fallback. When they do, they promote ahead of vibes.

| Category | Endpoint(s) |
|---|---|
| Cat 27 — Target audience | CTX |
| Cat 28 — Sensitive content | VWX |
| Cat 29 — Seasonal / holiday | CTX + P-EVT (proxy) |

### Tier 6 — Vibes / context fit

Pool-finder mode produces "movies with this feel across all
genres / eras" — broad, noisy, rarely the right starting set.
Promote only if literally no higher content tier fired.

| Category | Endpoint(s) |
|---|---|
| Cat 33 — Emotional / experiential | VWX + CTX + RCP |
| Cat 34 — Viewing occasion | CTX |

### Tier 7 — Global metadata priors / ordinals

These are corpus-level ordering signals, not content identifiers.
They are still promotable before the emergency fallback because
they can produce a coherent default pool for queries like "best
movies," "popular movies," or "newest movies." Promotion means
selecting the top-K candidates by the targeted metadata direction,
then letting every remaining reranker score that pool.

| Category | Endpoint / column | Promotion shape |
|---|---|---|
| Cat 38 — General appeal / quality baseline | META.reception_score / META.popularity_score | Top-K by requested prior direction |
| Cat 39 — Cultural status / canonical stature | META.popularity_score and/or RCP | Prefer RCP if present in the fired calls; otherwise top-K by popularity prior |
| Cat 44 — Chronological ordinal | META.release_date | Top-K by requested chronological direction |

### Tier NP — Never promote

These calls are pure rerankers only. If promotion reaches this tier
without any prior tier producing candidates, do **not** promote them.
Instead seed the pool with a neutral corpus prior: the top 2,000
movies by `0.8 * normalized_popularity_score + 0.2 *
normalized_reception_score`, with both component scores normalized
to `[0, 1]`, then run all rerankers against that seeded pool.

| Case | Why never promote |
|---|---|
| Any negative-polarity endpoint call | The call names what to penalize, not what to search for. Promoting "no horror" would fetch horror movies, which is backwards. |
| Positive-polarity metadata reranker with no usable direction | A prior without a resolved direction cannot define a meaningful pool. |

### Promotion rule

When no candidate-generating trait exists in the query:

1. Ignore every negative-polarity endpoint call for promotion.
   They remain rerankers.
2. Find the highest-tier positive-polarity reranker category
   present in the trait set. If none exists, go directly to Tier
   NP and seed the neutral fallback pool.
3. That category's primary promotable endpoint or metadata column
   promotes to candidate-finder duty.
4. If that promotion produces at least one candidate, stop. The
   promoted category(s) are the **minimum set** needed.
5. If it produces zero candidates, walk down to the next tier
   and add it to the promoted set. Continue until at least one
   candidate is produced or Tier NP is reached.
6. If multiple categories tie at the same top tier, all of them
   promote (each contributes its top-K), pools union.
7. If promotion reaches Tier NP without candidates, seed the pool
   with the top 2,000 movies by `0.8 *
   normalized_popularity_score + 0.2 *
   normalized_reception_score`, with both component scores
   normalized to `[0, 1]`. This is the "we need something to
   rerank" fallback, not a trait match.
8. Mark all semantic-promoted traits as **semantic-promoted** so
   the rarity calculation uses the semantic rule (1.0-scoring
   movies per §7). Metadata-promoted traits use the metadata
   rarity/count rule for the selected top-K or threshold shape.
9. Re-enter Step C with the updated trait set.

The intent: the broadest signals do the least narrowing work.
Promoting the highest tier first means setting / topic / plot
prose (which produce sharp, content-defined pools) carve the
candidate set, vibes / occasion only promote when no sharper
content signal exists, and global priors promote only before the
neutral top-2K seed would otherwise be needed.

### Worked example

Query: **"Slow-burn movies set in 1940s Berlin with a haunting
score"** — no candidate-generating traits fire (Cat 13
release_date covers production era, not narrative setting era;
nothing else applies).

Trait set:

- Cat 31 (narrative setting) — Tier 1
- Cat 33 (slow burn) — Tier 6
- Cat 36 (haunting score) — Tier 4

Resolution:

1. Highest tier present is Tier 1 — Cat 31 promotes.
2. P-EVT runs as candidate finder: top-K (via elbow
   normalization) for "set in 1940s Berlin" defines the pool.
3. Cat 31 is marked semantic-promoted; rarity uses 1.0-scoring
   movie count.
4. Cat 33 reranks the pool via VWX + CTX + RCP for slow-burn
   feel.
5. Cat 36 reranks the pool via RCP for haunting-score acclaim.

Counter-scenario (why the tiering matters): if Cat 33 had
promoted instead, VWX top-K for "slow burn" would surface slow-
burn movies across all eras and settings, and the "1940s Berlin"
reranking would have to do all the narrowing work against a pool
that's structurally wrong for it. Promoting Cat 31 puts the
narrowing work on the sharpest signal and lets the fuzzier
signals do what they're best at — refining order within a pool
already shaped by the right axis.

---

## 11. Edge cases

| Case | Rule |
|---|---|
| Pool-finder endpoint call returns 0 candidates | That granularity level produces an empty set. Connected rerankers within that scope **do not run** and **do not promote**. The empty set propagates upward. |
| Candidate-generating category returns 0 within a trait | That category contributes nothing to the trait. Other categories within the trait still execute. |
| All categories in a candidate-generating trait return 0 | The trait contributes 0 candidates and 0 score across the board. Other candidate-generating traits proceed normally. |
| All candidate-generating traits across the query return 0 | Union is empty. **Return empty results.** Do not run pure-reranker traits, do not fall back to tier promotion. ("If something truly doesn't exist, then it doesn't exist.") |
| No candidate-generating traits exist in the query (every trait is structurally pure-reranker) | Enter tier-fallback promotion (§10). Structurally distinct from the empty-result case above — this is "we have no way to find candidates," not "no candidates exist." |
| Tier-fallback promotion produces 0 from the highest tier | Walk down to the next tier and add it to the promoted set. Continue until at least one candidate exists or Tier NP is reached. |
| Promotion reaches Tier NP without candidates | Seed the pool with the top 2,000 movies by `0.8 * normalized_popularity_score + 0.2 * normalized_reception_score`, with both component scores normalized to `[0, 1]`, then run the pure rerankers against that pool. |
| Semantic reranker fetch with no clear elbow / very small / very noisy result | Use the elbow-normalization fallback shape (TBD — see §12). |
| Trait with multiple semantic fetches needing rarity (semantic-promoted case) | Rarity = (count of movies scoring 1.0 on **at least one** of those semantic fetches) / corpus size. |
| Cat 33 / Bucket 6 with KW present and tag matches | KW pool-finds; semantic rerankers refine within the KW result. The semantic long-tail outside the KW pool does not enter (precision-over-recall by design). |
| Cat 33 / Bucket 6 with KW present but 0 tag matches | Per the empty-set rule above: trait contributes nothing. Semantic does not promote within the trait. |
| Negative-polarity finder trait (e.g. "no horror") | Treated as pure-reranker for orchestration: does not generate candidates, waits for the union, applies negative contribution to candidates that match. Rarity computed as DF/corpus per finder rules. |
| Multi-endpoint trait scoring (e.g. Cat 33 firing VWX + CTX + RCP + KW) | All endpoints weighted equally in the composite trait_score. Per-endpoint base weights are not used. |
| Negative polarity required commitment (e.g. "absolutely no horror") | Acts as heavy downranker. The 3.0 commitment multiplier × negative polarity sign produces a large penalty per match, making horror movies very unlikely to surface — but not impossible. A horror-comedy that nails everything else can still appear. |

---

## 12. Open items to tune empirically

- **Commitment multipliers.** Set per §7 (3.0 / 1.75 / 1.0 / 0.6 /
  0.35) on a √e geometric scale. Eval-data tuning may shift the
  geometric ratio (try √2 if required dominates too aggressively,
  e if it doesn't bite hard enough), but the asymmetric spread
  (explicit at extremes, structural at inner half-step) is fixed
  by design.
- **Rarity tier boundaries.** Set per §7 at 0.1% / 1% / 10% / 30%
  of corpus, with factors 1.5 / 1.2 / 1.0 / 0.75 / 0.5. Boundaries
  are decade-spaced over corpus fraction; tune by shifting
  boundaries (e.g. tighten ultra-rare to 0.05% if specific-actor
  traits over-dominate) or by widening the [0.5, 1.5] bounds if
  the 3× span is too compressed.
- **Per-endpoint candidate-generation top-K sizes.** For finders
  this is a hit-list cap; for promoted semantic traits it's
  effectively determined by elbow normalization. Need empirical
  tuning to balance recall vs compute.
- **Elbow-normalization fallback shape.** What to use when
  semantic search has no clear elbow, returns very few results,
  or is very noisy. Options include fixed sigmoid, top-K with
  hard threshold, raw cosine. Decide based on observed failure
  modes.
- **Posting role-tier values.** Lead 1.0 / supporting 0.7 / minor
  0.3 / default 0.5 are starting values; revisit after eval data.
- **Implicit quality / popularity booster.** Deferred — handled
  as a separate concern outside this scoring framework.
