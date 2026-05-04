# Search Method Deterministic Logic

This document defines, per endpoint, when a search runs as a
**candidate finder** (produces its own pool by membership / top-K)
versus a **pool reranker** (scores an existing candidate pool with
additive signal). The decision is endpoint-driven, not category-
driven, and is deterministic from the trait surface.

The underlying principle: endpoints whose signal is discrete
membership and composes cleanly under additive scoring across
multiple traits are candidate finders. Endpoints whose signal is
continuous similarity, blurs across many candidates, or shifts
meaning depending on the rest of the trait set are rerankers —
their job is to sharpen ordering within an already-narrowed pool.

---

## Always candidate finder

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

---

## Always pool reranker

These endpoints always run as rerankers when at least one
candidate finder fires elsewhere in the query. Their signal is
continuous similarity (vector cosine) or continuous prior, and
their meaning shifts depending on the surrounding traits. They
sharpen ordering within an already-narrowed pool far better than
they define a set.

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

When **no** candidate finder fires anywhere in the query, the
fallback promotion rule (below) elevates one or more of these
endpoints to candidate-finder duty for that query only.

---

## META — split by attribute

META is heterogeneous and splits cleanly by column. The split is
deterministic from the category that fired the trait, so step 2
already labels the right mode.

### Pool-finder columns

These columns are filters or bucket lookups. Multiple traits
compose by AND without ambiguity ("90s + under 2 hrs + on Netflix
+ rated R" is unambiguously "matches all four").

| Column | Category | Mode |
|---|---|---|
| `release_date` | Cat 13 (era / range) | Range filter or soft-falloff range |
| `runtime` | Cat 14 | Range filter with soft-falloff; system 60-min floor enforced at dispatcher |
| `maturity_rank` | Cat 15, gate side of Cat 27 / Cat 28 | Hard gate |
| `audio_language_ids` | Cat 16 | Filter |
| `providers` | Cat 17 | Packed-uint32 decode filter |
| `budget_bucket` | Cat 18 | Bucket lookup |
| `box_office_bucket` | Cat 18 | Bucket lookup |
| `reception_score` (threshold) | Cat 19 | Numeric threshold filter |
| `country_of_origin` | Cat 20 | Filter |
| `media_type` | Cat 21 | Filter |

### Reranker columns

These columns are continuous priors. The doc defines them as
"additive lift, not hard threshold."

| Column | Category | Mode |
|---|---|---|
| `reception_score` (prior) | Cat 38 | Additive numeric prior |
| `popularity_score` | Cat 38, Cat 39 | Additive numeric prior |

### One column whose mode varies — `release_date`

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

## Reranker-promotion fallback (no candidate finders fired)

When a query produces no candidate-finder traits at all, the
query has no pool to rerank against. In that case, one or more
reranker endpoints temporarily promote to candidate-finder duty
for that query only. The promotion follows a tier system based
on how well each semantic category produces a useful pool by
top-K cosine.

The criterion for tier assignment:

1. **Sharpness** — does the category's prose unambiguously
   identify a small subset of the corpus?
2. **Context-independence** — does the trait mean the same thing
   without other traits present?
3. **Top-K density** — are the top-ranked candidates by cosine
   likely to all be genuine matches?

Categories that score high on all three (concrete subjects,
settings, plot events) belong in the top tier. Categories that
score low (vibes, watch-context fit) belong in the bottom tier.

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
Promote only if literally no higher tier fired.

| Category | Endpoint(s) |
|---|---|
| Cat 33 — Emotional / experiential | VWX + CTX + RCP |
| Cat 34 — Viewing occasion | CTX |

### Promotion rule

When no candidate finder fires anywhere in the query:

1. Find the highest-tier semantic category present in the trait
   set.
2. That category's primary semantic endpoint promotes to
   candidate finder (top-K).
3. All remaining semantic categories rerank against that pool as
   normal.
4. If multiple categories tie at the top tier, all of them run
   as candidate finders (each contributes its top-K), pools
   union, then everything reranks against the union.

The intent is that the broadest signals do the least narrowing
work. Promoting the highest tier first means setting / topic /
plot prose (which produce sharp, content-defined pools) carve
the candidate set, and vibes / occasion (which would produce
broad, fuzzy pools) only ever rerank.

### Worked example

Query: **"Slow-burn movies set in 1940s Berlin with a haunting
score"** — no structured carvers fire (Cat 13 release_date
covers production era, not narrative setting era; nothing else
applies).

Trait set:
- Cat 31 (narrative setting) — Tier 1
- Cat 33 (slow burn) — Tier 6
- Cat 36 (haunting score) — Tier 4

Resolution:
1. Highest tier present is Tier 1 — Cat 31 promotes.
2. P-EVT runs as candidate finder: top-K for "set in 1940s
   Berlin" defines the pool.
3. Cat 33 reranks the pool via VWX for slow-burn feel.
4. Cat 36 reranks the pool via RCP for haunting-score acclaim.

Counter-scenario (why the tiering matters): if Cat 33 had
promoted instead, VWX top-K for "slow burn" would surface slow-
burn movies across all eras and settings, and the "1940s Berlin"
reranking would have to do all the narrowing work against a pool
that's structurally wrong for it. Promoting Cat 31 puts the
narrowing work on the sharpest signal and lets the fuzzier
signals do what they're best at — refining order within a pool
already shaped by the right axis.
