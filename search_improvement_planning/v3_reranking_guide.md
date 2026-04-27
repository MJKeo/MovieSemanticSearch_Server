# v3 Reranking Guide

How a query's decomposed traits combine into a final rank-ordered
result list. This doc covers the rescoring layer that sits on top
of the carving / qualifying role taxonomy defined in
`carving_qualifying_boundaries.md`.

Read alongside:
- `carving_qualifying_boundaries.md` — the four-cell role taxonomy
  (positive/negative carver, positive/negative qualifier).
- `v3_proposed_changes.md` — the broader v3 change list. This doc
  expands #7 (trait grouping + within-group normalization), #9
  (5-state balance), #10 (multiplicative implicit priors), and #11
  (rarity-weighted scoring) into a coherent rescoring story.

---

## The picture in one paragraph

Each high-level trait the user expressed (after step 2 decomposition,
role classification, and grouping) contributes a weighted score to
one of two sides of the score: a carver score and a qualifier score.
Carver weights come from corpus-derived **rarity**. Qualifier weights
come from LLM-derived **salience**. The two sides are combined under
a per-query **balance state** (1 of 5 LLM-picked buckets) that
controls how much of the final rank order is driven by carving vs
qualifying. **Implicit priors** (popularity, quality, recency) sit
on top as a separate multiplicative layer with its own LLM-derived
strength state and dimension-level suppression by mode words in the
query.

Negative carvers do not contribute scores — they remove candidates
before scoring runs. Negative qualifiers contribute downward pushes
on the qualifier side (with a TODO to scale them asymmetrically vs
positive qualifiers).

---

## Carver score contribution

### Mechanism: rarity-weighted

Each positive carver has a weight derived from how rare its match
is across the corpus — specifically, the percentage of the corpus
that scores at or above the inclusion threshold for that carver.
A trait that 0.5% of the corpus matches contributes more per unit
of match strength than a trait 30% of the corpus matches, because
the rarer trait is more identifying.

Within the carver group, weighted scores are aggregated, then the
group is normalized (per v3 #7) so that expansion volume — a
multi-actor query, multi-vector-space semantic query, etc. — does
not distort the contribution of a single user-stated intent.

### Shape of the rarity-to-weight curve

**Threshold-then-flat with a soft ramp**, not linear.

- A floor on the common end (high-prevalence traits still help
  disambiguate; their contribution should plateau, not drop to zero).
- A ceiling on the rare end (past a certain rarity, additional
  rarity is noise; rewarding it incentivizes trivia matches).
- A ramp in between covering the "useful range" of prevalence.

The three tunable parameters are:
- Where the ramp begins (high-prevalence side floor).
- Where the ramp tops out (low-prevalence side ceiling).
- Max-to-min weight ratio (height of the ramp).

**All three must be derived empirically from the corpus + a benchmark
query set, not picked from the armchair.** See TODOs.

### Why carvers use rarity, not salience

Carving is binary at the gate (movie passes or doesn't). Among the
survivors, "this carver mattered more" is a fuzzier judgment than
for qualifiers. In practice, headline carvers tend to be rare —
people don't headline ubiquitous traits — so rarity already
approximates salience for carvers. Adding a salience knob on top
introduces another noisy LLM-derived axis without clear payoff.

Possible knob to add later if rarity proves insufficient. Not
included in v1.

---

## Qualifier score contribution

### Mechanism: salience-weighted

Each qualifier carries one of two discrete salience states,
LLM-picked from query intent:

- **Central** — a headline want; the query would feel fundamentally
  different without this qualifier.
- **Supporting** — meaningful but not load-bearing; rounds out the
  picture.

The states map to fixed weight multipliers. The exact ratio
(central:supporting) is empirical — somewhere between 1.5:1 and 4:1
based on prior intuition, peak determined by NDCG sweep on a
benchmark set.

Within the qualifier group, weighted scores aggregate and normalize
(per v3 #7), same shape as carvers.

### Why qualifiers don't use rarity

Rarity needs a corpus-wide denominator: "% of movies that match this
trait at all." Qualifiers don't have a clean denominator — they
score the candidate set on a continuous spectrum where essentially
every candidate gets some score. There is no clean "match / no
match" cut to count, so "rarity" of a qualifier match is
ill-defined. Salience does the necessary work of differentiating
qualifier weights, and it's the right primitive anyway — qualifier
strength is fundamentally about how the user phrased the want, not
about how unusual the trait is in the corpus.

### Negative qualifiers

Same shape with the polarity flipped: a strong match contributes a
downward push in the rerank rather than upward. See TODO on
asymmetric scaling — the cost of failing to demote a strong negative
match is plausibly higher than the cost of failing to credit a weak
positive one.

---

## Carver vs qualifier balance: the 5-state schedule

A per-query LLM-derived state controls how much of the final rank
order is driven by the carver score vs the qualifier score. Five
buckets:

- **Dealbreakers dominant.** Almost all weight on carvers; qualifiers
  barely steer. Example: "Show me horror movies from the 90s starring
  Anthony Hopkins."
- **Dealbreakers lean.** Carvers gate and dominate ranking; qualifiers
  steer at the margin. Example: "Horror movies from the 90s, ideally
  creepy and atmospheric."
- **Balanced.** Carvers and qualifiers contribute roughly equally
  once you're inside the carver-defined pool. Example: "Funny,
  feel-good 90s comedy for date night."
- **Preferences lean.** Qualifiers carry the headline; carvers act
  as soft guides. Example: "Something dark and slow-burn, ideally a
  thriller."
- **Preferences dominant.** Almost all weight on qualifiers; carvers
  (if any) barely participate. Example: "I want something cathartic
  and emotionally devastating but in a hopeful way."

The state maps to a fixed weight schedule (carver_weight,
qualifier_weight pair). The actual numbers are tunable; LLM picks
the state, not the numbers.

**Drive the state from intent, not from trait counts.** A query with
five carvers and one qualifier can still be qualifier-load-bearing
if that qualifier is the headline — count is a poor proxy for
balance.

---

## Implicit priors: the separate axis

### Why separate, not a qualifier

Treating implicit priors as "qualifiers we made explicit" is
tempting for consistency but conflates two different roles:

- An explicit qualifier is **user assertion** — "the user wants
  this."
- An implicit prior is **background inference** — "absent contrary
  signal, assume this."

Different epistemic status. Implicit priors should defer (not
compete) when explicit signal exists in the same dimension; that
deferral logic is awkward inside the qualifier mechanism and clean
in a separate one. They also don't have a corpus-rarity story (every
movie has a popularity score; "rare" doesn't apply meaningfully).

Conclusion: implicit priors are their own layer with their own
strength modulation, applied multiplicatively on top of the
combined carver+qualifier base score.

### Multiplicative application

Per v3 #10: `final = base * (1 + α * prior_factor)`.

Multiplicative scaling means a movie with a low base score doesn't
get pumped to relevance just for being popular — the prior's
influence scales with how well the movie matched explicit
constraints. A relevant-and-popular movie pulls ahead of a
relevant-and-obscure one without obscuring the relevance signal.

### Strength modulation: implicit priors get their own salience-style state

Independent of the carver/qualifier balance state. The implicit
prior's strength (the α coefficient, expressed as a percentage
boost rather than a fixed score addition) is determined by query
properties:

- **Query specificity / qualifier richness.** A query with many or
  rich qualifiers has already painted the picture; less room for
  priors to act. A sparse query leaves silence for priors to fill.
- **Explicit reranking-mode mentions.** Words like "best",
  "iconic", "underrated", "modern", "comfort watch" speak directly
  to prior dimensions. They cause two effects:
  - **Suppress** the corresponding prior when the user is now
    driving it explicitly (popularity prior defers to "underrated").
  - **Amplify** an aligned prior when the language reinforces it
    ("iconic" amplifies popularity rather than replacing it).
- **Sheer qualifier count.** Mostly redundant with specificity but
  provides a floor — even broad qualifiers in volume reduce
  headroom.

The "80s action movies" canonical case: pure carvers, no qualifiers,
no quality language. Implicit prior strength should land high — the
user obviously wants the genre canon ranked by popularity, not
direct-to-video obscurities. The carver/qualifier balance state
(dealbreakers dominant) is unrelated to this — implicit prior
strength is its own decision.

Pragmatic shape: discrete states (low / medium / high implicit-prior
strength), LLM-picked the same way salience is picked, plus
per-dimension suppression / amplification rules driven by mode-word
detection.

---

## Per-trait scoring summary

| Layer | Weight source | When |
|---|---|---|
| Carver score contribution | Corpus-derived rarity (per trait) | Always for positive carvers |
| Qualifier score contribution | LLM-derived salience (central / supporting) | Always for positive and negative qualifiers |
| Carver vs qualifier balance | LLM-derived 5-state schedule (per query) | Always |
| Implicit prior strength | LLM-derived state + mode-word rules (per query) | Always; multiplicative on top |
| Negative carvers | None — they remove candidates before scoring | Always for negative carvers |

Within-group normalization (per v3 #7) applies to both the carver
group and the qualifier group before they combine under the balance
state — expansion volume must not distort intent weight.

---

## New attributes the system needs to generate

Things the upstream stages must now emit that they don't today.

- **Per-trait rarity statistic.** % of the corpus that scores at or
  above the inclusion threshold for the trait. Computed offline from
  the corpus, looked up at query time. Different mechanic for
  different endpoints — a keyword tag has a literal count, a
  semantic carver needs a per-vector-space distribution probe.
- **Per-qualifier salience state.** `central` or `supporting`,
  LLM-emitted at trait extraction time alongside polarity and role.
- **Per-query carver/qualifier balance state.** One of 5,
  LLM-emitted at step 2 from the whole-query intent.
- **Per-query implicit-prior strength state.** Low / medium / high
  (or similar discrete states), LLM-emitted at step 2.
- **Mode-word detection.** A small registry of mode words
  ("best", "iconic", "underrated", "modern", "comfort", etc.)
  mapped to the prior dimension(s) they suppress or amplify.
  Probably a curated lexicon — see v3 #4 — rather than free-form
  LLM judgment.

---

## Data the system needs to gather (offline)

- **Per-trait corpus prevalence tables.** For every trait the
  rarity weight will look up at query time, the offline pipeline
  must compute and cache the prevalence statistic. Refreshed when
  the corpus changes materially.
- **Benchmark query set with hand-ranked ideal results.** Required
  to do the empirical sweeps that determine the rarity ramp shape,
  the salience ratio, the balance-state weight schedule, and the
  implicit-prior strength multipliers. Without this set there is
  no principled way to pick numbers.

---

## Open TODOs

In rough priority order:

- **Carver rarity ramp shape and weight bounds — empirical sweep on
  the corpus.** Three values: ramp-start prevalence, ramp-end
  prevalence, max:min weight ratio. Cannot be picked from intuition.
- **Salience weight ratio (central:supporting) — NDCG sweep.** Prior
  intuition is somewhere between 1.5:1 and 4:1. Pick the peak from
  benchmark.
- **5-state balance schedule weights — NDCG sweep.** What are the
  actual (carver_weight, qualifier_weight) pairs for each of the
  five states.
- **Implicit prior α values — NDCG sweep.** What % boost low /
  medium / high actually correspond to. Per-dimension or shared
  across popularity / quality / recency.
- **Per-dimension implicit-prior suppression / amplification rules.**
  Curated mapping of mode words to prior-dimension effects. Likely
  builds on v3 #4 (vague-quality lexicon).
- **Asymmetric scaling on negative-qualifier downrank contributions.**
  Probably a single multiplier > 1 applied to negative-qualifier
  weights vs positive-qualifier weights of the same salience.
  Empirical.
- **Build the benchmark query set.** Prerequisite to every empirical
  TODO above.
- **Per-trait rarity computation pipeline.** Offline job that emits
  the prevalence tables. Mechanic differs per endpoint (keyword
  count vs semantic distribution probe vs metadata histogram).

Notably **not** TODOs anymore (decided this round):

- Whether qualifiers also get rarity weighting — **no**, salience
  only. Rarity has no clean denominator on the qualifier side.
- Whether carvers also get salience — **no**, rarity does the work.
  Possible future knob if rarity proves insufficient.
- Whether implicit priors are a flavor of qualifier — **no**, their
  own layer with their own modulation.
- Whether trait-extraction confidence factors into the weight — **no**,
  treat extraction as truth.
