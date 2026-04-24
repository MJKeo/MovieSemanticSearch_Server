# Category Handler Planning

Finalized decisions from the category-handler design conversation.
Scope: how each category in [query_categories.md](query_categories.md)
translates into runnable code — what the handler looks like, what it
returns, and how the orchestrator composes handlers into a final
candidate set.

---

## The four handler types

Every category handler uses one of four orchestration shapes. The
shape governs how many endpoint queries can fire for that handler
on a given user request. Shapes fall into two families — shapes
that cap the handler at a single query, and shapes that allow
multiple queries to fire.

### At most one query fires

#### Type 1 — Single endpoint
Only one endpoint is ever applicable to the category. No routing
decision to make at query-generation time. This is the degenerate
case; the runtime codepath is identical to the other handler types
and returns the same four buckets.

Covers: Cats 1, 2, 3, 4, 5, 8, 9, 10, 13, 20, 22, 24, 27, 29, 30, 31.

#### Type 2 — Mutually exclusive
Two (or more) endpoints each individually capable of answering the
question, but they answer *different versions* of it. The
query-generation LLM picks whichever matches the user's framing and
ignores the others. Firing both would mix answers to different
questions rather than reinforce a single one.

Covers: Cats 11, 12.

#### Type 3 — Tiered
An ordered preference list of endpoints. The LLM walks the list at
query-generation time and fires whichever is the first *genuine fit*
for the user's phrasing. Earlier tiers are authoritative when they
apply; later tiers are fallbacks for cases the earlier tiers can't
cleanly express (typically spectrum-framed or long-tail asks outside
the canonical vocabulary).

Covers: Cats 6, 7, 14, 15, 16, 21, 26.

### More than one query may fire

#### Type 4 — Combo
Multiple endpoints apply to the same request and each carries
distinct, complementary signal that can't be collapsed into a single
call. All applicable endpoints fire in parallel; their outputs
populate the handler's return buckets. Reserved for categories where
forcing a single endpoint would drop real signal — either because no
single endpoint's data shape fully covers the question, or because
the question is inherently multi-faceted (e.g. seasonal intent spans
proxy tags, watch context, and narrative setting at once).

Covers: Cats 17, 18, 19, 23, 25, 28.

### Why gate + inclusion isn't a fifth type
An earlier "gate + inclusion" shape (for maturity / sensitive-content
gating, Cats 17 and 18) collapsed into Type 4 (Combo) plus the
`exclusion_ids` return bucket. The gate is not an orchestration
shape — it's a cross-cutting exclusion that applies to the final
candidate pool. Whether an endpoint's output lands in
`inclusion_candidates` or `exclusion_ids` is a bucket-population
choice driven by the query, not a separate runtime pattern.

### Why interpretation-first isn't a fifth type
An earlier "interpretation-first" shape (for Cats 28 and 31)
collapsed too. Interpretation is not a separate LLM stage — it's
extra schema fields the handler's existing LLM fills inline (intent
rationale, confidence, list-meaning decoding). No added latency, no
separate codepath.

---

## Modular handler construction

Handlers should not be hand-designed per category. They should be
assembled from composable pieces keyed off (a) the handler type
bucket and (b) the specific category being handled. Context that is
*inherent to an endpoint* (what keywords exist and what they mean,
what vector spaces are available and what each represents, what
Metadata columns can be predicated on, etc.) doesn't vary from
category to category — what varies is *which* endpoints are in
scope for a given category. So the static unit of reuse is the
endpoint-context chunk, dynamically appended to a system prompt
based on the category's endpoint set.

### The four plug-in pieces

Each handler's LLM prompt is composed of four pluggable pieces:

| Piece | Keyed by | Purpose |
|---|---|---|
| **Endpoint context** | Category (→ endpoint set) | Static per-endpoint reference: vocabulary, schema, parameter shapes, capabilities. Appended once per relevant endpoint. |
| **Output schema** | Bucket + category | What the LLM must produce. Bucket determines *shape* (one-of vs. best-fit vs. fan-out-set); category determines fields (which endpoints the schema includes, what hard-constraint sub-fields may appear). |
| **Core objective** | Bucket | The decision framing — what question the LLM is being asked to answer. Uniform across all categories in the same bucket. |
| **Additional objective notes** | Category | Category-specific nuances layered on top of the core objective: tiering bias orderings, when to emit `exclusion_ids`, interpretation/confidence guidance, etc. |

### Core objective per bucket

The LLM's decision-framing is bucket-level. Category-specific
notes adjust tone and nuances but don't change the core shape.

**Single endpoint.** "Given the user's query, decide whether this
endpoint *should* fire. Dispatch may have been wrong — answering
'no query needed' is a valid outcome and is preferable to inventing
a query that doesn't match the user's intent. If it should fire,
fill in the endpoint parameters so the query best satisfies the
attribute under consideration."

**Mutually exclusive.** "Given the user's query and the candidate
endpoints, decide which one best fits — *or* that none of them
should fire. Dispatch may have been wrong and 'none' is a valid
outcome. Once an endpoint is chosen, fill in its parameters so the
query best satisfies the attribute under consideration."

**Tiered.** "Given the user's query and the candidate endpoints,
decide which single endpoint best fits — *or* that none should
fire. You are provided with a preference bias favoring certain
endpoints when they can adequately satisfy the problem. Treat the
bias as a tiebreaker when multiple endpoints fit roughly equally;
do not treat it as an uphill battle for lower-preference endpoints.
A clearly-better lower-preference endpoint wins (e.g. 'lovable
rogue' has no canonical keyword tag, so Semantic beats Keyword
decisively despite the keyword-first bias). Once an endpoint is
chosen, fill in its parameters."

**Combo.** "Given the user's query and the full set of candidate
endpoints for this category, determine which *combination* of them
best covers the attribute under consideration. Any subset is valid
including the empty set (none fire). For each chosen endpoint, fill
in concrete parameters."

### Why tiered is mutually-exclusive-with-bias, not a walk-the-list

Tiered looks like "try tier 1, fall back to tier 2 if needed" but
in practice a single LLM call can evaluate all candidates jointly
and pick the best fit given a preference bias. That's operationally
identical to mutually-exclusive plus a bias prior on the scoring.
It's also strictly better than a sequential walk: the LLM sees all
options at once and can make a holistic judgment (is tier 1 a
*genuine* fit, or is tier 2 clearly a better match?) rather than
greedily taking the first "good enough" tier.

The bias exists to prevent the failure mode of "Semantic can
plausibly answer anything, so the LLM picks it even when a
canonical tag would be a cleaner hit." The bias puts a finger on
the scale toward the authoritative channel without locking out
the fallback for cases where the fallback is genuinely better.

### Why this modularity matters

Writing 31 bespoke handler prompts would leak category-specific
detail into endpoint-specific context and vice versa. The split
keeps each piece single-sourced:

- Endpoint vocabulary updates once per endpoint, not once per
  category that references it.
- Bucket core-objective changes (e.g. tightening the "none is
  valid" guidance) propagate to every category in the bucket.
- Category-specific quirks live in the "additional objective
  notes" chunk and nowhere else.

---

## Handler return contract

Every category handler returns the same four buckets:

| Bucket | Content | Merge op | Notes |
|---|---|---|---|
| `inclusion_candidates` | IDs + score | UNION across handlers, additive score | Drives candidate generation |
| `downrank_candidates` | IDs + score | Add (negative) score in final rerank | Semantic dealbreakers, soft "avoid" |
| `exclusion_ids` | IDs only, no score | Set-subtract from final pool | Hard dealbreakers (above PG-13, pre-1980, etc.) |
| `preference_specs` | Deferred query specs | Run after candidate consolidation | See preference-handling section |

Not every handler emits all four. Cat 10 (structured metadata) emits
`exclusion_ids` when the query implies a hard ceiling ("PG-13 max",
"under 2 hours"). Cat 17/18 emit `exclusion_ids` for maturity
ceilings alongside `inclusion_candidates` for family-friendly
scoring. Most purely-semantic categories emit only
`inclusion_candidates` and `preference_specs`.

---

## Hard vs soft exclusion: why the split

The distinction is not about where the exclusion comes from — it's
about the merge op.

- **Hard exclusion** (`exclusion_ids`) — absolute remove. "Must be at
  least PG", "released after 2000", "under 3 hours". Applied as set
  subtraction on the final candidate pool.
- **Soft exclusion** (`downrank_candidates`) — downrank with a
  negative score. "Avoid gore" as a semantic ask that returns IDs
  with negative contribution to the final score.

**Correctness point: hard dealbreakers must go through
`exclusion_ids`, not `inclusion_candidates`, regardless of which set
is smaller.** The merge math forces the direction:

- Maturity-as-inclusion: Cat 17 emits {G, PG}; Cat 11 emits horror.
  Union = G/PG ∪ horror. R-rated horror stays in the pool (via Cat
  11) and just misses Cat 17's additive bonus. Violates "PG max".
- Maturity-as-exclusion: Cat 17 emits {R, NC-17} as `exclusion_ids`;
  Cat 11 emits horror. Union-then-subtract = horror ∩ not-R = family
  horror. Correct.

The "pick the narrower direction" framing still applies *within the
exclusion side*: "at least PG" → exclude G only (small set); "PG
max" → exclude R/NC-17 (larger but workable at 100K). The LLM picks
whichever framing yields the smaller ID set, but the answer is
always an `exclusion_ids` emission, never an inclusion flip.

### Post-hoc subtraction quality note

Applying `exclusion_ids` after endpoints return their top-K means
what's left is "the family-safe fraction of the top-500 horror
matches," not "the top-500 family-safe horror matches." At 100K this
is tolerable — bump endpoint fetch size a bit when large exclusions
are active. Optionally, pass `exclusion_ids` into each endpoint's
query at retrieval time so filtering happens during search. Either
works; the latter is slightly better signal for no latency cost.

---

## Preference handling: deferred, per-search, additive

Preferences never run at handler time. Handlers emit
`preference_specs` (target vector space + prose fragment + role +
polarity). The orchestrator consolidates all preferences across
handlers, establishes the final candidate pool, then runs
preferences against it.

### Separate searches, not concatenated prose

For a stack like "horror movies that are dark yet funny and playful",
three separate vector searches (one per axis) with additive score
merge is the right model:

- Separate + additive → **conjunctive** ("strong on all axes wins").
- Concatenated prose → **centroid target** ("strong on one axis can
  match the centroid"). Worse for preferences, which are
  conjunctive by nature.

### Zero-candidate fallback

If inclusion produced no candidates and we need preferences to
generate them, don't jump straight to a mega-query. First try:
**run each preference search against the full corpus, take top-K
from each, union into a working candidate pool (~K·N items), then
additively re-score.** Qdrant is cheap enough to absorb the extra
searches at 100K scale.

Mega-query concatenation is a tier-below-that fallback if even the
per-preference union produces nothing coherent — not the default.

### When would prose concatenation help?

When a single LLM pass with full query context would write better
joint prose than N disconnected per-category prose chunks (e.g.
"gut-punch ending in a dark slow-burn horror" vs three separate
axis-strings). That's a real quality win, but it fights the
conjunctive-merge property above. For preferences, conjunctive
merge wins; keep them separate. If future evaluation shows per-axis
prose quality is the bottleneck, the right fix is to give the
per-handler LLM more query context, not to concatenate prose at
search time.

---

## Within-handler query interaction

No handler needs its inclusion queries to interact at runtime beyond
what the shape's merge rule defines.

- Types 1–3 (single endpoint / mutually exclusive / tiered): only
  one query fires per request — no runtime interaction possible.
- Type 4 (combo): scores sum across endpoints — queries are
  independent by construction.

Endpoint-internal multi-phase logic (Franchise's two-phase token
resolution, Keyword's DF-ceiling) lives inside the endpoint, not
the handler shape.

---

## Orchestrator flow (summary)

```
Step 2 pre-pass
    ↓ (fragments + coverage evidence)
Category dispatch (categorizer LLM)
    ↓ (per-category fragment assignments)
Run all category handlers in parallel
    ↓ (each returns 4 buckets)
Consolidate:
  - Union all inclusion_candidates, additive score
  - Set-subtract exclusion_ids from the pool
  - Keep downrank_candidates as score contributions
    ↓ (final candidate set established)
Run consolidated preference_specs against the candidate set
  - One vector search per preference (separate, not concatenated)
  - Additive merge of preference scores
    ↓
Final rerank:
  inclusion_score + downrank_score + preference_score
    ↓
Top-N → fetch display metadata → return
```

---

## Open items

- **Handler-level exclusion vs endpoint-level exclusion for
  `exclusion_ids`.** At 100K, post-hoc set subtraction is probably
  fine but produces lower-quality candidate pools when a large
  fraction of endpoint top-K gets kicked out. Revisit with real
  numbers once handlers are implemented.
- **Do we need `hard_inclusion_ids` (INTERSECT semantics)?** Not
  today. No category in the taxonomy requires "only these IDs, no
  others" positive-direction semantics. Every hard constraint we
  have is naturally expressible as an exclusion. Revisit if a new
  category breaks this.
- **Interpretation fields in handler output schemas.** Cat 28 and
  Cat 31 need intent-rationale and confidence fields on their LLM
  output so the orchestrator can lower confidence in the returned
  results. Design those schemas when those handlers are built.
