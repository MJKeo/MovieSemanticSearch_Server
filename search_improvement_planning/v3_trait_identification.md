# v3 Trait Identification Guide

How a query is decomposed into the discrete units ("traits") that
get classified, routed, and scored. This doc covers the upstream
decomposition step that feeds the rescoring layer described in
`v3_reranking_guide.md`.

Read alongside:
- `v3_reranking_guide.md` — the rescoring layer that consumes the
  classified traits this step emits.
- `carving_qualifying_boundaries.md` — the four-cell role taxonomy
  (positive/negative carver, positive/negative qualifier).
- `v3_proposed_changes.md` — the broader v3 change list. This doc
  expands #7 (trait grouping), #8 (inter-atom relationships — now
  largely obviated by programmatic merging), and #13 (interpretation
  deferral) into a coherent decomposition story.

---

## What a trait is

A **trait** is the smallest span of the query that carries a
coherent classification across the four per-trait attributes,
including any modifier tokens that attach to it.

The four attributes (defined below) are the atomicity test: if any
attribute would disagree across a span, that span must be split into
multiple traits. If all four agree and splitting would lose intended
meaning, the span is one trait.

"Trait" replaces the older "atom" vocabulary. Atom historically
suggested "indivisible by surface form," which led to over-splitting
on commas and conjunctions. Trait emphasizes the classification-
coherence test, which is the actual atomicity criterion.

---

## Per-trait data

Each trait carries exactly four classified attributes plus its
content. The grouping LLM's entire job is to emit this payload per
trait — nothing else.

| Attribute | Values | Purpose |
|---|---|---|
| **Role** | carver / qualifier | Decides which side of the rescore the trait contributes to. Carvers gate or sit on the rarity-weighted carver score; qualifiers steer the salience-weighted qualifier score. Different scoring mechanisms entirely. |
| **Polarity** | positive / negative | Whether the user wants this characteristic or wants to avoid it. Polarity is about *intent*, not surface grammar — "movies that aren't boring" is positive intent. Drives polarity-flip handling (negative carvers remove candidates; negative qualifiers downrank; tonal negations get rewritten to positive opposites in code). |
| **Category routing** | entity / metadata / keyword / semantic / parametric-knowledge / etc. | High-level handler the trait routes to. The grouping LLM picks the category; the handler-stage LLM does the per-handler interpretation. This decouples taxonomy from mechanism — the grouping LLM never needs DB-shaped context. |
| **Salience** (qualifier only) | central / supporting | LLM-derived weight on the qualifier side. Central qualifiers are headline wants; supporting ones round out the picture. Maps to fixed weight multipliers in scoring. Carvers do not get salience — they use corpus-derived rarity instead, computed downstream. |

The trait's textual content (the phrase itself, including absorbed
modifier tokens) rides along as the payload the downstream handler
will interpret. The grouping LLM does not resolve interpretation; it
only classifies.

---

## Modifier tokens

Modifier tokens **set or refine an attribute on a trait without
carrying their own classification**. They get absorbed into the
trait they modify rather than becoming traits of their own — they
fail the atomicity test because they have no standalone (role,
polarity, category, salience) classification.

Categories of modifiers:

- **Polarity setters** — "not," "without," "no," "avoid," "aren't,"
  "but not." Flip the polarity attribute on the trait they attach
  to. "Not depressing" → one tonal trait, polarity=negative.
- **Salience hints** — "ideally," "mainly," "especially," "must,"
  "need," "primarily," "particularly." Inform the central /
  supporting decision for qualifiers.
- **Role/category hints** — "starring," "directed by," "about,"
  "like," "from the," "in the style of," "set in." Help the grouping
  LLM pick the right category and (sometimes) the right role.
- **Range/intensity modifiers** — "very," "extremely," "around,"
  "roughly," "in the," "kind of," "a bit." Inform handler-stage
  interpretation (range widths, intensity targets) but don't change
  the trait's classification.

These tokens are extracted alongside the trait's content but live as
inline context on the trait, not as separate traits.

---

## Atomicity examples

Worked examples showing the splitting test in action.

- **"Horror movies starring Anthony Hopkins from the 90s"** → three
  traits. They disagree on category (keyword/genre vs entity vs
  metadata-date). Split-and-AND preserves meaning, so atomicity is
  forced. All three are positive carvers.
- **"Starring Anthony Hopkins"** → one trait. "Starring" is a
  role-hint modifier on the entity core (tells the entity handler
  this is an actor, not a director). No standalone classification
  for "starring."
- **"Not depressing"** → one trait. "Not" is a polarity-flip
  modifier on the tonal core. Polarity=negative; category=semantic;
  role=qualifier.
- **"About grief, not depressing"** → two traits. "About grief" is a
  thematic carver (binary "is this about grief?"); "not depressing"
  is a tonal qualifier with polarity=negative. Different roles, so
  must split. The fact that one steers the ranking among the other's
  survivors is downstream scoring behavior, not a grouping concern.
- **"Comedians doing serious roles"** → one trait. Routes to
  parametric-knowledge as a single composite reference class.
  Splitting into "comedians" + "serious roles" would lose the
  conjunction's meaning ("actors known for comedy taking dramatic
  parts" is not the AND of "comedians" and "serious roles"). The
  handler-stage LLM unpacks it later — interpretation is deferred.
- **"Mindless adrenaline-fueled action"** → three traits at
  grouping: action (positive carver, keyword/genre), mindless
  (positive qualifier, semantic), adrenaline-fueled (positive
  qualifier, semantic). The two qualifiers will programmatically
  merge later via vector-space overlap (see "Out of scope" below).
  The grouping LLM does not anticipate that merge.
- **"Modern classics"** → two traits. "Modern" is a positive carver
  (metadata-date, with the per-user default for "modern" supplying
  the range). "Classics" is a positive carver routed to reception
  (semantic — "widely regarded as a classic") or
  parametric-knowledge — the handler-stage LLM unpacks the exact
  mechanism. Splitting preserves meaning here: the user wants films
  that are modern AND classics, two independent constraints AND'd
  rather than a single textured concept. Do NOT pre-strip "classics"
  as a global popularity-prior amplification — the user is asking
  for movies that *are* classics, not for popular movies in general.
- **"Like Interstellar but not sci-fi"** → two traits. "Like
  Interstellar" is a positive carver routed to parametric-knowledge
  / entity-reference (single trait, not split — "like X" is one
  reference class). "Sci-fi" with polarity=negative is a negative
  carver routed to keyword/genre. The "but" is a relationship
  conjunction that doesn't itself become a trait.
- **"Iconic twist endings"** → one trait. "Iconic" here scopes to
  "twist endings" — the user wants movies *famous for their twist*
  (Sixth Sense, Fight Club, Usual Suspects), not generally popular
  movies that happen to have a twist. Splitting into "iconic" +
  "twist endings" would lose the scoping and produce the wrong
  result set (a famous movie with a forgettable twist would beat a
  less-famous movie defined by its twist). One positive carver;
  category routes to reception (semantic — "praised for the twist,"
  "famous for the reveal") or parametric-knowledge ("the canonical
  movies known for their endings"). The handler-stage LLM unpacks
  which exact mechanism. Critical illustration that mode-word
  handling is NOT just lexicon-strip-and-amplify — see the mode-word
  caveat in the per-query attributes section.

---

## What this step does NOT do

The grouping LLM's contract is narrow on purpose. The following are
explicitly *out of scope* for this step:

- **Interpretation of vague or composite traits.** "Like
  Interstellar," "comedians doing serious roles," "modern classics,"
  "iconic franchises" all stay as traits-with-context. The
  category routing assigns them to a handler, and the handler-stage
  LLM unpacks them with full per-handler context. This decouples
  taxonomy from interpretation and makes the grouping prompt small
  and stable — adding a new endpoint or expanding a reference-class
  table doesn't require re-prompting the grouping layer.
- **Mechanism awareness.** The grouping LLM does not need to know
  whether a trait routes to a structured filter, a vector space, a
  posting list, or a parametric-knowledge expansion. The category
  answer determines mechanism; downstream handlers carry the
  endpoint-shaped context.
- **Trait fusion.** Fusion (combining same-space qualifier bodies
  into one composite semantic query, per v3 #8) is handled
  programmatically downstream. Once the qualifying semantic traits
  are identified and routed, code merges any whose handler-emitted
  vector spaces overlap. Fusion is genuinely semantic-only —
  structured handlers don't have a meaningful fusion operation
  (atomic constraints, AND composition is natural).
- **Negation rewriting.** Tonal negations ("not depressing" →
  "hopeful, cathartic") are rewritten in code from a curated
  polarity-flip table, triggered by polarity=negative + category=
  tonal-semantic. The grouping LLM only emits the polarity flag; the
  rewrite is downstream.
- **Per-query state assignments.** The carver/qualifier balance
  state (one of 5) and the implicit-prior strength state are
  per-query, not per-trait. They live in a separate emission from
  the same step-2 process, not on individual traits.
- **Inter-trait relationship metadata.** v3 #8 originally proposed
  emitting qualifies/modifies/contradicts tags between traits to
  drive fusion. Given that fusion is now programmatic via
  space-overlap detection, and tonal-negation rewriting is triggered
  by polarity+category alone, explicit relationship tags are not
  needed in v1. They were a v3 #8 artifact obviated by the
  programmatic merging path.

---

## Why two-stage routing

The grouping LLM emits (role, polarity, category, salience) per
trait. The handler-stage LLM (one per category) translates the trait
into executable queries with full per-handler context.

This separation buys three things:

1. **Smaller, more stable grouping prompt.** The grouping LLM never
   sees endpoint-shaped context. Adding endpoints, expanding
   keyword tables, or reshaping vector spaces doesn't require
   re-prompting the grouping layer.
2. **Mechanism split for free.** Each handler invocation becomes its
   own group. The grouping LLM doesn't have to enforce "structured
   vs unstructured can't share a group" — the routing layer does.
3. **DB context lives where it can be used.** The handler-stage LLM
   only needs the slice of DB shape relevant to its own endpoint,
   keeping every prompt focused.

---

## Per-query attributes (separate from traits)

Step 2 also emits whole-query attributes that aren't per-trait.
These are listed for completeness but live elsewhere in the payload:

- **Carver / qualifier balance state.** One of 5 (dealbreakers
  dominant → preferences dominant). Drives the carver-vs-qualifier
  weight schedule in rescoring. Per `v3_reranking_guide.md`.
- **Implicit-prior strength state.** Low / medium / high. Drives
  the multiplicative implicit-prior layer's α coefficient. Per
  `v3_reranking_guide.md`.
- **Mode-word detections.** Curated-lexicon hits ("best," "iconic,"
  "underrated," "modern," "comfort watch," etc.) can drive
  per-dimension implicit-prior suppression / amplification rules
  (v3 #4). **Critical caveat: mode words have scope.** Sometimes the
  word applies globally to the whole query ("show me the best 80s
  action movies" — "best" applies to the entire result set; route
  to global popularity/quality prior amplification). Sometimes it
  scopes to a specific noun phrase inside the query ("iconic twist
  endings" — "iconic" modifies the twist itself, not the movie's
  overall popularity). Stripping the word in the second case
  actively destroys meaning and produces a different query. So
  mode-word handling is **not** a blind lexicon-strip; it requires a
  scope-resolution judgment first (global mode vs local modifier on
  a specific trait). When the scope is local, the word stays inside
  the trait it modifies and rides along into handler-stage
  interpretation. Only globally-scoped hits leave the trait stream
  and feed implicit-prior modulation. v3 #4 itself is still a
  proposal — see open TODOs.

---

## Open TODOs and reminders

Items captured during the design conversation that remain open or
deferred:

- **Define the category list.** The exact set of category-routing
  values the grouping LLM picks from. Working set: entity,
  metadata, keyword/genre, semantic, parametric-knowledge,
  award. Confirm against the existing stage-3 handler set in
  `search_v2/stage_3/category_handlers/`.
- **Define the modifier-token registry.** The curated lists of
  polarity setters, salience hints, role/category hints, and
  range/intensity modifiers. Some of these are already implicit in
  the current step-2 prompt; consolidate into one explicit registry
  the grouping prompt can reference.
- **Define the mode-word lexicon (v3 #4) AND its scope-resolution
  pass.** Curated mapping of "best," "iconic," "underrated,"
  "modern," etc. to (a) the implicit-prior dimension(s) they
  suppress or amplify when applied globally and (b) any pre-canned
  reception/quality sub-queries they trigger. **Mode-word handling
  is two stages, not one:** first detect the lexicon hit, then
  resolve scope (global query mode vs local modifier on a specific
  trait). Only globally-scoped hits leave the trait stream to feed
  implicit-prior modulation; locally-scoped hits stay inside the
  trait they modify ("iconic" in "iconic twist endings" rides along
  on the twist-endings trait and is interpreted by the handler).
  Scope resolution is a non-trivial LLM judgment — a curated lexicon
  alone cannot do it. Worked counterexample: "iconic twist endings"
  vs "iconic 80s action movies" — same word, different scope,
  different correct handling.
- **Programmatic same-space qualifier merge.** Post-grouping code
  that merges semantic-qualifier traits whose handler-emitted vector
  spaces overlap. Mechanism: union the bodies per shared space,
  track the merged trait under a single group identity for scoring.
- **Tonal-negation rewrite table.** Curated polarity-flip vocabulary
  (depressing → hopeful, cathartic; boring → engaging; predictable
  → surprising). Triggered by polarity=negative + category=
  tonal-semantic.
- **Salience guidance in the prompt.** Concrete language for when a
  qualifier is central vs supporting. Draft heuristics: headline
  wants ("really want," "must be," "above all"), structural
  centrality in the sentence (main clause vs trailing modifier),
  and explicit emphasis tokens ("especially," "primarily").
  Empirical tuning of the central:supporting ratio is a separate
  TODO in the rescoring doc.
- **Interpretation deferral discipline.** The grouping prompt must
  resist the temptation to resolve "like Interstellar" or
  "comedians doing serious roles" inline. Worth explicit instruction
  + examples in the prompt to prevent drift.
- **Whether the trait's textual content needs structured fields.**
  Open: do we just emit the raw phrase as a string for the handler,
  or do we pre-segment it into `core` + `modifiers`? Lean toward
  raw string for v1; the handler-stage LLM has the phrase and the
  classification and can re-parse if it needs to.

### Closed decisions (do not relitigate)

- Inter-trait relationship metadata is not in v1. Fusion is
  programmatic via space overlap.
- Fusion is semantic-only. Structured handlers don't fuse.
- Interpretation is deferred to the handler-stage LLM.
- Modifier tokens absorb into traits, not become traits.
- Per-query state assignments (balance, prior strength) live
  outside the per-trait payload.
- Trait-extraction confidence is not part of the per-trait payload
  — treat extraction as truth (per `v3_reranking_guide.md`).
- The unit name is "trait." "Atom" is retired vocabulary.
