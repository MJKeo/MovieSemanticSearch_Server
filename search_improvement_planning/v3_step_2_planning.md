# v3 Step 2 Planning

How step 2 (query pre-pass / trait identification) is being redesigned
for v3. Captures the responsibilities, decisions, and final output
schema, along with the reasoning behind each choice.

Read alongside:
- `query_categories.md` — the 1-43 category taxonomy step 2 routes to.
- `v3_trait_identification.md` — the broader decomposition story
  (modifier absorption, atomicity, interpretation deferral).
- `v3_reranking_guide.md` — the rescoring layer that consumes step 2's
  per-trait classifications and the followup LLM's per-query states.
- `carving_qualifying_boundaries.md` — the four-cell role taxonomy.

---

## Step 2's responsibilities

Step 2 takes the user's natural-language query and emits everything
step 3 (per-category handlers) needs to write actual queries against
the corpus. The responsibilities are:

1. **Identify and group high-level traits**, including absorbing the
   modifiers (polarity setters, salience hints, role markers) that
   attach to each trait.
2. **Mark each trait as carver vs qualifier.**
3. **Mark each trait as positive vs negative polarity.**
4. **Assign salience (central / supporting) to each qualifier.**
5. **Pick the best-fit category (1-43) for each trait.**
6. **Provide enough context that step 3 can interpret the trait
   without re-deriving step 2's reasoning.**

Per-query state assignments — carver/qualifier balance state and
implicit-prior strength state — are NOT done in this step. They are
emitted by a separate **followup LLM that runs in parallel with step
3** (see decisions below).

---

## Key decisions (with reasoning)

### Per-query state assignments → separate followup LLM, parallel to step 3

The carver/qualifier balance state and the implicit-prior strength
state are whole-query gestalt judgments, not per-trait classifications.
They benefit from seeing the trait list as input. Three reasons to
split them out:

- **Different reasoning shape.** Trait identification is mechanical
  (segmentation + classification). Per-query states are holistic
  (qualifier richness, headline-vs-supporting balance, mode-word
  presence).
- **Step 2's prompt is already heavy.** 43-category taxonomy +
  atomicity test + modifier registry + worked examples will dominate
  the prompt. Adding two whole-query gestalt judgments overloads it.
- **Per-query states don't gate handler dispatch.** They're consumed
  at rescoring time. So they can run in parallel with step 3 (handler
  dispatch). They converge with step 3's output at the merging stage.

The followup LLM takes (original query, trait list) → emits
(balance_state, implicit_prior_strength).

### Polarity is surface-grammar, not intent

If the query contains a polarity-setter ("not", "without", "no",
"avoid") on a trait, polarity = negative. No interpretation pass.
"Not too boring" → trait="boring", polarity=negative.

Resolves a tension in v3_trait_identification.md L47 (which suggested
intent-based polarity) vs L99 (which described surface-grammar
polarity). The surface-grammar rule is mechanical, supports interpretation
deferral, and leaves any rewrite (e.g. tonal-negation rewriting like
"not depressing" → "hopeful, cathartic") to downstream code.

### One trait : one category mapping (compound rule retired)

Earlier versions of the taxonomy let a single trait fan out to
multiple categories ("classic" → Cat 12 + Cat 38). v3 step 2 maps
each trait to exactly one category. Three implications:

- **The compound split rule is dead.** Replaced by an atomicity test
  on the span: split into multiple traits when each piece could
  classify independently AND splitting preserves meaning.
- **Bucket-B compounds (single concept → one category whose handler
  fans out) are fine.** Cat 32 (emotional/experiential) routes to
  VWX+CTX+RCP+KW from one trait — the handler does the fan-out, not
  step 2.
- **Audit confirmed no new categories are needed.** "Classic," "modern
  classic," "blockbuster," "indie," "popcorn movie," "hidden gem" all
  fit existing categories or split cleanly. Cat 38 is the natural home
  for compound reception-prose descriptors ("classic," "era-defining,"
  "stacked cast"); Cat 17 (financial scale, merged budget+box-office)
  owns "blockbuster"/"indie"; Cat 32 absorbs experiential compounds.

### Modifier absorption preserves meaning (atomicity test)

A token is absorbed as a modifier (not promoted to its own trait) if
it can't classify independently OR if splitting it off would damage
searchability. Examples:

- "lone female protagonist" → ONE trait (Cat 9 FEMALE_PROTAGONIST).
  "Lone" alone isn't a meaningful query span — it's a modifier on
  "female protagonist."
- "starring Tom Hanks" → ONE trait. "Starring" is a role-marker
  modifier, not its own trait.
- "not too dark or sad" → "not" applies to BOTH "dark" and "sad" via
  distribution, even though "not" is not adjacent to "sad" in the
  surface form.

### Interpretation deferral

Step 2 does NOT resolve "like Inception," "comedians doing serious
roles," "modern classics," etc. inline. Those stay as traits-with-
context. The category routing (e.g. Cat 41 like-X, Cat 43 catch-all)
sends them to handlers that have full per-handler context to unpack.

---

## Schema design reasoning

### Why a span_analysis pre-pass before the trait list

Small LLMs fail in characteristic ways on this kind of task:

1. **Skipping segmentation.** Going straight to classification while
   reading produces conflated decisions.
2. **Conflating attributes.** Deciding role and category in the same
   gestalt judgment — once one is committed it constrains the other,
   even when the evidence doesn't support both.
3. **Hallucinating modifiers as traits.** Promoting "ideally,"
   "starring," or "lone" to standalone traits.
4. **Picking the wrong category at boundaries.** 43 categories, with
   many semantically adjacent (Cat 3 vs Cat 5, Cat 7 vs Cat 8,
   Cat 12 vs Cat 42).

The pre-pass forces explicit segmentation reasoning *before*
classification. By the time the LLM commits to a final trait list,
each span has been examined for boundaries, modifiers, contextual
dependency, and split necessity.

### Why one trait list, not separate carver/qualifier lists

A single `traits` list with `role` as a per-trait field is preferred
over `carvers: [...]` and `qualifiers: [...]`. Three reasons:

- **No premature commitment.** Two-list shape forces role to be
  decided before placement, which inverts the natural reasoning order
  (category shape is a strong prior on role; role should be derived
  *after* category, not before).
- **Localized decisions.** Small LLMs are stronger at atomic per-trait
  classification than at global "which bucket" framing.
- **Handler dispatch doesn't care.** Step 3 dispatches on category,
  not role. Two lists is denormalization that buys nothing.

### Why structured `category_candidates` over freeform reasoning

Project learning: structured-field reasoning beats freeform strings
for small LLMs. `category_candidates` is a list of `{candidate, fits,
doesnt_fit}` objects so the LLM is forced to articulate the boundary
comparison between genuine contenders rather than emitting a vague
"this is a Cat X because it talks about Y" rationalization.

Rules for the list:
- Cap length (~2-4 entries). Don't enumerate every plausible cat —
  compare the genuine boundary contenders.
- Even single-candidate cases emit one entry. Keeps the schema
  uniform; small LLMs benefit from the consistency.

### Why no `role_cue` / `role_evidence` field

Reasoning fields that sit *after* the decision they justify drift
into post-hoc rationalization rather than pre-decision evidence.
Reasoning fields that sit *before* the decision must be evidence-
shaped, not decision-shaped.

The role decision is informed by:
- `query_context` and `contextual_dependency` from `span_analysis`
  (does this trait stand alone, or is it relativized by another?)
- `purpose_in_query` on the trait itself (carving vs qualifying vs
  reranking)
- Category prior (some categories are inherently carver-shaped,
  others qualifier-shaped)

All of those sit upstream of `role` in the schema. No additional
per-trait justification field is needed — adding one would either
restate upstream evidence or rationalize a committed decision.

### Field order encodes reasoning order

Within each trait:
1. `query_phrase` — what we're classifying
2. `modifiers` — what's been absorbed
3. `purpose_in_query` — what role this trait plays in the query (informs
   role-shape and constrains plausible categories)
4. `category_candidates` — boundary comparison reasoning
5. `best_fit_category` — committed pick, informed by candidates
6. `role` — informed by purpose_in_query
7. `polarity` — mechanical from modifiers
8. `salience` — mechanical from modifiers + position; null if carver

Each field appears once, in the location where it does the most work,
and ahead of any decision it informs.

---

## Final output schema

```python
{
  span_analysis: [
    {
      text: <span as it appears in query>,
      modifiers: [
        # Words inside the span OR elsewhere in the query that apply to
        # it. Includes non-adjacent applicability (e.g. "not" applies to
        # "sad" in "not too dark or sad" via distribution).
        <string>, ...
      ],
      query_context: <how other (non-modifier) parts of the query shape
                      this span's meaning, and/or how this span shapes
                      others. Empty when the span stands alone.>,
      decomposition: {
        possible_splits: [
          {
            isolated_span: <text that would be split out of the
                           parent span>,
            reasoning: <whether this isolated_span can stand as its
                        own query — would a user submit just this
                        piece and have it make sense, does it have
                        a category home, would pulling it out leave
                        a meaningful residual or an orphan>,
            should_split_out: <bool — final commit derived from
                              the reasoning above>
          },
          ...  # empty list when the span has no plausible split
               # points (e.g. "Tom Hanks")
        ]
      }
    },
    ...
  ],

  traits: [
    {
      query_phrase: <the trait text, modifiers excluded>,
      modifiers: [
        {
          text: <modifier word>,
          impact: <how it changes the trait's meaning in the query>
        },
        ...
      ],
      purpose_in_query: <reasoning about what role this trait plays in
                        the context of the query — carving the eligible
                        pool, qualifying within it, or reranking.
                        Grounded in query_context from span_analysis.>,
      category_candidates: [
        {
          candidate: <category enum, 1-43>,
          fits: <why this category could be the right home>,
          doesnt_fit: <why it might not be — boundary concern, missing
                      surface form, etc. Empty if no real concern>
        },
        ...  # cap ~2-4; even unambiguous traits emit one entry
      ],
      best_fit_category: <enum, 1-43>,
      role: <carver | qualifier>,
      polarity: <positive | negative>,
      salience: <central | supporting | null if carver>
    },
    ...
  ]
}
```

### Flow

`span_analysis` produces N entries (one per candidate span).
`traits` produces M entries, where M ≥ N when spans split via
`decomposition`. Each trait's `modifiers` and `purpose_in_query`
inherit from the corresponding span_analysis entry, narrowed to the
trait's subset after any decomposition splits.

---

## What step 2 does NOT emit (explicitly out of scope)

Captured here to prevent scope creep:

- **Per-query carver/qualifier balance state.** Followup LLM,
  parallel to step 3.
- **Per-query implicit-prior strength state.** Followup LLM,
  parallel to step 3.
- **Mode-word detection / scope resolution.** Step 3 / handler
  responsibility.
- **Tonal-negation rewriting.** Downstream code, triggered by
  polarity=negative + category=tonal-semantic.
- **Same-space qualifier fusion.** Programmatic, post-grouping.
- **Inter-trait relationship metadata.** Not in v1.
- **Trait-extraction confidence.** Treated as truth.
- **Interpretation of vague compounds.** ("Like Inception," "comedians
  doing drama," etc. stay as traits-with-context for handler-stage.)

---

## Prompt context: trait identification fundamentals

The five concepts below are what the system prompt has to convey
clearly before the LLM sees any worked examples. Each is written
in guiding-principle form rather than rule form — small LLMs are
better at reasoning from a clean principle plus calibration
examples than at executing a flowchart.

---

### 1. Trait unit identification (atomicity)

**Definition.** A trait is the smallest span of the query that
carries one coherent classification across role, category,
polarity, and salience — with any modifier tokens absorbed into
it. Splitting is meaning-preserving: split when each piece
independently survives as a meaningful query trait AND splitting
doesn't damage what the user was actually asking for.

**How to think.**
1. Look for explicit conjunctions ("and," "or") between distinct
   concepts — strong split signals.
2. For each potential split, ask: can each piece function as its
   own trait, classifying to its own category, with the same
   meaning it had inside the parent span?
3. If yes → split. If pulling a piece out turns the residual
   into something the user didn't ask for → don't split.

The deciding factor between **"modern classics"** (split) and
**"iconic twist endings"** (don't split) is what survives the
split:
- "Modern" + "classics" — each is a real trait the user is
  asking about. Two stacked constraints.
- "Iconic" + "twist endings" — pulling "iconic" out turns it
  into a vague "movie is iconic" trait while "twist endings"
  loses the scoping. The user wanted *iconic-because-of-the-
  twist*, not *iconic AND has a twist*. Compound concept stays
  whole.

**Boundaries.**
- **Bucket-B compounds.** Some single concepts route to one
  category whose handler fans out internally (the
  emotional/experiential category covers tone + pacing +
  experience goals + post-viewing resonance from one trait).
  Don't pre-split — the handler has more context.
- **Named entities never split mid-name.** "Stephen King" is
  one trait. "Coen Brothers" (a duo treated as one entity) is
  one trait. Multi-name compounds joined by an explicit "and"
  / "or" — "Hanks and Streep" — DO split per name.
- **"Based on" phrases always split** between the named
  referent and the medium ("Stephen King" + "novels" —
  separate traits routing to separate categories).

**Common pitfalls.**
- Splitting on commas / conjunctions reflexively. Read what
  the words do, not just the punctuation.
- Missing the "scoped adjective creates compound" pattern.
  "Iconic twist endings," "lovable rogue protagonist," "morally
  ambiguous lead" — front adjective scopes the noun, can't
  survive standalone without changing meaning. Keep whole.
- Pre-splitting Bucket-B compounds. "Slow-burn dread" routes
  to one emotional/experiential trait; splitting fragments
  what the handler is built to unpack as one.

**Examples.**

Split:
- "Modern classics" → "modern" + "classics"
- "Funny horror movies" → "funny" + "horror"
- "Movies starring Hanks and Streep" → two PERSON_CREDIT traits
- "Set in the 90s about grief" → "set in the 90s" + "about
  grief"

Don't split:
- "Iconic twist endings" → one trait (scoping is meaning-
  bearing)
- "Lone female protagonist" → one trait ("lone" doesn't
  survive standalone)
- "Darkly funny" → one trait ("darkly" reshapes "funny")
- "Comedians doing serious roles" → one trait
- "Stephen King" → one trait

---

### 2. Modifier vs trait

**Definition.** A modifier is a token that adjusts polarity or
strength of a trait without changing what the trait itself is
about. Modifiers absorb into the trait they attach to — they
don't become traits of their own. A token that fundamentally
reshapes meaning (forming a new compound concept) is NOT a
modifier; it's part of the trait, and the whole stays one trait.

**How to think.** For each candidate token next to a trait,
ask: does this word change *what the trait is about*, or just
*how strongly the user wants it / whether they want it at all*?

- "Not funny" — "not" changes intent (filter out funny things)
  but "funny" still means "funny." Modifier.
- "A bit funny" — "a bit" changes strength but "funny" still
  means "funny." Modifier.
- "Darkly funny" — "darkly" reshapes "funny" into a tonal
  compound. Not a modifier; the whole thing is one trait.
- "Starring Tom Hanks" — "starring" tells the entity handler
  this is an actor. Modifier.

**Boundaries.** Modifiers come in roughly four flavors:
polarity setters ("not," "without"), strength/hedge tokens
("ideally," "a bit," "really"), role markers ("starring,"
"directed by," "based on"), and explicit emphasis tokens
("must," "above all," "especially"). Don't try to memorize an
exhaustive list — recognize the principle: it adjusts polarity
or strength, doesn't redefine the trait.

The line between a meaning-shaping qualifier ("darkly funny")
and a strength modifier ("a bit funny") is whether removing the
front token leaves the trait pointing at the same thing.
"Funny" alone still points at humor — "a bit" is a modifier.
"Funny" alone is not the same trait as "darkly funny" — they're
different.

**Common pitfalls.**
- Promoting role markers to traits. "Starring," "directed by,"
  "from the studio of" — all modifiers, not traits.
- Promoting hedges to traits. "Ideally," "maybe," "kind of" —
  salience hints, not standalone traits.
- Splitting compound modifiers off as traits. "Lone female
  protagonist" — "lone" is not its own trait.
- Treating meaning-shaping qualifiers as modifiers. "Darkly
  funny" is one trait, not "funny" with a "darkly" modifier.

**Examples.**

Modifier (absorbed):
- "Not too violent" — polarity+strength modifier on "violent"
- "Ideally a slow-burn" — hedge on "slow-burn"
- "Starring Tom Hanks" — role marker on the entity
- "Movies based on a Stephen King novel" — "based on" role
  marker

Not a modifier (part of compound trait):
- "Darkly funny" — "darkly" reshapes "funny"
- "Iconic twist endings" — "iconic" scopes to "twist endings"
- "Morally ambiguous protagonist" — part of the archetype

---

### 3. Carver vs qualifier

**Definition.**
- **Carver** — defines what kinds of movies belong in the
  result set at all. Yes/no test ("does this movie have X?").
  A movie that fails is irrelevant, not just ranked lower.
- **Qualifier** — orders movies within a pool that other traits
  have already carved. Continuous test ("how much X does this
  movie have?"). A low-X movie is still a valid result, just
  ranked below higher-X ones.

Polarity is orthogonal — both can be positive or negative.
Negative carvers exclude; negative qualifiers downrank.

**How to think (the guiding principle).** For each trait, ask:
1. Is this trait qualifying another trait in the query? If
   it's narrowing, ranking, or steering the pool another trait
   defines — qualifier.
2. Is another trait qualifying this one? If yes, this one is
   a carver (it's the one being narrowed).
3. If it's not qualifying anything and nothing's qualifying
   it — it has to define the pool. Carver by default.

This handles the "popular movies" / "warm-hug movie" case
naturally: nothing else is in the query, so the trait can't be
qualifying anything and isn't being qualified — it must be the
carver. Don't reach for special promotion logic; the reasoning
principle resolves it.

**Boundaries.**
- Categorical traits (concrete facts: entities, dates, genres,
  settings, formats) tend toward carving — yes/no by nature.
- Gradient traits (experiential / evaluative qualities: mood,
  tone, popularity) tend toward qualifying when categoricals
  are present — continuous by nature.
- The decision is at the trait level, not the atom level.
  "Like Eternal Sunshine" decomposes internally into atoms
  (some categorical-looking) but the *trait* is one qualifier;
  internal categoricals don't escape and gate independently.

**Common pitfalls.**
- **Role-flipping on specificity when both are categorical.**
  "Horror movies set on a submarine" — both categorical, both
  carve. Submarine doesn't become "the real carver" while
  horror drops to qualifying. Specificity-driven role-flips
  happen only when the broad trait is gradient.
- **Letting an ordinal modifier promote a gradient to carver.**
  "Tarantino's least violent film" — "least violent" is still
  a gradient qualifier with an ordinal sort directive on top.
- **Confusing negation-as-carving with negation-as-qualifier-
  polarity.** "A non-violent crime thriller" — defining the
  pool by absence; carving by exclusion. "A rom-com that
  doesn't feel formulaic" — ranking direction over a
  continuous trait; negative qualifier. Test: if the trait
  appeared positively in this query, would it have been
  carving or qualifying? Same answer when negated.
- **Letting internal atoms of a parametric reference escape.**
  "Like Eternal Sunshine but set in space" — "like Eternal
  Sunshine" is one qualifier; the space-setting is the carver.
- **Same trait, different role via companions.** "A feel-good
  film" — feel-good carves (alone). "A feel-good comedy" —
  comedy carves, feel-good qualifies. Same word, different
  role.
- **Modifier-binding promoting a gradient to categorical.** "A
  quiet movie" — "quiet" attached to the whole movie reads
  gradient. "A movie with a quiet score" — "quiet" attached to
  the score is a categorical fact about a structural element.

**Examples.**

Carvers:
- "90s horror starring Anthony Hopkins" — three positive
  carvers (date, genre, entity)
- "A non-violent crime thriller" — "crime thriller" positive
  carver; "violent" negative carver (excluding by absence)
- "A feel-good film" (no other traits) — feel-good carves
  because nothing else does
- "Popular movies" (no other traits) — popular carves

Qualifiers:
- "Funny horror movies" — horror carves, "funny" qualifies
- "Dark slow-burn thriller" — thriller carves, "dark" +
  "slow-burn" qualify
- "A rom-com that doesn't feel formulaic" — rom-com carves,
  "formulaic" is a negative qualifier
- "Tarantino's least violent film" — Tarantino carves,
  "violent" qualifies (with ordinal sort downstream)
- "Like Eternal Sunshine but set in space" — space carves,
  "like Eternal Sunshine" qualifies

---

### 4. Salience

**Definition.** Per-qualifier weight on the qualifier side of
scoring. Two states:
- **Central** — a headline want; the query would feel
  fundamentally different without it.
- **Supporting** — meaningful but not load-bearing; rounds
  out the picture.

Carvers don't get salience (rarity does the weight work for
them downstream). Salience is qualifier-only.

**How to think (signals, in priority order).**
1. **Hedge / "nice to have" language — primary principle.**
   "Ideally," "if possible," "maybe," "would be nice," "kind
   of," "a bit." The user took the time to mark this as soft
   — strong signal that should never be ignored. Hedged →
   supporting. If a hedge is present, lean supporting even
   when other signals point central.
2. **Necessity language.** "Must," "need," "have to," "above
   all," "really want." Marks central explicitly.
3. **Corrective / contrastive structures.** "X but Y" — Y
   after "but" is often a corrective the user is tracking
   actively. Lean central for the corrective.
4. **Order of mention.** Earlier-mentioned qualifiers tend
   more central than later — first thing out of the user's
   mouth is often what they came to the search with.
5. **Headline position (clue, not rule).** Qualifiers in the
   adjective slot directly modifying the head noun
   ("slow-burn thriller") tend central; trailing modifier
   qualifiers ("a thriller, ideally slow-burn") tend
   supporting. A clue, override-able by other signals.

The unifying principle: salience tracks how much
**investment** the user put into the trait — words spent,
position chosen, hedge or emphasis added.

**Boundaries.**
- Don't force one qualifier in a query to be central. If only
  one qualifier exists, salience doesn't matter — a weighted
  sum of one item gets 100% regardless. If the only qualifier
  is hedged, mark it supporting.
- Salience is structural-importance, not strength-of-
  preference. "I really want a comedy" — "really" is intensity
  language but "comedy" is a carver, not a qualifier; salience
  doesn't apply.

**Common pitfalls.**
- Ignoring hedges because the trait is structurally prominent.
  "Ideally a slow-burn thriller" — supporting wins despite the
  adjective slot.
- Defaulting all qualifiers to central. Most queries have a
  mix.
- Treating salience as how much the user likes the trait.
  Salience is about how load-bearing the trait is.

**Examples.**

Central:
- "I really need a slow-burn thriller" — "really need" is
  necessity language
- "Above all I want it to be funny, and ideally short" —
  funny central; short supporting
- "A dark and brooding thriller" — both qualifiers in the
  adjective slot, no hedges, equal-weight central

Supporting:
- "Ideally a slow-burn thriller" — "ideally" hedges
- "A horror movie, maybe with some dark humor" — "maybe with
  some" hedges
- "A thriller, slow-burn would be nice" — "would be nice"
  hedges

---

### 5. Polarity

**Definition.** Whether the user wants the trait or wants to
avoid it.
- **Positive** — user wants the trait
- **Negative** — user wants to avoid / penalize the trait

Surface-grammar rule, not intent inference: if a polarity-
setter is present on a trait, polarity = negative. No second-
guessing of what the user "really" wants.

**How to think.** A polarity-setter is any token that signals
filter-out or downrank intent. The principle, not a fixed list:
tokens like "not," "without," "no," "avoid," "skip," "minus,"
"anything but," "spare me," "don't want" — anything reading as
"the user is calling out something to keep out or push down."
Recognize the function, don't memorize a taxonomy.

**"Not too X" is a special case.** Read as: polarity=negative,
salience=supporting. "Not too funny" means it's not a huge deal
if it's somewhat funny, but penalize when strongly present.
Negative direction, weak strength.

**Distribution scope.** When a polarity-setter sits at the
front of a coordinated phrase, does it apply to every conjunct?
- "Or" tends to distribute negation across both conjuncts.
  "Not too dark or sad" → both negative.
- "And" / "but" tends to break distribution. "Dark and funny"
  — only "dark" is governed by any preceding negation; "but"
  signals a pivot to a contrasting positive.

These are tendencies, not rules. Read the phrase as a person
and ask whether the negation reads naturally over each
conjunct. If it does, distribute.

**Boundaries.**
- Polarity is mechanical from surface grammar. Don't try to
  rewrite "movies that aren't boring" into positive intent for
  "engaging" — emit polarity=negative on "boring" and let
  downstream rewrite. Step 2's job is the surface signal.
- Polarity setters absorb as modifiers. "Without violence" →
  one trait ("violence," polarity=negative), not two.

**Common pitfalls.**
- Treating "not too X" as full negation. "Not too long" doesn't
  mean "must be short" — it means penalize long, don't kill
  mildly long. Polarity=negative, salience=supporting.
- Trying to memorize a closed list of polarity setters.
- Inferring polarity from intent rather than surface grammar.
- Mechanical if-statements for distribution. Read like a
  person.

**Examples.**

Positive:
- "Funny horror" — "funny" positive
- "A dark, brooding thriller" — both positive

Negative:
- "Not too violent" — "violent" negative, supporting
- "Without gore" — "gore" negative
- "Anything but a romcom" — "romcom" negative
- "Skip the jump scares" — "jump scares" negative
- "Not too dark or sad" — both negative (distribution over
  "or"); both supporting

Mixed:
- "A thriller that's tense but not too violent" — "tense"
  positive central; "violent" negative supporting

---

## Open items / TODOs

- **Modifier-token registry.** Curated list of polarity setters,
  salience hints, role markers, range/intensity modifiers the prompt
  can reference. Some implicit in current step-2 prompt; consolidate.
- **Atomicity test worked examples.** The prompt needs a tight set of
  examples covering modifier absorption, decomposition splits, and
  Bucket-B compounds (single concept, handler fans out).
- **Salience heuristics.** Concrete language for central vs
  supporting (headline wants, structural prominence, hedge tokens).
- **`category_candidates` length cap and prompt instruction.** Need
  to land on the exact wording that prevents enumeration ("list
  genuine contenders, skip obvious non-fits").
- **Followup LLM design.** Schema, prompt, and exact inputs for the
  per-query state assignments LLM that runs parallel to step 3.
