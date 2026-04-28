# v3 Step 2 Reasoning Fields

Companion to `v3_step_2_planning.md`. Per-field breakdown of every
free-text reasoning field in step 2's output schema:

- What the field accomplishes.
- How we want the LLM to think while filling it.
- Correct examples.
- Incorrect examples and what they get wrong.
- Small-LLM prompting guidance: pitfalls, biasing risks, how to
  scaffold without leading.

There are six reasoning fields, in schema order:

1. `span_analysis[].query_context`
2. `span_analysis[].decomposition.possible_splits[].reasoning`
3. `traits[].modifiers[].impact`
4. `traits[].purpose_in_query`
5. `traits[].category_candidates[].fits`
6. `traits[].category_candidates[].doesnt_fit`

Each field is positioned ahead of any decision it informs.

---

## 1. `query_context`

### What it accomplishes
Captures cross-span semantic dependencies — how a span's meaning is
shaped by other (non-modifier) spans in the query, or how it shapes
others. The upstream evidence that downstream `purpose_in_query`
(and transitively `role` / `category` / `polarity` / `salience`)
relies on. Catches relativized meanings — "funny" within a horror
frame, "dark" within a romcom frame — that change how the trait
should be searched.

### How to think
Walk every other content-bearing span in the query. For each, ask:
"if I removed that span, would this span's meaning change?" If yes,
articulate the shift. Bidirectional — note inbound shaping (others
affect me) and outbound shaping (I affect others). If the span
stands fully alone, leave the field empty.

### Correct
- Query: `"a funny horror movie"` / Span: `"funny"`
  > "Functioning relative to 'horror' — 'funny' here means tonal
  > lightness inside a horror frame, not standalone comedy. The
  > horror span sets the genre context this funniness operates
  > within."
- Query: `"horror movies from the 90s starring Anthony Hopkins"` /
  Span: `"Anthony Hopkins"`
  > `""` — actor name doesn't depend on or shape other spans.

### Incorrect
- "The user wants something that makes them laugh."
  *Restates the span's meaning, not its dependency on others.*
- "Anthony Hopkins is known for Silence of the Lambs."
  *Imports outside knowledge instead of describing intra-query
  dependencies.*
- "The span stands alone in the query."
  *Filler. The empty case is signaled by leaving the field blank,
  not by narrating that it's empty.*

### Small-LLM guidance
- **Confusion with "describe what the span means."** Most likely
  failure mode. Small LLMs default to describing content rather
  than relationships. Prompt has to repeatedly frame this as "how
  *other parts* shape this span" — not "what does this span say."
- **Outside-knowledge bleed.** "Anthony Hopkins is a famous actor
  known for…" — the LLM wants to be helpful and adds context the
  query doesn't actually invoke. Forbid information not present in
  the query itself.
- **Filling the field when it should be empty.** Small LLMs treat
  empty fields as failure. Provide multiple empty-case examples,
  label them as the correct response, and explicitly allow `""`.
- **Scaffold, don't lead.** Phrase the prompt as "if you removed
  [other span], would this span's meaning change?" rather than
  "what's the relationship between these spans?" The first forces a
  yes/no test; the second invites freeform speculation.

---

## 2. `decomposition.possible_splits[].reasoning`

### What it accomplishes
For each candidate piece that could be split out of the parent
span, evaluates whether that piece can stand as a query in its own
right. The per-piece atomicity test, surfaced as separate entries
so each candidate split is reasoned about independently. The
verdict (`should_split_out`) lives on the same object and emerges
*from* the reasoning.

### How to think
For each `isolated_span`, ask:
- Could a user submit just this piece as their entire query and
  have it make sense?
- Does this piece have a category home that could classify it
  standalone?
- If you pulled this piece out, would the residual span still carry
  the meaning the user intended?
- Would pulling this piece out create an orphan that no handler can
  act on?

If the piece passes all four → split it out. If any fails → don't.

### Correct
- Span: `"lone female protagonist"`, isolated_span: `"lone"`
  > "'Lone' alone has no category home — it's a relational
  > modifier (alone vs. ensemble). A user submitting just 'lone' as
  > a query makes no sense; the modifier needs a person-anchor.
  > Pulling 'lone' out would orphan it and lose the relational
  > meaning the full span carries — the user is asking for solo-
  > protagonist female-led movies, not just any female-led movie."
  >
  > `should_split_out: false`
- Span: `"creepy and atmospheric"`, isolated_span: `"creepy"`
  > "'Creepy' is a self-contained tonal want — a user could submit
  > just 'creepy movies' and it would classify as a viewer-
  > experience trait. Splitting it out from 'and atmospheric'
  > preserves both tonal wants distinctly rather than conflating
  > them into one ambiguous lookup."
  >
  > `should_split_out: true`

### Incorrect
- "Should not split because 'lone' is a modifier."
  *Verdict-first; the reasoning is just rationalization for an
  already-made decision.*
- "Two words, so should split."
  *Mechanical. Word count is not the test.*
- "Splitting is fine because each piece is a word."
  *Doesn't apply the standalone-query test.*

### Small-LLM guidance
- **Deciding the bool first, then writing the reasoning.** Schema
  position helps (reasoning comes before should_split_out), but
  small LLMs sometimes still write justification-shaped prose.
  Phrase the field as exploration: "describe whether this piece
  could work as a standalone query, then commit to the bool based
  on what you found."
- **Enumerating every possible split.** A 5-word span has many
  sub-spans, but most aren't meaningful split points. Tell the LLM
  to list only the splits a thoughtful reader would actually
  consider — not every adjacent word boundary. Empty list is fine
  when the span is unambiguously atomic ("Tom Hanks").
- **Structural test instead of usability test.** "Each piece is a
  word, so it can stand alone" is structural, not semantic. Anchor
  the reasoning in "could a user submit *just this piece* as their
  query?" That question forces a usability lens.
- **Calibration examples to include.** Modifier orphans (lone,
  very, classic), conjunction parallels (creepy and atmospheric,
  dark and slow), and residual cores (after pulling out a clear
  sub-trait, does the residual still mean what the user wanted?).
  Three failure modes, three example types.

---

## 3. `modifiers[].impact`

### What it accomplishes
For each absorbed modifier, articulates what the user is
communicating about how to treat the trait. Stays at user-intent
level — does not translate into system operations like "polarity
flip" or "weight reduction." Errs on the side of conservative
inference: a modifier suggests, it doesn't mandate.

### How to think
Read the modifier in the context of the full query. Ask: what is
the user telling the system about how to handle this trait? Stay
close to what's actually said. Don't over-claim — modifiers
typically weight or filter, not pin exact thresholds. Don't
translate into operational primitives.

### Correct
- Trait `"scary"`, modifier `"not"`
  > "Shows the user wants to filter out or downrank movies that
  > have this trait."
- Trait `"scary"`, modifier `"a bit"`
  > "Reduces the importance of scariness relative to other traits
  > in the query — wanted but not central."
- Trait `"boring"`, modifier `"not too"`
  > "User accepts some boredom but wants to avoid movies that lean
  > heavily into it. Mild is OK; heavy is not."
- Trait `"Tom Hanks"`, modifier `"starring"`
  > "Tom Hanks should be one of the most prominent actors in the
  > cast — featured rather than minor. Does not require him to be
  > the #1 lead."
- Trait `"creepy"`, modifier `"ideally"`
  > "User prefers but does not require this. A movie that misses
  > the mark on creepiness is still acceptable."

### Incorrect
- `"not"` → "Polarity flip to negative."
  *System operation, not user intent.*
- `"starring"` → "Tom Hanks must be the lead actor."
  *Over-infers — pins top billing when the modifier doesn't say
  that.*
- `"ideally"` → "User strongly prefers this."
  *Reads the hedge as strength; "ideally" is the opposite of
  strong.*
- `"not"` → "Means the user doesn't want it scary."
  *Tautological — restates surface meaning without articulating
  what the system should do.*

### Small-LLM guidance
- **Drift into system-operation language.** "Polarity flip,"
  "weight reduction," "salience marker" — small LLMs that have seen
  similar prompts in training will reach for these terms.
  Explicitly forbid them and require user-intent phrasing.
- **Over-inference.** "Starring" → top billing. "About" →
  exclusively about. "From the 90s" → exactly 1990-1999 inclusive.
  The LLM wants to be specific, but specificity the modifier
  doesn't actually carry is a mis-read. Provide explicit
  calibration: "starring means prominent, not top-billed."
- **Hedge mis-reading.** "Ideally," "preferably," "would love," "I
  think" — softeners that small LLMs sometimes read as enthusiasm
  markers. Pair softeners with their dampening effect in worked
  examples.
- **Calibration vocabulary.** Provide a curated modifier registry
  in the prompt — common polarity setters, hedges, role markers,
  intensity adjusters — with the intended impact framing for each.
  This is the modifier-token registry TODO from the planning doc.
- **Scaffold, don't lead.** Ask "what is the user telling you about
  how to handle this trait?" rather than "what operation does this
  modifier perform?" The first stays at user-intent level; the
  second invites system-operation language.

---

## 4. `purpose_in_query`

### What it accomplishes
A complete reading of what this trait is trying to accomplish in
the query. Not a role hint — the central reasoning hub for the
trait. Every committed downstream field (`category_candidates`,
`best_fit_category`, `role`, `polarity`, `salience`) draws on this.
It exists so the LLM does the real understanding work once, in
concrete user-want language, before being asked to commit to any
structured label.

### How to think
Cover four dimensions, in concrete user-want language (no category
labels, no system jargon):

1. **Concrete want or avoidance.** What specifically about a movie
   does the user want this trait to provide or rule out? Speak in
   things-in-the-movie terms: actors, eras, plot shapes, tonal
   qualities, audiences.
2. **Relationships to other traits.** Does this trait constrain
   another (genre frames it), is it constrained by another, paired
   with another (parallel wants), or redundant with another (mode
   word + dimension)?
3. **Criticality.** Would the user reject a movie that fails this
   trait, accept it grudgingly, or barely notice? Pull from query
   phrasing — hedges, headline position, repetition.
4. **Concreteness over abstraction.** Don't slip into category
   labels. Say "user wants this specific actor featured prominently,"
   not "this is a person-credit trait."

### Correct
- Query: `"horror movies from the 90s starring Anthony Hopkins"` /
  Trait: `"Anthony Hopkins"`
  > "User wants movies that feature Anthony Hopkins prominently in
  > the cast — the 'starring' modifier signals featured-not-cameo,
  > though not necessarily top-billed. Sits alongside the genre
  > constraint (horror) and the era constraint (90s) as one of
  > three intersecting requirements; together they define a narrow
  > eligible set. The actor is named directly and concretely with
  > no hedge, making it a hard requirement. A movie that doesn't
  > feature Anthony Hopkins doesn't partially satisfy the query —
  > it fails it."
- Query: `"ideally creepy and atmospheric horror"` / Trait: `"creepy"`
  > "User wants the horror to deliver sustained tonal unease — the
  > slow-burn, get-under-your-skin variety rather than jump-scare
  > adrenaline. Operates inside the horror frame (the genre is the
  > carver), so it's a quality of the eligible set rather than a
  > standalone constraint. Paired with 'atmospheric' as a parallel
  > tonal want; together they sketch a psychological / slow-burn
  > flavor of horror. The 'ideally' hedge marks it as preferred not
  > required; a horror movie that's less creepy is still
  > acceptable, just less wanted."

### Incorrect
- "Carving constraint."
  *Truncates to role-shape only; skips the four dimensions.*
- "Cat 1 person-credit constraint, hard, central."
  *Speaks in category and role labels — the structured fields'
  job, not this field's.*
- "User wants Anthony Hopkins in the movie."
  *Concrete but missing relationships, criticality, and modifier
  nuance.*
- "This is a positive carver in Cat 1, central salience."
  *Pre-commits the downstream fields it's supposed to feed.*

### Small-LLM guidance
- **Pre-committing structured fields.** Biggest risk for this
  field. The LLM has just seen the trait and wants to commit to
  category, role, polarity, salience. If it does that inside this
  field's prose, the downstream fields become rubber-stamps.
  Explicitly forbid naming any category, role, polarity, or
  salience inside this field, and require user-want phrasing only.
- **Truncation to one dimension.** Small LLMs default to short
  answers. The four-dimension scaffold (want / relationships /
  criticality / concreteness) is the antidote — make it a checklist
  the LLM walks through. Worked examples should show all four
  dimensions covered explicitly.
- **Category-language slip.** "This is a person-credit trait" is
  system-language thinking. Ban category names and the words
  "carver," "qualifier," "central," "supporting," "positive,"
  "negative" inside this field. Describe in user-facing terms what
  the trait wants from a movie.
- **Hedge mis-reading.** Same as for `modifiers[].impact`. If the
  trait has a softener attached, criticality should reflect that
  softener. Worked examples must include hedged traits ("ideally
  creepy") with appropriately soft criticality.
- **Scaffold, don't lead.** Phrase the four-dimension scaffold as
  questions — "What does the user want from a movie? How does this
  interact with other parts of the query? How critical is it?" —
  not as labels to fill in. Questions invite reasoning; labeled
  slots invite formula.
- **Length cap.** Substantive but not rambling. Target 3-5
  sentences. State the cap in the prompt.

---

## 5. `category_candidates[].fits`

### What it accomplishes
For each genuine contender category, articulates why the trait
could plausibly live there — pointing to the specific aspect of the
trait that lines up with the category's concept definition, *and*
how strongly that alignment holds. The strength signal is folded
into the prose: the field should say both "why fits" and "how
prototypically."

### How to think
Identify the specific axis of the candidate category. Identify the
specific feature of the trait. Articulate the alignment, then
qualify how strongly the trait represents that category — is it a
canonical surface form (prototypical), a clear secondary mention
(adjacent), or only an indirect implication (weak)? One unified
statement covers both pieces.

### Correct
- Trait `"blockbuster"`, candidate: Cat 17 (Financial scale)
  > "Cat 17 covers budget and box-office magnitude as a single
  > financial-scale axis. 'Blockbuster' is canonical high-budget /
  > high-grossing language — one of the prototypical surface forms
  > for this category. Strong alignment, no interpretation needed."
- Trait `"blockbuster"`, candidate: Cat 38 (Reception-prose
  descriptors)
  > "Cat 38 covers reception-prose framing ('classic',
  > 'era-defining'). 'Blockbuster' carries an implied reception
  > signal — a movie called blockbuster is by definition widely
  > received — but the financial-scale signal is more direct.
  > Plausible candidate; alignment is indirect rather than
  > prototypical."

### Incorrect
- "It's a kind of movie."
  *Vague — doesn't engage the category definition.*
- "Could fit."
  *Zero content.*
- "Cat 17 is for financial scale and blockbuster fits."
  *States the alignment but doesn't ground how strongly. Could be
  prototypical or weak; the merger downstream can't tell.*

### Small-LLM guidance
- **Enumerating every plausible category.** With 43 categories,
  several might pass a vibes-level "could fit" check. Require
  genuine boundary contenders only — categories the LLM seriously
  considered as the home, not categories it can construct an
  argument for. Cap at 2-4 entries; even unambiguous traits emit
  one entry for schema uniformity.
- **Vague alignment language.** "Talks about X-ish things," "is
  about that kind of stuff." Require specific mention of the
  category's axis and the trait's matching feature. Worked examples
  should illustrate axis-naming explicitly.
- **Missing the strength qualifier.** With confidence merged into
  the prose, the LLM may forget to include it. Require both pieces
  — alignment reason *and* alignment strength — in every entry.
  Worked examples must show both pieces.
- **Strength as filler.** "Strong alignment" with no grounding is
  no better than no qualifier at all. Tie strength to observable
  evidence: prototypicality of the surface form, axis directness,
  whether the trait names the category's home concept directly or
  only adjacent to it.
- **Scaffold, don't lead.** Ask "what specific axis of this
  category does the trait line up with, and how prototypical is
  that alignment?" — not "explain why this fits." The first forces
  specific reasoning; the second invites generic praise.

---

## 6. `category_candidates[].doesnt_fit`

### What it accomplishes
Surfaces the specific boundary concern that might disqualify this
candidate, *and* how serious the concern actually is. Empty when
there is no real concern. Makes the eventual `best_fit_category`
defensible against the alternative. The seriousness signal is
folded into the prose.

### How to think
What evidence — in the trait or in `query_context` — pulls away
from this category? Wrong axis (financial vs. reception vs.
experiential)? Missing surface form? Wrong role-shape (this trait
is gating, the category is qualifying-style)? Identify the concern,
then qualify how serious it is — does the evidence directly
contradict the category, or is the concern only theoretical?

If there's no real concern, leave the field empty. Do not
manufacture concerns to fill the slot.

### Correct
- Trait `"blockbuster"`, candidate: Cat 38
  > "Cat 38 captures reception-prose framing, which 'blockbuster'
  > adjacent-implies but doesn't directly assert. The financial-
  > scale axis is the dominant signal; reception is a downstream
  > consequence. Routing here would lose the budget / box-office
  > specificity. Moderate concern — there's genuine adjacency, but
  > financial scale is clearly the primary axis."
- Trait `"Tom Hanks"`, candidate: Cat 1 (Person credit)
  > `""` — actor name maps cleanly to person credit, no boundary
  > concern.

### Incorrect
- "Maybe doesn't fit."
  *Phantom concern, no specifics. Should have been left empty.*
- "Cat 38 is for reception."
  *Restates the cat's purpose without saying why this trait
  specifically pulls away.*
- "Some concern."
  *Says nothing.*
- "Could be wrong."
  *Manufactured uncertainty.*

### Small-LLM guidance
- **Manufacturing phantom concerns.** Small LLMs treat empty fields
  as failure. They invent concerns to feel productive. Explicitly
  say "leave empty when there is no real concern" and provide
  multiple empty-case examples labeled correct.
- **Restating the category instead of contrasting.** "Cat 38 is for
  reception" tells you nothing about why *this trait* doesn't fit.
  Require naming the trait's specific feature that pulls away from
  the category's specific axis.
- **Missing the seriousness qualifier.** With confidence merged
  into the prose, easy to forget. Require both pieces — concern
  reason *and* concern seriousness — in every non-empty entry.
  Empty stays empty for both.
- **False symmetry with `fits`.** The LLM may feel that every
  candidate needs both a fits and a doesnt_fit entry to be
  balanced. Wrong — clean fits with no real boundary concern should
  leave doesnt_fit empty. Call out asymmetry as the expected case.
- **Scaffold, don't lead.** Ask "what evidence in the trait or
  query pulls away from this category, and how serious is that
  concern?" Don't ask "why might this category not fit?" The first
  formulation requires specific evidence; the second invites
  speculation.

---

## Cross-cutting prompting principles

These apply to all reasoning fields, not just one:

- **Position reasoning before decisions.** Every reasoning field
  sits ahead of the structured field it informs. Reinforce this in
  the prompt: reasoning happens before the LLM commits to a labeled
  value. Prose-after-decision becomes rationalization;
  prose-before-decision becomes evidence.
- **Forbid label leakage.** Reasoning fields should not name the
  structured fields' enums (categories, roles, polarities, salience
  states) inside their prose. Doing so collapses reasoning into
  pre-commitment. The structured fields exist to capture
  commitment; the reasoning fields exist to support it.
- **Empty is a valid answer.** `query_context`,
  `decomposition.possible_splits` (whole list), and `doesnt_fit`
  all have legitimate empty cases. Small LLMs resist emptiness.
  Make emptiness a labeled-correct response with multiple worked
  empty-case examples.
- **Provide calibration examples, not category lists.** For
  `modifiers[].impact` especially, the LLM needs to see specific
  modifiers paired with their intended readings. A registry of
  common modifiers (polarity setters, hedges, role markers,
  intensity adjusters) with do/don't readings is more useful than
  abstract guidance.
- **Cap free-prose length.** Small LLMs ramble when given freeform
  prose freedom. Each reasoning field should have a soft length
  target appropriate to its scope: 1-2 sentences for `impact` and
  `fits` / `doesnt_fit`; 3-5 sentences for `purpose_in_query` and
  the exploratory reasoning fields.
- **Forbid outside knowledge.** Several fields tempt the LLM to
  import knowledge not in the query — actor filmographies, genre
  conventions, decade context. Constrain reasoning to evidence
  present in the query and (where relevant) the category
  definitions.
- **Phrase prompts as questions, not labels.** "What is the user
  telling you about how to handle this trait?" works better than
  "impact:". Questions invite reasoning; field labels invite
  formula. The field labels are for the schema; the prompt
  instructions should translate them into questions.
