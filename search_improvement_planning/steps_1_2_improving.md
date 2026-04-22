# Steps 1-2 Improving

Working note capturing what we learned while iterating on Search V2
Stage 1 and Stage 2. This is not a finalized spec; it is a practical
record of the failure modes we observed, what principles we extracted,
and what should guide the next revision.

## Current Problems

### Step 1 Problems

- `intent_rewrite` has been over-expanding the user query into inferred
  retrieval dimensions instead of staying a faithful rewrite.
- It has been injecting proxy traits that the user did not actually
  ask for, such as `iconic`, `highly-rated`, or similar evaluative
  language.
- It sometimes upgrades a literal clue into a judgment. Example:
  `"Indiana Jones movie where he runs from the boulder"` becoming
  something like `"iconic scene"`.
- It has shown a tendency to over-match on local patterns rather than
  applying broad evidence rules. Example: treating `"Marvel movies"`
  as if it might naturally be a title search because other `"X movies"`
  cases can involve title ambiguity.
- `primary_intent` drifted away from "most likely interpretation" and
  could become "most interesting" or "most useful" instead.
- The old top-level schema carried extra ambiguity scaffolding that was
  not used downstream, adding verbosity and token cost without runtime
  benefit.

### Step 2A Problems

- The ingredient inventory is often wrong, and once the inventory is
  wrong the rest of the Stage 2 chain is built on a bad foundation.
- Step 2A has been treating paraphrased or inferred rewrite content as
  if it were genuine user-backed ingredients.
- It can preserve ingredients that do not actually help retrieval.
  Example: broad or exhaustive phrases that do not narrow the search
  meaningfully.
- It can merge clearly distinct ideas into one ingredient or one
  concept. Example: `"animated and live-action films"` being treated
  as one concept rather than two distinct categories, even though as a
  pair they are nearly exhaustive and add little value.
- It has not had a strong enough notion of "actionable concept" versus
  "descriptive paraphrase from the rewrite."
- Step 2A has been too dependent on the exact phrasing of Step 1's
  rewrite, which means upstream overreach quickly turns into downstream
  garbage.

### Step 2A — Second-Round Problems (After The Rewrite)

After the Stage 2A rewrite (new `PlanningSlot` + `Step2AResponse`
schema, the `interpret` verdict, decompose-first-then-group flow,
capability-language endpoint descriptions, and branch-dynamic prompt
dispatch), empirical probing against a new feedback-query set surfaced
a distinct class of failures that the new shape partially fixed and
partially exposed:

- **Cross-family fusion.** Model would fuse atoms from different
  retrieval families (semantic + keyword) into one slot despite its
  own output naming two families. Surfaced the need for an explicit
  "cross-family atoms never fuse" rule and the "one slot = one family"
  framing.
- **Hallucinated family capabilities.** "Boomers love" was interpreted
  as "demographic popularity metrics" against the metadata family,
  which only supports global popularity. Surfaced the need for explicit
  per-family "what this family CANNOT do" notes (metadata popularity
  is global-only; keyword is closed-taxonomy; trending is real-time-only).
- **Idiom under-expansion.** Slang phrases like "main character energy"
  and "popcorn movies" passed through as single literal units. The
  verdict pattern implicitly capped output at one atom per phrase,
  even when the phrase genuinely decomposed into multiple retrieval
  targets. Surfaced the need for a verdict that accepts 1+ output
  atoms, with cardinality revealed by the phrase rather than
  pre-classified.
- **Compound interpretation strings.** A single `best_guess`/`interpret`
  output could mash multiple families into one string ("popcorn" →
  "blockbuster action with simple fun" packs semantic + keyword +
  metadata). Surfaced the need for single-family-per-atom discipline
  enforced by micro-format.
- **Multi-attribute fusion drift.** Same-family atoms on DIFFERENT
  sub-attributes (metadata.popularity + metadata.reception) sometimes
  fused into one slot, despite the downstream constraint that each
  metadata expression targets exactly one column. Observed
  inconsistency across runs (Big-budget + blockbuster fused one run,
  split the next). Surfaced the need to extend the fusion rule from
  "same family + ranking" to "same family + same sub-dimension +
  ranking".
- **Plot-scene over-translation.** Concrete plot events like "runs
  from a rolling boulder" could interpret into a specific movie title.
  Surfaced the need to keep `interpret` strictly for non-descriptive
  language (idiom, slang, metaphor, unsupported capability) and to
  default descriptive language to `literal`.

### Step 2B Problems

- Step 2B originally trusted Step 2A boundaries too passively and could
  drop important parts of a concept while still sounding coherent.
- Example failure: `"historically significant Disney movies"` produced
  expressions that preserved historical significance but forgot to
  preserve Disney.
- Even after adding required-ingredient coverage, Step 2B can still be
  forced to execute on a bad concept rather than challenge whether the
  concept should exist at all.
- Step 2B does not yet have a strong enough sanity-check on whether a
  concept is coherent, useful, and worth preserving.
- The chain is still vulnerable to "garbage in, garbage out": explicit
  coverage constraints help preserve a concept, but they do not solve
  Step 2A giving Step 2B the wrong concept in the first place.

## Step 1's True Role

### What Step 1 Should Handle

- Choose the correct major flow: `exact_title`, `similarity`, or
  `standard`.
- Decide whether additional branches would materially improve browsing.
- Identify the most likely reading of the query and make that the
  `primary_intent`.
- Produce optional alternatives only when they are genuinely useful and
  meaningfully different.
- Rewrite each branch into a clear, faithful search statement.
- Preserve the original level of specificity.
- Clarify wording when needed for readability or disambiguation.
- Recognize when the primary's intent describes a broad set worth
  subdividing, and emit `creative_alternatives` that propose
  productive sub-angles within that intent (animated / modern-era /
  etc.). Spins narrow within one intent; alternatives change the
  intent. Keep these two output channels semantically distinct.

### What Counts As Overstepping

- Decomposing the query into ingredients, filters, concepts, or
  retrieval axes.
- Turning one vague phrase into multiple inferred retrieval dimensions.
- Injecting proxy qualities like `iconic`, `highly-rated`,
  `prestigious`, `important`, or similar evaluative language unless the
  user directly asked for that idea.
- Turning a literal clue or scene description into a judgment.
- Narrowing a broad query into a subtype the user did not explicitly
  signal.
- Choosing a more interesting branch over the most likely branch.
- Treating every bare `"X movies"` query as if it carries meaningful
  exact-title ambiguity.

### Practical Framing

Stage 1 should route and lightly rewrite. It should not solve the query.
Its job is to preserve the user's request in a cleaner form, not to
pre-decompose it into what later stages might want.

## Working With Small LLMs

### Schema-Side Learnings

- Keep schemas lean. Extra fields that are not consumed downstream add
  token cost and cognitive load without enough value.
- Prefer one compact reasoning field over several overlapping ones when
  they serve the same purpose.
- Keep reasoning fields brief and structured. Small models drift when
  asked for open-ended prose explanations.
- Use field order as cognitive scaffolding: evidence or reasoning first,
  then the decision it supports.
- Avoid asking the model for values that code can derive.
- When optional sections may be absent, prefer empty collections rather
  than extra skip booleans.
- When a field is meant to constrain generation, define it as an
  upstream evidence inventory rather than a retrospective explanation.
- Add explicit validation where correctness matters. In Stage 2 this
  meant validating ingredient preservation and expression coverage in
  code, not relying only on prompt discipline.
- Reasoning trace fields need a defined micro-format. Free-form prose
  ("main=X; ambiguity=Y; alt=Z") lets the model write category labels
  ("vibe/preference") and feel done. Structured per-item lines with
  explicit verdicts (`<item> -> primary | emit | skip (reason)`)
  force per-item commitment that propagates to structured output by
  generation order.
- Field declaration order does the work that cross-field validators
  might. Putting derivative outputs AFTER prerequisite outputs (e.g.
  `creative_spin_analysis` after `alternative_intents`) decouples
  reasoning by construction — the model has already committed to the
  earlier output before generating the later one. This is especially
  useful when you don't want to add validators.
- When two output fields share a data shape but have distinct
  semantics, prefer separate classes over reusing one class.
  `AlternativeIntent` and `CreativeSpin` are nearly identical
  structurally, but downstream rendering, prompt discipline, and
  semantic intent diverge enough that one merged class would have
  cost more than it saved.
- Partition-invariant validators catch silent content drops that
  prompt discipline cannot. For Step 2A: `scope ⊆ inventory` AND
  `inventory ⊆ union-of-slot-scopes`. The first stops slots from
  citing phrases that were never committed; the second stops
  committed phrases from falling on the floor. Enforce the
  invariants in a Pydantic `model_validator` — the prompt enforces
  the shape, the validator enforces the math.
- Free-form trace fields can carry structured verdicts. When the
  verdict discipline needs to live in the output without introducing
  new schema types, a tight micro-format inside a free-form string
  (`<phrase> -> <verdict> (<family>, <≤8 words why>)`) is as
  reliable as a structured verdict enum — and cheaper. Reserve
  schema fields for content downstream actually reads.
- Branch-dynamic prompt dispatch, at the section level. When an
  upstream stage produces different input shapes for different
  branches (primary / alternative / spin), dispatch on branch kind
  in code and assemble only the prompt sections this branch's inputs
  justify. The model never sees instructions for fields it isn't
  receiving. Cost: a few section dicts and a dispatch function.
  Benefit: no wasted reasoning on absent fields, no self-classification
  the code could do.

### Prompt-Side Learnings

- Broad principles work better than large catalogs of examples.
- Examples should clarify boundary cases, not carry the main burden of
  teaching the behavior.
- Small models over-match on salient patterns, so too many narrow
  examples can make the prompt worse rather than better.
- The right structure is:
  1. define the task narrowly
  2. define what the model should not do
  3. define the evidence hierarchy or decision rules
  4. add a few boundary examples
  5. describe output fields in schema order
- Explicitly distinguish what counts as positive evidence for each
  reading instead of only listing bad outcomes.
- Tell the model how to abstain or emit `none` when there is no real
  ambiguity or no useful alternative.
- Keep reasoning outputs in compact trace form rather than paragraph
  prose to reduce latency and cost while still preserving the decision.
- Per-item verdicts beat global summaries. A trace that says
  `ambiguity=vibe/preference` lets the model think it's done. A trace
  that says `<reading> -> emit | <reading> -> skip (reason)` forces
  per-item commitment that mechanically lines up with structured
  output count.
- A "skip" verdict as a first-class option prevents both over-emission
  and under-emission. Without it, models either treat every
  identified candidate as worth emitting or refuse to enumerate at
  all to avoid being forced into emission. Skip lets the model
  identify a candidate AND explicitly reject it with a reason.
- Concrete-grounding requirements have to be specified per-field, not
  inferred from a general principle. `routing_signals` produced
  concrete content because it explicitly demanded query-text
  citation; `ambiguity_analysis` produced category labels because it
  didn't. Whenever a field needs grounding, name the grounding
  source it must cite.
- Worked positive-shape examples (showing the correct primary + alts
  together) shift behavior more than negative-shape examples or
  rules alone. Pair every "Do NOT" bullet with a worked failure → fix
  example. The boundary example for `"Disney live action movies
  millennials would love"` was inert when it only said "useful
  alternatives may vary only that phrase"; it became load-bearing
  once it showed the actual primary + alt rewrites.
- Conceptual distinctions need explicit names with behavioral tests.
  Naming "vagueness vs ambiguity" and giving the test (would the
  candidates retrieve different items?) handled an entire class of
  failures more effectively than additional rules.
- Hedging connectives ("or", "and") in a rewrite are a tell for
  unbranched ambiguity. Banning them with a worked example
  (`"popular with or nostalgic for"` → branch instead) forces
  branching.
- Evaluative-word substitution is a hidden form of enrichment that
  the standard "no proxy traits" rule does not catch. `"best"` →
  `"highly rated"` is the model picking one specific interpretation
  of a deliberately broad word. Needs a separate explicit rule
  listing the broad words to preserve verbatim (best, top, great,
  good, favorite, classic).
- Brevity needs explicit word caps. "Be brief" gets interpreted
  softly; "≤8 words, no full sentences" gets followed. Examples that
  demonstrate brevity help, but the explicit cap is cheap insurance.
- Decompose-first-then-group. When a field needs both expansion
  (idiom → atoms) and grouping (atoms → slots), force expansion as
  a prior pass via micro-format. Grouping decisions operate on
  atoms, not raw input phrases. Collapsing expansion and grouping
  into one verdict causes under-expansion — the model treats the
  input unit as the output unit even when the phrase genuinely
  contains multiple retrieval targets.
- Verdict unification over pre-classification. Rather than giving
  the model two verdicts ("translate to one thing" vs "decompose
  into many"), give one verdict with variable cardinality
  ("interpret into 1+ atoms"). The cardinality reveals itself from
  the phrase's structure; the model never has to pre-classify.
  Pre-classification is a common over-commitment trap because the
  pick gate runs before the content is written.
- Single-family-per-atom discipline. Every atom produced by an
  interpret-style verdict must carry exactly one retrieval-family
  tag. Enforce in the micro-format (one family per atom line) AND
  in an explicit "if you name two families, split into two atoms"
  rule. Compound atom strings that mash families are a common
  silent failure: downstream sees one item but the atom encodes
  multiple retrieval operations.
- Capability descriptions tuned to prevent observed hallucinations.
  For any family with internal structure (metadata columns, keyword
  categories, vector spaces), list the sub-dimensions as bullets
  with one-line descriptions, and include explicit "this family
  CANNOT do X" notes covering the likely hallucinations (global-only
  popularity, closed-taxonomy keyword, real-time-only trending). A
  generic one-line family description is not safe for families
  with internal structure.
- Principle-illustrating examples, not test-query-derived examples.
  Examples drawn from queries the prompt was built to address teach
  the model to pattern-match on those specific phrases, not the
  underlying principle. Draw example content from a disjoint pool.
  This is the Stage-2A analog of the example-eval separation
  convention already established for generation prompts.
- Fusion criteria need a sub-dimension clause when the family has
  internal structure. "Same family + ranking" was insufficient for
  fusion in multi-attribute families. Full rule: same family +
  same sub-dimension (where applicable) + joint ranking gradient.
  Two-condition rules let the model default in the fusing direction.
- "If in doubt, split" as explicit default for fusion decisions.
  Over-fusion failures outnumber under-fusion failures in probes,
  so make the conservative choice the explicit default. Two focused
  slots are strictly better than one compound slot.
- Observed inconsistency across runs is a signal of rule
  under-specification, not temperature variance. When the same
  input produces fused output one run and split output the next,
  the criterion has an ambiguity the model is resolving differently
  each time. Tighten the criterion; don't explain it away as sampling.
- Revert before you fix. When a prompt change produces wrong output,
  pause to identify the general pattern underlying the failure
  before applying a targeted fix. A fix that addresses one observed
  instance may paper over a deeper ambiguity the next instance will
  expose. The reliable loop is: probe → extract principle → modify
  prompt → re-probe, not probe → patch → re-probe.

## What We Learned From The Step 1 Schema Changes

### Specific Takeaways

- `ambiguity_level`, `hard_constraints`, and `ambiguity_sources` were
  useful as scaffolding during design, but once they were no longer used
  downstream they became expensive prompt baggage.
- `ambiguity_analysis` can absorb their value if it is tightly defined.
- `routing_signals` is still useful because it anchors each branch in
  concrete query evidence instead of free-floating interpretation.
- `difference_rationale` remains useful on alternatives because it helps
  prevent paraphrase branches.
- `primary_intent` should be explicitly defined as the most likely
  interpretation, not the most useful one.

### General Principles Going Forward

- If a field is not consumed downstream and does not materially improve
  model behavior, remove it.
- If several fields are trying to capture closely related reasoning,
  merge them into one better-defined field.
- Prefer compact, structured reasoning traces over freeform explanation
  prose.
- Keep the schema focused on the smallest set of fields that enforce the
  right behavior.
- Put broad behavioral control in the prompt and validation logic, not
  in a large pile of auxiliary reasoning fields.
- Use schema simplification as a latency and reliability tool, not just
  a cleanliness improvement.

## What We Learned From The Step 1 Prompt Iteration

These are the prompt-only lessons (no schema changes other than
adding the creative-spins fields) from the round of fixes that
followed the schema simplification.

### Specific Takeaways

- The original `ambiguity_analysis` failure mode was structural, not
  attitudinal. The model wasn't refusing to branch; it was writing
  category labels in the trace and then defaulting to no-alt because
  the trace format had no commitment hook into the structured output
  count. The fix was a format change, not stricter rules.
- Replacing the `main=...; ambiguity=...; alt=...` format with a
  `readings:` enumeration with per-item verdicts (primary / emit / skip)
  fixed four distinct queries that previously failed in the same way.
  This was the highest-leverage prompt change in the iteration.
- The same per-item-verdict pattern generalized to `creative_spin_analysis`
  with no additional design work. Once the pattern was established
  it became a reusable scaffold.
- Hedging in rewrites ("X or Y") was a silent failure mode. The model
  was branching internally but compressing the result into one
  rewrite string. Banning the connective with a worked example
  surfaced the branching into structured output.
- The "vagueness vs ambiguity" distinction did more work than any
  rule. Most of the original failures dissolved once the model could
  classify a phrase as one or the other.
- Creative spins are a different kind of branching than alternatives,
  not just more of the same. The cleanest path was a separate output
  field (and class) — mixing them into `alternative_intents` would
  have eroded the discipline we'd just built into that field.
- Evaluative-word substitution ("best" → "highly rated") is a distinct
  failure from proxy-trait addition and needed its own rule and
  example. The general "no enrichment" rule didn't catch it because
  the model treats substitution as clarification.
- Brevity drift is real. Once the spin format invited per-item
  reasoning, parentheticals ballooned to 15-20 words even though the
  readings format with the same structure stayed at 7-10 words. The
  difference was that the readings examples modeled brevity but the
  spin format didn't have an explicit cap. Adding "≤8 words" fixed it.

### General Principles Going Forward

- A trace field is only as useful as its format. Default to a
  micro-format with per-item verdicts before reaching for new fields.
- Prefer field ordering and separate classes over validators. The
  structured-output generation order does most of the work that
  cross-field validators would.
- When extending a working pattern (readings → spins), reuse the
  structural scaffold (per-item verdicts with skip option) rather
  than designing a new one. Reuse compounds.
- Pair every "Do NOT" bullet with a worked failure → fix example. The
  examples produce more behavior change than the bullet alone.
- Watch for substitution as a sneaky form of the failure modes you
  thought you'd already banned (substitution vs addition,
  compression vs decomposition, paraphrase vs interpretation).

## What We Learned From The Step 2A Rewrite

The Step 2A rewrite combined a schema change (new `PlanningSlot`
and `Step2AResponse` with a partition-completeness validator,
split into its own `search_v2/stage_2a.py` module with
branch-dynamic system-prompt dispatch) and a prompt-content
rewrite (the `interpret` verdict replacing `best_guess`,
decompose-first-then-group reasoning flow, capability-language
endpoint descriptions with per-attribute / per-category / per-
vector-space detail, a three-condition fusion criterion, and
boundary examples drawn from a disjoint pool).

### Specific Takeaways

- **`interpret` unified `best_guess` and decomposition.** The old
  schema had a verdict ("best_guess translates the phrase into a
  broader interpretation") that implicitly capped output at one
  atom. Phrases like "popcorn movies" that honestly decompose into
  a tone atom + a production-scale atom were forced into one
  compound string. Replacing with an `interpret → 1+ atoms`
  verdict lets cardinality reveal itself from the phrase structure.
- **Single-family-per-atom, enforced by micro-format.** The
  `(family: <family>, <≤8 words why>)` suffix per atom line, plus
  an explicit "if you name two families in one atom, split into
  two atoms" rule, prevents the compound-string failure where one
  interpret output mashes multiple retrieval operations into one
  item.
- **Descriptive language stays literal; interpret is the escape
  hatch.** Plot events, tonal adjectives, archetypes, named genres,
  and concrete scenes all default to `literal`. `interpret` fires
  only for idiom, slang, metaphor, or phrases naming an unsupported
  family capability. Earlier prompt drafts let `interpret` behave
  as a synonym-swap for descriptive language, which caused
  plot-scene over-translation into specific titles.
- **Three-condition fusion rule resolved observed inconsistency.**
  "Same family + ranking-style" was ambiguous on multi-attribute
  families (metadata's 10 attributes, keyword's 7 categories). Adding
  "same sub-dimension within multi-attribute families" as an explicit
  third condition eliminated the observed inconsistency (Big-budget +
  blockbuster fused one run, split the next). Semantic stays a
  single fusion unit because its vector spaces can be queried
  together in one slot.
- **Capability descriptions sized to prevent hallucination.** The
  generic family description ("metadata — release date, runtime,
  popularity, reception, etc.") was short enough to miss the fact
  that popularity is global-only. Expanding to per-attribute bullets
  with explicit "CANNOT do X" notes is the direct antidote to
  family-capability hallucinations like "demographic popularity
  metrics".
- **Boundary examples must not draw from production queries.**
  Examples drawn from queries being debugged taught the model to
  pattern-match on those phrases, not the underlying principle.
  Pulling example content from a disjoint pool ("a con-artist crew
  breaking into a bank vault", "an underground sleeper hit", "films
  especially beloved by younger viewers") keeps each example
  working as a principle illustration rather than a training echo.
- **Branch-dynamic prompt dispatch at the section level.** The
  model never sees instructions for fields it isn't receiving. This
  generalizes the Step 1 lesson (separate class for
  `AlternativeIntent` vs `CreativeSpin`) from the schema level to
  the prompt-section level. Cost: three section dicts and a
  dispatch function. Benefit: eliminates cross-branch prompt
  dilution and self-classification.
- **Free-form trace fields carrying structured verdicts.** The
  `unit_analysis` and `slot_analysis` fields are free-form strings,
  but their tight micro-formats make them function as ordered lists
  of structured verdicts. This avoided introducing new schema types
  (verdict enums, per-atom structured records) while preserving the
  per-item commitment pattern.
- **Partition-completeness validator catches silent drops.** "Every
  inventory entry must appear in some slot scope" is exactly the
  math the prompt can ask for but cannot enforce. The Pydantic
  validator closes the gap and fails loudly when a committed phrase
  falls on the floor.
- **The Step 1 per-item-verdict pattern transferred cleanly.**
  `unit_analysis` uses the same `<item> -> <verdict> (<why>)`
  scaffold as Step 1's `ambiguity_analysis`. No new prompt-authoring
  pattern was needed — the scaffold from Step 1 became a reusable
  form.
- **Empirical probing → principle extraction, not point fixes.**
  When probing surfaced failures, the reliable fix was to pause,
  identify the general pattern the failure represented, and update
  the prompt's principles. Point-fixing one observed instance
  consistently missed the deeper ambiguity the next probe would
  surface.

### General Principles Going Forward

- When a field requires both expansion and grouping, force expansion
  as a prior pass. Collapsing them into one verdict causes
  under-expansion.
- Prefer verdict unification over pre-classification. "One verdict
  with variable cardinality" beats "two verdicts the model picks
  between" when the distinction is really just output count.
- Enforce single-family-per-atom by micro-format plus explicit rule.
  The prompt names the constraint; the micro-format enforces it.
- Tune capability descriptions to the level that prevents the
  hallucinations actually observed. Short generic descriptions are
  unsafe for families with internal structure.
- Draw boundary examples from a pool disjoint from evaluation
  queries. Principle-illustrating examples generalize; test-query
  examples teach pattern-matching.
- Fusion criteria need a sub-dimension clause when the family has
  internal structure. Two-condition rules leave too much to the
  model's discretion.
- "If in doubt, split" as the explicit fusion default.
- Observed inconsistency across runs is under-specification, not
  variance. Tighten the rule.
- Revert before fix. Pause after a failure to identify the general
  pattern before applying a targeted change.
- Partition-invariant validators catch silent drops that prompt
  discipline cannot. Where coverage matters, enforce in code.

## Implications For Step 2A and 2B

The patterns that fixed Stage 1 transfer directly to Stage 2A and 2B.
These are hypotheses, not validated results — but they are
hypotheses backed by concrete pattern-matching, not just good vibes.

Note: Step 2A has since been rewritten in line with most of these
hypotheses (see "What We Learned From The Step 2A Rewrite" above).
The Step 2A subsection below is retained for historical continuity
— it shows the through-line from Step 1's patterns to the Step 2A
implementation. The Step 2B subsection remains forward-looking.

### For Step 2A (ingredient inventory + concept extraction)

- **Per-item verdict pattern for the ingredient inventory.** Instead
  of a flat list of "ingredients we extracted," structure as
  `<candidate ingredient> -> actionable | filler | skip (reason)`.
  Forces the model to evaluate each candidate against an explicit
  criterion rather than passing everything through. Directly
  addresses the "preserves ingredients that don't help retrieval"
  failure.
- **Concrete grounding per ingredient.** Each ingredient must cite
  the specific phrase from the rewrite it derives from. Prevents the
  "treats paraphrased rewrite content as user-backed ingredients"
  failure — the model can't substitute when it has to point at a
  source phrase.
- **Vagueness-vs-ambiguity at the concept level.** A rewrite phrase
  may be semantically vague (one concept, fuzzy edges → one
  ingredient) or contain genuinely distinct ideas that should be
  split (two ingredients). Naming this distinction explicitly,
  with the same behavioral test ("would these retrieve different
  movies?"), should help the "merges clearly distinct ideas" failure.
- **Worked failure → fix examples for the merge failure.** Show
  `"animated and live-action films"` being incorrectly merged into
  one concept, then fixed by either splitting or recognizing the
  pair as nearly exhaustive (and therefore filler).
- **A "skip" verdict on candidate concepts.** Lets Step 2A identify
  a candidate concept that doesn't earn its keep and reject it
  explicitly, rather than forcing it through to Step 2B.

### For Step 2B (expression planning)

- **A "skip" verdict on the concept itself.** Step 2B's existing
  problem with "trusting bad concepts" parallels Stage 1's old
  problem of taking the trace at face value. Give Step 2B explicit
  permission to challenge whether a concept can be coherently
  expressed and emit zero expressions for incoherent inputs, rather
  than force-execute and produce nonsense expressions.
- **Coverage validation as a downstream field, not an upstream
  rule.** Field-order scaffolding: put expression generation BEFORE
  coverage audit in the structured output. The model commits to
  expressions before auditing them, which is the correct order for
  catching coverage gaps. (This may already be the case — worth
  checking.)
- **The "stay inside the user's intent" principle from creative
  spins applies.** Step 2B's expressions should not drift to
  retrieval angles that change what Step 2A's concept meant. A spin-
  style "if your expression no longer matches the concept's required
  ingredients, you've drifted" rule with a worked drift example
  would help.
- **Same brevity discipline.** Reasoning fields in Step 2B should
  have explicit word caps. The pattern "small models inflate
  reasoning when given an open invitation" applies wherever there is
  an open-ended trace field.

## Working Hypothesis Going Forward

- Stage 1 is now in good shape: small, faithful, evidence-driven, with
  a separate creative-spins lane for broad-intent exploration that
  doesn't pollute `alternative_intents` semantics. The remaining
  watch-items are coexistence cluttering (when alts already exist),
  spin emission on semantic-vagueness queries (currently fires; may
  or may not match user expectations in production), and any
  re-emergence of evaluative-word substitution.
- Stage 2A has been rewritten and is producing clean output on the
  feedback-query probe. The per-item verdict scaffold, the
  `interpret` verdict with variable cardinality, the three-condition
  fusion rule, and the capability-language endpoint descriptions
  collectively resolved the observed failure modes. Watch-items:
  continued drift toward over-fusion on same-family-different-
  sub-dimension cases (mitigated by the "if in doubt, split" rule
  but not eliminated), and occasional interpret firing on borderline
  descriptive language (the boundary between "descriptive" and
  "idiomatic" is fuzzy for some phrases).
- Step 2B is now the highest-leverage next target. Its "trusting
  bad concepts" problem parallels Stage 1's old "taking the trace at
  face value" problem — the fix there was per-item verdicts with a
  skip option. Step 2B's input shape has also changed (it now
  receives `PlanningSlot` from 2A, not `ExtractedConcept`), which
  forces a rewrite anyway. When rewriting, adopt: the per-slot
  verdict pattern (plan / skip), an explicit "stay inside the slot's
  retrieval_shape" rule with a drift example, and the single-family
  discipline (each expression targets one family, which is already
  implicit in the `EndpointRoute` enum but should be stated
  explicitly in the prompt).
- The broader path forward continues to tighten responsibilities by
  stage: Stage 1 routes and lightly rewrites (done), Stage 2A
  partitions into single-family slots (done), and Stage 2B plans
  retrieval expressions inside each slot while retaining the right
  to refuse a slot that cannot be coherently expressed.
- Cross-stage pattern that has emerged: each stage's prompt gets a
  per-item verdict scaffold, a "skip / reject" verdict as a
  first-class option, concrete grounding requirements (cite the
  upstream artifact), principle-based constraints with one or two
  boundary examples, and explicit brevity caps. This is now the
  house style for search-pipeline prompts.

## Step 2B Redesign Proposal

This section is the consolidated conceptual proposal for rewriting
Step 2B from scratch. No concrete schema yet — that comes after
edge cases are fully mapped. The proposal captures role, inputs,
outputs, reasoning flow, principles, and execution notes.

### Role in one line

Given one Stage 2A slot, produce the set of concrete retrieval
actions that together execute that slot's intent — or refuse the
slot — with enough per-action metadata for Stage 3 to generate
the endpoint query and enough grouping metadata for Stage 4 to
combine results correctly.

### Execution shape

One LLM call per Stage 2A slot, run in parallel. A slot-count of 1
naturally collapses to one call. Failure of any single call is
isolated and retryable. Parallelism wins on less noise per call,
wall-clock latency (max-of-N not sum-of-N), and failure isolation.

Per-family prompt specialization via branch-dynamic dispatch was
considered and rejected: it would architecturally commit to 2A's
family choice at the dispatch level, which contradicts giving 2B
rerouting autonomy. One prompt covers all family capabilities.
The ~25-line cost is acceptable for preserving that autonomy.

### Inputs per call

**Anchoring context from Stage 1:**
- The user's rewritten query (`intent_rewrite`). Strongest grounding
  signal. Prevents drift into adjacent angles and lets each action
  resolve ambiguous words against the full query intent.

**The focal slot from Stage 2A, in full:**
- handle, scope atoms, retrieval_shape, cohesion, confidence
  (literal / inferred). Every field earns its keep. `retrieval_shape`
  is the load-bearing instruction for what the slot's calls should
  retrieve; `cohesion` prevents the model from splitting the group
  back apart; `confidence` is one signal among several for the
  hard-vs-soft decision.

**Sibling slots from Stage 2A, in compact form:**
- Per sibling: handle, retrieval_shape, scope. No cohesion, no
  confidence, no verdicts. Gives coverage awareness (so 2B doesn't
  duplicate work another slot handles) without reopening 2A's
  partitioning.

**Endpoint capabilities (prompt-resident):**
- Per-family capability descriptions at Stage 2A resolution:
  sub-dimension bullets plus explicit "CANNOT do X" notes. All
  families present in every call — 2B needs the full picture to
  exercise rerouting autonomy.

**Deliberately NOT passed:**
- Stage 2A's `unit_analysis` and `slot_analysis` traces. 2B trusts
  committed output or skips; it does not re-debate atom extraction
  or slot fusion. Passing predecessor reasoning invites re-litigation.
- Stage 2A's full `inventory` (already covered by sibling scopes).
- Stage 1's `ambiguity_analysis`, `query_traits`, `routing_signals`,
  `flow`, `display_phrase`, `title`, `alternative_intents`,
  `creative_spin_analysis`, `creative_alternatives`. Wrong altitude
  or separate pipelines.
- Any structural `family` field or suggested route. 2A's family
  choice reaches 2B only via the natural-language retrieval_shape —
  advisory, not architectural. This is the key "2A is context, not
  truth" commitment at the input level.

### Outputs

**Plan-level for the slot:**
- A sibling-group identity — implicitly "the slot." All actions
  produced by the call are siblings, i.e. alternative or
  complementary framings of the same user requirement. Stage 4
  uses this for MAX-within, additive-across combination.
- First-class skip outcome: zero actions plus a reason. Used when
  the slot cannot be coherently expressed or when 2A's grouping is
  judged unrecoverable.

**Per action:**
- Self-contained trait description. Must stand alone — a Stage 3
  generator that never saw the slot should be able to produce the
  right query body from the description plus the user's rewrite.
  Descriptions written in "remember the slot" shorthand silently
  degrade Stage 3 quality.
- Endpoint / family targeted.
- Role: hard filter (reshapes the candidate pool), soft scorer
  (ranks within the pool), or exclusion. This is the single most
  important downstream-facing decision.
- Strength modifier: for scorers, a core-vs-supporting tier; for
  filters, include-vs-exclude direction.
- Coverage grounding: which atoms from the slot's scope this action
  covers. Enables a code-side partition-completeness validator
  (every atom covered by at least one action, or the whole slot
  skipped). Same math as 2A's validator, one level deeper.

### How the model should think

One compact trace field per slot — a per-atom verdict scaffold
that commits the key decisions before any action is written.
Same structural pattern as Stage 1's `ambiguity_analysis` and
Stage 2A's `unit_analysis` / `slot_analysis`. For each atom in
the slot's scope, one line, tight format, explicit word cap
(target ≤15 words), committing three decisions:

1. **Coverage & expansion verdict.** Is one action enough to cover
   this atom cleanly, or does it warrant multiple? If multiple,
   which expansion motive applies:
   - **Ambiguity fan-out** — atom spans multiple distinct angles
     (`"biggest"` → budget / box-office / popularity)
   - **Paraphrase redundancy** — same target, multiple lexical
     framings (`"christmas OR holiday keyword"`)
   - **Defensive-retrieval expansion** — same requirement, multiple
     endpoints because each data source has gaps (`"Indiana Jones"`
     → franchise + character; `"scariest"` → horror filter +
     scariest preference)
   The reason must name a specific mechanism, not a vague appeal
   to thoroughness.
2. **Role verdict.** Is this atom a hard filter, a scorer, or an
   exclusion? Grounded in user phrasing strength, what the atom is,
   the slot's confidence, and how it fits the overall intent.
   Confidence is one input among several, not a gate — an inferred
   atom can still be a dealbreaker if the evidence supports it.
3. **Route commitment.** Which endpoint(s) will handle it. If
   rerouting away from 2A's advisory shape, a one-phrase reason
   tied to endpoint capability or a "CANNOT do X" note.

Structured-output generation order then produces the action list,
mechanically aligned with the trace by construction. A skip verdict
is a first-class option at the trace level — if every atom resolves
to "no coherent expression," the model emits zero actions and
states the reason once.

Brevity discipline: explicit word caps on every trace line. Fields
that justify decisions must cite user-text atoms or named endpoint
capabilities — not category labels.

### Key principles the design must enforce

1. **2A is context, not truth.** 2B may reroute, refuse, or expand
   beyond 2A's implied shape. Nothing in the input or prompt should
   architecturally commit to 2A's family choice.
2. **Concept = slot.** Each 2B call produces exactly one sibling
   group. If 2B would want to AND two actions as separate
   requirements, they should have been separate slots; the correct
   response to this mismatch is skip or partial-cover, not silently
   splitting the concept. Keeps Stage 4's MAX-within /
   additive-across math honest.
3. **No intersection combination mode needed.** AND across
   requirements is expressed by separate concepts, not a combination
   flag. Within one concept, all actions are framings of one
   requirement — MAX for filters, weighted sum for scorers — and
   that is sufficient.
4. **Expansion is a decision, not a reflex.** The three expansion
   motives each require a specific, named justification. The prompt
   must actively resist "more calls = more thorough." Reason fields
   must justify robustness appeals with a specific failure mode
   (known endpoint gap, known ambiguity, known noise risk), not a
   vague appeal to safety.
5. **Kind-layering is allowed within one concept.** A hard filter
   narrowing the pool plus a preference ranking within it (horror
   filter + scariest preference) is a legitimate, often superior
   pattern. The two kinds occupy different combination lanes in
   Stage 4 and don't conflict.
6. **Confidence is one signal, not a gate.** Inferred atoms are not
   barred from becoming dealbreakers — they're weighed against
   other evidence.
7. **Cross-slot consolidation is a code concern.** If 2A over-split
   (e.g., two semantic slots that should have been one combined
   preference string), post-2B code merges them before dispatch.
   2B stays a per-slot transform; it never reasons across slots.
8. **Partition-completeness at the action level.** Every atom in
   the slot's scope must be covered by at least one action, or the
   slot must be skipped as a whole. Enforce in a code-side
   validator. This is the same math as 2A's validator, one level
   deeper.
9. **Self-contained action descriptions.** Stage 3 must be able to
   produce a correct query body from the description plus the
   user's rewrite, without needing slot context.

### Execution notes

- **Parallelism shape:** N slots → N parallel LLM calls, each small
  and focused. Natural collapse to single call when N=1.
- **No structural `family` field on `PlanningSlot`.** Family stays
  advisory via retrieval_shape. Preserves rerouting autonomy at
  the architectural level.
- **One prompt, all family capabilities.** No branch-dynamic
  dispatch at the 2B level. Slightly larger prompt is the cost of
  rerouting autonomy.
- **Concept identity for Stage 4 is the slot handle.** Downstream
  grouping uses the slot as the concept key; no additional
  concept-id generation needed.
- **Reasoning discipline reuses established scaffolds.** Per-atom
  verdict trace, brevity caps, skip-as-first-class, grounding
  citations, principle-illustrating examples drawn from a disjoint
  pool — the house style established across Stages 1 and 2A.
- **Code-side validators:** partition-completeness (every atom
  covered or slot skipped), action-description non-emptiness, no
  duplicate endpoint+description pairs within a sibling group.
- **Code-side post-processing:** cross-slot semantic preference
  merging (comma-join into one semantic string before dispatch),
  deduping identical calls across slots, final concept-grouped
  packaging for Stage 3 dispatch.

### Open design questions to resolve before schema commit

- Precise format of the per-atom verdict trace line (target
  micro-format + word cap).
- Whether the skip verdict lives in the trace itself or as a
  separate outcome field — Stage 1 / 2A precedent suggests in-trace
  with structured-output genertion following, but 2B's shape may
  differ.
- Exact vocabulary for action roles (dealbreaker/preference/
  exclusion vs more granular terms) and strength tiers.
- Boundary examples drawn from a disjoint pool, covering: each
  expansion motive, kind-layering, rerouting away from 2A's shape,
  and a skip-the-slot case.
- Whether cross-family capability descriptions should be
  prioritized by some heuristic (e.g., `retrieval_shape` keyword
  match) or always presented in fixed order.
