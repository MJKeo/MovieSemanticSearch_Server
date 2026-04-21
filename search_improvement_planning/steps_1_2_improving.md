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

## Implications For Step 2A and 2B

The patterns that fixed Stage 1 transfer directly to Stage 2A and 2B.
These are hypotheses, not validated results — but they are
hypotheses backed by concrete pattern-matching, not just good vibes.

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
- Step 2A remains the weakest link. The per-item verdict pattern with
  a "skip" verdict and concrete-grounding requirement is the
  highest-leverage candidate intervention — it fixed the analogous
  problem in Stage 1.
- Step 2B's "incoherent-concept challenge" need parallels Stage 1's
  "skip when not worth emitting" — adopt the same pattern. Step 2B
  is also a good candidate for the spin-style "stay inside the
  source intent" rule.
- The best path forward is to keep tightening responsibilities by
  stage: Stage 1 should not over-decompose (done), Stage 2A should
  identify only actionable concepts (apply the verdict pattern), and
  Stage 2B should preserve every required part of a good concept
  while being explicitly empowered to refuse bad ones (apply the
  skip verdict + intent-anchoring pattern).
