# v3 Query Understanding Rethinking

> Originally titled "v3 Step 2 Rethinking" — what was scoped as a single
> Step 2 redesign has decomposed into a multi-stage pipeline. Title kept
> for historical continuity; the work below covers all stages.

## The problem this is correcting

The original Step 2 schema treated trait boundaries as a property of a
span: "is this span atomic, or should it split further?" That's only
one direction of correction (split). It can't merge chunks that the
LLM has already separated, and it can't recognize when a piece
searches fine in isolation but loses information that decomposition
can't recover.

Concrete failure: "John Wick but with kids, not too long". The
original flow extracted three spans, split them into separate traits,
and routed each. "With kids" found a search home (e.g. "kid
protagonist" in viewer-experience semantic space) — but the home it
found didn't match the parent unit's intent ("a kid version of John
Wick"). The pieces individually searched; the combination didn't
reconstruct what the user asked for.

The deeper insight: most multi-piece queries DO decompose cleanly into
additive traits. The narrow failure case is when either:

1. Decomposing destroys information that additive combination can't
   recover (the unit must stay whole), OR
2. A unit's literal text routes to the wrong handler because the user
   is implicitly relying on world knowledge to translate the phrase
   (the unit needs parametric expansion before it can be searched).

These are two distinct operations at two different levels, both
absent from the original schema.

## The pipeline

The first two stages (the holistic read AND the atomization) are
produced together by a single LLM call into one combined output
object (`QueryAnalysis`). The remaining stages are separate calls that
consume that output.

| # | Stage | Purpose |
|---|-------|---------|
| 1+2 | Query analysis (single call) | Holistic prose read of the query AND per-criterion atoms — each with the user's words for the criterion, every signal that shapes its evaluation, and a consolidated `evaluative_intent` statement. surface_text + modifying_signals stay descriptive; evaluative_intent is the one place where light inference is permitted |
| 3 | Reconstruction test | For each atom flagged with `candidate_internal_split`, would additive combination of the pieces reconstruct the user's intent? Sanity-check committed atoms |
| 4 | Literal test | For each atom, would its literal text route correctly, or does it need parametric resolution? Folds modification context into endpoint inputs based on category |
| 5 | Trait commitment | Final trait list with category, role (carver/qualifier), polarity, salience |

Cognitive sequence: holistic understanding first, atomization next
(in the same call), with verification, resolution, and commitment as
later layers.

## Stage 1+2: Query analysis (single LLM call)

The holistic read and the atomization are produced together. The LLM
reads the query, produces a faithful prose description, and uses
that read as the substrate for a structured atomization. Both
outputs stay strictly descriptive — recording what's in the query
without committing to downstream interpretations.

### Holistic read (the prose output)

**Purpose.** A faithful prose description of what the user is asking
for, in their own words, with the relationships between pieces of
that ask described only as far as the user themselves implied.

**Surface.** `holistic_read: str` field on `QueryAnalysis`. The
field description is the only documentation the LLM gets — its own
micro-prompt with NEVER and DO sections.

**The NEVER constraints (lead-with-constraints design):**

- CATEGORIZE — no system-shaped labels ('genre', 'runtime', 'actor',
  'tone', etc.).
- EXPAND NAMED THINGS — names stay as written; don't unpack what
  they evoke.
- INFER beyond what the user said — no 'i.e.', 'such as', 'meaning',
  parentheticals.
- IMPOSE STRUCTURE THAT ISN'T THERE — no 'primary anchor', no 'kept
  whole', no 'hybrid', no 'cross' unless the user's own phrasing
  puts the relationship on the page.

**The DO list:**

1. List the wants the user named, in their exact phrasing.
2. For each piece of modal language, name its effect using a fixed
   vocabulary verbatim: SOFTENS, HARDENS, FLIPS POLARITY, CONTRASTS.
3. Describe how the wants relate, only as plainly as the user
   implied — no relationship vocabulary the user didn't use.

**Design principles applied:** minimum context / maximum looseness;
lead with constraints; constrain vocabulary where downstream parsing
depends on it; no specific test queries in the description (eval
hygiene).

### Atomization (the structured output)

**Purpose.** Convert the user's evaluation intent into structured
form: a list of atoms (one per evaluative criterion at the user's
granularity), each with the modifying signals that shape it and a
consolidated statement of what evaluating the criterion actually
means.

**Surface.** `atoms: list[Atom]` on `QueryAnalysis`. Each `Atom`
carries: `surface_text` (verbatim user words), `modifying_signals`
(unified list of every signal in the query shaping this criterion's
evaluation), `evaluative_intent` (consolidated semantic rewrite),
and `candidate_internal_split` (deferral channel for Stage 3).
The supporting type is `ModifyingSignal` with `surface_phrase` +
freeform `effect` string.

**An atom is one evaluative criterion at the user's granularity** —
the unit the user is asking the system to score movies against.
Genre, era, named entity, mood word, comparison anchor, scoping
frame — anything the user is asking the system to check.

A compound stays as one atom when the pieces aren't separately
evaluable (only the whole names what the user is judging on, and
splitting loses what they actually asked for). Otherwise, distinct
evaluative criteria → distinct atoms.

**Why we stopped representing modifications as a graph.** The
earlier shape used `absorbed_modifiers` for in-atom qualifiers and
`modified_by` (with positional indices and a SHALLOW/DEEP enum) for
cross-atom edges. Running the 34-query test set showed systematic
problems with that shape: a closed `kind` enum bucket-forced
modifiers that didn't fit; positional `modifier_atom_index`
required the LLM to count and dropped forward-pointing edges in
multi-atom queries (q11, q15, q24, q32 all had zero edges); a
nullable `depth` field never actually got the null path
(every edge in the run got a value); the SHALLOW/DEEP binary
collapsed at least five distinct relationship shapes (subset
filter, context reframe, axis stack, counterfactual transposition,
style transfer). All four of those are shape problems, not
prompt-tuning problems.

The replacement framing is per-criterion, not graph-shaped: for
each atom, the LLM walks the whole query and records every signal
that shapes how this criterion should be evaluated, regardless of
where in surface order the signal sits. surface_phrase + freeform
effect string carries each signal; no kind enum to bucket-force
into, no depth binary to collapse diversity, no positional pointer
to count. Both adjacent qualifiers and cross-criterion modifiers
land on the same `modifying_signals` list — they were always the
same thing conceptually (something-shaping-this-criterion's-meaning).

**`evaluative_intent` is the load-bearing semantic field.** Once
the signals are catalogued, the LLM writes a 1-2 sentence prose
statement of what evaluating the criterion actually means once the
context is integrated. This is the one place where light inference
is permitted; surface_text and modifying_signals stay descriptive.

The intent rewrite is what normalizes raw-layer LLM variance for
downstream — small atomization-shape differences (whether
"not too long" splits into one effect or two, whether the trailing
"movie" noun gets its own atom, etc.) flatten when downstream
consumes the intent rather than the raw modifier structure. It also
absorbs distinctions the schema doesn't carry explicitly:
watch-context vs content-criterion, suspicious role bindings,
meta-reception properties — the intent prose can flag any of these
naturally even though there's no structured tag for them.

**Strict guardrails on intent inference.** Even though inference is
allowed inside `evaluative_intent`, downstream-shaped commitments
are not: no category labels, no concrete polarity / salience
numbers, no expansion of named things, no system vocabulary
(channel / vector / endpoint names). Translation into polarity /
salience / role / category happens at Stage 5.

**What the atomization records:**
- Atom boundaries (what the units are).
- Modifying signals on each atom (surface_phrase + concise effect
  string for every modifier, regardless of where in the query it
  came from).
- Per-atom evaluative intent (consolidated meaning, plain prose).
- Candidate internal splits (only when genuinely uncertain about a
  boundary — the deferral channel for Stage 3).

**Trust posture toward the holistic read: easy override.** The
holistic read makes weak structural claims by design (no kept-whole
calls, no anchor labels, no relationship-type vocabulary). The
atomization can restructure freely based on the principle sections
(atomicity, modifier vs trait, evaluative intent) loaded in the
system prompt.

## Stage 3: Reconstruction test

**Purpose unchanged from the original two-tests framing.** For each
atom flagged with `candidate_internal_split`, reason through whether
additive combination of the pieces would reconstruct the unit's
compound meaning.

> *If I split here and search each piece independently, would the
> system's additive combination of those results reconstruct what the
> user originally asked for?*

If yes → split. If no → keep whole.

**Inputs:** atoms list (with candidate_internal_split fields),
holistic_read, original query.

**Reference target.** The holistic read is the explicit reference
for "what the user originally asked for." The query remains the
ultimate ground truth as a second-line check.

**Stage 3 has shrunk to a sanity check.** Most atoms are committed
confidently by Stage 1+2. Stage 3 verifies the atoms with
candidate_internal_split populated, and may sanity-check committed
atoms against the holistic read for false positives.

## Stage 4: Literal test

**Purpose unchanged.** For each atom, decide:

> *Would the literal text of this atom, routed straight to the
> system, return movies that match the user's want?*

If yes → use surface text as the trait phrase.
If no → resolve parametrically: emit a concrete attribute description
as the trait phrase instead.

**Inputs are richer.** Stage 1+2 has already done structural work
that Stage 4 leverages directly:

- Atom `surface_text` is the literal candidate.
- `evaluative_intent` is the consolidated semantic load — Stage 4's
  primary input for deciding literal-vs-parametric and for building
  reformulated retrieval text when needed.
- `modifying_signals` carries provenance: each entry's
  `surface_phrase` + `effect` describes what's shaping the atom
  (role bindings, range calibrations, comparison frames, period
  transpositions, style references, polarity setters, hedges).
  Stage 4 reads the effect strings to decide endpoint routing
  (credit-style binding → person endpoint; comparison reference →
  parametric resolution; transposition → reformulated semantic
  query; etc.).
- `holistic_read` provides whole-query context for resolving
  preserved ambiguity (which dimension does "not Y" target?).

**Folding modification context into endpoint inputs.** Per (atom,
endpoint) pair, Stage 4 decides whether to incorporate the modifier
context recorded in `modifying_signals`:

- Structural-fact endpoints (genre enum, runtime range, year filter,
  person credit lookup): typically ignore semantic reshaping; may
  honor hard filters carried by modifiers.
- Semantic vector endpoints: incorporate context-reshaping modifiers
  (counterfactual transpositions, style references, scoping
  modifiers) by building context-aware retrieval text. The
  evaluative_intent statement is the most direct input here —
  it already reads as a context-integrated description of the
  evaluation target.

This is where Stage 1+2's preserved ambiguity gets resolved — loose
terms ("not Y", "underrated") get pinned to dimensions based on
whole-query context.

## Stage 5: Trait commitment

**Mechanical translation from the modifying_signals effect strings
plus evaluative_intent.** The recommended modal vocabulary
(SOFTENS / HARDENS / FLIPS POLARITY / CONTRASTS) still rides on each
atom's signals when applicable; Stage 5 translates:

- effect contains SOFTENS → salience = supporting
- effect contains HARDENS → salience = central
- effect contains FLIPS POLARITY → polarity = negative
- effect contains CONTRASTS → handled per atom context

For non-modal effects (role bindings, transpositions, comparison
references, scope narrowings) the effect string and the
evaluative_intent together describe what kind of trait this is and
how it should be weighted. Stage 5 reads both; the intent statement
is the canonical source of truth when an effect string is loose.

**Role determination from modifying_signals + intent.** Atoms whose
intent positions them as references (comparison anchors,
transposition operators) tend to be qualifiers; atoms whose intent
defines the result pool tend to be carvers. Stage 5 applies the
carver-vs-qualifier rules loaded in its prompt.

**Category commitment is unchanged.** The only piece that still
needs the full taxonomy and a per-trait decision.

**Stage 5's two sub-tasks:**

1. Category commitment (full taxonomy, original scope of work)
2. Polarity / salience / role formalization (translation from raw
   signal carried on atoms)

## Cross-cutting changes

**1. The holistic read makes minimal structural commitments.**
Earlier drafts had it flagging kept-whole units, picking primary
anchors, applying relationship-type labels. Those bins forced
queries into shapes that didn't fit — over-marking kept-whole on
clean compositional queries, picking primary anchors on parallel-
filter queries. The redesign strips structural commitments from the
read and pushes them to atomization where the principle sections
(atomicity, modifier vs trait, evaluative intent) live.

**2. The atomization separates description from interpretation,
with one designated place for inference.** surface_text and
modifying_signals stay strictly descriptive — verbatim user words,
freeform effect strings describing what each signal does without
categorizing it. `evaluative_intent` is the one designated place
where the LLM consolidates raw signals into per-criterion meaning;
even there, downstream-shaped commitments (concrete polarity /
salience numbers, category labels, system vocabulary) are
forbidden. `polarity_hint`, `salience_hint`, `role_marker`,
`AbsorbedModifierKind`, and `ModificationDepth` were all rejected
during design as prescriptive masquerading as recording — Stage 5
translates from raw effect strings + intent prose instead.

**3. Vocabulary is suggested where parsing depends on it, not
forced.** Modal effects (SOFTENS / HARDENS / FLIPS POLARITY /
CONTRASTS) remain the recommended phrasing for the cases they fit,
and Stage 5's translation rules key off those tokens. But the
effect field is freeform, not enum-enforced — the prior closed
enum on modifier kinds bucket-forced misclassifications (`or`
recorded as FLIPS POLARITY, "the one where" as ROLE_BINDING). The
rule now is: use the controlled vocabulary when it fits, describe
in plain words when it doesn't.

**4. Category taxonomy lives at Stage 5** (and a lightweight version
at Stage 4). Earlier stages don't load it — major prompt-size win.

**5. Each later stage is shorter, not longer.** The combined Stage
1+2 call absorbs structural work, modal-effect extraction, and
atomization. Stages 3-5 focus tighter:
- Stage 3 on verification of uncertain atoms
- Stage 4 on the literal/parametric axis + modification context
  folding
- Stage 5 on category + formalization

**6. No specific test queries in any prompt or schema description.**
Eval hygiene — examples in prompts contaminate the small-LLM
evaluation pattern.

## Implications for the category taxonomy

These changes still need to happen:

- **Parametric categories collapse.** Resolution happens in Stage 4;
  the trait that reaches categorization is always concrete.
  Categories like "stylistic signature transfer" or "creator out of
  context" can be removed — their work is absorbed into resolution.
- **Categories describe concrete attribute spaces only.** Genre, era,
  person credit, runtime, tonal vector, audience register, comparison
  anchor, etc. No category should encode "the LLM figures this out
  later."
- **Category fit becomes closer to mechanical routing.** Once traits
  are concrete, "this trait names a genre → genre category" is a much
  simpler decision than the current parametric-aware fitting. The
  square-peg pressure on category assignment drops sharply.

The taxonomy revision and Stage 5 schema design are coupled and have
to be done together.

## Remaining challenges

These don't go away under the new pipeline structure, but they're
tractable:

- **False positives on the reconstruction test.** The LLM can think
  reconstruction works when it doesn't. Mitigation: reasoning-before-
  commit pattern, worked examples in both directions in the Stage 3
  prompt (cases that look like they reconstruct but don't, and vice
  versa).

- **Reconstruction is fuzzy at the edges.** Borderline cases like
  "Inception but for kids" — the reconstruction test won't eliminate
  judgment calls; it just gives the LLM a more concrete tool than
  abstract category fit.

- **Resolution quality and confidence.** Whether Stage 4's parametric
  resolution of a creator's stylistic signatures is accurate is a
  separate axis from whether it correctly decided to resolve. Famous
  creators resolve reliably; obscure ones may resolve confidently
  but wrongly. Worth considering an explicit confidence signal with
  a documented fallback.

- **Resolution can multiply traits.** A resolved unit may decompose
  into more traits than the original surface chunk had — a kept-whole
  hypothetical might resolve into multiple concrete attributes.
  Decomposition needs to be allowed AFTER resolution, not just on
  surface text. Stage 4's output schema needs to support this.

- **Literal-vs-parametric is context-dependent for the same phrase.**
  A name in a credit context routes literally; the same name in a
  stylistic-anchor context routes parametrically. Stage 4 has access
  to atoms + holistic_read, which carry that context.

- **Multi-layer parametrics.** Dense hypothetical mashups will fail
  reconstruction (kept whole), then fail literal test (no such
  movie), then resolve as a single layered description. The framework
  handles it without special-case machinery, but resolution quality
  may be uneven.

- **Holistic-read failure modes (named-work expansion, anchor-pattern
  hallucination, use-case interpretation).** The lead-with-constraints
  redesign of the holistic read addresses the most common variants
  but doesn't eliminate them. Easy override at the atomization level
  partially mitigates remaining cases. Worth targeted prompt work if
  any persist in production traffic.

- **Evaluative intent is the new bottleneck.** With the schema's
  load-bearing semantic field consolidated into one prose statement
  per atom, downstream quality is bottlenecked on the LLM's intent-
  rewrite quality. Some test queries (q18 multi-anchor, q4 audience-
  age leak) still produce intent statements that miss the right
  reading even when the surface_text + signals are correct. These
  are prompt-tuning issues now, not schema-shape issues — but they
  warrant ongoing eval pressure.

- **Query-level meta-properties have no structured home.**
  Constraint-only queries with no positive ask (q23, q24, q31),
  multi-anchor reference sets (q18), meta-reception figurative
  queries (q33, q34), and suspicious role bindings (q28) are global
  query shapes that no per-atom field can carry without distortion.
  The `evaluative_intent` prose can mention them when relevant, but
  downstream consumes prose for these signals. Worth deciding
  later whether to add structured query-level annotations or accept
  prose-carried signals.

- **Eval coverage.** Test queries should deliberately include:
  parallel-filter queries with no anchor, multi-anchor sets,
  use-case scoping, pure tonal/mood, negation-heavy, mixed
  positive/negative carvers, counterfactuals, person-as-credit vs
  person-as-style, hedged queries, dense queries with 5+ wants,
  loose figurative language.

- **Combined Stage 1+2 call: prompt complexity.** Earlier drafts
  had Stage 1 and Stage 2 as separate calls. Combining them lowers
  call count and lets the LLM keep context coherent across the read
  and the atomization, but the prompt must support both jobs and
  the LLM does more in one shot. Worth measuring per-stage error
  rates against a hypothetical split-call baseline if quality
  regressions appear.
