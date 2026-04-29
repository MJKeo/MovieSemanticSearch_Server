# v3 Query Understanding Rethinking

> Originally titled "v3 Step 2 Rethinking" — what was scoped as a single
> Step 2 redesign has decomposed into a three-step query-understanding
> pipeline (Step 2 → Step 3 → Step 4). Title kept for historical
> continuity; the work below covers all three steps.

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

The query-understanding side now decomposes into three focused steps:

| # | Step | LLM calls | Output |
|---|------|-----------|--------|
| 2 | Query analysis + commit | 1 | `atoms` (analysis layer) + `traits` (committed layer) |
| 3 | Per-trait category-call generation | 1 per trait, parallel | List of category calls per trait |
| 4 | Per-endpoint query building | 1 per (category call, endpoint), parallel | Endpoint-specific structured queries |

Cognitive sequence: holistic understanding + atomization + commit
inside one Step 2 call (whole-query context); per-trait
categorization with parametric resolution baked in (Step 3); per-
endpoint structured-query generation (Step 4).

**Naming note.** Step 4 in this doc reuses the existing
`search_v2/stage_3/` endpoint generators (with light input-adapter
revision). The numerical overlap between "Step 4" (this doc, front-
end) and `stage_3/` (codebase, back-end) is unfortunate but
preserved — the codebase nomenclature reflects history; the doc
uses Step-N for the new query-understanding pipeline.

## Step 2: Query analysis + commit (single LLM call, two output layers)

The same LLM call produces three things in coupled layers:

1. `holistic_read`: faithful prose read of the query.
2. `atoms`: the **analysis layer** — descriptive, per-criterion
   atoms with `surface_text`, `modifying_signals`,
   `evaluative_intent`, and uncertainty signals
   (`split_exploration`, `standalone_check`).
3. `traits`: the **committed layer** — the search-ready units that
   flow downstream. Resolves splits, dedupes, and commits role,
   polarity, and salience.

Both layers ship from one LLM call so they share whole-query context
without a re-read. The deliberate name distinction (**atoms vs
traits**) forces the model to treat them as different shapes —
atoms are analysis units, traits are downstream commitments. Reusing
"atoms" for both layers would let the model treat the second list as
a copy of the first.

### Holistic read (the prose output)

**Purpose.** A faithful prose description of what the user is asking
for, in their own words, with the relationships between pieces of
that ask described only as far as the user themselves implied.

**Surface.** `holistic_read: str` on `QueryAnalysis`. The field
description is the only documentation the LLM gets — its own micro-
prompt with NEVER and DO sections.

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

### Atomization (the analysis layer)

**Purpose.** Convert the user's evaluation intent into descriptive
units: per-criterion atoms with the modifying signals that shape
each one and a consolidated statement of what evaluating the
criterion actually means.

**Surface.** `atoms: list[Atom]` on `QueryAnalysis`. Each `Atom`
carries:

- `surface_text` — verbatim user words for this criterion (modifying
  language stripped).
- `modifying_signals: list[ModifyingSignal]` — every signal in the
  query that shapes how this criterion is evaluated; each entry has
  `surface_phrase` + freeform `effect`.
- `evaluative_intent` — 1–2 sentence consolidated semantic statement.
- `split_exploration: str` — **always populated** evidence
  field. Walk through whether this atom could be subdivided into
  smaller pieces, each retrievable independently. For each
  plausible subdivision, describe what each piece would retrieve
  on its own and whether the combined retrieval captures the
  user's intent at this atom's granularity. Pure analysis — no
  "split" / "keep whole" verdict in the field; the commit phase
  reads and decides.
- `standalone_check: str` — **always populated** evidence field.
  Compare this atom's `evaluative_intent` against the user's
  articulated ask in `holistic_read`. Describe HOW (not if)
  retrieving this atom standalone would relate to the user's
  articulated intent for its part of the query: what population
  it would return; whether that population matches a
  user-articulated standalone-able criterion or shifts the
  meaning (introduces a hard requirement the user didn't ask for,
  loses a coupling the user did imply, narrows what the user
  kept loose); whether context the atom's intent integrates from
  another atom survives standalone or falls away. Pure analysis
  — no "redundant" / "not redundant" verdict; the commit phase
  reads and decides whether to merge. Forbidden short-circuits:
  uniqueness checks ("first mention", "primary subject", "no
  other atom captures this"), independent-retrievability-as-
  virtue, "while [coupling acknowledged] but [standalone value]"
  patterns.

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
nullable `depth` field never actually got the null path (every edge
in the run got a value); the SHALLOW/DEEP binary collapsed at least
five distinct relationship shapes (subset filter, context reframe,
axis stack, counterfactual transposition, style transfer). All four
of those are shape problems, not prompt-tuning problems.

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
the signals are catalogued, the LLM writes a 1–2 sentence prose
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
salience / role happens in the commit phase below; category
assignment happens at Step 3.

### Commit phase (the traits layer)

**Purpose.** Produce the final list of search-ready units (traits)
that Step 3 consumes. Three jobs in one pass:

1. **Act on `split_exploration` analyses** — every atom carries an
   exploration of plausible subdivisions. Apply the
   searchable-unit test using the analysis as evidence: if the
   analysis lays out pieces whose independent retrievals would
   combine to reconstruct the user's intent at this atom's
   granularity, emit each piece as a trait. Otherwise keep whole.
2. **Act on `standalone_check` analyses** — every atom carries an
   analysis of how its standalone retrieval relates to the user's
   articulated intent. If the analysis describes a meaning shift
   the user did not articulate (a hard requirement they didn't
   ask for, a coupling lost when read alone, a narrowing of
   loose intent), find the atom whose `evaluative_intent` already
   integrates this atom's content and merge: the surviving trait
   is the integrating atom (its evaluative_intent already
   absorbs the coupled atom's content). The coupled atom does
   NOT survive separately. If the standalone reading matches a
   user-articulated standalone-able criterion (including
   exclusions tied to a peer pool via negative polarity), keep
   as own trait.
3. **Commit per-trait fields** — assign role (carver | qualifier),
   polarity (positive | negative), `relevance_to_query` reasoning,
   and salience (central | supporting) for each trait. Salience
   commits as the natural conclusion of relevance_to_query.

**Surface.** `traits: list[Trait]` on `QueryAnalysis`. Each `Trait`
carries:

- `surface_text` — from the source atom (post-split if applicable).
- `evaluative_intent` — from the source atom, lightly edited if
  split or dedupe changed the unit.
- `role: Literal["carver", "qualifier"]` — carver if the trait
  defines the result pool; qualifier if it positions a reference
  (comparison anchor, transposition operator) shaping how other
  traits are evaluated.
- `polarity: Literal["positive", "negative"]` — committed directly
  from effect tokens (FLIPS POLARITY → negative; otherwise positive).
- `relevance_to_query: str` — reasoning step before salience.
  1–2 sentences walking through how the trait sits in the query as
  a whole: hedges or intensifiers on the source atom, position in
  surface order, words spent, whether removing it would
  meaningfully change the ask. Modal effect tokens are one signal
  but not the whole picture — within-query position and structural
  prominence contribute too.
- `salience: Literal["central", "supporting"]` — natural conclusion
  of `relevance_to_query`. CENTRAL = headline want; SUPPORTING =
  rounds out an already-defined ask. Applies to every trait
  regardless of role; a non-central carver acts as a lenient filter
  (downstream code reads salience and adjusts).

**Why the commit phase is in the same call.** The commit reads the
atoms layer as input. A separate LLM call would re-read the same
context for marginal benefit. One call with two output layers shares
context efficiently; the renaming (atoms vs traits) prevents the
model from treating them identically.

**Why polarity is committed here, not deferred.** Effect tokens on
`modifying_signals` already encode polarity. The LLM has the full
atom in hand at commit time. Committing directly removes a
downstream translation step. Polarity is atom-local in principle but
rides in the commit phase because the LLM is already consolidating
the atom's signals there — nearly free.

**Why role and salience need the commit phase, not per-atom
generation.** Carver vs qualifier depends on what other criteria
define the result pool. Salience is relative across the trait list
(read holistically via the `relevance_to_query` reasoning step).
Both fail when assigned per-atom in isolation. The commit phase has
the whole atoms list in front of it.

**Salience is consumed programmatically post-Step-2, not by
Step 3.** Step 3 reads role, polarity, surface_text,
evaluative_intent — the LLM-input contract. `relevance_to_query`
and `salience` are trait-level metadata for code to act on
between Step 2 and Step 3 (e.g. lenient-filter handling for
non-central carvers; preference-score weighting). This loosens
the Step 3 contract: the LLM doesn't need to decide whether to
treat a non-central carver differently — that's code's job.

**What the commit phase does NOT do.**
- No category assignment — Step 3.
- No parametric resolution — Step 3.
- No endpoint routing or query building — Steps 3 and 4.

The commit phase resolves uncertainty flags and assigns the per-trait
properties Step 3 needs as inputs. Nothing else.

## Step 3: Per-trait category-call generation

**Purpose.** For each trait, generate the minimum set of category
calls whose combined retrieval captures the trait's intent.

**Surface.** Per-trait LLM call, fanned out in parallel. Output is
`list[CategoryCall]` per trait. **Output schema deferred** — the
concrete shape is informed by what Step 4's existing endpoint
generators expect as input. Don't sketch the schema until Step 2's
commit phase is built and producing actual traits to design against.

**Each `CategoryCall` carries (sketch, not committed):**
- A category from the taxonomy.
- A sub-atom text/intent describing what to retrieve against under
  that category.
- Enough context for Step 4 to build per-endpoint queries from.

**Combining categorization + parametric resolution in one call.**
Categorization and resolution are the same cognitive move — deciding
"this is a comparison reference" and "interstellar means emotional
sci-fi + awe scope + family-relational core" are not separable.
Splitting them adds an LLM hop without buying anything since the
category decision already encodes "does this need resolution."

**Concrete examples of trait → category-call decomposition:**

- A concrete genre trait → single call to a genre category.
- A concrete person-credit trait → single call to a person category.
- A concrete era trait → single call to an era category.
- A comparison-anchor trait ("like X") → multiple category calls
  capturing X's defining attributes (tone, scope, genre, register).
- A figurative-anchor trait ("warm hug movie") → multiple category
  calls capturing the implied attributes.
- An out-of-context creator trait ("X does horror" where X isn't
  known for horror) → multiple category calls capturing X's
  signature plus the genre transposition.

**Polarity-agnostic.** Step 3 generates retrieval shapes; polarity
flips scoring direction at merge time. The convention "all endpoint
specs express presence of an attribute, not absence" already requires
this — Step 3 staying polarity-agnostic keeps the convention
consistent. The trait carries polarity from Step 2; Step 4 / merge
applies it.

**Aggregation rule (within a trait).** Unweighted sum across category
calls. Within a single category that fans to multiple endpoints,
max-pool — but that's handled inside the category, not by Step 3.
Unweighted is the simplest aggregation that could work; revisit if
eval shows over-decomposition.

**The minimum-set discipline.** Most traits resolve to one category
call. Parametric traits decompose into a few calls — but "few" is
the operating principle. Padding the list adds noise to the score
sum. Prompt the LLM to "generate the smallest set of category calls
whose combined retrieval would identify movies matching this trait's
intent. If a single call captures it, that's the answer — don't pad."

## Step 4: Per-endpoint query building

**Purpose.** For each (category call, endpoint) pair, generate the
endpoint-specific structured query.

**Surface.** Reuses existing `search_v2/stage_3/` endpoint
generators. Each endpoint gets its own LLM call with that endpoint's
schema. Mechanical-ish given concrete category calls — most of the
semantic load was paid in Step 3.

**Why per-endpoint.** Endpoint schemas are too large to bundle. Each
endpoint has its own field shape (entity, metadata, award, franchise,
studio, semantic, trending, media-type). A single LLM doing all of
them at once would have a bloated prompt and weak per-endpoint
reasoning.

**Why this stage stays small.** Step 3 already committed the
category and the sub-atom text/intent. Step 4 just translates that
into the endpoint's structured shape.

**Light fine-tuning needed.** The existing endpoint generators were
built against the old stage_2a/2b inputs. The Step 3 output shape
(once committed) will be different, so the input adapters on each
endpoint generator need light revision. The structural work of the
generators themselves — schema, prompt, retry semantics — is reused
unchanged.

## Build order

1. **Step 2 commit phase.** ✅ Landed. Atoms layer (descriptive)
   + traits layer (committed) emitted from one LLM call.
   Exploration fields (`split_exploration`, `standalone_check`)
   always populated with analysis (no embedded verdicts);
   commit phase reads them and decides. `relevance_to_query`
   reasoning before salience.
2. **Step 3 output schema next.** Design the schema based on
   (a) what Step 2 actually emits as traits, (b) what the existing
   endpoint generators expect as input. The Step 3 schema is the
   contract that bridges traits → endpoint generators. Don't
   speculatively design it before traits are real.
3. **Step 2 → Step 3 code layer.** The salience-driven preference /
   lenient-filter logic lives here (not in Step 3's LLM prompt).
   Reads `relevance_to_query` + `salience` from each trait and
   adjusts how the trait flows into Step 3 / merge.
4. **Step 3 prompt + Step 4 fine-tuning together.** Once the schema
   is committed, design the Step 3 prompt and adjust the endpoint
   generator input adapters in tandem.

## Outstanding changes for Step 2 (round 3)

Round 1 (commit-phase shape) and round 2 (prompt + schema alignment
to LLM-handling principles) are landed. Round 3 was a discipline-gate
+ holistic-salience refinement informed by 34-query re-runs. All
three rounds are now in. Items below are the iteration targets that
remain visible in the test set:

1. **Coupled-pair atomization** (comedians taking on serious roles,
   Q26 BB+1800s, Q27 succession+pirates). The standalone_check
   exploration fires on every atom (always-populated analysis),
   but the model can still degenerate the description into
   uniqueness checks and independent-retrievability appeals.
   Round-4 changes target this: the redundancy gate is replaced
   by the user-intent-comparison framing ("how does standalone
   retrieval relate to user-articulated intent?") and verdicts
   are stripped from the field so the commit phase makes the call
   on prose evidence rather than verifying claims.

2. **Q29 wes anderson does horror** is a role-assignment failure,
   not a coupling failure. Both atoms are correctly emitted; the
   wes anderson trait should commit role=qualifier (style
   reference) but commits carver. Separate fix on the role rule.

3. **Multi-anchor reference set** (Q18 inception interstellar
   tenet) → 3 separate title-search carvers. Per design decision,
   this is the desired output. No fix needed.

4. **Character-as-franchise disambiguation** (Q10 joker / phoenix
   one) and **meta-relation exclusion** (Q6 parody of godfather
   shouldn't surface Godfather itself) → Step 3 territory; needs
   the full taxonomy fitting machinery.

5. **Parametric resolution for scope phrases** ("the new ones",
   "the joaquin phoenix one") → Step 3 territory; set-intersection
   semantics already correct at Step 2.

Landed across rounds 3 and 4:

- `Atom.split_note` → `split_exploration: str`, required, pure
  evidence-gathering analysis with no embedded verdict.
- `Atom.redundancy_note` → `standalone_check: str`, required,
  reframed: compare atom's evaluative_intent against holistic_read
  and describe HOW standalone retrieval relates to user-articulated
  intent. NEVER list explicitly closes the verdict-first,
  uniqueness-check, independent-retrievability-as-virtue, and
  "while [coupling] but [standalone value]" rationalization
  patterns.
- `Trait.relevance_to_query: str` added before `salience`.
  Reasoning step that walks query holistically; salience drops
  out as natural conclusion.
- "Carvers don't get salience" rule lifted; salience applies to all
  traits regardless of role; non-central carver = lenient filter
  signal for code post-Step-2.
- `_ATOMICITY` SPLIT AND STANDALONE EXPLORATIONS section (renamed
  from SPLIT AND REDUNDANCY GATES); both fields framed as
  exploratory evidence rather than verdict gates.
- `_COMMIT_PHASE` ACT ON SPLIT EXPLORATIONS + ACT ON STANDALONE
  CHECKS replace verify-then-merge framings. Commit phase reads
  the exploration prose and makes structural decisions; the
  searchable-unit and user-intent-comparison tests apply at
  commit time, not atom time.
- `_SALIENCE` rewritten for holistic interpretation (modal tokens
  are one signal among several).
- `_TASK_FRAMING` updated to reflect atom phase = evidence
  (signals + intent + explorations) and commit phase = decisions
  (structural + per-trait commitments).

## Cross-cutting design choices

**1. The holistic read makes minimal structural commitments.**
Earlier drafts had it flagging kept-whole units, picking primary
anchors, applying relationship-type labels. Those bins forced
queries into shapes that didn't fit. The redesign strips structural
commitments from the read and pushes them to atomization where the
principle sections (atomicity, modifier vs trait, evaluative intent)
live.

**2. The atomization separates description from interpretation.**
`surface_text` and `modifying_signals` stay strictly descriptive;
`evaluative_intent` is the one designated place where the LLM
consolidates raw signals into per-criterion meaning. Even there,
downstream-shaped commitments (concrete polarity / salience numbers,
category labels, system vocabulary) are forbidden. `polarity_hint`,
`salience_hint`, `role_marker`, `AbsorbedModifierKind`, and
`ModificationDepth` were all rejected as prescriptive masquerading
as recording.

**3. The commit phase is where prescriptive translation lives.**
Role / polarity / salience are committed values, not hints — but
they live in a separate output layer (`traits`) from the descriptive
layer (`atoms`). The split keeps each layer's job tight: atoms
record, traits commit.

**4. Vocabulary is suggested where parsing depends on it, not
forced.** Modal effects (SOFTENS / HARDENS / FLIPS POLARITY /
CONTRASTS) remain the recommended phrasing for the cases they fit.
Polarity is a mechanical token-read of FLIPS POLARITY / negation
language. Salience reads modal tokens as one signal among several
via the `relevance_to_query` reasoning step (also accounting for
position, investment, structural prominence). The effect field is
freeform, not enum-enforced — the prior closed enum on modifier
kinds bucket-forced misclassifications. The rule now is: use the
controlled vocabulary when it fits, describe in plain words when
it doesn't.

**5. Category taxonomy lives at Step 3** (and only Step 3). Steps 2
and 4 don't load it — Step 2 is taxonomy-agnostic by design, and
Step 4 receives the category as part of its input.

**6. Each step is shorter than the previous monolith.** The combined
Step 2 call absorbs structural work, modal-effect extraction,
atomization, and commitment. Step 3 focuses tight on category
decomposition. Step 4 reuses existing endpoint generators.

**7. No specific test queries in any prompt or schema description.**
Eval hygiene — examples in prompts contaminate the small-LLM
evaluation pattern.

**8. Exploration before decision (no embedded verdicts).** Where
a commitment or structural decision depends on judgment rather
than mechanical token-mapping, the schema surfaces the analysis
as its own always-populated field with **no embedded verdict**.
The decision is made at a separate point that consumes the
exploration. Applies to:
- Salience: `relevance_to_query` (exploration) → `salience`
  (commitment).
- Splits: `split_exploration` (exploration) → commit-phase
  decision (split into traits or keep whole).
- Couplings: `standalone_check` (exploration) → commit-phase
  decision (merge into another atom or emit as own trait).
The verdict-laden gate shape ("redundant given X because [reason]"
+ "not redundant because [reason]") biased the model toward
committing first and rationalizing after. Pure exploration
fields force evidence-gathering before any verdict can latch in.
The commit phase makes the calls based on the analyses, applying
the same searchable-unit and user-intent tests that justify the
decisions in the first place.

**9. Salience consumed programmatically post-Step-2.** Step 3 is
fed `role`, `polarity`, `surface_text`, `evaluative_intent` —
the LLM-input contract. `relevance_to_query` and `salience` are
trait-level metadata for code to act on between Step 2 and
Step 3 (lenient-filter handling for non-central carvers,
preference-score weighting, etc.). This loosens the Step 3
contract: the LLM doesn't need to reason about salience, and the
preference-vs-filter distinction lives in code.

## Implications for the category taxonomy

These changes still need to happen:

- **Parametric categories collapse — but their work moves into
  Step 3, not into a separate resolution stage.** The trait that
  reaches Step 3 may be parametric (a comparison anchor, a figurative
  anchor); Step 3 resolves it as part of category-call generation.
  Categories like "stylistic signature transfer" or "creator out of
  context" can be removed — their work is absorbed into resolution
  inside Step 3.
- **Categories describe concrete attribute spaces only.** Genre, era,
  person credit, runtime, tonal vector, audience register, comparison
  anchor, etc. No category should encode "the LLM figures this out
  later."
- **Category fit becomes closer to mechanical routing.** Once Step 3
  is decomposing parametric traits into concrete category calls, "this
  call names a genre → genre category" is a much simpler decision
  than the current parametric-aware fitting.

The taxonomy revision and Step 3 schema design are coupled.

## Remaining challenges

These don't go away under the new pipeline structure, but they're
tractable:

- **Commit-phase complexity.** The Step 2 call now produces atoms
  AND traits in one shot. Worth measuring per-layer error rates if
  quality regressions appear. Mitigation available: split into
  Step 2a (analysis) + Step 2b (commit) if the combined call shows
  stress.

- **Commit-phase interpretive load.** Switching from
  verdict-shaped gates (`redundancy_note` with embedded
  "redundant given X" / "not redundant") to pure exploration
  fields (`standalone_check`) moves the structural decision from
  the atom phase to the commit phase. The commit phase now reads
  prose evidence and decides — more interpretive than verifying
  a structured claim. The risk is the commit phase rationalizing
  in the same direction the verdict-shaped gate did. Mitigation:
  the always-populated description-style exploration leaves a
  richer trace for the commit-phase prompt to act on, and the
  standalone_check NEVER list (no verdict, no uniqueness checks,
  no independent-retrievability-as-virtue, no "while [coupling]
  but [standalone value]") targets the specific rationalization
  patterns observed in the verdict-shaped runs. Worth measuring
  commit-phase merge accuracy on coupled-pair queries (comedians,
  Q26, Q27) once re-run.

- **Resolution quality and confidence.** Whether Step 3's parametric
  resolution of a creator's stylistic signatures is accurate is a
  separate axis from whether it correctly decided to resolve. Famous
  creators resolve reliably; obscure ones may resolve confidently
  but wrongly. Worth considering an explicit confidence signal with
  a documented fallback.

- **Resolution can multiply traits.** A resolved trait may decompose
  into multiple concrete category calls — a kept-whole hypothetical
  might emit 4–5 calls. Step 3's output schema needs to support
  this cleanly.

- **Literal-vs-parametric is context-dependent for the same phrase.**
  A name in a credit context routes literally; the same name in a
  stylistic-anchor context routes parametrically. Step 3 has access
  to traits + holistic_read, which carry that context.

- **Multi-layer parametrics.** Dense hypothetical mashups will
  resolve as a single layered description. The framework handles it
  without special-case machinery, but resolution quality may be
  uneven.

- **Step 3 over-decomposition.** Unweighted sum incentivizes more
  category calls (more = higher score ceiling). The "minimum set"
  prompt discipline is the first counterweight. If eval shows over-
  decomposition anyway, tighten the prompt before going to weighted
  sum.

- **Step 3 output schema depends on Step 4 input contracts.**
  Mismatched contracts force adapter logic in Step 4; large
  mismatches signal Step 3 is the wrong shape. Surface this risk
  during Step 3 schema design.

- **Holistic-read failure modes (named-work expansion, anchor-pattern
  hallucination, use-case interpretation).** The lead-with-constraints
  redesign of the holistic read addresses the most common variants
  but doesn't eliminate them. Easy override at the atomization level
  partially mitigates remaining cases. Worth targeted prompt work if
  any persist in production traffic.

- **Evaluative intent is the load-bearing semantic field.**
  Downstream quality is bottlenecked on the LLM's intent-rewrite
  quality. Some test queries (q18 multi-anchor, q4 audience-age
  leak) still produce intent statements that miss the right reading
  even when the surface_text + signals are correct. These are
  prompt-tuning issues now, not schema-shape issues — but they
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
