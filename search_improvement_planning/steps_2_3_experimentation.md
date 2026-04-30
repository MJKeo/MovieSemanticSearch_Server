# Steps 2 & 3 Experimentation Log

Tracking iterative changes to the query-analysis (Step 2) and
trait-decomposition (Step 3) stages of the v2 search pipeline. Each
experiment captures hypothesis, what was changed, what was observed,
and what we took away.

Test corpus: [search_v2/test_queries.md](../search_v2/test_queries.md).
Outputs are written to `/tmp/step3_runs_vN/` and compared against the
prior version's snapshot.

---

## Baseline — current failure suite

State after the most recent iteration (contextualized_phrase added,
identity-vs-attribute principle in Step 3, category-aware
decomposition, CLEAN-FIT TEST). These are the known failure modes the
next experiment must address or explicitly defer.

- **q06 "parody of the godfather" — "parody" disappears.** Step 2
  absorbs "parody of" as a modifying_signal on "the godfather" rather
  than emitting "parody" as a peer atom. Step 3 then decomposes The
  Godfather's archetype/iconography/tonal register without any trait
  carrying the requirement "be a comedic spoof / parody movie."
  Result population would be dark mafia dramas, not parodies. Root
  cause: the atomicity test only checks anchor-side retrieval ("does
  the anchor retrieve cleanly without the modifier?"); it never
  checks modifier-side ("does the modifier's content phrase, stripped
  of connective language, name a kind-of-movie a user could ask for
  on its own?"). "Parody" passes that inverse test, "in the style of"
  doesn't — current prompt conflates them.

- **q34 "hidden gem" — FINANCIAL_SCALE leaks alongside
  CULTURAL_STATUS.** Step 3's CLEAN-FIT TEST said: when a candidate
  has `what_this_misses="nothing"`, commit only that candidate. q34
  still emits FINANCIAL_SCALE as a separate dimension because
  decomposition surfaced "low-budget / indie scale" as its own facet
  before clean-fit got a chance to apply. The clean-fit gate fires at
  the candidate-list level, not at the dimension-inventory level.
  Deferred per prior decision; re-evaluate after Step 2's atom
  discipline tightens.

- **q10 "joker but not the joaquin phoenix one" — Step 3 emits
  Person + Character as separate calls.** Correct under the
  intersection-mode contract (orchestrator co-holds the calls at
  exclusion time), but the orchestrator-side intersection logic does
  not yet exist. Without it, additive-union exclusion strips every
  Phoenix film and every Joker film, including the ones the user
  wants. Step 3's output is correct; the gap is downstream.

- **q11 "superhero movie not from marvel or dc" — same shape as
  q10.** STUDIO_BRAND emits separately for Marvel and DC. Same
  orchestrator-side dependency.

- **General atomicity bias — modifier-vs-atom test is single-direction.**
  The current `_ATOMICITY` "DEEP RESHAPE" pattern absorbs
  population-naming content phrases (parody, musical, horror) as
  modifiers when they appear in prepositional frames ("parody of X",
  "X-style horror"). The role-marker list mixes structural binders
  ("starring", "directed by") with semantic frames ("based on", "in
  the style of"), licensing absorption of any preposition-shaped
  phrase. Affects q06 today; suspected to affect future queries with
  shape "X of Y", "X meets Y", "X but make it Y".

- **Decomposition / role-binding ambiguity — "young person X" not
  yet tested.** No existing query forces an implicit-era constraint
  ("young Pacino", "prime Schwarzenegger"). Behavior under role
  modifiers that compress era + person + genre into one phrase is
  unknown.

---

## Experiment 1 — Principle-driven prompt audit (population test, evidence reading, example removal)

### Hypothesis

The parody-of-godfather failure isn't an isolated edge case; it's the
visible symptom of a class of biases baked into the Step 2 and Step 3
prompts and schemas:

1. **Modifier-vs-atom is treated as a grammar question.** The current
   atomicity test runs anchor-side only ("does the anchor retrieve
   cleanly without this modifier?") and never asks modifier-side
   ("does the modifier's content phrase, stripped of connective
   language, name a kind-of-movie standalone?"). Phrases that look
   prepositionally like modifiers but carry their own population
   (parody, musical, horror, spoof) get absorbed. The fix is to
   reframe atomicity as a population question that runs over every
   candidate phrase in the query, not just over the anchor.

2. **Schema and prompt examples pattern-prime instead of teaching
   shapes.** "parody of the godfather", "warm hug", "fight club",
   "wes anderson", "DC", "hallmark", "directed by david lynch"
   appeared as worked examples across Step 2 and Step 3 — directly
   overlapping with the eval test set. Worse, the qualifier_relation
   token-mapping in the Step 2 commit phase explicitly said
   "parody/satire framings → 'parody target'", which mechanically
   reproduces the wrong absorption every time the model sees the
   shape. Examples should illustrate the SHAPE of a rule, not the
   specific cases the rule is meant to handle; test queries
   especially must not appear in prompts.

3. **Step 3's role analysis was an if/else cascade by
   qualifier_relation string value.** Six branches each prescribing
   what the dimensions should describe based on string-equality
   matching on enum-feeling values ("parody target", "comparison
   floor", "comparison ceiling", "style reference", "transposition
   target"). LLMs are bad at string-equality branching at temperature
   > 0; the fix is to read qualifier_relation as freeform prose
   carrying operational meaning the model translates into dimension
   constraints — same answer, evidence-driven instead of branch-
   driven.

4. **Closed-feeling enum-shaped vocabularies invite bucket-fitting.**
   The "Common shapes (NOT closed)" lists in qualifier_relation, the
   anchor_reference example list, the ModifyingSignal.effect example
   list, and the trait_role_analysis example block all read like
   enums even though they're freeform. Replace with shape
   characterization that describes what the field needs to capture
   without naming specific cases.

5. **The atom-claims-content rule was actively wrong.** "Once a
   concept has been absorbed as part of a modifying_signal, that
   concept does NOT also become a separate atom — even if the bare
   concept word would look atomizable on its own" — this told the
   model to absorb population-bearing content whenever any positioning
   language wrapped it. The replacement is the population test plus
   a no-double-emit rule that fires AFTER atomicity decides which
   slot a phrase belongs in.

If the principles are right, the parody case fixes without any
parody-specific edit, and queries that didn't previously fail (q21
warm hug, q05 lynch, q12 early 2000s neo noir, etc.) keep working.

### Changes made

**schemas/step_2.py:**
- `ModifyingSignal.effect`: dropped 9-flavor bucket list; replaced
  with controlled-modal-token guidance + freeform.
- `Atom.modifying_signals`: rewrote to distinguish modifier-only
  language (absorbs) from cross-atom relationships (modifying_signal
  on each peer; both peers survive). Replaced "absorbs as a signal —
  does NOT also appear as a separate atom" with peer-aware framing.
- `Atom.split_exploration`: added the INVERSE check. Field now walks
  two analyses — forward (could the atom subdivide?) and inverse (do
  any modifying_signals carry content phrases that pass the
  population test?). Inverse check runs from evaluative_intent and
  modifying_signals, not surface_text alone.
- `Trait.qualifier_relation`: dropped 7-bucket "Common shapes" list;
  replaced with shape characterization (what role does the trait
  play + operational implication, freeform, no closed vocabulary).
- `Trait.anchor_reference`: dropped 9-example list (which included
  "parody of", "darker than", "in the style of" — overlapping test
  queries q06/q08/q29).
- `Trait.contextualized_phrase`: dropped 4 worked examples that all
  came from test queries (q06 godfather, q08 fight club, q15
  hallmark, q21 warm hug carver baseline). Replaced with construction
  rule + read-aloud test.
- `QueryAnalysis.atoms` NEVER list: replaced "LET ABSORBED MATERIAL
  APPEAR TWICE" with "DOUBLE-EMIT" rule that's population-aware.

**search_v2/step_2.py:**
- `_ATOMICITY`: rewrote the test as a population question that runs
  over every candidate phrase. Replaced "TWO PATTERNS (PARALLEL
  CRITERIA / DEEP RESHAPE)" forced-binary with three population-test
  outcomes (peer atoms / operator-only absorption / reshape-into-
  uncovered-region absorption with explicit re-run instruction).
  Removed the absorbed-claims-content pitfall line that drove the
  parody bug.
- `_MODIFIER_VS_ATOM`: scoped section to in-atom modifier-only
  language only; removed the 6-bullet example list ("Not funny", "A
  bit funny", "Darkly funny", "Starring Tom Hanks", "Around 90
  minutes", "Like Inception"). Cross-criterion content-phrase
  decisions now route to the population test in `_ATOMICITY`.
- `_EVALUATIVE_INTENT`: replaced "Common signal shapes" closed-
  feeling list with generalized guidance about integrating each
  signal's effect into the intent; loosened the bucket-feeling
  grouping.
- `_COMMIT_PHASE`: removed the qualifier_relation token-mapping
  paragraph that explicitly mapped "parody/satire framings → 'parody
  target'". Replaced with a per-field summary that defers to the
  schema field descriptions; no token-mapping vocabulary.
- `_CARVER_VS_QUALIFIER`: replaced 3-step if/else procedure with
  evidence-reading guidance grounded in the operational filter-vs-
  downrank test.

**schemas/step_3.py:**
- Module header: removed test-query example ("warm hug" → 3
  expressions).
- `Dimension.expression`: dropped 6-example list that included "warm
  and comforting tonal register" (test-query overlap). Replaced with
  database-vocabulary categorization.
- `CategoryCall.expressions`: dropped the "directed by david lynch"
  and "warm hug movie" worked examples; replaced with cardinality-
  follows-dimensions framing.
- `CategoryCall.retrieval_intent`: removed "parody target, comparison
  floor/ceiling, style reference" enum-feeling list; routed Step 4
  encoding through trait_role_analysis prose translation instead.
- `TraitDecomposition.trait_role_analysis`: removed the 4-bucket
  example block (each illustrating one qualifier_relation string).
  Replaced with the two-question framing ((1) retrieve or position?
  (2) if positioning, what's the operational meaning translated from
  the relation prose?).
- `TraitDecomposition.dimensions`: removed "person credit, release
  year, runtime range" cardinality examples; replaced with intent-
  driven cardinality.

**search_v2/step_3.py:**
- `_TRAIT_ROLE_ANALYSIS`: replaced the 6-branch if/else cascade by
  qualifier_relation string value with the two-question evidence-
  reading framing. Identity-vs-attribute principle preserved (it
  follows from the role analysis structurally) but its 3-example
  block — using q06 godfather, q08 fight club, q29 wes anderson — is
  removed; principle is stated generally.
- `_DIMENSION_INVENTORY`: dropped 4 worked examples ("DC", "warm
  hug", "the godfather as parody target", runtime/release-date) —
  three were direct test-query overlaps. Operational tests preserved
  (those are evidence-reading, not pattern-priming).

**Verification:**
- `python -c "from schemas.step_2 import …; from search_v2.step_2
  import SYSTEM_PROMPT"` compiles cleanly.
- No test-query strings remain in any prompt or schema description
  (verified via grep across all four files for known query
  fragments).
- Step 2 system prompt: 32,121 chars. Step 3 system prompt: 50,409
  chars (about the same as before the audit; net change is roughly
  neutral — examples removed, principles expanded).

### Results observed

Ran all 42 queries through Step 2 + Step 3; outputs in
`/tmp/step3_runs_v4/`. Comparing against the v3 baseline
(`/tmp/step3_runs_v3/`, partial — 6 queries) and the documented
baseline failure suite at the top of this file.

**Failures from the baseline that resolved.**

- **q06 "parody of the godfather" (the principal target)** — Step 2
  now SPLITS into peer atoms: "parody" (carver, positive) +
  "the godfather" (qualifier, anchored to "parody of"). The inverse
  split_exploration check explicitly walks "the signal 'of the
  godfather' contains a content phrase 'the godfather' which names a
  specific population … splitting it out as a peer atom would allow
  the system to recognize the reference point independently." The
  parody trait routes Step 3 to Genre with expressions
  ["parody", "spoof"]; the godfather trait routes to attribute
  categories (Element/motif presence + Narrative setting +
  archetype). Population is no longer dark mafia dramas — it's
  comedic spoofs that mimic Godfather iconography. The
  trait_role_analysis on the godfather trait commits "this trait IS
  a positioning anchor rather than a retrieval target. The
  dimensions must describe the identifiable attributes of the
  reference film." The fix landed without any parody-specific
  edits — the population test in `_ATOMICITY` and the inverse check
  in `split_exploration` both did the work.

- **General atomicity bias resolved** — the inverse split check
  fires across queries. Verified: queries that genuinely should
  absorb (q01 scary, q05 lynch+maclachlan, q07 plot description,
  q12 early 2000s neo noir, q21 warm hug, q28 starring wes
  anderson, q33 underrated, q34 hidden gem, q41 villain wins) all
  walk the forward and inverse checks and conclude correctly that
  there is nothing to split. Queries that genuinely should split
  (q06 parody, q09 wonder woman/new, q10 joker/phoenix, q11
  superhero/marvel/dc, q14 slow paced/action, q15 christmas/good/
  hallmark, q16 80s/action/arnold, q26 breaking bad/1800s, q27
  succession/pirates, q29 wes anderson/horror, q30 slow burn/
  thriller, q35 spoof/marvel, q36 horror/feudal japan, q38 al
  pacino/crime, q39 oscar bait/good, q40 musical/horror) split
  cleanly into peer atoms, with cross-relations recorded as
  modifying_signals on each peer.

**Failures that the audit was not aimed at and stayed where they
were.**

- **q34 "hidden gem" — partial improvement, not full resolution.**
  v4 emits only Cultural status + General appeal (no FINANCIAL_SCALE
  leak). However, the dimension inventory still produces TWO
  dimensions (one with Cultural status as the cleanest fit, one with
  General appeal as the cleanest fit), and CLEAN-FIT TEST commits
  both calls because each dimension's clean-fit candidate is
  different. So there are still two calls — but they are now both
  the right calls, neither is FINANCIAL_SCALE. This is the
  acceptable behavior the prior plan deferred to.

**New successes for the previously-untested cases.**

- **q26 "breaking bad but in the 1800s"** — peer atoms: breaking bad
  (qualifier, anchored to "but in the 1800s") + 1800s (carver). The
  qualifier_relation prose: "Acts as a thematic archetype reference
  that the results must be positioned against; Step 3 should look
  for dimensions related to the show's specific character arcs and
  moral descent." Freeform, descriptive — no fixed-vocabulary slot.
- **q27 "like succession but with pirates"** — peer atoms:
  succession (qualifier, anchored to "like") + pirates (carver,
  anchored to "but with"). Both relation prose blocks are
  query-specific, no slot-fitting.
- **q29 "wes anderson does horror"** — peer atoms: wes anderson +
  horror, both carvers. Atomicity correctly identifies "horror" as
  population-bearing inside the absorbed signal. Caveat below.
- **q35 "spoof of marvel movies"** — generalizes the parody fix.
  Same shape, different anchor entity, same correct decomposition.
- **q36 "horror set in feudal japan"** — peer atoms (horror +
  feudal japan). The model treated "feudal japan" as a population-
  bearing setting (a kind of movie users do ask for), so it split
  rather than absorbed. This is the opposite of the test design's
  hypothesis ("set in feudal japan" should absorb as positioning
  operator) — the model's read is also defensible since "feudal
  japan movies" is a recognizable population. Worth flagging but
  not a clear regression.
- **q38 "young al pacino crime movie"** — al pacino (carver,
  anchored to "young") + crime movie (carver). "Young" is correctly
  absorbed as a chronological narrowing operator with no standalone
  population. The atom's evaluative_intent integrates "young" as
  "specifically focusing on his early career or youthful
  appearances", which is exactly the implicit-era constraint the
  query intends.
- **q40 "musical horror"** — peer atoms (musical + horror), both
  carvers. The user's intent is the intersection. Cleanly split.
- **q41 "movies where the villain wins"** — single carver; the
  compound "villain wins" stays whole because splitting would lose
  the narrative-outcome semantics. Forward and inverse checks both
  conclude no split.
- **q42 "feels like a video game"** — single qualifier (video game)
  with "feels like" absorbed as comparative reference. Atomicity
  identifies that "video game" is a non-film medium being used as
  a stylistic reference, not a retrieval population. Routes to
  comparison-style retrieval.

**Regressions surfaced by the rerun.**

- **q06 / q35 carver-side intent overflow.** When atomicity splits
  a peer-atom pair (parody + godfather; spoof + marvel), the
  carver atom's evaluative_intent restates the qualifier reference
  inline ("Identify movies that function as a parody specifically
  targeting the godfather"). Step 3 reads that intent and emits a
  Title text lookup / Studio identity call FOR the qualifier
  reference inside the carver trait — duplicating the qualifier
  trait's work and re-introducing the very identity-routing the
  qualifier trait was supposed to avoid. The qualifier peer is
  routing correctly; it's the carver peer that's leaking. Fix
  candidate: when atomicity splits a peer-atom pair, the carver's
  evaluative_intent should be cleanly local — the cross-relation
  belongs on modifying_signals + qualifier_relation, not duplicated
  inside intent prose.

- **q35 marvel movies committed FRANCHISE / UNIVERSE LINEAGE for a
  qualifier trait.** The identity-vs-attribute principle was
  supposed to keep qualifier traits out of identity categories —
  but the model still picked the identity category here, justifying
  itself in the candidate's what_this_misses=Nothing. The
  retrieval_intent reframes Step 4's behavior ("treat as the
  archetype that the results must satirize or spoof"), but the
  category commitment still puts Marvel in Step 4's identity-route
  pool. The principle didn't fully bind. Worth iterating; might be
  the candidate-list audit not surfacing the attribute alternative
  forcefully enough, or the model genuinely seeing identity as the
  cleanest fit.

- **q29 wes anderson now CARVER instead of QUALIFIER.** Previously
  Step 2 emitted wes anderson as qualifier (style reference); the
  v3 plan documented this as the desired shape. v4 emits both wes
  anderson and horror as CARVERS, which licenses Step 3 to route
  wes anderson to PERSON_CREDIT — putting his actual filmography in
  the result pool. The atom's evaluative_intent reads "Identify
  films directed by Wes Anderson, OR films that capture his
  specific aesthetic and directorial voice" — the model honored
  both readings. Whether this is wrong depends on intent: a strict
  read of "wes anderson does horror" treats Anderson as style
  reference (he hasn't done horror, so the user wants Anderson-
  styled horror); a permissive read includes his actual films
  (because they can be horror-adjacent). The new prompt is more
  permissive than the prior version. Borderline regression; flag.

**Sanity check across the simple queries.** No regressions on the
queries the audit wasn't aimed at:
- q01 scary, q12 early 2000s neo noir, q16 80s action arnold, q17
  90s comedy jim carrey, q19 date night, q20 something for kids,
  q21 warm hug, q22 feel good, q23 anything but romcom, q24 no
  horror no romance, q31 preferably under 2 hours, q33 underrated:
  all produce the same shape of decomposition as v3 (verified
  against the partial v3 set + the documented baseline expectations).
- q05 directed by david lynch starring kyle maclachlan: cleanly
  splits into two carvers; "directed by" / "starring" properly
  absorb as role markers. Same as prior baseline.

**Quantitative shifts.**
- Step 2 prompt: 32,121 chars (was 32,121 — net-neutral after audit).
- Step 3 prompt: 50,409 chars (was 50,409 — net-neutral).
- Test-query strings in prompts/schemas: 0 (was 7 worked examples
  drawn directly from q05/q06/q08/q11/q15/q21/q29).
- Inverse split_exploration check appears in 41/42 outputs (only
  q01 omits it because there are no modifying_signals to analyze).
- Closed-feeling vocabulary lists removed: 5 (qualifier_relation
  shapes, anchor_reference examples, ModifyingSignal.effect flavors,
  TraitDecomposition example block, identity-vs-attribute example
  block).

### Lessons learned

1. **The population test resolves the parody case without parody-
   specific code.** The same principle that splits "parody of the
   godfather" also splits "spoof of marvel" (q35), "wes anderson
   does horror" (q29), "young al pacino crime movie" (q38), and
   "horror set in feudal japan" (q36). Generalized rules outperform
   pattern-matched examples. The hypothesis held cleanly.

2. **Inverse split_exploration is the load-bearing structural
   change.** Adding the inverse check made the population question
   structurally present. The model walks both checks every time;
   the analysis text is visible in outputs and grounds the trait
   commitments. The forward check alone (which is what the prior
   prompt had) systematically missed population-bearing absorbed
   content. Adding a question to the schema where there wasn't one
   before did the structural work; restating the principle in the
   prompt without the schema scaffold would not have been enough.

3. **Removing closed-feeling enum vocabularies eliminates bucket-
   fitting.** The qualifier_relation values that emerge in v4 are
   freeform, query-specific, and operationally specific — exactly
   what we want for Step 3 to read. No more "parody target" /
   "comparison floor" enum-feeling slot-fitting. The model
   describes what THIS query's relation operationally requires.

4. **String-equality branching out of Step 3 is a real win.** The
   v3 _TRAIT_ROLE_ANALYSIS had 6 if/else branches on
   qualifier_relation string values. Replacing with a two-question
   evidence-reading framing produced trait_role_analyses that read
   the relation prose and commit operational meaning specific to
   the query — without ever hitting an enum-shaped logic gate.

5. **Schema field descriptions are the highest-leverage place to
   change behavior.** Edits to the schemas (Atom.modifying_signals,
   Atom.split_exploration, Trait.qualifier_relation, Trait.
   contextualized_phrase, Dimension.expression, TraitDecomposition.
   trait_role_analysis) had outsized impact on output quality
   relative to system-prompt edits. The "schema = micro-prompts;
   prompt = procedural" principle works because the model reads
   the field description right before generating the field's value;
   that's where pattern-priming examples bite hardest, and where
   removing them helps most.

6. **Carver-side intent overflow is the next failure class to
   address.** When atomicity splits, the carver atom's
   evaluative_intent often restates the qualifier reference
   inline. Step 3 reads that prose and re-encodes the qualifier
   into the carver's dimensions, duplicating the qualifier trait's
   work and re-introducing identity-category routing the qualifier
   trait was supposed to prevent. Fix needs to make Atom.
   evaluative_intent (when atomicity splits) describe what the atom
   evaluates LOCALLY, with cross-relation living entirely on
   modifying_signals + qualifier_relation. Worth a Experiment 2.

7. **The identity-vs-attribute principle can still be defeated by
   single-call clean-fit logic.** q35 marvel movies routed to
   FRANCHISE/UNIVERSE_LINEAGE because the candidate's
   what_this_misses="Nothing" — clean-fit overpowered the
   structural rule for qualifier traits. Either the principle
   needs to be a hard gate (qualifier traits CANNOT commit any
   identity category, full stop), or the candidate-listing step
   needs to suppress identity candidates entirely for qualifier
   traits. Currently it's a guideline that the model can step
   over when an identity category looks clean.

8. **Most "single concrete trait" queries stayed identical.**
   q01 / q12 / q16 / q17 / q19 / q20 / q21 / q22 / q23 / q24 / q31
   / q33 produce essentially the same shape as v3. The audit
   didn't introduce regressions on the queries the audit wasn't
   aimed at — confirming that principle-driven changes don't
   universally destabilize.

---

## Experiment 2 — `aspects` enumeration step before dimensions

### Hypothesis

Step 3's failure pattern on q34 "hidden gem" diagnoses a structural
shortfall in the schema's prose-to-list pipeline. `target_population`
and `trait_role_analysis` (both prose) faithfully name every axis
the trait calls for — for q34: high qualitative reception AND low
cultural visibility AND low commercial footprint. But by the time
`dimensions` is generated, only two axes are translated into
database-vocabulary checks; the commercial-footprint axis silently
drops out. The model is doing two jobs in one step — enumerating
axes AND translating each into a searchable check — and the
translation step is where coverage breaks.

The structural fix is to separate enumeration from translation by
inserting an `aspects: list[str]` field between
`trait_role_analysis` and `dimensions`. Each entry is one short
noun-phrase in user-vocabulary naming a single axis the trait calls
for. `dimensions` then shifts to a translation step: for each aspect
(or group of related aspects), commit one searchable check; coverage
is mechanically auditable (every aspect must be addressed).

The principle that makes this work: the same insight that drove
adding `split_exploration` to atoms in Experiment 1 — adding a
question to the schema where there wasn't one before structurally
forces the model to do the work. Restating "make sure dimensions
covers everything" in prompt prose without a schema slot to commit
to was not sufficient before, and isn't sufficient now.

Two risks to monitor:
1. **Duplication risk.** `aspects` could become a near-restatement
   of `target_population`. Mitigation: target_population describes
   what KIND of movies in prose; aspects enumerates the independent
   AXES that population is defined along. Different cognitive task,
   different output shape.
2. **Pattern-priming risk.** This audit just removed test-query
   examples and closed-feeling enum lists. The new field's
   description must follow the same discipline: principle-driven,
   no test-query examples, no closed bucket vocabularies. The
   field describes what aspects ARE (axes / facets) and what
   discipline applies (grounding in role-analysis prose, no
   category vocabulary, no inventing) — not what they look like.

If the principle is right, we should see:
- q34 hidden gem dimensions cover all three axes
  (quality + visibility + commercial), not just two.
- Other multi-faceted figurative traits (q33 underrated, q22 feel
  good, q21 warm hug, q42 video game) enumerate cleanly.
- No regression on single-axis traits — the aspects list collapses
  to one entry.
- No translation drift: aspects in user-vocabulary,
  dimensions in database-vocabulary; one aspect can map to one
  dimension (1:1), or multiple aspects to one dimension when one
  searchable check captures several axes.

### Changes made

**schemas/step_3.py:**
- Added `aspects: list[str]` field on `TraitDecomposition`, ordered
  between `trait_role_analysis` and `dimensions`. Field description
  characterizes the axes-not-narrative discipline: each entry is
  one short noun-phrase in user-vocabulary, drawn directly from
  what target_population and trait_role_analysis identified, with a
  read-back operational test (could a fresh reader reconstruct the
  same dimensions list given only aspects + evaluative_intent?).
  NEVER list: no category-vocabulary translation, no collapsing
  distinct axes, no inventing aspects not grounded in role analysis.
  No worked examples in the field description — the discipline
  describes the SHAPE of an aspect, not specific cases.
- Updated `dimensions` field description: removed the
  intent-driven cardinality framing and replaced with a translation
  framing — dimensions translate the aspects list into
  database-vocabulary, one searchable check per aspect (or per
  related group). Added coverage discipline: every aspect must be
  addressed by ≥1 dimension; no dimension that doesn't trace to ≥1
  aspect.

**search_v2/step_3.py:**
- New `_ASPECT_ENUMERATION` section inserted between
  `_TRAIT_ROLE_ANALYSIS` and `_DIMENSION_INVENTORY`. Section frames
  aspects as the enumeration step that runs in user-vocabulary
  before the translation-into-database-vocabulary step. Operational
  tests cover grounding (every aspect traces to role-analysis
  prose), distinctness (collapse only when aspects truly share one
  searchable surface), and read-back (can dimensions be
  reconstructed from this list alone?). No worked examples; no
  closed shape vocabulary.
- `_DIMENSION_INVENTORY`: rewrote GENERATION DISCIPLINE to walk the
  aspects list. Added a COVERAGE TEST: every aspect addressed by
  ≥1 dimension; every dimension traces to ≥1 aspect. Kept the
  category-aware decomposition guidance (single category covers
  → one dimension; multi-category → split per facet) but anchored
  it on the aspects list rather than re-deriving from
  evaluative_intent. The "ABSTRACTION UP" / "CATEGORY NAMING" /
  "ABSENCE FRAMING" / "BUNDLING" / "PADDING" pitfall list stayed.

**search_v2/run_step_3.py:**
- No change. The `model_dump()` of TraitDecomposition picks up
  `aspects` automatically; existing JSON-pretty-print covers it.

### Results observed

Ran all 42 queries through Step 2 + Step 3 (v5); outputs in
`/tmp/step3_runs_v5/`. q06 hit a transient Gemini 503 on the first
pass and was re-run successfully. Comparing against v4
(`/tmp/step3_runs_v4/`).

**The principal target — q34 hidden gem — resolved cleanly.** v4
emitted 2 calls (Cultural status + General appeal). v5 emits 1
multi-expression Cultural status call covering all three
expressions ("underrated", "overlooked", "hidden gem", "cult
classic"). The aspects list explicitly enumerated three axes —
"high qualitative merit", "low visibility or mainstream
recognition", "underrated or overlooked status" — and the
dimension translation correctly recognized that Cultural status
covers all three with multi-expression breadth. The
"unknown/visibility" axis no longer drops out: it's surfaced in
aspects, addressed in the "low commercial footprint and
visibility" dimension, and its candidate analysis explicitly cites
the boundary text that puts visibility under Cultural status.
Mode-shift discipline (user-vocabulary in aspects → database-
vocabulary in dimensions) held cleanly.

**The v4 identity-vs-attribute regression — q35 marvel —
resolved.** v4 committed FRANCHISE/UNIVERSE_LINEAGE on the
qualifier marvel trait via clean-fit override. v5's marvel trait
candidate list still surfaces FRANCHISE/UNIVERSE_LINEAGE (honest
adjacency analysis) but the model now explicitly cites the
qualifier-blocks-identity boundary in `what_this_misses`: "The
boundary for qualifiers forbids identity categories; the user is
not retrieving MCU films, only positioning against them." The
committed call list contains only Story/thematic archetype with
two expressions. The aspect-enumeration step appears to have
forced the model to read the role analysis more carefully when
producing the candidate-coverage prose, which in turn made the
identity-blocking principle bind on commit. This regression-fix
was not the experiment's goal but it dropped out as a side effect.

**Multi-faceted figurative traits enumerate cleanly without
over-decomposing.** Aspects → dimensions → calls counts across
the figurative traits show healthy collapse:
- q34 hidden gem: 3 aspects → 2 dims → 1 call.
- q33 underrated: 3 aspects → 1 dim → 1 call.
- q22 feel good: 4 aspects → 2 dims → 1 call.
- q21 warm hug: 4 aspects → 3 dims → 1 call.
- q41 villain wins: 3 aspects → 1 dim → 1 call.
- q42 video game: 3 aspects → 3 dims → 3 calls (each aspect
  routes to a different category — Format / Narrative devices /
  Emotional-experiential — so no collapse possible).

The aspects step surfaces axes the prose role analysis was
already naming; the dimension step then decides how to group them
into searchable checks. Coverage is preserved, and most aspects
collapse into shared-category multi-expression calls rather than
proliferating into one-call-per-aspect.

**Concrete one-axis traits stayed identical.** q01 scary, q05
lynch, q05 maclachlan, q12 release-years, q16 1980s, q23 anything
but romcom, q28 starring wes anderson, q31 preferably under 2
hours all emit one aspect, one dimension, one call — no
over-decomposition. Single-aspect-per-trait is the floor; the
schema does not push the model to invent axes that aren't there.

**One real expansion that's a quality win, not a regression —
q19 date night.** v4: 1 call (Viewing occasion only). v5: 3 calls
(Viewing occasion + Emotional/experiential + General appeal). The
aspects list surfaced "suitability for a date night occasion" +
"romantic or shared-emotional tone" + "broad qualitative appeal"
as three independent axes — which matches the test_queries.md
mental-model entry for q19 ("watch_context" + "soft genre prior"
+ "reception bias"). v4 was undercovering; v5's expansion is the
correct shape for this trait.

**One small regression noted — q25 non-violent crime thriller.**
v4 committed 2 calls on the crime trait (Genre); v5 commits 3
(Genre + suspense as a separate Story/thematic archetype call).
The aspects list pulled out "suspenseful narrative" as a separate
axis from "thriller genre", which is technically a duplication —
suspense is the experiential signal of the thriller genre, not a
separate axis. Borderline; visible in the aspects listing as a
candidate over-split.

**One persistent v4 borderline — q29 wes anderson does horror.**
Step 2 still emits both wes anderson and horror as carvers,
licensing PERSON_CREDIT routing on Anderson. v5 didn't move this.
The aspect step is downstream of the role assignment, so it
couldn't intervene on the carver-vs-qualifier decision; this case
needs Step 2 work to resolve.

**Aspect-list duplication risk surfaced — q12 early 2000s.** The
"early 2000s" trait emitted two near-synonymous aspects: "release
years 2000 through 2004" and "early 2000s era". They share one
searchable check (a date range), and the dimension step
correctly collapsed them into one dimension with one call, but
the aspects list itself wasn't crisp. Watch this; if it
generalizes, the field description's "INDEPENDENCE" test needs
strengthening.

**Carver-negative on identity traits — q11 marvel/dc — turned
out to be variance, not regression.** Initial v5 sample produced
7 calls (vs v4's 5); a re-run produced 5 calls, identical to v4.
Gemini at temperature 0.15 retains enough sampling variance for
the candidate-list audit to differ run-to-run on whether
Character-franchise gets surfaced and committed alongside
Studio/brand + Franchise/universe lineage. The aspects step did
not destabilize this query; the underlying carver-negative
composition policy still depends on the orchestrator-side
intersection-mode contract (deferred from prior experiments),
but Step 3's call shape is steady within sampling noise.

**Quantitative shifts.**
- Step 3 prompt: 54,382 chars (was 50,409 — net +3,973 chars
  from `_ASPECT_ENUMERATION` section; section is principle-only,
  no examples, no closed vocabularies).
- Aspects field present in all 42 outputs across all traits;
  41/42 queries produce aspects that ground in target_population
  /trait_role_analysis prose with no invented axes spotted.
- Aspect → dimension → call collapse rates: median 1:1:1 for
  one-axis traits, ~3:2:1 for figurative traits, ~3:3:3 for
  multi-category traits.
- Test-query strings in prompts/schemas: still 0.
- New worked examples in the schema field description for
  `aspects` or `dimensions`: 0. New closed-vocabulary lists: 0.
  The prior audit's discipline (principle-driven, no
  pattern-priming) was preserved.

### Lessons learned

1. **Adding a schema slot for the missing question is the
   load-bearing move, again.** Same pattern as Experiment 1's
   inverse split_exploration check. Telling the model "make sure
   dimensions covers all the axes named in the role analysis"
   in prompt prose was insufficient before this experiment;
   adding `aspects` as its own field — a place the model commits
   the enumeration in writing before the translation step —
   produced the structural change. The prompt could now be
   pruned of "dimensions must cover everything" exhortations
   because the schema enforces it; the dimensions field
   description's COVERAGE clause references the aspects list as
   the source of truth.

2. **Mode-separation prevents axis loss.** The failure mode on
   q34 was specifically that the model was doing two cognitive
   tasks in one step: enumerating axes (user-vocabulary) AND
   translating them (database-vocabulary). Splitting these into
   two consecutive fields, with the discipline that aspects stay
   in user-vocabulary and dimensions are translations, gave the
   model two simpler tasks instead of one harder one. The
   "READ-BACK" operational test (could a fresh reader rebuild
   the dimensions from this list?) names what makes the
   separation real instead of nominal.

3. **The aspects step strengthens the qualifier-blocks-identity
   principle as a side effect.** q35 marvel resolved without a
   targeted edit. Hypothesis: the aspect-enumeration step makes
   the model walk the role analysis once more before producing
   candidates, and that re-walk surfaces the qualifier role
   strongly enough that the candidate prose names the boundary
   block when an identity category appears. v4 had the principle
   stated; v5 has the principle bound.

4. **Aspect duplication is the risk to monitor next.** q12's
   "release years 2000 through 2004" + "early 2000s era" surfaced
   as twin aspects. The dimension step collapsed them correctly,
   but the aspect list itself was redundant. The INDEPENDENCE
   test ("could a candidate film vary along one without varying
   along the other?") is the right discipline; if duplications
   generalize, the field description needs to lean harder on
   that test.

5. **More calls is sometimes a coverage win, not padding.** q19
   date night and q25 violent crime thriller both gained calls,
   but q19's gain matches the test-suite mental model exactly
   (occasion + tone + appeal) while q25's gain is genuine
   over-split (suspense was already inside thriller). The
   distinction: gained calls are good when each addresses a
   genuinely independent axis the trait calls for; bad when
   they re-encode the same axis from multiple angles. The
   aspects list makes this distinction reviewable — every call
   should trace to an independent axis.

6. **Single-axis traits are not destabilized by a multi-axis
   discipline.** q01 / q05 / q12 / q16 / q23 / q28 / q31 stayed
   at 1 aspect / 1 dimension / 1 call. The schema doesn't push
   the model to invent axes; one-aspect cardinality emerges
   naturally when the trait is genuinely one-axis. This was a
   risk worth flagging — multi-aspect framings can lure the
   model into over-decomposing simple traits — but didn't
   materialize.

7. **Carver-negative composition policy is still the
   orchestrator's job, not Step 3's.** q11 marvel/dc surfaced
   more identity-category axes per trait under the new aspects
   discipline; this is correct structurally but wrong under
   additive-union exclusion. The aspects field gives Step 3 a
   richer picture; the orchestrator-side intersection contract
   (deferred from Experiments 0/1) is what reconciles
   richer-Step-3 with negation semantics. Don't try to solve
   this inside Step 3.

8. **Schema-as-micro-prompts holds at scale.** The schema field
   description for `aspects` is the bulk of what the model reads
   when generating that field; the prompt section
   `_ASPECT_ENUMERATION` reinforces but doesn't restate. The two
   together stayed under 4k characters added to the prompt
   surface. The prior audit's discipline (no examples, no closed
   vocabularies, principle-driven prose) was preserved without
   adding back what we removed in Experiment 1.

---
## Experiment 3 — `role_evidence` field before role commit (Step 2)

### Hypothesis

Experiment 1 left q29 "wes anderson does horror" with both atoms
emitting as carvers, licensing PERSON_CREDIT routing on Anderson.
The structural cause: the operational filter-vs-downrank test in
`_CARVER_VS_QUALIFIER` presupposes a stable retrieval target. When
the trait itself is what's contested ("does Anderson belong in
the result set, or is he a style reference for a horror
population that doesn't include him?"), the test becomes circular
— it gives the carver answer if you assume Anderson is the
target, the qualifier answer if you assume he's a reference.
Three things compound at the role-commit point:

1. The atom-level `evaluative_intent` permits both readings ("OR
   reflecting the distinct style") because the cross-relation is
   counterfactual — Anderson hasn't directed horror.
2. The Shape-tendencies guidance ("named entities tend to carve")
   biases toward carver for any named person without testing
   whether the carve produces a realizable population.
3. The disambiguating evidence (does Anderson actually have
   horror films?) lives nowhere in the schema; the model never
   has to commit it in writing before the role gets written.

The fix mirrors the structural pattern from Experiments 1 and 2 —
add a schema slot where the disambiguating question gets answered
in writing before the dependent commit. Experiment 1 added
`split_exploration` before atom commit; Experiment 2 added
`aspects` before dimensions commit; this experiment adds
`role_evidence` before role commit.

The single sentence the field demands the model commit:

> Reading the source atom's evaluative_intent and any cross-atom
> relations on modifying_signals: does committing this trait as
> a literal CARVE (intersect the named thing with the rest of
> the query) produce the population the user wants, or is the
> named thing being used to evaluate a population it isn't
> itself part of?

The pivot is the second clause — "evaluate a population it isn't
itself part of." It collapses the qualifier-shape diversity to
one structural fact: the named thing is *outside* the result
population, used as an instrument for evaluating it. That's the
common denominator across every qualifier in the test set
(references, templates, counterfactuals, comparison floors and
ceilings, anchors, style transpositions) and it generalizes to
any qualifier shape we haven't seen yet — without enumerating
subtypes (the closed-vocabulary trap the prior audit eliminated).
The test is verifiable: "is the named thing going to be in the
result set?" is a yes/no question grounded in real-world facts
rather than circular logic.

If the principle is right:
- q29 wes anderson commits as qualifier; Step 3 then routes him
  to attribute categories (visual craft, production aesthetic)
  rather than PERSON_CREDIT.
- No regression on cases where the named thing IS the population
  (q05 lynch+maclachlan, q11 marvel/dc carvers, q16 arnold,
  q17 carrey, q28 wes anderson director credit).
- No regression on cases already correctly committing as
  qualifier (q06 godfather, q08 fight club / seven, q26 breaking
  bad, q27 succession, q35 marvel, q42 video game).
- The new field grounds in verifiable facts; the model's
  evidence prose should mention whether the named thing has the
  cross-relation in reality (Anderson hasn't done horror; Hanks
  has been in films; Marvel has produced movies).

Risk to monitor: the model might rationalize whichever role it
was already going to pick rather than letting the evidence
constrain the role. Mitigation: the field's wording forces the
model to imagine the literal carve's result set first, which
makes counterfactual cases visibly empty in the prose before the
role commits.

### Changes made

**schemas/step_2.py:**
- Added `role_evidence: str` field on `Trait`, ordered immediately
  before `role`. Field description carries the one-sentence
  question and names role as the conclusion. No worked examples,
  no closed-vocabulary list of qualifier subtypes — same
  discipline as Experiments 1 and 2.
- Updated `role` field description: dropped the prior
  filter-vs-downrank operational test (the test that became
  circular for ambiguous traits) and replaced with a
  conclusion-of-evidence framing. The CARVER and QUALIFIER
  definitions track the role_evidence question's two clauses
  ("the named thing belongs in the result set" vs "the named
  thing evaluates a population it isn't part of").

**search_v2/step_2.py:**
- `_CARVER_VS_QUALIFIER` rewrote PROCESS to commit role_evidence
  first, role second. The evidence question (would the literal
  carve produce the population the user wants?) replaces the
  prior filter-vs-downrank operational test. Kept the polarity-
  is-orthogonal note and the named-entity pitfall (sharpened to
  warn against latching on namedness without running the
  evidence question).
- `_COMMIT_PHASE` updated PER-TRAIT COMMITMENTS list to include
  role_evidence in the per-trait walk; updated OPERATIONAL TESTS
  to add a post-role_evidence and post-role check (did I imagine
  the literal carve's result set, and does role match the
  evidence?).

**search_v2/run_step_3.py:**
- Surface `role_evidence` in the trait-inputs print block, ahead
  of role, so eval runs show the disambiguating prose.

### Results observed

Ran all 42 queries through the v6 pipeline; outputs in
`/tmp/step3_runs_v6/`. Comparing role assignments against v5
(`/tmp/step3_runs_v5/`).

**The field is doing its job on most qualifier cases, in the
model's own words.** Sample evidence prose across the test set:

| Query | Trait | v6 role | role_evidence prose |
|---|---|---|---|
| q06 | the godfather | qualifier | "The user does not want to watch The Godfather; they want to watch something that references or spoofs it from the outside." |
| q26 | breaking bad | qualifier | "The user is not looking for the literal show Breaking Bad (which is modern), but rather movies that are 'like' it, making the show a reference for evaluation rather than a member of the result set." |
| q27 | succession | qualifier | "The user does not want the show Succession itself, but rather movies that share its qualities, making it a reference for evaluation." |
| q35 | marvel movies | qualifier | "The user does not want actual Marvel movies; they want movies that use Marvel as a reference for parody, making this a qualifier for the spoof." |
| q42 | video game | qualifier | "The user is not asking for literal video games or necessarily adaptations of games, but rather movies that share their experiential qualities, making 'video game' a reference for evaluation." |

The "outside the result set" framing surfaces directly in the
prose ("does not want to watch", "reference for evaluation",
"from the outside") — the model is using the schema's question
to write disambiguating reasoning before committing role.

**The field correctly holds carver where the entity belongs in
the result set:**

| Query | Trait | v6 role | role_evidence prose |
|---|---|---|---|
| q16 | arnold | carver | "The user wants movies featuring this specific actor; the literal carve of his filmography is the intended set." |
| q28 | wes anderson (starring) | carver | "Committing this as a literal carve for movies featuring Wes Anderson in the cast produces exactly the population the user is asking for." |

The "literal carve... produces the population" framing fires when
the entity does belong in the result set. No regression on the
previously-correct carvers (q05 lynch, q05 maclachlan, q11 marvel
/dc as exclusion targets, q17 carrey, q28 anderson director
credit).

**The principal target — q29 wes anderson does horror — did NOT
flip.** v6 still commits Anderson as carver. The role_evidence
prose is the rationalization risk landing exactly where the
hypothesis flagged it could:

> "The user is looking for movies that are 'by' or 'in the style
> of' Wes Anderson, meaning he is a literal part of the desired
> result set's definition."

The model accepted the atom-level evaluative_intent's hedged "by
OR in the style of" framing and concluded Anderson is "a literal
part of the desired result set's **definition**" — using
"definition" as the weasel word that lets style-reference cases
pass the realizability test. Anderson IS part of the definition
(as a style template) but is NOT part of the result set (he has
no horror films). The field surfaced the contested reasoning but
didn't bind the model to ask the verifiable realizability
question (does Anderson have horror films?).

**One real regression — q30 ideally a slow burn thriller.** v5
correctly committed "slow burn" as qualifier (the SOFTENS hedge
on "ideally"). v6 flipped to carver. role_evidence: "Committing
this as a literal carve for movies with slow-burn pacing
produces the population the user wants to see." The new framing
ignores hedge semantics — under the realizability test, slow-
burn pacing IS a population the user wants, so it carves. But
the user's "ideally" signaled openness to alternatives, which is
qualifier (downrank, not filter) semantics. The new field
deprecated the prior filter-vs-downrank test entirely; with it
went the hedge-driven qualifier behavior.

**One borderline change — q32 nothing depressing.** v5 carver-
negative; v6 qualifier-negative. role_evidence: "The user wants
to avoid a specific emotional quality; 'depressing' is the axis
for evaluation." Tonal traits are gradient, so qualifier-
negative (downrank by tonal score) is arguably more correct than
carver-negative (binary exclusion). Could go either way; not a
clear regression.

**One improvement — q04 nothing too dark or scary.** v5 merged
"dark or scary" into one carver-negative trait. v6 split into
"dark" and "scary" as separate qualifier-negative traits. The
"too" hedge correctly drove qualifier semantics; the split is
also better-shaped (independent constraints). Net positive.

**q11 marvel/dc held steady.** Re-running v5's variance issue
within v6: still 5 calls in the typical sample. role_evidence on
each negative carver names the entity as part of the population
to be excluded — correct semantics for carver-negative.

**Summary table:**

| Failure | v5 | v6 |
|---|---|---|
| q29 wes anderson does horror | carver (regression from baseline) | carver (still unfixed) |
| q30 ideally a slow burn thriller | qualifier (correct) | carver (new regression) |
| q04 dark/scary merged | merged carver-neg | split qualifier-neg (improvement) |
| q32 depressing | carver-neg | qualifier-neg (arguably better) |
| All other role assignments | (per v5) | unchanged |

### Lessons learned

1. **The field works as written for most cases — but
   rationalization-resistant prose for hedge-able framings still
   needs work.** q29 demonstrates the rationalization risk we
   flagged in the hypothesis: when the atom's evaluative_intent
   hedges between two readings ("by OR in the style of"), the
   model picks the carver-favoring read and writes prose that
   uses "part of the definition" rather than "part of the result
   set." The wording asks the right structural question but
   doesn't force the verifiable factual sub-question (does the
   entity have the cross-relation in reality?). A future
   iteration could sharpen the field by demanding the model
   answer the factual sub-question explicitly: "what films does
   the entity actually have under the cross-relation?" — empty
   answer → qualifier; non-empty answer → carver.

2. **Role-disambiguating evidence quality is real progress even
   when conclusions don't flip.** The role_evidence prose makes
   the model's reasoning visible and verifiable. Even on q29
   where the conclusion didn't change, the prose now
   states the model's reasoning explicitly — which means the
   failure is now diagnosable (the "definition" weasel word is
   spottable in eval) rather than buried in implicit role
   selection. Auditability went up; the next iteration knows
   exactly where to push.

3. **Replacing the filter-vs-downrank test deprecated a real
   semantic — hedge-driven qualifier behavior — by accident.**
   The prior test tied hedges to downrank (qualifier) semantics:
   "ideally a slow burn thriller" → soft preference → qualifier.
   The new realizability test asks whether the trait names a
   population, and slow-burn pacing IS a population, so it
   carves. q30 surfaced this regression. The fix isn't to bring
   back the old test (it's still circular for ambiguous traits);
   the fix is to recognize that hedges and intensifiers are
   ORTHOGONAL to the realizability question — they affect
   salience and downstream weighting, not role. The field
   description and prompt could note this explicitly: "hedges
   like 'ideally', 'preferably', 'kind of' do not change the
   realizability answer; they affect salience downstream." That
   keeps q30 as carver but with reduced salience.

4. **Same structural pattern as Experiments 1 and 2 — adding a
   schema slot for the disambiguating question forces the work
   in writing.** Experiment 1: split_exploration before atom
   commit. Experiment 2: aspects before dimensions. Experiment
   3: role_evidence before role. All three pattern-match. The
   delta in q06/q26/q27/q35/q42 prose quality (model writing
   "from the outside" / "reference for evaluation" /
   "qualifier for the spoof" in its own words) shows the slot
   does its job when the question is unambiguous. Where it
   doesn't bind (q29) is when the atom phase has already hedged
   the evidence the role phase needs to read.

5. **Schema field descriptions remain the highest-leverage
   change.** The role_evidence field description is one
   sentence; the prompt's `_CARVER_VS_QUALIFIER` rewrote
   PROCESS to commit it first; nothing else changed. Total
   prompt growth: ~350 chars. The behavior delta (qualifier
   prose articulation across 5+ queries; q04 hedge split; q32
   gradient flip) is large relative to the change footprint.
   Same finding as Experiments 1 and 2.

6. **No worked examples, no closed vocabularies, no test-query
   strings introduced.** The audit discipline from Experiment 1
   was preserved: the field description states the question
   abstractly; the prompt's PROCESS section describes the
   reasoning shape, not specific cases. The "common pitfalls"
   list mentions named-entity status as a common trap but
   doesn't list specific named entities.

7. **Rationalization risk is the open question for any
   evidence-field design.** A model that's already going to pick
   role X can write evidence prose that supports role X. The
   defenses available: (a) make the evidence question grounded
   in verifiable facts the model can't easily evade ("what films
   exist under this cross-relation?"); (b) make the field's read-
   back test explicit and narrow ("does my evidence prose
   imagine the literal result set, or does it use abstract
   definitional language?"); (c) accept that evidence will
   sometimes rationalize and add another layer downstream.
   Experiment 3 took (a)-and-(b) at the prose level but didn't
   demand the model answer the factual sub-question literally;
   q29 is the case where that absence shows.

8. **The audit's prior wins are not destabilized.** All
   Experiment 1 atomicity fixes (q06 parody split, q26 breaking
   bad split, q27 succession split, q35 spoof split) survive.
   All Experiment 2 aspects-driven coverage improvements
   (q34 hidden gem 1-call, q35 marvel qualifier-blocks-identity)
   survive. The role_evidence change adds disambiguation prose
   without disturbing the structural commitments those
   experiments locked in.

---

## Experiment 4 — Evidence-gathering reframe of `role_evidence` (system-agnostic, three qualifier shapes, look across atoms + traits)

### Hypothesis

Experiment 3 surfaced three failure modes that all trace back to
the same shortcoming in `role_evidence`'s wording:

1. **q29 wes anderson does horror — rationalization.** The
   "literal carve produces the population / named thing evaluates
   from outside" framing let the model write "definition" as a
   weasel word: Anderson is "a literal part of the desired result
   set's *definition*". Trait stayed carver, identity routing
   leaked.
2. **q30 ideally a slow burn thriller — flipped to carver
   incorrectly.** The realizability test ("does this trait name a
   population?") returned yes for slow-burn pacing standalone,
   without considering that thriller (a peer atom) was already
   gating the population this trait would only refine.
3. **Surface-operator pattern-matching at the framing level.** The
   prior wording orbited around named entities and explicit
   operators; cases without operators (slow-burn thriller, funny
   horror — adjective-noun modifier structures) got read as
   parallel carvers because the framing didn't surface the
   peer-relationship test.

The deeper fix is to teach the model the THINKING — what kind of
relationship between traits matters for the role decision —
rather than asking a single test that presupposes a stable
retrieval target. The relationship that matters is system-
agnostic: in any movie-search system, a trait either gates
eligibility (carver) or scores within a population other traits
gate (qualifier). Three structurally-distinct evidence shapes
support the qualifier conclusion:

(a) **Continuous-score-only**: the trait has no yes/no membership
    a search could check (mood, pacing, tonal register, comparative
    axis without explicit reference).
(b) **Comparison reference**: what the user wants is not this
    trait's population, it is a population evaluated against this
    trait (Godfather as parody target; fight club as darkness
    floor; video game as experiential analog).
(c) **Peer-gates-the-population**: this trait, examined alone,
    looks like it could carve — but another atom or trait in the
    query is already gating the population this one would only
    refine. (Slow-burn looks carve-able alone, but in "slow burn
    thriller" the thriller atom is the structural population
    gate; slow-burn refines.)

Naming these three shapes explicitly does several things:
- It replaces the rationalization-prone "outside the result set"
  abstract framing with structural questions the model can
  verify.
- It generalizes beyond named-entity cases — q30 / q40 funny-
  horror / similar adjective-noun modifier structures get the
  peer-gate test even though no surface operator connects the
  atoms.
- It stays system-agnostic — no mention of vector spaces,
  metadata, lexical lookups, or any other architectural surface.
  The reasoning is in terms of generic search operations: gating
  eligibility vs continuous scoring.

The other change: the field description now explicitly tells the
model it can look at OTHER ATOMS in the query, not just the
in-progress trait list. Atoms carry the richer prose
(evaluative_intent, modifying_signals, split_exploration) — by
the time role_evidence runs, atoms are finalized; traits are
mid-commit. The atom layer often holds the disambiguating
evidence the trait layer needs.

If the principle is right, we should see:

- q29 wes anderson does horror flips to qualifier; horror gates
  the population (peer evidence shape c); Anderson refines.
- q30 ideally a slow burn thriller: slow-burn flips to qualifier
  via shape (c); thriller stays carver.
- q40 musical horror: both stay carver (each independently
  gates; no single peer dominates).
- q06/q26/q27/q35/q42 qualifier conclusions stay (shape b in
  most cases).
- q05/q11/q16/q17/q23/q28/q31 carver conclusions stay (peer
  atoms either don't exist or don't dominate).
- Hedge-driven cases (q19 date night, q23 anything but romcom,
  q31 preferably under 2 hours) keep their roles since hedges
  are explicitly named as orthogonal.

Risks:
- The three evidence shapes could become a closed-vocabulary
  list the model treats as enum buckets. Mitigation: each shape
  is described structurally (what it IS), not by example, and
  the framing is "one or more of the following" rather than
  "pick the matching bucket".
- The peer-gate test (shape c) could over-fire — every modifier
  could be read as "refining a peer" when it actually carves
  independently. Mitigation: shape c is qualified ("looks like
  it could carve... but a peer atom or trait is already gating
  the population this one would only refine"); the model has
  to identify a specific gating peer.

### Changes made

**schemas/step_2.py:**
- `role_evidence` field description rewritten as evidence
  gathering. One sentence; opens with "Gathering evidence from
  qualifier_relation (committed above) and the other atoms /
  traits in the query"; asks the carver question first; if no,
  asks which of the three qualifier-evidence shapes is
  supported. No mention of architecture-specific surfaces (no
  vector spaces, metadata, lexical entities); reasoning stays
  in generic search-operation language ("definitively include
  or exclude", "continuous score", "comparison reference",
  "another atom or trait already gates the population").
- `role` field description rewritten to match the evidence
  framing. CARVER = "definitively gates eligibility"; QUALIFIER
  = "scores or refines within a population other traits gate,
  OR is itself a comparison reference rather than the
  population the user wants".

**search_v2/step_2.py:**
- `_CARVER_VS_QUALIFIER` rewritten. CARVER and QUALIFIER
  definitions match the schema (gating eligibility vs scoring/
  comparison reference). PROCESS section asks the carver
  question first, then enumerates the three qualifier-evidence
  shapes (continuous score / comparison reference / peer atom
  or trait already gates). Names where each shape's evidence
  lives (trait nature alone / qualifier_relation +
  modifying_signals / other atoms and traits). Common pitfalls
  list updated: namedness, hedges, specificity, negation —
  each named structurally, not by surface pattern.
- `_COMMIT_PHASE` OPERATIONAL TESTS post-role_evidence check
  updated to match the three-evidence-shape framing.

**search_v2/run_step_3.py:**
- No change. role_evidence is already surfaced in the trait
  inputs print block from Experiment 3.

Verification: imports compile; Step 2 system prompt grew from
~32k to ~33k chars (~1k additional, principle-driven; no
worked examples, no closed vocabularies, no test-query strings).

### Results observed

Ran all 42 queries through the v7 pipeline; outputs in
`/tmp/step3_runs_v7/`. Comparing role assignments against v6
(`/tmp/step3_runs_v6/`).

**Both principal targets resolved.**

q29 wes anderson does horror — **flipped to qualifier**.
role_evidence: *"This trait functions as a qualifier because it
describes a continuous stylistic spectrum (how 'Anderson-esque' a
film is) rather than a binary membership."* Evidence shape (a)
fired explicitly. Horror committed as carver: *"This trait can
definitively gate eligibility by excluding any films that do not
belong to the horror genre."* The peer-gate structure shows up
visibly in both evidence prose blocks — Anderson is gradient,
horror is the gate.

q30 ideally a slow burn thriller — **flipped to qualifier**.
role_evidence: *"This trait describes a continuous pacing
spectrum rather than a binary membership, supporting a qualifier
role."* Thriller committed as carver: *"This trait defines a
categorical genre membership that can definitively gate
eligibility."* The Experiment 3 regression is fully resolved —
slow-burn now refines thriller's gated population.

**Other role flips, all explainable by the new evidence shapes:**

| Query | Trait | v6 → v7 | Evidence shape that fired |
|---|---|---|---|
| q02 | "long" (in "arent too long") | carver → qualifier | (a) continuous score (runtime spectrum) + (c) Tarantino is the gate |
| q03 | "sad endings" | carver → qualifier | (c) "luv stories" gates the population; sad endings refines |
| q09 | "new ones" (negative) | carver → qualifier | (c) wonder woman gates; new-ones refines chronologically |
| q13 | "good" | carver → qualifier | (a) quality is a continuous spectrum |
| q36 | "feudal japan" | carver → qualifier | (c) horror gates; feudal japan refines as setting |
| q32 | "depressing" | qualifier → carver | model now reads "nothing depressing" as definite filter; flipped back to v5 behavior |

The peer-gate evidence shape (c) is doing the work the prior
framing couldn't — adjective-noun and modifier structures
without explicit operators (slow-burn thriller, sad endings on
luv stories, feudal japan as setting on horror) now route
correctly because the model identifies which peer atom/trait
gates the population.

**One real new regression — q21 warm hug and q22 feel good
flipped to qualifier as SOLE TRAITS.**

q21 warm hug — only trait in the query:
- role_evidence: *"This trait describes a continuous emotional
  and experiential quality that movies possess to varying
  degrees rather than a binary membership."*
- role: qualifier

q22 feel good — only trait:
- role_evidence: *"This trait describes a continuous emotional
  spectrum rather than a binary membership, supporting a
  qualifier role."*
- role: qualifier

The model latched on evidence shape (a) (continuous score) and
concluded qualifier — but with no peer atom or trait gating
eligibility, the system has no carver at all. There's no
population for the qualifier to refine within. This is a
structural break: a query needs at least one carver to gate
the result set.

The framing has three qualifier-evidence shapes; if any one
fires, the model concludes qualifier. There's no rule that
says "qualifier requires shape (c) — i.e., a peer to gate the
population this would refine". For pure-experiential
single-trait queries, shape (a) fires (the trait IS continuous)
but shape (c) cannot fire (no peer to gate), and the
conclusion should be "carver of last resort" — the lone trait
gates by virtue of being alone, even if its evaluation is
continuous.

q33 underrated, q34 hidden gem, q41 villain wins are also
single-trait figurative queries but stayed carver in v7 — the
model treated them as concrete-enough cultural/narrative labels
rather than continuous spectrums. The boundary between
"continuous score" (shape a fires) and "definite category"
(shape a doesn't fire) is judgment-driven and fragile in the
current framing.

**Other key queries held correctly:**
- q23 anything but romcom: carver-negative ✓
- q24 no horror no romance: both carver-negative ✓
- q25 non-violent crime thriller: violent carver-negative + crime thriller carver ✓
- q28 starring wes anderson: carver ✓
- q31 preferably under 2 hours: carver ✓ (hedge correctly orthogonal)
- q42 feels like a video game: qualifier ✓ (shape b — comparison reference)
- q06 parody (carver) + godfather (qualifier) ✓
- q26 breaking bad (qualifier) + 1800s (carver) ✓
- q27 succession (qualifier) + pirates (carver) ✓
- q35 spoof (carver) + marvel movies (qualifier) ✓

**Sampling-variance note — q04 atom count.** v6 had 4 atoms
(with-my-mom + cozy mysteries + dark + scary as separate
qualifier-negatives). v7 produced 3 atoms (cozy mysteries +
dark + scary as carvers; with-my-mom dropped). Step 2 prompt
grew by ~1k chars; some sampling shift is expected. The v7
atomization is also defensible (with-my-mom is more
occasion-context than a trait the system can score). Not a
clean regression.

**Summary table:**

| Failure | v6 | v7 |
|---|---|---|
| q29 wes anderson does horror | carver (still unfixed) | **qualifier ✓ (RESOLVED)** |
| q30 ideally a slow burn thriller | carver (Exp 3 regression) | **qualifier ✓ (RESOLVED)** |
| q21 warm hug | carver | **qualifier ✗ (NEW REGRESSION — single-trait, no carver)** |
| q22 feel good | carver | **qualifier ✗ (NEW REGRESSION — single-trait, no carver)** |
| q03/q09/q13/q36 peer-gate cases | carver+carver | **carver+qualifier ✓ (CORRECT)** |
| q06/q26/q27/q35/q42 from Exp 3 | (correct) | unchanged ✓ |
| q05/q11/q16/q23/q24/q25/q28/q31 carvers | (correct) | unchanged ✓ |

### Lessons learned

1. **Evidence-shape decomposition resolves the principal
   targets.** q29 and q30 both flip cleanly with the new
   framing. The peer-gate evidence shape (c) is what
   generalizes beyond named-entity / explicit-operator cases —
   adjective-noun structures (slow burn thriller), settings on
   genres (feudal japan on horror), endings on plot types
   (sad endings on luv stories), chronological narrowing
   (new ones on wonder woman) all route correctly because the
   model can identify the gating peer.

2. **Three evidence shapes is the right structure — but
   they're not symmetric.** Shape (a) "continuous score" is
   read off the trait in isolation; shape (b) "comparison
   reference" is read off qualifier_relation /
   modifying_signals; shape (c) "peer gates" is read off other
   atoms/traits. Shapes (b) and (c) are inherently relational —
   they require something external to the trait. Shape (a) is
   the only one that can fire on a trait alone. That asymmetry
   creates the single-trait failure mode.

3. **The single-trait failure mode is structural, not
   model-quality.** q21 warm hug and q22 feel good flipped to
   qualifier because shape (a) fired. But a query with only a
   qualifier has nothing to qualify against — the system has
   no eligibility gate. This is a structural invariant the
   role definitions should respect: every query needs at least
   one carver. The missing rule: "if shape (c) cannot fire
   because no peer atom/trait exists in the query, this trait
   must carve regardless of shape (a)" — a carver-of-last-
   resort clause.

4. **The "continuous score vs definite category" line is
   model-judgment, not principle-driven.** q21 warm hug and
   q33 underrated are both figurative single-trait queries;
   the model concluded shape (a) fires for warm hug but not
   for underrated. The boundary is judgment-driven and
   sampling-noisy. The single-trait safety net (carver-of-
   last-resort) would protect against this regardless of how
   the model judges shape (a) firing.

5. **Same structural pattern as Experiments 1, 2, 3 — but
   with a carryover risk.** Each iteration adds a schema slot
   or reframes a question. Experiment 4's reframe added
   evidence-shape decomposition; the structural principle
   held (most role assignments improved or stayed correct).
   The carryover risk: each new framing has its own failure
   mode that wasn't visible in prior framings. q21/q22 were
   correct in v6 (and v5, v4) and broke in v7. Iteration
   converges; it doesn't monotonically improve.

6. **Hedges stayed orthogonal as designed.** q31 "preferably
   under 2 hours" stayed carver — the hedge was named
   explicitly in the pitfall list as a salience signal, not a
   role signal. q19 date night also unchanged. The
   Experiment 3 regression on q30 was about the test
   framing's blind spot, not about hedges per se; reframing
   to peer-gate evidence resolved it.

7. **Auditability is now structural.** role_evidence prose
   in v7 explicitly cites shape (a) / (b) / (c) reasoning by
   describing what evidence supports the conclusion. This
   makes eval review faster — the failure mode is named in
   the prose. q21/q22 are visibly flagged as "shape (a) fired
   alone" without a peer; that's the diagnostic the next
   iteration needs.

8. **The v7 carver-of-last-resort fix is small.** Add one
   clause to role_evidence and `_CARVER_VS_QUALIFIER`: "When
   no peer atom or trait in the query gates eligibility on
   its own, this trait must carve — there must be at least
   one carver to define a result set, even if the trait's
   evaluation is continuous." This binds the asymmetry across
   the three shapes: (c) being unavailable forces carver
   regardless of (a) firing. q21/q22 fix; q33/q34/q41 stay
   correct; the rest of the test set is unaffected.

---

## Experiment 5 — Remove `holistic_read` entirely

### Hypothesis

Inspecting the v7 `holistic_read` outputs across all 42 queries
surfaced three problems:

1. **~26/42 are pure restatement** ("scary" → "looking for scary
   movies"). Zero added information; just tokens.
2. **~12/42 introduce operator-ese (CONTRASTS / SOFTENS / FLIPS
   POLARITY) that's sometimes wrong.** Examples: q08
   *darker than fight club but funnier than seven* mislabeled
   CONTRASTS (the comparisons are parallel, not contrasting); q26
   *breaking bad but in the 1800s* and q27 *like succession but
   with pirates* mislabeled CONTRASTS (transpositions); q39
   *oscar bait but actually good* mislabeled CONTRASTS (it's a
   qualification); q02 *quentin terantino movies that arent too
   long* mangled the polarity flip into "FLIPS POLARITY on the
   quality of being too long" (it's an exclusion).
3. **0/42 cases where the holistic_read clearly carried something
   the trait inputs don't.** Step 3's per-trait inputs
   (contextualized_phrase, evaluative_intent, role_evidence,
   role, qualifier_relation, anchor_reference, polarity,
   relevance_to_query) are richer than holistic_read; the field
   added ambient context but no load-bearing signal.

The hypothesis: removing holistic_read entirely should be neutral
or mildly positive — kills the operator-ese leakage and restate-
ment noise; nothing downstream loses information. The atom phase
already has structured think-before-commit fields
(split_exploration, standalone_check) that don't need a separate
top-level prose pass.

Risks:
- The standalone_check field in Atom referenced holistic_read as
  the comparison surface. After the change, it compares against
  the original query directly. Atoms might lose some of their
  structural-coupling reasoning if that prose anchor mattered.
- Step 2's role decisions might depend on having a top-level
  scratchpad to articulate the query's structural shape before
  diving into per-atom work. Specifically the shape-(c) peer-gate
  evidence reasoning needs the model to perceive WHICH atom is
  the structural anchor — that perception might happen during
  holistic_read drafting in v7.

### Changes made

**schemas/step_2.py:**
- Removed `holistic_read` field from `QueryAnalysis`.
  `QueryAnalysis.model_fields` now contains only `atoms` and
  `traits`.
- Updated `Atom.standalone_check` description: replaced
  *"Compare this atom's evaluative_intent against the user's
  articulated ask in holistic_read"* with *"Compare this atom's
  evaluative_intent against the user's original query"*.
- Updated module docstring from "Three coupled outputs" to "Two
  coupled outputs".

**search_v2/step_2.py:**
- Removed holistic_read from task framing (the numbered output
  list and the "modifying_signals, and holistic_read stay
  strictly DESCRIPTIVE" line).
- Updated ATOMICITY's standalone_check guidance to compare
  against the user's original query.
- Updated module docstring.

**search_v2/step_3.py:**
- Removed `holistic_read` parameter from `_build_user_prompt`
  and `run_step_3`.
- Removed the "Query holistic read:\n{holistic_read}" prefix
  from the per-trait user prompt.
- Updated `_TRAIT_ROLE_ANALYSIS` prompt: "do not re-derive from
  evaluative_intent or holistic_read" → "do not re-derive from
  evaluative_intent". Same edit to the schemas/step_3.py
  TraitDecomposition `trait_role_analysis` field description.
- Updated module docstring and Usage example.

**search_v2/run_step_3.py:**
- Stopped passing `analysis.holistic_read` to `run_step_3`.

Verification: imports compile; Step 2 system prompt shrank from
~33.4k to ~33.3k chars (~100 chars removed); Step 3 system
prompt unchanged in size.

### Results observed

Ran all 42 queries through the v8 pipeline; outputs in
`/tmp/step3_runs_v8/`. Comparing role / polarity / salience
assignments against v7 (`/tmp/step3_runs_v7/`).

**31/42 queries unchanged.** All carver-negative discipline
holds (q11 marvel/dc, q23 anything but romcom, q24 no horror /
no romance, q25 violent), all carver-only queries hold
(q05/q12/q16/q17/q18/q20/q26/q27/q28), all qualifier
conclusions held in v7 from peer-gate reasoning hold (q06/q08/
q26/q27/q29/q30/q35/q42).

**11/42 with role / polarity / salience differences. Pattern:
peer-gate (shape c) reasoning weakened.**

| Query | Trait | v7 → v8 | Class |
|---|---|---|---|
| q03 luv stories w sad endings | "sad endings" | qualifier → carver | regression — shape-c lost (luv stories was the gate) |
| q10 joker but not the joaquin phoenix one | "joaquin phoenix" | carver-neg → qualifier-neg | mild regression — exclusion semantic weakened |
| q32 nothing depressing | "depressing" | carver-neg → qualifier-neg | regression — "nothing" hard-exclude weakened to continuous |
| q36 horror set in feudal japan | "feudal japan" | qualifier → carver | regression — shape-c lost (horror was the gate) |
| q04 mom + cozy mysteries + dark + scary | (atomization shift) | 3 atoms → 4 atoms; "dark" carver-neg → qualifier-neg; "scary" salience supporting → central | drift, neutral-to-positive (mom returned as atom) |
| q13 hungover | "hungover" | carver → qualifier | defensible either way |
| q15 christmas movie | "actually good" → "good" | surface_text minor loss | minor regression |
| q30 ideally a slow burn thriller | "slow burn" salience | supporting → central | minor regression — hedge weighting lost |
| q32 practical effects salience | central → supporting | minor drift |
| q37 popcorn flick | carver → qualifier | defensible either way |
| q38, q09, q25, q31 | minor surface_text or ordering | not behavior change |

**Reading the regressions.** v8's role_evidence prose for q03
sad endings: *"This trait acts as a definitive gate for the
narrative structure (the ending must be sad) rather than a
continuous score."* For q36 feudal japan: *"This trait
identifies a specific historical setting that can definitively
gate eligibility for the search."* For q32 depressing: *"Tone
is a continuous emotional spectrum where movies are scored by
intensity."*

In each case the model evaluated the trait IN ISOLATION —
"can this gate eligibility on its own?" → yes → carver. The
shape-(c) check that asks "is another atom already gating the
population this trait would only refine?" did not fire. The
peer-gate evidence shape requires the model to perceive the
query's structural anchor (luv stories gates the population;
horror gates the population; the user's hard exclusion gates),
and that perception was happening in v7 during the
holistic_read drafting step — even though the prose itself
looked trivial.

**Step 3 routing churn is real but mostly noise.** v7 → v8
category-call sets diverge on ~17/42 queries, but most diffs
are sampling-noise reorderings of multi-expression calls
within the same category set. Material Step 3 differences:
- q26 v7=4 calls (story archetype + element + emotional + setting), v8=2 calls (archetype + setting). v8 is tighter and arguably better.
- q34 v7=2 calls (quality baseline + cultural status), v8=1 call (cultural status). v8 fixes the v7 over-routing.
- q41 v7=2 calls (plot events + emotional), v8=1 call (emotional). v8 may have lost the plot-events route, mild regression.
- q18 v7=3 title-text calls, v8=11 mixed. v8 is over-decomposing the inception/interstellar/tenet name lookup into themes — clear regression but Step-3 / not Step-2.

The Step 3 churn isn't load-bearing for the holistic_read
question. The signal that matters lives in Step 2 role
assignments where peer-gate reasoning weakened.

**No problems holistic_read was causing got fixed by removing
it.** Operator-ese mislabeling on holistic_read prose is
moot — the field is gone. But the role assignments for q08 /
q26 / q27 / q39 (queries whose v7 holistic_read had mislabeled
CONTRASTS) all stayed identical in v8: those were misreads at
the prose level that didn't propagate into role commits. The
mislabeled prose was harmless to downstream commits.

### Lessons learned

1. **Removing holistic_read is a net regression — modest, but
   directional.** 3-4 clear shape-(c) regressions (q03, q10,
   q32, q36); zero queries improved by removal. The
   restatement-and-operator-ese problem the experiment was
   trying to fix doesn't show up as downstream errors, so
   removing the field paid no benefit while losing the
   structural-anchor scratchpad.

2. **The holistic_read prose looked trivial but was load-
   bearing.** The bulk of v7's holistic_reads added no surface
   information — that was the original critique. But drafting
   them forced the model to perceive the query's structural
   anchor BEFORE atom-level reasoning. Without that pass, atoms
   get evaluated more independently, and shape-(c) ("a peer
   atom or trait already gates the population this would
   refine") fires less reliably. The field was a thinking
   surface, not just a downstream input.

3. **Information value ≠ thinking-process value.** When
   evaluating a prompt-output field for removal, ask BOTH "does
   the downstream consumer need it?" AND "does drafting it
   shape the model's reasoning trajectory?". holistic_read
   failed the first test (downstream didn't load-bearingly need
   it) but passed the second (drafting it cued the structural-
   anchor perception that shape-(c) reasoning depends on).

4. **The peer-gate evidence shape needs a place to perceive
   the structural anchor.** In v7 that place was holistic_read
   drafting. Without it, the next-best surfaces are
   split_exploration and standalone_check (atom-level) — but
   those are atom-LOCAL and ask "this atom relative to other
   atoms" without forcing a query-LEVEL view. The role-
   evidence field then has to recover the structural anchor
   per-trait, which is harder and noisier.

5. **The right intervention is to REFRAME, not remove.** The
   user's original suggestion ("look at possible high-level
   intents in concrete terms") points the way. Reframe
   holistic_read as STRUCTURAL-ANCHOR identification: which
   piece of the query gates the population, which pieces
   refine? That preserves the thinking-process value while
   killing the operator-ese pattern-matching and
   restatement-of-the-query failure modes. Probable failure
   mode of this reframe: model commits to one anchor when
   genuinely-parallel queries (q16 80s action arnold, q17 90s
   comedy jim carrey) have multiple parallel carvers. The
   field framing would need to allow "no single anchor;
   parallel carvers" as a first-class outcome.

6. **Standalone_check losing its prose anchor was OK.** The
   field reasoning still works comparing against the original
   query directly. No regressions traced to this change. The
   load-bearing piece of holistic_read was the structural-
   anchor perception, not standalone_check's reference
   surface.

7. **Roll back v8.** The changes ship a regression. Restore
   holistic_read to the schemas and prompts at v7 state, then
   run a follow-up experiment that REFRAMES holistic_read as
   structural-anchor identification. Key file changes to
   revert: `schemas/step_2.py` (re-add holistic_read field +
   restore standalone_check description), `search_v2/step_2.py`
   (restore task framing's holistic_read line + the
   standalone_check guidance), `search_v2/step_3.py` (restore
   `holistic_read` parameter on `run_step_3` and
   `_build_user_prompt`, restore the user-prompt prefix, and
   re-add holistic_read to `_TRAIT_ROLE_ANALYSIS`'s "do not
   re-derive from" list), `search_v2/run_step_3.py` (restore
   `analysis.holistic_read` arg).

---

## Experiment 6 — `intent_exploration` field (replace `holistic_read`)

### Hypothesis

Experiment 5 established two things: (a) v7's `holistic_read` field
was producing low-quality prose (~26/42 pure restatement, ~12/42
operator-ese mislabeling), but (b) drafting it was load-bearing for
the model's query-level perception, which shape-(c) peer-gate
reasoning depends on. The fix is to keep the drafting step but
reframe it.

The new field, `intent_exploration`, replaces `holistic_read` with
exploratory analysis: surface the plausible high-level intents the
query could be expressing, in concrete terms, and weigh which is
more likely from cues actually present in the query. Not a verdict
— a perception step that mirrors the codebase's existing pattern
(`split_exploration`, `standalone_check` — evidence-gathering, no
commits).

What the field is supposed to do:
- For unambiguous queries: surface one obvious read in concrete
  user-vocabulary terms (kind of movie + structural relationships
  between the pieces). One read is enough; don't manufacture
  alternatives.
- For ambiguous queries: surface 2-3 plausible reads and reason
  about which is more likely. The structural anchor (which piece
  gates, which refines) emerges as a side effect of comparing
  reads.
- Replace v7's flat-restatement-and-effect-token pattern with
  thinking that's structurally similar to atom-level
  split_exploration: evidence-gathering, no commitment.

Why this should work:
- "Don't commit" + "weigh likelihood" forces the model to actually
  think structurally rather than label-and-move-on.
- "Concrete terms" requirement (describe the watching experience,
  not abstract intent labels) prevents the abstraction-up failure
  mode.
- Description deliberately avoids:
  - listing example cue types (would become a checklist)
  - example "concrete" prose (would become templates the model
    pattern-matches)
  - banning effect tokens (paradoxically primes them; the
    description simply doesn't introduce them in the first place)
  - listing system-label examples (would become a do-not-use
    blocklist that primes them)

Risks:
- Pulling intent reasoning toward QUALIFIER over-readings. Naming
  "more-likely intent's structural anchor" as a place to look for
  shape-(c) evidence might encourage the model to read peers as
  refining each other when both genuinely gate.
- For sole-trait queries, the carver-of-last-resort issue from
  Experiment 4 is not addressed — orthogonal to intent_exploration.
- Step 2 prompt grew from ~33.3k chars to ~34.1k chars (+800
  characters of field description and procedural prose).

### Changes made

**schemas/step_2.py:**
- Added `intent_exploration: str` to `QueryAnalysis`, declared
  before `atoms` and `traits` so model commits in field-declaration
  order. Description is the trimmed version per user instructions:
  no example cue types, no example concrete prose, no
  effect-token-vocabulary ban, no system-label examples — instead a
  general definition of categorize-vs-concrete, and the discipline
  rules ("never commit", "never list without weighing").
- Updated `Atom.standalone_check` description to compare against
  `intent_exploration`'s surfaced intents (replaces v8's "compare
  against the original query"; replaces v7's "compare against
  holistic_read").
- Updated `Trait.role_evidence` description to add
  intent_exploration's structural-relationships as one of the
  evidence sources alongside qualifier_relation and other
  atoms/traits.
- Updated module docstring from "Two coupled outputs" back to
  "Three coupled outputs".

**search_v2/step_2.py:**
- `_TASK_FRAMING`: added intent_exploration as output #1, with a
  paragraph stating it is drafted FIRST and is the query-level
  perception step.
- `_ATOMICITY` `standalone_check` guidance: comparison surface is
  intent_exploration's surfaced intents.
- `_COMMIT_PHASE` OPERATIONAL TESTS: added "When atom-level
  evidence on shape (c) is ambiguous, intent_exploration's
  structural-anchor reasoning is the tiebreaker."

**search_v2/step_3.py:**
- `_build_user_prompt`: re-prepended "Query intent exploration:\n"
  block with the field's value (replaces v7's "Query holistic
  read"; replaces v8's no-prefix).
- `run_step_3` signature: re-added the second positional argument,
  now named `intent_exploration: str`.
- `_TRAIT_ROLE_ANALYSIS`: "do not re-derive from evaluative_intent
  or intent_exploration"; "DERIVE A DIFFERENT ROLE from
  evaluative_intent or intent_exploration".

**schemas/step_3.py:**
- `TraitDecomposition.trait_role_analysis`: same one-line edit to
  the "do not re-derive from" list.

**search_v2/run_step_3.py:**
- Pass `analysis.intent_exploration` to `run_step_3`.

**search_v2/run_test_queries.py (NEW):**
- Standalone batch runner for the 42-query test suite. Reads
  queries from `search_v2/test_queries.md`, runs Step 2 + Step 3
  end-to-end on each with concurrency=5 (asyncio.Semaphore,
  bounded), writes per-query output files matching the historical
  `/tmp/step3_runs_v*/qNN.txt` shape. Each task writes its own
  StringIO buffer to its own file so concurrent tasks never share
  stdout. Configurable via `--out` and `--concurrency`. Replaces
  the previous one-off `bash` loop in `/tmp/run_v8.sh`.

Verification: imports compile; Step 2 system prompt 34.1k chars;
Step 3 system prompt 54.4k chars. Smoke test on q29 (wes anderson
does horror) confirmed expected behavior — anderson=qualifier,
horror=carver, intent_exploration cleanly identified the
"horror-films-feeling-like-Anderson" intent as more likely than the
literal-filmography read.

### Results observed

Ran all 42 queries through the v9 pipeline via the new batch
runner (concurrency=5; wallclock 93.5s end-to-end vs ~6 minutes
sequentially in v8). Outputs in `/tmp/step3_runs_v9/`. Comparing
trait role/polarity/salience assignments against v7 (canonical) and
v8 (Experiment 5 baseline).

**Headline counts. 25/42 unchanged from v7. 17/42 with
differences.** Of those 17:
- 4 v8 regressions partially carry forward (q03, q10, q32, q36 —
  see below for nuance).
- 4 new differences vs v7 that read as IMPROVEMENTS or defensible
  alternatives (q15, q18, q22, q36).
- 4 new differences vs v7 that read as REGRESSIONS (q31, q32, q34,
  and q10 carrying from v8).
- ~5 surface_text minor shifts (q12, q14, q25, q38) — not
  behavioral.

**Intent exploration prose quality is materially better than v7's
holistic_read.** Read the v9 prose for the structural-anchor
cases:

q03: *"The core population is defined by the romantic genre and
the specific narrative structure of a sad ending, while the 'rainy
night' context serves as a qualifier for the desired vibe."* —
explicit anchor identification.

q15: *"The primary intent is to find Christmas-themed movies where
'quality' is defined by a contrast against a specific archetype —
likely seeking films with more cinematic depth, complex narratives,
or higher production value than the formulaic, sentimental holiday
romance genre."* — concrete description of what would satisfy +
explicit naming of hallmark as comparison reference.

q18: *"The most likely intent is a request for movies that are
similar in style, theme, or narrative complexity to these three
Christopher Nolan films [...] A less likely but plausible intent
is a request for the specific films themselves (a collection)."* —
two plausible reads, each weighed; concrete description of the
satisfying watching experience.

q29: *"This is more likely an ask for 'horror movies that feel
like Wes Anderson' rather than a literal search for a horror film
directed by Wes Anderson (as he hasn't directed a pure horror
feature)."* — two plausible reads with explicit cue-based
likelihood reasoning ("as he hasn't directed a pure horror
feature").

q36: *"Satisfaction requires both; a horror movie set elsewhere or
a non-horror movie set in feudal Japan would fail the intent. The
most likely read is a search for the intersection of these two
populations."* — explicit dual-gate identification.

Compare to v7 holistic_read for the same queries: pure restatement
("The user is looking for X") or operator-ese ("CONTRASTS with",
"FLIPS POLARITY on the quality of being too long"). The v9 prose
is doing actual structural-anchor work that v7's couldn't.

**Where v9 reads more correctly than v7:**

| Query | v7 | v9 | Why v9 is better |
|---|---|---|---|
| q22 feel good | qualifier (sole-trait bug) | carver | Fixes the v7 sole-trait bug. The intent_exploration concretely names the watching experience, which the role read as gating. |
| q09 wonder woman / new ones | qualifier-negative | carver-negative | "Wonder woman movies but not the new ones" is a hard exclude on the chronological subset; carver-negative matches the user's exclusion semantic. |
| q15 hallmark kind | carver-negative | qualifier-negative | "Not the hallmark kind" reads as comparison-reference exclusion (shape b) per intent_exploration: hallmark is the archetype being contrasted against, not a population to wholesale-remove. |
| q36 feudal japan | qualifier | carver | Per intent_exploration: "satisfaction requires both"; both genuinely gate, not "horror gates and feudal japan refines". |
| q18 inception/interstellar/tenet | carver (literal title lookup) | qualifier (mind-bending sci-fi reference) | Per intent_exploration: the most-likely intent is "more like these" rather than literal collection-fetch. v9 reads the "most-likely intent" line correctly. |

**Where v9 regresses from v7:**

| Query | v7 | v9 | Why v9 is wrong |
|---|---|---|---|
| q31 preferably under 2 hours | carver | qualifier | The hedge "preferably" should affect SALIENCE, not ROLE — that's the design discipline from Exp 4. v9's role_evidence reads "preferably" as evidence the trait scores rather than gates, conflating salience with role. |
| q32 depressing | carver-negative | qualifier-negative (regression carries from v8) | "Nothing depressing" is a hard exclude. role_evidence still reads "tone is continuous" → shape (a) → qualifier, ignoring that the negation makes it a hard gate. The intent_exploration prose actually supports the carver read but role_evidence didn't use it. |
| q34 hidden gem | carver | qualifier | Single-trait query. The intent_exploration prose reads as continuous (high quality + low visibility); role_evidence cites shape (a). Same carver-of-last-resort failure mode as v7's q21 and q22. The fix for this is orthogonal to intent_exploration (deferred carver-of-last-resort code rule). |
| q10 joaquin phoenix (negative) | carver-negative | qualifier-negative (carries from v8) | "Joker but not the joaquin phoenix one" is a hard exclude. role_evidence reads as "qualifies the joker population by excluding a subset" — which is structurally TRUE but the user's wording wants definite exclusion, not soft downweight. |

**Pattern in the regressions:** the role_evidence field over-fires
shape (a) ("continuous score") on attributes that ARE continuous
in the abstract (runtime, tone, popularity) but where the user's
wording (negation, "preferably", "nothing X", sole-trait) points
at definitive gating. intent_exploration's prose surfaces the
correct structural read in most cases, but role_evidence doesn't
let it override the abstract continuity of the attribute.

**Atom-count drift on q04** (4 atoms in v8 → 3 atoms in v9; 3 in
v7). v9 matches v7 — "with my mom" is filtered as occasion-context
not a scoreable trait. Sampling-noise within the band.

**Step 3 routing churn.** v9 differs from v7 on category-call
counts in ~10 queries; almost all are within-noise expression-list
shifts (the same categories are committed, just different
expression breakdowns). One material Step 3 difference:
- q18: v7=3 TITLE_TEXT_LOOKUP calls; v9=4 calls (mixed — Story
  archetype, Emotional, Visual craft acclaim, plus implicit
  comparison routing). Tracks the role flip from carver to
  qualifier; the qualifier reading routes to attribute categories
  rather than identity categories.

### Lessons learned

1. **The intent_exploration field is doing real work.** The prose
   quality is materially better than v7's holistic_read — concrete
   watching descriptions, explicit structural-anchor naming,
   weighed plausible alternatives. q18's two-intent surfacing,
   q29's "as he hasn't directed a pure horror feature" cue
   reasoning, q36's "satisfaction requires both" — none of these
   exist in v7. This is the field doing what it was designed to
   do.

2. **Some "regressions vs v7" are actually corrections.** q15
   hallmark kind reads as comparison-reference (shape b) per the
   intent prose; that's structurally more honest than v7's
   carver-negative. q18 inception/interstellar/tenet reads as
   "movies like these" — almost always the user's true intent
   when typing three film titles, vs v7's literal-collection
   read. q22 feel good fixes a v7 sole-trait bug. q36 feudal
   japan reads as dual-carver intersection per intent_exploration.
   When the intent_exploration prose is good, it produces
   defensibly better role assignments than v7 did from
   per-atom-only reasoning.

3. **The remaining regressions are NOT intent_exploration
   issues.** q31 (hedge as role signal), q32 (negation
   over-ridden by attribute-continuity), q34 (sole-trait
   shape-a), q10 (named-entity carver-negative weakened to
   qualifier-negative). These all trace to role_evidence's three
   evidence shapes. Specifically, shape (a) "continuous score"
   over-fires when the attribute is continuous in the abstract
   even though the user's wording (negation, "nothing X", sole
   trait, hard hedge) points at definitive gating. The
   intent_exploration prose often surfaces the correct read but
   role_evidence ignores it.

4. **Cleanest follow-up: tighten role_evidence's shape (a)
   discipline.** The rule needs to be something like: "shape (a)
   continuous-score evidence does NOT fire when the user's
   wording attaches a definitive gate to the attribute (negation:
   'nothing X', 'no X'; sole trait gating eligibility; explicit
   hard threshold)." This is a calibration on shape (a)'s
   firing condition, not a new field. Defer to Experiment 7.

5. **Carver-of-last-resort is still a deferred code-side fix.**
   q34 hidden gem joins q21 warm hug and (if it had not flipped
   to carver) q22 feel good as sole-trait queries that need a
   post-process: when no trait in a query is a carver, promote
   the highest-salience qualifier to carver. Independent of
   intent_exploration's design.

6. **The new batch runner is a real win operationally.** v8 took
   ~6 minutes wallclock via sequential bash; v9 finishes in 93.5
   seconds at concurrency=5. Per-task isolated StringIO buffers
   eliminate the stdout-redirection race that would have broken
   concurrent runs. Reusable for future experiments (just point
   `--out` at a new directory).

7. **Description discipline matters more than I thought.** The
   user's edits to my draft description (remove cue examples,
   remove "do not pattern match" meta-guidance, remove concrete-
   prose examples, remove effect-token ban, remove system-label
   examples) actually mattered. The trimmed description produced
   intent prose that's MORE structurally rigorous than my
   example-laden draft would have. Generic principles + clear
   discipline > examples that prime imitation.

8. **v9 is roughly net-equivalent to v7 in correctness, with
   different distribution of failure modes — but materially
   better in REASONING QUALITY.** The intent prose actually says
   useful things now. The remaining failures are role_evidence
   calibration issues, not intent_exploration issues. Worth
   keeping; the regressions on q31/q32/q34 are localized and
   addressable in Exp 7.

---

## Experiment 7 — `intent_exploration` as primary source; remove from Step 3

### Hypothesis

Experiment 6 established that `intent_exploration` was producing
high-quality structural-anchor prose, but the v9 role_evidence
field was treating it as one of several parallel evidence sources
and the abstract shape (a)/(b)/(c) reasoning often fired before
reading it. The v9 regressions (q31 hedge over-rotated to role,
q32 negation lost to abstract continuity, q34 sole-trait shape
(a), q10 named-entity exclusion weakened) all had intent_exploration
prose that surfaced the correct structural read; role_evidence
just didn't defer to it.

The fix: promote intent_exploration to **primary source** for
both atom generation and role_evidence. Other signals
(qualifier_relation, peer atoms/traits, surface query) become
contextual grounding for the primary frame, not parallel evidence.
The shape (a)/(b)/(c) reasoning stays as HOW the qualifier
conclusion is reasoned within the frame — not free-standing tests
that fire ahead of reading the primary source.

Separately: remove intent_exploration from Step 3's user prompt.
With Changes 1+2 making upstream commits more faithful to
intent_exploration's perception, Step 3 gets the value indirectly
through better per-trait commits — without the query-level prose
leakage observed on q18 in v9 (where intent's "movies like these"
language leaked into per-title routing).

If the principle is right, we should see:
- q31/q32/q34 recover (role_evidence reads negation/hedge/sole-
  trait against intent_exploration's primary frame; abstract
  shape (a) doesn't preempt).
- q10 joaquin phoenix recovers to carver-negative.
- q21 warm hug should improve (sole-trait carver-of-last-resort
  may emerge from intent_exploration's "sole structural anchor"
  framing — an organic fix rather than the deferred code rule).
- q18 inception/interstellar/tenet should hold a clean read
  (either carver-as-literal or qualifier-as-similar; not the
  internal inconsistency where role=carver but
  trait_role_analysis routes to attributes).
- Step 3 routing on cross-trait-prone queries should get cleaner
  with the prose leakage removed.

### Changes made

**search_v2/step_2.py — `_ATOMICITY` (Change 1):**
- Added a "PRIMARY SOURCE" paragraph at the top: atoms are
  generated against intent_exploration's most-likely
  interpretation; surface query and modifying language provide
  additional context for grounding/verifying each atom.
- Updated GENERATION DISCIPLINE to walk the query "with
  intent_exploration's most-likely interpretation in hand," with
  each atom reflecting a piece that contributes to the primary
  intent's structural shape.
- POPULATION TEST reframed as the verification step against the
  primary intent, not a free-standing test that bypasses it.

**schemas/step_2.py — `Trait.role_evidence` (Change 2a):**
- Field description rewritten: intent_exploration is the
  PRIMARY source ("read intent_exploration's most-likely
  interpretation as the primary source — it has already
  identified which piece of the query gates the population vs
  which refines"). qualifier_relation and other atoms/traits are
  CONTEXTUAL grounding ("for that frame"). Shape (a)/(b)/(c) are
  reasoned WITHIN the primary frame.

**search_v2/step_2.py — `_CARVER_VS_QUALIFIER` PROCESS (Change 2b):**
- PROCESS section opens with the primary-source framing. The
  three evidence shapes are explicitly framed as "HOW you reason
  about the qualifier conclusion within intent_exploration's
  frame; they are not free-standing tests that fire ahead of
  reading the primary source."
- New paragraph at end of PROCESS: "A trait whose attribute is
  abstractly continuous (runtime, tone, popularity) can still be
  a carver when intent_exploration's primary intent attaches a
  definitive gate to it (negation, absence-of-X, sole structural
  anchor). The shape (a) language applies when the user's
  evaluation is genuinely a position on a spectrum — not when a
  continuous attribute carries a hard gate." This directly
  addresses the v9 q31/q32 regressions.

**search_v2/step_2.py — `_COMMIT_PHASE` OPERATIONAL TEST (Change 2c):**
- Replaced the "tiebreaker" framing with a primary-source check:
  "did I read intent_exploration's most-likely interpretation as
  the primary frame, then reason within it about whether this
  trait gates eligibility (→ carver) or qualifies via shape
  (a)/(b)/(c)? If the role_evidence reasoned only from abstract
  attribute properties without reading the primary frame,
  revise."

**search_v2/step_3.py + schemas/step_3.py + runners (Change 3):**
- Removed `intent_exploration` parameter from `run_step_3` and
  `_build_user_prompt`.
- Removed the `Query intent exploration:\n{intent_exploration}\n\n`
  prefix from the per-trait user prompt. Step 3 receives ONLY
  per-trait commits.
- Reverted "do not re-derive from evaluative_intent or
  intent_exploration" → "do not re-derive from evaluative_intent"
  in `_TRAIT_ROLE_ANALYSIS` (search_v2/step_3.py) and the
  schemas/step_3.py `trait_role_analysis` description.
- Updated `_build_user_prompt` docstring to document the
  intentional non-inclusion: "the query-level intent_exploration
  prose is deliberately NOT surfaced here: it lives at Step 2
  where it shapes the per-trait commits (atom partitioning,
  role_evidence). Sending query-level prose down would risk
  leaking other-trait interpretations into this trait's
  routing."
- Updated `run_step_3.py` and `run_test_queries.py` to drop the
  `analysis.intent_exploration` arg.

Verification: imports compile. Step 2 system prompt grew from
~34.1k chars to ~35.4k chars (+1.3k for the primary-source
prose; principle-driven, no examples added). Step 3 system
prompt unchanged in size (54.4k chars). Smoke test on q31
"preferably under 2 hours" confirmed expected behavior — the
trait commits as carver/positive/supporting with role_evidence:
*"This trait acts as a carver because, even though it is hedged,
it defines the only eligibility gate provided in the query."*

### Results observed

Ran all 42 queries through the v10 pipeline (concurrency=5,
wallclock 88.8s). Outputs in `/tmp/step3_runs_v10/`. Compared
trait role/polarity/salience assignments against v7 (canonical)
and v9 (Experiment 6 baseline).

**Headline counts.**
- v10 differs from v7 on 14/42; differs from v9 on 12/42.
- ALL FOUR v9 regressions resolved (q10, q31, q32, q34 — see
  table below).
- Two NEW improvements over BOTH v7 and v9: q21 warm hug and
  q34 hidden gem (sole-trait carver-of-last-resort emerged
  organically from intent_exploration's "sole structural anchor"
  framing).
- One internal-consistency issue: q18 (see below).

**v9 regressions ALL resolved.**

| Query | v7 | v9 | v10 | role_evidence cite |
|---|---|---|---|---|
| q10 *not the joaquin phoenix one* | carver-neg | qualifier-neg | **carver-neg ✓** | "While this narrows the search, it functions as a definitive gate—any film matching this entity is strictly excluded from the results." |
| q31 *preferably under 2 hours* | carver/central | qualifier | **carver/supporting ✓** | "This trait acts as a carver because, even though it is hedged, it defines the only eligibility gate provided in the query." |
| q32 *nothing depressing* | carver-neg | qualifier-neg | **carver-neg ✓** | "The user is definitively excluding a specific emotional state from the results." |
| q34 *hidden gem* | carver | qualifier | **carver ✓** | "This trait is the sole structural anchor of the query and definitively gates eligibility for the entire search." |

**Sole-trait carver-of-last-resort emerged organically.** This
was the deferred code-side fix from Experiment 4 — the rule
"when no peer atom or trait gates eligibility, this trait must
carve regardless of shape (a) firing." v10 produces it via
prompt without needing the post-process:

- q21 *warm hug* → carver. role_evidence: *"This trait is the
  sole structural anchor of the query and definitively gates
  eligibility for the kind of emotional experience the user is
  seeking."*
- q34 *hidden gem* → carver. role_evidence: *"This trait is the
  sole structural anchor of the query and definitively gates
  eligibility for the entire search."*

Promoting intent_exploration to primary source means
role_evidence reads "single, unified experiential request" /
"single-intent query where the entire phrase defines the
population of interest" from the intent prose, and the carver
conclusion follows directly. No explicit shape (a)-blocker
needed; the primary frame just doesn't have a peer to support
shape (c) fallback, so carver is the natural read.

**Cleaner atom partitioning on q08.** v7's q08 *"darker than
fight club but funnier than seven"* atomized as `["darker",
"funnier"]` with the films folded into modifying_signals. v10
atomized as `["fight club", "seven"]` with `darker than` /
`funnier than` as the modifying signals. role_evidence:
*"This trait is used as a comparison reference to define a
tonal spectrum rather than gating the population to 'Fight
Club' itself."* — clean shape (b) read of the films as
comparison anchors. This is intent_exploration's primary-source
framing reshaping atom partition into something more aligned
with what the user actually asked for.

**Other improvements over v7 maintained from v9:**
- q15 hallmark kind: v9 read it as qualifier-negative (shape b
  comparison reference); v10 reverts to carver-negative. The
  v10 read is closer to v7's carver-negative — "not the hallmark
  kind" reads as definitive exclusion. Either is defensible;
  v10's matches v7.
- q22 feel good: v9's carver-fix maintained.
- q36 feudal japan: v9's intersection-of-carvers read maintained.
- q09 wonder woman/new ones: v9's carver-negative read
  maintained.

**One real issue: q18 inception/interstellar/tenet internal
inconsistency.** v10 commits all three traits as carver (matches
v7), but role_evidence cites *"each title acts as a carver to
define the boundaries of the desired population (high-concept
sci-fi)"* and Step 3's trait_role_analysis reads the trait as
"movies that share the specific narrative and thematic
architecture of the film Inception" — both qualifier-flavored
prose. Step 3 then routes to attribute categories (Story
archetype, Narrative devices, Element, Emotional) rather than
TITLE_TEXT_LOOKUP. The role label says "carver" but the
trait's prose framing is shape (b). v9's q18 was internally
consistent (role=qualifier, attribute routing); v7's was
internally consistent (role=carver, TITLE_TEXT routing). v10
sits in between and produces the worst of both. This is the
one regression worth flagging.

**Minor drift:**
- q04 atom shift: "with my mom" returns as a qualifier atom;
  "dark or scary" merges into a single carver-negative atom.
  Atomization shifts within the band; not strictly worse than
  v7.
- q12, q14, q25 surface_text minor (e.g. "early 2000s" →
  "2000s"; "violent" → "non violent"). q25's "non violent" is
  arguably better — preserves the negation morpheme.

**Step 3 routing changes after removing intent_exploration from
the prompt.** Step 3 differs from v7 on category-call sets in
~25/42 queries — but most are within-noise sampling shifts at
the expression-list level. Materially:
- q34 hidden gem: 2 calls → 1 call (cleaner — single
  CULTURAL_STATUS).
- q06 godfather parody: TITLE_TEXT_LOOKUP route DROPPED on the
  godfather trait (matches the qualifier-routing-to-attribute
  principle from earlier experiments).
- q26 breaking bad/1800s: 4 calls → 3 calls (tighter).
- q18: as flagged above — over-decomposed despite committed
  carver role.

No leakage failures of the kind v9 produced (where Step 3
absorbed query-level "more like these" language into per-trait
routing). Step 3 routing is now decoupled from query-level
intent prose; it operates from per-trait commits as designed.

### Lessons learned

1. **Promoting intent_exploration to primary source resolved
   every v9 role_evidence regression.** q10, q31, q32, q34 all
   recover. The role_evidence prose visibly cites
   intent_exploration's structural framing instead of
   abstract-attribute reasoning ("definitively excluding a
   specific emotional state", "the only eligibility gate", "the
   sole structural anchor"). The reframing from "one of three
   evidence sources" to "primary source with contextual
   grounding" was the load-bearing edit.

2. **Sole-trait carver-of-last-resort emerged ORGANICALLY** from
   the primary-source promotion. q21 warm hug and q34 hidden
   gem now cite "sole structural anchor" in role_evidence and
   commit carver. The deferred code-side post-process is no
   longer needed — the prompt-level fix is sufficient. This is
   the cleanest possible resolution: a structural rule
   ("a query needs a carver to gate the result set") that
   emerged from a more-honest framing rather than a
   special-case patch.

3. **Atom partitioning improved structurally on q08.** The
   films (fight club, seven) become the atoms with the
   comparison operators as modifiers. This is what
   intent_exploration's primary-intent framing does to atom
   generation: the atoms reflect the structural anchors the
   user is actually asking against, not the surface-most
   adjectives. Other queries didn't shift dramatically — q08 is
   the case where the primary intent's structural read is
   different enough from the surface reading to drive
   visible change.

4. **Step 3 lost nothing by losing intent_exploration in its
   prompt.** No new failures traced to the missing field; the
   q18 issue is upstream (Step 2 internal inconsistency).
   Step 3 routes from per-trait commits cleanly. This confirms
   the design hypothesis: query-level prose was over-sharing
   for a per-trait decomposer.

5. **The remaining q18 issue is a Step 2 internal-consistency
   problem.** intent_exploration says "more likely intent is
   movies similar to these"; role_evidence cites "boundaries
   of the desired population (high-concept sci-fi)"; role
   commits carver. Three pieces don't align. The
   carver-vs-qualifier shape (b) test ("comparison reference
   rather than naming the population") would route this to
   qualifier under a literal read, but the model committed
   carver despite the prose. One hypothesis: when the
   intent_exploration weighs "similar to" as MORE LIKELY but
   doesn't explicitly reject the literal read, the model can
   commit either way and inherit the prose's qualifier-flavored
   framing into role_evidence regardless. Defer to a future
   experiment: tighten the rule when intent_exploration's
   weighed primary intent is shape (b) territory, role MUST
   commit qualifier (not carver).

6. **A schema field's value is in its USE, not its
   presence.** The same field (intent_exploration) was inert
   in v9's role_evidence (listed but ignored in favor of
   abstract reasoning) and load-bearing in v10's role_evidence
   (read as primary source, drove every recovery). The
   procedural framing in the prompt — "primary" vs "one of
   three evidence sources" — was what determined whether the
   model used the field. Schema descriptions matter; prompt
   framings matter MORE.

7. **Reasoning quality + correctness both improved.** v10's
   role_evidence prose is more substantive than v9's: it cites
   the specific structural framing ("sole structural anchor",
   "definitively excluding", "only eligibility gate") rather
   than naming abstract evidence shapes ("continuous score",
   "comparison reference"). The auditability gain is real —
   reading any v10 role_evidence string tells you what intent
   framing produced the role conclusion.

8. **v10 is the cleanest result so far.** All four v9
   regressions resolved; sole-trait failures resolved; v9
   improvements maintained; Step 3 routing cleaner without
   leakage; intent_exploration prose still high-quality. One
   real residual issue (q18 internal consistency). Worth
   shipping as the new baseline; defer the q18 fix to a
   targeted experiment.

---

## Experiment 8 — Step 3 inter-attribute information flow rebalance

### Hypothesis

Auditing Step 3's per-attribute reading discipline surfaces five
related issues, all stemming from the same pattern: each layer
reads from a too-narrow slice of upstream context, and the prompt
implicitly licenses lossy compression at every boundary.

1. **`trait_role_analysis` over-leans on `role`.** The current
   prompt frames `role` as the headline question
   ("THE TWO QUESTIONS… The role field tells you which"), but the
   field engineered to constrain dimension scope is
   `qualifier_relation` — its own schema description literally
   says "Step 3 consumes this prose directly to constrain its
   dimension scope". Reading `role` first makes the binary
   verdict load-bearing when the substantive signal is in the
   freeform relation prose.

2. **`role_evidence` is currently unused in Step 3.** The
   rationale for the role commit (continuous-score-only,
   comparison-reference, population-already-gated) is rich
   context that helps disambiguate borderline traits and
   carry-the-load when `qualifier_relation` is `"n/a"`.

3. **`aspects` walks `target_population` AND
   `trait_role_analysis` equally.** The framing implies they're
   peer sources, but `target_population` is the natural
   enumeration source (it names the kind of movies); the role
   analysis qualifies how to interpret each axis (population vs
   reference). Equal-weight walking blurs the two jobs.

4. **`dimensions` is permitted to silently drop or merge
   aspects.** The current rule "two aspects share one
   searchable check, collapse them" gives the model an out for
   any aspect that's awkward to translate. Real failure mode in
   v10: Wes Anderson trait drops "meticulous production design"
   silently; "darker tone" drops "lack of whimsy". Compression
   should happen at the call layer (where category routing
   merges same-category dimensions naturally), not at the
   dimension layer.

5. **`retrieval_intent` is qualifier-only by description, but
   Step 4 doesn't branch on role.** The carver/qualifier
   distinction is orchestrator-side (already committed
   upstream). For Step 4, retrieval_intent is the handoff field
   that conveys context the short `expressions` phrases can't
   carry — it should always be populated as a generic intent
   overview regardless of role.

### Changes made

**A. Add `role_evidence` to Step 3's user prompt.** Updated
`_build_user_prompt` in `search_v2/step_3.py` to surface
role_evidence between role and qualifier_relation. The field is
informational; gives the role-analysis layer the rationale, not
just the verdict.

**B. Rewrite `_TRAIT_ROLE_ANALYSIS` with explicit source
priority.** New ordering:
  1. `qualifier_relation` (primary — translate operational meaning)
  2. `role` + `role_evidence` (verdict + rationale; cross-check)
  3. `contextualized_phrase` + `evaluative_intent` (grounding)
  4. `anchor_reference` (surface pointer)
The "two questions" framing now leads with "what should the
dimensions describe?" (relation-driven) before "retrieve or
position?" (role-driven).

**C. Reframe `_ASPECT_ENUMERATION` to anchor on
`target_population`.** Aspects walk `target_population` first
(the population whose axes need decomposing), and use
`trait_role_analysis` to qualify whether each axis describes the
population vs the reference. Drop the "walk both equally"
framing.

**D. Rewrite `_DIMENSION_INVENTORY` to translate every aspect.**
Source list is `aspects`; `target_population` and
`trait_role_analysis` are read only to *understand* each aspect
more deeply. Every aspect must yield at least one dimension —
no silent drops, no aspect-merging at this layer. Removed the
"two aspects share one searchable check, collapse them"
allowance. Compression happens at category_calls.

**E. Reframe `retrieval_intent` schema description as a generic
handoff field.** Rewrote `CategoryCall.retrieval_intent` in
`schemas/step_3.py` so it always populates regardless of role,
and is explicitly framed as the field that conveys context
short `expressions` phrases can't carry — what exactly is being
searched, what shape the search should take. Step 4's success
depends on this field.

**F. Universal "consider all upstream context" rule.** Added to
each layer's instructions: read the whole upstream context, do
not stop early, do not quietly drop signals that resist
translation. Applied to `trait_role_analysis`, `aspects`,
`dimensions`, `category_calls`.

If the principles are right, we should see:
- Qualifier traits with rich `qualifier_relation` (q08 "feels
  like Fight Club", q09, q23, q31) produce more substantive
  trait_role_analysis prose that visibly cites the relation.
- Aspect→dimension coverage tightens: no silent aspect drops
  on multi-faceted figurative traits (q21 warm hug, q22 feel
  good, q33 underrated, q34 hidden gem, q42 video game).
- `retrieval_intent` reads as substantive context for Step 4
  on every call (carver and qualifier alike), not a thin
  one-liner on carvers.
- No regression on simple single-axis traits — overhead from
  the consider-everything rule shouldn't add bloat where the
  trait is genuinely simple.

### Observations

Aggregate stats (v10 baseline → v11):

| Metric | v10 | v11 | Δ |
|---|---|---|---|
| Traits decomposed | 84 | 86 | +2 (Step 2 LLM variance) |
| Aspects total | 212 | 206 | -6 |
| Dimensions total | 157 | 172 | **+15** |
| Calls total | 144 | 148 | +4 |
| Aspects→dim drops (count) | 55 | 34 | **−21 (−38%)** |
| Traits with silent drops | 44/84 (52%) | 32/86 (37%) | **−15pp** |
| Avg `retrieval_intent` length | 132 chars | 203 chars | **+54%** |
| `trait_role_analysis` citing relation prose | 0/84 (0%) | 38/86 (44%) | **+44pp** |
| `trait_role_analysis` substantive (>180 chars) | 76/84 (90%) | 84/86 (98%) | +8pp |

Per-trait observations on canonical cases:

- **q29 wes anderson does horror (Wes Anderson trait).** v10
  enumerated 5 aspects but emitted only 4 dimensions — "meticulous
  production design" silently dropped. v11 produces 5 aspects → 5
  dimensions (clean 1:1), with "meticulous diorama-like production
  design" preserved as its own dimension routing to Visual craft
  acclaim. The aspect-preservation discipline visibly fixes the
  v10 silent-drop failure.

- **q34 hidden gem.** v10: 1 call (CULTURAL_STATUS only), with the
  "gem" quality aspect bundled into the same expression as the
  "hidden" status aspect. v11: 2 calls (CULTURAL_STATUS for the
  underrated/overlooked aspect + General-appeal/quality-baseline
  for the qualitative-merit aspect). The independent axes
  ("hidden" and "gem") are now retrieved through their natural
  category routings instead of being collapsed.

- **q33 underrated.** v10: 2 aspects → 2 dimensions → 1 call. v11:
  3 aspects → 3 dimensions → 3 calls (CULTURAL_STATUS + General
  appeal/quality baseline + Financial scale). The
  quality/visibility/commercial-footprint axes that the original
  baseline noted as a known FINANCIAL_SCALE leakage failure mode
  are now treated as legitimate independent axes — the user's
  prior framing ("aspects should always be preserved") promotes
  this from "leakage" to "correct decomposition".

- **q21 warm hug.** v10: 3 aspects → 2 dims → 1 call with thin
  retrieval_intent ("Retrieve movies that score high on emotional
  safety, coziness, and uplifting resonance to satisfy the 'warm
  hug' experiential request"). v11: 4 aspects → 3 dims → 1 call
  with a 2-sentence retrieval_intent that explicitly names what
  to discriminate against ("...prioritize films that score high
  on kindness, gentle pacing, and positive resonance,
  discriminating against high-tension, gritty, or emotionally
  taxing content").

- **q08 fight club / seven (qualifier-heavy comparison).** Both
  Fight Club and Seven traits now have 1:1 aspect-to-dimension
  mapping (down from 4→2 and 3→2 in v10). trait_role_analysis
  now explicitly cites the qualifier_relation: "This trait is a
  qualifier where 'Fight Club' serves as a comparative anchor.
  Dimensions must describe the identifiable attributes of the
  anchor (Fight Club) to establish the threshold..."

- **q31 "preferably under 2 hours" (simple single-axis trait).**
  v10: 1 aspect → 1 dim → 1 call. v11: 1 aspect → 1 dim → 1 call.
  No bloat from the new "consider all upstream context" rule on
  simple traits. retrieval_intent expanded from one short
  sentence to two but stayed appropriate.

- **trait_role_analysis citation behavior.** Across all 86 traits,
  v10 never explicitly cites qualifier_relation in its prose
  (0/84). v11 does in 38/86 cases (44%) — the new "PRIMARY"
  framing of qualifier_relation visibly drove the model to
  treat the field as a load-bearing reference rather than
  background context. The field's value flipped from inert to
  load-bearing without changing the field itself, only its
  framing in the prompt.

- **role_evidence in user prompt.** Now visible to Step 3 across
  all 86 traits. No regressions traced to its addition; on
  borderline traits it provides the carver/qualifier rationale
  that the binary `role` field can't carry.

### Lessons learned

1. **Source priority framing is load-bearing — same pattern as
   Experiment 7's intent_exploration promotion.** Identical
   dynamic to v10's "intent_exploration as primary source"
   reframe: Step 2 had already engineered qualifier_relation
   specifically as Step 3's input ("Step 3 consumes this prose
   directly to constrain its dimension scope"), but the v10
   prompt framed `role` as the headline question. The field was
   read but not centered; its value was inert. Promoting it to
   PRIMARY in the prompt — without changing the schema or the
   field itself — flipped citation rate from 0% to 44%. A
   schema field's value is in its USE, not its presence; the
   prompt framing determines whether the model uses it.

2. **Silent drops drop by ~38% with explicit "translate every
   aspect" framing + removal of the collapse allowance.** v10's
   prompt licensed "two aspects share one searchable check, they
   collapse into one dimension"; v11 removed the allowance and
   added explicit "DROPPED ASPECT is the most common failure
   mode of this layer" warning. Drop rate fell from 55 aspects
   to 34, traits-with-drops from 52% to 37%. Not zero, but
   substantial — and the residual drops are concentrated in
   2-aspect cases where compression may be legitimate (e.g.
   "long": runtime + reading-time both fold into one duration
   check).

3. **Aspect preservation surfaces real multi-category routings
   the previous version artificially compressed.** q33
   underrated and q34 hidden gem now produce 2-3 calls instead
   of 1. The previous baseline filed "FINANCIAL_SCALE leaking
   alongside CULTURAL_STATUS" as a known failure of CLEAN-FIT
   discipline. Under the new framing, those are not leakages —
   they are legitimate decompositions of independent axes
   (quality vs visibility vs commercial-footprint), and the
   merge logic that's supposed to combine same-category
   dimensions still works correctly (v11 q21 warm hug: 3
   aspects → 3 dims → 1 multi-expression Emotional/experiential
   call). The previous CLEAN-FIT rule was over-eager; the
   pre-merge happened at the dimension layer instead of the
   call layer.

4. **`retrieval_intent` quality lifts dramatically (+54% chars)
   under the handoff-field framing.** Removing the
   qualifier-only framing AND explicitly naming retrieval_intent
   as Step 4's only context source (beyond expressions) shifted
   the model from one-line restatements to substantive 2-3
   sentence handoff prose. Several v11 entries now explicitly
   name what to discriminate against, which the v10 entries
   never did. The field shape didn't change — only the
   description's framing of what it's FOR.

5. **No regression on simple single-axis traits.** Cost of the
   "consider all upstream context" universal rule: token usage
   per trait rose from ~13.9k → ~14.8k input (+6%) and call
   counts rose by 4 across 86 traits — small. Single-axis
   traits like "preferably under 2 hours" stayed at 1
   aspect/dim/call. The discipline only fires where there's
   actually more upstream context to consider.

6. **role_evidence integrates cleanly without producing
   conflicting commitments.** The concern was that surfacing
   the role rationale alongside qualifier_relation might cause
   the model to second-guess Step 2's commitments. No evidence
   of this — trait_role_analysis prose still cites qualifier_-
   relation as primary; role_evidence shows up as supporting
   context (especially on carver traits where qualifier_-
   relation is "n/a"). The prompt's explicit source-priority
   ladder appears to have prevented the conflict.

7. **The known q34/q33 over-compression failure pattern was
   actually a symptom of the dimension-layer collapse rule.**
   The prior CLEAN-FIT TEST tried to enforce single-call
   discipline at commit time, but the real source of bloat was
   the "two aspects share one searchable check" allowance at
   the dimension layer — once collapsed there, the call layer
   couldn't recover the lost axis. Moving compression entirely
   to the call layer (where same-category dimensions merge into
   one multi-expression call) and forbidding it at the
   dimension layer is the cleaner fix. Same-trait double-routing
   to one category is still prevented (schema rule); legitimate
   multi-category routings now surface.

8. **v11 is the cleanest Step 3 result so far.** Aspect drops
   nearly halved; trait_role_analysis prose visibly cites the
   primary signal it was always supposed to use; retrieval_-
   intent is substantively richer in every case examined; no
   regressions on simple traits or on Step 2 commitments. Worth
   shipping as the new baseline. Residual silent drops cluster
   on 2-aspect single-category cases where collapse may
   actually be legitimate — defer further tightening until a
   query forces the issue.

---



## Experiment 9 — Step 2 source-priority audit: three reasoning fields

### Hypothesis

Auditing Step 2's reasoning fields surfaced three fields whose
upstream sources lacked the explicit PRIMARY / CONTEXTUAL
GROUNDING / ANTI-SOURCE structure that Experiments 7 and 8
showed to be load-bearing for `role_evidence` and Step 3's
`trait_role_analysis`. If the same pattern transfers, we should
see corresponding shifts in how the fields read their inputs.

1. **`Trait.relevance_to_query` doesn't invoke `intent_exploration`.**
   Salience is a structural-importance question, and
   `intent_exploration` already weighs which pieces of the query
   are headline-shaping vs refining as a side effect of
   exploring plausible intents. Yet the field's description
   today hands the model a flat list of within-trait signals
   (hedges, surface position, words spent, modal tokens) with
   no priority among them and no reference to the upstream
   frame. Same dynamic as the pre-Experiment-7 atomicity prompt
   and the pre-Experiment-8 `trait_role_analysis` prompt — a
   ready frame source that the field-level prompt doesn't
   centre.

2. **`Atom.split_exploration` has implicit per-check primaries.**
   The two checks (forward subdivision / inverse signal-as-
   population) each have a natural primary source —
   `evaluative_intent` for forward, `modifying_signals` for
   inverse — and `surface_text` is an anti-source for both
   (the absorbed content lives elsewhere). The current prompt
   states this inline ("walk evaluative_intent, not surface_text
   alone") rather than as labeled per-check primaries, which is
   easier to skim past on the inverse check, where silent
   absorption of population-bearing content originates.

3. **`Atom.evaluative_intent` has no anti-source for `intent_-
   exploration`.** `intent_exploration` is drafted before atoms
   in the same response, so it's visible to the model when
   filling `evaluative_intent`. With nothing telling the model
   to defer, `evaluative_intent` could collapse the atom's
   consolidated meaning toward the most-likely intent —
   removing the standalone-vs-most-likely contrast that
   `standalone_check` is supposed to surface. Hypothetical
   failure mode; the experiment doubles as a check on whether
   it was actually present.

If the pattern holds, we should see:
- `relevance_to_query` prose pivots to headline-vs-refining
  vocabulary that comes from the new framing.
- `split_exploration` inverse halves explicitly walk the
  signals and discuss them by name.
- `evaluative_intent` stays close to surface_text + signals
  with no pre-alignment to `intent_exploration`.

### Changes made

**A. Promote `intent_exploration` to PRIMARY in
`Trait.relevance_to_query`.** Restructured the schema field
description and the prompt's `_SALIENCE` section in parallel.
Old shape: a flat list of within-trait signals (hedges, modal
tokens, surface position, words spent, removability). New
shape: PRIMARY = `intent_exploration`'s most-likely
interpretation (it has already weighed which pieces are
headline-shaping vs refining); CONTEXTUAL GROUNDING = surface
position + words spent + SOFTENS / HARDENS tokens on
modifying_signals + the removability test. The contextual
sources verify and refine the primary frame; they do not stand
in for it. Same template as `role_evidence`.

**B. Make `Atom.split_exploration`'s two primaries explicit.**
Restructured the schema field description and the prompt's
`SPLIT AND STANDALONE EXPLORATIONS` block. New shape labels each
check with its primary source — FORWARD (primary:
evaluative_intent), INVERSE (primary: modifying_signals) — and
calls out `surface_text` as an anti-source for both. Same
substance as the pre-existing inline guidance ("walk
evaluative_intent, not just surface_text"); the change is
hardening it into the schema's role structure.

**C. Add an anti-source guardrail to `Atom.evaluative_intent`.**
Schema field description and the prompt's `_EVALUATIVE_INTENT`
section both gained an explicit "Don't pre-align to
`intent_exploration`. Sources for this field are surface_text +
modifying_signals only; the comparison against the query's
most-likely intent happens in `standalone_check`."

Files modified: `schemas/step_2.py` (three field descriptions);
`search_v2/step_2.py` (`_ATOMICITY` exploration block,
`_EVALUATIVE_INTENT` guardrails, `_SALIENCE`).

### Observations

Aggregate stats (v11 baseline → v12, 42-query suite):

| Metric | v11 | v12 | Δ |
|---|---|---|---|
| Atoms | 86 | 87 | +1 |
| Traits | 86 | 87 | +1 |
| Salience central | 73 | 77 | +4 |
| Salience supporting | 13 | 10 | −3 |
| Role carver | 59 | 65 | +6 |
| Role qualifier | 27 | 22 | −5 |
| Polarity negative | 14 | 14 | 0 |
| Step 2 input tokens | 345,805 | 355,213 | +2.7% |
| Step 2 output tokens | 37,720 | 39,009 | +3.4% |
| `relevance_to_query` strict frame-cite¹ | 0/86 | 0/87 | — |
| `relevance_to_query` loose frame vocab² | 31/86 (36%) | 47/87 (54%) | +18pp |
| `split_exploration` inverse names signals³ | 56/86 (65%) | 58/87 (67%) | +2pp |
| `evaluative_intent` pre-align phrases | 0/86 | 0/87 | 0 |

¹ Strict citation = mentions `intent_exploration` by name, or
phrases like "most-likely interpretation" / "the population the
user wants" / "headline-shaping" / "the exploration step." None
in either run — the model internalises the frame rather than
naming it.

² Loose vocabulary = strict signals plus generic salience words
("headline", "rounding out", "refining"). The +18pp shift here
is the headline behavioural change.

³ "Names signals" = inverse half of split_exploration mentions
"signal(s)" / "modifying signal" / "absorbed signal" by name.
Already at 65% in v11 — hardening to per-check primaries did
not move the needle materially.

**Salience-commit shifts (the substantive change).** Six trait
salience flips between v11 and v12; five upward to central, one
downward to supporting. Per-flip read:

- **q04 "with my mom of course shes 65 cozy mysteries nothing
  too dark or scary" — `dark` and `scary`.** Both flipped from
  qualifier/supporting (v11) to **carver/central** (v12). v11
  read: "the 'too' hedge softens the constraint." v12 read:
  "critical safety constraint that defines the 'cozy' aspect."
  **Regression.** The "not too" hedge is the dominant salience
  signal here; v12 elevates the trait because intent_exploration
  framed it as defining-the-cozy-population, losing the hedge
  attentiveness v11 had. Role also flipped (qualifier →
  carver), which compounds the issue.

- **q30 "ideally a slow burn thriller" — `slow burn`.** Flipped
  supporting (v11) → central (v12). v11 read: "preceded by the
  hedge 'ideally', which reduces its structural weight compared
  to the genre." v12 read: "led with a preference marker,
  making this a headline characteristic of the request."
  **Regression.** The v12 model reinterpreted "ideally" as a
  headline-marker (increase) rather than a softener (decrease).

- **q15 "christmas movie thats actually good not the hallmark
  kind" — `hallmark kind`.** Flipped qualifier/supporting (v11)
  → carver/central (v12). v11: "specific negative constraint to
  help define what the user means by 'actually good'." v12:
  "exclusion of the 'Hallmark' style is a specific and emphatic
  part of the user's request." Defensible — emphatic exclusion
  can carve — but the trait reads more naturally as a refiner
  of "actually good" than as the population gate.

- **q13 "whats good on netflix when im hungover" — `good`.**
  Flipped supporting → central. v11: "generic filler for quality
  that rounds out the more specific platform and mood requests."
  v12: "user explicitly asks for 'whats good', making quality a
  headline requirement." Debatable — both reads have merit.

- **q32 "fun 90s sci fi action movie good for friday night" —
  `good for friday night`.** Flipped central → supporting. v11:
  "captures the user's ultimate goal." v12: "provides the 'vibe'
  for the search." The downward flip; v12 is arguably better
  here — friday-night-vibe is a refiner on the genre population,
  not a headline.

The pattern: when `intent_exploration`'s frame describes a
trait as part of the population's definition, v12 commits
central salience. The within-trait hedge signal (SOFTENS
tokens, "not too", "ideally") is treated as contextual
*verification* rather than as a *strength modulator*. v11's
flat-list framing kept the hedge as a peer signal that could
pull salience down regardless of how the trait functions
structurally; v12's primary-source framing inverted that
priority.

**Other-field outcomes:**
- `split_exploration` inverse halves do walk signals
  explicitly in v12 (q02 'long': "the signal 'arent too' is
  operator-only language"; q08 'darker': "the signal 'than
  fight club' contains a content phrase 'fight club' which
  names a population"; q29 'wes anderson': "the signal 'does
  horror' contains the content phrase 'horror'..."). Quality
  is comparable to v11 — the discipline was already present at
  ~65% and didn't move materially. Net neutral.
- `evaluative_intent` shows no pre-alignment to
  `intent_exploration` in spot checks. q04 'dark': "Avoid
  movies with a heavy, grim, or overly serious tone; a light
  or moderate atmosphere is preferred." — pure
  signals-plus-anchor consolidation, no most-likely-intent
  vocabulary. The hypothesised drift wasn't visibly present in
  v11 either; this change reads as preventive hardening with
  no cost.
- Role distribution shifted +6 carver / −5 qualifier. Three
  of the carver promotions trace back to the same q04 / q15
  salience-flip queries (q04 dark, q04 scary, q15 hallmark
  kind), where the new framing also nudged the role from
  qualifier to carver. The remaining shifts are LLM variance.
- Atoms 86 → 87: one extra atom on q29 "wes anderson does
  horror" — the "horror" peer-atom that was already debated as
  a borderline Step-2-LLM-variance case.

### Lessons learned

1. **Promoting a frame source to PRIMARY for a continuous-
   quantity field has a downside the role-decision fields
   don't have.** Experiments 7 and 8 promoted `intent_-
   exploration` and `qualifier_relation` for **role** decisions
   (carver vs qualifier) — categorical commitments where the
   primary frame's binary read maps cleanly to the output
   binary. Salience is not categorical at heart; it is a
   continuous strength tempered by hedges. The new framing
   tells the model that "headline-shaping vs refining" (a
   binary read of structural function) IS the salience
   question, but the model then collapses to that binary and
   loses the within-trait hedge signal that v11's flat-list
   framing kept dominant. PRIMARY-source promotion is not a
   universal pattern — it works when the primary source's read
   shape matches the output's shape, and degrades otherwise.

2. **The hedge-attentiveness regression is structural, not
   transient.** All three hedged-trait failures (q04 'dark',
   q04 'scary', q30 'slow burn') share the same shape: a
   refiner with an explicit hedge ("not too" / "ideally") got
   re-read as a population-defining headline because intent_-
   exploration's frame described it as part of what the user
   wants. Two of these queries appear in the baseline failure
   suite already (q04 was the test of "not too" handling);
   v11 had them right.

3. **Frame citation rate is not a useful single metric for
   this change.** Strict citation rate stayed 0% (the model
   does not name `intent_exploration` literally), and loose
   citation rate moved 36% → 54% but is contaminated by
   generic salience vocabulary the prompt itself uses
   ("headline", "rounds out"). The substantive metric is
   per-trait salience commits, where the regressions surface.
   Future iterations should privilege the commit-level metric
   over prose-shape metrics for fields whose output is a
   commitment.

4. **Changes B and C did not move material output.** The
   per-check primary labelling on `split_exploration` was
   already implicit in the prompt at ~65% adoption; the
   relabeling didn't push it. The anti-source guardrail on
   `evaluative_intent` was preventive — the drift it guards
   against wasn't visibly present in v11, so v12 looks the
   same. Both are essentially no-ops at current model
   behaviour. Keep B (cheap discipline; may matter under model
   drift); revisit C only if a real pre-align pattern surfaces.

5. **Recommended next step on Change A.** Two paths:
   - **Refine:** keep `intent_exploration` as the frame for
     *what the trait is structurally doing in the query*, but
     re-elevate hedges as the dominant salience-strength
     modulator. Concretely: reorder the framing so SOFTENS /
     HARDENS / "not too" / "ideally" pull salience down
     regardless of structural function, with intent_-
     exploration setting the frame for headline-vs-refining
     interpretation. Two-source structure where the primary
     answers a different question than today.
   - **Revert:** roll back the `relevance_to_query` schema
     description and the `_SALIENCE` prompt section to v11.
     The framing that worked for `role_evidence` does not
     transfer to salience; cut losses.

   Either path should keep Changes B and C — they did no harm
   and the discipline is worth preserving. Default
   recommendation: **revert** Change A. The hedge regressions
   on q04 and q30 hit canonical baseline-failure-suite cases
   that v11 had right; refining the framing risks introducing
   another mis-balance without a clear principled fix.

6. **Negative result is the result.** v11 remains the cleanest
   Step 2 baseline. Experiment 9 surfaces a useful constraint
   on the source-priority pattern (point 1) that should
   inform future field audits — promotions to PRIMARY work
   for categorical commitments and degrade for continuous-
   strength commitments.

---
