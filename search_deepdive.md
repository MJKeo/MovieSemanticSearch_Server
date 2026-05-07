# Search Pipeline Deep Dive — Where Decomposition and Recombination Disagree

A diagnostic pass on the live Step 2 / Step 3 / Stage-4 stack as it
exists in the codebase today, focused on the contradictions you flagged
in your notes. Source-of-truth files inspected:

- [step_2.py](search_v2/step_2.py),
  [step_3.py](search_v2/step_3.py),
  [run_step_3.py](search_v2/run_step_3.py)
- [step_2 schemas](schemas/step_2.py),
  [step_3 schemas](schemas/step_3.py),
  [trait_category.py](schemas/trait_category.py)
- [stage_4_execution.py](search_v2/stage_4_execution.py),
  [full_pipeline_orchestrator.py](search_v2/full_pipeline_orchestrator.py)
- [rescore_overhaul.md](search_improvement_planning/rescore_overhaul.md)
  (planning context only)

## Test plan

Queries selected to exercise each pattern in your notes. All run via
`python -m search_v2.run_step_3 "<query>"` (Step 2 → Step 3 fan-out).

| # | Query | What it tests |
|---|---|---|
| 1 | `shitty shark movies` | Trait bleeding at Step 2; Step 3 over-generation on vague aesthetic |
| 2 | `bad bitch energy` | Compound concept under across-category MAX |
| 3 | `bro movie` | Compound concept; multi-category multiplicative implications |
| 4 | `horror not too gory` | Soft-negative polarity; over-generation control |
| 5 | `like zathura with jungles` | Positioning ("Like X by Y") + scope replacement |
| 6 | `dark gritty marvel movies` | Cross-trait scoping (control from rescore doc) |
| 7 | `feel-good Christmas movies` | Clean parallel-filter (control) |

Raw traces captured in `/tmp/deepdive/{shark,badbitch,bro,horror,zathura,marvel,xmas}.txt`.

## Where the system succeeds

- **Polarity + commitment commits cleanly.** Across all 7 queries the
  commit phase reads the explicit channel correctly. `not too gory` →
  `polarity=negative + commitment=diminished` ([horror.txt:46-47](#)),
  exactly the "softened preference against" the schema describes.
  Headlines commit `elevated`; co-equals commit `neutral`; `with`
  trailing-refinement commits `supporting` ([zathura.txt:59](#)).
  `commitment_evidence` always names which channel is firing — no
  default-fills observed.
- **Atomicity (the population test) works on simple cases.** `feel-good`
  and `Christmas` peer-atom cleanly ([xmas.txt:9-21](#)).
  `horror` / `gory` peer-atom cleanly with the "not too" hedge
  absorbing as operator-only ([horror.txt:17-22](#)). `with` is
  recognized as a binder in the Zathura trace.
- **Step 3 single-axis traits route precisely.** `horror` → 1 aspect →
  1 dimension → 1 GENRE call. `Christmas` → 1 aspect → 1 dimension →
  1 SEASONAL_HOLIDAY call. `shark movies` → 1 aspect → 1 ELEMENT_PRESENCE
  call. The decomposition pipeline does not over-fire when the trait is
  genuinely simple.
- **`contextualized_phrase` re-attaches positioning context.** `like
  zathura` and `with jungles` carry their operators into Step 3 ahead of
  bare surface text, which prevents Step 3 from treating Zathura as a
  literal title lookup.
- **The rescore doc's negative-trait gate × fuzzy formula is preserved
  in code** ([stage_4_execution.py:921](search_v2/stage_4_execution.py#L921)).
  The authoritative-vs-evidential split is implemented; soft hedges
  carry through correctly.

## Where the system fails

The seven failure patterns below were observed against the traces; each
points at a specific code or prompt site.

### 1. Step 2 atom bleeding — `evaluative_intent` is contaminated by peer atoms

**Observed.** When two peer atoms cross-modify, the schema asks each
atom to record the cross-modifier on `modifying_signals` and to
"integrate every signal" into `evaluative_intent`. For *scoping*
relations (where the modifier changes which population's instance to
score, but not what the atom itself means), this folding produces an
intent that bakes in the OTHER atom's identity.

| Query | Atom | Bled-in evaluative_intent (excerpt) |
|---|---|---|
| `shitty shark movies` | shark movies | "specifically those that fit the **'shitty' qualitative descriptor**" |
| `dark gritty marvel movies` | marvel movies | "specifically **for the purpose of filtering them by tone**" |
| `dark gritty marvel movies` | dark gritty | "**a lack of traditional superhero optimism**" |

**Why this matters.** Step 3 reads `evaluative_intent` as the semantic
seed for `target_population` and aspects. A bled intent silently
licenses Step 3 to invent dimensions that aren't really part of this
trait — and Step 3 will. Look at the dark-gritty Step 3 trace
([marvel.txt:75-136](#)): aspects include "mature thematic weight" and
"visceral or graphic intensity," and a SENSITIVE_CONTENT call gets
emitted. Those landed because the Step 2 intent carried Marvel-flavor
("superhero optimism") into the dark-gritty meaning. Run the same
trait standalone and SENSITIVE_CONTENT is much less likely to surface.

**Root cause (prompt + schema).** The `Atom.evaluative_intent` field
description ([schemas/step_2.py:131-158](schemas/step_2.py#L131-L158))
says intent must "reflect every modifying_signal" and the read-back
test asks "would the intent change noticeably if I removed this
signal? If no, you haven't integrated it." That test treats *any*
modifying signal as something the intent must visibly absorb, which is
right for transposition / hedging / role markers but wrong for peer
cross-relations that are purely *scoping*. The system prompt
([step_2.py:320-388](search_v2/step_2.py#L320-L388)) reinforces the
same instruction. There is no language separating "operator-shaped
modifier — fold in" from "peer atom acting as scope — record the
relation, but keep the atom's own identity intact."

**Cost.** This is the upstream source of half the bleeding in your
notes. Step 3 inherits the contamination and amplifies it.

### 2. Step 3 over-decomposes vague aesthetic traits

**Observed.** `shitty` produces 3 aspects → 3 dimensions → 3 distinct
category calls: GENRE ("schlocky B-movie"), GENERAL_APPEAL ("low
quality perception"), CULTURAL_STATUS ("so-bad-it's-good cult")
([shark.txt:127-149](#)).

**Why this matters.** The user's `shitty` is one criterion (a fused
aesthetic), not three. Each of the three calls fires a generator/
reranker that admits a different population:

- a critically-panned but tonally-sober drama clears GENERAL_APPEAL,
- a campy schlocky comedy clears GENRE,
- a cult-canonized failure (`The Room`) clears CULTURAL_STATUS.

Combined with across-category MAX (next failure), any single one of
those wins the trait at 1.0. The shark movies that match the user's
intent (low-budget schlocky shark schlock) score the same as a
critically-panned WWII drama on this trait.

**Root cause (prompt).** Step 3's `_PER_DIMENSION_CANDIDATES` section
([step_3.py:425-473](search_v2/step_3.py#L425-L473)) explicitly forces
"NO upper bound on candidate count" + "list every category whose
description, boundary, or edge_cases makes it a real candidate." For
single-axis concrete traits this is benign; for multi-faceted vague
ones it explodes. Then `_CATEGORY_ROUTING` commits one CategoryCall
per category that ended up with a clean fit. The routing prompt has
no "is this trait genuinely multi-faceted enough to warrant multiple
categories, or am I just expanding a single fuzzy concept across
adjacent homes?" check.

The aspect-enumeration prompt encourages the model toward this:
*"multi-faceted figurative traits ('hidden gem', 'feel-good',
'underrated') reliably encode three or more axes."* That is true for
some traits and false for others. Applied uniformly it pushes the
decomposition wider than the user's actual ask.

### 3. Across-category MAX is wrong for compound concepts

**Observed.**

| Query | Trait | Categories committed |
|---|---|---|
| `bad bitch energy` | bad bitch energy | CHARACTER_ARCHETYPE + EMOTIONAL_EXPERIENTIAL |
| `bro movie` | bro movie | STORY_THEMATIC_ARCHETYPE + EMOTIONAL_EXPERIENTIAL + NARRATIVE_DEVICES |
| `dark gritty marvel movies` | dark gritty | EMOTIONAL_EXPERIENTIAL + SENSITIVE_CONTENT + STORY_THEMATIC_ARCHETYPE |
| `shitty shark movies` | shitty | GENRE + GENERAL_APPEAL + CULTURAL_STATUS |

**Why this matters.** The Phase D combine
([stage_4_execution.py:735](search_v2/stage_4_execution.py#L735)) takes
`max(category_scores)` across the trait's categories, and the rescore
doc justifies it on the grounds that "Step 2's atomization rule
guarantees one trait = one criterion → categories are framings of the
same criterion." That premise holds for `marvel movies` (STUDIO_BRAND
and FRANCHISE_LINEAGE truly are framings of one identity, both 1.0 on
an MCU film). It does **not** hold here.

`bad bitch energy` decomposes into "assertive defiant female protag"
(CHARACTER) AND "stylish assertive aura" (EMOTIONAL). Both are necessary
features of the criterion. Under MAX, a stylish-but-male-led action
movie matches on EMOTIONAL alone and fires the trait at 1.0. A
defiant-female-protag biopic with a sober tonal register matches on
CHARACTER alone and also fires at 1.0. Neither is what was asked for.

`bro movie`: the male-friendship theme + frat humor + male-led ensemble
are *three independent facets that must compound*. Under MAX, an
ensemble heist with no male-bonding theme wins on NARRATIVE_DEVICES; a
crude solo comedy wins on EMOTIONAL; a Lord-of-the-Rings-style
fellowship wins on STORY_THEMATIC_ARCHETYPE. None is a bro movie.

This is exactly your note: *"this one shows we shouldn't just take the
max of all categories."*

**Root cause.** The rescore doc's premise is wrong for one important
class of traits. Step 2 keeps "bro movie" as a single atom (correct —
the population test passes once for the fused phrase) but Step 3 then
decomposes it into independently-varying facets within that single
trait. There's a missing distinction:

- *Same-criterion framings* (Marvel = STUDIO ∨ FRANCHISE) → MAX is
  right, because matching either is sufficient evidence of the
  identity.
- *Compound-criterion facets* (bro movie = bonding ∧ humor ∧
  ensemble) → MAX is too lax; the categories are alternative *axes*
  the user wants to see *all* hit, not alternative *evidence* for one
  attribute.

Today the schema doesn't carry this distinction, so the executor
can't switch combine modes. The candidates layer in Step 3 actually
hints at it indirectly — when one dimension's candidate has
`what_this_misses="nothing"`, the dimension is genuinely the same as
the category; when every dimension carries a different missing piece
about its routing target, the trait is plausibly compound. That signal
is present but unused at scoring time.

### 4. Routing taxonomy gaps cause forced-fit calls

**Observed.** `bro movie`'s "male-led ensemble" routes to
NARRATIVE_DEVICES with `what_this_misses: "The gender-specific nature
of the ensemble"` ([bro.txt:88-91](#)). Step 3's candidate audit even
records the gap honestly. Then commits NARRATIVE_DEVICES anyway and
fires a noisy semantic search for a "male ensemble" against a
narrative-craft vector space.

**Why this matters.** The category absorbs an axis that doesn't
belong, so the call's signal is at best low-quality and at worst
adversarial — it'll fire on any ensemble-structured film, gender
unaware. Then MAX makes that low-quality signal sufficient to fire the
whole trait.

**Root cause (taxonomy + prompt).** No category covers "demographic
makeup of the cast." Step 3's prompt instructs that an aspect that
"resists translation" was wrong — *"go back and revise aspects rather
than skipping it here"* ([step_3.py:368-369](search_v2/step_3.py#L368-L369)
and [schemas/step_3.py:420-424](schemas/step_3.py#L420-L424)). The
model interprets that as "find the closest category and call it good"
rather than "drop the aspect because the system can't really retrieve
this." A "drop and note" affordance for genuinely uncoverable axes
would help.

### 5. Positioning ("Like X by Y") exports the entire reference's identity

**Observed.** `like zathura with jungles`:

- Step 2 correctly identifies the comparative anchor and the scope
  shift, even surfaces both atoms' modifying signals
  ([zathura.txt:9-39](#)).
- Step 3 on the `zathura` trait then enumerates **5 aspects** for
  Zathura's full identity, including **"outer space or fantastical
  setting"** ([zathura.txt:83-89](#)).
- Step 3 commits a NARRATIVE_SETTING call for `outer space or
  fantastical peril` ([zathura.txt:161-167](#)) — *while the sibling
  trait is asking for jungles*.

The user's note: *"What core parts are being kept and should be
searched? How can we make this a single semantic query?"* — this is
exactly that.

**Why this matters.** Phase B/C dispatches both calls. The jungle
NARRATIVE_SETTING call retrieves jungle films; the Zathura
NARRATIVE_SETTING call retrieves space films. They are competing for
the same axis at the same step. Even with the trait_score MAX
recovery, the Zathura trait wastes a generator-class call on a
population the user actively excluded.

**Root cause.** Step 3 receives only per-trait inputs — see
[step_3.py:674-722](search_v2/step_3.py#L674-L722), where
`_build_user_prompt` deliberately omits sibling traits and the
query-level `intent_exploration`. The doc-string explains why: *"Sending
query-level prose down would risk leaking other-trait interpretations
into this trait's routing."* That argument is correct for purely
parallel filters (feel-good doesn't need to know about Christmas).
It's exactly wrong for positioning traits, where the WHOLE point of
the qualifier is to communicate a scope replacement to the reference.

The positioning case also surfaces a deeper architectural mismatch:
"like X" most cleanly serves as a *single similarity-against-X*
retrieval (essentially the existing similar-movies flow), with sibling
traits acting as soft modulators. Comprehensively decomposing the
reference film's identity into independent dimensions is the wrong
shape — it produces calls that retrieve the reference's literal
attributes (its space setting, its specific archetypes) instead of
films near it in similarity space. Your intuition about "single
semantic query" is right; the decomposition prompt has no off-ramp for
this case.

### 6. No cross-trait awareness anywhere in the LLM stages

The previous failures all converge here. Three call-counting
observations:

- **`marvel movies` emits both STUDIO_BRAND and FRANCHISE_LINEAGE**
  ([marvel.txt:205-220](#)). For the same brand identity. Across-
  category MAX makes this benign for MCU films (both 1.0); for
  peripheral cases (Sony Spider-Man, FOX X-Men) only one fires and MAX
  still gives full credit. So it's not actively wrong, but it's
  redundant — two generator calls when one would do.
- **`shitty` emits 3 calls in 3 categories** ([shark.txt:127-149](#))
  that don't know about each other or about `shark movies` →
  combinatorial expansion of the union pool with no compensating
  precision.
- **`like zathura` emits NARRATIVE_SETTING:outer-space while
  `with jungles` emits NARRATIVE_SETTING:jungle** — direct conflict.

Your notes already converged on the fix space: *"endpoint generators
can maybe have context on other routings... maybe even step 3 gets
context on other traits so it knows not to duplicate. Maybe do this at
the step 2 level?"* The data says yes — the strict per-trait isolation
is leaking into bad routing, and there are at least three places to
break it:

| Where to add awareness | Buys you |
|---|---|
| Step 2 itself (single LLM call holds the whole query) | Better scoping in `evaluative_intent` and `qualifier_relation` to communicate "kept axis" vs "replaced axis" |
| Step 3 prompt input | Avoids zathura-vs-jungles conflict; lets a positioning trait drop axes the sibling replaces |
| Handler-LLM input | Prevents over-firing of redundant routes when an adjacent trait already covered the population |

The most surgical fix is at Step 2: the trait the model writes for
`like zathura` should encode (in `qualifier_relation` prose) which
axes of Zathura the comparison preserves and which it doesn't.
Step 3's role analysis already reads `qualifier_relation` as truth —
if Step 2 wrote "preserve archetype/tone, replace setting," Step 3
would honor that. Today's `qualifier_relation` prose is too generic to
do this work — the schema description doesn't mention scope-
replacement at all.

### 7. ADDITIVE × is brittle when the handler fires both KW and SEM

**Observed.** Every CategoryCombineType.ADDITIVE category in the
taxonomy ([trait_category.py](schemas/trait_category.py)) has KW + SEM
endpoints — CENTRAL_TOPIC, ELEMENT_PRESENCE, EMOTIONAL_EXPERIENTIAL,
SEASONAL_HOLIDAY, TARGET_AUDIENCE, SENSITIVE_CONTENT, etc. The
handler-LLM elects which endpoints to fire. When it fires BOTH:
`category_score = KW × SEM` per
[stage_4_execution.py:184-188](search_v2/stage_4_execution.py#L184-L188).
A KW miss (vocabulary gap) with a strong SEM hit zeros the category.

In `feel-good Christmas`, `feel-good`'s sole CategoryCall is to
EMOTIONAL_EXPERIENTIAL with three expressions ([xmas.txt:118-127](#)).
Each expression is a separate description — `uplifting tone`,
`comforting cozy atmosphere`, `happy ending`. The semantic handler
collapses these to one orchestrator-visible spec (one body with
multiple internal terms — verified in the SEM prompt
[semantic.md](search_v2/endpoint_fetching/category_handlers/prompts/endpoints/semantic.md)),
so this *specific* call doesn't trigger ADDITIVE multiply. But the
moment a category is ADDITIVE and the handler fires KW alongside SEM,
multiply takes over.

**Why this matters.** The rescore doc accepts this strictness as a
feature (*"the multiply does the gating intentionally"*) and pushes
the remediation upstream to the handler-LLM (*"keep KW tag sets broad
enough to absorb realistic vocab variance"*). That's a defensible
position, but in practice it shifts the failure mode from the user's
view (they see a movie missing for an unobvious reason) to the
debugging surface (you have to inspect both KW and SEM). For traits
where the KW vocabulary is genuinely incomplete — and the most-
multiplexed categories (CENTRAL_TOPIC, ELEMENT_PRESENCE,
EMOTIONAL_EXPERIENTIAL) are exactly where vocabulary tends to be
incomplete — this multiplies (literal) the brittleness.

This isn't a bug, but it interacts badly with failures #2 and #3:
when Step 3 over-decomposes a vague trait into multiple ADDITIVE
categories and one of them fires both KW + SEM, that category's
multiply zeros under a vocab miss, MAX hides the loss, and the trait
ends up scored entirely by whichever single category happened to
match. The other failures make ADDITIVE strictness more visible than
the doc anticipates.

## Pattern map: which queries trigger which failures

|     | bleed | over-gen | MAX | tax-gap | pos-export | x-trait | ADD× |
|-----|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| shitty shark | ✅ | ✅ | ✅ | – | – | ✅ | – |
| bad bitch | – | – | ✅ | – | – | – | – |
| bro movie | – | – | ✅ | ✅ | – | – | possible |
| horror not too gory | – | – | – | – | – | – | – |
| like zathura w/ jungles | – | – | – | – | ✅ | ✅ | – |
| dark gritty marvel | ✅ | ✅ | ✅ | – | – | ✅ | – |
| feel-good Christmas | – | – | – | – | – | – | – |

`feel-good Christmas` and `horror not too gory` are the two queries
where the system holds together — both are clean parallel filters with
single-axis traits that hit categories with clean homes. That is the
implicit operating-point the prompts/schema/scoring were designed
for; the failures concentrate where queries deviate from it.

## Where queries are handled best vs worst

**Best:**

- **Parallel filters with single-axis traits** mapping to clean
  categories — `feel-good Christmas`, `horror not too gory`, `Tom Hanks
  WWII movies` (the rescore doc's reference case).
- **Negative polarity with explicit hedging** — soft hedges commit
  diminished correctly; the gate × fuzzy negative formula handles
  authoritative vs evidential properly.
- **Identity / lookup categories** with clear taxonomy homes —
  PERSON_CREDIT, STUDIO_BRAND, FRANCHISE_LINEAGE, RELEASE_DATE, GENRE,
  SEASONAL_HOLIDAY all route precisely.

**Worst:**

- **Compound aesthetic traits** — `bro movie`, `bad bitch energy`,
  `shitty`, `dark gritty`. Atomization keeps them whole (correctly),
  Step 3 decomposes them into independent facets, MAX rewards single-
  facet matches.
- **Cross-modifying parallel filters** — `shitty shark`, `dark gritty
  marvel`. Atom phase contaminates the modified atom's intent.
- **Positioning traits with scope replacement** — `like zathura with
  jungles`, `[X] but [Y]`-shaped queries. Step 3 has no per-trait scope
  awareness; "like X" exports the reference's full identity instead
  of being treated as a similarity anchor.
- **Traits that name attributes the taxonomy doesn't cleanly cover**
  — gender composition, vague qualitative descriptors (`shitty`,
  `bad`), traits that conflate multiple structural axes.

## Specific code/prompt sites to look at

The following point-fixes correspond 1:1 to the failures above. They
are not a remediation plan — just where the contradictions live.

- **Atom contamination ([schemas/step_2.py:131-158](schemas/step_2.py#L131-L158))**:
  the read-back test treats every modifying signal as something
  evaluative_intent must absorb. Distinguish "operator-style modifier
  on this atom" (absorb) from "peer-atom cross-relation" (record on
  signals, do **not** rewrite intent).
- **Step 3 candidate explosion ([search_v2/step_3.py:425-473](search_v2/step_3.py#L425-L473))**:
  candidate enumeration is open-ended; routing commit is per-clean-fit.
  No language pushes back on "are these multiple homes the same fuzzy
  concept spread, or genuinely independent criteria?"
- **Across-category combine ([stage_4_execution.py:735](search_v2/stage_4_execution.py#L735))**:
  `max(category_scores)` is uniform; Step 3 doesn't emit a signal that
  would let the executor switch to "compound" combine for compound
  traits.
- **`qualifier_relation` schema description ([schemas/step_2.py:277-313](schemas/step_2.py#L277-L313))**:
  no language about scope preservation vs scope replacement in
  positioning relations. Today's outputs say "comparative anchor for
  the search" without naming which axes survive the comparison.
- **Step 3 user prompt ([search_v2/step_3.py:674-722](search_v2/step_3.py#L674-L722))**:
  intentionally per-trait. For positioning traits this is too strict;
  the docstring's rationale (*"risk leaking other-trait
  interpretations"*) doesn't apply when the OTHER trait is precisely a
  scope qualifier on this one.
- **Aspect-translation discipline ([schemas/step_3.py:420-424](schemas/step_3.py#L420-L424))**:
  *"go back and revise aspects rather than skipping it here"* gives no
  off-ramp for axes the taxonomy doesn't cover. Forced-fit routing
  results.
- **ADDITIVE × strictness ([stage_4_execution.py:184-188](search_v2/stage_4_execution.py#L184-L188))**:
  documented as a feature; cost is moved upstream to handler-LLM tag
  breadth, but interacts adversarially with over-decomposition and
  MAX-hides-zero (#2 + #3).

## Round-2 investigation: trait-relationship typology

The user pushback after round 1 surfaced a paradox: in **shitty shark
movies** the bleed of `shitty` into `shark movies`'s `evaluative_intent`
is harmful, while in **like zathura with jungles** the *absence* of
bleed (Zathura doesn't know jungles is replacing its setting) is
harmful. So when *should* one trait inform another?

To pin down the pattern, I ran 7 more queries probing the hypothesis
that trait-pair relationships fall into three classes the system
currently treats identically. Raw traces in
`/tmp/deepdive/{violent,inception,tarantino,cottagecore,elevated,heist,godfather}.txt`.

### Test plan

| # | Query | Hypothesis being probed |
|---|---|---|
| 8 | `violent action movies` | Independent qualifier — should bleed-free |
| 9 | `movies like inception but funnier` | "but Y" = same scope-replacement as "with Y"? |
| 10 | `tarantino-style sci-fi` | Transposition — keep some axes, replace others |
| 11 | `cottagecore movies` | Single-atom compound aesthetic (kino-like) |
| 12 | `elevated horror` | Compound concept that *looks* parallel-filter |
| 13 | `ensemble heist movies` | Co-equal atoms; suspected cast-structure taxonomy gap |
| 14 | `the godfather but with cowboys` | Extreme transposition |

### What round 2 confirmed and what it changed

**1. Atom contamination is *selective*, not systemic — refines failure #1.**

- `violent action movies` came out clean: both `violent`'s and
  `action`'s `evaluative_intent` describe only their own axis. No
  cross-pollution.
- `feel-good Christmas` (round 1) was clean.
- `ensemble heist movies` was clean.
- `shitty shark`, `dark gritty marvel`, `elevated horror`, `inception`,
  `tarantino`, `godfather` all bled.

The trigger is `intent_exploration`'s framing. When the model says *"two
co-equal components"* (violent+action, ensemble+heist), no bleed
occurs. When it says *"X qualifies Y"* (shitty qualifies shark; dark
gritty qualifies marvel), bleed always occurs. The schema's
"integrate every signal" rule fires the model's qualifier reading;
when the reading is co-equal, it doesn't.

So failure #1 is *narrower* than originally reported: it triggers
specifically when the model perceives a hierarchical (qualifier-on-
population) relationship. That's also the case where the bleed is
*least* harmful in practice, because Step 3 often ignores it (e.g.
shark's Step 3 ignored "shitty" entirely). But Step 3 *sometimes*
absorbs the bleed and decomposes accordingly — `elevated horror`'s
horror-trait Step 3 includes aspects "sophisticated style",
"intellectual substance", "high production value" which are 100%
elevated, not horror. The bleed is downstream-corrosive when both
atoms cross-bleed.

**2. New failure: atomization-split decomposition duplication.**

This is the most important finding from round 2 and didn't appear in
round 1. When Step 2 splits a fused compound into two atoms with
strong cross-modifying signals, Step 3 effectively decomposes the
*compound* concept *twice* — once per trait — with overlapping calls.

- **`elevated horror`** → `horror` trait emits a GENRE call with
  expressions `["horror", "elevated horror", "psychological horror"]`
  + a VISUAL_CRAFT_ACCLAIM call. The `elevated` trait emits a GENRE
  call with `["psychological horror"]` + EMOTIONAL_EXPERIENTIAL +
  VISUAL_CRAFT_ACCLAIM + CULTURAL_STATUS. **GENRE and
  VISUAL_CRAFT_ACCLAIM are duplicated across both traits.**
- **`the godfather but with cowboys`** → godfather trait emits
  STORY_THEMATIC_ARCHETYPE (crime dynasty) + NARRATIVE_DEVICES
  (generational saga) + GENRE (**western**). cowboys trait emits GENRE
  (western) + STORY_THEMATIC_ARCHETYPE (**crime + dynasty**) +
  EMOTIONAL_EXPERIENTIAL (operatic tragic). Each trait emits the
  OTHER trait's central axis.

Both traits independently route the *same compound concept* through
their own per-trait prompt. Effects:

- Phase B/C dispatches duplicate generators. Pool union is
  inflated, semantic compute is wasted.
- Phase D scores each trait separately, both fetch the same scores via
  duplicate calls. Two traits each score 1.0 on the trait the user
  thought was one — total contribution doubled vs a single-trait
  version of the same content.
- Rarity weighting (pure-generator path) computes against the
  *single-trait* match set, so a duplicated GENRE generator with the
  same match set on both traits weights both at the same rarity tier
  — compounding the doubled contribution.

This is a different failure than the round-1 "no cross-trait
awareness." Round 1 was about *conflict* (zathura's space vs jungles).
This is about *redundancy* — both atoms decomposing the same
compound. The two failures share a root cause (trait-relationship
ambiguity), but they sit on opposite sides of it: too little flow
(zathura) vs too much overlap (godfather, elevated horror).

**3. "but Y" exhibits the same positioning-export failure as "with Y".**

`movies like inception but funnier` reproduced the zathura pattern
exactly — and on a *more telling* category:

- Inception trait emits an EMOTIONAL_EXPERIENTIAL call for `cerebral
  and mind-bending tone`.
- Funnier trait emits an EMOTIONAL_EXPERIENTIAL call for `comedic tone
  and humor`.

These are direct competitors on the **same category** for the **same
axis the user explicitly said to replace**. Phase D scores Inception's
`cerebral` reranker against every candidate, so even a sci-fi comedy
gets penalized on Inception's tone-axis call. (Recovered by MAX across
Inception's other categories — so ranking still works for sci-fi
comedies — but the cerebral reranker is wasted compute and noise on
every candidate.)

This generalizes failure #5 ("positioning exports the reference's
identity") beyond the `with Y` syntax to `but Y` and almost certainly
`X-style Y`, `X meets Y`, `X but Z`, etc. **Any** transposition where
Y replaces some axis of X triggers it.

**4. Transposition keeps SOME axes, replaces others.**

`tarantino-style sci-fi` makes the nuance explicit. The user wants:
- Tarantino's *style* (dialogue, structure, violence, character
  archetypes) — KEEP
- Tarantino's typical *genre/setting* (crime/thriller) — REPLACE with
  sci-fi

Step 3 on the tarantino trait emits 5 calls: NARRATIVE_DEVICES,
DIALOGUE_CRAFT_ACCLAIM, EMOTIONAL_EXPERIENTIAL, SENSITIVE_CONTENT,
CHARACTER_ARCHETYPE — *plus* a `crime-focused ensemble with eccentric
characters` for CHARACTER_ARCHETYPE. The crime dimension is half-
right (criminal archetypes are a Tarantino signature) but conflicts
mid-axis with the sci-fi setting.

So scope replacement isn't all-or-nothing. The model needs to know
*which* axes the modifier replaces and *which* it preserves — and
today there's no field to express that. This deepens the
qualifier_relation schema gap (failure #6).

**5. Compound-criterion-vs-MAX confirmed on single-atom aesthetics.**

`cottagecore` is one trait. Step 3 decomposes into 3 calls:
NARRATIVE_SETTING (rural), EMOTIONAL_EXPERIENTIAL (cozy), STORY_ARCHETYPE
(simple living harmony with nature). Under MAX:

- A documentary set in a city about traditional crafts → wins on
  STORY_ARCHETYPE alone.
- A war film set in countryside → wins on NARRATIVE_SETTING alone.
- A horror film with cozy domestic moments → wins on EMOTIONAL alone.

None of these is cottagecore. Same shape as `bro movie` from round 1.
Confirms the MAX failure isn't a multi-trait bug — it's a
single-trait-with-multi-facet bug. Step 2's atomization decision
doesn't affect it; Step 3's decomposition + Phase D's MAX produce
the failure regardless.

**6. Taxonomy gap is real and broader than expected.**

`ensemble heist movies` is the cleanest case I tested for "co-equal
parallel filters with no bleed" — and it *still* shows the taxonomy
gap. `ensemble`'s Step 3 emits a NARRATIVE_DEVICES call with
expressions `["ensemble cast structure", "group dynamics and
collective action"]` — but NARRATIVE_DEVICES is about *narrative
craft* (POV, framing, structure), not cast composition. The Step 3
candidate audit even says NARRATIVE_DEVICES *covers* cast composition
("nothing missing") — which is the audit lying to itself.

The same gap appears in `bro movie`'s "male-led ensemble" routing
(round 1). Cast-structure axes have no clean home; Step 3 forces them
into NARRATIVE_DEVICES and pretends the fit is clean. This isn't a
prompt bug — it's a taxonomy completeness bug. Either add a category
for cast composition, or extend `NARRATIVE_DEVICES`'s description to
explicitly cover it (so the candidate audit is honest about what it's
matching).

### The general pattern: trait-relationship typology

Combining round 1 and round 2, **trait pairs** fall on a relationship
spectrum the pipeline currently doesn't classify:

| Class | Example | Bleed? | Atomize? | Cross-trait info needed? |
|---|---|---|---|---|
| **Independent / parallel filter** | feel-good Christmas; Tom Hanks WWII; ensemble heist | No | Yes (separate) | No |
| **Qualifier-on-population** | violent action; raunchy comedy | No (selective) | Yes (separate) | No |
| **Qualifier-on-population (subordinated)** | shitty shark; dark gritty marvel | Yes (cosmetic) | Yes (separate) | No (but qualifier over-decomposes under MAX) |
| **Scope-replacement** | with jungles → setting; but funnier → tone | No (today) | Yes (separate) | **Yes — modified trait must drop replaced axis** |
| **Transposition** (partial replacement) | tarantino-style sci-fi; godfather but cowboys; X-meets-Y | Variable | **Better as one** | **Yes — keep-vs-replace per axis** |
| **Fused compound** (single concept, multi-faceted) | bro movie; bad bitch energy; cottagecore; elevated horror | n/a (one trait) | **Single atom** | n/a |

The system today picks a single default — atomize separately, decompose
per-trait — that's correct for the first two rows, mostly-correct for
the third, *partially wrong* for rows 4-5 (no cross-trait flow), and
*structurally wrong* for row 6 (the model sometimes splits when it
shouldn't, which manifests as the atomization-split duplication
failure).

The asymmetry you flagged maps onto this:

- **"Shitty shark"** is a row-3 case. The cosmetic bleed isn't the
  real problem; over-decomposition + MAX are.
- **"Zathura with jungles"** is a row-4 case. The system needs to
  communicate scope replacement that today's `qualifier_relation`
  prose can't carry mechanically.
- **"Elevated horror" / "godfather but cowboys"** are row-5/6
  borderline. The model split atoms it should have kept fused, and now
  Step 3 duplicates the compound's decomposition across both halves.

**One classification field could absorb all of this.** Step 2 already
writes `qualifier_relation` per trait; an additional field —
something like a relationship-class commit (independent / qualifier /
scope-replace / transpose / fused) — plus, for transposition, an
*axes-replaced* and *axes-kept* split — would let downstream stages
behave differently per class:

- **Independent / qualifier**: today's behavior (no info flow).
- **Scope-replace**: pass the replaced axis into the modified trait's
  Step 3 input so it can drop that axis from its candidate inventory.
- **Transpose**: same plus per-axis kept-vs-replaced communication.
- **Fused**: keep as one trait, single Step 3 decomposition. Don't
  atomize.

The atomization decision itself is also informed by this: row 6 cases
should commit *one atom* with the relationship class folded in, not
two with cross-modifying signals.

## Refined failure map

Reordered and renumbered to match what the data actually showed:

| # | Failure | Triggers when… | Status |
|---|---|---|---|
| 1 | Atom intent contamination | one atom is read as a qualifier of the other (shitty shark, dark gritty marvel, elevated horror) | Real but **selective**; downstream-cosmetic *or* corrosive depending on whether Step 3 absorbs the bleed |
| 2 | Step 3 over-decomposition | trait is genuinely vague / multi-faceted (shitty, dark gritty, cottagecore, tarantino) | Real, exacerbated by failure #3 |
| 3 | Across-category MAX rewards single-facet matches | trait decomposes into compound facets (bro movie, bad bitch, cottagecore, tarantino, shitty) | Real and load-bearing; the rescore doc's premise is wrong for compound traits |
| 4 | Routing taxonomy gaps | trait names a real axis the taxonomy doesn't cleanly own (cast composition, vague qualitative) | Real, broader than initially scoped — affects clean parallel-filter cases too |
| 5 | Positioning exports the full reference identity | "like X", "X-style Y", "X but Y" (zathura, inception+funnier, tarantino, godfather) | Real on **all three syntaxes**, not just `with Y` |
| 6 | No cross-trait awareness | scope-replacement / transposition queries | Real; the `qualifier_relation` prose can't carry it mechanically |
| **7 (new)** | **Atomization-split decomposition duplication** | a fused compound gets split into two cross-modifying atoms (elevated horror, godfather + cowboys) | Real — both traits emit overlapping calls; doubles trait contribution and inflates compute |
| 8 | ADDITIVE × strictness | handler fires both KW and SEM for an ADDITIVE category | Real but documented; interacts adversarially with #2 + #3 |

## Restructure implications (what the data points at, not a plan)

The failures aren't independent — they're symptoms of one underlying
mismatch: **the system has no way to commit *what kind of relationship*
two traits have, or whether a single multi-faceted trait should
decompose with conjunctive vs alternative semantics.**

Three places to break this open, ordered by surgical-ness:

- **Add a relationship-class commit at Step 2.** Lowest-cost change.
  Field is freeform-or-enum on each trait (or on the QueryAnalysis
  itself). Step 3's role analysis already reads `qualifier_relation`
  mechanically — adding a class read changes nothing about Step 3's
  shape, just gives it a tighter input. The hardest case is
  transposition's "kept axis vs replaced axis" — needs richer
  prose than today's qualifier_relation.

- **Atomization rule update.** Row-6 fused compounds (elevated
  horror, godfather but cowboys, bro movie, bad bitch energy) should
  commit as ONE atom even when the population test passes for both
  pieces. Today's population test treats "horror" and "elevated" as
  both passing; the rule needs an additional check for "do these
  pieces fuse into a compound concept whose cross-modifications are
  bidirectional?" — if yes, single atom. This avoids the
  atomization-split duplication failure at the source.

- **Combine-mode signal on Step 3 output.** Today every trait gets
  across-category MAX. Step 3 has the structural signal that
  distinguishes same-criterion-framings (Marvel = STUDIO ∨ FRANCHISE,
  both `what_this_misses="nothing"` for the same dimension) from
  compound-facets (bro movie's three categories each cover a
  *different* dimension). Surface that signal, and Phase D can switch
  combine modes per trait.

The asymmetry you noticed — bleed wanted in zathura, unwanted in
shitty shark — is what the relationship-class commit absorbs cleanly.
The same field that says "scope-replacement" tells Step 3 to drop the
replaced axis (zathura case); the field that says "qualifier-on-
population" tells Step 3 NOT to absorb the qualifier into the
population trait's decomposition (shitty case). Both directions of
the asymmetry collapse to "what does Step 3 do with the qualifier?"
and the answer falls out of the relationship class.

## Summary

The pipeline is well-shaped for **parallel-filter queries with simple
atomic traits whose dimensions fit cleanly to a single category**.
Round 2 confirmed `violent action movies` and `ensemble heist movies`
on top of round 1's `feel-good Christmas` and `horror not too gory`.

Everywhere else — compound aesthetics, scope-replacement / transposition
queries, fused compounds the LLM splits, and traits whose attributes
the taxonomy doesn't own — the system degrades through interlocking
failures. Two new findings from round 2 reshape the picture:

1. **Atomization-split duplication** (failure #7) is a real and
   previously-unreported failure mode for fused compounds like
   "elevated horror" and "godfather but with cowboys." Both atoms
   decompose the *same compound*, so generator pools, scoring, and
   rarity weighting all double-count.
2. **Bleed is selective** — failure #1's harm is narrower than first
   reported. The user's paradox (bleed harmful in shitty shark, beneficial
   in zathura) resolves when you classify trait pairs by relationship
   type: bleed is wanted exactly in scope-replacement / transposition
   cases, where today's `qualifier_relation` field can't carry it
   mechanically.

The unified picture: **trait relationships fall on a typology the
system today doesn't recognize.** Independent, qualifier-on-
population, scope-replacement, transposition, and fused-compound
each want different decomposition and recombination behavior; the
pipeline's single default works for the first two and breaks
elsewhere.

The cleanest place to introduce this distinction is at Step 2, where
the model already reads the whole query and writes
`qualifier_relation`. Adding a relationship-class commit (and, for
transposition, an axes-kept / axes-replaced split) lets every
downstream stage read the class mechanically and adjust behavior —
no separate cross-trait LLM hop required.

---

# V4 Plan

This section captures the agreed-upon set of changes for the next
iteration. Failures targeted, prompt/schema changes, and concrete
retest queries with success criteria.

## Design principles for prompt and schema changes

All new prompt language must follow the conventions established in
the V3 prompts:

- **Generalized principles, not few-shot examples.** The schema's
  field descriptions and the system prompt's procedural walks describe
  the *function* of a signal, not the surface forms it takes.
  Recognition is by what the language DOES, not what tokens it uses.
- **Operational tests, not specifications.** Each commitment carries
  a "read this back and ask…" check the model performs. The check is
  the discipline; the prompt does not enumerate cases.
- **Schema descriptions as micro-prompts.** Each new field's
  description carries its own NEVER list; the system prompt is
  procedural and does not duplicate field-shape guidance.
- **No closed-list slotting except where the slot is genuinely
  closed.** New enums are closed when the values are operationally
  exhaustive (the relationship-role enum); freeform prose remains
  freeform when the surface variability is unbounded
  (`replaces_axis`, `qualifier_relation`).

## Step 2 changes

### S2.1 — `Trait` schema additions

After `qualifier_relation` on `Trait` ([schemas/step_2.py:277](schemas/step_2.py#L277)),
add three coupled fields:

```python
class TraitRelationshipRole(str, Enum):
    INDEPENDENT          = "independent"
    POSITIONING_REFERENCE = "positioning_reference"
    POSITIONING_QUALIFIER = "positioning_qualifier"
```

- **`relationship_role: TraitRelationshipRole`** — hard commit to
  one role. Closed enum; no "other" escape hatch. INDEPENDENT covers
  parallel filters AND qualifier-on-population (failures #1's
  bleed-and-don't-act-on-it cases) — both shapes need no cross-trait
  flow downstream.
- **`replaces_axis: str | None`** — required when role is
  `POSITIONING_QUALIFIER`; `None` otherwise. A short user-vocabulary
  noun-phrase naming the axis on the sibling reference being
  substituted (e.g. `"setting"`, `"tone"`, `"genre/setting"`,
  `"comedic register"`). Field description forbids slotting into a
  fixed vocabulary — the surface variability is unbounded.
- **`axes_replaced_by_siblings: list[str]`** — populated when role
  is `POSITIONING_REFERENCE`; empty list otherwise. Carries the
  sibling qualifiers' `replaces_axis` values verbatim. Step 2 owns
  the cross-trait reasoning here, so the reference trait inherits
  the replacements rather than re-deriving them.

The reference trait's `axes_replaced_by_siblings` is the bridge that
lets Step 3 honor scope replacement without seeing siblings. Step 2
holds the whole query already; Step 3 reads its trait input and
behaves correctly per-trait.

### S2.2 — System prompt: relationship-role classification section

New section in the commit phase, immediately after the existing
qualifier_relation discussion. Procedural walk in three operational
shapes:

- **Independent** — the trait names its own evaluable population /
  qualifier-on-population, additively combining with siblings. No
  positioning relation. Default when no atom is being positioned
  against another.
- **Positioning-reference** — the trait names the anchor a sibling
  is comparing or transposing against. The trait's identity is being
  used as a template; specific axes of that template may be replaced
  by siblings.
- **Positioning-qualifier** — the trait modifies a sibling reference
  by replacing or adding an axis. The qualifier itself is independently
  scorable, but its meaning in the query is *substitution* on the
  reference.

The discriminator is functional, not surface-tokens. Recognize the
*role* the language plays (anchor / qualifier / standalone), not the
specific connective ("with", "but", "-style", "like"). Same operator
surface can carry independent or positioning relations depending on
the content phrases it joins.

For `replaces_axis`: the field names the AXIS being replaced, in
user vocabulary, not the substitute the qualifier provides. The
operational test on the field is read-back: "does this name a
*dimension* of evaluation, not a value on that dimension?"
(`"setting"` ✅; `"jungle setting"` ✗ — that's a value, not an axis).

For `axes_replaced_by_siblings` on positioning-reference traits: the
reference does not invent replacements; it copies sibling qualifiers'
`replaces_axis` values. Read-back test: "does every entry in this
list correspond to a sibling whose `replaces_axis` named exactly this
phrase?"

### S2.3 — Atomization rule: fused-compound merge

New section in the atom phase / commit phase, addressing failure #7
(atomization-split decomposition duplication).

The population test today produces two atoms when each piece passes
standalone. For fused compounds (`elevated horror`, `the godfather
but with cowboys`, `bro movie`-style compounds), the two-atom outcome
duplicates Step 3 work and double-counts in scoring.

The discriminator is **bidirectional identity-shaping**, observable
at the atom phase from each atom's `modifying_signals`. The merge
rule:

- After atom enumeration, walk pairs of atoms with cross-modifying
  signals on each other.
- For each such pair, ask: does the user mean piece A *specifically
  in the context of* piece B, AND piece B *specifically in the
  context of* piece A? If yes, the pieces fuse into one atom.
- Single-direction shaping (qualifier shapes population's instance;
  population doesn't reshape what qualifier means) is still two
  atoms — the row-3 cases (shitty shark, dark gritty marvel) stay
  separate atoms with INDEPENDENT relationship_role.

For the merge to be detectable, each atom needs to record what
signals from the other atom shape its identity, not just its
evaluation. The existing `modifying_signals.effect` field carries
this — but the atom-phase prompt's discipline today emphasizes
"how this signal SHAPES EVALUATION" rather than "whether the other
atom's content is part of THIS atom's identity." The prompt update
adds an explicit test on each atom: "does the other atom's content
need to be *present* for this atom's meaning to remain what the user
asked for, or does this atom's meaning survive the other atom's
absence?" Identity-shaping signals trigger the merge consideration.

### S2.4 — Removing the implicit 1:1 atom→trait mapping

Today's commit-phase language ("Splits add traits; merges combine.
Genuine criteria don't disappear" — [schemas/step_2.py:603-605](schemas/step_2.py#L603-L605))
treats merges as a known operation but does not surface them as a
default outcome of the fused-compound test. The prompt change makes
the merge an explicit branch the commit phase considers, parallel to
splits. Atom count ≠ trait count when the fuse rule fires.

The schema description on `traits` already permits merges; the
system prompt update strengthens the COMMIT PHASE section
([step_3.py:391-490 — equivalent for Step 2](search_v2/step_2.py#L391-L490))
to walk the fuse decision before walking polarity / commitment, so
fuse merges happen *before* per-trait commitments are computed.

## Step 3 changes

### S3.1 — User-prompt input: surface every sibling trait's role-related fields

`_build_user_prompt` ([search_v2/step_3.py:674-723](search_v2/step_3.py#L674-L723))
receives only the per-trait commits today. The V4 input adds a
sibling section listing every other trait's:

- `surface_text`
- `relationship_role`
- `replaces_axis` (if non-`None`)
- `axes_replaced_by_siblings` (if non-empty)

Crucially, *not* the siblings' `evaluative_intent` or
`contextualized_phrase` — those carry interpretation that would leak
cross-trait. The sibling section is structural-only: which trait
plays which role, and which axes are being replaced. This honors
the docstring's existing rationale (avoid leaking interpretive
prose) while giving the trait what it needs to act on its own role
correctly.

### S3.2 — Trait-role analysis branches on `relationship_role`

`_TRAIT_ROLE_ANALYSIS` ([search_v2/step_3.py:130-230](search_v2/step_3.py#L130-L230))
becomes a switch on the enum:

- **INDEPENDENT** → today's standard decomposition. Aspects describe
  the trait's own population.
- **POSITIONING_REFERENCE** → decompose the reference's identity.
  Read `axes_replaced_by_siblings` and EXCLUDE any aspect whose user-
  vocabulary phrasing matches a listed axis. The role analysis
  prose explicitly commits which axes are kept and which are
  dropped, so the audit trail makes the drops visible. The aspect
  list is then guaranteed to omit the replaced axes.
- **POSITIONING_QUALIFIER** → decompose the qualifier as a
  refinement. Aspects MUST cover the content of `replaces_axis` —
  the qualifier's job is to provide the substitute for the axis it
  named, so any decomposition that doesn't address the replacement
  axis fails the role.

The role-analysis prose carries the kept/dropped/covered decisions
explicitly, so a fresh reader (or the audit) can verify the
aspect list lines up with the role.

### S3.3 — Combine-mode commit BEFORE `category_calls`

This is the structural reorder. Today: aspects → dimensions →
candidates → `category_calls`. V4: aspects → dimensions →
candidates → **`combine_mode`** → `category_calls`.

```python
class TraitCombineMode(str, Enum):
    FRAMINGS = "framings"
    FACETS   = "facets"
```

The commit happens after candidates because the model needs the
candidate analysis to ground the decision; it happens before
`category_calls` because the commit *shapes how `category_calls` is
chosen*:

- **FRAMINGS** — categories are alternative homes for the same
  underlying thing. Matching any one is sufficient evidence.
  Authorizes the model to commit categories whose coverage *overlaps*
  — duplication is fine because Phase D will MAX over them, so
  multiple framings reinforce as alternative routes to the same
  signal. Marvel-shape (STUDIO_BRAND ∨ FRANCHISE_LINEAGE) lives here.
- **FACETS** — categories cover distinct axes of a compound concept.
  All facets must fire to a degree for the criterion to be met.
  Demands the model commit categories that *complement* rather than
  overlap — duplication under FACETS amplifies the wrong signals
  because Phase D will compound them. Bro-movie / cottagecore /
  bad-bitch / dark-gritty live here.

Procedural test the prompt walks: "Looking at the candidate analysis,
do the dimensions converge on equivalent meanings spread across
adjacent categories (FRAMINGS) or do they each name a different
identifiable axis of the trait the user wants compounded (FACETS)?"
Then: "If FRAMINGS, are there other equivalent-meaning categories I
should commit alongside the obvious ones? If FACETS, are any of the
candidate categories I'm considering duplicating coverage another
already provides?"

### S3.4 — Schema additions on `TraitDecomposition`

After `category_calls` add:

- **`combine_mode: TraitCombineMode`** — committed before
  `category_calls`; surfaced last in the schema for serialization
  order, but ordered upstream in the procedural walk.

The role-analysis fields (`target_population`, `trait_role_analysis`)
already exist; their content for V4 will reflect the role branches
above. No new role field beyond the relationship_role read upstream.

## Stage 4 (Phase D) changes

### S4.1 — Branch on `combine_mode` in `_score_positive_trait`

[stage_4_execution.py:735](search_v2/stage_4_execution.py#L735) today
returns `max(category_scores) if category_scores else 0.0`. V4:

- `combine_mode == FRAMINGS` → keep MAX behavior.
- `combine_mode == FACETS` → product over category scores
  (`reduce(operator.mul, category_scores, 1.0)`). Same shape as the
  existing within-category ADDITIVE rule, just lifted to the
  across-category level. Bounded `[0, 1]`; strict (any 0 zeros the
  trait); rewards conjunctive matching.

The `CategoryScore` breakdown stays the same; the only Phase D
change is the fold operator.

### S4.2 — No changes to negative-polarity scoring

Negative traits keep the gate × fuzzy three-bin formula
([stage_4_execution.py:921](search_v2/stage_4_execution.py#L921)).
The combine_mode signal is positive-trait-only (negative scoring
already partitions calls structurally and doesn't fold through
per-category scores).

## Hypotheses to validate in V4

| # | Hypothesis | Expected effect |
|---|---|---|
| H1 | Adding `relationship_role` lets Step 3 act per-trait correctly without sibling prose | Positioning-reference traits drop replaced axes; positioning-qualifier traits cover their replacement |
| H2 | The fused-compound atomization rule prevents row-6 splits | `elevated horror` and `godfather but cowboys` commit as ONE trait, not two |
| H3 | Reading sibling fields (structural-only, no prose) is enough information for Step 3 to honor scope replacement | No same-axis cross-trait conflicts (no zathura-space-vs-jungle pattern) |
| H4 | Combine-mode commit BEFORE `category_calls` produces tighter category sets for compound traits | FACETS-mode traits commit fewer redundant categories; FRAMINGS-mode traits commit overlapping categories deliberately |
| H5 | Phase D product combine for FACETS traits eliminates single-facet wins | Bro movies / cottagecore-shaped traits no longer fire 1.0 on a single category match |
| H6 | Bleed-style contamination in `evaluative_intent` becomes harmless once Step 3 reads role mechanically | Step 3's behavior is determined by `relationship_role`, not re-derived from intent prose |

## Failures targeted (with retest queries)

| Failure | Retest queries | Expected V4 behavior |
|---|---|---|
| #1 atom intent contamination | `shitty shark movies`, `dark gritty marvel movies` | Bleed may persist in intent prose, but `relationship_role=INDEPENDENT` makes it harmless — Step 3 doesn't act on the bleed. |
| #3 across-category MAX | `bro movie`, `bad bitch energy`, `cottagecore movies`, `dark gritty marvel movies` (dark gritty trait), `tarantino-style sci-fi` (tarantino trait), `shitty shark movies` (shitty trait) | All commit `combine_mode=FACETS`. Phase D scores via product. Single-facet movies no longer win at 1.0. |
| #5 positioning exports identity | `like zathura with jungles`, `movies like inception but funnier`, `tarantino-style sci-fi`, `the godfather but with cowboys` | Reference trait drops the replaced axis from its aspect list. No sibling-conflict calls emitted (no NARRATIVE_SETTING:space when sibling has NARRATIVE_SETTING:jungle; no EMOTIONAL:cerebral when sibling has EMOTIONAL:comedic). |
| #6 cross-trait awareness | Same as #5 | Solved structurally via the role+axes fields. |
| #7 atomization-split duplication | `elevated horror`, `the godfather but with cowboys` | Commit as ONE trait per query, not two. Step 3 produces one decomposition. No duplicate generators / scoring across siblings. |

## Failures NOT targeted in V4

| Failure | Why deferred |
|---|---|
| #2 Step 3 over-decomposition | Partially addressed by combine_mode (over-decomposition stops winning under MAX), but the candidate-explosion problem is independent. Needs a separate prompt tightening on `_PER_DIMENSION_CANDIDATES`. |
| #4 routing taxonomy gaps (cast composition, vague qualitative) | Independent problem. Needs taxonomy work or category-boundary updates. |
| #8 ADDITIVE × strictness | Within-category combine unchanged. May improve indirectly because compound traits stop hiding ADDITIVE × zeros under MAX. |

## Success criteria

Define success per query, scoped to what's testable from
`run_step_3.py` output (Step 2 + Step 3 traces; not requiring full-
pipeline scoring for V4 verification at this stage).

### Atomization

- **`elevated horror`** → 1 trait emitted from Step 2 (not 2). The
  trait's `surface_text` covers the compound; `relationship_role =
  INDEPENDENT`.
- **`the godfather but with cowboys`** → 1 trait emitted from Step
  2 (not 2). `relationship_role = INDEPENDENT`.
- **`shitty shark movies`** → 2 traits, both `INDEPENDENT` (not
  fused — single-direction shaping only).
- **`feel-good Christmas movies`** → 2 traits, both `INDEPENDENT`
  (no regression).
- **`violent action movies`** → 2 traits, both `INDEPENDENT`
  (no regression).

### Relationship roles + axes

- **`like zathura with jungles`** → zathura trait
  `relationship_role = POSITIONING_REFERENCE`,
  `axes_replaced_by_siblings = ["setting"]` (or close paraphrase
  matching jungles' `replaces_axis`). Jungles trait
  `relationship_role = POSITIONING_QUALIFIER`, `replaces_axis =
  "setting"`.
- **`movies like inception but funnier`** → inception trait =
  POSITIONING_REFERENCE with `axes_replaced_by_siblings = ["tone"]`.
  Funnier trait = POSITIONING_QUALIFIER with `replaces_axis =
  "tone"`.
- **`tarantino-style sci-fi`** → tarantino trait =
  POSITIONING_REFERENCE with `axes_replaced_by_siblings` containing
  `"genre/setting"` or equivalent. Sci-fi trait =
  POSITIONING_QUALIFIER with matching `replaces_axis`.
- **`shitty shark movies`** → both traits INDEPENDENT (the qualifier
  doesn't replace an axis on shark; it modifies how shark is scored
  but is independently scorable). No `replaces_axis` set.

### Step 3 axis honoring

- **`like zathura with jungles`** zathura trait → no aspect for
  "outer space / fantastical setting"; no NARRATIVE_SETTING call;
  `trait_role_analysis` prose explicitly notes the setting axis is
  dropped because a sibling replaces it.
- **`movies like inception but funnier`** inception trait → no
  aspect for "cerebral / mind-bending tone"; no
  EMOTIONAL_EXPERIENTIAL call.
- **`tarantino-style sci-fi`** tarantino trait → drops the
  genre/setting axis (no GENRE call for crime/thriller, no
  CHARACTER_ARCHETYPE for "crime-focused ensemble" — the criminal-
  archetype aspect is part of style and survives, but the
  genre-bound part doesn't). Sci-fi trait covers the replacement
  cleanly.

### Combine modes

- **`bro movie`**, **`bad bitch energy`**, **`cottagecore movies`**,
  **`dark gritty marvel movies`** dark gritty trait, **`shitty shark
  movies`** shitty trait → `combine_mode = FACETS`.
- **`marvel movies`** trait (in `dark gritty marvel movies`) →
  `combine_mode = FRAMINGS` (STUDIO_BRAND and FRANCHISE_LINEAGE are
  framings of one identity).
- **`feel-good Christmas`** feel-good trait → if multiple categories
  fire, `combine_mode` matches whichever is the right read for that
  query (likely FACETS, since cozy + heartwarming + happy-resolution
  are facets); no regression on the single-trait scoring.
- **`horror`** in `horror not too gory` → `combine_mode = FRAMINGS`
  (single category, trivially passthrough).

### Phase D behavior

Once V4 is fully wired:

- A movie that hits ONE of bro movie's three FACETS at 1.0 and
  scores 0 on the other two should produce trait_score ≈ 0
  (not 1.0). Verify via `score_breakdowns` on a known-bad candidate.
- A movie that hits ALL three FACETS at high scores produces a high
  trait_score. Verify via `score_breakdowns` on a known-good
  candidate.
- A non-Marvel film correctly scores 0 on the marvel trait under
  FRAMINGS-MAX (no regression).

### No-regression set

- `feel-good Christmas movies`, `horror not too gory`, `violent
  action movies`, `ensemble heist movies`, `Tom Hanks WWII movies`
  — every trait commits `INDEPENDENT`, V3 scoring shape preserved,
  no new role/axis fields fire.

## Out-of-scope for V4

- Step 3 candidate-explosion prompt tightening (failure #2).
- New taxonomy categories for cast composition or vague qualitative
  descriptors (failure #4).
- Within-category ADDITIVE × revisions (failure #8).
- Handler-LLM context awareness (the per-CategoryCall handler still
  receives only its own call's expressions + retrieval_intent).
- Similar-movies flow integration for "like X" queries that would
  benefit from full-similarity retrieval rather than decomposition
  (longer-term architectural question; V4 settles for "drop the
  replaced axis," which is the lowest-cost win).
