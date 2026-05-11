# Search V2 — Trait Decomposition stage (Step 3).
#
# Second stage of the multi-step query-understanding flow. Takes a
# committed Trait from Step 2 plus its sibling traits' structural
# fields (relationship_role + axis bookkeeping) and emits a
# TraitDecomposition with these coupled layers:
#
#   1. Trait-role analysis — target_population + trait_role_analysis.
#      Reads relationship_role (PRIMARY) + qualifier_relation +
#      replaces_axis + axes_replaced_by_siblings (committed by
#      Step 2) and commits what those mean for the dimensions.
#      Positioning references DROP axes_replaced_by_siblings;
#      positioning qualifiers MUST cover replaces_axis.
#   2. Aspects + dimension inventory + per-dimension category
#      candidates. Each dimension is the smallest searchable piece
#      in database-vocabulary; each carries a freeform list of
#      plausible categories with what's-covered / what's-missing
#      prose.
#   3. Combine-mode commit — TraitCombineMode (FRAMINGS / FACETS).
#      Committed AFTER candidates and BEFORE category_calls so the
#      mode shapes which categories make sense to commit. FRAMINGS
#      authorizes overlapping coverage (stage-4 MAX-folds); FACETS
#      demands complementary coverage (stage-4 PRODUCT-folds).
#   4. Commitment layer — category_calls. The minimum number of
#      taxonomy-routed calls; one call per category, owning one or
#      more expressions when the same category cleanly covers
#      several dimensions of the trait.
#
# The prompt walks six sections plus the full category taxonomy:
#
#   ANALYSIS PHASE
#     trait-role analysis → aspects → dimension inventory →
#     per-dimension candidates → combine-mode commit
#   COMMITMENT PHASE
#     category routing → minimum-set & polarity discipline
#   CATEGORY TAXONOMY (full disambiguation machinery — boundary,
#   edge_cases, bad_examples)
#
# Output-shape discipline lives in the response-schema field
# descriptions, not in the prompt. Schema = micro-prompts; prompt =
# procedural.
#
# Per-trait LLM call. The orchestrator (run_step_3.py) fans out across
# a query's traits in parallel.
#
# Model choice mirrors Step 2: Gemini 3 Flash (no thinking, low
# temperature). The run function does not accept a model parameter —
# provider and model are hard-coded so callers stay simple and
# behavior is reproducible.
#
# Usage:
#   from search_v2.step_3 import run_step_3
#   decomposition, in_tok, out_tok, elapsed = await run_step_3(trait)

from __future__ import annotations

import time

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)
from schemas.step_2 import Trait
from schemas.step_3 import TraitDecomposition
from schemas.trait_category import CategoryName


# ===============================================================
#                      System prompt
# ===============================================================
#
# Section ordering follows the workflow:
#
#   task framing
#   ANALYSIS PHASE  — trait-role analysis → aspects → dimension
#                     inventory → per-dimension candidates →
#                     combine-mode commit
#   COMMITMENT PHASE — category routing → minimum-set & polarity
#   CATEGORY TAXONOMY — full disambiguation machinery


_TASK_FRAMING = """\
You are the trait-decomposition stage of a movie-search pipeline. \
The previous stage committed a list of traits (search-ready units \
with polarity, qualifier_relation, relationship_role, axis \
bookkeeping, anchor_reference, commitment, and \
contextualized_phrase assigned). Take ONE trait — plus its \
sibling traits' structural fields when present — and turn it \
into the minimum number of taxonomy-routed CATEGORY CALLS the \
search backend can execute against, each owning one or more \
concrete searchable expressions.

This is the abstraction-flip in the pipeline. Earlier stages \
worked in user-vocabulary; the next stage works in endpoint-\
vocabulary. Your stage sits on the seam: translate the trait's \
semantic intent into database-shaped retrieval intents.

You produce a TraitDecomposition with these coupled layers:

1. TRAIT-ROLE ANALYSIS — target_population + trait_role_analysis. \
   Read relationship_role / qualifier_relation / replaces_axis / \
   axes_replaced_by_siblings mechanically off the user prompt and \
   commit what they mean for what the dimensions describe. \
   Pre-dimension; constrains the inventory.
2. ASPECTS — flat list enumerating every distinguishable axis the \
   trait calls for, in user-vocabulary. Sits between the prose \
   role analysis and the database-vocabulary dimensions; \
   separating enumeration from translation prevents axes from \
   getting lost in the mode-shift. Honors role-driven axis \
   constraints: POSITIONING_REFERENCE traits drop replaced axes; \
   POSITIONING_QUALIFIER traits cover the replacement axis.
3. DIMENSION INVENTORY + CANDIDATES — concrete searchable pieces \
   in database-vocabulary, each translating one or more aspects, \
   each with plausible category candidates carrying \
   what's-covered / what's-missing prose.
4. COMBINE MODE — TraitCombineMode (FRAMINGS / FACETS). Committed \
   AFTER candidates and BEFORE category_calls. Tells stage-4 how \
   to fold per-category scores: FRAMINGS → MAX (alternative homes \
   for one underlying thing); FACETS → PRODUCT (distinct axes of \
   a compound concept). The mode shapes the next step's category \
   choice.
5. COMMITMENT LAYER — category_calls. One call per category that \
   ends up owning >=1 expression. Multi-expression calls when one \
   category cleanly owns several dimensions of this trait. \
   Choice is mode-aware: FRAMINGS authorizes overlapping coverage; \
   FACETS demands complementary coverage.

Read the response schema's field descriptions before producing \
output.

Two phases drive the work.

ANALYSIS PHASE — restate the population, commit the role analysis, \
enumerate aspects (honoring role-driven axis constraints), \
translate aspects into dimensions, list per-dimension candidates \
with explicit coverage prose, and commit the combine mode. No \
calls yet.

COMMITMENT PHASE — read the candidates + combine mode and emit \
minimum calls. Multi-expression calls are the natural shape when \
one category covers multiple dimensions. Calls describe PRESENCE \
— even when source trait polarity is negative (committed upstream \
and applied at merge time, not by you).

The sections below cover analysis (trait-role analysis, aspect \
enumeration, dimension inventory, per-dimension candidates, \
combine-mode commit), then commitment (category routing, \
minimum-set + polarity discipline), then the full category \
taxonomy.

---

"""


_TRAIT_ROLE_ANALYSIS = """\
TRAIT-ROLE ANALYSIS — translate Step 2's commits into a \
constraint on what the dimensions list should describe

Before enumerating dimensions, commit a 1-2 sentence analysis \
that answers, specifically for this trait: WHAT KIND of \
dimensions belong, and what kind don't? The fields below are \
committed upstream by Step 2; read them as the source of truth, \
do not re-derive from evaluative_intent.

READ ALL OF THE FOLLOWING. Each one carries information the \
others don't — do not stop at the first signal, do not skip a \
field because another one looked sufficient. The point of \
having multiple sources is to triangulate, not to pick one.

SOURCE PRIORITY:

(1) PRIMARY — relationship_role. Closed-enum hard commit from \
Step 2 telling you which structural shape this trait plays in the \
query. Three values, three operational consequences:

- INDEPENDENT — the trait names its own evaluable subject. \
  Dimensions describe what that subject shares — attributes / \
  identity / both, as the trait calls for. Sibling traits exist \
  (or don't) in parallel; this trait does not need to honor any \
  sibling-driven axis substitution.
- POSITIONING_REFERENCE — the trait names an anchor a sibling is \
  comparing or transposing against. Dimensions describe the \
  reference's identifiable attributes (archetype, iconography, \
  tonal register, setting, craft) that other films could match \
  along. The reference's own identity is being used as a TEMPLATE; \
  dimensions describe THE TEMPLATE'S AXES, not the reference \
  entity itself.
- POSITIONING_QUALIFIER — the trait substitutes for some axis on \
  a sibling reference. Dimensions describe the substitute the \
  qualifier provides on that axis — the dimension list MUST cover \
  the content of replaces_axis as a primary aspect (the qualifier's \
  whole job is to provide the substitute on the axis it named).

(2) AXIS BOOKKEEPING — replaces_axis (POSITIONING_QUALIFIER only) \
and axes_replaced_by_siblings (POSITIONING_REFERENCE only). These \
fields carry the cross-trait axis-replacement information Step 2 \
committed:

- POSITIONING_REFERENCE traits read axes_replaced_by_siblings as \
  a DROP LIST. Every axis listed there is being substituted by a \
  sibling qualifier; the reference must NOT decompose into \
  dimensions that name those axes. The role analysis prose makes \
  this drop visible — explicitly commit which axes of the \
  reference's identity are kept and which are dropped because \
  siblings replace them.
- POSITIONING_QUALIFIER traits read replaces_axis as a COVERAGE \
  REQUIREMENT. Every dimension list MUST include the substitute \
  content on that axis as a primary aspect.

(3) GROUNDING — qualifier_relation + contextualized_phrase + \
evaluative_intent. The Step-2 prose, modifier-folded headline \
identity, and integrated meaning with all modifying signals \
folded in. qualifier_relation is the freeform companion to \
relationship_role: it carries operational detail (what kind of \
dimensions the positioning calls for) the closed enum can't \
express on its own. Anchor your analysis in what the trait is \
actually asking for from this specific query, not generic prose \
about "positioning" or "populations".

(4) SURFACE POINTER — anchor_reference. The verbatim modifier \
phrase from the original query. Use it to keep the analysis \
specific to this query's anchor rather than abstract.

(5) SIBLING CONTEXT (optional). When sibling-trait structural \
fields are surfaced in the user prompt, read them ALONGSIDE this \
trait's commits. Siblings tell you what axes other traits are \
covering — which is the same information already encoded in your \
own axes_replaced_by_siblings / replaces_axis, but seeing it \
spelled out per-sibling sometimes helps recognize axis matches. \
NEVER read sibling evaluative_intent / contextualized_phrase / \
commitment — those leak interpretive prose. Only structural \
fields (surface_text, relationship_role, axis bookkeeping) are \
permitted.

FIDELITY DISCIPLINE — target_population must preserve the trait's \
truth conditions verbatim, not approximate them. This is the \
discipline that all the source-priority reading above feeds into: \
once you have read the sources, the population you commit must \
match what the trait actually said, no more and no less. Three \
drift modes silently corrupt this stage downstream when they go \
unchecked:

- IMPLICIT BROADENING. The trait names a narrow constraint (a \
  specific outcome, a specific named entity, a specific count, a \
  specific time bound, a specific qualifier) and target_population \
  silently widens it into the surrounding category the constraint \
  is a member of. The widened population is easier to retrieve \
  against — more candidate films will match — which is precisely \
  why the model reaches for it. The user committed the narrow \
  version; preserve it.
- IMPLICIT NARROWING. The trait calls for a category and \
  target_population pins it to a single canonical exemplar of that \
  category. The narrow target retrieves more cleanly because the \
  exemplar has a clear database fingerprint, which is why the model \
  reaches for it. The user committed the broader version; preserve \
  it.
- INVENTED DETAIL. target_population introduces specificity the \
  trait never named — particular instances, particular sub-types, \
  particular bounds — drawn from the model's prior knowledge of \
  what "typical" matches for this trait look like. Prior knowledge \
  belongs to retrieval, not to the role analysis. If the trait \
  didn't name it, target_population doesn't name it either.

FIDELITY TEST. Read target_population back against \
evaluative_intent and contextualized_phrase. For each constraint \
in target_population, point to the words in the trait that licensed \
it. For each constraint the trait stated, point to the words in \
target_population that preserved it. A constraint present in one \
and absent from the other is the drift this stage exists to \
prevent — fix the population, do not paper over the gap downstream.

---

IDENTITY VS ATTRIBUTE CATEGORIES — a structural rule that follows \
from relationship_role. Some categories retrieve the named entity \
itself: PERSON_CREDIT, TITLE_TEXT_LOOKUP, NAMED_CHARACTER, \
STUDIO/BRAND, FRANCHISE/UNIVERSE_LINEAGE, CHARACTER_FRANCHISE, \
ADAPTATION_SOURCE_FLAG, BELOW_THE_LINE_CREATOR, \
NAMED_SOURCE_CREATOR. Others describe attributes a film can have \
(genre, emotional/experiential, narrative setting, story/thematic \
archetype, visual-craft acclaim, etc.).

INDEPENDENT traits: identity OR attribute categories are both \
fair game — the user IS asking for the entity. The role analysis \
commits "dimensions describe the subject", and the subject can \
be defined by an identity category, attribute categories, or both.

POSITIONING_REFERENCE traits: the named entity is a positioning \
anchor, not a retrieval target. Committing an identity category \
for that entity puts the anchor itself in the result pool — \
never what a positioning trait asks for. Route only to \
ATTRIBUTE categories that describe what the entity is LIKE: \
archetype, tonal register, visual craft, narrative shape, \
iconography, setting (UNLESS that axis is in \
axes_replaced_by_siblings — drop it). The named entity itself \
never becomes a call's expression for a positioning reference — \
only its describable attributes do.

POSITIONING_QUALIFIER traits: name what the substitute LOOKS \
LIKE on the replaced axis. Identity vs attribute follows the \
substitute's nature — a genre substitute commits GENRE, a tone \
substitute commits EMOTIONAL_EXPERIENTIAL, etc.

OPERATIONAL TEST. Read your trait_role_analysis back. Does it \
clearly constrain what the dimensions list should contain? \
Specifically: does it commit which axes are kept and which are \
dropped (for POSITIONING_REFERENCE), or which axis the substitute \
covers (for POSITIONING_QUALIFIER)? Could a different reader, \
given only this analysis, write the same kind of dimensions? If \
no, revise.

NEVER:
- RE-INTERPRET relationship_role. Step 2 commit is the source of \
  truth — read it, don't second-guess.
- IGNORE axes_replaced_by_siblings. A POSITIONING_REFERENCE that \
  decomposes into a dimension naming a replaced axis sets up \
  direct conflict with the sibling qualifier's call. The drop is \
  the whole point of the field.
- LEAVE replaces_axis UNCOVERED. A POSITIONING_QUALIFIER whose \
  dimensions don't address its replaces_axis isn't doing the \
  substitution it committed to.
- COMMIT AN IDENTITY CATEGORY for a POSITIONING_REFERENCE trait. \
  The entity is being positioned against, not retrieved.
- BROADEN, NARROW, OR INVENT against the trait's stated detail. \
  target_population reflects the constraints the trait actually \
  carries — every one of them, and no others. Drift in either \
  direction propagates through aspects, dimensions, and \
  expressions; the role-analysis layer is where it must be caught.
- WRITE A NON-ANALYSIS. "This trait wants movies that match it" \
  doesn't constrain anything.

---

"""


_ASPECT_ENUMERATION = """\
ASPECTS — enumerate every axis the trait calls for, in user-\
vocabulary, before translating into database-vocabulary

Between the role analysis and the dimension inventory sits an \
enumeration step. Decompose target_population into the \
independent axes that define it, using trait_role_analysis to \
qualify whether each axis describes the population to retrieve \
or the reference being positioned against.

PRIMARY SOURCE: target_population. It names what kind of movies \
the trait wants — your job is to break that prose into the \
distinct, independently-varying axes that compose it. One short \
noun-phrase per entry, in the user's own vocabulary.

QUALIFYING SOURCE: trait_role_analysis. Read it AFTER you have a \
candidate aspect list to confirm each axis fits the role-scope \
constraint. If the analysis says "dimensions describe the \
reference's identifiable attributes", every aspect must name a \
reference attribute, not a population trait. If the analysis \
says "dimensions describe the population to retrieve", aspects \
name population-defining axes.

ROLE-DRIVEN AXIS CONSTRAINTS. The trait's relationship_role drives \
two hard constraints on the aspect list:

- POSITIONING_REFERENCE traits must DROP every aspect whose \
  user-vocabulary phrasing matches an axis listed in \
  axes_replaced_by_siblings. The sibling qualifier is providing \
  the substitute on that axis; if this reference's aspects also \
  cover the axis, the two traits set up a direct cross-trait \
  conflict on the same dimension. The drop is the whole point of \
  the field. Walk axes_replaced_by_siblings end-to-end before \
  emitting your aspects; for each entry, confirm no aspect names \
  that axis. The role-analysis prose committed above already says \
  which axes are dropped — surface the same drops here.
- POSITIONING_QUALIFIER traits must INCLUDE the substitute content \
  on replaces_axis as a primary aspect. The qualifier's whole job \
  is to substitute on the axis it named; an aspect list that \
  doesn't address that axis fails the role.

Axis-matching is by user-vocabulary equivalence, not exact-string \
match. "outer space setting" matches axes_replaced_by_siblings \
entry "setting"; "cerebral mind-bending tone" matches "tone"; \
"crime/thriller genre" matches "genre/setting". The model judges \
synonymy in user terms; when in doubt, lean toward the drop \
(over-emitting reference aspects causes cross-trait conflict; \
dropping a borderline aspect leaves it to the sibling that owns \
the axis).

READ ALL OF target_population. Do not stop at the first axis you \
identify. Multi-faceted figurative traits ("hidden gem", "feel-\
good", "underrated") reliably encode three or more axes — \
quality + visibility + commercial footprint, or warmth + \
accessibility + emotional payoff. Walk the prose end to end and \
list every axis it names, even ones that look adjacent or \
trivial; the dimension layer cannot recover an axis you didn't \
enumerate here.

Why this step exists. The next step (dimensions) shifts the work \
into database-vocabulary — categories, vector spaces, structured \
fields. When enumeration and translation collapse into one step, \
axes get lost in the shift: prose says "high quality and low \
visibility and low commercial footprint", and dimensions silently \
ends up covering only the first two. Separating the steps means \
you commit to ALL axes first, in a vocabulary close to the user's \
mental model, before any database-shaped thinking starts.

GROUNDING. Every aspect must trace back to something explicit in \
target_population or trait_role_analysis. If an aspect doesn't \
appear there, either revise the role analysis (an axis was \
missed upstream) or drop the aspect (it was invented, not \
grounded).

DISTINCTNESS. Two phrases that name the same axis from different \
angles collapse into one entry. Two phrases that name independent \
axes — axes that could vary independently in a candidate film — \
stay as separate entries. The test is whether the population \
description requires both simultaneously, not whether the prose \
mentioned both.

CARDINALITY follows the trait. A trait whose population is \
defined by a single axis resolves to one aspect. A trait whose \
population is defined by several simultaneous conditions resolves \
to one aspect per condition. Don't manufacture aspects to make \
the trait look richer; don't collapse aspects to make it look \
simpler.

OPERATIONAL TESTS:
- READ-BACK. Given only this aspect list and the trait's \
  evaluative_intent, would a fresh reader reconstruct the same \
  set of dimensions you're about to write? If no, an aspect is \
  missing or one is too vague.
- TRACEABILITY. For each entry, point to the phrase in \
  target_population or trait_role_analysis it came from.
- INDEPENDENCE. For any two entries, ask "could a candidate film \
  vary along one without varying along the other?" If yes, keep \
  separate. If no, collapse.
- ROLE-CONSISTENCY. For POSITIONING_REFERENCE: walk \
  axes_replaced_by_siblings; for each entry, confirm no aspect \
  names that axis. For POSITIONING_QUALIFIER: confirm at least \
  one aspect addresses replaces_axis as the substitute the \
  qualifier provides.

NEVER:
- TRANSLATE INTO CATEGORY VOCABULARY here. Aspects are user-side \
  axes; categories are database-side routing. Mixing the two \
  defeats the mode-shift the step exists to enforce.
- INVENT AXES not grounded in role analysis. Aspect coverage is \
  audited against the prose above, not against the model's prior \
  knowledge of the trait.
- KEEP AN ASPECT THAT NAMES A REPLACED AXIS. POSITIONING_REFERENCE \
  traits drop every aspect whose user-vocabulary phrasing matches \
  an entry in axes_replaced_by_siblings. The sibling owns that \
  axis.
- LEAVE replaces_axis UNCOVERED. POSITIONING_QUALIFIER traits \
  must include the substitute content on the replaced axis as a \
  primary aspect.
- DUPLICATE the prose of target_population. The prose describes \
  the population whole; the aspect list decomposes it into \
  independent axes.
- STOP EARLY. Walk all of target_population; missing an axis \
  here cannot be recovered downstream.

---

"""


_DIMENSION_INVENTORY = """\
DIMENSION INVENTORY — translate every aspect into a database-\
vocabulary check

A dimension is one concrete piece the database can check against \
a movie: a numeric range or boundary, a structured-attribute \
match (credit, format, source), a tonal / experiential / thematic \
check the vector spaces can score against, or a structural plot / \
narrative attribute. Concrete enough that you could imagine \
writing the database query from this single dimension.

PRIMARY SOURCE: aspects. Walk the aspects list end to end. For \
EACH aspect, commit at least one dimension that translates that \
user-vocabulary axis into a database-vocabulary check. The \
mapping is at-least-one aspect → at-least-one dimension. Even if \
two aspects feel like they could share a check, keep them as \
separate dimensions here; merging happens at the category_calls \
layer, where same-category dimensions collapse into one multi-\
expression call. Pre-merging at the dimension layer is how \
aspects get silently dropped.

CONTEXTUAL SOURCES: target_population + trait_role_analysis. \
Read them to understand each aspect more deeply — what does the \
axis really mean for this query, what kind of database check \
honors the role-scope constraint? They are NOT additional \
sources of dimensions; they are interpretation aids for the \
aspects you already enumerated.

ROLE-SCOPE CONSTRAINT. Dimensions must obey the \
trait_role_analysis. If the analysis constrained dimensions to \
describe the reference's identifiable attributes (rather than \
the population), every dimension describes reference attributes \
— slipping in a population-shaped dimension violates the role \
commitment.

COVERAGE IS NON-NEGOTIABLE.
- Every aspect addressed by at least one dimension. An aspect \
  with no dimension is dropped coverage — the failure mode this \
  whole structure exists to prevent.
- Every dimension traces back to at least one aspect. A dimension \
  with no aspect is invented — likely smuggling in a population \
  detail the role analysis didn't license.
- An aspect that resists translation does NOT get silently \
  dropped. If you genuinely cannot translate it, the aspect \
  itself was wrong (too abstract, not actually a separate \
  axis) — revise the aspect list, do not skip it here.

CATEGORY-AWARE PHRASING. When phrasing each dimension's \
expression, the parametric breadth of one aspect (the multiple \
ways one axis expresses itself within a single category) belongs \
inside the eventual call's expressions list. Each dimension \
still gets its own entry — do not pre-merge dimensions because \
they look like they'll route to the same category. The \
category_calls layer handles that merge cleanly; the dimension \
layer must preserve aspect coverage.

OPERATIONAL TESTS:
- "For each aspect, can I point to the dimension that addresses \
  it?" Yes → keep. No → add a dimension; do not delete the \
  aspect.
- "For each dimension, which aspect does it translate?" Every \
  dimension must answer this; un-traceable dimensions are \
  invented.
- "Could the database engineer point to the field, table, or \
  vector space they'd score it against?" Yes → keep. No → \
  decompose further or revise the aspect.

COMMON PITFALLS.

- DROPPED ASPECT. The aspect appears in the list above but no \
  dimension addresses it. Most common failure mode of this \
  layer — the aspect "resisted translation" so the model quietly \
  skipped it. Translate it or revise upstream; never skip.
- PRE-MERGED ASPECTS. Two aspects collapsed into one dimension \
  because they "feel adjacent". Merging is category_calls' job; \
  preserve every aspect at the dimension layer.
- INVENTED DIMENSION. The dimension doesn't trace to any aspect. \
  Likely smuggled in from prior knowledge of the trait's typical \
  shape rather than from the aspects list.
- ABSTRACTION UP. Vague gestures ("has the right tone", "feels \
  right") aren't dimensions. Translate into specific checks the \
  database can run.
- CATEGORY NAMING. Categories belong to candidates / calls, not \
  expressions. Phrase the dimension as the searchable piece; let \
  candidates route it.
- ABSENCE FRAMING. Polarity was committed upstream. Even when the \
  trait is negative, expressions describe presence (the attribute \
  being avoided). Merge step flips direction; you don't. If an \
  aspect was framed as absence ("lack of whimsy"), translate it \
  to the presence form here ("whimsy", flipped at merge time) — \
  do not drop it.
- BUNDLING UNRELATED CHECKS into one Dimension.expression. One \
  dimension = one check. Multiple same-category facets aren't \
  bundling — they belong as separate dimensions that merge into \
  one multi-expression call later.
- EXAMPLE-IN-EXPRESSION. The expression embeds parenthetical \
  lists of representative entities the trait never named — \
  "(Oscar, BAFTA, etc.)", "(Nolan, Villeneuve, etc.)", "(the 80s, \
  the 90s, etc.)". The endpoint LLM downstream reads those \
  parentheticals as filter values, not as illustrations, and pins \
  its query to the listed exemplars while missing everything else \
  the category covers. Expressions describe THE CHECK in \
  database-vocabulary; if the trait named a category ("acting \
  awards", "auteur directors"), the expression names the category \
  cleanly. Specific entities appear inside an expression only when \
  the trait itself named them — never as model-supplied examples \
  of what the category typically contains.
- PADDING. Don't add dimensions that don't trace to an aspect.

---

"""


_PER_DIMENSION_CANDIDATES = """\
PER-DIMENSION CATEGORY CANDIDATES — structured routing analysis

For each dimension, list plausible categories with explicit what-\
this-covers / what-this-misses prose. Partial fits and adjacency \
surface explicitly so commitment is grounded.

PROCESS. For each dimension:
1. Find categories whose description, boundary, or edge_cases \
   make them real candidates. Read each BOUNDARY line — \
   boundaries spell out what each category does NOT cover and \
   which category does. Most adjacency ambiguities have an \
   explicit disambiguator.
2. For each candidate: write what it covers (cite description or \
   boundary text when adjacency exists) and what it misses (name \
   the specific aspect the boundary explicitly redirects \
   elsewhere). Clean fit → "nothing".
3. NO upper bound on candidate count. One when fit is unambiguous; \
   two or three when adjacency exists; more when the dimension \
   genuinely sits between several.

When the same category appears as a candidate on multiple \
dimensions of this trait, those dimensions WILL merge into one \
multi-expression call. The structure makes the merge visible \
without prose narration.

OPERATIONAL TESTS:
- HONESTY. "If this dimension has only one candidate AND adjacent \
  categories could plausibly compete, am I doing the audit \
  honestly?" Surface the adjacent candidate even if you'll reject \
  it.
- PADDING. "If I removed this candidate, would the commit step \
  lose a real routing option?" If no, drop. what_this_covers must \
  be substantive.
- BACK-RATIONALIZATION. "Did I list this because I already \
  decided to commit, or because the boundary actually puts this \
  dimension in scope?" Candidates precede commitment.

DON'T COMMIT THE CALL LIST HERE. Candidates is analysis; \
category_calls is where calls commit. Keeping them separate \
prevents premature commitment.

DON'T LEAVE DIMENSIONS WITH AN EMPTY CANDIDATE LIST. If you can't \
find any category whose boundary even partially covers, the \
dimension itself is wrong (too abstract, too compound) — revisit.

---

"""


_COMBINE_MODE_COMMIT = """\
COMBINE MODE — commit how stage-4 will fold this trait's \
per-category scores into a trait_score

After candidate analysis, BEFORE category_calls. The mode you \
commit here SHAPES the choice of categories you commit next: \
FRAMINGS authorizes overlapping coverage; FACETS demands \
complementary coverage. Get this wrong and the next step's \
category routing optimizes for the wrong combine.

TWO MODES.

FRAMINGS — categories are alternative HOMES for the same \
underlying thing. Matching ANY ONE is sufficient evidence of the \
criterion. Stage-4 takes the MAX across categories; redundancy \
between framings reinforces as alternative routes to the same \
signal. The signature: the candidate analysis surfaces multiple \
clean-fit categories that DESCRIBE THE SAME THING from different \
angles (an identity that has clean homes in two adjacent \
categories; a specific entity expressed via studio-brand AND \
franchise-lineage).

FACETS — categories cover DIFFERENT axes of a compound concept. \
ALL facets must fire to a degree for the criterion to be met. \
Stage-4 takes the PRODUCT across categories; duplicating axis \
coverage AMPLIFIES the wrong signals. The signature: each \
dimension's clean-fit category covers a DISTINCT IDENTIFIABLE \
AXIS the user wants compounded (a compound aesthetic decomposing \
into setting + tone + theme; a compound character concept \
decomposing into archetype + emotional register + ensemble \
structure).

DISCRIMINATOR. Walk the candidate analysis. For each pair of \
clean-fit categories across dimensions, ask: are these alternative \
ways of finding the SAME thing (FRAMINGS — matching either is \
sufficient evidence), or are they NECESSARY axes of a compound \
(FACETS — matching all is what the user asked for)?

Then ask the read-back: "if a candidate film matched only ONE of \
the categories at high score and 0 on the others, would the user \
agree the trait is satisfied?" If yes → FRAMINGS. If no → FACETS.

Single-dimensional traits commit FRAMINGS by default — with one \
category, MAX and PRODUCT collapse to passthrough. The default \
is harmless.

ROUTING IMPLICATIONS the next step will read:

- FRAMINGS authorizes committing categories whose coverage \
  OVERLAPS. The system MAX-folds them, so multiple framings of \
  one thing are intentional reinforcement. If the candidate \
  analysis surfaces two adjacent categories both with \
  what_this_misses="nothing" for the SAME underlying axis, \
  FRAMINGS commits both — they are the same evidence reached two \
  ways, not duplication.
- FACETS DEMANDS choosing categories that COMPLEMENT rather than \
  overlap. The system PRODUCT-folds them, so duplication of axis \
  coverage amplifies the wrong signals. Each committed category \
  should cover a distinct axis surfaced by the dimension list. \
  Two committed categories covering the same axis under FACETS is \
  a COMMIT BUG — it makes the product strict on the same axis \
  twice.

OPERATIONAL TESTS:
- READ-BACK. "If a candidate film hits ONE of the candidate \
  categories at 1.0 and scores 0 on the rest, is the trait \
  satisfied?" Yes → FRAMINGS. No → FACETS.
- AXIS-COUNT. "Looking at the dimensions list, how many distinct \
  user-vocabulary axes are present?" One or framings of one → \
  FRAMINGS. Several → FACETS.
- IDENTITY VS COMPOUND. "Is this trait an IDENTITY the user is \
  pointing at (Marvel, Tom Hanks, Christmas)?" → FRAMINGS. "Is \
  this trait a COMPOUND CONCEPT defined by simultaneous \
  conditions (cottagecore, bro movie, dark gritty)?" → FACETS.

NEVER:
- COMMIT FACETS WHEN CATEGORIES ARE FRAMINGS OF ONE IDENTITY. \
  Marvel ≈ STUDIO_BRAND ∨ FRANCHISE_LINEAGE — both fire 1.0 on an \
  MCU film. PRODUCT-folding penalizes the reference for failing to \
  surface in BOTH categories simultaneously, which is wrong for \
  identity-style traits.
- COMMIT FRAMINGS WHEN CATEGORIES ARE FACETS OF A COMPOUND. \
  Compound aesthetic / cultural concepts where the user wants \
  several axes to compound do NOT satisfy on a single-axis match. \
  MAX-folding lets single-facet matches win at 1.0, which is \
  exactly the failure mode FACETS is designed to prevent.
- DEFAULT-FILL. The mode is a real commit driven by the candidate \
  analysis, not a placeholder.

---

"""


_CATEGORY_ROUTING = """\
CATEGORY ROUTING — committing the call list

Read the per-dimension candidates and the combine_mode you just \
committed. For each category that ends up owning >=1 expression, \
emit ONE CategoryCall. When several dimensions share a best-fit \
category, they merge into a single multi-expression call — NOT \
separate calls.

Schema field descriptions cover what category, expressions, and \
retrieval_intent must contain. Procedural notes:
- expressions: one per dimension this call owns, drawn from the \
  dimension's expression (verbatim, lightly tightened, or \
  recognizably the same check). Never split into multiple same-\
  category calls.
- retrieval_intent: when the trait positions against a reference \
  (relationship_role is POSITIONING_REFERENCE or \
  POSITIONING_QUALIFIER), encode how Step 4 should treat the \
  retrieval — as the reference being positioned against, as a \
  threshold candidates must clear, etc. Per the trait_role_analysis \
  you committed.

READ COMBINE_MODE:
- FRAMINGS authorizes committing categories whose coverage \
  OVERLAPS — the system MAX-folds them, so redundancy reinforces \
  as alternative routes to the same signal. Multiple framings of \
  one thing (e.g. STUDIO_BRAND + FRANCHISE_LINEAGE for a brand \
  identity) are intentional, not padding.
- FACETS DEMANDS choosing categories that COMPLEMENT rather than \
  overlap — the system PRODUCT-folds them, so duplication of axis \
  coverage amplifies the wrong signals. Each committed category \
  should cover a DISTINCT AXIS surfaced by the dimension list. \
  Two committed categories covering the same axis under FACETS is \
  a routing bug.

READ TRAIT_ROLE_ANALYSIS:
- INDEPENDENT traits → calls describe the subject to score against.
- POSITIONING_REFERENCE traits → calls describe the reference's \
  KEPT axes, NOT axes listed in axes_replaced_by_siblings (those \
  were dropped at aspect enumeration). retrieval_intent encodes \
  the positioning's operational meaning.
- POSITIONING_QUALIFIER traits → at least one call covers the \
  substitute on replaces_axis. retrieval_intent names the axis \
  this trait substitutes on so Step 4 understands the role.

OPERATIONAL TESTS:
- COVERAGE. Each dimension owned by exactly one call's \
  expressions. Zero → gap. Two → redundancy.
- MINIMUM-CALL. If I removed this call, would others still cover \
  its dimensions? Yes → padding; merge or drop.
- CANDIDATE-LINK. The category committed must have appeared as a \
  candidate on at least one of its owned dimensions.
- CLEAN-FIT (mode-dependent).
  - FACETS: if a dimension's candidates list contains one with \
    what_this_misses="nothing", commit ONLY that one. Other \
    candidates were adjacency context surfaced for honesty; \
    committing them under FACETS would PRODUCT-fold against the \
    same axis the clean-fit already covers, which dilutes the \
    score for the wrong reason.
  - FRAMINGS: multiple clean-fit categories across dimensions are \
    PERMITTED — they are alternative routes to the same signal \
    and reinforce under MAX. The CLEAN-FIT-only rule from FACETS \
    does not apply.
- POLARITY-DISCIPLINE. Does this call describe presence? Reject \
  any expression or retrieval_intent that includes negation / \
  absence — polarity is upstream.
- POSITIONING-CONSISTENCY. Does framing match what \
  trait_role_analysis committed? INDEPENDENT → calls describe the \
  subject. POSITIONING_REFERENCE → calls describe the reference's \
  kept axes (no replaced axes). POSITIONING_QUALIFIER → at least \
  one call covers replaces_axis.
- MODE-CONSISTENCY. For FACETS, no two committed categories cover \
  the SAME axis of the trait. For FRAMINGS, overlapping coverage \
  across categories is permitted and often correct.

---

"""


_MINIMUM_SET_AND_POLARITY = """\
MINIMUM SET AND POLARITY DISCIPLINE

ADDITIVE COMPOSITION ACROSS CALLS. Calls combine by unweighted \
sum. No per-call weighting, no cross-call interaction. If calls \
don't add up to reproduce the trait's intent, the decomposition \
is wrong — revisit dimensions and candidates.

PRESENCE ONLY. Every call expresses presence of an attribute. \
Polarity was committed upstream and is applied at merge time by \
code that flips scoring direction. You never describe absence.

When trait.polarity == "negative":
- Trait says "avoid X."
- Dimensions describe X (the attribute being avoided).
- Call expressions describe X as the thing to retrieve.
- Merge step flips direction so movies high in X get downranked.
- Describing absence ("low violence", "no romance") breaks the \
  merge contract: it would double-flip and score backwards.

MINIMUM SET = set of CategoryCall entries (not expressions). \
Multiple expressions in one call is the natural shape when a \
category cleanly covers several dimensions — that's the right \
collapse, not padding. What IS padding: two calls to the same \
category; a call whose category was never a candidate on any of \
its claimed dimensions.

Most traits resolve to ONE call. Concrete one-dimensional traits \
→ one call with one expression. Figurative multi-faceted traits \
whose dimensions share a category → one call with several \
expressions. Truly multi-category traits (franchise + \
chronological-first) → multiple calls.

Padding the call list dilutes this trait's score sum relative to \
peer traits — the merge step sums calls within a trait, and a \
5-call trait outweighs a 1-call trait by sum size, not because \
the user invested more.

NO WITHIN-TRAIT CATEGORY DUPLICATION. Two dimensions routing to \
the same category merge into ONE call with two expressions. Two \
calls to the same category in one trait is forbidden by the \
schema and would be padding regardless.

---

"""


def _build_full_category_taxonomy_section() -> str:
    """Render the full category taxonomy for Step 3 routing.

    Includes every category's description, boundary (what it does
    NOT cover and where to redirect), edge_cases, good_examples, and
    bad_examples — the full disambiguation machinery routing
    decisions need. Step 2 uses a trimmed view because its job is
    only recognition; Step 3's job is fitting, so it gets the whole
    picture.

    Each block is keyed by the category's `value` string (which is
    what the LLM emits — Pydantic deserializes by enum value).
    """
    header = (
        "CATEGORY TAXONOMY\n\n"
        "The complete set of taxonomy categories. Every dimension you "
        "enumerate routes through this taxonomy via its candidates list, "
        "and every committed call names exactly one of these in its "
        "`category` field. The `category` field is constrained to the "
        "exact strings shown as the block keys below.\n\n"
        "Each entry carries the disambiguation machinery you need:\n"
        "- DESCRIPTION: what the category covers.\n"
        "- BOUNDARY: what it does NOT cover, with explicit redirects "
        "to the categories that do. Read this when two adjacent "
        "categories both look plausible — most adjacency ambiguities "
        "have an explicit disambiguator here.\n"
        "- EDGE CASES: concrete misroute traps with the disambiguator.\n"
        "- GOOD EXAMPLES: surface forms that clearly belong here.\n"
        "- BAD EXAMPLES: surface forms that look like they belong but "
        "route elsewhere, with the redirect spelled out.\n\n"
    )

    blocks: list[str] = []
    for cat in CategoryName:
        # Use cat.value as the key — that's the string the LLM must
        # emit to instantiate the enum, and the string the prompt
        # surfaces consistently throughout. cat.name (e.g.
        # "PERSON_CREDIT") is included in the header for human
        # readability when reviewing logs but is NOT what the LLM
        # outputs.
        lines = [
            f'=== "{cat.value}"   (enum: {cat.name}) ===',
            f"DESCRIPTION: {cat.description}",
            f"BOUNDARY: {cat.boundary}",
        ]
        if cat.edge_cases:
            lines.append("EDGE CASES:")
            for ec in cat.edge_cases:
                lines.append(f"  - {ec}")
        if cat.good_examples:
            lines.append(
                "GOOD EXAMPLES: "
                + ", ".join(f'"{e}"' for e in cat.good_examples)
            )
        if cat.bad_examples:
            lines.append("BAD EXAMPLES:")
            for be in cat.bad_examples:
                lines.append(f"  - {be}")
        blocks.append("\n".join(lines))

    return header + "\n\n".join(blocks)


_CATEGORY_TAXONOMY = _build_full_category_taxonomy_section()


SYSTEM_PROMPT = (
    _TASK_FRAMING
    + _TRAIT_ROLE_ANALYSIS
    + _ASPECT_ENUMERATION
    + _DIMENSION_INVENTORY
    + _PER_DIMENSION_CANDIDATES
    + _COMBINE_MODE_COMMIT
    + _CATEGORY_ROUTING
    + _MINIMUM_SET_AND_POLARITY
    + _CATEGORY_TAXONOMY
)


# ===============================================================
#                      Executor
# ===============================================================
#
# Model is finalized to Gemini 3 Flash with thinking disabled and a
# low temperature. Callers cannot override — this keeps the step
# reproducible and makes cost/latency predictable end-to-end.


_PROVIDER = LLMProvider.GEMINI
_MODEL = "gemini-3-flash-preview"
_MODEL_KWARGS: dict = {
    "thinking_config": {"thinking_budget": 0},
    "temperature": 0.15,
}


def _render_sibling_section(siblings: list[Trait]) -> str:
    """Render structural-only context about sibling traits in the query.

    Sibling info surfaces ONLY the structural fields downstream
    decomposition needs to act correctly:
    - surface_text: the sibling's identity at a glance
    - relationship_role: what role the sibling plays
    - replaces_axis: which axis the sibling substitutes (when it is
      a POSITIONING_QUALIFIER)
    - axes_replaced_by_siblings: which axes the sibling has marked
      as replaced (when it is a POSITIONING_REFERENCE)

    DELIBERATELY OMITTED: siblings' evaluative_intent, qualifier_relation,
    contextualized_phrase, commitment, commitment_evidence. Including
    those would leak the siblings' interpretive prose into this trait's
    routing — exactly the failure mode the per-trait isolation in V3
    was designed to avoid. The V4 sibling section is structural-only:
    enough to honor cross-trait scope replacement without leaking
    sibling decomposition decisions.

    Returns an empty string when no siblings exist (single-trait query)
    so the prompt stays clean.
    """
    if not siblings:
        return ""
    lines: list[str] = ["", "Sibling traits (structural context only):"]
    for sibling in siblings:
        lines.append(f'- surface_text: "{sibling.surface_text}"')
        lines.append(f"  relationship_role: {sibling.relationship_role.value}")
        if sibling.replaces_axis is not None:
            lines.append(f'  replaces_axis: "{sibling.replaces_axis}"')
        if sibling.axes_replaced_by_siblings:
            joined = ", ".join(
                f'"{axis}"' for axis in sibling.axes_replaced_by_siblings
            )
            lines.append(f"  axes_replaced_by_siblings: [{joined}]")
    return "\n".join(lines) + "\n"


def _build_user_prompt(trait: Trait, siblings: list[Trait]) -> str:
    """Render the per-trait user prompt.

    Step 3 receives the trait under decomposition plus a structural-
    only view of its sibling traits. The query-level intent_exploration
    prose is deliberately NOT surfaced here: it lives at Step 2 where
    it shapes the per-trait commits.

    Sibling info is structural-only (surface_text, relationship_role,
    replaces_axis, axes_replaced_by_siblings). Sibling evaluative_intent
    / contextualized_phrase / commitment are deliberately omitted to
    prevent cross-trait interpretation leakage. See
    `_render_sibling_section` for the rationale.

    The LLM gets the trait's contextualized_phrase (headline trait
    identity with modifiers folded in — Step 2's defense against
    shortcut routing on bare surface phrases), surface_text
    (verbatim grounding), evaluative_intent (the semantic seed),
    qualifier_relation, relationship_role, and any axis fields
    (drive trait_role_analysis and the identity-vs-attribute
    principle, plus axis-drop behavior for positioning references),
    anchor_reference, and polarity (informational — calls express
    presence regardless).

    `commitment` is intentionally NOT passed: it weights how the
    trait scores in the final query, but Step 3 acts identically on
    every trait regardless of importance level — decomposition
    discipline applies uniformly.

    contextualized_phrase appears AHEAD of surface_text — bare
    surface phrases stripped of their query context invite shortcut
    routing decisions that miss the modifier's role. surface_text
    is kept below for verbatim grounding.

    The qualifier_relation and anchor_reference fields are printed
    verbatim including the literal "n/a" sentinel — Step 3 reads
    "n/a" on qualifier_relation as an explicit "no positioning
    relationship; trait names a population to retrieve" signal, so
    we never conditionally hide it.

    Polarity is informational at this layer. Step 3 always
    describes presence of attributes; how the orchestrator composes
    calls is entirely orchestrator-side and does not need Step 3 to
    commit anything different in its decomposition.
    """
    base = (
        "Trait to decompose:\n"
        f'- contextualized_phrase: "{trait.contextualized_phrase}"\n'
        f'- surface_text (verbatim): "{trait.surface_text}"\n'
        f"- evaluative_intent: {trait.evaluative_intent}\n"
        f"- qualifier_relation: {trait.qualifier_relation}\n"
        f"- relationship_role: {trait.relationship_role.value}\n"
    )
    if trait.replaces_axis is not None:
        base += f'- replaces_axis: "{trait.replaces_axis}"\n'
    if trait.axes_replaced_by_siblings:
        joined = ", ".join(
            f'"{axis}"' for axis in trait.axes_replaced_by_siblings
        )
        base += f"- axes_replaced_by_siblings: [{joined}]\n"
    base += (
        f"- anchor_reference: {trait.anchor_reference}\n"
        f"- polarity: {trait.polarity}    "
        "(informational — calls express presence regardless)\n"
    )
    return base + _render_sibling_section(siblings)


async def run_step_3(
    trait: Trait,
    siblings: list[Trait] | None = None,
) -> tuple[TraitDecomposition, int, int, float]:
    """Run the trait-decomposition LLM on a single trait.

    Args:
        trait: the committed trait from Step 2.
        siblings: the other Step-2 traits in the same branch. Surfaced
            to the LLM as structural-only context (no evaluative_intent
            / contextualized_phrase / commitment) so the trait can
            honor cross-trait scope replacement without leaking
            sibling interpretive prose. Defaults to an empty list,
            which preserves V3 single-trait behavior for callers
            that haven't migrated.

    Returns:
        (response, input_tokens, output_tokens, elapsed_seconds) —
        elapsed measures wall-clock time spent inside the LLM call
        only, not prompt setup.
    """
    user_prompt = _build_user_prompt(trait, siblings or [])

    # perf_counter: monotonic, high-res wall-clock for short intervals.
    start = time.perf_counter()
    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=_PROVIDER,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=TraitDecomposition,
        model=_MODEL,
        **_MODEL_KWARGS,
    )
    elapsed = time.perf_counter() - start

    return response, input_tokens, output_tokens, elapsed
