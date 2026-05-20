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
#   3. Combine-mode commit — TraitCombineMode (SOLO / FRAMINGS /
#      FACETS). Committed AFTER candidates and BEFORE category_calls
#      so the mode shapes which categories make sense to commit. The
#      decision is hierarchical: SOLO when one category cleanly covers
#      every dimension (single committed call, passthrough at stage-4);
#      FRAMINGS when multiple categories are alternative homes for one
#      signal (stage-4 MAX-folds); FACETS when categories cover
#      distinct compounding axes (stage-4 PRODUCT-folds).
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

from google.genai import types

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)
from schemas.step_2 import Trait
from schemas.step_3 import TraitDecomposition
from schemas.trait_category import CategoryName
from search_v2.vague_temporal_vocabulary import (
    VAGUE_TEMPORAL_VOCABULARY_COMPACT,
)


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

0. TRAIT RESTATEMENT — verbatim reproduction of the trait's \
   upstream commit (contextualized_phrase, evaluative_intent, \
   relationship_role, axis bookkeeping). Produced FIRST, before \
   any inference. This is the anchor every later field reads \
   from; content not present here is content that was not in \
   the trait, and must not appear downstream.
1. TRAIT-ROLE ANALYSIS — target_population + trait_role_analysis. \
   Read relationship_role / qualifier_relation / replaces_axis / \
   axes_replaced_by_siblings mechanically off the trait_restatement \
   you just produced (these values were committed by Step 2 and \
   the restatement reproduces them verbatim) and commit what they \
   mean for what the dimensions describe. Pre-dimension; \
   constrains the inventory.
2. ASPECTS — flat list enumerating every distinguishable axis the \
   trait calls for, in user-vocabulary. Sits between the prose \
   role analysis and the dimension routing layer. Aspect strings \
   are LOAD-BEARING: the dimensions step below copies each into \
   its expression field VERBATIM, character-for-character. Honors \
   role-driven axis constraints: POSITIONING_REFERENCE traits \
   drop replaced axes; POSITIONING_QUALIFIER traits cover the \
   replacement axis.
3. DIMENSION INVENTORY + CANDIDATES — one dimension per aspect \
   routed. Each dimension's expression is the aspect string \
   verbatim (no rewriting, no paraphrasing, no merging). The \
   dimension carries plausible category candidates with what's-\
   covered / what's-missing prose — the routing decisions, not a \
   re-authoring of the aspect.
4. ROUTING EXPLORATION — exploratory prose between candidates and \
   combine_mode. Walks dedup, the granularity gate (identity vs \
   attribute categories filtered by aspect granularity), the \
   coverage check (does one surviving category cover every \
   dimension cleanly → SOLO), and only when no SOLO candidate \
   exists, the FRAMINGS-vs-FACETS relationship question plus \
   minimum-set commit. Forward reasoning that produces the \
   conclusions combine_mode and category_calls then read off.
5. COMBINE MODE — TraitCombineMode (SOLO / FRAMINGS / FACETS). \
   Mechanical translation of the routing_exploration conclusion. \
   Tells stage-4 how to fold per-category scores: SOLO → \
   passthrough (one category covers everything alone); FRAMINGS → \
   MAX (alternative homes for one signal, no single category \
   suffices); FACETS → PRODUCT (distinct compounding axes).
6. COMMITMENT LAYER — category_calls. One call per category that \
   ends up owning >=1 expression. Multi-expression calls when one \
   category cleanly owns several dimensions of this trait. The \
   set of categories and the dedup decisions were committed by \
   routing_exploration above; this layer is the mechanical \
   structured-call translation.

Read the response schema's field descriptions before producing \
output.

Two phases drive the work.

ANALYSIS PHASE — produce the trait restatement, then restate the \
population, commit the role analysis, enumerate aspects (honoring \
role-driven axis constraints), translate aspects into dimensions, \
list per-dimension candidates with explicit coverage prose, and \
finish with routing_exploration that prunes the candidate set and \
argues for SOLO vs FRAMINGS vs FACETS. No calls yet.

COMMITMENT PHASE — read routing_exploration's conclusions, commit \
combine_mode (mechanical from the relationship argument), then \
emit minimum calls (mechanical from the minimum-set conclusion). \
Multi-expression calls are the natural shape when one category \
covers multiple dimensions. Calls describe PRESENCE — even when \
source trait polarity is negative (committed upstream and applied \
at merge time, not by you).

The sections below cover analysis (trait-role analysis, aspect \
enumeration, dimension inventory, per-dimension candidates, \
combine-mode commit), then commitment (category routing, \
minimum-set + polarity discipline), then the full category \
taxonomy.

---

"""


_TRAIT_RESTATEMENT = """\
TRAIT RESTATEMENT — the first field you produce, before any \
inference

Your output schema declares trait_restatement BEFORE every other \
field. That ordering is load-bearing. The auto-regressive nature \
of structured generation means whatever you write first becomes \
the literal scaffold the rest of the response is conditioned on; \
this field exists to make that scaffold be the upstream commit, \
not a freely-elaborated paraphrase.

WHAT TO WRITE. Reproduce, in order, character-for-character:

(1) The trait's contextualized_phrase, in double quotes. Copy it \
    verbatim. No paraphrasing. No tightening. No expansion.
(2) The trait's evaluative_intent, in double quotes. Copy it \
    verbatim. No re-interpretation, no summary.
(3) The trait's relationship_role value, in single quotes.
(4) If replaces_axis is non-null: that value, in single quotes.
(5) If axes_replaced_by_siblings is non-empty: the list as a \
    bracketed series of single-quoted strings.

Items (4) and (5) are omitted when their source fields are \
empty / null. Items (1)–(3) are always present. Order is fixed.

WHAT THIS FIELD IS FOR. trait_restatement is the anchor every \
subsequent field reads from. target_population, trait_role_\
analysis, aspects, every dimension expression, every category \
call's expressions and retrieval_intent — each of those must \
describe the trait this restatement names, at the same width and \
scope. Content not present inside the quoted strings here is \
content that was not in the trait; do not introduce it \
downstream.

WHY THIS FIELD EXISTS. Earlier versions of this prompt asked the \
model to "anchor on upstream commits" via prose rules. The \
upstream evaluative_intent often described the criterion at the \
right width, but downstream fields silently broadened, narrowed, \
or invented detail anyway — because the prose rules sit alongside \
elaboration-positive language, and the model resolved the tension \
by elaborating in conventional directions drawn from training-\
data priors. A required restatement field, produced first, makes \
the upstream commit explicit in the model's own generation \
stream. The subsequent fields condition on what is literally \
present in the restatement, not on what the model "knows" the \
criterion typically means.

NEVER:
- PARAPHRASE THE QUOTED STRINGS. The whole point is verbatim \
  reproduction. "Films that won awards" is NOT a faithful \
  restatement of "films recognized with awards"; "criteria for X" \
  is NOT a faithful restatement of "the search prioritizes X". \
  Copy character-for-character.
- SUMMARIZE OR TIGHTEN. The downstream fields will paraphrase and \
  decompose; this field does not. If the upstream string is long, \
  it is still copied in full.
- ADD COMMENTARY. No "this means…", no "in other words…", no "the \
  user is asking for…". The restatement contains only the \
  enumerated items above, in order, with no surrounding gloss.
- OMIT FIELDS THAT ARE PRESENT. If replaces_axis or \
  axes_replaced_by_siblings has content, it appears. The \
  restatement reflects the actual upstream state.

---

"""


_TRAIT_ROLE_ANALYSIS = """\
TRAIT-ROLE ANALYSIS — translate Step 2's commits into a \
constraint on what the dimensions list should describe

Before enumerating dimensions, commit a 1-2 sentence analysis \
that answers, specifically for this trait: WHAT KIND of \
dimensions belong, and what kind don't? The fields below are \
committed upstream by Step 2; read them as the source of truth, \
do not re-derive from evaluative_intent. Anchor every clause of \
your trait_role_analysis (and the target_population you committed \
just above it) on the quoted upstream strings inside the \
trait_restatement field you produced first — they are the \
in-context record of what the trait actually said.

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

FIDELITY TEST. Read target_population back against the quoted \
strings inside trait_restatement (the verbatim contextualized_\
phrase and evaluative_intent the field reproduces). For each \
constraint in target_population, point to the words inside those \
quotes that licensed it. For each constraint inside those quotes, \
point to the words in target_population that preserved it. A \
constraint present in one and absent from the other is the drift \
this stage exists to prevent — fix the population, do not paper \
over the gap downstream. trait_restatement is the authoritative \
in-context record; if a clause cannot be anchored there, it does \
not belong here.

---

IDENTITY VS ATTRIBUTE CATEGORIES — a structural rule that combines \
relationship_role with aspect granularity. Some categories retrieve \
specific named entities: PERSON_CREDIT, TITLE_TEXT_LOOKUP, \
NAMED_CHARACTER, STUDIO/BRAND, FRANCHISE/UNIVERSE_LINEAGE, \
CHARACTER_FRANCHISE, ADAPTATION_SOURCE_FLAG, \
BELOW_THE_LINE_CREATOR, NAMED_SOURCE_CREATOR. Others describe \
attributes a film can have (genre, emotional/experiential, \
narrative setting, story/thematic archetype, visual-craft acclaim, \
etc.). Two layers govern when identity categories are admissible.

LAYER 1 — GRANULARITY GATE (applies to every trait, every role, \
both polarities). Identity categories retrieve specific named \
entities; they are admissible only when the aspect itself names a \
specific entity that lives in that category. When the aspect \
names a CATEGORY (a genre, an archetype, a kind-of-movie defined \
by what its members share), identity categories are out-of-scope \
even if prior knowledge associates specific named instances with \
that category. The discriminator is whether the aspect names a \
MEMBER of the category or names the category itself. This gate is \
the structural enforcement of Categorical-to-Specific Drift; the \
ROUTING EXPLORATION step below operates the gate per-dimension.

LAYER 2 — ROLE-DRIVEN RESTRICTIONS (applied after granularity \
passes).

INDEPENDENT traits: identity OR attribute categories are both \
fair game when the aspect passes the granularity gate. The role \
analysis commits "dimensions describe the subject", and the \
subject can be defined by an identity category (only when the \
aspect names a specific entity), attribute categories, or both. \
A negative-polarity INDEPENDENT trait that names a CATEGORY of \
films routes only to attribute categories — the granularity gate \
still applies.

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

Why this step exists. The next step (dimensions) routes each \
aspect to its plausible categories, with the aspect string itself \
copied VERBATIM into the dimension's expression field. The \
dimensions layer does not re-author or translate; it selects \
which aspects to route and which categories could own them. That \
makes the aspects you write here load-bearing in two ways: \
(1) any axis you skip cannot be recovered downstream — the \
routing step has nothing to route; (2) any awkwardness, vagueness, \
or drift in an aspect string propagates character-for-character \
into the expression and on into the endpoint handler — the \
routing step is forbidden from "fixing it up" later. Get the \
aspects right HERE so the verbatim copy downstream lands on \
clean strings.

GROUNDING. Every aspect must trace back to something explicit in \
target_population or trait_role_analysis. If an aspect doesn't \
appear there, either revise the role analysis (an axis was \
missed upstream) or drop the aspect (it was invented, not \
grounded).

TRUTH-CONDITION PRESERVATION. Tracing back to the population is \
not enough on its own — each aspect must preserve the constraint \
at the SAME granularity target_population stated it. Three drift \
modes corrupt the enumeration even when traceability passes:

- IMPLICIT BROADENING. target_population names a narrow truth \
  condition and the aspect widens it into the surrounding category \
  by adding an "or"-clause that admits neighboring sub-conditions \
  the population did not license.
- IMPLICIT NARROWING. target_population names a category and the \
  aspect pins it to a single canonical sub-type or exemplar of \
  that category.
- INVENTED ADDITIONAL CONDITION. The aspect bundles in a related-\
  but-not-stated condition the population never named, drawn from \
  the model's prior knowledge of what "typical" coverage of this \
  trait looks like.

FIDELITY TEST FOR ASPECTS. Read each aspect back against \
target_population. The aspect names the same constraint at the \
same width — no broader, no narrower, no extra conditions bundled \
in. If an aspect contains a clause target_population does not \
license, the aspect is drifted; tighten it before moving to \
dimensions. Drift here propagates directly into the dimension and \
expression layers — they cannot recover a constraint the aspect \
already lost.

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
- BROADEN, NARROW, OR BUNDLE EXTRA CONDITIONS against the \
  constraint target_population stated. Tracing back to the \
  population is not the same as preserving its width; an aspect \
  that names the same axis at a looser, tighter, or compounded \
  granularity is drifted even if its topic appears in the \
  population prose.
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
DIMENSION INVENTORY — emit one dimension per aspect routed, with \
expression as the aspect verbatim

A dimension is a routing slot for one aspect: it pairs the aspect \
(carried VERBATIM in the expression field) with the list of \
plausible categories that could own it (recorded in \
category_candidates). The dimension layer is where you decide WHICH \
aspects to route and WHAT categories they could route to; it is \
NOT where you re-author the aspects in different words.

VERBATIM EXPRESSION — the load-bearing rule of this layer.

Each Dimension.expression MUST be a character-for-character copy \
of one of the strings in the aspects list above. No rewriting. \
No tightening. No rewording. No "close enough" paraphrase. No \
merging two aspects into one phrase. If an aspect reads "highly \
praised for performances", the dimension that routes it emits \
expression="highly praised for performances" — exactly that, not \
"praised for performances" and not "performances are praised". \
The translation work — turning user-vocabulary into a database-\
runnable check — happens DOWNSTREAM in category_candidates and in \
the endpoint handler. expression is purely a pointer back to the \
aspect this dimension covers.

If an aspect feels awkward to copy verbatim — too vague, too \
narrow, oddly phrased — that is a signal the aspect itself is \
wrong. FIX IT UPSTREAM (revise the aspects list) instead of \
rephrasing it here. Rephrasing at the dimension layer is the \
silent path by which truth conditions, named entities, and width \
constraints get reshaped into something the user did not ask for.

PRIMARY SOURCE: aspects. Walk the aspects list end to end. For \
EACH aspect, commit at least one dimension whose expression is \
that aspect verbatim. The mapping is at-least-one aspect → at-\
least-one dimension. Even if two aspects feel like they could \
share a routing slot, keep them as separate dimensions here; \
merging happens at the category_calls layer, where same-category \
dimensions collapse into one multi-expression call. Pre-merging \
at the dimension layer is how aspects get silently dropped.

Multiple dimensions MAY share the same aspect-string when the \
aspect genuinely routes through more than one category — emit \
one dimension per routing option, each with the SAME verbatim \
aspect string in expression and DIFFERENT category_candidates \
lists. This is the rare case; one-dimension-per-aspect is the \
default.

CONTEXTUAL SOURCES: target_population + trait_role_analysis. \
Read them to understand each aspect more deeply — what does the \
axis really mean for this query, what kind of categories honor \
the role-scope constraint? They are NOT sources of new \
dimensions and they are NOT inputs that license rephrasing the \
aspect; they are interpretation aids for choosing category \
candidates.

ROLE-SCOPE CONSTRAINT. The set of dimensions you emit must obey \
the trait_role_analysis. If the analysis constrained the trait to \
describe the reference's identifiable attributes (rather than the \
population), every dimension's aspect is one of those attributes — \
slipping in a population-shaped aspect-routing violates the role \
commitment. The constraint applies to WHICH aspects you route, \
not to how you phrase them.

COVERAGE IS NON-NEGOTIABLE.
- Every aspect addressed by at least one dimension. An aspect \
  with no dimension is dropped coverage — the failure mode this \
  whole structure exists to prevent.
- Every dimension's expression is an exact copy of an entry in \
  the aspects list. A dimension whose expression is not present \
  verbatim in aspects is either invented (smuggling in content \
  not in aspects) or rewritten (paraphrasing an aspect into a \
  different phrase) — both violations.
- An aspect that seems hard to route does NOT get silently \
  dropped. If you cannot find any plausible category for it, the \
  aspect itself was wrong (too abstract, not actually a separate \
  axis) — revise the aspect list, do not skip it here.

OPERATIONAL TESTS:
- "For each aspect string, is there a dimension whose expression \
  is exactly that string?" Yes → keep. No → add a dimension; do \
  not delete the aspect.
- "For each dimension, can I find its expression verbatim in the \
  aspects list?" Yes → keep. No → fix the expression to match an \
  aspect verbatim, or fix the aspects list, or drop the dimension.
- "If I diff each dimension's expression against the aspects \
  list, is the match character-for-character?" Any non-match \
  (even a single deleted word, even a tightened phrase) is a \
  rewrite — undo it.

COMMON PITFALLS.

- DROPPED ASPECT. The aspect appears in the list above but no \
  dimension addresses it. Most common failure mode of this \
  layer — the aspect "felt hard to route" so the model quietly \
  skipped it. Route it or revise upstream; never skip.
- REWRITTEN ASPECT. The dimension's expression is recognizably \
  derived from an aspect but not identical to it ("highly praised \
  for performances" → "praised for performances"; "won acting \
  awards" → "acting award wins"). Even small edits violate the \
  verbatim rule. Copy the aspect string character-for-character.
- PRE-MERGED ASPECTS. Two aspects collapsed into one dimension \
  whose expression blends both ("acting awards" + "acting \
  nominations" → "acting award wins and nominations"). The \
  blended expression is no longer verbatim equal to either source \
  aspect. Emit TWO dimensions, each with its own verbatim aspect \
  string. Merging is the category_calls layer's job.
- INVENTED DIMENSION. The dimension's expression isn't present \
  verbatim in the aspects list. Likely smuggled in from prior \
  knowledge of the trait's typical shape rather than from the \
  aspects list. Drop or trace to an aspect.
- CATEGORY NAMING. Categories belong to candidates / calls, not \
  expressions. expression is the aspect verbatim; let \
  category_candidates record the routing options.
- ABSENCE FRAMING. Polarity was committed upstream. Even when the \
  trait is negative, expressions describe presence (the attribute \
  being avoided). Merge step flips direction; you don't. If an \
  aspect was framed as absence ("lack of whimsy"), the aspect \
  itself should be the presence form ("whimsy") — fix it \
  upstream, not by rephrasing here.
- BUNDLING UNRELATED CHECKS into one Dimension.expression. One \
  dimension = one aspect. Multiple same-category aspects aren't \
  bundling — they belong as separate dimensions that merge into \
  one multi-expression call later.
- EXAMPLE-IN-EXPRESSION. expression carries example entities the \
  trait never named — parenthetical lists, "such as X, Y, or Z" \
  formulations, canonical exemplars. Because expression is now \
  required to match an aspect verbatim, this pitfall is caught \
  earlier: if expression has parentheticals, either the aspect \
  itself has them (fix the aspect — apply the upstream fidelity \
  rules) or the expression has been rewritten (forbidden by the \
  verbatim rule). Either way, the fix is upstream, not here.
- PADDING. Don't add dimensions whose expression isn't present \
  in the aspects list. Every dimension's expression traces to a \
  verbatim aspect string.

---

"""


_PER_DIMENSION_CANDIDATES = """\
PER-DIMENSION CATEGORY CANDIDATES — broad routing analysis to feed \
the consolidation step

For each dimension, list plausible categories with explicit what-\
this-covers / what-this-misses prose. Partial fits and adjacency \
surface explicitly so the consolidation step that follows has \
real evidence to reason against.

LEAN BROAD HERE. The next step (ROUTING EXPLORATION) is where the \
candidate set gets pruned to the minimum useful commit. This \
layer's job is to surface ENOUGH options for that pruning to work \
honestly — not to pre-commit. When in doubt, surface the candidate \
and let the consolidation step decide.

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
dimensions of this trait, that dedup will be picked up by the \
consolidation step below. You do not need to narrate it here.

OPERATIONAL TESTS:
- HONESTY. "If this dimension has only one candidate AND adjacent \
  categories could plausibly compete, am I doing the audit \
  honestly?" Surface the adjacent candidate even if you'll reject \
  it.
- SUBSTANCE. what_this_covers must point to a real overlap with \
  the dimension; "this could fit" is not coverage.
- NO BACK-RATIONALIZATION. "Did I list this because I already \
  decided to commit, or because the boundary actually puts this \
  dimension in scope?" Pruning happens in the next step; do not \
  shortcut by emitting only what you intend to commit.
- NO INVENTED EXEMPLARS in what_this_covers. The candidate's \
  coverage is described structurally (what kind of dimension it \
  serves), not by naming specific entities the model thinks would \
  match. Naming a particular franchise / character / studio inside \
  what_this_covers leaks Categorical-to-Specific Drift into the \
  routing layer.

DON'T COMMIT THE CALL LIST HERE. Candidates is analysis; \
category_calls is where calls commit, after the consolidation step. \
Keeping them separate prevents premature commitment.

DON'T LEAVE DIMENSIONS WITH AN EMPTY CANDIDATE LIST. If you can't \
find any category whose boundary even partially covers, the \
dimension itself is wrong (too abstract, too compound) — revisit.

---

"""


_ROUTING_EXPLORATION = """\
ROUTING EXPLORATION — think before committing combine_mode and \
category_calls

After the per-dimension candidate analysis above, BEFORE committing \
combine_mode and category_calls, walk the candidate set as a whole \
and decide which categories actually deserve commits. The candidate \
analysis was deliberately allowed to be broad — this step is where \
breadth gets pruned to the minimum useful set, AND where the \
relationship among surviving categories gets named.

Write in FORWARD-REASONING shape: "I see X across these dimensions, \
so I'm leaning toward Y." Not back-rationalization: "My answer is \
Y because X." The fields below copy your conclusions; the \
conclusions themselves are reached here.

WALK FOUR CHECKS, IN ORDER. The order is load-bearing — earlier \
checks shrink the candidate set or short-circuit the decision \
entirely, so applying them out of sequence inflates the call list.

(1) DEDUP. Look at category_candidates across all dimensions. \
When the same category appears as a clean-fit (or near-clean-fit) \
on more than one dimension, note it — those dimensions will merge \
into one multi-expression call at the commitment step. Identifying \
merges here keeps the call list minimal without dropping coverage.

(2) GRANULARITY GATE. For each candidate still in play, ask \
whether the aspect it serves is at the granularity the category \
retrieves. Identity categories retrieve specific named entities; \
they are admissible ONLY when the aspect itself names a specific \
entity that lives in that category. When the aspect names a \
CATEGORY — a genre, an archetype, a kind-of-movie defined by what \
its members share — identity-category candidates are out-of-scope \
no matter how cleanly prior knowledge associates specific named \
instances with that category. Drop them here.

The discriminator: does the aspect string name a MEMBER of the \
category, or does it name the category itself? Members route to \
identity; categories route to attribute.

(3) COVERAGE — the SOLO short-circuit. After dedup and granularity \
pruning, ask whether a single surviving category cleanly covers \
EVERY dimension this trait routes. The per-dimension \
what_this_misses analysis is the source of truth: a candidate that \
named no substantive gap (the "Nothing" / no-real-gap form) across \
every dimension it could serve provides complete coverage by \
itself. When that is the case, the trait is SOLO — that category \
is the only commit, every other candidate was adjacency context \
that surfaced for honesty rather than coverage the primary lacked. \
State this conclusion explicitly when it applies.

The discipline of this check: a candidate whose what_this_misses \
names a real gap on any dimension does not heal the gap by being \
paired with another partial candidate that misses something else. \
What heals a gap is a single category whose own coverage closes it. \
When such a category exists, the trait commits SOLO regardless of \
how cleanly adjacent candidates might also have routed.

Only if NO single surviving candidate covers everything cleanly \
does step (4) come into play.

(4) RELATIONSHIP & MINIMUM SET (skipped when SOLO applies). Among \
the surviving candidates, are they alternative HOMES for one \
underlying thing the trait names (matching any one is sufficient \
evidence → leans FRAMINGS), or do they describe DIFFERENT axes of \
a compound concept the user wants compounded (all must fire to a \
degree → leans FACETS)? State which direction the dimensions \
actually support and why, then commit the minimum set of \
categories that together cover every dimension. Each surviving \
category must add NEW signal — if a category is redundant with a \
stronger sibling and FRAMINGS gives no incremental reinforcement \
(or FACETS would double-fire on the same axis), drop it.

CATEGORICAL-TO-SPECIFIC DRIFT — the named failure mode this step \
exists to catch. When the trait names a category (a genre, an \
archetype, a kind-of-movie defined by what its members share), do \
not commit a category whose job is to retrieve specific named \
instances of that category. The category is broader than any \
single instance; the specific instance covers only a part. \
Routing the trait to identity categories swaps the trait's truth \
conditions for a strictly narrower set, and adding multiple \
specific instances as 'framings' does not heal the gap — it just \
narrows the scope to a hand-picked subset.

This applies to BOTH polarities. For positive traits, the drift \
inflates the score on a narrow subset of films that happen to be \
canonical exemplars of the category, instead of scoring the whole \
category evenly. For negative traits, the drift over-excludes \
films that are adjacent to the canonical exemplars while missing \
other films that genuinely belong to the category. Polarity does \
not heal the drift; it changes which way the error tilts.

OPERATIONAL TESTS:
- GRANULARITY PASS. "Did I reach for a particular franchise / \
  character / studio / person name to justify a candidate the \
  trait did not itself name?" Yes → drop the candidate; the \
  Categorical-to-Specific Drift trap is firing.
- SAME-LEVEL CHECK. "Are the surviving candidates at the same \
  granularity as the aspects they serve?" If a candidate retrieves \
  specific instances but the aspect names a category, it does not \
  survive.
- COVERAGE CHECK. "Is there a single surviving category whose \
  what_this_misses analysis showed no substantive gap across every \
  dimension this trait routes?" Yes → SOLO; drop the other \
  candidates. No → proceed to FRAMINGS vs FACETS.
- MINIMUM-SET CHECK. "If I removed each surviving candidate one \
  at a time, would the remaining set lose meaningful coverage?" If \
  no for any candidate, that candidate was padding.

NEVER:
- SKIP THE COVERAGE CHECK and jump straight to FRAMINGS vs FACETS. \
  When one category cleanly covers the trait, the relationship \
  question has no work to do — the answer is SOLO.
- COMMIT EXTRA CATEGORIES AS HEDGES. When a clean primary covers \
  the trait, partial-coverage adjacents do not reinforce the \
  signal; they pad the call list. The commit is the minimum \
  sufficient set, not the broadest defensible set.
- WRITE BACKWARD FROM A COMMITTED ANSWER. The reasoning leads the \
  conclusion, not justifies it.
- ELABORATE BEYOND THE CANDIDATES. Only categories surfaced in \
  the candidate analysis above are in scope here.
- USE THIS FIELD TO INTRODUCE NEW NAMED ENTITIES. The trait \
  either named the entity or it didn't. The exploration is a \
  pruning pass, not a re-expansion pass.

---

"""


_COMBINE_MODE_COMMIT = """\
COMBINE MODE — commit how stage-4 will fold this trait's \
per-category scores into a trait_score

After the routing exploration above (which decided the surviving \
category set AND argued for SOLO vs FRAMINGS vs FACETS), this \
field is a MECHANICAL TRANSLATION of that conclusion into the \
closed enum. If you find yourself revisiting the question here, \
you didn't reason it through above — revise the exploration, then \
come back. Get this wrong and the next step's category routing \
optimizes for the wrong combine.

THREE MODES.

SOLO — exactly one surviving category cleanly covers every \
dimension the trait calls for. The other candidates surfaced as \
adjacency context but do not add coverage the primary doesn't \
already provide. Stage-4 has nothing to fold; the single \
category's score IS the trait_score. The signature: the routing \
exploration's coverage check found one candidate whose \
what_this_misses analysis named no substantive gap across every \
dimension it could serve.

FRAMINGS — multiple categories are alternative HOMES for the same \
underlying thing, AND no single category cleanly covers the trait \
on its own. Matching ANY ONE is sufficient evidence of the \
criterion. Stage-4 takes the MAX across categories; redundancy \
between framings reinforces as alternative routes to the same \
signal. The signature: every surviving candidate's \
what_this_misses named some gap that another surviving candidate's \
coverage fills — the candidates partition the trait's coverage \
along the same underlying axis from different angles.

FACETS — categories cover DIFFERENT axes of a compound concept. \
ALL facets must fire to a degree for the criterion to be met. \
Stage-4 takes the PRODUCT across categories; duplicating axis \
coverage AMPLIFIES the wrong signals. The signature: each \
dimension's clean-fit category covers a DISTINCT IDENTIFIABLE \
AXIS the user wants compounded — the categories partition the \
trait into independently-varying conditions, not alternative \
routes to one signal.

DISCRIMINATOR. The decision is hierarchical, not a three-way pick. \
Ask the coverage question first, then the relationship question \
only if needed:

(a) Did the routing exploration find one category whose coverage \
analysis closed every gap across every dimension this trait \
routes? Yes → SOLO. Other candidates were adjacency context and \
do not earn a commit. The downstream pipeline trims any extras \
emitted under SOLO before retrieval, so the trait_score depends \
on this single category alone.

(b) Only when no single-coverage category exists, ask the \
read-back: "if a candidate film matched only ONE of the categories \
at high score and 0 on the others, would the user agree the trait \
is satisfied?" Yes → FRAMINGS. No → FACETS.

ROUTING IMPLICATIONS the next step will read:

- SOLO means category_calls contains exactly ONE entry. List the \
  clean-fit primary first; extras are trimmed before retrieval and \
  never reach the endpoints.
- FRAMINGS authorizes committing categories whose coverage \
  OVERLAPS. The system MAX-folds them, so multiple framings of \
  one thing are intentional reinforcement when no single category \
  covers the trait cleanly on its own.
- FACETS DEMANDS choosing categories that COMPLEMENT rather than \
  overlap. The system PRODUCT-folds them, so duplication of axis \
  coverage amplifies the wrong signals. Each committed category \
  should cover a distinct axis surfaced by the dimension list. \
  Two committed categories covering the same axis under FACETS is \
  a COMMIT BUG — it makes the product strict on the same axis \
  twice.

OPERATIONAL TESTS:
- COVERAGE CHECK. "Does a single surviving category close every \
  gap across every dimension this trait routes?" Yes → SOLO. No → \
  proceed to the read-back.
- READ-BACK (when SOLO does not apply). "If a candidate film hits \
  ONE of the surviving categories at 1.0 and scores 0 on the rest, \
  is the trait satisfied?" Yes → FRAMINGS. No → FACETS.
- AXIS-COUNT (when SOLO does not apply). "Looking at the dimensions \
  list, how many distinct user-vocabulary axes remain uncovered by \
  any single category?" Multiple framings of one signal → FRAMINGS. \
  Multiple distinct compounding conditions → FACETS.

NEVER:
- COMMIT FRAMINGS WHEN ONE CATEGORY COVERS THE TRAIT CLEANLY. \
  Adding partial-coverage adjacents under FRAMINGS does not \
  reinforce the signal; it pads the call list. The correct commit \
  when one category suffices is SOLO.
- COMMIT FACETS WHEN CATEGORIES ARE FRAMINGS OF ONE IDENTITY. \
  PRODUCT-folding penalizes the reference for failing to surface \
  in EVERY category simultaneously, which is wrong for identity-\
  style traits with multiple plausible homes.
- COMMIT FRAMINGS WHEN CATEGORIES ARE FACETS OF A COMPOUND. \
  Compound concepts where the user wants several axes to compound \
  do NOT satisfy on a single-axis match. MAX-folding lets \
  single-facet matches win at 1.0, which is exactly the failure \
  mode FACETS is designed to prevent.
- DEFAULT-FILL. The mode is a real commit driven by the candidate \
  analysis, not a placeholder.

---

"""


_CATEGORY_ROUTING = """\
CATEGORY ROUTING — committing the call list

Read the routing_exploration's minimum-set conclusion and the \
combine_mode that followed. The category set, dedup decisions, and \
granularity pruning were all reasoned through upstream — this \
field is the mechanical translation into structured calls. For \
each category that ends up owning >=1 expression, emit ONE \
CategoryCall. When several dimensions share a best-fit category, \
they merge into a single multi-expression call — NOT separate \
calls.

If your call list diverges from the minimum-set the exploration \
committed (extra calls, missing categories, different dimension \
ownership), you skipped the upstream reasoning. Go back and \
revise the exploration; do not paper over the divergence here.

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
- retrieval_intent FIDELITY. Both expressions and retrieval_intent \
  are read verbatim by the downstream endpoint LLM, which treats \
  any named entity (ceremony, person, prize, franchise, decade, \
  studio, or other instance) as a literal filter value rather than \
  an illustration. Named entities therefore appear in either \
  string ONLY when the trait itself named them — never as \
  model-supplied examples of what the category typically contains. \
  When the trait named a category rather than specific instances, \
  retrieval_intent describes the category cleanly; parenthetical \
  example lists and "such as X, Y, or Z" formulations pin the \
  downstream query to those exemplars and miss everything else the \
  category covers.

READ COMBINE_MODE:
- SOLO means category_calls contains EXACTLY ONE entry. The single \
  clean-fit primary owns every dimension. Any extras emitted under \
  SOLO are trimmed before retrieval and never reach the endpoints; \
  list-ordering matters, so place the clean-fit primary first.
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
  - SOLO: exactly one category is committed — the one whose \
    coverage analysis closed every gap across every dimension. \
    Adjacency candidates surfaced earlier do not earn a call.
  - FACETS: if a dimension's candidates list contains one with \
    what_this_misses="nothing", commit ONLY that one. Other \
    candidates were adjacency context surfaced for honesty; \
    committing them under FACETS would PRODUCT-fold against the \
    same axis the clean-fit already covers, which dilutes the \
    score for the wrong reason.
  - FRAMINGS: multiple clean-fit categories across dimensions are \
    PERMITTED — they are alternative routes to the same signal \
    and reinforce under MAX. The CLEAN-FIT-only rule from FACETS \
    does not apply. FRAMINGS only applies when no single category \
    covers the trait on its own; if one does, the trait is SOLO.
- POLARITY-DISCIPLINE. Does this call describe presence? Reject \
  any expression or retrieval_intent that includes negation / \
  absence — polarity is upstream.
- POSITIONING-CONSISTENCY. Does framing match what \
  trait_role_analysis committed? INDEPENDENT → calls describe the \
  subject. POSITIONING_REFERENCE → calls describe the reference's \
  kept axes (no replaced axes). POSITIONING_QUALIFIER → at least \
  one call covers replaces_axis.
- MODE-CONSISTENCY. SOLO → exactly one entry. FACETS → no two \
  committed categories cover the SAME axis of the trait. FRAMINGS \
  → overlapping coverage across categories is permitted and often \
  correct.

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
collapse, not padding. What IS padding: extra calls beyond what \
coverage demands, two calls to the same category, or a call whose \
category was never a candidate on any of its claimed dimensions.

Most traits resolve to ONE call (combine_mode SOLO). Concrete \
single-axis traits → one call with one expression. Figurative \
multi-faceted traits whose dimensions all share one clean-fit \
category → one call with several expressions. Traits whose \
coverage genuinely splits across categories — alternative homes \
under FRAMINGS or distinct compounding axes under FACETS — are \
the cases that warrant multiple calls.

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
    + _TRAIT_RESTATEMENT
    + _TRAIT_ROLE_ANALYSIS
    + _ASPECT_ENUMERATION
    + _DIMENSION_INVENTORY
    + _PER_DIMENSION_CANDIDATES
    + _ROUTING_EXPLORATION
    + VAGUE_TEMPORAL_VOCABULARY_COMPACT
    + _COMBINE_MODE_COMMIT
    + _CATEGORY_ROUTING
    + _MINIMUM_SET_AND_POLARITY
    + _CATEGORY_TAXONOMY
)


# ===============================================================
#                      Executor
# ===============================================================
#
# Model is finalized to Gemini 3.5 Flash with thinking at minimal and
# a low temperature. Callers cannot override — this keeps the step
# reproducible and makes cost/latency predictable end-to-end.


_PROVIDER = LLMProvider.GEMINI
_MODEL = "gemini-3.5-flash"
_MODEL_KWARGS: dict = {
    "thinking_config": types.ThinkingConfig(thinking_level="minimal"),
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
