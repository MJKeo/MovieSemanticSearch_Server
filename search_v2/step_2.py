# Search V2 — Query Analysis stage.
#
# First stage of the multi-step query-understanding flow. Takes a
# raw natural-language query and emits a QueryAnalysis with three
# coupled outputs:
#
#   1. intent_exploration — query-level exploratory analysis.
#      Surface the plausible high-level intents and weigh which is
#      more likely. No verdict.
#   2. atoms — descriptive layer. Per-criterion records: surface_text,
#      modifying_signals, evaluative_intent. Atoms gather evidence.
#   3. traits — committed layer. Search-ready units derived from
#      atoms with polarity and commitment committed. Step 3
#      consumes traits.
#
# The prompt walks two phases:
#
#   ATOM PHASE — descriptive evidence gathering
#     atomicity → modifier vs atom → evaluative intent
#   COMMIT PHASE — read evidence, commit per trait
#     commit phase wrapper → polarity → commitment → relationship role
#   CATEGORY VOCABULARY — recognize-only; full taxonomy at Step 3
#
# Output-shape discipline lives in the response-schema field
# descriptions, not in the prompt. Schema = micro-prompts; prompt =
# procedural. Field-level "how to fill" guidance is not duplicated
# here.
#
# Model choice is finalized to Gemini 3.5 Flash with minimal
# thinking (`thinking_level="minimal"`) and modest temperature. The
# run function does not accept a model parameter — provider and
# model are hard-coded here so callers stay simple and behavior is
# reproducible.
#
# Usage:
#   python -m search_v2.step_2 "your query here"

from __future__ import annotations

import argparse
import asyncio
import json
import time

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)
from schemas.trait_category import CategoryName
from schemas.step_2 import QueryAnalysis
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
#   ATOM PHASE   — atomicity → modifier vs atom → evaluative intent
#   COMMIT PHASE — commit phase → polarity → commitment →
#                  relationship role
#   CATEGORY VOCABULARY (recognition-only)


_TASK_FRAMING = """\
You are the query-analysis stage of a movie-search pipeline. A raw \
natural-language query comes in; you produce three coupled outputs:

1. intent_exploration — query-level exploratory analysis. Surface \
   the plausible high-level intents the query could be expressing, \
   in concrete terms, and weigh which is more likely from the \
   query's context. Enumerate multiple reads only when they are \
   comparably plausible; when one read clearly dominates and the \
   alternative is a stretch, surface the dominant read alone rather \
   than manufacturing a low-probability alternative. No verdict; \
   downstream stages commit.
2. atoms — descriptive layer. surface_text + modifying_signals + \
   evaluative_intent + split_exploration + standalone_check. Atoms \
   record and analyze.
3. traits — committed layer. Search-ready units with polarity, \
   commitment, and structural relationship-role assigned. Step 3 \
   consumes this list.

Read the response schema's field descriptions before producing \
output.

intent_exploration is drafted FIRST, before atoms. It is the \
query-level perception step — the place where the query's \
structural shape is articulated as a side effect of weighing \
plausible reads. Atoms inherit this perception when comparing \
their evaluative_intent against the more-likely intent(s).

Two phases drive the rest of the work.

ATOM PHASE — gather evidence freeform. surface_text and \
modifying_signals stay strictly DESCRIPTIVE. evaluative_intent is \
the ONE place where light inference is permitted; its purpose is \
consolidating raw signals into per-criterion meaning. The \
exploration fields describe analyses without committing to verdicts.

COMMIT PHASE — read the evidence and commit. Structural decisions \
(split / keep whole; merge / own trait — including FUSE merges \
where two atoms with bidirectional IDENTITY-SHAPING signals \
collapse into one compound trait) act on the atom phase's \
exploration fields. Per-trait commitments (polarity, commitment, \
relationship-role and its axis bookkeeping) read mechanically off \
the source atom's intent shape and effect tokens, with commitment \
surveying both explicit phrasing signals and structural-prominence \
signals before settling on a level.

Don't re-interpret intent from scratch — the atom phase already \
did the interpretive work.

The sections below cover the atom phase first (atomicity, modifier \
vs atom, evaluative intent), then the commit phase (commit phase, \
polarity, commitment, relationship role), then category vocabulary \
for recognition checks.

---

"""


_ATOMICITY = """\
ATOMICITY — what counts as one atom

PRIMARY SOURCE: intent_exploration's most-likely interpretation. \
The exploration step has already identified what KIND of movie \
satisfies the query and which pieces of the query gate the \
population vs which refine within it. That perception is the \
starting frame for partitioning the query into atoms. The surface \
query, modifying language, and operator surface provide additional \
context that grounds and verifies each atom against the frame — \
they do not stand in for it.

An atom is a unit the system can search for and score \
INDEPENDENTLY. The decision is operational — a population question, \
not a grammar question.

THE POPULATION TEST. For each candidate phrase the query contains, \
ask: stripped of any connective language, would this phrase — on \
its own — name a kind-of-movie a user could ask for as a \
standalone search? Does it identify a population the user has \
experienced or could imagine experiencing as a coherent kind of \
movie? The test is how you VERIFY each candidate atom against the \
primary interpretation; it does not bypass it.

Run this question over the WHOLE query, including content sitting \
inside what looks grammatically like a modifier or operator phrase. \
The connective language ("of", "than", "but with", "set in", "in \
the style of", "starring") is operator surface; the question is \
about the content phrase the operator is positioning, not the \
operator itself.

OUTCOMES.

- Two or more phrases each pass the population test → those phrases \
  are PEER ATOMS, even when one is grammatically positioned as a \
  modifier of the other. The user's answer is the intersection of \
  the populations they each name (filtering, scoring, or both). The \
  cross-relation between them — whatever grammar carries it — \
  records as a modifying_signal on each peer atom, in the user's \
  words. Both peers survive; neither absorbs the other. This \
  outcome holds when the phrases name separable properties the \
  user wants together; when they instead form a description of \
  the plot's shape, the PLOT SHAPES rule governs and the \
  description stays one atom.

- One phrase passes; the other piece is operator-only language with \
  no standalone population (hedges, intensifiers, polarity setters, \
  role markers, range words, structural binders, pure positioning \
  prepositions) → ONE ATOM. The operator absorbs as a \
  modifying_signal on the population-bearing atom; \
  evaluative_intent describes the resulting evaluation.

- One phrase passes, but the cross-relation transposes its \
  evaluation surface into a region the bare population doesn't \
  contain (e.g. moves it to a period, medium, or counterfactual \
  setting that itself isn't a population the user is asking for) → \
  ONE ATOM. The reshape absorbs as a modifying_signal; \
  evaluative_intent describes the consolidated unit. Distinguish \
  this from the peer-atom case by re-running the population test \
  on the reshaping content: if THAT content also names a population \
  the user is asking for, peer atoms is the right call.

The trap to avoid: treating a content phrase as a modifier just \
because it sits in a prepositional / operator-shaped position. \
Position is grammar; population is meaning.

PLOT SHAPES — a described plot is one unit

Some queries describe the shape of a story rather than naming a \
property a movie has — a premise, an arc, a situation that plays \
out: who is involved and what happens to or between them, or how \
things change over the course of the film. When the query is \
doing this, the description of the plot shape is the searchable \
unit. Keep it whole as one atom and carry it through as one trait.

This holds even when the pieces of the description look \
searchable on their own. The characters, relationships, and \
events a plot description mentions will often each pass the \
population test individually — but the user is describing one \
story, not asking for the intersection of the populations those \
pieces name. Splitting the description searches for movies that \
contain the pieces and loses the plot the user described. \
surface_text spans the whole description; evaluative_intent \
restates the plot shape as one thing, not its pieces listed back.

Recognition is by what the content is doing, not by any signpost. \
A query need not announce that it is describing a plot for it to \
be one — a bare description, with nothing framing it as such, is \
still a description. Read whether the content tells you how a \
story goes; if it does, it is a plot shape and stays together.

The guard runs the other way too. Content that names a property \
the movie carries — something true of the film independent of how \
its story goes — is not a plot shape; it splits from the plot \
description as its own atom in the usual way. The question is \
whether the content describes how the story goes or names a \
separable property: describe-the-story stays whole, name-a-\
property splits.

GENERATION DISCIPLINE.

Walk the query in surface order with intent_exploration's \
most-likely interpretation in hand. Emit atoms one at a time, \
each reflecting a piece of the query that contributes to the \
primary intent's structural shape (the population the user wants, \
or a refinement / qualifier on it). For each atom, scan the WHOLE \
query — not just adjacent words — for anything that shapes its \
evaluation, and record those as modifying_signals.

Each phrase gets exactly ONE role: part of an atom's surface_text, \
a modifying_signal on some atom (or peer atoms in a cross-\
relation), or filler (articles, connectives, generic placeholders \
like "movies"). A modifier-only phrase does not also appear as a \
peer atom; a peer-atom phrase does not also appear as absorbed \
material inside another atom's modifying_signals. The atomicity \
decision picks one slot per phrase; don't double-emit.

COMMON PITFALLS.

- Running the test on grammar instead of populations. Whether two \
  pieces grammatically separate is irrelevant. The question is \
  whether each names a kind-of-movie standalone.

- Splitting reflexively on commas, conjunctions, or surface \
  separators. A conjunction supports a split only when both sides \
  genuinely name populations.

- Splitting compound concepts whose pieces don't mean individually \
  what the user meant collectively. Pulling an adjective out of a \
  tonal compound, separating a head noun from a defining qualifier \
  whose only function is to narrow it, splitting a named entity \
  mid-name.

- Breaking a described plot into the characters, relationships, or \
  events it mentions. When the content describes the shape of a \
  story, that description is the unit; pulling out the pieces it \
  names searches for movies that contain the pieces and loses the \
  plot. Recognize the description and keep it whole.

- Absorbing a content phrase as a modifier just because it sits in \
  an operator-shaped position. Re-run the population test on the \
  content, stripped of the operator. If it passes, peer atoms.

- Promoting operator-only language to atoms. Hedges, intensifiers, \
  polarity setters, role markers, range words, comparison \
  operators, and structural binders never have standalone \
  populations; they absorb as modifying_signals.

- Failing to absorb pure operator language. When a piece is \
  positioning-only and has no population of its own, record it as a \
  modifying_signal — don't emit it as an atom of its own.

SPLIT AND STANDALONE EXPLORATIONS. Every atom carries two analyses \
before it ships. Always populated — pure evidence gathering, no \
verdicts. The commit phase reads them and makes the structural \
decisions.

split_exploration walks two checks:
- FORWARD: could this atom's intent be subdivided into smaller \
  pieces, each retrievable independently? Walk evaluative_intent, \
  not just surface_text. When the atom describes the shape of a \
  plot, say so here: its pieces are the characters, relationships, \
  and events the story mentions, and subdividing would search for \
  those pieces instead of the plot itself. Leave it as evidence; \
  the commit phase decides.
- INVERSE: for each modifying_signal recorded on this atom, does \
  its content phrase (stripped of connective language) pass the \
  population test? If yes, that signal is carrying a peer-atom \
  candidate that should be split out; describe what each split \
  would retrieve. If no, describe why absorption is the only \
  sensible read. Skipping this check is how population-bearing \
  phrases get silently absorbed.

standalone_check compares evaluative_intent against the intents \
surfaced in intent_exploration. Describe HOW (not if) retrieving \
this atom standalone would relate to the user's articulated \
intent — what population standalone retrieval would return; \
whether that population fits naturally under the more-likely \
intent(s) or implicitly commits the search to a less-likely one \
(introducing a hard requirement the user didn't ask for, losing a \
coupling the user did imply, narrowing what the user kept loose); \
when the atom integrates context from another atom, whether that \
context survives or falls away.

Neither field writes a verdict. Their job is to leave evidence — \
not short-circuit with "first mention" / "primary subject" / "no \
other atom captures this" / "distinct concept" dismissals.

ORDERING. Atoms appear in the order their surface anchor appears \
in the query. Order is load-bearing downstream.

---

"""


_MODIFIER_VS_ATOM = """\
MODIFIER VS ATOM

This section covers IN-ATOM modifier-only language — words and \
short phrases that have no standalone population on their own and \
shape an adjacent atom's evaluation. Cross-criterion content-phrase \
relationships are governed by the population test in ATOMICITY \
(re-run there when the question is whether a content phrase has a \
population of its own).

Operator-only language never has a standalone population. It comes \
in stable functional shapes: polarity setters (negate or invert), \
hedges and intensifiers (soften or harden), role and binding \
markers (tell the handler which credit / source / format the \
adjacent atom binds to), range / approximation / ordinal words \
(calibrate a numeric scope), and pure comparison / scope operators \
("like", "similar to", "in the vein of"). Recognize the FUNCTION; \
specific surface tokens vary by query.

Compound atoms are different. When a token reshapes the atom's \
meaning into something new — when removing the token leaves the \
atom pointing at a different concept — the whole stays one atom. \
That's not modifier-on-atom; that's a multi-word atom that happens \
to look like adjective-plus-noun.

OPERATIONAL TEST. Ask of the candidate token, in plain words: does \
removing it leave the rest pointing at the same thing the user \
meant, just less precisely qualified? If yes → modifier. If no → \
the token is part of the atom's identity; whole is one atom.

Pitfalls:
- Promoting operator-only language to atoms. Hedges, polarity \
  setters, role markers, range words, comparison operators all \
  absorb as modifying_signals.
- Treating a meaning-shaping word as a modifier when it's actually \
  fused into the atom (the test above settles this).
- Treating every adjective-before-noun as modifier-on-atom by \
  default. If the adjective itself has standalone population (it \
  names a kind of person, kind of subject, etc.), the population \
  test in ATOMICITY governs — those split into peer atoms.

CROSS-RELATION CONTENT PHRASES. When the query contains content \
phrases that POSITION other atoms (transposing setting, comparing, \
qualifying, scoping), the population test in ATOMICITY decides \
whether each content phrase is a peer atom or absorbed material. \
Don't try to make that call here — this section is about \
operator-only language, not content phrases. Both kinds of \
absorbed material record on modifying_signals in the same shape \
(verbatim surface_phrase + concise effect); the difference between \
them is whether a peer atom was also emitted.

---

"""


_EVALUATIVE_INTENT = """\
EVALUATIVE INTENT

For each atom: (1) catalog every signal in the query that shapes \
how this criterion should be evaluated, (2) state — concisely, in \
plain prose — what evaluating this criterion actually means once \
those signals are integrated.

Mental model. Per-criterion, not directed graph. For this atom, \
what in the query reshapes how I'd score movies on it? Reshapes \
come from adjacent qualifying language, from polarity elsewhere \
distributing onto this criterion, or from cross-criterion content \
that scopes / transposes / narrows / qualifies this one. Position \
is irrelevant.

Recording signals. One entry on modifying_signals per signal:
- surface_phrase: verbatim user text. Adjacent operator → just \
  the phrase. Cross-criterion → the connecting language plus the \
  reference, in the user's words. Never a positional pointer or \
  atom index.
- effect: a few words describing what THIS signal does to THIS \
  atom's evaluation. Modal-language signals use the controlled \
  tokens SOFTENS / HARDENS / FLIPS POLARITY / CONTRASTS so the \
  commit phase can parse polarity and commitment. Otherwise \
  freeform; describe the specific effect, not a bucket.

Building evaluative_intent. 1-2 sentences. The field's purpose is \
a FAITHFUL restatement of what the criterion asks for, with each \
modifying_signal mechanically integrated by its effect. Inference \
is bounded to integrating signal effects — it is NOT a license to \
elaborate beyond surface_text and the signals. If the query is \
concrete and modifying_signals is empty, evaluative_intent is \
near-paraphrase of surface_text; that is the correct shape.

FIDELITY DISCIPLINE — light inference is for CONSOLIDATING the \
signals the query carries, not for elaborating beyond them. Every \
clause of evaluative_intent must trace back to either surface_text \
or a modifying_signal. evaluative_intent is the source the rest of \
the pipeline reads as the per-criterion contract; drift here \
propagates silently through every downstream step. Three drift \
modes corrupt this layer:

- IMPLICIT BROADENING. surface_text names a narrow constraint and \
  evaluative_intent widens it into the surrounding category that \
  contains it, often by adding an "or"-clause that admits \
  neighboring criteria the user did not articulate. The widened \
  intent is easier to retrieve against — more candidate films will \
  match — which is precisely why the model reaches for it.
- IMPLICIT NARROWING. surface_text names a category and \
  evaluative_intent pins it to a single canonical exemplar of that \
  category. The narrower intent retrieves more cleanly because the \
  exemplar has a clear database fingerprint, which is why the \
  model reaches for it.
- INVENTED DETAIL. evaluative_intent introduces specificity the \
  query never named — particular instances, particular sub-types, \
  particular bounds — drawn from the model's prior knowledge of \
  what "typical" instances of this criterion look like. Prior \
  knowledge belongs to retrieval, not to intent commitment.

FIDELITY TEST. Read evaluative_intent back against surface_text \
and the modifying_signals list. For each clause in \
evaluative_intent, point to the word in surface_text or the signal \
that licensed it. A clause with no anchor in the user's actual \
language is drift — remove it. The intent stays strictly inside \
what the user said; downstream steps depend on this layer being \
faithful.

OPERATIONAL TEST: read your modifying_signals list. For each \
entry, ask — does my intent statement reflect this signal's \
effect? Would changing the signal noticeably change the intent? If \
no for any signal, the intent has not consolidated it — revise.

Generalized guidance: integrate each signal's effect into the \
intent description. A softener appears as a softened evaluative \
direction; a negation appears as an avoid-direction; an absorbed \
reshape appears as the consolidated unit (not the bare anchor); a \
trait that is operating as a reference rather than a retrieval \
target says so, with the kind of scoring-against-reference made \
explicit. Don't slot into a fixed bucket — describe what the \
signals do for THIS atom.

Hard guardrails (these still apply with inference allowed):
- No category labels ("genre", "runtime", "actor", "tone").
- No concrete polarity / commitment values — describe in words; \
  commitments live on traits.
- No expansion of named things.
- No translation into system vocabulary.
- No broadening, narrowing, or invented detail against \
  surface_text. evaluative_intent stays inside the user's actual \
  words; light inference consolidates signals, it does not \
  elaborate beyond them.
- A near-paraphrase of surface_text is the CORRECT shape when the \
  criterion is concrete and modifying_signals is empty. Read this \
  guardrail in that direction — do NOT treat a near-paraphrase as a \
  smell that needs elaboration. The only smell is a rephrase that \
  IGNORES signals that DO exist; with no signals, the rephrase is \
  the answer.

Common pitfalls:
- Collapsing two population-bearing criteria into one atom because \
  they grammatically modify each other. They stay peer atoms; the \
  cross-relation goes on each peer's signals list.
- Promoting operator-only language to atoms.
- Treating a peer atom that operates as a comparison reference as \
  if it had to be scored against directly. Its intent reflects the \
  reference role; Step 3 reads that and routes accordingly.
- Empty modifying_signals on every atom of a parallel-filter query \
  is the COMMON case. Don't fabricate signals to make atoms look \
  connected.

PATTERNS — what faithful evaluative_intent looks like, as abstract \
shapes (not domain-specific examples).

- CONCRETE NARROW QUERY, NO SIGNALS. surface_text is a single \
  concrete criterion and modifying_signals is empty. \
  evaluative_intent restates surface_text in plain prose at the \
  same width — same constraint, same granularity, same scope. The \
  intent is essentially the surface phrase made readable. The \
  drift form is adding an "or"-clause that admits neighboring \
  criteria the user did not state, or appending canonical \
  exemplars / sub-types drawn from prior knowledge of what such \
  queries "typically" contain. Both are unanchored content; both \
  are drift.

- CONCRETE QUERY, SINGLE HARDENS/SOFTENS MODIFIER. surface_text \
  names a concrete criterion and one modifying_signal carries an \
  intensifying or softening effect. evaluative_intent restates \
  surface_text and folds in the strength language the effect \
  produces — preference-strength for SOFTENS, requirement-strength \
  for HARDENS, no further elaboration. The drift form is reading \
  the modifier as a license to enumerate canonical instances of \
  the surrounding category, or to list "or"-clauses that broaden \
  what the criterion covers. The modifier shapes STRENGTH on the \
  same criterion; it does not unlock new content.

- CROSS-CRITERION REFERENCE (POSITIONING). One atom refers to \
  another via cross-criterion language; modifying_signals carries \
  the reference and its effect. evaluative_intent restates the \
  criterion and describes the positioning operationally (the \
  named peer is being treated as a reference rather than a \
  retrieval target, with the comparison direction the signal \
  spelled out). No new content beyond what the surface and the \
  signal carry. The drift form is elaborating what the referenced \
  peer "really means" — that elaboration belongs to the peer's \
  own atom, not this one.

The throughline across all three patterns: faithful \
evaluative_intent never introduces content whose only source is \
the model's prior knowledge of the criterion. If a clause cannot \
be pointed back at a word in surface_text or an entry in \
modifying_signals, it is drift and is removed.

---

"""


_COMMIT_PHASE = """\
COMMIT PHASE — atoms → traits

Walk the atom list and produce a parallel trait list. Atoms \
gathered evidence (modifying_signals + evaluative_intent + \
exploration fields); traits commit values from that evidence. \
Reuse the work; don't re-interpret from scratch.

The Trait schema field descriptions specify what each field must \
contain. This section covers procedural decisions — when to split, \
when to merge, what tests each commitment must pass.

ACT ON SPLIT EXPLORATIONS. Read each atom's split_exploration. \
Apply the searchable-unit test yourself: if the exploration lays \
out pieces whose independent retrievals would combine to the \
user's intent at this atom's granularity, emit each piece as its \
own trait. Otherwise keep whole. The exploration is evidence; the \
decision is yours.

ACT ON STANDALONE CHECKS. Read each atom's standalone_check. If \
it describes a meaning shift the user did not articulate (a hard \
constraint they didn't ask for, a coupling lost when the atom is \
read alone, a narrowing the user kept loose), find the atom whose \
evaluative_intent already integrates this content and merge: the \
surviving trait is the integrating atom; the coupled atom does NOT \
survive separately, because emitting it separately would re-\
introduce the meaning shift the check identified. If standalone \
retrieval matches a user-articulated standalone-able criterion \
(including exclusions tied to an earlier atom's pool with negative \
polarity), keep as own trait.

ACT ON FUSED-COMPOUND CHECKS. Walk pairs of atoms with cross-\
modifying signals on each other. For each pair, ask: does each \
atom's modifying_signals carry an IDENTITY-SHAPING effect token \
referencing the other atom's content? Bidirectional IDENTITY-\
SHAPING is the signature for a fused compound — the user means \
piece A specifically in the context of piece B AND vice versa, \
such that A standalone ≠ A-in-the-compound and B standalone ≠ \
B-in-the-compound. When this fires, FUSE the pair into ONE trait: \
the population test's two-atom outcome is wrong for this query. \
Surface_text spans both source phrases as they appear in the query; \
evaluative_intent integrates both source intents into a single \
compound description.

Single-direction shaping is NOT a fuse trigger. When a qualifier \
shapes how the population's instance is scored but the population \
doesn't reshape what the qualifier means, the atoms stay separate \
— the qualifier-on-population case keeps two traits with \
INDEPENDENT relationship roles. The fuse rule fires only when \
removing either piece collapses both pieces' meaning into something \
the user wasn't asking for.

ACT ON PLOT-SHAPE SPLITS. Beyond the identity-shaping fuse, watch \
for sibling atoms the atom phase emitted separately that together \
describe one plot shape — the characters, relationships, and \
events a single story binds. When they do, merge them into ONE \
trait spanning their phrases, with a single evaluative_intent \
restating the plot as one thing. This is not gated on \
bidirectional signals: a plot whose pieces relate in one \
direction — one role acting on another, or a single subject \
changing over the story — is still one description. This does not \
disturb the qualifier-on-population case, where one population is \
scored on an added property rather than described as a plot — \
there nothing is merged.

The fuse decision happens BEFORE polarity / commitment / \
relationship-role commits, so the merged trait commits its \
own polarity, commitment, and role from scratch — it inherits \
neither source atom's slot.

When merging (either standalone-check-driven or fuse-driven), the \
merged trait absorbs both sources fully. Neither survives \
separately. For standalone-check merges: the host's surface_text \
and evaluative_intent stand; the coupled atom's content is \
integrated via the host's modifying_signals already. For fuse \
merges: both sources contribute equally to the merged \
surface_text and evaluative_intent.

DON'T DROP, DON'T INVENT. Every atom that survives the standalone \
check AND is not absorbed by a fuse merge produces at least one \
trait. Two atoms absorbed by a fuse merge produce ONE trait \
together — neither survives separately, the merged trait is the \
shared output. No trait without a source atom. Splits add traits; \
merges combine. Atom count is NOT constrained to equal trait \
count — splits push count up, merges push count down. Do NOT \
default to one-trait-per-atom; the count emerges from the splits \
and merges you commit, not from the atom list's length.

PER-TRAIT COMMITMENTS. qualifier_relation, relationship_role, \
replaces_axis, axes_replaced_by_siblings, anchor_reference, \
polarity, commitment_evidence, commitment, and \
contextualized_phrase commit per the schema field descriptions. \
The commitment order is: polarity (mechanical from FLIPS POLARITY \
signals) → commitment_evidence (surveys both signal channels) → \
commitment (natural conclusion of evidence) → qualifier_relation \
(freeform prose) → relationship_role (closed-enum hard commit) → \
replaces_axis / axes_replaced_by_siblings (axis bookkeeping for \
positioning roles) → anchor_reference (verbatim surface) → \
contextualized_phrase (user-voice restatement).

qualifier_relation is freeform prose describing how this trait \
positions against the rest of the query AND the operational \
meaning of that positioning (what kinds of dimensions Step 3 \
should produce). Read it off the source atom's modifying_signals \
in your own words — describe the specific relation this query \
contains, not a slot from a closed list. When no qualifier-style \
signal exists in the source atom's modifying_signals, commit the \
literal string "n/a".

relationship_role hard-commits the prose's structural shape into \
one of three closed values (INDEPENDENT / POSITIONING_REFERENCE / \
POSITIONING_QUALIFIER). The role is a structural classification, \
not a polarity / weight choice. See the RELATIONSHIP ROLE section \
below for the discriminator.

replaces_axis and axes_replaced_by_siblings carry the axis-level \
bookkeeping for positioning roles. See the RELATIONSHIP ROLE \
section.

anchor_reference is the modifier surface phrase carried verbatim \
from modifying_signals.surface_phrase. If no modifier acts on the \
trait, commit the literal string "n/a".

contextualized_phrase folds anchor_reference and meaning-shaping \
modifying_signals into one short phrase that restates the trait in \
user voice with its query context preserved. When no meaning-\
shaping modifier acts on the trait, copy surface_text verbatim.

OPERATIONAL TESTS (apply at the point of writing each value):
- After polarity: "is there a FLIPS POLARITY (or negation-flavored) \
  signal on the source atom?" Yes → negative; no → positive.
- After commitment_evidence: "did I read the explicit channel \
  first and note whether it fires, then read the structural \
  channel only as the fallback? Did I name which channel is doing \
  the work for this trait, without picking a level at the end?" \
  If the evidence consulted both channels symmetrically, or wrote \
  a verdict, revise.
- After commitment: "is this the natural conclusion of \
  commitment_evidence rather than a default-fill? If the explicit \
  channel fired, did the level commit at REQUIRED or DIMINISHED \
  regardless of where structural prominence sat? If the explicit \
  channel was silent, did the level fall to ELEVATED, NEUTRAL, or \
  SUPPORTING based on structural prominence alone?" REQUIRED and \
  DIMINISHED never commit without an explicit signal.
- After qualifier_relation / anchor_reference: "is the relation I \
  named, and the anchor I carried, present in modifying_signals?" \
  If not, revise to match or commit "n/a". NEVER fabricate to fill \
  the slot.
- After relationship_role: "if I asked a fresh reader to decompose \
  this trait standalone, would they need to know about a sibling \
  trait to do it correctly?" If no → INDEPENDENT. If yes and this \
  trait is the anchor → POSITIONING_REFERENCE. If yes and this \
  trait is the modifier → POSITIONING_QUALIFIER. See the \
  RELATIONSHIP ROLE section for the full discriminator and the \
  axis-bookkeeping rules.
- After contextualized_phrase: "if I read this phrase aloud out \
  of query context, can a fresh reader recover what the trait is \
  asking for?" If not, fold the modifier in more clearly. (When no \
  meaning-shaping modifier acts on the trait, this is automatically \
  yes — the field copies surface_text.)

TRAIT ORDERING. Source-atom order. Splits inherit position (each \
piece keeps the slot, in piece order). Merges take the earlier \
source's slot.

---

"""


_POLARITY = """\
POLARITY (commits Trait.polarity)

Read off the source atom's modifying_signals — effect tokens \
already mark polarity.

- Positive: user wants it.
- Negative: user wants to avoid / penalize it.

Mechanical rule. Any source signal's effect contains FLIPS \
POLARITY or recognizable negation language → negative. Otherwise \
positive. Don't re-interpret intent — the atom phase already \
recorded the polarity-setter.

Polarity-setter shapes (recognize): "not", "without", "no", \
"avoid", "skip", "minus", "anything but", "spare me", "don't \
want".

"Not too X" special case: polarity = negative, commitment = \
diminished. Softened preference against, not assertion against.

Boundaries:
- Polarity is mechanical from the recorded signal. Don't rewrite \
  "movies that aren't boring" into positive intent for "engaging".
- Hedges and intensifiers don't change polarity — they affect \
  commitment.

---

"""


_COMMITMENT = """\
COMMITMENT (commits Trait.commitment via Trait.commitment_evidence)

Five levels on a single importance axis. The axis applies to every \
trait regardless of polarity; reward or penalty direction is set \
by polarity, weight magnitude is set by commitment.

LEVELS.

- REQUIRED — explicit channel fires with strong-assertion language. \
  Largest weight in final score and reranking.
- ELEVATED — explicit channel silent; structural prominence reads \
  the trait as the load-bearing axis the search is fundamentally \
  about. Removing it would change WHAT KIND of movie the query is \
  asking for, not how that movie is qualified.
- NEUTRAL — explicit channel silent; structural prominence reads \
  balanced. Co-equal criterion among peers.
- SUPPORTING — explicit channel silent; structural prominence \
  reads as a refinement on a population other traits define.
- DIMINISHED — explicit channel fires with soft-framing language. \
  Lowest weight.

TWO SIGNAL CHANNELS. commitment_evidence surveys both, but they \
are not equal. The explicit channel dominates: when it fires, the \
trait commits at the level the explicit signal names regardless of \
where structural prominence sits. The structural channel sets the \
level only when the explicit channel is silent. This precedence is \
load-bearing — it is the difference between expressed and inferred \
strength.

(1) EXPLICIT signals. Walk the source atom's modifying_signals for \
language whose function is to fix the trait's strength.

Strong-assertion language takes several recognizable shapes — \
phrasing that names an inviolable constraint or non-negotiable; \
phrasing that frames the trait as a precondition for the candidate \
being a viable watch at all (an access, language, format, or \
viewer-fit gate) rather than a preference within the space of \
viable watches; phrasing that asserts an exclusion (the trait \
names something out of scope) rather than expressing a preference \
against (something to be downranked). The common thread is that \
the trait is positioned as defining the boundary of viable \
candidates, not as scoring within that boundary.

Soft-framing language takes one shape — phrasing whose function is \
to soften the trait's claim on the result and invite the system to \
set it aside in exchange for matches on other axes. Bonus-framing, \
idealization, conditionality, hedging.

Recognize the FUNCTION; specific surface tokens vary by query. \
Polarity (committed above) helps tell exclusion-assertion from \
preference-against — both attach to negative polarity, but the \
former asserts X is out of scope while the latter ranks X-bearing \
candidates lower.

(2) STRUCTURAL signals. Consulted only when the explicit channel \
is silent. Walk surface position (headline / leading vs. trailing), \
content load (bulk of the query's words vs. modest), positioning \
per qualifier_relation and anchor_reference (does the trait name \
the population the query is asking for, refine a population a peer \
defines, or sit coordinate with peers), and the removability test \
against intent_exploration's most-likely interpretation (would \
removing the trait collapse the structural ask, narrow it without \
collapse, or leave a refinement falling away). Headline / load-\
bearing → ELEVATED. Trailing refinement → SUPPORTING. Equal \
billing among peers → NEUTRAL.

RECOGNIZING REQUIRED AND DIMINISHED. Both extremes are reserved \
for traits where the user has expressed something explicit about \
the trait's strength. The explicit signal does not have to take \
the canonical "must" or "ideally" form — the recognition is by \
function, not surface token. Phrasings that declare a non-\
negotiable, name a watching precondition, or assert an exclusion \
all count as strong-assertion language even without "must." \
Phrasings that soften, idealize, or frame as a bonus all count as \
soft-framing language even without "ideally." Without an explicit \
signal of either kind, the trait cannot commit to either extreme — \
strong implicit prominence commits ELEVATED, structural triviality \
commits SUPPORTING.

BOUNDARIES.

- Commitment is not polarity. Negation flips polarity; it does \
  not lower the commitment level. An asserted exclusion at full \
  strength commits REQUIRED + negative; a softened exclusion \
  commits DIMINISHED + negative.
- Commitment is not atomicity. Atomicity decides whether a phrase \
  is a peer atom or absorbs into another atom's modifying_signals. \
  Once a trait exists, commitment weights it. A modifier-only \
  phrase that absorbs into a peer never gets its own commitment \
  because it never becomes a trait.
- Generic intensification is not strong assertion. A HARDENS \
  effect token that emphasizes without asserting inviolability, \
  declaring a precondition, or asserting an exclusion does not \
  commit REQUIRED. The explicit signal must function as one of \
  those three shapes.
- Structural prominence does not override expressed framing. A \
  trait the user explicitly hedged commits DIMINISHED even when \
  it sits at the head of the query; a trait the user explicitly \
  asserted as inviolable commits REQUIRED even when it sits in \
  trailing position.

Unifying principle: commitment tracks how strongly the user has \
attached themselves to the trait. The explicit channel reads what \
they said out loud about the trait's strength; the structural \
channel reads how much they invested in it (words spent, position \
chosen, what the query loses if the trait is removed). When the \
user has spoken, that is the answer. When the user has not, the \
investment is.

---

"""


_RELATIONSHIP_ROLE = """\
RELATIONSHIP ROLE (commits Trait.relationship_role,
Trait.replaces_axis, Trait.axes_replaced_by_siblings)

After qualifier_relation prose is written, hard-commit the \
structural shape into the closed enum. Three operational shapes:

INDEPENDENT — the trait stands on its own for retrieval and \
scoring. Sibling traits exist (or don't), but this trait does not \
need information from any sibling for Step 3 to decompose it \
correctly. Two cases land here:
- Parallel filters: each trait names its own evaluable population, \
  and the user wants the intersection.
- Qualifier-on-population: one trait modifies how another's \
  population is scored, but each is independently scorable. The \
  qualifier doesn't replace any axis of the population; it adds an \
  additional axis the user wants to score on. Single-direction \
  shaping with no axis-substitution is the signature.

POSITIONING_REFERENCE — the trait names an anchor a sibling is \
comparing, transposing, or scoping against. The trait's identity \
is being used as a TEMPLATE for matching other films; specific \
axes of that template may be replaced by sibling qualifiers. The \
user is not asking for the reference itself — they are asking for \
things that match the reference along the kept axes. This role is \
RECIPROCAL: it commits only when at least one sibling commits \
POSITIONING_QUALIFIER targeting this trait.

POSITIONING_QUALIFIER — the trait names a substitute for some \
axis on a sibling reference. The qualifier is independently \
scorable, but its meaning in this query is SUBSTITUTION on the \
reference. This role is RECIPROCAL: it commits only when at \
least one sibling commits POSITIONING_REFERENCE for it to point \
at.

DISCRIMINATOR. Read by what the trait DOES in the query, not by \
surface tokens. The same connective ("with", "but", "-style", \
"like") joins independent or positioning relations depending on \
the content phrases it joins. The connective is evidence; the role \
is what the trait is doing.

The operational test: "if I asked a fresh reader to decompose this \
trait standalone, would they need to know about a sibling trait to \
do it correctly?" If no → INDEPENDENT. If yes and this trait is \
the anchor → POSITIONING_REFERENCE. If yes and this trait is the \
modifier → POSITIONING_QUALIFIER.

REPLACES_AXIS (POSITIONING_QUALIFIER only). A short user-vocabulary \
noun-phrase naming the AXIS on the sibling reference being \
substituted. The axis is the DIMENSION of evaluation, not the \
VALUE this trait provides on that dimension. Axis = "setting"; \
value = "jungle setting". Axis = "tone"; value = "comedic". \
Multi-axis substitution → slash-joined phrase ("genre/setting") \
rather than emitting two qualifier traits.

AXES_REPLACED_BY_SIBLINGS (POSITIONING_REFERENCE only). VERBATIM \
copy of every sibling POSITIONING_QUALIFIER's replaces_axis. The \
reference does not invent or paraphrase replacements — it inherits \
them from siblings that committed them. Step 2 is the only stage \
that sees the whole query, so this list is where the cross-trait \
reasoning lands; Step 3 reads it mechanically when decomposing.

OPERATIONAL TESTS:
- After relationship_role: run the standalone-decomposition test \
  above. Does the role I committed match the answer?
- After replaces_axis: read it back. "Does this name a DIMENSION \
  of evaluation, or a VALUE on that dimension?" Dimension → \
  correct. Value → revise. (And: is this committed only when \
  relationship_role is POSITIONING_QUALIFIER? If not, revise — \
  None for the other roles.)
- After axes_replaced_by_siblings: walk each entry. "Does a \
  sibling trait commit replaces_axis equal to this exact phrase?" \
  If no, the entry was invented — drop it. (And: is this populated \
  only when relationship_role is POSITIONING_REFERENCE? If not, \
  revise — empty list for the other roles.)

BOUNDARIES.

- Role classification is reciprocal. POSITIONING_REFERENCE without \
  any sibling POSITIONING_QUALIFIER is wrong (no one is positioning \
  against it → it's just INDEPENDENT). POSITIONING_QUALIFIER \
  without any sibling POSITIONING_REFERENCE is wrong (nothing for \
  it to substitute on → it's just INDEPENDENT or a separate \
  filter).
- Single-direction shaping doesn't trigger positioning. When a \
  qualifier shapes how the population's instance is scored but the \
  population doesn't reshape what the qualifier means AND no axis \
  is being substituted, both atoms commit INDEPENDENT.
- Replaces_axis is at most ONE axis (or one slash-joined compound \
  axis) per qualifier. If a single qualifier substitutes on \
  multiple distinct axes that the user would consider separate, \
  consider whether atomization should have produced two qualifier \
  atoms instead.

Common pitfalls:
- Committing by connective surface. "with X" can be either an \
  independent qualifier-on-population or a positioning qualifier — \
  depends on whether X replaces an axis of a sibling reference.
- Naming the substitute instead of the axis in replaces_axis. \
  "jungle" / "comedic" / "sci-fi" are values; "setting" / "tone" / \
  "genre" are axes.
- Inventing entries in axes_replaced_by_siblings that no sibling \
  committed. The reference inherits; it doesn't generate.

---

"""


def _build_category_vocabulary_section() -> str:
    """Render category vocabulary for Step 2 atom-phase recognition.

    Trimmed view: name + description + good_examples only. The
    fitting machinery (boundary / edge_cases / bad_examples) lives
    at Step 3 where category routing actually happens; loading it
    here would force the atom phase into category-fitting mode
    when its job is only to recognize that a phrase has a home.
    """
    header = (
        "CATEGORY VOCABULARY\n\n"
        "What kinds of homes exist for query content. The atom "
        "phase uses this to recognize when a phrase has a category "
        "home (informs atomization and modifier-vs-atom). You do "
        "NOT pick categories at this stage — that's Step 3's job, "
        "with the full fitting machinery (boundaries, edge cases, "
        "misroute traps).\n\n"
    )
    blocks: list[str] = []
    for cat in CategoryName:
        lines = [f"{cat.name} — {cat.description}"]
        if cat.good_examples:
            lines.append(
                "  Examples: "
                + ", ".join(f'"{e}"' for e in cat.good_examples)
            )
        blocks.append("\n".join(lines))
    return header + "\n\n".join(blocks)


_CATEGORY_VOCABULARY = _build_category_vocabulary_section()


SYSTEM_PROMPT = (
    _TASK_FRAMING
    + _ATOMICITY
    + _MODIFIER_VS_ATOM
    + _EVALUATIVE_INTENT
    + VAGUE_TEMPORAL_VOCABULARY_COMPACT
    + _COMMIT_PHASE
    + _POLARITY
    + _COMMITMENT
    + _RELATIONSHIP_ROLE
    + _CATEGORY_VOCABULARY
)


# ===============================================================
#                      Executor
# ===============================================================
#
# Model is finalized to Gemini 3.5 Flash with minimal thinking
# (`thinking_level="minimal"` — the lowest non-disabled level, below
# "low") and a modest temperature. Callers cannot override — this
# keeps the step reproducible and makes cost/latency predictable
# end-to-end.


_PROVIDER = LLMProvider.GEMINI
_MODEL = "gemini-3.5-flash"
_MODEL_KWARGS: dict = {
    "thinking_config": {"thinking_level": "minimal"},
    "temperature": 0.35,
}


async def run_step_2(query: str) -> tuple[QueryAnalysis, int, int, float]:
    """Run the query-analysis LLM on a single query.

    Args:
        query: the raw user query.

    Returns:
        (response, input_tokens, output_tokens, elapsed_seconds) —
        elapsed measures wall-clock time spent inside the LLM call
        only, not prompt setup.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must be a non-empty string.")

    user_prompt = f"Query: {query}"

    # perf_counter: monotonic, high-res wall-clock for short intervals.
    start = time.perf_counter()
    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=_PROVIDER,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=QueryAnalysis,
        model=_MODEL,
        **_MODEL_KWARGS,
    )
    elapsed = time.perf_counter() - start

    return response, input_tokens, output_tokens, elapsed


# ===============================================================
#                      CLI
# ===============================================================


def _print_response(response: QueryAnalysis) -> None:
    """Pretty-print the structured response for terminal inspection."""
    payload = response.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the step-2 pre-pass on a single query and print "
            "the structured output."
        )
    )
    parser.add_argument(
        "query",
        type=str,
        help="The raw user query to process.",
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = _parse_args()
    response, in_tok, out_tok, elapsed = await run_step_2(args.query)
    _print_response(response)
    print(
        f"\n[tokens] input={in_tok} output={out_tok} "
        f"elapsed={elapsed:.2f}s"
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
