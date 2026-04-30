# Search V2 — Trait Decomposition stage (Step 3).
#
# Second stage of the multi-step query-understanding flow. Takes a
# committed Trait from Step 2 and emits a TraitDecomposition with
# three coupled layers:
#
#   1. Trait-role analysis — target_population + trait_role_analysis.
#      Reads role + qualifier_relation + anchor_reference (committed
#      by Step 2) and commits what those mean for the dimensions.
#   2. Dimension inventory + per-dimension category candidates.
#      Each dimension is the smallest searchable piece in database-
#      vocabulary; each carries a freeform list of plausible
#      categories with what's-covered / what's-missing prose.
#   3. Commitment layer — category_calls. The minimum number of
#      taxonomy-routed calls; one call per category, owning one or
#      more expressions when the same category cleanly covers
#      several dimensions of the trait.
#
# The prompt walks five sections plus the full category taxonomy:
#
#   ANALYSIS PHASE
#     trait-role analysis → dimension inventory → per-dimension
#     candidates
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
#   ANALYSIS PHASE  — trait-role analysis → dimension inventory →
#                     per-dimension candidates
#   COMMITMENT PHASE — category routing → minimum-set & polarity
#   CATEGORY TAXONOMY — full disambiguation machinery


_TASK_FRAMING = """\
You are the trait-decomposition stage of a movie-search pipeline. \
The previous stage committed a list of traits (search-ready units \
with role, polarity, salience, qualifier_relation, and \
anchor_reference assigned). Take ONE trait and turn it into the \
minimum number of taxonomy-routed CATEGORY CALLS the search \
backend can execute against, each owning one or more concrete \
searchable expressions.

This is the abstraction-flip in the pipeline. Earlier stages \
worked in user-vocabulary; the next stage works in endpoint-\
vocabulary. Your stage sits on the seam: translate the trait's \
semantic intent into database-shaped retrieval intents.

You produce a TraitDecomposition with these coupled layers:

1. TRAIT-ROLE ANALYSIS — target_population + trait_role_analysis. \
   Read role / qualifier_relation / anchor_reference mechanically \
   off the user prompt and commit what they mean for what the \
   dimensions describe. Pre-dimension; constrains the inventory.
2. ASPECTS — flat list enumerating every distinguishable axis the \
   trait calls for, in user-vocabulary. Sits between the prose \
   role analysis and the database-vocabulary dimensions; \
   separating enumeration from translation prevents axes from \
   getting lost in the mode-shift.
3. DIMENSION INVENTORY + CANDIDATES — concrete searchable pieces \
   in database-vocabulary, each translating one or more aspects, \
   each with plausible category candidates carrying \
   what's-covered / what's-missing prose.
4. COMMITMENT LAYER — category_calls. One call per category that \
   ends up owning >=1 expression. Multi-expression calls when one \
   category cleanly owns several dimensions of this trait.

Read the response schema's field descriptions before producing \
output.

Two phases drive the work.

ANALYSIS PHASE — restate the population, commit the role analysis, \
enumerate aspects, translate aspects into dimensions, list \
per-dimension candidates with explicit coverage prose. No calls \
yet.

COMMITMENT PHASE — read the candidates and emit minimum calls. \
Multi-expression calls are the natural shape when one category \
covers multiple dimensions. Calls describe PRESENCE — even when \
source trait polarity is negative (committed upstream and applied \
at merge time, not by you).

The sections below cover analysis (trait-role analysis, aspect \
enumeration, dimension inventory, per-dimension candidates), then \
commitment (category routing, minimum-set + polarity discipline), \
then the full category taxonomy.

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

(1) PRIMARY — qualifier_relation. When populated, this is the \
field Step 2 wrote specifically to constrain your dimension \
scope. It describes, in freeform user vocabulary, how this \
trait positions against the rest of the query AND the \
operational meaning of that role. Translate its prose into what \
KIND of dimensions belong: a measurable axis with a directional \
threshold; an archetype / iconography / tonal register the \
candidates need to satisfy; a setting / period / medium the \
candidates need to evaluate inside; a craft / aesthetic \
template the candidates need to match; etc. The relation is \
freeform; your translation is freeform too — describe what the \
dimensions need to capture for THIS query, not a slot from a \
fixed list. When qualifier_relation is "n/a", primary signal \
shifts to the grounding sources (4) below.

(2) VERDICT — role. The binary carver-vs-qualifier commitment.
- carver: dimensions describe the POPULATION the user wants \
  retrieved.
- qualifier: dimensions describe the REFERENCE / ANCHOR / \
  SHAPE the trait positions against, never the population to \
  recommend.

(3) RATIONALE — role_evidence. One sentence from Step 2 \
explaining WHY the role was committed: definitive eligibility \
gate (carver), or one of the qualifier shapes (continuous-score-\
only, used as a comparison reference rather than naming the \
population, or another trait already gates the population this \
one only refines). Read this even when role looks obvious — for \
borderline traits and for carvers where qualifier_relation is \
"n/a" it carries the load that primary doesn't.

(4) GROUNDING — contextualized_phrase + evaluative_intent. The \
modifier-folded headline identity and the integrated meaning \
with all modifying signals folded in. Anchor your analysis in \
what the trait is actually asking for from this specific query, \
not generic prose about "qualifiers" or "populations". Especially \
load-bearing when qualifier_relation is "n/a".

(5) SURFACE POINTER — anchor_reference. The verbatim modifier \
phrase from the original query. Use it to keep the analysis \
specific to this query's anchor rather than abstract.

IDENTITY VS ATTRIBUTE CATEGORIES — a structural rule that follows \
from "qualifier means positioning, not retrieval". Some categories \
retrieve the named entity itself: PERSON_CREDIT, TITLE_TEXT_LOOKUP, \
NAMED_CHARACTER, STUDIO/BRAND, FRANCHISE/UNIVERSE_LINEAGE, \
CHARACTER_FRANCHISE, ADAPTATION_SOURCE_FLAG, BELOW_THE_LINE_CREATOR, \
NAMED_SOURCE_CREATOR. Others describe attributes a film can have \
(genre, emotional/experiential, narrative setting, story/thematic \
archetype, visual-craft acclaim, etc.).

Carver traits: identity OR attribute categories are both fair \
game — the user IS asking for the entity. The role analysis \
commits "dimensions describe the population", and the population \
can be defined by an identity category (the user wants credits / \
title / studio / character / franchise / source) or by attribute \
categories (the user wants a kind of movie described by its \
qualities), or by both.

Qualifier traits: the named entity is a positioning anchor, not a \
retrieval target. Committing an identity category for that entity \
puts the anchor itself in the result pool — never what a \
qualifier asks for. Route only to ATTRIBUTE categories that \
describe what the entity is LIKE: archetype, tonal register, \
visual craft, narrative shape, iconography, setting. The named \
entity itself never becomes a call's expression for a qualifier \
trait — only its describable attributes do. This rule follows \
directly from what "qualifier" means; it inherits to every \
qualifier_relation, including ones the prompt has never seen \
before.

OPERATIONAL TEST. Read your trait_role_analysis back. Does it \
clearly constrain what the dimensions list should contain? Could \
a different reader, given only this analysis, write the same kind \
of dimensions? If no, revise.

NEVER:
- LEAD WITH role AS THE HEADLINE QUESTION. role is the verdict; \
  qualifier_relation is the substantive signal that tells you \
  what the dimensions should describe. Read all sources; don't \
  stop at the binary.
- RE-INTERPRET qualifier_relation. Step 2 commit is the source of \
  truth — read it, don't second-guess.
- DERIVE A DIFFERENT ROLE from evaluative_intent.
- COMMIT AN IDENTITY CATEGORY for a qualifier trait's named \
  entity. The entity is being positioned against, not retrieved.
- SLOT THE QUALIFIER_RELATION INTO A FIXED VOCABULARY. There is \
  no closed list of relation types; describe what THIS query's \
  relation operationally requires.
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
qualify whether each axis describes the population (carver) or \
the reference being positioned against (qualifier).

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

NEVER:
- TRANSLATE INTO CATEGORY VOCABULARY here. Aspects are user-side \
  axes; categories are database-side routing. Mixing the two \
  defeats the mode-shift the step exists to enforce.
- INVENT AXES not grounded in role analysis. Aspect coverage is \
  audited against the prose above, not against the model's prior \
  knowledge of the trait.
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


_CATEGORY_ROUTING = """\
CATEGORY ROUTING — committing the call list

Read the per-dimension candidates. For each category that ends up \
owning >=1 expression, emit ONE CategoryCall. When several \
dimensions share a best-fit category, they merge into a single \
multi-expression call — NOT separate calls.

Schema field descriptions cover what category, expressions, and \
retrieval_intent must contain. Procedural notes:
- expressions: one per dimension this call owns, drawn from the \
  dimension's expression (verbatim, lightly tightened, or \
  recognizably the same check). Never split into multiple same-\
  category calls.
- retrieval_intent: when source trait is a qualifier, encode how \
  Step 4 should treat the retrieval — as the reference being \
  positioned against, as a threshold candidates must clear, etc. \
  Per the trait_role_analysis you committed.

CARVER VS QUALIFIER (read off role analysis):
- Carver: dimensions describe the population. Calls describe the \
  population to score against.
- Qualifier: dimensions describe the reference / shape. Calls \
  describe the reference being positioned against, NOT a result \
  pool of their own. retrieval_intent must encode the qualifier's \
  operational meaning.

OPERATIONAL TESTS:
- COVERAGE. Each dimension owned by exactly one call's \
  expressions. Zero → gap. Two → redundancy.
- MINIMUM-CALL. If I removed this call, would others still cover \
  its dimensions? Yes → padding; merge or drop.
- CANDIDATE-LINK. The category committed must have appeared as a \
  candidate on at least one of its owned dimensions.
- CLEAN-FIT. If a dimension's candidates list contains one with \
  what_this_misses="nothing", commit ONLY that one. Other \
  candidates were adjacency context surfaced for honesty during \
  the audit, not parallel routes. Multiple categories on a single \
  dimension are valid only when no candidate cleanly owns it. \
  Failing this test is how unnecessary calls leak in.
- POLARITY-DISCIPLINE. Does this call describe presence? Reject \
  any expression or retrieval_intent that includes negation / \
  absence — polarity is upstream.
- ROLE-CONSISTENCY. Does framing match trait_role_analysis? \
  Carver → population. Qualifier → reference / shape with \
  retrieval_intent encoding operational meaning.

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


def _build_user_prompt(trait: Trait) -> str:
    """Render the per-trait user prompt.

    Step 3 receives only per-trait commits — every literal-vs-
    parametric disambiguation and structural-positioning fact this
    step needs is already encoded in the trait's fields. The
    query-level intent_exploration prose is deliberately NOT
    surfaced here: it lives at Step 2 where it shapes the per-trait
    commits (atom partitioning, role_evidence). Sending query-level
    prose down would risk leaking other-trait interpretations into
    this trait's routing.

    The LLM gets the trait's contextualized_phrase (headline trait
    identity with modifiers folded in — Step 2's defense against
    shortcut routing on bare surface phrases), surface_text
    (verbatim grounding), evaluative_intent (the semantic seed),
    role (the carver/qualifier verdict), role_evidence (one-
    sentence rationale for the role commit; load-bearing on
    borderline traits and on carvers where qualifier_relation is
    "n/a"), qualifier_relation and anchor_reference (drive
    trait_role_analysis and the identity-vs-attribute principle),
    polarity (informational — calls express presence regardless),
    and relevance_to_query (signals how aggressively the trait
    warrants decomposition).

    contextualized_phrase appears AHEAD of surface_text — bare
    surface phrases stripped of their query context invite shortcut
    routing decisions that miss the modifier's role. surface_text
    is kept below for verbatim grounding.

    The qualifier_relation and anchor_reference fields are printed
    verbatim including the literal "n/a" sentinel — Step 3 reads
    "n/a" as an explicit "no qualifier-style relation here" signal,
    so we never conditionally hide them.

    Polarity is informational at this layer. Step 3 always
    describes presence of attributes; how the orchestrator composes
    calls (additive sum for positive traits and qualifiers,
    intersection over calls for carver+negative exclusions) is
    entirely orchestrator-side and does not need Step 3 to commit
    anything different in its decomposition.
    """
    return (
        "Trait to decompose:\n"
        f'- contextualized_phrase: "{trait.contextualized_phrase}"\n'
        f'- surface_text (verbatim): "{trait.surface_text}"\n'
        f"- evaluative_intent: {trait.evaluative_intent}\n"
        f"- role: {trait.role}\n"
        f"- role_evidence: {trait.role_evidence}\n"
        f"- qualifier_relation: {trait.qualifier_relation}\n"
        f"- anchor_reference: {trait.anchor_reference}\n"
        f"- polarity: {trait.polarity}    "
        "(informational — calls express presence regardless)\n"
        f"- relevance_to_query: {trait.relevance_to_query}\n"
    )


async def run_step_3(
    trait: Trait,
) -> tuple[TraitDecomposition, int, int, float]:
    """Run the trait-decomposition LLM on a single trait.

    Args:
        trait: the committed trait from Step 2.

    Returns:
        (response, input_tokens, output_tokens, elapsed_seconds) —
        elapsed measures wall-clock time spent inside the LLM call
        only, not prompt setup.
    """
    user_prompt = _build_user_prompt(trait)

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
