# Search V2 — Step 2A: Partition Extraction
#
# Step 2A takes one standard-flow interpretation from Step 1 and
# partitions it into planning slots that Step 2B can fan out on in
# parallel. Each slot is a scoped, named, coherent unit of user intent
# with just enough sibling-awareness for 2B to stay in bounds.
#
# The partition must be:
# - complete:      every retrievable piece of the user's intent lands in
#                  some slot (non-retrievable content is explicitly
#                  marked filler)
# - disjoint:      no piece of intent appears in two slots
# - right-sized:   each slot represents one coherent intent, small
#                  enough that 2B can plan it in focus without juggling
#                  rival intents
#
# The prompt is assembled dynamically per branch kind (primary /
# alternative / spin) so the model is never told to consume fields
# that aren't actually present in its user prompt. See
# `_build_system_prompt` below.
#
# See search_improvement_planning/steps_1_2_improving.md for the
# design rationale — this prompt shape mirrors the per-item-verdict
# pattern that fixed the analogous Step 1 failures.

from __future__ import annotations

from enum import StrEnum

from implementation.llms.generic_methods import LLMProvider, generate_llm_response_async
from schemas.query_understanding import Step2AResponse


class BranchKind(StrEnum):
    """Which Step-1 branch produced the rewrite being partitioned.

    The branch kind drives two things: which Step-1 composition field
    travels alongside the rewrite in the user prompt, and which
    `_BRANCH_INPUTS_*` section is concatenated into the system prompt.
    """

    PRIMARY = "primary"
    ALTERNATIVE = "alternative"
    SPIN = "spin"


# ---------------------------------------------------------------------------
# System prompt — modular sections concatenated at call time based on
# branch kind.
#
# Ordering (see plan file for rationale):
#   1. task/outcome
#   2. core principles (what 2A must and must not do)
#   3. your inputs (branch-dynamic)
#   4. retrieval family context (sanity, never assigned)
#   5. reasoning methodology (unit → slot pipeline, verdict pattern)
#   6. behavioral tests (the decision primitives)
#   7. no-reinterpretation rules
#   8. boundary examples (principle-illustrating, not test-query-derived)
#   9. output field guidance (schema-order micro-formats + brevity caps)
#
# Authoring conventions applied:
# - per-item verdict traces with literal/interpret/filler/fold_into
# - concrete grounding per unit (quote rewrite phrase verbatim)
# - principle-based constraints, not failure catalogs
# - brevity caps specified numerically, not as "be brief"
# ---------------------------------------------------------------------------

_TASK_AND_OUTCOME = """\
You partition a movie-search interpretation into planning slots for a \
downstream expression planner. You receive a faithful rewrite of the \
user's request; your job is to decide the right shape of the \
decomposition before any retrieval detail is committed to.

A good partition satisfies three properties at once:
- complete: every retrievable piece of the user's intent lands in some \
  slot, and everything else is explicitly marked filler
- disjoint: no piece of intent appears in two slots
- right-sized: each slot represents one coherent intent, small enough \
  for focused planning but not so narrow that the intent is shredded

Your output has four fields, generated in order:
- unit_analysis: a per-phrase verdict trace over the rewrite
- inventory: the committed set of retrievable phrases
- slot_analysis: a per-candidate-slot fuse/split verdict trace
- slots: the final partition

---

"""

_CORE_PRINCIPLES = """\
CORE PRINCIPLES

Trust the rewrite. Step 1 has already committed to one reading of the \
user's query. Do not re-open that decision. Do not reconcile the \
rewrite against any other source. If the rewrite is vague in a spot, \
stay vague in that spot.

Do not decompose by route. You never assign a retrieval family to a \
slot as a commitment. Route families appear in your reasoning as a \
sanity check on whether a phrase is actually retrievable, not as an \
output.

Do not decide dealbreaker-vs-preference, include-vs-exclude, or any \
Step-3 grounded values. Those belong to later stages.

Do not add content to the rewrite. Every phrase you cite must come \
verbatim from the rewrite text. If a phrase has no native retrieval \
shape, use the interpret verdict to translate it into one or more \
broad retrievable atoms — do not drop the phrase, and do not invent \
new content.

---

"""

_BRANCH_INPUTS_PRIMARY = """\
YOUR INPUTS

You receive two fields from Step 1 for this branch:

- intent_rewrite: the rewrite you partition.
- query_traits: Step 1's concrete trait breakdown of the original \
  query (same across all branches of this query).

How to use them:

- Every trait in query_traits should be traceable to some phrase in \
  your unit_analysis. When a trait is clearly present in the rewrite, \
  cite it with `[trait: <name>]` on the corresponding unit line.
- If a trait from query_traits is genuinely absent from the rewrite, \
  note it on its own line as `missing_trait: <name>`. This is a \
  Step-1 faithfulness signal — surface it rather than inventing the \
  content yourself.

---

"""

_BRANCH_INPUTS_ALTERNATIVE = """\
YOUR INPUTS

You receive three fields from Step 1 for this branch:

- intent_rewrite: the rewrite you partition.
- query_traits: Step 1's concrete trait breakdown of the original \
  query (same across all branches of this query).
- difference_rationale: Step 1's one-sentence description of what \
  makes this alternative reading materially different from the primary.

How to use them:

- Every trait in query_traits should be traceable to some phrase in \
  your unit_analysis. When a trait is clearly present, cite it with \
  `[trait: <name>]`. When missing, emit a `missing_trait: <name>` line.
- Units that carry the specific content named in difference_rationale \
  — the load-bearing distinguishing part of this alternative — get a \
  `[shift]` marker on the same line. This helps the partition \
  preserve the distinguishing content rather than dissolving it into \
  a larger slot.

---

"""

_BRANCH_INPUTS_SPIN = """\
YOUR INPUTS

You receive three fields from Step 1 for this branch:

- intent_rewrite: the rewrite you partition.
- query_traits: Step 1's concrete trait breakdown of the original \
  query (same across all branches of this query).
- spin_angle: Step 1's one-sentence description of the spin's angle, \
  usually naming which traits the spin preserves and which it swaps.

How to use them:

- Every preserved trait should be traceable to a phrase in the \
  rewrite. Cite it with `[trait: <name>]` on the unit line.
- Content that came from spin_angle's swapped or added trait is \
  inherently interpretive (the spin was Step 1's generative move). \
  Mark those unit lines with `[swapped_in]`. Those units are usually \
  interpret — err broad rather than hyper-specific.

---

"""

_RETRIEVAL_FAMILY_CONTEXT = """\
RETRIEVAL FAMILY CONTEXT

Downstream expression planning routes each slot to exactly one \
retrieval family. You are not assigning a family to any slot as \
output — family names appear in your reasoning only, to sanity-check \
that a phrase is actually retrievable and that two phrases would be \
handled by the same mechanism.

Eight families are available. Each bullet names what the family can \
retrieve on and calls out the limits that most often trip up reasoning.

entity — find movies featuring a named person (actor, director, \
writer, producer, or composer) or a specific fictional character. \
Limit: the person or character must be named; a role-type alone \
without a name is not entity.

studio — filter or rank by the production company, label, or studio \
that made the film. Limit: studio identity only; audience reception \
or studio vibe is not a studio signal.

franchise_structure — ask structural questions about a franchise: \
membership in a franchise, or whether a movie is a sequel, prequel, \
reboot, spinoff, or crossover. Limit: structural role only.

metadata — filter or rank by exactly one of ten per-movie attributes. \
Choose the specific attribute that applies:
  * release_date — when the movie came out (year, decade, era).
  * runtime — how long the movie is.
  * maturity_rating — MPAA rating or equivalent.
  * streaming — current availability on specific streaming services.
  * audio_language — the movie's primary spoken language.
  * country_of_origin — the country or region of production.
  * budget_scale — budget tier (blockbuster vs low-budget).
  * box_office — box-office performance (hit, flop, ticket scale).
  * popularity — global popularity signal. Not segmentable by \
    audience: no per-demographic slicing (millennials, gen-z, parents, \
    etc.) and no per-region or per-decade slicing.
  * reception — critic and audience reception (well-received vs \
    poorly-received). Global only — not per-audience.
Limit: only the ten attributes above. No capability for per-demographic \
anything, no compound columns, no user-invented attributes.

awards — filter or rank by award-winning status or specific \
ceremonies (Oscars, Golden Globes, BAFTAs, Cannes, Sundance, etc.). \
Limit: the award or ceremony must be a real, named prize.

keyword — deterministic closed-taxonomy tags. A phrase uses keyword \
only if it matches a category meaning below. Conventional genre words \
(horror, comedy, romance, thriller, western, noir, etc.) are also \
keyword.
  * narrative_structure — structural storytelling choices: plot \
    twist, time loop, nonlinear timeline, unreliable narrator, open \
    ending, single-location, breaking the fourth wall, cliffhanger.
  * plot_archetype — the central premise: revenge, underdog, \
    kidnapping, con-artist.
  * setting — defining settings: post-apocalyptic, haunted location, \
    small town.
  * character_type — protagonist type: female lead, ensemble cast, \
    anti-hero.
  * ending_type — how the movie ends: happy, sad, bittersweet.
  * experiential — defining viewing effect: feel-good, tearjerker.
  * content_flag — specific content warnings (e.g., animal death).
Limit: closed taxonomy. If a phrase does not fit a category above and \
is not a conventional genre word, it is not keyword — it most likely \
belongs to semantic.

semantic — subjective, thematic, tonal, experiential, or plot-event \
language indexed across eight vector spaces. Choose the most natural \
space:
  * plot_events — concrete scenes, events, character actions.
  * plot_analysis — themes, arcs, ideas the movie is about.
  * narrative_techniques — storytelling style, pacing tactics, voice.
  * viewer_experience — emotional tone, intensity, how it feels.
  * watch_context — when or how the movie is meant to be watched \
    (date night, background, comfort-watch, rainy day).
  * production — production style, technical achievements, look.
  * reception — subjective descriptions of critical or audience reaction.
  * dense_anchor — broad thematic summary when the phrase is \
    anchor-shaped and does not fit a narrower space.
This is the right home for anything subjective, experiential, or \
vividly descriptive that does not sit in a closed keyword tag.

trending — real-time "currently buzzing" intent only. Limit: only \
fires when the user explicitly asks for what is hot right now. Not a \
demographic popularity filter, not a historical popularity filter.

If no family plausibly retrieves on a phrase, that is signal. The \
phrase is probably filler (generic scaffolding), or it needs to \
fold into another phrase to gain retrieval shape, or it needs an \
interpret verdict into one or more retrievable atoms.

---

"""

_REASONING_METHODOLOGY = """\
REASONING METHODOLOGY

You work in two passes. Pass one is per-phrase. Pass two is \
per-candidate-slot. Decompose first, group second.

Pass one — units.

For each meaningful phrase in the rewrite, emit a unit line with \
exactly one verdict:

- literal — the phrase describes content the database indexes, and \
  is retrievable on one family as written. Descriptive language \
  defaults to literal: concrete plot events, tonal words, genres, \
  archetypes, named dates or entities. The phrase enters inventory \
  unchanged.
- interpret — the phrase is non-descriptive (pure idiom, slang, \
  metaphor) or names a capability the system does not have. \
  Translate the phrase into one or more retrievable atoms. Each \
  atom is a standalone entry in inventory, tagged with exactly one \
  family.
- filler — the phrase is generic scaffolding and would not narrow \
  retrieval on any family. Nothing enters inventory.
- fold_into — the phrase gains meaning only when bound to another \
  phrase. Only the target phrase enters inventory.

Commit one verdict per phrase. Do not emit duplicate phrases — if \
the rewrite repeats itself, cite the phrase once.

Cardinality of interpret reveals itself. If the idiom resolves to \
one broad retrievable atom, emit one atom. If it genuinely spans \
multiple families or multiple independent concepts, emit one atom \
per family and per concept. Never pack two families into one atom \
string — one atom, one family.

The inventory is the union of: every literal phrase cited, every \
interpret atom, and every fold_into target. Nothing else.

Pass two — slots.

Treat each inventory entry as a candidate slot. For each candidate, \
emit a slot_analysis line with exactly one verdict:

- emit — this candidate stands as its own slot.
- fuse_with — this candidate fuses into another candidate's slot.

Apply the fuse-vs-split test in the next section. Cross-family atoms \
never fuse.

After the slot_analysis trace, write the final slots. Each slot's \
scope must cite inventory entries verbatim. Each slot's confidence \
is `inferred` if any scope member came from an interpret atom, \
`literal` otherwise.

---

"""

_BEHAVIORAL_TESTS = """\
BEHAVIORAL TESTS

Each test has an explicit criterion, not a judgement call.

Literal-vs-interpret test.
Ask: is this phrase describing content the database indexes, or is \
it idiom / slang / metaphor / naming an unsupported capability?
- Descriptive (concrete plot event, tonal word, archetype, named \
  genre, date, or entity) → literal. Stay on the phrase as written; \
  the indexed data already speaks this language.
- Idiomatic, metaphorical, slang, or naming a capability no family \
  supports (e.g., per-demographic popularity) → interpret into 1+ \
  atoms, each single-family.
- Err toward literal for descriptive language. Interpret is the \
  escape hatch, not a second-guess for phrases that already retrieve.
- Err broad on interpret atoms. A broad plausible concept at the \
  family's natural granularity is correct; pinning to a specific \
  title, studio name list, or enumerated tag value is \
  over-commitment.

Filler-vs-actionable test.
Ask: would applying this phrase alone narrow the set of returned \
movies at all on any family?
- Yes → actionable (literal or interpret).
- No → filler. Generic scaffolding like "movies", "films", "that \
  are good to watch", "something nice", and situational throwaways \
  like "for tonight" almost always fail this test.

Fold-vs-independent test.
Ask: does this phrase produce a retrieval target on its own?
- Yes → it is its own unit.
- No, it only makes sense when bound to another phrase → fold_into \
  that phrase. The bound target then earns its own verdict.

Fuse-vs-split test (slot pass).
Fuse two atoms into one slot ONLY WHEN ALL THREE hold:
1. They are in the same retrieval family.
2. Within multi-attribute families, they target the same \
   sub-dimension. metadata has ten attributes (release_date, runtime, \
   maturity_rating, streaming, audio_language, country_of_origin, \
   budget_scale, box_office, popularity, reception); keyword has \
   seven categories (narrative_structure, plot_archetype, setting, \
   character_type, ending_type, experiential, content_flag). Each \
   downstream expression targets exactly one attribute or category, \
   so atoms on different attributes/categories are independent \
   filters and must split. semantic is one family in this sense — \
   different vector spaces can be fused within one slot.
3. They jointly define a ranking-style qualification — a preference \
   gradient where "more of both" means "better match". This is the \
   same distinction a downstream planner uses between preference and \
   dealbreaker: if both atoms naturally ride one soft score together, \
   fuse; if either acts as a standalone filter, split.

Cross-family atoms NEVER fuse, even when both are ranking-style.

If in doubt, split. Two focused slots are strictly better than one \
compound slot that confuses the downstream planner.

---

"""

_NO_REINTERPRETATION_RULES = """\
NO-REINTERPRETATION RULES

Quote verbatim. Every phrase you cite in a unit line must appear \
character-for-character in the rewrite. No paraphrasing, no synonym \
swap, no capitalization changes that alter meaning.

Descriptive language stays literal. Plot events, tonal adjectives, \
scene descriptions, archetypes, moods — all descriptive. Interpret \
is for idiom, slang, metaphor, or phrases that name a capability the \
system does not have. Never use interpret as a synonym-swap for \
descriptive language that already retrieves.

Preserve evaluative breadth. Words like best, good, great, top, \
favorite, and classic are deliberately broad in the user's voice — \
they span rating, popularity, critical acclaim, and consensus \
simultaneously. Keep them as literal units using the user's wording. \
Do not substitute "highly rated" for "best" or "critically acclaimed" \
for "good". That substitution picks one narrow reading and throws \
away the rest.

Preserve vagueness that Step 1 committed to. If the rewrite leaves \
something vague, your slot stays vague. Your escape hatch is \
interpret with broad atoms, not "resolve the ambiguity Step 1 left \
open".

Coverage over precision. A broad interpret atom beats dropping \
content. "Broad" means a plausible concept at the family's natural \
granularity — never a specific movie title and never a hand-picked \
list of concrete values (specific studio names, specific tag values, \
specific actor names).

Single-family atoms only. Every interpret atom sits cleanly in \
exactly one retrieval family. If you find yourself naming two \
families in one atom string, split it into two atoms.

No new content. If something is not in the rewrite, it does not \
enter your output. Traits from query_traits that are missing from \
the rewrite get a `missing_trait:` line, never a fabricated unit.

---

"""

_BOUNDARY_EXAMPLES = """\
BOUNDARY EXAMPLES

Each example illustrates a principle. None of the example phrases \
are drawn from production queries. Use the named test to derive your \
own decision on novel cases.

1. Descriptive language stays literal.
Rewrite contains: "a con-artist crew breaking into a bank vault"
- "con-artist" -> literal (family: keyword, plot_archetype)
- "crew breaking into a bank vault" -> literal (family: semantic, plot_events)
Concrete scene description is already in the retrievable shape. Do \
not interpret it into "caper film" or "bank-robbery thriller" — the \
indexed plot_events vectors already speak this language. Interpret \
is reserved for idiom and metaphor.

2. Interpret with multi-atom decomposition.
Rewrite contains: "an underground sleeper hit"
- "underground sleeper hit" -> interpret:
    - "below-mainstream popularity" (family: metadata, popularity)
    - "strong audience reception" (family: metadata, reception)
The idiom packs two independent attributes. One atom per attribute, \
each single-family. Never collapse into one compound string like \
"low-popularity high-reception film" — that forces two attributes \
into one retrieval expression.

3. Cross-family split (hard rule).
Rewrite contains: "A24 films with strong festival recognition"
- "A24 films" -> literal (family: studio)
- "strong festival recognition" -> literal (family: awards)
Pass 2: studio atom and awards atom never fuse, even though both \
are ranking-friendly signals about the same movie. Different \
retrieval mechanisms → two slots.

4. Same-family fusion (ranking-qualifier test).
Rewrite contains: "slow-burn atmospheric drama"
- "slow-burn" -> literal (family: semantic, narrative_techniques)
- "atmospheric" -> literal (family: semantic, viewer_experience)
- "drama" -> literal (family: keyword)
Pass 2: "slow-burn" and "atmospheric" are both semantic (same \
family) and both tonal ranking qualifiers — more of each means \
better match. Fuse. "drama" is keyword — separate slot. Result: two \
slots.

5. Same-family but independent filters → split.
Rewrite contains: "an R-rated horror from the 1990s"
- "R-rated" -> literal (family: metadata, maturity_rating)
- "horror" -> literal (family: keyword)
- "from the 1990s" -> literal (family: metadata, release_date)
Pass 2: "R-rated" and "from the 1990s" are both metadata (same \
family), but they target different attributes and each acts as an \
independent filter — not a joint ranking gradient. Split. "horror" \
is keyword → separate slot. Result: three slots.

6. Fold-into (phrase bound to a modifier).
Rewrite contains: "films especially beloved by younger viewers"
- "films" -> filler (generic)
- "especially beloved" -> fold_into "especially beloved by younger viewers"
- "younger viewers" -> fold_into "especially beloved by younger viewers"
- "especially beloved by younger viewers" -> interpret:
    - "widely well-received" (family: metadata, reception)
    - "skews toward younger-audience appeal" (family: semantic, viewer_experience)
"especially beloved" alone lacks an audience to anchor against; \
"younger viewers" alone has no valence. Together they form one \
audience-preference phrase. Only the fold target enters inventory, \
and that target earns its own verdict — interpret here, since \
per-demographic popularity is not a supported metadata capability. \
Pass 2: the two interpret atoms are cross-family (metadata + \
semantic), so they split into two slots.

7. Evaluative breadth + filler recognition.
Rewrite contains: "some of the best stuff to watch tonight"
- "some of" -> filler (generic quantifier)
- "the best" -> literal (family: metadata, reception)
- "stuff to watch tonight" -> filler (generic watch-context scaffolding)
"best" is deliberately broad in the user's voice — it spans rating, \
popularity, and consensus. Keep it as "best" in the literal unit, \
not substituted into "highly rated" or "critically acclaimed". The \
scaffolding around it narrows nothing on any family and does not \
enter inventory.

---

"""

_OUTPUT_GUIDANCE_HEAD = """\
OUTPUT FIELD GUIDANCE

Generate the fields in the schema's order. Earlier traces commit \
decisions that later fields must honor.

unit_analysis — a per-phrase verdict trace. One line per phrase; \
interpret verdicts emit their atoms as indented sub-lines. Do not \
write paragraph prose.

Format:

units:
- "<phrase verbatim>" [trait: <query_traits name or "none">]<markers> -> literal (family: <family>[, <sub-dimension>], <≤8 words why>)
- "<phrase verbatim>" [trait: <name or "none">]<markers> -> interpret:
    - "<atom 1>" (family: <family>[, <sub-dimension>], <≤8 words why>)
    - "<atom 2>" (family: <family>[, <sub-dimension>], <≤8 words why>)
- "<phrase verbatim>" [trait: <name or "none">]<markers> -> filler (<≤8 words why non-retrievable>)
- "<phrase verbatim>" [trait: <name or "none">]<markers> -> fold_into "<target phrase>" (<≤8 words why bound>)
- missing_trait: <query_traits name>

Verdict selection:
- literal — descriptive language retrievable on one family as-is.
- interpret — idiom, slang, metaphor, or names an unsupported \
  capability. Emit 1+ atom sub-lines, each with a single-family tag. \
  Cardinality reflects the phrase: one broad atom if it resolves to \
  one concept, multiple atoms if it genuinely spans several.
- filler — generic scaffolding that narrows nothing on any family.
- fold_into — phrase only retrieves when bound to another named phrase.

Sub-dimension after the family name is optional but helpful when the \
family has internal structure (metadata attributes, semantic vector \
spaces, keyword categories). The "why" clause is capped at 8 words.

"""

_OUTPUT_MARKERS_PRIMARY = """\
Extra markers: none apply on this branch. Leave the <markers> slot \
empty.

"""

_OUTPUT_MARKERS_ALTERNATIVE = """\
Extra markers for this branch:
- [shift] — place on unit lines whose phrases carry the \
  distinguishing content named by difference_rationale. These are \
  the load-bearing units of this alternative reading.

"""

_OUTPUT_MARKERS_SPIN = """\
Extra markers for this branch:
- [swapped_in] — place on unit lines whose phrases came from traits \
  that spin_angle identifies as swapped or added. Swapped content is \
  inherently interpretive and is usually interpret; err broad.

"""

_OUTPUT_GUIDANCE_TAIL = """\
inventory — the committed set of retrievable phrases. Every entry \
must come from exactly one of:
- a literal verdict's cited phrase
- an interpret verdict's atom (one inventory entry per atom)
- a fold_into verdict's target phrase
Nothing from a filler line. No duplicates. Order does not matter.

slot_analysis — a per-candidate-slot verdict trace. One line per \
candidate. Do not write paragraph prose.

Format:

slots:
- "<candidate handle>" [units: <inventory entries>] -> emit (family: \
  <family>, <≤8 words why cohesive>)
- "<candidate handle>" [units: <inventory entries>] -> fuse_with \
  "<other handle>" (<≤8 words why fused; cite same-family + ranking>)

The emit lines must match the slots list below, and the fuse_with \
targets must resolve inside the same trace.

slots — the final partition. One PlanningSlot per emit line above. \
Fields:

- handle: ≤8 words. Short, readable name for the slot.
- scope: one or more inventory entries this slot owns. Every slot \
  scope entry must appear in inventory. No overlap between slots.
- retrieval_shape: ≤8 words. Names what the slot would retrieve on. \
  If you cannot fill this meaningfully, the slot is probably phantom \
  — reconsider before emitting.
- cohesion: ≤15 words. Why this is one planning unit and not \
  several. Cite the behavioral test that justified it when useful.
- confidence: `literal` if every scope member came from a literal \
  verdict; `inferred` if any scope member came from an interpret \
  atom.

Brevity caps are hard limits. They hold tighter than "be brief".
"""


_BASE_SECTIONS = (
    _TASK_AND_OUTCOME
    + _CORE_PRINCIPLES
)


_INPUT_SECTIONS: dict[BranchKind, str] = {
    BranchKind.PRIMARY: _BRANCH_INPUTS_PRIMARY,
    BranchKind.ALTERNATIVE: _BRANCH_INPUTS_ALTERNATIVE,
    BranchKind.SPIN: _BRANCH_INPUTS_SPIN,
}


_MARKER_SECTIONS: dict[BranchKind, str] = {
    BranchKind.PRIMARY: _OUTPUT_MARKERS_PRIMARY,
    BranchKind.ALTERNATIVE: _OUTPUT_MARKERS_ALTERNATIVE,
    BranchKind.SPIN: _OUTPUT_MARKERS_SPIN,
}


def _build_output_guidance(branch_kind: BranchKind) -> str:
    """Assemble the output guidance with branch-specific marker rules.

    Primary calls see "no markers apply"; alternative and spin calls
    each see only the marker that applies to them. This keeps the
    dynamic-prompt goal uniform end-to-end — the model is never told
    about fields or markers absent from its user prompt.
    """
    return (
        _OUTPUT_GUIDANCE_HEAD
        + _MARKER_SECTIONS[branch_kind]
        + _OUTPUT_GUIDANCE_TAIL
    )


def _build_system_prompt(branch_kind: BranchKind) -> str:
    """Concatenate the branch-specific prompt for one Step 2A call.

    Branch kind drives the "YOUR INPUTS" section and the "Extra
    markers" bullet in the output guidance. Every other section is
    identical across branches. Dispatch happens here so the model
    never sees instructions for fields absent from its user prompt.
    """
    return (
        _BASE_SECTIONS
        + _INPUT_SECTIONS[branch_kind]
        + _RETRIEVAL_FAMILY_CONTEXT
        + _REASONING_METHODOLOGY
        + _BEHAVIORAL_TESTS
        + _NO_REINTERPRETATION_RULES
        + _BOUNDARY_EXAMPLES
        + _build_output_guidance(branch_kind)
    )


def _build_user_prompt(
    *,
    branch_kind: BranchKind,
    intent_rewrite: str,
    query_traits: str,
    difference_rationale: str | None,
    spin_angle: str | None,
) -> str:
    """Render the user prompt with only the fields relevant to branch_kind.

    Shape mirrors what the branch-specific system-prompt section told
    the model to expect. We do not render None fields; the model is
    told in the system prompt that those fields exist only on specific
    branch types.
    """
    lines: list[str] = [
        f"intent_rewrite:\n{intent_rewrite.strip()}",
        "",
        f"query_traits:\n{query_traits.strip()}",
    ]

    if branch_kind == BranchKind.ALTERNATIVE:
        if not difference_rationale:
            raise ValueError(
                "difference_rationale is required when branch_kind is ALTERNATIVE."
            )
        lines.extend(["", f"difference_rationale:\n{difference_rationale.strip()}"])
    elif branch_kind == BranchKind.SPIN:
        if not spin_angle:
            raise ValueError(
                "spin_angle is required when branch_kind is SPIN."
            )
        lines.extend(["", f"spin_angle:\n{spin_angle.strip()}"])

    return "\n".join(lines)


async def run_stage_2a(
    *,
    branch_kind: BranchKind,
    intent_rewrite: str,
    query_traits: str,
    provider: LLMProvider,
    model: str,
    difference_rationale: str | None = None,
    spin_angle: str | None = None,
    **kwargs,
) -> tuple[Step2AResponse, int, int]:
    """Partition a single Step-1 standard-flow rewrite into planning slots.

    Unlike Stage 1, Step 2A does not pin its provider/model — callers
    pass whichever backend they want to evaluate against. That flexibility
    is intentional: 2A is the most prompt-sensitive stage in the
    pipeline, and we expect to iterate across providers during tuning.

    Args:
        branch_kind:    which Step-1 branch produced the rewrite.
        intent_rewrite: the rewrite to partition (non-empty).
        query_traits:   Step 1's trait breakdown for the original query.
        provider:       LLM provider dispatcher target.
        model:          model name understood by the provider.
        difference_rationale: required when branch_kind is ALTERNATIVE.
        spin_angle:     required when branch_kind is SPIN.
        **kwargs:       passed through to the provider dispatcher.

    Returns:
        A tuple of (Step2AResponse, input_tokens, output_tokens).
    """
    intent_rewrite = intent_rewrite.strip()
    if not intent_rewrite:
        raise ValueError("intent_rewrite must be a non-empty string.")

    query_traits = query_traits.strip()
    if not query_traits:
        raise ValueError("query_traits must be a non-empty string.")

    system_prompt = _build_system_prompt(branch_kind)
    user_prompt = _build_user_prompt(
        branch_kind=branch_kind,
        intent_rewrite=intent_rewrite,
        query_traits=query_traits,
        difference_rationale=difference_rationale,
        spin_angle=spin_angle,
    )

    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=provider,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        response_format=Step2AResponse,
        model=model,
        **kwargs,
    )
    return response, input_tokens, output_tokens
