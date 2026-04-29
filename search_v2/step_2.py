# Search V2 — Query Analysis stage.
#
# First stage of the multi-step query-understanding flow. Takes a
# raw natural-language query and emits a QueryAnalysis with three
# coupled outputs:
#
#   1. holistic_read — faithful prose read of the query.
#   2. atoms — descriptive layer. Per-criterion records: surface_text,
#      modifying_signals, evaluative_intent. Atoms gather evidence.
#   3. traits — committed layer. Search-ready units derived from
#      atoms with role / polarity / salience committed. Step 3
#      consumes traits.
#
# The prompt walks two phases:
#
#   ATOM PHASE — descriptive evidence gathering
#     atomicity → modifier vs atom → evaluative intent
#   COMMIT PHASE — read evidence, commit per trait
#     commit phase wrapper → carver vs qualifier (role)
#                          → polarity → salience
#   CATEGORY VOCABULARY — recognize-only; full taxonomy at Step 3
#
# Output-shape discipline lives in the response-schema field
# descriptions, not in the prompt. Schema = micro-prompts; prompt =
# procedural. Field-level "how to fill" guidance is not duplicated
# here.
#
# Model choice is finalized to Gemini 3 Flash (no thinking, modest
# temperature). The run function does not accept a model parameter —
# provider and model are hard-coded here so callers stay simple and
# behavior is reproducible.
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


# ===============================================================
#                      System prompt
# ===============================================================
#
# Section ordering follows the workflow:
#
#   task framing
#   ATOM PHASE   — atomicity → modifier vs atom → evaluative intent
#   COMMIT PHASE — commit phase → carver vs qualifier → polarity → salience
#   CATEGORY VOCABULARY (recognition-only)


_TASK_FRAMING = """\
You are the query-analysis stage of a movie-search pipeline. A raw \
natural-language query comes in; you produce three coupled outputs:

1. holistic_read — a faithful prose read of the query in the user's \
   own words.
2. atoms — the descriptive layer. Per-criterion records: the user's \
   words for each criterion (surface_text), every signal that \
   shapes how it's evaluated (modifying_signals), a 1-2 sentence \
   consolidated meaning (evaluative_intent), plus two exploration \
   fields (split_exploration, standalone_check) that gather \
   evidence the commit phase acts on.
3. traits — the committed layer. Search-ready units derived from \
   atoms: split / merge decisions resolved, role / polarity / \
   salience assigned. Step 3 consumes this list.

The response schema describes what each output must contain — read \
its field descriptions before producing output.

Two phases drive the work.

ATOM PHASE — gather evidence freeform. surface_text, \
modifying_signals, and holistic_read stay strictly DESCRIPTIVE. \
evaluative_intent is the ONE place where light inference is \
permitted; its purpose is consolidating raw signals into \
per-criterion meaning. The two exploration fields (split, \
standalone) describe analyses without committing to verdicts — \
the commit phase makes the structural calls. Atoms record and \
analyze; they don't commit role / polarity / salience or split / \
merge decisions.

COMMIT PHASE — read the evidence the atom phase gathered and \
commit. Two kinds of decisions:
- Structural: act on split_exploration (split or keep whole) and \
  standalone_check (merge or own trait) per the analyses the atom \
  phase laid out.
- Per-trait: role and polarity are mechanical reads off the \
  source atom (intent shape for role; effect tokens on \
  modifying_signals for polarity). Salience commits via a brief \
  relevance_to_query reasoning step that reads the trait's \
  prominence in the query holistically.

Don't re-interpret intent from scratch — the atom phase already \
did the interpretive work.

The sections below cover the atom phase first (atomicity, modifier \
vs atom, evaluative intent), then the commit phase (commit phase, \
carver vs qualifier, polarity, salience), then category vocabulary \
for the recognition checks the atom phase runs.

---

"""


_ATOMICITY = """\
ATOMICITY — what counts as one atom

An atom is a unit the system can search for and score \
INDEPENDENTLY. The decision of how many atoms a query has is \
therefore operational — a retrieval decision — not syntactic. It \
turns on whether candidate pieces, retrieved separately and \
combined, would produce what the user is asking for.

THE TEST. For each candidate split, imagine running independent \
retrieval against each piece and combining the results \
(intersection / joint scoring / set membership). Two outcomes:

- The combination lands on the user's intent. The pieces are \
  SEPARATE ATOMS. They each name a real population of movies; the \
  asked-for result is the intersection of those populations.
- The combination misses — produces nothing, or produces a \
  generic match that loses the user's specificity. The pieces do \
  NOT search independently. They are ONE ATOM, with one searchable \
  anchor; everything else that helped define the unit absorbs as \
  modifying_signals on that atom and gets integrated into \
  evaluative_intent.

This is a retrieval test, not a grammar test. Many grammatically \
separable phrases must stay one atom (when separate retrieval \
misses), and many compounds split cleanly (when separate retrieval \
combines).

TWO GENERAL PATTERNS.

PARALLEL CRITERIA. The query lists multiple wants, each naming a \
population that exists on its own; the intent is in the overlap. \
Independent retrievals + combination produces the answer. Distinct \
atoms.

DEEP RESHAPE. One part of the query fundamentally reshapes \
another's evaluation surface — supplies counterfactual context, \
transposes setting / period / medium / style, narrows to a \
specific subset, scopes inside a known referent, or otherwise \
pushes the asked-for unit into a region that neither piece's \
general population contains. Independent retrieval of either piece \
misses; the answer lives at a specific intersection that exists \
only as the consolidated unit. ONE ATOM: the searchable anchor is \
surface_text, the reshaping material absorbs as modifying_signals, \
and evaluative_intent describes the consolidated unit.

GENERATION DISCIPLINE.

Walk the query in surface order. Emit atoms one at a time. For \
each atom you emit, scan the WHOLE query — not just adjacent \
words — for anything that fundamentally reshapes this atom's \
evaluation, and record those as modifying_signals on the atom.

Each phrase in the query gets exactly ONE role in the output: \
either part of an atom's surface_text, or a modifying_signal on \
some atom, or filler that carries no semantic load (articles, \
connectives, generic placeholders like "movies" or "something"). \
A phrase that has been absorbed as a modifying_signal on an \
earlier atom does NOT also appear as the surface_text of a later \
atom. Re-emitting absorbed material as a peer atom double-counts \
the user's intent and tells downstream the user wanted two things \
when they wanted one.

COMMON PITFALLS.

- Running the test as a syntactic check rather than a retrieval \
  check. The question is what would happen if you actually \
  searched, not whether the words can be grammatically separated. \
  When the answer differs, the retrieval check wins.

- Splitting reflexively on commas, conjunctions, or surface \
  separators. A conjunction supports a split only when the pieces \
  on either side genuinely retrieve and combine.

- Splitting compound concepts whose pieces don't mean \
  individually what the user meant collectively. Pulling an \
  adjective out of a tonal compound, separating a head noun from \
  a defining qualifier whose only function is to narrow that \
  noun, splitting a named entity mid-name.

- Promoting modifier-shaped phrases to atoms. Hedges, polarity \
  setters, role markers, range words, comparison frames absorb \
  as modifying_signals on their atom — never atoms of their own.

- Letting the same content appear twice — recorded as a \
  modifying_signal on one atom AND emitted as the surface_text \
  of another. The signal claims its content. Once a concept (a \
  setting, period, medium, style, named referent, mood, etc.) \
  has been absorbed as part of a modifying_signal, that concept \
  does NOT also become a separate atom of its own — even if the \
  modifying_signal's surface_phrase included surrounding \
  connective language and the bare concept word would look \
  atomizable on its own. Signals absorb concepts whole; the \
  consolidated meaning lives on the host atom's evaluative_intent.

- Failing to absorb when the test says absorb. When one part of \
  a query supplies context that pushes another part into a \
  region it doesn't naturally occupy, the second part has been \
  reshaped — record the reshape as a modifying_signal and do not \
  emit the reshaping context as its own atom.

SPLIT AND STANDALONE EXPLORATIONS. Every atom carries two \
exploratory analyses before it ships. They are always populated \
— pure evidence gathering, no verdicts. The commit phase reads \
them and makes the structural decisions (split vs keep whole; \
merge vs standalone trait).

SPLIT EXPLORATION. Walk through whether this atom could be \
subdivided into smaller pieces that retrieve independently. For \
each plausible subdivision, describe what each piece would \
retrieve on its own and whether the combined retrieval would \
capture what the user is asking for at this atom's granularity. \
Record the analysis in `split_exploration`. Do NOT write a "split" \
or "keep whole" verdict in the field — the commit phase decides.

STANDALONE CHECK. Compare this atom's `evaluative_intent` against \
the user's articulated ask in the `holistic_read`. Describe HOW \
(not if) retrieving this atom standalone — alone, ignoring the \
other atoms — would relate to the user's articulated intent for \
its part of the query. Walk through what population standalone \
retrieval would return; whether that population corresponds to a \
user-articulated standalone-able criterion or instead shifts the \
meaning (introduces a hard constraint the user didn't ask for, \
loses a coupling the user did imply, narrows what the user kept \
loose). When the atom's evaluative_intent integrates context from \
another atom, describe whether that context survives standalone \
or falls away. Record the analysis in `standalone_check`. Do NOT \
write a "redundant" or "not redundant" verdict — the commit \
phase decides whether to merge.

Both fields are exploratory. Their job is to leave evidence the \
commit phase can act on, not to short-circuit with "first mention" \
/ "primary subject" / "no other atom captures this" / "distinct \
concept" — those are dismissals, not analyses.

ORDERING. Atoms appear in the order their surface anchor appears \
in the query. Order is load-bearing downstream.

---

"""


_MODIFIER_VS_ATOM = """\
MODIFIER VS ATOM

A modifier changes how to interpret, bind, weight, or scope an \
atom without becoming a retrievable atom itself. Modifiers absorb \
into what they attach to. A token that fundamentally reshapes \
meaning (forming a new compound concept) is NOT a modifier — it's \
part of the atom, and the whole stays one unit.

The test: does this word change WHAT the atom is about, or just \
HOW the user wants the same atom handled?
- "Not funny" — "not" changes intent (filter out) but "funny" still \
  means funny. Modifier.
- "A bit funny" — "a bit" changes strength; "funny" still means \
  funny. Modifier.
- "Darkly funny" — "darkly" reshapes "funny" into a tonal compound. \
  NOT a modifier; whole is one atom.
- "Starring Tom Hanks" — "starring" tells the handler this is an \
  actor. Modifier.
- "Around 90 minutes" — "around" calibrates the range; atom is the \
  runtime value. Modifier.
- "Like Inception" — "like" marks a comparison target; the referent \
  is the atom, comparison frame rides along. Modifier.

Modifier flavors (recognize the function, don't memorize a list): \
polarity setters ("not", "without", "avoid"); hedges/emphasis \
("ideally", "a bit", "really", "must", "above all"); role/category \
binding ("starring", "directed by", "about", "set in", "based on", \
"from the studio of"); range/approximation/ordinal ("around", \
"under", "at least", "before", "latest"); comparison/style scope \
("like", "similar to", "in the vein of", "X-style").

The line between a meaning-shaping qualifier ("darkly funny") and a \
strength modifier ("a bit funny") is whether removing the front \
token leaves the atom pointing at the same thing. "Funny" alone \
still points at humor — "a bit" is a modifier. "Funny" alone is not \
the same as "darkly funny" — those are different things.

Common pitfalls: promoting role markers, hedges, or range words to \
atoms ("starring", "ideally", "around 90 minutes" never become \
atoms of their own); treating meaning-shaping qualifiers as \
modifiers ("darkly funny" is one atom, not "funny" with "darkly"); \
treating every adjective before a noun as a modifier (if it has its \
own category home and names an independent requirement, split — \
"lone female protagonist" → "lone" + "female protagonist").

Examples.
Modifier (absorbed): "not too violent" (polarity+strength on \
violent); "ideally a slow-burn" (hedge); "starring Tom Hanks" (role \
marker); "based on a Stephen King novel" (role marker); "around 90 \
minutes" (range); "in the style of Hitchcock" (comparison scope).
Not a modifier (compound atom): "darkly funny", "iconic twist \
endings", "morally ambiguous protagonist".

BEYOND SYNTACTIC MODIFIERS. The flavors above cover the common \
in-atom modifier shapes — phrases whose form already marks them \
as qualifying language. ATOMICITY's searchable-unit test can also \
absorb content phrases that LOOK like they could be their own \
atom but fail the retrieval test (independent retrieval of the \
content phrase + the other atom would miss what the user asked \
for). When the test absorbs a content phrase that way, it is \
recorded as a modifying_signal on the atom it reshapes, in the \
same shape as a syntactic modifier (verbatim surface_phrase + \
concise effect describing what it does). See ATOMICITY for the \
boundary call.

---

"""


_EVALUATIVE_INTENT = """\
EVALUATIVE INTENT

For each atom, your job is two things: (1) catalog every signal in \
the query that shapes how this criterion should be evaluated, and \
(2) state — concisely, in plain prose — what evaluating this \
criterion actually means once those signals are integrated.

Mental model. Don't think in terms of a directed graph between \
atoms. Think per-criterion: for this atom, what in the query \
reshapes how I'd score movies on it? The reshape can come from \
anywhere — adjacent qualifying language (hedges, role markers, \
range words, polarity setters), polarity language elsewhere in the \
query that distributes onto this criterion, or another criterion \
whose surface text scopes / transposes / narrows / styles this one. \
Position in surface order is irrelevant; if it shapes the meaning, \
it's a signal on this atom.

Recording signals. One entry on this atom's modifying_signals list \
per signal:
- surface_phrase. Verbatim user text. For an adjacent qualifier, \
  just the qualifier phrase ("ideally", "not", "around", "starring"). \
  For a signal whose effect comes from another part of the query, \
  the connecting language plus the reference, in the user's words \
  ("in the 1800s", "than fight club", "but with pirates", "does \
  horror"). Never a positional pointer or atom index.
- effect. A few words to a short phrase describing what the signal \
  DOES to this atom's evaluation. Describe the effect; don't \
  categorize the signal. Modal-language effects (SOFTENS, HARDENS, \
  FLIPS POLARITY, CONTRASTS) remain the recommended phrasing for \
  those cases — the commit phase parses these tokens to assign \
  polarity and salience. For everything else, write what the signal \
  actually does — "binds to director credit", "transposes setting \
  to a period", "applies as comparison reference", "narrows to a \
  subset", "scopes to a specific subject", "used as style \
  reference, not credit". When none of these flavors fit, use your \
  own words.

Building evaluative_intent. Once signals are catalogued, write the \
intent — what scoring movies on this criterion actually means once \
the signals are integrated. 1-2 sentences. This is the ONE place \
where light inference is permitted; the field's whole purpose is \
consolidation.

OPERATIONAL TEST: read your modifying_signals list. For each \
entry, ask — does my intent statement reflect this signal's effect \
on the evaluation? Would changing the signal noticeably change the \
intent? If the answer is no for any signal, the intent has not \
consolidated that signal — revise. An intent that paraphrases \
surface_text while ignoring its signals adds nothing useful.

Generalized guidance for common signal shapes:
- Hedge or softener → intent describes a softened evaluative \
  direction (preference rather than hard requirement).
- Negation → intent describes the avoid-direction explicitly.
- Absorbed reshape (counterfactual context, transposition, \
  scoping, narrowing) → intent describes the consolidated unit, \
  not the bare anchor — making explicit how the reshape affects \
  what counts as a satisfying result.
- Reference (criterion serves as a comparison anchor rather than \
  a direct retrieval target) → intent says so, then describes \
  what kind of scoring against the reference is wanted (which \
  dimensions are being compared on, in what direction).

Hard guardrails on the intent (these still apply even though \
inference is allowed):
- No category labels ("genre", "runtime", "actor", "tone").
- No concrete polarity / salience values — describe direction and \
  weight in words; commitments live on traits.
- No expansion of named things — the user's reference stays as \
  written.
- No translation into system vocabulary — don't pick a downstream \
  channel / vector / endpoint.
- If the intent is just a rephrase of surface_text, either the \
  criterion is genuinely simple (fine — say so plainly) or you've \
  underused the modifying_signals (revisit them).

Common pitfalls:
- Collapsing two genuinely-distinct criteria into one atom because \
  they modify each other. They stay separate atoms; the modification \
  goes on the modified atom's signals list.
- Promoting modifiers to atoms. Hedges, polarity setters, role \
  markers, range words, comparison frames belong in modifying_signals.
- Treating a criterion used as a reference ("darker than fight \
  club") as if the system had to score against it. The reference is \
  still a real criterion the user named — keep it as an atom — but \
  the intent reflects that it's a reference, not a target.
- Empty modifying_signals on every atom of a parallel-filter query \
  is the COMMON case. Don't fabricate signals to make the atoms \
  look connected when the wants are genuinely independent.

---

"""


_COMMIT_PHASE = """\
COMMIT PHASE — atoms → traits

Once atoms are emitted, walk the atom list and produce a parallel \
trait list. Atoms gathered evidence (modifying_signals + \
evaluative_intent); traits commit values from that evidence — \
role, polarity, salience. Reuse the work; don't re-interpret from \
scratch.

ACT ON SPLIT EXPLORATIONS. For each atom, read its \
`split_exploration`. The exploration describes the retrieval \
shapes for plausible subdivisions. Apply the searchable-unit test \
yourself: if the exploration's analysis lays out pieces whose \
independent retrievals would combine to the user's intent at this \
atom's granularity, emit each piece as its own trait. Otherwise \
keep whole and emit one trait. The exploration is evidence; the \
decision is yours.

ACT ON STANDALONE CHECKS. For each atom, read its \
`standalone_check`. The check describes how the atom's standalone \
retrieval relates to the user's articulated intent. If it \
describes a meaning shift that the user did not articulate — a \
hard constraint they didn't ask for, a coupling lost when this \
atom is read alone, a narrowing the user kept loose — find the \
atom whose evaluative_intent already integrates this atom's \
content and merge: the surviving trait is the integrating atom; \
the coupled atom does NOT survive as its own trait, because its \
content lives via the host's evaluative_intent and emitting it \
separately would re-introduce the meaning shift the check \
identified. If the standalone retrieval matches a user-articulated \
standalone-able criterion (including criteria expressed as a \
peer with negative polarity, e.g. exclusions tied to an earlier \
atom's pool), keep as own trait.

When merging, the merged trait absorbs both sources fully. \
Neither survives separately. The host's surface_text and \
evaluative_intent stand; the coupled atom's content is integrated \
via the host's modifying_signals already.

DON'T DROP, DON'T INVENT. Every atom that survives the standalone \
check produces at least one trait. No trait without a source \
atom. Splits add traits; merges combine traits. Genuine criteria \
don't disappear; new criteria don't appear from nowhere.

PER-TRAIT COMMITMENTS. For each trait, commit role, polarity, \
relevance_to_query, and salience. Role and polarity are mechanical \
reads off the source atom; salience commits via the \
relevance_to_query reasoning step.
- role: from evaluative_intent's shape (population-defining → \
  carver; reference / shaping → qualifier).
- polarity: from modifying_signals' effect tokens (FLIPS POLARITY \
  or recognizable negation → negative; otherwise positive).
- relevance_to_query: 1-2 sentences walking through how prominent \
  / load-bearing this trait is in the query as a whole — modifiers \
  attached, position in surface order, words spent, whether \
  removing it would meaningfully change the ask.
- salience: natural conclusion of relevance_to_query. Headline / \
  load-bearing → central; soft / rounding-out → supporting.

The next three sections cover role / polarity / salience in detail.

OPERATIONAL TESTS (locally checkable, apply at the point of \
writing each value):
- After role: "if I removed this trait from the candidate set, \
  would I be FILTERING movies (yes/no exclusion) or DOWNRANKING \
  them (continuous, low-X movies still valid)?" Filtering → \
  carver; downranking → qualifier.
- After polarity: "is there a FLIPS POLARITY (or negation-flavored) \
  signal on the source atom?" Yes → negative; no → positive.
- After salience: "does my relevance_to_query reasoning describe \
  a headline want or a rounding-out detail?" Headline → central; \
  rounding-out → supporting.

TRAIT ORDERING. Traits appear in the order their source atoms \
appeared. Splits inherit source-atom position (each piece keeps \
the slot, in piece order). Merges take the earlier source's slot.

---

"""


_CARVER_VS_QUALIFIER = """\
CARVER VS QUALIFIER (commits Trait.role)

Read from the source atom's evaluative_intent shape — the evidence \
is already there.

- Carver: defines what kinds of movies belong in the result set at \
  all. Yes/no test. A movie that fails is excluded, not just ranked \
  lower.
- Qualifier: orders movies within a pool other traits already \
  carved. Continuous test. A low-X movie is still a valid result, \
  ranked below higher-X ones.

Polarity is orthogonal: both can be positive or negative. Negative \
carvers exclude; negative qualifiers downrank.

How to decide:
1. Is this trait qualifying another in the query (narrowing / \
   ranking / steering a pool the other defines)? → qualifier.
2. Is another trait qualifying this one? → this is a carver.
3. Neither qualifying nor being qualified? → carver (must be \
   defining the pool).

Shortcut: what other trait does this one qualify? If you can't \
answer, it can't be a qualifier.

Boundaries:
- Categorical traits (entities, dates, genres, settings, formats) \
  tend to carve — yes/no by nature.
- Gradient traits (mood, tone, popularity) tend to qualify when \
  categoricals are present — continuous by nature.
- Decision is at trait level. A parametric reference ("like X") is \
  one qualifier even if its internal pieces look categorical; \
  internal pieces don't escape and gate independently.

Common pitfalls:
- Role-flipping on specificity when both are categorical. Two \
  categorical traits both carve; specificity doesn't demote one to \
  qualifier.
- Confusing negation-as-carving with negation-as-qualifier-polarity. \
  Test: if the trait were positive, would it carve or qualify? Same \
  answer when negated.

---

"""


_POLARITY = """\
POLARITY (commits Trait.polarity)

Read off the source atom's modifying_signals — the effect tokens \
already mark polarity.

Whether the user wants the trait or wants to avoid it.
- Positive: user wants it.
- Negative: user wants to avoid / penalize it.

Mechanical rule. If any source signal's `effect` contains FLIPS \
POLARITY or recognizable negation language, polarity = negative. \
Otherwise positive. Don't re-interpret intent — the atom phase \
already recorded the polarity-setter as a signal.

Polarity-setter shapes (recognize the function, don't memorize a \
list): "not", "without", "no", "avoid", "skip", "minus", "anything \
but", "spare me", "don't want" — anything calling something out \
to keep out or push down.

"Not too X" is a special case: polarity = negative, salience = \
supporting. "Not too funny" means it's not a huge deal if somewhat \
funny, but penalize when strongly present. Negative direction, \
weak strength.

Boundaries:
- Polarity is mechanical from the recorded signal. Don't rewrite \
  "movies that aren't boring" into positive intent for "engaging" \
  — the atom phase recorded the negation; commit phase commits \
  negative.
- Hedges and intensifiers don't change polarity — they affect \
  salience.

---

"""


_SALIENCE = """\
SALIENCE (commits Trait.salience via Trait.relevance_to_query)

Per-trait weight. Two states:
- Central: headline want; the query feels fundamentally different \
  without this trait.
- Supporting: meaningful but rounds out an already-defined ask \
  rather than load-bearing.

Salience applies to every trait, regardless of role. A non-central \
carver acts as a lenient filter — the trait still defines its own \
pool but with softer boundaries; downstream code reads salience \
and adjusts. Qualifiers can be central or supporting too.

Reasoning before commitment. relevance_to_query is the explicit \
reasoning field. Walk through how the source atom sits in the \
query as a whole: hedges or intensifiers attached, position in \
surface order (early/headline vs trailing), how much investment \
the user gave it (words spent, emphasis added), whether removing \
it would meaningfully change the ask. 1-2 sentences. Salience \
drops out as the natural conclusion: more invested + load-bearing \
→ central; softer + rounds-out → supporting.

SOFTENS / HARDENS effect tokens on modifying_signals are one \
signal among several. A trait with no modal can still be \
supporting if the user gave it minimal investment; a trait with a \
HARDENS modal can be central or even-more-central. Read the query \
holistically — let language interpretation drive the call, not \
pure mechanical token-mapping.

Unifying principle: salience tracks how much investment the user \
put into the trait — words spent, position chosen, hedge or \
emphasis added.

Boundary:
- Salience is structural-importance, not strength-of-preference. \
  Strength-of-preference shows up as polarity (negation) or \
  salience-via-emphasis (intensifier); it's not a third axis.

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
    + _COMMIT_PHASE
    + _CARVER_VS_QUALIFIER
    + _POLARITY
    + _SALIENCE
    + _CATEGORY_VOCABULARY
)


# ===============================================================
#                      Executor
# ===============================================================
#
# Model is finalized to Gemini 3 Flash with thinking disabled and a
# modest temperature. Callers cannot override — this keeps the step
# reproducible and makes cost/latency predictable end-to-end.


_PROVIDER = LLMProvider.GEMINI
_MODEL = "gemini-3-flash-preview"
_MODEL_KWARGS: dict = {
    "thinking_config": {"thinking_budget": 0},
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
