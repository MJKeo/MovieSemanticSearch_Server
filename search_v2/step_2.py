# Search V2 — Query Analysis stage.
#
# First stage of the multi-step query-understanding flow. Takes a
# raw natural-language query and emits a QueryAnalysis: a faithful
# prose read of the query plus per-criterion atoms, each carrying
# the user's words for the criterion, every signal in the query
# that shapes how it should be evaluated, and a consolidated
# evaluative_intent statement. Downstream stages consume the
# evaluative_intent for evaluation; modifying_signals carries
# provenance.
#
# The system prompt loads principle sections this stage applies
# directly (atomicity, modifier vs trait, evaluative intent) plus
# background-context sections later stages use (carver vs
# qualifier, polarity, salience, category taxonomy). Keeping the
# background sections loaded here lets us measure prompt size as a
# baseline before deciding what to defer to per-stage prompts.
# Output-shape discipline lives in the response-schema field
# descriptions, not in the prompt.
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
# Section ordering:
#
#   task framing → atomicity → modifier vs trait → evaluative
#   intent → carver vs qualifier → polarity → salience →
#   category taxonomy
#
# This stage applies the first three principle sections (atomicity,
# modifier vs trait, evaluative intent) directly when producing
# atoms. The remaining sections (carver vs qualifier, polarity,
# salience, category taxonomy) are loaded as background context
# for downstream stages. Field-level "how to fill" guidance lives
# in the response schema, not here.


_TASK_FRAMING = """\
You are the query-analysis stage of a movie-search pipeline. A raw \
natural-language query comes in; you produce two coupled outputs:

1. holistic_read — a faithful prose read of the query in the user's \
   own words, describing what they're asking for and how the pieces \
   of that ask affect each other.
2. atoms — the query's evaluative criteria. For each criterion, \
   record the user's words for it (surface_text), catalog every \
   signal in the query that shapes how it should be evaluated \
   (modifying_signals), and state — concisely, in plain prose — \
   what evaluating it actually means once that context is \
   considered (evaluative_intent).

The response schema describes what each output must contain and how \
to fill it — read its descriptions before producing output.

surface_text and modifying_signals stay strictly DESCRIPTIVE. \
holistic_read stays strictly DESCRIPTIVE. evaluative_intent is the \
ONE place where light inference is permitted — that's the field's \
purpose: consolidating the raw signals into per-criterion meaning. \
Even there, do NOT commit to downstream decisions (concrete \
polarity / salience numbers, category labels, search strategy, \
weights). Those happen at later stages.

The sections below describe principles you APPLY directly (atomicity, \
modifier vs trait, evaluative intent) plus background context \
LATER stages use (carver vs qualifier, polarity, salience, category \
taxonomy). Apply the principle sections; the background sections are \
loaded for context but not used by you.

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

ORDERING. Atoms appear in the order their surface anchor appears \
in the query. Order is load-bearing downstream.

---

"""


_MODIFIER_VS_TRAIT = """\
MODIFIER VS TRAIT

A modifier changes how to interpret, bind, weight, or scope a \
trait without becoming a retrievable trait itself. Modifiers absorb \
into what they attach to. A token that fundamentally reshapes \
meaning (forming a new compound concept) is NOT a modifier — it's \
part of the trait, and the whole stays one unit.

The test: does this word change WHAT the trait is about, or just \
HOW the user wants the same trait handled?
- "Not funny" — "not" changes intent (filter out) but "funny" still \
  means funny. Modifier.
- "A bit funny" — "a bit" changes strength; "funny" still means \
  funny. Modifier.
- "Darkly funny" — "darkly" reshapes "funny" into a tonal compound. \
  NOT a modifier; whole is one trait.
- "Starring Tom Hanks" — "starring" tells the handler this is an \
  actor. Modifier.
- "Around 90 minutes" — "around" calibrates the range; trait is the \
  runtime value. Modifier.
- "Like Inception" — "like" marks a comparison target; the referent \
  is the trait, comparison frame rides along. Modifier.

Modifier flavors (recognize the function, don't memorize a list): \
polarity setters ("not", "without", "avoid"); hedges/emphasis \
("ideally", "a bit", "really", "must", "above all"); role/category \
binding ("starring", "directed by", "about", "set in", "based on", \
"from the studio of"); range/approximation/ordinal ("around", \
"under", "at least", "before", "latest"); comparison/style scope \
("like", "similar to", "in the vein of", "X-style").

The line between a meaning-shaping qualifier ("darkly funny") and a \
strength modifier ("a bit funny") is whether removing the front \
token leaves the trait pointing at the same thing. "Funny" alone \
still points at humor — "a bit" is a modifier. "Funny" alone is not \
the same as "darkly funny" — those are different things.

Common pitfalls: promoting role markers, hedges, or range words to \
traits ("starring", "ideally", "around 90 minutes" never become \
traits of their own); treating meaning-shaping qualifiers as \
modifiers ("darkly funny" is one trait, not "funny" with "darkly"); \
treating every adjective before a noun as a modifier (if it has its \
own category home and names an independent requirement, split — \
"lone female protagonist" → "lone" + "female protagonist").

Examples.
Modifier (absorbed): "not too violent" (polarity+strength on \
violent); "ideally a slow-burn" (hedge); "starring Tom Hanks" (role \
marker); "based on a Stephen King novel" (role marker); "around 90 \
minutes" (range); "in the style of Hitchcock" (comparison scope).
Not a modifier (compound trait): "darkly funny", "iconic twist \
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
  those cases. For everything else, write what the signal actually \
  does — "binds to director credit", "transposes setting to a \
  period", "applies as comparison reference", "narrows to a \
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
- No concrete polarity / salience numbers — describe direction and \
  weight in words.
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


_CARVER_VS_QUALIFIER = """\
CARVER VS QUALIFIER

- Carver: defines what kinds of movies belong in the result set at \
  all. Yes/no test. A movie that fails is excluded, not just ranked \
  lower.
- Qualifier: orders movies within a pool other traits already carved. \
  Continuous test. A low-X movie is still a valid result, ranked \
  below higher-X ones.

Polarity is orthogonal: both can be positive or negative. Negative \
carvers exclude; negative qualifiers downrank.

How to decide:
1. Is this trait qualifying another in the query (narrowing/ranking/\
   steering a pool the other defines)? → qualifier.
2. Is another trait qualifying this one? → this is a carver.
3. Neither qualifying nor being qualified? → carver (must be \
   defining the pool).

Shortcut: what other trait does this one qualify? If you can't \
answer, it can't be a qualifier. This handles "popular movies" / \
"warm-hug movie" naturally — nothing else in the query, so the \
trait must be the carver.

Boundaries:
- Categorical traits (entities, dates, genres, settings, formats) \
  tend to carve — yes/no by nature.
- Gradient traits (mood, tone, popularity) tend to qualify when \
  categoricals are present — continuous by nature.
- Decision is at trait level, not atom level. "Like Eternal Sunshine" \
  decomposes internally into atoms (some categorical-looking) but \
  the trait is one qualifier; internal categoricals don't escape and \
  gate independently.

Common pitfalls:
- Role-flipping on specificity when both are categorical. "Horror \
  movies set on a submarine" — both carve; submarine doesn't \
  demote horror to qualifier.
- Letting an ordinal modifier promote a gradient. "Tarantino's least \
  violent film" — violent is still a gradient qualifier with an \
  ordinal sort on top.
- Confusing negation-as-carving with negation-as-qualifier-polarity. \
  "A non-violent crime thriller" — defining the pool by absence; \
  carving by exclusion. "A rom-com that doesn't feel formulaic" — \
  ranking direction over a continuous trait; negative qualifier. \
  Test: if the trait were positive, would it carve or qualify? Same \
  answer when negated.
- Letting internal atoms of a parametric reference escape. "Like \
  Eternal Sunshine but set in space" — "like Eternal Sunshine" is \
  one qualifier; space is the carver.
- Same trait, different role via companions. "A feel-good film" — \
  feel-good carves alone. "A feel-good comedy" — comedy carves, \
  feel-good qualifies.
- Modifier-binding promoting gradient to categorical. "A quiet \
  movie" reads gradient. "A movie with a quiet score" — "quiet" \
  attached to the score is a categorical fact about a structural \
  element.

Examples.
Carvers: "90s horror starring Anthony Hopkins" (3 positive carvers); \
"a non-violent crime thriller" (positive crime-thriller carver, \
violent negative carver); "a feel-good film" alone (carves); \
"popular movies" alone (carves).
Qualifiers: "funny horror" (horror carves, funny qualifies); "dark \
slow-burn thriller" (thriller carves, dark + slow-burn qualify); "a \
rom-com that doesn't feel formulaic" (rom-com carves, formulaic \
negative qualifier); "Tarantino's least violent film" (Tarantino \
carves, violent qualifies with ordinal); "like Eternal Sunshine but \
set in space" (space carves, like-Eternal-Sunshine qualifies).

---

"""


_POLARITY = """\
POLARITY

Whether the user wants the trait or wants to avoid it.
- Positive: user wants it.
- Negative: user wants to avoid / penalize it.

Surface-grammar rule, not intent inference: if a polarity-setter is \
present, polarity = negative. Don't second-guess what the user \
"really" wants.

Polarity-setters (recognize the function, don't memorize a list): \
"not", "without", "no", "avoid", "skip", "minus", "anything but", \
"spare me", "don't want" — anything reading as "the user is calling \
something out to keep out or push down".

"Not too X" is a special case: polarity=negative, salience=supporting. \
"Not too funny" means it's not a huge deal if somewhat funny, but \
penalize when strongly present. Negative direction, weak strength.

Distribution scope. When a polarity-setter sits at the front of a \
coordinated phrase:
- "Or" tends to distribute. "Not too dark or sad" → both negative.
- "And" / "but" tends to break distribution. "Dark and funny" — only \
  "dark" governed; "but" pivots to a contrasting positive.
Tendencies, not rules. Read like a person.

Boundaries:
- Polarity is mechanical from surface grammar. Don't rewrite "movies \
  that aren't boring" into positive intent for "engaging" — read the \
  surface signal; downstream interprets it.
- Polarity setters absorb as modifiers. "Without violence" → one \
  trait ("violence", polarity=negative).

Common pitfalls: treating "not too X" as full negation ("not too \
long" doesn't mean "must be short"); inferring polarity from intent \
rather than surface grammar; mechanical if-statements for \
distribution.

Examples.
Positive: "funny horror" (funny positive); "a dark, brooding \
thriller" (both positive).
Negative: "not too violent" (violent negative, supporting); "without \
gore" (gore negative); "anything but a romcom" (romcom negative); \
"skip the jump scares" (jump scares negative); "not too dark or \
sad" (both negative, supporting).
Mixed: "a thriller that's tense but not too violent" — tense \
positive central; violent negative supporting.

---

"""


_SALIENCE = """\
SALIENCE

Per-qualifier weight. Two states:
- Central: a headline want; the query feels fundamentally different \
  without it.
- Supporting: meaningful but not load-bearing; rounds out the \
  picture.

Carvers don't get salience (rarity does the weight work downstream). \
Qualifier-only.

Signals, priority order:
1. Hedges ("ideally", "if possible", "maybe", "would be nice", \
   "kind of", "a bit") — primary signal. The user marked it soft; \
   supporting wins even when the trait is structurally prominent.
2. Supporting-language cues in the trait's role read ("rounds out", \
   "sits in service of", "marginal preference") — push toward \
   supporting when no hedge settled the call.
3. Necessity ("must", "need", "have to", "above all", "really \
   want") — central, when no supporting signal is present.
4. Corrective/contrastive ("X but Y") — Y after "but" is often a \
   corrective the user is tracking actively; lean central.
5. Order of mention — earlier qualifiers tend more central.
6. Headline position (clue, not rule) — adjective-slot qualifiers \
   ("slow-burn thriller") tend central; trailing qualifiers ("a \
   thriller, ideally slow-burn") tend supporting. Override-able by \
   hedges and supporting-language cues.

Unifying principle: salience tracks how much investment the user put \
into the trait — words spent, position chosen, hedge or emphasis \
added.

Boundaries:
- Don't force one qualifier per query to be central. Single-qualifier \
  queries get 100% weight regardless. Solo hedged qualifier → \
  supporting.
- Salience is structural-importance, not strength-of-preference. "I \
  really want a comedy" — "really" is intensity but "comedy" is a \
  carver, not qualifier; salience doesn't apply.

Common pitfalls: ignoring hedges because the trait is structurally \
prominent ("ideally a slow-burn thriller" — supporting wins); \
defaulting all qualifiers to central; treating salience as how much \
the user likes the trait.

Examples.
Central: "I really need a slow-burn thriller" (necessity); "above \
all I want it to be funny, and ideally short" (funny central; short \
supporting); "a dark and brooding thriller" (both adjective-slot, \
no hedges, equal-weight central).
Supporting: "ideally a slow-burn thriller" (ideally hedges); "a \
horror movie, maybe with some dark humor" (maybe with some hedges); \
"a thriller, slow-burn would be nice" (would be nice hedges); \
"ideally creepy and atmospheric horror" (both hedged by ideally).

---

"""


def _build_category_taxonomy_section() -> str:
    """Render the category taxonomy into a prompt-ready block.

    Sourced from the CategoryName enum so the prompt text and the
    structured-output vocabulary never drift apart. Compact format:
    one line per field; multi-item fields joined inline rather than
    bulleted, to keep token count down.
    """
    header = (
        "CATEGORY TAXONOMY\n\n"
        "Vocabulary for LATER pipeline steps; loaded here as "
        "context. Step 1 does not commit to a category. Enum values "
        "must match exactly when a downstream step references them.\n\n"
    )
    blocks: list[str] = []
    for cat in CategoryName:
        lines = [f"{cat.name} — {cat.description}"]
        lines.append(f"  Boundary: {cat.boundary}")
        if cat.edge_cases:
            lines.append("  Edge: " + " | ".join(cat.edge_cases))
        if cat.good_examples:
            lines.append(
                "  Good: " + ", ".join(f'"{e}"' for e in cat.good_examples)
            )
        if cat.bad_examples:
            lines.append("  Bad: " + " | ".join(cat.bad_examples))
        blocks.append("\n".join(lines))
    return header + "\n\n".join(blocks)


_CATEGORY_TAXONOMY = _build_category_taxonomy_section()


SYSTEM_PROMPT = (
    _TASK_FRAMING
    + _ATOMICITY
    + _MODIFIER_VS_TRAIT
    + _EVALUATIVE_INTENT
    + _CARVER_VS_QUALIFIER
    + _POLARITY
    + _SALIENCE
    + _CATEGORY_TAXONOMY
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
