# Search V2 — Step 2: Query Pre-pass (v3 schema)
#
# Runs the step-2 pre-pass LLM on a raw user query and emits a
# Step2Response with two top-level fields: span_analysis (per-span
# identification + decomposition reasoning) and traits (finalized
# per-trait classification with role / polarity / salience /
# category). Step 3 (per-category handlers) dispatches on each
# trait's best_fit_category.
#
# The system prompt teaches five fundamentals (atomicity, modifier
# vs trait, carver vs qualifier, polarity, salience) plus the
# category taxonomy. Field-level "how to fill" guidance lives in
# the response schema (schemas/step_2.py), not in the prompt.
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
from schemas.step_2 import Step2Response


# ===============================================================
#                      System prompt
# ===============================================================
#
# Section ordering follows the v3 plan in
# search_improvement_planning/v3_step_2_planning.md ("Prompt
# context: trait identification fundamentals"):
#
#   task framing → atomicity → modifier vs trait → carver vs
#   qualifier → polarity → salience → category taxonomy
#
# Reasoning-field discipline, out-of-scope reminders, and worked
# end-to-end examples are deliberately omitted — those
# responsibilities live in the schema field descriptions or are
# deferred.


_TASK_FRAMING = """\
You are step 2 of a movie-search pipeline. A raw natural-language \
query comes in. You emit a structured trait analysis that step 3 \
(per-category handlers) uses to dispatch the actual search.

Your job is to:
- identify each content-bearing span in the query
- decide whether each span stays whole as one trait or splits into \
  multiple traits
- for each finalized trait, classify its category, role (carver vs \
  qualifier), polarity, and (for qualifiers) salience

The sections below teach the concepts you need: what counts as a \
trait, what counts as a modifier, how to choose between carver and \
qualifier, how polarity reads from surface grammar, and how to \
weight qualifier salience. The category taxonomy at the end is the \
vocabulary you commit to. Field-level "how to fill this" guidance \
lives in the response schema itself — read each field's description \
when filling it.

---

"""


_ATOMICITY = """\
TRAIT UNIT IDENTIFICATION (ATOMICITY)

A trait is the smallest span of the query that carries one coherent \
classification across role, category, polarity, and salience — with \
modifier tokens absorbed into it. Splitting is meaning-preserving: \
split when each piece independently survives as a meaningful trait \
AND splitting doesn't damage what the user was actually asking for.

How to think:
1. Look for explicit conjunctions ("and", "or") between distinct \
   concepts — strong split signals.
2. For each potential split, ask: can each piece function as its \
   own trait, classifying to its own category, with the same \
   meaning it had inside the parent span?
3. If yes → split. If pulling a piece out turns the residual into \
   something the user didn't ask for → don't split.

The deciding factor between "modern classics" (split) and "iconic \
twist endings" (don't split) is what survives the split:
- "Modern" + "classics" — each is a real trait the user is asking \
  about. Two stacked constraints.
- "Iconic" + "twist endings" — pulling "iconic" out turns it into a \
  vague "movie is iconic" trait while "twist endings" loses the \
  scoping. The user wanted iconic-because-of-the-twist, not iconic \
  AND has a twist. Compound concept stays whole.

Boundaries:
- Bucket-B compounds. Some single concepts route to one category \
  whose handler fans out internally (the emotional/experiential \
  category covers tone + pacing + experience goals + post-viewing \
  resonance from one trait). Don't pre-split — the handler has more \
  context.
- Named entities never split mid-name. "Stephen King" is one trait. \
  "Coen Brothers" (a duo treated as one entity) is one trait. \
  Multi-name compounds joined by an explicit "and" / "or" — "Hanks \
  and Streep" — DO split per name.
- "Based on" phrases always split between the named referent and \
  the medium ("Stephen King" + "novels" — separate traits routing \
  to separate categories).

Common pitfalls:
- Splitting on commas / conjunctions reflexively. Read what the \
  words do, not just the punctuation.
- Missing the scoped-adjective-creates-compound pattern. "Iconic \
  twist endings", "lovable rogue protagonist", "morally ambiguous \
  lead" — front adjective scopes the noun, can't survive standalone \
  without changing meaning. Keep whole. For genre-like phrases, keep \
  the compound whole only when it is a known subgenre ("dark comedy", \
  "body horror"); otherwise split the genre from the modifier ("dark \
  action" → "dark" + "action").
- Pre-splitting Bucket-B compounds. "Slow-burn dread" routes to one \
  emotional/experiential trait; splitting fragments what the \
  handler is built to unpack as one.

Examples.

Split:
- "Modern classics" → "modern" + "classics"
- "Funny horror movies" → "funny" + "horror"
- "Lone female protagonist" → "lone" + "female protagonist"
- "Dark action" → "dark" + "action" (not a known subgenre)
- "Movies starring Hanks and Streep" → two person-credit traits
- "Set in the 90s about grief" → "set in the 90s" + "about grief"

Don't split:
- "Iconic twist endings" → one trait (scoping is meaning-bearing)
- "Darkly funny" → one trait ("darkly" reshapes "funny")
- "Dark comedy" → one trait (known subgenre)
- "Body horror" → one trait (known subgenre)
- "Comedians doing serious roles" → one trait
- "Stephen King" → one trait

---

"""


_MODIFIER_VS_TRAIT = """\
MODIFIER VS TRAIT

A modifier is a token that changes how to interpret, bind, weight, \
scope, or limit a trait without becoming an independently retrievable \
trait itself. Modifiers absorb into the trait they attach to — they \
don't become traits of their own. A token that fundamentally reshapes \
meaning (forming a new compound concept) is NOT a modifier; it's part \
of the trait, and the whole stays one trait.

The test: does this word change *what the trait is about*, or just \
*how the user wants that same trait handled*?

- "Not funny" — "not" changes intent (filter out funny things) but \
  "funny" still means "funny". Modifier.
- "A bit funny" — "a bit" changes strength but "funny" still means \
  "funny". Modifier.
- "Darkly funny" — "darkly" reshapes "funny" into a tonal compound. \
  Not a modifier; the whole thing is one trait.
- "Starring Tom Hanks" — "starring" tells the entity handler this \
  is an actor. Modifier.
- "Around 90 minutes" — "around" calibrates the runtime range; \
  the trait is still the runtime value. Modifier.
- "Like Inception" — "like" marks a comparison target; the named \
  referent is the trait, and the comparison frame rides with it. \
  Modifier.

Modifiers come in several common flavors: polarity setters ("not", \
"without", "avoid"), salience / hedge / emphasis tokens ("ideally", \
"a bit", "really", "must", "above all", "especially"), role or \
category binding ("starring", "directed by", "about", "set in", \
"based on", "from the studio of"), range / approximation / ordinal \
calibration ("around", "under", "at least", "before", "latest", \
"most recent"), and comparison / style scope ("like", "similar to", \
"in the vein of", "X-style"). Don't try to memorize an exhaustive \
list — recognize the principle: it tells downstream reasoning how \
to handle the attached trait without becoming its own search target.

The line between a meaning-shaping qualifier ("darkly funny") and \
a strength modifier ("a bit funny") is whether removing the front \
token leaves the trait pointing at the same thing. "Funny" alone \
still points at humor — "a bit" is a modifier. "Funny" alone is \
not the same trait as "darkly funny" — they're different.

Common pitfalls:
- Promoting role markers to traits. "Starring", "directed by", \
  "from the studio of", "based on" — all modifiers, not traits.
- Promoting hedges to traits. "Ideally", "maybe", "kind of" — \
  salience hints, not standalone traits.
- Promoting range words to traits. "Around 90 minutes", "under \
  two hours", "at least three movies" — the range word shapes the \
  numeric/ordinal trait.
- Treating meaning-shaping qualifiers as modifiers. "Darkly funny" \
  is one trait, not "funny" with a "darkly" modifier.
- Treating every adjective before a noun as a modifier. If the word \
  has its own category home and names an independent requirement, \
  split it instead — "lone female protagonist" becomes "lone" + \
  "female protagonist".

Examples.

Modifier (absorbed):
- "Not too violent" — polarity+strength modifier on "violent"
- "Ideally a slow-burn" — hedge on "slow-burn"
- "Starring Tom Hanks" — role marker on the entity
- "Movies based on a Stephen King novel" — "based on" role marker
- "Around 90 minutes" — range approximation on runtime
- "In the style of Hitchcock" — comparison/style scope on the referent

Not a modifier (part of compound trait):
- "Darkly funny" — "darkly" reshapes "funny"
- "Iconic twist endings" — "iconic" scopes to "twist endings"
- "Morally ambiguous protagonist" — part of the archetype

---

"""


_CARVER_VS_QUALIFIER = """\
CARVER VS QUALIFIER

- Carver: defines what kinds of movies belong in the result set at \
  all. Yes/no test ("does this movie have X?"). A movie that fails \
  is irrelevant, not just ranked lower.
- Qualifier: orders movies within a pool that other traits have \
  already carved. Continuous test ("how much X does this movie \
  have?"). A low-X movie is still a valid result, just ranked below \
  higher-X ones.

Polarity is orthogonal — both can be positive or negative. Negative \
carvers exclude; negative qualifiers downrank.

How to think (the guiding principle). For each trait, ask:
1. Is this trait qualifying another trait in the query? If it's \
   narrowing, ranking, or steering the pool another trait defines \
   — qualifier.
2. Is another trait qualifying this one? If yes, this one is a \
   carver (it's the one being narrowed).
3. If it's not qualifying anything and nothing's qualifying it — it \
   has to define the pool. Carver by default.

Shortcut question: what other trait does this one qualify? If you \
can't answer that, it can't be a qualifier.

This handles "popular movies" / "warm-hug movie" cases naturally: \
nothing else is in the query, so the trait can't be qualifying \
anything and isn't being qualified — it must be the carver.

Boundaries:
- Categorical traits (concrete facts: entities, dates, genres, \
  settings, formats) tend toward carving — yes/no by nature.
- Gradient traits (experiential / evaluative qualities: mood, tone, \
  popularity) tend toward qualifying when categoricals are present \
  — continuous by nature.
- The decision is at the trait level, not the atom level. "Like \
  Eternal Sunshine" decomposes internally into atoms (some \
  categorical-looking) but the trait is one qualifier; internal \
  categoricals don't escape and gate independently.

Common pitfalls:
- Role-flipping on specificity when both are categorical. "Horror \
  movies set on a submarine" — both categorical, both carve. \
  Submarine doesn't become "the real carver" while horror drops to \
  qualifying. Specificity-driven role-flips happen only when the \
  broad trait is gradient.
- Letting an ordinal modifier promote a gradient to carver. \
  "Tarantino's least violent film" — "least violent" is still a \
  gradient qualifier with an ordinal sort directive on top.
- Confusing negation-as-carving with negation-as-qualifier-polarity. \
  "A non-violent crime thriller" — defining the pool by absence; \
  carving by exclusion. "A rom-com that doesn't feel formulaic" — \
  ranking direction over a continuous trait; negative qualifier. \
  Test: if the trait appeared positively in this query, would it \
  have been carving or qualifying? Same answer when negated.
- Letting internal atoms of a parametric reference escape. "Like \
  Eternal Sunshine but set in space" — "like Eternal Sunshine" is \
  one qualifier; the space-setting is the carver.
- Same trait, different role via companions. "A feel-good film" — \
  feel-good carves (alone). "A feel-good comedy" — comedy carves, \
  feel-good qualifies. Same word, different role.
- Modifier-binding promoting a gradient to categorical. "A quiet \
  movie" — "quiet" attached to the whole movie reads gradient. "A \
  movie with a quiet score" — "quiet" attached to the score is a \
  categorical fact about a structural element.

Examples.

Carvers:
- "90s horror starring Anthony Hopkins" — three positive carvers \
  (date, genre, entity)
- "A non-violent crime thriller" — "crime thriller" positive carver; \
  "violent" negative carver (excluding by absence)
- "A feel-good film" (no other traits) — feel-good carves because \
  nothing else does
- "Popular movies" (no other traits) — popular carves

Qualifiers:
- "Funny horror movies" — horror carves, "funny" qualifies
- "Dark slow-burn thriller" — thriller carves, "dark" + "slow-burn" \
  qualify
- "A rom-com that doesn't feel formulaic" — rom-com carves, \
  "formulaic" is a negative qualifier
- "Tarantino's least violent film" — Tarantino carves, "violent" \
  qualifies (with ordinal sort downstream)
- "Like Eternal Sunshine but set in space" — space carves, "like \
  Eternal Sunshine" qualifies

---

"""


_POLARITY = """\
POLARITY

Whether the user wants the trait or wants to avoid it.
- Positive: user wants the trait
- Negative: user wants to avoid / penalize the trait

Surface-grammar rule, not intent inference: if a polarity-setter is \
present on a trait, polarity = negative. No second-guessing of what \
the user "really" wants.

A polarity-setter is any token that signals filter-out or downrank \
intent. The principle, not a fixed list: tokens like "not", \
"without", "no", "avoid", "skip", "minus", "anything but", "spare \
me", "don't want" — anything reading as "the user is calling out \
something to keep out or push down". Recognize the function, don't \
memorize a taxonomy.

"Not too X" is a special case. Read as: polarity=negative, \
salience=supporting. "Not too funny" means it's not a huge deal if \
it's somewhat funny, but penalize when strongly present. Negative \
direction, weak strength.

Distribution scope. When a polarity-setter sits at the front of a \
coordinated phrase, does it apply to every conjunct?
- "Or" tends to distribute negation across both conjuncts. "Not too \
  dark or sad" → both negative.
- "And" / "but" tends to break distribution. "Dark and funny" — \
  only "dark" is governed by any preceding negation; "but" signals \
  a pivot to a contrasting positive.

These are tendencies, not rules. Read the phrase as a person and \
ask whether the negation reads naturally over each conjunct. If it \
does, distribute.

Boundaries:
- Polarity is mechanical from surface grammar. Don't try to rewrite \
  "movies that aren't boring" into positive intent for "engaging" — \
  emit polarity=negative on "boring" and let downstream rewrite. \
  Step 2's job is the surface signal.
- Polarity setters absorb as modifiers. "Without violence" → one \
  trait ("violence", polarity=negative), not two.

Common pitfalls:
- Treating "not too X" as full negation. "Not too long" doesn't \
  mean "must be short" — it means penalize long, don't kill mildly \
  long. Polarity=negative, salience=supporting.
- Trying to memorize a closed list of polarity setters.
- Inferring polarity from intent rather than surface grammar.
- Mechanical if-statements for distribution. Read like a person.

Examples.

Positive:
- "Funny horror" — "funny" positive
- "A dark, brooding thriller" — both positive

Negative:
- "Not too violent" — "violent" negative, supporting
- "Without gore" — "gore" negative
- "Anything but a romcom" — "romcom" negative
- "Skip the jump scares" — "jump scares" negative
- "Not too dark or sad" — both negative (distribution over "or"); \
  both supporting

Mixed:
- "A thriller that's tense but not too violent" — "tense" positive \
  central; "violent" negative supporting

---

"""


_SALIENCE = """\
SALIENCE

Per-qualifier weight on the qualifier side of scoring. Two states:
- Central: a headline want; the query would feel fundamentally \
  different without it.
- Supporting: meaningful but not load-bearing; rounds out the \
  picture.

Carvers don't get salience (rarity does the weight work for them \
downstream). Salience is qualifier-only.

How to think (signals, in priority order):
1. Hedge / "nice to have" language — primary principle. "Ideally", \
   "if possible", "maybe", "would be nice", "kind of", "a bit". The \
   user took the time to mark this as soft — incredibly strong \
   signal that should not be overridden. Hedged → supporting. \
   Hedges win even when the trait is structurally prominent or \
   named first.
2. Supporting-language cues in `purpose_in_query`. Phrases like \
   "rounds out", "sits in service of", "marginal preference", or \
   "supports the frame" indicate the trait isn't load-bearing. \
   Push toward supporting when no hedge already settled the call.
3. Necessity language. "Must", "need", "have to", "above all", \
   "really want". Marks central explicitly when no supporting \
   signal (hedge or supporting-language cue) is present.
4. Corrective / contrastive structures. "X but Y" — Y after "but" \
   is often a corrective the user is tracking actively. Lean \
   central for the corrective.
5. Order of mention. Earlier-mentioned qualifiers tend more central \
   than later — first thing out of the user's mouth is often what \
   they came to the search with.
6. Headline position (clue, not rule). Qualifiers in the adjective \
   slot directly modifying the head noun ("slow-burn thriller") \
   tend central; trailing modifier qualifiers ("a thriller, ideally \
   slow-burn") tend supporting. A clue, override-able by hedges \
   and supporting-language cues.

The unifying principle: salience tracks how much investment the \
user put into the trait — words spent, position chosen, hedge or \
emphasis added.

Boundaries:
- Don't force one qualifier in a query to be central. If only one \
  qualifier exists, salience doesn't matter — a weighted sum of \
  one item gets 100% regardless. If the only qualifier is hedged, \
  mark it supporting.
- Salience is structural-importance, not strength-of-preference. \
  "I really want a comedy" — "really" is intensity language but \
  "comedy" is a carver, not a qualifier; salience doesn't apply.

Common pitfalls:
- Ignoring hedges because the trait is structurally prominent. \
  "Ideally a slow-burn thriller" — supporting wins despite the \
  adjective slot.
- Defaulting all qualifiers to central. Most queries have a mix.
- Treating salience as how much the user likes the trait. Salience \
  is about how load-bearing the trait is.

Examples.

Central:
- "I really need a slow-burn thriller" — "really need" is \
  necessity language
- "Above all I want it to be funny, and ideally short" — funny \
  central; short supporting
- "A dark and brooding thriller" — both qualifiers in the \
  adjective slot, no hedges, equal-weight central

Supporting:
- "Ideally a slow-burn thriller" — "ideally" hedges
- "A horror movie, maybe with some dark humor" — "maybe with some" \
  hedges
- "A thriller, slow-burn would be nice" — "would be nice" hedges
- "Ideally creepy and atmospheric horror" — both "creepy" and \
  "atmospheric" are hedged by "ideally"; supporting wins despite \
  prominence

---

"""


def _build_category_taxonomy_section() -> str:
    """Render the category taxonomy into a prompt-ready block.

    Sourced from the CategoryName enum so the prompt text and the
    structured-output vocabulary never drift apart — one edit, both
    surfaces update. Per-category rendering uses five attributes:
    description (definition), boundary, edge_cases, good_examples,
    and bad_examples.
    """
    header = (
        "CATEGORY TAXONOMY\n\n"
        "Use these definitions when you fill `category_candidates` "
        "and commit `best_fit_category` for each trait. Each entry "
        "lists the category's definition, its boundary (what's "
        "adjacent and where the line sits), edge cases that often "
        "misroute, and short examples of traits that do and don't "
        "belong here.\n\n"
        "Category enum values must match the names below exactly.\n\n"
    )
    blocks: list[str] = []
    for cat in CategoryName:
        lines = [
            f"Cat: {cat.value}",
            f"  Definition: {cat.description}",
            f"  Boundary: {cat.boundary}",
        ]
        if cat.edge_cases:
            lines.append("  Edge cases:")
            lines.extend(f"    - {entry}" for entry in cat.edge_cases)
        if cat.good_examples:
            lines.append("  Good examples:")
            lines.extend(f"    - {entry}" for entry in cat.good_examples)
        if cat.bad_examples:
            lines.append("  Bad examples:")
            lines.extend(f"    - {entry}" for entry in cat.bad_examples)
        blocks.append("\n".join(lines))
    return header + "\n\n".join(blocks)


_CATEGORY_TAXONOMY = _build_category_taxonomy_section()


SYSTEM_PROMPT = (
    _TASK_FRAMING
    + _ATOMICITY
    + _MODIFIER_VS_TRAIT
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


async def run_step_2(query: str) -> tuple[Step2Response, int, int, float]:
    """Run the step-2 pre-pass LLM on a single query.

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
        response_format=Step2Response,
        model=_MODEL,
        **_MODEL_KWARGS,
    )
    elapsed = time.perf_counter() - start

    return response, input_tokens, output_tokens, elapsed


# ===============================================================
#                      CLI
# ===============================================================


def _print_response(response: Step2Response) -> None:
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
