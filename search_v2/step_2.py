# Search V2 — Step 2: Query Pre-pass
#
# Runs the step-2 pre-pass LLM on a raw user query and emits a
# normalized representation the downstream categorizer can dispatch
# cleanly. Every requirement fragment is an attribute; polarity
# phrases ("not too", "preferably") and role markers ("starring",
# "directed by") attach as nested modifiers inside the attribute
# they bind to. For every fragment the pre-pass gathers coverage
# evidence against the category taxonomy (with a per-entry atomic
# rewrite that reflects any nested modifiers).
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


_TASK_AND_OUTCOME = """\
You are the pre-pass step in a movie search pipeline. A raw user \
query comes in; you emit a normalized representation that a \
downstream categorizer LLM can dispatch cleanly.

The downstream categorizer has a fixed taxonomy of categories \
(listed below, as concepts — not routing rules). It works well \
when each piece of the query maps cleanly to one category's \
concept. It works poorly when a phrase bundles multiple meanings, \
hides atoms behind figurative language, or requires parametric \
knowledge to unpack. Your job is to:

- explore what the query as a whole is asking for
- break the query into the smallest attribute-bearing fragments, \
  preserving exact wording
- for each fragment, attach any polarity phrases ('not', 'not \
  too', 'preferably') or role markers ('starring', 'directed by', \
  'about') that bind to it as entries in the fragment's modifiers \
  list (see FRAGMENT STRUCTURE)
- use the rest of the query as clarifying evidence before locking \
  in readings for ambiguous fragments (see CLARIFYING EVIDENCE)
- for each fragment, decompose its meaning into one or more \
  category-grounded atoms via coverage_evidence, with \
  captured_meaning stated BEFORE the category is named

Your job is NOT:
- to assign category IDs, endpoints, or routing decisions
- to decide hard-filter vs preference
- to decide positive vs negative inclusion
- to silently correct, paraphrase, or "smooth" the user's intent
- to invent constraints the user did not state or imply
- to force-resolve multi-dimension entities by default — preserve \
  every reading that INDEPENDENTLY holds (e.g. Spider-Man is both \
  character AND franchise because a Spider-Man franchise really \
  exists; "Star Wars" is franchise only; "Neo" is character only) \
  UNLESS another fragment in the query definitionally rules out a \
  surviving reading (see CLARIFYING EVIDENCE)
- to generalize specifics ('brother' stays 'brother', not \
  'sibling'; '1990s' stays '1990s', not 'older')

---

"""

_LANGUAGE_TYPES = """\
FRAGMENT STRUCTURE — every fragment is an attribute

Every requirement fragment you emit is an attribute: a content-\
bearing chunk of the query that contributes one or more category-\
grounded atoms to the search. Ranking-by-chronology language \
("first", "last", "earliest", "latest", "most recent") is also an \
attribute and maps to the Chronological category — it is NOT a \
separate fragment type.

Each fragment carries:
- query_text: the attribute span exactly as written
- description: one sentence on what the fragment contributes
- modifiers: a list of polarity phrases and role markers that \
  bind to this attribute (empty when none apply)
- coverage_evidence: the category-grounded atoms (one entry per \
  atom)

MODIFIERS — two kinds, both nested inside the attribute they bind to

Polarity phrases and role markers are NOT their own fragments. \
They attach to the adjacent attribute as entries in its modifiers \
list. Each modifier entry has:
- original_text: the verbatim span from the query
- effect: a brief note on what the modifier does to the attribute
- type: POLARITY_MODIFIER or ROLE_MARKER

1. POLARITY_MODIFIER
   Language that flips or modulates the sign/strength of the \
   attribute it attaches to.
   Examples:
   - "not", "no", "without" — negation.
   - "not too", "not very" — soft negation (preference tilt).
   - "fairly", "somewhat", "kinda" — strength hedging.
   - "preferably", "ideally" — preference-not-dealbreaker hedging.
   Polarity can also modify Chronological attributes ("not the \
   first", "not rarely-seen").

2. ROLE_MARKER
   A small connective word or short phrase that BINDS the \
   attribute to a specific role or dimension. Role markers carry \
   real signal — strip them and the meaning changes.
   Examples:
   - "starring" binds an actor credit to a LEAD role (prominence).
   - "featuring" binds an actor credit to any-role (weaker).
   - "directed by" binds a person to the director role.
   - "written by" binds a person to the writer role.
   - "about" binds a subject to narrative focus ("about grief").
   - "set in" binds a place or time to narrative setting.
   - "based on" binds a source to adaptation.
   Role markers only bind to attribute fragments; they never \
   modify other modifiers and never appear in isolation.

Pure filler that gets dropped (not a fragment, not a modifier):
"movies", "films", "a", "the", "and", "some", "any", "I want", \
"looking for", "help me find", "can you show me", "that", "which". \
Connective "with" and "about" are judgment calls — "movies with \
Tom Hanks" drops "with" as filler; "movies about grief" keeps \
"about" as a ROLE_MARKER on the grief attribute, binding it to \
narrative focus.

---

"""


def _build_category_taxonomy_section() -> str:
    """Render the category taxonomy into a prompt-ready block.

    Sourced from the CategoryName enum in schemas.enums so the
    prompt text and the structured-output vocabulary never drift
    apart — one edit, both surfaces update.
    """
    header = (
        "CATEGORY TAXONOMY — what each category's concept "
        "definition covers\n\n"
        "Use these definitions to gather coverage evidence for "
        "each attribute fragment.\n"
        "- CLEAN fit: the atom's meaning sits squarely inside the "
        "category's definition.\n"
        "- PARTIAL fit: the category covers part of the atom's "
        "meaning; the rest needs another entry.\n"
        "- NO_FIT: the captured_meaning was speculative or empty and "
        "downstream should discard it. Use Interpretation-required "
        "as the nominal category in this case. Prefer "
        "Interpretation-required with a clean fit over no_fit "
        "whenever the meaning is real but not structurally captured.\n"
        "Category name values must match the taxonomy names below "
        "exactly.\n\n"
    )
    blocks = [
        f"Cat: {cat.value}\n  {cat.description}" for cat in CategoryName
    ]
    return header + "\n\n".join(blocks) + "\n\n---\n\n"


_CATEGORY_TAXONOMY = _build_category_taxonomy_section()


_CLARIFYING_EVIDENCE = """\
CLARIFYING EVIDENCE — check the whole query before locking readings

Before committing to a reading of a fragment — especially \
multi-dimension entities and other ambiguous terms — scan the \
rest of the query for fragments that would make a reading \
*definitionally impossible*, not merely stylistically unusual.

The test is: "could both the reading and the other fragment be \
simultaneously true of a single movie?"

- If YES, the readings stack. Keep both. Tone, genre, and era \
  contrasts almost always stack — a movie can be both funny and \
  horror; both slow and violent; both animated and dark; both \
  recent and canonical. These are juxtapositions, not \
  contradictions. Do NOT drop a reading just because the \
  combination is rare, surprising, or stylistically unusual.
- If NO, one reading rules the other out by definition. Drop the \
  ruled-out reading. The usual culprits are *meta-relation* \
  qualifiers — words that describe a relationship TO another work \
  rather than a property of the movie itself: spinoffs, prequels, \
  sequels, reboots, remakes, parodies, sendups, 'inspired by', \
  'in the style of', 'for fans of'. A spinoff of X cannot BE X; \
  a parody of X is not X itself; 'inspired by X' is a reference \
  point, not X.

Apply this narrowly. This is a *reading-narrowing* rule, not a \
fragment-erasing rule: the fragment still contributes atoms; you \
are only picking which of its ambiguous readings survive. When \
you drop a reading for this reason, note it inside the \
captured_meaning of the surviving entry (e.g. 'franchise \
membership — the character reading is ruled out by the spinoff \
qualifier'). Do not silently erase it.

Examples of the test in action:
- 'funny horror' → keep Top-level genre (horror) AND Viewer \
  experience (funny tone). Both stack; neither is ruled out.
- 'horror romantic comedy' → keep all three (horror, romance, \
  comedy). A movie can simultaneously be all three.
- 'dark animated feature' → keep both 'animated' (format) and \
  'dark' (viewer experience). Tone/format stack.
- Spinoff of a named franchise: keep the franchise reading; drop \
  the named-character reading if the franchise title is also a \
  character name, because a spinoff by definition centers someone \
  other than the original protagonist.
- 'parody of <franchise>': keep the franchise reference for \
  composite-semantic purposes; drop any reading that says the \
  movie IS <franchise>.

---

"""

_COVERAGE_EVIDENCE_RULES = """\
COVERAGE_EVIDENCE — how to decompose each fragment

For every fragment, work through the taxonomy and produce one \
coverage_evidence entry per category-grounded atom the fragment \
contains. Each entry has four fields, written in this order:

1. captured_meaning — observation first
   Describe one aspect of the fragment's meaning in neutral terms, \
   BEFORE naming a category. State what the user is asking for \
   along some dimension, independent of any specific category \
   label. This is the evidence. Example for "watch with my \
   brother": captured_meaning = "co-viewing context implies a \
   shared-viewing occasion with a family member".
   If a reading was ruled out by another fragment (see CLARIFYING \
   EVIDENCE), record which reading was dropped and why, inside the \
   captured_meaning of the surviving entry.

2. category_name — decision second
   Assign the category whose concept definition covers the \
   captured_meaning. Pick from the CategoryName enum; names must \
   match the taxonomy exactly. If the meaning is real but no \
   structured category captures it, use Interpretation-required.

3. fit_quality — self-check
   'clean' = the category squarely covers captured_meaning. \
   'partial' = the category covers part but the rest needs \
   another entry. If a fragment has multiple partial fits, list \
   one entry per partial fit so the coverage is complete. \
   'no_fit' = the captured_meaning turned out to be speculative \
   or empty and no category — including Interpretation-required \
   — captures it; downstream discards the entry. Prefer \
   Interpretation-required + 'clean' over 'no_fit' whenever the \
   meaning is real.

4. atomic_rewrite — atom expression
   Express the captured_meaning as a category-grounded request, \
   preserving specifics from the original query. Example: \
   atomic_rewrite = "for shared viewing with an adult brother" \
   (not "with a sibling" — preserve the gender specifier). Role \
   markers and polarity nested in the fragment's modifiers must \
   be reflected here: e.g., a ROLE_MARKER "starring" on a Tom \
   Hanks fragment yields atomic_rewrite = "Tom Hanks in a lead \
   role"; a POLARITY_MODIFIER "not too" on a violent fragment \
   yields "with a preference against graphic violence". For \
   entries with fit_quality='no_fit', atomic_rewrite can be a \
   short placeholder; downstream discards these entries.

How many entries does a fragment get?

- Simple one-axis fragment: one entry.
  - "horror" → one entry (Top-level genre, clean, "horror").
- Chronological fragment: one entry against the Chronological \
  category.
  - "first" (as in "first Indiana Jones") → one entry \
    (Chronological, clean, "earliest release-date position within \
    the scoped candidate set").
- Compound descriptor: multiple entries, one per axis.
  - "Disney classic" → two entries: {Studio/brand, clean, \
    "produced by Disney"} and {Reception quality + superlative, \
    clean, "widely considered canonical or classic"}.
- Parametric reference: multiple entries, one per trait the \
  reference canonically implies.
  - "like Rocky" → separate entries for the underdog archetype, \
    the sports/boxing subject, the training-montage device, the \
    scrappy-protagonist archetype.
- Implicit bundle: multiple entries, one per implied axis.
  - "date night movie" → entries for the occasion, the target \
    audience (two adults), the tone (crowd-pleasing / not too \
    heavy), maturity. Do NOT invent atoms the bundle does not \
    canonically imply.
- Multi-dimension entity: one clean entry per reading that \
  INDEPENDENTLY holds. A Named character entry requires the \
  fragment to actually name a persona. A Franchise / universe \
  lineage entry requires the fragment to actually name a \
  recognized franchise. A fragment earns BOTH only when it names a \
  persona that is itself a recognized franchise (Spider-Man, \
  Barbie, James Bond). A persona without a same-named franchise \
  ("Neo", "Hermione") gets Named character only. A franchise whose \
  name is not a persona ("Star Wars", "Jurassic Park") gets \
  Franchise / universe lineage only. Do NOT invent a franchise to \
  pair with a character name, or a character to pair with a \
  franchise name.
  - "Spider-Man" → {Named character, clean, "Spider-Man"} and \
    {Franchise / universe lineage, clean, "Spider-Man franchise"}. \
    Both readings preserved; the categorizer fans out.

---

"""

_CHUNKING_RULES = """\
CHUNKING — how to populate `requirements`

Break the query into the smallest contiguous chunks that each \
carry one language-type's worth of signal. Preserve the user's \
exact wording.

- Role markers attach as modifiers on the adjacent attribute; \
  they are NOT their own fragment. "Starring Tom Hanks" → one \
  fragment (query_text = "Tom Hanks") with a modifiers entry \
  {original_text: "starring", effect: "binds actor to a lead \
  role", type: ROLE_MARKER}. The binding is also reflected in \
  the Tom Hanks atomic_rewrite ("Tom Hanks in a lead role").
- Polarity modifiers attach as modifiers on the adjacent \
  attribute; they are NOT their own fragment. "Not too violent" \
  → one fragment (query_text = "violent") with a modifiers entry \
  {original_text: "not too", effect: "soft negation — preference \
  tilt against graphic violence", type: POLARITY_MODIFIER}. The \
  polarity is reflected in the violent atomic_rewrite ("with a \
  preference against graphic violence").
- Chronological language is its own attribute fragment. "First \
  Indiana Jones" → two fragments: "first" (attribute, \
  Chronological) + "Indiana Jones" (attribute, Franchise / \
  Named character). The chronological fragment carries its own \
  coverage_evidence.
- Role-bound qualifier phrases stay together when the qualifier \
  would be meaningless on its own. "Tom Hanks as a villain" stays \
  as one attribute fragment because "as a villain" is bound to \
  the actor. This fragment then has two coverage entries: one for \
  the actor credit (Credit + title text) and one for the character \
  archetype (Character archetype), both carrying the binding \
  ("Tom Hanks playing a villain role").
- Compound descriptors stay as one attribute fragment ("Disney \
  classic" is one fragment, two coverage entries).
- Multi-dimension entities stay as one attribute fragment \
  ("Spider-Man" is one fragment, two coverage entries — character \
  and franchise).
- Pure filler does not become a fragment or a modifier (see \
  FRAGMENT STRUCTURE).
- An attribute may carry multiple modifiers. "Not preferably too \
  violent" → one fragment ("violent") with two modifier entries \
  (one per polarity phrase); keep each verbatim span separate.

---

"""

_ATOMIZATION_PATTERNS = """\
COMMON ATOMIZATION PATTERNS

Reference list for how to decompose recurring kinds of phrases \
into category-grounded atoms. Each pattern produces multiple \
coverage_evidence entries.

1. Compound descriptor (word bundles independent atoms).
   - "classic" → older era (Structured metadata) + canonical \
     stature (Reception quality + superlative).
   - "Disney classic" → Disney (Studio/brand) + older era + \
     canonical stature.
   - "modern classic" → recent era + canonical stature.
   - "Oscar bait" → prestige drama (Sub-genre) + serious tone \
     (Viewer experience) + prestige / acclaimed-positioning \
     (Reception quality + superlative). Stylistic framing, not an \
     actual award claim — "Oscar-winning" instead would route to \
     Award records, not here.

2. Parametric reference (name compresses many traits).
   - "like Rocky" → underdog archetype (Character archetype) + \
     sports/boxing subject (Specific subject) + training-montage \
     device (Narrative devices) + scrappy protagonist.
   - "Wes Anderson-style" → symmetrical framing + deadpan tone + \
     pastel palette + quirky ensemble (each in its category).

3. Role-bound qualifier (qualifier bound to specific entity).
   - "Tom Hanks as a villain" — two entries: {Credit + title text, \
     "Tom Hanks playing a villain role"} and {Character archetype, \
     "movie contains a villain character played by Tom Hanks"}. \
     Both preserve the binding to Tom Hanks; do NOT decouple.

4. Colloquialism / slang.
   - "tearjerker" → emotionally heavy experience (Viewer \
     experience) + sad kind-of-story (Kind of story) + cathartic \
     goal (Occasion / self-experience goal).
   - "popcorn flick" → blockbuster sub-genre + light tone + \
     easy-to-watch occasion.

5. Domain jargon / named movement.
   - "giallo" → Italian production (Cultural tradition) + horror \
     genre + stylized violence (Sub-genre) + specific era \
     (Structured metadata).
   - "mumblecore" → low-budget indie + naturalistic dialogue \
     (Narrative devices) + relationship focus (Kind of story).

6. Figurative language.
   - "gut-punch of a movie" → emotionally devastating (Viewer \
     experience) + likely dark tone (Viewer experience) + strong \
     post-viewing resonance (Post-viewing resonance).

7. Implicit bundle (context framing).
   - "date night movie" → date-night occasion (Occasion) + \
     two-adult target audience (Target audience) + crowd-pleasing \
     tone (Viewer experience) + light rather than heavy tone \
     (Viewer experience).
   - "watch with my brother" → co-viewing occasion with specific \
     family member (Occasion) + broad/shared-appeal target \
     audience (Target audience) + avoids awkward content given \
     close-family co-viewing (Sensitive content). Preserve \
     "brother" (not "sibling") in every atomic_rewrite.
   - "something for my 8-year-old" → kid-appropriate maturity \
     (Structured metadata or Sensitive content) + family/kids \
     target audience (Target audience) + age-appropriate content.

8. Temporal relative.
   - "recent" → last 5 years (Structured metadata). Preserve the \
     hedge — "roughly the last 5 years". Era-range framing, NOT \
     chronological ordinal.
   - "80s throwback" → 1980s era + retro aesthetic.

8b. Chronological ordinal (position within a scoped set).
   - "first" (as in "first Indiana Jones") → one entry \
     {Chronological, clean, "earliest release-date position within \
     the scoped candidate set"}.
   - "most recent" (as in "most recent Nolan film") → one entry \
     {Chronological, clean, "latest release-date position within \
     the scoped candidate set"}.
   - Always its own attribute fragment — never a modifier on the \
     adjacent attribute. Scope comes from the other fragments in \
     the query; this fragment only names the ordinal position.

9. Imprecise quantifier.
   - "fairly old" → older era as soft preference (Structured \
     metadata).
   - "somewhat scary" → horror/thriller (Top-level genre) with \
     moderate intensity (Sensitive content / Viewer experience).

10. Ambiguous scope.
    - "family movie" — usually {Target audience: for family \
      co-viewing}; can also mean {Kind of story: about a family}. \
      Pick the likely reading; if genuinely ambiguous, include \
      both as entries.

11. Multi-dimension entity — only when each reading INDEPENDENTLY \
    holds. Emit a Named character entry only if the fragment names \
    an actual persona; emit a Franchise / universe lineage entry \
    only if the fragment names a recognized franchise. Do NOT \
    invent a franchise around a character, and do NOT invent a \
    character around a franchise. There are three canonical shapes:

    a. Named persona that is ITSELF a recognized franchise — emit \
       both.
       - "Spider-Man" → {Named character, "Spider-Man"} + \
         {Franchise / universe lineage, "Spider-Man franchise"}.
       - "Barbie", "James Bond", "Batman" → same shape.

    b. Named persona whose name is NOT a franchise — emit Named \
       character only. Do not speculate a franchise into existence \
       just because the name could plausibly title one.
       - "movies with Neo in it" → {Named character, "Neo"} only. \
         Neo is a Matrix character; there is no "Neo franchise" — \
         the franchise is "The Matrix", which is not stated here.
       - "Hermione movies" → {Named character, "Hermione"} only. \
         The franchise would be "Harry Potter", not "Hermione".

    c. Franchise whose name is NOT a persona — emit Franchise / \
       universe lineage only. Do not invent a character named \
       after the franchise.
       - "Star Wars movies" → {Franchise / universe lineage, \
         "Star Wars franchise"} only. "Star Wars" is not a \
         character.
       - "Jurassic Park", "Fast & Furious", "Lord of the Rings" → \
         same shape (franchise only).

---

"""

_GROUND_RULES = """\
GROUND RULES

1. Stay within the user's stated or clearly-implied intent. Make \
   implicit things explicit; do not invent new constraints.
2. Small words matter. Role markers ("starring", "directed by", \
   "about", "set in") and polarity modifiers ("not", "not too", \
   "without") are signal, not filler — they belong inside the \
   adjacent attribute's modifiers list.
3. Chronological language ("first", "last", "earliest", "latest", \
   "most recent") is its own attribute fragment, mapped to the \
   Chronological category. It is not a modifier of another \
   attribute. "Best", "top", "rarely-seen", and other reception- \
   or popularity-framed superlatives are handled by Reception \
   quality + superlative or Structured metadata — they are \
   attributes too, not selection rules.
4. Multi-dimension entities default to preserving every reading \
   that INDEPENDENTLY holds as a clean-fit coverage entry. A Named \
   character reading requires the fragment to name an actual \
   persona; a Franchise / universe lineage reading requires the \
   fragment to name an actual recognized franchise. Spider-Man / \
   Barbie / James Bond are both (persona + same-named franchise); \
   "Neo" is character only; "Star Wars" is franchise only. Do NOT \
   invent a franchise to pair with a character name, or a \
   character to pair with a franchise name. Beyond that, narrow \
   only when another fragment in the query definitionally rules \
   out a surviving reading (see CLARIFYING EVIDENCE) — not merely \
   because the combination is unusual. When narrowing, record the \
   dropped reading inside the surviving captured_meaning.
5. Evidence before judgment. Within each coverage_evidence entry, \
   write captured_meaning before category_name. Do not decide the \
   category first and rationalize.
6. Modifier signal must be reflected in atomic_rewrite. A \
   ROLE_MARKER "starring" on a Tom Hanks fragment must surface as \
   "Tom Hanks in a lead role"; a POLARITY_MODIFIER "not too" on a \
   violent fragment must surface as "with a preference against \
   graphic violence". Specificity preserved: "brother" stays \
   "brother", "1990s" stays "1990s", "starring" stays "starring" \
   (or its lead-role equivalent) — never generalized to \
   "featuring" or "sibling".
7. Preserve verbatim user language in query_text and in each \
   modifier's original_text, typos included.
8. Do not emit category IDs, routing hints, dispatch metadata, or \
   references to endpoints.

---

"""

_OUTPUT_GUIDANCE = """\
OUTPUT FIELD GUIDANCE

overall_query_intention_exploration — 2 to 4 sentences describing \
what the query is asking for as a whole.

requirements — list of RequirementFragment. Fields in schema order:
  query_text: verbatim attribute span (no role markers or polarity \
    words — those go in modifiers).
  description: one sentence on what the fragment contributes.
  modifiers: list of Modifier entries (empty when none apply). \
    Each entry has:
      original_text: verbatim span from the query.
      effect: brief note on how the modifier changes the \
        attribute.
      type: POLARITY_MODIFIER | ROLE_MARKER.
  coverage_evidence: list of CoverageEvidence entries (one per \
    category-grounded atom). Each entry has:
      captured_meaning: observation before category.
      category_name: taxonomy name.
      fit_quality: clean | partial | no_fit.
      atomic_rewrite: category-grounded, specificity-preserving, \
        and reflects any nested modifiers.
"""


SYSTEM_PROMPT = (
    _TASK_AND_OUTCOME
    + _LANGUAGE_TYPES
    + _CATEGORY_TAXONOMY
    + _CLARIFYING_EVIDENCE
    + _COVERAGE_EVIDENCE_RULES
    + _CHUNKING_RULES
    + _ATOMIZATION_PATTERNS
    + _GROUND_RULES
    + _OUTPUT_GUIDANCE
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
