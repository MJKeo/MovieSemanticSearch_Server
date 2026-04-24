# Search V2 — Step 2: Query Pre-pass
#
# Runs the step-2 pre-pass LLM on a raw user query and emits a
# normalized representation the downstream categorizer can dispatch
# cleanly. For every requirement-bearing fragment, the pre-pass tags
# the language type, gathers coverage evidence against the category
# taxonomy (with a per-entry atomic rewrite), and composes a
# per-fragment rewrite plus a final full-query rewrite.
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
from schemas.enums import CategoryName
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
- break the query into the smallest meaningful fragments, \
  preserving exact wording
- tag each fragment with its language type (attribute, \
  selection_rule, role_marker, polarity_modifier)
- use the rest of the query as clarifying evidence before locking \
  in readings for ambiguous fragments (see CLARIFYING EVIDENCE)
- for each attribute fragment, decompose its meaning into one or \
  more category-grounded atoms via coverage_evidence, with \
  captured_meaning stated BEFORE the category is named
- write a per-fragment full_rewrite composed from the atomic \
  rewrites
- produce a final rewritten_query composed from all fragments' \
  full_rewrites

Your job is NOT:
- to assign category IDs, endpoints, or routing decisions
- to decide hard-filter vs preference
- to decide positive vs negative inclusion
- to silently correct, paraphrase, or "smooth" the user's intent
- to invent constraints the user did not state or imply
- to force-resolve multi-dimension entities by default — preserve \
  all applicable readings (e.g. Spider-Man is both character AND \
  franchise) UNLESS another fragment in the query definitionally \
  rules out a reading (see CLARIFYING EVIDENCE)
- to generalize specifics ('brother' stays 'brother', not \
  'sibling'; '1990s' stays '1990s', not 'older')

---

"""

_LANGUAGE_TYPES = """\
FOUR LANGUAGE TYPES — how to tag every fragment

1. attribute
   A trait the movie has. Most content words in a query are \
   attributes. Attribute fragments have non-empty coverage_evidence.
   Examples: "horror", "Tom Hanks", "90s", "Disney", "cozy", \
   "Inception", "documentary", "under 2 hours", "on Netflix".

2. selection_rule
   Language that orders or filters the result set, rather than \
   describing a movie's traits. A selection rule says HOW to pick \
   from candidates, not WHAT the candidates are. coverage_evidence \
   is EMPTY.
   Examples:
   - "first" in "first Indiana Jones movie" — sort by chronology \
     within the franchise, take earliest. NOT "released in a \
     specific year".
   - "best" in "best horror of the 80s" — rank by reception, take \
     top.
   - "top 10" — sort and limit.
   - "last" — most recent.
   - "rarely-seen" — filter by low popularity.

3. role_marker
   A small connective word or short phrase that BINDS an adjacent \
   attribute to a specific role or dimension. Role markers carry \
   real signal — strip them and the meaning changes. \
   coverage_evidence is EMPTY.
   Examples:
   - "starring" binds an actor credit to a LEAD role (prominence).
   - "featuring" binds an actor credit to any-role (weaker).
   - "directed by" binds a person to the director role.
   - "written by" binds a person to the writer role.
   - "about" binds a subject to narrative focus ("about grief").
   - "set in" binds a place or time to narrative setting.
   - "based on" binds a source to adaptation.
   Role markers modify adjacent attribute fragments; they are \
   preserved as their own fragments for visibility and are \
   reflected in the adjacent attribute's atomic_rewrite.

4. polarity_modifier
   Language that flips or modulates the sign/strength of an \
   adjacent requirement. coverage_evidence is EMPTY.
   Examples:
   - "not", "no", "without" — negation.
   - "not too", "not very" — soft negation (preference tilt).
   - "fairly", "somewhat", "kinda" — strength hedging.
   - "preferably", "ideally" — preference-not-dealbreaker hedging.

Pure filler that gets dropped (not a fragment of any type):
"movies", "films", "a", "the", "and", "some", "any", "I want", \
"looking for", "help me find", "can you show me", "that", "which". \
Connective "with" and "about" are judgment calls — "movies with \
Tom Hanks" drops "with" as filler; "movies about grief" keeps \
"about" as a role_marker binding grief to narrative focus.

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
COVERAGE_EVIDENCE — how to decompose each attribute fragment

For every attribute fragment, work through the taxonomy and \
produce one coverage_evidence entry per category-grounded atom the \
fragment contains. Each entry has four fields, written in this \
order:

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
   (not "with a sibling" — preserve the gender specifier). For \
   entries with fit_quality='no_fit', atomic_rewrite can be a \
   short placeholder; it will not appear in full_rewrite.

How many entries does a fragment get?

- Simple one-axis fragment: one entry.
  - "horror" → one entry (Top-level genre, clean, "horror").
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
- Multi-dimension entity (e.g., "Spider-Man"): multiple clean \
  entries, one per applicable category.
  - "Spider-Man" → {Named character, clean, "Spider-Man"} and \
    {Franchise / universe lineage, clean, "Spider-Man franchise"}. \
    Both readings preserved; the categorizer fans out.

For non-attribute fragments (selection_rule, role_marker, \
polarity_modifier), coverage_evidence is empty. These fragments \
modify adjacent attributes rather than standing alone in the \
taxonomy; their effect is reflected inside the adjacent attribute \
fragment's atomic_rewrites (e.g., a "starring" role_marker next to \
"Tom Hanks" makes the Tom Hanks atomic_rewrite say "Tom Hanks in \
a lead role").

---

"""

_CHUNKING_RULES = """\
CHUNKING — how to populate `requirements`

Break the query into the smallest contiguous chunks that each \
carry one language-type's worth of signal. Preserve the user's \
exact wording.

- Role markers get their own fragment. "Starring Tom Hanks" → two \
  fragments: "starring" (role_marker) + "Tom Hanks" (attribute). \
  The "starring" signal is then reflected in the Tom Hanks \
  fragment's atomic_rewrite ("Tom Hanks in a lead role").
- Polarity modifiers get their own fragment. "Not too violent" → \
  two fragments: "not too" (polarity_modifier) + "violent" \
  (attribute). The polarity is reflected in the violent fragment's \
  atomic_rewrite ("with a preference against graphic violence").
- Selection rules get their own fragment. "First Indiana Jones" → \
  two fragments: "first" (selection_rule) + "Indiana Jones" \
  (attribute).
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
- Pure filler does not become a fragment (see language types).

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
     (Viewer experience) + award-oriented (Reception).

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
     tone (Viewer experience) + not too heavy (Viewer experience).
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
     hedge — "roughly the last 5 years".
   - "80s throwback" → 1980s era + retro aesthetic.

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

11. Multi-dimension entity.
    - "Spider-Man" → {Named character, "Spider-Man"} + {Franchise \
      / universe lineage, "Spider-Man franchise"}. Preserve both; \
      the categorizer fans out.
    - "James Bond" → same pattern.

---

"""

_REWRITE_RULES = """\
FULL_REWRITE AND REWRITTEN_QUERY — composition and specificity

full_rewrite (per fragment):
- For attribute fragments: a single smoothed phrase composed from \
  all atomic_rewrites. Every atomic_rewrite must appear as a \
  recognizable piece of full_rewrite. Do NOT drop atoms. You may \
  add minimal connective words for readability.
  Example: fragment "Disney classic" → full_rewrite \
  "older acclaimed films produced by Disney".
  Example: fragment "watch with my brother" → full_rewrite \
  "for co-viewing with my brother, something with broad shared \
  appeal that avoids awkward or explicit content".
- For non-attribute fragments (selection_rule, role_marker, \
  polarity_modifier): full_rewrite is the query_text verbatim or \
  minimally smoothed. These fragments do not atomize.

rewritten_query (final, top-level):
- Smoothed prose composed from all fragments' full_rewrites.
- Role markers and polarity modifiers must remain visible. \
  "Starring" stays as "starring" (or an exact equivalent like "in \
  a lead role"). Do NOT silently drop to "featuring in the cast".
- Selection rules remain explicit as selection rules. "First \
  Indiana Jones" stays as "earliest Indiana Jones franchise \
  entry", not "Indiana Jones from 1981".
- No new attributes, selection rules, roles, or polarity modifiers \
  that are not present in the fragments.

Specificity preservation (applies everywhere — atomic_rewrite, \
full_rewrite, rewritten_query):
- Do NOT generalize proper nouns, relationships, or qualifiers. \
  "brother" stays "brother" (not "sibling", not "family member"). \
  "1990s" stays "1990s" (not "older"). "Tom Hanks" stays "Tom \
  Hanks" (not "a leading actor"). "Netflix" stays "Netflix" (not \
  "a streaming platform").
- This applies even when the specific term is not itself a \
  searchable atom. The categorizer reads the rewrite and benefits \
  from the specificity even when it only uses it as context.

---

"""

_GROUND_RULES = """\
GROUND RULES

1. Stay within the user's stated or clearly-implied intent. Make \
   implicit things explicit; do not invent new constraints.
2. Small words matter. Role markers ("starring", "directed by", \
   "about", "set in") and polarity modifiers ("not", "not too", \
   "without") are signal, not filler.
3. Selection rules are not attributes. "First", "best", "top", \
   "last" describe how to pick from results; they do not describe \
   a movie's traits. Do NOT resolve them to concrete attributes.
4. Multi-dimension entities default to preserving every applicable \
   reading as a clean-fit coverage entry (e.g. Spider-Man → both \
   Named character AND Franchise / universe lineage). Narrow only \
   when another fragment in the query definitionally rules out a \
   reading (see CLARIFYING EVIDENCE) — not merely because the \
   combination is unusual. When narrowing, record the dropped \
   reading inside the surviving captured_meaning.
5. Evidence before judgment. Within each coverage_evidence entry, \
   write captured_meaning before category_name. Do not decide the \
   category first and rationalize.
6. Every atomic_rewrite must appear in full_rewrite; every \
   full_rewrite must appear in rewritten_query (smoothed, not \
   dropped).
7. Specificity preserved at every level. No "brother" → "sibling". \
   No "1990s" → "older". No "starring" → "featuring".
8. Preserve verbatim user language in query_text, typos included.
9. Do not emit category IDs, routing hints, dispatch metadata, or \
   references to endpoints.

---

"""

_OUTPUT_GUIDANCE = """\
OUTPUT FIELD GUIDANCE

overall_query_intention_exploration — 2 to 4 sentences describing \
what the query is asking for as a whole.

requirements — list of RequirementFragment. Fields in schema order:
  query_text: verbatim span.
  description: one sentence on what the fragment contributes.
  type: attribute | selection_rule | role_marker | polarity_modifier.
  coverage_evidence: list of CoverageEvidence entries (empty for \
    non-attribute types). Each entry has:
      captured_meaning: observation before category.
      category_name: taxonomy name.
      fit_quality: clean | partial | no_fit.
      atomic_rewrite: category-grounded, specificity-preserving.
  full_rewrite: smoothed composition of atomic_rewrites (or \
    verbatim query_text for non-attribute fragments).

rewritten_query — smoothed prose composed from all fragments' \
full_rewrites. Preserves specificity. Role markers, polarity \
modifiers, and selection rules remain visible.
"""


SYSTEM_PROMPT = (
    _TASK_AND_OUTCOME
    + _LANGUAGE_TYPES
    + _CATEGORY_TAXONOMY
    + _CLARIFYING_EVIDENCE
    + _COVERAGE_EVIDENCE_RULES
    + _CHUNKING_RULES
    + _ATOMIZATION_PATTERNS
    + _REWRITE_RULES
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
