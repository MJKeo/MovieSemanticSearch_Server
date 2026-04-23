# Search V2 — Pre-pass Exploration (temporary test file)
#
# Runs the pre-pass LLM on a single query and prints the structured
# output. The pre-pass normalizes a raw user query into "simple"
# atom-sized fragments before the category dispatcher sees it,
# flagging compound words, parametric references, role-bound
# qualifiers, colloquialisms, figurative language, and similar
# patterns that are lossy to decompose at the categorizer stage.
#
# This file is a scratchpad — schema, prompt, and executor all live
# here so the prompt can be iterated without touching the permanent
# pipeline. It is not wired into the main search flow.
#
# Usage:
#   python -m search_v2.prepass_explorations "your query here" --gemini
#   python -m search_v2.prepass_explorations "your query here" --openai

import argparse
import asyncio
import json
from typing import List

from pydantic import BaseModel, Field

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)


# ===============================================================
#                      Output schema
# ===============================================================
#
# Field order matters — it scaffolds the LLM's reasoning:
#   1. Understand the whole query first (exploration).
#   2. Chunk it into fragments preserving original wording.
#   3. Flag problematic chunks and unpack them.
#   4. Emit a fully normalized rewrite built from simple atoms.
#
# The chunks in `requirements` preserve the user's exact wording;
# rewrites and decompositions happen in `problematic_requirements`
# and `expanded_rewritten_query`.


class RequirementFragment(BaseModel):
    """A contiguous chunk of the query that conveys one thing about
    the desired results. Preserves the user's exact wording.
    """
    query_text: str = Field(
        ...,
        description=(
            "The fragment exactly as it appears in the query, "
            "preserving the user's wording."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "One sentence describing what this fragment is asking "
            "for about the movie."
        ),
    )


class ProblematicRequirement(BaseModel):
    """A fragment (or phrase) that bundles multiple meanings or
    requires unpacking before the category dispatcher can handle it
    cleanly. Not every fragment is problematic — only the ones that
    would be lossy to pass downstream as-is.
    """
    query_text: str = Field(
        ...,
        description=(
            "The original phrase from the query that is problematic. "
            "Does not need to match a requirements entry exactly — "
            "this may be a larger span or a sub-span."
        ),
    )
    potential_problem: str = Field(
        ...,
        description=(
            "One-to-two-sentence explanation of why this phrase "
            "bundles multiple meanings or needs unpacking. Name the "
            "pattern (compound descriptor, parametric reference, "
            "role-bound qualifier, colloquialism, figurative "
            "language, implicit bundle, temporal relative, "
            "imprecise quantifier, etc.)."
        ),
    )
    distinct_meanings: List[str] = Field(
        ...,
        description=(
            "The simple atoms contained within this phrase. Each "
            "entry is a single plain-language attribute the "
            "downstream system can search on directly."
        ),
    )


class PrepassResponse(BaseModel):
    """Structured output for the pre-pass step."""

    overall_query_intention_exploration: str = Field(
        ...,
        description=(
            "A short exploration of what the query as a whole is "
            "asking for. Written for a downstream reader, not as "
            "stream-of-consciousness."
        ),
    )
    requirements: List[RequirementFragment] = Field(
        ...,
        description=(
            "Every chunk of the query that conveys a requirement, "
            "quoted with original wording. Use the smallest meaningful "
            "unit per chunk (prefer 'Disney' + 'animated' + "
            "'classics' over a single 'Disney animated classics' "
            "chunk when the parts stand alone)."
        ),
    )
    problematic_requirements: List[ProblematicRequirement] = Field(
        ...,
        description=(
            "The subset of phrases in the query that bundle multiple "
            "meanings or otherwise need unpacking. May be empty."
        ),
    )
    expanded_rewritten_query: str = Field(
        ...,
        description=(
            "A single prose sentence (or two) that rewrites the "
            "query using only simple atoms — every compound, "
            "reference, or colloquialism expanded into its explicit "
            "atomic components. Preserves the user's polarity and "
            "intent; adds nothing the user did not imply."
        ),
    )


# ===============================================================
#                      System prompt
# ===============================================================
#
# Authoring conventions:
#  - Describe dimensions, not categories or endpoints. The pre-pass
#    should NOT know about the 30-category taxonomy, vector spaces,
#    or the scoring model — those belong to later steps. Including
#    them here would encourage scope creep and couple the prompt to
#    downstream implementation.
#  - Lead with the problematic-patterns taxonomy. That is where the
#    LLM's judgment gets calibrated and most of the prompt budget
#    should go.
#  - End with explicit ground rules — what the pre-pass must NOT do.
#    These guard against the main failure modes: hallucinating
#    constraints, silently "correcting" the user, and drifting into
#    categorization.


_TASK_AND_OUTCOME = """\
You are the pre-pass step in a movie search pipeline. Your job is \
to read a raw user query and rewrite it into a normalized form that \
downstream steps can reason about cleanly.

Downstream, a second LLM will break the query into atomic \
requirements and map each to a search category. That second step \
works best when the input consists of simple attributes the search \
system can score against directly. It struggles with compound \
words, external references, figurative language, and qualifiers \
that are bound to other entities. Your job is to surface those \
problem cases and expand them into simple atoms.

Your job is:
- explore what the query as a whole is asking for
- chunk the query into distinct requirement fragments, preserving \
  the user's exact wording
- flag fragments (or broader phrases) that bundle multiple meanings \
  and unpack them into their distinct simple atoms
- produce a single rewritten-prose version of the query that uses \
  only simple atoms

Your job is NOT:
- to assign categories, endpoints, or scoring weights
- to decide whether a requirement is a hard filter or a preference
- to decide whether a requirement is positive or negative inclusion
- to invent constraints the user did not state or imply
- to silently "correct" the user's intent

---

"""

_ATOMIC_DIMENSIONS = """\
WHAT COUNTS AS A SIMPLE ATOM

A simple atom is a single attribute of a movie that the search \
system can score on directly. The dimensions the system treats as \
atomic include:

- Person credits: an actor, director, writer, producer, composer, \
  cinematographer, editor, or other named creator.
- Named characters: a specific character by name (e.g., "Batman").
- Studio or production brand: Disney, Pixar, A24, Studio Ghibli.
- Franchise or universe: MCU, Star Wars, James Bond.
- Release era or date: a year, decade, or range.
- Runtime: a length or range in minutes.
- Country of origin or audio language.
- Streaming platform.
- Maturity rating.
- Numeric reception score.
- Award records: wins, nominations, specific ceremonies.
- Top-level genre: horror, comedy, drama, action, sci-fi, romance, \
  animation.
- Sub-genre or story archetype: cozy mystery, space opera, heist, \
  underdog, revenge.
- Subject matter or motif: clowns, zombies, sharks, JFK, Vietnam War.
- Source of adaptation: novel, comic book, true story, remake.
- Format or visual style: documentary, black and white, found \
  footage, animation style.
- Tone or feel: dark, whimsical, gritty, cozy, tense, lighthearted.
- Pacing or cognitive demand: slow burn, fast, mindless, cerebral.
- Thematic archetype: grief, redemption, coming-of-age, man-vs-nature.
- Narrative device or structural form: nonlinear, ensemble, \
  unreliable narrator, twist ending.
- Viewing occasion: date night, rainy Sunday, background.
- Sensitive content levels: violence, gore, sex, language.
- Curated canon membership: Criterion, AFI Top 100, IMDb Top 250.

If a phrase in the query clearly names one of these dimensions with \
one concrete value, it is a simple atom. If a phrase bundles \
multiple of these dimensions into one word or expression, it is a \
candidate for the problematic list.

---

"""

_PROBLEMATIC_PATTERNS = """\
PROBLEMATIC PATTERNS — the phrases you should flag and unpack

A phrase is problematic when passing it to the category dispatcher \
as-is would lose meaning. The following patterns cover most cases. \
Name the pattern in `potential_problem` where it applies.

1. Compound descriptors.
   A single word or short phrase that denotes multiple independent \
   attributes at once.
   - "classic" → old era + canonical/acclaimed stature.
   - "Disney classic" → Disney + old era + canonical stature.
   - "modern classic" → recent era + canonical stature.
   - "Oscar bait" → prestige drama + award-oriented + serious tone.
   The parts stand alone; splitting is lossless.

2. Parametric references.
   Mentions of a specific film, show, actor style, or cultural \
   touchstone where the user expects the system to know what it \
   means. The reference compresses many attributes into one name.
   - "like Rocky" → underdog + boxing + training-montage + \
     scrappy-protagonist.
   - "Wes Anderson-style" → symmetrical framing + deadpan + \
     pastel palette + quirky ensemble + melancholic undertone.
   - "Bollywood" → Indian production + Hindi-language + musical \
     elements + specific melodrama conventions.
   Extract the traits from the reference so the downstream system \
   does not have to re-derive them.

3. Role-bound qualifiers.
   A qualifier attached to a specific entity rather than the movie \
   as a whole. Splitting naively breaks the binding.
   - "Tom Hanks as a villain" — the role ("villain") is bound to \
     Tom Hanks, not to the movie. Do NOT output "movies containing \
     a villain" as one atom and "starring Tom Hanks" as another. \
     Keep the binding explicit: "Tom Hanks playing a villain role" \
     stays as one bound atom in the rewrite.
   - "Spielberg directing a horror" — genre is bound to the \
     director, not the movie category. Keep the binding.

4. Colloquialisms, slang, and idioms.
   Single words or short phrases that film-literate users use as \
   shorthand for specific bundles.
   - "tearjerker" → emotionally heavy + sad + cathartic.
   - "popcorn flick" → action or blockbuster + light + easy-to-watch.
   - "sleeper hit" → commercially modest at release + strong \
     reception over time.
   - "guilty pleasure" → genre-trashy + personally enjoyable + \
     not critically acclaimed.
   - "so bad it's good" → widely considered poor-quality + \
     entertainingly so.

5. Domain jargon and named movements.
   Specialized film-history terms that require parametric \
   knowledge.
   - "giallo" → Italian horror + stylized violence + specific era.
   - "pre-code Hollywood" → American film + 1929-1934 + looser \
     content standards.
   - "mumblecore" → low-budget indie + naturalistic dialogue + \
     relationship-focused.
   - "neo-noir" → noir aesthetic + modern production + crime or \
     moral ambiguity.
   These are usually unpackable into a handful of simple atoms.

6. Figurative language.
   Metaphors for how a movie feels that need translation into \
   concrete feel/tone attributes.
   - "a gut-punch of a movie" → emotionally devastating + strong \
     impact + likely dark or tragic.
   - "a warm hug of a film" → comforting + feel-good + gentle tone.
   - "like a fever dream" → surreal + disorienting + dreamlike \
     visuals.

7. Implicit bundles (context framings).
   Phrases that imply several constraints simultaneously through \
   the viewing context they describe.
   - "something to watch with my 8-year-old" → family-appropriate \
     maturity rating + broad appeal + animated or live-action kids \
     genre + moderate runtime.
   - "a date night movie" → appealing to two adults + often \
     romantic or crowd-pleasing + not too heavy + not too long.
   - "my dad would love" → reflects the named person's tastes; \
     without specific context, unpack as a preference signal you \
     flag but cannot fully resolve — note the limitation rather \
     than inventing dad's taste.

8. Temporal relatives.
   References to time that need resolving to absolute ranges.
   - "recent" → roughly the last 5 years.
   - "last year" → the specific prior calendar year.
   - "80s throwback" → 1980s era + retro aesthetic.

9. Imprecise quantifiers.
   Hedged or graded language that needs normalizing.
   - "fairly old" → older era (e.g., pre-2000) as a soft preference.
   - "pretty long" → runtime above average, likely 2h+.
   - "somewhat scary" → horror or thriller but not extreme.

10. Ambiguous scope.
    Words with multiple plausible readings where you must pick or \
    surface the ambiguity.
    - "family movie" → usually "for a family to watch together" \
      but can mean "about a family."
    - "Christmas movie" → usually "for Christmas viewing" but can \
      mean "narratively set at Christmas."
    Pick the likely reading from context. If genuinely ambiguous, \
    note it in the problematic entry.

---

"""

_CHUNKING_RULES = """\
CHUNKING — how to populate `requirements`

Break the query into the smallest contiguous chunks that each \
carry one requirement about the movie. Preserve the user's exact \
wording; do not paraphrase inside `query_text`.

- Prefer smaller chunks when the parts stand alone. \
  "Disney animated classics" becomes three chunks: "Disney", \
  "animated", "classics" — each is a distinct requirement.
- Keep a phrase together when splitting would destroy its meaning. \
  "Tom Hanks as a villain" stays as one chunk because "as a \
  villain" has no meaning detached from "Tom Hanks".
- Ignore connective filler: words like "movies", "films", "that", \
  "and", "with", "about", "from", "some", "any", "I want", \
  "looking for", "help me find" do not become requirement chunks. \
  They can be present in the original query but should not be \
  quoted as requirements.
- A chunk can appear in `requirements` AND be referenced in \
  `problematic_requirements`. The problematic list is not \
  exclusive — it is the subset that needs unpacking.
- The `query_text` in a problematic_requirements entry does not \
  have to match a requirements chunk exactly. A longer phrase \
  ("my dad would love") or a single word inside a chunk can be \
  flagged independently.

---

"""

_REWRITE_RULES = """\
REWRITING — how to populate `expanded_rewritten_query`

Produce one or two sentences of prose that express the query using \
only simple atoms. Every problematic phrase should be expanded \
into its distinct_meanings. Role-bound qualifiers stay bound (keep \
"Tom Hanks in a villain role" as a bound phrase rather than two \
independent atoms).

The rewrite is for a downstream LLM, not the user. It can be \
longer than the original and does not need to sound natural — \
clarity and explicitness matter more than elegance.

Preserve the user's polarity and emphasis. If the user said "not \
too violent", the rewrite says "with a preference against graphic \
violence" — do not flip the sign or drop the hedge.

Do not add constraints the user did not state or imply. If the \
user said "date night movie", the rewrite expands the implicit \
bundle the phrase canonically carries; it does NOT add \
"preferably two hours or less" unless the user mentioned runtime.

---

"""

_GROUND_RULES = """\
GROUND RULES

1. Stay within the user's stated or clearly-implied intent. Make \
   implicit things explicit; do not invent new constraints.
2. When a reference or bundle is ambiguous or you are not confident \
   about its meaning, say so in `potential_problem` rather than \
   committing to a made-up expansion.
3. Do not assign search categories, endpoints, or scoring roles. \
   That is the next step's job. You describe WHAT the query means; \
   the next step decides HOW to search for it.
4. Do not silently correct the user. If they wrote a misspelled \
   word or an unusual phrasing, preserve `query_text` verbatim. \
   Corrections can appear in `description` or the rewrite, with the \
   original visible upstream.
5. Prefer noting a limitation over fabricating. If "my dad would \
   love" cannot be resolved to concrete attributes from the query \
   alone, flag it as a problematic reference you cannot fully \
   unpack, and the rewrite should leave it as an unresolved \
   preference cue rather than guessing at dad's taste.

---

"""

_OUTPUT_GUIDANCE = """\
OUTPUT FIELD GUIDANCE

overall_query_intention_exploration — 2 to 4 sentences. What the \
query is asking for as a whole, including any overarching framing \
(occasion, audience, mood) that colors the specific requirements.

requirements — list of RequirementFragment. Every requirement-\
bearing chunk of the query, with original wording in `query_text` \
and a one-sentence description of what the fragment is asking for.

problematic_requirements — list of ProblematicRequirement. Only \
include phrases that fall into one of the problematic patterns \
above. May be empty. For each, name the pattern in \
`potential_problem` and list the simple atoms in \
`distinct_meanings`.

expanded_rewritten_query — prose, one or two sentences, built \
entirely from simple atoms. Expansions of problematic phrases \
appear inline. Role-bound qualifiers stay bound. Polarity and \
emphasis preserved from the original.
"""


SYSTEM_PROMPT = (
    _TASK_AND_OUTCOME
    + _ATOMIC_DIMENSIONS
    + _PROBLEMATIC_PATTERNS
    + _CHUNKING_RULES
    + _REWRITE_RULES
    + _GROUND_RULES
    + _OUTPUT_GUIDANCE
)


# ===============================================================
#                      Executor
# ===============================================================


# Provider configs. Gemini is fast and cheap; OpenAI gpt-5-mini is
# the go-to mid-tier for structured-output tasks in this repo.
_PROVIDER_CONFIGS = {
    "gemini": {
        "provider": LLMProvider.GEMINI,
        "model": "gemini-3-flash-preview",
        "kwargs": {
            "thinking_config": {"thinking_budget": 0},
            "temperature": 0.35,
        },
    },
    "openai": {
        "provider": LLMProvider.OPENAI,
        "model": "gpt-5.4-mini",
        "kwargs": {"reasoning_effort": "none", "verbosity": "low"},
    },
}


async def run_prepass(
    query: str,
    provider_key: str,
) -> tuple[PrepassResponse, int, int]:
    """Run the pre-pass LLM on a single query.

    Args:
        query: the raw user query.
        provider_key: one of "gemini", "openai".

    Returns:
        (response, input_tokens, output_tokens)
    """
    query = query.strip()
    if not query:
        raise ValueError("query must be a non-empty string.")
    if provider_key not in _PROVIDER_CONFIGS:
        raise ValueError(
            f"unknown provider '{provider_key}'. "
            f"Valid options: {list(_PROVIDER_CONFIGS.keys())}"
        )

    cfg = _PROVIDER_CONFIGS[provider_key]
    user_prompt = f"Query: {query}"

    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=cfg["provider"],
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=PrepassResponse,
        model=cfg["model"],
        **cfg["kwargs"],
    )

    return response, input_tokens, output_tokens


# ===============================================================
#                      CLI
# ===============================================================


def _print_response(response: PrepassResponse) -> None:
    """Pretty-print the structured response for terminal inspection."""
    payload = response.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pre-pass exploration on a single query and "
            "print the structured output."
        )
    )
    parser.add_argument(
        "query",
        type=str,
        help="The raw user query to process.",
    )
    provider_group = parser.add_mutually_exclusive_group()
    provider_group.add_argument(
        "--gemini",
        action="store_const",
        dest="provider",
        const="gemini",
        help="Use Gemini (default).",
    )
    provider_group.add_argument(
        "--openai",
        action="store_const",
        dest="provider",
        const="openai",
        help="Use OpenAI gpt-5.4-mini (no thinking).",
    )
    parser.set_defaults(provider="gemini")
    return parser.parse_args()


async def _main_async() -> None:
    args = _parse_args()
    response, in_tok, out_tok = await run_prepass(
        query=args.query,
        provider_key=args.provider,
    )
    _print_response(response)
    print(
        f"\n[tokens] input={in_tok} output={out_tok} "
        f"provider={args.provider}"
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
