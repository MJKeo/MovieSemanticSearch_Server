# Search V2 — Step 1: Spin Generation
#
# Runs in parallel with step 0 on the raw user query. Treats every
# input as if the standard flow will execute and produces two
# creative spins — adjacent-but-distinct searches that promote
# browsing — plus a short UI label for the user's original query.
#
# Step 1 does NOT rewrite the original query (it passes through
# verbatim downstream), decide flow routing (step 0 owns that), or
# decompose into category-grounded atoms (step 2 owns that). Its
# scope is narrow: decompose the query into levers, then generate
# two distinct spins by pulling on two different levers.
#
# See search_improvement_planning/steps_1_2_improving.md (Step 1:
# Standard-Flow Intent Expansion) for the full design rationale.
#
# Usage:
#   python -m search_v2.step_1 "your query here"

from __future__ import annotations

import argparse
import asyncio
import json
import time

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)
from schemas.step_1 import Step1Response


# ===============================================================
#                      System prompt
# ===============================================================
#
# Modular sections concatenated at module level.
#
# Structure: task/outcome → observations-first principle → per-field
# decomposition rules → spin construction rules → distinctness
# discipline → UI label rules → boundary examples → output field
# guidance.
#
# Prompt authoring conventions applied:
# - Observations before decisions: decomposition fields come first
#   and ground the spin choices.
# - Principle-based constraints, not failure catalogs.
# - Field order in the schema scaffolds reasoning for small models.
# - Concrete examples of good vs bad distinctness.


_TASK_AND_OUTCOME = """\
You are the spin-generation step in a movie search pipeline. The \
user's raw query is already being searched verbatim as its own \
branch downstream. Your job is to propose two ADDITIONAL creative \
spins — adjacent but distinct searches the user might find \
interesting but wouldn't have typed themselves — plus short UI \
labels for all three branches.

The core purpose is to promote browsing and exploration. Two spins \
that retrieve largely the same movies as the original waste branch \
slots. Two spins that retrieve largely the same movies as each \
other waste one slot. Each spin must return a visibly different \
result set from both the original AND the sibling spin — if the \
tweaks are so minor the result set barely changes, that is \
boring and wrong.

Your job is:
- decompose the query into hard commitments, soft interpretations, \
  and open dimensions
- write a short UI label for the original query
- produce exactly two spins, each pulling on a different lever, \
  each with its own UI label

Your job is NOT:
- to rewrite or reinterpret the original query — it passes through \
  unchanged as its own branch
- to decide which flow the query belongs to — step 0 handles that
- to decompose the query into category-grounded atoms — step 2 \
  handles that
- to inject proxy quality signals like "highly-rated", "iconic", \
  "prestigious", or "cult-favorite" unless the user asked for them
- to produce spins that share a lever — two narrowings along the \
  same axis, or two reinterpretations of the same soft term, are \
  design errors

---

"""

_OBSERVATIONS_FIRST = """\
OBSERVATIONS FIRST, DECISIONS SECOND

The schema has two zones. Fill them in order.

Zone 1 — Observations (decomposition). Before committing to any \
spin, decompose the query along three axes:
- hard_commitments
- soft_interpretations
- open_dimensions

Zone 2 — Decisions. With the decomposition in hand, write:
- original_query_label
- spins (exactly two, each naming the specific lever it pulls, \
  each grounded in an item from Zone 1)

Every spin's branching_opportunity must name a specific item from \
either soft_interpretations or open_dimensions. If both lists are \
empty (the query is fully specified and untouched by soft terms), \
the spins swap in an adjacent concept for something the user left \
implicit — a different era, an adjacent actor/director, a shifted \
tone — but each spin still commits to ONE such swap and the two \
swaps are on different axes.

---

"""

_DECOMPOSITION_RULES = """\
HARD_COMMITMENTS — how to populate

Things the user explicitly named that any faithful spin must \
preserve. These are the anchors that keep spins recognizable as \
responses to the user's query.

Include:
- named entities: actors, directors, franchises, studios, \
  characters (e.g., "Tom Cruise", "Disney", "Spider-Man")
- explicit genres or formats: "horror", "animated", "documentary"
- explicit eras: "80s", "from 2020", "recent"
- explicit platforms or constraints: "on Netflix", "under 2 hours"

Do NOT include vague or inferential phrases — those go in \
soft_interpretations.

---

SOFT_INTERPRETATIONS — how to populate

Words or phrases the user wrote that carry inferential lifting. \
Different readings of these phrases would return meaningfully \
different movies.

Include:
- evaluative words: "classics", "best", "favorites", "iconic"
- mood/tone descriptors: "feel-good", "dark", "cozy", "intense"
- occasion or audience framings: "date night", "millennial \
  favorites", "family movie"
- vague scope words: "epic", "prestige", "underrated"

Each soft interpretation is a candidate lever a spin can pull by \
committing to one specific reading.

---

OPEN_DIMENSIONS — how to populate

Axes the query does NOT touch at all. These are the blank slots \
where a spin can add a new constraint to narrow the search into \
an interesting sub-angle.

Name dimensions concretely: not "a genre" — pick "sub-genre like \
thrillers" or "lead actor" or "decade" or "tone". Only list \
dimensions that would actually produce a browseable sub-angle on \
the primary query; do not pad the list.

A highly specific query may have zero or one open dimensions. A \
vague query may have three or four. Two is typical.

---

"""

_SPIN_CONSTRUCTION = """\
CONSTRUCTING A SPIN

Each spin is one of three shapes:

- Reinterpretation — take one item from soft_interpretations and \
  commit to a specific reading. Example: "Disney classics" with \
  "classics" committed to "the Disney Renaissance animated era".

- Narrowing — take one item from open_dimensions and add a \
  concrete constraint along it. Example: "Tom Cruise 80s movies" \
  with an added sub-genre constraint → "Tom Cruise 80s thrillers".

- Adjacent swap (only when soft_interpretations is empty AND no \
  open_dimension produces a useful narrowing) — shift exactly one \
  implicit axis to an adjacent value. Example: swapping decade, \
  swapping actor within the same era/genre. Never swap more than \
  one axis at a time.

A spin preserves ALL items in hard_commitments. Dropping a hard \
commitment produces a different query, not a spin.

A spin MUST pull on exactly ONE lever. If the spin description \
reads "narrows the genre AND reinterprets the mood", it bundles \
two levers — pick one; the other belongs in the sibling spin or \
is dropped.

branching_opportunity — name the lever concretely, grounded in \
the decomposition. "Reinterprets 'classics' (from \
soft_interpretations) as the Disney Renaissance era" is concrete. \
"Explores a classic angle" is not.

query — write the spin as a full search phrase, natural enough \
that step 2 can read and decompose it. Preserve hard commitments \
verbatim. Only the lever's content changes.

---

"""

_DISTINCTNESS = """\
DISTINCTNESS — the two spins must pull different levers

The two spins must not pull on the same lever. Common failure \
modes to avoid:

- Two narrowings along the same axis. "80s action thrillers" and \
  "80s action blockbusters" are both sub-genre narrowings on the \
  same query — redesign one onto a different lever.
- Two reinterpretations of the same soft term. Both spins reading \
  "classics" differently but within the same framing (e.g., \
  "Disney Renaissance" vs "Disney golden age") are two flavors of \
  one lever, not two.
- Near-synonym reinterpretations. "Intense" vs "gripping" as spin \
  angles collapse to the same retrieval direction.

Good distinctness patterns:
- One spin reinterprets a soft term; the other narrows an open \
  dimension.
- One spin shifts entity focus (actor, director); the other \
  shifts tone.
- One spin adds a sub-genre; the other shifts era.

Distinctness check before you commit: picture the two result \
lists side by side. Would a user scrolling both immediately see \
they're exploring different things? If not, pick a different \
lever for one of them.

The distinctness field on each spin must reference what makes \
THIS spin different from BOTH the original and the sibling spin, \
stated in retrieval terms (what movies appear in this spin's \
list that wouldn't in the others).

---

"""

_UI_LABELS = """\
UI LABELS — how to write

Labels appear in the browsing UI and describe what each branch is \
about. They guide the user's eye across the three result lists.

Rules:
- 2-5 words
- Title Case
- describe what the branch is ABOUT, not the retrieval mechanics
- for spins, lean into the distinguishing lever so the label makes \
  the spin's angle visible at a glance
- a little personality is welcome — these are human-facing, not \
  sterile summaries

Do not end every label with the same noun (e.g., all three ending \
in "Movies"). Variety makes the three branches scannable.

Examples:
- original "80s action classics" → "80s Action Classics"
- spin narrowing on actor → "Schwarzenegger's 80s Peak"
- spin reinterpreting "classics" as cult → "Cult 80s Gems"
- original "movies like Inception but funnier" → "Inception, But \
  Funnier"
- original "I need to feel something" → "Something Moving"
- spin on catharsis → "Cathartic Gut-Punches"
- spin on adrenaline → "Adrenaline Rushes"

---

"""

_BOUNDARY_EXAMPLES = """\
BOUNDARY EXAMPLES

Example 1 — broad query, many productive narrowings
Query: "Best Christmas movies for families"
- hard_commitments: ["Christmas", "families"]
- soft_interpretations: ["best"]
- open_dimensions: ["format (animated vs live-action)", "era"]
- original_query_label: "Christmas Family Picks"
- spin 1:
    branching_opportunity: "Narrows 'format' (open dimension) to \
      animated — produces a fully animated lineup distinct from \
      the mixed live-action/animated original."
    distinctness: "Animated-only cuts out every live-action \
      holiday staple the original returns (Home Alone, Elf, \
      National Lampoon's), yielding a disjoint lineup."
    query: "Best animated Christmas movies for families"
    ui_label: "Animated Christmas Classics"
- spin 2:
    branching_opportunity: "Reinterprets 'best' (soft \
      interpretation) as recent-streaming-era favorites rather \
      than the all-time canon."
    distinctness: "Filters to post-2015 titles, excluding the \
      older canonical films that dominate generic 'best Christmas \
      movie' lists — returns modern holiday content the other \
      two branches miss."
    query: "Recent streaming-era Christmas movies families love"
    ui_label: "Modern Christmas Hits"

Example 2 — narrow query, tangential spins
Query: "Tom Cruise 90s action movies"
- hard_commitments: ["Tom Cruise", "90s", "action"]
- soft_interpretations: []
- open_dimensions: ["sub-genre within action"]
- original_query_label: "Cruise's 90s Action"
- spin 1:
    branching_opportunity: "Narrows 'sub-genre within action' to \
      espionage/spy-thrillers — surfaces the Mission: Impossible \
      lineage as its own focused slice."
    distinctness: "Narrows to espionage-flavored titles inside \
      Cruise's 90s action output, excluding straight-action films \
      like Far and Away or Days of Thunder that the original \
      would include."
    query: "Tom Cruise 90s spy thrillers"
    ui_label: "Cruise Spy Thrillers"
- spin 2:
    branching_opportunity: "Adjacent swap on decade (no \
      soft_interpretations to pull from, sub-genre already taken \
      by spin 1) — asks what Cruise's 2000s action looks like as \
      an adjacent exploration."
    distinctness: "Shifts the decade to 2000s, returning a \
      disjoint Cruise lineup (Collateral, War of the Worlds, M:I \
      sequels) with zero overlap with the 90s-only original or \
      spin 1."
    query: "Tom Cruise 2000s action movies"
    ui_label: "Cruise In The 2000s"

Example 3 — vague query, alternate emotional readings
Query: "I need to feel something"
- hard_commitments: []
- soft_interpretations: ["feel something"]
- open_dimensions: ["target emotion", "pacing"]
- original_query_label: "Something Moving"
- spin 1:
    branching_opportunity: "Reinterprets 'feel something' (soft) \
      as catharsis — commits to emotionally heavy, tearjerker \
      territory."
    distinctness: "Returns sad, processing-oriented films (Manchester \
      by the Sea, Marriage Story) that a generic 'moving' query \
      would blend with uplifting ones."
    query: "Cathartic emotionally heavy movies that make you cry"
    ui_label: "Cathartic Gut-Punches"
- spin 2:
    branching_opportunity: "Reinterprets 'feel something' (soft) \
      as adrenaline — commits to high-intensity thrill rides."
    distinctness: "Surfaces thrillers and action set-pieces (Mad \
      Max: Fury Road, Uncut Gems) — a disjoint emotional register \
      from spin 1's sad films and from the vague original."
    query: "High-intensity adrenaline-rush movies"
    ui_label: "Adrenaline Rushes"

Example 4 — bare title (step 0 likely drops this output, but step \
1 still runs)
Query: "Interstellar"
- hard_commitments: ["Interstellar"]
- soft_interpretations: []
- open_dimensions: ["director lineage", "sub-genre of sci-fi"]
- original_query_label: "Interstellar"
- spin 1:
    branching_opportunity: "Narrows 'director lineage' — treats \
      the query as a pull toward Nolan's cerebral sci-fi universe."
    distinctness: "Returns Nolan's broader catalog (Inception, \
      Tenet, Memento) rather than the single-title original; \
      shares the directorial voice but not the space setting."
    query: "Christopher Nolan cerebral sci-fi films"
    ui_label: "Nolan Sci-Fi"
- spin 2:
    branching_opportunity: "Narrows 'sub-genre of sci-fi' to \
      grounded hard-science space — pulls on the scientific-\
      realism thread the user may be drawn to."
    distinctness: "Returns science-accurate space films across \
      directors (The Martian, Gravity, Ad Astra), a disjoint set \
      from spin 1's director-bounded list."
    query: "Grounded hard-science space movies"
    ui_label: "Hard-Science Space"

---

"""

_OUTPUT_GUIDANCE = """\
OUTPUT FIELD GUIDANCE

Generate fields in the schema's order. The decomposition fields \
come first and ground every downstream decision.

hard_commitments — list of strings. Named entities and explicit \
genres/formats/eras/platforms the user stated verbatim. Empty if \
the query names nothing concrete.

soft_interpretations — list of strings. Evaluative or inferential \
phrases from the query. Empty if the query has none.

open_dimensions — list of strings. Concretely-named axes the \
query doesn't touch, limited to those that would produce a useful \
sub-angle. 0-4 entries typical.

original_query_label — 2-5 word Title Case UI label for the \
user's raw query.

spins — exactly two entries. Each Spin has, in order:
  branching_opportunity: names the lever (one specific item from \
    soft_interpretations or open_dimensions) and why it yields an \
    interesting branch. One or two sentences.
  distinctness: how this spin's result set differs from BOTH the \
    original and the sibling spin, in concrete retrieval terms.
  query: the full spin query, preserving hard_commitments \
    verbatim.
  ui_label: 2-5 word Title Case label.
"""


SYSTEM_PROMPT = (
    _TASK_AND_OUTCOME
    + _OBSERVATIONS_FIRST
    + _DECOMPOSITION_RULES
    + _SPIN_CONSTRUCTION
    + _DISTINCTNESS
    + _UI_LABELS
    + _BOUNDARY_EXAMPLES
    + _OUTPUT_GUIDANCE
)


# ===============================================================
#                      Executor
# ===============================================================
#
# Model is finalized to Gemini 3 Flash with thinking disabled and a
# modest temperature — the same configuration steps 0 and 2 use, so
# the three prompts share a consistent backend profile. Callers
# cannot override — keeps the step reproducible and makes
# cost/latency predictable end-to-end.


_PROVIDER = LLMProvider.GEMINI
_MODEL = "gemini-3-flash-preview"
_MODEL_KWARGS: dict = {
    "thinking_config": {"thinking_budget": 0},
    "temperature": 0.35,
}


async def run_step_1(query: str) -> tuple[Step1Response, int, int, float]:
    """Run the step-1 spin-generation LLM on a single query.

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
        response_format=Step1Response,
        model=_MODEL,
        **_MODEL_KWARGS,
    )
    elapsed = time.perf_counter() - start

    return response, input_tokens, output_tokens, elapsed


# ===============================================================
#                      CLI
# ===============================================================


def _print_response(response: Step1Response) -> None:
    """Pretty-print the structured response for terminal inspection."""
    payload = response.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the step-1 spin generator on a single query and "
            "print the structured output."
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
    response, in_tok, out_tok, elapsed = await run_step_1(args.query)
    _print_response(response)
    print(
        f"\n[tokens] input={in_tok} output={out_tok} "
        f"elapsed={elapsed:.2f}s"
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
