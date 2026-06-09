# Search V2 — Step 1: Spin Generation
#
# Runs in parallel with step 0 on the raw user query. Treats every
# input as if the standard flow will execute and produces two
# creative spins — alternative queries the user didn't type that
# broaden their browsing without losing the thread of what they
# asked for.
#
# Step 1 does NOT rewrite the original query (it passes through
# verbatim downstream), decide flow routing (step 0 owns that), or
# decompose into category-grounded atoms (step 2 owns that). Its
# scope is narrow: read the query, understand what the user is
# really after, and emit two queries that explore adjacent
# territory which a verbatim search would miss.
#
# The schema scaffolds a three-step reasoning pattern: the model
# surfaces its read of the user and candidate adjacent directions
# visibly in an `exploration` field, then commits to two refined
# spins that each trace back to a candidate from exploration.
# Earlier iterations forced the model to dissect the query into
# hard-commitment / soft-term / open-dimension buckets before
# composing spins, which biased it toward single-token tweaks
# that collapsed back onto the original result set. The freeform
# pre-generation scratchpad replaces the slotted decomposition;
# the explicit refinement step keeps the spins anchored to
# exploration content rather than being generated independently.
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
from schemas.step_1 import Step1ClarificationResponse, Step1Response
from search_v2.query_input_validation import clean_clarification, clean_query


# ===============================================================
#                      System prompt
# ===============================================================
#
# Three blocks, concatenated at module level:
#   1. Task and outcome — what the step exists to do.
#   2. How to think about spins — the principles the model applies
#      while writing the exploration field and committing spins.
#   3. Output guidance — how to emit the exploration field and the
#      two committed spins.
#
# The prompt deliberately avoids enumerated decomposition rules.
# Branching depends on what each particular query is doing, and
# the model needs the freedom to read the query holistically rather
# than fill out fixed slots that prejudge the shape of the query.


_TASK_AND_OUTCOME = """\
You are the spin-generation step in a movie search pipeline. The \
user's raw query is already being searched verbatim as one branch \
downstream. Your job is to propose two ADDITIONAL spins — \
alternative queries the user didn't think to type that explore \
adjacent territory which a verbatim search of their query would \
miss.

The goal is to open up browsing. Three branches running on \
overlapping queries return overlapping movies, which wastes two of \
the three slots. Each spin must produce a visibly different result \
set from BOTH the original query AND the sibling spin. If two \
queries would surface largely the same titles, you only have one \
useful branch, not three.

---

"""


_HOW_TO_THINK = """\
HOW TO THINK ABOUT SPINS

Reasoning happens in three steps. Each step informs the next; \
don't skip ahead to candidate spins before steps 1 and 2 are done.

Step 1 — read the user. Work out what kind of person typed this \
query and what they're really after. Ground the read in cues from \
the query itself: the vocabulary used, the anchors named, the \
level of specificity, what is left implicit. Interpret the query \
— don't restate it. The same words can mean different things \
depending on how the rest of the language frames them, and \
finding the right adjacent territory depends on getting the read \
right.

Step 2 — find adjacent territory. Given who this user is and \
what they're after, what OTHER queries would they likely also \
enjoy seeing results for? Adjacent territory means a query a \
real viewer with this taste might type as a follow-up to the \
original — a logical leap, not a paraphrase. The shape of \
"adjacent" depends on whether the original is vague or specific.

For a VAGUE query (mood, occasion, loose hook, no named entities \
or hard constraints), the original leaves many possible result \
sets open. Here, spins ARE refinements: each adds specificity \
through interesting traits — committing to a concrete reading \
the user might really be after but didn't spell out, helping \
them clarify what they meant. The value to the user is seeing \
those specific readings made tangible. Each lane should be a \
different concrete experience the original left ambiguous.

For a SPECIFIC query (named entities, multi-anchor constraints, \
comparisons to other media — anything where the user has pinned \
down what they want), the user has already done the work of \
refining. The spin is not a refinement and not a transformation \
of their text. Build each spin FROM SCRATCH, starting from the \
viewer's underlying taste — the kind of person the original \
revealed, the mood, the interest — and construct an entirely \
new search this same person would plausibly also want to run. \
The spin's relationship to the original is semantic neighborhood \
(same viewer, same general territory of films), not textual \
descent (same anchors with edits). Don't audit what to keep or \
drop from the original; ignore its words and write a sibling \
search the user might try in the same browsing session as a \
fresh alternative.

Step 3 — pressure-test redundancy. For each candidate direction, \
picture the result set it would actually retrieve and compare \
against the original's. If overlap is large, sharpen the direction \
or replace it. Then compare candidates against each other; the \
two committed spins must surface visibly different lanes both \
from the original AND from each other.

Avoid:
- Spins on specific queries that are constructed as variations \
  of the original (paraphrases, narrower slices, "the original \
  but [adjective]", "the original with one word changed"). A \
  spin built from the original's text inherits the original's \
  retrieval shape and lands in the same neighborhood. Specific \
  queries always get from-scratch siblings, never derivatives.
- Naming specific movie or show titles inside the spin query. A \
  spin is a search request, not a recommendation list. Brand, \
  studio, director, and actor names are different — they are \
  legitimate search anchors when the spin pivots on such an \
  entity as a NEW, load-bearing center the original didn't \
  mention. If the original query already named the entity, the \
  spin should drop it: the original is already searching that \
  anchor, so keeping it just preserves the same retrieval. Never \
  use enumerated example-style listings of names ("from X, Y, \
  or Z", "like the ones starring A or B") — that's the same \
  failure as naming films.
- Evaluative quality language in spin queries or labels — words \
  that ask the search for prestige, popularity, or judgment rather \
  than describe what kind of movie the user wants. They leak \
  opinion into the search, don't carry retrieval signal, and \
  collapse the spin onto a "good movies in this space" filter.
- Drifting into a query that no longer connects to what the user \
  asked for.

---

"""


_OUTPUT_GUIDANCE = """\
OUTPUT

Emit fields in order: exploration first, then the two spins.

exploration — execute the three reasoning steps for this query \
in 2-3 compact sentences with telegraphic phrasing. Surface \
multiple candidate angles openly and weigh which would genuinely \
shift the result set — do not write a labeled preview of the two \
spins ("Direction 1: X. Direction 2: Y."), because that \
pre-commits without exploring and short-circuits the reasoning.

spins — exactly two committed alternatives, each refining one of \
the candidate angles surfaced in exploration into a full search \
query. Refine, do not copy: turn the candidate angle into the \
kind of natural-language phrase a user would actually type. The \
spin's content must trace back to a candidate from exploration. \
When writing the second spin, compare its query against the \
first's: if the two would return substantially the same result \
set, refine from a different exploration candidate instead.

For each spin:
- query: full natural-language search phrase, the kind of thing \
  the user could have typed themselves. Natural enough that step \
  2 can read and decompose it.
- ui_label: short Title Case label that captures the spin's \
  angle at a glance.
"""


SYSTEM_PROMPT = _TASK_AND_OUTCOME + _HOW_TO_THINK + _OUTPUT_GUIDANCE


# ===============================================================
#               Clarification-mode system prompt
# ===============================================================
#
# Fires only when the user supplied a follow-up clarification on
# top of the original query. The step now emits THREE things:
#   1. main_rewrite — a faithful merge of original + clarification
#      that replaces the verbatim-original slot in the branch plan.
#   2 & 3. two creative spins exploring around the rewritten intent.
#
# The principles that govern main_rewrite are different from the
# principles that govern spins: main must be a faithful translation
# (no hallucinated details, no abstractions, no attempts to resolve
# descriptions into specific titles), while spins keep their
# divergence discipline but operate over the rewritten intent rather
# than the raw original. Both responsibilities are taught here so a
# single LLM call can hold both stances at once.


_CLARIFICATION_TASK_AND_OUTCOME = """\
You are the query-refinement step in a movie search pipeline. The \
user just received search results, found them off the mark, and \
supplied a follow-up clarification correcting or refining their \
original query. Your job is to emit THREE things:

1. main_rewrite — the merged search representing the user's most \
   likely intent given the original query plus the clarification. \
   This becomes the primary branch the pipeline searches.
2-3. two creative spins exploring adjacent territory the rewritten \
   intent would otherwise miss. The spins broaden browsing the same \
   way they would for a fresh query — only the source intent has \
   shifted from the raw original to the rewritten merge.

main_rewrite is faithful; the spins take creative liberties. These \
are different stances and must not bleed into each other. Refine \
main_rewrite to capture exactly what the user asked for; refine \
spins to surface what a same-taste viewer might ALSO want to see \
that the rewritten search would not return.

---

"""


_CLARIFICATION_HOW_TO_THINK = """\
HOW TO THINK ABOUT main_rewrite

main_rewrite is a faithful translation of (original + clarification) \
into one natural-language search the user could have typed if they \
had known what to ask for the first time. The discipline:

- Treat the clarification as authoritative on any conflict with the \
  original. Where they disagree, the clarification wins.
- Preserve everything in the original the clarification is silent \
  about. Silence is not retraction; only explicit contradiction or \
  replacement counts as retraction, and retracted material must be \
  dropped from the rewrite.
- When the clarification flips polarity on something present in the \
  original, carry the flipped direction into the rewrite — do not \
  drop the underlying anchor, invert its sign.
- Stay inside the user's vocabulary. If they described an experience \
  in concrete terms, keep concrete terms. Do not abstract specifics \
  into broader categories, and do not narrow general descriptors \
  into specific genres or labels they did not state.
- Do not add facts, entities, qualifiers, or details that are not \
  directly present in one of the two inputs. The rewrite is a merge, \
  not an enrichment.
- Do not try to ANSWER the search by naming specific film titles. If \
  the user described a film through plot fragments, characters, or \
  era cues, leave those descriptions in the rewrite for the database \
  to resolve. Replacing descriptive phrasing with a specific title \
  is the most damaging failure mode — it destroys the user's intent \
  by collapsing a search into a guess.
- Match the length to (original + clarification) combined. A faithful \
  merge is not a paragraph; it is the smallest natural phrasing that \
  carries both inputs' signal.

ui_label for main_rewrite is a short Title Case label following the \
same style as spin ui_labels — pithy, scannable. The UI surfaces the \
full rewrite separately, so the label does not need to convey the \
full content.

---

HOW TO THINK ABOUT spins (clarification mode)

Spins behave the same way they do without a clarification, with one \
shift: their source intent is the rewritten merge, NOT the raw \
original. The original is already retracted/refined by the \
clarification, so spinning off the original would surface results \
the user just told you to move away from.

Reasoning happens in three steps. Each step informs the next.

Step 1 — read the (rewritten) user. Work out what kind of viewer is \
making this rewritten ask. Ground the read in cues from the rewrite: \
its vocabulary, anchors, level of specificity, what is left implicit.

Step 2 — find adjacent territory around the rewrite. Given who this \
viewer is and what the rewrite is asking for, what OTHER searches \
would they likely also enjoy seeing results for? Apply the same \
vague-vs-specific logic as the no-clarification flow: vague rewrites \
get spins that commit to concrete readings the rewrite left open; \
specific rewrites get from-scratch siblings built from the underlying \
taste, not textual edits of the rewrite.

Step 3 — pressure-test redundancy. Each spin's result set must be \
visibly different from main_rewrite and from the sibling spin. If \
two would surface largely the same titles, only one is doing useful \
work.

Avoid:
- Spins that paraphrase main_rewrite or carry the same anchors. They \
  collapse onto the same result set and waste branches.
- Spins that re-introduce material the clarification retracted. The \
  user said move away from that; do not bring it back through a spin.
- Naming specific movie or show titles inside a spin query. Brand, \
  studio, director, and actor names are allowed only when the spin \
  pivots on such an entity as a NEW load-bearing center the merged \
  intent did not already name.
- Evaluative quality language in spin queries or labels (prestige, \
  popularity, judgment words). They leak opinion into the search, \
  carry no retrieval signal, and collapse the spin onto a "good \
  movies in this space" filter.

---

"""


_CLARIFICATION_OUTPUT_GUIDANCE = """\
OUTPUT

Emit fields in order: exploration, then main_rewrite, then the two \
spins.

exploration — 2-3 compact telegraphic sentences that (a) read how \
the clarification reshapes the original (additions, retractions, \
polarity flips), (b) sketch the rewritten intent in plain words, \
(c) surface candidate adjacent angles for spins. Do not write a \
labeled preview of the spins; surface multiple candidate angles \
openly and weigh which would genuinely diverge from the rewrite.

main_rewrite — the faithful merge, with:
- query: full natural-language search phrase the user could have \
  typed themselves, applying the main_rewrite discipline above.
- ui_label: short Title Case label, spin-style.

spins — exactly two committed alternatives, each refining one of the \
candidate angles surfaced in exploration. Refine, do not copy. The \
spin's content must trace back to a candidate from exploration. \
Compare the second spin's query against the first's — if they would \
return substantially the same result set, refine from a different \
exploration candidate instead.

For each spin:
- query: full natural-language search phrase, the kind of thing the \
  user could have typed themselves. Natural enough that step 2 can \
  read and decompose it.
- ui_label: short Title Case label that captures the spin's angle at \
  a glance.
"""


CLARIFICATION_SYSTEM_PROMPT = (
    _CLARIFICATION_TASK_AND_OUTCOME
    + _CLARIFICATION_HOW_TO_THINK
    + _CLARIFICATION_OUTPUT_GUIDANCE
)


# ===============================================================
#                      Executor
# ===============================================================
#
# Thinking is set to `minimal` — the model's pre-generation
# reasoning happens visibly in the `exploration` field of the
# response, which acts as the chain-of-thought scaffold for
# committing the two spins. Spending tokens on hidden thinking
# on top of a freeform visible scratchpad is redundant. Callers
# cannot override the model config — keeps the step reproducible
# and makes cost/latency predictable end-to-end.


_PROVIDER = LLMProvider.GEMINI
_MODEL = "gemini-3.5-flash"
_MODEL_KWARGS: dict = {
    "thinking_config": {"thinking_level": "minimal"},
}


async def run_step_1(
    query: str,
    clarification: str | None = None,
) -> tuple[Step1Response | Step1ClarificationResponse, int, int, float]:
    """Run the step-1 spin-generation LLM on a single query.

    Args:
        query: the raw user query.
        clarification: optional follow-up clarification text. When
            present, swaps in the clarification-mode prompt and schema:
            the response carries a main_rewrite slot (faithful merge of
            original + clarification) alongside the two creative spins.
            When None, behavior is identical to the no-clarification
            path and the response is a Step1Response.

    Returns:
        (response, input_tokens, output_tokens, elapsed_seconds) —
        elapsed measures wall-clock time spent inside the LLM call
        only, not prompt setup. The response type discriminates on
        clarification presence.
    """
    # Validate/normalize at the boundary: strip, enforce non-empty +
    # length cap. Shared with every other public surface so the rules
    # live in one place (see search_v2/query_input_validation.py).
    query = clean_query(query)
    clarification = clean_clarification(clarification)

    if clarification:
        # Two labeled fields keep precedence resolution deterministic —
        # the prompt teaches "clarification wins on conflict" and the
        # model has to know which is which to apply that.
        user_prompt = (
            f"Original query: {query}\n"
            f"Clarification: {clarification}"
        )
        system_prompt = CLARIFICATION_SYSTEM_PROMPT
        response_format: type[Step1Response] | type[Step1ClarificationResponse] = (
            Step1ClarificationResponse
        )
    else:
        user_prompt = f"Query: {query}"
        system_prompt = SYSTEM_PROMPT
        response_format = Step1Response

    # perf_counter: monotonic, high-res wall-clock for short intervals.
    start = time.perf_counter()
    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=_PROVIDER,
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        response_format=response_format,
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
