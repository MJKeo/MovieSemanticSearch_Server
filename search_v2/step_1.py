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
from schemas.step_1 import Step1Response


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
