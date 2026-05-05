from __future__ import annotations

import argparse
import asyncio
import json
import time
from xml.sax.saxutils import escape as xml_escape

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)
from schemas.implicit_expectations import ImplicitExpectationsResult
from schemas.step_2 import QueryAnalysis, Trait
from search_v2.step_2 import run_step_2


# Search V2 — Implicit Expectations
#
# Runs after Step 2 and decides how much implicit quality/popularity
# prior pressure remains after the user's committed traits have spoken.
# This module intentionally reasons from QueryAnalysis.traits rather
# than re-deriving criteria from the raw query. The raw query is passed
# only for provenance and exact-phrasing disambiguation.
#
# Usage:
#   python -m search_v2.implicit_expectations "your query here"


# ===============================================================
#                      System prompt
# ===============================================================


_TASK_AND_OUTCOME = """\
You are the implicit-prior policy step in a movie search pipeline.

You receive:
- the raw user query for provenance only
- Step 2's intent_exploration
- Step 2's committed trait list

Your job is to determine the direction and strength of the implicit
quality prior and the implicit popularity prior that remain AFTER the
committed traits are respected.

The implicit priors represent default user expectations that were not
fully captured by explicit query wording:
- quality: general goodness, reception, acclaim, taste/value judgment,
  watch-worthiness, or avoiding low-quality results
- popularity: mainstream reach, fame, cultural familiarity, obscurity,
  hiddenness, or how well-known the result should be

Do not invent new criteria from the raw query. Treat Step 2's traits as
the closed set of explicit user intent. Use the raw query only to
preserve exact wording and resolve ambiguity in Step-2 fields.

---

"""

_OBSERVATIONS_FIRST = """\
OBSERVATIONS FIRST, DECISIONS SECOND

Fill the schema top to bottom:

1. query_intent_summary
2. explicit_signals
3. ordering_axis_analysis
4. query_specificity_analysis
5. quality_prior
6. popularity_prior

The final prior decisions must be direct consequences of the preceding
analysis. Do not jump to an answer and backfill explanations.

---

"""

_TRAIT_SIGNAL_RULES = """\
PER-TRAIT PRIOR SIGNALS

Produce exactly one explicit_signals row for every Step-2 committed
trait, in the same order provided.

For each trait, analyze the trait as Step 2 committed it:
- surface_text gives the user phrase
- evaluative_intent gives the integrated meaning
- qualifier_relation and anchor_reference say whether the trait is
  standalone, comparative, positioning, or only qualifying another ask
- polarity gives direction when the trait names an avoid/prefer-less
  axis
- commitment and commitment_evidence say how strongly this explicit
  trait should own ranking pressure

Classify explicit_axis:
- quality only when the trait itself speaks to goodness, badness,
  reception, acclaim, taste/value judgment, watch-worthiness, or
  avoiding low-quality results
- popularity only when the trait itself speaks to mainstreamness, fame,
  obscurity, hiddenness, cultural familiarity, or how well-known the
  results should be
- both only when the single trait bundles both axes
- neither for ordinary content, tone, era, entity, availability,
  format, occasion, or style traits

Classify direction:
- positive when the trait asks for more of the axis
- inverse when the trait asks for less of the axis
- none when explicit_axis is neither

Classify coverage:
- direct when this trait plainly names the axis and should fully
  replace an implicit prior on that axis
- partial when the trait shades into the axis but does not fully own it
- none when the trait does not cover quality or popularity

Keep specificity separate from prior coverage. A highly specific
non-prior trait may reduce prior_room later, but it is not itself an
explicit quality/popularity signal.

---

"""

_ORDERING_AXIS_RULES = """\
ORDERING AXIS ANALYSIS

Analyze whether the committed traits ask for a ranking order that is
different from generic quality or generic popularity.

Recognize ordering axes by function:
- chronology orders by time or sequence
- trending orders by current attention rather than static popularity
- semantic_extremeness orders by intensity on a requested semantic axis
- obscurity orders away from mainstream familiarity
- quality or popularity orders explicitly by one of the two prior axes
- other covers a real ordering objective not named above
- none means there are constraints/preferences but no explicit ranking
  order

Suppression is not automatic. A separate ordering axis should suppress
a prior only when that prior would fight the requested order rather
than serve as tie pressure among similarly relevant candidates.

---

"""

_SPECIFICITY_RULES = """\
QUERY SPECIFICITY AND PRIOR ROOM

Decide how much room remains for hidden priors after explicit traits
have spoken.

Use low explicit_trait_pressure when the query is broad, has few
committed traits, or leaves many equally plausible candidates.

Use medium explicit_trait_pressure when the query has a few meaningful
criteria but still leaves broad ranking discretion.

Use high explicit_trait_pressure when multiple traits are required or
elevated, when traits are highly specific, or when the query's own
ordering objective should dominate.

prior_room is the allowed strength envelope:
- high: broad discovery query; hidden quality/popularity expectations
  can meaningfully shape ordering
- normal: some explicit constraints, but default watch-worthiness or
  mainstream usefulness still matters
- light: explicit traits should dominate; priors should mostly break
  ties
- none: priors would duplicate or fight explicit intent

---

"""

_FINAL_DECISION_RULES = """\
FINAL PRIOR DECISIONS

For each axis independently, choose direction and strength.

Use direction=none and strength=none when:
- direct explicit coverage already owns that axis
- an ordering axis suppresses that prior
- prior_room is none
- applying the prior would fight the user's committed traits

Use direction=positive when the remaining implicit expectation is to
prefer higher quality or higher popularity.

Use direction=inverse when the committed traits imply an underseen,
obscure, less-mainstream, anti-prestige, or otherwise lower-axis
preference that should be represented as a prior rather than as
generic positive lift.

Strength must respect prior_room. Do not choose a strength above the
room available. When a trait provides partial explicit coverage on an
axis, reduce the implicit strength rather than treating the axis as
fully uncovered.

Final decisions are policy outputs only. Do not describe endpoint
routing, SQL, vector spaces, or score formulas.

---

"""

SYSTEM_PROMPT = "".join(
    [
        _TASK_AND_OUTCOME,
        _OBSERVATIONS_FIRST,
        _TRAIT_SIGNAL_RULES,
        _ORDERING_AXIS_RULES,
        _SPECIFICITY_RULES,
        _FINAL_DECISION_RULES,
    ]
)


# ===============================================================
#                      Prompt building
# ===============================================================


def build_user_prompt(raw_query: str, step_2: QueryAnalysis) -> str:
    """Serialize the query and Step-2 committed traits as XML."""
    parts = [
        _wrap_leaf("raw_query_for_provenance_only", raw_query),
        _wrap_leaf("intent_exploration", step_2.intent_exploration),
        _serialize_traits(step_2.traits),
    ]
    return "\n".join(parts)


def _wrap_leaf(tag: str, text: str) -> str:
    return f"<{tag}>{xml_escape(text)}</{tag}>"


def _serialize_traits(traits: list[Trait]) -> str:
    if not traits:
        return "<traits></traits>"

    blocks = "\n".join(_serialize_trait(trait) for trait in traits)
    return f"<traits>\n{blocks}\n</traits>"


def _serialize_trait(trait: Trait) -> str:
    lines = [
        "  <trait>",
        f"    <surface_text>{xml_escape(trait.surface_text)}</surface_text>",
        (
            "    <contextualized_phrase>"
            f"{xml_escape(trait.contextualized_phrase)}"
            "</contextualized_phrase>"
        ),
        (
            "    <evaluative_intent>"
            f"{xml_escape(trait.evaluative_intent)}"
            "</evaluative_intent>"
        ),
        (
            "    <qualifier_relation>"
            f"{xml_escape(trait.qualifier_relation)}"
            "</qualifier_relation>"
        ),
        (
            "    <anchor_reference>"
            f"{xml_escape(trait.anchor_reference)}"
            "</anchor_reference>"
        ),
        f"    <polarity>{xml_escape(trait.polarity.value)}</polarity>",
        f"    <commitment>{xml_escape(trait.commitment)}</commitment>",
        (
            "    <commitment_evidence>"
            f"{xml_escape(trait.commitment_evidence)}"
            "</commitment_evidence>"
        ),
        "  </trait>",
    ]
    return "\n".join(lines)


# ===============================================================
#                      Execution
# ===============================================================


_PROVIDER = LLMProvider.GEMINI
_MODEL = "gemini-3-flash-preview"
_MODEL_KWARGS: dict = {
    "thinking_config": {"thinking_budget": 0},
    "temperature": 0.35,
}


async def run_implicit_expectations(
    raw_query: str,
    step_2: QueryAnalysis,
) -> tuple[ImplicitExpectationsResult, int, int, float]:
    """Run the implicit-prior policy LLM on one query + Step-2 result."""
    raw_query = raw_query.strip()
    if not raw_query:
        raise ValueError("raw_query must be a non-empty string.")

    user_prompt = build_user_prompt(raw_query, step_2)

    start = time.perf_counter()
    response, input_tokens, output_tokens = await generate_llm_response_async(
        provider=_PROVIDER,
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PROMPT,
        response_format=ImplicitExpectationsResult,
        model=_MODEL,
        **_MODEL_KWARGS,
    )
    elapsed = time.perf_counter() - start

    if len(response.explicit_signals) != len(step_2.traits):
        raise ValueError(
            "Implicit expectations output must contain exactly one explicit "
            "signal per Step-2 committed trait."
        )
    for signal, trait in zip(
        response.explicit_signals,
        step_2.traits,
        strict=True,
    ):
        if signal.query_span != trait.surface_text:
            raise ValueError(
                "Implicit expectations output must preserve Step-2 trait order "
                "and copy each trait surface_text verbatim into query_span."
            )
    return response, input_tokens, output_tokens, elapsed


# ===============================================================
#                      CLI
# ===============================================================


def _print_response(response: ImplicitExpectationsResult) -> None:
    payload = response.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Step 2 followed by the implicit-prior policy step "
            "on a single query and print the structured output."
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
    step_2_response, step_2_in, step_2_out, step_2_elapsed = await run_step_2(
        args.query
    )
    response, in_tok, out_tok, elapsed = await run_implicit_expectations(
        args.query,
        step_2_response,
    )
    _print_response(response)
    print(
        f"\n[step_2 tokens] input={step_2_in} output={step_2_out} "
        f"elapsed={step_2_elapsed:.2f}s"
    )
    print(
        f"[implicit_expectations tokens] input={in_tok} output={out_tok} "
        f"elapsed={elapsed:.2f}s"
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
