from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Iterable
from xml.sax.saxutils import escape as xml_escape

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)
from schemas.implicit_expectations import ImplicitExpectationsResult
from schemas.step_2 import CoverageEvidence, Modifier, RequirementFragment, Step2Response
from search_v2.step_2 import run_step_2


# Search V2 — Implicit Expectations
#
# Runs after Step 2 and decides whether the query still leaves any
# implicit quality/notability gap that reranking should backfill.
# The module is intentionally narrow: it does not rewrite the query,
# route endpoints, or score movies. It only classifies the explicit
# Step-2 fragments and reports whether implicit priors still matter.
#
# Usage:
#   python -m search_v2.implicit_expectations "your query here"


# ===============================================================
#                      System prompt
# ===============================================================


_TASK_AND_OUTCOME = """\
You are the implicit-expectations step in a movie search pipeline.

You receive:
- the raw user query
- Step-2's fragment-by-fragment decomposition of that query

Your job is to determine whether the system should still apply an
implicit quality prior, an implicit notability prior, both, or neither.

Work in this order:
1. summarize the query's ranking intent at a high level
2. classify each Step-2 fragment for whether it explicitly addresses
   quality, notability, both, or neither
3. analyze whether the query explicitly asks for some OTHER ordering
   axis beyond quality/notability
4. then make the final booleans

Your job is NOT:
- to infer quality or notability from unstated cultural defaults at the
  fragment level
- to invent extra fragments not present in Step 2
- to merge multiple requirement fragments into one signal row
- to restate the whole query instead of classifying each fragment
- to force quality/notability labels onto fragments that are really
  about some other ordering criterion like trending, chronology, or
  semantic extremeness

---

"""

_OBSERVATIONS_FIRST = """\
OBSERVATIONS FIRST, DECISIONS SECOND

Fill the schema top to bottom.

First the exploratory fields:
- query_intent_summary
- explicit_signals
- explicit_ordering_axis_analysis

Then the boolean summaries and decisions:
- explicitly_addresses_quality
- explicitly_addresses_notability
- should_apply_quality_prior
- should_apply_notability_prior

The booleans must be a function of the earlier fields, not an
independent second guess.

---

"""

_SIGNAL_RULES = """\
EXPLICIT SIGNALS

Produce exactly one row for every Step-2 requirement fragment, in the
same order they are provided.

For each row:
1. Read the fragment plus its modifiers and coverage evidence.
2. Write normalized_description first as a short phrase describing
   what the fragment is asking for.
3. Then classify explicit_axis:
   - quality
   - notability
   - both
   - neither

Use 'quality' only when the fragment explicitly addresses the
goodness/badness/reception/value judgment of the movies.
Typical examples: worst, critically acclaimed, well reviewed, trashy,
masterpiece, so-bad-it's-good.

Use 'notability' only when the fragment explicitly addresses how
well-known, mainstream, obscure, hidden, or culturally familiar the
movies should be. Typical examples: underrated, obscure, mainstream,
blockbusters, everyone knows, lesser-known.

Use 'both' only when the fragment itself explicitly carries both
quality and notability in one bundle. Typical examples: classics,
best, hidden gems, must-see classics, iconic masterpieces.

Use 'neither' for every remaining fragment, including:
- genre
- tone
- plot
- era
- entity constraints
- format
- occasion
- stylistic or thematic preferences
- explicit ordering language like "trending", "most recent", or
  "scariest" that should be discussed in the top-level ordering
  analysis rather than labeled quality/notability here

Important: specificity alone does NOT make something quality or
notability. "Scary", "surreal", "political paranoia", "1970s", and
"comedies" are all 'neither' unless the fragment explicitly adds
quality/notability language. "Prestige" and "arthouse" by themselves
stay 'neither' unless the fragment directly adds a quality or
notability claim.

---

"""

_ORDERING_ANALYSIS_RULES = """\
EXPLICIT ORDERING AXIS ANALYSIS

Write a short analysis of whether the query explicitly asks to rank or
order results by something other than quality/notability.

Common examples:
- trending / popular right now
- chronology / most recent / earliest / in order
- semantic extremeness such as scariest, funniest, most disturbing,
  most romantic

If such an axis is present, name it plainly and say that it should take
precedence over default quality/notability priors.
If no such axis is present, say that directly.

This field is not about generic constraints like genre, era, or named
entities. Those do NOT count as explicit ordering axes by themselves.

---

"""

_FINAL_BOOLEAN_RULES = """\
FINAL BOOLEANS

- explicitly_addresses_quality is true iff at least one explicit_signals
  row is quality or both.
- explicitly_addresses_notability is true iff at least one
  explicit_signals row is notability or both.
- should_apply_quality_prior must be false whenever
  explicitly_addresses_quality is true.
- should_apply_notability_prior must be false whenever
  explicitly_addresses_notability is true.

If quality/notability is not already explicit, then decide the
should_apply_* booleans based on the ordering analysis above:
- if some other explicit ordering axis should drive ranking, the
  corresponding priors should be false
- if no other explicit ordering axis is present, the corresponding
  priors should be true

---

EXAMPLES

- "comedies"
  query_intent_summary: broad comedy discovery request
  explicit_signals: "comedies" -> neither
  explicit_ordering_axis_analysis: no explicit ordering axis beyond the
  basic request
  explicitly_addresses_quality=false
  explicitly_addresses_notability=false
  should_apply_quality_prior=true
  should_apply_notability_prior=true

- "best comedies"
  explicit_signals: "best" -> both, "comedies" -> neither
  explicit_ordering_axis_analysis: no separate ordering axis beyond the
  explicit best-ness request already captured in the signal rows
  explicitly_addresses_quality=true
  explicitly_addresses_notability=true
  should_apply_quality_prior=false
  should_apply_notability_prior=false

- "hidden gem comedies"
  explicit_signals: "hidden gem" -> both, "comedies" -> neither
  explicitly_addresses_quality=true
  explicitly_addresses_notability=true
  should_apply_quality_prior=false
  should_apply_notability_prior=false

- "trending comedies"
  explicit_signals: "trending" -> neither, "comedies" -> neither
  explicit_ordering_axis_analysis: explicit trending ordering axis
  explicitly_addresses_quality=false
  explicitly_addresses_notability=false
  should_apply_quality_prior=false
  should_apply_notability_prior=false

- "most recent horror movies"
  explicit_signals: "most recent" -> neither, "horror" -> neither
  explicit_ordering_axis_analysis: explicit chronology ordering axis
  explicitly_addresses_quality=false
  explicitly_addresses_notability=false
  should_apply_quality_prior=false
  should_apply_notability_prior=false

- "scariest horror movies"
  explicit_signals: "scariest" -> neither, "horror" -> neither
  explicit_ordering_axis_analysis: explicit semantic extremeness
  ordering axis
  explicitly_addresses_quality=false
  explicitly_addresses_notability=false
  should_apply_quality_prior=false
  should_apply_notability_prior=false

- "prestige thrillers"
  explicit_signals: "prestige" -> neither, "thrillers" -> neither
  explicit_ordering_axis_analysis: no explicit ordering axis beyond the
  descriptive request
  explicitly_addresses_quality=false
  explicitly_addresses_notability=false
  should_apply_quality_prior=true
  should_apply_notability_prior=true

---

"""

_GROUNDING_RULES = """\
GROUNDING RULES

- Prefer the fragment's exact meaning over the raw-query vibe.
- Use modifiers when they change what the fragment means.
- Coverage evidence exists to help you understand the fragment; do
  not copy category names into the answer unless they genuinely help
  the normalized description.
- Do not let one fragment's explicit quality/notability wording spill
  into neighboring rows that do not contain it.

---

"""

SYSTEM_PROMPT = "".join(
    [
        _TASK_AND_OUTCOME,
        _OBSERVATIONS_FIRST,
        _SIGNAL_RULES,
        _ORDERING_ANALYSIS_RULES,
        _FINAL_BOOLEAN_RULES,
        _GROUNDING_RULES,
    ]
)


# ===============================================================
#                      Prompt building
# ===============================================================


def build_user_prompt(raw_query: str, step_2: Step2Response) -> str:
    """Serialize the query and Step-2 output as an XML payload."""
    parts = [
        _wrap_leaf("raw_query", raw_query),
        _wrap_leaf(
            "overall_query_intention_exploration",
            step_2.overall_query_intention_exploration,
        ),
        _serialize_requirements(step_2.requirements),
    ]
    return "\n".join(parts)


def _wrap_leaf(tag: str, text: str) -> str:
    return f"<{tag}>{xml_escape(text)}</{tag}>"


def _serialize_requirements(
    requirements: Iterable[RequirementFragment],
) -> str:
    reqs = list(requirements)
    if not reqs:
        return "<requirements></requirements>"

    blocks = "\n".join(_serialize_requirement(req) for req in reqs)
    return f"<requirements>\n{blocks}\n</requirements>"


def _serialize_requirement(requirement: RequirementFragment) -> str:
    lines = [
        "  <requirement>",
        f"    <query_text>{xml_escape(requirement.query_text)}</query_text>",
        f"    <description>{xml_escape(requirement.description)}</description>",
        _serialize_modifiers(requirement.modifiers, indent="    "),
        _serialize_coverage_evidence(requirement.coverage_evidence, indent="    "),
        "  </requirement>",
    ]
    return "\n".join(lines)


def _serialize_modifiers(modifiers: Iterable[Modifier], indent: str) -> str:
    mods = list(modifiers)
    if not mods:
        return f"{indent}<modifiers></modifiers>"

    lines = [f"{indent}<modifiers>"]
    for modifier in mods:
        lines.append(f"{indent}  <modifier>")
        lines.append(f"{indent}    <type>{xml_escape(modifier.type.value)}</type>")
        lines.append(
            f"{indent}    <original_text>"
            f"{xml_escape(modifier.original_text)}"
            f"</original_text>"
        )
        lines.append(
            f"{indent}    <effect>{xml_escape(modifier.effect)}</effect>"
        )
        lines.append(f"{indent}  </modifier>")
    lines.append(f"{indent}</modifiers>")
    return "\n".join(lines)


def _serialize_coverage_evidence(
    evidence: Iterable[CoverageEvidence],
    indent: str,
) -> str:
    entries = list(evidence)
    if not entries:
        return f"{indent}<coverage_evidence></coverage_evidence>"

    lines = [f"{indent}<coverage_evidence>"]
    for entry in entries:
        lines.append(f"{indent}  <entry>")
        lines.append(
            f"{indent}    <captured_meaning>"
            f"{xml_escape(entry.captured_meaning)}"
            f"</captured_meaning>"
        )
        lines.append(
            f"{indent}    <category_name>{xml_escape(entry.category_name.value)}</category_name>"
        )
        lines.append(
            f"{indent}    <fit_quality>{xml_escape(entry.fit_quality.value)}</fit_quality>"
        )
        lines.append(
            f"{indent}    <atomic_rewrite>"
            f"{xml_escape(entry.atomic_rewrite)}"
            f"</atomic_rewrite>"
        )
        lines.append(f"{indent}  </entry>")
    lines.append(f"{indent}</coverage_evidence>")
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
    step_2: Step2Response,
) -> tuple[ImplicitExpectationsResult, int, int, float]:
    """Run the implicit-expectations LLM on one query + Step-2 result."""
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

    if len(response.explicit_signals) != len(step_2.requirements):
        raise ValueError(
            "Implicit expectations output must contain exactly one explicit "
            "signal per Step-2 requirement fragment."
        )
    for signal, requirement in zip(
        response.explicit_signals,
        step_2.requirements,
        strict=True,
    ):
        if signal.query_span != requirement.query_text:
            raise ValueError(
                "Implicit expectations output must preserve Step-2 "
                "requirement order and copy each fragment query_text "
                "verbatim into query_span."
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
            "Run Step 2 followed by the implicit-expectations step "
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
