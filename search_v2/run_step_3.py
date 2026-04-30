# Search V2 — Step 3 runner
#
# Thin CLI wrapper that runs Step 2 on a query, then fans out Step 3
# in parallel across the resulting traits and prints each
# decomposition. Mirrors the shape of `search_v2/run_step_2.py` so
# the two scripts feel symmetric for prototyping/eval.
#
# Usage:
#   python -m search_v2.run_step_3
#   python -m search_v2.run_step_3 "your query here"

from __future__ import annotations

import argparse
import asyncio
import json

from search_v2.step_2 import run_step_2
from search_v2.step_3 import run_step_3
from schemas.step_2 import QueryAnalysis, Trait
from schemas.step_3 import TraitDecomposition


# Default query — same one used by run_step_2.py so behavior stays
# comparable across runs (multiple language types in one call: role
# marker, polarity, chronological, multi-dimension entity).
_DEFAULT_QUERY = (
    "first Indiana Jones movie starring Harrison Ford, "
    "not too violent, preferably from the 1980s"
)


def _print_step_2_response(response: QueryAnalysis) -> None:
    """Pretty-print the Step 2 QueryAnalysis payload."""
    payload = response.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))



def _print_trait_decomposition(
    trait: Trait,
    decomposition: TraitDecomposition,
    in_tok: int,
    out_tok: int,
    elapsed: float,
) -> None:
    """Pretty-print a single trait's Step 3 decomposition with a
    header tying it back to the source trait's surface_text and the
    LLM call's per-trait timing and token usage."""
    print(f'\n--- Trait: "{trait.surface_text}" ---')
    print(
        f"[step 3 call] elapsed={elapsed:.2f}s "
        f"input_tokens={in_tok} output_tokens={out_tok}"
    )
    print(
        "[trait inputs]"
        f'\n  contextualized_phrase: "{trait.contextualized_phrase}"'
        f"\n  evaluative_intent: {trait.evaluative_intent}"
        f"\n  role_evidence: {trait.role_evidence}"
        f"\n  role: {trait.role}"
        f"\n  qualifier_relation: {trait.qualifier_relation}"
        f"\n  anchor_reference: {trait.anchor_reference}"
        f"\n  polarity: {trait.polarity}"
        f"\n  relevance_to_query: {trait.relevance_to_query}"
    )
    print("[decomposition]")
    payload = decomposition.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run search_v2.step_2 and search_v2.step_3 end-to-end on "
            "a query and print the analysis, per-trait decompositions, "
            "elapsed time, and token usage."
        )
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=_DEFAULT_QUERY,
        help=(
            "The raw user query to process. Defaults to a built-in "
            "sample query if omitted."
        ),
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = _parse_args()
    print(f"[query] {args.query}\n")

    # Step 2 first — we need traits before Step 3 can run.
    # Sequential by necessity (Step 3 fans out over Step 2's output).
    print("[step 2 response]")
    analysis, step_2_in, step_2_out, step_2_elapsed = await run_step_2(
        args.query
    )
    _print_step_2_response(analysis)

    # Step 3 fanned out across traits. Per-trait calls are independent
    # so we run them concurrently with asyncio.gather.
    print("\n[step 3 responses]")
    if not analysis.traits:
        print("(no traits emitted by Step 2 — nothing to decompose)")
        step_3_in_total = 0
        step_3_out_total = 0
        step_3_elapsed_max = 0.0
    else:
        results = await asyncio.gather(
            *(run_step_3(trait) for trait in analysis.traits)
        )
        # results[i] aligns with analysis.traits[i] because gather
        # preserves the order of its awaitables.
        step_3_in_total = 0
        step_3_out_total = 0
        step_3_elapsed_max = 0.0
        for trait, (decomp, in_tok, out_tok, elapsed) in zip(
            analysis.traits, results
        ):
            _print_trait_decomposition(
                trait, decomp, in_tok, out_tok, elapsed
            )
            step_3_in_total += in_tok
            step_3_out_total += out_tok
            # gather runs the per-trait calls concurrently, so the
            # wall-clock cost of the fan-out is the slowest call,
            # not the sum.
            if elapsed > step_3_elapsed_max:
                step_3_elapsed_max = elapsed

    in_total = step_2_in + step_3_in_total
    out_total = step_2_out + step_3_out_total
    print(
        "\n[stats]"
        f"\n  step 2:   elapsed={step_2_elapsed:.2f}s "
        f"input_tokens={step_2_in} output_tokens={step_2_out}"
        f"\n  step 3:   wallclock={step_3_elapsed_max:.2f}s "
        "(slowest of parallel calls) "
        f"input_tokens={step_3_in_total} "
        f"output_tokens={step_3_out_total}"
        f"\n  totals:   input_tokens={in_total} output_tokens={out_total}"
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
