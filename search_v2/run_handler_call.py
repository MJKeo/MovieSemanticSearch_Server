# Search V2 — focused per-handler tester.
#
# Skips Step 2 + Step 3 entirely and lets you inject a CategoryCall
# directly. Useful when iterating on the body-authoring side of the
# semantic / keyword / metadata handlers — Step 2/3 are deterministic
# variability we don't want when we're trying to verify what the
# handler emits for a fixed (category, expressions, retrieval_intent)
# triple.
#
# Reuses the helpers in run_query_generation.py rather than
# duplicating them. The whole point is to exercise the same code path
# the production handler runs.
#
# Usage:
#   python -m search_v2.run_handler_call \
#     --category EMOTIONAL_EXPERIENTIAL \
#     --expression "haunting bittersweet drama" \
#     --retrieval-intent "Films with a haunting, bittersweet during-viewing feel."
#
#   # Multiple expressions (one --expression flag each):
#   python -m search_v2.run_handler_call \
#     --category NARRATIVE_DEVICES \
#     --expression "non-linear timelines with flashbacks" \
#     --expression "unreliable narrators" \
#     --retrieval-intent "Films told via non-linear chronology with flashbacks and unreliable narrators."

from __future__ import annotations

import argparse
import asyncio

from schemas.enums import Polarity
from schemas.step_2 import Trait
from schemas.step_3 import CategoryCall
from schemas.trait_category import CategoryName

# Reuse — do not duplicate.
from search_v2.run_query_generation import (
    _dump_pydantic,
    _dump_spec,
    _run_handler_with_full_output,
)


def _build_synthetic_trait(
    expressions: list[str], retrieval_intent: str, polarity: Polarity
) -> Trait:
    """Build a Trait whose only load-bearing field is `polarity`.

    The handler reads `polarity` (via determine_operation_type to
    pick candidate_generator vs pool_reranker) and ignores the rest
    of the Trait's content. The other fields are populated with
    values that satisfy Pydantic validation but carry no semantic
    weight at the handler stage.
    """
    head = expressions[0] if expressions else "(injected)"
    return Trait(
        surface_text=head,
        evaluative_intent=retrieval_intent,
        qualifier_relation="n/a",
        anchor_reference="n/a",
        polarity=polarity,
        commitment_evidence="(injected for handler test — bypassing Step 2)",
        commitment="neutral",
        contextualized_phrase=head,
    )


async def _main_async() -> None:
    args = _parse_args()

    # Resolve the CategoryName from the string. Lets the user pass
    # the enum member name (e.g. EMOTIONAL_EXPERIENTIAL) on the CLI.
    try:
        category = CategoryName[args.category]
    except KeyError as exc:
        valid = ", ".join(sorted(c.name for c in CategoryName))
        raise SystemExit(
            f"Unknown category '{args.category}'. Valid: {valid}"
        ) from exc

    polarity = Polarity(args.polarity)

    category_call = CategoryCall(
        category=category,
        expressions=args.expression,
        retrieval_intent=args.retrieval_intent,
    )
    trait = _build_synthetic_trait(
        args.expression, args.retrieval_intent, polarity
    )

    print(f"[category]         {category.name}")
    print(f"[bucket]           {category.bucket.value}")
    print(f"[expressions]      {args.expression}")
    print(f'[retrieval_intent] "{args.retrieval_intent}"')
    print(f"[polarity]         {polarity.value}")
    print()

    raw_output, specs = await _run_handler_with_full_output(
        category_call=category_call, trait=trait
    )

    # Raw handler-LLM output — every reasoning field the LLM produced.
    # None for deterministic / no-op categories.
    print("========== RAW HANDLER OUTPUT ==========")
    if raw_output is None:
        print(
            "(deterministic / no-op category — no LLM call, no reasoning fields)"
        )
    else:
        print(_dump_pydantic(raw_output))

    # Extracted specs — what stage 4 would actually fire.
    print("\n========== EXTRACTED SPECS ==========")
    if not specs:
        print("(none — handler abstained)")
    else:
        for i, spec in enumerate(specs):
            print(f"\n[spec {i}]")
            print(_dump_spec(spec))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the per-handler stage with an injected CategoryCall, "
            "skipping Step 2 + Step 3. Prints the full raw handler "
            "output (every reasoning field) plus the extracted endpoint "
            "specs."
        )
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        help=(
            "CategoryName enum member (e.g. EMOTIONAL_EXPERIENTIAL, "
            "PLOT_EVENTS, FILMING_LOCATION)."
        ),
    )
    parser.add_argument(
        "--expression",
        type=str,
        action="append",
        required=True,
        help=(
            "One expression for CategoryCall.expressions. Pass multiple "
            "--expression flags to supply more than one."
        ),
    )
    parser.add_argument(
        "--retrieval-intent",
        type=str,
        required=True,
        help="The retrieval_intent string for the CategoryCall.",
    )
    parser.add_argument(
        "--polarity",
        type=str,
        choices=("positive", "negative"),
        default="positive",
        help="Trait polarity. Defaults to 'positive'.",
    )
    return parser.parse_args()


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
