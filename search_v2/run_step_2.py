# Search V2 — Step 2 runner
#
# Thin CLI wrapper around `search_v2.step_2.run_step_2` that prints the
# full structured response plus timing and token usage. The query is
# accepted as a positional argument with a built-in default so the
# script can be invoked with no arguments for a quick smoke test.
#
# Usage:
#   python -m search_v2.run_step_2
#   python -m search_v2.run_step_2 "your query here"

from __future__ import annotations

import argparse
import asyncio
import json

from search_v2.step_2 import run_step_2
from schemas.step_2 import QueryAnalysis


# Default query used when the user invokes the script with no
# argument. Picked to exercise multiple language types (role marker,
# polarity, chronological, multi-dimension entity) in one call.
_DEFAULT_QUERY = (
    "first Indiana Jones movie starring Harrison Ford, "
    "not too violent, preferably from the 1980s"
)


def _print_response(response: QueryAnalysis) -> None:
    """Pretty-print the full QueryAnalysis payload as indented JSON."""
    payload = response.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run search_v2.step_2 on a query and print the full "
            "response, elapsed time, and token usage."
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

    response, in_tok, out_tok, elapsed = await run_step_2(args.query)

    print("[response]")
    _print_response(response)

    print(
        f"\n[stats] elapsed={elapsed:.2f}s "
        f"input_tokens={in_tok} output_tokens={out_tok}"
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
