# Search V2 — Step 0 runner
#
# Thin CLI wrapper around `search_v2.step_0.run_step_0` that prints the
# full structured response plus timing and token usage. The query is
# accepted as a positional argument with a built-in default so the
# script can be invoked with no arguments for a quick smoke test.
#
# Usage:
#   python -m search_v2.run_step_0
#   python -m search_v2.run_step_0 "your query here"

from __future__ import annotations

import argparse
import asyncio
import json
import time

from search_v2.step_0 import run_step_0
from schemas.step_0_flow_routing import Step0Response


# Default query used when the user invokes the script with no
# argument. Picked to exercise a full-coverage title with genuine
# alternate reading — a classic boundary case for flow routing.
_DEFAULT_QUERY = "scary movie"


def _print_response(response: Step0Response) -> None:
    """Pretty-print the full Step0Response payload as indented JSON."""
    payload = response.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run search_v2.step_0 on a query and print the full "
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

    # run_step_0 returns (response, input_tokens, output_tokens) — no
    # elapsed time, so we measure it here.
    start = time.perf_counter()
    response, in_tok, out_tok = await run_step_0(args.query)
    elapsed = time.perf_counter() - start

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
