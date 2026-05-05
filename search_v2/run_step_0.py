# Search V2 — Step 0 runner
#
# Thin CLI wrapper around `search_v2.step_0.run_step_0` that prints the
# full structured response plus timing and token usage. The query is
# accepted as a positional argument with a built-in default so the
# script can be invoked with no arguments for a quick smoke test.
#
# When Step 0 sets exact_title_flow_data.should_be_searched=True, the
# runner additionally invokes search_v2.exact_title_search.run_exact_title_search
# with the resulting payload and pretty-prints the ranked candidates.
# This requires Postgres connectivity, so the pool is opened on demand.
#
# Usage:
#   python -m search_v2.run_step_0
#   python -m search_v2.run_step_0 "your query here"

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Load .env BEFORE importing db.postgres — its module-level
# `pool = AsyncConnectionPool(conninfo=_build_conninfo(), ...)` reads
# POSTGRES_* env vars at import time, so we need them in os.environ first.
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from db.postgres import fetch_movie_cards, pool as postgres_pool  # noqa: E402
from schemas.step_0_flow_routing import ExactTitleFlowData, Step0Response  # noqa: E402
from search_v2.exact_title_search import (  # noqa: E402
    ExactTitleSearchResult,
    run_exact_title_search,
)
from search_v2.step_0 import run_step_0  # noqa: E402


# Default query used when the user invokes the script with no
# argument. Picked to exercise a full-coverage title with genuine
# alternate reading — a classic boundary case for flow routing.
_DEFAULT_QUERY = "scary movie"

# Cap the printed exact-title result list. Long franchises (e.g.
# Marvel-universe queries) can produce 100+ rows; capping keeps the
# CLI output scannable while the full result set remains available
# in-memory for any caller that wants it.
_MAX_PRINTED_RESULTS = 25


def _print_response(response: Step0Response) -> None:
    """Pretty-print the full Step0Response payload as indented JSON."""
    payload = response.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run search_v2.step_0 on a query and print the full "
            "response, elapsed time, and token usage. If the exact-"
            "title flow fires, also run the exact-title search and "
            "print its ranked output."
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


def _year_of(release_ts: int | None) -> int | None:
    """Convert Unix-seconds release_ts → calendar year (UTC), or None."""
    if release_ts is None:
        return None
    return datetime.fromtimestamp(release_ts, tz=timezone.utc).year


async def _ensure_postgres_open() -> None:
    """Lazy-open the Postgres pool — no-op when already open.

    Mirrors run_orchestrator._ensure_db_ready: the pool is created with
    open=False at import time, so a CLI invocation has to open it
    before any query helper can run. We don't need Redis or Qdrant for
    the exact-title search.
    """
    if postgres_pool._closed:
        await postgres_pool.open()


async def _print_exact_title_results(
    flow_data: ExactTitleFlowData,
    result: ExactTitleSearchResult,
) -> None:
    """Render the exact-title ranked output as a compact table.

    Fetches movie cards in a single bulk call to resolve title and year
    for the top-N rows; the underlying ranked list is unmodified.
    """
    print("\n[exact_title_search]")
    print(
        f"input: title={flow_data.exact_title_to_search!r}, "
        f"release_year={flow_data.release_year}"
    )

    if not result.ranked:
        print("results: (no matches)")
        return

    top = result.ranked[:_MAX_PRINTED_RESULTS]
    movie_ids = [mid for mid, _ in top]
    cards = await fetch_movie_cards(movie_ids)
    by_id = {card["movie_id"]: card for card in cards}

    total = len(result.ranked)
    shown = len(top)
    suffix = f" (showing top {shown} of {total})" if total > shown else ""
    print(f"results: {total} total{suffix}")

    # Layout: rank | score | source | year | title (movie_id)
    # Column widths are sized to the constants and a typical title;
    # long titles are left to wrap naturally rather than truncated so
    # diagnostic output stays faithful.
    for rank, (mid, score) in enumerate(top, start=1):
        card = by_id.get(mid)
        title = card["title"] if card else "<missing card>"
        year = _year_of(card["release_ts"]) if card else None
        year_str = str(year) if year is not None else "----"
        source = result.score_source.get(mid, "?")
        print(
            f"  {rank:>2}. {score:>5.3f}  {source:<18}  "
            f"{year_str}  {title} ({mid})"
        )


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

    # Conditionally execute the exact-title search. We only fire when
    # Step 0 explicitly says the flow should run; an unconditional call
    # would defeat the purpose of the flow router and waste a Postgres
    # round-trip on standard / similarity queries.
    if response.exact_title_flow_data.should_be_searched:
        await _ensure_postgres_open()
        ets_start = time.perf_counter()
        ets_result = await run_exact_title_search(response.exact_title_flow_data)
        ets_elapsed = time.perf_counter() - ets_start
        await _print_exact_title_results(
            response.exact_title_flow_data, ets_result
        )
        print(f"[exact_title_stats] elapsed={ets_elapsed:.2f}s")


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
