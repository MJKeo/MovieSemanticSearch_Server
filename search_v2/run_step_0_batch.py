# Step 0 batch runner — runs Step 0 on N queries in parallel and writes
# each result (alongside its query) to its own file in step_0_results/.
#
# Usage:
#   python -m search_v2.run_step_0_batch "query 1" "query 2" "query 3"
#
# Each query produces a file step_0_results/<slug>.txt containing the
# query, the full Step0Response as JSON, the derived routing decisions,
# AND — when Step 0 selects the studio or person flow — the popularity-
# sorted (or bucketed) search results. Other entity flows currently
# dump only the Step 0 routing block; their search executors can be
# added here on the same pattern when verification of those flows lands.
#
# Files are overwritten on rerun so repeated invocations don't
# accumulate stale output.

from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

# Postgres pool is opened lazily — only studio-flow queries need it
# right now. Loading the module up front keeps the import surface
# consistent across batch entries even when no studio flow fires.
from db.postgres import fetch_movie_cards, pool as postgres_pool  # noqa: E402
from schemas.step_0_flow_routing import (  # noqa: E402
    EntityFlow,
    PersonFlowData,
    StudioFlowData,
)
from search_v2.step_0 import run_step_0  # noqa: E402
from search_v2.studio_search import run_studio_search, StudioSearchResult  # noqa: E402
from search_v2.person_search import run_person_search, PersonSearchResult  # noqa: E402

_OUT_DIR = Path(__file__).resolve().parents[1] / "step_0_results"

# Cap per-query rendered result rows — matches run_step_0.py so the
# two runners produce comparable output sizes.
_MAX_PRINTED_RESULTS = 25


def _slug(query: str) -> str:
    """Convert a query to a filesystem-safe slug, capped at 60 chars."""
    s = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
    return s[:60] if len(s) > 60 else s


def _year_of(release_ts: int | None) -> int | None:
    """Convert Unix-seconds release_ts → calendar year (UTC), or None."""
    if release_ts is None:
        return None
    return datetime.fromtimestamp(release_ts, tz=timezone.utc).year


async def _ensure_postgres_open() -> None:
    """Lazy-open the Postgres pool — no-op when already open.

    Mirrors run_step_0._ensure_postgres_open: the pool is created with
    open=False at import time, so a CLI invocation has to open it
    before any query helper can run.
    """
    if postgres_pool._closed:
        await postgres_pool.open()


async def _format_studio_block(
    flow_data: StudioFlowData,
    result: StudioSearchResult,
    elapsed: float,
) -> str:
    """Render the studio-search section for the per-query output file.

    Mirrors _print_studio_results in run_step_0.py but builds a string
    instead of streaming to stdout, so the batch runner can append it
    to the per-query file alongside the Step 0 block.
    """
    canonical_names = [ref.canonical_name for ref in flow_data.references]
    lines: list[str] = ["", "[studio_search]", f"input: canonical_names={canonical_names!r}"]

    total = len(result.ranked)
    if total == 0:
        lines.append("results: (no matches — none of the named studios resolved)")
        lines.append(f"[studio_stats] elapsed={elapsed:.2f}s")
        return "\n".join(lines)

    top = result.ranked[:_MAX_PRINTED_RESULTS]
    cards = await fetch_movie_cards(top)
    by_id = {card["movie_id"]: card for card in cards}

    suffix = f" (showing top {len(top)} of {total})" if total > len(top) else ""
    lines.append(f"results: {total} total{suffix}")
    for rank, mid in enumerate(top, start=1):
        card = by_id.get(mid)
        title = card["title"] if card else "<missing card>"
        year = _year_of(card["release_ts"]) if card else None
        year_str = str(year) if year is not None else "----"
        lines.append(f"  {rank:>2}. {year_str}  {title} ({mid})")
    lines.append(f"[studio_stats] elapsed={elapsed:.2f}s")
    return "\n".join(lines)


async def _format_person_block(
    flow_data: PersonFlowData,
    result: PersonSearchResult,
    elapsed: float,
) -> str:
    """Render the person-search section for the per-query output file.

    Mirrors _print_person_results in run_step_0.py but builds a string
    instead of streaming to stdout. Four prominence buckets, each
    capped at _MAX_PRINTED_RESULTS so distribution stays scannable in
    the per-query file.
    """
    canonical_names = [ref.canonical_name for ref in flow_data.references]
    lines: list[str] = ["", "[person_search]", f"input: canonical_names={canonical_names!r}"]

    buckets: list[tuple[str, list[int]]] = [
        ("bucket_1_lead",     result.bucket_1_lead),
        ("bucket_2_major",    result.bucket_2_major),
        ("bucket_3_relevant", result.bucket_3_relevant),
        ("bucket_4_minor",    result.bucket_4_minor),
    ]

    if not any(movie_ids for _, movie_ids in buckets):
        lines.append(
            "results: (no matches — none of the named people resolved to "
            "any credits across actor / director / writer / producer / "
            "composer postings)"
        )
        lines.append(f"[person_stats] elapsed={elapsed:.2f}s")
        return "\n".join(lines)

    # Bulk-fetch covering each bucket's print cap independently.
    movie_ids_to_hydrate: list[int] = []
    for _, movie_ids in buckets:
        movie_ids_to_hydrate.extend(movie_ids[:_MAX_PRINTED_RESULTS])
    cards = await fetch_movie_cards(movie_ids_to_hydrate)
    by_id = {card["movie_id"]: card for card in cards}

    for label, movie_ids in buckets:
        total = len(movie_ids)
        if total == 0:
            lines.append(f"\n{label}: (empty)")
            continue
        top = movie_ids[:_MAX_PRINTED_RESULTS]
        suffix = (
            f" (showing top {len(top)} of {total})"
            if total > len(top)
            else ""
        )
        lines.append(f"\n{label}: {total} total{suffix}")
        for rank, mid in enumerate(top, start=1):
            card = by_id.get(mid)
            title = card["title"] if card else "<missing card>"
            year = _year_of(card["release_ts"]) if card else None
            year_str = str(year) if year is not None else "----"
            lines.append(f"  {rank:>2}. {year_str}  {title} ({mid})")
    lines.append(f"[person_stats] elapsed={elapsed:.2f}s")
    return "\n".join(lines)


async def _run_one(query: str) -> tuple[str, str]:
    """Run Step 0 for one query and format the output block.

    When Step 0 routes the query into EntityFlow.STUDIO, also runs the
    studio search executor and appends its popularity-sorted ranked
    table to the output. Other entity flows currently dump only the
    Step 0 block — extend this dispatch the same way to wire them in.
    """
    start = time.perf_counter()
    try:
        response, in_tok, out_tok = await run_step_0(query)
        elapsed = time.perf_counter() - start
    except Exception as exc:  # noqa: BLE001 — surface any error per-query
        elapsed = time.perf_counter() - start
        return query, (
            f"[query] {query}\n"
            f"[stats] elapsed={elapsed:.2f}s\n\n"
            f"[error] {exc!r}\n"
        )

    block = [
        f"[query] {query}",
        f"[stats] elapsed={elapsed:.2f}s input_tokens={in_tok} output_tokens={out_tok}",
        "",
        "[response]",
        json.dumps(response.model_dump(), indent=2, ensure_ascii=False),
        "",
        f"[derived] fire_standard_flow={response.fire_standard_flow} "
        f"primary_flow={response.primary_flow.value}",
    ]

    # Per-flow executor dispatch. Studio and person are wired in here;
    # the other entity flows still print routing-only and execute via
    # run_step_0.py for live inspection.
    if response.selected_entity_flow == EntityFlow.STUDIO:
        flow_data = response.to_studio_flow_data()
        assert flow_data is not None
        try:
            await _ensure_postgres_open()
            std_start = time.perf_counter()
            std_result = await run_studio_search(flow_data)
            std_elapsed = time.perf_counter() - std_start
            block.append(
                await _format_studio_block(flow_data, std_result, std_elapsed)
            )
        except Exception as exc:  # noqa: BLE001 — keep batching even on per-query failure
            block.append(
                f"\n[studio_search]\n[error] {exc!r}"
            )
    elif response.selected_entity_flow == EntityFlow.PERSON:
        person_flow_data = response.to_person_flow_data()
        assert person_flow_data is not None
        try:
            await _ensure_postgres_open()
            ps_start = time.perf_counter()
            ps_result = await run_person_search(person_flow_data)
            ps_elapsed = time.perf_counter() - ps_start
            block.append(
                await _format_person_block(person_flow_data, ps_result, ps_elapsed)
            )
        except Exception as exc:  # noqa: BLE001 — keep batching even on per-query failure
            block.append(
                f"\n[person_search]\n[error] {exc!r}"
            )

    return query, "\n".join(block) + "\n"


async def _main_async(queries: list[str]) -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = await asyncio.gather(*(_run_one(q) for q in queries))
    for query, block in results:
        path = _OUT_DIR / f"{_slug(query)}.txt"
        path.write_text(block)
        print(f"  wrote {path.relative_to(_OUT_DIR.parent)}")


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: python -m search_v2.run_step_0_batch <query> [<query> ...]",
            file=sys.stderr,
        )
        sys.exit(2)
    queries = sys.argv[1:]
    from implementation.misc.event_loop import install_uvloop

    install_uvloop()
    asyncio.run(_main_async(queries))


if __name__ == "__main__":
    main()
