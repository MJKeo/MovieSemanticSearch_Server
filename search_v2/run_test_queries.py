"""Batch runner for the Step 2 + Step 3 test query suite.

Reads every query from `search_v2/test_queries.md` and runs the
end-to-end Step 2 + Step 3 pipeline against each one. Per-query
output mirrors what `python -m search_v2.run_step_3 "<query>"` would
print, so files written here are directly comparable to the
historical `/tmp/step3_runs_v*/` snapshots.

Concurrency is bounded by an asyncio.Semaphore — by default 5 queries
in flight at once. Each task writes its own output file when it
finishes, so concurrent tasks never share an output buffer (which
would race even within a single process).

Usage:
    python -m search_v2.run_test_queries
    python -m search_v2.run_test_queries --out /tmp/step3_runs_v9
    python -m search_v2.run_test_queries --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import re
import time
from pathlib import Path
from typing import TextIO

from search_v2.step_2 import run_step_2
from search_v2.step_3 import run_step_3
from schemas.step_2 import QueryAnalysis, Trait
from schemas.step_3 import TraitDecomposition


# Test-query corpus and default output location. Overridable via CLI.
_TEST_QUERIES_PATH = Path(__file__).resolve().parent / "test_queries.md"
_DEFAULT_OUT_DIR = Path("/tmp/step3_runs_latest")
_DEFAULT_CONCURRENCY = 5


# ---------------------------------------------------------------
# Output formatting — mirrors run_step_3.py's CLI shape exactly so
# files written here line up with historical batch outputs.
# ---------------------------------------------------------------


def _write_step_2_block(buf: TextIO, analysis: QueryAnalysis) -> None:
    """Pretty-print the Step 2 QueryAnalysis to buf."""
    print("[step 2 response]", file=buf)
    payload = analysis.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False), file=buf)


def _write_trait_decomposition(
    buf: TextIO,
    trait: Trait,
    decomposition: TraitDecomposition,
    in_tok: int,
    out_tok: int,
    elapsed: float,
) -> None:
    """Pretty-print one trait's Step 3 decomposition with its
    per-trait timing/token header — same layout as
    run_step_3._print_trait_decomposition."""
    print(f'\n--- Trait: "{trait.surface_text}" ---', file=buf)
    print(
        f"[step 3 call] elapsed={elapsed:.2f}s "
        f"input_tokens={in_tok} output_tokens={out_tok}",
        file=buf,
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
        f"\n  relevance_to_query: {trait.relevance_to_query}",
        file=buf,
    )
    print("[decomposition]", file=buf)
    payload = decomposition.model_dump()
    print(json.dumps(payload, indent=2, ensure_ascii=False), file=buf)


# ---------------------------------------------------------------
# Per-query orchestration
# ---------------------------------------------------------------


async def _run_one(
    query: str, idx: int, out_dir: Path
) -> tuple[int, str, float, bool]:
    """Run Step 2 + Step 3 on one query and write its output file.

    Returns (idx, query, elapsed_seconds, ok). `ok=False` indicates
    the run terminated with an error written into the output file —
    the caller logs it and moves on without taking down the batch.
    """
    out_path = out_dir / f"q{idx:02d}.txt"
    buf = io.StringIO()
    start = time.perf_counter()
    ok = True

    print(f"[query] {query}\n", file=buf)

    try:
        analysis, s2_in, s2_out, s2_elapsed = await run_step_2(query)
    except Exception as exc:
        print(f"[ERROR] step 2 failed: {exc!r}", file=buf)
        out_path.write_text(buf.getvalue())
        return idx, query, time.perf_counter() - start, False

    _write_step_2_block(buf, analysis)

    print("\n[step 3 responses]", file=buf)
    if not analysis.traits:
        print(
            "(no traits emitted by Step 2 — nothing to decompose)",
            file=buf,
        )
        s3_in_total, s3_out_total, s3_elapsed_max = 0, 0, 0.0
    else:
        try:
            results = await asyncio.gather(
                *(run_step_3(trait) for trait in analysis.traits)
            )
        except Exception as exc:
            print(f"\n[ERROR] step 3 failed: {exc!r}", file=buf)
            out_path.write_text(buf.getvalue())
            return idx, query, time.perf_counter() - start, False

        s3_in_total = 0
        s3_out_total = 0
        s3_elapsed_max = 0.0
        for trait, (decomp, in_tok, out_tok, elapsed) in zip(
            analysis.traits, results
        ):
            _write_trait_decomposition(
                buf, trait, decomp, in_tok, out_tok, elapsed
            )
            s3_in_total += in_tok
            s3_out_total += out_tok
            # gather runs per-trait calls concurrently, so the
            # wall-clock cost of the fan-out is the slowest call.
            if elapsed > s3_elapsed_max:
                s3_elapsed_max = elapsed

    in_total = s2_in + s3_in_total
    out_total = s2_out + s3_out_total
    print(
        "\n[stats]"
        f"\n  step 2:   elapsed={s2_elapsed:.2f}s "
        f"input_tokens={s2_in} output_tokens={s2_out}"
        f"\n  step 3:   wallclock={s3_elapsed_max:.2f}s "
        "(slowest of parallel calls) "
        f"input_tokens={s3_in_total} "
        f"output_tokens={s3_out_total}"
        f"\n  totals:   input_tokens={in_total} output_tokens={out_total}",
        file=buf,
    )

    out_path.write_text(buf.getvalue())
    return idx, query, time.perf_counter() - start, ok


async def _bounded_run(
    query: str, idx: int, out_dir: Path, sem: asyncio.Semaphore
) -> tuple[int, str, float, bool]:
    async with sem:
        return await _run_one(query, idx, out_dir)


# ---------------------------------------------------------------
# Query loading + main loop
# ---------------------------------------------------------------


def _load_queries() -> list[str]:
    """Extract the queries from test_queries.md.

    The file pairs each section header with a single line of the form
    `**Query:** \\`<query>\\``. The regex captures everything inside the
    backticks for every such occurrence, in document order.
    """
    text = _TEST_QUERIES_PATH.read_text()
    queries = re.findall(r"\*\*Query:\*\* `([^`]+)`", text)
    if not queries:
        raise RuntimeError(
            f"no queries parsed from {_TEST_QUERIES_PATH}"
        )
    return queries


async def _main_async(out_dir: Path, concurrency: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "_run.log"

    queries = _load_queries()
    sem = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(_bounded_run(q, i, out_dir, sem))
        for i, q in enumerate(queries, start=1)
    ]

    print(
        f"[start] {len(queries)} queries → {out_dir} "
        f"(concurrency={concurrency})"
    )

    # Write progress as queries finish, in completion order. Open the
    # log in line-buffered mode so a tail -f reads progress live.
    with log_path.open("w", buffering=1) as log_fp:
        completed = 0
        failed = 0
        wall_start = time.perf_counter()
        for coro in asyncio.as_completed(tasks):
            idx, query, elapsed, ok = await coro
            completed += 1
            if not ok:
                failed += 1
            status = "ok " if ok else "ERR"
            line = (
                f"[{time.strftime('%H:%M:%S')}] "
                f"({completed:>2}/{len(queries)}) "
                f"q{idx:02d} {status} ({elapsed:>5.1f}s): {query}"
            )
            print(line)
            log_fp.write(line + "\n")

        total = time.perf_counter() - wall_start
        done_line = (
            f"[{time.strftime('%H:%M:%S')}] DONE — "
            f"{completed - failed} ok, {failed} failed, "
            f"wallclock {total:.1f}s"
        )
        print(done_line)
        log_fp.write(done_line + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=_DEFAULT_OUT_DIR,
        help=(
            "directory for per-query output files "
            f"(default: {_DEFAULT_OUT_DIR})"
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=_DEFAULT_CONCURRENCY,
        help=(
            "number of queries to run in flight at once "
            f"(default: {_DEFAULT_CONCURRENCY})"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_main_async(args.out, args.concurrency))


if __name__ == "__main__":
    main()
