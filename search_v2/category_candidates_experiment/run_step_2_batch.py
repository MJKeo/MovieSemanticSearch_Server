"""Batch-run Step 2 across the experiment query suite in parallel
and persist a single consolidated JSON result file.

The output file is the canonical Step 2 input fed to
run_step_3_batch.py — running Step 3 against a fixed Step 2 output
removes one source of stochastic variance from the comparison across
schema variants.

Output schema (single JSON object):
    {
      "<query string>": {
        "step_2_output": {... QueryAnalysis.model_dump() ...},
        "input_tokens": int,
        "output_tokens": int,
        "elapsed_seconds": float
      },
      ...
    }

Usage:
    python -m search_v2.category_candidates_experiment.run_step_2_batch
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from search_v2.step_2 import run_step_2
from search_v2.category_candidates_experiment.queries import QUERIES


# Save next to this script so subsequent stages can find it
# unambiguously without an env-var or CLI knob.
_OUTPUT_PATH = Path(__file__).parent / "step_2_results.json"


async def _run_one(query: str) -> tuple[str, dict]:
    """Run Step 2 on a single query and return (query, payload)
    where payload mirrors the on-disk JSON entry."""
    try:
        analysis, in_tok, out_tok, elapsed = await run_step_2(query)
        payload = {
            "step_2_output": analysis.model_dump(mode="json"),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "elapsed_seconds": elapsed,
        }
    except Exception as exc:
        # Capture failures inline so a single bad query doesn't
        # nuke the whole batch. The next stage will surface the
        # error field and refuse to run for that query.
        payload = {
            "error": f"{type(exc).__name__}: {exc}",
            "step_2_output": None,
        }
    return query, payload


async def _main_async() -> None:
    print(f"[step 2 batch] running {len(QUERIES)} queries in parallel")
    started = time.perf_counter()

    results = await asyncio.gather(*(_run_one(q) for q in QUERIES))

    elapsed = time.perf_counter() - started

    # Preserve query order from QUERIES rather than gather's ordering
    # (which is already ordered, but be explicit).
    out: dict[str, dict] = {q: payload for q, payload in results}

    _OUTPUT_PATH.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    failures = sum(1 for v in out.values() if v.get("step_2_output") is None)
    print(
        f"[step 2 batch] done in {elapsed:.1f}s, "
        f"{len(out) - failures} ok, {failures} failed, "
        f"wrote {_OUTPUT_PATH}"
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
