"""Batch-run Step 3 against the consolidated Step 2 output, three
times per query, to measure how schema variants affect the
trait_decomposition shape.

Parallelism strategy:
- Queries are processed SEQUENTIALLY to keep upstream rate limits
  predictable.
- For each query, the THREE repeat runs fan out in parallel.
- Inside each run, the per-trait Step 3 calls fan out in parallel,
  mirroring the production pattern from `search_v2/run_step_3.py`.

Output layout: one JSON file per query, written to
`results/<prefix>_<first_4_words_of_query>.json`. The on-disk shape:

    {
      "query": "<query string>",
      "prefix": "<run prefix>",
      "step_2_output": {... QueryAnalysis dump ...},
      "runs": [
        {
          "run_index": 0,
          "elapsed_seconds": float,
          "trait_results": [
            {
              "trait_surface_text": "...",
              "trait": {... Trait.model_dump() ...},
              "decomposition": {... TraitDecomposition dump ...} | null,
              "input_tokens": int,
              "output_tokens": int,
              "elapsed_seconds": float,
              "error": "<only set on failure>"
            },
            ...
          ]
        },
        ...
      ]
    }

Usage:
    python -m search_v2.category_candidates_experiment.run_step_3_batch <prefix>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

from schemas.step_2 import QueryAnalysis, Trait
from schemas.step_3 import TraitDecomposition
from search_v2.step_3 import run_step_3
from search_v2.category_candidates_experiment.queries import (
    QUERIES,
    slugify_first_four,
)


_BASE_DIR = Path(__file__).parent
_STEP_2_PATH = _BASE_DIR / "step_2_results.json"
_RESULTS_DIR = _BASE_DIR / "results"

# Number of repeat Step 3 runs per query. Three is the user-requested
# balance between LLM stochasticity smoothing and total token spend.
_REPEAT_RUNS = 3


def _load_step_2_results() -> dict[str, dict]:
    if not _STEP_2_PATH.exists():
        raise FileNotFoundError(
            f"{_STEP_2_PATH} not found. Run `python -m "
            "search_v2.category_candidates_experiment.run_step_2_batch` first."
        )
    return json.loads(_STEP_2_PATH.read_text())


async def _run_single_trait(
    trait: Trait, siblings: list[Trait]
) -> dict:
    """Run Step 3 on a single trait; trap exceptions per-trait so
    one bad call doesn't poison the whole run."""
    try:
        decomposition, in_tok, out_tok, elapsed = await run_step_3(
            trait, siblings
        )
        return {
            "trait_surface_text": trait.surface_text,
            "trait": trait.model_dump(mode="json"),
            "decomposition": decomposition.model_dump(mode="json"),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "elapsed_seconds": elapsed,
        }
    except Exception as exc:
        return {
            "trait_surface_text": trait.surface_text,
            "trait": trait.model_dump(mode="json"),
            "decomposition": None,
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_seconds": 0.0,
            "error": f"{type(exc).__name__}: {exc}",
        }


async def _run_one_pass(traits: list[Trait], run_index: int) -> dict:
    """Run one full Step 3 pass over every trait of the query."""
    started = time.perf_counter()
    trait_results = await asyncio.gather(
        *(
            _run_single_trait(t, [s for s in traits if s is not t])
            for t in traits
        )
    )
    elapsed = time.perf_counter() - started
    return {
        "run_index": run_index,
        "elapsed_seconds": elapsed,
        "trait_results": list(trait_results),
    }


async def _run_query(
    prefix: str, query: str, step_2_entry: dict
) -> None:
    """Three parallel passes for one query; write the per-query file
    when all three finish."""
    slug = slugify_first_four(query)
    out_path = _RESULTS_DIR / f"{prefix}_{slug}.json"

    if step_2_entry.get("step_2_output") is None:
        # Step 2 failed for this query — record the gap and move on.
        out_path.write_text(
            json.dumps(
                {
                    "query": query,
                    "prefix": prefix,
                    "step_2_output": None,
                    "error": step_2_entry.get(
                        "error", "step_2_output missing"
                    ),
                    "runs": [],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"[{prefix}][skip] {query!r} — no step 2 output")
        return

    analysis = QueryAnalysis.model_validate(step_2_entry["step_2_output"])
    traits = list(analysis.traits)

    if not traits:
        out_path.write_text(
            json.dumps(
                {
                    "query": query,
                    "prefix": prefix,
                    "step_2_output": step_2_entry["step_2_output"],
                    "runs": [
                        {"run_index": i, "elapsed_seconds": 0.0,
                         "trait_results": []}
                        for i in range(_REPEAT_RUNS)
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        print(f"[{prefix}][skip] {query!r} — step 2 produced no traits")
        return

    started = time.perf_counter()
    runs = await asyncio.gather(
        *(_run_one_pass(traits, i) for i in range(_REPEAT_RUNS))
    )
    elapsed = time.perf_counter() - started

    payload = {
        "query": query,
        "prefix": prefix,
        "step_2_output": step_2_entry["step_2_output"],
        "runs": list(runs),
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # Quick per-query roll-up so terminal progress is informative
    # without needing a separate report step.
    total_traits = len(traits) * _REPEAT_RUNS
    failures = sum(
        1 for run in runs for tr in run["trait_results"]
        if tr.get("decomposition") is None
    )
    print(
        f"[{prefix}][done] {query!r} — "
        f"{total_traits - failures}/{total_traits} trait calls ok, "
        f"wall={elapsed:.1f}s, file={out_path.name}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch Step 3 runner: 3 parallel passes per query, "
            "queries processed sequentially. Reads step_2_results.json "
            "produced by run_step_2_batch.py."
        )
    )
    parser.add_argument(
        "prefix",
        type=str,
        help=(
            "File prefix for output JSONs (e.g. 'base', 'min3', "
            "'min5'). Output goes to "
            "results/<prefix>_<slug>.json."
        ),
    )
    return parser.parse_args()


async def _main_async() -> None:
    args = _parse_args()
    _RESULTS_DIR.mkdir(exist_ok=True)
    step_2 = _load_step_2_results()

    print(
        f"[step 3 batch] prefix={args.prefix!r} "
        f"queries={len(QUERIES)} repeats={_REPEAT_RUNS}"
    )
    overall = time.perf_counter()

    # Sequential across queries — rate-limit safety. Each iteration
    # fires 3 parallel passes which themselves fan out across traits.
    for query in QUERIES:
        entry = step_2.get(query)
        if entry is None:
            print(f"[{args.prefix}][skip] {query!r} — not in step_2 JSON")
            continue
        await _run_query(args.prefix, query, entry)

    print(
        f"[step 3 batch] all queries finished in "
        f"{time.perf_counter() - overall:.1f}s"
    )


def main() -> None:
    asyncio.run(_main_async())


if __name__ == "__main__":
    main()
