"""
run_orchestrator.py — CLI runner for the unified front-half orchestrator.

Runs `search_v2.full_pipeline_orchestrator.run_full_pipeline` with
`skip_bypass_steps_0_1=True`, so the raw query goes straight into
Step 2 as a single "original" branch. Prints:

  1. Per-step completion events with elapsed (real-time, via INFO logs).
  2. Total wall-clock elapsed for the whole pipeline.
  3. The full list of generated endpoint calls, grouped by
     trait → category → individual endpoint call. Per-call output
     names the endpoint route and dumps only the non-null parameters
     of its EndpointParameters wrapper.

Usage:
    python run_orchestrator.py "your query here"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

# Project root on sys.path so absolute imports resolve when the script
# is invoked directly (matches the convention in run_search.py).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from search_v2.full_pipeline_orchestrator import (  # noqa: E402
    CategoryCallWithEndpoints,
    FullPipelineResult,
    GeneratedEndpointSpec,
    TraitWithEndpoints,
    run_full_pipeline,
)


# ---------------------------------------------------------------------------
# Logging setup — surface per-step completion events from the
# orchestrator in real time. Filtering down to the orchestrator's own
# logger keeps unrelated INFO chatter (httpx, openai, etc.) silent.
# ---------------------------------------------------------------------------


def _configure_realtime_logging() -> None:
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("[%(relativeCreated)6.0fms] %(message)s")
    )

    orch_logger = logging.getLogger(
        "search_v2.full_pipeline_orchestrator"
    )
    orch_logger.setLevel(logging.INFO)
    orch_logger.addHandler(handler)
    orch_logger.propagate = False


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def _params_to_dict(params: BaseModel | None) -> dict | None:
    # exclude_none=True drops every field whose value is None at any
    # level of nesting, so the printed dict only carries the fields
    # the LLM (or deterministic builder) actually populated. mode="json"
    # coerces enums and other non-primitive types to their JSON forms
    # so json.dumps below doesn't choke.
    if params is None:
        return None
    return params.model_dump(mode="json", exclude_none=True)


def _print_endpoint_spec(spec: GeneratedEndpointSpec, indent: str) -> None:
    print(f"{indent}- endpoint route: {spec.route.value}")
    if spec.params is None:
        # TRENDING is the only route with no params today; surface
        # that explicitly rather than printing an empty dict.
        print(f"{indent}  params: (none — endpoint takes no parameters)")
        return

    params_dict = _params_to_dict(spec.params)
    print(f"{indent}  wrapper: {type(spec.params).__name__}")
    print(f"{indent}  params (non-null fields only):")
    # json.dumps with indent=2 gives a clean nested view; we then
    # re-indent each line so it sits under the "params:" label.
    rendered = json.dumps(params_dict, indent=2, ensure_ascii=False)
    for line in rendered.splitlines():
        print(f"{indent}    {line}")


def _print_category_call(
    cc: CategoryCallWithEndpoints, indent: str
) -> None:
    print(f"{indent}CATEGORY: {cc.category.value}")
    if cc.handler_error is not None:
        print(f"{indent}  handler_error: {cc.handler_error}")
    if not cc.generated_specs:
        # Empty is a valid outcome (EXPLICIT_NO_OP, MEDIA_TYPE with
        # nothing matchable, or LLM judged nothing to fire).
        print(f"{indent}  (no endpoint calls generated)")
        return
    for spec in cc.generated_specs:
        _print_endpoint_spec(spec, indent + "  ")


def _print_trait(trait: TraitWithEndpoints, indent: str = "") -> None:
    print(
        f'{indent}TRAIT: "{trait.surface_text}" '
        f"[polarity={trait.polarity.value}, "
        f"commitment={trait.commitment}]"
    )
    if trait.step_3_error is not None:
        print(f"{indent}  step_3_error: {trait.step_3_error}")
    if not trait.category_calls:
        print(f"{indent}  (no category calls — Step 3 produced nothing)")
        return
    for cc in trait.category_calls:
        _print_category_call(cc, indent + "  ")


def _print_auxiliary_specs(result: FullPipelineResult) -> None:
    # Auxiliary specs are global fetches not attached to any trait.
    # Today the only one is the default shorts-exclusion MEDIA_TYPE
    # fetch injected when no trait emitted a MEDIA_TYPE call.
    if not result.auxiliary_endpoint_specs:
        print(
            "AUXILIARY ENDPOINT CALLS (not attached to a trait): "
            "(none — a trait already covers MEDIA_TYPE)"
        )
        return
    print(
        f"AUXILIARY ENDPOINT CALLS (not attached to a trait) — "
        f"{len(result.auxiliary_endpoint_specs)} call(s):"
    )
    for spec in result.auxiliary_endpoint_specs:
        _print_endpoint_spec(spec, indent="  ")


def _print_full_result(result: FullPipelineResult) -> None:
    if not result.branches:
        print("\nNo branches produced — nothing to display.")
        return
    # skip_bypass_steps_0_1=True always yields exactly one branch;
    # no need to label by branch kind in the header for this script.
    branch = result.branches[0]
    if branch.branch_error is not None:
        print(f"\nBranch failed: {branch.branch_error}")
        return
    if not branch.traits:
        print("\nBranch produced zero traits.")
        _print_auxiliary_specs(result)
        return
    print(f"\nGenerated {len(branch.traits)} trait(s):\n")
    for trait in branch.traits:
        _print_trait(trait)
        print()
    _print_auxiliary_specs(result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _run(query: str) -> None:
    print(f'Query: "{query}"\n')
    print("Step completions (real-time):")

    pipeline_start = time.perf_counter()
    result = await run_full_pipeline(query, skip_bypass_steps_0_1=True)
    pipeline_elapsed = time.perf_counter() - pipeline_start

    # The orchestrator already records its own total_elapsed; print
    # both for sanity (the small delta covers script overhead).
    print(f"\nOrchestrator total_elapsed: {result.total_elapsed:.2f}s")
    print(f"Wall-clock elapsed:         {pipeline_elapsed:.2f}s")

    _print_full_result(result)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the unified front-half orchestrator with Steps 0/1 "
            "bypassed. Prints per-step completion timing and the "
            "generated endpoint calls grouped by trait → category → "
            "endpoint."
        )
    )
    parser.add_argument(
        "query",
        type=str,
        help="Raw user query to feed into the pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_realtime_logging()
    args = _parse_args()
    asyncio.run(_run(args.query))


if __name__ == "__main__":
    main()
