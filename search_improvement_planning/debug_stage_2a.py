# Debug driver: runs Step 2A (concept inventory extraction) against
# every standard-flow Step-1 rewrite produced for the curated
# queries in debug_stage_1.py.
#
# Why run 2A alone (not 2A + 2B): we want a clean lens on concept
# boundary behavior without 2B's expression-planning output masking
# the signal. If 2A hands 2B a bad inventory, everything after is
# built on sand — see steps_1_2_improving.md.
#
# Caching: Stage 1 results live in stage_1_debug_output.json (same
# file produced by debug_stage_1.py). This driver reuses them and
# only calls Stage 1 for queries missing from that cache. Delete the
# file to force a full re-run.
#
# Usage (from project root):
#   python -m search_improvement_planning.debug_stage_2a

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Ensure project root is on sys.path when invoked as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from implementation.llms.generic_methods import (
    LLMProvider,
    generate_llm_response_async,
)
from schemas.query_understanding import Step2AResponse

from search_improvement_planning.debug_stage_1 import (
    OUTPUT_PATH as STAGE_1_OUTPUT_PATH,
    QUERY_BUCKETS,
    _load_cache as _load_stage_1_cache,
    _run_bucket as _run_stage_1_bucket,
)
from search_v2.stage_2 import _STEP_2A_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Step 2A configuration.
#
# Match Stage 1's default debug model (Gemini 3 Flash with thinking
# disabled) so the two traces are directly comparable. The point of
# this driver is to exercise the prompt under the small-model regime
# we intend to run in production; a thinking model would paper over
# prompt weaknesses we are trying to observe.
# ---------------------------------------------------------------------------

_STEP_2A_PROVIDER = LLMProvider.GEMINI
_STEP_2A_MODEL = "gemini-3-flash-preview"
_STEP_2A_KWARGS: dict = {"thinking_config": {"thinking_budget": 0}}

STAGE_2A_OUTPUT_PATH = (
    PROJECT_ROOT / "search_improvement_planning" / "stage_2a_debug_output.json"
)


async def _run_step_2a(rewrite: str) -> dict:
    """Run Step 2A on a single Step-1 rewrite and return a serializable record."""
    started = time.perf_counter()
    try:
        user_prompt = f"Interpretation rewrite:\n{rewrite}"
        response, input_tokens, output_tokens = await generate_llm_response_async(
            provider=_STEP_2A_PROVIDER,
            user_prompt=user_prompt,
            system_prompt=_STEP_2A_SYSTEM_PROMPT,
            response_format=Step2AResponse,
            model=_STEP_2A_MODEL,
            **_STEP_2A_KWARGS,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "status": "ok",
            "elapsed_ms": round(elapsed_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response": response.model_dump(mode="json"),
        }
    except Exception as err:  # noqa: BLE001 — surface any failure in the report
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "status": "error",
            "elapsed_ms": round(elapsed_ms, 1),
            "error_type": type(err).__name__,
            "error_message": str(err),
        }


def _extract_standard_flow_branches(stage_1_response: dict) -> list[dict]:
    """Pick every standard-flow branch from a Stage 1 response payload.

    Step 2 only runs on standard flow. For each query we harvest the
    primary, any alternative_intents, and any creative_alternatives
    whose flow == "standard". Each branch becomes its own Step 2A
    input so we can observe concept-extraction behavior across the
    variants the pipeline actually produces for one raw query.
    """
    branches: list[dict] = []

    primary = stage_1_response.get("primary_intent", {})
    if primary.get("flow") == "standard":
        branches.append(
            {
                "branch_kind": "primary",
                "rewrite": primary["intent_rewrite"],
                "display_phrase": primary.get("display_phrase"),
            }
        )

    for idx, alt in enumerate(stage_1_response.get("alternative_intents", [])):
        if alt.get("flow") == "standard":
            branches.append(
                {
                    "branch_kind": f"alternative[{idx}]",
                    "rewrite": alt["intent_rewrite"],
                    "display_phrase": alt.get("display_phrase"),
                }
            )

    for idx, spin in enumerate(stage_1_response.get("creative_alternatives", [])):
        if spin.get("flow") == "standard":
            branches.append(
                {
                    "branch_kind": f"spin[{idx}]",
                    "rewrite": spin["intent_rewrite"],
                    "display_phrase": spin.get("display_phrase"),
                    "spin_angle": spin.get("spin_angle"),
                }
            )

    return branches


async def _ensure_stage_1_for_queries() -> dict[str, dict]:
    """Guarantee every debug query has a cached Stage 1 record.

    We rehydrate the existing stage_1 cache, run Stage 1 for any
    missing/errored queries, then rewrite the cache file so it stays
    authoritative for downstream steps (including repeated Step 2A
    runs).
    """
    cache = _load_stage_1_cache(STAGE_1_OUTPUT_PATH)

    # Collect bucket reports so we can re-serialize the cache after
    # backfilling anything that was missing.
    bucket_reports = []
    missing_any = False
    for bucket in QUERY_BUCKETS:
        if any(
            cache.get(q, {}).get("status") != "ok"
            for q in bucket["queries"]
        ):
            missing_any = True
        bucket_reports.append(await _run_stage_1_bucket(bucket, cache))

    if missing_any:
        # Refresh the cache on disk with any newly fetched records so
        # the next driver invocation can skip them.
        refreshed_report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_elapsed_s": 0.0,  # unknown: partially cached
            "stage_1_model": "gemini-3-flash-preview (thinking_budget=0)",
            "buckets": bucket_reports,
        }
        STAGE_1_OUTPUT_PATH.write_text(
            json.dumps(refreshed_report, indent=2, ensure_ascii=False)
        )

    # Flatten to {query: record} for easy lookup downstream.
    flat: dict[str, dict] = {}
    for bucket in bucket_reports:
        for record in bucket["results"]:
            flat[record["query"]] = record
    return flat


async def _run_bucket_stage_2a(
    bucket: dict, stage_1_by_query: dict[str, dict]
) -> dict:
    """Run Step 2A for every standard-flow branch of every query in a bucket."""
    query_reports = []
    for query in bucket["queries"]:
        stage_1_record = stage_1_by_query.get(query, {})
        if stage_1_record.get("status") != "ok":
            query_reports.append(
                {
                    "query": query,
                    "stage_1_status": stage_1_record.get("status", "missing"),
                    "stage_1_error": stage_1_record.get("error_message"),
                    "branches": [],
                }
            )
            continue

        stage_1_response = stage_1_record["response"]
        branches = _extract_standard_flow_branches(stage_1_response)

        # Fan out Step 2A calls for all branches concurrently. Step
        # 2A is independent across branches — one per rewrite.
        branch_results = await asyncio.gather(
            *(_run_step_2a(b["rewrite"]) for b in branches)
        )

        for branch_meta, result in zip(branches, branch_results):
            branch_meta["step_2a"] = result

        query_reports.append(
            {
                "query": query,
                "stage_1_status": "ok",
                "stage_1_primary_rewrite": stage_1_response["primary_intent"][
                    "intent_rewrite"
                ],
                "branches": branches,
            }
        )
    return {
        "bucket": bucket["bucket"],
        "hypothesis": bucket["hypothesis"],
        "queries": query_reports,
    }


def _render_compact_summary(report: dict) -> str:
    """Render a human-readable one-branch-per-line Step 2A summary."""
    lines: list[str] = []
    for bucket in report["buckets"]:
        lines.append("")
        lines.append(f"[{bucket['bucket']}] {bucket['hypothesis']}")
        for q_report in bucket["queries"]:
            lines.append(f"  query: {q_report['query']!r}")
            if q_report["stage_1_status"] != "ok":
                lines.append(
                    f"    ! Stage 1 {q_report['stage_1_status']}: "
                    f"{q_report.get('stage_1_error') or '(no error message)'}"
                )
                continue
            for branch in q_report["branches"]:
                header = f"    [{branch['branch_kind']}] {branch['rewrite']!r}"
                lines.append(header)
                s2a = branch.get("step_2a", {})
                if s2a.get("status") != "ok":
                    lines.append(
                        f"      ! Step 2A {s2a.get('status')}: "
                        f"{s2a.get('error_type')}: {s2a.get('error_message')}"
                    )
                    continue
                resp = s2a["response"]
                inv = ", ".join(resp["ingredient_inventory"]) or "(empty)"
                analysis = resp["concept_inventory_analysis"].replace("\n", " ")
                lines.append(f"      ingredient_inventory: {inv}")
                lines.append(f"      concept_inventory_analysis: {analysis}")
                for idx, concept in enumerate(resp["concepts"]):
                    ingredients = ", ".join(concept["required_ingredients"])
                    lines.append(
                        f"      concept[{idx}]: {concept['concept']!r} "
                        f"-> ingredients=[{ingredients}]"
                    )
                    lines.append(
                        f"        boundary_note: {concept['boundary_note']}"
                    )
    return "\n".join(lines)


async def main() -> None:
    overall_start = time.perf_counter()

    # Phase 1 — guarantee Stage 1 coverage for every debug query.
    stage_1_by_query = await _ensure_stage_1_for_queries()

    # Phase 2 — run Step 2A for every standard-flow branch per query.
    bucket_reports = []
    for bucket in QUERY_BUCKETS:
        bucket_reports.append(
            await _run_bucket_stage_2a(bucket, stage_1_by_query)
        )

    elapsed_s = time.perf_counter() - overall_start

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_s": round(elapsed_s, 2),
        "step_2a_model": f"{_STEP_2A_MODEL} (thinking_budget=0)",
        "buckets": bucket_reports,
    }

    STAGE_2A_OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )

    summary = _render_compact_summary(report)
    print(summary)
    print()
    print("-" * 72)
    print(f"Wrote full structured report to: {STAGE_2A_OUTPUT_PATH}")
    print(f"Total elapsed: {elapsed_s:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
