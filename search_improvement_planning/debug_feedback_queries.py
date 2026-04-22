# Debug driver: runs Stages 1, 2A, and 2B end-to-end against a curated
# feedback query set, capturing the full structured output at every
# stage so we can audit where each stage's behavior diverges from what
# the query alone would lead a human to expect.
#
# Unlike debug_stage_2a.py, this driver:
# - only operates on the primary_intent branch (keeps the report focused
#   on the default pipeline path)
# - carries outputs through Stage 2B so we can see the full action plan
# - matches Stage 2A and 2B to Gemini 3 Flash (small-model profile)
#
# Usage (from project root):
#   python -m search_improvement_planning.debug_feedback_queries

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

from implementation.llms.generic_methods import LLMProvider
from search_v2.stage_1 import route_query
from search_v2.stage_2a import BranchKind, run_stage_2a
from search_v2.stage_2b import run_stage_2b


# ---------------------------------------------------------------------------
# Feedback queries — the full list from the user's evaluation request.
# Order preserved so the output file is easy to scan top-down.
# ---------------------------------------------------------------------------

FEEDBACK_QUERIES: list[str] = [
    "Mindless action",
    "Christmas movies that make me cry in a happy way",
    "Popcorn movies",
    "Soulful",
    "Movies about running",
    "Disney classics",
    "A dystopian sci-fi thriller with a lone female protagonist",
    "A period drama about wartime love in 1940s England",
    "Movies that hit you in the gut",
    "Artful dramas with something to say",
    "Moody neo-noir with killer cinematography and bite",
    "Intimate, slow-moving character studies that linger with you",
    "Classic arnold schwarzenegger action movies",
    "Best christmas movies for families",
    "Indiana jones movie where he runs from the boulder",
    "Spider-man movies",
    "Shrek movies",
    "Mainline harry potter films",
]


# Match the existing debug drivers: small-model profile, no thinking.
# Stage 1 is already pinned to Gemini 3 Flash inside stage_1.py; we use
# the same for 2A and 2B so every stage runs under the same regime.
_STAGE_2_PROVIDER = LLMProvider.GEMINI
_STAGE_2_MODEL = "gemini-3-flash-preview"
_STAGE_2_KWARGS: dict = {"thinking_config": {"thinking_budget": 0}}


OUTPUT_PATH = (
    PROJECT_ROOT / "search_improvement_planning" / "feedback_queries_debug_output.json"
)

# Number of independent runs per query. Small-model stages are stochastic
# even with thinking disabled, so one run only catches the dominant mode;
# three runs surface behavior variance.
RUNS_PER_QUERY = 3


async def _run_stage_1(query: str) -> dict:
    """Run Stage 1 once and return a serializable record."""
    started = time.perf_counter()
    try:
        response, input_tokens, output_tokens = await route_query(query)
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


async def _run_stage_2a_primary(
    intent_rewrite: str,
    query_traits: str,
) -> dict:
    """Run Stage 2A on the primary-branch rewrite."""
    started = time.perf_counter()
    try:
        response, input_tokens, output_tokens = await run_stage_2a(
            branch_kind=BranchKind.PRIMARY,
            intent_rewrite=intent_rewrite,
            query_traits=query_traits,
            provider=_STAGE_2_PROVIDER,
            model=_STAGE_2_MODEL,
            **_STAGE_2_KWARGS,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "status": "ok",
            "elapsed_ms": round(elapsed_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response": response.model_dump(mode="json"),
            "_response_obj": response,  # carried forward for Stage 2B; dropped before serialization
        }
    except Exception as err:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "status": "error",
            "elapsed_ms": round(elapsed_ms, 1),
            "error_type": type(err).__name__,
            "error_message": str(err),
        }


async def _run_stage_2b(intent_rewrite: str, stage_2a_response) -> dict:
    """Run Stage 2B with the Stage 2A response object."""
    started = time.perf_counter()
    try:
        response, input_tokens, output_tokens = await run_stage_2b(
            intent_rewrite=intent_rewrite,
            stage_2a=stage_2a_response,
            provider=_STAGE_2_PROVIDER,
            model=_STAGE_2_MODEL,
            **_STAGE_2_KWARGS,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "status": "ok",
            "elapsed_ms": round(elapsed_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "response": response.model_dump(mode="json"),
        }
    except Exception as err:  # noqa: BLE001
        elapsed_ms = (time.perf_counter() - started) * 1000
        return {
            "status": "error",
            "elapsed_ms": round(elapsed_ms, 1),
            "error_type": type(err).__name__,
            "error_message": str(err),
        }


async def _run_one_query(query: str, run_idx: int) -> dict:
    """Run Stage 1 -> 2A -> 2B for one query. Stages are sequential
    since each depends on the previous stage's output. Non-standard
    flows (exact_title, similarity) skip 2A/2B entirely — those
    branches have no concept partitioning to do."""
    print(f"[running] run={run_idx}  {query!r}", flush=True)

    stage_1 = await _run_stage_1(query)

    if stage_1["status"] != "ok":
        return {
            "query": query,
            "stage_1": stage_1,
            "stage_2a": None,
            "stage_2b": None,
        }

    stage_1_response = stage_1["response"]
    primary = stage_1_response["primary_intent"]
    query_traits = stage_1_response["query_traits"]

    if primary["flow"] != "standard":
        # Non-standard flow — stages 2A/2B do not run for title
        # lookups or pure similarity queries. Record the skip explicitly
        # so the report is self-documenting.
        return {
            "query": query,
            "stage_1": stage_1,
            "stage_2a": {"status": "skipped", "reason": f"primary flow is {primary['flow']}"},
            "stage_2b": {"status": "skipped", "reason": f"primary flow is {primary['flow']}"},
        }

    stage_2a = await _run_stage_2a_primary(
        intent_rewrite=primary["intent_rewrite"],
        query_traits=query_traits,
    )
    if stage_2a["status"] != "ok":
        return {
            "query": query,
            "stage_1": stage_1,
            "stage_2a": _drop_response_obj(stage_2a),
            "stage_2b": None,
        }

    stage_2b = await _run_stage_2b(
        intent_rewrite=primary["intent_rewrite"],
        stage_2a_response=stage_2a["_response_obj"],
    )

    return {
        "query": query,
        "stage_1": stage_1,
        "stage_2a": _drop_response_obj(stage_2a),
        "stage_2b": stage_2b,
    }


def _drop_response_obj(stage_record: dict) -> dict:
    """Remove the live Pydantic object before JSON serialization."""
    clean = dict(stage_record)
    clean.pop("_response_obj", None)
    return clean


def _render_one_run(q: dict, run_idx: int) -> list[str]:
    """Render one run's stage-by-stage output as a list of lines."""
    lines: list[str] = []
    lines.append(f"  ---- run {run_idx} ----")

    s1 = q["stage_1"]
    if s1["status"] != "ok":
        lines.append(f"  [Stage 1] {s1['status']}: {s1.get('error_message')}")
        return lines
    r1 = s1["response"]
    lines.append(f"  [Stage 1]  flow={r1['primary_intent']['flow']}  "
                 f"alts={len(r1['alternative_intents'])}  "
                 f"spins={len(r1['creative_alternatives'])}")
    lines.append(f"    query_traits: {r1['query_traits']}")
    lines.append(f"    ambiguity_analysis: {r1['ambiguity_analysis'].replace(chr(10), ' ↵ ')}")
    lines.append(f"    primary.intent_rewrite: {r1['primary_intent']['intent_rewrite']}")
    for idx, alt in enumerate(r1["alternative_intents"]):
        lines.append(f"    alt[{idx}].intent_rewrite: {alt['intent_rewrite']}")
        lines.append(f"    alt[{idx}].difference_rationale: {alt['difference_rationale']}")
    lines.append(f"    creative_spin_analysis: {r1['creative_spin_analysis'].replace(chr(10), ' ↵ ')}")
    for idx, spin in enumerate(r1["creative_alternatives"]):
        lines.append(f"    spin[{idx}].spin_angle: {spin['spin_angle']}")
        lines.append(f"    spin[{idx}].intent_rewrite: {spin['intent_rewrite']}")

    s2a = q["stage_2a"]
    if s2a is None:
        lines.append("  [Stage 2A] (not run)")
    elif s2a.get("status") == "skipped":
        lines.append(f"  [Stage 2A] skipped — {s2a.get('reason')}")
    elif s2a.get("status") != "ok":
        lines.append(f"  [Stage 2A] {s2a.get('status')}: {s2a.get('error_message')}")
    else:
        r2a = s2a["response"]
        lines.append("  [Stage 2A]")
        lines.append(f"    unit_analysis: {r2a['unit_analysis'].replace(chr(10), ' ↵ ')}")
        lines.append(f"    inventory: {r2a['inventory']}")
        lines.append(f"    slot_analysis: {r2a['slot_analysis'].replace(chr(10), ' ↵ ')}")
        for idx, slot in enumerate(r2a["slots"]):
            lines.append(
                f"    slot[{idx}] {slot['handle']!r} (confidence={slot['confidence']})"
            )
            lines.append(f"      scope: {slot['scope']}")
            lines.append(f"      retrieval_shape: {slot['retrieval_shape']}")
            lines.append(f"      cohesion: {slot['cohesion']}")

    s2b = q["stage_2b"]
    if s2b is None:
        lines.append("  [Stage 2B] (not run)")
    elif s2b.get("status") == "skipped":
        lines.append(f"  [Stage 2B] skipped — {s2b.get('reason')}")
    elif s2b.get("status") != "ok":
        lines.append(f"  [Stage 2B] {s2b.get('status')}: {s2b.get('error_message')}")
    else:
        r2b = s2b["response"]
        lines.append("  [Stage 2B]")
        for idx, cs in enumerate(r2b["completed_slots"]):
            slot = cs["slot"]
            resp = cs["response"]
            lines.append(
                f"    slot[{idx}] {slot['handle']!r}  atoms={slot['scope']}"
            )
            lines.append(
                f"      atom_analysis: {resp['atom_analysis'].replace(chr(10), ' ↵ ')}"
            )
            if resp.get("skip_rationale"):
                lines.append(f"      SKIPPED — {resp['skip_rationale']}")
            for aidx, action in enumerate(resp.get("actions", []) or []):
                lines.append(
                    f"      action[{aidx}] route={action['route']} "
                    f"role={action['role']}"
                    + (f" strength={action['preference_strength']}"
                       if action.get('preference_strength') else "")
                )
                lines.append(f"        description: {action['description']}")
                lines.append(f"        coverage: {action['coverage_atoms']}")
                lines.append(f"        route_rationale: {action['route_rationale']}")
    return lines


def _render_compact_summary(report: dict) -> str:
    """Per-query multi-run summary for eyeballing the report."""
    lines: list[str] = []
    for q_entry in report["queries"]:
        lines.append("")
        lines.append("=" * 72)
        lines.append(f"QUERY: {q_entry['query']!r}")
        lines.append("=" * 72)
        for run_idx, run in enumerate(q_entry["runs"]):
            lines.extend(_render_one_run(run, run_idx))
    return "\n".join(lines)


async def main() -> None:
    overall_start = time.perf_counter()

    # Run each query RUNS_PER_QUERY times sequentially. Within a query,
    # runs are independent LLM calls — we record them as a list so
    # downstream analysis can diff behaviors across runs.
    query_reports: list[dict] = []
    for query in FEEDBACK_QUERIES:
        runs: list[dict] = []
        for run_idx in range(RUNS_PER_QUERY):
            runs.append(await _run_one_query(query, run_idx))
        query_reports.append({"query": query, "runs": runs})

    elapsed_s = time.perf_counter() - overall_start

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_elapsed_s": round(elapsed_s, 2),
        "stage_1_model": "gemini-3-flash-preview (thinking_budget=0, pinned in stage_1.py)",
        "stage_2_model": f"{_STAGE_2_MODEL} (thinking_budget=0)",
        "queries": query_reports,
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    summary = _render_compact_summary(report)
    print(summary)
    print()
    print("-" * 72)
    print(f"Wrote full structured report to: {OUTPUT_PATH}")
    print(f"Total elapsed: {elapsed_s:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
