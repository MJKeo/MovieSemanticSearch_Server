"""
run_search_json.py — JSON-dump variant of run_search.py.

Walks a single query through steps 2-4 of the search pipeline (step 0
bypassed, step 1 omitted by default since the user only cares about
2+) and emits a Markdown file per query containing a structured JSON
dump of every intermediate result:

  * Step 2: full Step2Response.
  * Step 3: per-CE handler LLM output, per-fired-endpoint params,
    per-fired-endpoint top-10 EndpointResult, implicit_expectations
    response.
  * Stage-3 final: per-movie score breakdowns (top_k) and the same
    final ranked table run_search.py renders (title + year + tmdb_id
    + score components).

Usage:
    python run_search_json.py "your query here"
        # → sample_thought_processes/<auto-slug>.md

    python run_search_json.py "your query here" --slug vague_title_recall
        # → sample_thought_processes/vague_title_recall.md

    python run_search_json.py --batch
        # runs every (slug, query) pair hard-coded in BATCH_QUERIES,
        # one output file each.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

from pydantic import BaseModel

from db.postgres import (
    pool as postgres_pool,
    fetch_movie_cards,
    fetch_quality_popularity_signals,
)
from db.qdrant import qdrant_client
from db.redis import init_redis, close_redis

from schemas.enums import (
    CategoryName,
    EndpointRoute,
    FitQuality,
    MatchMode,
    Polarity,
)
from schemas.endpoint_parameters import EndpointParameters
from schemas.endpoint_result import EndpointResult
from schemas.semantic_translation import SemanticEndpointParameters
from schemas.step_2 import CoverageEvidence, RequirementFragment, Step2Response

from search_v2.step_2 import run_step_2
from search_v2.implicit_expectations import run_implicit_expectations

from search_v2.stage_3.category_handlers.handler import (
    _run_handler_llm,
    _extract_fired_endpoints,
)
from search_v2.stage_3.category_handlers.handler_result import HandlerResult
from search_v2.stage_3.endpoint_executors import build_endpoint_coroutine
from search_v2.stage_3.orchestrator import (
    DEFAULT_TOP_K,
    Stage3Result,
    _consolidate,
    _resolve_active_priors,
    _run_deferred_preferences,
    _score_pool,
    _select_pool,
)
from search_v2.stage_3.trending_query_execution import execute_trending_query


# ============================================================================
# Hard-coded batch queries (from sample_thought_processes/claude_thinking.md)
# ============================================================================

BATCH_QUERIES: list[tuple[str, str]] = [
    (
        "vague_title_recall",
        "What's that movie where the guy can't form new memories and has tattoos all over his body to help him solve his wife's murder?",
    ),
    (
        "mood_practical_constraints",
        "I want something light and feel-good but with actual substance, around 90 minutes, on a streaming service.",
    ),
    (
        "actor_against_type",
        "What are the best dramatic performances from actors who are mostly known for comedy?",
    ),
    (
        "grief_not_depressing",
        "Films that deal with grief without being depressing.",
    ),
    (
        "2000s_super_hero",
        "Best superhero movies from the 2000s, before the MCU took over everything.",
    ),
]


# ============================================================================
# JSON helpers
# ============================================================================


def _to_jsonable(value: Any) -> Any:
    """Recursively convert Pydantic models, dataclasses, enums, sets, and
    tuples into plain JSON-friendly structures.

    The orchestrator surfaces a mix of types — Pydantic models for
    schema-bound shapes, dataclasses for Stage3Result / HandlerResult /
    ScoreBreakdown, and bare dicts/lists/sets for the consolidation
    buckets. We funnel everything through one converter so the resulting
    Markdown blocks are uniformly parseable.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: _to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, set):
        # Sort for stable output when elements are sortable; fall back
        # to a list of repr strings otherwise.
        try:
            return sorted(_to_jsonable(v) for v in value)
        except TypeError:
            return [_to_jsonable(v) for v in value]
    if hasattr(value, "value") and hasattr(value, "name"):
        # Enums.
        return value.value
    return str(value)


def _json_block(label: str, payload: Any, *, lines: list[str]) -> None:
    """Emit a fenced ```json``` block with the given payload."""
    lines.append(f"#### {label}")
    lines.append("```json")
    lines.append(json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")


def _truncate_endpoint_result(result: EndpointResult, *, top_n: int = 10) -> dict:
    """Top-N candidates from an EndpointResult, sorted by score desc."""
    top = sorted(result.scores, key=lambda sc: sc.score, reverse=True)[:top_n]
    return {
        "total_scored": len(result.scores),
        "top": [{"movie_id": sc.movie_id, "score": sc.score} for sc in top],
    }


# ============================================================================
# Filename derivation
# ============================================================================


def _slug_from_query(query: str) -> str:
    """Best-effort slug — used only when the caller doesn't pass --slug."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", query.lower()).strip("_")
    return slug[:60] or "query"


# ============================================================================
# Step 2
# ============================================================================


async def _step_2(query: str, lines: list[str]) -> Step2Response:
    lines.append("## Step 2 — Query Pre-pass")
    lines.append("")
    response, _in, _out, elapsed = await run_step_2(query)
    lines.append(f"_elapsed: {elapsed:.2f}s_")
    lines.append("")
    _json_block("Step 2 Response", response, lines=lines)
    return response


# ============================================================================
# Step 3 — per-CE handler runs
# ============================================================================


def _classify_into_handler_result(
    result: HandlerResult,
    wrapper: EndpointParameters,
    endpoint_result: EndpointResult,
) -> None:
    """Mirror handler.py's _assemble_result for the four-bucket fold,
    so the CLI's HandlerResult matches what the production orchestrator
    would have produced for this CE.
    """
    match_mode = wrapper.match_mode
    polarity = wrapper.polarity

    if match_mode == MatchMode.FILTER and polarity == Polarity.POSITIVE:
        for sc in endpoint_result.scores:
            result.inclusion_candidates[sc.movie_id] = (
                result.inclusion_candidates.get(sc.movie_id, 0.0) + sc.score
            )
        return

    if match_mode == MatchMode.FILTER and polarity == Polarity.NEGATIVE:
        if isinstance(wrapper, SemanticEndpointParameters):
            for sc in endpoint_result.scores:
                result.downrank_candidates[sc.movie_id] = (
                    result.downrank_candidates.get(sc.movie_id, 0.0) + sc.score
                )
        else:
            for sc in endpoint_result.scores:
                result.exclusion_ids.add(sc.movie_id)
        return

    if match_mode == MatchMode.TRAIT and polarity == Polarity.POSITIVE:
        result.preference_specs.append(wrapper)
        return

    if match_mode == MatchMode.TRAIT and polarity == Polarity.NEGATIVE:
        for sc in endpoint_result.scores:
            result.downrank_candidates[sc.movie_id] = (
                result.downrank_candidates.get(sc.movie_id, 0.0) + sc.score
            )
        return

    raise ValueError(
        f"Unhandled match_mode/polarity: {match_mode!r}/{polarity!r}"
    )


async def _run_one_endpoint(
    route: EndpointRoute,
    wrapper: EndpointParameters,
) -> tuple[dict, EndpointResult | None]:
    """Execute a single fired endpoint. Returns a JSON-friendly diagnostic
    block + the raw EndpointResult so the caller can fold it into the
    HandlerResult."""
    block: dict[str, Any] = {
        "endpoint": route.value,
        "match_mode": wrapper.match_mode.value,
        "polarity": wrapper.polarity.value,
        "wrapper_type": type(wrapper).__name__,
        "parameters": _to_jsonable(wrapper.parameters),
    }
    exec_start = time.perf_counter()
    try:
        endpoint_result = await build_endpoint_coroutine(
            route,
            wrapper,
            qdrant_client=qdrant_client,
            restrict_to_movie_ids=None,
        )
    except Exception as exc:  # noqa: BLE001
        block["exec_elapsed_s"] = round(time.perf_counter() - exec_start, 3)
        block["execution_failed"] = repr(exc)
        return block, None

    block["exec_elapsed_s"] = round(time.perf_counter() - exec_start, 3)
    block["result"] = _truncate_endpoint_result(endpoint_result, top_n=10)
    return block, endpoint_result


CEDiagEntry = tuple[
    RequirementFragment,
    CoverageEvidence,
    list[tuple[EndpointRoute | None, MatchMode, Polarity]],
]


async def _run_one_ce(
    *,
    query: str,
    step2_resp: Step2Response,
    parent_req: RequirementFragment,
    ce: CoverageEvidence,
    handler_result: HandlerResult,
    ce_diag_entry: list[tuple[EndpointRoute | None, MatchMode, Polarity]],
    ce_payload: dict,
) -> None:
    """Run one coverage_evidence handler and record its full diagnostic
    payload (LLM output + per-endpoint blocks) into ce_payload."""
    ce_payload["captured_meaning"] = ce.captured_meaning
    ce_payload["category_name"] = ce.category_name.value
    ce_payload["atomic_rewrite"] = ce.atomic_rewrite

    # TRENDING — no LLM, only execution.
    if ce.category_name == CategoryName.TRENDING:
        exec_start = time.perf_counter()
        result = await execute_trending_query(restrict_to_movie_ids=None)
        ce_payload["llm"] = None
        ce_payload["endpoints"] = [
            {
                "endpoint": "trending",
                "match_mode": "filter",
                "polarity": "positive",
                "parameters": "(no LLM — Redis read)",
                "exec_elapsed_s": round(time.perf_counter() - exec_start, 3),
                "result": _truncate_endpoint_result(result, top_n=10),
            }
        ]

        for sc in result.scores:
            handler_result.inclusion_candidates[sc.movie_id] = (
                handler_result.inclusion_candidates.get(sc.movie_id, 0.0) + sc.score
            )
        ce_diag_entry.append((None, MatchMode.FILTER, Polarity.POSITIVE))
        return

    siblings = [r for r in step2_resp.requirements if r is not parent_req]
    llm_start = time.perf_counter()
    output = await _run_handler_llm(
        category=ce.category_name,
        target_entry=ce,
        raw_query=query,
        overall_query_intention_exploration=(
            step2_resp.overall_query_intention_exploration
        ),
        parent_fragment=parent_req,
        sibling_fragments=siblings,
    )
    llm_elapsed = time.perf_counter() - llm_start

    ce_payload["llm"] = {
        "elapsed_s": round(llm_elapsed, 3),
        "output": _to_jsonable(output) if output is not None else None,
    }

    if output is None:
        ce_payload["endpoints"] = []
        ce_payload["note"] = "handler LLM failed after retry"
        return

    fired = _extract_fired_endpoints(ce.category_name, output)
    if not fired:
        ce_payload["endpoints"] = []
        ce_payload["note"] = "LLM returned no fired endpoints"
        return

    endpoint_outcomes = await asyncio.gather(
        *(_run_one_endpoint(route, wrapper) for route, wrapper in fired)
    )

    endpoints_payload: list[dict] = []
    for (route, wrapper), (block, endpoint_result) in zip(fired, endpoint_outcomes):
        endpoints_payload.append(block)
        if endpoint_result is None:
            continue
        _classify_into_handler_result(handler_result, wrapper, endpoint_result)
        ce_diag_entry.append((route, wrapper.match_mode, wrapper.polarity))

    ce_payload["endpoints"] = endpoints_payload


async def _run_ce_loop(
    query: str,
    step2_resp: Step2Response,
) -> tuple[list[HandlerResult], list[CEDiagEntry], list[dict]]:
    """Fan all non-NO_FIT coverage_evidence handlers out concurrently.

    Returns (handler_results, ce_diagnostics, ce_payloads). ce_payloads
    contains the full per-CE JSON-friendly diagnostic blob in canonical
    order — this is what the Markdown writer renders.
    """
    ce_specs: list[tuple[RequirementFragment, CoverageEvidence]] = []
    for parent_req in step2_resp.requirements:
        for ce in parent_req.coverage_evidence:
            if ce.fit_quality == FitQuality.NO_FIT:
                continue
            ce_specs.append((parent_req, ce))

    handler_results: list[HandlerResult] = [
        HandlerResult(category=ce.category_name) for _parent, ce in ce_specs
    ]
    ce_fired_lists: list[
        list[tuple[EndpointRoute | None, MatchMode, Polarity]]
    ] = [[] for _ in ce_specs]
    ce_payloads: list[dict] = [{} for _ in ce_specs]

    coros = [
        _run_one_ce(
            query=query,
            step2_resp=step2_resp,
            parent_req=parent_req,
            ce=ce,
            handler_result=handler_results[i],
            ce_diag_entry=ce_fired_lists[i],
            ce_payload=ce_payloads[i],
        )
        for i, (parent_req, ce) in enumerate(ce_specs)
    ]

    outcomes = await asyncio.gather(*coros, return_exceptions=True)
    for i, outcome in enumerate(outcomes):
        if isinstance(outcome, BaseException):
            ce_payloads[i].setdefault("captured_meaning", ce_specs[i][1].captured_meaning)
            ce_payloads[i].setdefault("category_name", ce_specs[i][1].category_name.value)
            ce_payloads[i]["handler_exception"] = repr(outcome)

    ce_diagnostics: list[CEDiagEntry] = [
        (parent_req, ce, ce_fired_lists[i])
        for i, (parent_req, ce) in enumerate(ce_specs)
    ]
    return handler_results, ce_diagnostics, ce_payloads


async def _run_implicit_with_timing(query: str, step2_resp: Step2Response):
    start = time.perf_counter()
    try:
        response, _in, _out, _elapsed = await run_implicit_expectations(
            query, step2_resp
        )
        return response, time.perf_counter() - start, None
    except Exception as exc:  # noqa: BLE001
        return None, time.perf_counter() - start, exc


# ============================================================================
# Step 3 — orchestration + Markdown output
# ============================================================================


async def _step_3(
    query: str,
    step2_resp: Step2Response,
    top_k: int,
    lines: list[str],
) -> tuple[list[CEDiagEntry], Stage3Result]:
    lines.append("## Step 3 — Endpoint Translation + Execution")
    lines.append("")

    (handler_results, ce_diagnostics, ce_payloads), (
        implicit_resp,
        impl_elapsed,
        impl_err,
    ) = await asyncio.gather(
        _run_ce_loop(query, step2_resp),
        _run_implicit_with_timing(query, step2_resp),
    )

    # Per-CE blocks.
    for i, payload in enumerate(ce_payloads):
        cat = payload.get("category_name", "?")
        rewrite = payload.get("atomic_rewrite", "")
        lines.append(f"### CE {i + 1} — {cat}")
        if rewrite:
            lines.append(f"_atomic_rewrite_: {rewrite}")
        lines.append("")
        _json_block(f"CE {i + 1} payload", payload, lines=lines)

    # Implicit expectations.
    lines.append("## Implicit Expectations")
    lines.append("")
    lines.append(f"_elapsed: {impl_elapsed:.2f}s_")
    if impl_err is not None:
        lines.append(f"_error: {impl_err!r}_")
    lines.append("")
    _json_block(
        "Implicit Expectations Response",
        implicit_resp,
        lines=lines,
    )

    # Reuse the orchestrator's post-fan-out phases directly.
    inclusion, downrank, exclusion, prefs = _consolidate(handler_results)
    pool_scores, deferred, fallback = await _select_pool(
        inclusion_aggregated=inclusion,
        preference_specs_pool=prefs,
        exclusion_set=exclusion,
        qdrant_client=qdrant_client,
    )

    if fallback == "empty":
        return ce_diagnostics, Stage3Result(
            movie_ids=[],
            breakdowns={},
            handler_results=handler_results,
            implicit_expectations=implicit_resp,
            used_fallback="empty",
        )

    for mid in exclusion:
        pool_scores.pop(mid, None)

    if not pool_scores:
        return ce_diagnostics, Stage3Result(
            movie_ids=[],
            breakdowns={},
            handler_results=handler_results,
            implicit_expectations=implicit_resp,
            used_fallback=fallback,
        )

    pool_set = set(pool_scores)
    deferred_outcomes = await _run_deferred_preferences(
        deferred, pool=pool_set, qdrant_client=qdrant_client
    )

    q_active, n_active = _resolve_active_priors(implicit_resp)
    prior_signals: dict[int, tuple[float | None, float | None]] = {}
    if q_active or n_active:
        prior_signals = await fetch_quality_popularity_signals(list(pool_scores))

    breakdowns = _score_pool(
        pool_inclusion_scores=pool_scores,
        downrank_aggregated=downrank,
        deferred_preference_outcomes=deferred_outcomes,
        quality_active=q_active,
        notability_active=n_active,
        prior_signals=prior_signals,
    )
    breakdowns.sort(key=lambda b: b.final_score, reverse=True)
    top = breakdowns[:top_k]

    stage3_result = Stage3Result(
        movie_ids=[b.movie_id for b in top],
        breakdowns={b.movie_id: b for b in top},
        handler_results=handler_results,
        implicit_expectations=implicit_resp,
        used_fallback=fallback,
    )

    # Consolidation summary block.
    lines.append("## Consolidation Buckets (post fan-out)")
    lines.append("")
    _json_block(
        "Consolidation summary",
        {
            "inclusion_unique_ids": len(inclusion),
            "downrank_unique_ids": len(downrank),
            "exclusion_unique_ids": len(exclusion),
            "preference_specs_count": len(prefs),
            "used_fallback": fallback,
        },
        lines=lines,
    )

    # Final score breakdowns (top_k).
    lines.append("## Final Score Breakdowns (top {0})".format(top_k))
    lines.append("")
    _json_block(
        f"Top {top_k} ScoreBreakdowns",
        [_to_jsonable(b) for b in top],
        lines=lines,
    )

    return ce_diagnostics, stage3_result


# ============================================================================
# Step 4 — final ranked table (mirrors run_search.py's _step_4_summary)
# ============================================================================


async def _step_4_summary(
    ce_diagnostics: list[CEDiagEntry],
    stage3_result: Stage3Result,
    top_k: int,
    total_start: float,
    lines: list[str],
) -> None:
    lines.append("## Step 4 — Summary")
    lines.append("")

    filters: list[str] = []
    traits: list[str] = []
    for _parent, ce, fired in ce_diagnostics:
        modes = {match_mode for _route, match_mode, _polarity in fired}
        if MatchMode.FILTER in modes and ce.atomic_rewrite not in filters:
            filters.append(ce.atomic_rewrite)
        if MatchMode.TRAIT in modes and ce.atomic_rewrite not in traits:
            traits.append(ce.atomic_rewrite)

    lines.append("### Filters")
    if filters:
        for ar in filters:
            lines.append(f"- {ar}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append("### Traits")
    if traits:
        for ar in traits:
            lines.append(f"- {ar}")
    else:
        lines.append("- (none)")
    lines.append("")

    lines.append(f"_used_fallback: {stage3_result.used_fallback}_")
    lines.append("")

    impl = stage3_result.implicit_expectations
    if impl is not None:
        lines.append("### query_intent_summary")
        lines.append(impl.query_intent_summary)
        lines.append("")
        quality_on = impl.should_apply_quality_prior
        popularity_on = impl.should_apply_notability_prior
    else:
        quality_on = False
        popularity_on = False
    lines.append(
        f"_implicit priors: quality={'on' if quality_on else 'off'}  "
        f"popularity={'on' if popularity_on else 'off'}_"
    )
    lines.append("")

    # Final ranked table.
    lines.append(f"### Top {min(top_k, len(stage3_result.movie_ids))} Results")
    lines.append("")

    if not stage3_result.movie_ids:
        lines.append("(no results — pool was empty)")
    else:
        top_ids = stage3_result.movie_ids[:top_k]
        cards = await fetch_movie_cards(top_ids)
        cards_by_id = {c["movie_id"]: c for c in cards}
        lines.append("| # | final | filter | pref | down | impl | title (year) | tmdb_id |")
        lines.append("|---|-------|--------|------|------|------|--------------|---------|")
        for rank, mid in enumerate(top_ids, 1):
            card = cards_by_id.get(mid, {})
            title = card.get("title") or "<untitled>"
            release_ts = card.get("release_ts")
            year = (
                datetime.fromtimestamp(release_ts, tz=timezone.utc).year
                if release_ts
                else "?"
            )
            b = stage3_result.breakdowns[mid]
            # Escape pipe characters in titles so they don't break the table.
            safe_title = title.replace("|", "\\|")
            lines.append(
                f"| {rank} | {b.final_score:.4f} | {b.inclusion_sum:.4f} | "
                f"{b.preference_contribution:.4f} | {-b.downrank_sum:.4f} | "
                f"{b.implicit_prior_contribution:.4f} | {safe_title} ({year}) | "
                f"{mid} |"
            )
    lines.append("")

    total_elapsed = time.perf_counter() - total_start
    lines.append(f"_total elapsed: {total_elapsed:.2f}s_")
    lines.append("")


# ============================================================================
# Top-level driver
# ============================================================================


async def _run_one_query(query: str, slug: str, top_k: int) -> Path:
    """Run the pipeline for a single (query, slug) pair and write the
    resulting Markdown file. Returns the output path."""
    total_start = time.perf_counter()
    lines: list[str] = []
    lines.append(f"# Query")
    lines.append("")
    lines.append(f"> {query}")
    lines.append("")

    step2_resp = await _step_2(query, lines)
    ce_diagnostics, stage3_result = await _step_3(
        query, step2_resp, top_k, lines
    )
    await _step_4_summary(
        ce_diagnostics, stage3_result, top_k, total_start, lines
    )

    out_dir = _PROJECT_ROOT / "sample_thought_processes"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{slug}.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


async def _main(args: argparse.Namespace) -> None:
    # DB lifecycle mirrors run_search.py.
    await postgres_pool.open()
    await init_redis()

    try:
        if args.batch:
            for slug, query in BATCH_QUERIES:
                print(f"\n=== {slug} ===")
                print(f"  query: {query}")
                out_path = await _run_one_query(query, slug, args.top_k)
                print(f"  → wrote {out_path.relative_to(_PROJECT_ROOT)}")
        else:
            slug = args.slug or _slug_from_query(args.query)
            out_path = await _run_one_query(args.query, slug, args.top_k)
            print(f"wrote {out_path.relative_to(_PROJECT_ROOT)}")
    finally:
        await close_redis()
        await postgres_pool.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "JSON-dump variant of run_search.py. Emits one Markdown "
            "file per query under sample_thought_processes/."
        )
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="The raw user query (omit when --batch is used).",
    )
    parser.add_argument(
        "--slug",
        type=str,
        default=None,
        help="Output filename stem (default: derived from the query).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of ranked results to display (default: {DEFAULT_TOP_K}).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help=(
            "Run every (slug, query) pair in BATCH_QUERIES (the five "
            "queries from sample_thought_processes/claude_thinking.md)."
        ),
    )
    args = parser.parse_args()
    if not args.batch and not args.query:
        parser.error("query is required unless --batch is passed.")
    return args


def main() -> None:
    args = _parse_args()
    if args.query:
        args.query = args.query.strip()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
