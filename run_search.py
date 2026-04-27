"""
run_search.py — CLI runner for the new search pipeline (steps 1-3 + summary).

Walks a single query through:
  Step 1: spin generation (alternate searches)
  Step 2: query pre-pass (intent + coverage evidence) — original query only
  Step 3: per-coverage_evidence endpoint translation + execution with
          per-CE LLM timing and per-endpoint exec timing
  Step 4: filter/trait grouping + final ranked table + total elapsed

Step 0 (flow routing) is bypassed; standard flow is assumed.

Usage:
    python run_search.py "your query here"
    python run_search.py "your query here" --top-k 25
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Project root on sys.path so absolute imports resolve when the script is
# invoked directly (`python run_search.py ...`) rather than via -m.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

load_dotenv(_PROJECT_ROOT / ".env")

# DB clients / lifecycle
from db.postgres import (
    pool as postgres_pool,
    fetch_movie_cards,
    fetch_quality_popularity_signals,
)
from db.qdrant import qdrant_client
from db.redis import init_redis, close_redis

# Schemas
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

# Pipeline steps
from search_v2.step_1 import run_step_1
from search_v2.step_2 import run_step_2
from search_v2.implicit_expectations import run_implicit_expectations

# Stage-3 building blocks. The underscore-prefixed helpers in handler.py
# and orchestrator.py are intentionally reused: this CLI is a development
# tool that needs the same internal seams the production orchestrator
# does, with diagnostic timing added between them.
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
# Display helpers
# ============================================================================


def _banner(title: str) -> None:
    """Section banner — visually separate the four step blocks."""
    bar = "=" * 72
    print(f"\n{bar}\n{title}\n{bar}")


def _format_params(wrapper: EndpointParameters) -> str:
    """Pretty-print only non-null fields of the wrapper's `parameters`.

    `model_dump(exclude_none=True)` cleanly hides empty axes for the
    structured wrappers (keyword/award/metadata/entity/franchise/studio)
    and preserves the populated semantic space_queries entries when the
    wrapper is semantic.
    """
    payload = wrapper.parameters.model_dump(exclude_none=True)
    return _stringify(payload, indent=8)


def _stringify(value, *, indent: int) -> str:
    """Render a nested dict/list as an indented multi-line string.

    Lightweight alternative to json.dumps so enum values and tuples
    survive without extra serializer hooks.
    """
    pad = " " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        lines = []
        for k, v in value.items():
            rendered = _stringify(v, indent=indent + 2)
            lines.append(f"{pad}{k}: {rendered}")
        return "\n" + "\n".join(lines)
    if isinstance(value, list):
        if not value:
            return "[]"
        lines = []
        for item in value:
            rendered = _stringify(item, indent=indent + 2)
            lines.append(f"{pad}- {rendered}")
        return "\n" + "\n".join(lines)
    return str(value)


# ============================================================================
# Step 1
# ============================================================================


async def _step_1(query: str) -> None:
    _banner("STEP 1 — Alternate Searches (spin generation)")
    response, _in, _out, elapsed = await run_step_1(query)

    for i, spin in enumerate(response.spins, 1):
        print(f"\nSpin {i}:")
        print(f"  query: {spin.query}")
        print(f"  branching_opportunity: {spin.branching_opportunity}")

    print(f"\n[step 1 elapsed: {elapsed:.2f}s]")


# ============================================================================
# Step 2
# ============================================================================


async def _step_2(query: str) -> Step2Response:
    _banner("STEP 2 — Query Pre-pass (intent + coverage evidence)")
    response, _in, _out, elapsed = await run_step_2(query)

    print(f"\noverall_query_intention_exploration:")
    print(f"  {response.overall_query_intention_exploration}")

    for req in response.requirements:
        print(f"\nRequirement: {req.query_text!r}")
        print(f"  description: {req.description}")
        if req.modifiers:
            print(f"  modifiers:")
            for m in req.modifiers:
                print(f"    - {m.original_text!r} ({m.type.value}) — {m.effect}")
        for ce in req.coverage_evidence:
            if ce.fit_quality == FitQuality.NO_FIT:
                continue
            print(f"    coverage_evidence:")
            print(f"      captured_meaning: {ce.captured_meaning}")
            print(f"      category_name: {ce.category_name.value}")
            print(f"      atomic_rewrite: {ce.atomic_rewrite}")

    print(f"\n[step 2 elapsed: {elapsed:.2f}s]")
    return response


# ============================================================================
# Step 3
# ============================================================================


def _classify_into_handler_result(
    result: HandlerResult,
    wrapper: EndpointParameters,
    endpoint_result: EndpointResult,
) -> None:
    """Mirror `_assemble_result` from handler.py: route a (wrapper, scores)
    pair into one of the four HandlerResult buckets.

    TRAIT+POSITIVE goes onto `preference_specs` and the diagnostic
    endpoint_result is intentionally discarded — the real pipeline runs
    preferences against the consolidated pool downstream and folding the
    full-corpus diagnostic scores in here would double-count.
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
        # Semantic FILTER+NEGATIVE is a soft downrank, not a hard exclude
        # — same override applied in handler.py:_classify_wrapper.
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


def _format_top5(result: EndpointResult) -> list[str]:
    """Return diagnostic lines for the 5 highest-scoring candidates.

    Returns a list of strings (no trailing newlines) rather than printing
    so callers can buffer per-CE/per-endpoint output and flush it in
    deterministic order after a parallel `asyncio.gather`.
    """
    if not result.scores:
        return ["        (no matches)"]
    top5 = sorted(result.scores, key=lambda sc: sc.score, reverse=True)[:5]
    return [f"        movie_id={sc.movie_id}  score={sc.score:.4f}" for sc in top5]


async def _run_one_endpoint(
    route: EndpointRoute,
    wrapper: EndpointParameters,
    llm_elapsed: float,
) -> tuple[list[str], EndpointResult | None]:
    """Execute a single fired endpoint and format its diagnostic block.

    Returns `(block_lines, result_or_None)`:
      - `block_lines` is a self-contained chunk of formatted lines for
        this endpoint (header + params + top 5 + trailing
        `LLM=X exec=Y` line). The caller appends the block to its
        parent CE buffer in original `fired` order so the visual
        layout is identical to the old sequential version.
      - `result_or_None` is the EndpointResult on success, or `None` if
        execution failed. The caller uses `None` to skip the
        `_classify_into_handler_result` call so a failed endpoint
        doesn't pollute the handler result.

    `llm_elapsed` is the parent CE's handler-LLM wall-clock; it's
    repeated on every endpoint's trailing line to preserve the
    original output format. Each invocation runs concurrently under
    the parent CE's `asyncio.gather` — `time.perf_counter()` is
    captured locally so the printed `exec=` number reflects only this
    coroutine's own wall-clock, not gather scheduling overhead.
    """
    exec_start = time.perf_counter()
    try:
        endpoint_result = await build_endpoint_coroutine(
            route,
            wrapper,
            qdrant_client=qdrant_client,
            restrict_to_movie_ids=None,
        )
    except Exception as exc:  # noqa: BLE001 — diagnostic, soft-fail
        exec_elapsed = time.perf_counter() - exec_start
        block = [
            f"    endpoint: {route.value}",
            f"      match_mode: {wrapper.match_mode.value}",
            f"      polarity: {wrapper.polarity.value}",
            f"      parameters: {_format_params(wrapper)}",
            f"      execution failed: {exc!r}",
            f"    LLM={llm_elapsed:.2f}s  exec={exec_elapsed:.2f}s",
        ]
        return block, None
    exec_elapsed = time.perf_counter() - exec_start

    block = [
        f"    endpoint: {route.value}",
        f"      match_mode: {wrapper.match_mode.value}",
        f"      polarity: {wrapper.polarity.value}",
        f"      parameters: {_format_params(wrapper)}",
        f"      top 5:",
        *_format_top5(endpoint_result),
        f"    LLM={llm_elapsed:.2f}s  exec={exec_elapsed:.2f}s",
    ]
    return block, endpoint_result


async def _run_one_ce(
    *,
    query: str,
    step2_resp: Step2Response,
    parent_req: RequirementFragment,
    ce: CoverageEvidence,
    handler_result: HandlerResult,
    ce_diag_entry: list[tuple[EndpointRoute | None, MatchMode, Polarity]],
    lines: list[str],
) -> None:
    """Run a single coverage_evidence handler, buffering diagnostics.

    All output is appended to `lines` (no print() calls) so the caller
    can run many CEs concurrently via `asyncio.gather` and still flush
    grouped per-CE blocks in original order. Mutates `handler_result`
    and appends one (route, match_mode, polarity) triple per fired
    endpoint to `ce_diag_entry`. Triples (rather than storing the
    wrapper) keep step-4 grouping uniform across LLM-driven handlers
    and the wrapper-less trending short-circuit.
    """
    # Re-record the step-2 fields so the reader has full context for
    # the endpoint output that follows.
    lines.append(f"\n  CE — {ce.category_name.value}")
    lines.append(f"      captured_meaning: {ce.captured_meaning}")
    lines.append(f"      category_name: {ce.category_name.value}")
    lines.append(f"      atomic_rewrite: {ce.atomic_rewrite}")

    # TRENDING — no LLM, only execution.
    if ce.category_name == CategoryName.TRENDING:
        exec_start = time.perf_counter()
        result = await execute_trending_query(restrict_to_movie_ids=None)
        exec_elapsed = time.perf_counter() - exec_start

        lines.append(f"    endpoint: trending")
        lines.append(f"      match_mode: filter")
        lines.append(f"      polarity: positive")
        lines.append(f"      parameters: (no LLM — Redis read)")
        lines.append(f"      top 5:")
        lines.extend(_format_top5(result))
        lines.append(f"    LLM=n/a  exec={exec_elapsed:.2f}s")

        for sc in result.scores:
            handler_result.inclusion_candidates[sc.movie_id] = (
                handler_result.inclusion_candidates.get(sc.movie_id, 0.0) + sc.score
            )
        # Record trending as filter+positive so step-4 grouping picks
        # up its atomic_rewrite under FILTERS.
        ce_diag_entry.append((None, MatchMode.FILTER, Polarity.POSITIVE))
        return

    # All other categories — call the per-category LLM, time the call.
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

    if output is None:
        lines.append(f"    handler LLM failed after retry — no endpoints fired.")
        lines.append(f"    LLM={llm_elapsed:.2f}s  exec=n/a")
        return

    fired = _extract_fired_endpoints(ce.category_name, output)
    if not fired:
        lines.append(f"    LLM returned no fired endpoints (judged nothing fits).")
        lines.append(f"    LLM={llm_elapsed:.2f}s  exec=n/a")
        return

    # Fan all fired endpoints out concurrently. Each `_run_one_endpoint`
    # records its own per-endpoint exec timing locally, so the printed
    # numbers reflect each call's own wall-clock duration regardless of
    # scheduling order. For TRAIT+POSITIVE wrappers the result is shown
    # here but discarded for scoring — the real pipeline defers
    # preferences to the consolidated pool downstream.
    endpoint_outcomes = await asyncio.gather(
        *(
            _run_one_endpoint(route, wrapper, llm_elapsed)
            for route, wrapper in fired
        )
    )

    # Walk outcomes in original `fired` order so the diagnostic order
    # is deterministic (matches the old sequential layout) and the
    # classification mutations happen in the same order they always did.
    for (route, wrapper), (block, endpoint_result) in zip(fired, endpoint_outcomes):
        lines.extend(block)
        if endpoint_result is None:
            continue
        _classify_into_handler_result(handler_result, wrapper, endpoint_result)
        ce_diag_entry.append((route, wrapper.match_mode, wrapper.polarity))


CEDiagEntry = tuple[
    RequirementFragment,
    CoverageEvidence,
    list[tuple[EndpointRoute | None, MatchMode, Polarity]],
]


async def _run_ce_loop(
    query: str,
    step2_resp: Step2Response,
) -> tuple[list[HandlerResult], list[CEDiagEntry]]:
    """Run every coverage_evidence handler concurrently.

    Mirrors the production orchestrator's `_fan_out` (search_v2/
    stage_3/orchestrator.py): build per-CE specs, dispatch all
    handlers in a single `asyncio.gather`, soft-fail individual
    handler exceptions so one bad CE doesn't tank the whole query.
    Diagnostic output is buffered per-CE and flushed in original spec
    order after gather completes — preserves the readable grouped
    layout while making the wall-clock numbers actually reflect
    production parallelism.
    """
    # Collect every non-NO_FIT (parent, ce) spec up front so the
    # gather list, the result list, and the flush order all share one
    # canonical ordering.
    ce_specs: list[tuple[RequirementFragment, CoverageEvidence]] = []
    for parent_req in step2_resp.requirements:
        for ce in parent_req.coverage_evidence:
            if ce.fit_quality == FitQuality.NO_FIT:
                continue
            ce_specs.append((parent_req, ce))

    # Pre-allocate per-CE state so each coroutine gets its own
    # HandlerResult / fired-triples list / diagnostic buffer. No
    # cross-coroutine sharing — safe under gather.
    handler_results: list[HandlerResult] = [
        HandlerResult(category=ce.category_name) for _parent, ce in ce_specs
    ]
    ce_fired_lists: list[
        list[tuple[EndpointRoute | None, MatchMode, Polarity]]
    ] = [[] for _ in ce_specs]
    line_buffers: list[list[str]] = [[] for _ in ce_specs]

    coros = [
        _run_one_ce(
            query=query,
            step2_resp=step2_resp,
            parent_req=parent_req,
            ce=ce,
            handler_result=handler_results[i],
            ce_diag_entry=ce_fired_lists[i],
            lines=line_buffers[i],
        )
        for i, (parent_req, ce) in enumerate(ce_specs)
    ]

    # `return_exceptions=True` mirrors the orchestrator's soft-fail:
    # one CE handler raising shouldn't take out the others. Each
    # caught exception gets recorded into that CE's buffer so the
    # operator can see what went wrong without losing the rest of
    # the run.
    outcomes = await asyncio.gather(*coros, return_exceptions=True)
    for i, outcome in enumerate(outcomes):
        if isinstance(outcome, BaseException):
            _parent, ce = ce_specs[i]
            line_buffers[i].append(
                f"\n  CE — {ce.category_name.value}"
            )
            line_buffers[i].append(
                f"    handler raised: {outcome!r} — empty result substituted."
            )

    # Flush per-CE blocks in canonical spec order so the visual layout
    # matches the old sequential run.
    for buf in line_buffers:
        if buf:
            print("\n".join(buf))

    ce_diagnostics: list[CEDiagEntry] = [
        (parent_req, ce, ce_fired_lists[i])
        for i, (parent_req, ce) in enumerate(ce_specs)
    ]
    return handler_results, ce_diagnostics


async def _run_implicit_with_timing(query: str, step2_resp: Step2Response):
    """Run implicit-expectations and return (response_or_None, elapsed_s).

    Soft-fails to None on any exception so a failed prior never tanks the
    whole query. Timing is captured here (not in `_step_3`) so it reflects
    only the implicit call's wall-clock, regardless of when the gathered
    CE loop finishes relative to it.
    """
    start = time.perf_counter()
    try:
        response, _in, _out, _elapsed = await run_implicit_expectations(
            query, step2_resp
        )
        return response, time.perf_counter() - start, None
    except Exception as exc:  # noqa: BLE001 — soft-fail by design
        return None, time.perf_counter() - start, exc


async def _step_3(
    query: str,
    step2_resp: Step2Response,
    top_k: int,
) -> tuple[list[CEDiagEntry], Stage3Result]:
    """Run step 3 with diagnostics and return (ce_diagnostics, Stage3Result).

    The CE loop and the implicit-expectations LLM run concurrently via
    asyncio.gather (mirrors the orchestrator's `_fan_out` parallelism).
    The CE loop owns all per-CE prints; the implicit-expectations
    timing line is printed once both finish.
    """
    _banner("STEP 3 — Endpoint Translation + Execution")

    (handler_results, ce_diagnostics), (
        implicit_resp,
        impl_elapsed,
        impl_err,
    ) = await asyncio.gather(
        _run_ce_loop(query, step2_resp),
        _run_implicit_with_timing(query, step2_resp),
    )

    print()
    if impl_err is not None:
        print(f"  implicit_expectations failed: {impl_err!r}")
    print(f"[implicit_expectations elapsed: {impl_elapsed:.2f}s]")

    # Reuse the orchestrator's post-fan-out phases directly so the
    # ranked output matches what run_stage_3 would produce.
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
    return ce_diagnostics, stage3_result


# ============================================================================
# Step 4
# ============================================================================


async def _step_4_summary(
    ce_diagnostics: list[CEDiagEntry],
    stage3_result: Stage3Result,
    top_k: int,
    total_start: float,
) -> None:
    _banner("STEP 4 — Summary")

    # Filter vs trait grouping. One CE can fire multiple endpoints; we
    # only want one rewrite per CE per bucket, so dedupe.
    filters: list[str] = []
    traits: list[str] = []
    for _parent, ce, fired in ce_diagnostics:
        modes = {match_mode for _route, match_mode, _polarity in fired}
        if MatchMode.FILTER in modes and ce.atomic_rewrite not in filters:
            filters.append(ce.atomic_rewrite)
        if MatchMode.TRAIT in modes and ce.atomic_rewrite not in traits:
            traits.append(ce.atomic_rewrite)

    print("\nFILTERS:")
    if filters:
        for ar in filters:
            print(f"  - {ar}")
    else:
        print("  (none)")

    print("\nTRAITS:")
    if traits:
        for ar in traits:
            print(f"  - {ar}")
    else:
        print("  (none)")

    print(f"\n[used_fallback: {stage3_result.used_fallback}]")

    # Implicit priors. `quality` is the reception prior, `notability` is
    # the popularity prior — names match ImplicitExpectationsResult's
    # should_apply_* fields. Both are inactive when the implicit-
    # expectations LLM call failed (implicit_expectations is None).
    impl = stage3_result.implicit_expectations
    if impl is not None:
        print(f"\n[query_intent_summary]")
        print(f"  {impl.query_intent_summary}")
        quality_on = impl.should_apply_quality_prior
        popularity_on = impl.should_apply_notability_prior
    else:
        quality_on = False
        popularity_on = False
    print(
        f"[implicit priors: quality={'on' if quality_on else 'off'}  "
        f"popularity={'on' if popularity_on else 'off'}]"
    )

    # Final ranked table.
    print()
    bar = "=" * 72
    print(bar)
    print(f"TOP {min(top_k, len(stage3_result.movie_ids))} RESULTS")
    print(bar)

    if not stage3_result.movie_ids:
        print("  (no results — pool was empty)")
    else:
        top_ids = stage3_result.movie_ids[:top_k]
        cards = await fetch_movie_cards(top_ids)
        cards_by_id = {c["movie_id"]: c for c in cards}
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
            # Final = inclusion_sum − downrank_sum + preference_contribution
            #         + implicit_prior_contribution. Each component prints
            #         alongside the final score so weak/strong contributors
            #         to a movie's rank are visible at a glance.
            print(
                f"  #{rank:<3d}  final={b.final_score:.4f}  "
                f"filter={b.inclusion_sum:.4f}  "
                f"pref={b.preference_contribution:.4f}  "
                f"down={-b.downrank_sum:.4f}  "
                f"impl={b.implicit_prior_contribution:.4f}  "
                f"{title} ({year})  tmdb_id={mid}"
            )

    total_elapsed = time.perf_counter() - total_start
    print(f"\n[total elapsed: {total_elapsed:.2f}s]")


# ============================================================================
# Entry point
# ============================================================================


async def _main(query: str, top_k: int) -> None:
    total_start = time.perf_counter()

    # DB lifecycle mirrors api/cli_search.py. AsyncConnectionPool.open()
    # is idempotent — safe to call on an already-open pool.
    await postgres_pool.open()
    await init_redis()

    try:
        await _step_1(query)
        step2_resp = await _step_2(query)
        ce_diagnostics, stage3_result = await _step_3(query, step2_resp, top_k)
        await _step_4_summary(ce_diagnostics, stage3_result, top_k, total_start)
    finally:
        await close_redis()
        await postgres_pool.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the new search pipeline (steps 1-3 + summary) on a "
            "single query. Step 0 is bypassed; standard flow is assumed."
        )
    )
    parser.add_argument("query", type=str, help="Raw user query.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of ranked results to display (default: {DEFAULT_TOP_K}).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(_main(args.query.strip(), args.top_k))


if __name__ == "__main__":
    main()
