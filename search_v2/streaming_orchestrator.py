# Search V2 — Streaming orchestrator.
#
# Yields progress events as the full pipeline executes, so an HTTP/SSE
# endpoint can surface partial state to the UI before the whole search
# finishes. Mirrors the orchestration in
# `search_v2.full_pipeline_orchestrator.run_full_pipeline`, but instead
# of returning a single FullPipelineResult, this generator emits one
# event per inflection point:
#
#   1. `fetches_ready`  — fires once after Steps 0+1. Lists every
#                          "fetch" (standard branches + non-standard
#                          flows) the pipeline will execute.
#   2. `branch_traits`  — fires per standard-flow branch when Step 2
#                          returns, before Step 3 fan-out. Skipped for
#                          exact-title / similarity (they have no
#                          traits stage).
#   3. `branch_results` — fires per fetch when its execution completes,
#                          in whatever order the underlying tasks
#                          finish.
#   4. `done`           — terminal event with total elapsed seconds.
#   5. `error`          — only on fatal failures (Step 0 fails both
#                          attempts). Per-fetch errors surface inside
#                          the corresponding `branch_results` event via
#                          a `branch_error` field instead.
#
# Concurrency:
#   - Non-standard flows (similarity, exact-title) launch immediately
#     after Step 0 succeeds, in parallel with the standard-flow Step 2
#     tasks.
#   - The auxiliary spec computation (shorts exclusion, reranker-only
#     fallback promotion) is GLOBAL across standard branches, so Stage 4
#     must wait for every standard-flow Step 2/3 task to resolve before
#     dispatching. Within that gate, branches are still parallelized.
#   - On client disconnect, the consuming endpoint cancels the generator,
#     which propagates `GeneratorExit` / `CancelledError`. The `finally`
#     block here cancels every still-pending task to avoid orphans.

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from db.postgres import fetch_movie_card_summaries
from schemas.api_responses import MovieCard
from schemas.step_0_flow_routing import (
    ExactTitleFlowData,
    SimilarityFlowData,
    Step0Response,
)
from schemas.step_1 import Step1Response
from schemas.step_2 import Trait

from search_v2.endpoint_fetching.category_handlers.generated_endpoint_spec import (
    GeneratedEndpointSpec,
)
from search_v2.exact_title_search import run_exact_title_search
from search_v2.full_pipeline_orchestrator import (
    BranchKind,
    Step2BranchResult,
    _apply_implicit_prior_rerank_for_branch,
    _apply_reranker_only_candidate_fallback,
    _build_auxiliary_specs,
    _call_with_retry,
    _finish_branch_after_step2,
    _plan_step2_branches,
    _run_step2_for_branch,
)
from search_v2.similar_movies import run_similarity_search
from search_v2.stage_4_execution import _run_branch as _stage4_run_branch
from search_v2.step_0 import run_step_0
from search_v2.step_1 import run_step_1

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fetch-id constants. Stable strings the UI keys events off of.
# ---------------------------------------------------------------------------

_FETCH_ID_EXACT_TITLE = "exact_title"
_FETCH_ID_SIMILARITY = "similarity"


def _standard_fetch_id(kind: BranchKind) -> str:
    return f"standard:{kind}"


# ---------------------------------------------------------------------------
# Internal task tagging
# ---------------------------------------------------------------------------


# Phase tags so the merge loop knows which branch of the state machine
# a completed task belongs to.
_PHASE_STEP2 = "step2"
_PHASE_STEP3 = "step3"
_PHASE_STAGE4 = "stage4"
_PHASE_NONSTANDARD = "nonstandard"


@dataclass
class _TaskInfo:
    fetch_id: str
    phase: str


@dataclass
class _FetchOutcome:
    """Uniform task-result shape for every Stage 4 / non-standard fetch.

    Each branch task — Stage 4 + implicit prior, similarity, exact-title
    — folds its native result down into this shape (cards + optional
    error). The merge-loop handler then only needs to serialize one
    structure rather than branching by phase.
    """

    cards: list[MovieCard] = field(default_factory=list)
    branch_error: str | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def stream_full_pipeline(
    query: str,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Run the full pipeline and yield `(event_name, payload)` tuples.

    Args:
        query: raw user query (non-empty after stripping).

    Yields:
        (event_name, payload) tuples. The endpoint encodes these as SSE
        wire frames; this generator is transport-agnostic.

    Raises:
        ValueError: if `query` is empty after stripping. Raised before
            the generator yields anything so the endpoint can convert
            it into a 400 response.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must be a non-empty string.")

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Steps 0 + 1 in parallel. Same retry/timeout discipline as the
    # non-streaming orchestrator. Step 0 failure is fatal — without
    # a routing decision there's nothing to dispatch.
    # ------------------------------------------------------------------
    step0_result, step1_result = await asyncio.gather(
        _call_with_retry(lambda: run_step_0(query), label="step_0"),
        _call_with_retry(lambda: run_step_1(query), label="step_1"),
        return_exceptions=True,
    )

    if isinstance(step0_result, BaseException):
        yield ("error", {"stage": "step_0", "message": repr(step0_result)})
        yield ("done", {"total_elapsed": time.perf_counter() - t0})
        return

    step0_response: Step0Response = step0_result[0]
    if isinstance(step1_result, BaseException):
        step1_response: Step1Response | None = None
    else:
        step1_response = step1_result[0]

    # ------------------------------------------------------------------
    # Build the fetch list: standard branches + non-standard flows.
    # ------------------------------------------------------------------
    branch_plan = _plan_step2_branches(step0_response, step1_response, query)
    # Preserve plan order so downstream `auxiliary_specs` sees branches
    # in the same order as the non-streaming orchestrator would.
    standard_fetch_ids_ordered: list[str] = [
        _standard_fetch_id(kind) for kind, _, _ in branch_plan
    ]

    exact_title_firing = step0_response.exact_title_flow_data.should_be_searched
    similarity_firing = step0_response.similarity_flow_data.should_be_searched

    fetches: list[dict[str, Any]] = []
    for kind, branch_query, ui_label in branch_plan:
        fetches.append(
            {
                "id": _standard_fetch_id(kind),
                "type": "standard",
                "label": ui_label,
                "query": branch_query,
            }
        )
    if exact_title_firing:
        et = step0_response.exact_title_flow_data
        fetches.append(
            {
                "id": _FETCH_ID_EXACT_TITLE,
                "type": "exact_title",
                "label": _exact_title_label(et.exact_title_to_search, et.release_year),
                "title": et.exact_title_to_search,
                "release_year": et.release_year,
            }
        )
    if similarity_firing:
        refs = step0_response.similarity_flow_data.references
        fetches.append(
            {
                "id": _FETCH_ID_SIMILARITY,
                "type": "similarity",
                "label": _similarity_label(refs),
                "references": [
                    {
                        "title": r.similar_search_title,
                        "release_year": r.release_year,
                    }
                    for r in refs
                ],
            }
        )

    yield ("fetches_ready", {"fetches": fetches})

    # ------------------------------------------------------------------
    # State for the merge loop.
    # ------------------------------------------------------------------
    # Maps live tasks → metadata so we can dispatch results by phase.
    task_info: dict[asyncio.Task, _TaskInfo] = {}
    # Step 3 results land here so we can pull all branches together
    # before computing auxiliary specs and dispatching Stage 4. Both
    # step2-failures and successful step3 completions populate this dict,
    # so `_all_standard_branches_done` is the single source of truth for
    # whether the Stage 4 gate can fire.
    completed_branches: dict[str, Step2BranchResult] = {}
    stage4_dispatched = False

    # Launch standard-flow Step 2 tasks.
    for kind, branch_query, ui_label in branch_plan:
        task = asyncio.create_task(
            _run_step2_for_branch(kind, branch_query, ui_label)
        )
        task_info[task] = _TaskInfo(
            fetch_id=_standard_fetch_id(kind),
            phase=_PHASE_STEP2,
        )

    # Launch non-standard tasks (independent of the standard-flow gate).
    # Each wrapper folds its native result into a _FetchOutcome (cards
    # + branch_error) so the merge-loop handler serializes one shape.
    if exact_title_firing:
        task = asyncio.create_task(
            _run_exact_title_with_hydration(step0_response.exact_title_flow_data)
        )
        task_info[task] = _TaskInfo(
            fetch_id=_FETCH_ID_EXACT_TITLE,
            phase=_PHASE_NONSTANDARD,
        )
    if similarity_firing:
        task = asyncio.create_task(
            _run_similarity_with_hydration(step0_response.similarity_flow_data)
        )
        task_info[task] = _TaskInfo(
            fetch_id=_FETCH_ID_SIMILARITY,
            phase=_PHASE_NONSTANDARD,
        )

    # Empty-plan short-circuit: no fetches at all → done immediately.
    if not task_info:
        yield ("done", {"total_elapsed": time.perf_counter() - t0})
        return

    needed_step3 = len(branch_plan)

    try:
        # Merge loop — process tasks as they complete, in any order.
        while task_info:
            done, _pending = await asyncio.wait(
                set(task_info.keys()),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for finished in done:
                info = task_info.pop(finished)
                async for event in _handle_finished_task(
                    finished,
                    info,
                    completed_branches=completed_branches,
                    task_info=task_info,
                ):
                    yield event

            # Dispatch Stage 4 once every standard branch has resolved
            # its Step 2 (and Step 3 where applicable). Skip when no
            # standard flow is firing — non-standard fetches drain on
            # their own and we exit when task_info empties.
            if (
                not stage4_dispatched
                and needed_step3 > 0
                and _all_standard_branches_done(
                    completed_branches, standard_fetch_ids_ordered
                )
            ):
                stage4_dispatched = True
                _dispatch_stage4(
                    completed_branches=completed_branches,
                    standard_fetch_ids_ordered=standard_fetch_ids_ordered,
                    task_info=task_info,
                )

        yield ("done", {"total_elapsed": time.perf_counter() - t0})
    finally:
        # Cancel any task that is still alive (client disconnect, or an
        # unexpected error above). Suppress per-task exceptions so
        # cleanup itself never raises.
        for pending_task in list(task_info.keys()):
            pending_task.cancel()
        for pending_task in list(task_info.keys()):
            try:
                await pending_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Merge-loop helpers
# ---------------------------------------------------------------------------


def _all_standard_branches_done(
    completed_branches: dict[str, Step2BranchResult],
    standard_fetch_ids_ordered: list[str],
) -> bool:
    """True iff every standard fetch_id has a Step2BranchResult ready.

    Step 2 failures populate completed_branches with a branch_error;
    successful branches populate after Step 3 fan-out completes. Either
    way, presence in the dict means we know the branch's final shape.
    """
    return all(fid in completed_branches for fid in standard_fetch_ids_ordered)


async def _handle_finished_task(
    task: asyncio.Task,
    info: _TaskInfo,
    *,
    completed_branches: dict[str, Step2BranchResult],
    task_info: dict[asyncio.Task, _TaskInfo],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Process one completed task; may launch follow-on tasks."""
    if info.phase == _PHASE_STEP2:
        async for event in _handle_step2_done(
            task, info, completed_branches=completed_branches, task_info=task_info
        ):
            yield event
        return

    if info.phase == _PHASE_STEP3:
        # Branch is fully prepared for Stage 4. Stash and let the
        # dispatch gate fire when every standard branch is here.
        try:
            branch_result: Step2BranchResult = task.result()
            completed_branches[info.fetch_id] = branch_result
        except Exception as exc:  # noqa: BLE001 — soft-fail per branch
            # Step 3 raising is unexpected (it has soft-fail discipline
            # internally), but treat it like a branch failure rather
            # than tanking the whole pipeline.
            logger.error(
                "Step 3 phase raised unexpectedly for %s: %r",
                info.fetch_id,
                exc,
            )
            completed_branches[info.fetch_id] = Step2BranchResult(
                kind=_kind_from_fetch_id(info.fetch_id),
                query="",
                ui_label="",
                traits=[],
                branch_error=repr(exc),
            )
            yield _branch_results_event(
                info.fetch_id, cards=[], branch_error=repr(exc)
            )
        return

    if info.phase == _PHASE_STAGE4 or info.phase == _PHASE_NONSTANDARD:
        # Both phases produce a _FetchOutcome from their wrapper task,
        # so the handler shape is identical. Unexpected exceptions get
        # caught here and surfaced as a branch_error on this fetch.
        try:
            outcome: _FetchOutcome = task.result()
        except Exception as exc:  # noqa: BLE001 — soft-fail per fetch
            logger.error(
                "%s task raised for %s: %r", info.phase, info.fetch_id, exc
            )
            yield _branch_results_event(
                info.fetch_id, cards=[], branch_error=repr(exc)
            )
            return
        yield _branch_results_event(
            info.fetch_id,
            cards=outcome.cards,
            branch_error=outcome.branch_error,
        )
        return


async def _handle_step2_done(
    task: asyncio.Task,
    info: _TaskInfo,
    *,
    completed_branches: dict[str, Step2BranchResult],
    task_info: dict[asyncio.Task, _TaskInfo],
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Emit branch_traits (or a branch_error result), then start Step 3."""
    try:
        partial, qa = task.result()
    except Exception as exc:  # noqa: BLE001 — _run_step2_for_branch
        # already soft-fails internally; this catches unexpected escapes.
        logger.error(
            "Step 2 phase raised unexpectedly for %s: %r",
            info.fetch_id,
            exc,
        )
        completed_branches[info.fetch_id] = Step2BranchResult(
            kind=_kind_from_fetch_id(info.fetch_id),
            query="",
            ui_label="",
            traits=[],
            branch_error=repr(exc),
        )
        yield _branch_results_event(
            info.fetch_id, cards=[], branch_error=repr(exc)
        )
        return

    if qa is None:
        # Step 2 soft-failed; emit a terminal branch_results event now
        # (no traits, no Stage 4) and record the branch so the global
        # gate still ticks.
        completed_branches[info.fetch_id] = partial
        yield _branch_results_event(
            info.fetch_id, cards=[], branch_error=partial.branch_error
        )
        return

    # Happy path — emit traits, then launch Step 3 fan-out as a task.
    yield (
        "branch_traits",
        {
            "fetch_id": info.fetch_id,
            "traits": [_trait_to_dict(t) for t in qa.traits],
        },
    )
    step3_task = asyncio.create_task(
        _finish_branch_after_step2(
            partial, qa, _kind_from_fetch_id(info.fetch_id)
        )
    )
    task_info[step3_task] = _TaskInfo(
        fetch_id=info.fetch_id, phase=_PHASE_STEP3
    )


def _dispatch_stage4(
    *,
    completed_branches: dict[str, Step2BranchResult],
    standard_fetch_ids_ordered: list[str],
    task_info: dict[asyncio.Task, _TaskInfo],
) -> None:
    """Compute auxiliary specs and launch one Stage 4 task per branch.

    Branches whose Step 2 failed have already emitted their final
    branch_results event; skip Stage 4 for them. The auxiliary
    computation still receives the full ordered list so its global view
    (shorts-exclusion detection, reranker-only fallback) matches the
    non-streaming orchestrator's behavior — empty-trait branches
    contribute no MEDIA_TYPE call and no rerank specs, so passing them
    in is a safe no-op.
    """
    branches_in_order = [
        completed_branches[fid] for fid in standard_fetch_ids_ordered
    ]
    auxiliary = (
        _apply_reranker_only_candidate_fallback(branches_in_order)
        + _build_auxiliary_specs(branches_in_order)
    )
    for fid in standard_fetch_ids_ordered:
        branch = completed_branches[fid]
        if branch.branch_error is not None:
            # Step 2 already failed — branch_results event already
            # emitted with the error. Skip Stage 4 dispatch.
            continue
        task = asyncio.create_task(
            _run_stage4_with_implicit_prior(branch, auxiliary)
        )
        task_info[task] = _TaskInfo(fetch_id=fid, phase=_PHASE_STAGE4)


async def _run_stage4_with_implicit_prior(
    branch: Step2BranchResult,
    auxiliary: list[GeneratedEndpointSpec],
) -> _FetchOutcome:
    """Run Stage 4 for one branch, rerank, then hydrate movie cards.

    Folding hydration into the task — rather than the merge-loop
    handler — lets parallel branches' Postgres lookups overlap.
    Returns the uniform _FetchOutcome shape so the handler doesn't
    need to discriminate on phase to serialize.
    """
    result = await _stage4_run_branch(branch, auxiliary)
    result = await _apply_implicit_prior_rerank_for_branch(branch, result)
    if result.branch_error is not None:
        return _FetchOutcome(cards=[], branch_error=result.branch_error)
    movie_ids = [int(mid) for mid, _ in result.ranked]
    cards = await fetch_movie_card_summaries(movie_ids)
    return _FetchOutcome(cards=cards, branch_error=None)


async def _run_similarity_with_hydration(
    flow_data: SimilarityFlowData,
) -> _FetchOutcome:
    """Run the similarity flow then hydrate the ranked tmdb_ids."""
    result = await run_similarity_search(flow_data)
    movie_ids = [int(r.movie_id) for r in result.ranked]
    cards = await fetch_movie_card_summaries(movie_ids)
    return _FetchOutcome(cards=cards, branch_error=None)


async def _run_exact_title_with_hydration(
    flow_data: ExactTitleFlowData,
) -> _FetchOutcome:
    """Run the exact-title flow then hydrate the ranked tmdb_ids."""
    result = await run_exact_title_search(flow_data)
    movie_ids = [int(mid) for mid, _ in result.ranked]
    cards = await fetch_movie_card_summaries(movie_ids)
    return _FetchOutcome(cards=cards, branch_error=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _kind_from_fetch_id(fetch_id: str) -> BranchKind:
    # Strip the "standard:" prefix to recover the BranchKind literal.
    # Only called on standard-flow fetch_ids.
    _, _, kind = fetch_id.partition("standard:")
    return kind  # type: ignore[return-value]


def _exact_title_label(title: str, release_year: int | None) -> str:
    if release_year is not None:
        return f"{title} ({release_year})"
    return title


def _similarity_label(references: list) -> str:
    """Build a UI label like 'Similar to: A (1999), B'."""
    parts: list[str] = []
    for ref in references:
        if ref.release_year is not None:
            parts.append(f"{ref.similar_search_title} ({ref.release_year})")
        else:
            parts.append(ref.similar_search_title)
    return "Similar to: " + ", ".join(parts) if parts else "Similar to: …"


# ---------------------------------------------------------------------------
# Serializers — kept module-private so the SSE payload shape stays
# centralized.
# ---------------------------------------------------------------------------


def _trait_to_dict(trait: Trait) -> dict[str, Any]:
    return {
        "surface_text": trait.surface_text,
        "polarity": trait.polarity.value,
        "commitment": trait.commitment,
    }


def _branch_results_event(
    fetch_id: str,
    *,
    cards: list[MovieCard],
    branch_error: str | None,
) -> tuple[str, dict[str, Any]]:
    """Build the `branch_results` SSE event for one completed fetch.

    Shape:
        {
          "fetch_id": "<id>",
          "results": [ {tmdb_id, title, release_date, poster_url}, ... ],
          "branch_error": null | "<repr>",
        }

    `results` is empty when the fetch failed (branch_error populated)
    or when no candidate matched.
    """
    return (
        "branch_results",
        {
            "fetch_id": fetch_id,
            "results": [card.model_dump() for card in cards],
            "branch_error": branch_error,
        },
    )
