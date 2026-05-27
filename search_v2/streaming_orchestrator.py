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
#   2. `branch_stage`   — fires whenever a single branch transitions
#                          between internal phases (e.g. "executing"
#                          → "rescoring"). Payload:
#                            { fetch_id, stage, label }
#                          where `stage` is a stable enum-like string
#                          and `label` is a human-readable phrase the
#                          UI can render verbatim. Fires multiple times
#                          per branch; no global ordering guarantees
#                          across branches (they run in parallel).
#   3. `branch_traits`  — fires per standard-flow branch when Step 2
#                          returns, before Step 3 fan-out. Skipped for
#                          exact-title / similarity (they have no
#                          traits stage).
#   4. `branch_categories` — fires per standard-flow branch when Step 3
#                          finishes (after `branch_traits`, before
#                          `branch_results`). Payload:
#                            { fetch_id, traits: [{ surface_text,
#                              polarity, step_3_error, category_calls:
#                              [{ category, expressions }] }] }
#                          Trait order matches the same branch's
#                          `branch_traits` payload. Per-trait
#                          `step_3_error` short-circuits that trait's
#                          `category_calls` to `[]`. EXPLICIT_NO_OP
#                          categories and calls with empty expressions
#                          are filtered out. Skipped for the
#                          non-standard flows (same set as
#                          `branch_traits`).
#   5. `branch_results` — fires per fetch when its execution completes,
#                          in whatever order the underlying tasks
#                          finish.
#   6. `done`           — terminal event with total elapsed seconds.
#   7. `error`          — only on fatal failures (Step 0 fails both
#                          attempts). Per-fetch errors surface inside
#                          the corresponding `branch_results` event via
#                          a `branch_error` field instead.
#
# `branch_stage` is emitted via two channels:
#   * Direct `yield` from the orchestrator's own coroutines for stages
#     that happen at task-completion boundaries (e.g. trait emission
#     after Step 2 finishes).
#   * An `asyncio.Queue` populated by worker-level emit callbacks for
#     stages that happen mid-task (e.g. "rescoring" between two awaits
#     inside `_run_stage4_with_implicit_prior`). The merge loop races
#     `asyncio.wait` on tasks against `queue.get()` so progress events
#     flush to the wire as soon as the worker pushes them.
#
# Concurrency:
#   - Non-standard flows (similarity, exact-title) launch immediately
#     after Step 0 succeeds, in parallel with the standard-flow Step 2
#     tasks.
#   - Auxiliary spec computation (shorts exclusion, reranker-only
#     fallback promotion) is PER-BRANCH, so each standard branch
#     dispatches its own Stage 4 the moment its Step 3 finishes — no
#     cross-branch gate. Branches finish at independent times and
#     their `branch_results` events fire as each one lands.
#   - On client disconnect, the consuming endpoint cancels the generator,
#     which propagates `GeneratorExit` / `CancelledError`. The `finally`
#     block here cancels every still-pending task to avoid orphans.

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from db.postgres import fetch_movie_card_summaries
from implementation.classes.schemas import MetadataFilters
from schemas.api_responses import MovieCard
from schemas.enums import HandlerBucket
from schemas.step_0_flow_routing import (
    CharacterFranchiseFlowData,
    ExactTitleFlowData,
    NonCharacterFranchiseFlowData,
    PersonFlowData,
    SimilarityFlowData,
    Step0Response,
    StudioFlowData,
)
from schemas.step_1 import Step1ClarificationResponse, Step1Response
from schemas.step_2 import Trait

from search_v2.endpoint_fetching.category_handlers.generated_endpoint_spec import (
    GeneratedEndpointSpec,
)
from search_v2.character_franchise_search import (
    run_character_franchise_search,
)
from search_v2.exact_title_search import run_exact_title_search
from search_v2.non_character_franchise_search import (
    run_non_character_franchise_search,
)
from search_v2.studio_search import run_studio_search
from search_v2.person_search import run_person_search
from search_v2.full_pipeline_orchestrator import (
    BranchKind,
    CategoryCallWithEndpoints,
    Step2BranchResult,
    TraitWithEndpoints,
    _apply_implicit_prior_rerank_for_branch,
    _call_with_retry,
    _compute_branch_auxiliary,
    _finish_branch_after_step2,
    _plan_step2_branches,
    _run_step2_for_branch,
    _step1_needed,
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
_FETCH_ID_NON_CHARACTER_FRANCHISE = "non_character_franchise"
_FETCH_ID_CHARACTER_FRANCHISE = "character_franchise"
_FETCH_ID_STUDIO = "studio"
_FETCH_ID_PERSON = "person"


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


# ---------------------------------------------------------------------------
# branch_stage labels. Stage keys are stable strings the UI can switch on;
# labels are user-facing prose the UI is free to render verbatim. Kept
# centralized so the wire vocabulary stays consistent across emit sites.
# ---------------------------------------------------------------------------


_STAGE_LABELS: dict[str, str] = {
    # Standard-flow stages. With per-branch Stage 4 dispatch there's
    # no longer a "queries_ready" / waiting state — Step 3 completion
    # immediately transitions the branch to "executing".
    "extracting_traits":    "Extracting traits…",
    "generating_endpoints": "Generating endpoint queries…",
    "executing":            "Executing queries…",
    "rescoring":            "Rescoring results…",
    "hydrating":            "Fetching movie details…",
    # Exact-title flow
    "searching_title":      "Searching by exact title…",
    # Similarity flow
    "resolving_anchors":    "Resolving similar movies…",
    # Person flow
    "resolving_person":     "Resolving person…",
}


def _stage_event(fetch_id: str, stage: str) -> tuple[str, dict[str, Any]]:
    """Build a `branch_stage` SSE event with a known label."""
    return (
        "branch_stage",
        {
            "fetch_id": fetch_id,
            "stage": stage,
            "label": _STAGE_LABELS.get(stage, stage),
        },
    )


# Worker-level emit callback. Synchronous (workers don't have to
# await just to report progress); the closure provided by
# `stream_full_pipeline` puts a pre-built event onto the shared
# progress queue, which the merge loop drains alongside task results.
# Signature: emit_stage(stage_key) -> None.
StageEmitter = Callable[[str], None]


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
    *,
    clarification: str | None = None,
    metadata_filters: MetadataFilters | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Run the full pipeline and yield `(event_name, payload)` tuples.

    Args:
        query: raw user query (non-empty after stripping).
        clarification: optional follow-up correction from the user. When
            present, Steps 0 and 1 switch to clarification-mode prompts;
            Step 1's main_rewrite becomes the slot-1 branch the pipeline
            searches (replacing the verbatim raw query).
        metadata_filters: Optional UI hard filters applied at every
            candidate-generation / reranker primitive. Threaded through
            standard Stage 4 branches and the entity-flow runners
            (exact_title, franchise, studio, person). Intentionally NOT
            threaded into the similarity flow — anchor-based "movies
            like X" is not filter-relevant.

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

    clarification = clarification.strip() if clarification else None

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Steps 0 + 1 in parallel, but as tasks so we can cancel Step 1
    # the moment Step 0's routing decision proves spins are unused
    # (no primary flow, or budget=1 because two non-standard flows
    # fire). Saves a full Step-1 wall-clock window on those queries.
    # Step 0 failure is fatal — without routing nothing dispatches.
    # ------------------------------------------------------------------
    step0_task = asyncio.create_task(
        _call_with_retry(
            lambda: run_step_0(query, clarification=clarification),
            label="step_0",
        )
    )
    step1_task = asyncio.create_task(
        _call_with_retry(
            lambda: run_step_1(query, clarification=clarification),
            label="step_1",
        )
    )

    try:
        step0_result = await step0_task
    except BaseException as step0_exc:
        step1_task.cancel()
        try:
            await step1_task
        except BaseException:
            pass
        yield ("error", {"stage": "step_0", "message": repr(step0_exc)})
        yield ("done", {"total_elapsed": time.perf_counter() - t0})
        return

    step0_response: Step0Response = step0_result[0]

    step1_response: Step1Response | Step1ClarificationResponse | None
    if _step1_needed(step0_response):
        try:
            step1_result = await step1_task
        except BaseException:
            step1_response = None
        else:
            step1_response = step1_result[0]
    else:
        # Step 1's spins would not feed any branch — cancel.
        step1_task.cancel()
        try:
            await step1_task
        except BaseException:
            pass
        step1_response = None

    # ------------------------------------------------------------------
    # Build the fetch list: standard branches + non-standard flows.
    # ------------------------------------------------------------------
    branch_plan = _plan_step2_branches(
        step0_response,
        step1_response,
        query,
        clarification=clarification,
    )

    # Map the new mutually-exclusive entity-flow choice back onto the
    # executors that exist today. Every entity flow now has its own
    # executor — exact_title, similarity, non_character_franchise, and
    # character_franchise all dispatch through this orchestrator.
    exact_title_flow_data: ExactTitleFlowData | None = (
        step0_response.to_exact_title_flow_data()
    )
    similarity_flow_data: SimilarityFlowData | None = (
        step0_response.to_similarity_flow_data()
    )
    non_character_franchise_flow_data: NonCharacterFranchiseFlowData | None = (
        step0_response.to_non_character_franchise_flow_data()
    )
    character_franchise_flow_data: CharacterFranchiseFlowData | None = (
        step0_response.to_character_franchise_flow_data()
    )
    studio_flow_data: StudioFlowData | None = (
        step0_response.to_studio_flow_data()
    )
    person_flow_data: PersonFlowData | None = (
        step0_response.to_person_flow_data()
    )
    exact_title_firing = exact_title_flow_data is not None
    similarity_firing = similarity_flow_data is not None
    non_character_franchise_firing = non_character_franchise_flow_data is not None
    character_franchise_firing = character_franchise_flow_data is not None
    studio_firing = studio_flow_data is not None
    person_firing = person_flow_data is not None

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
        et = exact_title_flow_data
        # Label gets the "Specific Title(s):" UI prefix; query carries
        # just the bare title text since that IS the resolved entity
        # expression. Both fall through `_exact_title_text` so the
        # per-title format stays in sync.
        bare_title = _exact_title_text(
            et.exact_title_to_search, et.release_year
        )
        fetches.append(
            {
                "id": _FETCH_ID_EXACT_TITLE,
                "type": "exact_title",
                "label": f"Specific Title(s): {bare_title}",
                "query": bare_title,
                "title": et.exact_title_to_search,
                "release_year": et.release_year,
            }
        )
    if similarity_firing:
        refs = similarity_flow_data.references
        fetches.append(
            {
                "id": _FETCH_ID_SIMILARITY,
                "type": "similarity",
                "label": _similarity_label(refs),
                "query": _similarity_query(refs),
                "references": [
                    {
                        "title": r.similar_search_title,
                        "release_year": r.release_year,
                    }
                    for r in refs
                ],
            }
        )
    if non_character_franchise_firing:
        canonical = non_character_franchise_flow_data.canonical_name
        fetches.append(
            {
                "id": _FETCH_ID_NON_CHARACTER_FRANCHISE,
                "type": "non_character_franchise",
                "label": f"Franchise: {canonical}",
                "query": _non_character_franchise_query(canonical),
                "canonical_name": canonical,
            }
        )
    if character_franchise_firing:
        canonical = character_franchise_flow_data.canonical_name
        fetches.append(
            {
                "id": _FETCH_ID_CHARACTER_FRANCHISE,
                "type": "character_franchise",
                "label": f"Character Franchise: {canonical}",
                "query": _character_franchise_query(canonical),
                "canonical_name": canonical,
            }
        )
    if studio_firing:
        canonical_names = [
            ref.canonical_name for ref in studio_flow_data.references
        ]
        fetches.append(
            {
                "id": _FETCH_ID_STUDIO,
                "type": "studio",
                "label": _studio_label(canonical_names),
                "query": _studio_query(canonical_names),
                "canonical_names": canonical_names,
            }
        )
    if person_firing:
        person_canonical_names = [
            ref.canonical_name for ref in person_flow_data.references
        ]
        fetches.append(
            {
                "id": _FETCH_ID_PERSON,
                "type": "person",
                "label": _person_label(person_canonical_names),
                "query": _person_query(person_canonical_names),
                "canonical_names": person_canonical_names,
            }
        )

    # Non-standard flows always precede standard-flow branches in the
    # wire payload — the UI surfaces them with priority. Stable sort
    # preserves within-group order: non-standard flows keep their
    # append order above, standard branches keep their planning order.
    fetches.sort(key=lambda f: f["type"] == "standard")

    yield ("fetches_ready", {"fetches": fetches})

    # ------------------------------------------------------------------
    # Progress queue + per-fetch emit callbacks.
    # ------------------------------------------------------------------
    # Workers that span multiple awaits (Stage 4, similarity, exact-
    # title) push `branch_stage` events onto this queue at each phase
    # boundary. The merge loop below races task completion against
    # `queue.get()` so progress flushes to the wire immediately rather
    # than piling up until the next task resolves.
    progress_queue: asyncio.Queue[tuple[str, dict[str, Any]]] = asyncio.Queue()

    def _make_emitter(fetch_id: str) -> StageEmitter:
        def _emit(stage: str) -> None:
            progress_queue.put_nowait(_stage_event(fetch_id, stage))
        return _emit

    # Emit each branch's starting stage right after fetches_ready so the
    # UI has something to show before any task completes. Standard
    # branches begin in Step 2 ("extracting traits"); non-standard flows
    # advertise their respective opening phases. These go through the
    # queue (rather than direct yields) so they interleave naturally
    # with later worker-pushed events in queue order.
    for fetch in fetches:
        ftype = fetch["type"]
        if ftype == "standard":
            progress_queue.put_nowait(_stage_event(fetch["id"], "extracting_traits"))
        elif ftype == "exact_title":
            progress_queue.put_nowait(_stage_event(fetch["id"], "searching_title"))
        elif ftype == "similarity":
            progress_queue.put_nowait(_stage_event(fetch["id"], "resolving_anchors"))
        elif ftype == "non_character_franchise":
            progress_queue.put_nowait(
                _stage_event(fetch["id"], "resolving_franchise")
            )
        elif ftype == "character_franchise":
            progress_queue.put_nowait(
                _stage_event(fetch["id"], "resolving_character_franchise")
            )
        elif ftype == "studio":
            progress_queue.put_nowait(
                _stage_event(fetch["id"], "resolving_studio")
            )
        elif ftype == "person":
            progress_queue.put_nowait(
                _stage_event(fetch["id"], "resolving_person")
            )

    # ------------------------------------------------------------------
    # State for the merge loop.
    # ------------------------------------------------------------------
    # Maps live tasks → metadata so we can dispatch results by phase.
    task_info: dict[asyncio.Task, _TaskInfo] = {}

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
    # Each wrapper also receives an emitter bound to its fetch_id so it
    # can push `branch_stage` events at its internal phase boundaries.
    if exact_title_firing:
        task = asyncio.create_task(
            _run_exact_title_with_hydration(
                exact_title_flow_data,
                emit_stage=_make_emitter(_FETCH_ID_EXACT_TITLE),
                metadata_filters=metadata_filters,
            )
        )
        task_info[task] = _TaskInfo(
            fetch_id=_FETCH_ID_EXACT_TITLE,
            phase=_PHASE_NONSTANDARD,
        )
    if similarity_firing:
        # Note: similarity flow intentionally NOT passed metadata_filters
        # — anchor-based "movies like X" search is not filter-relevant
        # (the user wants similar movies, not filtered ones).
        task = asyncio.create_task(
            _run_similarity_with_hydration(
                similarity_flow_data,
                emit_stage=_make_emitter(_FETCH_ID_SIMILARITY),
            )
        )
        task_info[task] = _TaskInfo(
            fetch_id=_FETCH_ID_SIMILARITY,
            phase=_PHASE_NONSTANDARD,
        )
    if non_character_franchise_firing:
        task = asyncio.create_task(
            _run_non_character_franchise_with_hydration(
                non_character_franchise_flow_data,
                emit_stage=_make_emitter(_FETCH_ID_NON_CHARACTER_FRANCHISE),
                metadata_filters=metadata_filters,
            )
        )
        task_info[task] = _TaskInfo(
            fetch_id=_FETCH_ID_NON_CHARACTER_FRANCHISE,
            phase=_PHASE_NONSTANDARD,
        )
    if character_franchise_firing:
        task = asyncio.create_task(
            _run_character_franchise_with_hydration(
                character_franchise_flow_data,
                emit_stage=_make_emitter(_FETCH_ID_CHARACTER_FRANCHISE),
                metadata_filters=metadata_filters,
            )
        )
        task_info[task] = _TaskInfo(
            fetch_id=_FETCH_ID_CHARACTER_FRANCHISE,
            phase=_PHASE_NONSTANDARD,
        )
    if studio_firing:
        task = asyncio.create_task(
            _run_studio_with_hydration(
                studio_flow_data,
                emit_stage=_make_emitter(_FETCH_ID_STUDIO),
                metadata_filters=metadata_filters,
            )
        )
        task_info[task] = _TaskInfo(
            fetch_id=_FETCH_ID_STUDIO,
            phase=_PHASE_NONSTANDARD,
        )
    if person_firing:
        task = asyncio.create_task(
            _run_person_with_hydration(
                person_flow_data,
                emit_stage=_make_emitter(_FETCH_ID_PERSON),
                metadata_filters=metadata_filters,
            )
        )
        task_info[task] = _TaskInfo(
            fetch_id=_FETCH_ID_PERSON,
            phase=_PHASE_NONSTANDARD,
        )

    # Empty-plan short-circuit: no fetches at all → done immediately.
    if not task_info:
        yield ("done", {"total_elapsed": time.perf_counter() - t0})
        return

    # Sentinel task that pulls the next progress event off the queue.
    # Re-created every time it fires so the merge loop always has
    # exactly one queue-waiter racing alongside the real worker tasks.
    progress_waiter: asyncio.Task = asyncio.create_task(progress_queue.get())

    try:
        # Merge loop — process tasks as they complete, in any order.
        # Each iteration races every worker task PLUS the queue waiter.
        while task_info:
            done, _pending = await asyncio.wait(
                set(task_info.keys()) | {progress_waiter},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for finished in done:
                # Queue-waiter resolved → yield the progress event and
                # rearm. Don't pop from task_info (it isn't in there).
                if finished is progress_waiter:
                    yield finished.result()
                    progress_waiter = asyncio.create_task(progress_queue.get())
                    continue

                info = task_info.pop(finished)
                async for event in _handle_finished_task(
                    finished,
                    info,
                    task_info=task_info,
                    make_emitter=_make_emitter,
                    metadata_filters=metadata_filters,
                ):
                    yield event

        # All worker tasks have drained. Flush any progress events that
        # landed in the queue in the same scheduling tick as the final
        # task completion so the UI sees them before `done`. The waiter
        # may already have a result we never had a chance to yield (it
        # resolved between the final asyncio.wait call and the loop
        # exit) — pull that out first, then drain the rest.
        if progress_waiter.done():
            try:
                yield progress_waiter.result()
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        else:
            progress_waiter.cancel()
        while not progress_queue.empty():
            yield progress_queue.get_nowait()

        yield ("done", {"total_elapsed": time.perf_counter() - t0})
    finally:
        # Cancel any task that is still alive (client disconnect, or an
        # unexpected error above). Suppress per-task exceptions so
        # cleanup itself never raises. The progress_waiter is cancelled
        # here too in case we hit the finally before the happy-path
        # cancel above (e.g. client disconnect mid-stream).
        progress_waiter.cancel()
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


async def _handle_finished_task(
    task: asyncio.Task,
    info: _TaskInfo,
    *,
    task_info: dict[asyncio.Task, _TaskInfo],
    make_emitter: Callable[[str], StageEmitter],
    metadata_filters: MetadataFilters | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """Process one completed task; may launch follow-on tasks.

    Step 3 completions now dispatch *this* branch's Stage 4
    immediately — no waiting for siblings. Auxiliary specs
    (shorts-exclusion, reranker-only fallback) are computed
    per-branch by `_compute_branch_auxiliary`, so each branch has
    everything it needs to execute on its own as soon as its Step 3
    resolves.
    """
    if info.phase == _PHASE_STEP2:
        async for event in _handle_step2_done(
            task, info, task_info=task_info
        ):
            yield event
        return

    if info.phase == _PHASE_STEP3:
        # Step 3 just finished for this branch. Dispatch its Stage 4
        # task right away — no cross-branch gate.
        try:
            branch_result: Step2BranchResult = task.result()
        except Exception as exc:  # noqa: BLE001 — soft-fail per branch
            # Step 3 raising is unexpected (it has soft-fail discipline
            # internally), but treat it like a branch failure rather
            # than tanking the whole pipeline.
            logger.error(
                "Step 3 phase raised unexpectedly for %s: %r",
                info.fetch_id,
                exc,
            )
            yield _branch_results_event(
                info.fetch_id, cards=[], branch_error=repr(exc)
            )
            return

        if branch_result.branch_error is not None:
            # Step 3 soft-failed (downstream LLM call exhausted retries
            # etc.). No Stage 4 to dispatch — emit terminal results now.
            yield _branch_results_event(
                info.fetch_id, cards=[], branch_error=branch_result.branch_error
            )
            return

        # Emit branch_categories before the next stage flips. This is
        # the progressive-reveal point between branch_traits (fired
        # after Step 2) and branch_results (fired after Stage 4): the
        # trait → category → expressions structure is fully resolved at
        # this point and the UI uses it to expand each trait before
        # results land. Direct `yield` lands on the wire ahead of the
        # `emitter("executing")` push below — the emitter routes via the
        # progress queue, which the merge loop drains on a later tick.
        yield (
            "branch_categories",
            {
                "fetch_id": info.fetch_id,
                "traits": [
                    _trait_with_endpoints_to_dict(t)
                    for t in branch_result.traits
                ],
            },
        )

        # Compute per-branch auxiliary specs and launch Stage 4 for
        # this branch. The fallback helper may mutate the branch's
        # own specs in place (reranker → candidate generator), so
        # this must run before the task starts.
        emitter = make_emitter(info.fetch_id)
        auxiliary = _compute_branch_auxiliary(branch_result)
        # Push "executing" through the queue before the task starts
        # so the UI flips out of "generating_endpoints" the instant
        # Step 3 completes (the task itself emits its later
        # sub-stages: rescoring → hydrating).
        emitter("executing")
        stage4_task = asyncio.create_task(
            _run_stage4_with_implicit_prior(
                branch_result, auxiliary, emitter,
                metadata_filters=metadata_filters,
            )
        )
        task_info[stage4_task] = _TaskInfo(
            fetch_id=info.fetch_id, phase=_PHASE_STAGE4
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
        yield _branch_results_event(
            info.fetch_id, cards=[], branch_error=repr(exc)
        )
        return

    if qa is None:
        # Step 2 soft-failed; emit a terminal branch_results event now
        # (no traits, no Stage 4). With per-branch dispatch there's no
        # cross-branch gate to keep ticking.
        yield _branch_results_event(
            info.fetch_id, cards=[], branch_error=partial.branch_error
        )
        return

    # Happy path — emit traits, signal the next phase, then launch
    # Step 3 fan-out as a task. The `generating_endpoints` stage runs
    # for as long as Step 3 takes (per-trait LLM decomposition +
    # handler-LLM endpoint translation). When Step 3 finishes,
    # `_handle_finished_task` dispatches this branch's Stage 4
    # without waiting for siblings.
    yield (
        "branch_traits",
        {
            "fetch_id": info.fetch_id,
            "traits": [_trait_to_dict(t) for t in qa.traits],
        },
    )
    yield _stage_event(info.fetch_id, "generating_endpoints")
    step3_task = asyncio.create_task(
        _finish_branch_after_step2(
            partial, qa, _kind_from_fetch_id(info.fetch_id)
        )
    )
    task_info[step3_task] = _TaskInfo(
        fetch_id=info.fetch_id, phase=_PHASE_STEP3
    )


async def _run_stage4_with_implicit_prior(
    branch: Step2BranchResult,
    auxiliary: list[GeneratedEndpointSpec],
    emit_stage: StageEmitter,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> _FetchOutcome:
    """Run Stage 4 for one branch, rerank, then hydrate movie cards.

    Folding hydration into the task — rather than the merge-loop
    handler — lets parallel branches' Postgres lookups overlap.
    Returns the uniform _FetchOutcome shape so the handler doesn't
    need to discriminate on phase to serialize.

    Emits `branch_stage` events at each internal await boundary so the
    UI can show "rescoring" / "hydrating" instead of going silent
    through the longest phase of the pipeline. The caller has already
    emitted "executing" before this task started.
    """
    # Execute candidate-generator + reranker specs and score traits.
    result = await _stage4_run_branch(
        branch, auxiliary, metadata_filters=metadata_filters,
    )
    emit_stage("rescoring")
    # Apply the implicit-prior post-rerank pass.
    result = await _apply_implicit_prior_rerank_for_branch(branch, result)
    if result.branch_error is not None:
        return _FetchOutcome(cards=[], branch_error=result.branch_error)
    emit_stage("hydrating")
    movie_ids = [int(mid) for mid, _ in result.ranked]
    cards = await fetch_movie_card_summaries(movie_ids)
    return _FetchOutcome(cards=cards, branch_error=None)


async def _run_similarity_with_hydration(
    flow_data: SimilarityFlowData,
    emit_stage: StageEmitter,
) -> _FetchOutcome:
    """Run the similarity flow then hydrate the ranked tmdb_ids.

    The caller emits the initial `resolving_anchors` stage before this
    task starts; we flip to `hydrating` between the similarity search
    and the Postgres card lookup so the UI sees the transition.

    Intentionally does NOT accept a metadata_filters arg — anchor-based
    similarity is not filter-relevant (user wants similar movies, not
    filtered ones).
    """
    result = await run_similarity_search(flow_data)
    emit_stage("hydrating")
    movie_ids = [int(r.movie_id) for r in result.ranked]
    cards = await fetch_movie_card_summaries(movie_ids)
    return _FetchOutcome(cards=cards, branch_error=None)


async def _run_exact_title_with_hydration(
    flow_data: ExactTitleFlowData,
    emit_stage: StageEmitter,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> _FetchOutcome:
    """Run the exact-title flow then hydrate the ranked tmdb_ids.

    The caller emits the initial `searching_title` stage before this
    task starts; we flip to `hydrating` between the title search and
    the Postgres card lookup so the UI sees the transition.
    """
    result = await run_exact_title_search(
        flow_data, metadata_filters=metadata_filters,
    )
    emit_stage("hydrating")
    movie_ids = [int(mid) for mid, _ in result.ranked]
    cards = await fetch_movie_card_summaries(movie_ids)
    return _FetchOutcome(cards=cards, branch_error=None)


async def _run_non_character_franchise_with_hydration(
    flow_data: NonCharacterFranchiseFlowData,
    emit_stage: StageEmitter,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> _FetchOutcome:
    """Run the non-character franchise flow then hydrate the bucketed result.

    The executor returns two ordered buckets (primary then secondary).
    We concatenate in that order and let fetch_movie_card_summaries
    preserve the input ordering — primary-by-popularity followed by
    secondary-by-popularity.
    """
    result = await run_non_character_franchise_search(
        flow_data, metadata_filters=metadata_filters,
    )
    emit_stage("hydrating")
    movie_ids = result.primary_franchise + result.secondary_franchise
    cards = await fetch_movie_card_summaries(movie_ids)
    return _FetchOutcome(cards=cards, branch_error=None)


async def _run_character_franchise_with_hydration(
    flow_data: CharacterFranchiseFlowData,
    emit_stage: StageEmitter,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> _FetchOutcome:
    """Run the character-franchise flow then hydrate the tiered result.

    The executor returns seven tiers in strict priority order
    (lineage-mainline, top-billed-appearance, lineage-ancillary,
    universe, prominent-appearance, relevant-appearance,
    minor-appearance). Concatenating in tier order yields the final
    ranked list; fetch_movie_card_summaries preserves input ordering
    so the tier-then-popularity invariant survives hydration.
    """
    result = await run_character_franchise_search(
        flow_data, metadata_filters=metadata_filters,
    )
    emit_stage("hydrating")
    movie_ids = (
        result.tier_1_lineage_mainline
        + result.tier_2_top_billed_appearance
        + result.tier_3_lineage_ancillary
        + result.tier_4_universe
        + result.tier_5_prominent_appearance
        + result.tier_6_relevant_appearance
        + result.tier_7_minor_appearance
    )
    cards = await fetch_movie_card_summaries(movie_ids)
    return _FetchOutcome(cards=cards, branch_error=None)


async def _run_studio_with_hydration(
    flow_data: StudioFlowData,
    emit_stage: StageEmitter,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> _FetchOutcome:
    """Run the studio flow then hydrate the single popularity-sorted list.

    The executor returns one flat `ranked` list — no tiers, no per-
    movie score. fetch_movie_card_summaries preserves input ordering,
    so the popularity-DESC invariant established by run_studio_search
    survives hydration.
    """
    result = await run_studio_search(
        flow_data, metadata_filters=metadata_filters,
    )
    emit_stage("hydrating")
    cards = await fetch_movie_card_summaries(result.ranked)
    return _FetchOutcome(cards=cards, branch_error=None)


async def _run_person_with_hydration(
    flow_data: PersonFlowData,
    emit_stage: StageEmitter,
    *,
    metadata_filters: MetadataFilters | None = None,
) -> _FetchOutcome:
    """Run the person flow then hydrate the bucketed result.

    The executor returns four buckets in strict priority order
    (lead, major, relevant, minor). Concatenating in bucket order
    yields the final ranked list; fetch_movie_card_summaries
    preserves input ordering so the bucket-then-overlap-then-popularity
    invariant survives hydration.
    """
    result = await run_person_search(
        flow_data, metadata_filters=metadata_filters,
    )
    emit_stage("hydrating")
    movie_ids = (
        result.bucket_1_lead
        + result.bucket_2_major
        + result.bucket_3_relevant
        + result.bucket_4_minor
    )
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


def _exact_title_text(title: str, release_year: int | None) -> str:
    """Format a single exact-title reference: 'Inception (2010)' or 'Inception' (no year)."""
    if release_year is not None:
        return f"{title} ({release_year})"
    return title


def _format_similarity_refs(references: list) -> list[str]:
    """Format each similarity ref as 'Title (Year)' or just 'Title'.

    Shared by `_similarity_label` and `_similarity_query` so the
    label and query stay aligned if the per-ref format changes.
    """
    return [
        f"{r.similar_search_title} ({r.release_year})"
        if r.release_year is not None
        else r.similar_search_title
        for r in references
    ]


def _similarity_label(references: list) -> str:
    """Build a UI label like 'Similar to: A (1999), B'."""
    parts = _format_similarity_refs(references)
    return "Similar to: " + ", ".join(parts) if parts else "Similar to: …"


def _studio_label(canonical_names: list[str]) -> str:
    """Build a UI label like 'A24' or 'A24 + Neon' for the studio fetch."""
    if not canonical_names:
        return "Studio"
    if len(canonical_names) == 1:
        return canonical_names[0]
    return " + ".join(canonical_names)


def _person_label(canonical_names: list[str]) -> str:
    """Build a UI label like 'Tom Hanks' or 'Tom Hanks + Meg Ryan' for the person fetch."""
    if not canonical_names:
        return "Person"
    if len(canonical_names) == 1:
        return canonical_names[0]
    return " + ".join(canonical_names)


# --- `query` builders for non-standard flows ---
# Natural-language descriptions of what each entity-flow branch is
# fetching. Surfaced verbatim in the `query` field of every fetch
# entry so the frontend can show the same shape it shows for
# standard-branch fetches (where `query` is the LLM-expanded sub-query).


def _join_canonical_names(canonical_names: list[str]) -> str:
    """Join 1+ names as 'A', 'A and B', or 'A, B, and C'."""
    n = len(canonical_names)
    if n == 0:
        return ""
    if n == 1:
        return canonical_names[0]
    if n == 2:
        return f"{canonical_names[0]} and {canonical_names[1]}"
    return ", ".join(canonical_names[:-1]) + f", and {canonical_names[-1]}"


def _similarity_query(references: list) -> str:
    """Build a query like 'movies similar to A (1999), B'."""
    parts = _format_similarity_refs(references)
    if not parts:
        return "movies similar to …"
    return "movies similar to " + ", ".join(parts)


def _non_character_franchise_query(canonical_name: str) -> str:
    """e.g. 'the Star Wars franchise'."""
    return f"the {canonical_name} franchise"


def _character_franchise_query(canonical_name: str) -> str:
    """e.g. 'films featuring Batman'."""
    return f"films featuring {canonical_name}"


def _studio_query(canonical_names: list[str]) -> str:
    """e.g. 'A24 films' or 'A24 and Neon films'."""
    joined = _join_canonical_names(canonical_names)
    return f"{joined} films" if joined else "studio films"


def _person_query(canonical_names: list[str]) -> str:
    """e.g. 'Christopher Nolan films' or 'Tom Hanks and Meg Ryan films'."""
    joined = _join_canonical_names(canonical_names)
    return f"{joined} films" if joined else "person films"


# ---------------------------------------------------------------------------
# Serializers — kept module-private so the SSE payload shape stays
# centralized.
# ---------------------------------------------------------------------------


def _trait_to_dict(trait: Trait) -> dict[str, Any]:
    return {
        "surface_text": trait.surface_text,
        "polarity": trait.polarity.value,
        "commitment": trait.commitment,
        # Folded user-intent restatement from Step 2 — frontend reveals
        # this when the user expands a trait node.
        "contextualized_phrase": trait.contextualized_phrase,
    }


def _category_call_to_dict(call: CategoryCallWithEndpoints) -> dict[str, Any]:
    return {
        "category": call.category.value,
        # User-facing description carried inline (cheap; saves the
        # frontend from maintaining a parallel category dictionary).
        "definition": call.category.description,
        "retrieval_intent": call.retrieval_intent,
        "expressions": call.expressions,
    }


def _trait_with_endpoints_to_dict(
    trait: TraitWithEndpoints,
) -> dict[str, Any]:
    if trait.step_3_error is not None:
        # Per-trait Step 3 failure — frontend renders an error leaf.
        category_calls: list[dict[str, Any]] = []
    else:
        # Independent filters (one does not imply the other):
        #   * EXPLICIT_NO_OP bucket — category deliberately not wired
        #     to an endpoint (e.g. BELOW_THE_LINE_CREATOR). Identity
        #     check matches handler.py's canonical pattern.
        #   * empty expressions — handler ran but resolved to nothing.
        category_calls = [
            _category_call_to_dict(call)
            for call in trait.category_calls
            if call.category.bucket is not HandlerBucket.EXPLICIT_NO_OP
            and len(call.expressions) > 0
        ]
    return {
        "surface_text": trait.surface_text,
        "polarity": trait.polarity.value,
        "step_3_error": trait.step_3_error,
        "category_calls": category_calls,
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
            # MovieCard is a msgspec.Struct — msgspec.json.encode at
            # the endpoint will encode it natively, no per-card
            # model_dump() materialization needed.
            "results": cards,
            "branch_error": branch_error,
        },
    )
