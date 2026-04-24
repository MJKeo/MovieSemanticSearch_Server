# Search V2 — Steps 0-2 Orchestrator
#
# Runs the front half of the V2 search pipeline for a single raw
# user query:
#   1. Steps 0 and 1 fire in parallel on the raw query. Step 0
#      decides which flows execute; step 1 speculatively generates
#      two spins against the standard flow.
#   2. Once both return, the orchestrator dispatches per-flow:
#        - exact_title flow: TODO (not yet implemented).
#        - similarity flow:  TODO (not yet implemented).
#        - standard flow:    fan out into step-2 branches. The branch
#          count is (3 - number of non-standard flows firing), drawing
#          from [original query, spin 1, spin 2] in that order. If the
#          standard flow is not enabled, step 2 is skipped entirely.
#
# Error discipline (treating each flow as independent):
#   - Step 0 failure is fatal — nothing can be dispatched without a
#     routing decision, so the error propagates to the caller.
#   - Step 1 failure is captured and does NOT block the exact_title
#     or similarity flows. It only blows up the step-2 branches that
#     would have used a spin; the original-query branch still runs.
#   - Step 2 failures are captured per-branch so one bad branch
#     doesn't take down the others.

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Literal

from schemas.step_0_flow_routing import Step0Response
from schemas.step_1 import Step1Response
from schemas.step_2 import Step2Response

from search_v2.step_0 import run_step_0
from search_v2.step_1 import run_step_1
from search_v2.step_2 import run_step_2


# ---------------------------------------------------------------------------
# Result dataclasses. Library-only (no CLI), so dataclasses over Pydantic —
# the orchestrator is always called from Python code that can introspect
# the structure directly.
# ---------------------------------------------------------------------------


# One step-2 branch: the input that was fed into step 2, the kind of
# branch (useful for labeling / debugging), and the outcome. Exactly
# one of `response` / `error` is populated.
BranchKind = Literal["original", "spin_1", "spin_2"]


@dataclass
class Step2Branch:
    kind: BranchKind
    # The exact query text handed to step 2. For "original" this is
    # the raw user query verbatim; for spin branches it is the spin's
    # `query` field (step 1 writes these as full search phrases).
    query: str
    # UI label for this branch. Taken from step 1's
    # original_query_label / spin.ui_label when available, otherwise
    # a fallback derived from `kind`.
    ui_label: str
    response: Step2Response | None
    error: str | None
    # Token / elapsed stats are None when the branch failed. Using
    # None (rather than 0) means aggregators that sum token counts
    # naturally skip failed branches instead of silently under-
    # counting as if the LLM had been called for free.
    input_tokens: int | None
    output_tokens: int | None
    elapsed: float | None


@dataclass
class OrchestratorResult:
    # The raw query exactly as the caller passed it (post-strip).
    query: str

    # Step 0 — always present; a failure here raises rather than
    # returning a result with a missing step0 field.
    step0_response: Step0Response
    step0_input_tokens: int
    step0_output_tokens: int
    step0_elapsed: float

    # Step 1 — may be absent if the LLM call failed. Populated
    # independently of whether the standard flow ends up firing
    # (step 1 runs speculatively in parallel with step 0).
    step1_response: Step1Response | None
    step1_error: str | None
    step1_input_tokens: int
    step1_output_tokens: int
    step1_elapsed: float

    # Flow placeholders — True when step 0 routed the flow for
    # execution. Both branches are no-ops today; the flags make it
    # possible for a caller to log what WOULD have run.
    exact_title_flow_executed: bool
    similarity_flow_executed: bool

    # Step 2 branches — empty list when the standard flow did not
    # fire. Ordered: original (if present), then spin_1, then spin_2.
    step2_branches: list[Step2Branch] = field(default_factory=list)

    # Total wall-clock time spent inside the orchestrator, from the
    # first LLM call to the last. Useful for latency accounting.
    total_elapsed: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _run_step_0_timed(query: str) -> tuple[Step0Response, int, int, float]:
    """Wrap run_step_0 to add a wall-clock elapsed measurement.

    run_step_0 does not return elapsed itself (unlike run_step_1 and
    run_step_2), so we time it here for consistency in the result.
    """
    start = time.perf_counter()
    response, in_tok, out_tok = await run_step_0(query)
    return response, in_tok, out_tok, time.perf_counter() - start


def _plan_step2_branches(
    step0: Step0Response,
    step1: Step1Response | None,
    raw_query: str,
) -> list[tuple[BranchKind, str, str]]:
    """Decide which step-2 branches should run.

    Returns a list of (kind, query, ui_label) tuples in the order the
    branches should execute. The standard flow gets a total branch
    budget of 3; every non-standard flow that also fires consumes one
    slot, so the available slots are pulled from [original, spin_1,
    spin_2] in that priority order.

    Returns an empty list when the standard flow is not enabled —
    step 2 is only for the standard flow.
    """
    if not step0.enable_primary_flow:
        return []

    # Count how many non-standard flows fire. Each one displaces a
    # step-2 branch so the overall UI still shows three result lists.
    non_standard_firing = int(step0.exact_title_flow_data.should_be_searched) + int(
        step0.similarity_flow_data.should_be_searched
    )
    budget = 3 - non_standard_firing  # 3, 2, or 1.

    branches: list[tuple[BranchKind, str, str]] = []

    # Slot 1 — original query branch always comes first when the
    # standard flow fires. Label falls back when step 1 failed.
    original_label = (
        step1.original_query_label if step1 is not None else "Original Query"
    )
    branches.append(("original", raw_query, original_label))

    # Slots 2 and 3 — spin branches, only available when step 1
    # succeeded. When step 1 failed we simply return the single
    # original branch; the caller still gets a working standard flow,
    # just without the creative spins.
    if step1 is None:
        return branches[:budget]

    if budget >= 2 and len(step1.spins) >= 1:
        spin = step1.spins[0]
        branches.append(("spin_1", spin.query, spin.ui_label))
    if budget >= 3 and len(step1.spins) >= 2:
        spin = step1.spins[1]
        branches.append(("spin_2", spin.query, spin.ui_label))

    return branches


async def _run_step2_branch(
    kind: BranchKind,
    query: str,
    ui_label: str,
) -> Step2Branch:
    """Run step 2 on one branch, capturing any exception.

    Per-branch error isolation: a failure here produces a Step2Branch
    with `response=None` and `error` populated. The orchestrator
    collects these alongside successful branches so one bad branch
    doesn't take down the sibling branches or the other flows.
    """
    try:
        response, in_tok, out_tok, elapsed = await run_step_2(query)
        return Step2Branch(
            kind=kind,
            query=query,
            ui_label=ui_label,
            response=response,
            error=None,
            input_tokens=in_tok,
            output_tokens=out_tok,
            elapsed=elapsed,
        )
    except Exception as exc:
        # repr keeps the exception type visible — useful for
        # distinguishing validation failures from network errors.
        # Token / elapsed stay None so downstream aggregators don't
        # count a failed branch as a zero-cost successful one.
        return Step2Branch(
            kind=kind,
            query=query,
            ui_label=ui_label,
            response=None,
            error=repr(exc),
            input_tokens=None,
            output_tokens=None,
            elapsed=None,
        )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


async def run_steps_0_to_2(query: str) -> OrchestratorResult:
    """Run steps 0-2 for a single raw user query.

    Steps 0 and 1 run concurrently against the raw query. Based on
    step 0's routing, the standard flow fans out into 1-3 step-2
    branches that also run concurrently. The two non-standard flows
    (exact_title, similarity) are TODO placeholders — their
    should_be_searched bits are surfaced on the result as
    exact_title_flow_executed / similarity_flow_executed so callers
    can log what WOULD have run.

    Error handling:
        - A step 0 failure raises to the caller (routing is required).
        - A step 1 failure is captured on the result; the standard
          flow falls back to just the original-query step-2 branch.
        - A step 2 failure is captured on the specific branch.

    Args:
        query: raw user query (non-empty after stripping).

    Returns:
        OrchestratorResult bundling every sub-step's response, token
        counts, elapsed times, and the dispatch outcomes.
    """
    query = query.strip()
    if not query:
        raise ValueError("query must be a non-empty string.")

    total_start = time.perf_counter()

    # Steps 0 and 1 run in parallel. return_exceptions=True so a
    # step-1 failure does not cancel step 0 (or vice versa) before
    # we can decide what to do with each.
    step0_result, step1_result = await asyncio.gather(
        _run_step_0_timed(query),
        run_step_1(query),
        return_exceptions=True,
    )

    # Step 0 failure is fatal — we cannot dispatch flows without a
    # routing decision. Re-raise the captured exception.
    if isinstance(step0_result, BaseException):
        raise step0_result

    step0_response, step0_in, step0_out, step0_elapsed = step0_result

    # Step 1 may have failed independently — capture the error but
    # keep going. The standard flow (if it fires) will fall back to
    # only the original-query branch, and the non-standard flows
    # don't use step 1 output at all.
    if isinstance(step1_result, BaseException):
        step1_response: Step1Response | None = None
        step1_error: str | None = repr(step1_result)
        step1_in = step1_out = 0
        step1_elapsed = 0.0
    else:
        step1_response, step1_in, step1_out, step1_elapsed = step1_result
        step1_error = None

    # TODO(exact_title flow): dispatch once the exact-title search
    # pipeline is implemented. Today we only surface the routing
    # decision so callers can log what WOULD have run.
    exact_title_flow_executed = (
        step0_response.exact_title_flow_data.should_be_searched
    )

    # TODO(similarity flow): dispatch once the similarity search
    # pipeline is implemented.
    similarity_flow_executed = (
        step0_response.similarity_flow_data.should_be_searched
    )

    # Standard flow — plan the step-2 branches based on the flow-
    # budget rule, then run them in parallel with per-branch error
    # isolation.
    branch_plan = _plan_step2_branches(step0_response, step1_response, query)
    step2_branches: list[Step2Branch] = []
    if branch_plan:
        step2_branches = list(
            await asyncio.gather(
                *(
                    _run_step2_branch(kind, q, label)
                    for kind, q, label in branch_plan
                )
            )
        )

    total_elapsed = time.perf_counter() - total_start

    return OrchestratorResult(
        query=query,
        step0_response=step0_response,
        step0_input_tokens=step0_in,
        step0_output_tokens=step0_out,
        step0_elapsed=step0_elapsed,
        step1_response=step1_response,
        step1_error=step1_error,
        step1_input_tokens=step1_in,
        step1_output_tokens=step1_out,
        step1_elapsed=step1_elapsed,
        exact_title_flow_executed=exact_title_flow_executed,
        similarity_flow_executed=similarity_flow_executed,
        step2_branches=step2_branches,
        total_elapsed=total_elapsed,
    )
