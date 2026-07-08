"""Request-scoped LLM + embedding usage accumulation (cost + tokens).

A single `/query_search` request fans out into many LLM calls (Steps 0-3,
implicit expectations, per-trait decomposition, per-category handlers, Stage-4
query translation) and many embedding calls (the semantic endpoints). Each of
those knows its own USD cost and token usage, but nothing sums them onto the
request's server span. This module provides that rollup — one object carrying
both the total dollar cost and the total input / cached-input / output tokens
for the whole request.

Mechanism — a `ContextVar` holding a STACK of accumulators:
  - The `/query_search` handler enters `track_request_cost()` once, at the top
    of its streaming generator, BEFORE the pipeline spawns any `asyncio` tasks.
    That pushes the request-level (root) accumulator onto the stack.
  - `asyncio.create_task` snapshots the current context, so every downstream
    task (at any depth) inherits the SAME stack by reference and can
    `add_request_cost(...)` into it. The whole `/query_search` path is pure
    asyncio (no threads), so a plain, non-locked `+=` is safe.
  - Because the contextvar defaults to an empty stack, `add_request_cost(...)`
    is a no-op outside a tracked request — so the router and embedder can call
    it unconditionally without affecting the offline ingestion / eval paths,
    which never enter `track_request_cost()`.

Per-stage attribution — `track_stage_cost()`:
  - A pipeline stage (Step 2, decomposition, candidate generation, rerankers)
    enters `track_stage_cost()` INSIDE its own coroutine, before it spawns its
    fan-out. That pushes a fresh CHILD accumulator onto the stack.
  - `add_request_cost` / `add_request_tokens` add to EVERY accumulator on the
    stack, so the root always holds the full request total (unchanged for the
    server-span rollup) while each child holds only what was incurred within
    its scope. No delta arithmetic, no roll-up-on-exit — the child is complete
    the moment its scope ends, and because context is snapshotted at task
    creation, a stage entered before its fan-out isolates its subtree even when
    sibling branches run the same stage concurrently.

Dependency-free (stdlib only), mirroring `observability/names.py`, so the
low-level LLM router can import it without dragging in the OTel SDK.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Iterator, Optional


@dataclass
class RequestCostAccumulator:
    """Running cost + token totals for one request's LLM + embedding calls.

    Mutated in place from many `asyncio` tasks that share this object by
    reference (never rebound via `ContextVar.set`, which would not propagate
    back to the parent). Safe without a lock under the single-threaded asyncio
    invariant of the search path.

    Both cost and tokens sum over ALL billed attempts (a retried/failed-but-
    billed attempt still counts) — the accounting call sites fire per attempt,
    before the parse that may fail, because a billed attempt is paid for
    regardless of whether it ultimately succeeds.
    """

    total_usd: float = 0.0
    # Token totals. `total_cached_input_tokens` is a SUBSET of
    # `total_input_tokens` (cached input tokens are a discounted slice of the
    # input, never additive to it — the same "cached ⊆ input" convention the
    # per-call `gen_ai.usage.*` attributes use), so it is tracked alongside,
    # never on top.
    total_input_tokens: int = 0
    total_cached_input_tokens: int = 0
    total_output_tokens: int = 0

    def add(self, cost_usd: Optional[float]) -> None:
        """Add one call's cost. `None` (unpriced model) and 0 are ignored."""
        if cost_usd:
            self.total_usd += cost_usd

    def add_tokens(
        self,
        input_tokens: Optional[int],
        cached_input_tokens: Optional[int],
        output_tokens: Optional[int],
    ) -> None:
        """Add one call's token usage.

        Accumulated UNCONDITIONALLY — unlike `add()`, this is not gated on the
        model being priced: tokens are billed (and worth counting) whether or
        not the pricing table knows the model, so an unpriced model still
        contributes to the token rollup even though it contributes `0` to cost.
        `None`/`0` fields are treated as zero.
        """
        self.total_input_tokens += input_tokens or 0
        self.total_cached_input_tokens += cached_input_tokens or 0
        self.total_output_tokens += output_tokens or 0


# Defaults to an empty stack: no accumulator means "not inside a tracked
# request", so add_request_cost() short-circuits and the offline callers are
# unaffected. The stack is stored as an immutable tuple and only ever rebound
# via `ContextVar.set` (never mutated in place), so pushes/pops in a child task
# never leak back into a parent's view — while the accumulator OBJECTS the tuple
# holds are shared by reference, which is exactly how downstream tasks add into
# the same root.
_request_cost: contextvars.ContextVar[tuple[RequestCostAccumulator, ...]] = (
    contextvars.ContextVar("request_cost_stack", default=())
)


@contextlib.contextmanager
def track_request_cost() -> Iterator[RequestCostAccumulator]:
    """Scope a fresh request-level cost accumulator to the current context.

    Enter this ONCE per request, before the pipeline spawns its tasks, so every
    task inherits the stack. Pushes the root accumulator and yields it so the
    caller can read `.total_usd` at the end (e.g. to write it onto the request
    span) before the context var is reset. Nested `track_stage_cost()` scopes
    push children above this root; because every add fans out to the whole
    stack, the root still sees the full request total.
    """
    accumulator = RequestCostAccumulator()
    token = _request_cost.set(_request_cost.get() + (accumulator,))
    try:
        yield accumulator
    finally:
        _request_cost.reset(token)


@contextlib.contextmanager
def track_stage_cost() -> Iterator[RequestCostAccumulator]:
    """Scope a child accumulator capturing one pipeline stage's cost.

    Enter this INSIDE the stage's own coroutine, BEFORE the stage spawns its
    fan-out, so the child is on the stack when the fan-out tasks snapshot the
    context. Yields the child accumulator, whose `.total_usd` after the scope is
    exactly this stage's LLM + embedding cost — isolated from sibling branches
    running the same stage concurrently, because each gets its own child on its
    own context copy. A no-op outside a tracked request (empty stack), matching
    `track_request_cost()`.
    """
    accumulator = RequestCostAccumulator()
    token = _request_cost.set(_request_cost.get() + (accumulator,))
    try:
        yield accumulator
    finally:
        _request_cost.reset(token)


def add_request_cost(cost_usd: Optional[float]) -> None:
    """Add one call's USD cost to every active accumulator on the stack, if any.

    Fanning out to the whole stack keeps the request-level root total complete
    while each nested stage accumulator collects only its own scope's cost.
    No-op when called outside `track_request_cost()` (offline ingestion / eval),
    so LLM/embedding call sites can invoke it unconditionally.
    """
    for accumulator in _request_cost.get():
        accumulator.add(cost_usd)


def add_request_tokens(
    input_tokens: Optional[int],
    cached_input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> None:
    """Add one call's token usage to every active accumulator on the stack.

    Fans out to the whole stack (see `add_request_cost`). No-op when called
    outside `track_request_cost()` (offline ingestion / eval), so LLM/embedding
    call sites can invoke it unconditionally. See
    `RequestCostAccumulator.add_tokens` for the cached-subset and unpriced-model
    semantics.
    """
    for accumulator in _request_cost.get():
        accumulator.add_tokens(input_tokens, cached_input_tokens, output_tokens)
