"""Request-scoped LLM + embedding usage accumulation (cost + tokens).

A single `/query_search` request fans out into many LLM calls (Steps 0-3,
implicit expectations, per-trait decomposition, per-category handlers, Stage-4
query translation) and many embedding calls (the semantic endpoints). Each of
those knows its own USD cost and token usage, but nothing sums them onto the
request's server span. This module provides that rollup — one object carrying
both the total dollar cost and the total input / cached-input / output tokens
for the whole request.

Mechanism — a `ContextVar` holding a mutable accumulator:
  - The `/query_search` handler enters `track_request_cost()` once, at the top
    of its streaming generator, BEFORE the pipeline spawns any `asyncio` tasks.
  - `asyncio.create_task` snapshots the current context, so every downstream
    task (at any depth) inherits the SAME accumulator object by reference and
    can `add_request_cost(...)` into it. The whole `/query_search` path is pure
    asyncio (no threads), so a plain, non-locked `+=` is safe.
  - Because the contextvar defaults to `None`, `add_request_cost(...)` is a
    no-op outside a tracked request — so the router and embedder can call it
    unconditionally without affecting the offline ingestion / eval paths, which
    never enter `track_request_cost()`.

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


# Defaults to None: no accumulator means "not inside a tracked request", so
# add_request_cost() short-circuits and the offline callers are unaffected.
_request_cost: contextvars.ContextVar[Optional[RequestCostAccumulator]] = (
    contextvars.ContextVar("request_cost_accumulator", default=None)
)


@contextlib.contextmanager
def track_request_cost() -> Iterator[RequestCostAccumulator]:
    """Scope a fresh cost accumulator to the current context.

    Enter this ONCE per request, before the pipeline spawns its tasks, so every
    task inherits the accumulator. Yields the accumulator so the caller can read
    `.total_usd` at the end (e.g. to write it onto the request span) before the
    context var is reset.
    """
    accumulator = RequestCostAccumulator()
    token = _request_cost.set(accumulator)
    try:
        yield accumulator
    finally:
        _request_cost.reset(token)


def add_request_cost(cost_usd: Optional[float]) -> None:
    """Add one call's USD cost to the active request accumulator, if any.

    No-op when called outside `track_request_cost()` (offline ingestion / eval),
    so LLM/embedding call sites can invoke it unconditionally.
    """
    accumulator = _request_cost.get()
    if accumulator is not None:
        accumulator.add(cost_usd)


def add_request_tokens(
    input_tokens: Optional[int],
    cached_input_tokens: Optional[int],
    output_tokens: Optional[int],
) -> None:
    """Add one call's token usage to the active request accumulator, if any.

    No-op when called outside `track_request_cost()` (offline ingestion / eval),
    so LLM/embedding call sites can invoke it unconditionally. See
    `RequestCostAccumulator.add_tokens` for the cached-subset and unpriced-model
    semantics.
    """
    accumulator = _request_cost.get()
    if accumulator is not None:
        accumulator.add_tokens(input_tokens, cached_input_tokens, output_tokens)
