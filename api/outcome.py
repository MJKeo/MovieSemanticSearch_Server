"""Per-request outcome instrumentation for the API endpoints.

Records a single, uniform verdict on every instrumented request — `outcome.success`
(bool) and, when that is false, `outcome.failure_reason` — on the FastAPI server
span. The verdict is written in exactly ONE place (`record_outcome`); each failure
site only declares *why* it failed by raising `EndpointFailure` with a
`FailureReason`, and that reason bubbles up the call stack to be recorded at the
boundary. Keeping the write single-sourced is deliberate: the same fact (this
request failed, for reason X) written at many exit points is exactly the kind of
duplication that drifts out of sync.

Layering: the outcome *names* (`outcome.success` / `outcome.failure_reason`) live in
`observability/names.py`, the dependency-free name registry. This module is the
API-layer *mechanism* — it imports FastAPI and the OTel SDK — so it sits under
`api/`, not `observability/` (which we keep SDK-plumbing / dependency-free).
"""

from __future__ import annotations

from enum import Enum
from functools import wraps

from fastapi import HTTPException
from opentelemetry import trace

from observability.names import OUTCOME_FAILURE_REASON, OUTCOME_SUCCESS


class FailureReason(str, Enum):
    """Why a request failed, recorded as `outcome.failure_reason` (only when
    `outcome.success` is false).

    Kept intentionally coarse: each member is a distinct, *actionable* class of
    failure — the span's other attributes (which parameter, which tmdb_id) and the
    recorded exception carry the specifics to drill into, so there is no need to
    split a reason finer than the action it implies. Every member is
    low-cardinality, so it stays safe as a future metric label (names.py rule F).
    str-valued so a member serializes directly via `.value`.
    """

    INVALID_PARAMETERS = "invalid_parameters"  # 422 — bad/missing request params
    NOT_INDEXED = "not_indexed"                # 404 — absent from our movie_card index
    TMDB_REMOVED = "tmdb_removed"              # 404 — in our index, gone upstream at TMDB
    TMDB_FETCH_FAILED = "tmdb_fetch_failed"    # 502 — TMDB fetch failed after retries
    INTERNAL_ERROR = "internal_error"          # 500 — unexpected server-side failure


class EndpointFailure(HTTPException):
    """An expected, request-terminating failure that carries its `FailureReason`.

    Subclasses `HTTPException` so FastAPI still renders the intended status code and
    any existing `except HTTPException` handling keeps working unchanged; the added
    `failure_reason` is what `record_outcome` reads to annotate the server span.
    Raise this (instead of a bare `HTTPException`) at every *known* failure site so
    the reason travels with the exception to the single recording point — the site
    that knows *what* failed never has to know *how* the outcome is recorded.
    """

    def __init__(
        self, *, status_code: int, failure_reason: FailureReason, detail: str
    ) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.failure_reason = failure_reason


def record_outcome(handler):
    """Decorate an async endpoint so its `outcome.*` verdict is recorded exactly once.

    Reads the active FastAPI server span (the current span when the handler runs)
    and, around the handler body:
      - `EndpointFailure` -> `success=false` + its carried `failure_reason`;
      - any other `Exception` -> `success=false` + `internal_error` (a genuine bug;
        it is also surfaced via the span error contract / logs elsewhere);
      - clean return -> `success=true`. Reaching the end with no failure raised *is*
        the success verdict (the "final analysis" of the request).
    Every branch re-raises or returns unchanged, so HTTP behavior and the
    auto-instrumentation server-span ERROR status are untouched — this only adds the
    semantic `outcome.*` attributes.

    Apply it *inside* the route decorator so it wraps the handler, not the route
    registration::

        @app.get("/title_search")
        @record_outcome
        async def title_search(...): ...
    """

    @wraps(handler)
    async def wrapper(*args, **kwargs):
        # The auto-instrumented server span is the active span while the handler
        # runs, so this returns the request root — the same reference the handlers
        # capture as `request_span` for their other request-scoped attributes.
        span = trace.get_current_span()
        try:
            result = await handler(*args, **kwargs)
        except EndpointFailure as exc:
            # Expected failure — the reason was decided at the raising site.
            span.set_attribute(OUTCOME_SUCCESS, False)
            span.set_attribute(OUTCOME_FAILURE_REASON, exc.failure_reason.value)
            raise
        except Exception:
            # Unexpected failure — collapse to the single internal_error class; the
            # stack/root cause lives on the recorded exception, not on this key.
            span.set_attribute(OUTCOME_SUCCESS, False)
            span.set_attribute(OUTCOME_FAILURE_REASON, FailureReason.INTERNAL_ERROR.value)
            raise
        span.set_attribute(OUTCOME_SUCCESS, True)
        return result

    return wrapper
