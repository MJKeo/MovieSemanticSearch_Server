"""Process-wide event-loop bootstrap.

Centralizes the uvloop install so every entry point — CLI runners, the
FastAPI app, batch scripts — picks the same accelerated event loop
without each file duplicating the boilerplate. uvloop reports a ~2x
throughput win over the default selector loop on socket-heavy workloads
(many concurrent httpx / qdrant / psycopg calls), which is exactly the
shape of the similarity-search hot path.

`install_uvloop()` is idempotent and silently no-ops on platforms where
uvloop is unavailable (e.g., Windows). Callers must invoke it BEFORE
the event loop is started — i.e., before `asyncio.run(...)` or before
the ASGI runner (uvicorn) is launched.
"""

from __future__ import annotations

import asyncio


def install_uvloop() -> None:
    """Set the asyncio event-loop policy to uvloop's when available.

    Safe to call multiple times. No-ops cleanly if uvloop can't be
    imported on the current platform.
    """
    try:
        import uvloop
    except ImportError:
        return
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
