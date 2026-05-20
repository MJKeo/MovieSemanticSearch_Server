"""
run_gradio_ui.py — Streaming Gradio frontend for the search API.

Hits the FastAPI `POST /query_search` endpoint (api/main.py) as a
Server-Sent Events client, then progressively renders per-branch
results as each SSE event arrives. The API is assumed to be running
externally (e.g. `uvicorn api.main:app --port 8000`); the Gradio
process itself does not touch Postgres / Redis / Qdrant.

SSE event grammar (from search_v2/streaming_orchestrator.py):
  fetches_ready  → { fetches: [{id, type, label, ...}] }
  branch_stage   → { fetch_id, stage, label }  (fine-grained progress)
  branch_traits  → { fetch_id, traits: [{surface_text, polarity, commitment}] }
  branch_results → { fetch_id, results: [MovieCard], branch_error: str|None }
  done           → { total_elapsed: float }
  error          → { stage, message }    (fatal Step 0 failure only)

Usage:
    python run_gradio_ui.py
    MOVIE_SEARCH_API_URL=http://otherhost:8000 python run_gradio_ui.py
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import Any, AsyncIterator

import gradio as gr
import httpx
from dotenv import load_dotenv

# Project root on sys.path + .env loading so this script behaves like
# run_search.py / run_orchestrator.py when invoked directly.
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
load_dotenv(_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("MOVIE_SEARCH_API_URL", "http://localhost:8000")

# Cap on movies rendered per branch panel. The API can return long
# ranked lists; the UI only needs the top of each to be useful, and
# capping keeps the @gr.render rebuild cheap (one <img> tag per movie
# per yield).
_MAX_RESULTS_PER_BRANCH = 100

# The API returns TMDB poster paths like "/abc123.jpg" — i.e. just the
# image filename with a leading slash, not a full URL. w500 is the
# standard poster size in TMDB's image config.
_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


def _resolve_poster_url(raw: str | None) -> str | None:
    """Normalize an API poster_url field into a full URL, or None.

    - Already-absolute http(s):// URLs pass through unchanged.
    - Relative TMDB paths ("/abc.jpg") get the TMDB CDN prefix.
    - None / empty → None (caller renders an empty placeholder cell).
    """
    if not raw:
        return None
    if raw.startswith(("http://", "https://", "data:")):
        return raw
    # Anything else is treated as a TMDB relative path. Normalize the
    # leading slash so we don't end up with "..//abc.jpg".
    return _TMDB_IMAGE_BASE + ("" if raw.startswith("/") else "/") + raw


# ---------------------------------------------------------------------------
# SSE consumer
# ---------------------------------------------------------------------------


async def stream_query_search(
    query: str, base_url: str
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """POST to /query_search and yield (event_name, payload) tuples.

    The FastAPI endpoint emits the canonical SSE framing
    `event: NAME\\ndata: <json>\\n\\n`, which we parse line-by-line.
    `read=None` disables the read timeout so long-running pipelines
    don't fail mid-stream.
    """
    timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", f"{base_url}/query_search", json={"query": query}
        ) as resp:
            resp.raise_for_status()
            event_name: str | None = None
            data_buf: list[str] = []
            async for line in resp.aiter_lines():
                if line == "":
                    # Blank line terminates an SSE event frame.
                    if event_name and data_buf:
                        yield event_name, json.loads("\n".join(data_buf))
                    event_name, data_buf = None, []
                elif line.startswith("event:"):
                    event_name = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    # Strip exactly one leading space per the SSE spec.
                    chunk = line[len("data:") :]
                    if chunk.startswith(" "):
                        chunk = chunk[1:]
                    data_buf.append(chunk)
                # Ignore comment lines (": ...") and any other field
                # names; the endpoint doesn't emit them today.


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def _fresh_state() -> dict[str, Any]:
    return {
        "fetches": [],
        "traits": {},
        "results": {},
        # Per-branch current stage. Maps fetch_id → {"stage": str, "label": str}.
        # Cleared (per branch) when `branch_results` arrives for that fetch.
        "stages": {},
        "running": False,
        "total_elapsed": None,
        "fatal_error": None,
    }


def _apply_event(
    state: dict[str, Any], event: str, payload: dict[str, Any]
) -> None:
    """Reduce one SSE event into the UI state dict (in place)."""
    if event == "fetches_ready":
        state["fetches"] = payload.get("fetches", [])
    elif event == "branch_stage":
        # Stage transitions are per-branch and out-of-order across
        # branches; just overwrite. The latest stage for any fetch_id
        # wins, which is what the UI wants to display.
        state["stages"][payload["fetch_id"]] = {
            "stage": payload.get("stage", ""),
            "label": payload.get("label", ""),
        }
    elif event == "branch_traits":
        state["traits"][payload["fetch_id"]] = payload.get("traits", [])
    elif event == "branch_results":
        fid = payload["fetch_id"]
        state["results"][fid] = {
            "movies": payload.get("results", []),
            "error": payload.get("branch_error"),
        }
        # Terminal event for this branch — drop its stage so the UI
        # stops showing a stale progress label next to the gallery.
        state["stages"].pop(fid, None)
    elif event == "done":
        state["total_elapsed"] = payload.get("total_elapsed")
    elif event == "error":
        # Fatal Step 0 failure. `done` still fires afterward.
        state["fatal_error"] = (
            f"[{payload.get('stage', 'unknown')}] {payload.get('message', '')}"
        )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _status_text(state: dict[str, Any]) -> str:
    """One-line status string for the top of the page."""
    if state["fatal_error"]:
        return f"**Error:** {state['fatal_error']}"
    if state["total_elapsed"] is not None:
        n_branches = len(state["results"])
        return f"**Done** in {state['total_elapsed']:.2f}s — {n_branches} branch(es)."
    if state["running"]:
        n_fetches = len(state["fetches"])
        n_results = len(state["results"])
        if n_fetches == 0:
            return "_Running — waiting for fetch plan…_"
        return f"_Running — {n_results} / {n_fetches} branch(es) returned._"
    return "_Idle. Enter a query and hit Search._"


def _header_md(fetch: dict[str, Any]) -> str:
    """Per-branch header line. Shape varies by fetch type."""
    ftype = fetch.get("type", "?")
    fid = fetch.get("id", "?")
    if ftype == "standard":
        q = fetch.get("query", "")
        return f"### `{fid}`\n_query:_ **{q}**"
    if ftype == "exact_title":
        title = fetch.get("title", "?")
        year = fetch.get("release_year")
        year_suffix = f" ({year})" if year else ""
        return f"### `{fid}`\n_exact title:_ **{title}{year_suffix}**"
    if ftype == "similarity":
        refs = fetch.get("references", []) or []
        refs_fmt = ", ".join(
            f"{r.get('title', '?')}"
            + (f" ({r['release_year']})" if r.get("release_year") else "")
            for r in refs
        )
        return f"### `{fid}`\n_references:_ **{refs_fmt or '(none)'}**"
    return f"### `{fid}` _{ftype}_"


def _traits_md(traits: list[dict[str, Any]] | None) -> str | None:
    """Compact one-line trait summary, or None if no traits present."""
    if not traits:
        return None
    parts = []
    for t in traits:
        sign = "+" if t.get("polarity") == "positive" else "−"
        text = t.get("surface_text", "")
        commit = " (committed)" if t.get("commitment") else ""
        parts.append(f"{sign}{text}{commit}")
    return "_traits:_ " + ", ".join(parts)


def _gallery_html(movies: list[dict[str, Any]]) -> str:
    """Build a 6-column poster grid as raw HTML.

    We deliberately avoid gr.Gallery here: Gallery downloads every
    remote URL server-side into its temp cache and re-serves it via
    /file=, so each @gr.render rebuild (one per SSE event) re-pays
    that cost for every poster. Raw <img> tags let the browser hit
    TMDB's CDN directly, cache by URL, and skip the proxy entirely.
    """
    cells: list[str] = []
    for m in movies:
        url = _resolve_poster_url(m.get("poster_url"))
        title = m.get("title") or "—"
        release = m.get("release_date") or ""
        year = release[:4] if release else ""
        caption = html.escape(title + (f" ({year})" if year else ""))
        # 2:3 aspect ratio matches TMDB poster art; padding-top hack
        # reserves the slot before the image loads so the grid doesn't
        # reflow as posters stream in.
        if url:
            img = (
                f'<img src="{html.escape(url, quote=True)}" loading="lazy" '
                f'alt="{caption}" '
                f'style="position:absolute;top:0;left:0;width:100%;height:100%;'
                f'object-fit:contain;border-radius:4px;" />'
            )
        else:
            img = ""
        cells.append(
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'text-align:center;font-size:12px;">'
            '<div style="position:relative;width:100%;padding-top:150%;'
            'background:#222;border-radius:4px;">'
            f'{img}'
            '</div>'
            f'<div style="margin-top:4px;line-height:1.2;">{caption}</div>'
            '</div>'
        )
    return (
        '<div style="display:grid;grid-template-columns:repeat(6,1fr);gap:12px;">'
        + "".join(cells)
        + "</div>"
    )


# ---------------------------------------------------------------------------
# Submit handler
# ---------------------------------------------------------------------------


async def on_search(query: str, state: dict[str, Any]):
    """Async generator driving the streaming UI.

    Yields three outputs each step: (state_dict, status_markdown,
    submit_button_update). State changes drive the @gr.render block;
    the button is disabled while a search is in flight.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        # Don't reset state on empty input — keep prior results visible.
        yield state, "**Enter a query.**", gr.update()
        return

    state = _fresh_state()
    state["running"] = True
    yield state, _status_text(state), gr.update(interactive=False, value="Searching…")

    try:
        async for event, payload in stream_query_search(cleaned, API_BASE_URL):
            _apply_event(state, event, payload)
            yield state, _status_text(state), gr.update()
    except httpx.HTTPStatusError as exc:
        # The API returned 4xx/5xx (e.g. 400 on empty query, although
        # we already guard above). Surface the body.
        body = ""
        try:
            body = exc.response.text
        except Exception:  # noqa: BLE001
            pass
        state["fatal_error"] = f"HTTP {exc.response.status_code}: {body or exc!r}"
    except httpx.HTTPError as exc:
        # Network / connection error (API down, DNS, etc.)
        state["fatal_error"] = f"Network error: {exc!r}"
    finally:
        state["running"] = False
        yield state, _status_text(state), gr.update(interactive=True, value="Search")


# ---------------------------------------------------------------------------
# Gradio Blocks
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Movie Search — Streaming") as demo:
        gr.Markdown(
            f"# Movie Search — Streaming\n"
            f"Hitting `POST {API_BASE_URL}/query_search`. Set "
            f"`MOVIE_SEARCH_API_URL` to point elsewhere."
        )

        with gr.Row():
            query_tb = gr.Textbox(
                label="Query",
                placeholder="e.g. feel-good 90s romcoms, Tom Hanks WWII movies, movies like Inception",
                scale=5,
                autofocus=True,
            )
            submit_btn = gr.Button("Search", variant="primary", scale=1)

        status_md = gr.Markdown("_Idle. Enter a query and hit Search._")

        # gr.State holds the streaming reduction. The @gr.render block
        # below redraws the per-branch panels whenever this changes.
        state = gr.State(value=_fresh_state())

        @gr.render(inputs=state)
        def _render_branches(s: dict[str, Any] | None):
            if not s or not s.get("fetches"):
                return
            for fetch in s["fetches"]:
                fid = fetch["id"]
                with gr.Group():
                    gr.Markdown(_header_md(fetch))
                    traits_line = _traits_md(s["traits"].get(fid))
                    if traits_line:
                        gr.Markdown(traits_line)
                    branch = s["results"].get(fid)
                    if branch is None:
                        # Show the current pipeline stage if we have
                        # one (from a branch_stage event); fall back to
                        # a generic spinner-style line otherwise.
                        stage = s["stages"].get(fid)
                        if stage and stage.get("label"):
                            gr.Markdown(f"⏳ _{stage['label']}_")
                        else:
                            gr.Markdown("⏳ _Starting…_")
                    elif branch["error"]:
                        gr.Markdown(
                            f"**Branch error:** `{branch['error']}`"
                        )
                    elif not branch["movies"]:
                        gr.Markdown("_No results._")
                    else:
                        gr.HTML(
                            _gallery_html(
                                branch["movies"][:_MAX_RESULTS_PER_BRANCH]
                            )
                        )

        # Submit on click or Enter. Both fire the same async generator.
        submit_btn.click(
            on_search,
            inputs=[query_tb, state],
            outputs=[state, status_md, submit_btn],
        )
        query_tb.submit(
            on_search,
            inputs=[query_tb, state],
            outputs=[state, status_md, submit_btn],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    # queue() is required for async generator streaming handlers in
    # Gradio 5.x — without it, only the final yield reaches the client.
    # inbrowser=True opens the default browser tab automatically once
    # the server is ready, so the user doesn't have to click the URL.
    app.queue().launch(inbrowser=True)
