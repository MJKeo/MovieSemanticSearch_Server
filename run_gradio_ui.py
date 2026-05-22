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
_MAX_RESULTS_PER_BRANCH = 25

# The API returns TMDB poster paths like "/abc123.jpg" — i.e. just the
# image filename with a leading slash, not a full URL. w500 is the
# standard poster size in TMDB's image config.
_TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"


# ---------------------------------------------------------------------------
# Hard-filter choices
# ---------------------------------------------------------------------------
# These mirror the six hard filters defined in
# implementation/classes/schemas.py (release_ts range, runtime range,
# maturity range, genres, audio languages, watch offer keys) — the
# pre-filters applied to both Postgres and Qdrant before scoring.
#
# The choices below are duplicated here (rather than imported from the
# enums) to keep this UI module import-light: it talks to the API over
# HTTP and otherwise has no need for direct schema access. When these
# filters are wired into the request payload, swap to enum-driven
# choices so the labels can't drift.

# Genre enum values (implementation/classes/enums.py:117-183).
_GENRE_CHOICES = [
    "Action", "Adventure", "Animation", "Biography", "Comedy", "Crime",
    "Documentary", "Drama", "Family", "Fantasy", "Film-Noir", "Game-Show",
    "History", "Horror", "Music", "Musical", "Mystery", "News",
    "Reality-TV", "Romance", "Sci-Fi", "Short", "Sport", "Talk-Show",
    "Thriller", "War", "Western",
]

# Common audio languages. The real enum has 200+ entries; we surface
# a representative subset here until the filter is wired up — at which
# point this should be sourced from implementation/classes/languages.py.
_LANGUAGE_CHOICES = [
    "English", "Spanish", "French", "German", "Italian", "Japanese",
    "Korean", "Mandarin Chinese", "Cantonese", "Portuguese", "Russian",
    "Hindi", "Arabic", "Turkish", "Dutch", "Swedish", "Polish",
    "Danish", "Norwegian", "Finnish", "Thai", "Vietnamese", "Hebrew",
]

# Maturity labels in ascending order (rank 1 → 5 — see MaturityRating
# in implementation/classes/enums.py). Backend stores the ordinal, but
# users pick the human-readable rating; the UI maps label → rank via
# _MATURITY_TO_RANK below.
_MATURITY_CHOICES = ["G", "PG", "PG-13", "R", "NC-17"]
_MATURITY_TO_RANK: dict[str, int] = {
    "G": 1, "PG": 2, "PG-13": 3, "R": 4, "NC-17": 5,
}

# UI display label → StreamingService enum value (string). The API
# expands each enum value into the underlying TMDB provider-id list
# via STREAMING_PROVIDER_MAP. Source of truth for both sides is
# implementation/classes/watch_providers.py:STREAMING_SERVICE_DISPLAY_NAMES;
# we mirror it here to keep the UI module import-light (no schema
# imports — the UI only talks to the API over HTTP).
_PROVIDER_DISPLAY_TO_SERVICE: dict[str, str] = {
    "Netflix": "netflix",
    "Amazon Prime Video": "amazon",
    "Hulu": "hulu",
    "Disney+": "disney",
    "Max": "max",
    "Peacock": "peacock",
    "Paramount+": "paramount",
    "Apple TV+": "apple",
    "Crunchyroll": "crunchyroll",
    "fuboTV": "fubotv",
    "YouTube": "youtube",
    "AMC+": "amc",
    "Starz": "starz",
    "Tubi": "tubi",
    "Pluto TV": "pluto",
    "The Roku Channel": "roku",
    "Plex": "plex",
    "Shudder": "shudder",
    "MGM+": "mgm",
    "Fandango at Home": "vudu",
}
_PROVIDER_CHOICES = list(_PROVIDER_DISPLAY_TO_SERVICE.keys())

# Runtime slider bounds in minutes. 0–360 covers everything from
# shorts to the longest features without making the slider unwieldy.
# The "any max" sentinel is _RUNTIME_MAX: when the right-hand slider
# sits at that value, the filter is interpreted as no upper bound
# (the wire payload omits max_runtime entirely in that case).
_RUNTIME_MIN, _RUNTIME_MAX = 0, 360


def _collect_filters(
    release_mode: str,
    release_before_date: float | None,
    release_after_date: float | None,
    release_between_from: float | None,
    release_between_to: float | None,
    runtime_enable: bool,
    min_runtime: float | None,
    max_runtime: float | None,
    maturity_mode: str,
    maturity_anchor: str | None,
    genres_enable: bool,
    genres_selected: list[str] | None,
    languages_enable: bool,
    languages_selected: list[str] | None,
    providers_enable: bool,
    providers_selected: list[str] | None,
) -> dict[str, Any] | None:
    """Build the `filters` JSON payload from the current UI state.

    Returns None when no filter is active so the request body omits the
    field entirely (the API skips filter-construction overhead on
    unfiltered queries).

    Each section is self-contained: a checkbox / mode selector decides
    whether to emit the field, and individual sliders / dropdowns
    contribute None when at their "no constraint" sentinel value
    (e.g. max_runtime == 360 → omit the upper bound entirely).
    """
    filters: dict[str, Any] = {}

    # Release date — mode radio decides which window we emit. DateTime
    # widgets return Unix timestamps (seconds since epoch); the API
    # expects ints. We `int(...)` to drop any fractional part Gradio
    # may emit when include_time=False.
    if release_mode == "Before" and release_before_date is not None:
        filters["max_release_ts"] = int(release_before_date)
    elif release_mode == "After" and release_after_date is not None:
        filters["min_release_ts"] = int(release_after_date)
    elif release_mode == "Between":
        # If both bounds are set in reverse order, swap so the API
        # always sees min <= max. Equal values (single-day window)
        # are kept as-is. Without this guard a reversed selection
        # would produce an empty BETWEEN range silently.
        lo = release_between_from
        hi = release_between_to
        if lo is not None and hi is not None and lo > hi:
            lo, hi = hi, lo
        if lo is not None:
            filters["min_release_ts"] = int(lo)
        if hi is not None:
            filters["max_release_ts"] = int(hi)

    # Runtime — enabled iff the checkbox is checked. min=0 and max=360
    # are sentinels for "no lower / upper bound" so we omit them rather
    # than sending the default values.
    if runtime_enable:
        if min_runtime is not None and int(min_runtime) > _RUNTIME_MIN:
            filters["min_runtime"] = int(min_runtime)
        if max_runtime is not None and int(max_runtime) < _RUNTIME_MAX:
            filters["max_runtime"] = int(max_runtime)

    # Maturity — comparison-type radio folds the inactive state in.
    # "At most X" → max_maturity_rank=rank(X); "At least X" → min.
    if maturity_mode != "Inactive" and maturity_anchor:
        rank = _MATURITY_TO_RANK.get(maturity_anchor)
        if rank is not None:
            if maturity_mode == "At most":
                filters["max_maturity_rank"] = rank
            elif maturity_mode == "At least":
                filters["min_maturity_rank"] = rank

    # Genres / languages — labels pass through verbatim; the API
    # resolves them to Genre / Language enum members.
    if genres_enable and genres_selected:
        filters["genres"] = list(genres_selected)
    if languages_enable and languages_selected:
        filters["audio_languages"] = list(languages_selected)

    # Streaming providers — translate UI labels to StreamingService
    # enum values so the API can do a clean `StreamingService(token)`
    # lookup. Unknown labels are silently dropped (shouldn't happen
    # since the dropdown is constrained to _PROVIDER_CHOICES).
    if providers_enable and providers_selected:
        services = [
            _PROVIDER_DISPLAY_TO_SERVICE[label]
            for label in providers_selected
            if label in _PROVIDER_DISPLAY_TO_SERVICE
        ]
        if services:
            filters["streaming_services"] = services

    return filters or None


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
    query: str,
    base_url: str,
    filters: dict[str, Any] | None = None,
) -> AsyncIterator[tuple[str, dict[str, Any]]]:
    """POST to /query_search and yield (event_name, payload) tuples.

    The FastAPI endpoint emits the canonical SSE framing
    `event: NAME\\ndata: <json>\\n\\n`, which we parse line-by-line.
    `read=None` disables the read timeout so long-running pipelines
    don't fail mid-stream.

    When `filters` is not None, the dict is sent as the `filters` field
    of the request body — its shape mirrors MetadataFiltersInput in
    api/main.py. When None, the field is omitted so the API
    short-circuits filter plumbing.
    """
    payload: dict[str, Any] = {"query": query}
    if filters:
        payload["filters"] = filters
    timeout = httpx.Timeout(connect=10.0, read=None, write=10.0, pool=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", f"{base_url}/query_search", json=payload
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


async def on_search(
    query: str,
    state: dict[str, Any],
    # Filter widget values. Gradio passes these as positional args in
    # the order their components are listed in the submit handler's
    # `inputs=` array — see build_app() below for the binding order.
    release_mode: str,
    release_before_date: float | None,
    release_after_date: float | None,
    release_between_from: float | None,
    release_between_to: float | None,
    runtime_enable: bool,
    min_runtime: float | None,
    max_runtime: float | None,
    maturity_mode: str,
    maturity_anchor: str | None,
    genres_enable: bool,
    genres_selected: list[str] | None,
    languages_enable: bool,
    languages_selected: list[str] | None,
    providers_enable: bool,
    providers_selected: list[str] | None,
):
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

    # Collapse every widget's current value into the wire-shape filters
    # dict. None means "no filter active anywhere"; the helper handles
    # all the enable/mode gating internally.
    filters = _collect_filters(
        release_mode=release_mode,
        release_before_date=release_before_date,
        release_after_date=release_after_date,
        release_between_from=release_between_from,
        release_between_to=release_between_to,
        runtime_enable=runtime_enable,
        min_runtime=min_runtime,
        max_runtime=max_runtime,
        maturity_mode=maturity_mode,
        maturity_anchor=maturity_anchor,
        genres_enable=genres_enable,
        genres_selected=genres_selected,
        languages_enable=languages_enable,
        languages_selected=languages_selected,
        providers_enable=providers_enable,
        providers_selected=providers_selected,
    )

    try:
        async for event, payload in stream_query_search(
            cleaned, API_BASE_URL, filters=filters,
        ):
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

        # ---------------------------------------------------------------
        # Hard filters
        # ---------------------------------------------------------------
        # Six pre-filters mirroring the schema in
        # implementation/classes/schemas.py. Each filter has an enable
        # checkbox that gates its controls — when unchecked, the
        # controls render as disabled (interactive=False), giving each
        # filter a clearly "inactive" state. The values are not yet
        # forwarded to the API; wiring happens in a follow-up change.
        with gr.Accordion("Filters", open=False):

            # --- Release date ---------------------------------------
            # Three comparison modes (Before / After / Between) plus an
            # Inactive option folded into the same radio. The date
            # pickers return Unix timestamps (matching the release_ts
            # column / Qdrant payload key), and only the pickers used
            # by the current mode are shown — everything else is
            # hidden so the panel stays compact.
            with gr.Group():
                release_mode = gr.Radio(
                    choices=["Inactive", "Before", "After", "Between"],
                    value="Inactive",
                    label="Release date filter",
                )
                release_before_date = gr.DateTime(
                    label="Released before",
                    include_time=False,
                    type="timestamp",
                    visible=False,
                )
                release_after_date = gr.DateTime(
                    label="Released after",
                    include_time=False,
                    type="timestamp",
                    visible=False,
                )
                with gr.Row():
                    release_between_from = gr.DateTime(
                        label="From",
                        include_time=False,
                        type="timestamp",
                        visible=False,
                    )
                    release_between_to = gr.DateTime(
                        label="To",
                        include_time=False,
                        type="timestamp",
                        visible=False,
                    )

                # Visibility map: each picker is shown only when the
                # mode that consumes it is active.
                def _on_release_mode(mode_value: str):
                    return (
                        gr.update(visible=mode_value == "Before"),
                        gr.update(visible=mode_value == "After"),
                        gr.update(visible=mode_value == "Between"),
                        gr.update(visible=mode_value == "Between"),
                    )

                release_mode.change(
                    _on_release_mode,
                    inputs=[release_mode],
                    outputs=[
                        release_before_date,
                        release_after_date,
                        release_between_from,
                        release_between_to,
                    ],
                )

            # --- Runtime --------------------------------------------
            # Gradio doesn't ship a native dual-handle range slider, so
            # the closest UX is two stacked sliders that together act
            # as the "left knob" and "right knob". Parking the right
            # knob at _RUNTIME_MAX is the sentinel for "no upper bound"
            # (see the comment on _RUNTIME_MAX above). The enable
            # checkbox provides the inactive state since neither
            # slider has a natural "off" position on its own.
            with gr.Group():
                runtime_enable = gr.Checkbox(
                    label="Filter by runtime", value=False,
                )
                with gr.Row():
                    min_runtime = gr.Slider(
                        label="Min runtime (min)",
                        minimum=_RUNTIME_MIN, maximum=_RUNTIME_MAX,
                        value=_RUNTIME_MIN, step=5,
                        interactive=False,
                    )
                    max_runtime = gr.Slider(
                        label=f"Max runtime (min) — {_RUNTIME_MAX} = Any",
                        minimum=_RUNTIME_MIN, maximum=_RUNTIME_MAX,
                        value=_RUNTIME_MAX, step=5,
                        interactive=False,
                    )
                runtime_enable.change(
                    lambda on: [gr.update(interactive=on),
                                gr.update(interactive=on)],
                    inputs=[runtime_enable],
                    outputs=[min_runtime, max_runtime],
                )

            # --- Maturity rating ------------------------------------
            # Single comparison-type selector with the inactive state
            # folded in, paired with an anchor rating. "At most X" maps
            # to max_maturity_rank=X, "At least X" maps to
            # min_maturity_rank=X (both bounds are inclusive in the
            # backend schema).
            with gr.Group():
                with gr.Row():
                    maturity_mode = gr.Dropdown(
                        label="Maturity comparison",
                        choices=["Inactive", "At most", "At least"],
                        value="Inactive",
                    )
                    maturity_anchor = gr.Dropdown(
                        label="Anchor rating",
                        choices=_MATURITY_CHOICES,
                        value="PG-13",
                        interactive=False,
                    )
                # Anchor is only meaningful when a comparison is
                # actually selected; gray it out otherwise.
                maturity_mode.change(
                    lambda mode: gr.update(interactive=mode != "Inactive"),
                    inputs=[maturity_mode],
                    outputs=[maturity_anchor],
                )

            # --- Genres (match any) ----------------------------------
            with gr.Group():
                genres_enable = gr.Checkbox(
                    label="Filter by genres", value=False,
                )
                genres = gr.Dropdown(
                    label="Genres (match any)",
                    choices=_GENRE_CHOICES, multiselect=True,
                    value=[], interactive=False,
                )
                genres_enable.change(
                    lambda on: gr.update(interactive=on),
                    inputs=[genres_enable], outputs=[genres],
                )

            # --- Audio languages (match any) -------------------------
            with gr.Group():
                languages_enable = gr.Checkbox(
                    label="Filter by audio language", value=False,
                )
                languages = gr.Dropdown(
                    label="Audio languages (match any)",
                    choices=_LANGUAGE_CHOICES, multiselect=True,
                    value=[], interactive=False,
                )
                languages_enable.change(
                    lambda on: gr.update(interactive=on),
                    inputs=[languages_enable], outputs=[languages],
                )

            # --- Streaming providers (match any) ---------------------
            with gr.Group():
                providers_enable = gr.Checkbox(
                    label="Filter by streaming provider", value=False,
                )
                providers = gr.Dropdown(
                    label="Streaming providers (match any)",
                    choices=_PROVIDER_CHOICES, multiselect=True,
                    value=[], interactive=False,
                )
                providers_enable.change(
                    lambda on: gr.update(interactive=on),
                    inputs=[providers_enable], outputs=[providers],
                )

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
        # `inputs` order must match on_search()'s positional signature
        # exactly — Gradio binds component values positionally. Filter
        # components follow query / state in the order the helper
        # function _collect_filters() expects them.
        _filter_inputs = [
            release_mode,
            release_before_date,
            release_after_date,
            release_between_from,
            release_between_to,
            runtime_enable,
            min_runtime,
            max_runtime,
            maturity_mode,
            maturity_anchor,
            genres_enable,
            genres,
            languages_enable,
            languages,
            providers_enable,
            providers,
        ]
        submit_btn.click(
            on_search,
            inputs=[query_tb, state, *_filter_inputs],
            outputs=[state, status_md, submit_btn],
        )
        query_tb.submit(
            on_search,
            inputs=[query_tb, state, *_filter_inputs],
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
