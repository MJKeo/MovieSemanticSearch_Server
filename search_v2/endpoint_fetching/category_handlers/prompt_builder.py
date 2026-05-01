# Assembles the 8-chunk handler system prompt + user message pair
# from markdown chunks in prompts/, keyed off category, bucket, and
# the category's endpoint set.
#
# See search_improvement_planning/category_handler_planning.md
# §"Prompt assembly" → "Finalized section table" for the ordered
# layout and keying rules, and §"Prompt builder behavior" for the
# import-time caching + TRENDING-raises conventions this module
# implements.

from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import escape as xml_escape

from schemas.award_category_tags import render_taxonomy_for_prompt
from schemas.award_surface_forms import (
    render_award_name_surface_forms_for_prompt,
    render_ceremony_mappings_for_prompt,
)
from schemas.enums import EndpointRoute, HandlerBucket
from schemas.production_brand_surface_forms import render_brand_registry_for_prompt
from schemas.streaming_service_surface_forms import (
    render_tracked_streaming_services_for_prompt,
)
from schemas.step_3 import CategoryCall
from schemas.trait_category import CategoryName
from schemas.unified_classification_families import (
    render_classification_registry_for_prompt,
)


# ── Paths ─────────────────────────────────────────────────────────

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_SHARED_DIR = _PROMPTS_DIR / "shared"
_BUCKETS_DIR = _PROMPTS_DIR / "buckets"
_ENDPOINTS_DIR = _PROMPTS_DIR / "endpoints"
_CATEGORIES_DIR = _PROMPTS_DIR / "categories"
_NOTES_DIR = _CATEGORIES_DIR / "additional_objective_notes"
_EXAMPLES_DIR = _CATEGORIES_DIR / "few_shot_examples"


def _read(path: Path) -> str:
    # Strip trailing whitespace so joined sections don't drift into
    # extra blank lines; leading whitespace is preserved because some
    # chunks start with markdown headers that the LLM keys off.
    return path.read_text(encoding="utf-8").rstrip()


# ── Eager-loaded static chunks ────────────────────────────────────
# These are invariants of the handler stack — every handler call uses
# them. Load once at import; any missing file fails loudly at startup,
# mirroring schema_factories.py's eager-build convention.

_ROLE: str = _read(_SHARED_DIR / "role.md")
_SHARED_VOCAB: str = _read(_SHARED_DIR / "shared_vocabulary.md")
_INPUT_SPEC: str = _read(_SHARED_DIR / "input_spec.md")

_BUCKET_OBJECTIVES: dict[HandlerBucket, str] = {
    bucket: _read(_BUCKETS_DIR / f"{bucket.value}_objective.md")
    for bucket in HandlerBucket
}
_BUCKET_GUARDRAILS: dict[HandlerBucket, str] = {
    bucket: _read(_BUCKETS_DIR / f"{bucket.value}_guardrails.md")
    for bucket in HandlerBucket
}

# Endpoint chunks are keyed by EndpointRoute. TRENDING has no LLM
# codepath at all. MEDIA_TYPE will be routed deterministically by code
# (matching surface phrases against the ReleaseFormat enum) rather than
# through the LLM handler — the deterministic routing path is pending,
# but in the meantime no prompt is authored and the route is skipped on
# eager load. Both routes are short-circuited to no-op results inside
# handler.run_handler, so reaching the LLM codepath for either should
# never happen.
_ENDPOINT_PROMPTLESS: frozenset[EndpointRoute] = frozenset({
    EndpointRoute.TRENDING,
    EndpointRoute.MEDIA_TYPE,
})


# Per-endpoint placeholder substitution. Each entry maps a route to
# the {{TOKEN}} → renderer callable mapping its .md file expects. The
# renderer is called once at import time and its output replaces the
# token in the cached chunk, so handler-time prompt builds are pure
# string concatenation. Renderers are co-located with their data
# under schemas/ and run their own consistency checks at call time
# (e.g., every registry member placed in a family, every enum value
# carrying a display-name + alias entry), so a stale registry fails
# loudly here rather than silently shipping broken prompts.
#
# Routes absent from this map have no placeholders — their .md is
# loaded verbatim. The leftover-placeholder scan below catches the
# case where someone adds a new {{TOKEN}} to a .md without
# registering a renderer here.
_ENDPOINT_PLACEHOLDER_RENDERERS: dict[
    EndpointRoute, dict[str, Callable[[], str]]
] = {
    EndpointRoute.AWARDS: {
        "{{CEREMONY_MAPPINGS}}":        render_ceremony_mappings_for_prompt,
        "{{AWARD_NAME_SURFACE_FORMS}}": render_award_name_surface_forms_for_prompt,
        "{{CATEGORY_TAG_TAXONOMY}}":    render_taxonomy_for_prompt,
    },
    EndpointRoute.STUDIO: {
        "{{BRAND_REGISTRY}}": render_brand_registry_for_prompt,
    },
    EndpointRoute.KEYWORD: {
        "{{CLASSIFICATION_REGISTRY}}": render_classification_registry_for_prompt,
    },
    EndpointRoute.METADATA: {
        "{{TRACKED_STREAMING_SERVICES}}": render_tracked_streaming_services_for_prompt,
    },
}


# Permissive — catches any double-brace token regardless of casing or
# punctuation. Used after substitution to surface unregistered
# placeholders before they reach the LLM as literal `{{...}}` text.
_PLACEHOLDER_PATTERN = re.compile(r"\{\{[^{}]+\}\}")


def _load_endpoint_chunk(route: EndpointRoute) -> str:
    """Read a route's .md file, substitute its registered placeholders,
    then assert no double-brace tokens remain.

    A registered placeholder that isn't found in the .md raises — the
    map and the file have drifted. A token that survives substitution
    raises — someone added a new placeholder without registering a
    renderer.
    """
    text = _read(_ENDPOINTS_DIR / f"{route.value}.md")
    for token, renderer in _ENDPOINT_PLACEHOLDER_RENDERERS.get(route, {}).items():
        if token not in text:
            raise RuntimeError(
                f"Endpoint chunk {route.value}.md is missing the registered "
                f"placeholder {token!r}; the .md and the renderer dispatch "
                f"in prompt_builder._ENDPOINT_PLACEHOLDER_RENDERERS have "
                f"drifted out of sync."
            )
        text = text.replace(token, renderer())
    leftover = _PLACEHOLDER_PATTERN.search(text)
    if leftover is not None:
        raise RuntimeError(
            f"Endpoint chunk {route.value}.md contains an unregistered "
            f"placeholder {leftover.group(0)!r} after substitution. Add a "
            f"renderer to prompt_builder._ENDPOINT_PLACEHOLDER_RENDERERS "
            f"or remove the placeholder from the .md."
        )
    return text


_ENDPOINT_CHUNKS: dict[EndpointRoute, str] = {
    route: _load_endpoint_chunk(route)
    for route in EndpointRoute
    if route not in _ENDPOINT_PROMPTLESS
}


# Per-category chunks. Loaded at import for every category that has
# files on disk — missing files are stored as absent keys and surface
# as a clear FileNotFoundError at build_system_prompt() call time.
# This keeps import from breaking while categories are being authored.
def _load_category_chunks(directory: Path) -> dict[CategoryName, str]:
    chunks: dict[CategoryName, str] = {}
    for category in CategoryName:
        if category is CategoryName.TRENDING:
            # No LLM handler — build_system_prompt raises for TRENDING
            # regardless of whether a file exists.
            continue
        path = directory / f"{category.name.lower()}.md"
        if path.exists():
            chunks[category] = _read(path)
    return chunks


_ADDITIONAL_NOTES: dict[CategoryName, str] = _load_category_chunks(_NOTES_DIR)
_FEW_SHOT_EXAMPLES: dict[CategoryName, str] = _load_category_chunks(_EXAMPLES_DIR)


# ── System prompt assembly ────────────────────────────────────────


@functools.cache
def build_system_prompt(category: CategoryName) -> str:
    """Assemble the 8-chunk system prompt for a given category.

    Order matches the finalized section table in the planning doc:
    Role → Shared vocab → Endpoint context → Input spec → Core
    objective → Additional objective notes → Guardrails → Few-shot
    examples.

    The result is memoized per ``CategoryName`` — the output is
    deterministic given the category, and callers typically invoke
    the builder once per CategoryCall inside the same query (many
    calls, same category). The memo also removes the per-caller
    caching burden flagged in the planning doc §"Prompt builder
    behavior".

    Raises:
        ValueError: if ``category`` is ``CategoryName.TRENDING`` —
            TRENDING has no LLM codepath and should be dispatched to
            a deterministic handler before reaching this function.
        ValueError: if the category's endpoint set resolves to zero
            LLM-wrapper endpoints (e.g. a future category that
            lists only TRENDING). Silently returning an empty
            endpoint-context slot would mislead the LLM about what
            it can call.
        FileNotFoundError: if the per-category ``notes`` or
            ``examples`` chunk is missing, naming the expected path.
    """

    # TRENDING has no LLM handler — the explicit raise catches the
    # mistake if dispatch forgets to route it to the deterministic
    # codepath.
    if category is CategoryName.TRENDING:
        raise ValueError(
            "CategoryName.TRENDING has no LLM handler and therefore no "
            "system prompt. Dispatch should route TRENDING to the "
            "deterministic codepath before reaching the prompt builder."
        )

    # Per-category chunks are required. Missing files surface here
    # with a clear error naming the expected path — this is how the
    # authoring gap for un-finished categories becomes visible.
    notes = _require_category_chunk(
        category, _ADDITIONAL_NOTES, _NOTES_DIR, "additional_objective_notes",
    )
    examples = _require_category_chunk(
        category, _FEW_SHOT_EXAMPLES, _EXAMPLES_DIR, "few_shot_examples",
    )

    # Endpoint context: one chunk per route in the category's priority
    # order. Routes in _ENDPOINT_PROMPTLESS (TRENDING, MEDIA_TYPE) are
    # silently skipped because no LLM chunk exists for them; if that
    # leaves zero endpoints the category has nothing to dispatch to and
    # the builder must surface the misconfiguration rather than emit an
    # empty section.
    endpoint_chunks = [
        _ENDPOINT_CHUNKS[route]
        for route in category.endpoints
        if route in _ENDPOINT_CHUNKS
    ]
    if not endpoint_chunks:
        raise ValueError(
            f"CategoryName.{category.name} has no LLM-wrapper endpoints "
            f"(declared: {[r.value for r in category.endpoints]}). A "
            f"handler with no endpoint context cannot produce a useful "
            f"prompt — route this category to a deterministic codepath "
            f"or add a real endpoint."
        )
    endpoint_context = "\n\n".join(endpoint_chunks)

    core_objective = _BUCKET_OBJECTIVES[category.bucket]
    guardrails = _BUCKET_GUARDRAILS[category.bucket]

    # Two blank lines between sections so markdown headers at the top
    # of each chunk remain visually distinct in the assembled prompt.
    return "\n\n".join(
        [
            _ROLE,
            _SHARED_VOCAB,
            endpoint_context,
            _INPUT_SPEC,
            core_objective,
            notes,
            guardrails,
            examples,
        ]
    )


def _require_category_chunk(
    category: CategoryName,
    cache: dict[CategoryName, str],
    directory: Path,
    subdir_label: str,
) -> str:
    try:
        return cache[category]
    except KeyError:
        expected = directory / f"{category.name.lower()}.md"
        raise FileNotFoundError(
            f"Missing {subdir_label} chunk for CategoryName.{category.name}. "
            f"Expected at: {expected}"
        ) from None


# ── User message (per-call XML payload) ───────────────────────────


def build_user_message(category_call: CategoryCall) -> str:
    """Serialize the per-call input payload as the XML block the
    handler LLM consumes.

    Two sections only — ``<retrieval_intent>`` and ``<expressions>``
    — both pulled from the CategoryCall committed by Step 3. The LLM
    is intentionally not given raw_query, query-level intent
    framing, sibling traits, or the upstream modifier signals: those
    are either already folded into ``retrieval_intent`` by Step 3 or
    deliberately out of scope for this stage (Step 4 translates the
    committed call; it does not re-interpret the query). Match_mode
    and polarity are stamped onto the wrapper post-hoc by
    ``handler.run_handler`` from the parent Trait, so they are not
    emitted by the LLM either.
    """

    intent = xml_escape(category_call.retrieval_intent)
    expression_lines = "\n".join(
        f"  <expression>{xml_escape(expr)}</expression>"
        for expr in category_call.expressions
    )
    return (
        f"<retrieval_intent>{intent}</retrieval_intent>\n"
        f"<expressions>\n{expression_lines}\n</expressions>"
    )
