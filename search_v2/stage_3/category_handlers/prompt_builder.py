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
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape as xml_escape

from schemas.enums import CategoryName, EndpointRoute, HandlerBucket
from schemas.step_2 import CoverageEvidence, Modifier, RequirementFragment


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
# codepath and therefore no prompt file, so it is excluded here.
_ENDPOINT_CHUNKS: dict[EndpointRoute, str] = {
    route: _read(_ENDPOINTS_DIR / f"{route.value}.md")
    for route in EndpointRoute
    if route is not EndpointRoute.TRENDING
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
    the builder once per coverage_evidence atom inside the same
    query (many atoms, same category). The memo also removes the
    per-caller caching burden flagged in the planning doc
    §"Prompt builder behavior".

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
    # order. TRENDING routes are silently skipped (no LLM chunk exists
    # for them); if that leaves zero endpoints the category has
    # nothing to dispatch to and the builder must surface the
    # misconfiguration rather than emit an empty section.
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


def build_user_message(
    raw_query: str,
    overall_query_intention_exploration: str,
    target_entry: CoverageEvidence,
    parent_fragment: RequirementFragment,
    sibling_fragments: list[RequirementFragment],
) -> str:
    """Serialize the per-call input payload as the XML block the
    handler LLM consumes.

    Structure matches the XML schema in the planning doc and
    `prompts/shared/input_spec.md`. All user-derived leaf text is
    XML-escaped via ``xml.sax.saxutils.escape``; modifiers use the
    fully-tagged nested-element form (not attributes).

    ``parent_fragment`` and ``sibling_fragments`` may be passed
    verbatim from Step 2 output — the builder serializes only
    ``query_text``, ``description``, and ``modifiers`` and ignores
    ``coverage_evidence`` regardless of whether it's populated. The
    target atom is surfaced separately via ``target_entry``; any
    other atoms on those fragments belong to other handler calls
    and simply don't appear in the payload.
    """

    parts = [
        _wrap_leaf("raw_query", raw_query),
        _wrap_leaf(
            "overall_query_intention_exploration",
            overall_query_intention_exploration,
        ),
        _serialize_target_entry(target_entry),
        _serialize_fragment("parent_fragment", parent_fragment, indent=""),
        _serialize_siblings(sibling_fragments),
    ]
    return "\n".join(parts)


def _wrap_leaf(tag: str, text: str) -> str:
    # Single-line element for scalar text. Escape even though the
    # typical query is plain ASCII — '&', '<', '>' in a user query
    # would otherwise break XML parsing on the LLM side.
    return f"<{tag}>{xml_escape(text)}</{tag}>"


def _serialize_target_entry(entry: CoverageEvidence) -> str:
    # Four sub-fields exactly as described in input_spec.md. Enum
    # leaves carry their `.value` (e.g. 'clean' / 'partial') so the
    # text matches the shared-vocabulary definitions literally.
    lines = [
        "<target_entry>",
        f"  <captured_meaning>{xml_escape(entry.captured_meaning)}</captured_meaning>",
        f"  <category_name>{xml_escape(entry.category_name.value)}</category_name>",
        f"  <fit_quality>{xml_escape(entry.fit_quality.value)}</fit_quality>",
        f"  <atomic_rewrite>{xml_escape(entry.atomic_rewrite)}</atomic_rewrite>",
        "</target_entry>",
    ]
    return "\n".join(lines)


def _serialize_fragment(
    tag: str, fragment: RequirementFragment, *, indent: str
) -> str:
    # Fragments carry query_text / description / modifiers. The
    # coverage_evidence list on RequirementFragment is deliberately
    # not serialized — the per-call target atom is already surfaced
    # in <target_entry>, and any other atoms belong to other handler
    # calls.
    #
    # ``indent`` is the outer indent applied to the wrapper tag
    # itself; sub-elements are indented by two additional spaces.
    # The parent_fragment case uses indent="" (top-level element);
    # sibling fragments use indent="  " so they nest visually under
    # the <sibling_fragments> wrapper.
    inner = indent + "  "
    modifiers_block = _serialize_modifiers(fragment.modifiers, indent=inner)
    lines = [
        f"{indent}<{tag}>",
        f"{inner}<query_text>{xml_escape(fragment.query_text)}</query_text>",
        f"{inner}<description>{xml_escape(fragment.description)}</description>",
        modifiers_block,
        f"{indent}</{tag}>",
    ]
    return "\n".join(lines)


def _serialize_modifiers(modifiers: Iterable[Modifier], indent: str) -> str:
    # Fully-tagged nested form (Option A in the planning doc): every
    # field on a modifier is its own element. Chosen over attributes
    # for two reasons: element text parses more reliably on small
    # models, and escaping is uniform across all leaves.
    mods = list(modifiers)
    if not mods:
        # Keep an explicit empty tag so the LLM sees the slot exists.
        return f"{indent}<modifiers></modifiers>"

    lines = [f"{indent}<modifiers>"]
    for modifier in mods:
        lines.append(f"{indent}  <modifier>")
        lines.append(f"{indent}    <type>{xml_escape(modifier.type.value)}</type>")
        lines.append(
            f"{indent}    <original_text>"
            f"{xml_escape(modifier.original_text)}"
            f"</original_text>"
        )
        lines.append(
            f"{indent}    <effect>{xml_escape(modifier.effect)}</effect>"
        )
        lines.append(f"{indent}  </modifier>")
    lines.append(f"{indent}</modifiers>")
    return "\n".join(lines)


def _serialize_siblings(siblings: list[RequirementFragment]) -> str:
    # An empty list is legitimate (single-fragment queries). Emit an
    # explicit empty tag so the LLM sees the slot rather than having
    # to infer absence.
    if not siblings:
        return "<sibling_fragments></sibling_fragments>"

    # Inner fragments are indented two spaces so their visual nesting
    # matches the rest of the XML (target_entry, parent_fragment,
    # modifiers all indent sub-elements by two).
    fragment_blocks = "\n".join(
        _serialize_fragment("fragment", sibling, indent="  ")
        for sibling in siblings
    )
    return f"<sibling_fragments>\n{fragment_blocks}\n</sibling_fragments>"
