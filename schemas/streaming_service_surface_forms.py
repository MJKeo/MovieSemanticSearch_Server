"""Prompt-rendering helper for the stage-3 metadata translator.

The metadata endpoint's `streaming.services` field is typed as
`list[StreamingService]` (20 enum values). The structured-output JSON
schema already exposes the full enum to the LLM, so the prompt's job
is not to enumerate valid values — it is to give the LLM the
display-name and alias mappings so colloquial phrasing
("HBO Max" → `max`, "Prime Video" → `amazon`, "Disney+" → `disney`)
resolves to the right enum slug.

This module renders a compact display-name + alias listing suitable
for direct inclusion in the metadata system prompt via the
{{TRACKED_STREAMING_SERVICES}} placeholder.

Structural mirror of schemas/production_brand_surface_forms.py and
schemas/award_surface_forms.py — same "render function returns a
string to embed in the prompt" pattern. Imports the source data
from implementation/classes/watch_providers.py, matching the
cross-package dependency that schemas/metadata_translation.py
already takes on the same module.

Consistency: iterates StreamingService directly, so adding a new
enum value automatically flows into the prompt. The renderer raises
at import time if a service is missing a display name or alias
entry — those maps are hand-maintained alongside the enum and
should never silently drift.
"""

from __future__ import annotations

from implementation.classes.watch_providers import (
    STREAMING_SERVICE_ALIASES,
    STREAMING_SERVICE_DISPLAY_NAMES,
    StreamingService,
)


def render_tracked_streaming_services_for_prompt() -> str:
    """Render every StreamingService as one prompt line.

    Format:
        <slug> (<display_name>; aliases: <a>, <b>, ...)
        ...

    `<slug>` is the enum value (what the LLM emits in the structured
    output). `<display_name>` is the user-facing brand name. Aliases
    are colloquial phrasings the LLM should map onto this slug.
    Aliases that duplicate the slug are kept in the list for
    completeness — the source-of-truth STREAMING_SERVICE_ALIASES
    map already deduplicates display variants per service.

    Raises RuntimeError at call time (which is module-import time
    for the prompt builder) if any StreamingService member lacks a
    display-name or alias entry — the underlying maps must stay in
    sync with the enum.
    """
    lines: list[str] = []
    for service in StreamingService:
        if service not in STREAMING_SERVICE_DISPLAY_NAMES:
            raise RuntimeError(
                f"StreamingService.{service.name} is missing a "
                f"STREAMING_SERVICE_DISPLAY_NAMES entry — update "
                f"watch_providers.py to keep the maps in sync with "
                f"the enum."
            )
        if service not in STREAMING_SERVICE_ALIASES:
            raise RuntimeError(
                f"StreamingService.{service.name} is missing a "
                f"STREAMING_SERVICE_ALIASES entry — update "
                f"watch_providers.py to keep the maps in sync with "
                f"the enum."
            )
        display = STREAMING_SERVICE_DISPLAY_NAMES[service]
        aliases = ", ".join(STREAMING_SERVICE_ALIASES[service])
        lines.append(f"{service.value} ({display}; aliases: {aliases})")
    return "\n".join(lines)
