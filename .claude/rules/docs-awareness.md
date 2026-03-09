# Documentation Awareness

## Reading Docs

- Read docs/PROJECT.md at the start of every session for product
  context, priorities, and constraints
- The first time you work in a module's directory during a session,
  read its doc in docs/modules/ if one exists
- When the module doc references a guide in guides/, read that
  guide before making non-trivial changes
- When making a tradeoff decision (performance vs readability,
  complexity vs simplicity, etc.), check docs/PROJECT.md priorities
  and scan docs/decisions/ for relevant precedent before choosing
- Read docs/personal_preferences.md at the start of every session
  to understand how the user prefers to work and communicate

## Updating Docs

You MAY autonomously update:
- docs/modules/ — if you discover a module doc is stale while
  working in that module, fix it as part of your current changeset.
  Keep updates proportional: don't rewrite the doc, just correct
  the inaccurate parts.

You must NEVER autonomously modify:
- docs/PROJECT.md — human-only via /update-project
- docs/conventions.md — flag inconsistencies to me, don't fix
- docs/decisions/ — only written via /save-decisions, /promote,
  or the docs-maintainer subagent
