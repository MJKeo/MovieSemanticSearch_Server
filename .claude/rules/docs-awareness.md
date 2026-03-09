# Documentation Awareness

## Reading Docs

- Read docs/PROJECT.md at the start of every session for product
  context, priorities, and constraints
- The first time you work in a module's directory during a session,
  read its doc in docs/modules/ if one exists
- When making a tradeoff decision (performance vs readability,
  complexity vs simplicity, etc.), check docs/PROJECT.md priorities
  and scan docs/decisions/ for relevant precedent before choosing
- Read docs/personal_preferences.md at the start of every session
  to understand how the user prefers to work and communicate
- Scan docs/TODO.md when starting work in an area — existing TODOs
  may be relevant to the current task

## Updating Docs

You MAY autonomously update:
- docs/modules/ — if you discover a module doc is stale while
  working in that module, fix it as part of your current changeset.
  Keep updates proportional: don't rewrite the doc, just correct
  the inaccurate parts.
- docs/TODO.md — add entries when you discover actionable items
  during implementation work. Follow the existing entry format.

You must NEVER autonomously modify:
- docs/PROJECT.md — human-only via /extract-finalized-decisions
- docs/conventions.md — only written via /solidify-draft-conventions
- docs/decisions/ — only written via /extract-finalized-decisions
  or the docs-maintainer subagent
