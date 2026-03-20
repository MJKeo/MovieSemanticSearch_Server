---
name: "audit-personal-preferences"
description: "Use when the user wants to review, confirm, update, or prune the collaboration preferences stored in docs/personal_preferences.md."
---

# Audit Personal Preferences

Review and clean up docs/personal_preferences.md one entry at a time.

## Workflow
1. Read `docs/personal_preferences.md` and the exact legacy prompt in `references/original-command.md`.
2. Present each stored preference one at a time and explicitly route the user toward `Keep`, `Update`, or `Remove`.
3. If the user updates an entry, rewrite that entry in place rather than appending a duplicate.
4. End with a concise kept/updated/removed summary.

## References
- `references/original-command.md` -> exact Claude command behavior to preserve
- `docs/personal_preferences.md` -> source of truth for stored preferences
