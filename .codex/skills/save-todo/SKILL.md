---
name: "save-todo"
description: "Use when the user wants an actionable follow-up captured in docs/TODO.md without duplicating an existing entry."
---

# Save TODO

Append a deduplicated TODO entry to docs/TODO.md from a user-supplied idea.

## Workflow
1. Start from the user argument, but enrich it with session context when useful.
2. Deduplicate against existing TODO entries before writing.
3. Use the exact entry shape from `references/original-command.md`.
4. Do not add items that are already being completed in the current session.

## References
- `references/original-command.md` -> entry format and deduping rules
- `docs/TODO.md` -> persistent TODO queue
