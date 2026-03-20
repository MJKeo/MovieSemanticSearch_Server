---
name: "extract-finalized-decisions"
description: "Use when a design conversation has concluded and the user wants finalized decisions captured in docs/decisions/ with follow-on doc impact called out."
---

# Extract Finalized Decisions

Turn settled discussion outcomes into numbered decision records.

## Workflow
1. Treat only committed choices as decisions; ignore exploration and rejected options.
2. Use the exact record format in `references/original-command.md`.
3. Update `docs/modules/` when a decision changes expected module behavior.
4. Never modify `docs/PROJECT.md` or `CLAUDE.md` without explicit approval; instead surface targeted proposals.

## References
- `references/original-command.md` -> exact extraction and approval workflow
- `docs/decisions/` -> numbering source and existing precedent
- `docs/modules/` -> module docs to refresh when decisions change behavior
