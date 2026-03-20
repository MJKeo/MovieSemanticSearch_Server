---
name: "force-diff-context-update"
description: "Use when the user wants DIFF_CONTEXT.md reconciled with the current session before wrapping up or switching tasks."
---

# Force DIFF Context Update

Audit DIFF_CONTEXT.md and append anything still missing from the session.

## Workflow
1. Read the current `DIFF_CONTEXT.md` plus any new work from the session.
2. Preserve prior entries; append only.
3. Include decisions and tradeoffs, not just file edits.
4. Match entry size to scope exactly as described in the legacy prompt.

## References
- `references/original-command.md` -> required reconciliation behavior
- `DIFF_CONTEXT.md` -> active session context log
