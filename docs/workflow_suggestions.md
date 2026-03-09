# Workflow Suggestions

Potential automations and workflow improvements observed across
sessions. Run /review-workflow to go through them interactively.

Entries are added automatically during /safe-clear when repeated
manual patterns are detected.

## SQLite health check command
**Pattern observed:** User encountered DB corruption and we
manually ran `PRAGMA integrity_check`, checked for WAL/SHM files,
inspected extended attributes, and attempted `.recover` / `.dump`.
This was a multi-step diagnostic sequence.
**Suggested implementation:** command (`/check-tracker-db`)
**Rationale:** A single command that runs integrity check, reports
WAL/SHM presence, shows row counts by status, and flags any
extended attributes would save time on future corruption events
or routine health checks.
