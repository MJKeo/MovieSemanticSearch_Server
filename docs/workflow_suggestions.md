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

## New metadata evaluation command
**Pattern observed:** Starting a new metadata type evaluation
always follows the same sequence: (1) run /initiate-spec-understanding-conversation
against the evaluation guide, (2) enter plan mode and design the
rubric and table schema, (3) implement shared.py stubs and the
per-type evaluation file. This three-step flow was followed
explicitly for plot_events and will repeat for plot_analysis,
viewer_experience, etc.
**Suggested implementation:** command (`/new-metadata-evaluation <type>`)
**Rationale:** The command would scaffold the per-type evaluation
file from a template, add the three table CREATE statements, wire
in EvaluationCandidate, and open the spec conversation. Saves
the repetitive boilerplate that each new type requires.
