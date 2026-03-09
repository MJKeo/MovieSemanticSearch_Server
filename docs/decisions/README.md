# Decision Records

This directory contains numbered records of significant decisions —
product, architectural, and technical.

Decisions are append-only: old decisions are superseded by new ones,
never edited. To supersede a decision, create a new record that
references the old one and update the old record's Status to
"Superseded by [NNN]".

## How Decisions Are Created

- **Automatically** by the docs-maintainer subagent when processing
  DIFF_CONTEXT.md at commit time (drafted for human review)
- **Manually** via `/save-decisions` after brainstorming sessions
- **Manually** via `/promote` when graduating a DIFF_CONTEXT entry

## Format

See the `/save-decisions` command for the full template. Each
record includes: Status, Context, Decision, Alternatives Considered,
Consequences, and References.
