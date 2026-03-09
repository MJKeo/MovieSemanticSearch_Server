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
- **Manually** via `/extract-finalized-decisions` after brainstorming sessions

## Format

See the `/extract-finalized-decisions` command for the full template.
Each record includes: Status, Context, Decision, Alternatives
Considered, Consequences, and References.

## Note on `guides/` References

Many ADRs reference files in a `guides/` directory that no longer
exists. The deep technical content from those guides has been folded
into the module docs in `docs/modules/`. Treat `guides/` references
in older ADRs as historical context, not live links.
