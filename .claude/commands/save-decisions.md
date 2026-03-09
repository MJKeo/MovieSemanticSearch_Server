Review the current conversation and extract all finalized decisions.

A "finalized decision" is a choice we explicitly agreed on — not
exploration, not options we rejected, not open questions. Look for
moments where we committed to an approach.

## For Each Decision

Determine the next available number in docs/decisions/ and create
a new file with this format:

```markdown
# [NNN] — [Short title describing the decision]

## Status
Active

## Context
[What problem or question prompted this decision. 2-3 sentences.]

## Decision
[What we decided. Be specific and concrete.]

## Alternatives Considered
[What other options we discussed and why they were rejected.]

## Consequences
[What this decision enables, constrains, or changes.
Reference docs/PROJECT.md priorities if the decision involves
a tradeoff between them.]

## References
[Links to relevant docs: PROJECT.md sections, other decisions,
module docs, guides/]
```

## After Writing Decisions

- If any decision changes how a module works, update the relevant
  doc in docs/modules/ to reflect the new expected behavior
- If any decision affects product priorities or constraints, tell
  me — I will update PROJECT.md myself via /update-project
- Report what you wrote: number of decisions saved, filenames,
  and a one-line summary of each
