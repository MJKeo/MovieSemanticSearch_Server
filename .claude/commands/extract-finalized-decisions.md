# .claude/commands/extract-finalized-decisions.md

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
module docs]
```

## After Writing Decisions

- If any decision changes how a module works, update the relevant
  doc in docs/modules/ to reflect the new expected behavior

## Documentation Impact Check

After writing all decisions, read docs/PROJECT.md and CLAUDE.md
and check whether any decisions from this session affect them.

**PROJECT.md** — Flag if any decision:
- Changes the target audience or problem statement
- Shifts priority ordering or adds/removes a priority
- Introduces or removes a constraint
- Changes the system overview or module map
- Supersedes information currently stated in PROJECT.md

**CLAUDE.md** — Flag if any decision:
- Adds new commands, tools, or workflows Claude should know about
- Changes build commands, test commands, or environment setup
- Alters the architecture overview or key directory descriptions
- Introduces new cross-codebase invariants
- Changes how Claude should interact with the codebase

For each potential update, present it as:
```
📌 [PROJECT.md or CLAUDE.md] — [section name]
Current: [what the doc currently says, briefly]
Suggested update: [what it should say based on this decision]
Reason: [which decision necessitates this change]
```

Wait for my confirmation on each before making any changes to
PROJECT.md or CLAUDE.md. I may approve all, some, or none.

## Summary

Report:
- Decisions saved: count, filenames, one-line summary of each
- Module docs updated: which ones and what changed
- PROJECT.md updates proposed: count (pending my approval)
- CLAUDE.md updates proposed: count (pending my approval)