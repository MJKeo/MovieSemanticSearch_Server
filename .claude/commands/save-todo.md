Add a TODO entry to docs/TODO.md.

Argument: $ARGUMENTS

## Instructions

1. Read docs/TODO.md to see existing entries and avoid duplicates
2. If the argument matches or substantially overlaps with an
   existing entry, report that and do not add a duplicate
3. Otherwise, append a new entry using the format below

## Entry Format

```
## [Brief description derived from the argument]
**Context:** [Expand the argument into a clear explanation of
what needs to be done and why, using context from the current
session if available]
**When:** [Infer when this should be addressed — e.g., "When
building feature X", "Next refactoring pass", "Before production
deploy". If unclear, use "When relevant work begins."]
**See:** [Reference relevant files, memory entries, or docs.
If no specific reference exists, omit this line.]
```

## Rules

- Keep entries concise but informative — a fresh reader should
  understand what to do without needing session context
- Use the argument as the starting point but enrich it with
  context from the current conversation when available
- Do not add entries for things that are already being done in
  the current session
- After adding, confirm what was written
