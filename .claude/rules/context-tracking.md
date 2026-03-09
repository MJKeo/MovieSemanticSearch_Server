# Context Tracking

After completing each implementation task (writing or modifying
code — not answering questions, not planning, not reading), append
a structured entry to DIFF_CONTEXT.md before reporting results.

If DIFF_CONTEXT.md does not exist, create it with the header:
```
# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.
```

## Entry Format

Organize entries by intent, not by file. Scale the entry to match
the scope of the change:

**Small change** (single method fix, minor refactor) — 1-2 lines:
```
## [brief description of intent]
Files: [paths] | [one sentence: what and why]
```

**Medium change** (new endpoint, feature addition) — 5-10 lines:
```
## [intent]
Files: [paths]
Why: [motivation]
Approach: [what you did and why you chose this approach over alternatives]
Design context: [reference to docs/decisions/ or docs/PROJECT.md if relevant]
Testing notes: [what needs coverage]
```

**Large change** (new feature, multi-module work) — up to 25 lines:
```
## [intent]
Files: [paths]

### Intent
[What this achieves and why it matters]

### Key Decisions
[Each decision with justification, alternatives considered,
and references to permanent docs where relevant]

### Planning Context
[Any brainstorming or planning decisions that shaped this work]

### Testing Notes
[What needs coverage, what's risky, edge cases to watch]
```

## Rules

- Match entry length to the amount of reasoning a fresh reader
  would need to understand what happened and why
- Include decisions and justifications even for planning-level
  choices that shaped the implementation approach
- Reference permanent docs (docs/decisions/, docs/PROJECT.md)
  rather than restating their content
- Do not rewrite or reorganize previous entries
- Do not include code snippets — reference file paths instead
