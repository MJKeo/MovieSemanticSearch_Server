Before I clear context, perform two tasks: save context and
extract learnings from this session.

## Task 1: Save Context

1. Review the work and discussions in this session
2. If DIFF_CONTEXT.md is missing entries for work done or
   decisions made in this session, append them now
3. If DIFF_CONTEXT.md is already current, confirm that

## Task 2: Extract Session Learnings

Scale this analysis to the length and complexity of the session.
A quick 5-minute fix needs a brief scan. A long session with
multiple corrections and discussions warrants deeper reflection.

Read these files for context before analyzing:
- docs/personal_preferences.md
- docs/conventions_draft.md
- docs/workflow_suggestions.md
- docs/TODO.md

Then review the session for three types of learnings:

### Personal Preferences
Look for signals about how the user prefers to work:
- Communication preferences (verbosity, format, tone)
- Workflow habits (when they want to be asked vs told)
- Response format preferences (diffs, explanations, code-first)
- Any explicit or implicit feedback about how you responded

If you identify new preferences not already in
docs/personal_preferences.md, add them directly to the file.
If an existing preference should be updated based on this
session, update it in place.

### Convention Candidates
Look for patterns that could be codebase conventions:
- Code style corrections the user made
- Patterns the user asked you to follow or avoid
- Approaches the user consistently preferred
- Architectural patterns that emerged as decisions

For each candidate not already in docs/conventions_draft.md or
docs/conventions.md, append it to docs/conventions_draft.md in
this format:
```
## [Brief description]
**Observed:** [What happened in the session that suggests this]
**Proposed convention:** [The specific rule to follow]
**Sessions observed:** 1
```

If a pattern already exists in conventions_draft.md, increment
its session count. Patterns with higher counts are stronger
candidates.

### Workflow Suggestions
Look for friction or repetition that could be automated:
- Tasks I performed manually that could be a command or skill
- Questions I asked repeatedly that could be a rule
- Processes I followed step-by-step that could be a single command
- Anything I asked you to do 2+ times in the same way

For each suggestion not already in docs/workflow_suggestions.md,
append it in this format:
```
## [Brief description of the automation]
**Pattern observed:** [What the user did manually]
**Suggested implementation:** [rule / command / skill / subagent / hook]
**Rationale:** [Why this would save time]
```

### TODO Items
Look for actionable items that came up but were deferred:
- Things the user said to do later or "when we get to X"
- Known issues discovered but not fixed in this session
- Improvements identified but out of scope for current work

For each item not already in docs/TODO.md, append it in this
format:
```
## [Brief description]
**Context:** [What was discovered and why it matters]
**When:** [When this should be addressed]
**See:** [Reference to relevant files, memory, or docs if any]
```

## After Both Tasks

Report a brief summary:
- Context: saved / already current
- Preferences: N added or updated (list them)
- Convention candidates: N added or updated (list them)
- Workflow suggestions: N added (list them)
- TODO items: N added (list them)
- If nothing was learned, say "No new learnings from this session"

Then tell me: "Context saved and learnings extracted.
Run /clear when ready."

Do NOT attempt to run /clear yourself.
