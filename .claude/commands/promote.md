I want to promote content from DIFF_CONTEXT.md into permanent
storage. If I specify which entry, promote that one. If not,
show me the current entries and let me choose.

For each entry being promoted, determine the right target:

**→ docs/decisions/** if the entry describes a significant choice
with alternatives, tradeoffs, or justification. Create a numbered
decision record following the format in /save-decisions.

**→ docs/modules/** if the entry describes how a module works,
its boundaries, patterns, or gotchas. Update the existing module
doc or create a new one.

After promoting:
- Mark the promoted entry in DIFF_CONTEXT.md by prepending
  `[PROMOTED → decisions/NNN]` or `[PROMOTED → modules/X.md]`
  to its heading
- Report what was promoted and where

Focus on: $ARGUMENTS
