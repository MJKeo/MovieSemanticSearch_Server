Read docs/workflow_suggestions.md.

If the file is empty (no suggestions beyond the header), say
"No workflow suggestions to review" and stop.

Present each suggestion one at a time. For each suggestion:

1. Explain what was observed and what automation is proposed
2. Ask me to choose:
   - **Implement** — build the rule/command/skill/subagent/hook
   - **Discuss** — talk through the best approach before building
   - **Skip** — not worth automating, remove from the list
   - **Keep** — interesting but not ready, leave it for later

If I choose Implement:
- Create the appropriate file (.claude/rules/, .claude/commands/,
  .claude/agents/, or .claude/skills/)
- If the new automation affects how other parts of the system
  work, update CLAUDE.md or relevant docs to reference it
- Remove the suggestion from workflow_suggestions.md

If I choose Discuss:
- Have a conversation about the best approach
- Once we agree, implement it and remove from the list

If I choose Skip:
- Remove the suggestion from workflow_suggestions.md

If I choose Keep:
- Leave the suggestion in workflow_suggestions.md and move to
  the next one

After processing all suggestions, report what was implemented,
skipped, and kept.
