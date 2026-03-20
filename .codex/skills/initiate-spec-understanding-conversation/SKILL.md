---
name: "initiate-spec-understanding-conversation"
description: "Use when the user presents a spec and wants a thorough understanding pass before planning or implementation begins."
---

# Initiate Spec Understanding

Run a deep pre-implementation spec review conversation against the codebase.

## Workflow
1. Start by restating the spec intent in one paragraph.
2. Read the code paths and docs the spec would affect.
3. Organize findings into `Must resolve before implementation`, `Should discuss`, and `Worth noting`.
4. Keep the output discussion-oriented; do not jump to implementation.

## References
- `references/original-command.md` -> exact discussion workflow and output shape
- `docs/PROJECT.md`, `DIFF_CONTEXT.md`, and relevant `docs/modules/*.md` -> required startup context
