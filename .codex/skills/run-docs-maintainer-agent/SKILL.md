---
name: "run-docs-maintainer-agent"
description: "Use when the user wants DIFF_CONTEXT.md processed into module-doc updates and draft decision records, typically near commit time."
---

# Run Docs Maintainer

Process DIFF_CONTEXT.md into durable documentation updates using the legacy maintainer workflow.

## Workflow
1. Read both `references/original-command.md` and `references/legacy-agent.md`.
2. Keep the hard guardrails from the legacy agent: do not touch `docs/PROJECT.md` or `docs/conventions.md`, and do not modify source code.
3. Use DIFF entries to drive proportional doc updates and draft decisions.
4. Finish by resetting `DIFF_CONTEXT.md` exactly as specified if the workflow runs to completion.

## References
- `references/original-command.md` -> command wrapper behavior
- `references/legacy-agent.md` -> full docs-maintainer workflow
- `DIFF_CONTEXT.md` -> transient source of truth for pending doc work
