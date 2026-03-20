---
name: "run-docs-auditor-agent"
description: "Use when the user wants a read-only audit of module docs, conventions, decisions, and priority drift against the current codebase."
---

# Run Docs Auditor

Perform a full permanent-doc staleness audit modeled on the legacy docs-auditor subagent.

## Workflow
1. Read both `references/original-command.md` and `references/legacy-agent.md` before auditing.
2. Treat this as a read-only analysis pass.
3. Be specific with file paths and why each doc is stale, incomplete, or current.
4. Return the report for user triage rather than making doc edits automatically.

## References
- `references/original-command.md` -> command wrapper behavior
- `references/legacy-agent.md` -> full docs-auditor analysis contract
- `docs/` and relevant source modules -> audit inputs
