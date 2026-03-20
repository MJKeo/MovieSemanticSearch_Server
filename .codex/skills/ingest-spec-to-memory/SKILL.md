---
name: "ingest-spec-to-memory"
description: "Use when the user provides a spec and wants permanent project docs updated carefully, with conflicts surfaced before any writes."
---

# Ingest Spec To Docs

Reconcile a product spec against the project's documentation system before writing updates.

## Workflow
1. Read the full documentation baseline named in `references/original-command.md` before analysis.
2. Separate new information from conflicts.
3. Stop after the conflict report until the user resolves every blocking issue.
4. After resolution, write only the allowed docs and summarize what changed.

## References
- `references/original-command.md` -> phased workflow and approval gates
- `docs/PROJECT.md`, `docs/conventions.md`, `docs/decisions/`, `docs/modules/` -> comparison baseline
