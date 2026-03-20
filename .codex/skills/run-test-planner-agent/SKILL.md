---
name: "run-test-planner-agent"
description: "Use when the user wants a test planning pass driven by changed source files and current test coverage, without implementing tests yet."
---

# Run Test Planner

Produce an independent TEST_PLAN audit using the legacy test-planner workflow.

## Workflow
1. Read both `references/original-command.md` and `references/legacy-agent.md`.
2. Diff the codebase against `main` unless the user narrows the scope.
3. Keep the work analysis-only; do not write test code.
4. Follow the exact TEST_PLAN structure and closing summary sentence from the legacy workflow.

## References
- `references/original-command.md` -> command wrapper behavior
- `references/legacy-agent.md` -> detailed planning workflow
- `unit_tests/` -> existing coverage and conventions
