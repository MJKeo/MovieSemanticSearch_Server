---
name: "create-unit-test-plan"
description: "Use when the user explicitly wants a unit test plan for recent code changes, without implementing the tests yet."
---

# Create Unit Test Plan

Write a source-driven testing plan into unit_tests/TEST_PLAN.md.

## Workflow
1. Follow `references/original-command.md` for the required output structure and reporting sentence.
2. Respect repo test boundaries unless the user explicitly requested this workflow.
3. Read `DIFF_CONTEXT.md` to recover intent before planning cases.
4. Design cases against intended behavior, not current implementation.

## References
- `references/original-command.md` -> exact planning checklist and output format
- `DIFF_CONTEXT.md` -> recent change intent
- `unit_tests/` -> existing test conventions and coverage patterns
