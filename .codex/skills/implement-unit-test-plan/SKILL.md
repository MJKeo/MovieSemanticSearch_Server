---
name: "implement-unit-test-plan"
description: "Use when the user explicitly wants the approved test plan implemented under unit_tests/."
---

# Implement Unit Test Plan

Turn unit_tests/TEST_PLAN.md into actual tests without touching source code.

## Workflow
1. Treat `unit_tests/TEST_PLAN.md` as the contract; do not add or skip cases.
2. Keep all edits inside `unit_tests/`.
3. If tests fail, report them and do not fix source in the same pass unless the user separately asks.
4. Delete `TEST_PLAN.md` only if the legacy command conditions are met.

## References
- `references/original-command.md` -> exact implementation rules
- `unit_tests/TEST_PLAN.md` -> approved plan to implement
