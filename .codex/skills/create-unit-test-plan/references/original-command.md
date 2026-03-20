# .claude/commands/create-unit-test-plan.md

Analyze the code you implemented in this session. For each new or
modified method, create a comprehensive test plan.

Read DIFF_CONTEXT.md for context on the intent behind recent changes.

## Process

1. Identify all methods you created or modified in this session
2. Read existing test files in unit_tests/ to understand current
   coverage, conventions, and the base_movie_factory fixture in
   conftest.py
3. Identify gaps: what is untested, what tests are now stale, what
   edge cases are missing

## Output

Write your plan to unit_tests/TEST_PLAN.md using this structure:

For each file containing changes, create a section. For each method
in that file that needs testing, document:
- Intent: what the method is supposed to do
- Existing coverage: what tests already exist (or "none")
- New test cases needed: each as a descriptive test name with a
  brief explanation of what it verifies and why

At the top, include a summary with total files changed, methods
needing testing, and existing test files needing updates.

At the end of each method section, include a coverage checklist:
- Happy path
- Boundary values
- Type/format violations
- Error paths
- State-dependent behavior

## Rules

- Do NOT write test code, only the plan
- Do NOT modify any source files
- Design test cases against intended behavior, not current
  implementation — treat the source as potentially flawed
- Be specific in test case descriptions — "handles edge cases"
  is not a test case, "returns 0 when input array is empty" is
- Flag any methods where the intended behavior is ambiguous
- Note which tests will be async (asyncio_mode is "auto")

After writing the plan, report:
"Test plan written to unit_tests/TEST_PLAN.md — X files,
Y test cases planned. Review the plan, then run /test-implement."

Focus on: $ARGUMENTS