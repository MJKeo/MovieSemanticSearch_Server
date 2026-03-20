Run the test-planner subagent to analyze code changes and produce
a comprehensive test plan.

The planner will:
1. Diff against main to identify all changed/new source files
2. Read each changed file to understand intended behavior
3. Review existing tests for current coverage and conventions
4. Identify gaps, stale tests, and missing edge cases

Output is written to ./unit_tests/TEST_PLAN.md.

Review the plan and tell me what needs attention.
I will decide what to implement.
