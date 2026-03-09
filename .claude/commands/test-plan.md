# .claude/commands/test-plan.md

Analyze the code you implemented in this session. For each new or
modified method, create a comprehensive test plan.

For each method that needs testing, identify:
- What behavior it implements and what its contract is
- What existing tests in ./unit_tests/ already cover it
- What specific test cases are needed per the coverage categories
  below

Present the plan organized by file.
Do NOT write any test code. Output ONLY the plan.

## Coverage Categories

- Happy path with representative inputs
- Boundary values (zero, empty, min/max, off-by-one)
- Type and format violations (null, undefined, wrong type, malformed)
- Error paths (exceptions, timeouts, permission failures)
- State-dependent behavior (empty vs populated, first-call vs repeat)