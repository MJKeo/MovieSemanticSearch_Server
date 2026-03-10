# .claude/commands/implement-unit-test-plan.md

Implement the test plan ./unit_tests/TEST_PLAN.md.
Write all tests in ./unit_tests/.

## Rules

- Follow the approved plan — do not add or skip test cases
- Design tests against the method's intended contract, not the
  current implementation
- If a test fails, the source is wrong, not the test
- Do NOT modify any source code outside of ./unit_tests/
- Each test verifies ONE behavior with a descriptive name that
  reads as a specification
- Use Arrange-Act-Assert structure
- No test should depend on execution order or shared mutable state
- Prefer explicit values over random/generated data
- Match patterns and conventions already used in ./unit_tests/

After implementation, run the test suite and report results.
Do NOT fix any failures. If no failures are detected then delete TEST_PLAN.md.