# Coding Standards

## Priority Order (when in tension)
1. Correctness and security
2. Readability and maintainability
3. Performance and efficiency

## Architecture
- Prefer small, single-responsibility functions (under ~40 lines)
- Extract shared logic into reusable utilities rather than duplicating
- Use dependency injection over hard-coded dependencies
- Design public interfaces to be stable; keep implementation details private
- When adding new code, follow the conventions already established
  in the surrounding codebase

## Error Handling
- Validate inputs at system boundaries (API endpoints, CLI args, external data)
- Use typed errors/custom exceptions over generic throws
- Handle failures gracefully with meaningful error messages that aid debugging
- Never silently swallow exceptions; log or propagate them

## Security
- Sanitize all user-provided input before use
- Never log secrets, tokens, or PII
- Use parameterized queries; never interpolate values into SQL or commands
- If you encounter a security vulnerability in existing code, flag it and fix it

## Performance
- Prefer lazy evaluation and streaming for large data sets
- Avoid unnecessary allocations in hot paths
- Use appropriate data structures (e.g., Set for lookups, Map for key-value)
- Add complexity only when profiling justifies it; prefer readable code
  over premature optimization

## Code Quality
- Name variables and functions to describe *what* and *why*, not *how*
- Liberally include comments explaining what you are doing and why
- No magic numbers or strings; use named constants
- Keep nesting depth to 2-3 levels max; use early returns to flatten logic
- Helpers and abstractions are fine when the pattern is likely to
  recur — even if currently used in one spot. When adding a new
  class or base class, first check if an existing class can absorb
  the responsibility. Only introduce a new class when it has a
  clear, distinct responsibility that doesn't overlap.
