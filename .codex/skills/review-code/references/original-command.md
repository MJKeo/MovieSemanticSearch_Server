# .claude/commands/review-code.md

Review the code changes in this session (or the files specified)
against the project's standards and design intent.

Before reviewing, read:
- docs/PROJECT.md for priorities and constraints
- DIFF_CONTEXT.md for the intent behind recent changes
- The relevant module doc in docs/modules/ for expected patterns
- docs/conventions.md for cross-codebase invariants

## Review Process

1. Intent alignment: Do the changes accomplish what DIFF_CONTEXT.md
   says they should? Is anything missing or divergent from the
   stated intent? If no DIFF_CONTEXT.md exists, ask me what the
   changes were intended to accomplish before proceeding.

2. Bugs: Identify concrete bugs — not hypothetical concerns,
   actual code paths that will produce wrong results. For each:
   - What triggers it (specific input, state, or sequence)
   - What happens vs what should happen
   - Severity: critical (data corruption, security) / high
     (blocks functionality) / medium (wrong output under
     specific conditions) / low (edge case, cosmetic)

3. Logic errors: Identify flawed reasoning in the code. Off-by-one,
   incorrect boundary conditions, wrong operator, inverted boolean,
   race conditions, incorrect null handling. For each:
   - The specific location (file and function)
   - What it does vs what it should do
   - A concrete fix

4. Efficiency: Identify inefficiencies that matter given the
   priority ordering in docs/PROJECT.md. Ignore micro-optimizations.
   Focus on:
   - Unnecessary API/DB calls, N+1 queries, unbounded loops
   - Wasteful allocations in hot paths
   - Missed opportunities to use existing utilities in the codebase
   - Violations of cross-codebase invariants (e.g., per-candidate
     Postgres queries instead of bulk fetch, partial DAG caching)

5. Standards compliance: Check against .claude/rules/coding-standards.md
   and docs/conventions.md. Flag violations only — do not list
   things that are already correct.

6. Security: Check for unsanitized inputs, leaked secrets or PII
   in logs, SQL injection vectors, and any authentication or
   authorization gaps. Only flag concrete issues.

## Output

Organize findings by severity. Lead with what matters most:

**Critical** — Will cause incorrect behavior, data corruption,
or security vulnerability. Must fix before commit.

**Warning** — Potential issue under specific conditions, or
significant inefficiency worth addressing.

**Suggestion** — Improvement that isn't blocking. Style, minor
efficiency, or readability.

For each finding: state the file and function, describe the
problem concretely, and propose a specific fix.

Do not pad the review. If a category has no findings, skip it.
If the code is clean, say so in one sentence.

Focus on: $ARGUMENTS