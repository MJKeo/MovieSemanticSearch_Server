Run the docs-auditor subagent to perform a full staleness scan
of all permanent documentation.

The auditor will check:
- Module docs against actual code
- Conventions against codebase patterns
- Decision records for references to deprecated/changed code
- PROJECT.md priorities against recent decision patterns

Review the report and tell me what needs attention.
I will decide what to act on.
