---
name: "review-code"
description: "Use when the user asks for a review of current-session changes or specific files and wants concrete bugs, logic errors, inefficiencies, and standards violations."
---

# Review Code

Perform a standards- and intent-aware code review focused on real findings.

## Workflow
1. Prioritize real bugs and regressions over style commentary.
2. Validate intent against `DIFF_CONTEXT.md` before judging implementation quality.
3. Follow the severity buckets and finding format from `references/original-command.md`.
4. If the code is clean, say so briefly rather than padding the review.

## References
- `references/original-command.md` -> exact review checklist and output format
- `docs/PROJECT.md`, `DIFF_CONTEXT.md`, `docs/modules/`, `docs/conventions.md` -> review baseline
