# ADR-013: Structured Documentation System

**Status:** Active

## Context

The codebase previously kept deep technical detail in 23 markdown
files under `guides/`. These were authoritative but had no structure
for product context, architectural rationale, or per-module summaries.
As the project grows and Claude Code agents work across the codebase,
unstructured guides make it hard to find the right information quickly,
hard to detect staleness, and impossible for agents to update docs
autonomously without risk of overwriting stable design decisions.

## Decision

Replace the flat `guides/` directory with a structured docs/ hierarchy:

- `docs/PROJECT.md` — product context, priorities, constraints,
  module map. Human-only updates via `/extract-finalized-decisions`.
- `docs/conventions.md` — cross-codebase invariants and patterns.
  Human-only; agents flag inconsistencies rather than fixing.
- `docs/modules/` — concise per-module summaries (what, boundaries,
  patterns, gotchas). Agents may update autonomously when they find
  staleness while working in a module.
- `docs/decisions/` — append-only numbered ADRs for significant
  architectural choices. Written only via docs-maintainer subagent
  or explicit human commands (`/save-decisions`, `/promote`).
- `DIFF_CONTEXT.md` — transient working context cleared after each
  commit by the docs-maintainer subagent.

Deep technical detail from `guides/` is preserved in context by
referencing it from CLAUDE.md and module docs, with guides/ kept
as an archive (or deleted after migration is verified complete).

## Alternatives Considered

1. **Keep guides/ as-is**: No structure for product priorities or
   architectural rationale. Agents must read many large files to
   orient themselves. No separation between stable decisions and
   mutable module-level notes.
2. **Single wiki-style flat file**: Easy to write, impossible to
   maintain at scale. No clear ownership rules for which sections
   agents may touch autonomously.
3. **Inline comments only**: Doesn't capture product priorities,
   cross-cutting invariants, or decisions that span multiple files.

## Consequences

- Agents can orient themselves quickly: read PROJECT.md for context,
  the relevant module doc for boundaries, check decisions/ for
  precedent — all before touching code.
- Module docs are explicitly agent-writable, keeping them fresh
  without human overhead.
- Decision records are append-only and agent-protected, preventing
  accidental loss of rationale.
- The docs-maintainer subagent processes DIFF_CONTEXT.md at commit
  time, closing the loop between code changes and documentation.
- guides/ references embedded in older ADRs become stale if guides/
  is deleted; those references should be updated when the guides/
  directory is formally removed.

## References

- docs/PROJECT.md (product context and priorities)
- docs/conventions.md (cross-codebase invariants)
- CLAUDE.md (autonomous documentation rules for agents)
- .claude/rules/docs-awareness.md (what agents may/may not update)
- .claude/rules/context-tracking.md (DIFF_CONTEXT entry format)
