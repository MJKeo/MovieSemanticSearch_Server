# Module Documentation

This directory contains concise documentation for each major module
in the codebase. These docs describe *what* a module does and *how*
to work with it — not exhaustive API references.

## What Belongs Here

- What the module does (1-2 sentences)
- Module boundaries: what's in scope, what's not
- Key internal patterns and conventions
- Interactions with other modules
- Known gotchas and footguns

## How Module Docs Are Updated

- **Autonomously** by Claude when it discovers a stale doc while
  working in that module (included in the same changeset)
- **Autonomously** by the docs-maintainer subagent at commit time
- **On demand** via the docs-auditor subagent (reports staleness)

Keep each module doc under 60 lines.
