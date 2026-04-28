# Workflow Suggestions

Potential automations and workflow improvements observed across
sessions. Run /review-workflow to go through them interactively.

Entries are added automatically during /safe-clear when repeated
manual patterns are detected.

## ~~Backfill command for re-fetching data into tracker DB~~ IMPLEMENTED
Implemented as `movie_ingestion/imdb_scraping/backfill_awards_boxoffice.py` (2026-04-08).
Standalone script that queries `status='ingested'`, reuses process_movie pipeline,
writes only imdb_data (INSERT OR REPLACE), skips status updates and filter logging.
Pattern is reusable for future backfills — copy and adjust the candidate query.

## Doc folder consolidation command
**Pattern observed:** User asked to review each file in a planning folder one-by-one, present a brief description, ask keep/consume, then extract knowledge and delete consumed files. This was a ~45-minute manual process across 13 files with repeated read-present-ask-extract-delete cycles.
**Suggested implementation:** command — `/consolidate-docs <folder>` that iterates files in a folder, presents each with a summary, asks keep/consume/defer, and for consumed files asks which target docs should receive extracted knowledge.
**Rationale:** The review-present-decide-extract loop is mechanical once the user makes the keep/consume decision. A command could automate the file iteration, summary generation, and deletion bookkeeping while keeping the human decision point.

## Audit script for LLM-facing enum prompt fields
**Pattern observed:** After editing `schemas/trait_category.py`, I manually wrote a one-shot Python script to verify (a) all 44 members have all 7 fields populated, (b) no project-internal acronyms (ENT/FRA/META/KW/SEM) appear in any prompt-injected field, (c) no `→ CAT_A + CAT_B` shorthand exists outside an explicit "two traits" / "three traits" / "splits" framing. Each check was a few lines of regex over the enum's prompt fields. The same script — or one parameterized over the target enum — would be useful every time an LLM-facing taxonomy enum is modified, since this codebase has several (`CategoryName`, `OverallKeyword`, `NarrativeStructureTag`, etc.).
**Suggested implementation:** skill or command — `/audit-llm-enum <enum-import-path>` that imports the enum, iterates members, checks all prompt fields for acronym leakage (configurable list per project), unmarked multi-category mappings, completeness of fields, and stale references to non-existent enum values. Print findings grouped by member.
**Rationale:** Manual scripts get rewritten each session and miss prior gotchas (FEMALE_LEAD-style stale references, internal jargon like "handler stage" / "dispatcher default"). A dedicated audit lets the rules accumulate as the project's LLM-facing enums grow, and it's natural to run as a pre-commit on changes to schemas/.
