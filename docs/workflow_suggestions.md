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

