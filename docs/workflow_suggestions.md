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

