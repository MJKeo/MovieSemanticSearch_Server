# DIFF_CONTEXT
Active context for uncommitted changes in the current working session.

## Early truncation in build_requests to avoid wasted work
Files: `movie_ingestion/metadata_generation/request_builder.py`, `movie_ingestion/metadata_generation/run.py` | Added `max_batches` param to `build_requests` so it truncates the eligible movie list before loading data and building prompts, instead of building everything then discarding excess batches in the caller.
